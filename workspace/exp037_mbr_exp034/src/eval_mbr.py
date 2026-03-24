"""
exp037: exp034モデルでの単純MBR（3候補）評価
- beam4で3候補生成 → chrF++ consensus で最良を選択
- ペナルティなし、動的パディング+長さソート
- greedy/beam4ベースラインも同時評価

Usage: python eval_mbr.py [--fold N]
"""
import os
import re
import sys
import math
import json
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=3)
args = parser.parse_args()
FOLD = args.fold

# ============================================================
# Config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = str(PROJECT_ROOT / "workspace" / "exp034_st_pretrain" / "results" / f"fold{FOLD}" / "last_model")
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_LENGTH = 512
BATCH_SIZE = 4

# MBR config
NUM_BEAMS = 4
NUM_RETURN_SEQUENCES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Preprocessing (exp023)
# ============================================================
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002


def _decimal_to_fraction_approx(match):
    dec_str = match.group(0)
    try:
        value = float(dec_str)
    except ValueError:
        return dec_str
    int_part = int(value)
    frac_part = value - int_part
    if frac_part < 0.001:
        return dec_str
    best_frac, best_dist = None, float('inf')
    for target, symbol in FRACTION_TARGETS.items():
        dist = abs(frac_part - target)
        if dist < best_dist:
            best_dist, best_frac = dist, symbol
    if best_dist <= APPROX_TOLERANCE:
        return best_frac if int_part == 0 else f"{int_part} {best_frac}"
    return dec_str


def preprocess_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction_approx, text)
    return text


# ============================================================
# Data preparation (GroupKFold)
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))


def simple_sentence_aligner(df, keep_oare_id=False):
    aligned_data = []
    for _, row in df.iterrows():
        src = str(row["transliteration"])
        tgt = str(row["translation"])
        tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split("\n") if s.strip()]
        entry_base = {"oare_id": row["oare_id"]} if keep_oare_id else {}
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({**entry_base, "transliteration": s, "translation": t})
        else:
            aligned_data.append({**entry_base, "transliteration": src, "translation": tgt})
    return pd.DataFrame(aligned_data)


# GroupKFold
akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_expanded = simple_sentence_aligner(train_df, keep_oare_id=True)
train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=train_expanded["akt_group"].values))
_, val_idx = splits[FOLD]
val_data = train_expanded.iloc[val_idx].copy()
logger.info(f"GroupKFold fold={FOLD}, val={len(val_data)} samples")

# sentence_aligned
sent_aligned = pd.read_csv(str(SENTENCE_ALIGNED_PATH))
oare_to_row = {row['oare_id']: row for _, row in train_df.iterrows()}
alignment_dict = {}
for oare_id, group in sent_aligned.groupby('oare_id'):
    group = group.sort_values('sent_idx')
    alignment_dict[oare_id] = [
        {'akk_segment': str(r['akk_segment']), 'eng_sentence': str(r['eng_sentence'])}
        for _, r in group.iterrows()
    ]
translit_to_oare = {str(row['transliteration']): oare_id for oare_id, row in oare_to_row.items()}

prefix = "translate Akkadian to English: "

# --- sent-CV ---
sent_inputs, sent_refs = [], []
for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    oare_id = translit_to_oare.get(translit)
    if oare_id and oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            for b in boundaries:
                if b['akk_segment'].strip() and b['eng_sentence'].strip():
                    sent_inputs.append(prefix + preprocess_transliteration(b['akk_segment']))
                    sent_refs.append(b['eng_sentence'])

# --- doc-CV ---
doc_inputs, doc_refs = [], []
for _, row in val_data.iterrows():
    translit = str(row['transliteration'])
    translation = str(row['translation'])
    if translit.strip() and translation.strip():
        doc_inputs.append(prefix + preprocess_transliteration(translit))
        doc_refs.append(translation)

logger.info(f"sent-CV: {len(sent_inputs)} sents, doc-CV: {len(doc_inputs)} docs")

# ============================================================
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info(f"Model loaded from {MODEL_PATH}")

# ============================================================
# Dynamic padding + length sort inference
# ============================================================
class DynamicPaddingDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length):
        self.items = []
        for t in texts:
            enc = tokenizer(t, max_length=max_length, truncation=True, return_tensors="pt")
            self.items.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def dynamic_collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_id = tokenizer.pad_token_id or 0
    input_ids, attention_mask = [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask)}


def run_greedy(inputs, desc):
    """Greedy推論（num_beams=1）"""
    ds = DynamicPaddingDataset(inputs, tokenizer, MAX_LENGTH)
    lengths = [ds.items[i]["input_ids"].size(0) for i in range(len(ds))]
    sorted_indices = sorted(range(len(ds)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(ds, sorted_indices)
    loader = DataLoader(sorted_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_fn)
    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=1,
            )
            sorted_preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    preds = [""] * len(inputs)
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    return preds


def run_beam(inputs, desc):
    """Beam search推論（num_beams=4, 1候補）"""
    ds = DynamicPaddingDataset(inputs, tokenizer, MAX_LENGTH)
    lengths = [ds.items[i]["input_ids"].size(0) for i in range(len(ds))]
    sorted_indices = sorted(range(len(ds)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(ds, sorted_indices)
    loader = DataLoader(sorted_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_fn)
    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=NUM_BEAMS, early_stopping=True,
            )
            sorted_preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    preds = [""] * len(inputs)
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    return preds


# ============================================================
# MBR
# ============================================================
chrfpp = sacrebleu.metrics.CHRF(word_order=2)


def mbr_pick(candidates):
    """chrF++ consensus selection from candidates."""
    # Dedup keeping order
    cands = list(dict.fromkeys(candidates))
    n = len(cands)
    if n <= 1:
        return cands[0] if cands else ""
    scores = []
    for i in range(n):
        s = sum(float(chrfpp.sentence_score(cands[i], [cands[j]]).score) for j in range(n) if j != i)
        scores.append(s / (n - 1))
    return cands[int(np.argmax(scores))]


def run_mbr(inputs, desc):
    """MBR推論: beam4で3候補生成 → chrF++ consensus"""
    ds = DynamicPaddingDataset(inputs, tokenizer, MAX_LENGTH)
    lengths = [ds.items[i]["input_ids"].size(0) for i in range(len(ds))]
    sorted_indices = sorted(range(len(ds)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(ds, sorted_indices)
    # batch_size=2 for MBR (3 return sequences → 6 outputs per batch)
    mbr_batch_size = max(1, BATCH_SIZE // 2)
    loader = DataLoader(sorted_ds, batch_size=mbr_batch_size, shuffle=False, collate_fn=dynamic_collate_fn)

    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            B = ids.shape[0]
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                num_return_sequences=NUM_RETURN_SEQUENCES,
                early_stopping=True,
            )
            decoded = [d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)]
            # Group by sample: each sample has NUM_RETURN_SEQUENCES candidates
            for i in range(B):
                cands = decoded[i * NUM_RETURN_SEQUENCES: (i + 1) * NUM_RETURN_SEQUENCES]
                sorted_preds.append(mbr_pick(cands))

    preds = [""] * len(inputs)
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    return preds


# ============================================================
# Metrics
# ============================================================
def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i + n] == words[i + n:i + 2 * n]:
                return " ".join(words[:i + n])
    return text


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


def calc_metrics(preds, refs, label):
    preds_clean = [repeat_cleanup(p) for p in preds]
    chrf = sacrebleu.corpus_chrf(preds_clean, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds_clean, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in preds_clean) / len(preds_clean)
    logger.info(f"  {label}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep_rate:.1f}%")
    return {"chrf": round(chrf.score, 2), "bleu": round(bleu.score, 2), "geo": round(geo, 2), "rep": round(rep_rate, 1)}


# ============================================================
# Sampling MBR
# ============================================================
TEMPERATURES = [0.6, 0.8, 1.05]
NUM_SAMPLE_PER_TEMP = 2


def _make_sorted_loader(inputs, batch_size):
    """動的パディング+ソート済みloaderとインデックスマップを返す"""
    ds = DynamicPaddingDataset(inputs, tokenizer, MAX_LENGTH)
    lengths = [ds.items[i]["input_ids"].size(0) for i in range(len(ds))]
    sorted_indices = sorted(range(len(ds)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(ds, sorted_indices)
    loader = DataLoader(sorted_ds, batch_size=batch_size, shuffle=False, collate_fn=dynamic_collate_fn)
    return loader, idx_map, len(inputs)


def run_sampling_mbr(inputs, desc, include_greedy=False):
    """Sampling MBR: multi-temp sampling候補 → chrF++ consensus"""
    loader, idx_map, n = _make_sorted_loader(inputs, BATCH_SIZE)

    # Collect sampling candidates per temperature
    all_cands = [[] for _ in range(n)]

    # Optionally add greedy as anchor
    if include_greedy:
        logger.info(f"[{desc}] Generating greedy anchor...")
        sorted_greedy = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{desc} greedy", leave=False):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                out = model.generate(input_ids=ids, attention_mask=mask, max_length=MAX_LENGTH, num_beams=1)
                sorted_greedy.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
        for new_idx, pred in enumerate(sorted_greedy):
            all_cands[idx_map[new_idx]].append(pred)

    for temp in TEMPERATURES:
        logger.info(f"[{desc}] Sampling temp={temp}, n_ret={NUM_SAMPLE_PER_TEMP}...")
        sorted_samp = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{desc} t={temp}", leave=False):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                out = model.generate(
                    input_ids=ids, attention_mask=mask,
                    max_length=MAX_LENGTH,
                    do_sample=True, num_beams=1, top_p=0.9, temperature=temp,
                    num_return_sequences=NUM_SAMPLE_PER_TEMP,
                )
                sorted_samp.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
        # Reshape and assign to original indices
        for new_idx in range(n):
            orig_idx = idx_map[new_idx]
            cands = sorted_samp[new_idx * NUM_SAMPLE_PER_TEMP: (new_idx + 1) * NUM_SAMPLE_PER_TEMP]
            all_cands[orig_idx].extend(cands)

    # MBR selection
    logger.info(f"[{desc}] MBR selection...")
    preds = [mbr_pick(cands) for cands in all_cands]
    return preds


# ============================================================
# Run sampling MBR strategies
# ============================================================
results = {}

# 1. Sampling only (3 temps × 2 = 6 candidates)
logger.info("=" * 60)
logger.info(f"=== Sampling MBR (3temp×{NUM_SAMPLE_PER_TEMP} = {len(TEMPERATURES) * NUM_SAMPLE_PER_TEMP}候補) ===")
t0 = time.time()
samp_sent = run_sampling_mbr(sent_inputs, "sample_6 sent-CV", include_greedy=False)
samp_doc = run_sampling_mbr(doc_inputs, "sample_6 doc-CV", include_greedy=False)
t_samp = time.time() - t0
results["sample_6"] = {
    "sent_cv": calc_metrics(samp_sent, sent_refs, f"sample_6 sent-CV ({len(sent_inputs)} sents)"),
    "doc_cv": calc_metrics(samp_doc, doc_refs, f"sample_6 doc-CV ({len(doc_inputs)} docs)"),
    "time_s": round(t_samp, 0),
}

# 2. Greedy + Sampling (1 + 3 temps × 2 = 7 candidates)
logger.info("=" * 60)
logger.info(f"=== Greedy + Sampling MBR (1+{len(TEMPERATURES) * NUM_SAMPLE_PER_TEMP} = {1 + len(TEMPERATURES) * NUM_SAMPLE_PER_TEMP}候補) ===")
t0 = time.time()
gsamp_sent = run_sampling_mbr(sent_inputs, "greedy+sample_7 sent-CV", include_greedy=True)
gsamp_doc = run_sampling_mbr(doc_inputs, "greedy+sample_7 doc-CV", include_greedy=True)
t_gsamp = time.time() - t0
results["greedy_plus_sample_7"] = {
    "sent_cv": calc_metrics(gsamp_sent, sent_refs, f"greedy+sample_7 sent-CV ({len(sent_inputs)} sents)"),
    "doc_cv": calc_metrics(gsamp_doc, doc_refs, f"greedy+sample_7 doc-CV ({len(doc_inputs)} docs)"),
    "time_s": round(t_gsamp, 0),
}

# ============================================================
# Summary
# ============================================================
logger.info("=" * 60)
logger.info(f"=== Summary (fold{FOLD}) ===")
logger.info("  (prior results: greedy=37.38/24.70, beam4=36.71/24.31, mbr_3cand=37.40/24.71)")
for name, r in results.items():
    sg = r["sent_cv"]["geo"]
    dg = r["doc_cv"]["geo"]
    logger.info(f"  {name:25s}: sent-CV={sg:.2f}, doc-CV={dg:.2f}, time={r['time_s']:.0f}s")

# Save
results["fold"] = FOLD
results["model_path"] = MODEL_PATH
with open(str(RESULTS_DIR / f"sampling_mbr_fold{FOLD}.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
logger.info(f"Results saved to {RESULTS_DIR / f'sampling_mbr_fold{FOLD}.json'}")
logger.info("Done.")
