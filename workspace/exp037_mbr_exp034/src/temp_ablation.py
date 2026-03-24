"""
exp037: 温度別候補品質 + MBR組み合わせグリッドサーチ
- 各温度で1候補生成 → 個別スコア・rep%を比較
- MBR組み合わせを網羅的に試してスコア比較

Usage: python temp_ablation.py [--fold N]
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
from itertools import combinations

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

# 温度リスト: 0はgreedy扱い
TEMPERATURES = [0, 0.2, 0.4, 0.6, 0.8, 1.05]

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


akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_expanded = simple_sentence_aligner(train_df, keep_oare_id=True)
train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=train_expanded["akt_group"].values))
_, val_idx = splits[FOLD]
val_data = train_expanded.iloc[val_idx].copy()
logger.info(f"GroupKFold fold={FOLD}, val={len(val_data)} samples")

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

# sent-CV only (faster, and LB correlates better with sent-CV)
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

logger.info(f"sent-CV: {len(sent_inputs)} sents")

# ============================================================
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info(f"Model loaded from {MODEL_PATH}")

# ============================================================
# Dynamic padding + length sort
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


def _make_sorted_loader(inputs, batch_size):
    ds = DynamicPaddingDataset(inputs, tokenizer, MAX_LENGTH)
    lengths = [ds.items[i]["input_ids"].size(0) for i in range(len(ds))]
    sorted_indices = sorted(range(len(ds)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(ds, sorted_indices)
    loader = DataLoader(sorted_ds, batch_size=batch_size, shuffle=False, collate_fn=dynamic_collate_fn)
    return loader, idx_map, len(inputs)


# ============================================================
# MBR
# ============================================================
chrfpp = sacrebleu.metrics.CHRF(word_order=2)


def mbr_pick(candidates):
    cands = list(dict.fromkeys(candidates))
    n = len(cands)
    if n <= 1:
        return cands[0] if cands else ""
    scores = []
    for i in range(n):
        s = sum(float(chrfpp.sentence_score(cands[i], [cands[j]]).score) for j in range(n) if j != i)
        scores.append(s / (n - 1))
    return cands[int(np.argmax(scores))]


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


def calc_metrics(preds, refs):
    preds_clean = [repeat_cleanup(p) for p in preds]
    chrf = sacrebleu.corpus_chrf(preds_clean, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds_clean, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in preds_clean) / len(preds_clean)
    return {
        "chrf": round(chrf.score, 2),
        "bleu": round(bleu.score, 2),
        "geo": round(geo, 2),
        "rep": round(rep_rate, 1),
    }


# ============================================================
# Phase 1: Generate 1 candidate per temperature
# ============================================================
logger.info("=" * 60)
logger.info("=== Phase 1: Temperature ablation (1 candidate each) ===")

loader, idx_map, n = _make_sorted_loader(sent_inputs, BATCH_SIZE)
torch.manual_seed(42)  # reproducibility

# Dict: temp_label -> list of predictions (original order)
temp_preds = {}

for temp in TEMPERATURES:
    label = f"t={temp}" if temp > 0 else "greedy"
    logger.info(f"Generating {label}...")
    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=label, leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            if temp == 0:
                out = model.generate(input_ids=ids, attention_mask=mask,
                                     max_length=MAX_LENGTH, num_beams=1)
            else:
                out = model.generate(input_ids=ids, attention_mask=mask,
                                     max_length=MAX_LENGTH,
                                     do_sample=True, num_beams=1,
                                     top_p=0.9, temperature=temp,
                                     num_return_sequences=1)
            sorted_preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])

    preds = [""] * n
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    temp_preds[label] = preds

# Evaluate each temperature individually
logger.info("")
logger.info("=== Individual temperature scores (sent-CV) ===")
logger.info(f"{'temp':<10} {'chrF++':>8} {'BLEU':>8} {'geo':>8} {'rep%':>8}")
logger.info("-" * 46)

individual_results = {}
for label, preds in temp_preds.items():
    m = calc_metrics(preds, sent_refs)
    individual_results[label] = m
    logger.info(f"{label:<10} {m['chrf']:>8.2f} {m['bleu']:>8.2f} {m['geo']:>8.2f} {m['rep']:>7.1f}%")

# ============================================================
# Phase 2: MBR combinations
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== Phase 2: MBR combinations (sent-CV) ===")

temp_labels = list(temp_preds.keys())

# Try all combinations of size 2 to 6
mbr_results = {}

for size in range(2, len(temp_labels) + 1):
    for combo in combinations(range(len(temp_labels)), size):
        combo_labels = [temp_labels[i] for i in combo]
        combo_name = "+".join(combo_labels)

        # Build candidate pool per sample
        mbr_preds = []
        for sample_idx in range(n):
            cands = [temp_preds[label][sample_idx] for label in combo_labels]
            mbr_preds.append(mbr_pick(cands))

        m = calc_metrics(mbr_preds, sent_refs)
        mbr_results[combo_name] = m

# Sort by geo score and print
logger.info(f"{'combination':<55} {'chrF++':>8} {'BLEU':>8} {'geo':>8} {'rep%':>8}")
logger.info("-" * 91)

sorted_mbr = sorted(mbr_results.items(), key=lambda x: -x[1]["geo"])
for name, m in sorted_mbr:
    logger.info(f"{name:<55} {m['chrf']:>8.2f} {m['bleu']:>8.2f} {m['geo']:>8.2f} {m['rep']:>7.1f}%")

# ============================================================
# Save results
# ============================================================
output = {
    "individual": individual_results,
    "mbr_combinations": mbr_results,
    "fold": FOLD,
    "temperatures": TEMPERATURES,
    "top10_mbr": [{"combo": name, **m} for name, m in sorted_mbr[:10]],
}
out_path = RESULTS_DIR / f"temp_ablation_fold{FOLD}.json"
with open(str(out_path), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

logger.info("")
logger.info(f"Results saved to {out_path}")
logger.info("Done.")
