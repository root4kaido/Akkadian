"""
exp037: chrF++ MBR vs Weighted MBR 比較
- temp_ablationと同じ候補を再生成（seed=42固定）
- MBRスコアリング3種を比較:
  1. chrF++ only
  2. weighted: chrF++(0.55) + BLEU(0.25) + Jaccard(0.20) + LenBonus(0.10)
  3. weighted_no_len: chrF++(0.6) + BLEU(0.25) + Jaccard(0.15) (length bonus抜き)

Usage: python weighted_mbr_ablation.py [--fold N]
"""
import os
import re
import sys
import math
import json
import time
import logging
import argparse
import pickle
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

TEMPERATURES = [0, 0.2, 0.4, 0.6, 0.8, 1.05]
# beam search candidates: num_beams=4, top-3 return
BEAM_NUM_BEAMS = 4
BEAM_NUM_RETURN = 3
CACHE_PATH = RESULTS_DIR / f"candidates_fold{FOLD}.pkl"

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
# Data preparation (GroupKFold) — sent-CV only
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
# Model & inference helpers
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


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


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
# Phase 1: Generate candidates (or load from cache)
# ============================================================
n = len(sent_inputs)

if CACHE_PATH.exists():
    logger.info(f"Loading cached candidates from {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        temp_preds = pickle.load(f)
    logger.info(f"Loaded {len(temp_preds)} temperature predictions")
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    logger.info(f"Model loaded from {MODEL_PATH}")

    logger.info("=" * 60)
    logger.info("=== Phase 1: Generating candidates (seed=42) ===")

    loader, idx_map, _ = _make_sorted_loader(sent_inputs, BATCH_SIZE)
    torch.manual_seed(42)

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

    # Beam search candidates (num_beams=4, top-3)
    logger.info(f"Generating beam4 (top-{BEAM_NUM_RETURN} candidates)...")
    sorted_beam_preds = []
    mbr_batch_size = max(1, BATCH_SIZE // 2)
    loader_beam, idx_map_beam, _ = _make_sorted_loader(sent_inputs, mbr_batch_size)
    with torch.no_grad():
        for batch in tqdm(loader_beam, desc="beam4", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            B = ids.shape[0]
            out = model.generate(input_ids=ids, attention_mask=mask,
                                 max_length=MAX_LENGTH,
                                 num_beams=BEAM_NUM_BEAMS,
                                 num_return_sequences=BEAM_NUM_RETURN,
                                 early_stopping=True)
            decoded = [d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)]
            for i in range(B):
                cands = decoded[i * BEAM_NUM_RETURN: (i + 1) * BEAM_NUM_RETURN]
                sorted_beam_preds.append(cands)
    # Reorder to original indices
    beam_cands_per_sample = [None] * n
    for new_idx, cands in enumerate(sorted_beam_preds):
        beam_cands_per_sample[idx_map_beam[new_idx]] = cands
    # Store as beam4_0, beam4_1, beam4_2
    for rank in range(BEAM_NUM_RETURN):
        label = f"beam4_{rank}"
        temp_preds[label] = [cands[rank] for cands in beam_cands_per_sample]

    # Save cache
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(temp_preds, f)
    logger.info(f"Candidates cached to {CACHE_PATH}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

# ============================================================
# MBR scoring functions
# ============================================================
chrfpp_metric = sacrebleu.metrics.CHRF(word_order=2)
bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)  # sentence-level smoothed BLEU


def score_chrf(hyp, ref):
    return float(chrfpp_metric.sentence_score(hyp, [ref]).score) / 100.0


def score_bleu(hyp, ref):
    return float(bleu_metric.sentence_score(hyp, [ref]).score) / 100.0


def score_jaccard(hyp, ref):
    """Word-level Jaccard similarity"""
    h_set = set(hyp.lower().split())
    r_set = set(ref.lower().split())
    if not h_set and not r_set:
        return 1.0
    if not h_set or not r_set:
        return 0.0
    return len(h_set & r_set) / len(h_set | r_set)


def score_length_bonus(hyp, ref):
    """Length similarity: 1 - |len_h - len_r| / max(len_h, len_r)"""
    len_h = len(hyp.split())
    len_r = len(ref.split())
    if len_h == 0 and len_r == 0:
        return 1.0
    if len_h == 0 or len_r == 0:
        return 0.0
    return 1.0 - abs(len_h - len_r) / max(len_h, len_r)


def mbr_pick_chrf(candidates):
    """chrF++ only consensus"""
    cands = list(dict.fromkeys(candidates))
    nc = len(cands)
    if nc <= 1:
        return cands[0] if cands else ""
    scores = []
    for i in range(nc):
        s = sum(score_chrf(cands[i], cands[j]) for j in range(nc) if j != i)
        scores.append(s / (nc - 1))
    return cands[int(np.argmax(scores))]


def mbr_pick_weighted(candidates, w_chrf=0.55, w_bleu=0.25, w_jaccard=0.20, w_len=0.10):
    """Weighted MBR: chrF++ + BLEU + Jaccard + Length bonus"""
    cands = list(dict.fromkeys(candidates))
    nc = len(cands)
    if nc <= 1:
        return cands[0] if cands else ""
    scores = []
    for i in range(nc):
        total = 0.0
        for j in range(nc):
            if j == i:
                continue
            s = (w_chrf * score_chrf(cands[i], cands[j])
                 + w_bleu * score_bleu(cands[i], cands[j])
                 + w_jaccard * score_jaccard(cands[i], cands[j])
                 + w_len * score_length_bonus(cands[i], cands[j]))
            total += s
        scores.append(total / (nc - 1))
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
# Phase 2: Compare MBR scoring methods
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== Phase 2: MBR scoring comparison ===")

temp_labels = list(temp_preds.keys())

# Define MBR methods
mbr_methods = {
    "chrf_only": lambda cands: mbr_pick_chrf(cands),
    "weighted_full": lambda cands: mbr_pick_weighted(cands, 0.55, 0.25, 0.20, 0.10),
    "weighted_no_len": lambda cands: mbr_pick_weighted(cands, 0.60, 0.25, 0.15, 0.00),
    "bleu_heavy": lambda cands: mbr_pick_weighted(cands, 0.40, 0.40, 0.10, 0.10),
    "jaccard_heavy": lambda cands: mbr_pick_weighted(cands, 0.40, 0.20, 0.30, 0.10),
}

# Key combinations to test (top performers from temp_ablation + smaller combos)
key_combos = [
    # --- baselines (single) ---
    ["greedy"],
    ["t=0.4"],
    ["beam4_0"],  # beam4 top-1
    # --- sampling only ---
    ["t=0.2", "t=0.4"],
    ["t=0.2", "t=0.4", "t=0.6"],
    ["t=0.2", "t=0.4", "t=0.8", "t=1.05"],  # best from temp_ablation
    # --- beam only ---
    ["beam4_0", "beam4_1", "beam4_2"],
    # --- beam + sampling mix ---
    ["beam4_0", "t=0.4"],
    ["beam4_0", "t=0.2", "t=0.4"],
    ["beam4_0", "t=0.4", "t=0.6"],
    ["beam4_0", "beam4_1", "t=0.4", "t=0.6"],
    ["beam4_0", "t=0.2", "t=0.4", "t=0.6", "t=0.8"],
    # --- greedy + beam + sampling ---
    ["greedy", "beam4_0", "t=0.4"],
    ["greedy", "beam4_0", "t=0.2", "t=0.4", "t=0.6"],
    # --- large pool ---
    ["greedy", "t=0.2", "t=0.4", "t=0.6", "t=0.8", "t=1.05"],
    ["greedy", "beam4_0", "beam4_1", "t=0.2", "t=0.4", "t=0.6", "t=0.8"],
]

results = {}

for combo_labels in key_combos:
    combo_name = "+".join(combo_labels)
    n_cands = len(combo_labels)

    if n_cands == 1:
        # Single candidate — no MBR needed, just evaluate
        preds = temp_preds[combo_labels[0]]
        m = calc_metrics(preds, sent_refs)
        results[f"{combo_name} (single)"] = m
        logger.info(f"{combo_name:<50} {'single':<18} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")
        continue

    for method_name, mbr_fn in mbr_methods.items():
        logger.info(f"  Computing {combo_name} x {method_name}...")
        mbr_preds = []
        for sample_idx in range(n):
            cands = [temp_preds[label][sample_idx] for label in combo_labels]
            mbr_preds.append(mbr_fn(cands))
        m = calc_metrics(mbr_preds, sent_refs)
        key = f"{combo_name} | {method_name}"
        results[key] = m
        logger.info(f"{combo_name:<50} {method_name:<18} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

    logger.info("")  # separator between combos

# ============================================================
# Summary: sorted by geo
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== Summary: all results sorted by geo ===")
logger.info(f"{'config':<70} {'chrF++':>8} {'BLEU':>8} {'geo':>8} {'rep%':>8}")
logger.info("-" * 98)

sorted_results = sorted(results.items(), key=lambda x: -x[1]["geo"])
for name, m in sorted_results:
    logger.info(f"{name:<70} {m['chrf']:>8.2f} {m['bleu']:>8.2f} {m['geo']:>8.2f} {m['rep']:>7.1f}%")

# ============================================================
# Save
# ============================================================
output = {"results": results, "fold": FOLD}
out_path = RESULTS_DIR / f"weighted_mbr_fold{FOLD}.json"
with open(str(out_path), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

logger.info(f"\nResults saved to {out_path}")
logger.info("Done.")
