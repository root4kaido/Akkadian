"""
exp037: eval_full — 6構成のsent-CV + doc-CV評価
  1. beam4 (baseline)
  2. t=0.4 single
  3. t=0.2+0.4+0.6 | rt_weighted
  4. t=0.2+0.4+0.6 | rt_chrf
  5. t=0.2+0.4+0.6 | rt_bleu
  6. beam4_0+t=0.4 | eng_MBR_chrf

sent-CV: sentence_aligned.csvベースの文単位評価（既存キャッシュ活用）
doc-CV: train.csvのドキュメント単位評価（新規生成）

Usage: python eval_full_rerank.py [--fold N] [--batch_size 4]
"""
import os
import re
import sys
import math
import json
import logging
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()
FOLD = args.fold
BATCH_SIZE = args.batch_size

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

CAND_CACHE_PATH = RESULTS_DIR / f"candidates_fold{FOLD}.pkl"
BT_CACHE_PATH = RESULTS_DIR / f"backtrans_fold{FOLD}.pkl"
DOC_CACHE_PATH = RESULTS_DIR / f"doc_candidates_fold{FOLD}.pkl"
DOC_BT_CACHE_PATH = RESULTS_DIR / f"doc_backtrans_fold{FOLD}.pkl"

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
# Data: GroupKFold split
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
val_sent_data = train_expanded.iloc[val_idx].copy()
logger.info(f"GroupKFold fold={FOLD}, val_sent={len(val_sent_data)} samples")

# Doc-level val data (from train.csv directly, not sentence-aligned)
val_oare_ids = set(val_sent_data["oare_id"].unique())
val_doc_data = train_df[train_df["oare_id"].isin(val_oare_ids)].copy()
logger.info(f"Doc-level val: {len(val_doc_data)} docs")

# sentence_aligned info
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

prefix_fwd = "translate Akkadian to English: "
prefix_rev = "translate English to Akkadian: "

# ============================================================
# Sent-CV inputs/refs
# ============================================================
sent_inputs_raw = []
sent_inputs = []
sent_refs = []
for idx, row in val_sent_data.iterrows():
    translit = str(row['transliteration'])
    oare_id = translit_to_oare.get(translit)
    if oare_id and oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            for b in boundaries:
                if b['akk_segment'].strip() and b['eng_sentence'].strip():
                    sent_inputs_raw.append(preprocess_transliteration(b['akk_segment']))
                    sent_inputs.append(prefix_fwd + preprocess_transliteration(b['akk_segment']))
                    sent_refs.append(b['eng_sentence'])

n_sent = len(sent_inputs)
logger.info(f"sent-CV: {n_sent} sents")

# ============================================================
# Doc-CV inputs/refs
# ============================================================
doc_inputs_raw = []
doc_inputs = []
doc_refs = []
for _, row in val_doc_data.iterrows():
    translit = str(row['transliteration'])
    translation = str(row['translation'])
    if translit.strip() and translation.strip():
        doc_inputs_raw.append(preprocess_transliteration(translit))
        doc_inputs.append(prefix_fwd + preprocess_transliteration(translit))
        doc_refs.append(translation)

n_doc = len(doc_inputs)
logger.info(f"doc-CV: {n_doc} docs")

# ============================================================
# Model & inference helpers
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info(f"Model loaded from {MODEL_PATH}")


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


def generate_batch(texts, desc, num_beams=1, temperature=None, do_sample=False, top_p=0.9):
    """Generate translations with dynamic padding + length sorting."""
    ds = DynamicPaddingDataset(texts, tokenizer, MAX_LENGTH)
    lengths = [ds.items[i]["input_ids"].size(0) for i in range(len(ds))]
    sorted_indices = sorted(range(len(ds)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(ds, sorted_indices)
    loader = DataLoader(sorted_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_fn)

    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            gen_kwargs = dict(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH,
            )
            if do_sample:
                gen_kwargs.update(num_beams=1, do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gen_kwargs.update(num_beams=num_beams)

            out = model.generate(**gen_kwargs)
            sorted_preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])

    preds = [""] * len(texts)
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    return preds


def back_translate_batch(texts, desc="back-translate"):
    """Back-translate English texts to Akkadian."""
    rev_inputs = [prefix_rev + t for t in texts]
    return generate_batch(rev_inputs, desc=desc, num_beams=1)


# ============================================================
# Scoring functions
# ============================================================
chrfpp_metric = sacrebleu.metrics.CHRF(word_order=2)
bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)


def score_chrf(hyp, ref):
    return float(chrfpp_metric.sentence_score(hyp, [ref]).score) / 100.0


def score_bleu(hyp, ref):
    return float(bleu_metric.sentence_score(hyp, [ref]).score) / 100.0


def score_jaccard(hyp, ref):
    h_set = set(hyp.lower().split())
    r_set = set(ref.lower().split())
    if not h_set and not r_set:
        return 1.0
    if not h_set or not r_set:
        return 0.0
    return len(h_set & r_set) / len(h_set | r_set)


def score_length_bonus(hyp, ref):
    len_h = len(hyp.split())
    len_r = len(ref.split())
    if len_h == 0 and len_r == 0:
        return 1.0
    if len_h == 0 or len_r == 0:
        return 0.0
    return 1.0 - abs(len_h - len_r) / max(len_h, len_r)


# ============================================================
# Reranking methods (only the 4 we need)
# ============================================================
def eng_mbr_chrf(candidates, back_translations, source):
    """Standard chrF++ MBR on English candidates."""
    nc = len(candidates)
    if nc <= 1:
        return 0
    scores = []
    for i in range(nc):
        s = sum(score_chrf(candidates[i], candidates[j])
                for j in range(nc) if j != i)
        scores.append(s / (nc - 1))
    return int(np.argmax(scores))


def rt_pick_chrf(candidates, back_translations, source):
    if len(candidates) <= 1:
        return 0
    scores = [score_chrf(back_translations[i], source) for i in range(len(candidates))]
    return int(np.argmax(scores))


def rt_pick_bleu(candidates, back_translations, source):
    if len(candidates) <= 1:
        return 0
    scores = [score_bleu(back_translations[i], source) for i in range(len(candidates))]
    return int(np.argmax(scores))


def rt_pick_weighted(candidates, back_translations, source,
                     w_chrf=0.55, w_bleu=0.25, w_jaccard=0.20, w_len=0.10):
    if len(candidates) <= 1:
        return 0
    scores = []
    for i in range(len(candidates)):
        bt = back_translations[i]
        s = (w_chrf * score_chrf(bt, source)
             + w_bleu * score_bleu(bt, source)
             + w_jaccard * score_jaccard(bt, source)
             + w_len * score_length_bonus(bt, source))
        scores.append(s)
    return int(np.argmax(scores))


# ============================================================
# Metrics
# ============================================================
def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for ng in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * ng + 1):
            if words[i:i + ng] == words[i + ng:i + 2 * ng]:
                return " ".join(words[:i + ng])
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


def apply_rerank(cand_dict, bt_dict, inputs_raw, combo_labels, method_fn):
    """Apply reranking method to select best candidates."""
    n = len(inputs_raw)
    selected = []
    for i in range(n):
        cands = [cand_dict[label][i] for label in combo_labels]
        bts = [bt_dict[label][i] for label in combo_labels]
        src = inputs_raw[i]
        # Deduplicate
        seen = {}
        unique_cands, unique_bts = [], []
        for c, bt in zip(cands, bts):
            if c not in seen:
                seen[c] = True
                unique_cands.append(c)
                unique_bts.append(bt)
        if len(unique_cands) <= 1:
            selected.append(unique_cands[0] if unique_cands else "")
        else:
            best_idx = method_fn(unique_cands, unique_bts, src)
            selected.append(unique_cands[best_idx])
    return selected


# ============================================================
# The 6 configurations to evaluate
# ============================================================
CONFIGS = [
    {"name": "beam4", "type": "single", "label": "beam4_0"},
    {"name": "t=0.4_single", "type": "single", "label": "t=0.4"},
    {"name": "t=0.2+0.4+0.6|rt_weighted", "type": "rerank",
     "combo": ["t=0.2", "t=0.4", "t=0.6"], "method": rt_pick_weighted},
    {"name": "t=0.2+0.4+0.6|rt_chrf", "type": "rerank",
     "combo": ["t=0.2", "t=0.4", "t=0.6"], "method": rt_pick_chrf},
    {"name": "t=0.2+0.4+0.6|rt_bleu", "type": "rerank",
     "combo": ["t=0.2", "t=0.4", "t=0.6"], "method": rt_pick_bleu},
    {"name": "beam4_0+t=0.4|eng_MBR_chrf", "type": "rerank",
     "combo": ["beam4_0", "t=0.4"], "method": eng_mbr_chrf},
]

# Needed candidate labels
NEEDED_LABELS = {"beam4_0", "t=0.2", "t=0.4", "t=0.6"}

# ============================================================
# SENT-CV: Load from existing results (skip re-computation)
# ============================================================
logger.info("=" * 60)
logger.info("=== SENT-CV: Loading from existing roundtrip_rerank results ===")
sent_results_path = RESULTS_DIR / f"roundtrip_rerank_fold{FOLD}.json"
if sent_results_path.exists():
    with open(sent_results_path) as f:
        prev = json.load(f)["results"]
    # Map config names to previous result keys
    SENT_KEY_MAP = {
        "beam4": "beam4_0 (single)",
        "t=0.4_single": "t=0.4 (single)",
        "t=0.2+0.4+0.6|rt_weighted": "t=0.2+t=0.4+t=0.6 | rt_weighted",
        "t=0.2+0.4+0.6|rt_chrf": "t=0.2+t=0.4+t=0.6 | rt_chrf",
        "t=0.2+0.4+0.6|rt_bleu": "t=0.2+t=0.4+t=0.6 | rt_bleu",
        "beam4_0+t=0.4|eng_MBR_chrf": "beam4_0+t=0.4 | eng_MBR_chrf",
    }
    sent_results = {}
    for cfg in CONFIGS:
        key = SENT_KEY_MAP[cfg["name"]]
        sent_results[cfg["name"]] = prev[key]
        m = prev[key]
        logger.info(f"  {cfg['name']:<40} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")
else:
    logger.warning("No previous sent-CV results found, skipping.")
    sent_results = {}

# ============================================================
# DOC-CV: Generate or load doc-level candidates
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== DOC-CV: Generating candidates ===")

# Generation configs for each needed label
GEN_CONFIGS = {
    "beam4_0": {"num_beams": 4, "do_sample": False},
    "t=0.2": {"num_beams": 1, "do_sample": True, "temperature": 0.2, "top_p": 0.9},
    "t=0.4": {"num_beams": 1, "do_sample": True, "temperature": 0.4, "top_p": 0.9},
    "t=0.6": {"num_beams": 1, "do_sample": True, "temperature": 0.6, "top_p": 0.9},
}

if DOC_CACHE_PATH.exists():
    logger.info(f"Loading cached doc candidates from {DOC_CACHE_PATH}")
    with open(DOC_CACHE_PATH, "rb") as f:
        doc_cand = pickle.load(f)
    logger.info(f"Doc candidates loaded: {list(doc_cand.keys())}")
else:
    logger.info("Generating doc-level candidates...")
    doc_cand = {}

    # seed for reproducibility of sampling
    torch.manual_seed(42)

    for label, gen_cfg in GEN_CONFIGS.items():
        logger.info(f"  Generating {label} for {n_doc} docs...")
        doc_cand[label] = generate_batch(
            doc_inputs, desc=f"doc {label}",
            num_beams=gen_cfg.get("num_beams", 1),
            temperature=gen_cfg.get("temperature"),
            do_sample=gen_cfg.get("do_sample", False),
            top_p=gen_cfg.get("top_p", 0.9),
        )
        # Log sample
        for i in range(min(2, n_doc)):
            logger.info(f"    [{i}] {doc_cand[label][i][:100]}")

    with open(DOC_CACHE_PATH, "wb") as f:
        pickle.dump(doc_cand, f)
    logger.info(f"Doc candidates cached to {DOC_CACHE_PATH}")

# ============================================================
# DOC-CV: Back-translate candidates
# ============================================================
if DOC_BT_CACHE_PATH.exists():
    logger.info(f"Loading cached doc back-translations from {DOC_BT_CACHE_PATH}")
    with open(DOC_BT_CACHE_PATH, "rb") as f:
        doc_bt = pickle.load(f)
    logger.info(f"Doc BTs loaded: {list(doc_bt.keys())}")
else:
    logger.info("Back-translating doc candidates...")
    doc_bt = {}
    for label in doc_cand:
        logger.info(f"  Back-translating {label} ({len(doc_cand[label])} docs)...")
        doc_bt[label] = back_translate_batch(doc_cand[label], desc=f"doc BT {label}")
        for i in range(min(2, n_doc)):
            logger.info(f"    [{i}] eng: {doc_cand[label][i][:80]}")
            logger.info(f"         bt:  {doc_bt[label][i][:80]}")
            logger.info(f"         src: {doc_inputs_raw[i][:80]}")

    with open(DOC_BT_CACHE_PATH, "wb") as f:
        pickle.dump(doc_bt, f)
    logger.info(f"Doc BTs cached to {DOC_BT_CACHE_PATH}")

# ============================================================
# DOC-CV: Evaluate 6 configs
# ============================================================
logger.info("")
logger.info("=== DOC-CV Results ===")
doc_results = {}

for cfg in CONFIGS:
    if cfg["type"] == "single":
        preds = doc_cand[cfg["label"]]
        m = calc_metrics(preds, doc_refs)
    else:
        preds = apply_rerank(doc_cand, doc_bt, doc_inputs_raw,
                             cfg["combo"], cfg["method"])
        m = calc_metrics(preds, doc_refs)
    doc_results[cfg["name"]] = m
    logger.info(f"  {cfg['name']:<40} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

# ============================================================
# Combined summary
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== COMBINED RESULTS ===")
logger.info(f"{'config':<45} {'sent-geo':>10} {'sent-rep':>10} {'doc-geo':>10} {'doc-rep':>10}")
logger.info("-" * 90)

for cfg in CONFIGS:
    name = cfg["name"]
    s = sent_results[name]
    d = doc_results[name]
    logger.info(f"{name:<45} {s['geo']:>10.2f} {s['rep']:>9.1f}% {d['geo']:>10.2f} {d['rep']:>9.1f}%")

# ============================================================
# Save
# ============================================================
output = {
    "fold": FOLD,
    "sent_cv": sent_results,
    "doc_cv": doc_results,
}
out_path = RESULTS_DIR / f"eval_full_rerank_fold{FOLD}.json"
with open(str(out_path), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

logger.info(f"\nResults saved to {out_path}")
logger.info("Done.")
