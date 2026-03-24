"""
exp037: 2-model ensemble + round-trip weighted rerank — sent-CV & doc-CV evaluation
- Model A: exp034_st_pretrain (ByT5-base) beam4
- Model B: s1_exp007_large_lr1e4 (ByT5-large) beam4
- 各モデルで順翻訳beam4 + 逆翻訳beam4 → rt_weighted / rt_chrf 等で選択
- モデルは1つずつロード→推論→保存→解放（メモリ効率）

Usage: python eval_ensemble_rt.py [--fold N] [--batch_size 4]
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
MODEL_A_PATH = str(PROJECT_ROOT / "workspace" / "exp034_st_pretrain" / "results" / f"fold{FOLD}" / "last_model")
MODEL_B_PATH = str(PROJECT_ROOT / "workspace" / "s1_exp007_large_lr1e4" / "results" / f"fold{FOLD}" / "last_model")
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_LENGTH = 512
NUM_BEAMS_FWD = 4
NUM_BEAMS_BT = 4

CACHE_DIR = RESULTS_DIR / "ensemble_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_df["akt_group"] = train_df["oare_id"].map(oare_to_group).fillna("None")

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_df, groups=train_df["akt_group"].values))
_, val_idx = splits[FOLD]
val_doc_data = train_df.iloc[val_idx].copy()
val_oare_ids = set(val_doc_data["oare_id"].unique())
logger.info(f"Fold {FOLD}: val_docs={len(val_doc_data)}")

# sentence_aligned for sent-CV
sent_aligned = pd.read_csv(str(SENTENCE_ALIGNED_PATH))
oare_to_row = {row['oare_id']: row for _, row in train_df.iterrows()}
alignment_dict = {}
for oare_id, group in sent_aligned.groupby('oare_id'):
    group = group.sort_values('sent_idx')
    alignment_dict[oare_id] = [
        {'akk_segment': str(r['akk_segment']), 'eng_sentence': str(r['eng_sentence'])}
        for _, r in group.iterrows()
    ]

prefix_fwd = "translate Akkadian to English: "
prefix_rev = "translate English to Akkadian: "

# ============================================================
# Sent-CV inputs/refs (short docs ≤6 sents)
# ============================================================
sent_inputs_raw = []
sent_inputs = []
sent_refs = []
for _, row in val_doc_data.iterrows():
    oare_id = row['oare_id']
    if oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            for b in boundaries:
                if b['akk_segment'].strip() and b['eng_sentence'].strip():
                    raw = preprocess_transliteration(b['akk_segment'])
                    sent_inputs_raw.append(raw)
                    sent_inputs.append(prefix_fwd + raw)
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
        raw = preprocess_transliteration(translit)
        doc_inputs_raw.append(raw)
        doc_inputs.append(prefix_fwd + raw)
        doc_refs.append(translation)

n_doc = len(doc_inputs)
logger.info(f"doc-CV: {n_doc} docs")


# ============================================================
# Inference helpers (tokenizer passed as param for multi-model)
# ============================================================
class DynamicPaddingDataset(TorchDataset):
    def __init__(self, texts, tok, max_length):
        self.items = []
        for t in texts:
            enc = tok(t, max_length=max_length, truncation=True, return_tensors="pt")
            self.items.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_collate_fn(pad_id):
    def dynamic_collate_fn(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids, attention_mask = [], []
        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask)}
    return dynamic_collate_fn


def generate_sorted(texts, tok, mdl, desc, num_beams=4, batch_size=BATCH_SIZE):
    """Generate with length-sorted batching, return in original order."""
    ds = DynamicPaddingDataset(texts, tok, MAX_LENGTH)
    lengths = [ds.items[i]["input_ids"].size(0) for i in range(len(ds))]
    sorted_indices = sorted(range(len(ds)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(ds, sorted_indices)
    collate_fn = make_collate_fn(tok.pad_token_id or 0)
    loader = DataLoader(sorted_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = mdl.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=num_beams, early_stopping=True,
            )
            sorted_preds.extend([d.strip() for d in tok.batch_decode(out, skip_special_tokens=True)])

    preds = [""] * len(texts)
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    return preds


def run_model_inference(model_path, model_name, batch_size=BATCH_SIZE):
    """Load model, generate fwd+bt for both sent-CV and doc-CV, save, release."""
    cache_path = CACHE_DIR / f"{model_name}_fold{FOLD}.pkl"

    if cache_path.exists():
        logger.info(f"Loading cached results for {model_name} from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"Loading model: {model_name} from {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    mdl.eval()
    logger.info(f"  Model loaded ({sum(p.numel() for p in mdl.parameters()) / 1e6:.0f}M params)")

    results = {}

    # --- sent-CV: forward beam4 ---
    logger.info(f"  [{model_name}] sent-CV forward beam4 ({n_sent} sents)...")
    results["sent_fwd"] = generate_sorted(sent_inputs, tok, mdl,
                                           desc=f"{model_name} sent fwd",
                                           num_beams=NUM_BEAMS_FWD, batch_size=batch_size)

    # --- sent-CV: back-translate beam4 ---
    logger.info(f"  [{model_name}] sent-CV back-translate beam4...")
    sent_rev_inputs = [prefix_rev + p for p in results["sent_fwd"]]
    results["sent_bt"] = generate_sorted(sent_rev_inputs, tok, mdl,
                                          desc=f"{model_name} sent bt",
                                          num_beams=NUM_BEAMS_BT, batch_size=batch_size)

    # --- doc-CV: forward beam4 ---
    logger.info(f"  [{model_name}] doc-CV forward beam4 ({n_doc} docs)...")
    results["doc_fwd"] = generate_sorted(doc_inputs, tok, mdl,
                                          desc=f"{model_name} doc fwd",
                                          num_beams=NUM_BEAMS_FWD, batch_size=batch_size)

    # --- doc-CV: back-translate beam4 ---
    logger.info(f"  [{model_name}] doc-CV back-translate beam4...")
    doc_rev_inputs = [prefix_rev + p for p in results["doc_fwd"]]
    results["doc_bt"] = generate_sorted(doc_rev_inputs, tok, mdl,
                                         desc=f"{model_name} doc bt",
                                         num_beams=NUM_BEAMS_BT, batch_size=batch_size)

    # Log samples
    for i in range(min(2, n_sent)):
        logger.info(f"  sent[{i}] fwd: {results['sent_fwd'][i][:80]}")
        logger.info(f"           bt:  {results['sent_bt'][i][:80]}")
        logger.info(f"           src: {sent_inputs_raw[i][:80]}")

    # Cache
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"  [{model_name}] Results cached to {cache_path}")

    # Release
    del mdl, tok
    torch.cuda.empty_cache()
    logger.info(f"  [{model_name}] Released from GPU")

    return results


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
# Reranking methods
# ============================================================
def rt_pick_chrf(candidates, back_translations, source):
    scores = [score_chrf(back_translations[i], source) for i in range(len(candidates))]
    return int(np.argmax(scores))


def rt_pick_bleu(candidates, back_translations, source):
    scores = [score_bleu(back_translations[i], source) for i in range(len(candidates))]
    return int(np.argmax(scores))


def rt_pick_weighted(candidates, back_translations, source,
                     w_chrf=0.55, w_bleu=0.25, w_jaccard=0.20, w_len=0.10):
    scores = []
    for i in range(len(candidates)):
        bt = back_translations[i]
        s = (w_chrf * score_chrf(bt, source)
             + w_bleu * score_bleu(bt, source)
             + w_jaccard * score_jaccard(bt, source)
             + w_len * score_length_bonus(bt, source))
        scores.append(s)
    return int(np.argmax(scores))


def eng_mbr_chrf(candidates, back_translations, source):
    nc = len(candidates)
    if nc <= 1:
        return 0
    scores = []
    for i in range(nc):
        s = sum(score_chrf(candidates[i], candidates[j])
                for j in range(nc) if j != i)
        scores.append(s / (nc - 1))
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


def apply_rerank(cands_list, bts_list, inputs_raw, method_fn):
    """Apply reranking. cands_list/bts_list: list of (model_a_pred, model_b_pred) per sample."""
    selected = []
    pick_stats = {"A": 0, "B": 0}
    for i in range(len(inputs_raw)):
        cands = list(cands_list[i])
        bts = list(bts_list[i])
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
            pick_stats["A"] += 1
        else:
            best_idx = method_fn(unique_cands, unique_bts, src)
            selected.append(unique_cands[best_idx])
            if best_idx == 0:
                pick_stats["A"] += 1
            else:
                pick_stats["B"] += 1
    return selected, pick_stats


# ============================================================
# Run inference for both models
# ============================================================
logger.info("=" * 60)
logger.info("=== Step 1: Model A (exp034 ByT5-base) ===")
logger.info("=" * 60)
model_a = run_model_inference(MODEL_A_PATH, "model_a", batch_size=BATCH_SIZE)

logger.info("")
logger.info("=" * 60)
logger.info("=== Step 2: Model B (s1_exp007 ByT5-large) ===")
logger.info("=" * 60)
model_b = run_model_inference(MODEL_B_PATH, "model_b", batch_size=BATCH_SIZE)

# ============================================================
# Reranking methods to evaluate
# ============================================================
METHODS = {
    "rt_weighted": rt_pick_weighted,
    "rt_chrf": rt_pick_chrf,
    "rt_bleu": rt_pick_bleu,
    "eng_MBR_chrf": eng_mbr_chrf,
}

# ============================================================
# Evaluate sent-CV
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== SENT-CV Results ===")
logger.info("=" * 60)

sent_results = {}

# Singles
for label, data in [("model_A (exp034 base)", model_a), ("model_B (s1_exp007 large)", model_b)]:
    m = calc_metrics(data["sent_fwd"], sent_refs)
    sent_results[label] = m
    logger.info(f"  {label:<45} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

# Ensemble reranking
sent_cands = list(zip(model_a["sent_fwd"], model_b["sent_fwd"]))
sent_bts = list(zip(model_a["sent_bt"], model_b["sent_bt"]))

for method_name, method_fn in METHODS.items():
    preds, pick = apply_rerank(sent_cands, sent_bts, sent_inputs_raw, method_fn)
    m = calc_metrics(preds, sent_refs)
    label = f"A+B | {method_name}"
    sent_results[label] = m
    logger.info(f"  {label:<45} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%  (A={pick['A']}, B={pick['B']})")

# ============================================================
# Evaluate doc-CV
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== DOC-CV Results ===")
logger.info("=" * 60)

doc_results = {}

# Singles
for label, data in [("model_A (exp034 base)", model_a), ("model_B (s1_exp007 large)", model_b)]:
    m = calc_metrics(data["doc_fwd"], doc_refs)
    doc_results[label] = m
    logger.info(f"  {label:<45} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

# Ensemble reranking
doc_cands = list(zip(model_a["doc_fwd"], model_b["doc_fwd"]))
doc_bts = list(zip(model_a["doc_bt"], model_b["doc_bt"]))

for method_name, method_fn in METHODS.items():
    preds, pick = apply_rerank(doc_cands, doc_bts, doc_inputs_raw, method_fn)
    m = calc_metrics(preds, doc_refs)
    label = f"A+B | {method_name}"
    doc_results[label] = m
    logger.info(f"  {label:<45} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%  (A={pick['A']}, B={pick['B']})")

# ============================================================
# Combined summary
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== COMBINED RESULTS ===")
logger.info(f"{'config':<50} {'sent-geo':>10} {'sent-rep':>10} {'doc-geo':>10} {'doc-rep':>10}")
logger.info("-" * 95)

for label in sent_results:
    s = sent_results[label]
    d = doc_results[label]
    logger.info(f"{label:<50} {s['geo']:>10.2f} {s['rep']:>9.1f}% {d['geo']:>10.2f} {d['rep']:>9.1f}%")

# ============================================================
# Save
# ============================================================
output = {
    "fold": FOLD,
    "num_beams_fwd": NUM_BEAMS_FWD,
    "num_beams_bt": NUM_BEAMS_BT,
    "model_a": MODEL_A_PATH,
    "model_b": MODEL_B_PATH,
    "sent_cv": sent_results,
    "doc_cv": doc_results,
}
out_path = RESULTS_DIR / f"ensemble_rt_fold{FOLD}.json"
with open(str(out_path), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

logger.info(f"\nResults saved to {out_path}")
logger.info("Done.")
