"""
exp037: kNN pseudo-reference rerank
- sentence_aligned.csvからtransliterationの近傍を検索
- 近傍の英訳を疑似参照として候補をスコアリング
- chrF++ MBRとの比較

Usage: python knn_rerank.py [--fold N] [--top_k 5]
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import sacrebleu

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=3)
parser.add_argument("--top_k", type=int, default=5)
args = parser.parse_args()
FOLD = args.fold
TOP_K = args.top_k

# ============================================================
# Config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CACHE_PATH = RESULTS_DIR / f"candidates_fold{FOLD}.pkl"
MAX_LENGTH = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

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
# Data preparation
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


# GroupKFold split
akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_expanded = simple_sentence_aligner(train_df, keep_oare_id=True)
train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=train_expanded["akt_group"].values))
train_idx, val_idx = splits[FOLD]
val_data = train_expanded.iloc[val_idx].copy()
train_fold_data = train_expanded.iloc[train_idx].copy()
logger.info(f"GroupKFold fold={FOLD}, train={len(train_fold_data)}, val={len(val_data)}")

# sentence_aligned.csv — train fold only for reference corpus
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

# Val sent-CV inputs/refs
sent_inputs_raw = []  # raw transliteration (before prefix, for kNN search)
sent_inputs = []      # with prefix (for matching with cached candidates)
sent_refs = []
for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    oare_id = translit_to_oare.get(translit)
    if oare_id and oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            for b in boundaries:
                if b['akk_segment'].strip() and b['eng_sentence'].strip():
                    sent_inputs_raw.append(preprocess_transliteration(b['akk_segment']))
                    sent_inputs.append(prefix + preprocess_transliteration(b['akk_segment']))
                    sent_refs.append(b['eng_sentence'])

n = len(sent_inputs)
logger.info(f"sent-CV: {n} sents")

# ============================================================
# Build kNN reference corpus from train fold sentence_aligned
# ============================================================
logger.info("Building kNN reference corpus from train fold sentences...")

# Get train fold oare_ids
train_fold_oare_ids = set(train_fold_data["oare_id"].unique())

ref_corpus_akk = []  # preprocessed transliterations
ref_corpus_eng = []  # corresponding translations

for oare_id in train_fold_oare_ids:
    if oare_id in alignment_dict:
        for entry in alignment_dict[oare_id]:
            akk = preprocess_transliteration(entry['akk_segment'])
            eng = entry['eng_sentence']
            if akk.strip() and eng.strip():
                ref_corpus_akk.append(akk)
                ref_corpus_eng.append(eng)

# Also add document-level train data (for coverage)
for _, row in train_fold_data.iterrows():
    translit = preprocess_transliteration(str(row['transliteration']))
    translation = str(row['translation'])
    if translit.strip() and translation.strip():
        ref_corpus_akk.append(translit)
        ref_corpus_eng.append(translation)

logger.info(f"Reference corpus: {len(ref_corpus_akk)} entries "
            f"({len([a for a in ref_corpus_akk if len(a) < 200])} short, "
            f"{len([a for a in ref_corpus_akk if len(a) >= 200])} long)")

# ============================================================
# TF-IDF char n-gram index for kNN search
# ============================================================
logger.info("Building TF-IDF index (char 3-6 grams)...")
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), max_features=50000)
ref_matrix = tfidf.fit_transform(ref_corpus_akk)
query_matrix = tfidf.transform(sent_inputs_raw)

logger.info(f"TF-IDF matrix: query={query_matrix.shape}, ref={ref_matrix.shape}")

# Find top-K neighbors for each query
logger.info(f"Finding top-{TOP_K} neighbors...")
# Batch cosine similarity to avoid memory issues
BATCH = 100
all_neighbor_indices = []
all_neighbor_sims = []

for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    sims = cosine_similarity(query_matrix[start:end], ref_matrix)
    for i in range(end - start):
        top_indices = np.argsort(sims[i])[::-1][:TOP_K]
        all_neighbor_indices.append(top_indices)
        all_neighbor_sims.append(sims[i][top_indices])

# Log neighbor quality
avg_top1_sim = np.mean([s[0] for s in all_neighbor_sims])
avg_top5_sim = np.mean([s.mean() for s in all_neighbor_sims])
logger.info(f"Avg top-1 similarity: {avg_top1_sim:.4f}, avg top-{TOP_K} mean: {avg_top5_sim:.4f}")

# ============================================================
# Load cached candidates
# ============================================================
if not CACHE_PATH.exists():
    logger.error(f"Candidate cache not found: {CACHE_PATH}")
    logger.error("Run weighted_mbr_ablation.py first to generate candidates.")
    sys.exit(1)

with open(CACHE_PATH, "rb") as f:
    temp_preds = pickle.load(f)
logger.info(f"Loaded candidates: {list(temp_preds.keys())}")

# ============================================================
# Rerank methods
# ============================================================
chrfpp_metric = sacrebleu.metrics.CHRF(word_order=2)
bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)


def rerank_knn(candidates, neighbor_translations, method="chrf"):
    """Score candidates against kNN pseudo-references, pick best."""
    cands = list(dict.fromkeys(candidates))
    if len(cands) <= 1:
        return cands[0] if cands else ""
    if not neighbor_translations:
        return cands[0]

    scores = []
    for cand in cands:
        if method == "chrf":
            s = np.mean([float(chrfpp_metric.sentence_score(cand, [ref]).score)
                         for ref in neighbor_translations])
        elif method == "geo":
            chrf_scores = [float(chrfpp_metric.sentence_score(cand, [ref]).score)
                          for ref in neighbor_translations]
            bleu_scores = [float(bleu_metric.sentence_score(cand, [ref]).score)
                          for ref in neighbor_translations]
            avg_chrf = np.mean(chrf_scores)
            avg_bleu = np.mean(bleu_scores)
            s = math.sqrt(avg_chrf * avg_bleu) if avg_chrf > 0 and avg_bleu > 0 else 0.0
        elif method == "chrf_weighted":
            # Weight by similarity
            sims = neighbor_translations  # actually (ref, sim) tuples
            total_w = sum(sim for _, sim in sims)
            if total_w == 0:
                return cands[0]
            s = sum(sim * float(chrfpp_metric.sentence_score(cand, [ref]).score)
                    for ref, sim in sims) / total_w
        else:
            raise ValueError(f"Unknown method: {method}")
        scores.append(s)
    return cands[int(np.argmax(scores))]


def mbr_pick_chrf(candidates):
    """Standard chrF++ MBR consensus"""
    cands = list(dict.fromkeys(candidates))
    nc = len(cands)
    if nc <= 1:
        return cands[0] if cands else ""
    scores = []
    for i in range(nc):
        s = sum(float(chrfpp_metric.sentence_score(cands[i], [cands[j]]).score)
                for j in range(nc) if j != i)
        scores.append(s / (nc - 1))
    return cands[int(np.argmax(scores))]


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


# ============================================================
# Evaluate: kNN rerank vs MBR for various candidate combos
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== kNN Rerank vs MBR Comparison ===")

key_combos = [
    ["greedy"],
    ["t=0.4"],
    ["beam4_0"],
    ["beam4_0", "t=0.4"],
    ["t=0.2", "t=0.4"],
    ["beam4_0", "t=0.4", "t=0.6"],
    ["t=0.2", "t=0.4", "t=0.6"],
    ["t=0.2", "t=0.4", "t=0.8", "t=1.05"],
    ["beam4_0", "t=0.2", "t=0.4", "t=0.6", "t=0.8"],
]

results = {}

for combo_labels in key_combos:
    combo_name = "+".join(combo_labels)
    n_cands = len(combo_labels)

    if n_cands == 1:
        preds = temp_preds[combo_labels[0]]
        m = calc_metrics(preds, sent_refs)
        key = f"{combo_name} (single)"
        results[key] = m
        logger.info(f"{key:<65} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")
        continue

    # --- MBR (chrF++ consensus) ---
    mbr_preds = []
    for sample_idx in range(n):
        cands = [temp_preds[label][sample_idx] for label in combo_labels]
        mbr_preds.append(mbr_pick_chrf(cands))
    m_mbr = calc_metrics(mbr_preds, sent_refs)
    key_mbr = f"{combo_name} | MBR_chrf"
    results[key_mbr] = m_mbr

    # --- kNN rerank (chrF++ against pseudo-refs) ---
    knn_preds_chrf = []
    for sample_idx in range(n):
        cands = [temp_preds[label][sample_idx] for label in combo_labels]
        neighbor_refs = [ref_corpus_eng[ni] for ni in all_neighbor_indices[sample_idx]]
        knn_preds_chrf.append(rerank_knn(cands, neighbor_refs, method="chrf"))
    m_knn_chrf = calc_metrics(knn_preds_chrf, sent_refs)
    key_knn_chrf = f"{combo_name} | kNN_chrf(k={TOP_K})"
    results[key_knn_chrf] = m_knn_chrf

    # --- kNN rerank (geo_mean against pseudo-refs) ---
    knn_preds_geo = []
    for sample_idx in range(n):
        cands = [temp_preds[label][sample_idx] for label in combo_labels]
        neighbor_refs = [ref_corpus_eng[ni] for ni in all_neighbor_indices[sample_idx]]
        knn_preds_geo.append(rerank_knn(cands, neighbor_refs, method="geo"))
    m_knn_geo = calc_metrics(knn_preds_geo, sent_refs)
    key_knn_geo = f"{combo_name} | kNN_geo(k={TOP_K})"
    results[key_knn_geo] = m_knn_geo

    # --- kNN rerank (similarity-weighted chrF++) ---
    knn_preds_w = []
    for sample_idx in range(n):
        cands = [temp_preds[label][sample_idx] for label in combo_labels]
        neighbor_refs_w = [(ref_corpus_eng[ni], all_neighbor_sims[sample_idx][j])
                           for j, ni in enumerate(all_neighbor_indices[sample_idx])]
        knn_preds_w.append(rerank_knn(cands, neighbor_refs_w, method="chrf_weighted"))
    m_knn_w = calc_metrics(knn_preds_w, sent_refs)
    key_knn_w = f"{combo_name} | kNN_wchrf(k={TOP_K})"
    results[key_knn_w] = m_knn_w

    logger.info(f"{key_mbr:<65} chrF++={m_mbr['chrf']:>6.2f}  BLEU={m_mbr['bleu']:>6.2f}  geo={m_mbr['geo']:>6.2f}  rep={m_mbr['rep']:>5.1f}%")
    logger.info(f"{key_knn_chrf:<65} chrF++={m_knn_chrf['chrf']:>6.2f}  BLEU={m_knn_chrf['bleu']:>6.2f}  geo={m_knn_chrf['geo']:>6.2f}  rep={m_knn_chrf['rep']:>5.1f}%")
    logger.info(f"{key_knn_geo:<65} chrF++={m_knn_geo['chrf']:>6.2f}  BLEU={m_knn_geo['bleu']:>6.2f}  geo={m_knn_geo['geo']:>6.2f}  rep={m_knn_geo['rep']:>5.1f}%")
    logger.info(f"{key_knn_w:<65} chrF++={m_knn_w['chrf']:>6.2f}  BLEU={m_knn_w['bleu']:>6.2f}  geo={m_knn_w['geo']:>6.2f}  rep={m_knn_w['rep']:>5.1f}%")
    logger.info("")

# ============================================================
# Summary sorted by geo
# ============================================================
logger.info("=" * 60)
logger.info("=== Summary sorted by geo ===")
logger.info(f"{'config':<65} {'chrF++':>8} {'BLEU':>8} {'geo':>8} {'rep%':>8}")
logger.info("-" * 95)

sorted_results = sorted(results.items(), key=lambda x: -x[1]["geo"])
for name, m in sorted_results:
    logger.info(f"{name:<65} {m['chrf']:>8.2f} {m['bleu']:>8.2f} {m['geo']:>8.2f} {m['rep']:>7.1f}%")

# Save
output = {"results": results, "fold": FOLD, "top_k": TOP_K,
          "avg_top1_sim": float(avg_top1_sim), "avg_topk_sim": float(avg_top5_sim)}
out_path = RESULTS_DIR / f"knn_rerank_fold{FOLD}.json"
with open(str(out_path), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
logger.info(f"\nResults saved to {out_path}")
logger.info("Done.")
