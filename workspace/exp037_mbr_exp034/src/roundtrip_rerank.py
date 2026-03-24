"""
exp037: Round-trip reranking
- 候補の英語テキストを逆翻訳 (English → Akkadian) し、元のアッカド語ソースとの類似度で候補を選択
- 選択手法:
  1. Direct: 逆翻訳 vs ソースの直接スコアリング (chrF++, BLEU, geo, Jaccard, weighted)
  2. MBR on back-translations: 逆翻訳同士のconsensusスコアリング
  3. Clustering: TF-IDF cosine空間でのmedoid / source proximity
  4. Hybrid: direct + MBR の混合

Usage: python roundtrip_rerank.py [--fold N] [--batch_size 4]
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sacrebleu

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
# Data preparation (GroupKFold) — sent-CV
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

prefix_fwd = "translate Akkadian to English: "
prefix_rev = "translate English to Akkadian: "

# Val sent-CV inputs/refs
sent_inputs_raw = []  # raw Akkadian transliteration (for round-trip comparison)
sent_inputs = []      # with prefix (for matching with cached candidates)
sent_refs = []        # English references
for idx, row in val_data.iterrows():
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

n = len(sent_inputs)
logger.info(f"sent-CV: {n} sents")

# ============================================================
# Load cached candidates
# ============================================================
if not CAND_CACHE_PATH.exists():
    logger.error(f"Candidate cache not found: {CAND_CACHE_PATH}")
    logger.error("Run weighted_mbr_ablation.py first to generate candidates.")
    sys.exit(1)

with open(CAND_CACHE_PATH, "rb") as f:
    temp_preds = pickle.load(f)
logger.info(f"Loaded candidates: {list(temp_preds.keys())}")

# ============================================================
# Model & inference helpers for back-translation
# ============================================================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


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


def back_translate_batch(texts, model, batch_size):
    """Back-translate English texts to Akkadian using reverse prefix."""
    rev_inputs = [prefix_rev + t for t in texts]
    ds = DynamicPaddingDataset(rev_inputs, tokenizer, MAX_LENGTH)
    lengths = [ds.items[i]["input_ids"].size(0) for i in range(len(ds))]
    sorted_indices = sorted(range(len(ds)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(ds, sorted_indices)
    loader = DataLoader(sorted_ds, batch_size=batch_size, shuffle=False, collate_fn=dynamic_collate_fn)

    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="back-translate", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(input_ids=ids, attention_mask=mask,
                                 max_length=MAX_LENGTH, num_beams=1)
            sorted_preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])

    preds = [""] * len(texts)
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    return preds


# ============================================================
# Generate or load back-translations for all candidates
# ============================================================
if BT_CACHE_PATH.exists():
    logger.info(f"Loading cached back-translations from {BT_CACHE_PATH}")
    with open(BT_CACHE_PATH, "rb") as f:
        bt_preds = pickle.load(f)
    logger.info(f"Loaded back-translations for: {list(bt_preds.keys())}")
else:
    logger.info("Generating back-translations for all candidates...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    logger.info(f"Model loaded from {MODEL_PATH}")

    bt_preds = {}
    for label, preds in temp_preds.items():
        logger.info(f"Back-translating {label} ({len(preds)} texts)...")
        bt_preds[label] = back_translate_batch(preds, model, BATCH_SIZE)
        # Log a few examples
        for i in range(min(3, n)):
            logger.info(f"  [{i}] eng: {preds[i][:80]}")
            logger.info(f"       bt:  {bt_preds[label][i][:80]}")
            logger.info(f"       src: {sent_inputs_raw[i][:80]}")

    # Save cache
    with open(BT_CACHE_PATH, "wb") as f:
        pickle.dump(bt_preds, f)
    logger.info(f"Back-translations cached to {BT_CACHE_PATH}")

    del model
    torch.cuda.empty_cache()

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

# --- 1. Direct round-trip scoring (back-translation vs source) ---

def rt_pick_chrf(candidates, back_translations, source):
    """Pick candidate whose back-translation has highest chrF++ vs source."""
    cands = list(range(len(candidates)))
    if len(cands) <= 1:
        return 0
    scores = [score_chrf(back_translations[i], source) for i in cands]
    return int(np.argmax(scores))


def rt_pick_bleu(candidates, back_translations, source):
    scores = [score_bleu(back_translations[i], source) for i in range(len(candidates))]
    return int(np.argmax(scores))


def rt_pick_geo(candidates, back_translations, source):
    scores = []
    for i in range(len(candidates)):
        c = score_chrf(back_translations[i], source)
        b = score_bleu(back_translations[i], source)
        scores.append(math.sqrt(c * b) if c > 0 and b > 0 else 0.0)
    return int(np.argmax(scores))


def rt_pick_jaccard(candidates, back_translations, source):
    scores = [score_jaccard(back_translations[i], source) for i in range(len(candidates))]
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


# --- 2. MBR on back-translations (consensus among back-translations) ---

def rt_mbr_chrf(candidates, back_translations, source):
    """MBR consensus: pairwise chrF++ among back-translations."""
    nc = len(candidates)
    if nc <= 1:
        return 0
    scores = []
    for i in range(nc):
        s = sum(score_chrf(back_translations[i], back_translations[j])
                for j in range(nc) if j != i)
        scores.append(s / (nc - 1))
    return int(np.argmax(scores))


def rt_mbr_weighted(candidates, back_translations, source):
    """MBR consensus with weighted scoring on back-translations."""
    nc = len(candidates)
    if nc <= 1:
        return 0
    scores = []
    for i in range(nc):
        total = 0.0
        for j in range(nc):
            if j == i:
                continue
            total += (0.55 * score_chrf(back_translations[i], back_translations[j])
                      + 0.25 * score_bleu(back_translations[i], back_translations[j])
                      + 0.20 * score_jaccard(back_translations[i], back_translations[j]))
        scores.append(total / (nc - 1))
    return int(np.argmax(scores))


# --- 3. Hybrid: direct + MBR ---

def rt_hybrid_chrf(candidates, back_translations, source, alpha=0.5):
    """Hybrid: alpha * direct_score + (1-alpha) * mbr_consensus_score."""
    nc = len(candidates)
    if nc <= 1:
        return 0
    direct_scores = np.array([score_chrf(back_translations[i], source) for i in range(nc)])
    mbr_scores = np.zeros(nc)
    for i in range(nc):
        s = sum(score_chrf(back_translations[i], back_translations[j])
                for j in range(nc) if j != i)
        mbr_scores[i] = s / (nc - 1)
    # Normalize each to [0, 1]
    if direct_scores.max() > direct_scores.min():
        d_norm = (direct_scores - direct_scores.min()) / (direct_scores.max() - direct_scores.min())
    else:
        d_norm = np.ones(nc)
    if mbr_scores.max() > mbr_scores.min():
        m_norm = (mbr_scores - mbr_scores.min()) / (mbr_scores.max() - mbr_scores.min())
    else:
        m_norm = np.ones(nc)
    combined = alpha * d_norm + (1 - alpha) * m_norm
    return int(np.argmax(combined))


# --- 4. Clustering-based ---

def rt_cluster_cosine(candidates, back_translations, source, tfidf_vec=None):
    """Pick candidate whose back-translation is closest to source in TF-IDF cosine space."""
    nc = len(candidates)
    if nc <= 1:
        return 0
    all_texts = back_translations + [source]
    if tfidf_vec is not None:
        vecs = tfidf_vec.transform(all_texts)
    else:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), max_features=10000)
        vecs = vec.fit_transform(all_texts)
    source_vec = vecs[-1]
    bt_vecs = vecs[:-1]
    sims = cosine_similarity(bt_vecs, source_vec).flatten()
    return int(np.argmax(sims))


def rt_cluster_medoid(candidates, back_translations, source):
    """Pick candidate whose back-translation has highest avg cosine sim to all others (medoid)."""
    nc = len(candidates)
    if nc <= 1:
        return 0
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), max_features=10000)
    vecs = vec.fit_transform(back_translations)
    sim_matrix = cosine_similarity(vecs)
    avg_sims = []
    for i in range(nc):
        avg_sims.append(sum(sim_matrix[i][j] for j in range(nc) if j != i) / (nc - 1))
    return int(np.argmax(avg_sims))


# --- 5. Standard MBR on English side (baseline) ---

def eng_mbr_chrf(candidates, back_translations, source):
    """Standard chrF++ MBR on English candidates (baseline for comparison)."""
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


# ============================================================
# Build global TF-IDF for clustering methods
# ============================================================
logger.info("Building global TF-IDF for clustering methods...")
all_bt_texts = []
for label in temp_preds:
    all_bt_texts.extend(bt_preds[label])
all_bt_texts.extend(sent_inputs_raw)
global_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), max_features=50000)
global_tfidf.fit(all_bt_texts)
logger.info(f"Global TF-IDF vocabulary: {len(global_tfidf.vocabulary_)} features")

# ============================================================
# Evaluate: all methods × candidate combinations
# ============================================================
rerank_methods = {
    "eng_MBR_chrf": eng_mbr_chrf,            # baseline: standard MBR on English
    "rt_chrf": rt_pick_chrf,                  # direct: chrF++ of BT vs source
    "rt_bleu": rt_pick_bleu,                  # direct: BLEU of BT vs source
    "rt_geo": rt_pick_geo,                    # direct: geo of BT vs source
    "rt_jaccard": rt_pick_jaccard,            # direct: Jaccard of BT vs source
    "rt_weighted": rt_pick_weighted,          # direct: weighted combo
    "rt_mbr_chrf": rt_mbr_chrf,              # MBR consensus on BT side
    "rt_mbr_weighted": rt_mbr_weighted,       # weighted MBR on BT side
    "rt_hybrid_chrf": rt_hybrid_chrf,         # hybrid: direct + MBR
    "rt_cosine": lambda c, bt, s: rt_cluster_cosine(c, bt, s, tfidf_vec=global_tfidf),
    "rt_medoid": rt_cluster_medoid,           # clustering: medoid on BT side
}

key_combos = [
    # --- baselines (single) ---
    ["greedy"],
    ["t=0.4"],
    ["beam4_0"],
    # --- 2 candidates ---
    ["beam4_0", "t=0.4"],
    ["t=0.2", "t=0.4"],
    # --- 3 candidates ---
    ["beam4_0", "t=0.4", "t=0.6"],
    ["t=0.2", "t=0.4", "t=0.6"],
    # --- 4+ candidates ---
    ["t=0.2", "t=0.4", "t=0.8", "t=1.05"],
    ["beam4_0", "t=0.2", "t=0.4", "t=0.6", "t=0.8"],
]

logger.info("")
logger.info("=" * 60)
logger.info("=== Round-trip Reranking Evaluation ===")
logger.info("")

# Log back-translation quality stats first
logger.info("--- Back-translation quality (BT vs source, mean chrF++) ---")
for label in ["greedy", "t=0.4", "beam4_0", "t=0.2", "t=0.6", "t=0.8", "t=1.05"]:
    if label not in bt_preds:
        continue
    bt_scores = [score_chrf(bt_preds[label][i], sent_inputs_raw[i]) for i in range(n)]
    mean_score = np.mean(bt_scores)
    logger.info(f"  {label:<12} mean BT chrF++ vs source: {mean_score:.4f}")
logger.info("")

results = {}

for combo_labels in key_combos:
    combo_name = "+".join(combo_labels)
    n_cands = len(combo_labels)
    logger.info(f"--- {combo_name} ({n_cands} candidates) ---")

    if n_cands == 1:
        preds = temp_preds[combo_labels[0]]
        m = calc_metrics(preds, sent_refs)
        key = f"{combo_name} (single)"
        results[key] = m
        logger.info(f"  {'single':<20} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")
        logger.info("")
        continue

    for method_name, method_fn in rerank_methods.items():
        selected_preds = []
        for sample_idx in range(n):
            cands = [temp_preds[label][sample_idx] for label in combo_labels]
            bts = [bt_preds[label][sample_idx] for label in combo_labels]
            src = sent_inputs_raw[sample_idx]

            # Deduplicate candidates (preserve order)
            seen = {}
            unique_cands = []
            unique_bts = []
            for c, bt in zip(cands, bts):
                if c not in seen:
                    seen[c] = True
                    unique_cands.append(c)
                    unique_bts.append(bt)

            if len(unique_cands) <= 1:
                selected_preds.append(unique_cands[0] if unique_cands else "")
            else:
                best_idx = method_fn(unique_cands, unique_bts, src)
                selected_preds.append(unique_cands[best_idx])

        m = calc_metrics(selected_preds, sent_refs)
        key = f"{combo_name} | {method_name}"
        results[key] = m
        logger.info(f"  {method_name:<20} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

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

# ============================================================
# Summary sorted by rep% (ascending)
# ============================================================
logger.info("")
logger.info("=== Summary sorted by rep% (ascending) ===")
logger.info(f"{'config':<65} {'chrF++':>8} {'BLEU':>8} {'geo':>8} {'rep%':>8}")
logger.info("-" * 95)

sorted_by_rep = sorted(results.items(), key=lambda x: x[1]["rep"])
for name, m in sorted_by_rep:
    logger.info(f"{name:<65} {m['chrf']:>8.2f} {m['bleu']:>8.2f} {m['geo']:>8.2f} {m['rep']:>7.1f}%")

# ============================================================
# Save
# ============================================================
output = {"results": results, "fold": FOLD}
out_path = RESULTS_DIR / f"roundtrip_rerank_fold{FOLD}.json"
with open(str(out_path), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

logger.info(f"\nResults saved to {out_path}")
logger.info("Done.")
