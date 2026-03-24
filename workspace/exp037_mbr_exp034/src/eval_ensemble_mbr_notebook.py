"""
exp037: Notebook-style ensemble MBR evaluation (beam4x4 + sampling2) x 2 models
- Model A: exp034_st_pretrain (ByT5-base)
- Model B: s1_exp007_large_lr1e4 (ByT5-large)
- 各モデルから beam4 top-4候補 + stochastic sampling 2候補 = 6候補/model
- 候補プールをマージして chrF++ MBR で最終翻訳を選択
- 比較: 1-best MBR (2cands), beam4x4 MBR (8cands), full pool MBR (12cands)

Usage: python eval_ensemble_mbr_notebook.py [--fold N] [--batch_size 4]
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

# Generation config (matching notebook)
NUM_BEAMS = 8
NUM_BEAM_RETURN = 4
NUM_SAMPLE_RETURN = 2
SAMPLE_TOP_P = 0.92
SAMPLE_TEMPERATURE = 0.75
SAMPLE_REP_PENALTY = 1.2
LENGTH_PENALTY = 1.3

CACHE_DIR = RESULTS_DIR / "ensemble_mbr_cache"
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
alignment_dict = {}
for oare_id, group in sent_aligned.groupby('oare_id'):
    group = group.sort_values('sent_idx')
    alignment_dict[oare_id] = [
        {'akk_segment': str(r['akk_segment']), 'eng_sentence': str(r['eng_sentence'])}
        for _, r in group.iterrows()
    ]

prefix_fwd = "translate Akkadian to English: "

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
# Inference helpers
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


def generate_multi_sorted(texts, tok, mdl, desc,
                          num_beams=4, num_return_sequences=1,
                          do_sample=False, temperature=1.0,
                          top_p=1.0, repetition_penalty=1.0,
                          length_penalty=1.0,
                          batch_size=BATCH_SIZE):
    """Generate with length-sorted batching, returning list of lists.

    Returns: list[list[str]] — outer list has len(texts) items,
             each inner list has num_return_sequences candidates.
    """
    ds = DynamicPaddingDataset(texts, tok, MAX_LENGTH)
    lengths = [ds.items[i]["input_ids"].size(0) for i in range(len(ds))]
    sorted_indices = sorted(range(len(ds)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(ds, sorted_indices)
    collate_fn = make_collate_fn(tok.pad_token_id or 0)

    # Reduce effective batch size for multi-candidate generation
    effective_bs = max(1, batch_size // num_return_sequences)
    loader = DataLoader(sorted_ds, batch_size=effective_bs, shuffle=False, collate_fn=collate_fn)

    sorted_multi_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            gen_kwargs = dict(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH,
                num_return_sequences=num_return_sequences,
                use_cache=True,
            )

            if do_sample:
                gen_kwargs.update(
                    do_sample=True,
                    num_beams=1,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )
            else:
                gen_kwargs.update(
                    do_sample=False,
                    num_beams=max(num_beams, num_return_sequences),
                    early_stopping=True,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                )

            out = mdl.generate(**gen_kwargs)
            decoded = [d.strip() for d in tok.batch_decode(out, skip_special_tokens=True)]

            B = ids.shape[0]
            for b in range(B):
                cands = decoded[b * num_return_sequences: (b + 1) * num_return_sequences]
                sorted_multi_preds.append(cands)

    # Unsort to original order
    multi_preds = [[] for _ in range(len(texts))]
    for new_idx, cands in enumerate(sorted_multi_preds):
        multi_preds[idx_map[new_idx]] = cands
    return multi_preds


# ============================================================
# Model inference with caching
# ============================================================
def run_model_multi_inference(model_path, model_name, batch_size=BATCH_SIZE):
    """Load model, generate beam4x4 + sampling2 for sent/doc, cache, release."""
    cache_path = CACHE_DIR / f"{model_name}_multi_fold{FOLD}.pkl"

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
    torch.manual_seed(42)

    # --- sent-CV: beam top-4 ---
    logger.info(f"  [{model_name}] sent-CV beam8 top-4 ({n_sent} sents)...")
    results["sent_beam4"] = generate_multi_sorted(
        sent_inputs, tok, mdl, desc=f"{model_name} sent beam4x4",
        num_beams=NUM_BEAMS, num_return_sequences=NUM_BEAM_RETURN,
        length_penalty=LENGTH_PENALTY, repetition_penalty=SAMPLE_REP_PENALTY,
        batch_size=batch_size)

    # --- sent-CV: stochastic sampling x2 ---
    torch.manual_seed(42)
    logger.info(f"  [{model_name}] sent-CV sampling x2...")
    results["sent_sample"] = generate_multi_sorted(
        sent_inputs, tok, mdl, desc=f"{model_name} sent sample",
        do_sample=True, num_return_sequences=NUM_SAMPLE_RETURN,
        temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P,
        repetition_penalty=SAMPLE_REP_PENALTY,
        batch_size=batch_size)

    # --- doc-CV: beam top-4 ---
    logger.info(f"  [{model_name}] doc-CV beam8 top-4 ({n_doc} docs)...")
    results["doc_beam4"] = generate_multi_sorted(
        doc_inputs, tok, mdl, desc=f"{model_name} doc beam4x4",
        num_beams=NUM_BEAMS, num_return_sequences=NUM_BEAM_RETURN,
        length_penalty=LENGTH_PENALTY, repetition_penalty=SAMPLE_REP_PENALTY,
        batch_size=batch_size)

    # --- doc-CV: stochastic sampling x2 ---
    torch.manual_seed(42)
    logger.info(f"  [{model_name}] doc-CV sampling x2...")
    results["doc_sample"] = generate_multi_sorted(
        doc_inputs, tok, mdl, desc=f"{model_name} doc sample",
        do_sample=True, num_return_sequences=NUM_SAMPLE_RETURN,
        temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P,
        repetition_penalty=SAMPLE_REP_PENALTY,
        batch_size=batch_size)

    # Log samples
    for i in range(min(2, n_sent)):
        logger.info(f"  sent[{i}] beam4[0]: {results['sent_beam4'][i][0][:80]}")
        logger.info(f"           beam4[3]: {results['sent_beam4'][i][-1][:80]}")
        logger.info(f"           sample[0]: {results['sent_sample'][i][0][:80]}")

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
# Proper noun lexicon (from notebook)
# ============================================================
LEXICON_PATH = PROJECT_ROOT / "datasets" / "raw" / "OA_Lexicon_eBL.csv"


def build_proper_noun_lexicon(csv_path):
    """Extract PN/GN/DN/RN mappings from OA_Lexicon_eBL.csv."""
    df = pd.read_csv(csv_path, encoding='utf-8')
    target_types = ['PN', 'GN', 'DN', 'RN']
    entity_df = df[df['type'].isin(target_types)].copy()
    lexicon_map = {}
    for _, row in entity_df.iterrows():
        form = str(row['form']).strip()
        norm = str(row['norm']).strip()
        if pd.isna(form) or pd.isna(norm) or form == 'nan' or norm == 'nan':
            continue
        clean_form = re.sub(r'[\[\]\(\)\?\!]', '', form).lower()
        if clean_form:
            lexicon_map[clean_form] = norm
    logger.info(f"Loaded {len(lexicon_map)} proper nouns into lexicon map.")
    return lexicon_map


PROPER_NOUN_MAP = build_proper_noun_lexicon(str(LEXICON_PATH))


# ============================================================
# chrF++ MBR selection
# ============================================================
chrfpp_metric = sacrebleu.metrics.CHRF(word_order=2)


def _dedup_candidates(candidates):
    seen = set()
    unique = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def mbr_select(candidates):
    """Pick candidate with highest average chrF++ against other candidates."""
    unique = _dedup_candidates(candidates)
    if len(unique) == 0:
        return ""
    if len(unique) == 1:
        return unique[0]

    nc = len(unique)
    scores = []
    for i in range(nc):
        s = sum(float(chrfpp_metric.sentence_score(unique[i], [unique[j]]).score)
                for j in range(nc) if j != i)
        scores.append(s / (nc - 1))
    return unique[int(np.argmax(scores))]


def _lexical_fidelity_score(source_text, candidate):
    """Score how well the candidate preserves proper nouns from the source."""
    if not source_text or not candidate:
        return 0.0
    clean_source = re.sub(r'[^\w\-\s]', '', source_text.lower())
    source_tokens = clean_source.split()
    expected_entities = []
    for token in source_tokens:
        if token in PROPER_NOUN_MAP:
            expected_entities.append(PROPER_NOUN_MAP[token].lower())
    if not expected_entities:
        return 100.0
    cand_lower = candidate.lower()
    match_count = sum(1 for entity in expected_entities if entity in cand_lower)
    return (match_count / len(expected_entities)) * 100.0


def mbr_select_with_fidelity(candidates, source_text, w_chrf=0.8, w_fidelity=0.2):
    """MBR with lexical fidelity scoring (notebook style)."""
    unique = _dedup_candidates(candidates)
    if len(unique) == 0:
        return ""
    if len(unique) == 1:
        return unique[0]

    nc = len(unique)
    best_i, best_s = 0, -1e9
    for i in range(nc):
        consensus = sum(float(chrfpp_metric.sentence_score(unique[i], [unique[j]]).score)
                        for j in range(nc) if j != i) / (nc - 1)
        fidelity = _lexical_fidelity_score(source_text, unique[i])
        total = w_chrf * consensus + w_fidelity * fidelity
        if total > best_s:
            best_s, best_i = total, i
    return unique[best_i]


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
# Run inference for both models
# ============================================================
logger.info("=" * 60)
logger.info("=== Step 1: Model A (exp034 ByT5-base) ===")
logger.info("=" * 60)
model_a = run_model_multi_inference(MODEL_A_PATH, "model_a", batch_size=BATCH_SIZE)

logger.info("")
logger.info("=" * 60)
logger.info("=== Step 2: Model B (s1_exp007 ByT5-large) ===")
logger.info("=" * 60)
model_b = run_model_multi_inference(MODEL_B_PATH, "model_b", batch_size=BATCH_SIZE)

# ============================================================
# Build candidate pools and evaluate
# ============================================================
CONFIGS = {
    "1-best MBR (2 cands)": lambda a, b, key: [
        [a[f"{key}_beam4"][i][0], b[f"{key}_beam4"][i][0]]
        for i in range(len(a[f"{key}_beam4"]))
    ],
    "beam4x4 MBR (8 cands)": lambda a, b, key: [
        a[f"{key}_beam4"][i] + b[f"{key}_beam4"][i]
        for i in range(len(a[f"{key}_beam4"]))
    ],
    "full pool MBR (12 cands)": lambda a, b, key: [
        a[f"{key}_beam4"][i] + a[f"{key}_sample"][i]
        + b[f"{key}_beam4"][i] + b[f"{key}_sample"][i]
        for i in range(len(a[f"{key}_beam4"]))
    ],
}

# ============================================================
# Evaluate sent-CV
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== SENT-CV Results ===")
logger.info("=" * 60)

sent_results = {}

# Singles (beam4 1-best)
for label, data in [("model_A (exp034 base)", model_a), ("model_B (s1_exp007 large)", model_b)]:
    preds = [d[0] for d in data["sent_beam4"]]
    m = calc_metrics(preds, sent_refs)
    sent_results[label] = m
    logger.info(f"  {label:<50} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

# MBR configurations (chrF++ only)
for config_name, pool_fn in CONFIGS.items():
    pools = pool_fn(model_a, model_b, "sent")
    preds = [mbr_select(pool) for pool in tqdm(pools, desc=f"sent MBR {config_name}", leave=False)]
    m = calc_metrics(preds, sent_refs)
    label = f"A+B | {config_name}"
    sent_results[label] = m
    avg_pool = np.mean([len(set(p)) for p in pools])
    logger.info(f"  {label:<50} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%  (avg_pool={avg_pool:.1f})")

# MBR + lexical fidelity (notebook style: 80% chrF++ + 20% fidelity)
for config_name, pool_fn in CONFIGS.items():
    pools = pool_fn(model_a, model_b, "sent")
    preds = [mbr_select_with_fidelity(pool, src)
             for pool, src in tqdm(zip(pools, sent_inputs_raw), total=len(pools),
                                   desc=f"sent MBR+lex {config_name}", leave=False)]
    m = calc_metrics(preds, sent_refs)
    label = f"A+B | {config_name} +lex"
    sent_results[label] = m
    avg_pool = np.mean([len(set(p)) for p in pools])
    logger.info(f"  {label:<50} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%  (avg_pool={avg_pool:.1f})")

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
    preds = [d[0] for d in data["doc_beam4"]]
    m = calc_metrics(preds, doc_refs)
    doc_results[label] = m
    logger.info(f"  {label:<50} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

# MBR configurations (chrF++ only)
for config_name, pool_fn in CONFIGS.items():
    pools = pool_fn(model_a, model_b, "doc")
    preds = [mbr_select(pool) for pool in tqdm(pools, desc=f"doc MBR {config_name}", leave=False)]
    m = calc_metrics(preds, doc_refs)
    label = f"A+B | {config_name}"
    doc_results[label] = m
    avg_pool = np.mean([len(set(p)) for p in pools])
    logger.info(f"  {label:<50} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%  (avg_pool={avg_pool:.1f})")

# MBR + lexical fidelity (notebook style)
for config_name, pool_fn in CONFIGS.items():
    pools = pool_fn(model_a, model_b, "doc")
    preds = [mbr_select_with_fidelity(pool, src)
             for pool, src in tqdm(zip(pools, doc_inputs_raw), total=len(pools),
                                   desc=f"doc MBR+lex {config_name}", leave=False)]
    m = calc_metrics(preds, doc_refs)
    label = f"A+B | {config_name} +lex"
    doc_results[label] = m
    avg_pool = np.mean([len(set(p)) for p in pools])
    logger.info(f"  {label:<50} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%  (avg_pool={avg_pool:.1f})")

# ============================================================
# Combined summary
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("=== COMBINED RESULTS ===")
logger.info(f"{'config':<55} {'sent-geo':>10} {'sent-rep':>10} {'doc-geo':>10} {'doc-rep':>10}")
logger.info("-" * 100)

for label in sent_results:
    s = sent_results[label]
    d = doc_results[label]
    logger.info(f"{label:<55} {s['geo']:>10.2f} {s['rep']:>9.1f}% {d['geo']:>10.2f} {d['rep']:>9.1f}%")

# Phase 8 reference
logger.info("-" * 100)
logger.info("--- Phase 8 reference (eval_ensemble_rt.py) ---")
logger.info(f"{'A+B | rt_weighted (Phase 8)':<55} {'37.90':>10} {'12.1':>9}% {'26.32':>10} {'68.7':>9}%")
logger.info(f"{'A+B | eng_MBR_chrf (Phase 8)':<55} {'39.00':>10} {'13.9':>9}% {'26.47':>10} {'70.4':>9}%")

# ============================================================
# Save
# ============================================================
output = {
    "fold": FOLD,
    "models": {"A": MODEL_A_PATH, "B": MODEL_B_PATH},
    "generation_config": {
        "beam": {"num_beams": NUM_BEAMS, "num_return_sequences": NUM_BEAM_RETURN,
                 "length_penalty": LENGTH_PENALTY, "repetition_penalty": SAMPLE_REP_PENALTY},
        "sample": {"top_p": SAMPLE_TOP_P, "temperature": SAMPLE_TEMPERATURE,
                   "repetition_penalty": SAMPLE_REP_PENALTY, "num_return_sequences": NUM_SAMPLE_RETURN},
    },
    "sent_cv": sent_results,
    "doc_cv": doc_results,
}
out_path = RESULTS_DIR / f"ensemble_mbr_notebook_fold{FOLD}.json"
with open(str(out_path), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

logger.info(f"\nResults saved to {out_path}")
logger.info("Done.")
