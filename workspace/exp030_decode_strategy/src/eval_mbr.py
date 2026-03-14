"""
exp030: バッチ化MBRデコーディング評価（sent-CV + doc-CV）
ベスト設定: beam4候補 + sample3×3temp(0.6/0.8/1.05) = 13候補, ペナルティなし

Usage: python eval_mbr.py [--fold N]
"""
import os
import re
import sys
import math
import json
import logging
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import GroupKFold
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
MODEL_PATH = str(PROJECT_ROOT / "workspace" / "exp023_full_preprocessing" / "results" / f"fold{FOLD}" / "last_model")
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MAX_LENGTH = 512

# MBR config (best from sweep)
NUM_BEAM_CANDS = 4
NUM_SAMPLE_CANDS = 3
TEMPERATURES = [0.6, 0.8, 1.05]
# total candidates = 4 + 3*3 = 13

BATCH_SIZE_BEAM = 2   # beam with num_return_sequences=4 → 8 outputs per batch
BATCH_SIZE_SAMP = 4   # sampling with num_return_sequences=3 → 12 outputs per batch

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
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _decimal_to_fraction(match):
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


def clean_transliteration(text):
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


# ============================================================
# Data preparation (GroupKFold)
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
train_df["transliteration"] = train_df["transliteration"].astype(str).apply(clean_transliteration)

akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))


def simple_sentence_aligner(df):
    aligned_data = []
    for _, row in df.iterrows():
        src = str(row["transliteration"])
        tgt = str(row["translation"])
        oare_id = row["oare_id"]
        tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split("\n") if s.strip()]
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({"transliteration": s, "translation": t, "oare_id": oare_id})
        else:
            aligned_data.append({"transliteration": src, "translation": tgt, "oare_id": oare_id})
    return pd.DataFrame(aligned_data)


train_expanded = simple_sentence_aligner(train_df)
train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=train_expanded["akt_group"].values))
_, val_idx = splits[FOLD]
val_data = train_expanded.iloc[val_idx].copy()
logger.info(f"Fold {FOLD}: val={len(val_data)} samples")
logger.info(f"Val groups: {val_data['akt_group'].value_counts().to_dict()}")

# Build sent-CV inputs
sent_aligned = pd.read_csv(str(SENTENCE_ALIGNED_PATH))
alignment_dict = {}
for oare_id, group in sent_aligned.groupby('oare_id'):
    group = group.sort_values('sent_idx')
    alignment_dict[oare_id] = [
        {'akk_segment': str(r['akk_segment']), 'eng_sentence': str(r['eng_sentence'])}
        for _, r in group.iterrows()
    ]

translit_to_oare = {str(row['transliteration']): row['oare_id'] for _, row in train_df.iterrows()}

prefix = "translate Akkadian to English: "

# --- sent-CV ---
sent_inputs = []
sent_refs = []
for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    oare_id = translit_to_oare.get(translit)
    if oare_id and oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            for b in boundaries:
                if b['akk_segment'].strip() and b['eng_sentence'].strip():
                    sent_inputs.append(prefix + clean_transliteration(b['akk_segment']))
                    sent_refs.append(b['eng_sentence'])

# --- doc-CV ---
doc_inputs = []
doc_refs = []
for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    translation = str(row['translation'])
    if translit.strip() and translation.strip():
        doc_inputs.append(prefix + clean_transliteration(translit))
        doc_refs.append(translation)

logger.info(f"sent-CV: {len(sent_inputs)} sents, doc-CV: {len(doc_inputs)} docs")

# ============================================================
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info("Model loaded")


# ============================================================
# Batched MBR inference
# ============================================================
class InferenceDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, max_length=max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def batched_generate(inputs, batch_size, gen_kwargs, num_return):
    """Generate with batching. Returns list of lists (per-sample candidates)."""
    ds = InferenceDataset(inputs, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_texts = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"generate(n_ret={num_return})", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH,
                num_return_sequences=num_return,
                **gen_kwargs,
            )
            decoded = [d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)]
            all_texts.extend(decoded)

    # Reshape: flat list → list of lists (num_return per sample)
    n = len(inputs)
    candidates_per_sample = []
    for i in range(n):
        candidates_per_sample.append(all_texts[i * num_return: (i + 1) * num_return])
    return candidates_per_sample


chrfpp = sacrebleu.metrics.CHRF(word_order=2)


def mbr_pick(candidates):
    """chrF++ consensus selection."""
    cands = list(dict.fromkeys(candidates))  # dedup keeping order
    cands = cands[:32]
    n = len(cands)
    if n <= 1:
        return cands[0] if cands else ""
    scores = []
    for i in range(n):
        s = sum(float(chrfpp.sentence_score(cands[i], [cands[j]]).score) for j in range(n) if j != i)
        scores.append(s / (n - 1))
    return cands[int(np.argmax(scores))]


def run_mbr(inputs, desc=""):
    """Batched MBR: generate all candidates first, then select."""
    logger.info(f"[MBR] {desc}: {len(inputs)} samples, {NUM_BEAM_CANDS}beam + {NUM_SAMPLE_CANDS}samp×{len(TEMPERATURES)}temp = {NUM_BEAM_CANDS + NUM_SAMPLE_CANDS * len(TEMPERATURES)} candidates")
    t0 = time.time()

    # 1) Beam candidates (batch)
    logger.info(f"[MBR] Generating beam candidates (batch={BATCH_SIZE_BEAM})...")
    beam_cands = batched_generate(
        inputs, BATCH_SIZE_BEAM,
        gen_kwargs={"do_sample": False, "num_beams": max(8, NUM_BEAM_CANDS), "early_stopping": True},
        num_return=NUM_BEAM_CANDS,
    )
    t_beam = time.time() - t0
    logger.info(f"[MBR] Beam done: {t_beam:.0f}s")

    # 2) Sampling candidates per temperature (batch)
    samp_cands_all = [[] for _ in range(len(inputs))]
    for temp in TEMPERATURES:
        logger.info(f"[MBR] Generating sampling candidates temp={temp} (batch={BATCH_SIZE_SAMP})...")
        t_s = time.time()
        samp_cands = batched_generate(
            inputs, BATCH_SIZE_SAMP,
            gen_kwargs={"do_sample": True, "num_beams": 1, "top_p": 0.9, "temperature": temp},
            num_return=NUM_SAMPLE_CANDS,
        )
        for i in range(len(inputs)):
            samp_cands_all[i].extend(samp_cands[i])
        logger.info(f"[MBR] Sampling temp={temp} done: {time.time() - t_s:.0f}s")

    # 3) MBR selection
    logger.info(f"[MBR] Running MBR consensus selection...")
    t_mbr = time.time()
    preds = []
    for i in range(len(inputs)):
        candidates = beam_cands[i] + samp_cands_all[i]
        preds.append(mbr_pick(candidates))
    t_sel = time.time() - t_mbr
    total = time.time() - t0
    logger.info(f"[MBR] MBR selection: {t_sel:.0f}s, total: {total:.0f}s")
    return preds, total


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
    return chrf.score, bleu.score, geo, rep_rate


# ============================================================
# Run
# ============================================================
sent_preds, sent_time = run_mbr(sent_inputs, desc="sent-CV")
doc_preds, doc_time = run_mbr(doc_inputs, desc="doc-CV")

logger.info("=" * 60)
logger.info(f"=== exp030 MBR 13cand nopen (fold{FOLD}) ===")
sc, sb, sg, sr = calc_metrics(sent_preds, sent_refs, f"sent-CV ({len(sent_inputs)} sents, {sent_time:.0f}s)")
dc, db, dg, dr = calc_metrics(doc_preds, doc_refs, f"doc-CV  ({len(doc_inputs)} docs, {doc_time:.0f}s)")

# Save
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
metrics = {
    "fold": FOLD,
    "config": f"beam{NUM_BEAM_CANDS}+samp{NUM_SAMPLE_CANDS}x{len(TEMPERATURES)}temp=13cand_nopen",
    "sent_cv": {"chrf": round(sc, 2), "bleu": round(sb, 2), "geo": round(sg, 2), "rep": round(sr, 1),
                "n": len(sent_inputs), "time_s": round(sent_time, 0)},
    "doc_cv": {"chrf": round(dc, 2), "bleu": round(db, 2), "geo": round(dg, 2), "rep": round(dr, 1),
               "n": len(doc_inputs), "time_s": round(doc_time, 0)},
}
with open(str(RESULTS_DIR / f"mbr_eval_fold{FOLD}.json"), "w") as f:
    json.dump(metrics, f, indent=2)
logger.info(f"Metrics saved to {RESULTS_DIR / f'mbr_eval_fold{FOLD}.json'}")
logger.info("Done.")
