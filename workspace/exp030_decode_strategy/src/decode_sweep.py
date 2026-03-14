"""
デコード戦略グリッドサーチ: sent-CVのみで高速評価
exp023 fold3 last_modelを使い、各種generate()パラメータとMBRの効果を測定
"""
import os
import re
import sys
import math
import logging
import time
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
# Config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = str(PROJECT_ROOT / "workspace" / "exp023_full_preprocessing" / "results" / "fold3" / "last_model")
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"
MAX_LENGTH = 512
FOLD = 3

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

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def clean_transliteration(text):
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text

# ============================================================
# Data preparation
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
train_df["transliteration"] = train_df["transliteration"].astype(str).apply(clean_transliteration)

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
akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=train_expanded["akt_group"].values))
_, val_idx = splits[FOLD]
val_data = train_expanded.iloc[val_idx].copy()
logger.info(f"Fold {FOLD}: val={len(val_data)} samples")

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

logger.info(f"sent-CV: {len(sent_inputs)} sents")

# ============================================================
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info("Model loaded")

# ============================================================
# Utilities
# ============================================================
def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return " ".join(words[:i+n])
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
    return chrf.score, bleu.score, geo, rep_rate

chrfpp = sacrebleu.metrics.CHRF(word_order=2)

def mbr_pick(candidates):
    """MBR: chrF++のconsensusで最良候補を選択"""
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

# ============================================================
# Inference functions
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

def run_beam(inputs, num_beams=4, length_penalty=1.0, repetition_penalty=1.0):
    ds = InferenceDataset(inputs, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=num_beams,
                early_stopping=True,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )
            preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    return preds

def run_mbr(inputs, num_beam_cands=4, num_sample_cands=2, temperatures=[0.7],
            length_penalty=1.3, repetition_penalty=1.2):
    """MBR: beam候補 + sampling候補からconsensus選択"""
    ds = InferenceDataset(inputs, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=1, shuffle=False)  # batch=1 for multi-candidate
    preds = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            candidates = []

            # Beam candidates
            beam_out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH,
                do_sample=False,
                num_beams=max(8, num_beam_cands),
                num_return_sequences=num_beam_cands,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                early_stopping=True,
            )
            candidates.extend([d.strip() for d in tokenizer.batch_decode(beam_out, skip_special_tokens=True)])

            # Sampling candidates (multi-temperature)
            for temp in temperatures:
                samp_out = model.generate(
                    input_ids=ids, attention_mask=mask,
                    max_length=MAX_LENGTH,
                    do_sample=True, num_beams=1,
                    top_p=0.9, temperature=temp,
                    num_return_sequences=num_sample_cands,
                    repetition_penalty=repetition_penalty,
                )
                candidates.extend([d.strip() for d in tokenizer.batch_decode(samp_out, skip_special_tokens=True)])

            preds.append(mbr_pick(candidates))
    return preds

# ============================================================
# Sweep configs
# ============================================================
configs = [
    # === exp012ベスト再現: beam4 + sample3×3temp = 13候補 ===
    {"name": "mbr_b4s3_13cand", "method": "mbr", "num_beam_cands": 4, "num_sample_cands": 3,
     "temperatures": [0.6, 0.8, 1.05], "length_penalty": 1.0, "repetition_penalty": 1.2},

    # === exp012ベスト + lp1.3 ===
    {"name": "mbr_b4s3_13cand_lp1.3", "method": "mbr", "num_beam_cands": 4, "num_sample_cands": 3,
     "temperatures": [0.6, 0.8, 1.05], "length_penalty": 1.3, "repetition_penalty": 1.2},

    # === 上位NB設定: beam8候補 + sample3×3temp = 17候補 ===
    {"name": "mbr_b8s3_17cand", "method": "mbr", "num_beam_cands": 8, "num_sample_cands": 3,
     "temperatures": [0.6, 0.8, 1.05], "length_penalty": 1.3, "repetition_penalty": 1.2},

    # === ペナルティなし13候補（比較用） ===
    {"name": "mbr_b4s3_13cand_nopen", "method": "mbr", "num_beam_cands": 4, "num_sample_cands": 3,
     "temperatures": [0.6, 0.8, 1.05], "length_penalty": 1.0, "repetition_penalty": 1.0},
]

# ============================================================
# Run sweep
# ============================================================
results = []
for cfg in configs:
    name = cfg["name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {name}")
    t0 = time.time()

    if cfg["method"] == "beam":
        preds = run_beam(sent_inputs, cfg["num_beams"], cfg["length_penalty"], cfg["repetition_penalty"])
    else:
        preds = run_mbr(sent_inputs, cfg["num_beam_cands"], cfg["num_sample_cands"],
                        cfg["temperatures"], cfg["length_penalty"], cfg["repetition_penalty"])

    elapsed = time.time() - t0
    chrf_score, bleu_score, geo, rep = calc_metrics(preds, sent_refs)
    results.append({
        "name": name, "chrF++": chrf_score, "BLEU": bleu_score,
        "geo": geo, "rep%": rep, "time_s": elapsed,
    })
    logger.info(f"  {name}: chrF++={chrf_score:.2f}, BLEU={bleu_score:.2f}, geo={geo:.2f}, rep={rep:.1f}%, time={elapsed:.0f}s")

# ============================================================
# Summary
# ============================================================
logger.info(f"\n{'='*60}")
logger.info("=== Decode Strategy Sweep Results (sent-CV, fold3) ===")
logger.info(f"{'='*60}")
logger.info(f"{'name':25s} {'chrF++':>8s} {'BLEU':>8s} {'geo':>8s} {'rep%':>6s} {'time':>6s}")
logger.info("-" * 65)
for r in results:
    logger.info(f"{r['name']:25s} {r['chrF++']:8.2f} {r['BLEU']:8.2f} {r['geo']:8.2f} {r['rep%']:5.1f}% {r['time_s']:5.0f}s")

# Save (append mode to preserve previous results)
out_path = Path(__file__).parent.parent / "results" / "decode_sweep_results2.txt"
with open(str(out_path), "w") as f:
    f.write(f"{'name':25s} {'chrF++':>8s} {'BLEU':>8s} {'geo':>8s} {'rep%':>6s} {'time':>6s}\n")
    f.write("-" * 65 + "\n")
    for r in results:
        f.write(f"{r['name']:25s} {r['chrF++']:8.2f} {r['BLEU']:8.2f} {r['geo']:8.2f} {r['rep%']:5.1f}% {r['time_s']:5.0f}s\n")
logger.info(f"Results saved to {out_path}")
