"""
exp035 MBR-FT Step1: 訓練データに対してMBR候補を生成し、最良翻訳を選択。
MBR選択された翻訳をターゲットとしてCSV出力する。

各訓練文に対し:
1. greedy (1候補)
2. beam=4 (4候補)
3. sampling temp=0.7, top_p=0.9 (8候補)
→ 計13候補からchrF++ベースのMBRで最良を選択
"""
import os
import re
import sys
import math
import json
import logging
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
from tqdm import tqdm

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
cmd_args = parser.parse_args()
FOLD = cmd_args.fold

# ============================================================
# Paths
# ============================================================
EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parent.parent
RESULTS_DIR = EXP_DIR / "results" / f"fold{FOLD}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"
BASE_MODEL_DIR = str(PROJECT_ROOT / "workspace" / "exp034_st_pretrain" / "results" / f"fold{FOLD}" / "last_model")

log_file = str(RESULTS_DIR / "generate_mbr.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    force=True,
)
logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================
MAX_LENGTH = 512
NUM_BEAM_CANDIDATES = 4
NUM_SAMPLE_CANDIDATES = 8
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.9
SEED = 42
PREFIX = "translate Akkadian to English: "

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}, Fold: {FOLD}")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Preprocessing (exp023-identical)
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

ROMAN_TO_INT = {
    "XII": "12", "XI": "11", "VIII": "8", "VII": "7",
    "VI": "6", "IV": "4", "IX": "9", "III": "3",
    "II": "2", "X": "10", "V": "5", "I": "1",
}

MONTH_NAMES_TRANSLATION = {
    r"B[eē]lat[\s-]ekall[ie]m": "1", r"[Šš]a[\s-]sarr[aā]tim": "2",
    r"[Kk]en[aā]tim": "3", r"[Šš]a[\s-]k[eē]n[aā]tim": "3",
    r"Ma[hḫ]h?ur[\s-]il[iī]": "4", r"Ab[\s-]?[šš]arr[aā]ni": "5",
    r"[Aa]b[sš]arrani": "5", r"[Hh]ubur": "6",
    r"[Ṣṣ]ip['\u2019]?um": "7", r"[Qq]arr[aā]['\u2019]?[aā]tum": "8",
    r"[Qq]arr[aā]tum": "8", r"[Kk]an[wm]arta": "9",
    r"[Tt]e['\u2019\u02BE]?in[aā]tum": "10",
    r"[Tt][eē]['\u2019\u02BE]?in[aā]tum": "10",
    r"[Kk]uzall?[iu]m?": "11", r"[Aa]llan[aā]tum": "12",
}

def clean_translation(text):
    if not isinstance(text, str) or not text.strip():
        return text
    text = re.sub(r'\bfem\.\s*', '', text)
    text = re.sub(r'\bsing\.\s*', '', text)
    text = re.sub(r'\bpl\.\s*', '', text)
    text = re.sub(r'\bplural\b\s*', '', text)
    text = text.replace('(?)', '')
    text = re.sub(r'<<\s*>>', '', text)
    text = re.sub(r'<\s+>', '', text)
    text = re.sub(r'(?<!\.)\.\.(?!\.)', '', text)
    text = re.sub(r'\bxx?\b', '', text)
    text = re.sub(r'\bPN\b', '<gap>', text)
    text = re.sub(r'\b-gold\b', 'pašallum gold', text)
    text = re.sub(r'\b-tax\b', 'šadduātum tax', text)
    text = re.sub(r'\b-textiles\b', 'kutānum textiles', text)
    text = re.sub(r'(\S+)\s*/\s*\S+', r'\1', text)
    text = re.sub(r'\(m\)', '{m}', text)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    for roman, integer in sorted(ROMAN_TO_INT.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf'\bmonth\s+{roman}(?=[\s,.:;!?\)]|$)', f'month {integer}', text)
    for pattern, number in MONTH_NAMES_TRANSLATION.items():
        text = re.sub(rf'\bmonth\s+{pattern}\b', f'month {number}', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_transliteration(text):
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


# ============================================================
# MBR utilities
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


def sentence_chrf(hyp, ref):
    """sentence-level chrF++ score (0-100)."""
    return sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score


def mbr_pick(cands):
    """MBR selection using chrF++ as utility function."""
    # Deduplicate
    seen = set()
    unique = []
    for c in cands:
        c = repeat_cleanup(c.strip())
        if c and c not in seen:
            unique.append(c)
            seen.add(c)

    if len(unique) == 0:
        return ""
    if len(unique) == 1:
        return unique[0]

    n = len(unique)
    scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = sentence_chrf(unique[i], unique[j])
            scores[i][j] = s
            scores[j][i] = s

    best_i, best_avg = 0, -1.0
    for i in range(n):
        avg = sum(scores[i]) / (n - 1)
        if avg > best_avg:
            best_avg, best_i = avg, i
    return unique[best_i]


# ============================================================
# Data loading
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
logger.info(f"Original Train Data: {len(train_df)} docs")

akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))

train_df["translation"] = train_df["translation"].astype(str).apply(clean_translation)
train_df["transliteration"] = train_df["transliteration"].astype(str).apply(clean_transliteration)


def simple_sentence_aligner(df):
    aligned_data = []
    for idx, row in df.iterrows():
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
logger.info(f"Expanded Train Data: {len(train_expanded)} sentences")

train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")
groups = train_expanded["akt_group"].values
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=groups))
train_idx, val_idx = splits[FOLD]

train_split = train_expanded.iloc[train_idx].copy().reset_index(drop=True)
val_split = train_expanded.iloc[val_idx].copy().reset_index(drop=True)
logger.info(f"Fold {FOLD}: train={len(train_split)}, val={len(val_split)}")


# ============================================================
# Model
# ============================================================
logger.info(f"Loading model from {BASE_MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR)
model.to(device)
model.eval()


# ============================================================
# Generation
# ============================================================
class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            truncation=True, padding=False, return_tensors=None,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def collate_simple(batch, tokenizer):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    attention_mask = []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
        attention_mask.append(b["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def generate_candidates(df, desc="Generating"):
    """Generate MBR candidates for each row. Returns list of (candidates_list, ref)."""
    texts = [PREFIX + str(t) for t in df["transliteration"].tolist()]
    refs = df["translation"].astype(str).tolist()

    ds = SimpleDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=lambda b: collate_simple(b, tokenizer),
                        num_workers=4, pin_memory=True)

    all_candidates = []  # list of list of strings
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            B = ids.shape[0]

            # 1. Greedy (1 candidate)
            greedy_out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=1,
            )
            greedy_txts = tokenizer.batch_decode(greedy_out, skip_special_tokens=True)

            # 2. Beam (4 candidates)
            beam_out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=NUM_BEAM_CANDIDATES,
                num_return_sequences=NUM_BEAM_CANDIDATES,
                early_stopping=True,
            )
            beam_txts = tokenizer.batch_decode(beam_out, skip_special_tokens=True)

            # 3. Sampling (8 candidates)
            torch.manual_seed(SEED + sample_idx)
            samp_out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, do_sample=True,
                temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P,
                num_return_sequences=NUM_SAMPLE_CANDIDATES,
            )
            samp_txts = tokenizer.batch_decode(samp_out, skip_special_tokens=True)

            for i in range(B):
                cands = [greedy_txts[i]]
                cands += beam_txts[i * NUM_BEAM_CANDIDATES:(i + 1) * NUM_BEAM_CANDIDATES]
                cands += samp_txts[i * NUM_SAMPLE_CANDIDATES:(i + 1) * NUM_SAMPLE_CANDIDATES]
                all_candidates.append(cands)

            sample_idx += B

    return all_candidates, refs


# ============================================================
# Main: generate and select MBR targets
# ============================================================
logger.info("Generating MBR candidates for training data...")
train_candidates, train_refs = generate_candidates(train_split, "train MBR candidates")

logger.info("Selecting MBR targets...")
mbr_targets = []
n_changed = 0
for i, (cands, ref) in enumerate(tqdm(zip(train_candidates, train_refs), total=len(train_refs), desc="MBR selection")):
    selected = mbr_pick(cands)
    greedy = repeat_cleanup(cands[0].strip())

    # Check if MBR selected something different from greedy
    if selected != greedy:
        n_changed += 1

    mbr_targets.append(selected)

logger.info(f"MBR selection done. Changed from greedy: {n_changed}/{len(mbr_targets)} ({100*n_changed/len(mbr_targets):.1f}%)")

# Save MBR targets
output_df = train_split[["transliteration", "translation", "oare_id"]].copy()
output_df["mbr_target"] = mbr_targets
output_df["original_target"] = train_refs

output_path = str(RESULTS_DIR / "mbr_targets.csv")
output_df.to_csv(output_path, index=False)
logger.info(f"MBR targets saved to {output_path}")

# Also compute stats: chrF++ improvement
greedy_chrfs = []
mbr_chrfs = []
for i, (cands, ref) in enumerate(zip(train_candidates, train_refs)):
    greedy = repeat_cleanup(cands[0].strip())
    greedy_chrfs.append(sentence_chrf(greedy, ref))
    mbr_chrfs.append(sentence_chrf(mbr_targets[i], ref))

logger.info(f"Average chrF++ — greedy: {np.mean(greedy_chrfs):.2f}, MBR: {np.mean(mbr_chrfs):.2f}")
logger.info(f"chrF++ improvement: {np.mean(mbr_chrfs) - np.mean(greedy_chrfs):.2f}")

# Save stats
stats = {
    "fold": FOLD,
    "n_train": len(train_split),
    "n_changed": n_changed,
    "pct_changed": round(100 * n_changed / len(mbr_targets), 1),
    "avg_greedy_chrf": round(float(np.mean(greedy_chrfs)), 2),
    "avg_mbr_chrf": round(float(np.mean(mbr_chrfs)), 2),
    "chrf_improvement": round(float(np.mean(mbr_chrfs) - np.mean(greedy_chrfs)), 2),
}
with open(str(RESULTS_DIR / "mbr_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

logger.info(f"Stats: {json.dumps(stats, indent=2)}")
logger.info("Done!")
