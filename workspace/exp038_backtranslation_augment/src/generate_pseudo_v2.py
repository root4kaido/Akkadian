"""
exp038: 擬似ラベル再生成 (v2)
- s1_exp007(byt5-large) fold3 last_model で published_texts を英訳
- train.csvとの重複を除外
- repeat_cleanupではなく、繰り返し検出→除外（品質フィルタ）
- 出力: dataset/pseudo_labels_v2.csv
"""
import os
import re
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ============================================================
# Paths
# ============================================================
EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parent.parent
DATASET_DIR = EXP_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

PUBLISHED_TEXTS_PATH = PROJECT_ROOT / "datasets" / "raw" / "published_texts.csv"
TRAIN_PATH = PROJECT_ROOT / "datasets" / "raw" / "train.csv"
MODEL_PATH = str(PROJECT_ROOT / "workspace" / "s1_exp007_large_lr1e4" / "results" / "fold3" / "last_model")

# ============================================================
# Logging
# ============================================================
log_file = str(EXP_DIR / "results" / "generate_pseudo_v2.log")
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
BATCH_SIZE = 16
NUM_BEAMS = 4
PREFIX = "translate Akkadian to English: "
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ============================================================
# 前処理（exp023と同一）
# ============================================================
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

ROMAN_TO_INT = {
    "XII": "12", "XI": "11", "VIII": "8", "VII": "7",
    "VI": "6", "IV": "4", "IX": "9", "III": "3",
    "II": "2", "X": "10", "V": "5", "I": "1",
}

MONTH_NAMES_TRANSLATION = {
    r"B[eē]lat[\s-]ekall[ie]m": "1",
    r"[Šš]a[\s-]sarr[aā]tim": "2",
    r"[Kk]en[aā]tim": "3",
    r"[Šš]a[\s-]k[eē]n[aā]tim": "3",
    r"Ma[hḫ]h?ur[\s-]il[iī]": "4",
    r"Ab[\s-]?[šš]arr[aā]ni": "5",
    r"[Aa]b[sš]arrani": "5",
    r"[Hh]ubur": "6",
    r"[Ṣṣ]ip['\u2019]?um": "7",
    r"[Qq]arr[aā]['\u2019]?[aā]tum": "8",
    r"[Qq]arr[aā]tum": "8",
    r"[Kk]an[wm]arta": "9",
    r"[Tt]e['\u2019\u02BE]?in[aā]tum": "10",
    r"[Tt][eē]['\u2019\u02BE]?in[aā]tum": "10",
    r"[Kk]uzall?[iu]m?": "11",
    r"[Aa]llan[aā]tum": "12",
}


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


def clean_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


def clean_translation(text: str) -> str:
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


def has_repetition(text):
    """繰り返しパターンを検出（品質不良として除外用）"""
    words = str(text).split()
    if len(words) < 6:
        return False
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return True
    return False


# ============================================================
# Dataset
# ============================================================
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = [PREFIX + t for t in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx], max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


# ============================================================
# Data loading
# ============================================================
logger.info(f"Loading published_texts from {PUBLISHED_TEXTS_PATH}")
pub_df = pd.read_csv(str(PUBLISHED_TEXTS_PATH))
logger.info(f"Published texts: {len(pub_df)}")

train_df = pd.read_csv(str(TRAIN_PATH))
train_oare_ids = set(train_df["oare_id"].tolist())
logger.info(f"Train oare_ids: {len(train_oare_ids)}")

# 重複除外
pub_df = pub_df[~pub_df["oare_id"].isin(train_oare_ids)].reset_index(drop=True)
logger.info(f"After excluding train overlap: {len(pub_df)}")

# 前処理
pub_df["transliteration"] = pub_df["transliteration"].astype(str).apply(clean_transliteration)

# 空文字列・短すぎるテキストを除外
pub_df = pub_df[pub_df["transliteration"].str.len() >= 10].reset_index(drop=True)
logger.info(f"After filtering short texts: {len(pub_df)}")

texts = pub_df["transliteration"].tolist()
oare_ids = pub_df["oare_id"].tolist()

# ============================================================
# Generate with s1_exp007 fold3 last_model
# ============================================================
logger.info(f"Loading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

dataset = InferenceDataset(texts, tokenizer, MAX_LENGTH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

predictions = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Generating pseudo labels (s1_exp007)"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend([d.strip() for d in decoded])

logger.info(f"{len(predictions)} predictions generated")

del model
torch.cuda.empty_cache()

# ============================================================
# 品質フィルタ: 繰り返しのある出力を除外
# ============================================================
pseudo_df = pd.DataFrame({
    "oare_id": oare_ids,
    "transliteration": texts,
    "translation": predictions,
})

# clean_translation適用
pseudo_df["translation"] = pseudo_df["translation"].apply(clean_translation)

n_before = len(pseudo_df)
has_rep = pseudo_df["translation"].apply(has_repetition)
pseudo_df = pseudo_df[~has_rep].reset_index(drop=True)
n_filtered = n_before - len(pseudo_df)
logger.info(f"Repetition filter: {n_filtered}/{n_before} removed ({n_filtered/n_before*100:.1f}%)")
logger.info(f"Remaining: {len(pseudo_df)} samples")

# ============================================================
# Save
# ============================================================
output_path = DATASET_DIR / "pseudo_labels_v2.csv"
pseudo_df.to_csv(str(output_path), index=False)
logger.info(f"Pseudo labels v2 saved to {output_path}: {len(pseudo_df)} samples")

# サンプル表示
logger.info("--- Sample pseudo labels ---")
for i in range(min(5, len(pseudo_df))):
    row = pseudo_df.iloc[i]
    logger.info(f"  AK: {row['transliteration'][:80]}...")
    logger.info(f"  EN: {row['translation'][:100]}")
    logger.info("")

logger.info("Done.")
