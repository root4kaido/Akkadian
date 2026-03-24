"""
exp038: Back-Translation — generated_english.csv を s1_exp007(byt5-large) でアッカド語に逆翻訳
- 方向: English → Akkadian
- モデル: s1_exp007_large_lr1e4/results/fold3/last_model (byt5-large, sent-CV chrF++ 49.83)
- 入力: datasets/processed/generated_english.csv (2,120件)
- 出力: dataset/backtranslated.csv (transliteration, translation ペア)
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

INPUT_CSV = PROJECT_ROOT / "datasets" / "processed" / "generated_english.csv"
MODEL_PATH = str(PROJECT_ROOT / "workspace" / "s1_exp007_large_lr1e4" / "results" / "fold3" / "last_model")

# ============================================================
# Logging
# ============================================================
log_file = str(EXP_DIR / "results" / "generate_backtranslation.log")
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
PREFIX = "translate English to Akkadian: "
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ============================================================
# 前処理（exp023と同一 — 英語テキストにclean_translationを適用）
# ============================================================
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002

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


def clean_translation(text: str) -> str:
    """英語テキストの前処理（exp023と同一）"""
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
    words = text.split()
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
logger.info(f"Loading generated English from {INPUT_CSV}")
eng_df = pd.read_csv(str(INPUT_CSV))
logger.info(f"Generated English: {len(eng_df)} samples")
logger.info(f"Columns: {eng_df.columns.tolist()}")

# 英語テキストに前処理適用
eng_df["cleaned_translation"] = eng_df["generated_translation"].astype(str).apply(clean_translation)

# 空文字列・短すぎるテキストを除外
eng_df = eng_df[eng_df["cleaned_translation"].str.len() >= 10].reset_index(drop=True)
logger.info(f"After filtering short texts: {len(eng_df)}")

texts = eng_df["cleaned_translation"].tolist()

# ============================================================
# Generate back-translations (English → Akkadian)
# ============================================================
logger.info(f"Loading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

dataset = InferenceDataset(texts, tokenizer, MAX_LENGTH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

predictions = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Back-translating English → Akkadian"):
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

logger.info(f"{len(predictions)} back-translations generated")

del model
torch.cuda.empty_cache()

# ============================================================
# 品質フィルタ: 繰り返しのある出力は除外
# ============================================================
bt_df = pd.DataFrame({
    "transliteration": predictions,                # モデル生成のアッカド語
    "translation": eng_df["cleaned_translation"].tolist(),  # 元の英語
})

n_before = len(bt_df)
has_rep = bt_df["transliteration"].apply(has_repetition)
bt_df = bt_df[~has_rep].reset_index(drop=True)
n_filtered = n_before - len(bt_df)
logger.info(f"Repetition filter: {n_filtered}/{n_before} removed ({n_filtered/n_before*100:.1f}%)")
logger.info(f"Remaining: {len(bt_df)} samples")

# ============================================================
# Save
# ============================================================
output_path = DATASET_DIR / "backtranslated.csv"
bt_df.to_csv(str(output_path), index=False)
logger.info(f"Back-translated data saved to {output_path}: {len(bt_df)} samples")

# サンプル表示
logger.info("--- Sample back-translations ---")
for i in range(min(10, len(bt_df))):
    row = bt_df.iloc[i]
    logger.info(f"  EN: {row['translation'][:100]}...")
    logger.info(f"  AK: {row['transliteration'][:100]}")
    logger.info("")

# 統計
akk_lengths = [len(t) for t in predictions]
eng_lengths = [len(t) for t in texts]
logger.info(f"Akkadian output length: mean={np.mean(akk_lengths):.0f}, median={np.median(akk_lengths):.0f}, max={max(akk_lengths)}")
logger.info(f"English input length:   mean={np.mean(eng_lengths):.0f}, median={np.median(eng_lengths):.0f}, max={max(eng_lengths)}")

logger.info("Done.")
