"""
exp030: MBR 13候補ペナルティなし提出スクリプト
- beam4候補 + sample3×3temp(0.6/0.8/1.05) = 13候補
- chrF++ consensus MBR
- fold3 last_model
- バッチ化推論
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

# ============================================================
# Paths
# ============================================================
ON_KAGGLE = False

if ON_KAGGLE:
    MODEL_BASE = "/kaggle/input/akkadianmodels/pytorch/exp023_full_preprocessing/1"
    MODEL_PATH = f"{MODEL_BASE}/fold3/last_model"
    TEST_PATH = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    OUTPUT_PATH = "submission.csv"
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXP_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "workspace", "exp023_full_preprocessing", "results", "fold3", "last_model")
    TEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "test.csv")
    OUTPUT_PATH = os.path.join(EXP_DIR, "results", "submission.csv")

MAX_LENGTH = 512
PREFIX = "translate Akkadian to English: "

# MBR config
NUM_BEAM_CANDS = 4
NUM_SAMPLE_CANDS = 3
TEMPERATURES = [0.6, 0.8, 1.05]
BATCH_SIZE_BEAM = 2
BATCH_SIZE_SAMP = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# 前処理（exp023と同一）
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


def clean_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


# ============================================================
# 出力後処理（exp023と同一）
# ============================================================
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


def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i + n] == words[i + n:i + 2 * n]:
                return " ".join(words[:i + n])
    return text


# ============================================================
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# ============================================================
# Data
# ============================================================
test_df = pd.read_csv(TEST_PATH)
print(f"Test samples: {len(test_df)}")

texts = test_df["transliteration"].astype(str).apply(clean_transliteration).tolist()
texts = [PREFIX + t for t in texts]


# ============================================================
# Batched MBR inference
# ============================================================
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, max_length=max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def batched_generate(texts, batch_size, gen_kwargs, num_return):
    ds = InferenceDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_texts = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"generate(n_ret={num_return})", leave=False):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH,
                num_return_sequences=num_return,
                **gen_kwargs,
            )
            decoded = [d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)]
            all_texts.extend(decoded)

    n = len(texts)
    candidates_per_sample = []
    for i in range(n):
        candidates_per_sample.append(all_texts[i * num_return: (i + 1) * num_return])
    return candidates_per_sample


chrfpp = sacrebleu.metrics.CHRF(word_order=2)


def mbr_pick(candidates):
    cands = list(dict.fromkeys(candidates))
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
# Run MBR
# ============================================================
t0 = time.time()
total_cands = NUM_BEAM_CANDS + NUM_SAMPLE_CANDS * len(TEMPERATURES)
print(f"MBR: {NUM_BEAM_CANDS}beam + {NUM_SAMPLE_CANDS}samp×{len(TEMPERATURES)}temp = {total_cands} candidates")

# 1) Beam candidates
print("Generating beam candidates...")
beam_cands = batched_generate(
    texts, BATCH_SIZE_BEAM,
    gen_kwargs={"do_sample": False, "num_beams": max(8, NUM_BEAM_CANDS), "early_stopping": True},
    num_return=NUM_BEAM_CANDS,
)
print(f"Beam done: {time.time() - t0:.0f}s")

# 2) Sampling candidates per temperature
samp_cands_all = [[] for _ in range(len(texts))]
for temp in TEMPERATURES:
    print(f"Generating sampling candidates temp={temp}...")
    t_s = time.time()
    samp_cands = batched_generate(
        texts, BATCH_SIZE_SAMP,
        gen_kwargs={"do_sample": True, "num_beams": 1, "top_p": 0.9, "temperature": temp},
        num_return=NUM_SAMPLE_CANDS,
    )
    for i in range(len(texts)):
        samp_cands_all[i].extend(samp_cands[i])
    print(f"Sampling temp={temp} done: {time.time() - t_s:.0f}s")

# 3) MBR selection
print("Running MBR consensus selection...")
t_mbr = time.time()
all_predictions = []
for i in tqdm(range(len(texts)), desc="MBR select"):
    candidates = beam_cands[i] + samp_cands_all[i]
    all_predictions.append(mbr_pick(candidates))
print(f"MBR selection: {time.time() - t_mbr:.0f}s")
print(f"Total inference: {time.time() - t0:.0f}s")

# ============================================================
# Post-processing
# ============================================================
all_predictions = [repeat_cleanup(p) for p in all_predictions]
all_predictions = [clean_translation(p) for p in all_predictions]

# ============================================================
# Submission
# ============================================================
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": all_predictions,
})
submission["translation"] = submission["translation"].apply(
    lambda x: x if len(x) > 0 else "broken text"
)
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Submission saved to {OUTPUT_PATH}")
print(f"Shape: {submission.shape}")
print(f"Empty translations: {(submission['translation'] == 'broken text').sum()}")
