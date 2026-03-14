"""
exp024_pn_gn_tagging: CV評価スクリプト
- exp023ベース + PN/GNタグ付加
- beam4, max_length=512, early_stopping=True
- sent-level評価 (入力200B截断 + 最初の文抽出)
"""
import os
import re
import sys
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parent.parent
RESULTS_DIR = EXP_DIR / "results"
MODEL_PATH = str(RESULTS_DIR / "best_model")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / "eval_cv.log")),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = 512
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


# ============================================================
# 前処理: train.pyと同一
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
    best_frac = None
    best_dist = float('inf')
    for target, symbol in FRACTION_TARGETS.items():
        dist = abs(frac_part - target)
        if dist < best_dist:
            best_dist = dist
            best_frac = symbol
    if best_dist <= APPROX_TOLERANCE:
        if int_part == 0:
            return best_frac
        else:
            return f"{int_part} {best_frac}"
    return dec_str

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

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


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


def clean_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


# ============================================================
# PN/GN tagging
# ============================================================
import json

def load_form_tag_dict(dict_path: str) -> dict:
    with open(dict_path) as f:
        return json.load(f)

def tag_transliteration(text: str, form_tag_dict: dict) -> str:
    tokens = text.split()
    tagged_tokens = []
    for token in tokens:
        tag = form_tag_dict.get(token)
        if tag:
            tagged_tokens.append(f"{token}[{tag}]")
        else:
            tagged_tokens.append(token)
    return " ".join(tagged_tokens)


# ============================================================
# Data: exp022と同一のsplitを再現 + 前処理適用 + PN/GNタグ
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))

train_df["translation"] = train_df["translation"].astype(str).apply(clean_translation)
train_df["transliteration"] = train_df["transliteration"].astype(str).apply(clean_transliteration)

# PN/GNタグ付加
FORM_TAG_DICT_PATH = str(EXP_DIR / "dataset" / "form_type_dict.json")
form_tag_dict = load_form_tag_dict(FORM_TAG_DICT_PATH)
logger.info(f"Loaded form_tag_dict: {len(form_tag_dict)} entries")
train_df["transliteration"] = train_df["transliteration"].astype(str).apply(
    lambda t: tag_transliteration(t, form_tag_dict)
)


def simple_sentence_aligner(df):
    aligned_data = []
    for idx, row in df.iterrows():
        src = str(row["transliteration"])
        tgt = str(row["translation"])
        tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split("\n") if s.strip()]
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({"transliteration": s, "translation": t})
        else:
            aligned_data.append({"transliteration": src, "translation": tgt})
    return pd.DataFrame(aligned_data)


train_expanded = simple_sentence_aligner(train_df)
dataset = Dataset.from_pandas(train_expanded)
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)

val_data = split_datasets["test"].to_pandas()
logger.info(f"Val samples (doc-level): {len(val_data)}")


# ============================================================
# Sent-level eval
# ============================================================
def extract_first_sentence(text: str) -> str:
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()


def truncate_akkadian_to_sentence(translit: str, max_bytes: int = 200) -> str:
    encoded = str(translit).encode('utf-8')
    if len(encoded) <= max_bytes:
        return str(translit)
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    last_space = truncated.rfind(' ')
    return truncated[:last_space].strip() if last_space > 0 else truncated.strip()


def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return " ".join(words[:i+n])
    return text


prefix = "translate Akkadian to English: "
sent_inputs, sent_refs = [], []
for _, row in val_data.iterrows():
    t = str(row["transliteration"])
    tr = str(row["translation"])
    eng = extract_first_sentence(tr)
    akk = truncate_akkadian_to_sentence(t)
    if eng.strip() and akk.strip():
        sent_inputs.append(prefix + akk)
        sent_refs.append(eng)

logger.info(f"Sent-level val samples: {len(sent_inputs)}")


# ============================================================
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info(f"Model loaded from {MODEL_PATH}")


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


# ============================================================
# Inference (beam4)
# ============================================================
dataset_eval = InferenceDataset(sent_inputs, tokenizer, MAX_LENGTH)
loader = DataLoader(dataset_eval, batch_size=4, shuffle=False)

preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc="CV eval (beam4)"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model.generate(
            input_ids=ids, attention_mask=mask,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
        )
        preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])

preds_clean = [repeat_cleanup(p) for p in preds]

# ============================================================
# 予測結果保存
# ============================================================
val_preds_df = pd.DataFrame({
    "input": sent_inputs,
    "reference": sent_refs,
    "prediction_raw": preds,
    "prediction_clean": preds_clean,
})
val_preds_path = os.path.join(RESULTS_DIR, "val_predictions.csv")
val_preds_df.to_csv(val_preds_path, index=False)
logger.info(f"Val predictions saved to {val_preds_path} ({len(val_preds_df)} rows)")


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


# ============================================================
# Metrics
# ============================================================
logger.info("=" * 60)
logger.info("=== Sent-level CV evaluation (beam4) ===")

for plabel, pred_list in [("raw", preds), ("clean", preds_clean)]:
    chrf = sacrebleu.corpus_chrf(pred_list, [sent_refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(pred_list, [sent_refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in pred_list) / len(pred_list)
    logger.info(
        f"  {plabel:5s}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, "
        f"geo={geo:.2f}, rep={rep_rate:.1f}%"
    )

logger.info("")
logger.info("=== Reference: exp022 CV = 47.09, exp016 CV = 45.78 ===")
