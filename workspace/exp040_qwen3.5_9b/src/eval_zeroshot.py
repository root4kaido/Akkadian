"""
exp040: Qwen3.5-9B ゼロショット翻訳評価
- fold3 valデータに対してAkkadian→Englishのゼロショット翻訳
- sent-CV / doc-CV をbyt5ベースラインと比較
- Qwen3_5ForConditionalGeneration + AutoProcessor
- 必要: pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"
"""
import os
import re
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import sacrebleu

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=3, help="Fold index (0-4)")
parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B", help="Model name")
parser.add_argument("--quantize", type=str, default="none", choices=["none", "int4", "int8"])
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
SENT_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"

log_file = str(RESULTS_DIR / "eval_zeroshot.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    force=True,
)
logger = logging.getLogger(__name__)

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


# ============================================================
# Data loading — fold3 val (exp023と同一split)
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
train_df["translation"] = train_df["translation"].astype(str).apply(clean_translation)
train_df["transliteration"] = train_df["transliteration"].astype(str).apply(clean_transliteration)

akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_df["akt_group"] = train_df["oare_id"].map(oare_to_group).fillna("None")

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_df, groups=train_df["akt_group"].values))
_, val_idx = splits[FOLD]
val_df = train_df.iloc[val_idx].copy().reset_index(drop=True)
logger.info(f"Fold {FOLD}: val={len(val_df)} docs")

# sent-CV用: sentence_aligned.csvからval docsの文を取得
sent_df = pd.read_csv(str(SENT_ALIGNED_PATH))
val_oare_ids = set(val_df["oare_id"].tolist())
sent_val = sent_df[sent_df["oare_id"].isin(val_oare_ids)].reset_index(drop=True)

# 6文以下のdocのみ(既存eval_full_doc.pyと同条件)
doc_sent_counts = sent_val.groupby("oare_id").size()
short_docs = doc_sent_counts[doc_sent_counts <= 6].index
sent_val_short = sent_val[sent_val["oare_id"].isin(short_docs)].reset_index(drop=True)

# 前処理適用
sent_val_short["akk_segment"] = sent_val_short["akk_segment"].astype(str).apply(clean_transliteration)
sent_val_short["eng_sentence"] = sent_val_short["eng_sentence"].astype(str).apply(clean_translation)

logger.info(f"sent-CV: {len(sent_val_short)} sents from {sent_val_short['oare_id'].nunique()} docs")
logger.info(f"doc-CV: {len(val_df)} docs")

# ============================================================
# Model loading
# ============================================================
from transformers import Qwen3_5ForConditionalGeneration

logger.info(f"Loading model: {cmd_args.model} (quantize={cmd_args.quantize})")

load_kwargs = {
    "device_map": "auto",
    "torch_dtype": torch.bfloat16,
}

if cmd_args.quantize == "int4":
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
elif cmd_args.quantize == "int8":
    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(cmd_args.model)
model = Qwen3_5ForConditionalGeneration.from_pretrained(cmd_args.model, **load_kwargs)
model.eval()
logger.info("Model loaded")

# ============================================================
# Translation function
# ============================================================
SYSTEM_PROMPT = """You are an expert Assyriologist specializing in Old Assyrian texts. Translate the following Akkadian transliteration into English. Output ONLY the English translation, nothing else."""


def translate(text):
    """Translate a single Akkadian transliteration to English."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
        )
    # 入力部分をスキップして生成部分のみデコード
    output_ids = generated_ids[0][model_inputs["input_ids"].shape[1]:]
    decoded = processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return decoded


# ============================================================
# sent-CV evaluation
# ============================================================
logger.info("=== sent-CV evaluation ===")
sent_sources = sent_val_short["akk_segment"].tolist()
sent_references = sent_val_short["eng_sentence"].tolist()

sent_predictions = []
for src in tqdm(sent_sources, desc="sent-CV"):
    pred = translate(src)
    sent_predictions.append(clean_translation(pred))

# chrF++ (word_order=2)
chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
chrf_score = chrf_metric.corpus_score(sent_predictions, [sent_references]).score
bleu_score = sacrebleu.corpus_bleu(sent_predictions, [sent_references]).score
geo_mean = (chrf_score * bleu_score) ** 0.5 if chrf_score > 0 and bleu_score > 0 else 0.0


def has_repetition(text):
    words = str(text).split()
    if len(words) < 6:
        return False
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return True
    return False


rep_rate = sum(has_repetition(p) for p in sent_predictions) / len(sent_predictions) * 100
logger.info(f"  sent-CV ({len(sent_predictions)} sents): chrF++={chrf_score:.2f}, BLEU={bleu_score:.2f}, geo={geo_mean:.2f}, rep={rep_rate:.1f}%")

# ============================================================
# doc-CV evaluation
# ============================================================
logger.info("=== doc-CV evaluation ===")
doc_sources = val_df["transliteration"].tolist()
doc_references = val_df["translation"].tolist()

doc_predictions = []
for src in tqdm(doc_sources, desc="doc-CV"):
    pred = translate(src)
    doc_predictions.append(clean_translation(pred))

doc_chrf = chrf_metric.corpus_score(doc_predictions, [doc_references]).score
doc_bleu = sacrebleu.corpus_bleu(doc_predictions, [doc_references]).score
doc_geo = (doc_chrf * doc_bleu) ** 0.5 if doc_chrf > 0 and doc_bleu > 0 else 0.0
doc_rep = sum(has_repetition(p) for p in doc_predictions) / len(doc_predictions) * 100

logger.info(f"  doc-CV ({len(doc_predictions)} docs): chrF++={doc_chrf:.2f}, BLEU={doc_bleu:.2f}, geo={doc_geo:.2f}, rep={doc_rep:.1f}%")

# ============================================================
# Summary
# ============================================================
logger.info("============================================================")
logger.info(f"=== exp040 Qwen3.5-9B zero-shot (fold{FOLD}) ===")
logger.info(f"  sent-CV: chrF++={chrf_score:.2f}, BLEU={bleu_score:.2f}, geo={geo_mean:.2f}, rep={rep_rate:.1f}%")
logger.info(f"  doc-CV:  chrF++={doc_chrf:.2f}, BLEU={doc_bleu:.2f}, geo={doc_geo:.2f}, rep={doc_rep:.1f}%")

# Save predictions
sent_out = pd.DataFrame({
    "source": sent_sources,
    "reference": sent_references,
    "prediction": sent_predictions,
})
sent_out.to_csv(str(RESULTS_DIR / "sent_predictions.csv"), index=False)

doc_out = pd.DataFrame({
    "oare_id": val_df["oare_id"].tolist(),
    "source": doc_sources,
    "reference": doc_references,
    "prediction": doc_predictions,
})
doc_out.to_csv(str(RESULTS_DIR / "doc_predictions.csv"), index=False)

import json
metrics = {
    "sent_chrf": chrf_score, "sent_bleu": bleu_score, "sent_geo": geo_mean, "sent_rep": rep_rate,
    "doc_chrf": doc_chrf, "doc_bleu": doc_bleu, "doc_geo": doc_geo, "doc_rep": doc_rep,
}
with open(str(RESULTS_DIR / "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

logger.info(f"Results saved to {RESULTS_DIR}")
logger.info("Done.")
