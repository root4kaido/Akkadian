"""
exp044_bt_augment_v3: Full-data finetune
- exp044 fold3 pretrained model → finetune on ALL train data (no validation split)
- Save last model only
"""
import os
import gc
import re
import sys
import logging
import argparse
from pathlib import Path

os.environ["WANDB_PROJECT"] = "akkadian-translation"

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_callback import TrainerCallback
import json

# ============================================================
# Paths
# ============================================================
EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parent.parent
RESULTS_DIR = EXP_DIR / "results" / "fulldata_ft"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRETRAIN_MODEL_DIR = EXP_DIR / "results" / "fold3" / "pretrain_ft" / "pretrain_model"

log_file = str(RESULTS_DIR / "train.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    force=True,
)
logger = logging.getLogger(__name__)

_tf_logger = logging.getLogger("transformers")
_tf_logger.setLevel(logging.INFO)
_fh = logging.FileHandler(log_file)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_tf_logger.addHandler(_fh)


class MetricsLogger(TrainerCallback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.train_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step, **logs}
        if "loss" in logs and "eval_loss" not in logs:
            self.train_logs.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        data = {"train": self.train_logs}
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics log saved to {self.output_path}")


# ============================================================
# Config
# ============================================================
MODEL_NAME = "google/byt5-base"
MAX_LENGTH = 512
FINETUNE_EPOCHS = 5
FINETUNE_LR = 5e-5
SEED = 42

LAST_MODEL_DIR = str(RESULTS_DIR / "last_model")
FINETUNE_CKPT_DIR = str(RESULTS_DIR / "checkpoints_finetune")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything(SEED)


# ============================================================
# 前処理: exp023と同一
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
    r"[Tt]e['\u2019\u02BE]?in[aā]tum": "10", r"[Tt][eē]['\u2019\u02BE]?in[aā]tum": "10",
    r"[Kk]uzall?[iu]m?": "11", r"[Aa]llan[aā]tum": "12",
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
# Data: ALL train data (no validation split)
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
logger.info(f"Original Train Data: {len(train_df)} docs")

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
logger.info(f"Full Train Data: {len(train_expanded)} sentences (ALL, no validation split)")


def create_bidirectional_data_from_df(df):
    df = df[["transliteration", "translation"]].copy()
    df_fwd = df.copy()
    df_fwd["input_text"] = "translate Akkadian to English: " + df_fwd["transliteration"].astype(str)
    df_fwd["target_text"] = df_fwd["translation"].astype(str)
    df_bwd = df.copy()
    df_bwd["input_text"] = "translate English to Akkadian: " + df_bwd["translation"].astype(str)
    df_bwd["target_text"] = df_bwd["transliteration"].astype(str)
    df_combined = pd.concat([df_fwd, df_bwd], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return Dataset.from_pandas(df_combined)


finetune_dataset = create_bidirectional_data_from_df(train_expanded)
logger.info(f"Finetune samples: {len(finetune_dataset)} (full train, bidirectional)")

# ============================================================
# Tokenization
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_function(examples):
    inputs = [str(ex) for ex in examples["input_text"]]
    targets = [str(ex) for ex in examples["target_text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_finetune = finetune_dataset.map(preprocess_function, batched=True)

# ============================================================
# Finetune on full data
# ============================================================
logger.info("=" * 60)
logger.info("=== Full-data Finetune from exp044 pretrained model ===")
logger.info("=" * 60)

if not PRETRAIN_MODEL_DIR.exists():
    logger.error(f"Pretrain model not found: {PRETRAIN_MODEL_DIR}")
    logger.error("Run exp044 pretrain first!")
    sys.exit(1)

gc.collect()
torch.cuda.empty_cache()
model = AutoModelForSeq2SeqLM.from_pretrained(str(PRETRAIN_MODEL_DIR))
logger.info(f"Model loaded from {PRETRAIN_MODEL_DIR} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

finetune_args = Seq2SeqTrainingArguments(
    output_dir=FINETUNE_CKPT_DIR,
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=FINETUNE_LR,
    optim="adafactor",
    bf16=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    generation_max_length=MAX_LENGTH,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=FINETUNE_EPOCHS,
    predict_with_generate=False,
    logging_steps=10,
    report_to="wandb",
    run_name="exp044_fulldata_ft",
    seed=SEED,
    dataloader_num_workers=12,
)

metrics_logger = MetricsLogger(str(RESULTS_DIR / "metrics_log.json"))

trainer = Seq2SeqTrainer(
    model=model,
    args=finetune_args,
    train_dataset=tokenized_finetune,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[metrics_logger],
)

# チェックポイント自動再開
ckpt_dir = Path(FINETUNE_CKPT_DIR)
resume_ckpt = None
if ckpt_dir.exists():
    checkpoints = [d for d in ckpt_dir.iterdir() if d.name.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
        resume_ckpt = str(latest)
        logger.info(f"Auto-resuming from: {resume_ckpt}")

logger.info("Starting full-data finetune...")
if resume_ckpt:
    trainer.train(resume_from_checkpoint=resume_ckpt)
else:
    trainer.train()

# Save last model only
logger.info(f"Saving last model to {LAST_MODEL_DIR}")
trainer.save_model(LAST_MODEL_DIR)
tokenizer.save_pretrained(LAST_MODEL_DIR)

logger.info("Training complete.")
