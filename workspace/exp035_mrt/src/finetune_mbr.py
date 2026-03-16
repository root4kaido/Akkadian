"""
exp035 MBR-FT Step2: MBR選択翻訳をターゲットとしてcross-entropy fine-tune。

- データ: MBR選択翻訳（mbr_targets.csv）をターゲットとして使用
- モデル: exp034 fold{N} last_model からスタート
- 設定: exp034のfinetune.pyとほぼ同一（短いepoch, 低いlr）
"""
import os
import gc
import re
import sys
import logging
import json
import argparse
from pathlib import Path

os.environ["WANDB_PROJECT"] = "akkadian-translation"

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
from transformers.trainer_callback import TrainerCallback
import evaluate

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
cmd_args = parser.parse_args()
FOLD = cmd_args.fold

# ============================================================
# Paths
# ============================================================
EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parent.parent
RESULTS_DIR = EXP_DIR / "results" / f"fold{FOLD}"

AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"
BASE_MODEL_DIR = str(PROJECT_ROOT / "workspace" / "exp034_st_pretrain" / "results" / f"fold{FOLD}" / "last_model")
MBR_TARGETS_PATH = str(RESULTS_DIR / "mbr_targets.csv")

log_file = str(RESULTS_DIR / "finetune_mbr.log")
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
        self.eval_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step, **logs}
        if "eval_loss" in logs:
            self.eval_logs.append(entry)
        elif "loss" in logs:
            self.train_logs.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        data = {"train": self.train_logs, "eval": self.eval_logs}
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# Config
# ============================================================
MAX_LENGTH = 512
EPOCHS = 3               # short, MBR-FT needs few epochs
LEARNING_RATE = 5e-5     # same as exp034 finetune
SEED = 42

BEST_MODEL_DIR = str(RESULTS_DIR / "best_model")
LAST_MODEL_DIR = str(RESULTS_DIR / "last_model")
CHECKPOINT_DIR = str(RESULTS_DIR / "mbr_checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}, Fold: {FOLD}")

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


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

def clean_transliteration(text):
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


# ============================================================
# Data: MBR targets for train, original refs for val
# ============================================================
if not Path(MBR_TARGETS_PATH).exists():
    logger.error(f"MBR targets not found at {MBR_TARGETS_PATH}. Run generate_mbr_targets.py first.")
    sys.exit(1)

mbr_df = pd.read_csv(MBR_TARGETS_PATH)
logger.info(f"MBR targets: {len(mbr_df)} rows")

# Reconstruct val split for evaluation
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))

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
akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")
groups = train_expanded["akt_group"].values

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=groups))
train_idx, val_idx = splits[FOLD]
val_split = train_expanded.iloc[val_idx].copy().reset_index(drop=True)
logger.info(f"Val split: {len(val_split)} sentences")


# ============================================================
# Create datasets
# ============================================================
PREFIX = "translate Akkadian to English: "


def create_train_dataset(mbr_df):
    """Train: Akkadian→MBR target (forward only, no backward)."""
    df = mbr_df.copy()
    df["input_text"] = PREFIX + df["transliteration"].astype(str)
    df["target_text"] = df["mbr_target"].astype(str)
    return Dataset.from_pandas(df[["input_text", "target_text"]])


def create_val_dataset(val_df):
    """Val: Akkadian→original reference."""
    df = val_df.copy()
    df["input_text"] = PREFIX + df["transliteration"].astype(str)
    df["target_text"] = df["translation"].astype(str)
    return Dataset.from_pandas(df[["input_text", "target_text"]])


train_dataset = create_train_dataset(mbr_df)
val_dataset = create_val_dataset(val_split)
logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")


# ============================================================
# Model & Tokenizer
# ============================================================
gc.collect()
torch.cuda.empty_cache()
logger.info(f"Loading model from {BASE_MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def preprocess_function(examples):
    inputs = [str(ex) for ex in examples["input_text"]]
    targets = [str(ex) for ex in examples["target_text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# ============================================================
# Metrics
# ============================================================
metric_chrf = evaluate.load("chrf")
metric_bleu = evaluate.load("sacrebleu")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    if hasattr(preds, "ndim") and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    preds = preds.astype(np.int64)
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    chrf = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    bleu = metric_bleu.compute(predictions=decoded_preds, references=[[x] for x in decoded_labels])["score"]
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    return {"chrf": round(chrf, 2), "bleu": round(bleu, 2), "geo_mean": round(geo_mean, 2)}


# ============================================================
# Training
# ============================================================
args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    optim="adafactor",
    bf16=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    generation_max_length=MAX_LENGTH,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    logging_steps=10,
    report_to="wandb",
    run_name=f"exp035_mbr_ft_fold{FOLD}",
    load_best_model_at_end=True,
    metric_for_best_model="geo_mean",
    greater_is_better=True,
    label_smoothing_factor=0.0,  # no label smoothing for MBR-FT
    seed=SEED,
    dataloader_num_workers=12,
)

metrics_logger = MetricsLogger(str(RESULTS_DIR / "mbr_ft_metrics_log.json"))

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[metrics_logger],
)

# Checkpoint auto-resume
ckpt_dir_path = Path(CHECKPOINT_DIR)
resume_ckpt = None
if ckpt_dir_path.exists():
    checkpoints = [d for d in ckpt_dir_path.iterdir() if d.name.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
        resume_ckpt = str(latest)
        logger.info(f"Auto-resuming from checkpoint: {resume_ckpt}")

logger.info(f"Starting MBR-FT (fold{FOLD}, {EPOCHS} epochs)...")
if resume_ckpt:
    trainer.train(resume_from_checkpoint=resume_ckpt)
else:
    trainer.train()

# ============================================================
# Save
# ============================================================
logger.info(f"Saving best model to {BEST_MODEL_DIR}")
trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)

logger.info(f"Saving last model to {LAST_MODEL_DIR}")
last_ckpt = None
ckpt_dir = Path(CHECKPOINT_DIR)
if ckpt_dir.exists():
    ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        last_ckpt = ckpts[-1]

if last_ckpt and last_ckpt.exists():
    last_model = AutoModelForSeq2SeqLM.from_pretrained(str(last_ckpt))
    last_model.save_pretrained(LAST_MODEL_DIR)
    tokenizer.save_pretrained(LAST_MODEL_DIR)
    del last_model
else:
    trainer.save_model(LAST_MODEL_DIR)
    tokenizer.save_pretrained(LAST_MODEL_DIR)

metrics = trainer.evaluate()
logger.info(f"Final eval metrics: {metrics}")

import yaml
with open(str(RESULTS_DIR / "mbr_ft_eval_metrics.yaml"), "w") as f:
    yaml.dump(metrics, f, default_flow_style=False)

logger.info("MBR-FT complete!")
