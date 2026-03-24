"""
s1_exp011_large_bt_pretrain: Stage1 — pseudo_labels_v2 + backtranslated_v3でpre-training
- 開始モデル: google/byt5-large (HuggingFace pretrained weights)
- pseudo_labels_v2.csv (6,109件) + backtranslated.csv (62,655件) を結合して双方向学習
- lr=1e-4, 5epoch
- real dataのfold3 valでevalモニタリング
"""
import os
import gc
import re
import sys
import logging
import shutil
import argparse
from pathlib import Path

os.environ["WANDB_PROJECT"] = "akkadian-translation"

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_callback import TrainerCallback
import evaluate
import json

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

# Pseudo data sources
PSEUDO_LABELS_V2_PATH = PROJECT_ROOT / "workspace" / "exp038_backtranslation_augment" / "dataset" / "pseudo_labels_v2.csv"
BACKTRANSLATED_V3_PATH = PROJECT_ROOT / "workspace" / "exp044_bt_augment_v3" / "dataset" / "backtranslated.csv"

# HuggingFace pretrained weightsから開始
MODEL_NAME = "google/byt5-large"

log_file = str(RESULTS_DIR / "pretrain.log")
log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
    force=True,
)
logger = logging.getLogger(__name__)

_tf_logger = logging.getLogger("transformers")
_tf_logger.setLevel(logging.INFO)
_file_handler = logging.FileHandler(log_file)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_tf_logger.addHandler(_file_handler)

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
        logger.info(f"Metrics log saved to {self.output_path}")


# ============================================================
# Config
# ============================================================
MAX_LENGTH = 512
EPOCHS = 5  # Stage1: pseudo pretrainは5エポック
LEARNING_RATE = 1e-4
SEED = 42

PRETRAIN_MODEL_DIR = str(RESULTS_DIR / "pretrain_model")
CHECKPOINT_DIR = str(RESULTS_DIR / "pretrain_checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
logger.info(f"Fold: {FOLD}")
logger.info(f"Base model: {MODEL_NAME}")


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
# Data: pseudo data (2ソース結合) で学習、real dataでeval
# ============================================================
# Source 1: pseudo_labels_v2 (oare_id付き)
logger.info(f"Loading pseudo_labels_v2 from {PSEUDO_LABELS_V2_PATH}")
pseudo_v2 = pd.read_csv(str(PSEUDO_LABELS_V2_PATH))
logger.info(f"pseudo_labels_v2: {len(pseudo_v2)} docs")
pseudo_v2 = pseudo_v2[["transliteration", "translation"]].copy()

# Source 2: backtranslated_v3 (oare_idなし)
logger.info(f"Loading backtranslated_v3 from {BACKTRANSLATED_V3_PATH}")
bt_v3 = pd.read_csv(str(BACKTRANSLATED_V3_PATH))
logger.info(f"backtranslated_v3: {len(bt_v3)} rows")
bt_v3 = bt_v3[["transliteration", "translation"]].copy()

# 結合
pseudo_combined = pd.concat([pseudo_v2, bt_v3], ignore_index=True)
logger.info(f"Combined pseudo data: {len(pseudo_combined)} rows")


# pseudo_labels_v2はdoc-levelなので展開する、backtranslated_v3は既にsentence-level
def simple_sentence_aligner_no_oare(df):
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


pseudo_expanded = simple_sentence_aligner_no_oare(pseudo_combined)
logger.info(f"Pseudo expanded: {len(pseudo_expanded)} sentences")

# Real data (eval用)
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
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
_, val_idx = splits[FOLD]
val_split = train_expanded.iloc[val_idx].copy()

logger.info(f"Fold {FOLD}: pseudo_train={len(pseudo_expanded)}, real_val={len(val_split)}")


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


def create_unidirectional_data_from_df(df):
    df = df[["transliteration", "translation"]].copy()
    df["input_text"] = "translate Akkadian to English: " + df["transliteration"].astype(str)
    df["target_text"] = df["translation"].astype(str)
    return Dataset.from_pandas(df)


# 学習: pseudo data (bidirectional)
bidirectional_train = create_bidirectional_data_from_df(pseudo_expanded)
# 評価: real val data (unidirectional)
unidirectional_val = create_unidirectional_data_from_df(val_split)
logger.info(f"Train samples: {len(bidirectional_train)} (Bidirectional pseudo)")
logger.info(f"Val samples:   {len(unidirectional_val)} (Unidirectional real)")

# ============================================================
# Tokenization
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Pre-cache special token IDs to avoid slow repeated property lookups in batch_decode
_special_ids = set(tokenizer.all_special_ids)


def _fast_batch_decode(ids_array):
    """batch_decode without skip_special_tokens to avoid O(n*m) property rebuild."""
    results = []
    for ids in ids_array:
        filtered = [int(i) for i in ids if int(i) not in _special_ids]
        results.append(tokenizer.decode(filtered, skip_special_tokens=False))
    return results


def preprocess_function(examples):
    inputs = [str(ex) for ex in examples["input_text"]]
    targets = [str(ex) for ex in examples["target_text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_train = bidirectional_train.map(preprocess_function, batched=True)
tokenized_val = unidirectional_val.map(preprocess_function, batched=True)

# ============================================================
# Training
# ============================================================
gc.collect()
torch.cuda.empty_cache()
logger.info(f"Loading base model: {MODEL_NAME}")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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
    decoded_preds = _fast_batch_decode(preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = _fast_batch_decode(labels)
    chrf = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    bleu = metric_bleu.compute(predictions=decoded_preds, references=[[x] for x in decoded_labels])["score"]
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    return {"chrf": round(chrf, 2), "bleu": round(bleu, 2), "geo_mean": round(geo_mean, 2)}


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
    run_name=f"s1_exp011_pretrain_fold{FOLD}",
    load_best_model_at_end=False,  # Stage1はlast modelを使う
    seed=SEED,
    dataloader_num_workers=12,
)

metrics_logger = MetricsLogger(str(RESULTS_DIR / "pretrain_metrics_log.json"))

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

# チェックポイント自動再開
ckpt_dir_path = Path(CHECKPOINT_DIR)
resume_ckpt = None
if ckpt_dir_path.exists():
    checkpoints = [d for d in ckpt_dir_path.iterdir() if d.name.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
        resume_ckpt = str(latest)
        logger.info(f"Auto-resuming from checkpoint: {resume_ckpt}")

logger.info(f"Starting Stage1 Pretrain (s1_exp011 fold{FOLD}, {EPOCHS} epochs on pseudo data)...")
if resume_ckpt:
    trainer.train(resume_from_checkpoint=resume_ckpt)
else:
    trainer.train()

# ============================================================
# Save pretrain model (last checkpoint)
# ============================================================
logger.info(f"Saving pretrain model to {PRETRAIN_MODEL_DIR}")
last_ckpt = None
ckpt_dir = Path(CHECKPOINT_DIR)
if ckpt_dir.exists():
    ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        last_ckpt = ckpts[-1]

if last_ckpt and last_ckpt.exists():
    last_model = AutoModelForSeq2SeqLM.from_pretrained(str(last_ckpt))
    last_model.save_pretrained(PRETRAIN_MODEL_DIR)
    tokenizer.save_pretrained(PRETRAIN_MODEL_DIR)
    del last_model
else:
    trainer.save_model(PRETRAIN_MODEL_DIR)
    tokenizer.save_pretrained(PRETRAIN_MODEL_DIR)

metrics = trainer.evaluate()
logger.info(f"Pretrain model eval metrics: {metrics}")

import yaml
with open(str(RESULTS_DIR / "pretrain_eval_metrics.yaml"), "w") as f:
    yaml.dump(metrics, f, default_flow_style=False)

logger.info(f"Stage1 Pretrain complete. Fold {FOLD}")
logger.info(f"Pretrain model: {PRETRAIN_MODEL_DIR}")
logger.info("Next: run finetune.py to fine-tune on real data")
