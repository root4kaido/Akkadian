"""
exp022_train_cleaning: exp016 + Host推奨のtrain前処理
- translation: (?), PN→<gap>, /代替→先頭選択, 小数→分数, ローマ数字月→整数,
               -gold/-tax/-textiles置換, stray marks除去
- transliteration: Ḫ→H, ḫ→h, 下付き数字→整数, 小数→分数
- モデル・ハイパラはexp016と完全同一
"""
import os
import gc
import re
import sys
import logging
import shutil
from pathlib import Path

os.environ["WANDB_PROJECT"] = "akkadian-translation"

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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
# Paths
# ============================================================
EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parent.parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

log_file = str(RESULTS_DIR / "train.log")
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
    """学習メトリクスをJSONファイルに保存"""

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
# Config — exp016と同一
# ============================================================
MODEL_NAME = "google/byt5-base"
MAX_LENGTH = 512
EPOCHS = 20
LEARNING_RATE = 2e-4
SEED = 42

BEST_MODEL_DIR = str(RESULTS_DIR / "best_model")
LAST_MODEL_DIR = str(RESULTS_DIR / "last_model")
CHECKPOINT_DIR = str(RESULTS_DIR / "checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


# ============================================================
# Seed
# ============================================================
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything(SEED)


# ============================================================
# Host推奨前処理 (Discussion 678899 "A Stitch in Time")
# ============================================================

# 小数→分数マッピング
DECIMAL_TO_FRACTION = {
    "0.5": "½",
    "0.25": "¼",
    "0.3333": "⅓",
    "0.6666": "⅔",
    "0.8333": "⅚",
    "0.75": "¾",
    "0.1666": "⅙",
    "0.625": "⅝",
}

# ローマ数字→整数（月名用、I〜XII）
ROMAN_TO_INT = {
    "XII": "12", "XI": "11", "VIII": "8", "VII": "7",
    "VI": "6", "IV": "4", "IX": "9", "III": "3",
    "II": "2", "X": "10", "V": "5", "I": "1",
}

# 下付き数字→通常数字
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def clean_translation(text: str) -> str:
    """Host推奨のtranslation前処理"""
    if not isinstance(text, str) or not text.strip():
        return text

    # fem., sing., pl., plural 除去（念のため）
    text = re.sub(r'\bfem\.\s*', '', text)
    text = re.sub(r'\bsing\.\s*', '', text)
    text = re.sub(r'\bpl\.\s*', '', text)
    text = re.sub(r'\bplural\b\s*', '', text)

    # (?) 除去
    text = text.replace('(?)', '')

    # stray marks除去: << >> と < >（<gap>は残す）
    text = re.sub(r'<<\s*>>', '', text)
    text = re.sub(r'<\s+>', '', text)  # < > but not <gap>

    # 孤立 .. 除去（... も含む、ただし文末ピリオドは残す）
    text = re.sub(r'(?<!\.)\.\.(?!\.)', '', text)

    # 孤立 x, xx 除去（単語境界で）
    text = re.sub(r'\bxx?\b', '', text)

    # PN → <gap>
    text = re.sub(r'\bPN\b', '<gap>', text)

    # -gold → pašallum gold
    text = re.sub(r'\b-gold\b', 'pašallum gold', text)

    # -tax → šadduātum tax
    text = re.sub(r'\b-tax\b', 'šadduātum tax', text)

    # -textiles → kutānum textiles
    text = re.sub(r'\b-textiles\b', 'kutānum textiles', text)

    # / による代替翻訳 → 最初の選択肢を採用
    # パターン: "word1 / word2" → "word1"
    text = re.sub(r'(\S+)\s*/\s*\S+', r'\1', text)

    # 小数→分数変換（長い小数から先にマッチ）
    for decimal, fraction in sorted(DECIMAL_TO_FRACTION.items(), key=lambda x: -len(x[0])):
        text = text.replace(decimal, fraction)

    # ローマ数字月→整数（"month V" → "month 5"）
    # 長いローマ数字から先にマッチ
    for roman, integer in sorted(ROMAN_TO_INT.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf'\bmonth\s+{roman}\b', f'month {integer}', text)

    # 連続空白を1つに
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_transliteration(text: str) -> str:
    """Host推奨のtransliteration前処理（Optional項目）"""
    if not isinstance(text, str) or not text.strip():
        return text

    # Ḫ→H, ḫ→h
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')

    # 下付き数字→通常整数
    text = text.translate(SUBSCRIPT_MAP)

    # 小数→分数（transliterationでも同様）
    for decimal, fraction in sorted(DECIMAL_TO_FRACTION.items(), key=lambda x: -len(x[0])):
        text = text.replace(decimal, fraction)

    return text


# ============================================================
# Data
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
logger.info(f"Original Train Data: {len(train_df)} docs")

# 前処理適用
logger.info("Applying Host-recommended preprocessing...")
train_df["translation"] = train_df["translation"].astype(str).apply(clean_translation)
train_df["transliteration"] = train_df["transliteration"].astype(str).apply(clean_transliteration)
logger.info("Preprocessing complete.")


def simple_sentence_aligner(df):
    """starterと完全同一のアライナー（実質的にはマッチ0件で素通り）"""
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
logger.info(f"Expanded Train Data: {len(train_expanded)} sentences")

dataset = Dataset.from_pandas(train_expanded)
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)


def create_bidirectional_data(dataset_split):
    """starterと完全同一の双方向データ作成"""
    df = dataset_split.to_pandas()
    df_fwd = df.copy()
    df_fwd["input_text"] = "translate Akkadian to English: " + df_fwd["transliteration"].astype(str)
    df_fwd["target_text"] = df_fwd["translation"].astype(str)
    df_bwd = df.copy()
    df_bwd["input_text"] = "translate English to Akkadian: " + df_bwd["translation"].astype(str)
    df_bwd["target_text"] = df_bwd["transliteration"].astype(str)
    df_combined = pd.concat([df_fwd, df_bwd], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return Dataset.from_pandas(df_combined)


def create_unidirectional_data(dataset_split):
    """starterと完全同一のvalidation用データ作成"""
    df = dataset_split.to_pandas()
    df["input_text"] = "translate Akkadian to English: " + df["transliteration"].astype(str)
    df["target_text"] = df["translation"].astype(str)
    return Dataset.from_pandas(df)


bidirectional_train = create_bidirectional_data(split_datasets["train"])
unidirectional_val = create_unidirectional_data(split_datasets["test"])
logger.info(f"Train samples: {len(bidirectional_train)} (Bidirectional)")
logger.info(f"Val samples:   {len(unidirectional_val)} (Unidirectional)")

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


tokenized_train = bidirectional_train.map(preprocess_function, batched=True)
tokenized_val = unidirectional_val.map(preprocess_function, batched=True)

# ============================================================
# Training — BF16でByT5-base
# ============================================================
gc.collect()
torch.cuda.empty_cache()
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
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
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
    run_name="exp022_train_cleaning",
    load_best_model_at_end=True,
    metric_for_best_model="geo_mean",
    greater_is_better=True,
    seed=SEED,
    dataloader_num_workers=12,
)

metrics_logger = MetricsLogger(str(RESULTS_DIR / "metrics_log.json"))

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

logger.info("Starting Training (BF16, ByT5-base, Host preprocessing)...")
if resume_ckpt:
    trainer.train(resume_from_checkpoint=resume_ckpt)
else:
    trainer.train()

# ============================================================
# Save best model
# ============================================================
logger.info(f"Saving best model to {BEST_MODEL_DIR}")
trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)

# ============================================================
# Save last model
# ============================================================
logger.info(f"Saving last model to {LAST_MODEL_DIR}")
last_ckpt = None
ckpt_dir = Path(CHECKPOINT_DIR)
if ckpt_dir.exists():
    ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        last_ckpt = ckpts[-1]

if last_ckpt and last_ckpt.exists():
    logger.info(f"Last checkpoint found: {last_ckpt}")
    last_model = AutoModelForSeq2SeqLM.from_pretrained(str(last_ckpt))
    last_model.save_pretrained(LAST_MODEL_DIR)
    tokenizer.save_pretrained(LAST_MODEL_DIR)
    del last_model
else:
    logger.warning("Last checkpoint not found, saving current (best) model as last")
    trainer.save_model(LAST_MODEL_DIR)
    tokenizer.save_pretrained(LAST_MODEL_DIR)

# 最終評価
metrics = trainer.evaluate()
logger.info(f"Best model eval metrics: {metrics}")

import yaml
results_path = str(RESULTS_DIR / "eval_metrics.yaml")
with open(results_path, "w") as f:
    yaml.dump(metrics, f, default_flow_style=False)
logger.info(f"Metrics saved to {results_path}")

logger.info("Training complete.")
logger.info(f"Best model: {BEST_MODEL_DIR}")
logger.info(f"Last model: {LAST_MODEL_DIR}")
