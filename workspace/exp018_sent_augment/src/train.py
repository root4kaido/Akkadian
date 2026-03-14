"""
exp018_sent_augment: doc-levelデータ + 確率的開始位置シフトaugmentation
- ベース: exp016 (ByT5-base, bf16, lr=2e-4, 20epoch)
- データ: train.csv doc-level（6文以下のドキュメントのみ）
- augmentation: 確率0.5でsentence_aligned.csvの境界情報を使い、
  Akkadianの開始位置をランダムな文境界にずらす。英語側も対応する文以降に。
  逆翻訳方向(Eng→Akk)も同様にずらす。
"""
import os
import gc
import re
import sys
import random
import logging
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

SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"

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
MODEL_NAME = "google/byt5-base"
MAX_LENGTH = 512
EPOCHS = 20
LEARNING_RATE = 2e-4
SEED = 42
AUGMENT_PROB = 0.5  # 確率でずらす

BEST_MODEL_DIR = str(RESULTS_DIR / "best_model")
LAST_MODEL_DIR = str(RESULTS_DIR / "last_model")
CHECKPOINT_DIR = str(RESULTS_DIR / "checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


seed_everything(SEED)

# ============================================================
# Data: exp016と同一のval splitを再現
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
logger.info(f"Original Train Data: {len(train_df)} docs")


def split_sentences_eng(text):
    text = str(text).strip()
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sents


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


# exp016と同一のval splitを再現
train_expanded = simple_sentence_aligner(train_df)
dataset = Dataset.from_pandas(train_expanded)
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)

# ============================================================
# アライメント情報のロードと辞書化
# ============================================================
sent_aligned = pd.read_csv(str(SENTENCE_ALIGNED_PATH))
logger.info(f"Sentence aligned data: {len(sent_aligned)} rows, {sent_aligned['oare_id'].nunique()} docs")

# oare_id → 文境界情報の辞書
# 各ドキュメントについて、(start_tok, end_tok, eng_sentence)のリストを保持
alignment_dict = {}
for oare_id, group in sent_aligned.groupby('oare_id'):
    group = group.sort_values('sent_idx')
    boundaries = []
    for _, row in group.iterrows():
        boundaries.append({
            'sent_idx': int(row['sent_idx']),
            'start_tok': int(row['start_tok']),
            'end_tok': int(row['end_tok']),
            'eng_sentence': str(row['eng_sentence']),
        })
    alignment_dict[oare_id] = boundaries

logger.info(f"Alignment dict: {len(alignment_dict)} docs with boundaries")

# ============================================================
# oare_id → train_df行のマッピング
# ============================================================
oare_to_row = {}
for _, row in train_df.iterrows():
    oare_to_row[row['oare_id']] = row

# ============================================================
# Augmentedデータ作成: 6文以下のdocのみ対象
# ============================================================
train_pandas = split_datasets["train"].to_pandas()


def create_augmented_bidirectional_data(train_pandas, train_df):
    """
    各ドキュメントについて:
    - 確率(1-p): そのまま（先頭から）= exp016と同じ
    - 確率p: ランダムな文境界から開始（Akk/Eng両方をずらす）
    """
    rows_fwd = []
    rows_bwd = []

    for _, row in train_pandas.iterrows():
        translit = str(row['transliteration'])
        translation = str(row['translation'])

        # このドキュメントのoare_idを特定
        # train_expandedにはoare_idがないので、transliterationでマッチ
        matched_oare = None
        for oare_id, orig_row in oare_to_row.items():
            if str(orig_row['transliteration']) == translit:
                matched_oare = oare_id
                break

        n_eng_sents = len(split_sentences_eng(translation))
        has_alignment = (matched_oare is not None and
                         matched_oare in alignment_dict and
                         n_eng_sents <= 6 and n_eng_sents >= 2)

        if has_alignment and random.random() < AUGMENT_PROB:
            # ランダムな文境界にずらす
            boundaries = alignment_dict[matched_oare]
            # 0以外の開始文を選ぶ（0は通常と同じなので1以降）
            if len(boundaries) > 1:
                shift_idx = random.randint(1, len(boundaries) - 1)

                # Akkadian: shift_idx文目以降
                akk_tokens = translit.split()
                new_start_tok = boundaries[shift_idx]['start_tok']
                new_start_tok = min(new_start_tok, len(akk_tokens) - 1)
                shifted_akk = " ".join(akk_tokens[new_start_tok:])

                # English: shift_idx文目以降の文を結合
                shifted_eng_parts = [b['eng_sentence'] for b in boundaries[shift_idx:]]
                shifted_eng = " ".join(shifted_eng_parts)

                if shifted_akk.strip() and shifted_eng.strip():
                    # Forward: Akk→Eng
                    rows_fwd.append({
                        "input_text": "translate Akkadian to English: " + shifted_akk,
                        "target_text": shifted_eng,
                    })
                    # Backward: Eng→Akk
                    rows_bwd.append({
                        "input_text": "translate English to Akkadian: " + shifted_eng,
                        "target_text": shifted_akk,
                    })
                    continue

        # 通常: そのまま
        rows_fwd.append({
            "input_text": "translate Akkadian to English: " + translit,
            "target_text": translation,
        })
        rows_bwd.append({
            "input_text": "translate English to Akkadian: " + translation,
            "target_text": translit,
        })

    df_combined = pd.DataFrame(rows_fwd + rows_bwd)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return Dataset.from_pandas(df_combined)


bidirectional_train = create_augmented_bidirectional_data(train_pandas, train_df)

# augmented stats
n_total = len(train_pandas)
logger.info(f"Train docs: {n_total}")
logger.info(f"Bidirectional train samples: {len(bidirectional_train)}")


def create_unidirectional_data(dataset_split):
    df = dataset_split.to_pandas()
    df["input_text"] = "translate Akkadian to English: " + df["transliteration"].astype(str)
    df["target_text"] = df["translation"].astype(str)
    return Dataset.from_pandas(df)


unidirectional_val = create_unidirectional_data(split_datasets["test"])
logger.info(f"Train samples: {len(bidirectional_train)} (Bidirectional with augment)")
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
# Training
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
    run_name="exp018_sent_augment",
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

ckpt_dir_path = Path(CHECKPOINT_DIR)
resume_ckpt = None
if ckpt_dir_path.exists():
    checkpoints = [d for d in ckpt_dir_path.iterdir() if d.name.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
        resume_ckpt = str(latest)
        logger.info(f"Auto-resuming from checkpoint: {resume_ckpt}")

logger.info("Starting Training (exp018: sent_augment)...")
if resume_ckpt:
    trainer.train(resume_from_checkpoint=resume_ckpt)
else:
    trainer.train()

# ============================================================
# Save models
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
logger.info(f"Best model eval metrics: {metrics}")

import yaml
with open(str(RESULTS_DIR / "eval_metrics.yaml"), "w") as f:
    yaml.dump(metrics, f, default_flow_style=False)

logger.info("Training complete.")
