"""
exp026_denoising: Denoising Augmentation + GroupKFold
- ベース: exp023 GKF fold0
- 変更点: ソース側（翻字）にランダムノイズを注入して学習
  - バイトレベルでdrop / swap / insert を確率的に適用
  - モデルが表層パターンに依存せず文脈から翻訳する力をつける
  - ByT5のspan corruption事前学習と相性が良い
"""
import os
import gc
import re
import sys
import logging
import shutil
import argparse
import random
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
parser.add_argument("--noise_prob", type=float, default=0.1,
                    help="Per-character noise probability (default: 0.1)")
cmd_args = parser.parse_args()
FOLD = cmd_args.fold
NOISE_PROB = cmd_args.noise_prob

# ============================================================
# Paths
# ============================================================
EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parent.parent
RESULTS_DIR = EXP_DIR / "results" / f"fold{FOLD}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"

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

BEST_MODEL_DIR = str(RESULTS_DIR / "best_model")
LAST_MODEL_DIR = str(RESULTS_DIR / "last_model")
CHECKPOINT_DIR = str(RESULTS_DIR / "checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
logger.info(f"Fold: {FOLD}, Noise prob: {NOISE_PROB}")


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


seed_everything(SEED)


# ============================================================
# Denoising Augmentation
# ============================================================
# アッカド語翻字でよく使われる文字群（ノイズ挿入時の候補）
AKKADIAN_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789-šṣṭāēīū ")


def add_noise_to_transliteration(text: str, prob: float = 0.1) -> str:
    """翻字テキストに文字レベルのノイズを注入する。

    各文字に対して確率probで以下のいずれかを適用:
    - drop (33%): 文字を削除
    - swap (33%): 隣の文字と入れ替え
    - insert (33%): ランダムな文字を挿入

    改行(\n)とプレフィックス("translate ...")は保持。
    """
    if not isinstance(text, str) or len(text) < 5:
        return text

    chars = list(text)
    result = []
    i = 0
    while i < len(chars):
        if chars[i] == '\n':
            result.append(chars[i])
            i += 1
            continue

        if random.random() < prob:
            action = random.random()
            if action < 0.33:
                # drop: skip this character
                i += 1
                continue
            elif action < 0.66:
                # swap: swap with next character
                if i + 1 < len(chars) and chars[i + 1] != '\n':
                    result.append(chars[i + 1])
                    result.append(chars[i])
                    i += 2
                    continue
                else:
                    result.append(chars[i])
                    i += 1
                    continue
            else:
                # insert: add a random character before current
                result.append(random.choice(AKKADIAN_CHARS))
                result.append(chars[i])
                i += 1
                continue
        else:
            result.append(chars[i])
            i += 1

    return "".join(result)


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
# Data
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
logger.info(f"Original Train Data: {len(train_df)} docs")

akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))

logger.info("Applying full preprocessing (exp023)...")
train_df["translation"] = train_df["translation"].astype(str).apply(clean_translation)
train_df["transliteration"] = train_df["transliteration"].astype(str).apply(clean_transliteration)
logger.info("Preprocessing complete.")


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
logger.info(f"Expanded Train Data: {len(train_expanded)} sentences")

train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")

groups = train_expanded["akt_group"].values
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=groups))
train_idx, val_idx = splits[FOLD]

train_split = train_expanded.iloc[train_idx].copy()
val_split = train_expanded.iloc[val_idx].copy()

val_groups = val_split["akt_group"].value_counts()
train_groups = train_split["akt_group"].value_counts()
logger.info(f"Fold {FOLD}: train={len(train_split)}, val={len(val_split)}")
logger.info(f"Val groups:\n{val_groups.to_string()}")
logger.info(f"Train groups:\n{train_groups.to_string()}")


def create_bidirectional_data_from_df(df, apply_noise=False):
    """双方向データ作成。apply_noise=Trueの場合、ソース側にノイズを注入。

    - Akk→Eng: transliterationにノイズ注入（ソース側）
    - Eng→Akk: translationはクリーンのまま（英語にノイズを入れても意味がない）
    - ノイズ版とクリーン版の両方を含める（データ量2倍ではなく、50%にノイズ適用）
    """
    df = df[["transliteration", "translation"]].copy()

    # Forward: Akk → Eng
    df_fwd = df.copy()
    if apply_noise:
        # 50%のサンプルにノイズを適用
        noisy_mask = np.random.random(len(df_fwd)) < 0.5
        df_fwd_clean = df_fwd[~noisy_mask].copy()
        df_fwd_noisy = df_fwd[noisy_mask].copy()
        df_fwd_noisy["transliteration"] = df_fwd_noisy["transliteration"].apply(
            lambda x: add_noise_to_transliteration(x, prob=NOISE_PROB)
        )
        df_fwd = pd.concat([df_fwd_clean, df_fwd_noisy], ignore_index=True)

    df_fwd["input_text"] = "translate Akkadian to English: " + df_fwd["transliteration"].astype(str)
    df_fwd["target_text"] = df_fwd["translation"].astype(str)

    # Backward: Eng → Akk (ノイズなし)
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


bidirectional_train = create_bidirectional_data_from_df(train_split, apply_noise=True)
unidirectional_val = create_unidirectional_data_from_df(val_split)
logger.info(f"Train samples: {len(bidirectional_train)} (Bidirectional + Denoising)")
logger.info(f"Val samples:   {len(unidirectional_val)} (Unidirectional, clean)")

# ノイズの例をログに出力
logger.info("=== Denoising examples ===")
example_texts = train_split["transliteration"].head(3).tolist()
for orig in example_texts:
    noisy = add_noise_to_transliteration(str(orig), prob=NOISE_PROB)
    logger.info(f"  Original: {str(orig)[:80]}")
    logger.info(f"  Noisy:    {noisy[:80]}")
    logger.info("")

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
    run_name=f"exp026_denoising_p{NOISE_PROB}_fold{FOLD}",
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

logger.info(f"Starting Training (exp026 Denoising noise_prob={NOISE_PROB} fold{FOLD})...")
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

logger.info(f"Training complete. Fold {FOLD}")
logger.info(f"Best model: {BEST_MODEL_DIR}")
logger.info(f"Last model: {LAST_MODEL_DIR}")
