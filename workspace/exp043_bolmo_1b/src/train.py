"""
exp043_bolmo_1b: Bolmo-1B (byte-level decoder-only LM) for Akkadian translation
- CausalLM fine-tuning with input masking (loss only on target tokens)
- Same preprocessing as exp023
- GroupKFold(n_splits=5) with akt_groups
- Single stage (real data only) as initial test
"""
import os
import gc
import re
import sys
import logging
import argparse
import json
from pathlib import Path

os.environ["WANDB_PROJECT"] = "akkadian-translation"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import GroupKFold
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from torch.nn import CrossEntropyLoss

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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"

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
MODEL_NAME = "allenai/Bolmo-1B"
MAX_INPUT_LENGTH = 512   # bytes for prompt (prefix + source)
MAX_TARGET_LENGTH = 512  # bytes for target
MAX_LENGTH = MAX_INPUT_LENGTH + MAX_TARGET_LENGTH  # total after concat
EPOCHS = 10
LEARNING_RATE = 1e-4
SEED = 42

BEST_MODEL_DIR = str(RESULTS_DIR / "best_model")
LAST_MODEL_DIR = str(RESULTS_DIR / "last_model")
CHECKPOINT_DIR = str(RESULTS_DIR / "checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
logger.info(f"Fold: {FOLD}")


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything(SEED)

# ============================================================
# Prompt template
# ============================================================
PROMPT_TEMPLATE = "Translate Akkadian to English.\nSource: {src}\nTarget: {tgt}"
PROMPT_PREFIX = "Translate Akkadian to English.\nSource: {src}\nTarget: "

# ============================================================
# Preprocessing (same as exp023)
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
# Data
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
logger.info(f"Original Train Data: {len(train_df)} docs")

akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))

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
logger.info(f"Expanded Train Data: {len(train_expanded)} sentences")

train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")
groups = train_expanded["akt_group"].values
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=groups))
train_idx, val_idx = splits[FOLD]

train_split = train_expanded.iloc[train_idx].copy()
val_split = train_expanded.iloc[val_idx].copy()
logger.info(f"Fold {FOLD}: train={len(train_split)}, val={len(val_split)}")

# ============================================================
# Tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Bolmo already has pad_token='<pad>'(id=0) != eos_token='<bos>'(id=1)
logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}, len={len(tokenizer)}")
logger.info(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
logger.info(f"PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
logger.info(f"BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")


# ============================================================
# Custom Dataset for CausalLM with input masking
# ============================================================
class TranslationCausalLMDataset(TorchDataset):
    """
    CausalLM dataset: concatenate prompt + target, mask prompt portion in labels.
    Format: "Translate Akkadian to English.\nSource: {src}\nTarget: {tgt}<eos>"
    Labels: -100 for prompt portion, actual token ids for target portion.
    """

    def __init__(self, df, tokenizer, max_length, bidirectional=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for _, row in df.iterrows():
            src = str(row["transliteration"])
            tgt = str(row["translation"])
            # Forward: Akkadian -> English
            self.samples.append((src, tgt, "akk2en"))
            # Backward: English -> Akkadian
            if bidirectional:
                self.samples.append((tgt, src, "en2akk"))

        if bidirectional:
            np.random.seed(SEED)
            indices = np.random.permutation(len(self.samples))
            self.samples = [self.samples[i] for i in indices]

        logger.info(f"Dataset created: {len(self.samples)} samples (bidirectional={bidirectional})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt, direction = self.samples[idx]

        if direction == "akk2en":
            prompt = f"Translate Akkadian to English.\nSource: {src}\nTarget: "
        else:
            prompt = f"Translate English to Akkadian.\nSource: {src}\nTarget: "

        # Tokenize prompt (with BOS) and target (WITHOUT BOS) separately
        prompt_enc = self.tokenizer(prompt, max_length=MAX_INPUT_LENGTH, truncation=True)
        # add_special_tokens=False to avoid double BOS at prompt-target boundary
        target_enc = self.tokenizer(tgt + self.tokenizer.eos_token, max_length=MAX_TARGET_LENGTH, truncation=True, add_special_tokens=False)

        prompt_ids = prompt_enc["input_ids"]
        target_ids = target_enc["input_ids"]

        # Concatenate
        input_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Labels: -100 for prompt, actual ids for target
        labels = torch.cat([
            torch.full((len(prompt_ids),), -100, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        ])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# Custom data collator with padding
# ============================================================
class CausalLMDataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(f["input_ids"].shape[0] for f in features)
        max_len = min(max_len, self.max_length)
        # Bolmo's xlstm requires sequence length divisible by 64
        max_len = ((max_len + 63) // 64) * 64

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            input_ids = f["input_ids"][:max_len]
            attention_mask = f["attention_mask"][:max_len]
            labels = f["labels"][:max_len]

            pad_len = max_len - input_ids.shape[0]
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


# ============================================================
# Create datasets
# ============================================================
train_dataset = TranslationCausalLMDataset(train_split, tokenizer, MAX_LENGTH, bidirectional=True)
val_dataset = TranslationCausalLMDataset(val_split, tokenizer, MAX_LENGTH, bidirectional=False)

data_collator = CausalLMDataCollator(tokenizer, MAX_LENGTH)

# ============================================================
# Model
# ============================================================
gc.collect()
torch.cuda.empty_cache()

logger.info(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
# No resize needed — Bolmo vocab already includes pad(0) and eos/bos(1)
logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
logger.info(f"pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

# ============================================================
# Training
# ============================================================
# ============================================================
# Custom Trainer: Bolmo's forward doesn't accept labels
# ============================================================
class BolmoTrainer(Trainer):
    def _compute_bolmo_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fn = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Handle NaN: all labels were -100 (prompt exceeded max_length)
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
        return loss, outputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = self._compute_bolmo_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, outputs = self._compute_bolmo_loss(model, inputs)
        if torch.isnan(loss):
            logger.warning(f"NaN loss detected in prediction_step")
        return (loss.detach(), None, None)


metrics_logger = MetricsLogger(str(RESULTS_DIR / "metrics_log.json"))

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    bf16=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    report_to="wandb",
    run_name=f"exp043_bolmo_1b_fold{FOLD}",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=SEED,
    dataloader_num_workers=4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    remove_unused_columns=False,
)

trainer = BolmoTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[metrics_logger],
)

# チェックポイント自動再開
ckpt_dir = Path(CHECKPOINT_DIR)
resume_ckpt = None
if ckpt_dir.exists():
    checkpoints = [d for d in ckpt_dir.iterdir() if d.name.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
        resume_ckpt = str(latest)
        logger.info(f"Auto-resuming from: {resume_ckpt}")

logger.info("Starting Training (Bolmo-1B, CausalLM)...")
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
ckpts = sorted(Path(CHECKPOINT_DIR).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
if ckpts:
    last_model = AutoModelForCausalLM.from_pretrained(str(ckpts[-1]), trust_remote_code=True)
    last_model.save_pretrained(LAST_MODEL_DIR)
    tokenizer.save_pretrained(LAST_MODEL_DIR)
    del last_model
else:
    trainer.save_model(LAST_MODEL_DIR)
    tokenizer.save_pretrained(LAST_MODEL_DIR)

final_metrics = trainer.evaluate()
logger.info(f"Final eval metrics: {final_metrics}")

import yaml
with open(str(RESULTS_DIR / "eval_metrics.yaml"), "w") as f:
    yaml.dump(final_metrics, f, default_flow_style=False)

logger.info("Training complete. Run eval_cv.py for generation-based evaluation.")
