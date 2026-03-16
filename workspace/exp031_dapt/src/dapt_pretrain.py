"""
exp031: Domain-Adaptive Pre-Training (DAPT)
- published_texts.csv の翻字テキスト7,953件でByT5-baseを継続事前学習
- ByT5のspan corruption目的関数（noise_density=0.15, mean_noise_span_length=20）
- 教師なし学習なのでラベル不要、リークなし
"""
import os
import re
import sys
import logging
import json
import time
from pathlib import Path

os.environ["WANDB_PROJECT"] = "akkadian-translation"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_callback import TrainerCallback

# ============================================================
# Paths
# ============================================================
EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parent.parent
RESULTS_DIR = EXP_DIR / "results" / "dapt_model"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = EXP_DIR / "results" / "dapt_checkpoints"

PUBLISHED_TEXTS_PATH = PROJECT_ROOT / "datasets" / "raw" / "published_texts.csv"

# ============================================================
# Logging
# ============================================================
log_file = str(RESULTS_DIR / "dapt_pretrain.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    force=True,
)
logger = logging.getLogger(__name__)

_tf_logger = logging.getLogger("transformers")
_tf_logger.setLevel(logging.INFO)
_tf_logger.addHandler(logging.FileHandler(log_file))

# ============================================================
# Config
# ============================================================
MODEL_NAME = "google/byt5-base"
MAX_LENGTH = 512
NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 20
MIN_TEXT_LENGTH = 50
VAL_RATIO = 0.1
SEED = 42

EPOCHS = 10
LEARNING_RATE = 5e-4
BATCH_SIZE = 4
GRAD_ACCUM = 2

np.random.seed(SEED)
torch.manual_seed(SEED)

logger.info(f"Config: noise_density={NOISE_DENSITY}, mean_span_length={MEAN_NOISE_SPAN_LENGTH}")
logger.info(f"Config: epochs={EPOCHS}, lr={LEARNING_RATE}, batch={BATCH_SIZE}, accum={GRAD_ACCUM}")

# ============================================================
# Preprocessing (exp023 clean_transliteration)
# ============================================================
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


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
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


# ============================================================
# Data
# ============================================================
logger.info(f"Loading published_texts from {PUBLISHED_TEXTS_PATH}")
df = pd.read_csv(str(PUBLISHED_TEXTS_PATH))
logger.info(f"Total documents: {len(df)}")

texts = df["transliteration"].astype(str).apply(clean_transliteration).tolist()
texts = [t for t in texts if len(t) >= MIN_TEXT_LENGTH]
logger.info(f"After filtering (>={MIN_TEXT_LENGTH} chars): {len(texts)} documents")
logger.info(f"Mean length: {np.mean([len(t) for t in texts]):.0f} chars")

# Train/val split
np.random.shuffle(texts)
n_val = max(1, int(len(texts) * VAL_RATIO))
val_texts = texts[:n_val]
train_texts = texts[n_val:]
logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

# ============================================================
# Tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
logger.info(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

# ByT5 sentinel token IDs
# ByT5 has extra_id_0 to extra_id_124 (IDs 259+)
SENTINEL_START_ID = tokenizer.convert_tokens_to_ids("<extra_id_0>")
logger.info(f"Sentinel start ID (<extra_id_0>): {SENTINEL_START_ID}")


# ============================================================
# Span Corruption Dataset
# ============================================================
def compute_noise_mask(length, noise_density, mean_noise_span_length):
    """Compute a random noise mask for span corruption."""
    num_noise_tokens = max(1, int(round(length * noise_density)))
    num_noise_spans = max(1, int(round(num_noise_tokens / mean_noise_span_length)))
    num_nonnoise_tokens = length - num_noise_tokens

    # Interleave spans of noise and non-noise
    # Each noise span has at least 1 token, each non-noise span has at least 0 tokens
    def _random_segmentation(num_items, num_segments):
        """Randomly split num_items into num_segments segments."""
        bars = sorted(np.random.choice(range(1, num_items), num_segments - 1, replace=False))
        segments = []
        prev = 0
        for b in bars:
            segments.append(b - prev)
            prev = b
        segments.append(num_items - prev)
        return segments

    if num_noise_spans > 1 and num_nonnoise_tokens > 1:
        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    else:
        noise_span_lengths = [num_noise_tokens]
        nonnoise_span_lengths = [num_nonnoise_tokens]

    # Build mask: interleave non-noise and noise spans
    mask = np.zeros(length, dtype=bool)
    pos = 0
    for i in range(len(nonnoise_span_lengths)):
        pos += nonnoise_span_lengths[i]
        if i < len(noise_span_lengths):
            end = min(pos + noise_span_lengths[i], length)
            mask[pos:end] = True
            pos = end
    return mask


def create_span_corruption_example(token_ids, noise_density, mean_noise_span_length):
    """Create span corruption input/target from token IDs."""
    length = len(token_ids)
    if length < 5:
        return token_ids, token_ids

    mask = compute_noise_mask(length, noise_density, mean_noise_span_length)

    # Build input (replace noise spans with sentinels)
    input_ids = []
    target_ids = []
    sentinel_idx = 0
    in_noise = False

    for i in range(length):
        if mask[i]:
            if not in_noise:
                # Start of new noise span
                sentinel_id = SENTINEL_START_ID + sentinel_idx
                input_ids.append(sentinel_id)
                target_ids.append(sentinel_id)
                sentinel_idx += 1
                in_noise = True
            target_ids.append(token_ids[i])
        else:
            input_ids.append(token_ids[i])
            in_noise = False

    # Add final sentinel to target
    target_ids.append(SENTINEL_START_ID + sentinel_idx)

    # Add EOS
    input_ids.append(tokenizer.eos_token_id)
    target_ids.append(tokenizer.eos_token_id)

    return input_ids, target_ids


class SpanCorruptionDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length, noise_density, mean_noise_span_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length

        # Pre-tokenize all texts
        self.all_token_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            # Truncate to max_length to leave room for sentinels
            ids = ids[:max_length - 50]  # reserve space for sentinels
            if len(ids) >= 5:
                self.all_token_ids.append(ids)

        logger.info(f"SpanCorruptionDataset: {len(self.all_token_ids)} samples")

    def __len__(self):
        return len(self.all_token_ids)

    def __getitem__(self, idx):
        token_ids = self.all_token_ids[idx]
        input_ids, target_ids = create_span_corruption_example(
            token_ids, self.noise_density, self.mean_noise_span_length
        )

        # Truncate to max_length
        input_ids = input_ids[:self.max_length]
        target_ids = target_ids[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long),
        }


# ============================================================
# Data Collator (dynamic padding)
# ============================================================
class DataCollatorForSpanCorruption:
    def __init__(self, tokenizer, max_length):
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, features):
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]

        # Pad to max length in batch
        max_input_len = min(max(len(ids) for ids in input_ids_list), self.max_length)
        max_label_len = min(max(len(ids) for ids in labels_list), self.max_length)

        padded_inputs = []
        attention_masks = []
        padded_labels = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            # Pad input
            input_pad_len = max_input_len - len(input_ids)
            padded_input = torch.cat([input_ids, torch.full((input_pad_len,), self.pad_token_id, dtype=torch.long)])
            attn_mask = torch.cat([torch.ones(len(input_ids), dtype=torch.long), torch.zeros(input_pad_len, dtype=torch.long)])

            # Pad labels (-100 for padding)
            label_pad_len = max_label_len - len(labels)
            padded_label = torch.cat([labels, torch.full((label_pad_len,), -100, dtype=torch.long)])

            padded_inputs.append(padded_input)
            attention_masks.append(attn_mask)
            padded_labels.append(padded_label)

        return {
            "input_ids": torch.stack(padded_inputs),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(padded_labels),
        }


# ============================================================
# Metrics Logger
# ============================================================
class MetricsLogger(TrainerCallback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logs.append({"step": state.global_step, **logs})

    def on_train_end(self, args, state, control, **kwargs):
        with open(self.output_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        logger.info(f"Metrics log saved to {self.output_path}")


# ============================================================
# Build datasets
# ============================================================
logger.info("Building span corruption datasets...")
train_dataset = SpanCorruptionDataset(
    train_texts, tokenizer, MAX_LENGTH, NOISE_DENSITY, MEAN_NOISE_SPAN_LENGTH
)
val_dataset = SpanCorruptionDataset(
    val_texts, tokenizer, MAX_LENGTH, NOISE_DENSITY, MEAN_NOISE_SPAN_LENGTH
)

# Show example
if len(train_dataset) > 0:
    example = train_dataset[0]
    logger.info(f"Example input length: {len(example['input_ids'])}")
    logger.info(f"Example target length: {len(example['labels'])}")
    # Decode for inspection
    input_decoded = tokenizer.decode(example["input_ids"], skip_special_tokens=False)
    target_decoded = tokenizer.decode(example["labels"], skip_special_tokens=False)
    logger.info(f"Example input (first 200 chars): {input_decoded[:200]}")
    logger.info(f"Example target (first 200 chars): {target_decoded[:200]}")

# ============================================================
# Model
# ============================================================
logger.info(f"Loading model: {MODEL_NAME}")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

data_collator = DataCollatorForSpanCorruption(tokenizer, MAX_LENGTH)

# ============================================================
# Training
# ============================================================
args = Seq2SeqTrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    optim="adafactor",
    bf16=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    report_to="wandb",
    run_name="exp031_dapt_pretrain",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=SEED,
    dataloader_num_workers=4,
)

metrics_logger = MetricsLogger(str(RESULTS_DIR / "metrics_log.json"))

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[metrics_logger],
)

# Checkpoint auto-resume
ckpt_dir_path = Path(str(CHECKPOINT_DIR))
resume_ckpt = None
if ckpt_dir_path.exists():
    checkpoints = [d for d in ckpt_dir_path.iterdir() if d.name.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
        resume_ckpt = str(latest)
        logger.info(f"Auto-resuming from checkpoint: {resume_ckpt}")

logger.info("Starting DAPT pre-training...")
t0 = time.time()
if resume_ckpt:
    trainer.train(resume_from_checkpoint=resume_ckpt)
else:
    trainer.train()
elapsed = time.time() - t0
logger.info(f"DAPT training complete: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

# ============================================================
# Save
# ============================================================
logger.info(f"Saving DAPT model to {RESULTS_DIR}")
trainer.save_model(str(RESULTS_DIR))
tokenizer.save_pretrained(str(RESULTS_DIR))

# Also save last checkpoint model
last_ckpt = None
if ckpt_dir_path.exists():
    ckpts = sorted(ckpt_dir_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        last_ckpt = ckpts[-1]

last_model_dir = EXP_DIR / "results" / "dapt_model_last"
last_model_dir.mkdir(parents=True, exist_ok=True)
if last_ckpt:
    last_model = AutoModelForSeq2SeqLM.from_pretrained(str(last_ckpt))
    last_model.save_pretrained(str(last_model_dir))
    tokenizer.save_pretrained(str(last_model_dir))
    del last_model
    logger.info(f"Last model saved to {last_model_dir}")

metrics = trainer.evaluate()
logger.info(f"Final eval metrics: {metrics}")

with open(str(RESULTS_DIR / "final_metrics.json"), "w") as f:
    json.dump({"elapsed_s": elapsed, **metrics}, f, indent=2)

logger.info("Done.")
