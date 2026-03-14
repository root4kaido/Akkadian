"""
eda017: starterノートブック完全再現 (takamichitoda/dpc-starter-train)
- byt5-small, 20epoch, 双方向学習, simple_sentence_aligner
- LB=26のモデルのCV基準値を得る（CV-LBキャリブレーション）

唯一の変更: generation_max_length=512追加（starterのバグ修正）
"""
import os
import gc
import re
import sys
import json
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sacrebleu
import evaluate
from sklearn.model_selection import train_test_split
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from tqdm import tqdm

# ============================================================
EDA_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EDA_DIR.parent.parent
RESULTS_DIR = EDA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "train_starter.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Config（starterと完全同一）
# ============================================================
MODEL_NAME = "google/byt5-small"
MAX_LENGTH = 512
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42
OUTPUT_DIR = str(RESULTS_DIR / "byt5-starter-model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything(SEED)

# ============================================================
# Data（starterと完全同一）
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
logger.info(f"Original Train Data: {len(train_df)} docs")


def simple_sentence_aligner(df):
    aligned_data = []
    for idx, row in df.iterrows():
        src = str(row['transliteration'])
        tgt = str(row['translation'])
        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({'transliteration': s, 'translation': t})
        else:
            aligned_data.append({'transliteration': src, 'translation': tgt})
    return pd.DataFrame(aligned_data)


train_expanded = simple_sentence_aligner(train_df)
logger.info(f"Expanded Train Data: {len(train_expanded)} sentences")

dataset = Dataset.from_pandas(train_expanded)
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)


def create_bidirectional_data(dataset_split):
    df = dataset_split.to_pandas()
    df_fwd = df.copy()
    df_fwd['input_text'] = "translate Akkadian to English: " + df_fwd['transliteration'].astype(str)
    df_fwd['target_text'] = df_fwd['translation'].astype(str)
    df_bwd = df.copy()
    df_bwd['input_text'] = "translate English to Akkadian: " + df_bwd['translation'].astype(str)
    df_bwd['target_text'] = df_bwd['transliteration'].astype(str)
    df_combined = pd.concat([df_fwd, df_bwd], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return Dataset.from_pandas(df_combined)


def create_unidirectional_data(dataset_split):
    df = dataset_split.to_pandas()
    df['input_text'] = "translate Akkadian to English: " + df['transliteration'].astype(str)
    df['target_text'] = df['translation'].astype(str)
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
# Training（starterと同一 + generation_max_length修正）
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
    chrf = metric_chrf.compute(predictions=decoded_preds, references=[[x] for x in decoded_labels])["score"]
    bleu = metric_bleu.compute(predictions=decoded_preds, references=[[x] for x in decoded_labels])["score"]
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    return {"chrf": round(chrf, 2), "bleu": round(bleu, 2), "geo_mean": round(geo_mean, 2)}


args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    optim="adafactor",
    label_smoothing_factor=0.2,
    fp16=False,
    per_device_train_batch_size=2,    # 4090はメモリ余裕あり、starterの1→2に
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,    # effective batch = 8（starterと同一）
    generation_max_length=MAX_LENGTH,  # ★starterのバグ修正
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="geo_mean",
    greater_is_better=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

logger.info("Starting Training (FP32 mode, starter reproduction)...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logger.info(f"Model saved to {OUTPUT_DIR}")

# ============================================================
# 我々のCV条件で評価
# ============================================================
logger.info("=" * 60)
logger.info("=== Our CV evaluation (sent-level, extract_first_sentence) ===")

sys.path.insert(0, str(PROJECT_ROOT / "workspace" / "exp007_mbr_postprocess" / "src"))
from infer_mbr import repeat_cleanup


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


# Use same val split as training (from split_datasets["test"])
val_data = split_datasets["test"].to_pandas()
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


model.eval()
dataset_eval = InferenceDataset(sent_inputs, tokenizer, MAX_LENGTH)
loader = DataLoader(dataset_eval, batch_size=4, shuffle=False)

preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Our CV eval"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        # starter-inferと完全同一の生成設定
        out = model.generate(
            input_ids=ids, attention_mask=mask,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
        )
        preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])

preds_clean = [repeat_cleanup(p) for p in preds]


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


for plabel, pred_list in [("raw", preds), ("clean", preds_clean)]:
    chrf = sacrebleu.corpus_chrf(pred_list, [sent_refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(pred_list, [sent_refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in pred_list) / len(pred_list)
    logger.info(
        f"  starter_{plabel:5s}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, "
        f"geo={geo:.2f}, rep={rep_rate:.1f}%"
    )

logger.info("")
logger.info("=== Calibration reference ===")
logger.info("  This model's LB = 26")
logger.info("  exp011_tag_clean CV = 33.45")
logger.info("  llkh0a_clean CV = 41.50, LB ~32")
