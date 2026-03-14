"""
eda020: 複数モデルでsentence-level CV (split docsのみ) を比較
Usage: python eval_multi_models.py <model_path> <exp_name>
"""
import os
import re
import sys
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

MODEL_PATH = sys.argv[1]
EXP_NAME = sys.argv[2]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Data: 同一val split
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))


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


train_expanded = simple_sentence_aligner(train_df)
dataset = Dataset.from_pandas(train_expanded)
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)
val_data = split_datasets["test"].to_pandas()

# ============================================================
# sentence_aligned.csv
# ============================================================
sent_aligned = pd.read_csv(str(SENTENCE_ALIGNED_PATH))

oare_to_row = {}
for _, row in train_df.iterrows():
    oare_to_row[row['oare_id']] = row

alignment_dict = {}
for oare_id, group in sent_aligned.groupby('oare_id'):
    group = group.sort_values('sent_idx')
    boundaries = []
    for _, row in group.iterrows():
        boundaries.append({
            'sent_idx': int(row['sent_idx']),
            'akk_segment': str(row['akk_segment']),
            'eng_sentence': str(row['eng_sentence']),
        })
    alignment_dict[oare_id] = boundaries

translit_to_oare = {}
for oare_id, row in oare_to_row.items():
    translit_to_oare[str(row['transliteration'])] = oare_id

# ============================================================
# Build eval samples: split docs only
# ============================================================
prefix = "translate Akkadian to English: "
split_inputs = []
split_refs = []

split_doc_set = set()
for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    oare_id = translit_to_oare.get(translit)
    if oare_id and oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            split_doc_set.add(idx)
            for b in boundaries:
                akk_seg = b['akk_segment']
                eng_sent = b['eng_sentence']
                if akk_seg.strip() and eng_sent.strip():
                    split_inputs.append(prefix + akk_seg)
                    split_refs.append(eng_sent)

logger.info(f"[{EXP_NAME}] Split docs: {len(split_doc_set)}, sentences: {len(split_inputs)}")

# ============================================================
# Model & inference
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info(f"[{EXP_NAME}] Model loaded from {MODEL_PATH}")


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


dataset_eval = InferenceDataset(split_inputs, tokenizer, MAX_LENGTH)
loader = DataLoader(dataset_eval, batch_size=4, shuffle=False)

preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc=f"{EXP_NAME} sent-CV"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model.generate(
            input_ids=ids, attention_mask=mask,
            max_length=MAX_LENGTH, num_beams=4, early_stopping=True,
        )
        preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])


def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return " ".join(words[:i+n])
    return text


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


preds_clean = [repeat_cleanup(p) for p in preds]

# ============================================================
# Metrics
# ============================================================
chrf = sacrebleu.corpus_chrf(preds_clean, [split_refs], word_order=2)
bleu = sacrebleu.corpus_bleu(preds_clean, [split_refs])
geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
rep_rate = 100 * sum(has_repetition(p) for p in preds_clean) / len(preds_clean)

logger.info(f"[{EXP_NAME}] RESULT: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep_rate:.1f}%")

# Save to file
with open(str(RESULTS_DIR / "multi_model_results.txt"), "a") as f:
    f.write(f"{EXP_NAME}\tchrF++={chrf.score:.2f}\tBLEU={bleu.score:.2f}\tgeo={geo:.2f}\trep={rep_rate:.1f}%\n")
