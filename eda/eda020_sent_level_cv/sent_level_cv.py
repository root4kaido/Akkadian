"""
eda020: sentence_aligned.csvを使ってvalドキュメントを文分割し、文単位でCV計測
- exp016モデルを使用
- val内の6文以下ドキュメントを文分割して個別に推論
- 文レベルCVとdocレベルCVの差を定量化
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(PROJECT_ROOT / "workspace" / "exp016_byt5_base" / "results" / "best_model")
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = 512
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Data: exp016と同一のval splitを再現
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
logger.info(f"Val docs: {len(val_data)}")

# ============================================================
# sentence_aligned.csv をロード
# ============================================================
sent_aligned = pd.read_csv(str(SENTENCE_ALIGNED_PATH))
logger.info(f"Sentence aligned: {len(sent_aligned)} rows, {sent_aligned['oare_id'].nunique()} docs")

# oare_id → train_df行
oare_to_row = {}
for _, row in train_df.iterrows():
    oare_to_row[row['oare_id']] = row

# oare_id → alignment boundaries
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

# ============================================================
# Val docsをsentence_aligned情報で文分割
# ============================================================
# transliteration → oare_id のマッピング
translit_to_oare = {}
for oare_id, row in oare_to_row.items():
    translit_to_oare[str(row['transliteration'])] = oare_id

sent_inputs = []
sent_refs = []
sent_doc_ids = []  # どのval docから来たか
sent_indices = []  # 何文目か

doc_level_inputs = []
doc_level_refs = []

n_split_docs = 0
n_unsplit_docs = 0
n_split_sents = 0

prefix = "translate Akkadian to English: "

for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    translation = str(row['translation'])

    # oare_idを特定
    oare_id = translit_to_oare.get(translit)

    if oare_id and oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            # 文分割して個別に追加
            n_split_docs += 1
            for b in boundaries:
                akk_seg = b['akk_segment']
                eng_sent = b['eng_sentence']
                if akk_seg.strip() and eng_sent.strip():
                    sent_inputs.append(prefix + akk_seg)
                    sent_refs.append(eng_sent)
                    sent_doc_ids.append(idx)
                    sent_indices.append(b['sent_idx'])
                    n_split_sents += 1
            continue

    # alignment情報がない or 7文以上 → docそのまま（1文抽出）
    n_unsplit_docs += 1
    # 先頭200Bで切って最初の文を使う（従来方式）
    encoded = translit.encode('utf-8')
    if len(encoded) > 200:
        truncated = encoded[:200].decode('utf-8', errors='ignore')
        last_space = truncated.rfind(' ')
        akk = truncated[:last_space].strip() if last_space > 0 else truncated.strip()
    else:
        akk = translit

    m = re.search(r'^(.*?[.!?])(?:\s|$)', translation)
    eng = m.group(1).strip() if m else translation.strip()

    if akk.strip() and eng.strip():
        sent_inputs.append(prefix + akk)
        sent_refs.append(eng)
        sent_doc_ids.append(idx)
        sent_indices.append(0)

logger.info(f"Split docs: {n_split_docs} ({n_split_sents} sentences)")
logger.info(f"Unsplit docs (fallback): {n_unsplit_docs}")
logger.info(f"Total sentence-level eval samples: {len(sent_inputs)}")

# ============================================================
# Model load & inference
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info(f"Model loaded from {MODEL_PATH}")


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


dataset_eval = InferenceDataset(sent_inputs, tokenizer, MAX_LENGTH)
loader = DataLoader(dataset_eval, batch_size=4, shuffle=False)

preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Sent-level CV (beam4)"):
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


preds_clean = [repeat_cleanup(p) for p in preds]


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


# ============================================================
# Save predictions
# ============================================================
results_df = pd.DataFrame({
    "input": sent_inputs,
    "reference": sent_refs,
    "prediction_raw": preds,
    "prediction_clean": preds_clean,
    "doc_id": sent_doc_ids,
    "sent_idx": sent_indices,
})
results_df.to_csv(str(RESULTS_DIR / "sent_level_predictions.csv"), index=False)

# ============================================================
# Metrics: 全体
# ============================================================
logger.info("=" * 60)
logger.info(f"=== Sentence-level CV (beam4, exp016) ===")
logger.info(f"    Total: {len(preds)} samples ({n_split_docs} split docs + {n_unsplit_docs} unsplit docs)")

for plabel, pred_list in [("raw", preds), ("clean", preds_clean)]:
    chrf = sacrebleu.corpus_chrf(pred_list, [sent_refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(pred_list, [sent_refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in pred_list) / len(pred_list)
    logger.info(f"  {plabel:5s}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep_rate:.1f}%")

# ============================================================
# Metrics: split docsのみ（文分割されたもの）
# ============================================================
split_mask = [i for i, d in enumerate(sent_doc_ids) if d in
    [idx for idx, row in val_data.iterrows()
     if str(row['transliteration']) in translit_to_oare
     and translit_to_oare.get(str(row['transliteration'])) in alignment_dict
     and len(alignment_dict.get(translit_to_oare.get(str(row['transliteration'])), [])) <= 6]]

# Simpler approach: use the split tracking
split_doc_set = set()
for idx, row in val_data.iterrows():
    t = str(row['transliteration'])
    oid = translit_to_oare.get(t)
    if oid and oid in alignment_dict and len(alignment_dict[oid]) <= 6:
        split_doc_set.add(idx)

split_preds = [preds_clean[i] for i in range(len(preds_clean)) if sent_doc_ids[i] in split_doc_set]
split_refs = [sent_refs[i] for i in range(len(sent_refs)) if sent_doc_ids[i] in split_doc_set]

if split_preds:
    logger.info("")
    logger.info(f"=== Split docs only ({len(split_preds)} sentences from {len(split_doc_set)} docs) ===")
    chrf = sacrebleu.corpus_chrf(split_preds, [split_refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(split_preds, [split_refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in split_preds) / len(split_preds)
    logger.info(f"  clean: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep_rate:.1f}%")

# unsplit docsのみ
unsplit_preds = [preds_clean[i] for i in range(len(preds_clean)) if sent_doc_ids[i] not in split_doc_set]
unsplit_refs = [sent_refs[i] for i in range(len(sent_refs)) if sent_doc_ids[i] not in split_doc_set]

if unsplit_preds:
    logger.info(f"=== Unsplit docs only ({len(unsplit_preds)} samples) ===")
    chrf = sacrebleu.corpus_chrf(unsplit_preds, [unsplit_refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(unsplit_preds, [unsplit_refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in unsplit_preds) / len(unsplit_preds)
    logger.info(f"  clean: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep_rate:.1f}%")

# ============================================================
# 比較: 従来のdoc-level CV (exp016の値)
# ============================================================
logger.info("")
logger.info("=== Reference ===")
logger.info("  exp016 doc-level CV (clean): chrF++=53.54, BLEU=39.21, geo=45.78, rep=26.8%")
logger.info("  exp016 LB: 29.5")
logger.info(f"  CV/LB ratio (doc): {45.78/29.5:.2f}")

logger.info("\nDone.")
