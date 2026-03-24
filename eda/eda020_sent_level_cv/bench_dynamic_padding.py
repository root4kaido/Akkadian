"""
動的パディング vs 固定パディングの速度比較ベンチマーク
対象: exp034_st_pretrain fold3 last_model
eval_full_doc.py と同じデータ・推論設定で、パディング方式のみ変更して時間を比較
"""
import os
import re
import sys
import math
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import sacrebleu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(PROJECT_ROOT / "workspace" / "exp034_st_pretrain" / "results" / "fold3" / "last_model")
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"
FOLD = 3
MAX_LENGTH = 512
BATCH_SIZE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 前処理 (exp023相当)
# ============================================================
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002

def _decimal_to_fraction_approx(match):
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
            best_dist = dist
            best_frac = symbol
    if best_dist <= APPROX_TOLERANCE:
        return best_frac if int_part == 0 else f"{int_part} {best_frac}"
    return dec_str

def preprocess_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction_approx, text)
    return text

# ============================================================
# データ準備 (eval_full_doc.pyと同一)
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))

def simple_sentence_aligner(df, keep_oare_id=False):
    aligned_data = []
    for _, row in df.iterrows():
        src = str(row["transliteration"])
        tgt = str(row["translation"])
        tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split("\n") if s.strip()]
        entry_base = {"oare_id": row["oare_id"]} if keep_oare_id else {}
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({**entry_base, "transliteration": s, "translation": t})
        else:
            aligned_data.append({**entry_base, "transliteration": src, "translation": tgt})
    return pd.DataFrame(aligned_data)

# GroupKFold fold=3
akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_expanded = simple_sentence_aligner(train_df, keep_oare_id=True)
train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=train_expanded["akt_group"].values))
_, val_idx = splits[FOLD]
val_data = train_expanded.iloc[val_idx].copy()
logger.info(f"GroupKFold fold={FOLD}, val={len(val_data)} samples")

# sentence_aligned
sent_aligned = pd.read_csv(str(SENTENCE_ALIGNED_PATH))
oare_to_row = {row['oare_id']: row for _, row in train_df.iterrows()}
alignment_dict = {}
for oare_id, group in sent_aligned.groupby('oare_id'):
    group = group.sort_values('sent_idx')
    alignment_dict[oare_id] = [
        {'akk_segment': str(r['akk_segment']), 'eng_sentence': str(r['eng_sentence'])}
        for _, r in group.iterrows()
    ]
translit_to_oare = {str(row['transliteration']): oare_id for oare_id, row in oare_to_row.items()}

prefix = "translate Akkadian to English: "
sent_inputs, sent_refs = [], []
split_doc_ids = set()
for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    oare_id = translit_to_oare.get(translit)
    if oare_id and oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            split_doc_ids.add(idx)
            for b in boundaries:
                if b['akk_segment'].strip() and b['eng_sentence'].strip():
                    sent_inputs.append(prefix + preprocess_transliteration(b['akk_segment']))
                    sent_refs.append(b['eng_sentence'])

doc_inputs, doc_refs = [], []
for _, row in val_data.iterrows():
    translit = str(row['transliteration'])
    translation = str(row['translation'])
    if translit.strip() and translation.strip():
        doc_inputs.append(prefix + preprocess_transliteration(translit))
        doc_refs.append(translation)

logger.info(f"sent-CV: {len(sent_inputs)} sents, doc-CV: {len(doc_inputs)} docs")

# ============================================================
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info("Model loaded")

# ============================================================
# 方式A: 固定パディング (現行)
# ============================================================
class FixedPaddingDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, max_length=max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

def run_fixed(inputs, desc):
    ds = FixedPaddingDataset(inputs, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=4, early_stopping=True,
            )
            preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    return preds

# ============================================================
# 方式B: 動的パディング + 長さソート
# ============================================================
class DynamicPaddingDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length):
        self.items = []
        for t in texts:
            enc = tokenizer(t, max_length=max_length, truncation=True, return_tensors="pt")
            self.items.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

def dynamic_collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_id = tokenizer.pad_token_id or 0
    input_ids, attention_mask = [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask)}

def run_dynamic(inputs, desc):
    ds = DynamicPaddingDataset(inputs, tokenizer, MAX_LENGTH)
    # 長さソート → 元の順序復元
    lengths = [ds.items[i]["input_ids"].size(0) for i in range(len(ds))]
    sorted_indices = sorted(range(len(ds)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}

    sorted_ds = torch.utils.data.Subset(ds, sorted_indices)
    loader = DataLoader(sorted_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_fn)
    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=4, early_stopping=True,
            )
            sorted_preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    preds = [""] * len(inputs)
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    return preds

# ============================================================
# 後処理・評価
# ============================================================
def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return " ".join(words[:i+n])
    return text

def calc_metrics(preds, refs, label):
    preds_clean = [repeat_cleanup(p) for p in preds]
    chrf = sacrebleu.corpus_chrf(preds_clean, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds_clean, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    logger.info(f"  {label}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}")
    return geo

# ============================================================
# ベンチマーク実行
# ============================================================
# ウォームアップ (最初のGPU起動コストを除外)
logger.info("=== Warmup ===")
_ = run_fixed(sent_inputs[:4], "warmup")

# --- 固定パディング ---
logger.info("=== Fixed Padding (現行方式) ===")
torch.cuda.synchronize()
t0 = time.time()
sent_preds_fixed = run_fixed(sent_inputs, "fixed sent-CV")
torch.cuda.synchronize()
t1 = time.time()
doc_preds_fixed = run_fixed(doc_inputs, "fixed doc-CV")
torch.cuda.synchronize()
t2 = time.time()

fixed_sent_time = t1 - t0
fixed_doc_time = t2 - t1
fixed_total = t2 - t0

# --- 動的パディング ---
logger.info("=== Dynamic Padding + Length Sort ===")
torch.cuda.synchronize()
t0 = time.time()
sent_preds_dynamic = run_dynamic(sent_inputs, "dynamic sent-CV")
torch.cuda.synchronize()
t1 = time.time()
doc_preds_dynamic = run_dynamic(doc_inputs, "dynamic doc-CV")
torch.cuda.synchronize()
t2 = time.time()

dynamic_sent_time = t1 - t0
dynamic_doc_time = t2 - t1
dynamic_total = t2 - t0

# ============================================================
# 結果比較
# ============================================================
logger.info("=" * 60)
logger.info("=== 速度比較 ===")
logger.info(f"{'':20s} {'Fixed':>10s} {'Dynamic':>10s} {'Speedup':>10s}")
logger.info(f"{'sent-CV':20s} {fixed_sent_time:>9.1f}s {dynamic_sent_time:>9.1f}s {fixed_sent_time/dynamic_sent_time:>9.2f}x")
logger.info(f"{'doc-CV':20s} {fixed_doc_time:>9.1f}s {dynamic_doc_time:>9.1f}s {fixed_doc_time/dynamic_doc_time:>9.2f}x")
logger.info(f"{'Total':20s} {fixed_total:>9.1f}s {dynamic_total:>9.1f}s {fixed_total/dynamic_total:>9.2f}x")

# 入力長の統計
all_texts = sent_inputs + doc_inputs
all_lengths = [len(tokenizer(t, truncation=True, max_length=MAX_LENGTH)["input_ids"]) for t in all_texts]
logger.info(f"\n入力長統計: mean={np.mean(all_lengths):.0f}, median={np.median(all_lengths):.0f}, "
            f"max={np.max(all_lengths)}, min={np.min(all_lengths)} (MAX_LENGTH={MAX_LENGTH})")

# スコア一致確認
logger.info("\n=== スコア確認 (結果が一致するか) ===")
sg_f = calc_metrics(sent_preds_fixed, sent_refs, "fixed  sent-CV")
sg_d = calc_metrics(sent_preds_dynamic, sent_refs, "dynamic sent-CV")
dg_f = calc_metrics(doc_preds_fixed, doc_refs, "fixed  doc-CV")
dg_d = calc_metrics(doc_preds_dynamic, doc_refs, "dynamic doc-CV")

# 予測一致率
sent_match = sum(a == b for a, b in zip(sent_preds_fixed, sent_preds_dynamic)) / len(sent_preds_fixed) * 100
doc_match = sum(a == b for a, b in zip(doc_preds_fixed, doc_preds_dynamic)) / len(doc_preds_fixed) * 100
logger.info(f"\n予測一致率: sent={sent_match:.1f}%, doc={doc_match:.1f}%")
