"""
eda020: sent-CV vs doc-CV（切り詰めなし）を複数モデルで比較
- sent-CV: 6文以下docをsentence_aligned.csvで分解して文単位評価
- doc-CV: ドキュメント全体を入力し、翻訳全体と比較（truncationなし）
Usage: python eval_full_doc.py <model_path> <exp_name> [--preprocess none|exp022|exp023] [--fold N]
"""
import os
import re
import sys
import math
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("exp_name", type=str)
parser.add_argument("--preprocess", type=str, default="none",
                    choices=["none", "exp022", "exp023"],
                    help="Transliteration preprocessing to apply (default: none)")
parser.add_argument("--fold", type=int, default=-1,
                    help="GroupKFold fold index (0-4). -1 = random split (default)")
args = parser.parse_args()

MODEL_PATH = args.model_path
EXP_NAME = args.exp_name
PREPROCESS = args.preprocess
FOLD = args.fold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"

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
# Transliteration preprocessing (matches training preprocessing)
# ============================================================
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# exp022: 完全一致小数→分数
DECIMAL_TO_FRACTION = {
    "0.5": "½", "0.25": "¼", "0.3333": "⅓", "0.6666": "⅔",
    "0.8333": "⅚", "0.75": "¾", "0.1666": "⅙", "0.625": "⅝",
}

# exp023: 近似マッチ小数→分数
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


def preprocess_transliteration(text: str) -> str:
    """Apply transliteration preprocessing matching the training config."""
    if PREPROCESS == "none":
        return text
    if not isinstance(text, str) or not text.strip():
        return text

    # Common to exp022 and exp023
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)

    if PREPROCESS == "exp022":
        for decimal, fraction in sorted(DECIMAL_TO_FRACTION.items(), key=lambda x: -len(x[0])):
            text = text.replace(decimal, fraction)
    elif PREPROCESS == "exp023":
        # Turkish character normalization
        text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
        text = re.sub(r'\(ki\)', '{ki}', text)
        text = re.sub(r'\d+\.\d+', _decimal_to_fraction_approx, text)

    return text

# ============================================================
# Data: val split (random or GroupKFold)
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))


def simple_sentence_aligner(df, keep_oare_id=False):
    aligned_data = []
    for idx, row in df.iterrows():
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


if FOLD >= 0:
    # GroupKFold
    from sklearn.model_selection import GroupKFold
    akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
    oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
    train_expanded = simple_sentence_aligner(train_df, keep_oare_id=True)
    train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(train_expanded, groups=train_expanded["akt_group"].values))
    _, val_idx = splits[FOLD]
    val_data = train_expanded.iloc[val_idx].copy()
    logger.info(f"GroupKFold fold={FOLD}, val={len(val_data)} samples")
    logger.info(f"Val groups: {val_data['akt_group'].value_counts().to_dict()}")
else:
    # Original random split
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
# Build eval samples
# ============================================================
prefix = "translate Akkadian to English: "

# --- sent-CV: split docs (6文以下) ---
sent_inputs = []
sent_refs = []
split_doc_ids = set()

for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    oare_id = translit_to_oare.get(translit)
    if oare_id and oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            split_doc_ids.add(idx)
            for b in boundaries:
                akk_seg = b['akk_segment']
                eng_sent = b['eng_sentence']
                if akk_seg.strip() and eng_sent.strip():
                    sent_inputs.append(prefix + preprocess_transliteration(akk_seg))
                    sent_refs.append(eng_sent)

# --- doc-CV: 全val docs、切り詰めなし ---
doc_inputs = []
doc_refs = []

for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    translation = str(row['translation'])
    if translit.strip() and translation.strip():
        doc_inputs.append(prefix + preprocess_transliteration(translit))
        doc_refs.append(translation)

logger.info(f"[{EXP_NAME}] preprocess={PREPROCESS}")
logger.info(f"[{EXP_NAME}] sent-CV: {len(sent_inputs)} sents from {len(split_doc_ids)} docs")
logger.info(f"[{EXP_NAME}] doc-CV: {len(doc_inputs)} docs (no truncation)")

# ============================================================
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info(f"[{EXP_NAME}] Model loaded")


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


def run_inference(inputs, desc):
    ds = InferenceDataset(inputs, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
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


def calc_metrics(preds, refs, label):
    preds_clean = [repeat_cleanup(p) for p in preds]
    chrf = sacrebleu.corpus_chrf(preds_clean, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds_clean, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in preds_clean) / len(preds_clean)
    logger.info(f"  {label}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep_rate:.1f}%")
    return chrf.score, bleu.score, geo, rep_rate


# ============================================================
# Run inference
# ============================================================
sent_preds = run_inference(sent_inputs, f"{EXP_NAME} sent-CV")
doc_preds = run_inference(doc_inputs, f"{EXP_NAME} doc-CV")

# ============================================================
# Metrics
# ============================================================
logger.info("=" * 60)
logger.info(f"=== {EXP_NAME} ===")

sc, sb, sg, sr = calc_metrics(sent_preds, sent_refs, f"sent-CV ({len(sent_inputs)} sents)")
dc, db, dg, dr = calc_metrics(doc_preds, doc_refs, f"doc-CV  ({len(doc_inputs)} docs, no trunc)")

# Save metrics
prep_tag = f"(prep={PREPROCESS})" if PREPROCESS != "none" else ""
with open(str(RESULTS_DIR / "full_comparison_results.txt"), "a") as f:
    f.write(f"{EXP_NAME}{prep_tag}\tsent-CV\tgeo={sg:.2f}\tchrF++={sc:.2f}\tBLEU={sb:.2f}\trep={sr:.1f}%\n")
    f.write(f"{EXP_NAME}{prep_tag}\tdoc-CV\tgeo={dg:.2f}\tchrF++={dc:.2f}\tBLEU={db:.2f}\trep={dr:.1f}%\n")

# Save predictions to CSV
sent_preds_clean = [repeat_cleanup(p) for p in sent_preds]
sent_pred_df = pd.DataFrame({
    "input": sent_inputs,
    "reference": sent_refs,
    "prediction_raw": sent_preds,
    "prediction_clean": sent_preds_clean,
})
sent_csv_path = RESULTS_DIR / f"{EXP_NAME}_sent_predictions.csv"
sent_pred_df.to_csv(str(sent_csv_path), index=False)
logger.info(f"Sent predictions saved to {sent_csv_path}")

doc_preds_clean = [repeat_cleanup(p) for p in doc_preds]
doc_pred_df = pd.DataFrame({
    "input": doc_inputs,
    "reference": doc_refs,
    "prediction_raw": doc_preds,
    "prediction_clean": doc_preds_clean,
})
doc_csv_path = RESULTS_DIR / f"{EXP_NAME}_doc_predictions.csv"
doc_pred_df.to_csv(str(doc_csv_path), index=False)
logger.info(f"Doc predictions saved to {doc_csv_path}")

# Save metrics to JSON
import json
metrics_dict = {
    "exp_name": EXP_NAME,
    "preprocess": PREPROCESS,
    "fold": FOLD,
    "sent_cv": {"chrf": round(sc, 2), "bleu": round(sb, 2), "geo": round(sg, 2), "rep": round(sr, 1), "n": len(sent_inputs)},
    "doc_cv": {"chrf": round(dc, 2), "bleu": round(db, 2), "geo": round(dg, 2), "rep": round(dr, 1), "n": len(doc_inputs)},
}
metrics_json_path = RESULTS_DIR / f"{EXP_NAME}_metrics.json"
with open(str(metrics_json_path), "w") as f:
    json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
logger.info(f"Metrics saved to {metrics_json_path}")

logger.info("Done.")
