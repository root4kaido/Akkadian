"""
exp043_bolmo_1b: CausalLM用 sent-CV / doc-CV 評価スクリプト
eval_full_doc.py と同じ評価ロジック、Bolmo (CausalLM) 用に推論部分を変更

Usage:
  python workspace/exp043_bolmo_1b/src/eval_cv.py <model_path> <exp_name> --fold 3
  python workspace/exp043_bolmo_1b/src/eval_cv.py workspace/exp043_bolmo_1b/results/fold3/best_model exp043_bolmo_1b_best --fold 3
"""
import os
import re
import sys
import math
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import sacrebleu

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("exp_name", type=str)
parser.add_argument("--fold", type=int, default=3)
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (CausalLM, default=1)")
parser.add_argument("--int8", action="store_true", help="Load model in int8 quantization (for Kaggle submission sim)")
args = parser.parse_args()

MODEL_PATH = args.model_path
EXP_NAME = args.exp_name
FOLD = args.fold
MAX_NEW_TOKENS = args.max_new_tokens
BATCH_SIZE = args.batch_size
USE_INT8 = args.int8

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXP_DIR / "results" / f"fold{FOLD}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Preprocessing (same as exp023 / train.py)
# ============================================================
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


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
            best_dist, best_frac = dist, symbol
    if best_dist <= APPROX_TOLERANCE:
        return best_frac if int_part == 0 else f"{int_part} {best_frac}"
    return dec_str


def preprocess_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction_approx, text)
    return text


# ============================================================
# Data: GroupKFold val split
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

# ============================================================
# sentence_aligned.csv for sent-CV
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
# --- sent-CV ---
sent_sources = []
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
                    sent_sources.append(preprocess_transliteration(akk_seg))
                    sent_refs.append(eng_sent)

# --- doc-CV ---
doc_sources = []
doc_refs = []

for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    translation = str(row['translation'])
    if translit.strip() and translation.strip():
        doc_sources.append(preprocess_transliteration(translit))
        doc_refs.append(translation)

logger.info(f"[{EXP_NAME}] sent-CV: {len(sent_sources)} sents from {len(split_doc_ids)} docs")
logger.info(f"[{EXP_NAME}] doc-CV: {len(doc_sources)} docs (no truncation)")

# ============================================================
# Model (CausalLM)
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# Bolmo already has pad_token='<pad>'(id=0) != eos_token='<bos>'(id=1)

if USE_INT8:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    logger.info(f"[{EXP_NAME}] CausalLM model loaded in INT8 from {MODEL_PATH}")
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    logger.info(f"[{EXP_NAME}] CausalLM model loaded in BF16 from {MODEL_PATH}")
model.eval()


# ============================================================
# Inference (CausalLM generation)
# ============================================================
def manual_generate(input_ids, max_new_tokens, eos_token_id):
    """Manual autoregressive generation bypassing Bolmo's custom generate().

    Right-pads to multiples of 64 (xlstm requirement) and passes attention_mask.
    """
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        seq_len = generated.shape[1]
        # Right-pad to multiple of 64 for xlstm (same as training)
        padded_len = ((seq_len + 63) // 64) * 64
        if padded_len > seq_len:
            pad_size = padded_len - seq_len
            padded_ids = torch.cat([
                generated,
                torch.full((1, pad_size), tokenizer.pad_token_id, dtype=generated.dtype, device=generated.device),
            ], dim=1)
            attn_mask = torch.cat([
                torch.ones((1, seq_len), dtype=torch.long, device=generated.device),
                torch.zeros((1, pad_size), dtype=torch.long, device=generated.device),
            ], dim=1)
        else:
            padded_ids = generated
            attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=generated.device)

        outputs = model(padded_ids, attention_mask=attn_mask)
        # Take logits at the last REAL token position (not the padded position)
        next_token_logits = outputs.logits[0, seq_len - 1, :]
        next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)

        if next_token.item() == eos_token_id:
            break

        generated = torch.cat([generated, next_token], dim=1)

    return generated


def run_inference(sources, desc):
    preds = []
    for src in tqdm(sources, desc=desc):
        prompt = f"Translate Akkadian to English.\nSource: {src}\nTarget: "
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = manual_generate(input_ids, MAX_NEW_TOKENS, tokenizer.eos_token_id)

        generated_ids = outputs[0][prompt_len:]
        pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        preds.append(pred)

        # Debug: print first 5 predictions
        if len(preds) <= 5:
            logger.info(f"  [sample {len(preds)}] prompt_len={prompt_len}, gen_len={len(generated_ids)}, pred='{pred[:200]}'")

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
# Run
# ============================================================
sent_preds = run_inference(sent_sources, f"{EXP_NAME} sent-CV")
doc_preds = run_inference(doc_sources, f"{EXP_NAME} doc-CV")

# ============================================================
# Metrics
# ============================================================
logger.info("=" * 60)
logger.info(f"=== {EXP_NAME} ===")

sc, sb, sg, sr = calc_metrics(sent_preds, sent_refs, f"sent-CV ({len(sent_sources)} sents)")
dc, db, dg, dr = calc_metrics(doc_preds, doc_refs, f"doc-CV  ({len(doc_sources)} docs, no trunc)")

# Save metrics
with open(str(RESULTS_DIR / "cv_metrics.json"), "w") as f:
    json.dump({
        "exp_name": EXP_NAME,
        "fold": FOLD,
        "sent_cv": {"chrf": round(sc, 2), "bleu": round(sb, 2), "geo": round(sg, 2), "rep": round(sr, 1), "n": len(sent_sources)},
        "doc_cv": {"chrf": round(dc, 2), "bleu": round(db, 2), "geo": round(dg, 2), "rep": round(dr, 1), "n": len(doc_sources)},
    }, f, indent=2, ensure_ascii=False)

# Save predictions
sent_pred_df = pd.DataFrame({
    "source": sent_sources,
    "reference": sent_refs,
    "prediction_raw": sent_preds,
    "prediction_clean": [repeat_cleanup(p) for p in sent_preds],
})
sent_pred_df.to_csv(str(RESULTS_DIR / f"{EXP_NAME}_sent_predictions.csv"), index=False)

doc_pred_df = pd.DataFrame({
    "source": doc_sources,
    "reference": doc_refs,
    "prediction_raw": doc_preds,
    "prediction_clean": [repeat_cleanup(p) for p in doc_preds],
})
doc_pred_df.to_csv(str(RESULTS_DIR / f"{EXP_NAME}_doc_predictions.csv"), index=False)

# Also append to full_comparison_results.txt (same format as eval_full_doc.py)
comparison_path = PROJECT_ROOT / "eda" / "eda020_sent_level_cv" / "full_comparison_results.txt"
with open(str(comparison_path), "a") as f:
    f.write(f"{EXP_NAME}\tsent-CV\tgeo={sg:.2f}\tchrF++={sc:.2f}\tBLEU={sb:.2f}\trep={sr:.1f}%\n")
    f.write(f"{EXP_NAME}\tdoc-CV\tgeo={dg:.2f}\tchrF++={dc:.2f}\tBLEU={db:.2f}\trep={dr:.1f}%\n")

logger.info(f"Results saved to {RESULTS_DIR}")
logger.info("Done.")
