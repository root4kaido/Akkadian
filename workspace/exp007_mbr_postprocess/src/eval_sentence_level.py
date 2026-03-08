"""
exp007: 文レベル推論 + 評価
入力アッカド語を先頭200バイトでカット、参照英語の最初の文を抽出し、
greedy / MBR / greedy+post / MBR+post の4手法で推論・評価する。
"""
import os
import re
import sys
import math
import yaml
import logging
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import sacrebleu
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# Setup
# ============================================================
EXP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(RESULTS_DIR / "eval_sentence.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

with open(EXP_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ============================================================
# Sentence extraction utilities
# ============================================================
def extract_first_sentence(text: str) -> str:
    """英語テキストの最初の文を抽出"""
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    if m:
        return m.group(1).strip()
    return str(text).strip()


def truncate_akkadian_to_sentence(translit: str, max_bytes: int = 200) -> str:
    """アッカド語transliterationを先頭max_bytesバイトでカット（スペース境界）"""
    encoded = str(translit).encode('utf-8')
    if len(encoded) <= max_bytes:
        return str(translit)
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space].strip()
    return truncated.strip()


# ============================================================
# Dataset
# ============================================================
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, max_length=max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# ============================================================
# Import post-processing from infer_mbr
# ============================================================
sys.path.insert(0, str(EXP_DIR / "src"))
from infer_mbr import (
    OALexiconPostProcessor,
    TranslationMemory,
    repeat_cleanup,
    MBRDecoder,
)


# ============================================================
# Evaluation
# ============================================================
def evaluate_cv(preds, refs, label=""):
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    logger.info(f"CV {label}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo_mean={geo:.2f}")
    return chrf.score, bleu.score, geo


# ============================================================
# Main
# ============================================================
def main():
    # Load model
    model_path = str(EXP_DIR / config["model"]["checkpoint"])
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    # Prepare validation data (same split as exp005)
    prefix = config["model"].get("prefix", "translate Akkadian to English: ")
    if not prefix:
        prefix = "translate Akkadian to English: "

    train_df = pd.read_csv(str(EXP_DIR / config["data"]["train_path"]))
    train_df = train_df[
        (train_df["transliteration"].astype(str).str.len() > 0)
        & (train_df["translation"].astype(str).str.len() > 0)
    ]

    seed = config["data"]["seed"]
    val_ratio = config["data"].get("val_ratio", 0.1)
    actual_train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=seed)
    actual_train_df = actual_train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Build sentence-level pairs
    sent_inputs = []
    sent_refs = []
    sent_translits_full = []  # full translit for TM lookup
    for _, row in val_df.iterrows():
        translit = str(row["transliteration"])
        translation = str(row["translation"])
        first_sent_eng = extract_first_sentence(translation)
        first_sent_akk = truncate_akkadian_to_sentence(translit, max_bytes=200)
        if len(first_sent_eng.strip()) > 0 and len(first_sent_akk.strip()) > 0:
            sent_inputs.append(prefix + first_sent_akk)
            sent_refs.append(first_sent_eng)
            sent_translits_full.append(translit)

    logger.info(f"Sentence-level val samples: {len(sent_inputs)}")

    max_length = config["model"].get("params", {}).get("max_output_length", 512)

    # Initialize post-processors
    oa_proc = None
    if config["postprocess"]["oa_lexicon"]["enabled"]:
        oa_proc = OALexiconPostProcessor(
            lexicon_path=str(EXP_DIR / config["postprocess"]["oa_lexicon"]["path"]),
            train_df=actual_train_df,
        )

    tm = None
    if config["postprocess"]["translation_memory"]["enabled"]:
        tm = TranslationMemory(train_df=actual_train_df)

    use_repeated_removal = config["postprocess"]["repeated_removal"]

    # Build dataset
    max_input_length = config["model"].get("params", {}).get("max_input_length", 512)
    dataset = InferenceDataset(sent_inputs, tokenizer, max_input_length)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)  # sentence-level is shorter, can use larger batch

    # ============================================================
    # 1. Greedy decoding
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Sentence-level Greedy ===")
    greedy_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Greedy"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=1,
                do_sample=False,
                repetition_penalty=config["inference"]["mbr"].get("repetition_penalty", 1.2),
            )
            greedy_preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # ============================================================
    # 2. MBR decoding
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Sentence-level MBR ===")
    mbr_decoder = MBRDecoder(model, tokenizer, config)

    mbr_preds = []
    # MBR processes one sample at a time (batch=1 for candidate generation)
    for i in tqdm(range(len(sent_inputs)), desc="MBR"):
        enc = tokenizer(
            sent_inputs[i], max_length=max_input_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        results = mbr_decoder.generate_batch(input_ids, attention_mask)
        mbr_preds.append(results[0])

    # ============================================================
    # 3. Apply post-processing
    # ============================================================
    def apply_postprocess(preds, translits):
        results = []
        for pred, translit in zip(preds, translits):
            out = pred
            # Translation Memory (use full translit for lookup)
            if tm:
                tm_result = tm.lookup(translit)
                if tm_result is not None:
                    out = extract_first_sentence(tm_result)
                    results.append(out)
                    continue
            # OA Lexicon
            if oa_proc:
                out = oa_proc.process(translit, out)
            # Repeated removal
            if use_repeated_removal:
                out = repeat_cleanup(out)
            results.append(out)
        return results

    greedy_post = apply_postprocess(greedy_preds, sent_translits_full)
    mbr_post = apply_postprocess(mbr_preds, sent_translits_full)

    # ============================================================
    # Evaluate all 4 methods
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Sentence-level Results ===")
    logger.info("=" * 60)
    evaluate_cv(greedy_preds, sent_refs, "greedy_raw")
    evaluate_cv(mbr_preds, sent_refs, "mbr_raw")
    evaluate_cv(greedy_post, sent_refs, "greedy_post")
    evaluate_cv(mbr_post, sent_refs, "mbr_post")

    # ============================================================
    # Save CSV
    # ============================================================
    results_df = pd.DataFrame({
        "input": sent_inputs,
        "reference": sent_refs,
        "greedy_pred": greedy_preds,
        "mbr_pred": mbr_preds,
        "greedy_post_pred": greedy_post,
        "mbr_post_pred": mbr_post,
    })
    csv_path = RESULTS_DIR / "val_predictions_sentence.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path} ({len(results_df)} rows)")

    # Repetition stats
    def has_repetition(text, min_repeat=3):
        words = str(text).split()
        for i in range(len(words) - min_repeat):
            chunk = " ".join(words[i:i + min_repeat])
            rest = " ".join(words[i + min_repeat:])
            if chunk in rest:
                return True
        return False

    for col_name, preds in [("greedy", greedy_preds), ("mbr", mbr_preds),
                             ("greedy_post", greedy_post), ("mbr_post", mbr_post)]:
        n_rep = sum(1 for p in preds if has_repetition(str(p)))
        logger.info(f"Repetitions in {col_name}: {n_rep}/{len(preds)} ({100*n_rep/len(preds):.1f}%)")


if __name__ == "__main__":
    main()
