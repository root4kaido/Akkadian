"""
exp010: 文レベル推論 + 評価（PN/GNタグ付加対応）
入力アッカド語を先頭200バイトでカットし、PN/GNタグを付加してから推論。
greedy / MBR の2手法で推論・評価する。
MBRDecoderはexp007から再利用。
"""
import json
import os
import re
import sys
import math
import yaml
import logging
from pathlib import Path
from typing import List

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
PROJECT_ROOT = EXP_DIR.parent.parent
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


def tag_transliteration(text: str, form_tag_dict: dict) -> str:
    """翻字テキストのトークンにPN/GNタグを付加する。"""
    tokens = text.split()
    tagged_tokens = []
    for token in tokens:
        tag = form_tag_dict.get(token)
        if tag:
            tagged_tokens.append(f"{token}[{tag}]")
        else:
            tagged_tokens.append(token)
    return " ".join(tagged_tokens)


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
# Import MBR from exp007
# ============================================================
sys.path.insert(0, str(PROJECT_ROOT / "workspace" / "exp007_mbr_postprocess" / "src"))
from infer_mbr import MBRDecoder, repeat_cleanup


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
    model_path = str(EXP_DIR / "results" / "best_model")
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    # Prepare validation data (same split as training)
    prefix = config["model"].get("prefix", "translate Akkadian to English: ")

    # Load PN/GN tag dictionary
    dict_path = EXP_DIR / "dataset" / "form_type_dict.json"
    with open(dict_path) as f:
        form_tag_dict = json.load(f)
    logger.info(f"Loaded form_tag_dict: {len(form_tag_dict)} entries")

    train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
    train_df = train_df[
        (train_df["transliteration"].astype(str).str.len() > 0)
        & (train_df["translation"].astype(str).str.len() > 0)
    ]

    seed = config["training"]["seed"]
    val_ratio = config["training"].get("val_ratio", 0.1)
    actual_train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=seed)
    val_df = val_df.reset_index(drop=True)

    # Build sentence-level pairs (with PN/GN tagging)
    sent_inputs = []
    sent_refs = []
    for _, row in val_df.iterrows():
        translit = str(row["transliteration"])
        translation = str(row["translation"])
        first_sent_eng = extract_first_sentence(translation)
        first_sent_akk = truncate_akkadian_to_sentence(translit, max_bytes=200)
        # Apply PN/GN tagging
        first_sent_akk_tagged = tag_transliteration(first_sent_akk, form_tag_dict)
        if len(first_sent_eng.strip()) > 0 and len(first_sent_akk_tagged.strip()) > 0:
            sent_inputs.append(prefix + first_sent_akk_tagged)
            sent_refs.append(first_sent_eng)

    logger.info(f"Sentence-level val samples: {len(sent_inputs)}")

    max_length = config["model"].get("max_length", 512)
    max_input_length = config["model"].get("max_length", 512)

    # Build dataset
    dataset = InferenceDataset(sent_inputs, tokenizer, max_input_length)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

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
                repetition_penalty=1.2,
            )
            greedy_preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # ============================================================
    # 2. MBR decoding
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Sentence-level MBR ===")

    # MBR config compatible with exp007's MBRDecoder
    mbr_config = {
        "inference": {
            "max_length": max_length,
            "mbr": {
                "num_beams": 4,
                "num_sampling": 2,
                "sampling_temperature": 1.0,
                "sampling_top_p": 0.95,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0,
            },
        },
        "model": config["model"],
    }
    mbr_decoder = MBRDecoder(model, tokenizer, mbr_config)

    mbr_preds = []
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
    # 3. Greedy + repeated_removal
    # ============================================================
    greedy_clean = [repeat_cleanup(p) for p in greedy_preds]
    mbr_clean = [repeat_cleanup(p) for p in mbr_preds]

    # ============================================================
    # Evaluate all methods
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Sentence-level Results ===")
    logger.info("=" * 60)
    evaluate_cv(greedy_preds, sent_refs, "greedy_raw")
    evaluate_cv(mbr_preds, sent_refs, "mbr_raw")
    evaluate_cv(greedy_clean, sent_refs, "greedy_clean")
    evaluate_cv(mbr_clean, sent_refs, "mbr_clean")

    # ============================================================
    # Save CSV
    # ============================================================
    results_df = pd.DataFrame({
        "input": sent_inputs,
        "reference": sent_refs,
        "greedy_pred": greedy_preds,
        "mbr_pred": mbr_preds,
        "greedy_clean": greedy_clean,
        "mbr_clean": mbr_clean,
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
                             ("greedy_clean", greedy_clean), ("mbr_clean", mbr_clean)]:
        n_rep = sum(1 for p in preds if has_repetition(str(p)))
        logger.info(f"Repetitions in {col_name}: {n_rep}/{len(preds)} ({100*n_rep/len(preds):.1f}%)")


if __name__ == "__main__":
    main()
