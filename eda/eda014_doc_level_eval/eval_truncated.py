"""
eda014: Truncated ref評価
doc-level推論結果に対して、参照テキストも512バイトにtruncateして評価する。
llkh0a notebook (geo_mean=43.46) との比較用。

Trainerのcompute_metricsはlabelsを512トークンにtruncateするため、
参照テキストが短くなりスコアが水増しされる。同条件で我々のモデルを評価する。
"""
import json
import math
import re
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sacrebleu
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EDA_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(EDA_DIR / "eval_truncated.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tag_transliteration(text, form_tag_dict):
    tokens = text.split()
    return " ".join(f"{t}[{form_tag_dict[t]}]" if t in form_tag_dict else t for t in tokens)


sys.path.insert(0, str(PROJECT_ROOT / "workspace" / "exp007_mbr_postprocess" / "src"))
from infer_mbr import repeat_cleanup


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def main():
    model_path = str(PROJECT_ROOT / "workspace" / "exp011_additional_data" / "results" / "best_model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    prefix = "translate Akkadian to English: "
    max_length = 512

    # PN/GN dict
    with open(PROJECT_ROOT / "workspace" / "exp011_additional_data" / "dataset" / "form_type_dict.json") as f:
        form_tag_dict = json.load(f)

    # Doc-level val split
    train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
    train_df = train_df[(train_df["transliteration"].astype(str).str.len() > 0) & (train_df["translation"].astype(str).str.len() > 0)]
    _, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    val_df = val_df.reset_index(drop=True)

    doc_inputs, doc_refs_full = [], []
    for _, row in val_df.iterrows():
        src = tag_transliteration(str(row["transliteration"]), form_tag_dict)
        doc_inputs.append(prefix + src)
        doc_refs_full.append(str(row["translation"]))

    N = len(doc_inputs)
    logger.info(f"Val samples: {N}")

    # Truncate refs to 512 ByT5 tokens (simulating Trainer behavior)
    doc_refs_trunc = []
    for r in doc_refs_full:
        toks = tokenizer(r, max_length=max_length, truncation=True)["input_ids"]
        doc_refs_trunc.append(tokenizer.decode(toks, skip_special_tokens=True))

    n_trunc = sum(1 for a, b in zip(doc_refs_full, doc_refs_trunc) if a != b)
    logger.info(f"Refs truncated: {n_trunc}/{N} ({100*n_trunc/N:.1f}%)")
    logger.info(f"Ref len full: mean={np.mean([len(r) for r in doc_refs_full]):.0f}")
    logger.info(f"Ref len trunc: mean={np.mean([len(r) for r in doc_refs_trunc]):.0f}")

    # Greedy decoding
    logger.info("=== Greedy Decoding (doc-level, rp=1.2) ===")
    dataset = InferenceDataset(doc_inputs, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Greedy"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(input_ids=ids, attention_mask=mask, max_length=max_length, num_beams=1, do_sample=False, repetition_penalty=1.2)
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    preds_clean = [repeat_cleanup(p) for p in preds]

    # Evaluate with both full and truncated refs
    logger.info("=" * 60)
    logger.info("=== Results ===")

    for ref_label, refs in [("full_ref", doc_refs_full), ("trunc_ref (512 tokens)", doc_refs_trunc)]:
        logger.info(f"--- {ref_label} ---")
        for pred_label, pred_list in [("greedy_raw", preds), ("greedy_clean", preds_clean)]:
            chrf = sacrebleu.corpus_chrf(pred_list, [refs], word_order=2)
            bleu = sacrebleu.corpus_bleu(pred_list, [refs])
            geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
            rep_rate = 100 * sum(has_repetition(p) for p in pred_list) / N
            logger.info(f"  {pred_label:15s}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep_rate:.1f}%")

    logger.info("=" * 60)
    logger.info("Comparison: llkh0a (LB~32): chrF++=57.02, BLEU=33.12, geo=43.46 (trunc ref)")
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
