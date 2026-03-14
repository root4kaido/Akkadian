"""
eda014: Doc-level評価
exp011モデルをdoc単位のval splitでgreedy推論し、
他ノートブック(llkh0a: geo_mean=43.46)と比較可能な数値を出す。

条件:
- train.csvをdoc単位でtrain_test_split(test_size=0.1, seed=42)
- PN/GNタグ付き入力
- greedy推論 (rp=1.2)
- sacrebleuでchrF++/BLEU/geo_mean計算
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
# Setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
EDA_DIR = RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(EDA_DIR / "eval_doc_level.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


# ============================================================
# Utilities
# ============================================================
def tag_transliteration(text: str, form_tag_dict: dict) -> str:
    tokens = text.split()
    return " ".join(
        f"{t}[{form_tag_dict[t]}]" if t in form_tag_dict else t
        for t in tokens
    )


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
        self.encodings = tokenizer(
            texts, max_length=max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# ============================================================
# Main
# ============================================================
def main():
    # Load model
    model_path = str(PROJECT_ROOT / "workspace" / "exp011_additional_data" / "results" / "best_model")
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    prefix = "translate Akkadian to English: "
    max_length = 512

    # Load PN/GN dict
    dict_path = PROJECT_ROOT / "workspace" / "exp011_additional_data" / "dataset" / "form_type_dict.json"
    with open(dict_path) as f:
        form_tag_dict = json.load(f)
    logger.info(f"Loaded form_tag_dict: {len(form_tag_dict)} entries")

    # ============================================================
    # Doc-level val split (same as llkh0a notebook)
    # ============================================================
    train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
    train_df = train_df[
        (train_df["transliteration"].astype(str).str.len() > 0)
        & (train_df["translation"].astype(str).str.len() > 0)
    ]

    _, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    val_df = val_df.reset_index(drop=True)
    logger.info(f"Doc-level val samples: {len(val_df)}")

    # Prepare doc-level inputs (full document, with PN/GN tags)
    doc_inputs = []
    doc_refs = []
    for _, row in val_df.iterrows():
        src = str(row["transliteration"])
        tgt = str(row["translation"])
        tagged = tag_transliteration(src, form_tag_dict)
        doc_inputs.append(prefix + tagged)
        doc_refs.append(tgt)

    N = len(doc_inputs)

    # Log input/ref length stats
    input_lens = [len(s.encode('utf-8')) for s in doc_inputs]
    ref_lens = [len(s) for s in doc_refs]
    logger.info(f"Input byte lengths: mean={np.mean(input_lens):.0f}, max={max(input_lens)}, >512: {sum(1 for l in input_lens if l > 512)}/{N}")
    logger.info(f"Ref char lengths: mean={np.mean(ref_lens):.0f}, max={max(ref_lens)}, >512: {sum(1 for l in ref_lens if l > 512)}/{N}")

    # ============================================================
    # Greedy decoding (rp=1.2)
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Greedy Decoding (doc-level, rp=1.2) ===")

    dataset = InferenceDataset(doc_inputs, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Greedy"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=max_length, num_beams=1, do_sample=False,
                repetition_penalty=1.2,
            )
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    preds_clean = [repeat_cleanup(p) for p in preds]

    # ============================================================
    # Evaluation
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Results ===")

    for label, pred_list in [("greedy_raw", preds), ("greedy_clean", preds_clean)]:
        chrf = sacrebleu.corpus_chrf(pred_list, [doc_refs], word_order=2)
        bleu = sacrebleu.corpus_bleu(pred_list, [doc_refs])
        geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
        rep_rate = 100 * sum(has_repetition(p) for p in pred_list) / N
        mean_len = np.mean([len(p) for p in pred_list])

        logger.info(
            f"  {label:15s}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, "
            f"geo_mean={geo:.2f}, rep={rep_rate:.1f}%, len={mean_len:.0f}"
        )

    # ============================================================
    # Comparison context
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Comparison ===")
    logger.info("  llkh0a notebook (LB~32): chrF++=57.02, BLEU=33.12, geo_mean=43.46")
    logger.info("  (Note: their score includes Trainer truncation bias)")
    logger.info("  Our sent-level best (exp012): geo_mean=34.47 (13-cand MBR)")

    # ============================================================
    # Sample outputs
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Sample Predictions (first 5) ===")
    chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
    for i in range(min(5, N)):
        score = float(chrf_metric.sentence_score(preds_clean[i], [doc_refs[i]]).score)
        logger.info(f"  [{i}] chrF++={score:.1f}")
        logger.info(f"    ref : {doc_refs[i][:150]}...")
        logger.info(f"    pred: {preds_clean[i][:150]}...")

    logger.info("=" * 60)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
