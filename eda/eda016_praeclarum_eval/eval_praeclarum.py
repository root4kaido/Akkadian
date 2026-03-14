"""
eda016: praeclarum/cuneiform モデル（T5-base 223M）を我々のCV条件で評価
- 文レベルCV（extract_first_sentence + truncate_akkadian_to_sentence 200bytes）
- seed=42, val 10%
- greedy推論 + repeat_cleanup
- PN/GNタグなし（praeclarumモデルはタグ非対応）
- 比較: eda015と同一条件
"""
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EDA_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(EDA_DIR / "eval_praeclarum.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def evaluate_preds(preds, refs, label):
    preds_clean = [repeat_cleanup(p) for p in preds]
    N = len(preds)
    for plabel, pred_list in [("raw", preds), ("clean", preds_clean)]:
        chrf = sacrebleu.corpus_chrf(pred_list, [refs], word_order=2)
        bleu = sacrebleu.corpus_bleu(pred_list, [refs])
        geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
        rep_rate = 100 * sum(has_repetition(p) for p in pred_list) / N
        mean_len = np.mean([len(p) for p in pred_list])
        logger.info(
            f"  {label}_{plabel:5s}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, "
            f"geo={geo:.2f}, rep={rep_rate:.1f}%, len={mean_len:.0f}"
        )
    return preds, preds_clean


def main():
    prefix = "translate Akkadian to English: "
    max_length = 512

    # Val split（eda015と同一条件）
    train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
    train_df = train_df[
        (train_df["transliteration"].astype(str).str.len() > 0)
        & (train_df["translation"].astype(str).str.len() > 0)
    ]
    _, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    val_df = val_df.reset_index(drop=True)

    sent_inputs, sent_refs = [], []
    for _, row in val_df.iterrows():
        t = str(row["transliteration"])
        tr = str(row["translation"])
        eng = extract_first_sentence(tr)
        akk = truncate_akkadian_to_sentence(t)
        if eng.strip() and akk.strip():
            sent_inputs.append(prefix + akk)
            sent_refs.append(eng)

    logger.info(f"Val samples: {len(sent_inputs)}")

    # praeclarum/cuneiform
    logger.info("=" * 60)
    logger.info("=== praeclarum/cuneiform (T5-base 223M) ===")
    tokenizer = AutoTokenizer.from_pretrained("praeclarum/cuneiform")
    model = AutoModelForSeq2SeqLM.from_pretrained("praeclarum/cuneiform").to(device)
    model.eval()

    dataset = InferenceDataset(sent_inputs, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="praeclarum greedy"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=max_length, num_beams=1, do_sample=False,
            )
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    _, preds_clean = evaluate_preds(preds, sent_refs, "praeclarum")

    # Sample predictions
    logger.info("")
    logger.info("=== Sample predictions (praeclarum) ===")
    for i in range(min(15, len(preds))):
        logger.info(f"  INPUT: {sent_inputs[i][len(prefix):][:100]}")
        logger.info(f"  REF:   {sent_refs[i][:100]}")
        logger.info(f"  PRED:  {preds[i][:100]}")
        logger.info("")

    # eda015比較表（手動追記用）
    logger.info("=" * 60)
    logger.info("=== Comparison with eda015 results ===")
    logger.info("  llkh0a_clean:      chrF++=50.44, BLEU=34.15, geo=41.50")
    logger.info("  exp011_tag_clean:  chrF++=42.23, BLEU=26.49, geo=33.45")
    logger.info("  exp011_notag_clean:chrF++=37.95, BLEU=20.89, geo=28.15")
    logger.info("  praeclarum_clean:  (see above)")


if __name__ == "__main__":
    main()
