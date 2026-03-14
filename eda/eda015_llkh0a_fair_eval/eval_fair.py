"""
eda015: llkh0aモデルを我々のCV条件で公平に評価する
- 文レベルCV（extract_first_sentence + truncate_akkadian_to_sentence）
- greedy (rp=1.2) + repeat_cleanup
- PN/GNタグなし（llkh0aモデルはタグなし学習）
- 比較対象: 我々のexp011モデル（タグあり/なし）
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
        logging.FileHandler(EDA_DIR / "eval_fair.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# repeat_cleanup
sys.path.insert(0, str(PROJECT_ROOT / "workspace" / "exp007_mbr_postprocess" / "src"))
from infer_mbr import repeat_cleanup


# ============================================================
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


def tag_transliteration(text: str, form_tag_dict: dict) -> str:
    tokens = text.split()
    return " ".join(
        f"{t}[{form_tag_dict[t]}]" if t in form_tag_dict else t
        for t in tokens
    )


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


def evaluate_model(model, tokenizer, inputs, refs, label, max_length=512):
    """Greedy推論 + 評価"""
    dataset = InferenceDataset(inputs, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=label):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=max_length, num_beams=1, do_sample=False,
                repetition_penalty=1.2,
            )
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

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

    # ============================================================
    # Val split（同一条件: seed=42, test_size=0.1）
    # ============================================================
    train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
    train_df = train_df[
        (train_df["transliteration"].astype(str).str.len() > 0)
        & (train_df["translation"].astype(str).str.len() > 0)
    ]
    _, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    val_df = val_df.reset_index(drop=True)

    # 文レベルペア（タグなし）
    sent_inputs_notag, sent_refs = [], []
    for _, row in val_df.iterrows():
        t = str(row["transliteration"])
        tr = str(row["translation"])
        eng = extract_first_sentence(tr)
        akk = truncate_akkadian_to_sentence(t)
        if eng.strip() and akk.strip():
            sent_inputs_notag.append(prefix + akk)
            sent_refs.append(eng)

    # 文レベルペア（タグあり — 我々のモデル用）
    dict_path = PROJECT_ROOT / "workspace" / "exp011_additional_data" / "dataset" / "form_type_dict.json"
    with open(dict_path) as f:
        form_tag_dict = json.load(f)

    sent_inputs_tagged = []
    for _, row in val_df.iterrows():
        t = str(row["transliteration"])
        tr = str(row["translation"])
        eng = extract_first_sentence(tr)
        akk = tag_transliteration(truncate_akkadian_to_sentence(t), form_tag_dict)
        if eng.strip() and akk.strip():
            sent_inputs_tagged.append(prefix + akk)

    N = len(sent_refs)
    logger.info(f"Val samples: {N}")

    # ============================================================
    # Model 1: llkh0a（タグなし）
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== llkh0a model (no PN/GN tags) ===")
    llkh0a_path = str(EDA_DIR / "llkh0a_model")
    tokenizer_ll = AutoTokenizer.from_pretrained(llkh0a_path)
    model_ll = AutoModelForSeq2SeqLM.from_pretrained(llkh0a_path).to(device)
    model_ll.eval()
    evaluate_model(model_ll, tokenizer_ll, sent_inputs_notag, sent_refs, "llkh0a", max_length)
    del model_ll
    torch.cuda.empty_cache()

    # ============================================================
    # Model 2: 我々のexp011（タグあり）
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== exp011 model (with PN/GN tags) ===")
    exp011_path = str(PROJECT_ROOT / "workspace" / "exp011_additional_data" / "results" / "best_model")
    tokenizer_ours = AutoTokenizer.from_pretrained(exp011_path)
    model_ours = AutoModelForSeq2SeqLM.from_pretrained(exp011_path).to(device)
    model_ours.eval()
    evaluate_model(model_ours, tokenizer_ours, sent_inputs_tagged, sent_refs, "exp011_tag", max_length)

    # ============================================================
    # Model 2b: 我々のexp011（タグなし — 公平比較用）
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== exp011 model (no PN/GN tags, for fair comparison) ===")
    evaluate_model(model_ours, tokenizer_ours, sent_inputs_notag, sent_refs, "exp011_notag", max_length)

    logger.info("=" * 60)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
