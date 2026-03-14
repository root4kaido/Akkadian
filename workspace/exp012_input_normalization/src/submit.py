"""
exp012: Kaggle提出用推論スクリプト
- 13候補chrF++ MBR (CV=34.47) をメイン
- greedy_clean (CV=33.45) をフォールバック

Usage:
  python submit.py                  # MBR (default)
  python submit.py --mode greedy    # greedy_clean
"""
import json
import re
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sacrebleu
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ============================================================
# Paths — ローカル用。Kaggle環境では以下を書き換える
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "workspace" / "exp011_additional_data" / "results" / "best_model"
DICT_PATH = PROJECT_ROOT / "workspace" / "exp011_additional_data" / "dataset" / "form_type_dict.json"
TEST_CSV = PROJECT_ROOT / "datasets" / "raw" / "test.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"

# Kaggle環境用（アップロード後にコメント解除）
# KAGGLE_DATASET = Path("/kaggle/input/exp011-best-model")
# MODEL_PATH = KAGGLE_DATASET / "best_model"
# DICT_PATH = KAGGLE_DATASET / "form_type_dict.json"
# TEST_CSV = Path("/kaggle/input/deep-past-initiative-machine-translation/test.csv")
# OUTPUT_DIR = Path("/kaggle/working")

# repeat_cleanup用スクリプトパス
sys.path.insert(0, str(PROJECT_ROOT / "workspace" / "exp007_mbr_postprocess" / "src"))
# Kaggle環境用: sys.path.insert(0, str(KAGGLE_DATASET / "src"))

# ============================================================
# Setup
# ============================================================
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utilities (self-contained for Kaggle)
# ============================================================
def tag_transliteration(text: str, form_tag_dict: dict) -> str:
    tokens = text.split()
    return " ".join(
        f"{t}[{form_tag_dict[t]}]" if t in form_tag_dict else t
        for t in tokens
    )


# repeat_cleanup — infer_mbr.pyからインポート。失敗時は自前定義
try:
    from infer_mbr import repeat_cleanup
except ImportError:
    FUNC_WORDS = {
        'the','a','an','of','to','and','in','on','for','with','at','by','from','as',
        'is','are','was','were','be','been','being','that','this','these','those',
        'it','its','his','her','their','your','my','our','or','not','no'
    }

    def dedup_consecutive_tokens(tokens, mode='function_only'):
        out, prev_core, prev_punct_score = [], None, 0
        for t in tokens:
            m = re.match(r"^(\W*)(.*?)(\W*)$", t)
            pre, core, suf = m.group(1), m.group(2), m.group(3)
            core_l = core.lower()
            punct_score = len(pre) + len(suf)
            if prev_core is not None and core_l and core_l == prev_core:
                if (mode == 'all') or (core_l in FUNC_WORDS):
                    if punct_score > prev_punct_score and out:
                        out[-1] = t
                        prev_punct_score = punct_score
                    continue
            out.append(t)
            prev_core = core_l if core_l else None
            prev_punct_score = punct_score
        return out

    def remove_repeated_suffix(tokens, min_k=3, max_k=12):
        while True:
            n = len(tokens)
            found = False
            for k in range(min(max_k, n // 2), min_k - 1, -1):
                if tokens[n-2*k:n-k] == tokens[n-k:n]:
                    seg = tokens[n-k:n]
                    if sum(bool(re.search(r"[A-Za-z]", x)) for x in seg) >= 2:
                        tokens = tokens[:n-k]
                        found = True
                        break
            if not found:
                break
        return tokens

    def repeat_cleanup(s: str) -> str:
        toks = str(s).split()
        if len(toks) < 2:
            return s
        toks = dedup_consecutive_tokens(toks, mode='function_only')
        if len(toks) >= 6:
            toks = remove_repeated_suffix(toks, min_k=3, max_k=12)
        return ' '.join(toks)


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


def _dedup(xs):
    seen, out = set(), []
    for x in xs:
        x = str(x).strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


class ChrFMBRSelector:
    def __init__(self, pool_cap=32):
        self._chrf = sacrebleu.metrics.CHRF(word_order=2)
        self.pool_cap = pool_cap

    def _score(self, a, b):
        if not a or not b:
            return 0.0
        return float(self._chrf.sentence_score(a, [b]).score)

    def pick(self, candidates):
        cands = _dedup(candidates)[:self.pool_cap]
        n = len(cands)
        if n <= 1:
            return cands[0] if cands else ""
        scores = [
            sum(self._score(cands[i], cands[j]) for j in range(n) if j != i) / (n - 1)
            for i in range(n)
        ]
        return cands[int(np.argmax(scores))]


# ============================================================
# Inference functions
# ============================================================
def greedy_inference(model, tokenizer, inputs, max_length=512, batch_size=4):
    dataset = InferenceDataset(inputs, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
    return [repeat_cleanup(p) for p in preds]


def mbr_inference(model, tokenizer, inputs, max_length=512, batch_size=1):
    """13候補chrF++ MBR: beam4 + 3temps×3 sampling"""
    selector = ChrFMBRSelector(pool_cap=32)
    preds = []
    with torch.no_grad():
        for text in tqdm(inputs, desc="MBR"):
            enc = tokenizer(
                text, max_length=max_length,
                truncation=True, padding="max_length", return_tensors="pt",
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            pool = []
            # Beam candidates (4)
            beam_out = model.generate(
                input_ids=ids, attention_mask=mask,
                do_sample=False, num_beams=8,
                num_return_sequences=4,
                max_new_tokens=max_length,
                length_penalty=1.3, early_stopping=True,
                repetition_penalty=1.2, use_cache=True,
            )
            pool.extend(tokenizer.batch_decode(beam_out, skip_special_tokens=True))

            # Sampling candidates (3 temps × 3 = 9)
            for temp in [0.6, 0.8, 1.05]:
                samp_out = model.generate(
                    input_ids=ids, attention_mask=mask,
                    do_sample=True, num_beams=1, top_p=0.92, temperature=temp,
                    num_return_sequences=3,
                    max_new_tokens=max_length,
                    repetition_penalty=1.2, use_cache=True,
                )
                pool.extend(tokenizer.batch_decode(samp_out, skip_special_tokens=True))

            preds.append(selector.pick(pool))
    return preds


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mbr", "greedy"], default="mbr")
    args = parser.parse_args()

    logger.info(f"Device: {device}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {MODEL_PATH}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH)).to(device)
    model.eval()

    # Load PN/GN dict
    with open(DICT_PATH) as f:
        form_tag_dict = json.load(f)
    logger.info(f"form_tag_dict: {len(form_tag_dict)} entries")

    # Load test data
    test_df = pd.read_csv(str(TEST_CSV))
    logger.info(f"Test samples: {len(test_df)}")

    prefix = "translate Akkadian to English: "
    test_inputs = []
    for _, row in test_df.iterrows():
        src = tag_transliteration(str(row["transliteration"]), form_tag_dict)
        test_inputs.append(prefix + src)

    # Inference
    if args.mode == "mbr":
        preds = mbr_inference(model, tokenizer, test_inputs)
    else:
        preds = greedy_inference(model, tokenizer, test_inputs)

    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "translation": preds,
    })

    # Validation
    assert len(submission) == len(test_df), f"Row mismatch: {len(submission)} vs {len(test_df)}"
    assert submission["translation"].isna().sum() == 0, "Found NaN"
    assert (submission["translation"].str.len() > 0).all(), "Found empty translation"

    out_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path} ({len(submission)} rows)")

    # Preview
    for i in range(min(5, len(submission))):
        logger.info(f"  [{i}] {submission.iloc[i]['translation'][:120]}")


if __name__ == "__main__":
    main()
