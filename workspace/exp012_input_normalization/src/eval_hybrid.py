"""
exp012: Greedy-MBR Hybrid推論
- 繰り返しが検出されない → greedy出力を使用（長く豊かな出力、平均162文字）
- 繰り返しが検出される → 13候補chrF++ MBRで安全な出力を選択

期待: greedyの繰り返しなし出力(38%で繰り返し)のリッチさと、MBRの安定性を両立。
"""
import json
import math
import os
import re
import sys
import time
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
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "eval_hybrid.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


# ============================================================
# Utilities (from grid_search_mbr.py)
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


sys.path.insert(0, str(PROJECT_ROOT / "workspace" / "exp007_mbr_postprocess" / "src"))
from infer_mbr import repeat_cleanup


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


def _dedup(xs):
    seen, out = set(), []
    for x in xs:
        x = str(x).strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


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
# MBR Selector (chrF++ only)
# ============================================================
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
# Evaluation
# ============================================================
def evaluate_cv(preds, refs):
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    return chrf.score, bleu.score, geo


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

    # Prepare val data
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
        akk = tag_transliteration(truncate_akkadian_to_sentence(t), form_tag_dict)
        if eng.strip() and akk.strip():
            sent_inputs.append(prefix + akk)
            sent_refs.append(eng)

    N = len(sent_inputs)
    logger.info(f"Val samples: {N}")

    # ============================================================
    # Step 1: Greedy decoding (rp=1.2)
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Step 1: Greedy Decoding (rp=1.2) ===")
    t0 = time.time()

    dataset = InferenceDataset(sent_inputs, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    greedy_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Greedy"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=max_length, num_beams=1, do_sample=False,
                repetition_penalty=1.2,
            )
            greedy_preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))

    greedy_time = time.time() - t0
    logger.info(f"Greedy time: {greedy_time:.0f}s")

    # Detect repetitions
    greedy_has_rep = [has_repetition(p) for p in greedy_preds]
    n_rep = sum(greedy_has_rep)
    logger.info(f"Greedy repetitions: {n_rep}/{N} ({100*n_rep/N:.1f}%)")

    # ============================================================
    # Step 2: MBR candidates (only for repetition samples)
    # ============================================================
    rep_indices = [i for i, has in enumerate(greedy_has_rep) if has]
    logger.info("=" * 60)
    logger.info(f"=== Step 2: MBR for {len(rep_indices)} repetition samples ===")

    # Generate 13 candidates: beam4 + 3temps×3
    mbr_selector = ChrFMBRSelector(pool_cap=32)
    mbr_preds = {}  # index -> prediction

    t0_mbr = time.time()
    with torch.no_grad():
        for idx in tqdm(rep_indices, desc="MBR Pool"):
            enc = tokenizer(
                sent_inputs[idx], max_length=max_length,
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

            mbr_preds[idx] = mbr_selector.pick(pool)

    mbr_time = time.time() - t0_mbr
    logger.info(f"MBR time (for {len(rep_indices)} samples): {mbr_time:.0f}s")

    # ============================================================
    # Step 3: Hybrid merge
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Step 3: Hybrid Results ===")

    hybrid_preds = []
    for i in range(N):
        if greedy_has_rep[i]:
            hybrid_preds.append(mbr_preds[i])
        else:
            hybrid_preds.append(greedy_preds[i])

    total_time = greedy_time + mbr_time

    # ============================================================
    # Step 4: Evaluate all variants
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Step 4: Evaluation ===")

    variants = {
        "greedy_raw": greedy_preds,
        "greedy_clean": [repeat_cleanup(p) for p in greedy_preds],
        "hybrid_raw": hybrid_preds,
        "hybrid_clean": [repeat_cleanup(p) for p in hybrid_preds],
    }

    # Also compute full MBR (13 cands) for all samples as reference
    logger.info("--- Computing full MBR (13 cands) for all samples as reference ---")
    t0_full = time.time()
    full_mbr_preds = []
    with torch.no_grad():
        for i in tqdm(range(N), desc="Full MBR"):
            if i in mbr_preds:
                # Already computed
                full_mbr_preds.append(mbr_preds[i])
            else:
                enc = tokenizer(
                    sent_inputs[i], max_length=max_length,
                    truncation=True, padding="max_length", return_tensors="pt",
                )
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)

                pool = []
                beam_out = model.generate(
                    input_ids=ids, attention_mask=mask,
                    do_sample=False, num_beams=8,
                    num_return_sequences=4,
                    max_new_tokens=max_length,
                    length_penalty=1.3, early_stopping=True,
                    repetition_penalty=1.2, use_cache=True,
                )
                pool.extend(tokenizer.batch_decode(beam_out, skip_special_tokens=True))

                for temp in [0.6, 0.8, 1.05]:
                    samp_out = model.generate(
                        input_ids=ids, attention_mask=mask,
                        do_sample=True, num_beams=1, top_p=0.92, temperature=temp,
                        num_return_sequences=3,
                        max_new_tokens=max_length,
                        repetition_penalty=1.2, use_cache=True,
                    )
                    pool.extend(tokenizer.batch_decode(samp_out, skip_special_tokens=True))

                full_mbr_preds.append(mbr_selector.pick(pool))

    full_mbr_time = time.time() - t0_full + mbr_time  # total including rep samples
    variants["full_mbr_raw"] = full_mbr_preds
    variants["full_mbr_clean"] = [repeat_cleanup(p) for p in full_mbr_preds]

    for name, preds in variants.items():
        chrf, bleu, geo = evaluate_cv(preds, sent_refs)
        rep_rate = 100 * sum(has_repetition(p) for p in preds) / N
        mean_len = np.mean([len(p) for p in preds])
        logger.info(
            f"  {name:20s}: chrF++={chrf:.2f}, BLEU={bleu:.2f}, geo={geo:.2f}, "
            f"rep={rep_rate:.1f}%, len={mean_len:.0f}"
        )

    # ============================================================
    # Step 5: Per-sample comparison (hybrid vs baselines)
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Step 5: Per-sample analysis ===")

    chrf_metric = sacrebleu.metrics.CHRF(word_order=2)

    # hybrid vs greedy_clean
    hybrid_wins, hybrid_loses, ties = 0, 0, 0
    for i in range(N):
        h = hybrid_preds[i]
        g = repeat_cleanup(greedy_preds[i])
        hs = float(chrf_metric.sentence_score(h, [sent_refs[i]]).score)
        gs = float(chrf_metric.sentence_score(g, [sent_refs[i]]).score)
        if hs > gs + 0.5:
            hybrid_wins += 1
        elif gs > hs + 0.5:
            hybrid_loses += 1
        else:
            ties += 1
    logger.info(f"Hybrid vs greedy_clean: wins={hybrid_wins}, loses={hybrid_loses}, ties={ties}")

    # hybrid vs full_mbr
    hybrid_wins2, hybrid_loses2, ties2 = 0, 0, 0
    for i in range(N):
        h = hybrid_preds[i]
        m = full_mbr_preds[i]
        hs = float(chrf_metric.sentence_score(h, [sent_refs[i]]).score)
        ms = float(chrf_metric.sentence_score(m, [sent_refs[i]]).score)
        if hs > ms + 0.5:
            hybrid_wins2 += 1
        elif ms > hs + 0.5:
            hybrid_loses2 += 1
        else:
            ties2 += 1
    logger.info(f"Hybrid vs full_mbr: wins={hybrid_wins2}, loses={hybrid_loses2}, ties={ties2}")

    # ============================================================
    # Step 6: Timing summary
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Timing Summary ===")
    logger.info(f"  Greedy (all {N} samples): {greedy_time:.0f}s")
    logger.info(f"  MBR (rep {len(rep_indices)} samples only): {mbr_time:.0f}s")
    logger.info(f"  Hybrid total: {total_time:.0f}s")
    logger.info(f"  Full MBR (all {N} samples): {full_mbr_time + greedy_time:.0f}s (for reference)")
    logger.info(f"  Hybrid speedup vs full MBR: {(full_mbr_time + greedy_time) / max(total_time, 1):.1f}x")

    # ============================================================
    # Save predictions
    # ============================================================
    out_df = pd.DataFrame({
        "input": sent_inputs,
        "reference": sent_refs,
        "greedy": greedy_preds,
        "greedy_clean": [repeat_cleanup(p) for p in greedy_preds],
        "hybrid": hybrid_preds,
        "full_mbr": full_mbr_preds,
        "has_repetition": greedy_has_rep,
    })
    csv_path = RESULTS_DIR / "val_predictions_hybrid.csv"
    out_df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    logger.info("=" * 60)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
