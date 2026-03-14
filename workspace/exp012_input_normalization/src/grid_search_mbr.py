"""
exp012: MBR/greedy推論バリエーションのグリッドサーチ
exp011のモデル1つで、候補生成・スコアリング・後処理の全組合せを試す。

テスト項目:
A. Greedy系
  A1. greedy (rep_penalty=1.2) — baseline
  A2. greedy (rep_penalty=1.3)
  A3. greedy (rep_penalty=1.5)

B. 候補プールサイズ × chrF++単体MBR
  B1. beam4 + sampling2 = 6候補 (exp011相当)
  B2. beam4 + multi-temp×2 = 10候補 (exp012で検証済み)
  B3. beam4 + multi-temp×3 = 13候補
  B4. beam8 + multi-temp×2 = 14候補
  B5. beam4 + multi-temp×4 = 16候補

C. スコアリング改善 (10候補ベース)
  C1. chrF++単体 (=B2)
  C2. chrF++ + BLEU (length bonus なし)
  C3. chrF++ + BLEU + Jaccard (length bonus なし)

D. 全てにrepeat_cleanup適用版も評価

候補生成は一度だけ行い、各スコアリング手法に使い回す。
"""
import json
import math
import os
import re
import sys
import time
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Tuple

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
        logging.FileHandler(RESULTS_DIR / "grid_search_mbr.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


# ============================================================
# Utilities
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
# MBR Selectors
# ============================================================
class ChrFMBRSelector:
    """chrF++単体MBR"""
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


class WeightedMBRSelector:
    """複合スコアMBR（length bonusなし）"""
    def __init__(self, w_chrf=0.55, w_bleu=0.25, w_jaccard=0.20, pool_cap=32):
        self._chrf = sacrebleu.metrics.CHRF(word_order=2)
        self._bleu = sacrebleu.metrics.BLEU(effective_order=True)
        self.w_chrf = w_chrf
        self.w_bleu = w_bleu
        self.w_jaccard = w_jaccard
        self.pool_cap = pool_cap
        self._total = max(w_chrf + w_bleu + w_jaccard, 1e-9)

    def _pairwise(self, a, b):
        if not a or not b:
            return 0.0
        chrf = float(self._chrf.sentence_score(a, [b]).score)
        try:
            bleu = float(self._bleu.sentence_score(a, [b]).score)
        except Exception:
            bleu = 0.0
        ta, tb = set(a.lower().split()), set(b.lower().split())
        jac = 100.0 * len(ta & tb) / len(ta | tb) if (ta or tb) else 100.0
        return (self.w_chrf * chrf + self.w_bleu * bleu + self.w_jaccard * jac) / self._total

    def pick(self, candidates):
        cands = _dedup(candidates)[:self.pool_cap]
        n = len(cands)
        if n <= 1:
            return cands[0] if cands else ""
        scores = [
            sum(self._pairwise(cands[i], cands[j]) for j in range(n) if j != i) / (n - 1)
            for i in range(n)
        ]
        return cands[int(np.argmax(scores))]


def _dedup(xs):
    seen, out = set(), []
    for x in xs:
        x = str(x).strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ============================================================
# Candidate Generation
# ============================================================
def generate_greedy(model, tokenizer, input_ids, attention_mask, max_length, rep_penalty):
    out = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_length=max_length, num_beams=1, do_sample=False,
        repetition_penalty=rep_penalty,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def generate_pool(model, tokenizer, input_ids, attention_mask, max_length,
                  num_beam_cands, num_beams, temps, num_per_temp, top_p=0.92):
    """beam + multi-temp samplingで候補プール生成"""
    B = input_ids.shape[0]

    # Beam
    beam_out = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        do_sample=False, num_beams=num_beams,
        num_return_sequences=num_beam_cands,
        max_new_tokens=max_length,
        length_penalty=1.3, early_stopping=True,
        repetition_penalty=1.2, use_cache=True,
    )
    beam_texts = tokenizer.batch_decode(beam_out, skip_special_tokens=True)

    pools = [[] for _ in range(B)]
    for i in range(B):
        pools[i].extend(beam_texts[i * num_beam_cands:(i + 1) * num_beam_cands])

    # Sampling
    for temp in temps:
        samp_out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=True, num_beams=1, top_p=top_p, temperature=temp,
            num_return_sequences=num_per_temp,
            max_new_tokens=max_length,
            repetition_penalty=1.2, use_cache=True,
        )
        samp_texts = tokenizer.batch_decode(samp_out, skip_special_tokens=True)
        for i in range(B):
            pools[i].extend(samp_texts[i * num_per_temp:(i + 1) * num_per_temp])

    return pools


# ============================================================
# Evaluation
# ============================================================
def evaluate_cv(preds, refs):
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    return chrf.score, bleu.score, geo


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


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

    logger.info(f"Val samples: {len(sent_inputs)}")
    N = len(sent_inputs)

    results = []  # list of dicts

    # ============================================================
    # A. Greedy variants
    # ============================================================
    dataset = InferenceDataset(sent_inputs, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for rep_pen in [1.2, 1.3, 1.5]:
        label = f"greedy_rp{rep_pen}"
        logger.info(f"=== {label} ===")
        t0 = time.time()
        preds = []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                preds.extend(generate_greedy(model, tokenizer, ids, mask, max_length, rep_pen))
        elapsed = time.time() - t0

        chrf, bleu, geo = evaluate_cv(preds, sent_refs)
        preds_clean = [repeat_cleanup(p) for p in preds]
        chrf_c, bleu_c, geo_c = evaluate_cv(preds_clean, sent_refs)
        rep_rate = 100 * sum(has_repetition(p) for p in preds) / N
        rep_rate_c = 100 * sum(has_repetition(p) for p in preds_clean) / N
        mean_len = np.mean([len(p) for p in preds])

        results.append({
            "method": label, "geo_raw": geo, "geo_clean": geo_c,
            "chrf_raw": chrf, "bleu_raw": bleu, "chrf_clean": chrf_c, "bleu_clean": bleu_c,
            "rep_raw": rep_rate, "rep_clean": rep_rate_c,
            "mean_len": mean_len, "time_sec": elapsed, "n_cands": 1,
        })
        logger.info(f"  raw={geo:.2f}, clean={geo_c:.2f}, rep={rep_rate:.1f}%, time={elapsed:.0f}s")

    # ============================================================
    # B. Pool generation (max superset, then subset for each config)
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Generating max candidate pool (beam8 + temps[0.6,0.8,1.05] × 4) ===")

    # Generate the largest pool: beam8 + 3temps × 4 = 20 candidates
    t0_pool = time.time()
    max_pools = []
    with torch.no_grad():
        for i in tqdm(range(N), desc="MaxPool"):
            enc = tokenizer(
                sent_inputs[i], max_length=max_length,
                truncation=True, padding="max_length", return_tensors="pt",
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            pools = generate_pool(
                model, tokenizer, ids, mask, max_length,
                num_beam_cands=8, num_beams=12,
                temps=[0.6, 0.8, 1.05], num_per_temp=4,
            )
            max_pools.append(pools[0])
    pool_time = time.time() - t0_pool
    logger.info(f"Pool generation time: {pool_time:.0f}s")

    # Pool configs: (label, beam_cands, temps_to_use, samples_per_temp)
    pool_configs = [
        ("B1_beam4_samp2",   4, [0.8], 2),           # 4+2=6
        ("B2_beam4_mt2",     4, [0.6, 0.8, 1.05], 2), # 4+6=10
        ("B3_beam4_mt3",     4, [0.6, 0.8, 1.05], 3), # 4+9=13
        ("B4_beam8_mt2",     8, [0.6, 0.8, 1.05], 2), # 8+6=14
        ("B5_beam4_mt4",     4, [0.6, 0.8, 1.05], 4), # 4+12=16
    ]

    # Map temps to indices in max_pool
    # max_pool structure: [beam0..beam7, temp0.6_s0..s3, temp0.8_s0..s3, temp1.05_s0..s3]
    TEMP_LIST = [0.6, 0.8, 1.05]

    def extract_subpool(full_pool, beam_cands, temps_to_use, samp_per_temp):
        """最大プールからサブセットを抽出"""
        sub = full_pool[:beam_cands]  # beam candidates
        for temp in temps_to_use:
            if temp in TEMP_LIST:
                tidx = TEMP_LIST.index(temp)
                start = 8 + tidx * 4  # 8 beam cands in max pool
                sub.extend(full_pool[start:start + samp_per_temp])
        return sub

    # MBR selectors
    chrf_selector = ChrFMBRSelector(pool_cap=32)
    chrf_bleu_selector = WeightedMBRSelector(w_chrf=0.65, w_bleu=0.35, w_jaccard=0.0, pool_cap=32)
    chrf_bleu_jac_selector = WeightedMBRSelector(w_chrf=0.55, w_bleu=0.25, w_jaccard=0.20, pool_cap=32)

    # ============================================================
    # B. Pool size × chrF++ MBR
    # ============================================================
    for label, bc, temps, spt in pool_configs:
        n_cands = bc + len(temps) * spt
        logger.info(f"=== {label} ({n_cands} cands) + chrF++ MBR ===")

        t0 = time.time()
        preds = []
        for pool in max_pools:
            sub = extract_subpool(pool, bc, temps, spt)
            preds.append(chrf_selector.pick(sub))
        elapsed = time.time() - t0 + pool_time * (n_cands / 20)  # proportional pool time

        chrf, bleu, geo = evaluate_cv(preds, sent_refs)
        preds_clean = [repeat_cleanup(p) for p in preds]
        chrf_c, bleu_c, geo_c = evaluate_cv(preds_clean, sent_refs)
        rep_rate = 100 * sum(has_repetition(p) for p in preds) / N
        rep_rate_c = 100 * sum(has_repetition(p) for p in preds_clean) / N
        mean_len = np.mean([len(p) for p in preds])

        results.append({
            "method": f"{label}_chrf", "geo_raw": geo, "geo_clean": geo_c,
            "chrf_raw": chrf, "bleu_raw": bleu, "chrf_clean": chrf_c, "bleu_clean": bleu_c,
            "rep_raw": rep_rate, "rep_clean": rep_rate_c,
            "mean_len": mean_len, "time_sec": elapsed, "n_cands": n_cands,
        })
        logger.info(f"  raw={geo:.2f}, clean={geo_c:.2f}, rep={rep_rate:.1f}%, time={elapsed:.0f}s")

    # ============================================================
    # C. Scoring variants (10候補ベース)
    # ============================================================
    for sel_label, selector in [
        ("C2_chrf_bleu", chrf_bleu_selector),
        ("C3_chrf_bleu_jac", chrf_bleu_jac_selector),
    ]:
        logger.info(f"=== {sel_label} (10 cands) ===")
        t0 = time.time()
        preds = []
        for pool in max_pools:
            sub = extract_subpool(pool, 4, [0.6, 0.8, 1.05], 2)
            preds.append(selector.pick(sub))
        elapsed = time.time() - t0 + pool_time * (10 / 20)

        chrf, bleu, geo = evaluate_cv(preds, sent_refs)
        preds_clean = [repeat_cleanup(p) for p in preds]
        chrf_c, bleu_c, geo_c = evaluate_cv(preds_clean, sent_refs)
        rep_rate = 100 * sum(has_repetition(p) for p in preds) / N
        rep_rate_c = 100 * sum(has_repetition(p) for p in preds_clean) / N
        mean_len = np.mean([len(p) for p in preds])

        results.append({
            "method": sel_label, "geo_raw": geo, "geo_clean": geo_c,
            "chrf_raw": chrf, "bleu_raw": bleu, "chrf_clean": chrf_c, "bleu_clean": bleu_c,
            "rep_raw": rep_rate, "rep_clean": rep_rate_c,
            "mean_len": mean_len, "time_sec": elapsed, "n_cands": 10,
        })
        logger.info(f"  raw={geo:.2f}, clean={geo_c:.2f}, rep={rep_rate:.1f}%, time={elapsed:.0f}s")

    # ============================================================
    # Summary ranking
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== RANKING (by geo_clean) ===")
    logger.info("=" * 60)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("geo_clean", ascending=False).reset_index(drop=True)

    for i, row in results_df.iterrows():
        logger.info(
            f"  #{i+1:2d} {row['method']:25s} | "
            f"geo_clean={row['geo_clean']:.2f} | geo_raw={row['geo_raw']:.2f} | "
            f"rep={row['rep_clean']:.0f}% | len={row['mean_len']:.0f} | "
            f"time={row['time_sec']:.0f}s | cands={row['n_cands']}"
        )

    # Save
    csv_path = RESULTS_DIR / "grid_search_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved: {csv_path}")

    logger.info("=" * 60)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
