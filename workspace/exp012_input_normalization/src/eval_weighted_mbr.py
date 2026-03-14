"""
exp012: Weighted MBR評価
exp011のモデルを使い、候補生成+スコアリングの改善を検証。

比較対象:
1. greedy (num_beams=1, repetition_penalty=1.2)
2. 既存MBR (chrF++単体, beam4+sampling2)
3. weighted MBR (chrF++ + BLEU + Jaccard + length bonus, beam4+multi-temp sampling6)

全てexp011のbest_modelを使用。学習は行わない。
"""
import json
import math
import os
import re
import sys
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
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "eval_weighted_mbr.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

with open(EXP_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


# ============================================================
# Utilities
# ============================================================
def extract_first_sentence(text: str) -> str:
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    if m:
        return m.group(1).strip()
    return str(text).strip()


def truncate_akkadian_to_sentence(translit: str, max_bytes: int = 200) -> str:
    encoded = str(translit).encode('utf-8')
    if len(encoded) <= max_bytes:
        return str(translit)
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space].strip()
    return truncated.strip()


def tag_transliteration(text: str, form_tag_dict: dict) -> str:
    tokens = text.split()
    tagged_tokens = []
    for token in tokens:
        tag = form_tag_dict.get(token)
        if tag:
            tagged_tokens.append(f"{token}[{tag}]")
        else:
            tagged_tokens.append(token)
    return " ".join(tagged_tokens)


# Import repeat_cleanup from exp007
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
# Weighted MBR Selector
# ============================================================
class WeightedMBRSelector:
    """chrF++ + BLEU + Jaccard + length bonus による候補選択"""

    def __init__(self, w_chrf=0.55, w_bleu=0.25, w_jaccard=0.20, w_length=0.10, pool_cap=32):
        self._chrf = sacrebleu.metrics.CHRF(word_order=2)
        self._bleu = sacrebleu.metrics.BLEU(effective_order=True)
        self.w_chrf = w_chrf
        self.w_bleu = w_bleu
        self.w_jaccard = w_jaccard
        self.w_length = w_length
        self.pool_cap = pool_cap
        self._pw_total = max(w_chrf + w_bleu + w_jaccard, 1e-9)

    def _chrfpp(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(self._chrf.sentence_score(a, [b]).score)

    def _bleu_score(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        try:
            return float(self._bleu.sentence_score(a, [b]).score)
        except Exception:
            return 0.0

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta and not tb:
            return 100.0
        if not ta or not tb:
            return 0.0
        return 100.0 * len(ta & tb) / len(ta | tb)

    def _pairwise_score(self, a: str, b: str) -> float:
        s = (
            self.w_chrf * self._chrfpp(a, b)
            + self.w_bleu * self._bleu_score(a, b)
            + self.w_jaccard * self._jaccard(a, b)
        )
        return s / self._pw_total

    @staticmethod
    def _length_bonus(lengths: List[int], idx: int) -> float:
        if not lengths:
            return 100.0
        median = float(np.median(lengths))
        sigma = max(median * 0.4, 5.0)
        z = (lengths[idx] - median) / sigma
        return 100.0 * math.exp(-0.5 * z * z)

    @staticmethod
    def _dedup(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def pick(self, candidates: List[str]) -> str:
        cands = self._dedup(candidates)
        if self.pool_cap:
            cands = cands[:self.pool_cap]

        n = len(cands)
        if n == 0:
            return ""
        if n == 1:
            return cands[0]

        lengths = [len(c.split()) for c in cands]
        scores = []

        for i in range(n):
            pw = sum(
                self._pairwise_score(cands[i], cands[j])
                for j in range(n) if j != i
            ) / max(1, n - 1)

            lb = self._length_bonus(lengths, i)
            total = pw + self.w_length * lb
            scores.append(total)

        return cands[int(np.argmax(scores))]


# ============================================================
# Legacy MBR (chrF++ only) for comparison
# ============================================================
class LegacyMBRSelector:
    """exp007/exp011で使用していたchrF++単体のMBR"""

    def __init__(self, pool_cap=32):
        self._chrf = sacrebleu.metrics.CHRF(word_order=2)
        self.pool_cap = pool_cap

    def _chrfpp(self, a: str, b: str) -> float:
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return 0.0
        return float(self._chrf.sentence_score(a, [b]).score)

    @staticmethod
    def _dedup(xs):
        seen, out = set(), []
        for x in xs:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def pick(self, candidates: List[str]) -> str:
        cands = self._dedup(candidates)
        cands = cands[:self.pool_cap]

        n = len(cands)
        if n == 0:
            return ""
        if n == 1:
            return cands[0]

        best_i, best_s = 0, -1e9
        for i in range(n):
            s = sum(self._chrfpp(cands[i], cands[j]) for j in range(n) if i != j) / max(1, n - 1)
            if s > best_s:
                best_s, best_i = s, i
        return cands[best_i]


# ============================================================
# Candidate Generation
# ============================================================
def generate_candidate_pool(
    model, tokenizer, input_ids, attention_mask, config,
) -> List[List[str]]:
    """beam + multi-temperature samplingで候補プールを生成"""
    cand_cfg = config["inference"]["candidates"]
    B = input_ids.shape[0]

    num_beam_cands = cand_cfg["num_beam_cands"]
    num_beams = cand_cfg["num_beams"]

    # Beam candidates
    beam_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=num_beam_cands,
        max_new_tokens=config["inference"]["max_length"],
        length_penalty=cand_cfg["length_penalty"],
        early_stopping=cand_cfg["early_stopping"],
        repetition_penalty=config["inference"]["repetition_penalty"],
        use_cache=True,
    )
    beam_texts = tokenizer.batch_decode(beam_out, skip_special_tokens=True)

    pools = [[] for _ in range(B)]
    for i in range(B):
        pools[i].extend(beam_texts[i * num_beam_cands:(i + 1) * num_beam_cands])

    # Sampling candidates
    if cand_cfg.get("use_sampling", False):
        temps = cand_cfg["sample_temperatures"]
        num_per_temp = cand_cfg["num_sample_per_temp"]
        top_p = cand_cfg.get("sampling_top_p", 0.92)

        for temp in temps:
            samp_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                num_beams=1,
                top_p=top_p,
                temperature=temp,
                num_return_sequences=num_per_temp,
                max_new_tokens=config["inference"]["max_length"],
                repetition_penalty=config["inference"]["repetition_penalty"],
                use_cache=True,
            )
            samp_texts = tokenizer.batch_decode(samp_out, skip_special_tokens=True)
            for i in range(B):
                pools[i].extend(samp_texts[i * num_per_temp:(i + 1) * num_per_temp])

    return pools


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
    # Load model (exp011's best_model)
    model_path = str(PROJECT_ROOT / "workspace" / "exp011_additional_data" / "results" / "best_model")
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    prefix = config["model"].get("prefix", "translate Akkadian to English: ")

    # Load PN/GN tag dictionary (from exp011)
    dict_path = PROJECT_ROOT / "workspace" / "exp011_additional_data" / "dataset" / "form_type_dict.json"
    with open(dict_path) as f:
        form_tag_dict = json.load(f)
    logger.info(f"Loaded form_tag_dict: {len(form_tag_dict)} entries")

    # Prepare validation data (same split as exp011)
    train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
    train_df = train_df[
        (train_df["transliteration"].astype(str).str.len() > 0)
        & (train_df["translation"].astype(str).str.len() > 0)
    ]

    seed = config["training"]["seed"]
    val_ratio = config["training"].get("val_ratio", 0.1)
    _, val_df = train_test_split(train_df, test_size=val_ratio, random_state=seed)
    val_df = val_df.reset_index(drop=True)

    # Build sentence-level pairs (with PN/GN tagging, same as exp011)
    sent_inputs = []
    sent_refs = []
    for _, row in val_df.iterrows():
        translit = str(row["transliteration"])
        translation = str(row["translation"])
        first_sent_eng = extract_first_sentence(translation)
        first_sent_akk = truncate_akkadian_to_sentence(translit, max_bytes=200)
        first_sent_akk_tagged = tag_transliteration(first_sent_akk, form_tag_dict)
        if len(first_sent_eng.strip()) > 0 and len(first_sent_akk_tagged.strip()) > 0:
            sent_inputs.append(prefix + first_sent_akk_tagged)
            sent_refs.append(first_sent_eng)

    logger.info(f"Sentence-level val samples: {len(sent_inputs)}")

    max_length = config["inference"]["max_length"]

    # ============================================================
    # 1. Greedy decoding
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== 1. Greedy Decoding ===")

    dataset = InferenceDataset(sent_inputs, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

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
                repetition_penalty=config["inference"]["repetition_penalty"],
            )
            greedy_preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # ============================================================
    # 2. Generate candidate pools (for both MBR methods)
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== 2. Generating Candidate Pools ===")

    all_pools = []
    # batch_size=1 for candidate generation (num_return_sequences > 1)
    with torch.no_grad():
        for i in tqdm(range(len(sent_inputs)), desc="Candidates"):
            enc = tokenizer(
                sent_inputs[i], max_length=max_length,
                truncation=True, padding="max_length", return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            pools = generate_candidate_pool(model, tokenizer, input_ids, attention_mask, config)
            all_pools.append(pools[0])

    # Log pool statistics
    pool_sizes = [len(p) for p in all_pools]
    dedup_sizes = [len(set(p)) for p in all_pools]
    logger.info(f"Pool sizes: mean={np.mean(pool_sizes):.1f}, unique mean={np.mean(dedup_sizes):.1f}")

    # ============================================================
    # 3. Legacy MBR (chrF++ only)
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== 3. Legacy MBR (chrF++ only) ===")

    legacy_selector = LegacyMBRSelector(pool_cap=32)
    legacy_preds = []
    for pool in tqdm(all_pools, desc="Legacy MBR"):
        legacy_preds.append(legacy_selector.pick(pool))

    # ============================================================
    # 4. Weighted MBR
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== 4. Weighted MBR ===")

    mbr_cfg = config["inference"]["mbr"]
    weighted_selector = WeightedMBRSelector(
        w_chrf=mbr_cfg["w_chrf"],
        w_bleu=mbr_cfg["w_bleu"],
        w_jaccard=mbr_cfg["w_jaccard"],
        w_length=mbr_cfg["w_length"],
        pool_cap=mbr_cfg["pool_cap"],
    )
    weighted_preds = []
    for pool in tqdm(all_pools, desc="Weighted MBR"):
        weighted_preds.append(weighted_selector.pick(pool))

    # ============================================================
    # 5. Apply repeat_cleanup
    # ============================================================
    greedy_clean = [repeat_cleanup(p) for p in greedy_preds]
    legacy_clean = [repeat_cleanup(p) for p in legacy_preds]
    weighted_clean = [repeat_cleanup(p) for p in weighted_preds]

    # ============================================================
    # 6. Evaluate all methods
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Results ===")
    logger.info("=" * 60)

    evaluate_cv(greedy_preds, sent_refs, "greedy_raw")
    evaluate_cv(legacy_preds, sent_refs, "legacy_mbr_raw")
    evaluate_cv(weighted_preds, sent_refs, "weighted_mbr_raw")
    evaluate_cv(greedy_clean, sent_refs, "greedy_clean")
    evaluate_cv(legacy_clean, sent_refs, "legacy_mbr_clean")
    evaluate_cv(weighted_clean, sent_refs, "weighted_mbr_clean")

    # ============================================================
    # 7. Analysis: where does weighted MBR differ from legacy?
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Weighted vs Legacy MBR Differences ===")

    n_diff = sum(1 for a, b in zip(legacy_preds, weighted_preds) if a != b)
    logger.info(f"Different predictions: {n_diff}/{len(legacy_preds)} ({100*n_diff/len(legacy_preds):.1f}%)")

    # Show examples where they differ
    shown = 0
    for i, (leg, wt, ref) in enumerate(zip(legacy_preds, weighted_preds, sent_refs)):
        if leg != wt and shown < 5:
            leg_chrf = sacrebleu.sentence_chrf(leg, [ref], word_order=2).score
            wt_chrf = sacrebleu.sentence_chrf(wt, [ref], word_order=2).score
            logger.info(f"  [{i}] ref: {ref[:100]}")
            logger.info(f"       legacy  (chrF={leg_chrf:.1f}): {leg[:100]}")
            logger.info(f"       weighted(chrF={wt_chrf:.1f}): {wt[:100]}")
            shown += 1

    # Repetition stats
    def has_repetition(text, min_repeat=3):
        words = str(text).split()
        for i in range(len(words) - min_repeat):
            chunk = " ".join(words[i:i + min_repeat])
            rest = " ".join(words[i + min_repeat:])
            if chunk in rest:
                return True
        return False

    for col_name, preds in [
        ("greedy", greedy_preds), ("legacy_mbr", legacy_preds), ("weighted_mbr", weighted_preds),
        ("greedy_clean", greedy_clean), ("legacy_mbr_clean", legacy_clean), ("weighted_mbr_clean", weighted_clean),
    ]:
        n_rep = sum(1 for p in preds if has_repetition(str(p)))
        logger.info(f"Repetitions in {col_name}: {n_rep}/{len(preds)} ({100*n_rep/len(preds):.1f}%)")

    # Output length comparison
    for col_name, preds in [
        ("greedy", greedy_preds), ("legacy_mbr", legacy_preds), ("weighted_mbr", weighted_preds),
        ("reference", sent_refs),
    ]:
        lengths = [len(p) for p in preds]
        logger.info(f"Length {col_name}: mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # ============================================================
    # 8. Save results
    # ============================================================
    results_df = pd.DataFrame({
        "input": sent_inputs,
        "reference": sent_refs,
        "greedy_pred": greedy_preds,
        "legacy_mbr_pred": legacy_preds,
        "weighted_mbr_pred": weighted_preds,
        "greedy_clean": greedy_clean,
        "legacy_mbr_clean": legacy_clean,
        "weighted_mbr_clean": weighted_clean,
    })
    csv_path = RESULTS_DIR / "val_predictions_weighted_mbr.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path} ({len(results_df)} rows)")

    logger.info("=" * 60)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
