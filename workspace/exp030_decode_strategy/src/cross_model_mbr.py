"""
Cross-model MBR: 5foldモデルから候補を生成しchrF++ consensusで選択
exp030_decode_strategy の一環として sent-CV (fold3) で評価

Usage:
    python cross_model_mbr.py [--mode greedy|sampling|both] [--num_samples 2]
"""
import os
import re
import sys
import math
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import GroupKFold
import sacrebleu

# ============================================================
# Config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP023_DIR = PROJECT_ROOT / "workspace" / "exp023_full_preprocessing"
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"
MAX_LENGTH = 512
EVAL_FOLD = 3  # fold3のvalで評価

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Preprocessing (exp023 と同一)
# ============================================================
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002

def _decimal_to_fraction(match):
    dec_str = match.group(0)
    try:
        value = float(dec_str)
    except ValueError:
        return dec_str
    int_part = int(value)
    frac_part = value - int_part
    if frac_part < 0.001:
        return dec_str
    best_frac, best_dist = None, float('inf')
    for target, symbol in FRACTION_TARGETS.items():
        dist = abs(frac_part - target)
        if dist < best_dist:
            best_dist, best_frac = dist, symbol
    if best_dist <= APPROX_TOLERANCE:
        return best_frac if int_part == 0 else f"{int_part} {best_frac}"
    return dec_str

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def clean_transliteration(text):
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text

# ============================================================
# Utilities
# ============================================================
def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return " ".join(words[:i+n])
    return text

def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False

chrfpp = sacrebleu.metrics.CHRF(word_order=2)

def mbr_pick(candidates):
    """MBR: chrF++のconsensusで最良候補を選択"""
    cands = list(dict.fromkeys(candidates))  # dedup keeping order
    cands = cands[:32]
    n = len(cands)
    if n <= 1:
        return cands[0] if cands else ""
    scores = []
    for i in range(n):
        s = sum(float(chrfpp.sentence_score(cands[i], [cands[j]]).score) for j in range(n) if j != i)
        scores.append(s / (n - 1))
    return cands[int(np.argmax(scores))]

def calc_metrics(preds, refs):
    preds_clean = [repeat_cleanup(p) for p in preds]
    chrf = sacrebleu.corpus_chrf(preds_clean, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds_clean, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in preds_clean) / len(preds_clean)
    return chrf.score, bleu.score, geo, rep_rate

# ============================================================
# Data preparation (fold3 sent-CV)
# ============================================================
def prepare_val_data():
    train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
    train_df["transliteration"] = train_df["transliteration"].astype(str).apply(clean_transliteration)

    def simple_sentence_aligner(df):
        aligned_data = []
        for _, row in df.iterrows():
            src = str(row["transliteration"])
            tgt = str(row["translation"])
            oare_id = row["oare_id"]
            tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
            src_lines = [s.strip() for s in src.split("\n") if s.strip()]
            if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
                for s, t in zip(src_lines, tgt_sents):
                    if len(s) > 3 and len(t) > 3:
                        aligned_data.append({"transliteration": s, "translation": t, "oare_id": oare_id})
            else:
                aligned_data.append({"transliteration": src, "translation": tgt, "oare_id": oare_id})
        return pd.DataFrame(aligned_data)

    train_expanded = simple_sentence_aligner(train_df)
    akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
    oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
    train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(train_expanded, groups=train_expanded["akt_group"].values))
    _, val_idx = splits[EVAL_FOLD]
    val_data = train_expanded.iloc[val_idx].copy()

    sent_aligned = pd.read_csv(str(SENTENCE_ALIGNED_PATH))
    alignment_dict = {}
    for oare_id, group in sent_aligned.groupby('oare_id'):
        group = group.sort_values('sent_idx')
        alignment_dict[oare_id] = [
            {'akk_segment': str(r['akk_segment']), 'eng_sentence': str(r['eng_sentence'])}
            for _, r in group.iterrows()
        ]

    translit_to_oare = {str(row['transliteration']): row['oare_id'] for _, row in train_df.iterrows()}

    prefix = "translate Akkadian to English: "
    sent_inputs = []
    sent_refs = []
    for _, row in val_data.iterrows():
        translit = str(row['transliteration'])
        oare_id = translit_to_oare.get(translit)
        if oare_id and oare_id in alignment_dict:
            boundaries = alignment_dict[oare_id]
            if len(boundaries) <= 6:
                for b in boundaries:
                    if b['akk_segment'].strip() and b['eng_sentence'].strip():
                        sent_inputs.append(prefix + clean_transliteration(b['akk_segment']))
                        sent_refs.append(b['eng_sentence'])

    logger.info(f"Fold {EVAL_FOLD}: val sents={len(sent_inputs)}")
    return sent_inputs, sent_refs

# ============================================================
# Candidate generation
# ============================================================
def generate_candidates_single_model(model, tokenizer, texts, mode="both",
                                     num_samples=2, temperatures=[0.6, 0.8, 1.05]):
    """1モデルから候補生成。返り値: list of list of str (各入力の候補リスト)"""
    all_candidates = [[] for _ in range(len(texts))]

    # Encode inputs
    encodings = tokenizer(
        texts, max_length=MAX_LENGTH, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    ds = torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"])
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    idx = 0
    with torch.no_grad():
        for batch_ids, batch_mask in loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            bs = batch_ids.shape[0]

            if mode in ("greedy", "both"):
                # Greedy
                out = model.generate(
                    input_ids=batch_ids, attention_mask=batch_mask,
                    max_length=MAX_LENGTH, do_sample=False, num_beams=1,
                )
                decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
                for i, d in enumerate(decoded):
                    all_candidates[idx + i].append(d.strip())

            if mode in ("sampling", "both"):
                # Multi-temperature sampling
                for temp in temperatures:
                    for sample_idx in range(bs):
                        ids_i = batch_ids[sample_idx:sample_idx+1]
                        mask_i = batch_mask[sample_idx:sample_idx+1]
                        samp_out = model.generate(
                            input_ids=ids_i, attention_mask=mask_i,
                            max_length=MAX_LENGTH,
                            do_sample=True, num_beams=1,
                            top_p=0.9, temperature=temp,
                            num_return_sequences=num_samples,
                            repetition_penalty=1.2,
                        )
                        decoded = tokenizer.batch_decode(samp_out, skip_special_tokens=True)
                        for d in decoded:
                            all_candidates[idx + sample_idx].append(d.strip())

            idx += bs

    return all_candidates

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both", choices=["greedy", "sampling", "both"],
                        help="Candidate generation mode per model")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per temperature per model")
    parser.add_argument("--folds", default="0,1,2,3,4",
                        help="Comma-separated fold indices to use")
    parser.add_argument("--model_type", default="best", choices=["best", "last"],
                        help="Which checkpoint to load")
    args = parser.parse_args()

    fold_indices = [int(x) for x in args.folds.split(",")]
    model_name = f"{args.model_type}_model"

    # Prepare val data
    sent_inputs, sent_refs = prepare_val_data()

    # Load models one-at-a-time to save VRAM, generate candidates, then unload
    all_candidates = [[] for _ in range(len(sent_inputs))]
    temperatures = [0.6, 0.8, 1.05]

    for fold_i in fold_indices:
        model_path = str(EXP023_DIR / "results" / f"fold{fold_i}" / model_name)
        logger.info(f"Loading fold{fold_i} model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        model.eval()

        t0 = time.time()
        fold_candidates = generate_candidates_single_model(
            model, tokenizer, sent_inputs,
            mode=args.mode, num_samples=args.num_samples,
            temperatures=temperatures,
        )
        elapsed = time.time() - t0

        # Merge candidates
        for i in range(len(sent_inputs)):
            all_candidates[i].extend(fold_candidates[i])

        n_cands = len(fold_candidates[0]) if fold_candidates else 0
        logger.info(f"  fold{fold_i}: {n_cands} candidates/sample, {elapsed:.0f}s")

        # Unload model to free VRAM
        del model
        torch.cuda.empty_cache()

    # Log candidate pool stats
    pool_sizes = [len(c) for c in all_candidates]
    logger.info(f"Candidate pool: mean={np.mean(pool_sizes):.1f}, "
                f"min={min(pool_sizes)}, max={max(pool_sizes)}")
    unique_sizes = [len(set(c)) for c in all_candidates]
    logger.info(f"Unique candidates: mean={np.mean(unique_sizes):.1f}, "
                f"min={min(unique_sizes)}, max={max(unique_sizes)}")

    # MBR selection
    logger.info("Running MBR selection...")
    t0 = time.time()
    preds = [mbr_pick(cands) for cands in all_candidates]
    mbr_time = time.time() - t0
    logger.info(f"MBR selection: {mbr_time:.0f}s")

    # Evaluate
    chrf_score, bleu_score, geo, rep = calc_metrics(preds, sent_refs)
    logger.info(f"\n{'='*60}")
    logger.info(f"Cross-model MBR Results (fold{EVAL_FOLD} sent-CV)")
    logger.info(f"  Folds: {fold_indices}")
    logger.info(f"  Mode: {args.mode}, samples/temp: {args.num_samples}")
    logger.info(f"  chrF++={chrf_score:.2f}, BLEU={bleu_score:.2f}, geo={geo:.2f}, rep={rep:.1f}%")
    logger.info(f"{'='*60}")

    # Also evaluate greedy-only (fold3) as baseline
    logger.info("\n--- Baselines (from candidates) ---")
    # fold3 greedy = first candidate if fold3 is included and mode includes greedy
    if EVAL_FOLD in fold_indices and args.mode in ("greedy", "both"):
        fold3_idx = fold_indices.index(EVAL_FOLD)
        if args.mode == "greedy":
            greedy_preds = [c[fold3_idx] if fold3_idx < len(c) else "" for c in all_candidates]
        else:
            # "both" mode: greedy is the first candidate per fold
            n_per_fold = 1 + args.num_samples * len(temperatures)
            greedy_idx = fold3_idx * n_per_fold
            greedy_preds = [c[greedy_idx] if greedy_idx < len(c) else "" for c in all_candidates]
        chrf_g, bleu_g, geo_g, rep_g = calc_metrics(greedy_preds, sent_refs)
        logger.info(f"  fold3 greedy: chrF++={chrf_g:.2f}, BLEU={bleu_g:.2f}, geo={geo_g:.2f}, rep={rep_g:.1f}%")

    # Save results
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "cross_model_mbr_results.txt"
    with open(str(out_path), "a") as f:
        f.write(f"folds={fold_indices} mode={args.mode} samples={args.num_samples} "
                f"chrF++={chrf_score:.2f} BLEU={bleu_score:.2f} geo={geo:.2f} rep={rep:.1f}%\n")
    logger.info(f"Results appended to {out_path}")


if __name__ == "__main__":
    main()
