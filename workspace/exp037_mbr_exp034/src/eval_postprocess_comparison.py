"""
exp037: 後処理の差分比較 — MBR前にノートブック式後処理を適用する効果を測定
- キャッシュ済み候補を使うため再推論不要
- (A) 後処理なし → MBR → repeat_cleanup (現状)
- (B) ノートブック式後処理 → MBR → repeat_cleanup
- (C) 自前(v8)式後処理 → MBR → repeat_cleanup

Usage: python eval_postprocess_comparison.py [--fold N]
"""
import os
import re
import sys
import math
import json
import logging
import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import sacrebleu

# ============================================================
# Args & Config
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=3)
args = parser.parse_args()
FOLD = args.fold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CACHE_DIR = RESULTS_DIR / "ensemble_mbr_cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ============================================================
# Preprocessing (same as eval_ensemble_mbr_notebook.py)
# ============================================================
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002


def _decimal_to_fraction_approx(match):
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


def preprocess_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction_approx, text)
    return text


# ============================================================
# Data: GroupKFold split
# ============================================================
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"

train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_df["akt_group"] = train_df["oare_id"].map(oare_to_group).fillna("None")

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_df, groups=train_df["akt_group"].values))
_, val_idx = splits[FOLD]
val_doc_data = train_df.iloc[val_idx].copy()

sent_aligned = pd.read_csv(str(SENTENCE_ALIGNED_PATH))
alignment_dict = {}
for oare_id, group in sent_aligned.groupby('oare_id'):
    group = group.sort_values('sent_idx')
    alignment_dict[oare_id] = [
        {'akk_segment': str(r['akk_segment']), 'eng_sentence': str(r['eng_sentence'])}
        for _, r in group.iterrows()
    ]

prefix_fwd = "translate Akkadian to English: "

sent_inputs_raw, sent_inputs, sent_refs = [], [], []
for _, row in val_doc_data.iterrows():
    oare_id = row['oare_id']
    if oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            for b in boundaries:
                if b['akk_segment'].strip() and b['eng_sentence'].strip():
                    raw = preprocess_transliteration(b['akk_segment'])
                    sent_inputs_raw.append(raw)
                    sent_inputs.append(prefix_fwd + raw)
                    sent_refs.append(b['eng_sentence'])

doc_inputs_raw, doc_inputs, doc_refs = [], [], []
for _, row in val_doc_data.iterrows():
    translit = str(row['transliteration'])
    translation = str(row['translation'])
    if translit.strip() and translation.strip():
        raw = preprocess_transliteration(translit)
        doc_inputs_raw.append(raw)
        doc_inputs.append(prefix_fwd + raw)
        doc_refs.append(translation)

n_sent = len(sent_inputs)
n_doc = len(doc_inputs)
logger.info(f"Fold {FOLD}: sent-CV={n_sent}, doc-CV={n_doc}")

# ============================================================
# Load cached candidates
# ============================================================
with open(CACHE_DIR / f"model_a_multi_fold{FOLD}.pkl", "rb") as f:
    model_a = pickle.load(f)
with open(CACHE_DIR / f"model_b_multi_fold{FOLD}.pkl", "rb") as f:
    model_b = pickle.load(f)
logger.info("Loaded cached candidates for model_a and model_b")

# ============================================================
# Postprocessor A: None (current approach)
# ============================================================


# ============================================================
# Postprocessor B: Notebook-style (VectorizedPostprocessor)
# ============================================================
_PN_RE = re.compile(r"\bPN\b")
_WS_RE = re.compile(r"\s+")

# Gap normalization (from notebook preprocessing, reused in postprocessing)
_GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I
)

_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)", re.I
)
_BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE = re.compile(r"\(\?\)")
_CURLY_QUOTES_RE = re.compile("[\u201c\u201d\u2018\u2019]")

_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
              "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}

_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")

_FORBIDDEN_TRANS = str.maketrans("", "", '()——<>⌈⌋⌊[]+ʾ;')

_COMMODITY_RE = re.compile(r'-(gold|tax|textiles)\b')
_COMMODITY_REPL = {"gold": "pašallum gold", "tax": "šadduātum tax", "textiles": "kutānum textiles"}

_SHEKEL_REPLS = [
    (re.compile(r'5\s+11\s*/\s*12\s+shekels?', re.I), '6 shekels less 15 grains'),
    (re.compile(r'5\s*/\s*12\s+shekels?', re.I), '⅔ shekel 15 grains'),
    (re.compile(r'7\s*/\s*12\s+shekels?', re.I), '½ shekel 15 grains'),
    (re.compile(r'1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?', re.I), '15 grains'),
]

_SLASH_ALT_RE = re.compile(r'(?<!\d)\s*/\s*(?!\d)\S+')
_STRAY_MARKS_RE = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')
_MULTI_GAP_RE = re.compile(r'(?:<gap>\s*){2,}')

_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚", "0.6666": "⅔", "0.3333": "⅓", "0.1666": "⅙",
    "0.625": "⅝", "0.75": "¾", "0.25": "¼", "0.5": "½",
}
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")

_ALLOWED_FRACS = [
    (1/6, "0.16666"), (1/4, "0.25"), (1/3, "0.33333"),
    (1/2, "0.5"), (2/3, "0.66666"), (3/4, "0.75"), (5/6, "0.83333"),
]
_FRAC_TOL = 2e-3


def _frac_repl(m):
    return _EXACT_FRAC_MAP[m.group(0)]


def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))
    if abs(frac - best[0]) <= _FRAC_TOL:
        dec = best[1]
        if ip == 0:
            return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    return f"{x:.5f}".rstrip("0").rstrip(".")


def _commodity_repl_fn(m):
    return _COMMODITY_REPL[m.group(1)]


def _month_repl(m):
    return f"Month {_ROMAN2INT.get(m.group(1).upper(), m.group(1))}"


def notebook_postprocess(translations: List[str]) -> List[str]:
    """Notebook-style VectorizedPostprocessor."""
    s = pd.Series(translations).fillna("").astype(str)

    s = s.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)
    s = s.str.replace(_PN_RE, "<gap>", regex=True)
    s = s.str.replace(_COMMODITY_RE, _commodity_repl_fn, regex=True)

    for pat, repl in _SHEKEL_REPLS:
        s = s.str.replace(pat, repl, regex=True)

    s = s.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
    s = s.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)

    s = s.str.replace(_SOFT_GRAM_RE, " ", regex=True)
    s = s.str.replace(_BARE_GRAM_RE, " ", regex=True)
    s = s.str.replace(_UNCERTAIN_RE, "", regex=True)

    s = s.str.replace(_STRAY_MARKS_RE, "", regex=True)
    s = s.str.replace(_SLASH_ALT_RE, "", regex=True)
    s = s.str.replace(_CURLY_QUOTES_RE, "", regex=True)

    s = s.str.replace(_MONTH_RE, _month_repl, regex=True)
    s = s.str.replace(_MULTI_GAP_RE, "<gap>", regex=True)

    s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
    s = s.str.translate(_FORBIDDEN_TRANS)
    s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)

    s = s.str.replace(_REPEAT_WORD_RE, r"\1", regex=True)
    for n in range(4, 1, -1):
        pat = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
        s = s.str.replace(pat, r"\1", regex=True)

    s = s.str.replace(_PUNCT_SPACE_RE, r"\1", regex=True)
    s = s.str.replace(_REPEAT_PUNCT_RE, r"\1", regex=True)
    s = s.str.replace(_WS_RE, " ", regex=True).str.strip()

    return s.tolist()


# ============================================================
# Postprocessor C: Our v8 style (clean_translation)
# ============================================================
ROMAN_TO_INT = {
    "XII": "12", "XI": "11", "VIII": "8", "VII": "7",
    "VI": "6", "IV": "4", "IX": "9", "III": "3",
    "II": "2", "X": "10", "V": "5", "I": "1",
}
MONTH_NAMES_TRANSLATION = {
    r"B[eē]lat[\s-]ekall[ie]m": "1",
    r"[Šš]a[\s-]sarr[aā]tim": "2",
    r"[Kk]en[aā]tim": "3",
    r"[Šš]a[\s-]k[eē]n[aā]tim": "3",
    r"Ma[hḫ]h?ur[\s-]il[iī]": "4",
    r"Ab[\s-]?[šš]arr[aā]ni": "5",
    r"[Aa]b[sš]arrani": "5",
    r"[Hh]ubur": "6",
    r"[Ṣṣ]ip['\u2019]?um": "7",
    r"[Qq]arr[aā]['\u2019]?[aā]tum": "8",
    r"[Qq]arr[aā]tum": "8",
    r"[Kk]an[wm]arta": "9",
    r"[Tt]e['\u2019\u02BE]?in[aā]tum": "10",
    r"[Tt][eē]['\u2019\u02BE]?in[aā]tum": "10",
    r"[Kk]uzall?[iu]m?": "11",
    r"[Aa]llan[aā]tum": "12",
}


def v8_postprocess(translations: List[str]) -> List[str]:
    """Our v8 clean_translation applied to a list."""
    results = []
    for text in translations:
        if not isinstance(text, str) or not text.strip():
            results.append(text)
            continue
        text = re.sub(r'\bfem\.\s*', '', text)
        text = re.sub(r'\bsing\.\s*', '', text)
        text = re.sub(r'\bpl\.\s*', '', text)
        text = re.sub(r'\bplural\b\s*', '', text)
        text = text.replace('(?)', '')
        text = re.sub(r'<<\s*>>', '', text)
        text = re.sub(r'<\s+>', '', text)
        text = re.sub(r'(?<!\.)\.\.(?!\.)', '', text)
        text = re.sub(r'\bxx?\b', '', text)
        text = re.sub(r'\bPN\b', '<gap>', text)
        text = re.sub(r'\b-gold\b', 'pašallum gold', text)
        text = re.sub(r'\b-tax\b', 'šadduātum tax', text)
        text = re.sub(r'\b-textiles\b', 'kutānum textiles', text)
        text = re.sub(r'(\S+)\s*/\s*\S+', r'\1', text)
        text = re.sub(r'\(m\)', '{m}', text)
        text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
        text = re.sub(r'\d+\.\d+', _decimal_to_fraction_approx, text)
        for roman, integer in sorted(ROMAN_TO_INT.items(), key=lambda x: -len(x[0])):
            text = re.sub(rf'\bmonth\s+{roman}(?=[\s,.:;!?\)]|$)', f'month {integer}', text)
        for pattern, number in MONTH_NAMES_TRANSLATION.items():
            text = re.sub(rf'\bmonth\s+{pattern}\b', f'month {number}', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        results.append(text)
    return results


# ============================================================
# MBR selection
# ============================================================
chrfpp_metric = sacrebleu.metrics.CHRF(word_order=2)


def mbr_select(candidates):
    seen = set()
    unique = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    if len(unique) == 0:
        return ""
    if len(unique) == 1:
        return unique[0]
    nc = len(unique)
    scores = []
    for i in range(nc):
        s = sum(float(chrfpp_metric.sentence_score(unique[i], [unique[j]]).score)
                for j in range(nc) if j != i)
        scores.append(s / (nc - 1))
    return unique[int(np.argmax(scores))]


# ============================================================
# Metrics
# ============================================================
def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for ng in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * ng + 1):
            if words[i:i + ng] == words[i + ng:i + 2 * ng]:
                return " ".join(words[:i + ng])
    return text


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


def calc_metrics(preds, refs):
    preds_clean = [repeat_cleanup(p) for p in preds]
    chrf = sacrebleu.corpus_chrf(preds_clean, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds_clean, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in preds_clean) / len(preds_clean)
    return {
        "chrf": round(chrf.score, 2),
        "bleu": round(bleu.score, 2),
        "geo": round(geo, 2),
        "rep": round(rep_rate, 1),
    }


# ============================================================
# Build full pool (12 candidates) and run comparison
# ============================================================
POSTPROCESSORS = {
    "none": lambda cands: cands,
    "notebook": notebook_postprocess,
    "v8": v8_postprocess,
}

logger.info("=" * 60)
logger.info("=== Postprocessing comparison: full pool MBR (12 cands) ===")
logger.info("=" * 60)

# --- SENT-CV ---
logger.info("\n=== SENT-CV ===")
sent_results = {}

for pp_name, pp_fn in POSTPROCESSORS.items():
    pools_raw = [
        model_a["sent_beam4"][i] + model_a["sent_sample"][i]
        + model_b["sent_beam4"][i] + model_b["sent_sample"][i]
        for i in range(n_sent)
    ]
    # Apply postprocessing to candidates BEFORE MBR
    pools_pp = [pp_fn(pool) for pool in pools_raw]
    preds = [mbr_select(pool) for pool in tqdm(pools_pp, desc=f"sent {pp_name}", leave=False)]
    m = calc_metrics(preds, sent_refs)
    sent_results[pp_name] = m
    logger.info(f"  postprocess={pp_name:<12} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

# Also test: postprocess applied AFTER MBR (to final output only)
for pp_name, pp_fn in [("notebook_after", notebook_postprocess), ("v8_after", v8_postprocess)]:
    pools_raw = [
        model_a["sent_beam4"][i] + model_a["sent_sample"][i]
        + model_b["sent_beam4"][i] + model_b["sent_sample"][i]
        for i in range(n_sent)
    ]
    # MBR on raw candidates, then postprocess the selected output
    preds_raw = [mbr_select(pool) for pool in tqdm(pools_raw, desc=f"sent {pp_name}", leave=False)]
    preds = pp_fn(preds_raw)
    m = calc_metrics(preds, sent_refs)
    sent_results[pp_name] = m
    logger.info(f"  postprocess={pp_name:<12} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

# --- DOC-CV ---
logger.info("\n=== DOC-CV ===")
doc_results = {}

for pp_name, pp_fn in POSTPROCESSORS.items():
    pools_raw = [
        model_a["doc_beam4"][i] + model_a["doc_sample"][i]
        + model_b["doc_beam4"][i] + model_b["doc_sample"][i]
        for i in range(n_doc)
    ]
    pools_pp = [pp_fn(pool) for pool in pools_raw]
    preds = [mbr_select(pool) for pool in tqdm(pools_pp, desc=f"doc {pp_name}", leave=False)]
    m = calc_metrics(preds, doc_refs)
    doc_results[pp_name] = m
    logger.info(f"  postprocess={pp_name:<12} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

for pp_name, pp_fn in [("notebook_after", notebook_postprocess), ("v8_after", v8_postprocess)]:
    pools_raw = [
        model_a["doc_beam4"][i] + model_a["doc_sample"][i]
        + model_b["doc_beam4"][i] + model_b["doc_sample"][i]
        for i in range(n_doc)
    ]
    preds_raw = [mbr_select(pool) for pool in tqdm(pools_raw, desc=f"doc {pp_name}", leave=False)]
    preds = pp_fn(preds_raw)
    m = calc_metrics(preds, doc_refs)
    doc_results[pp_name] = m
    logger.info(f"  postprocess={pp_name:<12} chrF++={m['chrf']:>6.2f}  BLEU={m['bleu']:>6.2f}  geo={m['geo']:>6.2f}  rep={m['rep']:>5.1f}%")

# --- COMBINED ---
logger.info("\n" + "=" * 60)
logger.info("=== COMBINED ===")
logger.info(f"{'postprocess':<20} {'sent-geo':>10} {'sent-rep':>10} {'doc-geo':>10} {'doc-rep':>10}")
logger.info("-" * 65)
for pp_name in sent_results:
    s = sent_results[pp_name]
    d = doc_results[pp_name]
    logger.info(f"{pp_name:<20} {s['geo']:>10.2f} {s['rep']:>9.1f}% {d['geo']:>10.2f} {d['rep']:>9.1f}%")

# Save
output = {"fold": FOLD, "sent_cv": sent_results, "doc_cv": doc_results}
out_path = RESULTS_DIR / f"postprocess_comparison_fold{FOLD}.json"
with open(str(out_path), "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
logger.info(f"\nSaved to {out_path}")
