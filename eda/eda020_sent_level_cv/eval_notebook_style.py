"""
exp034 fold3 last_modelを、notebookと同じ推論方法(MBR)+後処理で評価。
現行のeval_full_doc.py(beam=4, repeat_cleanup)との比較用。

推論設定の比較:
  現行:     beam=4, early_stopping=True
  notebook: beam=8, MBR(beam4+sample8), length_penalty=1.3, repetition_penalty=1.2

後処理の比較:
  現行:     repeat_cleanup のみ
  notebook: VectorizedPostprocessor(引用符除去、括弧除去、繰り返し除去、gap正規化等)
"""
import os
import re
import sys
import math
import logging
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("--fold", type=int, default=3)
args = parser.parse_args()

MODEL_PATH = args.model_path
FOLD = args.fold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
SENTENCE_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"
AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"

MAX_LENGTH = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Preprocessing (exp023 style)
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
    best_frac = None
    best_dist = float('inf')
    for target, symbol in FRACTION_TARGETS.items():
        dist = abs(frac_part - target)
        if dist < best_dist:
            best_dist = dist
            best_frac = symbol
    if best_dist <= APPROX_TOLERANCE:
        if int_part == 0:
            return best_frac
        else:
            return f"{int_part} {best_frac}"
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
# Data: GroupKFold val split
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))


def simple_sentence_aligner(df, keep_oare_id=False):
    aligned_data = []
    for idx, row in df.iterrows():
        src = str(row["transliteration"])
        tgt = str(row["translation"])
        tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split("\n") if s.strip()]
        entry_base = {"oare_id": row["oare_id"]} if keep_oare_id else {}
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({**entry_base, "transliteration": s, "translation": t})
        else:
            aligned_data.append({**entry_base, "transliteration": src, "translation": tgt})
    return pd.DataFrame(aligned_data)


from sklearn.model_selection import GroupKFold
akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_expanded = simple_sentence_aligner(train_df, keep_oare_id=True)
train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=train_expanded["akt_group"].values))
_, val_idx = splits[FOLD]
val_data = train_expanded.iloc[val_idx].copy()
logger.info(f"GroupKFold fold={FOLD}, val={len(val_data)} samples")

# Build eval samples
sent_aligned = pd.read_csv(str(SENTENCE_ALIGNED_PATH))
oare_to_row = {}
for _, row in train_df.iterrows():
    oare_to_row[row['oare_id']] = row

alignment_dict = {}
for oare_id, group in sent_aligned.groupby('oare_id'):
    group = group.sort_values('sent_idx')
    boundaries = []
    for _, row in group.iterrows():
        boundaries.append({
            'sent_idx': int(row['sent_idx']),
            'akk_segment': str(row['akk_segment']),
            'eng_sentence': str(row['eng_sentence']),
        })
    alignment_dict[oare_id] = boundaries

translit_to_oare = {}
for oare_id, row in oare_to_row.items():
    translit_to_oare[str(row['transliteration'])] = oare_id

prefix = "translate Akkadian to English: "
sent_inputs, sent_refs = [], []
split_doc_ids = set()
for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    oare_id = translit_to_oare.get(translit)
    if oare_id and oare_id in alignment_dict:
        boundaries = alignment_dict[oare_id]
        if len(boundaries) <= 6:
            split_doc_ids.add(idx)
            for b in boundaries:
                akk_seg = b['akk_segment']
                eng_sent = b['eng_sentence']
                if akk_seg.strip() and eng_sent.strip():
                    sent_inputs.append(prefix + preprocess_transliteration(akk_seg))
                    sent_refs.append(eng_sent)

doc_inputs, doc_refs = [], []
for idx, row in val_data.iterrows():
    translit = str(row['transliteration'])
    translation = str(row['translation'])
    if translit.strip() and translation.strip():
        doc_inputs.append(prefix + preprocess_transliteration(translit))
        doc_refs.append(translation)

logger.info(f"sent-CV: {len(sent_inputs)} sents, doc-CV: {len(doc_inputs)} docs")

# ============================================================
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info(f"Model loaded from {MODEL_PATH}")


class InferenceDataset(TorchDataset):
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
# Postprocessing: existing (repeat_cleanup)
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


# ============================================================
# Postprocessing: notebook style (VectorizedPostprocessor)
# ============================================================
_QUOTES_RE = re.compile(r'["""'']')
_SOFT_GRAM_PARENS_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)",
    re.I,
)
_REPEATED_WORDS_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:;])")
_REPEATED_PUNCT_RE = re.compile(r"([.,:;])\1+")
_WS_RE = re.compile(r"\s+")
_PN_RE = re.compile(r"\bPN\b")
_GAP_LEGACY_RE = re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I)
_BIG_GAP_LEGACY_RE = re.compile(r"(\.{3,}|…|\[\.+\])")
FORBIDDEN_CHARS = "()""''—–⌈⌋⌊+ʾ"
FORBIDDEN_TRANS = str.maketrans("", "", FORBIDDEN_CHARS)

ROMAN_TO_INT = {
    "XII": "12", "XI": "11", "VIII": "8", "VII": "7",
    "VI": "6", "IV": "4", "IX": "9", "III": "3",
    "II": "2", "X": "10", "V": "5", "I": "1",
}
_MONTH_ROMAN_RE = re.compile(
    r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.IGNORECASE
)


def notebook_postprocess(text):
    """Notebook's VectorizedPostprocessor logic (single-string version)."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # PN -> <gap>
    text = _PN_RE.sub("<gap>", text)
    # legacy gap patterns
    text = _GAP_LEGACY_RE.sub("<gap>", text)
    text = _BIG_GAP_LEGACY_RE.sub("<gap>", text)

    # soft grammatical parentheses removal
    text = _SOFT_GRAM_PARENS_RE.sub(" ", text)
    # quote removal
    text = _QUOTES_RE.sub("", text)

    # collapse consecutive <gap>
    text = re.sub(r"(<gap>\s*){2,}", "<gap> ", text)

    # protect <gap>, remove forbidden chars, restore
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.translate(FORBIDDEN_TRANS)
    text = text.replace("\x00GAP\x00", " <gap> ")

    # month roman -> int
    def _month_repl(m):
        r = m.group(1).upper()
        return f"Month {ROMAN_TO_INT.get(r, r)}"
    text = _MONTH_ROMAN_RE.sub(_month_repl, text)

    # word-level repeat removal
    text = _REPEATED_WORDS_RE.sub(r"\1", text)
    # phrase-level repeat removal (2-4 words)
    for n in range(4, 1, -1):
        pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
        text = re.sub(pattern, r"\1", text)

    # punctuation cleanup
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)

    text = _WS_RE.sub(" ", text).strip()
    return text


# ============================================================
# MBR: notebook style (pure-Python BLEU + chrF++)
# ============================================================
def _get_ngrams(tokens, n):
    counts = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _sentence_bleu(hypothesis, reference, max_n=4):
    hyp_tok = hypothesis.strip().split()
    ref_tok = reference.strip().split()
    if not hyp_tok or not ref_tok:
        return 0.0
    bp = min(1.0, math.exp(1.0 - len(ref_tok) / max(len(hyp_tok), 1)))
    log_avg = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = _get_ngrams(hyp_tok, n)
        ref_ng = _get_ngrams(ref_tok, n)
        clipped = sum(min(cnt, ref_ng.get(ng, 0)) for ng, cnt in hyp_ng.items())
        total = sum(hyp_ng.values())
        if n == 1:
            if total == 0:
                return 0.0
            precision = clipped / total
            if precision == 0:
                return 0.0
        else:
            precision = (clipped + 1) / (total + 1)
        log_avg += math.log(max(precision, 1e-100)) / max_n
    return bp * math.exp(log_avg)


def _char_ngrams(text, n):
    counts = {}
    for i in range(len(text) - n + 1):
        ng = text[i:i + n]
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _sentence_chrfpp(hypothesis, reference, max_char_n=6, max_word_n=2, beta=2.0):
    hyp = hypothesis.strip()
    ref = reference.strip()
    if not hyp or not ref:
        return 0.0
    total_prec_num = total_prec_den = total_rec_num = total_rec_den = 0.0
    for n in range(1, max_char_n + 1):
        hyp_ng = _char_ngrams(hyp, n)
        ref_ng = _char_ngrams(ref, n)
        matched = sum(min(hyp_ng.get(ng, 0), ref_ng.get(ng, 0)) for ng in hyp_ng)
        total_prec_num += matched
        total_prec_den += sum(hyp_ng.values())
        total_rec_num += matched
        total_rec_den += sum(ref_ng.values())
    hyp_tok = hyp.split()
    ref_tok = ref.split()
    for n in range(1, max_word_n + 1):
        hyp_ng = _get_ngrams(hyp_tok, n)
        ref_ng = _get_ngrams(ref_tok, n)
        matched = sum(min(hyp_ng.get(ng, 0), ref_ng.get(ng, 0)) for ng in hyp_ng)
        total_prec_num += matched
        total_prec_den += sum(hyp_ng.values())
        total_rec_num += matched
        total_rec_den += sum(ref_ng.values())
    precision = total_prec_num / max(total_prec_den, 1e-100)
    recall = total_rec_num / max(total_rec_den, 1e-100)
    if precision + recall == 0:
        return 0.0
    beta2 = beta * beta
    score = (1 + beta2) * precision * recall / (beta2 * precision + recall)
    return score * 100.0


def _competition_metric(hypothesis, reference):
    bleu = _sentence_bleu(hypothesis, reference)
    chrfpp = _sentence_chrfpp(hypothesis, reference) / 100.0
    if bleu <= 0 or chrfpp <= 0:
        return 0.0
    return math.sqrt(bleu * chrfpp)


def _dedup_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        x = str(x).strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def mbr_pick(cands, pool_cap=32):
    cands = _dedup_keep_order(cands)
    if pool_cap:
        cands = cands[:pool_cap]
    n = len(cands)
    if n == 0:
        return ""
    if n == 1:
        return cands[0]
    scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _competition_metric(cands[i], cands[j])
            scores[i][j] = s
            scores[j][i] = s
    best_i, best_avg = 0, -1.0
    for i in range(n):
        avg = sum(scores[i]) / (n - 1)
        if avg > best_avg:
            best_avg, best_i = avg, i
    return cands[best_i]


# ============================================================
# Inference strategies
# ============================================================
def run_beam4(inputs, desc):
    """現行: beam=4"""
    ds = InferenceDataset(inputs, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=4, early_stopping=True,
            )
            preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    return preds


def run_greedy(inputs, desc):
    """greedy decoding"""
    ds = InferenceDataset(inputs, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH,
            )
            preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    return preds


def run_notebook_mbr(inputs, desc):
    """Notebook style: beam=8 candidates (return 4) + sampling 8 candidates, MBR select."""
    ds = InferenceDataset(inputs, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=2, shuffle=False)  # smaller batch for MBR
    preds = []
    gen_common = dict(
        max_new_tokens=256,
        repetition_penalty=1.2,
        use_cache=True,
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            B = ids.shape[0]

            # beam candidates
            beam_out = model.generate(
                input_ids=ids, attention_mask=mask,
                do_sample=False,
                num_beams=8,
                num_return_sequences=4,
                length_penalty=1.3,
                early_stopping=True,
                **gen_common,
            )
            beam_txt = tokenizer.batch_decode(beam_out, skip_special_tokens=True)

            # sampling candidates
            torch.manual_seed(42)
            random.seed(42)
            samp_out = model.generate(
                input_ids=ids, attention_mask=mask,
                do_sample=True,
                num_beams=1,
                top_p=0.9,
                temperature=0.7,
                num_return_sequences=8,
                max_new_tokens=256,
                use_cache=True,
            )
            samp_txt = tokenizer.batch_decode(samp_out, skip_special_tokens=True)

            # MBR pick per example
            for i in range(B):
                pool = beam_txt[i*4:(i+1)*4] + samp_txt[i*8:(i+1)*8]
                chosen = mbr_pick(pool)
                preds.append(chosen)

    return preds


# ============================================================
# Metrics
# ============================================================
def calc_metrics(preds, refs, label):
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in preds) / len(preds)
    logger.info(f"  {label}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep_rate:.1f}%")
    return chrf.score, bleu.score, geo, rep_rate


# ============================================================
# Main: run all strategies and compare
# ============================================================
logger.info("=" * 60)
logger.info("Strategy comparison for exp034 fold3 last_model")
logger.info("=" * 60)

# 1. Current: beam=4 + repeat_cleanup
logger.info("\n[1] beam=4 + repeat_cleanup (current)")
sent_beam4 = run_beam4(sent_inputs, "beam4 sent")
doc_beam4 = run_beam4(doc_inputs, "beam4 doc")
calc_metrics([repeat_cleanup(p) for p in sent_beam4], sent_refs, "sent-CV")
calc_metrics([repeat_cleanup(p) for p in doc_beam4], doc_refs, "doc-CV")

# 2. Greedy + repeat_cleanup (exp030 showed greedy > beam4)
logger.info("\n[2] greedy + repeat_cleanup")
sent_greedy = run_greedy(sent_inputs, "greedy sent")
doc_greedy = run_greedy(doc_inputs, "greedy doc")
calc_metrics([repeat_cleanup(p) for p in sent_greedy], sent_refs, "sent-CV")
calc_metrics([repeat_cleanup(p) for p in doc_greedy], doc_refs, "doc-CV")

# 3. Notebook MBR + repeat_cleanup (existing postprocess)
logger.info("\n[3] notebook MBR + repeat_cleanup")
sent_mbr = run_notebook_mbr(sent_inputs, "MBR sent")
doc_mbr = run_notebook_mbr(doc_inputs, "MBR doc")
calc_metrics([repeat_cleanup(p) for p in sent_mbr], sent_refs, "sent-CV")
calc_metrics([repeat_cleanup(p) for p in doc_mbr], doc_refs, "doc-CV")

# 4. Notebook MBR + notebook postprocess
logger.info("\n[4] notebook MBR + notebook postprocess")
calc_metrics([notebook_postprocess(p) for p in sent_mbr], sent_refs, "sent-CV")
calc_metrics([notebook_postprocess(p) for p in doc_mbr], doc_refs, "doc-CV")

logger.info("\nDone!")
