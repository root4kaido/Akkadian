"""
exp007: MBRデコーディング + OA_Lexicon後処理 + repeated_removal + Translation Memory
exp005のモデルを使い、推論パイプラインのみ改善
"""
import os
import re
import sys
import yaml
import logging
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import sacrebleu
import evaluate
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================
# Setup
# ============================================================
EXP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "infer_mbr.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

with open(EXP_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ============================================================
# Dataset
# ============================================================
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }

# ============================================================
# MBR Decoding
# ============================================================
class MBRDecoder:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config["inference"]["mbr"]
        self.max_length = config["inference"].get("max_length", 512)
        self._chrfpp = sacrebleu.metrics.CHRF(word_order=2)

    def _sim_chrfpp(self, a: str, b: str) -> float:
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return 0.0
        return float(self._chrfpp.sentence_score(a, [b]).score)

    @staticmethod
    def _dedup_keep_order(xs):
        seen = set()
        out = []
        for x in xs:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def _mbr_pick(self, cands: List[str]) -> str:
        cands = self._dedup_keep_order(cands)
        cands = cands[:32]  # pool cap

        n = len(cands)
        if n == 0:
            return ""
        if n == 1:
            return cands[0]

        best_i, best_s = 0, -1e9
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i == j:
                    continue
                s += self._sim_chrfpp(cands[i], cands[j])
            s /= max(1, n - 1)
            if s > best_s:
                best_s, best_i = s, i

        return cands[best_i]

    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask) -> List[str]:
        gen_common = {
            "max_new_tokens": self.max_length,
            "repetition_penalty": 1.2,
            "use_cache": True,
        }

        B = input_ids.shape[0]
        num_beam_cands = self.cfg["num_beams"]  # 4
        num_sample_cands = self.cfg["num_sampling"]  # 2
        nb = max(8, num_beam_cands)  # num_beams >= num_return_sequences

        # Beam candidates
        beam_out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=nb,
            num_return_sequences=num_beam_cands,
            length_penalty=1.3,
            early_stopping=True,
            **gen_common,
        )
        beam_txt = self.tokenizer.batch_decode(beam_out, skip_special_tokens=True)

        pools = [[] for _ in range(B)]
        for i in range(B):
            pools[i].extend(beam_txt[i * num_beam_cands:(i + 1) * num_beam_cands])

        # Sampling candidates
        if num_sample_cands > 0:
            samp_out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                num_beams=1,
                top_p=self.cfg.get("sampling_top_p", 0.9),
                temperature=self.cfg.get("sampling_temperature", 0.7),
                num_return_sequences=num_sample_cands,
                max_new_tokens=self.max_length,
                use_cache=True,
                repetition_penalty=1.2,
            )
            samp_txt = self.tokenizer.batch_decode(samp_out, skip_special_tokens=True)

            for i in range(B):
                pools[i].extend(samp_txt[i * num_sample_cands:(i + 1) * num_sample_cands])

        chosen = [self._mbr_pick(p) for p in pools]
        return chosen


# ============================================================
# OA Lexicon Post-processing
# ============================================================
_DIACRITIC_MAP = str.maketrans({
    "š": "s", "Š": "s", "ṣ": "s", "Ṣ": "s",
    "ṭ": "t", "Ṭ": "t", "ḫ": "h", "Ḫ": "h",
    "ā": "a", "Ā": "a", "ē": "e", "Ē": "e",
    "ī": "i", "Ī": "i", "ū": "u", "Ū": "u",
    "ʾ": "", "ʼ": "", "'": "", "'": "",
})

_DIACRITIC_CHARS = set("šŠṣṢṭṬḫḪāēīūĀĒĪŪ")

SUB_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

EXPLICIT_NE_TYPES = {"DN", "GN", "PN", "MN", "ON", "TN"}
EN_STOPWORDS = {
    'the','a','an','and','or','of','to','in','on','at','by','for','from','with','as','but','not','no','nor',
    'is','are','was','were','be','been','being',
    'i','you','he','she','it','we','they','me','him','her','us','them','my','your','his','their','our','its',
    'this','that','these','those','there','here',
    'who','whom','which','what','when','where','why','how',
}


def _strip_disambig(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"(?<=\D)\d+$", "", s)
    return s


def norm_key_token(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s).translate(SUB_DIGITS).strip()
    s = re.sub(r"^[\"'\u201c\u201d\u2018\u2019()\[\]{}<>]+", "", s)
    s = re.sub(r"[\"'\u201c\u201d\u2018\u2019()\[\]{}<>]+$", "", s)
    s = s.strip(".,;:!?")
    return s.lower()


def fold_for_match(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = _strip_disambig(s)
    s = s.translate(_DIACRITIC_MAP)
    s = s.lower()
    s = s.replace("sh", "s").replace("kh", "h")
    s = re.sub(r"[^a-z]+", "", s)
    return s


def looks_like_name(lexeme: str, typ: str) -> bool:
    if not lexeme:
        return False
    t = (typ or "").strip().upper()
    if t in EXPLICIT_NE_TYPES:
        return True
    if any(ch.isupper() for ch in lexeme):
        return True
    if any(ch in _DIACRITIC_CHARS for ch in lexeme):
        return True
    return False


class OALexiconPostProcessor:
    def __init__(self, lexicon_path: str, train_df: pd.DataFrame = None, train_path: str = None):
        logger.info(f"Loading OA Lexicon: {lexicon_path}")
        oa = pd.read_csv(lexicon_path)
        logger.info(f"OA Lexicon rows: {len(oa)}")

        self.token2lexemes = defaultdict(list)
        for _, r in oa.iterrows():
            typ = "" if pd.isna(r.get("type")) else str(r["type"]).strip()
            lex = "" if pd.isna(r.get("lexeme")) else str(r["lexeme"]).strip()
            if not lex:
                continue
            for col in ["form", "norm", "Alt_lex"]:
                if col not in oa.columns:
                    continue
                v = r.get(col)
                if pd.isna(v):
                    continue
                for tok in str(v).split():
                    k = norm_key_token(tok)
                    if k:
                        self.token2lexemes[k].append((lex, typ))

        # Dedup
        for k, v in list(self.token2lexemes.items()):
            seen = set()
            uniq = []
            for lex, typ in v:
                key = (lex, typ)
                if key not in seen:
                    seen.add(key)
                    uniq.append((lex, typ))
            self.token2lexemes[k] = uniq
        logger.info(f"OA token keys indexed: {len(self.token2lexemes)}")

        # Learn surface forms from train
        self.fold2surface = {}
        self.fold2freq = {}
        self._learn_surface_forms(train_df=train_df, train_path=train_path)

    def _learn_surface_forms(self, train_df: pd.DataFrame = None, train_path: str = None):
        if train_df is None and train_path is not None:
            train_df = pd.read_csv(train_path)
        if train_df is None:
            logger.warning("No train data for surface forms")
            return
        col = "translation" if "translation" in train_df.columns else train_df.columns[-1]

        surf_counter = defaultdict(Counter)
        token_re = re.compile(r"[A-Za-zšṣṭḫāēīūŠṢṬḪĀĒĪŪ''\-]+")

        for text in train_df[col].astype(str).tolist():
            for tok in token_re.findall(text):
                if len(tok) < 3:
                    continue
                if not (tok[:1].isupper() or any(ch in _DIACRITIC_CHARS for ch in tok)):
                    continue
                f = fold_for_match(tok)
                if len(f) < 4:
                    continue
                surf_counter[f][tok] += 1

        for f, counter in surf_counter.items():
            tok, cnt = counter.most_common(1)[0]
            self.fold2surface[f] = tok
            self.fold2freq[f] = cnt

        logger.info(f"Learned surface forms from train: {len(self.fold2surface)} folds")

    def extract_name_targets(self, translit: str, max_targets: int = 50) -> Dict[str, str]:
        translit = "" if translit is None else str(translit)
        targets = {}
        seen = set()

        for tok in translit.split():
            k = norm_key_token(tok)
            for lex, typ in self.token2lexemes.get(k, []):
                if lex in seen:
                    continue
                seen.add(lex)
                if not looks_like_name(lex, typ):
                    continue

                lex_clean = _strip_disambig(lex)
                f = fold_for_match(lex_clean)
                if len(f) < 4:
                    continue

                min_freq = 2 if (typ or "").strip().upper() in EXPLICIT_NE_TYPES else 3
                if f in self.fold2surface and self.fold2freq.get(f, 0) >= min_freq:
                    targets[f] = self.fold2surface[f]

            if len(targets) >= max_targets:
                break

        return targets

    def normalize_names(self, pred: str, targets: Dict[str, str]) -> str:
        if not pred or not targets:
            return pred

        parts = str(pred).split()
        out = []

        for p in parts:
            m = re.match(r"^(\W*)(.*?)(\W*)$", p)
            pre, core, suf = m.group(1), m.group(2), m.group(3)
            if not core:
                out.append(p)
                continue

            poss = ""
            core_base = core
            pm = re.match(r"^(.*?)(['']s)$", core)
            if pm:
                core_base = pm.group(1)
                poss = pm.group(2)

            f = fold_for_match(core_base)
            if len(f) < 4:
                out.append(p)
                continue

            is_cap = core_base[:1].isupper()
            if f in targets and is_cap:
                out.append(pre + targets[f] + poss + suf)
                continue
            # Allow lowercased proper names if long enough and not stopword
            if f in targets and len(f) >= 4 and core_base.lower() not in EN_STOPWORDS:
                out.append(pre + targets[f] + poss + suf)
                continue

            out.append(p)

        return " ".join(out)

    def process(self, translit: str, pred: str) -> str:
        targets = self.extract_name_targets(translit)
        return self.normalize_names(pred, targets)


# ============================================================
# Translation Memory
# ============================================================
class TranslationMemory:
    def __init__(self, train_df: pd.DataFrame = None, train_path: str = None):
        if train_df is None and train_path is not None:
            train_df = pd.read_csv(train_path)
        logger.info(f"Building Translation Memory from DataFrame ({len(train_df)} rows)")

        # Build exact match map
        tmp = defaultdict(Counter)
        for src, tgt in zip(
            train_df["transliteration"].astype(str).tolist(),
            train_df["translation"].astype(str).tolist(),
        ):
            k = re.sub(r"\s+", " ", src.strip())
            if k:
                tmp[k][tgt] += 1

        self.exact_map = {k: c.most_common(1)[0][0] for k, c in tmp.items()}
        logger.info(f"TM entries: {len(self.exact_map)}")

    def lookup(self, src: str) -> Optional[str]:
        k = re.sub(r"\s+", " ", src.strip())
        return self.exact_map.get(k)


# ============================================================
# Repeated Removal
# ============================================================
FUNC_WORDS = {
    'the','a','an','of','to','and','in','on','for','with','at','by','from','as',
    'is','are','was','were','be','been','being','that','this','these','those',
    'it','its','his','her','their','your','my','our','or','not','no'
}


def dedup_consecutive_tokens(tokens, mode='function_only'):
    out = []
    prev_core = None
    prev_punct_score = 0

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
        max_k_eff = min(max_k, n // 2)
        for k in range(max_k_eff, min_k - 1, -1):
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


# ============================================================
# CV Evaluation
# ============================================================
def compute_cv(preds, refs, label=""):
    metric_chrf = evaluate.load("chrf")
    metric_bleu = evaluate.load("sacrebleu")

    chrf = metric_chrf.compute(predictions=preds, references=refs)["score"]
    bleu = metric_bleu.compute(
        predictions=preds, references=[[x] for x in refs]
    )["score"]
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0

    logger.info(f"CV {label}: chrF={chrf:.2f}, BLEU={bleu:.2f}, geo_mean={geo_mean:.2f}")
    return {"chrf": round(chrf, 2), "bleu": round(bleu, 2), "geo_mean": round(geo_mean, 2)}


def extract_first_sentence(text: str) -> str:
    m = re.search(r'^(.*?[.!?])(?:\s|$)', text)
    if m:
        return m.group(1).strip()
    return text.strip()


def truncate_akkadian_to_sentence(translit: str, max_bytes: int = 200) -> str:
    encoded = translit.encode('utf-8')
    if len(encoded) <= max_bytes:
        return translit
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space].strip()
    return truncated.strip()


# ============================================================
# Main
# ============================================================
def main():
    # Load model
    model_path = str(EXP_DIR / config["model"]["checkpoint"])
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    # Prepare validation data (same split as exp005)
    prefix = config["model"].get("prefix", "translate Akkadian to English: ")
    if not prefix:
        prefix = "translate Akkadian to English: "
    train_df = pd.read_csv(str(EXP_DIR / config["data"]["train_path"]))
    train_df = train_df[
        (train_df["transliteration"].astype(str).str.len() > 0)
        & (train_df["translation"].astype(str).str.len() > 0)
    ]

    seed = config["data"]["seed"]
    val_ratio = config["data"].get("val_ratio", 0.1)
    actual_train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=seed)
    actual_train_df = actual_train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    val_texts = [prefix + t for t in val_df["transliteration"].astype(str)]
    val_refs = val_df["translation"].astype(str).tolist()
    val_translits = val_df["transliteration"].astype(str).tolist()

    logger.info(f"Validation samples: {len(val_df)}")

    max_length = config["model"].get("params", {}).get("max_output_length", 512)
    batch_size = config["inference"].get("batch_size", 2)  # MBR needs more VRAM

    # Initialize post-processors
    oa_proc = None
    if config["postprocess"]["oa_lexicon"]["enabled"]:
        oa_proc = OALexiconPostProcessor(
            lexicon_path=str(EXP_DIR / config["postprocess"]["oa_lexicon"]["path"]),
            train_df=actual_train_df,
        )

    tm = None
    if config["postprocess"]["translation_memory"]["enabled"]:
        tm = TranslationMemory(train_df=actual_train_df)

    use_repeated_removal = config["postprocess"]["repeated_removal"]

    # Build dataset
    dataset = InferenceDataset(val_texts, tokenizer, config["model"].get("params", {}).get("max_input_length", 512))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ============================================================
    # 1. Greedy decoding (baseline)
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Greedy Decoding (baseline) ===")
    greedy_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Greedy"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
            )
            greedy_preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    compute_cv(greedy_preds, val_refs, "greedy_raw_doc")

    # Sentence-level
    sent_preds = [extract_first_sentence(p) for p in greedy_preds]
    sent_refs = [extract_first_sentence(r) for r in val_refs]
    compute_cv(sent_preds, sent_refs, "greedy_raw_sent")

    # ============================================================
    # 2. MBR Decoding
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== MBR Decoding ===")
    mbr_decoder = MBRDecoder(model, tokenizer, config)

    mbr_preds = []
    # MBR needs batch_size=1 or small due to num_return_sequences
    mbr_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(mbr_loader, desc="MBR"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            chosen = mbr_decoder.generate_batch(input_ids, attention_mask)
            mbr_preds.extend(chosen)

    compute_cv(mbr_preds, val_refs, "mbr_raw_doc")

    sent_preds_mbr = [extract_first_sentence(p) for p in mbr_preds]
    compute_cv(sent_preds_mbr, sent_refs, "mbr_raw_sent")

    # ============================================================
    # 3. Apply post-processing to MBR preds
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Post-processing MBR predictions ===")

    mbr_post = list(mbr_preds)  # copy

    # 3a. Translation Memory override
    n_tm = 0
    if tm:
        for i, translit in enumerate(val_translits):
            match = tm.lookup(translit)
            if match:
                mbr_post[i] = match
                n_tm += 1
        logger.info(f"TM exact matches: {n_tm}/{len(mbr_post)}")

    # 3b. Repeated removal
    if use_repeated_removal:
        mbr_post = [repeat_cleanup(p) for p in mbr_post]
        logger.info("Applied repeat_cleanup")

    # 3c. OA Lexicon name normalization
    if oa_proc:
        mbr_post = [
            oa_proc.process(translit, pred)
            for translit, pred in zip(val_translits, mbr_post)
        ]
        logger.info("Applied OA Lexicon normalization")

    compute_cv(mbr_post, val_refs, "mbr_post_doc")

    sent_preds_mbr_post = [extract_first_sentence(p) for p in mbr_post]
    compute_cv(sent_preds_mbr_post, sent_refs, "mbr_post_sent")

    # ============================================================
    # 4. Greedy + post-processing (for comparison)
    # ============================================================
    logger.info("=" * 60)
    logger.info("=== Greedy + Post-processing ===")

    greedy_post = list(greedy_preds)
    if tm:
        n_tm_g = 0
        for i, translit in enumerate(val_translits):
            match = tm.lookup(translit)
            if match:
                greedy_post[i] = match
                n_tm_g += 1
        logger.info(f"TM exact matches (greedy): {n_tm_g}")

    if use_repeated_removal:
        greedy_post = [repeat_cleanup(p) for p in greedy_post]

    if oa_proc:
        greedy_post = [
            oa_proc.process(translit, pred)
            for translit, pred in zip(val_translits, greedy_post)
        ]

    compute_cv(greedy_post, val_refs, "greedy_post_doc")

    sent_preds_g_post = [extract_first_sentence(p) for p in greedy_post]
    compute_cv(sent_preds_g_post, sent_refs, "greedy_post_sent")

    # ============================================================
    # Save results
    # ============================================================
    results_df = val_df.copy()
    results_df["greedy_pred"] = greedy_preds
    results_df["mbr_pred"] = mbr_preds
    results_df["mbr_post_pred"] = mbr_post
    results_df["greedy_post_pred"] = greedy_post
    results_df.to_csv(RESULTS_DIR / "val_predictions.csv", index=False)
    logger.info(f"Saved predictions to {RESULTS_DIR / 'val_predictions.csv'}")

    logger.info("=" * 60)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
