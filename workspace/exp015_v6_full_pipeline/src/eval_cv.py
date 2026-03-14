"""
exp015_v6_full_pipeline: CV評価スクリプト
eda017と同一条件でCV評価する。PN/GNタグ付きで推論。
- beam4, max_length=512, early_stopping=True (submit条件と同一)
- sent-level評価 (入力200B截断 + 最初の文抽出)
"""
import os
import re
import sys
import json
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(RESULTS_DIR, "eval_cv.log")),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# repeat_cleanup
sys.path.insert(0, os.path.join(PROJECT_ROOT, "workspace", "exp007_mbr_postprocess", "src"))
from infer_mbr import repeat_cleanup

# Config
config_path = os.path.join(EXP_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

MAX_LENGTH = config["model"]["max_length"]
SEED = config["training"]["seed"]
PREFIX = config["model"]["prefix"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ============================================================
# Data: train.csvのみでval split (追加データなし、exp011と同一分割)
# ============================================================
train_path = os.path.join(EXP_DIR, config["data"]["train_path"])
df = pd.read_csv(train_path)
df = df[(df["transliteration"].astype(str).str.len() > 0) & (df["translation"].astype(str).str.len() > 0)]

val_ratio = config["training"]["val_ratio"]
train_split, val_split = train_test_split(df, test_size=val_ratio, random_state=SEED)
logger.info(f"Val samples (doc-level): {len(val_split)}")

# PN/GNタグ辞書
dict_path = os.path.join(EXP_DIR, "dataset", "form_type_dict.json")
with open(dict_path) as f:
    form_tag_dict = json.load(f)
logger.info(f"Loaded form_tag_dict: {len(form_tag_dict)} entries")


def tag_transliteration(text, form_tag_dict):
    tokens = text.split()
    tagged = []
    for token in tokens:
        tag = form_tag_dict.get(token)
        if tag:
            tagged.append(f"{token}[{tag}]")
        else:
            tagged.append(token)
    return " ".join(tagged)


# ============================================================
# Sent-level eval
# ============================================================
def extract_first_sentence(text):
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()


def truncate_akkadian_to_sentence(translit, max_bytes=200):
    encoded = str(translit).encode('utf-8')
    if len(encoded) <= max_bytes:
        return str(translit)
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    last_space = truncated.rfind(' ')
    return truncated[:last_space].strip() if last_space > 0 else truncated.strip()


sent_inputs, sent_refs = [], []
for _, row in val_split.iterrows():
    t = str(row["transliteration"])
    tr = str(row["translation"])
    eng = extract_first_sentence(tr)
    akk_tagged = tag_transliteration(truncate_akkadian_to_sentence(t), form_tag_dict)
    if eng.strip() and akk_tagged.strip():
        sent_inputs.append(PREFIX + akk_tagged)
        sent_refs.append(eng)

logger.info(f"Sent-level val samples: {len(sent_inputs)}")

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
# Inference (beam4)
# ============================================================
dataset_eval = InferenceDataset(sent_inputs, tokenizer, MAX_LENGTH)
loader = DataLoader(dataset_eval, batch_size=4, shuffle=False)

preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc="CV eval (beam4)"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model.generate(
            input_ids=ids, attention_mask=mask,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
        )
        preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])

preds_clean = [repeat_cleanup(p) for p in preds]

# ============================================================
# 予測結果保存
# ============================================================
val_preds_df = pd.DataFrame({
    "input": sent_inputs,
    "reference": sent_refs,
    "prediction_raw": preds,
    "prediction_clean": preds_clean,
})
val_preds_path = os.path.join(RESULTS_DIR, "val_predictions.csv")
val_preds_df.to_csv(val_preds_path, index=False)
logger.info(f"Val predictions saved to {val_preds_path} ({len(val_preds_df)} rows)")


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


# ============================================================
# Metrics
# ============================================================
logger.info("=" * 60)
logger.info("=== Sent-level CV evaluation (beam4) ===")

for plabel, pred_list in [("raw", preds), ("clean", preds_clean)]:
    chrf = sacrebleu.corpus_chrf(pred_list, [sent_refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(pred_list, [sent_refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in pred_list) / len(pred_list)
    logger.info(
        f"  {plabel:5s}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, "
        f"geo={geo:.2f}, rep={rep_rate:.1f}%"
    )
