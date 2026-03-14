"""
training eval条件を手動再現して、eval_cvとの差の原因を特定する。
"""
import os, sys, json, math, re
import pandas as pd
import numpy as np
import yaml
import torch
import sacrebleu
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model")

config_path = os.path.join(EXP_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

SEED = config["training"]["seed"]
MAX_LENGTH = config["model"]["max_length"]
PREFIX = config["model"]["prefix"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Val split (preprocess.pyと同一)
train_path = os.path.join(EXP_DIR, config["data"]["train_path"])
df = pd.read_csv(train_path)
df = df[(df["transliteration"].astype(str).str.len() > 0) & (df["translation"].astype(str).str.len() > 0)]
_, val_split = train_test_split(df, test_size=config["training"]["val_ratio"], random_state=SEED)

# PN/GNタグ
dict_path = os.path.join(EXP_DIR, "dataset", "form_type_dict.json")
with open(dict_path) as f:
    form_tag_dict = json.load(f)

def tag_transliteration(text):
    tokens = text.split()
    return " ".join(f"{t}[{form_tag_dict[t]}]" if t in form_tag_dict else t for t in tokens)

# Training evalと同一条件の入力 (full doc, tagged, prefix付き)
full_inputs = [PREFIX + tag_transliteration(str(row["transliteration"])) for _, row in val_split.iterrows()]
refs_full = [str(row["translation"]) for _, row in val_split.iterrows()]
refs_512B = [t.encode('utf-8')[:512].decode('utf-8', errors='ignore') for t in refs_full]

# eval_cv条件の入力 (200B truncated)
def truncate_200B(text):
    enc = str(text).encode('utf-8')
    if len(enc) <= 200:
        return str(text)
    trunc = enc[:200].decode('utf-8', errors='ignore')
    last = trunc.rfind(' ')
    return trunc[:last].strip() if last > 0 else trunc.strip()

def extract_first_sentence(text):
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()

sent_inputs = [PREFIX + tag_transliteration(truncate_200B(str(row["transliteration"]))) for _, row in val_split.iterrows()]
refs_first = [extract_first_sentence(str(row["translation"])) for _, row in val_split.iterrows()]

# Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
print(f"Model loaded from {MODEL_PATH}")

def run_inference(inputs, max_length, num_beams=1, label=""):
    preds = []
    batch_size = 4
    for i in tqdm(range(0, len(inputs), batch_size), desc=label):
        batch = inputs[i:i+batch_size]
        enc = tokenizer(batch, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True if num_beams > 1 else False,
            )
        preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    return preds

def eval_metrics(preds, refs, label):
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0
    pred_lens = [len(p.encode('utf-8')) for p in preds]
    ref_lens = [len(r.encode('utf-8')) for r in refs]
    print(f"\n=== {label} ===")
    print(f"  chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}")
    print(f"  pred len: mean={np.mean(pred_lens):.0f}B, median={np.median(pred_lens):.0f}B")
    print(f"  ref len: mean={np.mean(ref_lens):.0f}B, median={np.median(ref_lens):.0f}B")

# === 条件1: training eval再現 (greedy, full input, 512B ref) ===
print("\n" + "="*60)
print("Condition 1: training eval (greedy, full input, 512B-truncated ref)")
preds_greedy_full = run_inference(full_inputs, MAX_LENGTH, num_beams=1, label="greedy full")
eval_metrics(preds_greedy_full, refs_512B, "greedy full vs 512B ref")
eval_metrics(preds_greedy_full, refs_first, "greedy full vs first-sentence ref")

# === 条件2: eval_cv再現 (beam4, 200B input, first-sentence ref) ===
print("\n" + "="*60)
print("Condition 2: eval_cv (beam4, 200B input, first-sentence ref)")
preds_beam4_sent = run_inference(sent_inputs, MAX_LENGTH, num_beams=4, label="beam4 sent")
eval_metrics(preds_beam4_sent, refs_first, "beam4 sent vs first-sentence ref")
eval_metrics(preds_beam4_sent, refs_512B, "beam4 sent vs 512B ref")

# === 条件3: greedy, 200B input (decodingの影響を分離) ===
print("\n" + "="*60)
print("Condition 3: greedy, 200B input")
preds_greedy_sent = run_inference(sent_inputs, MAX_LENGTH, num_beams=1, label="greedy sent")
eval_metrics(preds_greedy_sent, refs_first, "greedy sent vs first-sentence ref")

# === 条件4: beam4, full input (入力長の影響を分離) ===
print("\n" + "="*60)
print("Condition 4: beam4, full input")
preds_beam4_full = run_inference(full_inputs, MAX_LENGTH, num_beams=4, label="beam4 full")
eval_metrics(preds_beam4_full, refs_first, "beam4 full vs first-sentence ref")
eval_metrics(preds_beam4_full, refs_512B, "beam4 full vs 512B ref")
