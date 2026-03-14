"""exp014: training eval vs eval_cv のギャップ原因調査"""
import os, sys, re, math
import pandas as pd
import numpy as np
import torch
import sacrebleu
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "last_model")

MAX_LENGTH = 512
PREFIX = "translate Akkadian to English: "
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Val split (exp014 training と同一: starter方式)
train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "datasets", "raw", "train.csv"))

def simple_sentence_aligner(df):
    aligned = []
    for _, row in df.iterrows():
        src, tgt = str(row["transliteration"]), str(row["translation"])
        tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split("\n") if s.strip()]
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned.append({"transliteration": s, "translation": t})
        else:
            aligned.append({"transliteration": src, "translation": tgt})
    return pd.DataFrame(aligned)

train_expanded = simple_sentence_aligner(train_df)
dataset = Dataset.from_pandas(train_expanded)
split = dataset.train_test_split(test_size=0.1, seed=42)
val_data = split["test"].to_pandas()
print(f"Val samples: {len(val_data)}")

# References
refs_full = val_data["translation"].astype(str).tolist()
refs_512B = [t.encode('utf-8')[:512].decode('utf-8', errors='ignore') for t in refs_full]

ref_lengths = [len(r.encode('utf-8')) for r in refs_full]
print(f"\nRef length: mean={np.mean(ref_lengths):.0f}B, >512B: {sum(1 for l in ref_lengths if l > 512)}/{len(ref_lengths)}")

def extract_first_sentence(text):
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()

def truncate_200B(text):
    enc = str(text).encode('utf-8')
    if len(enc) <= 200: return str(text)
    trunc = enc[:200].decode('utf-8', errors='ignore')
    last = trunc.rfind(' ')
    return trunc[:last].strip() if last > 0 else trunc.strip()

refs_first = [extract_first_sentence(t) for t in refs_full]
full_inputs = [PREFIX + str(row["transliteration"]) for _, row in val_data.iterrows()]
sent_inputs = [PREFIX + truncate_200B(str(row["transliteration"])) for _, row in val_data.iterrows()]

# Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

def run_inference(inputs, num_beams=1, label=""):
    preds = []
    for i in tqdm(range(0, len(inputs), 4), desc=label):
        batch = inputs[i:i+4]
        enc = tokenizer(batch, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                                 max_length=MAX_LENGTH, num_beams=num_beams, early_stopping=True if num_beams > 1 else False)
        preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    return preds

def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i+min_repeat])
        if chunk in " ".join(words[i+min_repeat:]): return True
    return False

def show(preds, refs, label):
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0
    pred_lens = [len(p.encode('utf-8')) for p in preds]
    rep = 100 * sum(has_repetition(p) for p in preds) / len(preds)
    print(f"  {label}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, pred={np.mean(pred_lens):.0f}B, rep={rep:.1f}%")

# Condition 1: training eval (greedy, full input, 512B ref)
print("\n=== Condition 1: greedy, full input ===")
p1 = run_inference(full_inputs, num_beams=1, label="greedy full")
show(p1, refs_512B, "vs 512B ref")
show(p1, refs_first, "vs first-sent ref")

# Condition 2: eval_cv (beam4, 200B input, first-sent ref)
print("\n=== Condition 2: beam4, 200B input ===")
p2 = run_inference(sent_inputs, num_beams=4, label="beam4 sent")
show(p2, refs_first, "vs first-sent ref")
show(p2, refs_512B, "vs 512B ref")

# Condition 3: greedy, 200B input
print("\n=== Condition 3: greedy, 200B input ===")
p3 = run_inference(sent_inputs, num_beams=1, label="greedy sent")
show(p3, refs_first, "vs first-sent ref")
