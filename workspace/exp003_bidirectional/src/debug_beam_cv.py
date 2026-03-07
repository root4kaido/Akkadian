"""beam search CV問題のデバッグ: raw vs postprocess比較"""
import os
import sys
import torch
import pandas as pd
import yaml
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)
from postprocess import postprocess_batch

# config
with open(os.path.join(EXP_DIR, "config.yaml")) as f:
    config = yaml.safe_load(f)

model_path = os.path.join(EXP_DIR, "results", "best_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
model.eval()

# val data
os.chdir(EXP_DIR)
from preprocess import prepare_data
_, val_df = prepare_data(config)

prefix = config["model"]["prefix"]
val_src = [t.replace(prefix, "", 1) for t in val_df["input_text"].tolist()]
val_ref = val_df["target_text"].tolist()


class InfDS(Dataset):
    def __init__(self, texts):
        self.texts = [prefix + str(t) for t in texts]
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        inputs = tokenizer(self.texts[idx], max_length=512, padding="max_length",
                           truncation=True, return_tensors="pt")
        return {"input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0)}


loader = DataLoader(InfDS(val_src), batch_size=16, shuffle=False)
raw_preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Raw inference"):
        outputs = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_length=512, num_beams=4, early_stopping=True,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        raw_preds.extend([d.strip() for d in decoded])

metric_chrf = evaluate.load("chrf")
metric_bleu = evaluate.load("sacrebleu")

# 1. NO postprocess (Starter方式)
chrf = metric_chrf.compute(predictions=raw_preds, references=val_ref)["score"]
bleu = metric_bleu.compute(predictions=raw_preds, references=[[x] for x in val_ref])["score"]
geo = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
print(f"\nNO postprocess (Starter方式): chrF={chrf:.2f}, BLEU={bleu:.2f}, geo_mean={geo:.2f}")

# 2. WITH postprocess
pp_preds = postprocess_batch(raw_preds)
chrf2 = metric_chrf.compute(predictions=pp_preds, references=val_ref)["score"]
bleu2 = metric_bleu.compute(predictions=pp_preds, references=[[x] for x in val_ref])["score"]
geo2 = (chrf2 * bleu2) ** 0.5 if chrf2 > 0 and bleu2 > 0 else 0.0
print(f"WITH postprocess: chrF={chrf2:.2f}, BLEU={bleu2:.2f}, geo_mean={geo2:.2f}")

# 3. サンプル比較
print(f"\n--- Raw vs Postprocessed (problematic samples) ---")
for i in range(min(5, len(raw_preds))):
    if raw_preds[i] != pp_preds[i] or len(raw_preds[i]) > 300:
        print(f"[{i}] Raw  ({len(raw_preds[i]):4d} chars): {raw_preds[i][:200]}")
        print(f"[{i}] Post ({len(pp_preds[i]):4d} chars): {pp_preds[i][:200]}")
        print(f"[{i}] Ref  ({len(val_ref[i]):4d} chars): {val_ref[i][:200]}")
        print()
