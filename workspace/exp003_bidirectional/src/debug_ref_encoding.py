"""training evalのcompute_metricsとの差を検証
training evalではlabelsをdecodeした参照を使う → encode→decode劣化が起きるか確認"""
import os
import sys
import yaml
import evaluate
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

os.chdir(EXP_DIR)
from preprocess import prepare_data

with open(os.path.join(EXP_DIR, "config.yaml")) as f:
    config = yaml.safe_load(f)

model_path = os.path.join(EXP_DIR, "results", "best_model")
tokenizer = AutoTokenizer.from_pretrained(model_path)

_, val_df = prepare_data(config)
val_ref_raw = val_df["target_text"].tolist()

# training evalと同じ: tokenize → decode
encoded = tokenizer(val_ref_raw, max_length=512, truncation=True)
val_ref_roundtrip = tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True)

# 差があるか確認
diffs = 0
for i, (raw, rt) in enumerate(zip(val_ref_raw, val_ref_roundtrip)):
    if raw != rt:
        diffs += 1
        if diffs <= 3:
            print(f"[{i}] DIFF:")
            print(f"  Raw: {raw[:150]}")
            print(f"  R/T: {rt[:150]}")
            print()

print(f"\nTotal diffs: {diffs}/{len(val_ref_raw)}")

# 参照テキストの違いによるスコア差を検証
# 同じ予測に対して、raw refとroundtrip refで比較
import pandas as pd
preds_df = pd.read_csv(os.path.join(EXP_DIR, "results", "val_predictions.csv"))
preds = preds_df["prediction"].tolist()

metric_chrf = evaluate.load("chrf")
metric_bleu = evaluate.load("sacrebleu")

# vs raw ref
chrf1 = metric_chrf.compute(predictions=preds, references=val_ref_raw)["score"]
bleu1 = metric_bleu.compute(predictions=preds, references=[[x] for x in val_ref_raw])["score"]
geo1 = (chrf1 * bleu1) ** 0.5 if chrf1 > 0 and bleu1 > 0 else 0.0
print(f"vs raw ref:       chrF={chrf1:.2f}, BLEU={bleu1:.2f}, geo_mean={geo1:.2f}")

# vs roundtrip ref
chrf2 = metric_chrf.compute(predictions=preds, references=val_ref_roundtrip)["score"]
bleu2 = metric_bleu.compute(predictions=preds, references=[[x] for x in val_ref_roundtrip])["score"]
geo2 = (chrf2 * bleu2) ** 0.5 if chrf2 > 0 and bleu2 > 0 else 0.0
print(f"vs roundtrip ref: chrF={chrf2:.2f}, BLEU={bleu2:.2f}, geo_mean={geo2:.2f}")
