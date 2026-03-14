"""training eval vs eval_cv のギャップ原因調査"""
import os, sys, json, math, re
import pandas as pd
import numpy as np
import yaml
import sacrebleu
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))

config_path = os.path.join(EXP_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

SEED = config["training"]["seed"]
val_ratio = config["training"]["val_ratio"]

# 1. Val splitの確認
train_path = os.path.join(EXP_DIR, config["data"]["train_path"])
df = pd.read_csv(train_path)
df = df[(df["transliteration"].astype(str).str.len() > 0) & (df["translation"].astype(str).str.len() > 0)]
_, val_split = train_test_split(df, test_size=val_ratio, random_state=SEED)
print(f"Val samples: {len(val_split)}")

# 2. 参照テキストの長さ分布
ref_lengths = val_split["translation"].astype(str).apply(lambda t: len(t.encode('utf-8')))
print(f"\n=== Reference length (bytes) ===")
print(f"  mean: {ref_lengths.mean():.0f}")
print(f"  median: {ref_lengths.median():.0f}")
print(f"  max: {ref_lengths.max()}")
print(f"  >512B: {(ref_lengths > 512).sum()} / {len(ref_lengths)} ({100*(ref_lengths>512).sum()/len(ref_lengths):.1f}%)")

# 3. Training evalの512B truncation影響
refs_full = val_split["translation"].astype(str).tolist()
refs_truncated = [t.encode('utf-8')[:512].decode('utf-8', errors='ignore') for t in refs_full]
print(f"\n=== 512B truncation effect ===")
n_truncated = sum(1 for f, t in zip(refs_full, refs_truncated) if f != t)
print(f"  Truncated: {n_truncated} / {len(refs_full)} ({100*n_truncated/len(refs_full):.1f}%)")

# 4. eval_cvのfirst sentence抽出
def extract_first_sentence(text):
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()

refs_first_sent = [extract_first_sentence(t) for t in refs_full]
first_sent_lengths = [len(r.encode('utf-8')) for r in refs_first_sent]
print(f"\n=== First sentence reference length (bytes) ===")
print(f"  mean: {np.mean(first_sent_lengths):.0f}")
print(f"  median: {np.median(first_sent_lengths):.0f}")

# 5. 参照の違いでスコアがどう変わるか、val_predictions.csvがあれば比較
pred_path = os.path.join(EXP_DIR, "results", "val_predictions.csv")
if os.path.exists(pred_path):
    preds_df = pd.read_csv(pred_path)
    preds_clean = preds_df["prediction_clean"].tolist()
    refs_first = preds_df["reference"].tolist()  # eval_cvのfirst sentence ref

    # eval_cv条件: preds_clean vs first_sentence refs (beam4)
    chrf_sent = sacrebleu.corpus_chrf(preds_clean, [refs_first], word_order=2)
    bleu_sent = sacrebleu.corpus_bleu(preds_clean, [refs_first])
    geo_sent = math.sqrt(chrf_sent.score * bleu_sent.score) if chrf_sent.score > 0 and bleu_sent.score > 0 else 0
    print(f"\n=== eval_cv (beam4, sent-level) ===")
    print(f"  chrF++={chrf_sent.score:.2f}, BLEU={bleu_sent.score:.2f}, geo={geo_sent:.2f}")

    # 同じbeam4 predsをfull reference (truncated 512B)で評価
    chrf_trunc = sacrebleu.corpus_chrf(preds_clean, [refs_truncated], word_order=2)
    bleu_trunc = sacrebleu.corpus_bleu(preds_clean, [refs_truncated])
    geo_trunc = math.sqrt(chrf_trunc.score * bleu_trunc.score) if chrf_trunc.score > 0 and bleu_trunc.score > 0 else 0
    print(f"\n=== beam4 preds vs 512B-truncated refs (training eval相当の参照) ===")
    print(f"  chrF++={chrf_trunc.score:.2f}, BLEU={bleu_trunc.score:.2f}, geo={geo_trunc:.2f}")

    # 同じbeam4 predsをfull reference (no truncation)で評価
    chrf_full = sacrebleu.corpus_chrf(preds_clean, [refs_full], word_order=2)
    bleu_full = sacrebleu.corpus_bleu(preds_clean, [refs_full])
    geo_full = math.sqrt(chrf_full.score * bleu_full.score) if chrf_full.score > 0 and bleu_full.score > 0 else 0
    print(f"\n=== beam4 preds vs full refs (truncationなし) ===")
    print(f"  chrF++={chrf_full.score:.2f}, BLEU={bleu_full.score:.2f}, geo={geo_full:.2f}")

    # prediction長
    pred_lengths = [len(p.encode('utf-8')) for p in preds_clean]
    print(f"\n=== Prediction length (bytes) ===")
    print(f"  mean: {np.mean(pred_lengths):.0f}")
    print(f"  median: {np.median(pred_lengths):.0f}")
else:
    print("\nval_predictions.csv not found")
