"""同じ予測を異なる参照で評価し、参照の影響を分離する"""
import os, sys, re, json, math
import pandas as pd, numpy as np, yaml, sacrebleu
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(EXP_DIR, "results")

with open(os.path.join(EXP_DIR, "config.yaml")) as f:
    config = yaml.safe_load(f)
SEED = config["training"]["seed"]

# Val split (eval_cvと同一)
df = pd.read_csv(os.path.join(EXP_DIR, config["data"]["train_path"]))
df = df[(df["transliteration"].astype(str).str.len() > 0) & (df["translation"].astype(str).str.len() > 0)]
_, val_split = train_test_split(df, test_size=config["training"]["val_ratio"], random_state=SEED)

def extract_first_sentence(text):
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()

# 3種類の参照を作成
refs_first_sent = [extract_first_sentence(str(row["translation"])) for _, row in val_split.iterrows()]
refs_512B = [str(row["translation"]).encode('utf-8')[:512].decode('utf-8', errors='ignore') for _, row in val_split.iterrows()]
refs_full = [str(row["translation"]) for _, row in val_split.iterrows()]

print(f"Ref lengths (bytes):")
print(f"  first-sent: mean={np.mean([len(r.encode('utf-8')) for r in refs_first_sent]):.0f}")
print(f"  512B trunc: mean={np.mean([len(r.encode('utf-8')) for r in refs_512B]):.0f}")
print(f"  full:       mean={np.mean([len(r.encode('utf-8')) for r in refs_full]):.0f}")

def has_repetition(text, mr=3):
    w = str(text).split()
    for i in range(len(w)-mr):
        if " ".join(w[i:i+mr]) in " ".join(w[i+mr:]): return True
    return False

def show(preds, refs, label):
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0
    rep = 100 * sum(has_repetition(p) for p in preds) / len(preds)
    pred_len = np.mean([len(p.encode('utf-8')) for p in preds])
    ref_len = np.mean([len(r.encode('utf-8')) for r in refs])
    print(f"  {label}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep:.1f}%, pred={pred_len:.0f}B, ref={ref_len:.0f}B")

# === 実験1: eval_cvの予測(beam4, 200B入力)を異なる参照で評価 ===
print("\n" + "="*60)
print("実験1: eval_cv predictions (beam4, 200B input) × 3 references")
preds_df = pd.read_csv(os.path.join(RESULTS_DIR, "val_predictions.csv"))
preds_clean = preds_df["prediction_clean"].tolist()
print(f"  predictions: {len(preds_clean)} samples, mean={np.mean([len(p.encode('utf-8')) for p in preds_clean]):.0f}B")

show(preds_clean, refs_first_sent, "vs first-sent ref")
show(preds_clean, refs_512B, "vs 512B ref    ")
show(preds_clean, refs_full, "vs full ref    ")
