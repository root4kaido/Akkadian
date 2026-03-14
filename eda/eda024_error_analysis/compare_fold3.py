"""fold3の4実験(exp019, exp023, exp026, exp027)を比較"""
import pandas as pd
import numpy as np
import evaluate
import re

metric_chrf = evaluate.load("chrf")
metric_bleu = evaluate.load("sacrebleu")

BASE_DIR = "/home/user/work/Akkadian/eda/eda020_sent_level_cv"

experiments = {
    "exp019": "exp019_gkf_fold3_last",
    "exp023": "exp023_gkf_fold3_last",
    "exp026": "exp026_gkf_fold3_last",
    "exp027": "exp027_gkf_fold3_last",
}

def calc_per_sample(df):
    scores = []
    for _, row in df.iterrows():
        pred = str(row["prediction_raw"])
        ref = str(row["reference"])
        chrf = metric_chrf.compute(predictions=[pred], references=[ref])["score"]
        bleu = metric_bleu.compute(predictions=[pred], references=[[ref]])["score"]
        geo = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
        scores.append({"chrf": chrf, "bleu": bleu, "geo": geo})
    return pd.DataFrame(scores)

def has_repetition(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r'(\b\w+(?:\s+\w+){0,2}?)(?:\s+\1){2,}', text))

def has_akkadian_output(text):
    """翻字がそのまま出力されているか"""
    if not isinstance(text, str):
        return False
    akk_patterns = [r'DUMU', r'KÙ\.BABBAR', r'IGI\b', r'GÍN', r'KIŠIB', r'-im\b', r'-šu\b']
    matches = sum(1 for p in akk_patterns if re.search(p, text))
    return matches >= 2

# ============================================================
# sent-CV比較
# ============================================================
print("=" * 100)
print("=== Fold3 sent-CV比較 ===")
print("=" * 100)

sent_data = {}
for exp_name, file_prefix in experiments.items():
    path = f"{BASE_DIR}/{file_prefix}_sent_predictions.csv"
    df = pd.read_csv(path)
    scores = calc_per_sample(df)
    df = pd.concat([df.reset_index(drop=True), scores], axis=1)
    df["pred_len"] = df["prediction_raw"].astype(str).str.len()
    df["ref_len"] = df["reference"].astype(str).str.len()
    df["len_ratio"] = df["pred_len"] / df["ref_len"].clip(lower=1)
    df["has_rep"] = df["prediction_raw"].apply(has_repetition)
    df["akk_output"] = df["prediction_raw"].apply(has_akkadian_output)
    df["ref_words"] = df["reference"].astype(str).str.split().str.len()
    df["input_bytes"] = df["input"].astype(str).str.len()
    sent_data[exp_name] = df

# 全体スコア
print(f"\n{'metric':<25} {'exp019':>10} {'exp023':>10} {'exp026':>10} {'exp027':>10}")
print("-" * 70)

for metric_name in ["geo", "chrf", "bleu"]:
    vals = [sent_data[e][metric_name].mean() for e in experiments]
    print(f"{metric_name + ' mean':<25} {vals[0]:10.2f} {vals[1]:10.2f} {vals[2]:10.2f} {vals[3]:10.2f}")

for metric_name in ["geo"]:
    vals = [sent_data[e][metric_name].median() for e in experiments]
    print(f"{metric_name + ' median':<25} {vals[0]:10.2f} {vals[1]:10.2f} {vals[2]:10.2f} {vals[3]:10.2f}")

# geo=0の数
vals = [(sent_data[e]["geo"] == 0).sum() for e in experiments]
print(f"{'geo=0 count':<25} {vals[0]:10d} {vals[1]:10d} {vals[2]:10d} {vals[3]:10d}")

# geo<10の数
vals = [(sent_data[e]["geo"] < 10).sum() for e in experiments]
print(f"{'geo<10 count':<25} {vals[0]:10d} {vals[1]:10d} {vals[2]:10d} {vals[3]:10d}")

# 繰り返し率
vals = [sent_data[e]["has_rep"].mean()*100 for e in experiments]
print(f"{'repetition %':<25} {vals[0]:10.1f} {vals[1]:10.1f} {vals[2]:10.1f} {vals[3]:10.1f}")

# アッカド語出力率
vals = [sent_data[e]["akk_output"].mean()*100 for e in experiments]
print(f"{'akkadian output %':<25} {vals[0]:10.1f} {vals[1]:10.1f} {vals[2]:10.1f} {vals[3]:10.1f}")

# 短すぎる予測
vals = [(sent_data[e]["len_ratio"] < 0.3).mean()*100 for e in experiments]
print(f"{'too short (<0.3) %':<25} {vals[0]:10.1f} {vals[1]:10.1f} {vals[2]:10.1f} {vals[3]:10.1f}")

# 長すぎる予測
vals = [(sent_data[e]["len_ratio"] > 3.0).mean()*100 for e in experiments]
print(f"{'too long (>3.0) %':<25} {vals[0]:10.1f} {vals[1]:10.1f} {vals[2]:10.1f} {vals[3]:10.1f}")

# ============================================================
# スコア帯別の比較
# ============================================================
print(f"\n=== スコア帯別分布 (geo) ===")
print(f"{'range':<10} {'exp019':>10} {'exp023':>10} {'exp026':>10} {'exp027':>10}")
print("-" * 55)
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for i in range(len(bins)-1):
    vals = []
    for e in experiments:
        count = ((sent_data[e]["geo"] >= bins[i]) & (sent_data[e]["geo"] < bins[i+1])).sum()
        pct = count / len(sent_data[e]) * 100
        vals.append(pct)
    print(f"{bins[i]:3d}-{bins[i+1]:3d}     {vals[0]:9.1f}% {vals[1]:9.1f}% {vals[2]:9.1f}% {vals[3]:9.1f}%")

# ============================================================
# 入力バイト長別
# ============================================================
print(f"\n=== 入力バイト長別 geo mean ===")
print(f"{'range':<15} {'exp019':>10} {'exp023':>10} {'exp026':>10} {'exp027':>10}")
print("-" * 60)
for lo, hi in [(0, 100), (100, 200), (200, 300), (300, 500)]:
    vals = []
    for e in experiments:
        sub = sent_data[e][(sent_data[e]["input_bytes"] >= lo) & (sent_data[e]["input_bytes"] < hi)]
        vals.append(sub["geo"].mean() if len(sub) > 0 else 0)
    print(f"{lo:4d}-{hi:4d}       {vals[0]:10.2f} {vals[1]:10.2f} {vals[2]:10.2f} {vals[3]:10.2f}")

# ============================================================
# ref単語数別
# ============================================================
print(f"\n=== ref単語数別 geo mean ===")
print(f"{'range':<15} {'exp019':>10} {'exp023':>10} {'exp026':>10} {'exp027':>10}")
print("-" * 60)
for lo, hi in [(1, 10), (10, 20), (20, 30), (30, 50), (50, 999)]:
    vals = []
    for e in experiments:
        sub = sent_data[e][(sent_data[e]["ref_words"] >= lo) & (sent_data[e]["ref_words"] < hi)]
        vals.append(sub["geo"].mean() if len(sub) > 0 else 0)
    n = len(sent_data["exp023"][(sent_data["exp023"]["ref_words"] >= lo) & (sent_data["exp023"]["ref_words"] < hi)])
    print(f"{lo:3d}-{hi:3d} (n={n:3d})  {vals[0]:10.2f} {vals[1]:10.2f} {vals[2]:10.2f} {vals[3]:10.2f}")

# ============================================================
# パターン別
# ============================================================
print(f"\n=== パターン別 geo mean ===")
print(f"{'pattern':<25} {'exp019':>10} {'exp023':>10} {'exp026':>10} {'exp027':>10}")
print("-" * 70)
patterns = {
    "<gap>": r"<gap>",
    "shekels/minas": r"\bshekel|mina|talent\b",
    "month": r"\bmonth\b",
    "witness/seal": r"\bwitness|seal\b",
    "says/said": r"\bsays?|said|speak\b",
    "owe/debt": r"\bowe|debt|credit\b",
    "silver/gold": r"\bsilver|gold\b",
}
for name, pat in patterns.items():
    vals = []
    for e in experiments:
        mask = sent_data[e]["reference"].astype(str).str.contains(pat, case=False, regex=True)
        sub = sent_data[e][mask]
        vals.append(sub["geo"].mean() if len(sub) > 0 else 0)
    n = sent_data["exp023"]["reference"].astype(str).str.contains(pat, case=False, regex=True).sum()
    print(f"{name + f' (n={n})':<25} {vals[0]:10.2f} {vals[1]:10.2f} {vals[2]:10.2f} {vals[3]:10.2f}")

# ============================================================
# doc-CV比較
# ============================================================
print(f"\n{'='*100}")
print("=== Fold3 doc-CV比較 ===")
print(f"{'='*100}")

for exp_name, file_prefix in experiments.items():
    path = f"{BASE_DIR}/{file_prefix}_doc_predictions.csv"
    df = pd.read_csv(path)
    scores = calc_per_sample(df)
    df = pd.concat([df.reset_index(drop=True), scores], axis=1)
    df["has_rep"] = df["prediction_raw"].apply(has_repetition)

    chrf_all = metric_chrf.compute(predictions=df["prediction_raw"].tolist(), references=df["reference"].tolist())["score"]
    bleu_all = metric_bleu.compute(predictions=df["prediction_raw"].tolist(), references=[[r] for r in df["reference"].tolist()])["score"]
    geo_all = (chrf_all * bleu_all) ** 0.5 if chrf_all > 0 and bleu_all > 0 else 0.0

    rep_pct = df["has_rep"].mean() * 100
    geo0 = (df["geo"] == 0).sum()
    print(f"{exp_name}: corpus chrF++={chrf_all:.2f}, BLEU={bleu_all:.2f}, geo={geo_all:.2f} | rep={rep_pct:.1f}% | geo=0: {geo0}/{len(df)}")

# ============================================================
# サンプル単位でexp023 vs 他を比較（改善/悪化ケース）
# ============================================================
print(f"\n{'='*100}")
print("=== exp023 vs exp027: サンプル単位の変化 ===")
print(f"{'='*100}")

df23 = sent_data["exp023"]
df27 = sent_data["exp027"]

# 同じ入力を持つサンプルを突合
merged = df23.merge(df27, on="input", suffixes=("_23", "_27"))
merged["geo_diff"] = merged["geo_27"] - merged["geo_23"]

improved = merged[merged["geo_diff"] > 5].sort_values("geo_diff", ascending=False)
degraded = merged[merged["geo_diff"] < -5].sort_values("geo_diff")

print(f"改善(>5pt): {len(improved)} samples, 悪化(<-5pt): {len(degraded)} samples, 中立: {len(merged) - len(improved) - len(degraded)} samples")

print(f"\n--- 最も悪化したケース (exp023→exp027) ---")
for _, row in degraded.head(10).iterrows():
    print(f"\n  geo: {row['geo_23']:.1f} → {row['geo_27']:.1f} (diff={row['geo_diff']:.1f})")
    print(f"  REF:  {str(row['reference_23'])[:120]}")
    print(f"  PRED23: {str(row['prediction_raw_23'])[:120]}")
    print(f"  PRED27: {str(row['prediction_raw_27'])[:120]}")

print(f"\n--- 最も改善したケース (exp023→exp027) ---")
for _, row in improved.head(10).iterrows():
    print(f"\n  geo: {row['geo_23']:.1f} → {row['geo_27']:.1f} (diff={row['geo_diff']:.1f})")
    print(f"  REF:  {str(row['reference_23'])[:120]}")
    print(f"  PRED23: {str(row['prediction_raw_23'])[:120]}")
    print(f"  PRED27: {str(row['prediction_raw_27'])[:120]}")
