"""exp023 GKF全5foldのsent予測を分析し、エラーパターンを特定する"""
import pandas as pd
import numpy as np
import evaluate
import re
from collections import Counter

metric_chrf = evaluate.load("chrf")
metric_bleu = evaluate.load("sacrebleu")

BASE_DIR = "/home/user/work/Akkadian/eda/eda020_sent_level_cv"
OUT_DIR = "/home/user/work/Akkadian/eda/eda024_error_analysis"

# ============================================================
# 全foldのsent予測を結合
# ============================================================
all_dfs = []
for fold in range(5):
    path = f"{BASE_DIR}/exp023_gkf_fold{fold}_last_sent_predictions.csv"
    df = pd.read_csv(path)
    df["fold"] = fold
    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"Total sent predictions: {len(df_all)} from 5 folds")

# ============================================================
# 個別文のスコア計算
# ============================================================
scores = []
for idx, row in df_all.iterrows():
    pred = str(row["prediction_raw"])
    ref = str(row["reference"])
    chrf = metric_chrf.compute(predictions=[pred], references=[ref])["score"]
    bleu = metric_bleu.compute(predictions=[pred], references=[[ref]])["score"]
    geo = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    scores.append({"chrf": chrf, "bleu": bleu, "geo": geo})

scores_df = pd.DataFrame(scores)
df_all = pd.concat([df_all.reset_index(drop=True), scores_df], axis=1)

print(f"\n=== 全体統計 ===")
print(f"geo mean: {df_all['geo'].mean():.2f}, median: {df_all['geo'].median():.2f}")
print(f"geo=0 の割合: {(df_all['geo'] == 0).mean()*100:.1f}%")

# ============================================================
# fold別スコア分布
# ============================================================
print(f"\n=== fold別 ===")
for fold in range(5):
    sub = df_all[df_all["fold"] == fold]
    print(f"fold{fold}: n={len(sub)}, geo mean={sub['geo'].mean():.2f}, median={sub['geo'].median():.2f}, geo=0: {(sub['geo']==0).sum()}/{len(sub)}")

# ============================================================
# スコア分布
# ============================================================
print(f"\n=== スコア分布 ===")
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for i in range(len(bins)-1):
    count = ((df_all["geo"] >= bins[i]) & (df_all["geo"] < bins[i+1])).sum()
    pct = count / len(df_all) * 100
    print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {count:4d} ({pct:5.1f}%)")

# ============================================================
# worst cases (geo < 10)
# ============================================================
worst = df_all[df_all["geo"] < 10].sort_values("geo")
print(f"\n=== Worst cases (geo < 10): {len(worst)} samples ===")
for _, row in worst.head(20).iterrows():
    pred_short = str(row["prediction_raw"])[:100]
    ref_short = str(row["reference"])[:100]
    print(f"\n  fold{row['fold']} | geo={row['geo']:.1f} | chrf={row['chrf']:.1f} | bleu={row['bleu']:.1f}")
    print(f"  REF: {ref_short}")
    print(f"  PRED: {pred_short}")

# ============================================================
# エラーパターン分類
# ============================================================
print(f"\n=== エラーパターン分析 ===")

# 1. 繰り返し検出
def has_repetition(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r'(\b\w+(?:\s+\w+){0,2}?)(?:\s+\1){2,}', text))

df_all["has_rep"] = df_all["prediction_raw"].apply(has_repetition)
rep_count = df_all["has_rep"].sum()
print(f"1. 繰り返しあり: {rep_count} ({rep_count/len(df_all)*100:.1f}%)")
rep_geo = df_all[df_all["has_rep"]]["geo"].mean()
norep_geo = df_all[~df_all["has_rep"]]["geo"].mean()
print(f"   繰り返しあり geo: {rep_geo:.2f} vs なし: {norep_geo:.2f}")

# 2. 予測が空 or 極端に短い
df_all["pred_len"] = df_all["prediction_raw"].astype(str).str.len()
df_all["ref_len"] = df_all["reference"].astype(str).str.len()
df_all["len_ratio"] = df_all["pred_len"] / df_all["ref_len"].clip(lower=1)

short_pred = df_all[df_all["len_ratio"] < 0.3]
print(f"\n2. 予測が短すぎる (pred/ref < 0.3): {len(short_pred)} ({len(short_pred)/len(df_all)*100:.1f}%)")
if len(short_pred) > 0:
    print(f"   平均geo: {short_pred['geo'].mean():.2f}")

long_pred = df_all[df_all["len_ratio"] > 3.0]
print(f"3. 予測が長すぎる (pred/ref > 3.0): {len(long_pred)} ({len(long_pred)/len(df_all)*100:.1f}%)")
if len(long_pred) > 0:
    print(f"   平均geo: {long_pred['geo'].mean():.2f}")

# 3. 数字の有無による差
def has_numbers(text):
    return bool(re.search(r'\d', str(text)))

df_all["ref_has_num"] = df_all["reference"].apply(has_numbers)
num_geo = df_all[df_all["ref_has_num"]]["geo"].mean()
nonum_geo = df_all[~df_all["ref_has_num"]]["geo"].mean()
print(f"\n4. 数字あり ref: geo={num_geo:.2f} (n={df_all['ref_has_num'].sum()})")
print(f"   数字なし ref: geo={nonum_geo:.2f} (n={(~df_all['ref_has_num']).sum()})")

# 4. 文長別スコア
df_all["ref_words"] = df_all["reference"].astype(str).str.split().str.len()
print(f"\n5. 文長別スコア (ref単語数)")
for lo, hi in [(1, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 999)]:
    sub = df_all[(df_all["ref_words"] >= lo) & (df_all["ref_words"] < hi)]
    if len(sub) > 0:
        print(f"   {lo:3d}-{hi:3d} words: n={len(sub):4d}, geo={sub['geo'].mean():.2f}, median={sub['geo'].median():.2f}")

# 5. 入力長別スコア
df_all["input_len"] = df_all["input"].astype(str).str.len()
print(f"\n6. 入力バイト長別スコア")
for lo, hi in [(0, 100), (100, 200), (200, 300), (300, 500), (500, 1000), (1000, 9999)]:
    sub = df_all[(df_all["input_len"] >= lo) & (df_all["input_len"] < hi)]
    if len(sub) > 0:
        print(f"   {lo:4d}-{hi:4d} bytes: n={len(sub):4d}, geo={sub['geo'].mean():.2f}")

# 6. 特定単語パターン
print(f"\n7. 特定パターン別スコア")
patterns = {
    "<gap>": r"<gap>",
    "shekels/minas (数量)": r"\b(shekel|mina|talent)\b",
    "month (月名)": r"\bmonth\b",
    "witness (証人)": r"\b(witness|seal)\b",
    "says/said": r"\b(says?|said|speak)\b",
}
for name, pat in patterns.items():
    mask = df_all["reference"].astype(str).str.contains(pat, case=False, regex=True)
    sub = df_all[mask]
    if len(sub) > 0:
        print(f"   {name}: n={len(sub)}, geo={sub['geo'].mean():.2f}")

# ============================================================
# best cases (geo > 70)
# ============================================================
best = df_all[df_all["geo"] > 70].sort_values("geo", ascending=False)
print(f"\n=== Best cases (geo > 70): {len(best)} samples ===")
for _, row in best.head(10).iterrows():
    pred_short = str(row["prediction_raw"])[:100]
    ref_short = str(row["reference"])[:100]
    print(f"\n  fold{row['fold']} | geo={row['geo']:.1f}")
    print(f"  REF: {ref_short}")
    print(f"  PRED: {pred_short}")

# ============================================================
# 保存
# ============================================================
df_all.to_csv(f"{OUT_DIR}/all_folds_sent_with_scores.csv", index=False)
print(f"\nSaved to {OUT_DIR}/all_folds_sent_with_scores.csv")

# サマリー統計
summary = {
    "total_samples": len(df_all),
    "geo_mean": round(df_all["geo"].mean(), 2),
    "geo_median": round(df_all["geo"].median(), 2),
    "geo_zero_pct": round((df_all["geo"] == 0).mean() * 100, 1),
    "repetition_pct": round(df_all["has_rep"].mean() * 100, 1),
    "short_pred_pct": round(len(short_pred) / len(df_all) * 100, 1),
    "long_pred_pct": round(len(long_pred) / len(df_all) * 100, 1),
}
import json
with open(f"{OUT_DIR}/summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)
