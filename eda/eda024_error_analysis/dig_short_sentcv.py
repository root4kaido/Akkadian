"""
sent-CVの短すぎ予測(ratio<0.5)と低スコア(<100B入力)を深掘り
"""
import re
import pandas as pd
import sacrebleu
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
df = pd.read_csv(RESULTS_DIR / "sentcv_predictions_scored.csv")

# ============================================================
# 1. 学習データの入力長分布（doc-level）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
train_df = pd.read_csv(PROJECT_ROOT / "datasets" / "raw" / "train.csv")

train_bytes = train_df["transliteration"].astype(str).apply(lambda x: len(x.encode("utf-8")))
print("=== 学習データ（doc-level）入力長 ===")
print(f"  mean={train_bytes.mean():.0f}B, median={train_bytes.median():.0f}B")
print(f"  min={train_bytes.min()}B, max={train_bytes.max()}B")
print(f"  <100B: {(train_bytes < 100).sum()}/{len(train_bytes)} ({100*(train_bytes < 100).mean():.1f}%)")
print(f"  <200B: {(train_bytes < 200).sum()}/{len(train_bytes)} ({100*(train_bytes < 200).mean():.1f}%)")

# ============================================================
# 2. sent-CV入力長分布
# ============================================================
print("\n=== sent-CV入力長 ===")
print(f"  mean={df['inp_bytes'].mean():.0f}B, median={df['inp_bytes'].median():.0f}B")
print(f"  min={df['inp_bytes'].min()}B, max={df['inp_bytes'].max()}B")
print(f"  <100B: {(df['inp_bytes'] < 100).sum()}/{len(df)} ({100*(df['inp_bytes'] < 100).mean():.1f}%)")

# ============================================================
# 3. too_short予測の詳細パターン分類
# ============================================================
print("\n" + "=" * 70)
print("too_short (ratio<0.5) 60件の詳細パターン")
print("=" * 70)

short = df[df["ratio"] < 0.5].copy()

def classify_error(row):
    pred = str(row["pred"])
    ref = str(row["ref"])
    # アッカド語がそのまま出力されている
    if re.search(r'[a-z]-[a-z]', pred) and not re.search(r'[a-z]-[a-z]', ref):
        return "akkadian_output"
    # <gap>だらけ
    if pred.count("<gap>") >= 2 and len(pred.split()) <= 5:
        return "gap_only"
    # 予測が極端に短い（5語以下）
    if len(pred.split()) <= 3:
        return "too_few_words"
    # 内容が全く違う
    return "wrong_content"

short["error_type"] = short.apply(classify_error, axis=1)

for etype, group in short.groupby("error_type"):
    print(f"\n  {etype}: {len(group)}件, chrF++={group['chrf'].mean():.2f}")
    print(f"    avg inp_bytes={group['inp_bytes'].mean():.0f}B, avg ref_words={group['ref_words'].mean():.1f}")
    for _, row in group.head(3).iterrows():
        print(f"    - Ref:  {str(row['ref'])[:100]}")
        print(f"      Pred: {str(row['pred'])[:100]}")
        print(f"      inp={row['inp_bytes']}B, chrF++={row['chrf']:.1f}")

# ============================================================
# 4. 入力が短い(<100B)のに正しく翻訳できたケースとの比較
# ============================================================
print("\n" + "=" * 70)
print("<100B入力: 成功 vs 失敗")
print("=" * 70)

short_input = df[df["inp_bytes"] < 100].copy()
good = short_input[short_input["chrf"] >= 50]
bad = short_input[short_input["chrf"] < 30]

print(f"\n  <100B全体: {len(short_input)}件")
print(f"  chrF++>=50 (成功): {len(good)}件")
print(f"  chrF++<30  (失敗): {len(bad)}件")

print("\n  --- 成功例 ---")
for _, row in good.head(5).iterrows():
    print(f"    chrF++={row['chrf']:.1f} | inp={row['inp_bytes']}B")
    print(f"      Ref:  {str(row['ref'])[:100]}")
    print(f"      Pred: {str(row['pred'])[:100]}")

print("\n  --- 失敗例 ---")
for _, row in bad.head(5).iterrows():
    print(f"    chrF++={row['chrf']:.1f} | inp={row['inp_bytes']}B")
    print(f"      Ref:  {str(row['ref'])[:100]}")
    print(f"      Pred: {str(row['pred'])[:100]}")

# ============================================================
# 5. 参照のword数 vs 入力bytes の散布（短入力・長参照 = 難ケース）
# ============================================================
print("\n" + "=" * 70)
print("短入力 + 長参照（情報不足ケース）")
print("=" * 70)

hard = df[(df["inp_bytes"] < 100) & (df["ref_words"] > 15)]
print(f"  <100B入力 & 参照>15語: {len(hard)}件, chrF++={hard['chrf'].mean():.2f}")
for _, row in hard.head(5).iterrows():
    print(f"    inp={row['inp_bytes']}B, ref={row['ref_words']}w, chrF++={row['chrf']:.1f}")
    print(f"      Ref:  {str(row['ref'])[:120]}")
    print(f"      Pred: {str(row['pred'])[:120]}")

# ============================================================
# 6. sent_idx別スコア（文書の何番目の文か）
# ============================================================
print("\n" + "=" * 70)
print("sent_idx別スコア")
print("=" * 70)

for idx in sorted(df["sent_idx"].unique()):
    subset = df[df["sent_idx"] == idx]
    print(f"  sent_idx={idx}: {len(subset)}件, chrF++={subset['chrf'].mean():.2f}, avg inp={subset['inp_bytes'].mean():.0f}B")
