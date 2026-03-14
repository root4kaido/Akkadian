"""
eda024 追加分析: 「短すぎ予測」19件の内訳を分類
+ worst全体を人間が読める形で出力
"""
import re
import pandas as pd
import sacrebleu
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
df = pd.read_csv(RESULTS_DIR / "val_predictions_scored.csv")

PREFIX = "translate Akkadian to English: "
df["transliteration"] = df["input"].str.replace(PREFIX, "", n=1)
df["pred"] = df["prediction_clean"].astype(str)
df["ref"] = df["reference"].astype(str)
df["ref_len"] = df["ref"].apply(lambda x: len(x.split()))
df["pred_len"] = df["pred"].apply(lambda x: len(x.split()))
df["len_ratio"] = df["pred_len"] / df["ref_len"].clip(lower=1)
df["input_bytes"] = df["transliteration"].apply(lambda x: len(str(x).encode("utf-8")))

# ============================================================
# 1. 短すぎ19件の詳細分類
# ============================================================
print("=" * 70)
print("1. 「短すぎ」予測 (pred/ref < 0.5) の詳細分類")
print("=" * 70)

short = df[df["len_ratio"] < 0.5].copy()
print(f"\n全{len(short)}件\n")

gap_caused = 0
truncated = 0
other = 0

for i, (_, row) in enumerate(short.iterrows()):
    n_gaps_ref = row["ref"].count("<gap>")
    ref_words_no_gap = len([w for w in row["ref"].split() if w != "<gap>"])

    # 分類
    if n_gaps_ref >= 2 and ref_words_no_gap < 15:
        category = "GAP欠損（参照がgapだらけで短い）"
        gap_caused += 1
    elif row["pred_len"] < row["ref_len"] * 0.3:
        category = "途中打ち切り（モデルが早期停止）"
        truncated += 1
    else:
        category = "その他"
        other += 1

    print(f"[{i+1}] chrF++={row['chrf']:.1f} | ratio={row['len_ratio']:.2f} | {category}")
    print(f"  Input ({row['input_bytes']}B): {row['transliteration'][:120]}")
    print(f"  Ref  ({row['ref_len']}w): {row['ref'][:150]}")
    print(f"  Pred ({row['pred_len']}w): {row['pred'][:150]}")
    print()

print(f"分類結果: GAP欠損={gap_caused}, 途中打ち切り={truncated}, その他={other}")

# ============================================================
# 2. 長すぎ45件の分析
# ============================================================
print("\n" + "=" * 70)
print("2. 「長すぎ」予測 (pred/ref > 1.5) の詳細")
print("=" * 70)

long = df[df["len_ratio"] > 1.5].copy().sort_values("len_ratio", ascending=False)
print(f"\n全{len(long)}件")

# 長すぎの原因分類
hallucination = 0  # 全く関係ない内容を生成
elaboration = 0    # 正しい内容を冗長に生成
ref_gap_short = 0  # 参照がgap欠損で短い

for i, (_, row) in enumerate(long.head(15).iterrows()):
    n_gaps_ref = row["ref"].count("<gap>")

    if n_gaps_ref >= 2:
        cat = "参照GAP欠損"
        ref_gap_short += 1
    elif row["chrf"] > 50:
        cat = "内容は合っているが冗長"
        elaboration += 1
    else:
        cat = "内容ずれ/幻覚"
        hallucination += 1

    print(f"\n[{i+1}] chrF++={row['chrf']:.1f} | ratio={row['len_ratio']:.2f} | {cat}")
    print(f"  Ref  ({row['ref_len']}w): {row['ref'][:150]}")
    print(f"  Pred ({row['pred_len']}w): {row['pred'][:150]}")

# ============================================================
# 3. <gap>を除外した場合のスコア再計算
# ============================================================
print("\n\n" + "=" * 70)
print("3. <gap>含有文を除外した場合のスコア")
print("=" * 70)

no_gap = df[~df["ref"].str.contains("<gap>", regex=False)]
with_gap = df[df["ref"].str.contains("<gap>", regex=False)]

print(f"\n<gap>なし: {len(no_gap)}件, chrF++={no_gap['chrf'].mean():.2f}, BLEU={no_gap['bleu'].mean():.2f}, geo={no_gap['geo'].mean():.2f}")
print(f"<gap>あり: {len(with_gap)}件, chrF++={with_gap['chrf'].mean():.2f}, BLEU={with_gap['bleu'].mean():.2f}, geo={with_gap['geo'].mean():.2f}")

# gap除外の短すぎ・長すぎ
no_gap_short = no_gap[no_gap["len_ratio"] < 0.5]
no_gap_long = no_gap[no_gap["len_ratio"] > 1.5]
print(f"\n<gap>なしの中で:")
print(f"  短すぎ: {len(no_gap_short)}件")
print(f"  長すぎ: {len(no_gap_long)}件")
print(f"  適正: {len(no_gap) - len(no_gap_short) - len(no_gap_long)}件")

# ============================================================
# 4. chrF++ < 40 の全ケース詳細（改善余地の大きいもの）
# ============================================================
print("\n" + "=" * 70)
print("4. chrF++ < 40 の全ケース（改善余地が大きい）")
print("=" * 70)

low = df[df["chrf"] < 40].sort_values("chrf")
print(f"\n全{len(low)}件\n")

for i, (_, row) in enumerate(low.iterrows()):
    n_gaps = row["ref"].count("<gap>")
    print(f"[{i+1}] chrF++={row['chrf']:.1f} | BLEU={row['bleu']:.1f} | ratio={row['len_ratio']:.2f} | gaps={n_gaps}")
    print(f"  Input ({row['input_bytes']}B): {row['transliteration'][:130]}")
    print(f"  Ref:  {row['ref'][:180]}")
    print(f"  Pred: {row['pred'][:180]}")
    print()
