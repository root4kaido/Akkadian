"""train.csvのサンプルデータ確認"""
import pandas as pd

tr = pd.read_csv("datasets/raw/train.csv")
# 短めのサンプルを見る
short = tr[tr["transliteration"].str.len() < 200].head(3)
for i, row in short.iterrows():
    print(f"[{i}]")
    print(f"  translit: {row['transliteration'][:300]}")
    print(f"  translat: {row['translation'][:300]}")
    print()

# 文区切りの手がかりを探す
print("=== 改行の有無 ===")
has_newline = tr["transliteration"].str.contains("\n").sum()
print(f"transliteration with newline: {has_newline}/{len(tr)}")
has_newline_t = tr["translation"].str.contains("\n").sum()
print(f"translation with newline: {has_newline_t}/{len(tr)}")

# ピリオドの有無
print("\n=== ピリオド(.) の有無 ===")
has_period_trans = tr["transliteration"].str.contains(r"\.", regex=True).sum()
print(f"transliteration with period: {has_period_trans}/{len(tr)}")
has_period_transl = tr["translation"].str.contains(r"\.", regex=True).sum()
print(f"translation with period: {has_period_transl}/{len(tr)}")

# 英語翻訳のピリオド数分布
period_counts = tr["translation"].str.count(r"\.")
print(f"\n=== translation内のピリオド数 ===")
print(f"mean: {period_counts.mean():.1f}, median: {period_counts.median():.0f}, max: {period_counts.max()}")
print(period_counts.describe())
