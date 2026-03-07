"""学習データのテキスト長分析 — ByT5のmax_length設定根拠"""
import pandas as pd
import numpy as np

train = pd.read_csv("datasets/raw/train.csv")
test = pd.read_csv("datasets/raw/test.csv")

prefix_len = len("translate Akkadian to English: ")

for name, df, cols in [
    ("train", train, ["transliteration", "translation"]),
    ("test", test, ["transliteration"]),
]:
    print(f"\n=== {name} ({len(df)} rows) ===")
    for col in cols:
        s = df[col].astype(str)
        lens = s.str.len()
        # ByT5入力: prefix + text
        if col == "transliteration":
            lens_with_prefix = lens + prefix_len
        else:
            lens_with_prefix = lens

        print(f"\n  {col} (raw chars):")
        print(f"    mean={lens.mean():.0f}, median={lens.median():.0f}, "
              f"max={lens.max()}, min={lens.min()}")
        print(f"    p90={lens.quantile(0.9):.0f}, p95={lens.quantile(0.95):.0f}, "
              f"p99={lens.quantile(0.99):.0f}")

        if col == "transliteration":
            print(f"  {col} (with prefix, ≈ ByT5 tokens):")
            print(f"    mean={lens_with_prefix.mean():.0f}, median={lens_with_prefix.median():.0f}, "
                  f"max={lens_with_prefix.max()}, min={lens_with_prefix.min()}")
            print(f"    p90={lens_with_prefix.quantile(0.9):.0f}, p95={lens_with_prefix.quantile(0.95):.0f}, "
                  f"p99={lens_with_prefix.quantile(0.99):.0f}")

        # truncation影響: 512超のサンプル数
        for threshold in [512, 1024, 2048]:
            over = (lens > threshold).sum()
            pct = over / len(df) * 100
            print(f"    >{threshold}: {over} ({pct:.1f}%)")

# 入力と出力の長さ相関
print("\n=== train: 入力長 vs 出力長 ===")
src_len = train["transliteration"].astype(str).str.len()
tgt_len = train["translation"].astype(str).str.len()
print(f"  correlation: {src_len.corr(tgt_len):.3f}")
print(f"  出力/入力 ratio: mean={( tgt_len / src_len).mean():.2f}, "
      f"median={(tgt_len / src_len).median():.2f}")

# 入力512超 かつ 出力512超 のサンプル
both_over = ((src_len + prefix_len > 512) & (tgt_len > 512)).sum()
print(f"  入力>512 AND 出力>512: {both_over} ({both_over/len(train)*100:.1f}%)")
