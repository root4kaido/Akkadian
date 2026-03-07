"""英語翻訳をピリオド分割した場合の文レベル長さ分布"""
import pandas as pd
import re
import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "../../datasets/raw"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

# === 英語翻訳のピリオド分割後の文長 ===
print("=== train.csv 英語翻訳: ピリオド分割後の文長 ===")
all_sentences = []
for _, row in train.iterrows():
    trans = str(row["translation"])
    # ピリオド・!・?で分割
    sents = re.split(r'(?<=[.!?])\s+', trans.strip())
    sents = [s.strip() for s in sents if s.strip()]
    all_sentences.extend(sents)

lens = [len(s.encode('utf-8')) for s in all_sentences]
lens = np.array(lens)
print(f"総文数: {len(lens)}")
print(f"mean: {lens.mean():.0f}, median: {np.median(lens):.0f}")
print(f"max: {lens.max()}, min: {lens.min()}")
print(f"p90: {np.percentile(lens, 90):.0f}, p95: {np.percentile(lens, 95):.0f}, p99: {np.percentile(lens, 99):.0f}")

for threshold in [128, 256, 384, 512, 768, 1024]:
    pct = (lens <= threshold).mean() * 100
    print(f"  <= {threshold} bytes: {pct:.1f}%")

# === アッカド語transliterationの文長（ドキュメント全体） ===
print("\n=== train.csv アッカド語: ドキュメント全体のバイト長 ===")
akk_lens = train["transliteration"].astype(str).apply(lambda x: len(x.encode('utf-8'))).values
print(f"mean: {akk_lens.mean():.0f}, median: {np.median(akk_lens):.0f}")
print(f"max: {akk_lens.max()}, min: {akk_lens.min()}")
print(f"p90: {np.percentile(akk_lens, 90):.0f}, p95: {np.percentile(akk_lens, 95):.0f}")

for threshold in [128, 256, 384, 512, 768, 1024]:
    pct = (akk_lens <= threshold).mean() * 100
    print(f"  <= {threshold} bytes: {pct:.1f}%")

# === テストデータのtransliteration長 ===
print("\n=== test.csv アッカド語: transliterationバイト長 ===")
test_lens = test["transliteration"].astype(str).apply(lambda x: len(x.encode('utf-8'))).values
print(f"総件数: {len(test_lens)}")
print(f"mean: {test_lens.mean():.0f}, median: {np.median(test_lens):.0f}")
print(f"max: {test_lens.max()}, min: {test_lens.min()}")
print(f"p90: {np.percentile(test_lens, 90):.0f}, p95: {np.percentile(test_lens, 95):.0f}, p99: {np.percentile(test_lens, 99):.0f}")

for threshold in [64, 128, 256, 384, 512]:
    pct = (test_lens <= threshold).mean() * 100
    print(f"  <= {threshold} bytes: {pct:.1f}%")

# === Sentences_Oareの翻訳文長（参考: 文レベル） ===
print("\n=== Sentences_Oare 英語翻訳: バイト長 ===")
so = pd.read_csv(f"{DATA_DIR}/Sentences_Oare_FirstWord_LinNum.csv")
so_lens = so["translation"].astype(str).apply(lambda x: len(x.encode('utf-8'))).values
print(f"mean: {so_lens.mean():.0f}, median: {np.median(so_lens):.0f}")
print(f"max: {so_lens.max()}, min: {so_lens.min()}")
print(f"p90: {np.percentile(so_lens, 90):.0f}, p95: {np.percentile(so_lens, 95):.0f}")
