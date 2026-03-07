"""
EDA004: アッカド語に対して英語翻訳が極端に短いケースの分析
- train.csv と Sentences_Oare の両方を調査
- 長さ比率の分布、外れ値の特定、具体例の出力
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = "/home/user/work/Akkadian/datasets/raw"

# ============================================================
# 1. train.csv の分析
# ============================================================
print("=" * 60)
print("1. train.csv の分析")
print("=" * 60)

train = pd.read_csv(f"{RAW_DIR}/train.csv")
print(f"Total rows: {len(train)}")

# 長さ計算（単語数ベース・文字数ベース）
train["src_words"] = train["transliteration"].fillna("").apply(lambda x: len(x.split()))
train["tgt_words"] = train["translation"].fillna("").apply(lambda x: len(x.split()))
train["src_chars"] = train["transliteration"].fillna("").str.len()
train["tgt_chars"] = train["translation"].fillna("").str.len()

# 比率（英語/アッカド語）
train["word_ratio"] = train["tgt_words"] / train["src_words"].replace(0, np.nan)
train["char_ratio"] = train["tgt_chars"] / train["src_chars"].replace(0, np.nan)

print(f"\n--- 単語数の基本統計 ---")
print(f"Akkadian words: mean={train['src_words'].mean():.1f}, median={train['src_words'].median():.1f}")
print(f"English words:  mean={train['tgt_words'].mean():.1f}, median={train['tgt_words'].median():.1f}")
print(f"Word ratio (en/akk): mean={train['word_ratio'].mean():.3f}, median={train['word_ratio'].median():.3f}")

# 分布の可視化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 単語数比率のヒストグラム
axes[0, 0].hist(train["word_ratio"].dropna(), bins=100, edgecolor="black", alpha=0.7)
axes[0, 0].set_title("Word Ratio (English/Akkadian) Distribution")
axes[0, 0].set_xlabel("Ratio")
axes[0, 0].axvline(x=0.3, color="red", linestyle="--", label="threshold=0.3")
axes[0, 0].legend()

# 散布図
axes[0, 1].scatter(train["src_words"], train["tgt_words"], alpha=0.3, s=10)
axes[0, 1].set_xlabel("Akkadian word count")
axes[0, 1].set_ylabel("English word count")
axes[0, 1].set_title("Source vs Target Word Count")
axes[0, 1].plot([0, train["src_words"].max()], [0, train["src_words"].max() * 0.3],
                "r--", label="ratio=0.3")
axes[0, 1].legend()

# 文字数比率
axes[1, 0].hist(train["char_ratio"].dropna(), bins=100, edgecolor="black", alpha=0.7)
axes[1, 0].set_title("Char Ratio (English/Akkadian) Distribution")
axes[1, 0].set_xlabel("Ratio")
axes[1, 0].axvline(x=0.3, color="red", linestyle="--", label="threshold=0.3")
axes[1, 0].legend()

# 文字数散布図
axes[1, 1].scatter(train["src_chars"], train["tgt_chars"], alpha=0.3, s=10)
axes[1, 1].set_xlabel("Akkadian char count")
axes[1, 1].set_ylabel("English char count")
axes[1, 1].set_title("Source vs Target Char Count")
axes[1, 1].plot([0, train["src_chars"].max()], [0, train["src_chars"].max() * 0.3],
                "r--", label="ratio=0.3")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/length_ratio_distribution.png", dpi=150)
plt.close()
print(f"\nSaved: length_ratio_distribution.png")

# ============================================================
# 2. 外れ値の特定（英語が極端に短い）
# ============================================================
print("\n" + "=" * 60)
print("2. 英語が極端に短いケースの特定")
print("=" * 60)

# 複数の閾値で確認
for threshold in [0.1, 0.2, 0.3]:
    count = (train["word_ratio"] < threshold).sum()
    print(f"  word_ratio < {threshold}: {count} rows ({count/len(train)*100:.1f}%)")

# word_ratio < 0.3 の詳細
short_train = train[train["word_ratio"] < 0.3].sort_values("word_ratio")
print(f"\n--- word_ratio < 0.3 のケース ({len(short_train)} rows) ---")

# 具体例を出力
print("\n=== 最も比率が低い上位20件 ===")
for i, (_, row) in enumerate(short_train.head(20).iterrows()):
    print(f"\n[{i+1}] oare_id={row['oare_id']}, ratio={row['word_ratio']:.3f} "
          f"(src={row['src_words']}w, tgt={row['tgt_words']}w)")
    src_preview = row["transliteration"][:200] + ("..." if len(str(row["transliteration"])) > 200 else "")
    tgt_preview = row["translation"][:200] + ("..." if len(str(row["translation"])) > 200 else "")
    print(f"  SRC: {src_preview}")
    print(f"  TGT: {tgt_preview}")

# ============================================================
# 3. パターン分析
# ============================================================
print("\n" + "=" * 60)
print("3. 短い翻訳のパターン分析")
print("=" * 60)

# 翻訳が空 or 非常に短い
empty_or_very_short = train[train["tgt_words"] <= 3]
print(f"英語が3単語以下: {len(empty_or_very_short)} rows")
for _, row in empty_or_very_short.iterrows():
    print(f"  oare_id={row['oare_id']}: '{row['translation']}' (src={row['src_words']}w)")

# gapが大量に含まれるケース
train["gap_count"] = train["transliteration"].fillna("").str.count("<gap>")
train["gap_ratio"] = train["gap_count"] / train["src_words"].replace(0, np.nan)

print(f"\n--- <gap>の影響 ---")
short_with_gap = short_train.merge(train[["oare_id", "gap_count", "gap_ratio"]], on="oare_id")
if len(short_with_gap) > 0:
    print(f"ratio<0.3のうちgapあり: {(short_with_gap['gap_count'] > 0).sum()} / {len(short_with_gap)}")
    print(f"平均gap数: {short_with_gap['gap_count'].mean():.1f}")

# 翻訳に特定パターンがあるか
print(f"\n--- 短い翻訳の内容パターン ---")
if len(short_train) > 0:
    # 翻訳テキストのユニーク度
    print(f"ユニークな翻訳テキスト数: {short_train['translation'].nunique()} / {len(short_train)}")

    # よく出るフレーズ
    from collections import Counter
    tgt_texts = short_train["translation"].tolist()
    counter = Counter(tgt_texts)
    print("頻出する翻訳テキスト:")
    for text, count in counter.most_common(10):
        preview = str(text)[:100]
        print(f"  ({count}回) {preview}")

# ============================================================
# 4. Sentences_Oare の分析
# ============================================================
print("\n" + "=" * 60)
print("4. Sentences_Oare の分析")
print("=" * 60)

sent = pd.read_csv(f"{RAW_DIR}/Sentences_Oare_FirstWord_LinNum.csv")
print(f"Total rows: {len(sent)}")

# translationカラムの長さ
sent["tgt_words"] = sent["translation"].fillna("").apply(lambda x: len(x.split()))
sent["tgt_chars"] = sent["translation"].fillna("").str.len()

# first_word_spellingだけでは文全体の長さがわからないが、翻訳が空 or 極端に短いのは確認可能
short_sent = sent[sent["tgt_words"] <= 2]
print(f"翻訳が2単語以下: {len(short_sent)} rows ({len(short_sent)/len(sent)*100:.1f}%)")

# 翻訳が空
empty_sent = sent[sent["translation"].fillna("").str.strip() == ""]
print(f"翻訳が空: {len(empty_sent)} rows")

print("\n--- 翻訳が短い例 (上位20件) ---")
for i, (_, row) in enumerate(short_sent.head(20).iterrows()):
    print(f"[{i+1}] display_name={row['display_name']}, "
          f"translation='{row['translation']}', "
          f"first_word='{row.get('first_word_spelling', 'N/A')}'")

# ============================================================
# 5. 逆パターン：英語が極端に長い
# ============================================================
print("\n" + "=" * 60)
print("5. 逆パターン：英語が極端に長いケース")
print("=" * 60)

long_train = train[train["word_ratio"] > 5].sort_values("word_ratio", ascending=False)
print(f"word_ratio > 5: {len(long_train)} rows")
for i, (_, row) in enumerate(long_train.head(10).iterrows()):
    print(f"\n[{i+1}] oare_id={row['oare_id']}, ratio={row['word_ratio']:.1f} "
          f"(src={row['src_words']}w, tgt={row['tgt_words']}w)")
    src_preview = str(row["transliteration"])[:150]
    tgt_preview = str(row["translation"])[:150]
    print(f"  SRC: {src_preview}")
    print(f"  TGT: {tgt_preview}")

# ============================================================
# 6. サマリー統計の保存
# ============================================================
# word_ratioの分位数
print("\n" + "=" * 60)
print("6. word_ratio パーセンタイル")
print("=" * 60)
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = train["word_ratio"].quantile(p / 100)
    print(f"  {p}th percentile: {val:.3f}")

# 外れ値候補リストをCSVに保存
short_train_out = train[train["word_ratio"] < 0.3][
    ["oare_id", "src_words", "tgt_words", "word_ratio", "gap_count", "transliteration", "translation"]
].sort_values("word_ratio")
short_train_out.to_csv(f"{OUT_DIR}/short_translation_candidates.csv", index=False)
print(f"\nSaved: short_translation_candidates.csv ({len(short_train_out)} rows)")

long_train_out = train[train["word_ratio"] > 5][
    ["oare_id", "src_words", "tgt_words", "word_ratio", "gap_count", "transliteration", "translation"]
].sort_values("word_ratio", ascending=False)
long_train_out.to_csv(f"{OUT_DIR}/long_translation_candidates.csv", index=False)
print(f"Saved: long_translation_candidates.csv ({len(long_train_out)} rows)")

print("\n=== 分析完了 ===")
