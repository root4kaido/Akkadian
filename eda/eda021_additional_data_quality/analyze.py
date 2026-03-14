"""
eda021: additional_train.csv vs train.csv の品質・特性比較
- テキスト長分布（Akkadian / English）
- 語彙のオーバーラップ
- 英訳の文体・品質指標
- ドメイン（テキストジャンル）の違い
- ノイズ率（空文、極端に短い/長い、特殊文字）
"""
import pandas as pd
import numpy as np
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent

# ============================================================
# Load data
# ============================================================
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
additional_df = pd.read_csv(str(PROJECT_ROOT / "workspace" / "exp017_additional_data" / "dataset" / "additional_train.csv"))
test_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "test.csv"))

print(f"Train: {len(train_df)} rows")
print(f"Additional: {len(additional_df)} rows")
print(f"Test: {len(test_df)} rows")
print()

# ============================================================
# 1. Basic stats
# ============================================================
def text_stats(df, col, label):
    texts = df[col].astype(str)
    lengths_char = texts.str.len()
    lengths_word = texts.str.split().str.len()
    lengths_byte = texts.apply(lambda x: len(x.encode('utf-8')))
    print(f"=== {label} - {col} ===")
    print(f"  char len: mean={lengths_char.mean():.1f}, median={lengths_char.median():.1f}, "
          f"min={lengths_char.min()}, max={lengths_char.max()}, std={lengths_char.std():.1f}")
    print(f"  word len: mean={lengths_word.mean():.1f}, median={lengths_word.median():.1f}, "
          f"min={lengths_word.min()}, max={lengths_word.max()}, std={lengths_word.std():.1f}")
    print(f"  byte len: mean={lengths_byte.mean():.1f}, median={lengths_byte.median():.1f}, "
          f"min={lengths_byte.min()}, max={lengths_byte.max()}, std={lengths_byte.std():.1f}")
    return lengths_char, lengths_word, lengths_byte

print("=" * 70)
print("1. テキスト長分布")
print("=" * 70)
for col in ['transliteration', 'translation']:
    text_stats(train_df, col, "Train")
    text_stats(additional_df, col, "Additional")
    if col == 'transliteration':
        text_stats(test_df, col, "Test")
    print()

# ============================================================
# 2. 長さ分布のヒストグラム比較（テキスト出力）
# ============================================================
print("=" * 70)
print("2. Akkadian transliteration 単語数分布")
print("=" * 70)

def print_histogram(data, label, bins):
    counts, edges = np.histogram(data, bins=bins)
    total = len(data)
    for i, c in enumerate(counts):
        pct = c / total * 100
        bar = "#" * int(pct)
        print(f"  {edges[i]:5.0f}-{edges[i+1]:5.0f}: {c:5d} ({pct:5.1f}%) {bar}")

bins_akk = [0, 5, 10, 20, 30, 50, 100, 200, 500, 10000]

train_akk_wlen = train_df['transliteration'].astype(str).str.split().str.len()
add_akk_wlen = additional_df['transliteration'].astype(str).str.split().str.len()
test_akk_wlen = test_df['transliteration'].astype(str).str.split().str.len()

print("\nTrain:")
print_histogram(train_akk_wlen, "Train", bins_akk)
print("\nAdditional:")
print_histogram(add_akk_wlen, "Additional", bins_akk)
print("\nTest:")
print_histogram(test_akk_wlen, "Test", bins_akk)

print()
print("=" * 70)
print("3. English translation 単語数分布")
print("=" * 70)

bins_eng = [0, 5, 10, 20, 30, 50, 100, 200, 500, 10000]

train_eng_wlen = train_df['translation'].astype(str).str.split().str.len()
add_eng_wlen = additional_df['translation'].astype(str).str.split().str.len()

print("\nTrain:")
print_histogram(train_eng_wlen, "Train", bins_eng)
print("\nAdditional:")
print_histogram(add_eng_wlen, "Additional", bins_eng)

# ============================================================
# 3. 語彙分析
# ============================================================
print()
print("=" * 70)
print("4. Akkadian 語彙オーバーラップ")
print("=" * 70)

def get_vocab(series):
    words = Counter()
    for text in series.astype(str):
        words.update(text.lower().split())
    return words

train_akk_vocab = get_vocab(train_df['transliteration'])
add_akk_vocab = get_vocab(additional_df['transliteration'])
test_akk_vocab = get_vocab(test_df['transliteration'])

train_set = set(train_akk_vocab.keys())
add_set = set(add_akk_vocab.keys())
test_set = set(test_akk_vocab.keys())

print(f"Train unique tokens: {len(train_set)}")
print(f"Additional unique tokens: {len(add_set)}")
print(f"Test unique tokens: {len(test_set)}")
print()

overlap_train_add = train_set & add_set
print(f"Train ∩ Additional: {len(overlap_train_add)} ({len(overlap_train_add)/len(train_set)*100:.1f}% of train, {len(overlap_train_add)/len(add_set)*100:.1f}% of additional)")

overlap_train_test = train_set & test_set
print(f"Train ∩ Test: {len(overlap_train_test)} ({len(overlap_train_test)/len(test_set)*100:.1f}% of test covered by train)")

overlap_add_test = add_set & test_set
print(f"Additional ∩ Test: {len(overlap_add_test)} ({len(overlap_add_test)/len(test_set)*100:.1f}% of test covered by additional)")

overlap_all = train_set | add_set
print(f"(Train ∪ Additional) ∩ Test: {len(overlap_all & test_set)} ({len(overlap_all & test_set)/len(test_set)*100:.1f}% of test)")

# Additional にしかないtest語彙
add_only_test = (add_set & test_set) - train_set
print(f"\nAdditionalのみにあるTest語彙: {len(add_only_test)}")
if add_only_test:
    print(f"  例: {list(add_only_test)[:20]}")

# Test にしかない語彙
test_only = test_set - (train_set | add_set)
print(f"Train+Additionalにないtest語彙: {len(test_only)}")
if test_only:
    print(f"  例: {list(test_only)[:20]}")

# ============================================================
# 4. 英語翻訳の語彙分析
# ============================================================
print()
print("=" * 70)
print("5. English 語彙分析")
print("=" * 70)

train_eng_vocab = get_vocab(train_df['translation'])
add_eng_vocab = get_vocab(additional_df['translation'])

train_eng_set = set(train_eng_vocab.keys())
add_eng_set = set(add_eng_vocab.keys())

print(f"Train eng unique tokens: {len(train_eng_set)}")
print(f"Additional eng unique tokens: {len(add_eng_set)}")

# Additional にしかない英語語彙（上位20）
add_only_eng = add_eng_set - train_eng_set
print(f"\nAdditionalにしかない英語語彙: {len(add_only_eng)}")
add_only_eng_freq = {w: add_eng_vocab[w] for w in add_only_eng}
top_add_only = sorted(add_only_eng_freq.items(), key=lambda x: -x[1])[:30]
print(f"  上位30: {top_add_only}")

# ============================================================
# 5. ノイズ分析
# ============================================================
print()
print("=" * 70)
print("6. ノイズ・品質指標")
print("=" * 70)

def noise_analysis(df, label):
    akk = df['transliteration'].astype(str)
    eng = df['translation'].astype(str)

    # 極端に短い
    short_akk = (akk.str.len() < 5).sum()
    short_eng = (eng.str.len() < 5).sum()

    # 極端に長い
    long_akk = (akk.str.split().str.len() > 100).sum()
    long_eng = (eng.str.split().str.len() > 100).sum()

    # 空・NaN
    empty_akk = (akk.str.strip() == '').sum() + df['transliteration'].isna().sum()
    empty_eng = (eng.str.strip() == '').sum() + df['translation'].isna().sum()

    # 破損っぽいもの（翻訳に楔形文字やUnicode文字が混入）
    non_ascii_eng = eng.apply(lambda x: bool(re.search(r'[^\x00-\x7F]', x))).sum()

    # 英語翻訳に"..."や"[...]"が含まれるもの（欠損・不完全翻訳のサイン）
    has_ellipsis = eng.str.contains(r'\.\.\.', regex=True).sum()
    has_brackets = eng.str.contains(r'\[\.+\]|\[\?\]|\[x\]|\[\s*\]', regex=True, case=False).sum()

    # Akkadian側に数字のみ・記号のみ
    digit_only_akk = akk.apply(lambda x: bool(re.match(r'^[\d\s\.\-]+$', x.strip()))).sum()

    # src/tgt長さ比（極端な不均衡）
    ratio = eng.str.len() / akk.str.len().clip(lower=1)
    extreme_ratio = ((ratio > 5) | (ratio < 0.1)).sum()

    total = len(df)
    print(f"\n--- {label} ({total} rows) ---")
    print(f"  短いAkk (<5char): {short_akk} ({short_akk/total*100:.1f}%)")
    print(f"  短いEng (<5char): {short_eng} ({short_eng/total*100:.1f}%)")
    print(f"  長いAkk (>100words): {long_akk} ({long_akk/total*100:.1f}%)")
    print(f"  長いEng (>100words): {long_eng} ({long_eng/total*100:.1f}%)")
    print(f"  空/NaN Akk: {empty_akk}")
    print(f"  空/NaN Eng: {empty_eng}")
    print(f"  Non-ASCII Eng: {non_ascii_eng} ({non_ascii_eng/total*100:.1f}%)")
    print(f"  省略記号 '...': {has_ellipsis} ({has_ellipsis/total*100:.1f}%)")
    print(f"  角括弧欠損 [...]: {has_brackets} ({has_brackets/total*100:.1f}%)")
    print(f"  数字のみAkk: {digit_only_akk}")
    print(f"  極端な長さ比(>5x or <0.1x): {extreme_ratio} ({extreme_ratio/total*100:.1f}%)")

noise_analysis(train_df, "Train")
noise_analysis(additional_df, "Additional")

# ============================================================
# 6. サンプル比較
# ============================================================
print()
print("=" * 70)
print("7. サンプル比較")
print("=" * 70)

print("\n--- Train: ランダム5件 ---")
for _, row in train_df.sample(5, random_state=42).iterrows():
    akk = str(row['transliteration'])[:100]
    eng = str(row['translation'])[:100]
    print(f"  AKK: {akk}")
    print(f"  ENG: {eng}")
    print()

print("\n--- Additional: ランダム5件 ---")
for _, row in additional_df.sample(5, random_state=42).iterrows():
    akk = str(row['transliteration'])[:100]
    eng = str(row['translation'])[:100]
    print(f"  AKK: {akk}")
    print(f"  ENG: {eng}")
    print()

print("\n--- Test: 全件 ---")
for _, row in test_df.iterrows():
    akk = str(row['transliteration'])[:150]
    print(f"  AKK: {akk}")
    print()

# ============================================================
# 7. src/tgt 長さ比の分布
# ============================================================
print()
print("=" * 70)
print("8. Akk/Eng 長さ比（word数ベース）")
print("=" * 70)

def length_ratio_analysis(df, label):
    akk_wlen = df['transliteration'].astype(str).str.split().str.len()
    eng_wlen = df['translation'].astype(str).str.split().str.len()
    ratio = eng_wlen / akk_wlen.clip(lower=1)
    print(f"\n{label}:")
    print(f"  Eng/Akk ratio: mean={ratio.mean():.2f}, median={ratio.median():.2f}, "
          f"std={ratio.std():.2f}, min={ratio.min():.2f}, max={ratio.max():.2f}")

    bins_ratio = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 100]
    print_histogram(ratio, label, bins_ratio)

length_ratio_analysis(train_df, "Train")
length_ratio_analysis(additional_df, "Additional")

# ============================================================
# 8. 文の数の分布（英訳に何文含まれるか）
# ============================================================
print()
print("=" * 70)
print("9. 英訳の文数分布（ピリオドで分割）")
print("=" * 70)

def sentence_count_dist(df, label):
    eng = df['translation'].astype(str)
    sent_counts = eng.apply(lambda x: len([s for s in re.split(r'[.!?]+', x) if s.strip()]))
    print(f"\n{label}:")
    bins_sent = [0, 1, 2, 3, 4, 5, 10, 50]
    print_histogram(sent_counts, label, bins_sent)

sentence_count_dist(train_df, "Train")
sentence_count_dist(additional_df, "Additional")

# ============================================================
# 9. Akkadian特殊記号の頻度
# ============================================================
print()
print("=" * 70)
print("10. Akkadian特殊記号・パターン")
print("=" * 70)

def akk_pattern_analysis(df, label):
    akk = df['transliteration'].astype(str)

    # ハイフン接続（楔形文字の音節区切り）
    has_hyphen = akk.str.contains('-').sum()
    # 大文字始まり（固有名詞系）
    has_upper = akk.apply(lambda x: bool(re.search(r'[A-Z]', x))).sum()
    # 破損記号 [...] や x
    has_damage = akk.str.contains(r'\[', regex=True).sum()
    # 数字を含む
    has_digit = akk.str.contains(r'\d').sum()

    total = len(df)
    print(f"\n{label} ({total} rows):")
    print(f"  ハイフン含む: {has_hyphen} ({has_hyphen/total*100:.1f}%)")
    print(f"  大文字含む: {has_upper} ({has_upper/total*100:.1f}%)")
    print(f"  破損記号[...]: {has_damage} ({has_damage/total*100:.1f}%)")
    print(f"  数字含む: {has_digit} ({has_digit/total*100:.1f}%)")

akk_pattern_analysis(train_df, "Train")
akk_pattern_analysis(additional_df, "Additional")
akk_pattern_analysis(test_df, "Test")

# ============================================================
# Save summary
# ============================================================
print()
print("=" * 70)
print("分析完了")
print("=" * 70)
