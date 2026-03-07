"""
EDA001: 初期探索 - 全データセットの基本統計とテキスト特性分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 12
import os
import re
from collections import Counter

RAW_DIR = "/home/user/work/Akkadian/dataset/raw"
FIG_DIR = "/home/user/work/Akkadian/eda/eda001_initial_exploration/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. 全ファイルの基本情報
# ============================================================
print("=" * 60)
print("1. 全ファイルの基本情報")
print("=" * 60)

files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.csv')])
for f in files:
    path = os.path.join(RAW_DIR, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    try:
        df = pd.read_csv(path, nrows=0)
        nrows = sum(1 for _ in open(path)) - 1
        print(f"\n{f}: {size_mb:.2f} MB, {nrows} rows, {len(df.columns)} cols")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"\n{f}: {size_mb:.2f} MB - Error: {e}")

# ============================================================
# 2. train.csv 詳細分析
# ============================================================
print("\n" + "=" * 60)
print("2. train.csv 詳細分析")
print("=" * 60)

train = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
print(f"\nShape: {train.shape}")
print(f"\nColumns: {list(train.columns)}")
print(f"\ndtypes:\n{train.dtypes}")
print(f"\nNull counts:\n{train.isnull().sum()}")
print(f"\nDuplicate oare_ids: {train['oare_id'].duplicated().sum()}")

# テキスト長の分析
train['translit_len'] = train['transliteration'].str.len()
train['translit_words'] = train['transliteration'].str.split().apply(len)
train['translation_len'] = train['translation'].str.len()
train['translation_words'] = train['translation'].str.split().apply(len)

print(f"\n--- Transliteration Length (chars) ---")
print(train['translit_len'].describe())
print(f"\n--- Transliteration Length (words) ---")
print(train['translit_words'].describe())
print(f"\n--- Translation Length (chars) ---")
print(train['translation_len'].describe())
print(f"\n--- Translation Length (words) ---")
print(train['translation_words'].describe())

# 比率
train['ratio_words'] = train['translation_words'] / train['translit_words']
print(f"\n--- Translation/Transliteration word ratio ---")
print(train['ratio_words'].describe())

# テキスト長分布のプロット
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(train['translit_words'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Transliteration - Word Count Distribution')
axes[0, 0].set_xlabel('Word count')
axes[0, 0].axvline(train['translit_words'].median(), color='red', linestyle='--', label=f"median={train['translit_words'].median():.0f}")
axes[0, 0].legend()

axes[0, 1].hist(train['translation_words'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Translation - Word Count Distribution')
axes[0, 1].set_xlabel('Word count')
axes[0, 1].axvline(train['translation_words'].median(), color='red', linestyle='--', label=f"median={train['translation_words'].median():.0f}")
axes[0, 1].legend()

axes[1, 0].scatter(train['translit_words'], train['translation_words'], alpha=0.3, s=10, color='steelblue')
axes[1, 0].set_title('Transliteration vs Translation Word Count')
axes[1, 0].set_xlabel('Transliteration words')
axes[1, 0].set_ylabel('Translation words')
# 対角線
max_val = max(train['translit_words'].max(), train['translation_words'].max())
axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

axes[1, 1].hist(train['ratio_words'], bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Translation/Transliteration Word Ratio')
axes[1, 1].set_xlabel('Ratio')
axes[1, 1].axvline(train['ratio_words'].median(), color='red', linestyle='--', label=f"median={train['ratio_words'].median():.2f}")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "train_text_length_dist.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Saved] train_text_length_dist.png")

# ============================================================
# 3. test.csv 詳細分析
# ============================================================
print("\n" + "=" * 60)
print("3. test.csv 詳細分析（ダミーデータ）")
print("=" * 60)

test = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))
print(f"\nShape: {test.shape}")
print(f"\nColumns: {list(test.columns)}")
print(f"\ndtypes:\n{test.dtypes}")
print(f"\nNull counts:\n{test.isnull().sum()}")
print(f"\nUnique text_ids: {test['text_id'].nunique()}")
print(f"\nSentences per document:")
print(test.groupby('text_id').size().describe())

test['translit_len'] = test['transliteration'].str.len()
test['translit_words'] = test['transliteration'].str.split().apply(len)

print(f"\n--- Test Transliteration Length (words) ---")
print(test['translit_words'].describe())

# train vs test の比較プロット
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train['translit_words'], bins=50, alpha=0.6, label='Train (doc-level)', color='steelblue', density=True)
axes[0].hist(test['translit_words'], bins=50, alpha=0.6, label='Test (sent-level)', color='coral', density=True)
axes[0].set_title('Transliteration Word Count: Train vs Test')
axes[0].set_xlabel('Word count')
axes[0].set_ylabel('Density')
axes[0].legend()

# test: sentences per document
sents_per_doc = test.groupby('text_id').size()
axes[1].hist(sents_per_doc, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
axes[1].set_title('Test: Sentences per Document')
axes[1].set_xlabel('Number of sentences')
axes[1].axvline(sents_per_doc.median(), color='red', linestyle='--', label=f"median={sents_per_doc.median():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "train_vs_test_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[Saved] train_vs_test_comparison.png")

# ============================================================
# 4. テキスト中の特殊文字・記号分析
# ============================================================
print("\n" + "=" * 60)
print("4. 翻字テキストの特殊文字・記号分析")
print("=" * 60)

all_translits = pd.concat([train['transliteration'], test['transliteration']])

# 波括弧(限定詞)の出現
det_pattern = r'\{[^}]+\}'
train['det_count'] = train['transliteration'].str.findall(det_pattern).apply(len)
all_dets = []
for t in all_translits:
    all_dets.extend(re.findall(det_pattern, str(t)))
det_counter = Counter(all_dets)
print("\n限定詞(Determinatives) Top 20:")
for det, count in det_counter.most_common(20):
    print(f"  {det}: {count}")

# 大文字語(シュメール語ロゴグラム)の検出
logo_pattern = r'\b[A-Z][A-Z0-9.]+\b'
train['logo_count'] = train['transliteration'].str.findall(logo_pattern).apply(len)
all_logos = []
for t in train['transliteration']:
    all_logos.extend(re.findall(logo_pattern, str(t)))
logo_counter = Counter(all_logos)
print("\nシュメール語ロゴグラム Top 20:")
for logo, count in logo_counter.most_common(20):
    print(f"  {logo}: {count}")

# 破損マーカーの分析
gap_markers = ['[x]', '...', '[…', '…]', '<gap>', '<big_gap>']
print("\n破損マーカーの出現:")
for marker in gap_markers:
    train_count = train['transliteration'].str.count(re.escape(marker)).sum()
    print(f"  '{marker}': train={train_count}")

# Ḫ/ḫ の出現
h_special = train['transliteration'].str.contains('[Ḫḫ]', regex=True).sum()
print(f"\nḪ/ḫ を含むtrainテキスト数: {h_special}/{len(train)} ({h_special/len(train)*100:.1f}%)")
h_special_test = test['transliteration'].str.contains('[Ḫḫ]', regex=True).sum()
print(f"Ḫ/ḫ を含むtestテキスト数: {h_special_test}/{len(test)} ({h_special_test/len(test)*100:.1f}%)")

# 角括弧（破損テキスト）
bracket_train = train['transliteration'].str.contains(r'\[', regex=True).sum()
print(f"\n[ ] を含むtrainテキスト数: {bracket_train}/{len(train)} ({bracket_train/len(train)*100:.1f}%)")

# ============================================================
# 5. 翻訳テキストの分析
# ============================================================
print("\n" + "=" * 60)
print("5. 翻訳テキストの特徴分析")
print("=" * 60)

# 翻訳中の特殊記号
print("\n翻訳テキスト内の特殊パターン:")
patterns_in_trans = {
    '...': r'\.\.\.',
    '(...)': r'\(\.\.\.\)',
    '[...]': r'\[\.\.\.\]',
    'parentheses ()': r'\([^)]+\)',
    'brackets []': r'\[[^\]]+\]',
}
for name, pat in patterns_in_trans.items():
    count = train['translation'].str.contains(pat, regex=True, na=False).sum()
    print(f"  {name}: {count}/{len(train)} ({count/len(train)*100:.1f}%)")

# サンプルの表示
print("\n--- Train サンプル (最初の5件) ---")
for i, row in train.head(5).iterrows():
    print(f"\n[{row['oare_id']}]")
    print(f"  Translit: {row['transliteration'][:200]}...")
    print(f"  Translation: {row['translation'][:200]}...")

print("\n--- Test サンプル (最初の5件) ---")
for i, row in test.head(5).iterrows():
    print(f"\n[id={row['id']}, text_id={row['text_id']}, lines={row['line_start']}-{row['line_end']}]")
    print(f"  Translit: {row['transliteration'][:200]}")

# ============================================================
# 6. 補足データの分析
# ============================================================
print("\n" + "=" * 60)
print("6. 補足データの分析")
print("=" * 60)

# published_texts.csv
pub_texts = pd.read_csv(os.path.join(RAW_DIR, "published_texts.csv"))
print(f"\npublished_texts.csv: {pub_texts.shape}")
print(f"  oare_ids in train: {pub_texts['oare_id'].isin(train['oare_id']).sum()}")
print(f"  oare_ids NOT in train: {(~pub_texts['oare_id'].isin(train['oare_id'])).sum()}")
print(f"  genre_label distribution:")
print(pub_texts['genre_label'].value_counts(dropna=False).head(10).to_string())

# note列にテキストがあるもの
has_note = pub_texts['note'].notna().sum()
print(f"\n  noteが存在: {has_note}/{len(pub_texts)}")

# AICC_translation
has_aicc = pub_texts['AICC_translation'].notna().sum()
print(f"  AICC_translationが存在: {has_aicc}/{len(pub_texts)}")

# publications.csv
pubs = pd.read_csv(os.path.join(RAW_DIR, "publications.csv"))
print(f"\npublications.csv: {pubs.shape}")
print(f"  Unique PDFs: {pubs['pdf_name'].nunique()}")
print(f"  has_akkadian分布:")
print(pubs['has_akkadian'].value_counts().to_string())
print(f"  page_text平均長: {pubs['page_text'].str.len().mean():.0f} chars")

# bibliography.csv
bib = pd.read_csv(os.path.join(RAW_DIR, "bibliography.csv"))
print(f"\nbibliography.csv: {bib.shape}")
print(f"  Year distribution:")
print(pd.to_numeric(bib['year'], errors='coerce').describe())

# OA_Lexicon_eBL.csv
lex = pd.read_csv(os.path.join(RAW_DIR, "OA_Lexicon_eBL.csv"))
print(f"\nOA_Lexicon_eBL.csv: {lex.shape}")
print(f"  Type distribution:")
print(lex['type'].value_counts().head(10).to_string())

# Sentences alignment aid
sents = pd.read_csv(os.path.join(RAW_DIR, "Sentences_Oare_FirstWord_LinNum.csv"))
print(f"\nSentences_Oare_FirstWord_LinNum.csv: {sents.shape}")
print(f"  Columns: {list(sents.columns)}")
print(f"  Sample:")
print(sents.head(3).to_string())

# ============================================================
# 7. Train-Test ギャップ分析
# ============================================================
print("\n" + "=" * 60)
print("7. Train-Test ギャップ分析")
print("=" * 60)

# trainの語彙
train_vocab = set()
for t in train['transliteration']:
    train_vocab.update(str(t).split())

test_vocab = set()
for t in test['transliteration']:
    test_vocab.update(str(t).split())

print(f"\nTrain vocabulary size: {len(train_vocab)}")
print(f"Test vocabulary size: {len(test_vocab)}")
print(f"Overlap: {len(train_vocab & test_vocab)}")
print(f"Test-only words: {len(test_vocab - train_vocab)}")
print(f"Test vocab coverage by train: {len(train_vocab & test_vocab) / len(test_vocab) * 100:.1f}%")

# Test-only words sample
test_only = sorted(test_vocab - train_vocab)[:30]
print(f"\nTest-only words (sample): {test_only}")

# ============================================================
# 8. 可視化: 補足データ
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Genre distribution
genre_counts = pub_texts['genre_label'].value_counts(dropna=False).head(10)
genre_counts.plot.barh(ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Published Texts: Genre Distribution (Top 10)')
axes[0, 0].set_xlabel('Count')

# Lexicon type distribution
lex_type_counts = lex['type'].value_counts().head(10)
lex_type_counts.plot.barh(ax=axes[0, 1], color='coral')
axes[0, 1].set_title('OA Lexicon: Type Distribution (Top 10)')
axes[0, 1].set_xlabel('Count')

# Bibliography year distribution
bib_years = pd.to_numeric(bib['year'], errors='coerce').dropna()
bib_years.astype(int).hist(bins=30, ax=axes[1, 0], color='mediumpurple', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Bibliography: Publication Year Distribution')
axes[1, 0].set_xlabel('Year')

# Determinative count distribution in train
axes[1, 1].hist(train['det_count'], bins=30, color='teal', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Train: Determinative Count per Document')
axes[1, 1].set_xlabel('Determinative count')
axes[1, 1].axvline(train['det_count'].median(), color='red', linestyle='--', label=f"median={train['det_count'].median():.0f}")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "supplemental_data_overview.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Saved] supplemental_data_overview.png")

# ============================================================
# 9. sample_submission確認
# ============================================================
print("\n" + "=" * 60)
print("9. sample_submission.csv")
print("=" * 60)
sub = pd.read_csv(os.path.join(RAW_DIR, "sample_submission.csv"))
print(f"Shape: {sub.shape}")
print(f"Columns: {list(sub.columns)}")
print(sub.head(3).to_string())

print("\n" + "=" * 60)
print("EDA完了!")
print("=" * 60)
