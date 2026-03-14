"""
フィルタ済み追加データの文数分布を調べる
- sentence_aligned.csvと突合して、何文構成のドキュメントかを確認
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# フィルタ済みデータ
filtered_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "processed" / "additional_train_filtered.csv"))
additional_df = pd.read_csv(str(PROJECT_ROOT / "workspace" / "exp017_additional_data" / "dataset" / "additional_train.csv"))
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
sent_aligned = pd.read_csv(str(PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"))

print(f"フィルタ済み: {len(filtered_df)} rows")
print(f"元additional: {len(additional_df)} rows")
print(f"train: {len(train_df)} rows")
print()

# ============================================================
# 英訳をピリオド区切りで文数推定
# ============================================================
def count_sentences(text):
    text = str(text)
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return max(len(sents), 1)

filtered_df['n_sents'] = filtered_df['translation'].apply(count_sentences)
additional_df['n_sents'] = additional_df['translation'].apply(count_sentences)
train_df['n_sents'] = train_df['translation'].apply(count_sentences)

print("=== 英訳の文数分布 ===")
for label, df in [("Train", train_df), ("Additional(元)", additional_df), ("Filtered", filtered_df)]:
    counts = df['n_sents'].value_counts().sort_index()
    total = len(df)
    print(f"\n{label} ({total} rows):")
    for n in range(1, 11):
        c = counts.get(n, 0)
        pct = c / total * 100
        bar = "#" * int(pct)
        print(f"  {n:2d}文: {c:5d} ({pct:5.1f}%) {bar}")
    over10 = df[df['n_sents'] > 10]
    print(f"  11+: {len(over10):5d} ({len(over10)/total*100:5.1f}%)")

# ============================================================
# sentence_aligned.csvとの突合（oare_id経由）
# ============================================================
print("\n\n=== sentence_aligned.csvとの突合 ===")

# additional_train.csvにはoare_idがないので、published_textsで突合
published = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "published_texts.csv"))

# additional_dfのtransliterationでpublished_textsとマッチ
filtered_with_id = filtered_df.merge(
    published[['oare_id', 'transliteration']],
    on='transliteration',
    how='left'
)
matched = filtered_with_id[filtered_with_id['oare_id'].notna()]
print(f"published_textsとマッチ: {len(matched)}/{len(filtered_df)} ({len(matched)/len(filtered_df)*100:.1f}%)")

if len(matched) > 0:
    # sentence_aligned.csvの文数情報と突合
    sent_counts = sent_aligned.groupby('oare_id')['sent_idx'].max().reset_index()
    sent_counts.columns = ['oare_id', 'max_sent_idx']
    sent_counts['n_sents_aligned'] = sent_counts['max_sent_idx'] + 1

    matched_with_sents = matched.merge(sent_counts, on='oare_id', how='left')
    has_alignment = matched_with_sents[matched_with_sents['n_sents_aligned'].notna()]
    print(f"sentence_alignedにもある: {len(has_alignment)}/{len(matched)} ({len(has_alignment)/len(matched)*100:.1f}%)")

    if len(has_alignment) > 0:
        print(f"\nsentence_aligned上の文数分布:")
        sa_counts = has_alignment['n_sents_aligned'].astype(int).value_counts().sort_index()
        for n, c in sa_counts.items():
            pct = c / len(has_alignment) * 100
            bar = "#" * int(pct)
            print(f"  {n:2d}文: {c:5d} ({pct:5.1f}%) {bar}")

# ============================================================
# Akkadian単語数の分布（フィルタ後）
# ============================================================
print("\n\n=== Akkadian単語数分布 ===")
for label, df in [("Train", train_df), ("Filtered", filtered_df)]:
    wlen = df['transliteration'].astype(str).str.split().str.len()
    print(f"\n{label}:")
    bins = [0, 10, 20, 30, 50, 100, 200, 10000]
    counts, edges = np.histogram(wlen, bins=bins)
    total = len(df)
    for i, c in enumerate(counts):
        pct = c / total * 100
        bar = "#" * int(pct)
        print(f"  {edges[i]:5.0f}-{edges[i+1]:5.0f}: {c:5d} ({pct:5.1f}%) {bar}")
