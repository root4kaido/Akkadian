"""
EDA003: Sentences_Oareの活用可能性 + train.csvの文分割手がかり調査

目的:
1. Sentences_Oare.csvの構造・カラム・transliteration有無を詳細調査
2. train.csvとSentences_Oareのtext_uuid重複を確認（join可能性）
3. train.csvの英語翻訳から文分割の手がかりを探る
4. published_texts.csvとの連携可能性
"""
import os
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "../../datasets/raw"

# =============================================================================
# 1. Sentences_Oare.csv 詳細分析
# =============================================================================
print("=" * 60)
print("1. Sentences_Oare.csv 詳細分析")
print("=" * 60)

so = pd.read_csv(f"{DATA_DIR}/Sentences_Oare_FirstWord_LinNum.csv")
print(f"\nShape: {so.shape}")
print(f"Columns: {list(so.columns)}")
print(f"\nDtypes:\n{so.dtypes}")
print(f"\nNull counts:\n{so.isnull().sum()}")

print(f"\n--- ユニーク数 ---")
for col in so.columns:
    print(f"  {col}: {so[col].nunique()}")

print(f"\n--- translationの統計 ---")
tlen = so["translation"].astype(str).str.len()
print(f"  mean: {tlen.mean():.0f}, median: {tlen.median():.0f}, max: {tlen.max()}, min: {tlen.min()}")
print(f"  p90: {tlen.quantile(0.9):.0f}, p95: {tlen.quantile(0.95):.0f}")

print(f"\n--- サンプル (最初の5件) ---")
for i in range(min(5, len(so))):
    row = so.iloc[i]
    print(f"\n[{i}] display_name: {row['display_name']}")
    print(f"    text_uuid: {row['text_uuid']}")
    print(f"    sentence_uuid: {row['sentence_uuid']}")
    print(f"    sentence_obj_in_text: {row['sentence_obj_in_text']}")
    print(f"    translation: {str(row['translation'])[:150]}")
    print(f"    first_word_transcription: {row['first_word_transcription']}")
    print(f"    first_word_spelling: {row['first_word_spelling']}")
    print(f"    line_number: {row['line_number']}")

# =============================================================================
# 2. train.csv との連携可能性
# =============================================================================
print("\n" + "=" * 60)
print("2. train.csv との連携可能性")
print("=" * 60)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
print(f"\ntrain.csv columns: {list(train.columns)}")
print(f"train.csv shape: {train.shape}")

# text_uuidの重複確認
if "text_uuid" in train.columns:
    train_uuids = set(train["text_uuid"].unique())
    so_uuids = set(so["text_uuid"].unique())
    overlap = train_uuids & so_uuids
    print(f"\ntrain text_uuid count: {len(train_uuids)}")
    print(f"Sentences_Oare text_uuid count: {len(so_uuids)}")
    print(f"Overlap: {len(overlap)} ({len(overlap)/len(train_uuids)*100:.1f}% of train)")

    # overlap分のSentences_Oareを確認
    so_overlapping = so[so["text_uuid"].isin(overlap)]
    print(f"\nOverlapping Sentences_Oare rows: {so_overlapping.shape[0]}")

    # 1つのtext_uuidでtrain.csvのドキュメントとSentences_Oareの文を比較
    if len(overlap) > 0:
        sample_uuid = list(overlap)[0]
        print(f"\n--- サンプル text_uuid: {sample_uuid} ---")
        train_row = train[train["text_uuid"] == sample_uuid].iloc[0]
        so_rows = so[so["text_uuid"] == sample_uuid].sort_values("sentence_obj_in_text")

        print(f"\ntrain transliteration ({len(str(train_row['transliteration']))} chars):")
        print(f"  {str(train_row['transliteration'])[:300]}")
        print(f"\ntrain translation ({len(str(train_row['translation']))} chars):")
        print(f"  {str(train_row['translation'])[:300]}")
        print(f"\nSentences_Oare ({len(so_rows)} sentences):")
        for _, srow in so_rows.iterrows():
            print(f"  [{srow['sentence_obj_in_text']}] {str(srow['translation'])[:150]}")
            print(f"       first_word: {srow['first_word_spelling']} (line: {srow['line_number']})")
else:
    print("\ntrain.csv has no text_uuid column")
    print(f"train columns: {list(train.columns)}")

    # IDベースの連携を試みる
    if "id" in train.columns:
        print(f"\ntrain id sample: {train['id'].head(3).tolist()}")

    # published_texts.csvとの連携
    pt = pd.read_csv(f"{DATA_DIR}/published_texts.csv")
    print(f"\npublished_texts columns: {list(pt.columns)}")
    if "text_uuid" in pt.columns:
        pt_uuids = set(pt["text_uuid"].unique())
        so_uuids = set(so["text_uuid"].unique())
        overlap_pt_so = pt_uuids & so_uuids
        print(f"published_texts text_uuid count: {len(pt_uuids)}")
        print(f"Overlap with Sentences_Oare: {len(overlap_pt_so)}")

# =============================================================================
# 3. train.csv の文分割手がかり
# =============================================================================
print("\n" + "=" * 60)
print("3. train.csv の文分割手がかり")
print("=" * 60)

# 英語翻訳のピリオドベース分割
translations = train["translation"].astype(str)
transliterations = train["transliteration"].astype(str)

# ピリオドで文数を推定
import re
sentence_counts = translations.apply(lambda x: len(re.split(r'(?<=[.!?])\s+', x.strip())))
print(f"\n英語翻訳の推定文数:")
print(f"  mean: {sentence_counts.mean():.1f}, median: {sentence_counts.median():.0f}")
print(f"  1文のみ: {(sentence_counts == 1).sum()} ({(sentence_counts == 1).mean()*100:.1f}%)")
print(f"  2-5文: {((sentence_counts >= 2) & (sentence_counts <= 5)).sum()}")
print(f"  6-10文: {((sentence_counts >= 6) & (sentence_counts <= 10)).sum()}")
print(f"  11文以上: {(sentence_counts >= 11).sum()}")

# アッカド語翻字の区切り文字候補
print(f"\n--- アッカド語翻字の区切り候補 ---")
# ピリオド
has_period = transliterations.str.contains(r"\.").sum()
print(f"  ピリオド(.): {has_period}/{len(train)} ({has_period/len(train)*100:.1f}%)")

# ピリオドのコンテキストを確認（数字の小数点？文末？）
period_samples = transliterations[transliterations.str.contains(r"\.")].head(5)
for idx, s in period_samples.items():
    # ピリオド周辺のコンテキストを表示
    matches = list(re.finditer(r'.{0,20}\..{0,20}', s))
    if matches:
        print(f"  [{idx}] ...{matches[0].group()}...")

# コロン
has_colon = transliterations.str.contains(r":").sum()
print(f"\n  コロン(:): {has_colon}/{len(train)} ({has_colon/len(train)*100:.1f}%)")

# セミコロン
has_semi = transliterations.str.contains(r";").sum()
print(f"  セミコロン(;): {has_semi}/{len(train)} ({has_semi/len(train)*100:.1f}%)")

# 二重スペース
has_double_space = transliterations.str.contains(r"  ").sum()
print(f"  二重スペース: {has_double_space}/{len(train)} ({has_double_space/len(train)*100:.1f}%)")

# <gap>
has_gap = transliterations.str.contains(r"<gap>").sum()
print(f"  <gap>: {has_gap}/{len(train)} ({has_gap/len(train)*100:.1f}%)")

# =============================================================================
# 4. Sentences_Oare transliteration取得可能性
# =============================================================================
print("\n" + "=" * 60)
print("4. published_texts.csv との連携でtransliteration取得可能性")
print("=" * 60)

pt = pd.read_csv(f"{DATA_DIR}/published_texts.csv")
print(f"published_texts columns: {list(pt.columns)}")
print(f"published_texts shape: {pt.shape}")

if "text_uuid" in pt.columns and "transliteration" in pt.columns:
    # Sentences_OareのテキストIDでpublished_textsからtransliterationを取得
    so_uuids = set(so["text_uuid"].unique())
    pt_with_translit = pt[pt["text_uuid"].isin(so_uuids) & pt["transliteration"].notna()]
    print(f"\nSentences_Oare text_uuid in published_texts (with transliteration): {len(pt_with_translit)}")

    if len(pt_with_translit) > 0:
        # サンプル: ドキュメント全体のtransliterationと個別sentenceのtranslation
        sample_uuid = pt_with_translit["text_uuid"].iloc[0]
        pt_sample = pt_with_translit[pt_with_translit["text_uuid"] == sample_uuid].iloc[0]
        so_sample = so[so["text_uuid"] == sample_uuid].sort_values("sentence_obj_in_text")

        print(f"\n--- サンプル: {sample_uuid} ---")
        print(f"published_texts transliteration ({len(str(pt_sample['transliteration']))} chars):")
        print(f"  {str(pt_sample['transliteration'])[:300]}")
        if "translation" in pt.columns:
            print(f"published_texts translation ({len(str(pt_sample.get('translation', '')))} chars):")
            print(f"  {str(pt_sample.get('translation', ''))[:300]}")

        print(f"\nSentences_Oare ({len(so_sample)} sentences):")
        for _, srow in so_sample.head(5).iterrows():
            print(f"  [{srow['sentence_obj_in_text']}] translation: {str(srow['translation'])[:100]}")
            print(f"       first_word_spelling: {srow['first_word_spelling']}, line: {srow['line_number']}")

# =============================================================================
# 5. train.csvとpublished_textsの連携
# =============================================================================
print("\n" + "=" * 60)
print("5. train.csv と published_texts.csv の連携")
print("=" * 60)

if "text_uuid" in pt.columns:
    # train.csvにtext_uuidがあるか確認（なければdisplay_nameなどで結合）
    print(f"train columns: {list(train.columns)}")

    # trainのIDとpublished_textsのIDの対応を探る
    if "display_name" in train.columns and "display_name" in pt.columns:
        train_names = set(train["display_name"].unique()) if "display_name" in train.columns else set()
        pt_names = set(pt["display_name"].unique()) if "display_name" in pt.columns else set()
        overlap_names = train_names & pt_names
        print(f"display_name overlap: {len(overlap_names)}")

    # train.csvのid列をチェック
    print(f"\ntrain id samples: {train['id'].head(5).tolist()}")
    if "display_name" in so.columns:
        print(f"Sentences_Oare display_name samples: {so['display_name'].head(5).tolist()}")

# =============================================================================
# 6. Sentences_Oareの翻訳テキスト品質
# =============================================================================
print("\n" + "=" * 60)
print("6. Sentences_Oare翻訳テキストの品質・特徴")
print("=" * 60)

so_trans = so["translation"].astype(str)
print(f"Total sentences: {len(so)}")
print(f"Unique translations: {so_trans.nunique()}")
print(f"Empty/nan: {(so_trans.isin(['', 'nan'])).sum()}")

# 長さ分布
so_lens = so_trans.str.len()
print(f"\n翻訳長: mean={so_lens.mean():.0f}, median={so_lens.median():.0f}, max={so_lens.max()}, min={so_lens.min()}")
print(f"  p90={so_lens.quantile(0.9):.0f}, p95={so_lens.quantile(0.95):.0f}")

# <gap>の含有率
has_gap_so = so_trans.str.contains("<gap>", regex=False).sum()
print(f"\n<gap>含有: {has_gap_so}/{len(so)} ({has_gap_so/len(so)*100:.1f}%)")

# 言語チェック（英語以外が混在していないか）
non_ascii = so_trans.apply(lambda x: any(ord(c) > 127 for c in str(x))).sum()
print(f"non-ASCII含有: {non_ascii}/{len(so)} ({non_ascii/len(so)*100:.1f}%)")

# ランダムサンプル
print(f"\n--- ランダムサンプル (5件) ---")
samples = so.sample(5, random_state=42)
for _, row in samples.iterrows():
    print(f"  [{row['display_name']}] {str(row['translation'])[:150]}")
