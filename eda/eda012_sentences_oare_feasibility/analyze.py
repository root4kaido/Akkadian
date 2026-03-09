"""
eda012: Sentences_Oare + published_texts 結合による追加翻訳ペア構築の実現可能性調査

調査項目:
1. Sentences_OareのUUIDでpublished_textsのtransliterationを引けるか
2. 引けた場合の品質・カバレッジ
3. 文レベルの対応付けは可能か（Sentences_Oareは文単位、published_textsはdoc単位）
4. 実際に使えるペア数の見積もり
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "datasets/raw"

# --- Load data ---
print("=" * 60)
print("=== データ読み込み ===")
print("=" * 60)

sent = pd.read_csv(f"{DATA_DIR}/Sentences_Oare_FirstWord_LinNum.csv")
pub = pd.read_csv(f"{DATA_DIR}/published_texts.csv")
train = pd.read_csv(f"{DATA_DIR}/train.csv")

print(f"Sentences_Oare: {len(sent)} rows")
print(f"published_texts: {len(pub)} rows")
print(f"train: {len(train)} rows")
print()

# --- 1. UUID結合可能性 ---
print("=" * 60)
print("=== 1. UUID結合可能性 ===")
print("=" * 60)

# Sentences_Oareのtext_uuidでpublished_textsのoare_idを引く
sent_uuids = set(sent["text_uuid"].unique())
pub_uuids = set(pub["oare_id"].unique())
train_uuids = set(train["oare_id"].unique())

matched = sent_uuids & pub_uuids
not_in_pub = sent_uuids - pub_uuids

print(f"Sentences_Oare ユニークtext_uuid: {len(sent_uuids)}")
print(f"published_texts ユニークoare_id: {len(pub_uuids)}")
print(f"マッチ数: {len(matched)} ({len(matched)/len(sent_uuids)*100:.1f}%)")
print(f"pub_textsに存在しない: {len(not_in_pub)}")
print()

# trainとの重複確認
matched_not_in_train = matched - train_uuids
matched_in_train = matched & train_uuids
print(f"マッチのうちtrainに存在: {len(matched_in_train)}")
print(f"マッチのうちtrainに非存在（新規データ）: {len(matched_not_in_train)}")
print()

# --- 2. transliterationの有無チェック ---
print("=" * 60)
print("=== 2. transliterationの有無 ===")
print("=" * 60)

pub_matched = pub[pub["oare_id"].isin(matched)]
has_translit = pub_matched["transliteration"].notna() & (pub_matched["transliteration"].str.strip() != "")
print(f"マッチした文書のうちtransliterationあり: {has_translit.sum()}/{len(pub_matched)} ({has_translit.mean()*100:.1f}%)")

pub_new = pub[pub["oare_id"].isin(matched_not_in_train)]
has_translit_new = pub_new["transliteration"].notna() & (pub_new["transliteration"].str.strip() != "")
print(f"新規文書のうちtransliterationあり: {has_translit_new.sum()}/{len(pub_new)} ({has_translit_new.mean()*100:.1f}%)")
print()

# --- 3. 文レベル対応付けの可能性 ---
print("=" * 60)
print("=== 3. 文レベル対応付けの可能性 ===")
print("=" * 60)

# Sentences_Oareの文数分布（text_uuid別）
sent_per_doc = sent.groupby("text_uuid").size()
print(f"文書あたり文数: mean={sent_per_doc.mean():.1f}, median={sent_per_doc.median():.0f}, "
      f"min={sent_per_doc.min()}, max={sent_per_doc.max()}")

# 1文のみの文書 → doc transliterationをそのまま使える
single_sent = set(sent_per_doc[sent_per_doc == 1].index)
single_sent_new = single_sent & matched_not_in_train
print(f"\n1文のみの文書（分割不要）: {len(single_sent)}")
print(f"  うち新規（trainに非存在）: {len(single_sent_new & pub_uuids)}")

# 1文のみの新規文書でtransliterationあり
single_new_with_translit = pub[
    (pub["oare_id"].isin(single_sent_new)) &
    (pub["transliteration"].notna()) &
    (pub["transliteration"].str.strip() != "")
]
print(f"  うちtransliterationあり: {len(single_new_with_translit)}")

# 複数文の文書
multi_sent = set(sent_per_doc[sent_per_doc > 1].index)
multi_sent_new = multi_sent & matched_not_in_train
print(f"\n複数文の文書: {len(multi_sent)}")
print(f"  うち新規: {len(multi_sent_new & pub_uuids)}")

# --- 4. 1文文書のペア品質サンプル ---
print("=" * 60)
print("=== 4. 1文文書のペア品質サンプル ===")
print("=" * 60)

# 1文の新規文書からサンプル取得
if len(single_new_with_translit) > 0:
    sample_ids = single_new_with_translit["oare_id"].head(10).tolist()
    for uid in sample_ids:
        translit = pub[pub["oare_id"] == uid]["transliteration"].values[0]
        translation = sent[sent["text_uuid"] == uid]["translation"].values[0]
        translit_len = len(str(translit))
        translation_len = len(str(translation))
        print(f"\n[{uid[:8]}...]")
        print(f"  Akk ({translit_len}B): {str(translit)[:150]}...")
        print(f"  Eng ({translation_len}B): {str(translation)[:150]}...")

# --- 5. 複数文文書の分割可能性 ---
print()
print("=" * 60)
print("=== 5. 複数文文書の分割可能性 ===")
print("=" * 60)

# Sentences_Oareにはline_number, side, column, sentence_obj_in_text がある
# これを使って文書内の文の順序を特定可能
print("Sentences_Oareのカラム:", list(sent.columns))
print()

# line_numberとfirst_word情報の有無
print(f"line_number非null率: {sent['line_number'].notna().mean()*100:.1f}%")
print(f"first_word_spelling非null率: {sent['first_word_spelling'].notna().mean()*100:.1f}%")
print(f"first_word_number非null率: {sent['first_word_number'].notna().mean()*100:.1f}%")
print(f"sentence_obj_in_text非null率: {sent['sentence_obj_in_text'].notna().mean()*100:.1f}%")
print()

# 複数文文書の例を見る
print("--- 複数文文書の構造例 ---")
multi_example_ids = list(multi_sent_new & pub_uuids)[:3]
for uid in multi_example_ids:
    doc_sents = sent[sent["text_uuid"] == uid].sort_values("sentence_obj_in_text")
    translit = pub[pub["oare_id"] == uid]["transliteration"].values
    if len(translit) == 0 or pd.isna(translit[0]):
        continue
    translit = translit[0]

    print(f"\n[{uid[:8]}...] transliteration ({len(translit)}B):")
    print(f"  {translit[:200]}...")
    print(f"  文数: {len(doc_sents)}")
    for _, row in doc_sents.iterrows():
        print(f"    sent_obj={row['sentence_obj_in_text']}, "
              f"line={row['line_number']}, "
              f"first_word={row.get('first_word_spelling', 'N/A')}: "
              f"{str(row['translation'])[:80]}...")

# --- 6. doc-levelペア（分割なし）の可能性 ---
print()
print("=" * 60)
print("=== 6. doc-levelペア（分割なし）の可能性 ===")
print("=" * 60)

# 文書全体のtransliteration + 全文translationを結合
# これなら文分割不要でtrain.csvと同じフォーマットになる
new_docs_with_translit = pub[
    (pub["oare_id"].isin(matched_not_in_train)) &
    (pub["transliteration"].notna()) &
    (pub["transliteration"].str.strip() != "")
]
print(f"新規文書でtransliterationあり: {len(new_docs_with_translit)}")

# 各文書の全文translationを結合
doc_translations = sent[sent["text_uuid"].isin(new_docs_with_translit["oare_id"])].groupby("text_uuid").agg(
    n_sents=("translation", "count"),
    all_translation=("translation", lambda x: " ".join(x.dropna().astype(str)))
).reset_index()

print(f"translation結合可能な文書: {len(doc_translations)}")

# 長さ分布
if len(doc_translations) > 0:
    doc_pairs = doc_translations.merge(
        new_docs_with_translit[["oare_id", "transliteration"]],
        left_on="text_uuid", right_on="oare_id"
    )
    doc_pairs["translit_len"] = doc_pairs["transliteration"].str.len()
    doc_pairs["trans_len"] = doc_pairs["all_translation"].str.len()

    print(f"\n構築可能なdoc-levelペア数: {len(doc_pairs)}")
    print(f"  transliteration長: mean={doc_pairs['translit_len'].mean():.0f}, "
          f"median={doc_pairs['translit_len'].median():.0f}")
    print(f"  translation長: mean={doc_pairs['trans_len'].mean():.0f}, "
          f"median={doc_pairs['trans_len'].median():.0f}")
    print(f"  文数: mean={doc_pairs['n_sents'].mean():.1f}, "
          f"median={doc_pairs['n_sents'].median():.0f}")

# --- 7. trainとの比較 ---
print()
print("=" * 60)
print("=== 7. trainデータとの比較 ===")
print("=" * 60)

train["translit_len"] = train["transliteration"].str.len()
train["trans_len"] = train["translation"].str.len()
print(f"train transliteration長: mean={train['translit_len'].mean():.0f}, "
      f"median={train['translit_len'].median():.0f}")
print(f"train translation長: mean={train['trans_len'].mean():.0f}, "
      f"median={train['trans_len'].median():.0f}")

# --- 8. AICC_translationとの比較 ---
print()
print("=" * 60)
print("=== 8. AICC_translation（機械翻訳）の活用可能性 ===")
print("=" * 60)

# trainに無い文書でtransliteration + AICC_translationがあるもの
pub_not_in_train = pub[~pub["oare_id"].isin(train_uuids)]
has_both = (
    pub_not_in_train["transliteration"].notna() &
    (pub_not_in_train["transliteration"].str.strip() != "") &
    pub_not_in_train["AICC_translation"].notna() &
    (pub_not_in_train["AICC_translation"].str.strip() != "")
)
print(f"train外でtransliteration+AICC_translationあり: {has_both.sum()}")

# Sentences_Oareとの比較
has_sent_oare = pub_not_in_train["oare_id"].isin(sent_uuids)
has_translit_only = (
    pub_not_in_train["transliteration"].notna() &
    (pub_not_in_train["transliteration"].str.strip() != "") &
    ~has_sent_oare
)
print(f"train外でtransliterationあり+Sentences_Oareなし: {has_translit_only.sum()}")
print(f"  うちAICC_translationあり: {(has_translit_only & has_both).sum()}")

# --- 9. 最終サマリー ---
print()
print("=" * 60)
print("=== 最終サマリー ===")
print("=" * 60)

print(f"""
■ 結合経路: Sentences_Oare.text_uuid → published_texts.oare_id

■ カバレッジ:
  - Sentences_Oare {len(sent_uuids)}文書 → published_textsにマッチ: {len(matched)} ({len(matched)/len(sent_uuids)*100:.1f}%)
  - うちtrain外（新規）: {len(matched_not_in_train)}文書
  - 新規でtransliterationあり: {len(new_docs_with_translit)}文書

■ 構築可能なペア:
  A) doc-level（文分割なし）: {len(doc_pairs) if 'doc_pairs' in dir() else 'N/A'}ペア
     → train.csvと同じ形式。即座に学習データに追加可能
  B) 1文文書（そのまま文レベル）: {len(single_new_with_translit)}ペア
  C) 複数文文書の分割: line_number + first_word_spelling で分割可能性あり（要詳細調査）

■ train {len(train)}件に対する増加率: +{len(doc_pairs) if 'doc_pairs' in dir() else 'N/A'}件
""")
