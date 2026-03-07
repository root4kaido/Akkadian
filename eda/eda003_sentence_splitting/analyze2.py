"""
EDA003 追加分析: oare_idベースの連携 + first_word_spellingによる文分割可能性
"""
import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "../../datasets/raw"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
so = pd.read_csv(f"{DATA_DIR}/Sentences_Oare_FirstWord_LinNum.csv")
pt = pd.read_csv(f"{DATA_DIR}/published_texts.csv")

# =============================================================================
# 1. oare_id ベースの連携
# =============================================================================
print("=" * 60)
print("1. oare_id ベースの連携")
print("=" * 60)

print(f"train oare_id samples: {train['oare_id'].head(5).tolist()}")

# Sentences_Oareにoare_idはないが、text_uuidがある
# published_textsにはoare_idとtext_uuidの両方があるか？
print(f"\npublished_texts has oare_id: {'oare_id' in pt.columns}")
print(f"Sentences_Oare has text_uuid: {'text_uuid' in so.columns}")

# published_textsを経由してtrain.csvとSentences_Oareを結合
train_oare_ids = set(train["oare_id"].unique())
pt_oare_ids = set(pt["oare_id"].unique())
overlap_train_pt = train_oare_ids & pt_oare_ids
print(f"\ntrain oare_id: {len(train_oare_ids)}")
print(f"published_texts oare_id: {len(pt_oare_ids)}")
print(f"Overlap: {len(overlap_train_pt)} ({len(overlap_train_pt)/len(train_oare_ids)*100:.1f}% of train)")

# published_texts経由でtext_uuidを取得
if "text_uuid" not in pt.columns:
    print("\npublished_texts has no text_uuid!")
    # display_nameなど他のキーで結合可能か
    print(f"Sentences_Oare display_name: {so['display_name'].head(3).tolist()}")

    # oare_idの形式を確認
    print(f"\ntrain oare_id format: {train['oare_id'].head(5).tolist()}")
    print(f"published_texts oare_id format: {pt['oare_id'].head(5).tolist()}")
else:
    pt_bridge = pt[pt["oare_id"].isin(train_oare_ids)][["oare_id"]].drop_duplicates()
    print(f"\nBridge possible via published_texts: {len(pt_bridge)} train texts")

# =============================================================================
# 2. display_nameで直接結合を試みる
# =============================================================================
print("\n" + "=" * 60)
print("2. display_nameでの結合")
print("=" * 60)

# Sentences_Oareのdisplay_nameからoare_idに近いものを抽出
so_display = so["display_name"].unique()
print(f"Sentences_Oare display_name samples:")
for d in so_display[:10]:
    print(f"  '{d}'")

# published_textsのlabelやdescription
print(f"\npublished_texts label samples:")
for l in pt["label"].head(10):
    print(f"  '{l}'")

# =============================================================================
# 3. Sentences_Oareのtext_uuid → published_texts oare_id → train oare_id
# =============================================================================
print("\n" + "=" * 60)
print("3. text_uuid → oare_id 連携チェーン")
print("=" * 60)

# published_textsにtext_uuidがないなら別ルートを探す
# eBL_idやcdli_idでの連携を試みる
for col in ["text_uuid", "oare_id", "cdli_id", "eBL_id"]:
    if col in pt.columns:
        non_null = pt[col].notna().sum()
        print(f"published_texts.{col}: {non_null}/{len(pt)} non-null")

# Sentences_OareのユニークテキストIDとpublished_textsのサイズ
so_text_count = so["text_uuid"].nunique()
print(f"\nSentences_Oare unique texts: {so_text_count}")
print(f"published_texts texts: {len(pt)}")
print(f"train texts: {len(train)}")

# =============================================================================
# 4. first_word_spellingで文分割の手がかり
# =============================================================================
print("\n" + "=" * 60)
print("4. first_word_spellingで文分割の手がかり")
print("=" * 60)

# 最頻出のfirst_word_spelling
fw_counts = so["first_word_spelling"].value_counts().head(20)
print("最頻出 first_word_spelling:")
for word, count in fw_counts.items():
    pct = count / len(so) * 100
    # 対応するtranscriptionも表示
    trans = so[so["first_word_spelling"] == word]["first_word_transcription"].dropna().iloc[0] if so[so["first_word_spelling"] == word]["first_word_transcription"].notna().any() else "N/A"
    print(f"  {word}: {count} ({pct:.1f}%) — transcription: {trans}")

# =============================================================================
# 5. 実際にtransliterationで文頭語を検索
# =============================================================================
print("\n" + "=" * 60)
print("5. train.csv transliteration内でのfirst_word_spelling出現")
print("=" * 60)

# 上位の文頭語がtransliterationに出現するか確認
top_first_words = fw_counts.head(10).index.tolist()
for fw in top_first_words:
    # transliteration内でこのspellingが出現する回数
    count = train["transliteration"].str.contains(fw, regex=False).sum()
    # 出現位置（文頭以外にも出現するか）
    total_occurrences = train["transliteration"].str.count(fw.replace("(", r"\(").replace(")", r"\)")).sum()
    print(f"  {fw}: {count} texts contain, {total_occurrences} total occurrences")

# =============================================================================
# 6. Sentences_OareのtransliterationなしでもByT5の学習に使えるか？
# =============================================================================
print("\n" + "=" * 60)
print("6. Sentences_Oare: translationのみの活用方法")
print("=" * 60)

# 英語翻訳のみを持つので、以下の用途が考えられる:
# A) 英語→アッカド語の逆翻訳学習データ（translationは英語のみなので使えない）
# B) published_textsからtransliterationを取得してペアを作る

# published_textsにtransliterationがあるテキストのうち、Sentences_Oareのtext_uuidと一致するものは？
# → published_textsにtext_uuidがないので直接結合できない

# oare_idベースで可能か？
# Sentences_Oareにoare_idがない → display_nameで結合が必要

# display_nameの形式を詳しく確認
print("Sentences_Oare display_name format analysis:")
so_dn = so["display_name"].astype(str)
# 括弧内のテキストを抽出
import re
bracketed = so_dn.str.extract(r'\(([^)]+)\)')
print(f"  括弧内テキスト抽出: {bracketed[0].notna().sum()}/{len(so)}")
print(f"  サンプル: {bracketed[0].dropna().head(10).tolist()}")

# published_textsのlabelとの類似性
pt_labels = pt["label"].astype(str)
print(f"\npublished_texts label samples:")
print(f"  {pt_labels.head(10).tolist()}")

# 完全一致の試み: Sentences_Oareのdisplay_nameからラベルを抽出
# 例: " (HS 2931)" → "HS 2931" が published_texts.label にあるか
so_labels = bracketed[0].dropna().str.strip()
pt_label_set = set(pt_labels.str.strip().unique())
matches = so_labels.isin(pt_label_set).sum()
print(f"\nSentences_Oare label → published_texts label 完全一致: {matches}/{len(so_labels)}")

# 部分一致を試みる
if matches < 100:
    # published_textsのlabelをもう少し詳しく見る
    print("\n括弧内ラベルのサンプル:")
    for label in so_labels.unique()[:10]:
        # published_textsで部分一致を検索
        pt_match = pt[pt_labels.str.contains(str(label), regex=False, na=False)]
        if len(pt_match) > 0:
            print(f"  '{label}' → matched {len(pt_match)} in published_texts")
        else:
            print(f"  '{label}' → no match")
