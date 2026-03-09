"""
Sentences_Oare + published_texts から追加doc-levelペアを構築する。
出力: dataset/additional_train.csv (oare_id, transliteration, translation)

結合ロジック:
1. Sentences_Oare.text_uuid → published_texts.oare_id でUUID結合
2. trainに存在するUUIDは除外（重複防止）
3. published_textsからtransliterationを取得
4. Sentences_Oareの文翻訳をsentence_obj_in_text順に結合してdoc-level translation構築
"""
import os
import sys

import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)


def main():
    os.chdir(EXP_DIR)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # データ読み込み
    train = pd.read_csv(config["data"]["train_path"])
    sent = pd.read_csv(config["additional_data"]["sentences_oare_path"])
    pub = pd.read_csv(config["additional_data"]["published_texts_path"])

    print(f"train: {len(train)} rows")
    print(f"Sentences_Oare: {len(sent)} rows")
    print(f"published_texts: {len(pub)} rows")

    train_uuids = set(train["oare_id"].unique())
    pub_uuids = set(pub["oare_id"].unique())
    sent_uuids = set(sent["text_uuid"].unique())

    # train外 & published_textsに存在するUUID
    new_uuids = (sent_uuids & pub_uuids) - train_uuids
    print(f"New UUIDs (not in train, in published_texts): {len(new_uuids)}")

    # published_textsからtransliterationを取得
    pub_new = pub[pub["oare_id"].isin(new_uuids)].copy()
    pub_new = pub_new[
        pub_new["transliteration"].notna() &
        (pub_new["transliteration"].str.strip() != "")
    ]
    print(f"New docs with transliteration: {len(pub_new)}")

    # Sentences_Oareの文翻訳を結合（sentence_obj_in_text順）
    sent_new = sent[sent["text_uuid"].isin(pub_new["oare_id"])].copy()
    sent_new = sent_new.sort_values(["text_uuid", "sentence_obj_in_text"])

    # NaN translationを除外
    sent_new = sent_new[sent_new["translation"].notna()]

    doc_translations = sent_new.groupby("text_uuid").agg(
        translation=("translation", lambda x: " ".join(x.astype(str))),
        n_sents=("translation", "count"),
    ).reset_index()

    # published_textsと結合
    additional = doc_translations.merge(
        pub_new[["oare_id", "transliteration"]],
        left_on="text_uuid",
        right_on="oare_id",
        how="inner",
    )

    # 空のtranslationを除外
    additional = additional[
        (additional["translation"].str.strip().str.len() > 0) &
        (additional["transliteration"].str.strip().str.len() > 0)
    ]

    # train.csvと同じフォーマットで出力
    output = additional[["oare_id", "transliteration", "translation"]].reset_index(drop=True)

    # 統計
    print(f"\n=== Additional data stats ===")
    print(f"Total additional pairs: {len(output)}")
    print(f"Transliteration length: mean={output['transliteration'].str.len().mean():.0f}, "
          f"median={output['transliteration'].str.len().median():.0f}")
    print(f"Translation length: mean={output['translation'].str.len().mean():.0f}, "
          f"median={output['translation'].str.len().median():.0f}")
    print(f"Sentences per doc: mean={additional['n_sents'].mean():.1f}, "
          f"median={additional['n_sents'].median():.0f}")

    # 保存
    output_path = os.path.join(EXP_DIR, "dataset", "additional_train.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(output)} rows)")


if __name__ == "__main__":
    main()
