"""
EDA007: 未使用CSVファイルの活用可能性調査
- Sentences_Oare_FirstWord_LinNum.csv
- OA_Lexicon_eBL.csv
- eBL_Dictionary.csv
- published_texts.csv
- publications.csv
- resources.csv
+ ディスカッション/ノートブックでの言及状況
"""

import pandas as pd
import numpy as np
import json
import os
import re
from collections import Counter

RAW = "/home/user/work/Akkadian/datasets/raw"
SURVEY = "/home/user/work/Akkadian/docs/survey"
OUT = "/home/user/work/Akkadian/eda/eda007_unused_csv_analysis"

# ============================================================
# 0. train/test読み込み（参照用）
# ============================================================
print("=" * 70)
print("=== 0. train/test 基本情報 ===")
print("=" * 70)
train = pd.read_csv(f"{RAW}/train.csv")
test = pd.read_csv(f"{RAW}/test.csv")
print(f"train: {len(train)} rows, columns={list(train.columns)}")
print(f"test: {len(test)} rows, columns={list(test.columns)}")
print(f"train oare_id sample: {train['oare_id'].iloc[:3].tolist()}")
print(f"test text_id sample: {test['text_id'].iloc[:3].tolist()}")
print()

# ============================================================
# 1. Sentences_Oare_FirstWord_LinNum.csv
# ============================================================
print("=" * 70)
print("=== 1. Sentences_Oare_FirstWord_LinNum.csv ===")
print("=" * 70)
sent = pd.read_csv(f"{RAW}/Sentences_Oare_FirstWord_LinNum.csv")
print(f"行数: {len(sent)}")
print(f"カラム: {list(sent.columns)}")
print(f"dtypes:\n{sent.dtypes}\n")
print("--- head 5 ---")
print(sent.head().to_string())
print()

# text_uuidとtrain.oare_idの突合
train_ids = set(train['oare_id'])
sent_text_ids = set(sent['text_uuid'].dropna())
overlap_train = train_ids & sent_text_ids
print(f"train oare_id の数: {len(train_ids)}")
print(f"Sentences_Oare text_uuid ユニーク数: {len(sent_text_ids)}")
print(f"trainと一致するtext_uuid: {len(overlap_train)} ({100*len(overlap_train)/len(train_ids):.1f}%)")

# test.text_idとの突合
test_text_ids = set(test['text_id'].dropna())
overlap_test = test_text_ids & sent_text_ids
print(f"test text_id ユニーク数: {len(test_text_ids)}")
print(f"testと一致するtext_uuid: {len(overlap_test)} ({100*len(overlap_test)/max(len(test_text_ids),1):.1f}%)")

# translationの分析
has_translation = sent['translation'].notna() & (sent['translation'].str.strip() != '')
print(f"\ntranslation非空: {has_translation.sum()} / {len(sent)} ({100*has_translation.mean():.1f}%)")
print(f"translation平均文字数: {sent.loc[has_translation, 'translation'].str.len().mean():.1f}")
print(f"translation中央値文字数: {sent.loc[has_translation, 'translation'].str.len().median():.1f}")

# 1 text_uuidあたりのsentence数
sents_per_doc = sent.groupby('text_uuid').size()
print(f"\n1文書あたりsentence数: mean={sents_per_doc.mean():.1f}, median={sents_per_doc.median():.1f}, max={sents_per_doc.max()}")

# trainと一致する文書のsentence情報
if overlap_train:
    train_sents = sent[sent['text_uuid'].isin(overlap_train)]
    print(f"trainと一致する文書のsentence数合計: {len(train_sents)}")
    spt = train_sents.groupby('text_uuid').size()
    print(f"  mean={spt.mean():.1f}, median={spt.median():.1f}, max={spt.max()}")

    # サンプル: 1つのtrain文書のsentence一覧
    sample_uuid = list(overlap_train)[:1][0]
    sample_sents = sent[sent['text_uuid'] == sample_uuid][['translation', 'first_word_spelling', 'line_number']].head(10)
    print(f"\nサンプル文書 ({sample_uuid}):")
    print(sample_sents.to_string())

    # 対応するtrain翻訳
    train_row = train[train['oare_id'] == sample_uuid]
    if len(train_row) > 0:
        print(f"\n対応train翻訳 (先頭500文字):")
        print(train_row['translation'].iloc[0][:500])

# first_word_spellingの分布
print(f"\nfirst_word_spelling上位20:")
fw_counts = sent['first_word_spelling'].value_counts().head(20)
for w, c in fw_counts.items():
    print(f"  {w}: {c}")

# 重要: trainにない文書のsentence（追加データの可能性）
extra_text_ids = sent_text_ids - train_ids
extra_sents = sent[sent['text_uuid'].isin(extra_text_ids)]
has_trans_extra = extra_sents['translation'].notna() & (extra_sents['translation'].str.strip() != '')
print(f"\ntrainに無い文書のsentence数: {len(extra_sents)} (文書数: {len(extra_text_ids)})")
print(f"  うちtranslation非空: {has_trans_extra.sum()}")

print()

# ============================================================
# 2. OA_Lexicon_eBL.csv
# ============================================================
print("=" * 70)
print("=== 2. OA_Lexicon_eBL.csv ===")
print("=" * 70)
lex = pd.read_csv(f"{RAW}/OA_Lexicon_eBL.csv")
print(f"行数: {len(lex)}")
print(f"カラム: {list(lex.columns)}")
print(f"dtypes:\n{lex.dtypes}\n")
print("--- head 10 ---")
print(lex.head(10).to_string())
print()

# type分布
print("type分布:")
type_counts = lex['type'].value_counts()
for t, c in type_counts.items():
    print(f"  {t}: {c} ({100*c/len(lex):.1f}%)")

# form/norm/lexemeの関係
print(f"\nform ユニーク数: {lex['form'].nunique()}")
print(f"norm ユニーク数: {lex['norm'].nunique()}")
print(f"lexeme ユニーク数: {lex['lexeme'].nunique()}")

# 同じlexemeに複数のformがある例
lex_grouped = lex.groupby('lexeme')['form'].apply(list)
multi_form = lex_grouped[lex_grouped.apply(len) > 3]
print(f"\n3つ以上のformを持つlexeme数: {len(multi_form)}")
if len(multi_form) > 0:
    for lexeme, forms in multi_form.head(5).items():
        print(f"  {lexeme}: {forms[:10]}")

# PN (固有名詞) の分析
pn = lex[lex['type'] == 'PN']
print(f"\nPN (固有名詞): {len(pn)} entries")
print(f"  form sample: {pn['form'].head(10).tolist()}")
print(f"  norm sample: {pn['norm'].head(10).tolist()}")

# GN (地名) の分析
gn = lex[lex['type'] == 'GN']
print(f"\nGN (地名): {len(gn)} entries")
print(f"  form sample: {gn['form'].head(10).tolist()}")

# train翻字内でのLexiconマッチング（サンプルベース）
print("\n--- train翻字とLexiconのマッチング ---")
all_forms = set(lex['form'].dropna().str.lower())
sample_train = train.sample(min(50, len(train)), random_state=42)
match_rates = []
for _, row in sample_train.iterrows():
    words = row['transliteration'].lower().replace('-', ' ').split()
    matches = sum(1 for w in words if w in all_forms)
    match_rates.append(matches / max(len(words), 1))
print(f"train翻字の語彙マッチ率 (50サンプル): mean={np.mean(match_rates):.3f}, median={np.median(match_rates):.3f}")

# test翻字でも同様
match_rates_test = []
for _, row in test.iterrows():
    words = row['transliteration'].lower().replace('-', ' ').split()
    matches = sum(1 for w in words if w in all_forms)
    match_rates_test.append(matches / max(len(words), 1))
print(f"test翻字の語彙マッチ率 (全件): mean={np.mean(match_rates_test):.3f}, median={np.median(match_rates_test):.3f}")

# Female(f)カラム
if 'Female(f)' in lex.columns:
    print(f"\nFemale(f)カラム非空: {lex['Female(f)'].notna().sum()}")
# Alt_lexカラム
if 'Alt_lex' in lex.columns:
    print(f"Alt_lex カラム非空: {lex['Alt_lex'].notna().sum()}")
    alt = lex[lex['Alt_lex'].notna()]
    if len(alt) > 0:
        print(f"Alt_lex サンプル:")
        print(alt[['form', 'norm', 'lexeme', 'Alt_lex']].head(10).to_string())

print()

# ============================================================
# 3. eBL_Dictionary.csv
# ============================================================
print("=" * 70)
print("=== 3. eBL_Dictionary.csv ===")
print("=" * 70)
ebl = pd.read_csv(f"{RAW}/eBL_Dictionary.csv")
print(f"行数: {len(ebl)}")
print(f"カラム: {list(ebl.columns)}")
print(f"dtypes:\n{ebl.dtypes}\n")
print("--- head 10 ---")
print(ebl.head(10).to_string())
print()

# definition分析
has_def = ebl['definition'].notna() & (ebl['definition'].str.strip() != '')
print(f"definition非空: {has_def.sum()} / {len(ebl)} ({100*has_def.mean():.1f}%)")
print(f"definition平均長: {ebl.loc[has_def, 'definition'].str.len().mean():.1f}")

# word→definition のサンプル（一般的な語彙）
print("\n--- word→definition サンプル (先頭20) ---")
for _, row in ebl.head(20).iterrows():
    print(f"  {row['word']}: {str(row['definition'])[:100]}")

# derived_from の分析
has_derived = ebl['derived_from'].notna()
print(f"\nderived_from 非空: {has_derived.sum()} ({100*has_derived.mean():.1f}%)")

# trainの英語翻訳に出る語とeBL definitionの対応
print("\n--- eBL Dictionaryの実用性チェック ---")
# trainの翻字で使われている語がeBLにあるか
ebl_words = set(ebl['word'].dropna().str.lower().str.strip())
# 簡易マッチ: 翻字の各トークンがeBL wordに部分一致するか
sample_translit = train['transliteration'].iloc[:5]
for i, t in enumerate(sample_translit):
    tokens = t.lower().replace('-', ' ').split()[:10]
    matches = [tok for tok in tokens if any(tok in w for w in list(ebl_words)[:1000])]
    print(f"  train[{i}] 先頭10トークン中マッチ: {len(matches)}/{min(len(tokens),10)}")

print()

# ============================================================
# 4. published_texts.csv
# ============================================================
print("=" * 70)
print("=== 4. published_texts.csv ===")
print("=" * 70)
pub_texts = pd.read_csv(f"{RAW}/published_texts.csv")
print(f"行数: {len(pub_texts)}")
print(f"カラム: {list(pub_texts.columns)}")
print(f"dtypes:\n{pub_texts.dtypes}\n")
print("--- head 3 (主要カラムのみ) ---")
key_cols = [c for c in ['oare_id', 'label', 'genre_label', 'transliteration', 'transliteration_orig'] if c in pub_texts.columns]
print(pub_texts[key_cols].head(3).to_string())
print()

# transliterationの分析
has_translit = pub_texts['transliteration'].notna() & (pub_texts['transliteration'].str.strip() != '')
print(f"transliteration 非空: {has_translit.sum()} / {len(pub_texts)} ({100*has_translit.mean():.1f}%)")
print(f"transliteration 平均文字数: {pub_texts.loc[has_translit, 'transliteration'].str.len().mean():.1f}")

# translationカラムがあるか
if 'translation' in pub_texts.columns:
    has_trans = pub_texts['translation'].notna() & (pub_texts['translation'].str.strip() != '')
    print(f"translation 非空: {has_trans.sum()} / {len(pub_texts)} ({100*has_trans.mean():.1f}%)")
else:
    print("translation カラムなし")

# trainとの突合
pub_ids = set(pub_texts['oare_id'].dropna()) if 'oare_id' in pub_texts.columns else set()
overlap_pub_train = train_ids & pub_ids
print(f"\ntrain oare_idと一致: {len(overlap_pub_train)} / {len(train_ids)} ({100*len(overlap_pub_train)/len(train_ids):.1f}%)")

# testとの突合
overlap_pub_test = test_text_ids & pub_ids
print(f"test text_idと一致: {len(overlap_pub_test)} / {len(test_text_ids)} ({100*len(overlap_pub_test)/max(len(test_text_ids),1):.1f}%)")

# trainにもtestにもない追加テキスト
extra_pub = pub_ids - train_ids - test_text_ids
print(f"train/testどちらにもない: {len(extra_pub)} テキスト")

# genre_label分布
if 'genre_label' in pub_texts.columns:
    print(f"\ngenre_label分布:")
    for g, c in pub_texts['genre_label'].value_counts().head(10).items():
        print(f"  {g}: {c}")

# online_transcript カラム
if 'online_transcript' in pub_texts.columns:
    has_online = pub_texts['online_transcript'].notna()
    print(f"\nonline_transcript 非空: {has_online.sum()}")

# AICC翻訳カラム
aicc_cols = [c for c in pub_texts.columns if 'aicc' in c.lower() or 'machine' in c.lower() or 'mt' in c.lower()]
print(f"\nAICC/機械翻訳関連カラム: {aicc_cols}")
for c in aicc_cols:
    has_val = pub_texts[c].notna() & (pub_texts[c].str.strip() != '')
    print(f"  {c} 非空: {has_val.sum()}")

# 全カラム表示（translationに相当するものを探す）
print(f"\n全カラム一覧:")
for c in pub_texts.columns:
    non_null = pub_texts[c].notna().sum()
    print(f"  {c}: {non_null} non-null ({100*non_null/len(pub_texts):.0f}%)")

print()

# ============================================================
# 5. publications.csv（先頭5000行）
# ============================================================
print("=" * 70)
print("=== 5. publications.csv (先頭5000行) ===")
print("=" * 70)
pubs = pd.read_csv(f"{RAW}/publications.csv", nrows=5000)
pubs_full_count = sum(1 for _ in open(f"{RAW}/publications.csv")) - 1
print(f"全行数: {pubs_full_count}")
print(f"カラム: {list(pubs.columns)}")
print(f"dtypes:\n{pubs.dtypes}\n")

# has_akkadianの分布
if 'has_akkadian' in pubs.columns:
    akk_dist = pubs['has_akkadian'].value_counts()
    print(f"has_akkadian分布 (先頭5000行):")
    for v, c in akk_dist.items():
        print(f"  {v}: {c} ({100*c/len(pubs):.1f}%)")

    # has_akkadian=True のサンプル
    akk_pages = pubs[pubs['has_akkadian'] == True]
    print(f"\nhas_akkadian=True のpage_text サンプル (先頭500文字 x3):")
    for i, (_, row) in enumerate(akk_pages.head(3).iterrows()):
        text = str(row.get('page_text', ''))[:500]
        print(f"  [{i}] pdf={row.get('pdf_name','')[:60]}, page={row.get('page','')}:")
        print(f"      {text[:300]}")
        print()

# bibliography.csv との関係
print("--- bibliography.csv ---")
bib = pd.read_csv(f"{RAW}/bibliography.csv")
print(f"行数: {len(bib)}")
print(f"カラム: {list(bib.columns)}")
print(bib.head(3).to_string())

# publications.csv の pdf_name と bibliography の pdf_name の突合
if 'pdf_name' in pubs.columns and 'pdf_name' in bib.columns:
    pub_pdfs = set(pubs['pdf_name'].dropna())
    bib_pdfs = set(bib['pdf_name'].dropna())
    overlap_bib = pub_pdfs & bib_pdfs
    print(f"\npublications pdf_name ユニーク: {len(pub_pdfs)}")
    print(f"bibliography pdf_name ユニーク: {len(bib_pdfs)}")
    print(f"一致: {len(overlap_bib)}")

print()

# ============================================================
# 6. resources.csv
# ============================================================
print("=" * 70)
print("=== 6. resources.csv ===")
print("=" * 70)
res = pd.read_csv(f"{RAW}/resources.csv")
print(f"行数: {len(res)}")
print(f"カラム: {list(res.columns)}")
print(f"dtypes:\n{res.dtypes}\n")
print("--- head 5 ---")
print(res.head().to_string())
# Topics/Methods分布
if 'Topics' in res.columns:
    print(f"\nTopics分布 (上位10):")
    for t, c in res['Topics'].value_counts().head(10).items():
        print(f"  {t}: {c}")
if 'Methods' in res.columns:
    print(f"\nMethods分布 (上位10):")
    for m, c in res['Methods'].value_counts().head(10).items():
        print(f"  {m}: {c}")

print()

# ============================================================
# 7. ディスカッション/ノートブックでの言及状況
# ============================================================
print("=" * 70)
print("=== 7. ディスカッション/ノートブックでの言及状況 ===")
print("=" * 70)

keywords = {
    'Sentences_Oare': ['sentences_oare', 'sentence_oare', 'firstword', 'first_word', 'sentence alignment'],
    'OA_Lexicon': ['oa_lexicon', 'lexicon_ebl', 'lexicon'],
    'eBL_Dictionary': ['ebl_dictionary', 'ebl dictionary', 'ebl_dict'],
    'published_texts': ['published_texts', 'published_text', 'published.csv'],
    'publications': ['publications.csv', 'publications csv', 'page_text', 'ocr'],
    'onomasticon': ['onomasticon', 'proper name', 'proper noun', 'pn list'],
}

# ディスカッション詳細から検索
disc_path = f"{SURVEY}/discussion/snapshot_20260306_details.json"
with open(disc_path) as f:
    disc_data = json.load(f)

discussions = disc_data.get('discussions', [])
print(f"ディスカッション数: {len(discussions)}")

for csv_name, kws in keywords.items():
    mentions = []
    for d in discussions:
        body = str(d.get('body', '')).lower()
        title = str(d.get('title', '')).lower()
        text = title + ' ' + body
        for kw in kws:
            if kw in text:
                mentions.append(d.get('title', '')[:80])
                break
    if mentions:
        print(f"\n【{csv_name}】 {len(mentions)}件の言及:")
        for m in mentions:
            print(f"  - {m}")
    else:
        print(f"\n【{csv_name}】 言及なし")

# ノートブック詳細から検索
nb_path = f"{SURVEY}/notebooks/snapshot_20260306_details.json"
with open(nb_path) as f:
    nb_data = json.load(f)

notebooks = nb_data.get('notebooks', [])
print(f"\nノートブック数: {len(notebooks)}")

for csv_name, kws in keywords.items():
    mentions = []
    for nb in notebooks:
        code = str(nb.get('analysis', {}).get('code_summary', '') if isinstance(nb.get('analysis'), dict) else '').lower()
        title = str(nb.get('title', '')).lower()
        text = title + ' ' + code
        for kw in kws:
            if kw in text:
                mentions.append((nb.get('title', '')[:60], nb.get('votes', 0)))
                break
    if mentions:
        print(f"\n【{csv_name}】 {len(mentions)}件の言及:")
        for title, votes in mentions:
            print(f"  - {title} (votes: {votes})")
    else:
        print(f"\n【{csv_name}】 言及なし")

# ダウンロード済みノートブックの中を直接検索
print("\n--- ダウンロード済みノートブック内の直接検索 ---")
nb_dir = f"{SURVEY}/notebooks/downloaded"
if os.path.exists(nb_dir):
    for root, dirs, files in os.walk(nb_dir):
        for fname in files:
            if fname.endswith('.ipynb'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read().lower()
                    for csv_name, kws in keywords.items():
                        for kw in kws:
                            if kw in content:
                                print(f"  {fname}: 【{csv_name}】 (keyword: {kw})")
                                # 該当部分を抽出
                                idx = content.find(kw)
                                snippet = content[max(0,idx-100):idx+200]
                                # 改行・エスケープを除去して表示
                                snippet = snippet.replace('\\n', ' ').replace('\\t', ' ')
                                print(f"    ...{snippet[:250]}...")
                                break
                except:
                    pass

print()
print("=" * 70)
print("=== 分析完了 ===")
print("=" * 70)
