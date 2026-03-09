# SESSION_NOTES: exp011_additional_data

## セッション情報
- **日付**: 2026-03-09
- **作業フォルダ**: workspace/exp011_additional_data
- **目標**: Sentences_Oare + published_textsで追加doc-levelペア(+1,166件)を投入し、データ量増加による精度向上を検証

## 仮説
- train 1,561件は少なく、データ不足がボトルネックの一つ
- eda012で1,166件の新規doc-levelペアが構築可能と判明
- Sentences_Oare（文翻訳）を結合 + published_texts（翻字）で、trainと同形式のペアを作成
- 学習データ+74.7%で汎化性能が向上すると期待
- PN/GNタグ付加はexp010から継承

## 追加データの概要
- Sentences_Oare.text_uuid → published_texts.oare_id でUUID結合
- trainに存在しない1,164文書 → published_textsでtransliterationあり1,166件
- 各文書のSentences_Oare翻訳を結合してdoc-levelのtranslation構築
- 注意: 異なるアノテーションパイプライン由来（eda011: 類似度0.544）

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| 追加データ+1,165件 | Sentences_Oare+published_textsでdoc-levelペア追加 | greedy=33.13, MBR=31.54, greedy_clean=**33.45** | - | **全指標で大幅改善。繰り返しも30.6%に低減** |

## ファイル構成
- `src/build_lexicon.py` — OA_Lexiconからform→type辞書構築（exp010と同一）
- `src/build_additional_data.py` — Sentences_Oare + published_textsから追加データ構築
- `src/preprocess.py` — 追加データ結合 + PN/GNタグ付加
- `src/train.py` — exp010ベース
- `src/eval_sentence_level.py` — 文レベル推論・評価（タグ付加対応）

## 重要な知見

- 追加データ+1,165件（train 1,404→2,569件、+83%）で**greedy_clean=33.45**（exp010比+3.53pt）
- **全指標で大幅改善**: greedy +4.17pt, MBR +2.97pt
- **繰り返し率が大幅改善**: greedy 50.3%→38.9%, MBR 41.4%→30.6%
- greedyがMBRを上回る傾向がさらに強まった（33.13 vs 31.54）
- ベストモデル: best_model（training eval geo_mean=31.77）
- 追加データとtrain間のリーク確認済み: oare_id重複0、transliteration完全一致0、高類似度(>0.7)も0

## 性能変化の記録

| 指標 | exp010 | exp011 | 差分 |
|------|--------|--------|------|
| greedy_raw | 28.96 | **33.13** | **+4.17** |
| mbr_raw | 28.57 | **31.54** | **+2.97** |
| greedy_clean | 29.92 | **33.45** | **+3.53** |
| mbr_clean | 28.57 | 31.51 | +2.94 |
| repetition (greedy) | 50.3% | **38.9%** | -11.4pt |
| repetition (mbr) | 41.4% | **30.6%** | -10.8pt |
| training eval geo_mean | 25.55 | 31.77 | +6.22 |

## コマンド履歴
```bash
# 辞書構築 + 追加データ構築 + 学習（dev0, 約4時間20分）
python workspace/exp011_additional_data/src/build_lexicon.py
python workspace/exp011_additional_data/src/build_additional_data.py
python workspace/exp011_additional_data/src/train.py

# 文レベル評価
python workspace/exp011_additional_data/src/eval_sentence_level.py
```

## 次のステップ
