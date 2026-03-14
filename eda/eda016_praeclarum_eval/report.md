# eda016: praeclarum/cuneiform モデル評価

## 目的

praeclarum/cuneiform（T5-base 223M、HuggingFace公開のアッカド語→英語翻訳モデル）を我々のCV条件で評価し、
直接提出に使えるか、または追加データ生成用として使えるかを判断する。

## 分析概要

- モデル: praeclarum/cuneiform (T5-base, 223M params, MIT license)
- 評価条件: eda015と完全同一（文レベル、extract_first_sentence + truncate 200bytes, seed=42, val 10%）
- 推論: greedy + repeat_cleanup

## 結果

| モデル | chrF++ | BLEU | geo_mean | 繰返率 | 平均長 |
|--------|--------|------|----------|--------|--------|
| **llkh0a** (eda015) | 50.44 | 34.15 | **41.50** | 34.4% | 172 |
| **exp011 (タグあり)** (eda015) | 42.23 | 26.49 | **33.45** | 38.2% | 159 |
| **exp011 (タグなし)** (eda015) | 37.95 | 20.89 | **28.15** | 35.7% | 127 |
| **praeclarum/cuneiform** | 13.19 | 0.94 | **3.53** | 62.4% | 204 |

## 発見事項

1. **praeclarumモデルは壊滅的に低スコア**: geo_mean=3.53で、我々のexp011 (33.45)の1/10以下
2. **繰り返し率62.4%**: 出力の大半が繰り返しで崩壊している
3. **翻訳内容が的外れ**: サンプルを見ると、古アッシリア語（Old Assyrian）の文脈を全く理解できていない
   - 固有名詞（Enna-Suen, Ennam-Aššur等）を認識できず
   - 数値表現（ma-na, GÍN等）の理解が不正確
   - 文構造（手紙の定型句 "um-ma X-ma a-na Y qí-bi₄-ma"）を把握していない
4. **praeclarumはCDLI ATF形式で学習**: 我々のデータはOARE形式で、transliterationのフォーマットが異なる可能性が高い
5. **直接提出には使えない**: スコアが低すぎる
6. **追加データ生成用としても疑問**: 翻訳品質が非常に低く、ノイジーラベルとしても有害な可能性

## 結論

- praeclarum/cuneiformは直接利用・データ生成いずれの用途にも**不適**
- 古アッシリア語（Old Assyrian）に特化していないことが主因と思われる
- データ拡充は publications.csv からの対訳抽出に注力すべき
