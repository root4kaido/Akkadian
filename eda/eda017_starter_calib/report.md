# eda017: CV-LBキャリブレーション

## 目的

starterノートブック（takamichitoda/dpc-starter-train、LB=26）を我々のデータ分割で再現学習し、
我々のCV評価条件でスコアを出すことで、CV→LBの変換係数を推定する。

## 分析概要

- starterと完全同一のコード再現（byt5-small, 20epoch, 双方向学習, simple_sentence_aligner）
- 唯一の変更: generation_max_length=512追加（starterのバグ修正、学習には影響なし）
- 学習後、我々のCV条件（文レベル、extract_first_sentence + truncate 200bytes, beam4）で評価
- exp011モデルのサブミット結果（greedy LB=20.0, MBR LB=23.8）も判明

## 結果

### Trainer内eval（参考: decoded_labelsとの比較、doc-level、512切り詰め）
- eval_chrf: 38.64, eval_bleu: 16.98, eval_geo_mean: 25.62

### 我々のCV条件（文レベル、extract_first_sentence）
- starter_raw: chrF++=41.41, BLEU=24.80, geo=32.04, rep=52.2%
- starter_clean: chrF++=41.55, BLEU=24.55, geo=31.94, rep=47.8%

### CV-LB対応表

| モデル | CV (our method) | LB | CV/LB比 | 備考 |
|--------|----------------|-----|---------|------|
| starter (byt5-small) | 32.04 | 26.0 | 1.23 | eda017で再現 |
| exp011 greedy_clean | 33.45 | 20.0 | 1.67 | PN/GNタグあり |
| exp011 MBR (13cand) | 34.47 | 23.8 | 1.45 | PN/GNタグあり |
| llkh0a (jeanjean111) | 41.50 | ~32 | 1.30 | jeanjean111 pretrained |

## 発見事項

1. **starterのCV=32.04に対しLB=26** — CV/LB比は1.23
2. **llkh0aもCV/LB比は1.30と近い値**
3. **exp011のCV/LB比は1.45-1.67と異常に高い** — CVが不当に高く出ているか、テスト本番で何かが壊れている
4. **MBR (LB=23.8) > greedy (LB=20.0)** — MBRの+3.8ptの効果はLBでも確認
5. **しかしstarterのLB=26にすら及ばない** — 我々のモデルに根本的な問題がある

## 考察: exp011のCV/LB乖離の原因

1. **CVが「最初の1文」限定**: extract_first_sentenceで文書冒頭のみ評価。テストは文書中の任意位置の文。冒頭定型句はモデルが得意で、中盤以降は苦手の可能性
2. **追加データ（Sentences_Oare +1,166件）がテスト分布と不整合**: eda012で追加したデータの分布がテストと異なる可能性
3. **PN/GNタグの逆効果**: タグ付きで学習したがテストデータの語彙がOA_Lexiconに不十分にマッチ
4. **byt5-base vs byt5-small**: smallの方がLBが良い可能性？（過学習？）

## 次のステップ

- CV評価方法の改善（最初の1文ではなく、ランダムな文での評価）
- starterと同条件でbyt5-baseを試す（モデルサイズの影響切り分け）
- exp011の追加データの影響切り分け（追加データなしでbase学習）
- submit.pyの推論パイプラインの問題切り分け（タグの影響等）
