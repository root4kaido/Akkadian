# SESSION_NOTES: exp020_doc_level_additional

## セッション情報
- **日付**: 2026-03-11
- **作業フォルダ**: workspace/exp020_doc_level_additional
- **目標**: 元doc-levelデータに、開始位置をずらしたdoc-levelデータを追加

## 仮説
- exp019では2文目以降を個別文ペアとして追加 → sent-CV=39.50で改善
- doc-levelペアの方が学習データの形式と一致し、文脈保持の観点でも有利
- 3文docなら (sent2+sent3)→(eng2+eng3), (sent3)→(eng3) の2件をdoc-levelで追加

## exp019からの変更点
| パラメータ | exp019 | exp020 |
|---|---|---|
| 追加データ形式 | 個別文ペア（sent_idx>=1の各文） | **doc-levelペア（各開始位置から末尾まで結合）** |
| val split | train.csvから10% | 同一（公平比較） |
| その他 | 全てexp016と同一 | 同一 |

## 試したアプローチと結果

| アプローチ | 変更点 | sent-CV(geo) | doc-CV(geo) | LB | 備考 |
|-----------|--------|-------------|-------------|-----|------|
| doc-levelずらし追加 | 各開始位置からdoc-level | 38.87 | 26.62 | **26.2** | rep=12.7%/59.9% |

## 訓練eval
- chrf=59.79, bleu=37.67, geo=47.46
- 旧CV (clean): geo=44.26, rep=35.7%

## 比較（ベースラインとの差分）
| 実験 | sent-CV | doc-CV | 旧CV |
|------|---------|--------|------|
| exp016 (base) | 36.95 | 25.41 | 45.78 |
| exp019 (sent追加) | 39.50 | 27.21 | 46.44 |
| **exp020 (docずらし)** | **38.87** | **26.62** | 44.26 |

- sent-CV: exp019 > exp020 > exp016。個別文ペア追加の方がdoc-levelずらしより効果的
- doc-CV: exp019 > exp020 > exp016。同傾向

## ファイル構成
- src/train.py — 学習スクリプト
- src/eval_cv.py — CV評価

## コマンド履歴
```bash
CUDA_VISIBLE_DEVICES=0 python workspace/exp020_doc_level_additional/src/train.py
```

## 次のステップ
- sent-CV / doc-CVでexp016, exp019と比較
