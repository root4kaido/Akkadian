# SESSION_NOTES: exp019_sent_additional

## セッション情報
- **日付**: 2026-03-11
- **作業フォルダ**: workspace/exp019_sent_additional
- **目標**: 元doc-levelデータに文レベル分割データの2文目以降を追加してデータ量増加

## 仮説
- 元doc-levelデータでは512B超のドキュメントが34.7%truncationされている
- 文分割の2文目以降を追加することで、truncationで失われていた後半部分のデータを回収
- train.csv自体の分割なのでドメインミスマッチリスクなし（exp017の教訓）
- 1文目は元docのtruncation内に含まれるため重複を避けて除外

## exp016からの変更点
| パラメータ | exp016 | exp019 |
|---|---|---|
| data | train.csv doc-level のみ | **train.csv + sentence_aligned.csv (sent_idx>=1)** |
| val split | train.csvから10% | 同一（公平比較） |
| その他 | 全てexp016と同一 | 同一 |

## 試したアプローチと結果

| アプローチ | 変更点 | CV(geo) | LB | 備考 |
|-----------|--------|---------|-----|------|
| 分割データ追加 (2文目以降) | sentence_aligned.csv sent_idx>=1 | **46.44** (clean) | **28.7** | exp016比+0.66pt。rep=29.3% (exp016=26.8%) |

## ファイル構成
- src/train.py — 学習スクリプト
- src/eval_cv.py — CV評価

## アライメント前処理結果
- 全2,311文ペアのうち sent_idx >= 1: 推定~1,643件
- val splitのドキュメントは除外済み

## 詳細スコア
- training eval: chrf=60.88, bleu=38.62, geo=48.49
- raw:   chrF++=55.76, BLEU=37.72, geo=45.86, rep=31.8%
- clean: chrF++=55.46, BLEU=38.89, geo=46.44, rep=29.3%

## 重要な知見
- 2文目以降の分割データ追加でCV +0.66pt改善（45.78→46.44）
- ただし繰り返し率がexp016(26.8%)より悪化(29.3%)
- データ量増加（+~1,643件）の効果はあるが、繰り返し問題が顕在化

## コマンド履歴
```bash
CUDA_VISIBLE_DEVICES=0 python workspace/exp019_sent_additional/src/train.py  # dev1
CUDA_VISIBLE_DEVICES=0 python workspace/exp019_sent_additional/src/eval_cv.py  # dev1
```

## 次のステップ
- LB提出して実際のスコアを確認
- 繰り返し抑制と組み合わせる可能性
