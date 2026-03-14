# SESSION_NOTES: exp017_additional_data

## セッション情報
- **日付**: 2026-03-10
- **作業フォルダ**: workspace/exp017_additional_data
- **目標**: 追加データのみの効果を分離測定（exp014ベース + 追加データのみ）

## 仮説
- exp014(ByT5-small, v6設定)のCV=39.89, LB=26.9
- exp015(ByT5-base + タグ + 追加データ)のCV=43.26, LB=27.3
- 3つの変更(モデル大/タグ/追加データ)のうち、追加データの効果を切り分ける
- 追加データ(+1,165件)でCV改善が見られるか？

## exp014からの変更点
| パラメータ | exp014 | exp017 |
|---|---|---|
| data | train.csvのみ (1,561件) | **train.csv + additional_train.csv (+1,165件)** |
| val split | train.csvのみから分割 | 同一（公平な評価のため） |
| その他 | 全てexp014と同一 | 同一 |

## 試したアプローチと結果

| アプローチ | 変更点 | CV(geo) | LB | 備考 |
|-----------|--------|---------|-----|------|
| 追加データ only | train.csvに追加データ結合 | 42.20 (clean) | **25.4** | CV+2.31ptだがLB-1.5pt。追加データはLBで逆効果 |

## ファイル構成
- src/train.py — 学習スクリプト (exp014ベース、追加データ読み込みのみ変更)
- src/eval_cv.py — CV評価
- src/submit.py — Kaggle提出用推論
- dataset/additional_train.csv — 追加データ (exp011から)

## 重要な知見

### アブレーション結果
- 追加データ単体でCV=42.20 (exp014比+2.31pt)
- ByT5-base(exp016, +5.89pt)と比較して効果は半分以下
- rep率: raw=36.9%, clean=29.9%（exp014よりやや悪化）

### 詳細スコア
- raw:   chrF++=51.35, BLEU=34.05, geo=41.81, rep=36.9%
- clean: chrF++=51.19, BLEU=34.79, geo=42.20, rep=29.9%
- training eval: chrF=54.07, BLEU=31.25, geo=41.10

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp014 | ベースライン (ByT5-small, v6) | geo=39.89 | - |
| exp017 | + 追加データ (+1,165件) | geo=42.20 (clean) | +2.31pt |

## コマンド履歴
```bash
CUDA_VISIBLE_DEVICES=1 python workspace/exp017_additional_data/src/train.py  # dev1
CUDA_VISIBLE_DEVICES=1 python workspace/exp017_additional_data/src/eval_cv.py  # dev1
```

## 次のステップ
- LB提出で実際のスコアを確認
- exp016との比較でどちらの効果がLBに反映されるか確認
