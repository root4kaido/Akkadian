# s1_exp005_large_freeze

## 実験概要
- **日付**: 2026-03-16
- **親実験**: s1_exp002_large_lowlr
- **目的**: encoder+decoder下層凍結でpretrained汎化能力を保持しつつタスク適応
- **マシン**: A100

## 変更点
- encoder全体を凍結
- decoder下位12層（block[0:11]）を凍結
- decoder上位12層（block[12:23]） + lm_head のみ学習
- その他はs1_exp002と同一（epoch=5, lr=5e-5, byt5-large, fold3）

## 結果
- **eval metrics**:

## メモ
