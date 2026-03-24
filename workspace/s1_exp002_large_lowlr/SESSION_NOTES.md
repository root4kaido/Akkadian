# s1_exp002_large_lowlr

## 実験概要
- **日付**: 2026-03-16
- **親実験**: s1_exp001_byt5_large_fold3
- **目的**: byt5-largeの過学習対策 — epoch削減 + 学習率低下
- **マシン**: A100

## 変更点
- epochs: 20 → 5
- lr: 2e-4 → 5e-5
- その他はs1_exp001と同一（byt5-large, fold3, exp023前処理）

## 結果
- **eval metrics**:

## メモ
