# s1_exp006_large_mixout

## 実験概要
- **日付**: 2026-03-16
- **親実験**: s1_exp002_large_lowlr
- **目的**: Mixoutでpretrained weightsに確率的リセットし過学習を抑制
- **マシン**: A100

## 変更点
- Mixout(p=0.5): 全Linearレイヤーに適用。学習中に50%の確率でpretrained weightにリセット
- その他はs1_exp002と同一（epoch=5, lr=5e-5, byt5-large, fold3）

## 結果
- **eval metrics**:

## メモ
