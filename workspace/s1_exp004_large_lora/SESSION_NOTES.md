# s1_exp004_large_lora

## 実験概要
- **日付**: 2026-03-16
- **親実験**: s1_exp002_large_lowlr
- **目的**: LoRAでパラメータ数を大幅削減し過学習を防ぐ
- **マシン**: A100

## 変更点
- LoRA: r=16, alpha=32, dropout=0.05, target_modules=[q, v]
- その他はs1_exp002と同一（epoch=5, lr=5e-5, byt5-large, fold3）

## 結果
- **eval metrics**:

## メモ
