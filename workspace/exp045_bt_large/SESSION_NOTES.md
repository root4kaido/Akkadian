# SESSION_NOTES: exp045_bt_large

## セッション情報
- **日付**: 2026-03-23
- **作業フォルダ**: workspace/exp045_bt_large
- **目標**: byt5-largeでBT augment 2段階学習（BT 1/4サンプリング + pseudo pretrain → real finetune）

## 仮説
- byt5-largeの方がbaseより表現力が高く、BT augmentの恩恵も大きい
- largeモデルにはBT全量(62k)は過剰な可能性があり、1/4サンプリング(~15k)で十分
- pretrain lr=1e-4, finetune lr=2e-5（s1_exp007の知見を活用）

## 試したアプローチと結果

| アプローチ | 変更点 | CV (trainer eval) | LB | 備考 |
|-----------|--------|-----|-----|------|
| pretrain_ft fold3 | BT 1/4(~15k) + pseudo(6k) pretrain → real ft | sent-geo=40.03, doc-geo=27.87 | - | chrf=51.57/40.74, bleu=31.07/19.07, rep=8.3%/56.9% |

## ファイル構成
- `src/train_pretrain_ft.py` — 2段階学習スクリプト（large版）

## 重要な知見
- byt5-large (geo_mean=51.04) がbyt5-base (geo_mean=52.10) に負けている
- lr設定やBTサンプリング量の調整が必要かもしれない

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp044 (base) | BT 62k + pseudo 6k | sent-geo=40.59, doc-geo=29.56 | baseline |
| exp045 (large) | BT 1/4 + pseudo, lr 1e-4/2e-5 | sent-geo=40.03, doc-geo=27.87 | -0.56/-1.69 |

## コマンド履歴
```bash
# 学習 (fold 3)
python workspace/exp045_bt_large/src/train_pretrain_ft.py --fold 3
```

## 次のステップ
- exp044(base)との比較
- アンサンブル検討
