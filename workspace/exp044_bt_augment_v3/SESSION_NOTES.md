# SESSION_NOTES: exp044_bt_augment_v3

## セッション情報
- **日付**: 2026-03-22
- **作業フォルダ**: workspace/exp044_bt_augment_v3
- **目標**: generate_v3を使ったBT augment拡大（v2+v3マージ62k件）で2段階学習

## 仮説
- exp041(v2: 24k件)に対し、v3新規分(38k件)を追加した62k件のBTデータでpretrainすることで翻訳品質が向上する

## 試したアプローチと結果

| アプローチ | 変更点 | CV (sent-geo / doc-geo) | LB | 備考 |
|-----------|--------|-----|-----|------|
| pretrain_ft fold3 | BT 62k + pseudo 6k pretrain → real ft | sent-geo=40.59, doc-geo=29.56 | - | chrf=51.95/42.27, bleu=31.71/20.67, rep=7.3%/57.9% |
| fulldata_ft | 上記pretrained → train全体でft (val無し) | CVなし | - | 提出用モデル |

## ファイル構成
- `src/generate_backtranslation.py` — 逆翻訳スクリプト（v3新規分生成、v2とマージ）
- `src/train_pretrain_ft.py` — 2段階学習スクリプト
- `dataset/backtranslated.csv` — マージ済みBTデータ (62,655件)
- `dataset/backtranslated_v3_only.csv` — v3新規分のみ (38,683件)

## 重要な知見
- BT拡大(24k→62k)でexp041比 sent-geo +0.12改善 (40.47→40.59)
- Trainer eval: geo_mean 50.83→52.10 (+1.27) と大きく改善

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp041 (base) | BT 24k + pseudo 6k | sent-geo=40.47, doc-geo=29.18 | baseline |
| exp044 fold3 | BT 62k + pseudo 6k | sent-geo=40.59, doc-geo=29.56 | +0.12/+0.38 |

## コマンド履歴
```bash
# BT生成
python workspace/exp044_bt_augment_v3/src/generate_backtranslation.py
# 学習 (fold 3)
python workspace/exp044_bt_augment_v3/src/train_pretrain_ft.py --fold 3
# fulldata finetune
python workspace/exp044_bt_augment_v3/src/finetune_fulldata.py
```

## 次のステップ
- 全foldの学習・CV確認
- exp041との比較
