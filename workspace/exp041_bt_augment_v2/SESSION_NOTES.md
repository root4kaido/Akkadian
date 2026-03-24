# SESSION_NOTES: exp041_bt_augment_v2

## セッション情報
- **日付**: 2026-03-20
- **作業フォルダ**: workspace/exp041_bt_augment_v2
- **目標**: generate_v2 BT(23k) + pseudo_labels_v2(6k) pretrain 5ep → real finetune 5ep（exp034式2段階学習）

## 仮説
- exp034の2段階学習（pseudo pretrain→real finetune）がLBベスト
- generate_v2のBT 23k件 + published_texts pseudo 6k件で合計約30kのpretrain
- exp038(BT混合)より2段階の方が効果的な可能性

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| BT 23k+pseudo 6k pretrain 5ep → real ft 5ep (lr=5e-5) | exp034式2段階 | sent-geo=40.71, doc-geo=28.28 | **33.4** | **LBベスト更新**。rep=59.3%(doc), 8.5%(sent) |

## ファイル構成
<!-- 作成したスクリプト、可視化結果、データファイル -->

## 重要な知見
- **LB 33.4（単体モデルでLBベスト更新）**: 前回ベストの2モデルアンサンブルrt_weighted(32.6)を単体で超えた
- **pretrain→ft が混合学習より汎化性能高い**: exp042(混合)はCV同等(geo≒28.5)だがLB 30.9で大きく劣る
- **BT拡大(2k→23k)の効果**: exp038(BT 2k, LB未提出)→exp041(BT 23k, LB 33.4)でCV/LB両軸大幅改善

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp023 (baseline) | - | sent-geo=35.44, doc-geo=25.52 | - |
| exp038 (BT 2k件) | generated_english BT augment | sent-geo=37.22, doc-geo=26.64 | sent+1.78, doc+1.12 |
| **exp041 (BT 23k 2段階)** | pretrain 5ep + ft 5ep (lr=5e-5) | sent-geo=40.71, doc-geo=28.28 | sent+5.27, doc+2.76 |

## コマンド履歴
```bash
# BT生成 (dev0)
python workspace/exp041_bt_augment_v2/src/generate_backtranslation.py 2>&1 | tee workspace/exp041_bt_augment_v2/results/generate_backtranslation.log

# 学習: 2段階 pretrain→ft (dev0)
python workspace/exp041_bt_augment_v2/src/train_pretrain_ft.py --fold 3 2>&1 | tee workspace/exp041_bt_augment_v2/results/fold3/pretrain_ft/train.log
```

## 次のステップ
<!-- 結果を受けて、次の実験の候補や残課題など -->
