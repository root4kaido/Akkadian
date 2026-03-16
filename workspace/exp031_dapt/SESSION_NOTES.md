# SESSION_NOTES: exp031_dapt

## セッション情報
- **日付**: 2026-03-14
- **作業フォルダ**: workspace/exp031_dapt
- **目標**: Domain-Adaptive Pre-Trainingでモデルの学習精度を改善する

## 仮説
- exp025-029で教師あり学習の工夫（正則化・augmentation・label smoothing）が全て失敗した
- 原因: 1,561件の小データでは教師ありの工夫が効かない
- DAPTは教師なしでByT5をアッカド語ドメインに適応させる
- published_textsの7,953件（train含む、5x増）でspan corruption事前学習
- ByT5はmC4で学習されておりアッカド語翻字は未見 → ドメイン適応の余地が大きい

## 試したアプローチと結果

| アプローチ | 変更点 | sent-CV | doc-CV | 備考 |
|-----------|--------|---------|--------|------|
| exp023 baseline (fold3) | - | 35.44 | 25.52 | 比較対象 |
| DAPT + fine-tune (last) | +DAPT事前学習 10ep | 32.75 | 20.59 | **悪化** rep=67.3% |

## ファイル構成
- `src/dapt_pretrain.py` — DAPT事前学習（span corruption）
- `src/finetune.py` — DAPT済みモデルでexp023相当のfine-tune
- `results/dapt_model/` — DAPT済みモデル
- `results/fold3/` — fine-tune結果

## 重要な知見
- **DAPTは失敗**: sent-CV 32.75 (-2.69)、doc-CV 20.59 (-4.93)
- DAPT事前学習自体はeval_loss 0.871→0.609と順調に収束（10ep, 38min）
- fine-tune時のtraining eval_geoも41.20まで到達（exp023は43.73）
- しかし実際の生成評価では大幅悪化。rep率67.3%（繰り返し生成が多い）
- 原因候補: (1) 7,953件はDAPTには少なすぎ過学習 (2) span corruptionでByT5の翻訳能力が損なわれた (3) best checkpoint消失（save_total_limit=2）

## コマンド履歴
```bash
# Step 1: DAPT事前学習
python workspace/exp031_dapt/src/dapt_pretrain.py

# Step 2: fine-tune (fold3)
python workspace/exp031_dapt/src/finetune.py --fold 3

# Step 3: 評価
python eda/eda020_sent_level_cv/eval_full_doc.py workspace/exp031_dapt/results/fold3/last_model exp031_dapt --preprocess exp023 --fold 3
```

## 次のステップ
- DAPTは失敗。この方向は打ち切り
- 別アプローチを検討: ByT5-large、Retrieval-Augmented Translation、外部データ活用など
