# SESSION_NOTES: exp032_self_training

## セッション情報
- **日付**: 2026-03-14
- **作業フォルダ**: workspace/exp032_self_training
- **目標**: Self-Trainingで擬似並列データを作成し学習データを5倍に増やす

## 仮説
- exp025-031で教師あり/教師なしの工夫が全て失敗
- 根本原因: 1,561件の並列コーパスが少なすぎる
- Self-training: exp023モデルでpublished_textsを英訳し擬似並列データ~6,400件を作成
- real 1,561 + pseudo 6,400 = ~8,000件で再学習

## 試したアプローチと結果

| アプローチ | 変更点 | sent-CV | doc-CV | 備考 |
|-----------|--------|---------|--------|------|
| exp023 baseline (fold3) | - | 35.44 | 25.52 | 比較対象 |
| Self-training (last) | +pseudo 6,360件 | 35.71 | 23.75 | sent微増, doc悪化, rep=62.0% |

## ファイル構成
- `src/generate_pseudo.py` — 擬似ラベル生成（fold3 last_model単体）
- `src/train_gkf.py` — 再学習（exp023ベース + pseudo data統合）
- `results/pseudo_labels.csv` — 擬似ラベル

## 重要な知見
- pseudo 6,360件を追加（real 1,264 + pseudo 6,360 = bidirectional合計 15,248で学習）
- training eval geo_mean=45.08（exp023の43.73より+1.35）と改善したが...
- 実際の生成評価ではsent-CVほぼ横ばい(+0.27)、doc-CV悪化(-1.77)
- pseudo dataのノイズが長文（doc）生成に悪影響。rep率62.0%も依然高い
- training metricsと生成評価の乖離が大きい（過学習の兆候）

## コマンド履歴
```bash
# Step 1: 擬似ラベル生成
python workspace/exp032_self_training/src/generate_pseudo.py

# Step 2: 再学習 (fold3)
python workspace/exp032_self_training/src/train_gkf.py --fold 3

# Step 3: 評価
python eda/eda020_sent_level_cv/eval_full_doc.py workspace/exp032_self_training/results/fold3/last_model exp032_self_training --preprocess exp023 --fold 3
```

## 次のステップ
- 現状では効果不十分。改善の方向性:
  - pseudoデータの品質フィルタリング（短文のみ、confidence高いもののみ）
  - pseudo/realの比率調整（現在4:1 → 1:1にダウンサンプリング）
  - pseudo行のloss weightを下げる
  - 別の方向性を検討
