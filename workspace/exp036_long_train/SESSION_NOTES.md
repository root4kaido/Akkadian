# SESSION_NOTES: exp036_long_train

## セッション情報
- **日付**: 2026-03-16
- **作業フォルダ**: workspace/exp036_long_train
- **目標**: exp023と同一設定でepoch数を100に増やし、長時間学習の効果を検証

## 仮説
- exp023は20epochで学習。ByT5-baseはbyte-levelで収束が遅い可能性がある
- epoch数を5倍にすることでさらなる改善が得られるか検証

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| exp023 fold3 (20ep) | baseline | sent=34.35, doc=24.87 | 30.03 | 比較対象 |
| exp036 fold3 (100ep) | epoch 20→100 | sent=35.99, doc=25.32 | - | sent +0.55, doc -0.20 |

## ファイル構成
- `src/train_gkf.py` — exp023のtrain_gkf.pyベース、EPOCHS=100のみ変更

## 重要な知見
- 100epochでもsent-CV +0.55pt、doc-CV -0.20ptとほぼ変わらず。20epochで十分収束している
- last_model eval_geo_mean: 20ep=43.73 → 100ep=44.55（+0.82pt）だが実評価では微差
- rep率: sent=13.9%, doc=69.4%。長時間学習でも繰り返し問題は解消されない

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp023 fold3 | 20 epoch | best geo_mean=43.84 | baseline |
| exp036 fold3 | 100 epoch | last eval_geo=44.55, best=43.84 | sent +0.55, doc -0.20 |

## コマンド履歴
```bash
python workspace/exp036_long_train/src/train_gkf.py --fold 3
python eda/eda020_sent_level_cv/eval_full_doc.py workspace/exp036_long_train/results/fold3/best_model exp036_long_train --preprocess exp023 --fold 3
```

## 次のステップ
