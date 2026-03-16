# SESSION_NOTES: exp033_st_filtered

## セッション情報
- **日付**: 2026-03-15
- **作業フォルダ**: workspace/exp033_st_filtered
- **目標**: pseudo dataの品質フィルタ+比率調整でself-trainingを改善

## 仮説
- exp032はpseudo:real=4:1で比率が偏りすぎ、ノイジーなpseudoに引っ張られた
- 品質フィルタ（短文除外）+ 1:1比率にすればpseudoの悪影響を抑えつつ恩恵を得られる

## 試したアプローチと結果

| アプローチ | 変更点 | sent-CV | doc-CV | 備考 |
|-----------|--------|---------|--------|------|
| exp023 baseline (fold3) | - | 35.44 | 25.52 | 比較対象 |
| exp032 (全pseudo) | +pseudo 6,360件 | 35.71 | 23.75 | doc悪化 |
| exp033 (filtered 1:1) | +pseudo ~1,264件 (>=20文字, 1:1) | 35.22 | 23.38 | sent-0.22, doc-2.14悪化。rep=59.6% |

## ファイル構成
- `src/train_gkf.py` — exp023ベース + filtered pseudo統合

## 重要な知見
- 品質フィルタ+ダウンサンプリングでもexp023を超えられず
- exp032（全pseudo）より更に悪化。pseudo dataの量を減らしても質の問題は解決しない
- self-trainingの混合学習アプローチは根本的に限界がある

## コマンド履歴
```bash
python workspace/exp033_st_filtered/src/train_gkf.py --fold 3
python eda/eda020_sent_level_cv/eval_full_doc.py workspace/exp033_st_filtered/results/fold3/last_model exp033_st_filtered --preprocess exp023 --fold 3
```

## 次のステップ
- 棄却。混合学習でのself-trainingは効果なし
