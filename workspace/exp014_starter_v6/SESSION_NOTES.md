# SESSION_NOTES: exp014_starter_v6

## セッション情報
- **日付**: 2026-03-10
- **作業フォルダ**: workspace/exp014_starter_v6
- **目標**: Starterノートブック v6(ピン留め版, LB=26)を正確に再現する

## 仮説
- exp013はlatestバージョン(lr=1e-4, label_smoothing=0.2)で学習 → LB=21.6
- v6(ピン留め版)はlr=2e-4, label_smoothing=0, batch=4×accum=2
- v6設定で再学習すればLB=26を再現できるはず

## v6 vs latest の差分

| パラメータ | v6 (ピン留め) | latest |
|---|---|---|
| LEARNING_RATE | 2e-4 | 1e-4 |
| label_smoothing | なし | 0.2 |
| batch_size | 4 | 1 |
| grad_accum | 2 | 8 |
| effective batch | 8 | 8 |

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| starter v6再現 | lr=2e-4, no label_smoothing | 39.89 | 26.9 | CV/LB比=1.48。starter LB=26に近い |

## ファイル構成
- src/train.py — 学習スクリプト (v6設定)
- src/eval_cv.py — CV評価 (eda017と同一条件)
- src/submit.py — Kaggle提出用推論スクリプト

## 重要な知見

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp014 | starter v6 (lr=2e-4, LS=0) | CV=39.89, LB=26.9 | vs exp013: CV+7.85, LB+5.3 |

## コマンド履歴
```bash
# 学習
python workspace/exp014_starter_v6/src/train.py
# CV評価
python workspace/exp014_starter_v6/src/eval_cv.py
```

## 次のステップ
- LB=26を再現確認
- 再現できたら、ここをベースにアブレーション開始
