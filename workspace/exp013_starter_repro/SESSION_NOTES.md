# SESSION_NOTES: exp013_starter_repro

## セッション情報
- **日付**: 2026-03-10
- **作業フォルダ**: workspace/exp013_starter_repro
- **目標**: Starterノートブック(LB=26)を完全再現し、今後のアブレーションのベースラインとする

## 仮説
- Starter (byt5-small, train.csv only, beam4) のLB=26を再現できるはず
- eda017で同条件のCV=32.04を確認済み
- 今後、ここに1つずつ改善を加えてLBで効果検証する

## 注意: starterのlatestバージョンで学習した
- ピン留め版(v6)はlr=2e-4, label_smoothing=0, batch=4×accum=2
- latestはlr=1e-4, label_smoothing=0.2, batch=1×accum=8
- **exp013はlatestで学習してしまった** → exp014でv6を再現

## Starterとの差分（インフラのみ）
- wandb logging追加
- best model + last model 両方保存
- generation_max_length=512追加（starterのバグ修正、学習には影響なし）
- batch_size=2 + grad_accum=4（effective=8はstarterと同一）

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| starter再現(latest版) | lr=1e-4, label_smoothing=0.2 | 32.04 | 21.6 | ピン留め版(v6)とは設定が異なっていた |

## ファイル構成
- src/train.py — 学習スクリプト
- src/submit.py — Kaggle提出用推論スクリプト

## 重要な知見

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp013 | starter latest版 (lr=1e-4, LS=0.2) | CV=32.04, LB=21.6 | CV/LB比=1.48 |

## コマンド履歴
```bash
# 学習
python workspace/exp013_starter_repro/src/train.py
```

## 次のステップ
- LB=26を再現確認
- 再現できたら、MBR追加/追加データ追加のアブレーション
