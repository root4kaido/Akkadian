# SESSION_NOTES: exp016_byt5_base

## セッション情報
- **日付**: 2026-03-10
- **作業フォルダ**: workspace/exp016_byt5_base
- **目標**: ByT5-baseのみの効果を分離測定（exp014ベース + モデルサイズ変更のみ）

## 仮説
- exp014(ByT5-small, v6設定)のCV=39.89, LB=26.9
- exp015(ByT5-base + タグ + 追加データ)のCV=43.26, LB=27.3
- 3つの変更(モデル大/タグ/追加データ)のうち、モデルサイズの効果を切り分ける
- ByT5-base単体でCV改善が見られるか？

## exp014からの変更点
| パラメータ | exp014 | exp016 |
|---|---|---|
| model | google/byt5-small | **google/byt5-base** |
| precision | FP32 | **BF16** (メモリ制約) |
| その他 | 全てexp014と同一 | 同一 |

## 試したアプローチと結果

| アプローチ | 変更点 | CV(geo) | LB | 備考 |
|-----------|--------|---------|-----|------|
| ByT5-base only | モデルサイズ変更のみ | 45.78 (clean) | **29.5** | **LBベスト**。exp014比 CV+5.89pt, LB+2.6pt |

## ファイル構成
- src/train.py — 学習スクリプト (exp014ベース、MODEL_NAME + BF16のみ変更)
- src/eval_cv.py — CV評価
- src/submit.py — Kaggle提出用推論

## 重要な知見

### アブレーション結果
- ByT5-base単体でCV=45.78 (exp014比+5.89pt)
- **exp015(全部入り、43.26)よりexp016(base only、45.78)の方がスコアが高い**
- → タグ・追加データがByT5-baseと組み合わせた時に逆効果になっている可能性
- rep率: raw=32.5%, clean=26.8%（exp014と同等水準）

### 詳細スコア
- raw:   chrF++=54.63, BLEU=35.09, geo=43.78, rep=32.5%
- clean: chrF++=54.61, BLEU=38.38, geo=45.78, rep=26.8%
- training eval: chrF=58.95, BLEU=37.04, geo=46.73

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp014 | ベースライン (ByT5-small, v6) | geo=39.89 | - |
| exp016 | ByT5-base + BF16 | geo=45.78 (clean) | +5.89pt |

## コマンド履歴
```bash
CUDA_VISIBLE_DEVICES=0 python workspace/exp016_byt5_base/src/train.py  # dev0
CUDA_VISIBLE_DEVICES=0 python workspace/exp016_byt5_base/src/eval_cv.py  # dev0
```

## 次のステップ
- LB提出で実際のスコアを確認
- exp015よりLBが高ければ、タグ・追加データは不要と判断
