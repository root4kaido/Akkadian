# SESSION_NOTES: exp018_sent_augment

## セッション情報
- **日付**: 2026-03-11
- **作業フォルダ**: workspace/exp018_sent_augment
- **目標**: doc-levelデータ + 確率的開始位置シフトaugmentation

## 仮説
- eda019でfuzzy matchアライメント精度が2-6文で86-100%と確認
- 512B truncationで先頭部分しか使われない長文ドキュメントに対し、
  確率的に開始位置を文境界でずらすことで、後半部分のデータも学習に活用
- Akk/Eng両方向を同時にずらすので逆翻訳学習にも対応

## exp016からの変更点
| パラメータ | exp016 | exp018 |
|---|---|---|
| data | train.csv doc-level (全件) | train.csv doc-level (全件、**同じ**) |
| augmentation | なし | **確率0.5で6文以下のdocの開始位置を文境界にずらす** |
| val split | train.csvから10% | 同一（公平比較） |
| その他 | 全てexp016と同一 | 同一 |

## augmentationの詳細
- 6文以下 & sentence_aligned.csvにアライメントがあるドキュメントが対象
- 確率0.5で、ランダムに2文目〜最終文の開始位置にAkk入力をシフト
- 英語側も対応する文以降を結合
- 逆翻訳方向(Eng→Akk)も同様にシフト
- データ量はexp016と同じ（augmentationはデータ内容の変更のみ）

## 試したアプローチと結果

| アプローチ | 変更点 | CV(geo) | LB | 備考 |
|-----------|--------|---------|-----|------|
| 開始位置シフトaugment | prob=0.5, 文境界シフト | **45.37** (clean) | - | exp016比-0.41pt。微減。rep=28.0% (exp016=26.8%) |

## ファイル構成
- src/train.py — 学習スクリプト
- src/eval_cv.py — CV評価

## コマンド履歴
```bash
CUDA_VISIBLE_DEVICES=0 python workspace/exp018_sent_augment/src/train.py  # dev0
CUDA_VISIBLE_DEVICES=0 python workspace/exp018_sent_augment/src/eval_cv.py  # dev0
```

## 詳細スコア
- training eval: chrf=57.46, bleu=35.77, geo=45.33
- raw:   chrF++=53.69, BLEU=34.70, geo=43.16, rep=36.3%
- clean: chrF++=53.69, BLEU=38.33, geo=45.37, rep=28.0%

## 重要な知見
- 開始位置シフトaugmentationはCV微減(-0.41pt)。効果なし。
- 繰り返し率もexp016(26.8%)より悪化(28.0%)
- ずらしたデータは短くなるため、元のdoc-level全文を見る機会が減る副作用か

## 次のステップ
- exp019の結果と合わせて判断
