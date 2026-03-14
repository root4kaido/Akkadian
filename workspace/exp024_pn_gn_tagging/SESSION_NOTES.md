# SESSION_NOTES: exp024_pn_gn_tagging

## セッション情報
- **日付**: 2026-03-12
- **作業フォルダ**: workspace/exp024_pn_gn_tagging
- **目標**: exp023(全前処理) + PN/GNタグ付加で固有名詞の翻訳精度を改善

## 仮説
exp023のval予測で固有名詞の25.5%が不正解（648名中165名ミス）。
exp010ではPN/GNタグでgreedy_clean=29.92（当時ベスト）と繰り返し率改善を確認。
ByT5-base + 全前処理ベースにPN/GNタグを再適用すれば、固有名詞精度が向上しLBも改善を期待。

## 親実験: exp023_full_preprocessing
- CV: 46.85, LB: 30.03
- eda022全改善（近似小数、月名→番号、regex修正、gap重複、(ki)対応）

## exp023からの変更点
1. OA_Lexicon form_type_dict.jsonでtransliterationのトークンにPN/GNタグ付加
2. submit.pyの入力前処理にもタグ付加を追加

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| exp024 | exp023 + PN/GNタグ | training=47.52, sent-CV=44.62 | - | exp023比-2.23pt悪化。棄却 |

## ファイル構成
- `src/train.py` — exp023ベース + PN/GNタグ付加
- `src/eval_cv.py` — 同上
- `src/submit.py` — タグ付加対応
- `dataset/form_type_dict.json` — exp010から流用

## 重要な知見
- PN/GNタグ付加はexp010（lr=5e-5, LS=0.2, cosine）では有効だったが、exp023ベース（lr=2e-4, LS=0, linear）では悪化
- 入力長増加（平均+23B, 512B超え614→649件）によるtruncation悪化が一因の可能性
- eda023でPN→英語名のルールベース置換も調査したが、辞書カバレッジ42%で実現困難

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp023 | 全前処理 | CV=46.85, LB=30.03 | baseline |
| exp024 | + PN/GNタグ | training=47.52, sent-CV=44.62 | **-2.23pt悪化** |

## コマンド履歴
```bash
# 学習（dev0で実行中）
tmux send-keys -t dev0 'python workspace/exp024_pn_gn_tagging/src/train.py' Enter
# eval_cv
tmux send-keys -t dev1 'python workspace/exp024_pn_gn_tagging/src/eval_cv.py' Enter
```

## 次のステップ
