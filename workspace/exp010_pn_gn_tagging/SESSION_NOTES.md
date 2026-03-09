# SESSION_NOTES: exp010_pn_gn_tagging

## セッション情報
- **日付**: 2026-03-08
- **作業フォルダ**: workspace/exp010_pn_gn_tagging
- **目標**: OA_Lexiconを用いたPN/GNタグ付加で固有名詞翻訳を改善

## 仮説
- Hostが「固有名詞が最大のボトルネック」と明言、Gutherz 2023でも固有名詞誤りが主因
- ByT5はバイトレベルなので、翻字トークンが人名か一般語か区別する手がかりがない
- OA_Lexiconのform→typeマッピングで、PN/GNトークンに[PN]/[GN]タグを付加
- タグによりモデルが「これは音写すべき固有名詞」と学習できる
- OA_Lexiconカバレッジ: train PN 2,165/11,583トークン、GN 107トークン
- 曖昧なform（PN+word両方に該当する488件）はタグ付加しない（ambiguous_strategy=skip）

## OA_Lexiconの統計
- 全39,332エントリ: word 25,574 / PN 13,424 / GN 334
- ユニークform: 35,049
- PN only: 12,640 / GN only: 310 / word only: 21,611 / Mixed: 488
- Train tokens matching PN: 2,165 / GN: 107 / Any: 8,111/11,583

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| PN/GNタグ付加 | OA_Lexiconでtransliterationにタグ付加 | greedy=28.96, MBR=28.57, greedy_clean=**29.92** | - | greedy_cleanがベスト。繰り返し大幅改善 |

## ファイル構成
- `src/build_lexicon.py` — OA_Lexiconからform→type辞書を構築しJSONで保存
- `src/preprocess.py` — PN/GNタグ付加付きデータ準備
- `src/train.py` — exp008ベース
- `src/eval_sentence_level.py` — 文レベル推論・評価（タグ付加対応）

## 重要な知見

- PN/GNタグ付加により**greedy_clean=29.92**（全実験ベスト）
- **繰り返し率が大幅改善**: greedy 54.8%→50.3%、MBR 51.0%→41.4%（exp008比）
- greedy系が大きく改善（+0.82pt raw, +2.49pt clean）、MBR rawは微減(-0.37pt)
- MBRが効かないのは、タグ付加によりgreedy出力が既に安定しているため
- ベストモデル: checkpoint-3344 (epoch 19)、training eval geo_mean=25.55

## 性能変化の記録

| 指標 | exp008 | exp010 | 差分 |
|------|--------|--------|------|
| greedy_raw | 28.14 | 28.96 | **+0.82** |
| mbr_raw | 28.94 | 28.57 | -0.37 |
| greedy_clean | ~27.43 | **29.92** | **+2.49** |
| mbr_clean | - | 28.57 | - |
| repetition (greedy) | 54.8% | 50.3% | -4.5pt |
| repetition (mbr) | 51.0% | **41.4%** | **-9.6pt** |
| training eval geo_mean | 24.33 | 25.55 | +1.22 |

## コマンド履歴
```bash
# 辞書構築
python workspace/exp010_pn_gn_tagging/src/build_lexicon.py

# 学習（dev0, 約3時間）
python workspace/exp010_pn_gn_tagging/src/train.py

# 文レベル評価
python workspace/exp010_pn_gn_tagging/src/eval_sentence_level.py
```

## 次のステップ
- [x] 辞書構築
- [x] 学習実行（best: checkpoint-3344, epoch 19）
- [x] sent-level評価（greedy_clean=29.92がベスト）
