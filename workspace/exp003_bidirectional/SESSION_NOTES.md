# SESSION_NOTES: exp003_bidirectional

## セッション情報
- **日付**: 2026-03-07
- **作業フォルダ**: workspace/exp003_bidirectional
- **目標**: Starter完全再現（双方向学習ON + 最終epochモデル使用）でgeo_mean ~30を目指す

## 仮説

exp002(単方向, geo_mean=21.16)に対し、Starterと同じ双方向学習ONにすればgeo_mean ~25-30に到達するはず。

## 試したアプローチと結果

| アプローチ | 変更点 | CV (inference greedy) | CV (beam4) | LB | 備考 |
|-----------|--------|----------------------|------------|-----|------|
| exp003 | 双方向学習ON + 最終epoch使用 | **17.87** | 14.37 | - | training eval(23.55)はtruncation水増し |

※ training eval の23.55は参照テキストが512バイト切り詰めにより水増し。inference greedy(17.87)が正確なCV。
※ exp001/exp002のtraining eval CVも同様に水増しされていた可能性が高い。

## Epoch別スコア推移

| Epoch | eval_loss | chrF | BLEU | geo_mean |
|-------|-----------|------|------|----------|
| 1 | 2.5671 | 7.84 | 0.51 | 2.00 |
| 2 | 2.3954 | 14.27 | 2.45 | 5.91 |
| 3 | 2.3308 | 20.83 | 4.55 | 9.73 |
| 4 | 2.2837 | 26.43 | 6.86 | 13.47 |
| 5 | 2.2560 | 27.96 | 8.37 | 15.29 |
| 6 | 2.2291 | 29.60 | 9.66 | 16.91 |
| 7 | 2.2092 | 30.49 | 10.45 | 17.85 |
| 8 | 2.1984 | 30.74 | 11.13 | 18.50 |
| 9 | 2.1911 | 31.77 | 12.60 | 20.01 |
| 10 | 2.1808 | 31.95 | 12.16 | 19.71 |
| 11 | 2.1730 | 32.58 | 13.38 | 20.88 |
| 12 | 2.1668 | 33.22 | 13.56 | 21.23 |
| 13 | 2.1637 | 33.59 | 14.07 | 21.74 |
| 14 | 2.1608 | 34.15 | 14.60 | 22.33 |
| 15 | 2.1564 | 34.29 | 14.72 | 22.46 |
| 16 | 2.1530 | 34.64 | 15.25 | 22.99 |
| 17 | 2.1524 | 34.82 | 15.47 | 23.20 |
| 18 | 2.1502 | 35.53 | 16.14 | **23.95** |
| 19 | 2.1475 | 35.32 | 16.62 | 24.22 |
| 20 | 2.1476 | 34.79 | 15.94 | 23.55 |

**最終epoch (20) モデル使用**: greedy CV geo_mean=23.55

注意: epoch 19 がピーク(24.22)だが、Starterに合わせて最終epoch(20)のモデルを使用。
load_best_model_at_end=Trueにすれば+0.67pt改善の余地あり。

## デバッグ結果

### Training eval vs Inference eval の乖離

training evalのcompute_metricsでは、参照テキストがtokenize時にmax_length=512バイトで切り詰められる。
ByT5はバイトレベルなので512トークン=512バイト≒512文字。長い参照テキスト（66/157=42%が影響）が短くなった状態で比較するため、training evalスコアが水増しされていた。

| 評価方法 | chrF | BLEU | geo_mean | 備考 |
|---------|------|------|----------|------|
| training eval (greedy) | 34.79 | 15.94 | 23.55 | 参照truncationで水増し |
| vs roundtrip ref | 32.10 | 13.12 | 20.52 | truncation再現で近い値 |
| **inference greedy (正確)** | **27.85** | **11.46** | **17.87** | **正確なCV** |
| inference beam4 | 25.62 | 8.06 | 14.37 | 繰り返し劣化 |
| inference beam4 + postprocess | 25.02 | 6.96 | 13.20 | postprocessも微減 |

### Beam search劣化

ByT5でbeam search(num_beams=4)はgreedy(17.87)より3.5pt低い(14.37)。
長文サンプルで "its import duty added," 等の無限反復が発生。
no_repeat_ngram_size=3はByT5バイトレベルでは3バイト=1-2文字の制約となり壊滅的(3.23)。

Starterもbeam4を使うが、**テストデータは文レベル（短い）ので影響小**。valデータは文書レベル（長い）のため影響大。

## 最終評価

- eval_loss: 2.1476
- eval_chrf: 34.79
- eval_bleu: 15.94
- eval_geo_mean: **23.55** (greedy)
- 学習時間: 2h36m

## ファイル構成

- `src/preprocess.py` — 双方向データ作成対応
- `src/train.py` — 双方向学習 + save_total_limit=1 + MetricsLoggerコールバック
- `src/postprocess.py` — exp002から流用
- `src/infer.py` — val CV評価 + test推論
- `src/plot_metrics.py` — 学習曲線可視化
- `results/best_model/` — 最終epochモデル
- `results/eval_metrics.yaml` — greedy eval指標
- `results/cv_beam_metrics.yaml` — beam search CV指標
- `results/val_predictions.csv` — val予測結果
- `results/submission.csv` — テスト予測結果
- `results/metrics_log.json` — 学習メトリクスログ

## 重要な知見

- **training evalのCVスコアは水増し**: ByT5の512バイトtruncationにより参照テキストが切り詰められ、実際より高いスコアが出る
- **正確なCV(inference greedy)は17.87**: training eval(23.55)との差は5.68pt
- **beam search(14.37)はgreedy(17.87)より3.5pt低い**: ByT5の長文繰り返し問題
- **no_repeat_ngram_sizeはByT5に使えない**: バイトレベルのため3-gram=1-2文字で壊滅的
- **テストデータは短い文レベル**なので、beam search劣化の影響はval(文書レベル)より小さいはず
- exp001/002のtraining eval CVも同様に水増しの可能性高い

## コマンド履歴
```bash
cd /home/user/work/Akkadian && python workspace/exp003_bidirectional/src/train.py
python workspace/exp003_bidirectional/src/infer.py
```

## 次のステップ
- [ ] no_repeat_ngram_size追加でビームサーチ改善
- [ ] load_best_model_at_end=Trueに戻す（+0.67pt）
- [ ] wandb対応
- [ ] Sentences_Oare.csv追加データ
