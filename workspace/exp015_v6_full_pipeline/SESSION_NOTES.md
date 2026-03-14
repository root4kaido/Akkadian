# SESSION_NOTES: exp015_v6_full_pipeline

## セッション情報
- **日付**: 2026-03-10
- **作業フォルダ**: workspace/exp015_v6_full_pipeline
- **目標**: exp014(v6設定)にexp011の全改善を統合し、LBスコアを最大化

## 仮説
- exp014のv6設定(lr=2e-4, LS=0)がexp011の改善群と相乗効果を発揮するはず
- exp011はlr=5e-5, LS=0.2で学習していた → v6設定に変更で大幅改善期待
- exp011 CV=33.45 × (exp014/exp013比 39.89/32.04 = 1.245) ≒ CV=41.6が期待値
- LB ≈ CV × 0.675 ≒ 28前後を期待

## exp011からの変更点
| パラメータ | exp011 | exp015 |
|---|---|---|
| learning_rate | 5e-5 | **2e-4** |
| label_smoothing | 0.2 | **0** |
| batch_size × accum | 1×16 | **2×4** |
| effective batch | 16 | **8** |

## exp011から継承する要素
- ByT5-base (google/byt5-base)
- PN/GNタグ付加
- 追加データ (Sentences_Oare + published_texts, +1,165件)
- Cosine scheduler + warmup 10%
- BF16
- 双方向学習
- ラベルマスキング (翻訳文を512バイトでtruncate)

## 試したアプローチと結果

| アプローチ | 変更点 | CV(geo) | LB | 備考 |
|-----------|--------|---------|-----|------|
| v6 full pipeline | exp011 + v6設定 | 43.26 | 27.3 | CV/LB比=0.63。exp014(26.9)から+0.4ptのみ |

## ファイル構成
- src/train.py — 学習スクリプト (exp011ベース + v6設定)
- src/preprocess.py — 前処理 (exp011と同一)
- src/eval_cv.py — CV評価
- src/submit.py — Kaggle提出用推論

## 重要な知見

### Training eval vs eval_cv の乖離分析

training eval=48.44 vs eval_cv=43.26 の5pt差を徹底調査。因子分解の結果:

| 因子 | 寄与 |
|------|------|
| メトリクス: chrF(word_order=0) vs chrF++(word_order=2) | +0.66 |
| デコーディング: beam4→greedy | -2.18 |
| 入力: 200B→full | -5.79 |
| 参照: first-sent→512B-tok | **+12.55** |
| **合計** | **+5.24** |

- **主因は参照の違い**: 長い予測(331B) + 長い参照(344B) のペアは、短い予測(162B) + 短い参照(171B) のペアより geo_mean が系統的に高く出る
- `evaluate.load("chrf")` のデフォルトは word_order=0 (chrF) であり、コンペのchrF++ (word_order=2) と異なる（バグ）
- greedy は beam4 より悪い（-2.18pt）
- 入力をfullにすると repetition が増加して悪化（-5.79pt）
- Eng/Akk バイト比 ≈ 1.0（英語が長いわけではない、eda002確認済み）

### Truncation閾値比較

| truncation | chrF++ | BLEU | geo | rep率 | pred長 |
|-----------|--------|------|-----|-------|--------|
| 200B | 52.15 | 35.89 | **43.26** | 20.4% | 161B |
| 300B | 55.22 | 31.20 | 41.51 | 38.2% | 222B |
| 512B | 55.86 | 24.63 | 37.09 | 50.3% | 302B |

入力を長くするとchrF++は上がるがrepetitionが急増しBLEUが下がる。200B truncationが最適。

### best_model vs last_model
同一スコア (geo=43.25/43.26)。

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| training eval | greedy, full input, 512B ref, chrF | geo=48.44 | - |
| eval_cv (200B, beam4) | first-sent ref, chrF++ | geo=43.26 | 信頼できるCV |
| eval_cv (300B, beam4) | 300B truncation | geo=41.51 | rep増で悪化 |
| eval_cv (512B, beam4) | no truncation | geo=37.09 | rep 50%で大幅悪化 |

## コマンド履歴
```bash
python workspace/exp015_v6_full_pipeline/src/train.py  # dev0
python workspace/exp015_v6_full_pipeline/src/eval_cv.py  # dev0
python workspace/exp015_v6_full_pipeline/src/eval_cv_300B.py  # dev1
python workspace/exp015_v6_full_pipeline/src/eval_cv_last.py  # dev1
python workspace/exp015_v6_full_pipeline/src/debug_full_decomposition.py  # dev0
python workspace/exp015_v6_full_pipeline/src/debug_metric_diff.py  # dev1
python workspace/exp015_v6_full_pipeline/src/debug_byte_ratio.py  # dev1
bash tools/upload_model.sh exp015_v6_full_pipeline  # dev1
```

## 次のステップ
- LBスコア確認（提出中）
- MBRデコーディング試行
- compute_metrics を次実験で修正: chrF++ (word_order=2) + first-sent ref
- repetition対策の強化（300B入力でrep 38%は高すぎる）
