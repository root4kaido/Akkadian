# SESSION_NOTES: exp008_byt5_base

## セッション情報
- **日付**: 2026/03/08
- **作業フォルダ**: workspace/exp008_byt5_base
- **目標**: ByT5-baseへのスケールアップでモデル性能を大幅改善

## 仮説
- ByT5-small(300M)→ByT5-base(580M)でモデル容量が約2倍。翻訳品質が大幅に向上する見込み
- トップNBは全てByT5-base以上を使用（LB ~35.3）。current best 27.31との差はモデルサイズが主因
- cosine schedulerにより学習後半のlr decayが改善、収束が安定する
- BF16を使用（RTX4090対応）。FP16はByT5でNaN頻発のためNG
- exp005の学習設定（label masking + bidirectional）を踏襲

## 試したアプローチと結果

| アプローチ | 変更点 | CV (sent) | LB | 備考 |
|-----------|--------|-----------|-----|------|
| exp005 (small) | ByT5-small baseline | 26.56 (greedy) / 27.31 (MBR) | - | 比較基準 |
| **exp008 (base)** | ByT5-base + cosine + BF16 | **28.14 (greedy) / 28.94 (MBR)** | - | **+1.63pt (MBR)** |

## 学習設定
- ByT5-base (580M params)
- Adafactor, lr=5e-5, cosine scheduler, warmup_ratio=0.1
- BF16, batch_size=1, grad_accum=16 (effective=16)
- label_smoothing=0.2, weight_decay=0.01
- bidirectional + label masking (exp005踏襲)
- 20 epochs, best model = epoch 18 (by geo_mean)

## 学習推移（training eval）
- epoch 1: geo_mean=0.9
- epoch 5: geo_mean=15.36
- epoch 10: geo_mean=22.18
- epoch 15: geo_mean=24.11
- epoch 18: geo_mean=**24.33** (best)
- epoch 19: geo_mean=24.02
- epoch 20: geo_mean=24.26

## Sent-level推論結果（正確なCV）

| 手法 | chrF++ | BLEU | geo_mean | 繰り返し率 |
|------|--------|------|----------|-----------|
| greedy_raw | 36.76 | 21.54 | **28.14** | 48.4% |
| **mbr_raw** | **38.36** | **21.83** | **28.94** | 45.2% |
| greedy_clean | 36.98 | 21.52 | 28.21 | 45.2% |
| mbr_clean | 38.46 | 21.72 | 28.90 | 44.6% |

## ファイル構成
- `src/train.py` — 学習スクリプト（cosine scheduler + BF16 + checkpoint resume対応）
- `src/preprocess.py` — 前処理（exp005と同一ロジック）
- `src/eval_sentence_level.py` — sent-level推論+評価スクリプト
- `results/best_model/` — ベストモデル（epoch 18）
- `results/val_predictions_sentence.csv` — sent-level全4手法の予測結果（157サンプル）
- `results/metrics_log.json` — 学習メトリクスログ
- `results/eval_metrics.yaml` — ベストモデル評価結果

## 重要な知見

1. **ByT5-base (580M) は ByT5-small (300M) に対し+1.63pt改善**（MBR: 27.31→28.94）
2. **BLEUの改善が顕著**: 20.11→21.83 (+1.72pt)。chrF++も36.08→38.36 (+2.28pt)
3. **cosine scheduler + warmup で安定収束**: epoch 18がbest、その後は微減
4. **BF16で問題なく学習可能**: RTX4090でOOM回避しつつ高速化
5. **batch_size=1 + grad_accum=16が必要**: batch_size=4ではOOM
6. **繰り返し問題は依然深刻**: greedy 48.4%, MBR 45.2%。ByT5-smallより悪化
7. **repeat_cleanupの効果は限定的**: MBR上でほぼ中立（28.94→28.90）
8. **LB ~35.3との差は依然大きい**: 追加データ、アンサンブル、後処理の改善が必要

## 性能変化の記録

| 実験 | 変更内容 | sent CV (MBR) | 改善幅 |
|------|---------|---------------|--------|
| exp005 (small) | ByT5-small baseline | 27.31 | - |
| **exp008 (base)** | ByT5-base + cosine + BF16 | **28.94** | **+1.63** |

## コマンド履歴
```bash
# 学習（tmux dev0, GPU0）
CUDA_VISIBLE_DEVICES=0 python workspace/exp008_byt5_base/src/train.py

# sent-level推論+評価
CUDA_VISIBLE_DEVICES=0 python workspace/exp008_byt5_base/src/eval_sentence_level.py
```

## 次のステップ
- [ ] submissionを生成してPublic LBを確認
- [ ] Sentences_Oare.csv等の追加データ活用
- [ ] アンサンブル（ByT5-small + ByT5-base）
- [ ] repetition_penaltyの調整（1.5〜2.0）
- [ ] MBR候補数増加（sampling 4〜8個）
