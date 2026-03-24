# SESSION_NOTES: s1_exp007_large_lr1e4

## セッション情報
- **日付**: 2026-03-17
- **作業フォルダ**: workspace/s1_exp007_large_lr1e4
- **目標**: byt5-large fold3をlr=1e-4（s1_exp001の半分）で学習し、過学習を抑制する

## 仮説
- s1_exp001 (lr=2e-4) はeval_lossが過学習していた
- lr=1e-4に下げることで、lossが健全に推移し、LBとの乖離が縮小するはず

## 設定
- モデル: google/byt5-large
- lr: 1e-4（s1_exp001の2e-4から半減）
- epochs: 20
- batch_size: 4, grad_accum: 2
- GatedBestModelCallback (loss_gate=1.2)
- その他はexp023/s1_exp001と同一

## 結果

### 学習経過
- lossが最後まで改善し続け、**gated_best = last（epoch 20）**
- min_loss = 0.3960 (epoch 8)、最終loss = 0.4283
- gated bestが20回更新（ほぼ毎epoch）→ lossの悪化がゲート閾値内に収まり続けた
- geo_mean: 14.95 (ep1) → 46.61 (ep20) で単調増加

### eval_full_doc評価

| モデル | sent-CV chrF++ | sent-CV BLEU | sent-CV geo | sent rep% | doc-CV geo | doc rep% |
|--------|---------------|-------------|-------------|-----------|------------|----------|
| last (=gated_best) | 49.83 | 29.00 | 38.01 | 10.8% | 26.05 | 65.0% |

### s1シリーズ比較

| 実験 | 変更点 | sent-CV geo | doc-CV geo | sent rep% |
|------|--------|-------------|------------|-----------|
| s1_exp001 gated_best (lr=2e-4, ep13) | baseline large | 36.17 | 23.88 | 11.2% |
| s1_exp001 last (lr=2e-4, ep20) | baseline large | 36.97 | 24.62 | 13.9% |
| **s1_exp007 last (lr=1e-4, ep20)** | **lr半減** | **38.01** | **26.05** | **10.8%** |

## 重要な知見
- lr=1e-4ではlossの過学習が大幅に緩和され、gated_best=lastとなった
- sent-CV/doc-CV両方でs1_exp001を上回り、rep率も最低
- ただしbaseのexp023 fold3 (sent-CV geo=35.44, LB=30.4) と比較してsent-CVは+2.57ptだが、largeのCV-LB乖離が大きい懸念あり
- LB結果待ち

## コマンド履歴
```bash
python workspace/s1_exp007_large_lr1e4/src/train_gkf.py --fold 3

# 評価
python eda/eda020_sent_level_cv/eval_full_doc.py workspace/s1_exp007_large_lr1e4/results/fold3/last_model s1_exp007_large_lr1e4_last --preprocess exp023 --fold 3
python eda/eda020_sent_level_cv/eval_full_doc.py workspace/s1_exp007_large_lr1e4/results/fold3/gated_best_model s1_exp007_large_lr1e4_gated_best --preprocess exp023 --fold 3
```
