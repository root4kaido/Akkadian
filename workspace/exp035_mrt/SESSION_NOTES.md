# SESSION_NOTES: exp035_mrt

## セッション情報
- **日付**: 2026-03-15
- **作業フォルダ**: workspace/exp035_mrt
- **目標**: MRT系手法（SCST → MBR-FT）でコンペ指標を直接最適化し、LBスコア改善

## 仮説
- exp034はcross-entropy損失で学習 → 生成品質（BLEU/chrF++）との乖離が確認済み
- Kaggle notebook (byt5-akkadian-mbr-v2) はLB=34.4で、我々のLB=31.7と+2.7ptの差
- notebookモデル名の"mbr"はMBR/MRT的訓練を示唆
- 評価指標を直接最適化する訓練が必要

## 試したアプローチと結果

| アプローチ | 変更点 | val geo_mean | 備考 |
|-----------|--------|-------------|------|
| exp034 baseline (greedy) | - | 25.33 (initial eval) | 比較対象 |
| SCST temp=0.3 | α=0.5, K=4, lr=5e-6 | 25.18 (step50) | advantage≈0.001、信号弱すぎ |
| SCST temp=0.7 | α=0.5, K=4, lr=5e-6 | (未eval) | advantage≈0.002、改善なし |
| SCST temp=1.0 | α=0.7, K=4, lr=2e-5 | 24.99 (step50) | advantage≈-0.03、悪化 |
| MBR-FT (best_model) | MBR選択翻訳でcross-entropy fine-tune, 3ep | sent=36.21, doc=23.32 | **悪化** sent -1.17, doc -1.38 |

## SCST失敗の分析
- **根本原因**: ByT5のbyte-levelサンプリングとREINFORCE勾配の相性が悪い
  - temp低(0.3): サンプルがgreedyとほぼ同一 → advantage≈0、勾配信号なし
  - temp高(1.0): サンプルがgreedyより悪い → 負のadvantageのみ、良い方向への学習信号なし
- **gradient checkpointing + train()モードのバグ**: generate()がtrain()モードだと1文字しか生成しない。eval()切替が必須
- **結論**: REINFORCE勾配推定はbyte-levelモデルには不向き。MBR-FT（蒸留ベース）に切替

## ファイル構成
- `src/train_mrt.py` — SCST訓練スクリプト（失敗、参考用）
- `src/generate_mbr_targets.py` — MBR-FT Step1: 候補生成 + MBR選択
- `src/finetune_mbr.py` — MBR-FT Step2: MBR選択翻訳でcross-entropy fine-tune

## MBR-FT失敗の分析
- MBRターゲット生成: greedy chrF++=60.39 → MBR chrF++=61.12（+0.73ptのみ）
- 68.7%の文でMBRターゲットがgreedyと異なるが、改善幅が小さすぎる
- MBRターゲットでfine-tuneすると参照翻訳から離れる方向に学習 → eval_loss悪化（0.58→0.64）
- 生成メトリクス（Trainer内eval geo_mean）は45.08→45.34と微増だが、eval_full_doc.pyでは悪化
- **結論**: ByT5のsampling多様性が低いため、MBR選択でも大幅な改善は得られない。モデル自身の出力空間にはgreedyより有意に良い翻訳がほとんど存在しない

## 重要な知見
- gradient_checkpointing_enable() + model.train() だとmodel.generate()が壊れる（1文字しか生成しない）→ 生成時はmodel.eval()必須
- ByT5のbyte-levelサンプリングはtoken-levelと比べて多様性が低い（多くのバイトが確定的）
- sentence-level BLEUは短文でほぼ0 → reward関数にはchrF++単体が適切
- SCST/REINFORCEはbyte-levelモデルには不向き
- MBR-FTも効果なし: MBRターゲットの改善幅が+0.73pt chrF++と小さすぎ、fine-tuneで悪化

## コマンド履歴
```bash
# SCST訓練（失敗）
python workspace/exp035_mrt/src/train_mrt.py --fold 3

# MBR-FT Step1: ターゲット生成
python workspace/exp035_mrt/src/generate_mbr_targets.py --fold 3

# MBR-FT Step2: fine-tune
python workspace/exp035_mrt/src/finetune_mbr.py --fold 3

# 評価
python eda/eda020_sent_level_cv/eval_full_doc.py workspace/exp035_mrt/results/fold3/best_model exp035_mrt --preprocess exp023 --fold 3
```

## 次のステップ
- exp035はSCST、MBR-FTともに失敗 → MRT系アプローチは断念
- LB=34.4との差は訓練手法ではなく、データ or モデルアーキテクチャの差の可能性
- 別アプローチを検討: 5fold ensemble、知識蒸留、データ拡張改善 等
