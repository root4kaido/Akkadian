# SESSION_NOTES: exp034_st_pretrain

## セッション情報
- **日付**: 2026-03-15
- **作業フォルダ**: workspace/exp034_st_pretrain
- **目標**: pseudo dataをpre-trainingとして使い、real dataでfine-tune（2段階学習）

## 仮説
- exp032はpseudoとrealを混ぜて学習→pseudoのノイズがrealの学習を阻害
- 2段階にすれば: Stage1でpseudoからドメイン知識を獲得 → Stage2でrealの正確な翻訳を学習
- DAPTと似た発想だが、span corruptionではなく実際の翻訳タスクでpretrain

## 試したアプローチと結果

| アプローチ | 変更点 | sent-CV | doc-CV | 備考 |
|-----------|--------|---------|--------|------|
| exp023 baseline (fold3) | - | 35.44 | 25.52 | 比較対象 |
| exp032 (混合学習) | +pseudo 6,360件 | 35.71 | 23.75 | doc悪化 |
| exp034 (2段階, last) | pseudo 5ep pretrain → real 5ep finetune (lr=5e-5) | **36.71** | 24.31 | **sent +1.27**。rep=64.6% |
| exp034 (2段階, best_loss) | 同上、best eval_loss epoch選択 | 36.40 | 24.27 | last modelとほぼ同等 |

### デコード戦略比較（fold3 last_model）

| 推論方法 | 後処理 | sent-CV | doc-CV | 備考 |
|----------|--------|---------|--------|------|
| beam=4 (現行) | repeat_cleanup | 36.71 | 24.31 | eval_full_doc.pyのデフォルト |
| **greedy** | repeat_cleanup | **37.38** | **24.70** | **sent +0.67, doc +0.39** |
| notebook MBR (beam8+sample8) | repeat_cleanup | 35.23 | 11.79 | MBR+repetition_penaltyで長文崩壊 |
| notebook MBR | notebook postprocess | 34.90 | 11.84 | 後処理も逆効果 |

## ファイル構成
- `src/pretrain.py` — Stage1: pseudo dataでpretrain (5ep, lr=2e-4)
- `src/finetune.py` — Stage2: pretrain済みモデルでreal dataをfine-tune (5ep, lr=5e-5)

## 重要な知見
- Stage1 pretrain (pseudo 5ep): eval geo_mean=43.93（exp023の43.73とほぼ同等）
- Stage2 finetune epoch1: eval geo_mean=44.36（既にexp023超え）
- Stage2 finetune epoch5: eval geo_mean=44.93（best loss model）
- **sent-CV=36.71はexp023比+1.27ptで全self-training実験中ベスト**
- doc-CVは24.31（baseline 25.52より-1.21pt）、rep=64.6%と高い
- teacher-forcing eval lossとmetrics: pseudo pretrainでlossは0.1低いのにmetricsはほぼ同じ → cross-entropy lossは生成品質と乖離
- 混合学習（exp032/033）より2段階学習の方が明確に効果的
- **greedy推論がbeam=4より+0.67pt良い**（exp030のexp023モデルと同じ傾向）
- notebook MBR（beam8+sample8, repetition_penalty=1.2）はdoc-CVが24.31→11.79と壊滅。ByT5+MBRは長文生成を破壊
- notebook後処理（引用符・括弧除去）は参照翻訳との不一致を起こし逆効果

## コマンド履歴
```bash
# Stage 1: pseudo pretrain
python workspace/exp034_st_pretrain/src/pretrain.py --fold 3

# Stage 2: real finetune
python workspace/exp034_st_pretrain/src/finetune.py --fold 3

# 評価
python eda/eda020_sent_level_cv/eval_full_doc.py workspace/exp034_st_pretrain/results/fold3/last_model exp034_st_pretrain --preprocess exp023 --fold 3
```

## LB結果
- **LB 31.7** (exp023の30.03から+1.67pt、LBベスト更新)
- sent-CV/LB比 = 36.71/31.7 = 1.16（exp023の1.21より改善、CV-LB乖離が縮小）

## 次のステップ
- greedy推論でsubmission再作成 → LB改善確認
- 5fold展開してsubmission作成
- MRT（生成メトリクス直接最適化）でさらなる改善検討
