# SESSION_NOTES: s1_exp001_byt5_large_fold3

## セッション情報
- **日付**: 2026-03-16〜17
- **作業フォルダ**: workspace/s1_exp001_byt5_large_fold3
- **目標**: byt5-largeでfold3を学習し、baseとの比較を行う
- **マシン**: 別マシン（A100）で実行

## 仮説
- byt5-largeはbaseより表現力が高く、翻訳精度が向上するはず
- ただしデータ量（~1500サンプル）に対してパラメータ数が多すぎる懸念あり

## 設定
- モデル: google/byt5-large (1.2B params)
- lr: 2e-4（exp023 baseと同一）
- epochs: 20
- batch_size: 4, grad_accum: 2
- GatedBestModelCallback (loss_gate=1.2) ※再学習時に追加
- その他はexp023と同一

## 結果

### 学習経過
- eval_lossはepoch 7 (0.4197) で最小、以降は過学習
- geo_meanはepoch 20まで微改善し続けた（loss悪化でもargmaxは安定する現象）
- GatedBestModelCallbackはepoch 13で最終更新（geo=44.34, loss=0.4672）
- epoch 14以降はloss > min_loss * 1.2 でゲートされた

### eval_full_doc評価

| モデル | sent-CV chrF++ | sent-CV BLEU | sent-CV geo | sent rep% | doc-CV geo | doc rep% |
|--------|---------------|-------------|-------------|-----------|------------|----------|
| gated_best (ep13) | 47.64 | 27.46 | 36.17 | 11.2% | 23.88 | 59.3% |
| last (ep20) | 48.33 | 28.28 | 36.97 | 13.9% | 24.62 | 61.6% |

### LB結果
- **last model: LB 25.3**（exp023 base fold3 last LB=30.4から-5.1pt）
- sent-CVでは+1.53ptなのにLBでは-5.1pt → CV-LB乖離が深刻

## 重要な知見
- byt5-largeはlr=2e-4ではeval_lossが激しく過学習する
- geo_meanはloss悪化中も改善し続ける（teacher-forcingのcross-entropy lossと生成品質の乖離）
  - 過学習で確率分布がシャープになり、loss（対数尤度）は悪化するが、argmax（生成トークン）は変わらない
- CV-LB乖離: sent-CVが上がってもLBは下がる → val分布内では正しいargmaxを出すが、未知パターンで過信した分布が間違ったargmaxを出す
- このデータ規模ではlr=2e-4はlargeモデルに対して高すぎる → s1_exp007でlr=1e-4に改善
- batch_decode(skip_special_tokens=True)がbyt5-largeで極端に遅くなる問題あり → _fast_batch_decodeで回避

## コマンド履歴
```bash
python workspace/s1_exp001_byt5_large_fold3/src/train_gkf.py --fold 3

# 評価
python eda/eda020_sent_level_cv/eval_full_doc.py workspace/s1_exp001_byt5_large_fold3/results/fold3/gated_best_model s1_exp001_byt5_large_fold3_gated_best --preprocess exp023 --fold 3
python eda/eda020_sent_level_cv/eval_full_doc.py workspace/s1_exp001_byt5_large_fold3/results/fold3/last_model s1_exp001_byt5_large_fold3_last --preprocess exp023 --fold 3
```
