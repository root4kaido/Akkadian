# SESSION_NOTES: exp004_longer_context

## セッション情報
- **日付**: 2026-03-07
- **作業フォルダ**: workspace/exp004_longer_context
- **目標**: max_length拡大でtruncation解消 + best model選択 + wandb導入

## 仮説

exp003ではmax_length=512で入力35%・出力37%がtruncateされ、長文の学習情報が失われていた。
加えてtraining evalのlabel truncationによりCVが水増しされていた(23.55 vs 実際17.87)。

encoder_max_length=1024, decoder_max_length=2048にすることで:
1. 長文データの学習品質が向上（入力99.8%, 出力98.6%カバー）
2. training evalが正確になる（水増し解消）
3. load_best_model_at_end=Trueで+0.67pt程度の改善

## exp003からの変更点

| 変更 | exp003 | exp004 |
|------|--------|--------|
| encoder_max_length | 512 | 1024 |
| decoder_max_length | 512 | 2048 |
| load_best_model_at_end | False | True (geo_meanベース) |
| save_total_limit | 1 | 2 |
| report_to | none | wandb |
| inference num_beams | 4 | 1 (greedy) |
| inference batch_size | 16 | 8 |
| plot_metrics.py | あり | なし（wandbで代替） |

## 試したアプローチと結果

| アプローチ | 変更点 | CV (inference greedy raw) | CV (inference greedy post) | CV (beam4 post) | LB | 備考 |
|-----------|--------|--------------------------|---------------------------|-----------------|-----|------|
| exp004 | max_length拡大 + best model | **10.68** | **11.90** | **14.00** | - | exp003(17.87)から大幅悪化 |

## Inference CV詳細

| 方法 | chrF | BLEU | geo_mean | exp003比較 |
|------|------|------|----------|-----------|
| greedy raw | 23.48 | 4.86 | 10.68 | 17.87 → 10.68 (**-7.19**) |
| greedy post | 24.20 | 5.85 | 11.90 | - |
| beam4 raw | 23.95 | 7.63 | 13.52 | 14.37 → 13.52 (-0.85) |
| beam4 post | 23.72 | 8.26 | 14.00 | 13.20 → 14.00 (+0.80) |

## Training Eval（truncationなし、正確）

| Epoch | eval_loss | chrF | BLEU | geo_mean |
|-------|-----------|------|------|----------|
| 17 | 2.2030 | 22.66 | 4.51 | 10.11 |
| 18 | 2.2018 | 22.79 | 4.66 | 10.30 |
| 19 | 2.2001 | 22.89 | 4.76 | 10.44 |
| 20 | 2.1998 | 23.50 | 4.86 | 10.69 |

Best model = epoch 20 (geo_mean=10.69)。training evalとinference evalの一致を確認済み（10.69 vs 10.68）。

## ファイル構成

- `src/preprocess.py` — exp003と同一（双方向データ作成）
- `src/train.py` — encoder/decoder別max_length, wandb, load_best_model_at_end=True, eval_batch_size=8
- `src/postprocess.py` — exp003と同一（s.strip()→s.str.strip()修正）
- `src/infer.py` — greedy推論 + beam4比較オプション付き

## 重要な知見

- **training evalとinference evalの乖離が解消**: label truncation修正により一致（10.69 vs 10.68）
- **max_length拡大でスコアが大幅悪化**: greedy 17.87→10.68（-7.19pt）
- **beam4はほぼ同等**: 14.37→13.52（-0.85pt）、postprocess付きだと13.20→14.00（+0.80pt）
- **greedyがbeam4より悪い（逆転）**: exp003ではgreedy>beam4だったが、exp004ではbeam4>greedy
- **chrFは同等（23-24）だがBLEUが激減**: greedy BLEU 11.46→4.86。長文生成の精度が低下
- ByT5-smallにはmax_length=512が適切な可能性。長い系列の学習はモデル容量不足で逆効果

## コマンド履歴
```bash
cd /home/user/work/Akkadian && python workspace/exp004_longer_context/src/train.py
python workspace/exp004_longer_context/src/infer.py
```

## 次のステップ
- [ ] 予測サンプルを比較してexp003との差異を分析
- [ ] max_length=512に戻してbest model+wandbのみの効果を検証（exp005）
