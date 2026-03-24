# SESSION_NOTES: exp038_backtranslation_augment

## セッション情報
- **日付**: 2026-03-19
- **作業フォルダ**: workspace/exp038_backtranslation_augment
- **目標**: generated_english.csvをs1_exp007(byt5-large)でアッカド語に逆翻訳し、追加データとしてbyt5-baseをスクラッチ学習

## 仮説
- generated_english.csv(2,120件)はLLM生成の英語翻訳。これをbyt5-largeで逆翻訳すればアッカド語の疑似データが得られる
- exp032(pseudo label)はpublished_textsの翻字→英訳だったが、今回は英語→翻字の逆方向
- real:pseudo ≈ 1:1.4の比率で、exp032の4:1よりバランスが良い
- exp034の2段階学習が有効だったが、まず混合学習でベースラインを確認

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| BT混合学習 (fold3) | generated_english 2,014件(rep除外後)をBTデータとして追加、byt5-baseスクラッチ | sent-geo=37.22, doc-geo=26.64 | - | exp023比 sent+1.78, doc+1.12 |

## ファイル構成
- `src/generate_backtranslation.py` — 英語→アッカド語の逆翻訳生成(s1_exp007 byt5-large)
- `src/generate_pseudo_v2.py` — published_texts→英訳の疑似ラベル再生成(s1_exp007, rep除外)
- `src/train.py` — 追加データ込みのbyt5-base学習(GroupKFold)
- `dataset/backtranslated.csv` — BT生成データ(2,014件、rep除外済み)
- `dataset/pseudo_labels_v2.csv` — 疑似ラベルv2(s1_exp007生成、rep除外済み)

## 重要な知見
- BT混合学習でsent-CV +1.78pt, doc-CV +1.12ptと両軸で改善。rep=10.2%も良好
- generated_english 2,120件中106件(5.0%)がrepetitionでフィルタ除外
- exp032のpseudo_labelsはrepeat_cleanup(修復)済みで品質判別不可→s1_exp007で再生成(pseudo_labels_v2)

## 性能変化の記録

| 実験 | sent-geo | doc-geo | rep% | 備考 |
|------|----------|---------|------|------|
| exp023 (baseline, fold3) | 35.44 | 25.52 | - | ベースライン |
| exp034 (pseudo pretrain→real ft, fold3) | 36.71 | 24.31 | - | 2段階学習 |
| **exp038 (BT混合, fold3)** | **37.22** | **26.64** | **10.2%** | **sent+1.78, doc+1.12 vs exp023** |

## コマンド履歴
```bash
# Step 1: 逆翻訳生成 (dev0)
tmux send-keys -t dev0 'python workspace/exp038_backtranslation_augment/src/generate_backtranslation.py' Enter

# Step 2: 学習 fold3 (dev0)
tmux send-keys -t dev0 'python workspace/exp038_backtranslation_augment/src/train.py --fold 3' Enter

# Step 3: 評価 (dev0)
tmux send-keys -t dev0 'python eda/eda020_sent_level_cv/eval_full_doc.py workspace/exp038_backtranslation_augment/results/fold3/last_model exp038_bt_augment --preprocess exp023 --fold 3' Enter

# pseudo labels v2 生成 (dev1, 並行)
tmux send-keys -t dev1 'python workspace/exp038_backtranslation_augment/src/generate_pseudo_v2.py' Enter
```

## 次のステップ
- exp034式の2段階学習(BT+pseudo pretrain → real finetune)の検討
- pseudo_labels_v2 + backtranslated の全データ混合学習
- LB提出で実力確認
