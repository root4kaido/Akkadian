# SESSION_NOTES: exp039_all_data_mix

## セッション情報
- **日付**: 2026-03-19
- **作業フォルダ**: workspace/exp039_all_data_mix
- **目標**: 全データ混合学習（train + additional_train_filtered + pseudo_labels_v2 + backtranslated）

## 仮説
- exp038でBT 2,014件の混合がsent+1.78, doc+1.12と有効だった
- さらにpseudo_labels_v2(6,109件)とadditional_train_filtered(778件)を追加し全データ投入
- 合計10,462件（train比6.7倍）。5 epochsで学習
- データ量増加で汎化性能が上がるか、ノイズで悪化するかの検証

## データソース

| ソース | 件数 | 内容 |
|--------|------|------|
| train.csv | 1,561 | コンペ本体(doc単位) |
| additional_train_filtered | 778 | フィルタ済み追加データ |
| pseudo_labels_v2 | 6,109 | published_texts→s1_exp007英訳(rep除外) |
| backtranslated | 2,014 | generated_english→s1_exp007逆翻訳(rep除外) |
| **合計** | **10,462** | |

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| 全データ混合 (fold3) | train+additional+pseudo_v2+BT, 5ep, lr=1e-4 | sent-geo=37.37, doc-geo=25.42 | - | sent微増だがdoc悪化 |

## ファイル構成
- `src/train.py` — 全データ混合学習スクリプト

## 重要な知見
- sent-CVはexp038(BT混合)と同等(37.37 vs 37.22, +0.15pt)だが、doc-CVが25.42と-1.22pt悪化
- pseudo_labels_v2(6,109件)とadditional_train_filtered(778件)の追加はdoc品質を悪化させる
- exp032の教訓と同パターン: pseudo dataの比率が高すぎるとノイズが支配的になりdoc-CV悪化
- **BT 2,014件のみの混合(exp038)が最もバランスが良い**

## 性能変化の記録

| 実験 | sent-geo | doc-geo | rep% | 備考 |
|------|----------|---------|------|------|
| exp023 (baseline, fold3) | 35.44 | 25.52 | - | ベースライン |
| exp034 (pseudo 2段階) | 36.71 | 24.31 | - | 2段階学習 |
| exp038 (BT混合) | 37.22 | 26.64 | 10.2% | **BT 2,014件のみ** |
| **exp039 (全データ混合)** | **37.37** | **25.42** | **11.0%** | sent微増、doc悪化 |

## コマンド履歴
```bash
# 学習(fold3) lr=1e-4
tmux send-keys -t dev0 'python workspace/exp039_all_data_mix/src/train.py --fold 3' Enter

# 評価
tmux send-keys -t dev0 'python eda/eda020_sent_level_cv/eval_full_doc.py workspace/exp039_all_data_mix/results/fold3/last_model exp039_all_data --preprocess exp023 --fold 3' Enter
```

## 次のステップ
- exp038(BT混合)の方がdoc-CVで優位 → exp038ベースで進める方が良い
- 2段階学習(全データpretrain → real finetune)は検討の余地あり
