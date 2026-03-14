# SESSION_NOTES: exp027_filtered_additional

## セッション情報
- **日付**: 2026-03-13
- **作業フォルダ**: workspace/exp027_filtered_additional
- **目標**: exp023(前処理あり)にフィルタ済み追加データを追加し、CV改善を検証

## 仮説
- exp019(未フィルタ追加データ)ではLB改善せず（CV+2.31pt → LB-1.5pt）
- eda021で特定したノイズ（省略記号18.8%、ドイツ語混入、長さ比異常）を除去した778件なら有効
- クリーンな追加データ + exp023前処理の組み合わせで改善が見込める

## フィルタ内容（eda021）
- 省略記号 `...` を含む翻訳を除外（~219件）
- ドイツ語翻訳を除外（langdetect）
- Eng/Akk ratio < 0.3 のペアを除外
- 結果: 1161件 → 778件（約33%削除）

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| fold3 | exp023+filtered add data (778件) | sent-geo=35.45, doc-geo=24.06 | - | sent横ばい(+0.01pt)、doc悪化(-1.46pt) |

## ファイル構成
- `src/train_gkf.py` — exp023 GKFベース + additional_train_filtered.csv追加
- `config.yaml` — 実験設定
- `results/fold3/` — fold3の結果

## 重要な知見
- **フィルタ済み追加データ(778件)はほぼ中立〜微悪化**
- sent-geoはexp023と同値（35.45 vs 35.44）、doc-geoは-1.46pt悪化
- rep率は改善（10.2%）しているがスコア改善には繋がらず
- ノイズ除去しても追加データの本質的な問題（テスト分布との乖離、数字含有率98.6%、語彙カバレッジ0件）は解決していない
- **追加データ路線は効果が見込めない**。別のアプローチが必要

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp023 fold3 baseline | 前処理のみ | sent-geo=35.44, doc-geo=25.52 | - |
| exp027 fold3 | +filtered add data (778件) | sent-geo=35.45, doc-geo=24.06 | sent +0.01pt, doc -1.46pt |

## コマンド履歴
```bash
# fold3学習
tmux send-keys -t dev0 'python workspace/exp027_filtered_additional/src/train_gkf.py --fold 3' Enter
```

## 次のステップ
- **棄却**。追加データ路線は中立〜悪化。別アプローチを検討
