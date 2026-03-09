# SESSION_NOTES: exp009_input_truncation

## セッション情報
- **日付**: 2026-03-08
- **作業フォルダ**: workspace/exp009_input_truncation
- **目標**: 入力側の文レベルaugmentationでテスト条件（文レベル入力）への適応を改善

## 仮説
- 現状、学習は常にdoc全文入力（平均490バイト）だが、テストは文レベル入力（~200バイト）
- exp005でラベル（出力側）は文マスクしたが、入力側のミスマッチは未解決
- 各エポックで50%の確率でAkkadian入力を先頭200バイトに切り詰めることで、テスト条件に適応
- 逆変換（Eng→Akk）も対称的に、英語1文目→Akkadian先頭200バイトのsent版を適用
- データ量は変わらず（augmentation）、エポックごとに異なるバリエーション → 正則化効果も期待

## 試したアプローチと結果

| アプローチ | 変更点 | CV (greedy) | CV (MBR) | 備考 |
|-----------|--------|-------------|----------|------|
| input truncation aug (p=0.5, 200B) | 50%の確率でAkk入力を先頭200Bに切り詰め | 24.11 | **29.08** | MBR微増、greedy悪化 |

## ファイル構成
- `src/preprocess.py` — 確率的input truncation augmentation付きデータ準備（AugmentedTrainDataset）
- `src/train.py` — exp008ベース、カスタムDatasetで動的augmentation
- `src/eval_sentence_level.py` — 文レベル推論・評価（exp008と同一ロジック）
- `config.yaml` — augmentation設定追加

## 重要な知見
- **MBR=29.08（+0.14pt vs exp008）**: わずかに改善だが、有意差とは言い難い
- **greedy=24.11（-4.03pt vs exp008）**: 大幅悪化。繰り返し率も45%→54.8%に増加
- augmentationにより出力が不安定化 → greedyでは繰り返しが増えたが、MBRが吸収
- training eval geo_mean=21.70（exp008: 24.33）。ベストモデルはepoch 17（checkpoint-2992）
- **入力側の文レベル最適化は効果限定的**。出力側のラベルマスキング（exp005）の方が効果大

## 性能変化の記録

| 実験 | 変更内容 | greedy | MBR | 改善幅(MBR) |
|------|---------|--------|-----|------------|
| exp008 | ByT5-base baseline | 28.14 | 28.94 | - |
| exp009 | +input truncation aug (p=0.5) | 24.11 | **29.08** | +0.14pt |

### Sent-level全結果

| 手法 | chrF++ | BLEU | geo_mean |
|------|--------|------|----------|
| greedy_raw | 35.09 | 16.56 | 24.11 |
| mbr_raw | 37.94 | 22.30 | **29.08** |
| greedy_clean | 36.52 | 20.61 | 27.43 |
| mbr_clean | 37.98 | 22.24 | 29.06 |

### 繰り返し率

| 手法 | exp008 | exp009 |
|------|--------|--------|
| greedy | 45% | 54.8% |
| MBR | - | 51.0% |

## コマンド履歴
```bash
# 学習
python workspace/exp009_input_truncation/src/train.py
# 評価
python workspace/exp009_input_truncation/src/eval_sentence_level.py
```

## 次のステップ
- [x] 学習実行
- [x] sent-level評価
- [ ] 結果を踏まえた次の実験検討
