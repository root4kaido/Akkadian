# Deep Past Challenge - Translate Akkadian to English

> Bringing Bronze Age Voices Back to Life – Machine Translation of Old Assyrian Cuneiform

## コンペ概要

- **URL**: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation
- **主催**: Deep Past Initiative
- **タイプ**: Featured Code Competition
- **タスク**: 翻字(transliteration)されたアッカド語（古アッシリア方言）→ 英語への機械翻訳
- **データライセンス**: CC BY-SA 4.0
- **勝者ライセンス**: CC-BY 4.0

## 背景

- 4,000年前のアッシリア商人が残した粘土板アーカイブ（債務・隊商・日常の家族の事柄を記録）
- 約23,000枚の粘土板が現存し、半分のみ翻訳済み
- 世界で十数名の専門家しか読めない → AIで解読を支援
- 古アッシリア語はアッカド語の初期方言で、最古の記録されたセム語族の言語

## 評価指標

**BLEU と chrF++ のジオメトリック平均（Geometric Mean）**

- 各スコアの十分統計量（sufficient statistics）をコーパス全体で集約（**micro-average**）
- 実装参考: [SacreBLEU](https://github.com/mjpost/sacrebleu/)
- 評価メトリックNotebook: [Geometric Mean of BLEU and chrF++](https://www.kaggle.com/code/metric/dpi-bleu-chrf)

### 提出ファイル形式

```csv
id,translation
0,Thus Kanesh, say to the -payers, our messenger, every single colony, and the...
1,In the letter of the City (it is written): From this day on, whoever buys meteoric...
2,As soon as you have heard our letter, who(ever) over there has either sold it to...
```

- テストセットの各 `id` に対して英語翻訳を予測
- 各翻訳は1文（single sentence）

## タイムライン

| イベント | 日付 |
|---------|------|
| 開始 | 2025年12月16日 |
| エントリー締切 | 2026年3月16日 |
| チームマージ締切 | 2026年3月16日 |
| **最終提出締切** | **2026年3月23日** |

※すべて UTC 23:59

## 賞金（合計 $50,000）

| 順位 | 賞金 |
|------|------|
| 1st | $15,000 |
| 2nd | $10,000 |
| 3rd | $8,000 |
| 4th | $7,000 |
| 5th | $5,000 |
| 6th | $5,000 |

## コード要件（Code Competition）

- **CPU Notebook**: 9時間以内
- **GPU Notebook**: 9時間以内
- **インターネット接続**: 無効
- **外部データ**: 無料で公開されている外部データ・事前学習済みモデルは使用可
- **提出ファイル名**: `submission.csv`

## 参加状況（2026/3/6時点）

- エントラント: 10,415
- 参加者: 3,082
- チーム: 2,715
- 提出数: 44,015

## Citation

Abdulla, F., Agarwal, R., Anderson, A., Barjamovic, G., Lassen, A., Ryan Holbrook, and María Cruz. Deep Past Challenge - Translate Akkadian to English. https://kaggle.com/competitions/deep-past-initiative-machine-translation, 2025. Kaggle.
