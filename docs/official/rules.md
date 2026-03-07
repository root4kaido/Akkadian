# コンペティションルール

## 基本情報

- **コンペ名**: Deep Past Challenge - Translate Akkadian to English
- **URL**: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation
- **主催**: Deep Past Initiative（234 Front St, New Haven CT 06513）
- **最終提出期限**: 2026年3月23日 23:59 UTC

## 評価指標

**BLEU と chrF++ のジオメトリック平均**（micro-average）

- 実装: [SacreBLEU](https://github.com/mjpost/sacrebleu/)
- 評価Notebook: [Geometric Mean of BLEU and chrF++](https://www.kaggle.com/code/metric/dpi-bleu-chrf)

## 提出形式

- ファイル名: `submission.csv`
- フォーマット: `id,translation`（テストセットの各idに英語翻訳を1文ずつ）
- Code Competition: Notebook経由で提出

## チーム・提出制限

- **最大チームサイズ**: 5名
- **チームマージ**: 可（マージ後の合計提出数が上限以内であること）
- **1日あたり最大提出数**: 5回
- **最終提出選択数**: 2つ

## コード要件

- CPU Notebook: 9時間以内
- GPU Notebook: 9時間以内
- インターネット接続: **無効**
- 外部データ: 無料で公開されているものは使用可（事前学習済みモデル含む）
- 提出ファイル名: `submission.csv`

## 外部データ・ツール

- **外部データ使用可**: ただし全参加者に公開・無料でアクセス可能であること
- **Reasonableness Standard**: 過度なコストのLLM・データ・ツールは制限される可能性あり
  - 例: Gemini Advancedの少額サブスクはOK、コンペ賞金を超えるデータセットライセンスはNG
- **AutoML**: 使用可（適切なライセンスが必要）

## データアクセス・使用

- ライセンス: **CC-BY-SA 4.0**
- 競技目的・商用目的いずれも使用可
- データセキュリティ: ルールに同意していない者へのデータ提供・再配布は禁止

## 勝者の義務

- 勝者ライセンス: **CC-BY 4.0**（Submissionおよびソースコード）
- 最終モデルのソフトウェアコード（学習コード・推論コード・計算環境の説明）を提供
- 方法論の詳細な説明（アーキテクチャ、前処理、損失関数、学習詳細、ハイパーパラメータ等）
- 再現可能なコードリポジトリとドキュメントを提出

## コード共有ルール

- **Private共有**: チーム外での非公開コード共有は**禁止**
- **Public共有**: Kaggle上のフォーラム・Notebookでの公開共有は**可**（OSSライセンス必須）
- **オープンソース**: OSI承認ライセンスで商用制限なし

## 賞金（合計 $50,000）

| 順位 | 賞金 |
|------|------|
| 1st | $15,000 |
| 2nd | $10,000 |
| 3rd | $8,000 |
| 4th | $7,000 |
| 5th | $5,000 |
| 6th | $5,000 |

## 準拠法

コネチカット州法、ニューヘイブン郡の連邦・州裁判所
