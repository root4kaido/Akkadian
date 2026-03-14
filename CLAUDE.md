# CLAUDE.md

## コンペ概要

- **コンペ名**: 
Deep Past Challenge - Translate Akkadian to English

- **コンペURL**: 
https://www.kaggle.com/competitions/deep-past-initiative-machine-translation

## 実行環境

- **GPU**: NVIDIA GeForce RTX 4090 x2（各24GB）
- **Python**: 3.10.12

### tmux運用ルール（必須）

Python関連のコマンド（学習・推論・EDA等）は、**必ずtmuxセッション `dev0` または `dev1` にsend-keysで実行する**こと。
直接bashで実行してはいけない。

```bash
# 実行前: セッションの状態を確認（実行中のプロセスがないかチェック）
tmux capture-pane -t dev0 -p | tail -20

# 実行: send-keysで投入
tmux send-keys -t dev0 'python workspace/exp001_baseline/src/train.py' Enter

# 結果確認: capture-paneで出力を取得
tmux capture-pane -t dev0 -p | tail -50
```

- 実行前に必ず `capture-pane` で実行中のプロセスがないか確認する
- `dev0` が使用中なら `dev1` を使う
- 長時間学習の場合はどちらのセッションで実行中かをSESSION_NOTES.mdに記録する
- ライブラリが不足している場合も、tmux上で `pip install` すること（直接bashで実行しない）

## ディレクトリマップ

```
./
├── CLAUDE.md                  # この作業規約
├── .claude/                   # Claude Code設定・スキル
├── myMemo.md                  # 人間用メモ（Claudeは編集しない）
├── datasets/                  # データセット
│   ├── raw/                   #   配布データ（加工禁止）
│   └── processed/             #   前処理済みデータ
├── docs/                      # ドキュメント
│   ├── official/              #   コンペ公式情報（ルール・データ定義）
│   ├── paper/                 #   論文
│   └── survey/                #   技術サーベイ（Deep Researchの結果など）
├── eda/                       # EDA分析結果
│   └── eda_summary.md         #   EDA知見の集約
├── workspace/                 # 実験フォルダ群
│   └── experiment_summary.md  #   全実験のサマリー・知見集約
├── src/                       # 共有ユーティリティ
└── tools/                     # ツールスクリプト
```

## アイデア提案の原則（堅実＋爆発）

**アプローチやアイデアを提案するときは、必ず「堅実案」と「爆発案」の両方を出すこと。**

- **堅実案**: 既知の手法、定石、段階的改善。確実にスコアが上がる見込みがあるもの
- **爆発案**: 常識外れ、異分野からの転用、誰もやらなそうなアプローチ。失敗リスクは高いが当たれば大きいもの

局所解に陥らないために、爆発案は「それは普通やらないだろう」くらいがちょうどいい。

## 前処理・評価・提出

- 正規化はtrainデータの統計量で計算する（testの情報を使わない）
- コンペの評価指標を正確に再現する（既存実装のパラメータも確認）
- Augmentationはまず弱めで、過学習が確認されてから強める
- single modelのCV/LBを記録してからアンサンブルする
- 提出前に行数・カラム名・欠損値・値の範囲を確認する

## 禁止事項

- テストデータでfit/学習しない
- 今回、testデータはサンプルなので分析に使用しないこと
- CUDA_VISIBLE_DEVICES=1はつけないでください
- `datasets/raw/` の中身を直接加工・変更しない
- `myMemo.md` をClaude側で編集しない
- 実験フォルダ外にモデルや中間ファイルを散らかさない
- /tmpにファイルを作らない

## 変更の作法

- 小さく・頻繁にコミットする
- 良かれと思って余計な変更はせず、必要な変更のみ行う
- コミットメッセージは変更内容がわかるようにする
- 実験結果は必ず `SESSION_NOTES.md` と `workspace/experiment_summary.md` の両方に記録する

## 利用可能なSkills

- `/eda [データパス]` — EDA実行 → `eda/edaXXX_xxx/` に結果保存 → `eda/eda_summary.md` に知見集約
- `/experiment [実験名]` — 新規実験フォルダ作成（SESSION_NOTES.md + config.yaml の雛形生成）
- `/evaluate` — 全実験のSESSION_NOTESを読み取り `workspace/experiment_summary.md` に集約
- `/survey-discussion` — Kaggleディスカッションの定点観測・差分分析
- `/survey-notebooks` — Kaggleノートブックの調査・手法分析 → `docs/survey/notebooks/` に保存
- `/survey-papers [キーワード]` — 論文・類似コンペ解法の調査 → `docs/paper/` に保存
