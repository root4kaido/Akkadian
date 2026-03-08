# ディスカッション活動サマリー

> 最終更新: 2026/03/08（2回目スナップショット）

## 概要

- トピック数: 20（うち pinned: 7）
- 最もコメントの多いトピック: "A Stitch in Time Saves Nine"（64 comments）
- 活発なトピック: "Two practical stumbling blocks"（52 comments）、"Dataset Update - Mind the Gaps"（44 comments）

## 3/8 新規トピック（3/6以降）

### eBL text divergence（1 comment, David Ochoa Corrales）
- **eBL→OARE Mimation Restorer**: Standard Babylonian語形→Old Assyrian語形変換スクリプト
- Old AssyrianはMimation（語末"-m"）を保持、SBは失っている
- 例: eBL `šakānu I` → OARE `šakānum`
- **585ペアの検証で92.8%精度**
- **重要度**: 中 — eBL_Dictionaryの活用可能性を大幅改善

### Notebook Submission not throwing errors（0 comments, The Swimmer）
- 投稿不具合の質問。技術的価値なし

---

## 重要トピック分析

### [Pinned] A Stitch in Time Saves Nine（64 comments）
- **投稿者**: Adam Anderson（Competition Host）
- **内容**: 最終データ更新のアナウンス
  - gapマーカーを2種類→1種類(`<gap>`)に統一
  - trainデータをtestデータのフォーマットに合わせるための推奨事項リスト
  - 追加変更で新たな不整合が生じないよう、最小限の修正に留めた
- **重要度**: **極めて高い** — trainデータの前処理方針に直結

### [Pinned] Dataset Update - Mind the Gaps（44 comments）
- **投稿者**: Ryan Holbrook（Kaggle Staff）
- **内容**: `<gap>`表記の統一アップデート
  - train.csv, published.csv, hidden test.csv, test set labelsすべて更新
  - 全提出のリスコアを実施
  - コメントで議論: 分数表記（0.5 vs ½）、PN=`<gap>`の意味、アポストロフィの扱い
- **重要度**: **高** — データ不整合の修正

### [Pinned] Two practical stumbling blocks in Akkadian → English MT（52 comments）
- **投稿者**: Adam Anderson（Competition Host）
- **内容**: コンペの**2大ボトルネック**を指摘
  1. **固有名詞（人名・地名・神名）**: 翻字が不統一、モデルが hallucinate しやすい、BLEUスコアへの影響大
  2. **ASCII/翻字フォーマットの不一致**: データセット間でエンコーディングが異なる
  - 固有名詞の扱いがトークナイゼーション、アラインメント、報酬安定性に最も大きく影響
  - モデルサイズやオプティマイザーの選択よりもデータ表現レベルの問題が支配的
- **重要度**: **極めて高い** — 戦略決定に直結する技術的知見

### [Pinned] Welcome（16 comments）
- **投稿者**: Adam Anderson
- **内容**: コンペ開始のウェルカムメッセージ、リソース案内
- **重要度**: 低（情報的）

### [Pinned] Old Assyrian dataset updates（4 comments）
- **投稿者**: Adam Anderson
- **内容**: 追加データセットの公開
  1. **Old Assyrian grammars + onomasticon.csv**: https://www.kaggle.com/datasets/deeppast/old-assyrian-grammars-and-other-resources/data
     - 文法書、アッカド語辞書zip、**onomasticon（人名リスト）**
     - onomasticonはOA_Lexicon_eBL.csvとマッチングで「別名」発見可能
  2. **Old Assyrian Kültepe tablets PDF**: https://www.kaggle.com/datasets/deeppast/old-assyrian-kltepe-tablets-in-pdf/data
     - 主要テキストエディションのPDF（publications.csvのOCRが不十分だったため）
- **重要度**: **高** — 追加学習データの源泉

### [Pinned] Data Update（33 comments）
- **投稿者**: Ryan Holbrook
- **内容**: eBL_Dictionary.csv, resources.csv, Sentences_Oare_FirstWord_LinNum.csv の公開
  - コメントで議論: 限定詞の `{d}` vs `(d)` の不一致、gap表記の違い
- **重要度**: 中

### [Pinned] How to get started + Discord（2 comments）
- **内容**: 初心者向け案内、Discord: https://discord.gg/kaggle
- **重要度**: 低

---

## 注目の非pinnedトピック

### Insights from the Akkademia Codebase & PNAS Paper（5 comments）
- **投稿者**: James McGuigan
- **内容**: Akkademiaプロジェクトの分析
  - 先行研究はCNNベース（Fairseq fconv）で**BLEU4=37.47**を達成（2023年PNAS論文）
  - 2つのパイプライン: C2E（楔形文字→英語）とT2E（翻字→英語）
  - **ByT5, mT5, NLLBなどの最新Transformerで超えるべきベンチマーク**
  - BPEトークナイゼーション: 翻字1,000語彙、英語10,000語彙
- **重要度**: **高** — ベースライン・モデル選択の参考

### Is this competition becoming a 'Regex Guessing Game'?（6 comments）
- **投稿者**: DaylightH（28位）
- **内容**: データフォーマットの不統一に対する不満
  - Regex前処理に時間を取られ、NMT戦略の実験に集中できない
  - 反論: 「現実のデータクレンジングの学び」「最新updateでフォーマットは明確になった」
- **重要度**: 中 — データ前処理が重要な競技要素であることを示唆

### Lora on ByT5 large（1 comment）
- **投稿者**: PUN
- **内容**: ByT5-largeにLoRAを適用、スコア上限18.0
  - コメント: 「LoRAは既存知識の適応に向いており、新知識の注入はフルファインチューニングが必要」
- **重要度**: 中 — ByT5-largeのLoRAは効果薄

### Low LB score（2 comments）
- **内容**: 前処理したのにLBスコア10程度
  - 回答: 「スターターnotebookは前処理なしで30近く出る」「文分割が重要」
- **重要度**: 低〜中 — スターターで30が出ることを確認

### Massive bot attack（17 comments）
- **投稿者**: Yurnero（3位）
- **内容**: 140+のbot アカウントが同時にコンペに参加。Kaggle Staff調査中
- **重要度**: 低（競技自体には影響なし）

### Old Assyrian Tokens（1 comment）
- **投稿者**: David Ochoa Corrales
- **内容**: OAREデータベースの全音節リスト公開
  - https://www.kaggle.com/datasets/davidochoacorrales/oare-tokens
  - Unicode正規化（NFC）の推奨
- **重要度**: 中 — トークナイゼーションの参考

---

## 戦略的インサイト（ディスカッションから得られた知見）

1. **固有名詞が最大のボトルネック**（Host直言）→ onomasticon + OA_Lexiconの活用が鍵
2. **データ前処理が極めて重要**: gap統一、Ḫ→H変換、フォーマット不整合の対処
3. **ByT5-largeのLoRAは効果薄**（18.0止まり）→ フルファインチューニングが必要
4. **スターターnotebookで約30**: ベースラインとして参考に
5. **先行研究（Akkademia/PNAS）**: CNN fconvでBLEU4=37.47、BPE vocab 1K/10K
6. **追加データ**: onomasticon.csv（人名リスト）と PDFテキストエディションが公開済み
7. **Unicode正規化（NFC）** が推奨される
