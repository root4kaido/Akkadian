# 文献サーベイ: Akkadian-to-English 機械翻訳

コンペ（Deep Past Challenge）に関連する論文・リソースの調査結果。
コンペへの実用性が高い順に整理。

---

## 1. Gutherz et al. (2023) — 最重要先行研究

**Translating Akkadian to English with neural machine translation**
- 著者: Gai Gutherz, Shai Gordin, Luis Sáenz, Omer Levy, Jonathan Berant
- 掲載: PNAS Nexus, Vol.2, No.5, pgad096
- URL: https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349
- 補助資料: https://github.com/DigitalPasts/AkkadiantoEnglish_NMT_SI

### アーキテクチャ
- **Fairseq fconv（CNN）** を採用。低リソース・文字レベルMTではTransformerより学習が速いと判断
- 2タスク: T2E（翻字→英語）、C2E（楔形文字→英語）

### 学習データ
- **ORACC** から取得: RINAP, RIAo, RIBo, SAAo, Suhu
- 合計 **8,056テキスト** / T2E: 56,160文, C2E: 50,299文
- **Neo-Assyrian中心（7,327件）、Old Assyrianはわずか122件**
- 分割: 90/5/5

### トークナイゼーション
- T2E: **BPE (SentencePiece)** — 翻字1,000語彙 / 英語10,000語彙
- C2E: 文字ベース（400語彙）

### 結果
| タスク | NMT BLEU4 | 翻訳メモリ BLEU4 |
|--------|-----------|-----------------|
| T2E    | **37.47** | 23.51           |
| C2E    | **36.52** | 27.09           |

### コンペへの重要知見
1. **翻字不要**: C2EとT2Eはほぼ同等 → 楔形文字からの直接翻訳も有効
2. **定型ジャンルが高スコア**: 王碑文・行政文書・天文予兆で安定
3. **短文依存**: 中央値118文字以下が最適。長文でリピート・省略発生
4. **ハルシネーション**: 内在的（ソース改変）と外在的（ソース無視）の2種
5. **固有名詞が最大の課題**: 人名・地名を意味翻訳しようとして誤訳
6. **ドメイン差**: Neo-Assyrian中心学習 → Old Assyrian商業文書への転移が必要

### 補助資料リポジトリの内容
- 人間対機械比較（50文・5テキスト）: **適切翻訳16、ハルシネーション12、不適切22**
- ハルシネーション率約24% → 品質面ではまだ大きな課題

---

## 2. Gordin et al. (2020) — 楔形文字の自動翻字

**Reading Akkadian cuneiform using natural language processing**
- 著者: Shai Gordin, Gai Gutherz ら
- 掲載: PLOS ONE, 10.1371/journal.pone.0240511
- URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0240511

### 手法
3モデル比較:
1. HMM (Viterbi + trigram)
2. MEMM (最大エントロピー + ログラムフィーチャー)
3. **BiLSTM** (200次元embedding)

### データ
- ORACC Neo-Assyrian王碑文: **23,526行**
- 分割: 80/10/10

### 結果
| モデル | 翻字+分割精度 | 翻字のみ | 分割F1 |
|--------|-------------|----------|--------|
| HMM    | 89.5%       | 93.6%    | 91.8%  |
| MEMM   | 94.0%       | 96.4%    | 95.9%  |
| **BiLSTM** | **96.7%** | **97.8%** | **97.9%** |

### コンペへの適用
- **直接適用不要**（コンペデータは翻字済み）
- ただしAkkademiaツール（Python package）が公開済みで、追加データ処理に使える可能性
- 異時代テキストへの汎化: 68-84% → ドメイン差の影響を示す

---

## 3. Homburg & Chiarcos (2016) — 語分割

**Word Segmentation for Akkadian Cuneiform**
- 著者: Timo Homburg, Christian Chiarcos
- 掲載: LREC 2016
- URL: https://aclanthology.org/L16-1642.pdf

### 手法
- ルールベース・辞書ベース・統計的・ML手法を比較
- 中国語・日本語用の辞書ベースアルゴリズムを楔形文字に転用

### 結果
- 最良: **辞書ベースアルゴリズム**、F-score: 60-80%（時代で変動）

### コンペへの適用
- **コンペデータは分割済みのため直接適用不要**
- ByT5のバイトレベルアプローチが辞書不要で語分割問題を回避する理由の背景
- tokenization設計がスコアに直結するという示唆

---

## 4. Sahala et al. (2020) — BabyFST形態素解析

**BabyFST - Towards a Finite-State Based Computational Model of Ancient Babylonian**
- 著者: Aleksi Sahala, Miikka Silfverberg, Antti Arppe, Krister Lindén
- 掲載: LREC 2020
- URL: https://aclanthology.org/2020.lrec-1.479.pdf

### 手法
- 有限状態トランスデューサ (FST) によるBabylonian方言の形態素解析器

### 結果
- カバレッジ: 97.3%
- lemma/POS再現率: 93.7%
- **単一の曖昧さのない解析になるのは20.1%のみ** → 語形だけで一発翻訳は困難

### コンペへの適用
- **形態的多義性の高さ**がアッカド語MTの本質的困難さ
- ByT5は形態解析なしで文字パターンから学習 → end-to-end学習の妥当性を支持
- Old Assyrian ≠ Babylonian（方言差に注意）

---

## 5. Smidt et al. (2024) — POSタギングと前処理

**At the Crossroad of Cuneiform and NLP: Challenges for Fine-grained Part-of-speech Tagging**
- 著者: Gustav Ryberg Smidt, Els Lefever, Katrien de Graef
- 掲載: LREC-COLING 2024
- URL: https://aclanthology.org/2024.lrec-main.154.pdf

### 主要知見
- Old Babylonian書簡テキストのPOSタギング
- **文分割戦略の選択がモデル性能に大きく影響**
- 入力形式（翻字 vs Unicode記号）がモデル性能に影響

### コンペへの適用
- 文分割の重要性はコンペにも直結（train=doc-level, test=sent-level問題）
- 追加データ活用時のフォーマット不整合に注意

---

## 6. 2025 Transfer Learning論文

**Translating Akkadian Transliterations to English with Transfer Learning**
- 掲載: ICAART 2025 (SciTePress)
- URL: https://www.scitepress.org/Papers/2025/132572/132572.pdf

### 手法
- **mBART-50 many-to-many** のfine-tuning
- ORACC由来 約20,000行で学習

### 結果
- BLEU **34.09**（Gutherz 2023の37.47には未到達）

### コンペへの適用
- 多言語モデルの転移学習はアプローチの一つだが、ByT5ベースの現行手法のほうが有望
- mBART-50にはアッカド語が事前学習に含まれない点が弱み

---

## 7. Jones & Mitkov (2025) — Transformer評価

**Evaluating the Performance of Transformers in Translating Low-Resource Languages through Akkadian**
- 掲載: R2LM 2025 (ACL Anthology)
- URL: https://aclanthology.org/2025.r2lm-1.5.pdf

### 概要
- 低リソースMTにおけるTransformerアーキテクチャの性能評価
- ETCSL、CDLI、ORACCのデータ活用
- ATF（Annotated Text Format）形式への対応が鍵

---

## 外部リソース

### ORACC (Open Richly Annotated Cuneiform Corpus)
- URL: https://oracc.museum.upenn.edu/
- rich annotation + open licensingの楔形文字コーパス
- 検索・注釈・グロッサリーあり
- Gutherz 2023の学習データの源泉

### eBL (electronic Babylonian Library)
- 約25,000 tablets、35万行超の翻字
- 辞書 + コーパス + API提供
- retrieval augmentation / 辞書lookup の母体として有用

### Akkademia / Babylonian Engine
- 楔形文字→翻字→翻訳の全体パイプライン
- Gordin et al.のグループが開発

---

## コンペ向け統合知見

### 先行研究 vs コンペの条件差

| 項目 | Gutherz 2023 | Deep Past (Kaggle) |
|------|-------------|-------------------|
| データ量 | 50,544文 (train) | 1,561文書 (≈数千文) |
| 時代 | Neo-Assyrian中心 | **Old Assyrian** |
| ジャンル | 王碑文・行政書簡・混合 | **商業文書（契約・書簡・借金証書）** |
| 入力形式 | 翻字 or 楔形文字 | 翻字のみ |
| 粒度 | 文レベル | train=doc, test=**文レベル** |
| 評価 | BLEU4 | **√(BLEU × chrF++)** |
| ベストスコア | BLEU4=37.47 | 現時点トップLB: ~35.3 |

### 文献から得られる実装指針

1. **短文が有利**: Gutherz 2023で確認。コンペのtestも文レベル → 文分割学習が重要
2. **固有名詞処理**: 最大のボトルネック。OA_Lexiconやonomasticonの活用が鍵
3. **ドメイン特化**: 先行研究のNeo-Assyrian偏りがコンペでは不利。Old Assyrian商業文書への適応必要
4. **BPE vs バイトレベル**: Gutherz 2023はBPE 1K/10K、コンペ上位はByT5。低リソースではバイトレベルが優位
5. **ハルシネーション対策**: 先行研究で24%のハルシネーション率。MBRデコーディングやreranking等の対策が有効
6. **形態的多義性**: 単一解析になるのは20%のみ（BabyFST）→ 文脈依存の翻訳が本質的に必要
7. **辞書・注釈リソース**: ORACC/eBLを補助損失、retrieval、reranking等に活用する設計が研究レベルでも推奨

### 未探索だが有望なアプローチ（文献ベース）

| # | アプローチ | 根拠 | 難易度 | 検証状況 |
|---|-----------|------|--------|---------|
| A | **入力にlemma/辞書情報を付加** | BabyFST: 80%の語が多義。後処理でなく前処理で辞書を使う | 中 | 未検証 |
| B | **定型パターンのテンプレート活用** | Gutherz 2023: 定型ジャンルほど高スコア。OA商業文書は定型的 | 中 | 未検証 |
| C | **BPE tokenization** | Gutherz 2023: BPE 1K/10Kで37.47。音節文字はBPEと相性が良い可能性 | 中 | 未検証 |
| D | Retrieval-Augmented Translation | eBL/ORACC辞書をretrievalソースに | 高 | 未検証 |
| E | 多段パイプライン（分割→解析→翻訳） | Gutherz 2023, Gordin 2020 | 高 | 未検証 |
| F | 形態素情報の補助タスク | Smidt 2024, Sahala 2020 | 中 | 未検証 |
| G | ORACC追加データでの事前学習 | Gutherz 2023のデータソース | 中 | 未検証 |
| H | Translation Memory + NMTのハイブリッド | Gutherz 2023でTM baseline=23.51 | 低 | exp007で限定的 |
| I | **CNNアーキテクチャ (fconv)** | Gutherz 2023: 低リソースでTransformerより学習効率が良い | 中 | 未検証 |
| J | **楔形文字Unicode追加入力（マルチビュー学習）** | Gutherz 2023: C2E≈T2E。2表現から学習できる可能性 | 高 | 未検証 |

### 文献知見の詳細メモ

**A. 入力にlemma/辞書情報を付加（BabyFST → 前処理応用）**
- BabyFST(Sahala 2020): 80%の語が複数の形態解析を持ち、単一解析は20%のみ
- exp007でOA_Lexicon後処理は中立だった → 後処理ではなく**入力側に辞書引き結果を付加**する発想
- 例: `ta-aq-bi-a-am` → `ta-aq-bi-a-am [qabûm:speak]` のようにlemma情報をインライン付加
- 「人間の研究者が辞書を見ながら読む」をモデルに再現（ChatGPT会話の方向性）

**B. 定型パターンのテンプレート活用（Gutherz 2023 → ドメイン応用）**
- Gutherz 2023: 「定型ジャンルほどスコアが高い」
- Old Assyrian商業文書は高度に定型的（契約冒頭句・結語・証人リスト・Seal文書）
- テンプレート+穴埋め方式: `Seal of X, son of Y` のようなパターンを明示的にモデルに教える
- eda004で発見した「Seal文書の定型パターン」と接続

**C. BPE tokenization（Gutherz 2023 → トークナイゼーション再考）**
- Gutherz 2023: 翻字側BPE 1K語彙 / 英語側10K語彙でBLEU4=37.47
- アッカド語は音節文字 → BPEの分割単位が音節境界と合致しやすい
- ByT5のバイトレベルは汎用的だが、アッカド語の構造を活かせていない可能性
- ByT5 + BPE比較、またはsentencepieceベースのモデル検討

**I. CNNアーキテクチャ fconv（Gutherz 2023 → アーキテクチャ再考）**
- 「低リソース・文字レベルMTではTransformerより学習が速い」と明言
- 1,561件という極小データではCNNのほうが過学習しにくい可能性
- Fairseq fconvは実装が容易

**ハルシネーション分析（Gutherz 2023 補助資料）**
- 内在的（ソース改変）vs 外在的（ソース無視）の2種、人間評価で率24%
- 我々の「繰り返し問題 80.9%」は外在的ハルシネーションの一種
- 論文で**短文化が最も効果的な対策** → 文レベル学習の追加根拠

---

*調査日: 2026-03-08*
*ソース: ChatGPT共有会話 + 各論文の直接参照*
https://chatgpt.com/share/69ad2190-d420-800f-8c5a-6aa40caafd70
