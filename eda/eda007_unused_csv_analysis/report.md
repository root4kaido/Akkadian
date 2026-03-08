# EDA007: 未使用CSVファイルの活用可能性調査

## 分析の目的
train.csv / test.csv 以外のCSVファイル（6種 + 外部データ1種）がどう活用できるかを、
データ内容・train/testとの突合・ディスカッション/ノートブックでの実使用例の3方面から分析する。

## 分析概要

### 対象ファイル一覧

| ファイル | 行数 | 現状 |
|---------|------|------|
| Sentences_Oare_FirstWord_LinNum.csv | 9,782 | 未使用 |
| OA_Lexicon_eBL.csv | 39,332 | 未使用 |
| eBL_Dictionary.csv | 19,215 | 未使用 |
| published_texts.csv | 7,953 | 未使用 |
| publications.csv | 240,510 | 未使用 |
| resources.csv | 292 | 未使用 |
| onomasticon.csv (外部) | 未取得 | 未使用 |

---

## 発見事項

### 1. Sentences_Oare_FirstWord_LinNum.csv — ★★★ 最重要

**概要**: 9,782件の文レベル翻訳データ（1,700文書由来）

| 指標 | 値 |
|------|-----|
| translation非空率 | 99.9% (9,771/9,782) |
| translation平均文字数 | 73.6 |
| trainと一致する文書 | 253/1,561 (16.2%) |
| trainと一致する文書の文数 | 1,213 |
| **trainに無い文書の文数** | **8,564 (1,447文書)** |
| **うちtranslation非空** | **8,564** |
| testと一致 | 0 (testは短縮IDのため突合不可) |

**重大な発見**:
- **英語翻訳付きの文レベルデータが8,564件もtrainに含まれていない**
- ただし**transliterationカラムが存在しない**（翻訳のみ）
  - `first_word_spelling` と `line_number` で原文テキストとの対応を取る必要がある
- trainと一致する253文書では、Sentences_Oareの文翻訳を結合するとdoc→sentの分割が可能

**1文書あたりsentence数**: mean=5.8, median=4.0, max=201

**first_word_spelling上位**: `ma-na`(941), `a-na`(675), `iš-tù`(619), `šu-ma`(548), `um-ma`(536)

**ディスカッション/ノートブックでの言及**:
- Data Update (pinned) でRyan Holbrookが公開を告知
- starter-train.ipynbで「simple sentence alignment」として言及（但し具体実装は文レベル分割のみ）

**活用案**:
1. **train.csvの文書をsentence分割するためのガイド** → trainの253文書について、Sentences_Oareの文境界情報（first_word_spelling + line_number）を使ってdoc-level → sent-level分割を実現
2. **published_texts.csvとの組み合わせ** → Sentences_Oareの1,447文書（train外）の翻字をpublished_textsから取得し、翻訳ペアを構築

### 2. OA_Lexicon_eBL.csv — ★★★ 重要（後処理で活用済みNB有）

**概要**: 39,332件の古アッシリア語語彙データベース

| type | 件数 | 割合 |
|------|------|------|
| word | 25,574 | 65.0% |
| PN (固有名詞) | 13,424 | 34.1% |
| GN (地名) | 334 | 0.8% |

| 指標 | 値 |
|------|-----|
| form ユニーク数 | 35,048 |
| norm ユニーク数 | 19,662 |
| lexeme ユニーク数 | 6,353 |
| 3+形態を持つlexeme | 1,955 |
| train翻字マッチ率 | 43.3% |
| test翻字マッチ率 | 33.0% |
| Alt_lex非空 | 886件 |

**ノートブックでの実使用** (`dpc-infer-with-post-processing-by-llm` by takamichitoda, 193 votes):
- OA_Lexiconを**後処理の辞書ベース固有名詞正規化**に使用
- `form`, `norm`, `Alt_lex` からtoken→lexemeインデックスを構築
- ダイアクリティクス除去 + fold_for_match で表記揺れを吸収
- 翻訳中の固有名詞のスペルをLexiconの正規形に修正
- **Translation Memory**（trainデータの完全一致）との併用

**表記揺れの例**:
```
Abela: a-pì-la, a-pì-lá, a-bi₄-lá, a-bi-la, a-bi-lá, a-be-la, ...
Abatanānu: a-ba-ta-na-nim, a-ba-ta-na-nu-um, a-pá-ta-na-nim, ...
```

**活用案**:
1. **後処理での固有名詞修正**（takamichitodaのアプローチを踏襲）
2. **学習データの前処理** → 翻字の正規化（form→normマッピング）で表記揺れを統一
3. **固有名詞タグの付与** → PN/GNをタグ（`<PN>xxx</PN>`等）で囲み、モデルの固有名詞認識を補助

### 3. eBL_Dictionary.csv — ★★ 参考価値（新情報あり）

**概要**: 19,215件のアッカド語辞書（word→英語definition）

| 指標 | 値 |
|------|-----|
| definition非空率 | 79.9% (15,344件) |
| definition平均長 | 72.5文字 |
| derived_from非空 | 55.7% |
| train翻字との部分マッチ | 6-8/10トークン |

**ディスカッション言及**:
- 「0.78% Train Transliterations Found in eBL_Dictionary?」 — wordカラムとの直接マッチ率はわずか0.78%
  - **原因**: eBLは辞書形（語幹）、trainの翻字は活用形（音節表記）→ マッチしにくい
  - コメント: 「翻字システムが異なる。文法的文脈で形が変わる。eBLで手動検索すれば見つかる」

- **[NEW] 「eBL text divergence」（2026/03/07投稿, David Ochoa Corrales）**
  - **eBL→OARE Mimation Restorer**: eBL辞書のStandard Babylonian語形をOld Assyrian語形に変換するスクリプト
  - Old Assyrianはproto-semitic mimation（語末の"-m"）を保持、Standard Babylonianは失っている
  - 例: eBL `šakānu I` → OARE `šakānum`
  - **585ペアの検証で92.8%の精度**
  - これによりeBL Dictionaryの直接マッチ率が大幅改善する可能性

**活用案**:
1. **Mimation Restorerを適用してeBL→OAREの変換後、辞書引き** — マッチ率が0.78%→大幅改善の可能性
2. **word→definitionを使った知識注入** — T5のpre-trainingやprompt補助に利用可能だが、効果は不透明
3. **OA_Lexiconとの組み合わせ** — OA_Lexiconのlexeme→eBLのwordで辞書引き

### 4. published_texts.csv — ★★★ 重要（追加データ源泉）

**概要**: 7,953件のアッカド語テキスト（翻字付き、人手翻訳なし）

| 指標 | 値 |
|------|-----|
| transliteration非空率 | 100% (7,953件) |
| transliteration平均文字数 | 458.8 |
| trainとの一致 | 1,561/1,561 (**100%**) |
| **train/test外のテキスト** | **6,388件** |
| AICC_translation非空 | 7,702 (97%) |

**重大な発見**:
- **trainの全1,561テキストはpublished_textsに含まれる**（完全包含）
- **AICC_translation**: 7,702件の**機械翻訳**が付いている
  - 品質は不明だが、弱い教師信号として活用可能
- **6,388件の追加翻字テキスト**（翻訳なし）が存在
- `transliteration_orig`（原始版）と`transliteration`（正規化版）の2カラム

**genre_label分布**: unknown(4,046), letter(2,261), debt_note(527), note(218), ...

**Sentences_Oareとの組み合わせ**:
- Sentences_Oareの1,447文書（train外）のtext_uuidがpublished_textsのoare_idと突合可能
- → published_textsの翻字 + Sentences_Oareの文翻訳 = **新たな翻訳ペア**

**活用案**:
1. **Sentences_Oareとの結合** → 追加翻訳ペアの構築（最大8,564文）
2. **AICC_translationをnoisy labelとして活用** → 機械翻訳品質のフィルタリング付きで学習データに追加
3. **翻字のみのデータでの自己教師あり学習** → Denoising autoencoderやMLMで翻字の表現学習

### 5. publications.csv — ★ 活用困難

**概要**: 240,510行の学術論文OCRテキスト

| 指標 | 値 |
|------|-----|
| 全行数 | 240,510 |
| has_akkadian=True | ~0.2% (先頭5000行中10件) |
| bibliography.csvとの一致 | 18 PDF |

**実態**: ほぼ英語の学術テキスト。アッカド語含有ページは極めて少ない。

**活用案**:
- has_akkadian=Trueのページからアッカド語翻字＋英語翻訳ペアを抽出する試みは可能だが、OCR品質やフォーマットの問題で**コスパが悪い**
- Hostがpublications.csvのOCR品質問題を認め、PDF版を別途公開済み

### 6. resources.csv — ★ 直接的価値なし

**概要**: 292件の学術論文・プロジェクトの書誌情報

Topics上位: OCR(44), Tablet digitization(20), Cuneiform analysis(18)
Methods上位: Neural(9), Statistical(7), 3D scanning(6)

**活用案**: 関連論文のリサーチガイドとしてのみ有用。学習データには使えない。

---

## 総合評価と活用優先度

| ファイル | 活用可能性 | 優先度 | 主な活用方法 |
|---------|-----------|--------|------------|
| **Sentences_Oare** | ★★★ | **極高** | 文アライメント + 追加翻訳ペア |
| **published_texts** | ★★★ | **極高** | Sentences_Oareとの結合 + AICC翻訳 |
| **OA_Lexicon** | ★★★ | **高** | 後処理の固有名詞正規化 |
| eBL_Dictionary | ★★ | 中 | OA_Lexiconの補助辞書 |
| publications | ★ | 低 | コスパ悪い |
| resources | ★ | 低 | 論文リサーチのみ |

---

## 精度向上の仮説

### 仮説1: Sentences_Oare + published_textsで追加学習データ構築 (+2-5pt?)
- published_textsから翻字を取得 + Sentences_Oareから文翻訳を取得
- first_word_spelling + line_numberで翻字内の文境界を特定
- 最大8,564件の文レベル翻訳ペアを追加
- **testが文レベルであるため、文レベル学習データは特に有効**

### 仮説2: OA_Lexiconによる後処理の固有名詞修正 (+0.5-1pt?)
- takamichitodaのノートブック（193 votes）で実証済みのアプローチ
- 翻訳中の固有名詞のスペルを辞書の正規形に修正
- Host直言「固有名詞が最大のボトルネック」への対策

### 仮説3: AICC_translationをnoisy training dataとして活用 (+0-2pt?)
- 7,702件の機械翻訳（品質不明）
- 品質フィルタリング（BLEUスコアで足切り等）後に学習データに追加
- リスク: 低品質データ混入でモデル性能悪化の可能性
