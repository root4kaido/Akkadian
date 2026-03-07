# EDA003: Sentences_Oareの活用可能性 + train.csvの文分割手がかり調査

## 分析の目的

テストデータは文レベル（約4,000文）だがtrain.csvはドキュメントレベル（1,561テキスト）。
このドメインミスマッチを解消するために:
1. Sentences_Oare.csvから文レベルの学習データを作れるか
2. train.csvを文レベルに分割できるか

## 分析概要

### データ構造

**Sentences_Oare.csv** (9,782行, 1,700ユニークテキスト)

| カラム | 内容 | 欠損 |
|--------|------|------|
| display_name | テキスト名 " (HS 2931)" 形式 | 0 |
| text_uuid | テキストUUID | 0 |
| sentence_uuid | 文UUID（全件ユニーク） | 0 |
| sentence_obj_in_text | テキスト内の文番号 | 0 |
| **translation** | **英語翻訳（文レベル）** | 10 |
| first_word_transcription | 文頭語の転写 | **1,247** |
| first_word_spelling | 文頭語のスペリング（翻字） | 0 |
| first_word_number | 文頭語の番号 | 0 |
| line_number | 行番号 | 0 |
| side / column | タブレット面・列 | 0 |

**重要**: transliteration（翻字全文）カラムは**存在しない**。英語翻訳と文頭語情報のみ。

**train.csv** (1,561行)

| カラム | 内容 |
|--------|------|
| oare_id | UUID形式のID |
| transliteration | アッカド語翻字（ドキュメント全体） |
| translation | 英語翻訳（ドキュメント全体） |

**published_texts.csv** (7,953行)

| 主要カラム | 内容 |
|------------|------|
| oare_id | UUID形式のID |
| transliteration | アッカド語翻字 |
| AICC_translation | AICC機械翻訳 |

### テーブル間の連携

```
train.csv (oare_id) ←→ published_texts.csv (oare_id): 100%一致（1,561件全て）
Sentences_Oare (text_uuid) ←→ published_texts: text_uuidカラムなし → 直接結合不可
Sentences_Oare (display_name) → published_texts (label): 部分一致で結合可能（要加工）
```

- train.csvの全1,561件がpublished_textsに存在する
- Sentences_Oareとpublished_textsの結合は`display_name`→`label`の部分一致で可能だが、完全自動化は困難
  - 例: Sentences_Oare `"(HS 2931)"` → published_texts label `"Cuneiform Tablet HS 2931"` で検索可能
  - ただし一部マッチしないケースあり（`"Adana 237s"` → no match）

### train.csvの文分割手がかり

| 区切り候補 | 出現率 | 実用性 |
|-----------|--------|--------|
| 改行 | 0% | 使えない |
| コロン(:) | 0% | 使えない |
| セミコロン(;) | 0% | 使えない |
| 二重スペース | 0% | 使えない |
| ピリオド(.) | 83.3% | **数字表記・略語の一部であり、文区切りではない** |
| <gap> | 42.9% | 文区切りではない |

**アッカド語翻字には明示的な文区切り記号がない。**

### 英語翻訳の文分割

| 統計 | 値 |
|------|---|
| 推定文数(平均) | 4.4文/ドキュメント |
| 推定文数(中央値) | 3文 |
| 1文のみ | 370件 (23.7%) |
| 2-5文 | 759件 |
| 6-10文 | 327件 |
| 11文以上 | 105件 |

### Sentences_Oareの文頭語（first_word_spelling）

| スペリング | 出現回数 | 転写 | 意味の手がかり |
|-----------|---------|------|--------------|
| ma-na | 941 (9.6%) | manā | 「マナ」（重量単位） |
| a-na | 675 (6.9%) | ana | 「～に」（前置詞） |
| iš-tù | 619 (6.3%) | ištu | 「～から」（前置詞） |
| šu-ma | 548 (5.6%) | šumma | 「もし」（条件節） |
| um-ma | 536 (5.5%) | umma | 「曰く」（引用導入） |
| IGI | 471 (4.8%) | maḫar | 「～の前で」 |
| KIŠIB | 294 (3.0%) | kunuk | 「印章」 |

これらの語がtransliteration内に頻出するが、文頭以外にも出現するため（例: a-naは7,630回出現、1,443テキストに含まれる）、単純なキーワードマッチでの文分割は困難。

### Sentences_Oareの翻訳品質

- 9,782件中、空/nan: 10件のみ
- 翻訳長: mean=74文字, median=61文字（文レベルなので短い）
- <gap>含有: 0件（train.csvの翻訳にはある）
- non-ASCII含有: 60.6%（アッカド語固有名詞のdiacritics: Šalim-aḫum等）
- ユニーク翻訳: 9,075/9,782（重複少ない）

## 発見事項

### 主要な発見

1. **Sentences_Oareにはtransliteration（翻字）がない** — 英語翻訳と文頭語情報のみ。そのまま学習データとしては使えない

2. **train.csvとSentences_Oareの直接結合は困難** — train.csvはoare_id、Sentences_Oareはtext_uuidで、published_textsにtext_uuidカラムがないため連結できない。display_name→label経由の部分一致は可能だが信頼性が低い

3. **アッカド語翻字には文区切り記号がない** — 改行・コロン・セミコロンは0%、ピリオドは数字表記の一部

4. **英語翻訳のピリオド分割は可能**だが、対応するアッカド語の分割点が特定できない

5. **文頭語（first_word_spelling + line_number）による分割**が最も有望
   - Sentences_Oareは各文の「first_word_spelling」と「line_number」を持つ
   - train.csvのtransliterationから文頭語を検索し、line_number情報と組み合わせれば分割位置を推定できる可能性
   - ただし同じ語が文中にも出現するため、位置情報の照合が必要

### 活用方法の現実的な選択肢

| 方法 | 難易度 | データ量 | 品質 |
|------|--------|---------|------|
| A) 英語翻訳のみで逆翻訳学習 | 低 | 9,782文 | 低（transliterationなし） |
| B) display_name経由でpublished_textsのtransliterationを取得→文レベルペア作成 | 中〜高 | 未知 | 中 |
| C) first_word_spelling + line_numberでtrain.csvを文分割 | 高 | ~6,800文(1,561×4.4) | 高（正確に分割できれば） |
| D) ドキュメントレベルのまま学習（現状維持） | 最低 | 1,561 | 確立済み |

## 精度向上の仮説

1. **方法B（published_texts経由）が最も現実的**: Sentences_Oareのtext_uuidに対応するpublished_textsのテキストを、display_name→label部分一致で特定し、transliterationと文レベルtranslationのペアを作成する。ただし結合精度の検証が必要。

2. **方法Cは高精度だが実装が複雑**: first_word_spellingの出現位置とline_number情報を使ってtransliterationを分割する。アッカド語学の知識が必要。

3. **テストが文レベルならドキュメントレベル学習でも問題ない可能性**: exp003でmax_length=512のtruncationが入っても実質的に文レベル相当の学習になっている。テスト入力は短い（推定~100-300文字）のでtruncationの影響なし。
