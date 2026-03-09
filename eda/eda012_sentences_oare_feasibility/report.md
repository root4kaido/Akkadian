# eda012: Sentences_Oare + published_texts 結合による追加データ構築の実現可能性調査

## 分析の目的

Sentences_Oare（文レベル翻訳）とpublished_texts（翻字）を結合して、trainに存在しない追加の翻字-翻訳ペアを構築できるか調査する。

## 分析概要

- Sentences_Oare.text_uuid → published_texts.oare_id でUUID結合
- カバレッジ、transliteration有無、文レベル対応付けの可能性を調査
- doc-levelペアと1文ペアの品質をサンプル確認

## 発見事項

### 1. UUID結合: 高カバレッジで成功

| 指標 | 値 |
|------|-----|
| Sentences_Oare ユニーク文書数 | 1,700 |
| published_textsにマッチ | 1,417 (83.4%) |
| うちtrain外（新規） | 1,164文書 |
| 新規でtransliterationあり | **1,166文書 (100%)** |

→ **UUID結合は問題なく機能する。新規1,166文書の全てにtransliterationが存在。**

### 2. 構築可能なペア

| ペアの種類 | 数 | 備考 |
|-----------|-----|------|
| A) doc-level（文分割なし） | **1,166ペア** | train.csvと同じ形式。即座に使用可能 |
| B) 1文文書（文レベル） | 146ペア | 分割不要 |
| C) 複数文文書の分割 | ~1,018文書 | line_number + first_word_spellingで分割可能性あり |

- train 1,561件に対し+1,166件 = **+74.7%のデータ増加**

### 3. doc-levelペアの品質

| 指標 | 新規ペア | train |
|------|---------|-------|
| transliteration長 (mean) | 566 | 429 |
| translation長 (mean) | 459 | 500 |
| 文書あたり文数 (mean) | 6.2 | - |

→ transliterationはtrainより長め、translationはやや短い。全体的にtrainと同等のドメイン。

### 4. 1文文書ペアの品質（サンプル確認）

- 多くが **Seal/Witness記述のみ** の短文（定型的な封筒テキスト）
- 例: 「Seal of Ennam-Suen s. Aššur-mālik...」「Witnesses: Buzuzum, Tūrāya」
- 翻字-翻訳の対応は正確だが、**翻訳が1文のみ＝本体ではなく末尾部分のみの可能性が高い**
- 品質にばらつきあり。一部は空（1バイト）のtranslationも存在

### 5. 複数文文書の分割可能性

- Sentences_Oareにはline_number (非null率75%), first_word_spelling (37.5%), sentence_obj_in_text (100%)あり
- **sentence_obj_in_text** で文の順序は確定可能
- ただし **transliterationの文分割方法が最大の課題**: アッカド語翻字には文区切り記号がない（eda003確認済み）
- line_numberが文の開始行に対応するなら、published_textsの翻字をline_numberで分割できる可能性があるが、published_textsの翻字にはline区切り情報がない（全て1行の連続テキスト）

### 6. AICC機械翻訳の追加データ

| 指標 | 値 |
|------|-----|
| train外でtransliteration+AICC_translationあり | 6,141文書 |
| Sentences_Oareにもない文書でAICC_translationあり | 5,004文書 |

→ Sentences_Oareを超える量だが品質未検証。noisy labelとしての活用は別途検討。

## 結論

### すぐにできること（doc-level追加）
- **1,166件のdoc-levelペアは即座に構築・学習投入可能**
- Sentences_Oareの全文translationを結合 + published_textsのtransliteration
- train 1,561件 → 2,727件 (+74.7%)
- 実装コスト: 低（preprocess.pyでpublished_textsとSentences_Oareを読み込み、結合するだけ）

### 課題・注意点
- Sentences_Oareのtranslationは文ごとに格納されているが、結合後はdoc全体の翻訳として使う必要がある
- 文の順序はsentence_obj_in_textで保証可能
- **ただし、Sentences_Oareの翻訳とpublished_textsの翻字は異なるアノテーションパイプライン由来（eda011: 類似度0.544）** → 品質にばらつきの可能性
- 1文文書は封筒・Witness等の断片的テキストが多く、質が低い可能性

### 文レベル分割は困難
- アッカド語翻字に文区切りがなく、published_textsの翻字は全て連続テキスト
- 文分割には高度なアライメントが必要（line_number情報だけでは不十分）
- **doc-levelで追加し、既存のlabel maskingで対応するのが現実的**
