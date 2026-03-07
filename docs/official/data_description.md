# データ定義

> 合計10ファイル、600.78 MB、CSV形式、ライセンス: CC BY-SA 4.0

## 概要

8,000以上の古アッシリア楔形文字テキストの翻字（transliteration）とメタデータ。一部にアライン済み英語翻訳を提供。
約900件の学術論文のOCRテキストも含まれ、追加学習データの作成に利用可能。

**重要**: Code Competitionのため、`test.csv` はダミーデータ。スコアリング時に本番テストデータに置換される。

## メインファイル

### train.csv
約1,500件の古アッシリア語テキストの翻字＋英語翻訳（**ドキュメントレベル**でアライン）。

| カラム | 型 | 説明 |
|--------|-----|------|
| `oare_id` | str | [OARE](https://deeppast.org/oare) データベースの識別子。テキストを一意に特定 |
| `transliteration` | str | アッカド語の翻字テキスト |
| `translation` | str | 対応する英語翻訳 |

### test.csv
約4,000文（約400ドキュメント由来）。**文レベル**でアライン（trainとは異なる粒度に注意）。

| カラム | 型 | 説明 |
|--------|-----|------|
| `id` | int | 各文の一意な識別子 |
| `text_id` | str | 各ドキュメントの一意な識別子 |
| `line_start` | str | タブレット上の文の開始行番号（`1`, `1'`, `1''` 等） |
| `line_end` | str | タブレット上の文の終了行番号 |
| `transliteration` | str | アッカド語の翻字テキスト（→ `translation` を予測する） |

**注意**: `line_start` / `line_end` は str 型。`'`（破損行）や `''`（二重破損行）を含む。

### sample_submission.csv
正しい提出フォーマットのサンプル。

## 補足データ（Supplemental Data）

### published_texts.csv
約8,000件の古アッシリア語テキストの翻字 + メタデータ（翻訳なし）。

| カラム | 説明 |
|--------|------|
| `oare_id` | OAREデータベースID（train.csvと同一体系） |
| `online transcript` | [DPIウェブサイト](https://deeppast.org/oare)の翻字URL |
| `cdli_id` | [CDLI](https://cdli.earth/)のID（複数は`\|`区切り） |
| `aliases` | 他の出版ラベル（出版番号、博物館IDなど、`\|`区切り） |
| `label` | テキストの主要ラベル |
| `publication_catalog` | 出版物・博物館記録のラベル（`\|`区切り） |
| `description` | テキストの基本的な説明 |
| `genre_label` | ジャンルラベル（全テキストにはない） |
| `inventory_position` | 博物館内ラベル（`\|`区切り） |
| `online_catalog` | Yale CollectionのURL（CC-0メタデータ・画像） |
| `note` | 専門家によるコメント・翻訳ノート |
| `interlinear_commentary` | 特定行を議論する出版物の参照 |
| `online_information` | British MuseumのURL（著作権はBM） |
| `excavation_no` | 発掘時の識別子 |
| `oatp_key` | Old Assyrian Text ProjectのID |
| `eBL_id` | [eBL](https://www.ebl.lmu.de/library/)のID |
| `AICC_translation` | [初期オンライン機械翻訳](https://aicuneiform.com/)のURL（品質は低い） |
| `transliteration_orig` | OAREデータベースの元翻字 |
| `transliteration` | フォーマット提案に基づくクリーン版翻字 |

### publications.csv
約880件の学術論文のOCRテキスト（古アッシリア語→多言語翻訳を含む）。

| カラム | 説明 |
|--------|------|
| `pdf_name` | 元PDFファイル名 |
| `page` | ページ番号 |
| `page_text` | 記事テキスト |
| `has_akkadian` | アッカド語翻字を含むかどうか |

### bibliography.csv
publications.csvの書誌情報。

| カラム | 説明 |
|--------|------|
| `pdf_name` | publications.csvに対応するID |
| `title`, `author`, `author_place`, `journal`, `volume`, `year`, `pages` | 標準書誌データ |

### OA_Lexicon_eBL.csv
全古アッシリア語の翻字語と辞書的等価語のリスト。

| カラム | 説明 |
|--------|------|
| `type` | 語のタイプ（word, `PN`=人名, `GN`=地名 等） |
| `form` | 翻字のまま（string literal） |
| `norm` | ハイフン除去・母音長表示の正規化形 |
| `lexeme` | 辞書形（見出し語） |
| `eBL` | eBLオンライン辞書URL |
| `I_IV` | 同音異義語のローマ数字表記（CDA準拠） |
| `A_D` | 同音異義語のアルファベット表記（[CAD](https://isac.uchicago.edu/research/publications/chicago-assyrian-dictionary)準拠） |
| `Female(f)` | 女性ジェンダー表記 |
| `Alt_lex` | 代替正規化形 |

### eBL_Dictionary.csv
eBLデータベースの完全なアッカド語辞書。OA_Lexicon_eBL.csvの`eBL`URLが指す詳細データ。

### resources.csv
追加データとして利用可能なリソース一覧。

### Sentences_Oare_FirstWord_LinNum.csv
train.csvの文レベルアラインメント補助。各文の最初の単語とタブレット上の位置を示す。

## 追加学習データ構築の推奨ワークフロー

1. **各テキストと翻訳の特定**: ドキュメントID（ID, aliases, museum numbers）を使って、OCR出力内の翻字と翻訳をマッチング
2. **全翻訳を英語に変換**: 原文翻訳は多言語（英語、フランス語、ドイツ語、トルコ語等）→ 英語に統一
3. **文レベルアラインメント作成**: アッカド語翻字と英語翻訳を文ごとに分割・ペアワイズアライン

## 参考文献

- https://cdli.earth/publications
- https://cdli.ox.ac.uk/wiki/abbreviations_for_assyriology
