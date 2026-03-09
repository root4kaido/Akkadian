# EDA011: UUID Matching Investigation

## 分析の目的

train.csv (1561文書) と Sentences_Oare_FirstWord_LinNum.csv の間で、UUID直接マッチが253件(16.2%)しかない原因を調査する。

## 分析概要

3つのファイル間のID関係を体系的に調査:
- train.csv (1561文書, oare_id)
- Sentences_Oare_FirstWord_LinNum.csv (9782行, 1700ユニークtext_uuid)
- published_texts.csv (7953行, 7949ユニークoare_id)

## 発見事項

### 1. UUIDフォーマットは完全に同一
- 全ファイルで標準UUID v4フォーマット（36文字、`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`）
- train: 1561/1561 (100%) 有効UUID
- Sentences_Oare text_uuid: 9782/9782 (100%) 有効UUID
- published_texts: 7953/7953 (100%) 有効UUID
- **フォーマットの違いが原因ではない**

### 2. train ↔ published_texts: 完全一致
- train oare_ids 1561件中 **1561件が published_texts に存在 (100%)**
- trainの文書は全てpublished_textsのサブセット

### 3. Sentences_Oare ↔ published_texts: 83.4%一致
- Sentences_Oare のユニークtext_uuid 1700件中 **1417件が published_texts に存在 (83.4%)**
- 283件はpublished_textsにも存在しない（別のデータソース由来？）

### 4. 根本原因: 異なるサブセット
```
published_texts: 7949 texts (全体)
├── train.csv:        1561 texts (20%)
├── Sentences_Oare:   1700 texts (21%)  ← 1417がpublished内
└── 重複:              253 texts (train ∩ Sentences)
```

**train と Sentences_Oare は published_texts の largely DIFFERENT subsets**。
- trainは7949件中1561件を使用
- Sentences_Oareは7949件中1417件をカバー（+283件は外部）
- 重複はわずか253件

### 5. Sentences_Oare の他カラムにもマッチなし
- display_name, sentence_uuid, その他全カラム: train oare_ids とのマッチ 0件
- text_uuid のみが253件マッチ

### 6. 253件のUUIDマッチも翻訳の質にばらつき
翻訳テキストの類似度（SequenceMatcher、先頭500文字）:

| 類似度閾値 | 件数 |
|-----------|------|
| >= 0.95   | 53   |
| >= 0.9    | 65   |
| >= 0.8    | 84   |
| >= 0.7    | 100  |
| >= 0.5    | 133  |
| >= 0.3    | 163  |
| < 0.3     | 90   |

- 平均類似度: 0.544
- 1文のみのテキスト (55件) は平均類似度 0.282（多くが断片的・省略された翻訳）
- 2文以上のテキスト (198件) は平均類似度が高い

### 7. テキスト内容ベースのマッチングも効果薄
- 先頭30文字一致: +1件
- 先頭文の部分一致: +1件
- published_texts transliteration経由: +0件
- **合計: 253 → 254件 (16.3%)、実質的に増えない**

### 8. 高品質マッチの実数
- UUID一致 AND 2文以上 AND 類似度>=0.5: **123件**
- これが実用的にsentence splittingに使える文書数

### 9. Sentences_Oareの文数分布
| 文数 | テキスト数 |
|------|-----------|
| 1文  | 271       |
| 2-5文 | 836      |
| 6-10文 | 364     |
| 10文超 | 229     |
| 平均  | 5.8文    |

## 結論

**253/1561の低マッチ率は、IDフォーマットの問題ではなく、2つのファイルがpublished_textsの異なるサブセットをカバーしているため。** テキスト内容ベースのマッチングも追加で1-2件しか見つからず、この結論を裏付ける。

## 精度向上への示唆

1. **Sentences_Oare によるsentence splitting は最大253文書（実用的には123文書）にしか適用できない**
2. 残り84%のtrain文書には**ルールベースのsentence splitting**（ピリオド区切り、コロン区切り等）が必要
3. Sentences_Oareの1447件のtrain外テキストは**追加の学習データ**として利用可能（sentence-level翻訳ペア）
4. UUID一致しても翻訳テキストが異なるケースが多い → Sentences_Oareの翻訳はtrainと**異なるアノテーションパイプライン**由来の可能性
