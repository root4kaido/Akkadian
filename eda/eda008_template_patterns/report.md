# eda008: OA商業文書の定型パターン分析

## 目的
Gutherz 2023「定型ジャンルほどスコアが高い」を受けて、train.csvの翻訳テキストから定型構造を抽出・分類し、テンプレート化による精度向上の可能性を調査する。

## 分析概要
- train.csv 1,561文書 → 6,878文に分割して分析
- 英語翻訳・アッカド語翻字の両方で文頭パターンを抽出
- 文書タイプの自動分類、定型度の定量評価を実施

## 発見事項

### 1. 明確な定型パターンが存在する

**英語翻訳の頻出文頭（5-word prefix）:**

| 出現数 | パターン | 意味 |
|--------|---------|------|
| 45x | "Reckoned from the week of" | 支払期限の起算（債務文書） |
| 42x | "If he has not paid" | 支払遅延条件（利息条項） |
| 34x | "The Kanesh colony gave us" | 植民地からの命令/証言（法的手続き） |
| 19x | "If he does not pay" | 支払条件（バリエーション） |
| 12x | "My dear brothers, take care" | 書簡の懇願定型句 |

**アッカド語翻字の頻出パターン:**

| 出現数 | パターン | 対応する英語 |
|--------|---------|------------|
| 24x | "a-na a-lá-ḫi-im qí-bi-ma" | "Say to Ali-ahum" （書簡冒頭） |
| 12x | "1 ma-na kù.babbar" | "1 mina of silver" |
| 12x | "um-ma ša-lim-a-šùr-ma a-na" | "Thus Šalim-Aššur to..." |
| 10x | "10 gín kù.babbar" | "10 shekels of silver" |

### 2. 文書タイプの分布

| タイプ | 件数 | 割合 | 定型度 |
|--------|------|------|--------|
| Debt/Loan（借金・貸付） | 594 | 38.1% | **高** — 金額+債務者+期限+利息条項の定型構造 |
| Other/Unknown | 478 | 30.6% | 不明 |
| Envelope/Tablet | 380 | 24.3% | 中 |
| Contract/Agreement | 220 | 14.1% | 高 |
| Shipping/Transport | 219 | 14.0% | 中 |
| Seal | 172 | 11.0% | **極高** — "Seal of X, son of Y" |
| Receipt | 153 | 9.8% | 高 |
| Letter/Message | 114 | 7.3% | **冒頭のみ高**、本文は自由記述 |
| Legal/Court | 59 | 3.8% | 高 |

※35.6%が複数タイプに分類される（例: Debt + Contract）

### 3. 定型度の定量評価

| 指標 | 値 |
|------|-----|
| 定型文(formulaic, 5回以上一致) | 4.3% |
| 半定型(semi-formulaic, 2-4回一致) | 14.4% |
| 自由記述(free) | 77.2% |
| 短文(5語未満) | 4.1% |
| 定型文に属するトークンカバレッジ | **19.8%** |
| >50%定型の文書 | 9.4% |
| 100%定型の文書 | 5.4% |

**結論: 全体の約20%のトークンが定型パターンでカバーできる。**

### 4. テンプレート化が有効な3つの構造

#### A. 債務文書テンプレート（最も定型的、38.1%の文書に該当）
```
{NUM} mina(s) of {refined} silver {DEBTOR} owes to {CREDITOR}.
Reckoned from the week of {EPONYM}, month of {MONTH}, in the eponymy of {EPONYM2},
he will pay in {NUM} weeks.
If he has not paid in time, he will add interest at the rate {RATE} shekel per mina per month.
Witnessed by {W1}, by {W2}, by {W3}.
Seal of {S1}, {S2}.
```

#### B. 書簡テンプレート（冒頭・結語が定型的）
```
[冒頭] Say to {RECIPIENT}, thus {SENDER}:
[冒頭] To {RECIPIENT} from {SENDER}:
[書き出し] My dear brother(s), ...
[懇願] Take care to/so ...
[結語] Send me word about ...
```

#### C. 法的手続きテンプレート
```
The {COLONY} colony gave us for these proceedings
and we gave our testimony before Aššur's dagger.
```

### 5. 数値・単位パターン

翻訳全体で非常に高頻度:
- shekels: 1,539回 / minas: 1,454回 / mina: 853回
- 「{NUM} mina(s)/shekel(s) of (refined) silver」が商業文書の基本構造
- `kù.babbar` (翻字) → "silver" (英語) の対応が安定

### 6. Witness/Sealセクション

- 全文書の**22.2%**にwitness条項が含まれる
- 構造: "Witnessed by {NAME}, by {NAME}, by {NAME}." が基本形
- Sealも同様: "Seal of {NAME}, son of {NAME}."
- 固有名詞の数・構造は文書ごとに異なるが、枠組みは完全に定型

## 実験への示唆

### 直接活用できる知見

1. **定型テンプレートを学習データに付加する**
   - 債務文書テンプレートを分解して、構造学習用のデータとして追加
   - 「{NUM} ma-na kù.babbar → {NUM} mina(s) of silver」のようなサブ翻訳を追加学習

2. **構造化プロンプト/タグ**
   - 入力に文書タイプタグ（`<debt>`, `<letter>`, `<legal>`等）を付加
   - モデルが文書タイプごとの翻訳パターンを学習しやすくする

3. **数値・単位の規則ベース翻訳**
   - `gín` → "shekel(s)", `ma-na` → "mina(s)", `kù.babbar` → "silver" は完全に規則的
   - 後処理で数値+単位パターンを補正可能

4. **Witness/Seal部分の定型翻訳**
   - 固有名詞リスト（OA_Lexicon）と組み合わせて、Witness/Seal部分はほぼテンプレートで生成可能

### 限界

- **77%の文は自由記述** — テンプレートだけでは不十分
- 5-wordプレフィックスでは捕捉できない構造的類似性がある可能性
- 固有名詞のテンプレート化（{NAME}置換）が不十分（大文字始まりのみで検出）
