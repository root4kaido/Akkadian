# eda022: Host推奨前処理の網羅性監査

## 目的
exp022で実装したHost推奨前処理の網羅性を検証し、未対応パターンを洗い出す。

## 調査対象
- Host Discussion 678899 "A Stitch in Time" の推奨前処理
- Host Discussion 665209 "Two practical stumbling blocks" の仕様情報
- MPWAREが指摘した追加の不整合パターン

---

## 1. Translation前処理

### 1.1 除去対象

| 項目 | train内件数 | exp022対応 | 備考 |
|------|-----------|-----------|------|
| fem. | 0 | ✅ | Host更新で既に除去済み |
| sing. | 0 | ✅ | 同上 |
| pl. | 0 | ✅ | 同上 |
| plural | 5 | ✅ | exp022で除去 |
| (?) | 61 | ✅ | exp022で除去 |
| stray `..` | 0 | ✅ | Host更新で既に除去済み |
| stray x/xx | 7 | ✅ | exp022で除去 |
| `<< >>` | 0 | ✅ | Host更新で既に除去済み |

### 1.2 置換対象

| 項目 | train内件数 | exp022対応 | 備考 |
|------|-----------|-----------|------|
| PN → `<gap>` | 8 | ✅ | |
| -gold → pašallum gold | 4 | ✅ | |
| -tax → šadduātum tax | 29 | ✅ | |
| -textiles → kutānum textiles | 23 | ✅ | |
| `/` 代替翻訳 → 先頭選択 | 112 | ✅ | |

### 1.3 小数→分数変換

#### exp022の完全一致マッピング
```python
DECIMAL_TO_FRACTION = {
    "0.5": "½", "0.25": "¼", "0.3333": "⅓", "0.6666": "⅔",
    "0.8333": "⅚", "0.75": "¾", "0.1666": "⅙", "0.625": "⅝",
}
```

#### trainデータの小数分布（translation、小数部分別）

| 小数部分 | 件数 | exp022で変換 | あるべき分数 |
|---------|------|------------|------------|
| 0.5 | 744 | ✅ | ½ |
| 0.3333 | 209 | ✅ | ⅓ |
| 0.6666 | 138 | ✅ | ⅔ |
| **0.3332** | **123** | **❌** | ⅓ |
| 0.8333 | 55 | ✅ | ⅚ |
| **0.6665** | **51** | **❌** | ⅔ |
| **0.8332** | **27** | **❌** | ⅚ |
| **0.1665** | **10** | **❌** | ⅙ |
| 0.1666 | 7 | ✅ | ⅙ |
| 0.333 | 2 | ❌ | ⅓ |

Total: 1,366件中 1,153件が完全一致、**213件が丸め誤差で未変換**（15.6%）。

#### transliterationも同様
| 小数部分 | 件数 | exp022で変換 |
|---------|------|------------|
| 0.5 | 829 | ✅ |
| 0.3333 | 270 | ✅ |
| 0.6666 | 174 | ✅ |
| **0.3332** | **141** | **❌** |
| 0.8333 | 80 | ✅ |
| 0.25 | 57 | ✅ |
| **0.6665** | **57** | **❌** |
| **0.8332** | **28** | **❌** |
| 0.1666 | 18 | ✅ |
| 0.75 | 15 | ✅ |
| **0.1665** | **10** | **❌** |
| 0.2999 | 1 | ❌ |
| 0.333 | 1 | ❌ |
| 0.3000 | 1 | ❌ |

#### 対策
完全一致を**近似マッチ**に変更。小数部分を最も近い既知分数にマッピング（許容誤差0.002以内）。

**注意**: eda022のanalyze.pyでのカウント（approx=5件）は手動カウント（213件）と矛盾。analyze.pyにバグがある可能性あり（要修正）。

### 1.4 ローマ数字→整数（月名）

#### exp022の実装
```python
for roman, integer in sorted(ROMAN_TO_INT.items(), key=lambda x: -len(x[0])):
    text = re.sub(rf'\bmonth\s+{roman}\b', f'month {integer}', text)
```

#### 残存パターン（exp022適用後も残る）

| パターン | 件数 | 原因 |
|---------|------|------|
| month I, | 4 | `\b`がカンマの前で終端しない |
| month II, | 4 | 同上 |
| month III, | 4 | 同上 |
| month IV, | 6 | 同上 |
| month V, | 4 | 同上 |
| month VI, | 3 | 同上 |
| month VII, | 3 | 同上 |
| month VIII, | 2 | 同上 |
| month IX, | 4 | 同上 |
| month X, | 4 | 同上 |
| month XI, | 3 | 同上 |
| month XII, | 3 | 同上 |

合計: **34件が取りこぼし**

#### 対策
正規表現を `\bmonth\s+{roman}\b` → `\bmonth\s+{roman}(?=[\b,.\s:;])` に修正。

### 1.5 月名→月番号（**新規発見・未実装**）

Hostの指示は「month V → month 5」だが、trainには**月名がそのまま残っている**。
テストでは `month 5` のように数字化されている可能性が高い。

#### trainデータ内の月名出現

**Translation内:**
| 月名 | 月番号 | 出現パターン例 | 件数 |
|------|--------|--------------|------|
| Bēlat-ekallim | 1 | "month Bēlat-ekallim, eponymy" | 3+ |
| Ša-sarrātim / Ša-kēnātim | 2/3 | "month Ša-kēnātim, in the eponymy" | 6+ |
| Mahhur-ilī | 4 | "month Mahhur-ilī, eponymy" | 4 |
| Ab-šarrāni / Abšarrani | 5 | "month Ab-šarrāni, eponymy" | 3+ |
| Hubur / hubur | 6 | "month hubur, eponymy" | 6 |
| Ṣip'um | 7 | — | 少 |
| Qarra'ātum / Qarrātum | 8 | "month Qarra'ātum, eponymy" | 3+ |
| Kanwarta | 9 | — | 2 |
| Te'inātum / Tē'inātum / Teʾinātum | 10 | "month Te'inātum, eponymy" | 9+ |
| Kuzallu / Kuzallum | 11 | "month Kuzallu, eponymy" | 7+ |
| Allanātum | 12 | "month Allanātum, eponymy" | 9+ |

**`month` を含む行: 171件**

**Transliteration内（月名の翻字形）:**
| 翻字形 | 月番号 | 件数 |
|--------|--------|------|
| té-i-na-tim | 10 | 10 |
| ku-zal-li | 11 | 10 |
| a-lá-na-tim | 12 | 8 |
| a-lá-na-tum | 12 | 1 |
| ku-zal-lu | 11 | 2 |

#### 注意点
- 月名の**表記揺れが激しい**（Ša-sarrātim / Ša-kēnātim / ša-sarratim）
- 大文字/小文字の違い（Hubur / hubur）
- アポストロフィの種類（Te'inātum / Teʾinātum）
- `month` が前に付かない場合もある（"Contracted in Kuzallu, eponymy..."）

#### 対策
月名→月番号のマッピング辞書を作成し、`month {名前}` パターンを `month {番号}` に置換。
表記揺れ対応のため、正規表現で柔軟にマッチさせる。

---

## 2. Transliteration前処理

### 2.1 実装済み

| 項目 | train内件数 | exp022対応 |
|------|-----------|-----------|
| Ḫ→H, ḫ→h | 1,313 | ✅ |
| 下付き数字→整数 | 1,173 | ✅ |
| 小数→分数 | 1,682 | ✅（丸め誤差は未対応） |

### 2.2 確認済み・対応不要

| 項目 | train内件数 | 判定 |
|------|-----------|------|
| KÙ.B.(短縮形) | 0 | 不要（KÙ.BABBAR形式に統一済み） |
| (d) | 0 | 不要（Host更新済み） |
| (ki) | 0 | 不要（Host更新済み。submit.pyで(ki)→{ki}変換は念のため実装） |
| {d} | 332 | — |
| {ki} | 250 | — |
| Unicode NFC差分 | 0 | 不要 |

---

## 3. その他の確認事項

### 3.1 shekel/grains分数変換
- `N/12 shekel` パターン: **0件**（Host更新で変換済み）
- `grains` 表記: 54件（既にgrains形式で存在。変換不要）

### 3.2 stray ? の分析
- `?` を含む行: 263件
- `(?)` を含む行: 61件（exp022で除去済み）
- `(?)` 以外の `?`: 202件
  - 文中の `?`: 116件
  - 文末の `?`: 3件
- **例を見ると大半が意味のある疑問符**（"Why is that you..."の文末?、引用中の?等）
- Hostも「Do NOT remove meaningful ?」と明言 → **除去は非推奨**

### 3.4 Gap統合の検証（Host更新で実施済み）
- `<big_gap>` 残存: train **0件** ✅
- `<gap> <gap>` 重複: train/translation に **2件** 残存
  - row 310: `of grain <gap> owed by Galgaliya...`（スペース区切りの重複gap）
  - row 843: `<gap><gap>`（連結）
  - 影響は微少（translationのみ2件）

### 3.5 (TÚG) 確認
- `(TÚG)` 残存: train **0件** ✅（Host更新済み）
- `(m)` がtrain/translationに **5件** 残存（determinative `(m)` は未変換）

### 3.6 Long float（小数点以下5桁以上）
- **train/transliteration**: **11件** 残存
  - 例: `0.83334`, `4.66667`, `1.666699999999999`
  - `1.666699999999999` は浮動小数点の丸め誤差がそのまま残っている
- **train/translation**: **1件**（row 825に `1.66666`, `2.66666` 等）
- **対策**: 小数→分数の近似マッチで同時に吸収可能

### 3.7 `< >` stray marks
- 空白入り `< >`: **0件**
- `<>` 系（`<gap>`除外後）: train/translation に **33件**
  - 内容: `<and>`, `<of silver>`, `<lil>` 等 — **学術的復元マーク**（欠損テキストの補完を示す慣例表記）
  - **除去すべきではない**（意味のある記号）

### 3.8 Quotation marks / Apostrophes（除去禁止の確認）
- Hostは「Do NOT remove quotation marks " " and apostrophes '」と明言
- train/translation内:
  - Double quote `"`: **394件**
  - Single quote `'`: **615件**（多くはアッカド語名の一部、例: `tappa'i`）
  - 左右ダブルクォート `"` `"`: 各1件
  - Right single quote `'`: 2件
- transliterationには0件
- **対応: 除去しないことを確認済み** ✅

### 3.9 Exclamation marks `!`（除去禁止の確認）
- Hostは「Do NOT remove meaningful ! 」と明言
- train/translation: **205件**（文末の感嘆符が主、例: `Do not fear!`, `Send me silver!`）
- transliterationには0件
- **対応: 除去しないことを確認済み** ✅

---

## 4. 改善優先度まとめ

| 優先度 | 項目 | 影響規模 | 対応方法 |
|--------|------|---------|---------|
| **HIGH** | 月名→月番号変換 | 171行中の多数 | train.pyに月名辞書追加 |
| **HIGH** | 小数→分数の丸め誤差吸収 | ~213件(translation) + ~237件(transliteration) + long float 12件 | 近似マッチに変更 |
| **MED** | ローマ数字月の正規表現修正 | 34件 | 正規表現のword boundary修正 |
| **MED** | (ki)→{ki} | — | submit.pyの入力前処理（Host仕様に合わせて念のため実装） |
| **LOW** | `<gap>` 重複 | 2件 | `<gap> <gap>` → `<gap>` |
| **LOW** | stray ? | 除去非推奨 | 対応しない |
| **LOW** | (m) determinative | 5件 | (m)→{m}変換を実装 |
| **不要** | NFC, (d), (TÚG), KÙ.B., shekel分数 | 0件 | 対応不要 |
| **除去禁止** | `" ' ! < >復元マーク` | 多数 | 除去しないこと確認済み |
