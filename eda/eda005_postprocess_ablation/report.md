# EDA005: 後処理Ablationテスト

## 分析の目的

exp005の後処理がdoc-levelで悪化(-1.18pt)、sent-levelでわずかに改善(+0.19pt)する原因を特定し、最適な後処理構成を決定する。

## 分析概要

exp005の保存済み予測CSV（doc-level: 157件, sent-level: 157件）を使い、以下3種のテストを実施:
1. **単独適用**: rawに各コンポーネント1つだけ適用
2. **全適用**: 全コンポーネント適用（従来のpostprocess_batch相当）
3. **除外テスト**: 全適用から1つずつ除外

### 後処理コンポーネント一覧

| # | コンポーネント | 内容 |
|---|--------------|------|
| 1 | special_chars | Ḫ→H, 下付き数字→通常数字 |
| 2 | gap_normalization | [x]/(x)/x → `<gap>`, ...→`<big_gap>` |
| 3 | annotation_removal | (fem), (plur) 等のアノテーション除去 |
| 4 | forbidden_chars | `!?()"——<>⌈⌋⌊[]+ʾ/;` の除去 |
| 5 | fraction_conversion | 0.5→½, 0.25→¼, 0.75→¾ |
| 6 | repeated_removal | 繰り返し単語・n-gram除去 |
| 7 | punctuation_fixes | `スペース+句読点` → `句読点`, 連続句読点除去 |

## 発見事項

### 単独適用テスト（rawからの差分）

| コンポーネント | doc diff | sent diff | 判定 |
|--------------|----------|-----------|------|
| special_chars | 0.00 | 0.00 | 影響なし |
| gap_normalization | 0.00 | +0.01 | 影響なし |
| annotation_removal | 0.00 | 0.00 | 影響なし |
| **forbidden_chars** | **-0.26** | **-0.23** | **有害** |
| **fraction_conversion** | **-0.15** | **-0.27** | **有害** |
| **repeated_removal** | **-0.76** | **+0.70** | **doc有害/sent有効** |
| punctuation_fixes | 0.00 | 0.00 | 影響なし |

### 除外テスト（全適用からの改善幅）

| 除外コンポーネント | doc geo_mean | doc改善 | sent geo_mean | sent改善 |
|------------------|-------------|---------|--------------|---------|
| (全適用) | 17.10 | - | 27.22 | - |
| w/o special_chars | 17.10 | 0.00 | 27.22 | 0.00 |
| w/o gap_norm | 17.10 | 0.00 | 27.21 | -0.01 |
| w/o annotation | 17.10 | 0.00 | 27.22 | 0.00 |
| **w/o forbidden_chars** | **17.37** | **+0.27** | **27.45** | **+0.23** |
| **w/o fraction_conv** | **17.25** | **+0.15** | **27.50** | **+0.28** |
| **w/o repeated_removal** | **17.86** | **+0.76** | **26.53** | **-0.69** |
| w/o punct_fixes | 17.10 | 0.00 | 27.22 | 0.00 |

### 主要な知見

1. **repeated_removal（繰り返し除去）が最大の影響要因**
   - doc-levelでは-0.76pt有害（繰り返しがある長文で過剰に削除）
   - sent-levelでは+0.70pt有効（短文の繰り返しを適切に除去）
   - テスト条件（短文）ではsent-levelに近いので **採用すべき**

2. **forbidden_chars（禁止文字除去）は両方で有害**
   - doc: -0.26, sent: -0.23
   - 正解テキストに含まれる文字（括弧、引用符等）も消してしまうため
   - **除外すべき**

3. **fraction_conversion（分数変換）も両方で有害**
   - doc: -0.15, sent: -0.27
   - 正解テキストが小数表記のままの場合、不一致になる
   - **除外すべき**

4. **影響なしのコンポーネント**: special_chars, gap_normalization, annotation_removal, punctuation_fixes
   - 現在のモデル出力にこれらが該当するケースがほぼないと推測

## 推奨後処理構成

テスト条件（短文）を考慮した最適構成:

| コンポーネント | 採用 | 理由 |
|--------------|------|------|
| special_chars | △ | 影響なしだが念のため |
| gap_normalization | △ | 影響なしだが念のため |
| annotation_removal | △ | 影響なしだが念のため |
| forbidden_chars | ✗ | 両方で有害 |
| fraction_conversion | ✗ | 両方で有害 |
| **repeated_removal** | **✓** | **sent-levelで+0.70pt** |
| punctuation_fixes | △ | 影響なし |

**推定改善幅**: repeated_removalのみ適用で sent-level raw 27.03 → 27.73 (+0.70pt)

## 精度向上の仮説

- forbidden_charsとfraction_conversionを除外し、repeated_removalのみの後処理に変更すれば、sent-level CVで+0.5〜0.7ptの改善が見込める
- 現在の繰り返し除去は正規表現ベースだが、ByT5のバイトレベル繰り返しに特化した除去（例: 一定バイト以上の繰り返しパターン検出）でさらに改善の余地あり
