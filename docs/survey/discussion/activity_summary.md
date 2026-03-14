# ディスカッション活動サマリー

> 最終更新: 2026/03/13（4回目・LBリーダーボード+最新トピック追加）

## 概要

- トピック数: 25+（うち pinned: 7、8ページ分）
- 最もコメントの多いトピック: "A Stitch in Time Saves Nine"（67 comments）
- 活発なトピック: "Two practical stumbling blocks"（54 comments, 73 votes）、"Dataset Update - Mind the Gaps"（46 comments）
- **残り10日**（2026/03/23締切）

---

## テストデータに関する確実な情報（Host直接発言）

以下はHostの投稿・コメントから得た**本番テストセット全体**に関する情報。ダミー4件の分析ではない。

### テストのtransliteration形式
| 項目 | テストの状態 | 根拠 |
|------|-------------|------|
| ダイアクリティクス | š ṣ ṭ á à í ì ú ù あり（ḫは例外） | 665209 Host回答 |
| 下付き数字 | なし（DU10形式で整数表記） | 665209 Host回答 |
| Ḫ/ḫ | Hostは「Ḫ→H変換はOptional」とのみ言及。テストにḪがないとは断言していない | 678899 Host本文 |
| gap | `<gap>`に統一済み（`<big_gap>`廃止） | 678899 Host本文 |
| 限定詞 | `{d}` `{ki}` 形式（`(d)` `(ki)` ではない） | 678899 Host本文 |

### テストのtranslation形式
| 項目 | テストの状態 | 根拠 |
|------|-------------|------|
| 引用符 `" "` | あり（除去してはいけない） | 678899 Host本文 |
| アポストロフィ `'` | あり（除去してはいけない） | 678899 Host本文 |
| 意味のある `?` `!` | あり（除去してはいけない） | 678899 Host本文 |
| `fem.` `sing.` `pl.` `(?)` | なし（trainから除去推奨） | 678899 Host本文 |
| `PN` | `<gap>` に置換済み | 678899 Host本文 |
| 分数表記 | Unicode分数使用（½ ⅓ ⅔ etc.）| 678899 Host本文 |
| ローマ数字 | 整数化済み（month V → month 5）| 678899 Host本文 |
| 固有名詞 | ダイアクリティクス正規化あり（例: A-šùr-DU10 → Aššur-ṭāb）| 665209 Host回答 |

### テストの構造
| 項目 | 内容 | 根拠 |
|------|------|------|
| 粒度 | **文レベル**（trainはdoc-level） | data_description.md |
| 規模 | 約4,000文、約400ドキュメント由来 | data_description.md |

### 注意: ダミーテストからの外挿は信頼できない
test.csvはダミー4件のみ。以下はダミー分析で「0%」だったが本番を保証しない：
- 数字含有率、Ḫ含有率、小数/分数含有率、文長分布

---

## 重要トピック詳細

### [Pinned] Two practical stumbling blocks (665209, Host, 71 votes, 52+ comments)

**本文要旨:**
コンペの2大ボトルネックを指摘：

**1. 固有名詞（人名・地名・神名）が支配的なエラー源**
- 翻字が版ごとに不統一、古い正書法を保持していることが多い
- 明示的サポートなしではモデルにとって意味的に不透明
- 名前がmangled/dropped/hallucinatedされるだけで翻訳全体が失敗
- **対策: onomasticon** を補足データとして公開済み
  - 入手先: https://www.kaggle.com/datasets/deeppast/old-assyrian-grammars-and-other-resources/data
  - 活用法: ルックアップ/制約レイヤー、デコーディングバイアス、生成後修復（post-generation repair）

**2. ASCII/翻字フォーマット正規化は必須**
- 異なるコーパスが同じテキストを異なる規約でエンコード
- `sz→š` 等のASCII化は**逆方向**（評価データにダイアクリティクスあり）
- `s / ṣ / š`, `t / ṭ` の区別を保持すべき
- **テストデータの添字規則**:
  - 2/3の読みはダイアクリティクスで表記: `á à é è í ì ú ù`（`a2, a3` や `a₂` **ではない**）
  - 4以上の記号値は整数を使用: `DU10`（添え字なし）
  - 例: `A-šùr-DU10`（`A-szur-DU₁₀` ではない）

**3. Gap/damage markers** (2/18更新)
- `x` = 単一破損, `x x x x` や `...` = 大きな欠損
- 全て `<gap>` に統一。`<big_gap>` は廃止
- 連続gap重複は排除済み
- `<gap>-A-šùr` のようにワードに付着したgapは保持すべき（盲目的に除去しない）
- **gapのtransliteration-translation間アライメントは不完全** → 制御した参加者に有利

**4. RL/報酬ベース手法への影響**
- SFTは妥当なloss曲線を出すが、RL/preferenceベース手法が改善しない原因はほぼすべて**output non-conformance**
- 正規化とアライメントを先に対処すれば報酬がスムーズになる

---

#### Host提供: テスト文字リスト

**Transliterations Characters:**
```
!+-.0123456789:<>ABDEGHIKLMNPQRSTUWZ_abdeghiklmnpqrstuwz{}¼½ÀÁÈÉÌÍÙÚàáèéìíùúİışŠšṢṣṬṭ…⅓⅔⅙⅚
```

**Translations Characters:**
```
!"\\'()+,-.0123456789:;<>?ABCDEFGHIJKLMNOPQRSTUWYZ[]_abcdefghijklmnopqrstuvwxyz¼½àâāēğīışŠšūṢṣṬṭ–—''""⅓⅔⅙⅚
```

注意:
- Transliterationに `v` がない（translationのみ）
- Transliterationに `ā ī ū` がない（translationのみ。固有名詞の正規化形）
- `şİı` はトルコ語由来の表記（`ş=ṣ`, `İ=I`, `ı=i`）
- Transliterationに `…` が含まれるが `<gap>` に変換すべき

---

#### Host提供: 限定詞の正しい形式

テストデータの限定詞形式（Host回答）:
- `{d}`, `{ki}` → そのまま（括弧ではなく中括弧）
- `(TÚG)` → `TÚG`（括弧なし）
- **`{tug₂}` でも `{tug2}` でもなく `{túg}`** がこのデータセットの形式
- CDLIデータ: `{tug2}`, ORACCデータ: `{tug₂}` → 各自変換が必要

全限定詞リスト（概要タブより、テストでの形式は要確認）:
| 限定詞 | 意味 | テスト形式 |
|--------|------|-----------|
| {d} | 神名 | {d} |
| {ki} | 地名 | {ki} |
| TÚG | 織物 | TÚG（括弧なし） |
| {m} | 男性人名 | 要確認 |
| {mul} | 天体 | 要確認 |
| {lu₂} | 人・職業 | 要確認 |
| その他 | {e₂},{uru},{kur},{mi},{geš},{dub},{id₂},{mušen},{na₄},{kuš},{u₂} | 要確認 |

---

#### 重要コメント詳細

**Souhardya (4 votes, 188位)**: onomasticonとtrainラベルの命名規則不一致を指摘
- `šu-{d}EN.LÍL` → Onomasticon: `Šu-Enlil` / Train: `Šu-Illil` — 名前自体が違う
- `šu-ku-tum` → Onomasticon: `Šukatum` / Train: `Šukkutum` — 促音kk欠落
- `ṣí-lu-lu` → Onomasticon: `Ṣilulu` / Train: `Ṣilūlu` — 長母音ū欠落
- テストラベルはどちらの規約に従うか？ → **Host未回答**

**Bilzard (12 votes)**: テストのtransliterationとtranslation両方が正規化済みか質問
- Case 2（翻訳のみ正規化）なら「前処理宝くじ」になりかねない

**MPWARE (1 vote, 290位)**: 翻訳中の `?` `!` `:` の扱いを質問
- `(?)` `(!)` `(fem. plur.)` 等の括弧付き注記のみ除去すべきでは？
- 文字リストの頻度カウントがおかしい（`'`=1,651,390は不可能）→ Host認め修正

**Angantyr (4 votes)**: 文字リストをソート整理して再投稿。さらに:
- `şİı` の正体を特定（トルコ語由来）
- `…:!{}` の正規化ルールを提案
- **分数問題を指摘**: `1.83333 ma-na 3.5 GÍN` → transliterationでは `1⅚ ma-na 3½ GÍN` だが、trainのtranslationでは `1 mina 56.5 shekels` に再計算されている。テストではどちらか？

**Anil Ozturk (8 votes, 467位)**: gap正規化のエッジケースを詳細に質問
- `<big_gap> <gap>-A-šùr` → そのまま保持？
- `xxxx-kam` → `<big_gap>-kam`？
- 「正規化ロジックのリバースエンジニアリングを強いられている」と不満

**耶✌ (1-3 votes, 37位)**: LBの実験結果共有
- gap/big_gapを**マージしない方がLBが良い**結果

**Adam Anderson (Host回答)**: gap共起パターンの頻度リストを共有(9 votes)
- `someword-<gap> <gap>-someword` と `someword-<big_gap>-someword` 両方ある
- `xxxx-kam` → `<big_gap>-kam`, `x-kam` → `<gap>-kam`

**Adam Anderson (Host回答)**: 添え字の詳細説明(4 votes)
- ASCII ATF(CDLI): `ú=u2, ù=u3` → ORACC: `ú=u₂` → **このデータセット: ダイアクリティクス使用**
- 4以上は整数（`DU10`）。添え字数字は使わない

**Wisdom Aduah (0 votes, 20位)**: テスト翻訳にUnicode分数(`¼`)が使われるか、通常テキスト(`1/4`)が使われるか質問 → **Host未回答**

**Adam Anderson (Host, 最新回答)**: onomasticon.csvの入手先を共有
- https://www.kaggle.com/datasets/deeppast/old-assyrian-grammars-and-other-resources/data

### [Pinned] A Stitch in Time Saves Nine (678899, Host, 33 votes, 14日前)

**最終データ更新アナウンス。以下の変更をtrain+test両方に適用:**

**確定変更（既に適用済み）:**
- 全gap種 → `<gap>` に統一（x, [x], …, (break), (large break), (n broken lines), `<big_gap>`等すべて）
- 重複gap → 単一`<gap>`
- 限定詞: `(d)` → `{d}`, `(ki)` → `{ki}` （テストに合わせる）
- 長い浮動小数点 → 小数4桁に丸め（1.3333300000000001 → 1.3333）

**推奨前処理（参加者が適用すべき）:**

Remove from translations:
- `fem.`, `sing.`, `pl.`, `plural`, `(?)`
- stray marks: `..`, `?`, `x`, `xx`, `<< >>`, `< >` (ただし`<gap>`は残す)
- `/` による代替翻訳 → 一方を選ぶ（"you / she brought" → "you brought"）

Do NOT remove from translations (テストにもある):
- 引用符 `" "`、アポストロフィ `'`
- 意味のある `?` `!`

Replace in translations:
- `PN` → `<gap>`
- `-gold` → `pašallum gold`
- `-tax` → `šadduātum tax`
- `-textiles` → `kutānum textiles`
- 分数変換: `1/12 shekel` → `15 grains` 等
- 小数→分数: 0.5→½, 0.25→¼, 0.3333→⅓, 0.6666→⅔, 0.8333→⅚, 0.75→¾, 0.1666→⅙, 0.625→⅝
- ローマ数字→整数（月名: month V → month 5）

Optional in transliterations:
- `Ḫ` → `H`, `ḫ` → `h`
- `KÙ.B.` → `KÙ.BABBAR`
- Unicode下付き数字→通常整数（₀→0, ₁→1, ...）
- 小数→分数（transliterationでも同様）

**重要コメント:**
- **Yurnero (5位)**: 良い変更だが、GPU/人的時間のコストが大きかったと不満。「新テストでもtransitivityは保たれるはず」
- **Jack (8位)**: 「良い変更だけど、uselessになった作業が多い」
- **MPWARE (290位)**: trainの多数の不整合を詳細に列挙（分数の非一貫性、英語化されたtransliteration、ヘブライ文字ד混入、括弧の不統一等）

### [Pinned] Dataset Update - Mind the Gaps (674136)

データ更新アナウンス（678899の前段階）。gap表記統一の詳細。

---

## 3/8時点の非pinnedトピック

### eBL text divergence（1 comment, David Ochoa Corrales）
- eBL→OARE Mimation Restorer: SB語形→OA語形変換（585ペアで92.8%精度）
- **重要度**: 中

### Insights from the Akkademia Codebase & PNAS Paper（5 comments）
- 先行研究はCNN fconv BLEU4=37.47（2023年PNAS論文）
- **重要度**: 高

### Is this competition becoming a 'Regex Guessing Game'?（6 comments）
- データフォーマット不統一への不満
- **重要度**: 中

### Lora on ByT5 large（1 comment）
- ByT5-largeにLoRA適用、スコア上限18.0
- フルファインチューニングが必要との結論
- **重要度**: 中

### Old Assyrian Tokens（1 comment, David Ochoa Corrales）
- OAREの全音節リスト公開、Unicode正規化（NFC）推奨
- **重要度**: 中

### Massive bot attack（17 comments, Yurnero 3位）
- 140+ bot アカウント参加。Kaggle Staff調査中
- **重要度**: 低

---

## リーダーボード状況（2026/03/13時点）

| 順位 | チーム | スコア | 特記 |
|------|--------|--------|------|
| 1 | KE WU | 40.6 | 133提出 |
| 2 | How I Learned to Stop Worrying (Darragh + Raja) | 39.9 | 125提出 |
| 3 | Hrithik Reddy | 39.5 | 223提出 |
| 4 | Yurnero | 39.5 | 271提出 |
| 5 | DataTech Club (4人) | 39.4 | 208提出 |
| 6-7 | kwa / John Doe | 39.0 | |
| 8 | It's showtime! (AK + Priyanshu) | 38.9 | |
| 9 | M & J & M | 38.7 | 326提出 |
| 10 | not free lunch2.0 (heng他) | 38.6 | 292提出 |

- Gold Medal圏: 1-15位（38.1+）
- Silver Medal圏: 16-49位（36.1-38.0）
- Public LB 34%のみで評価。Private（66%）で大きく変動する可能性あり
- 公開notebookアンサンブルでLB 35-36が大量参入中

---

## 3/13時点の新トピック

### Regarding the open-source issue（680686, mochi 29位, 38 votes, 11 comments）
- 締切14日前に公開されたLB 35.9のアンサンブルnotebookへの抗議
- 既存公開モデルの重み平均/ランクブレンドのみで技術的貢献ゼロ
- **LB 35.9 notebook**: `vitorhugobarbedo/lb-35-9-with-corrections-public-model`
  - 既存公開モデル2つのcross-model MBR + regex後処理
- 注目コメント:
  - Giovanny Rodriguez (26位): 「公開モデルはpublic LBにoverfitしている。private shake-upで落ちる」
  - Cody_Null (78位): 「メダル圏のnotebookを最終月に公開禁止にすべき」
  - freeze periodは既に発動済み（新規公開notebook不可）
- **重要度**: 中（戦略的には、public notebookアンサンブルはprivate LBで落ちる可能性高い）

### Long Submission Queue and Unexpected Timeouts（680699, Musa Peker, 6 votes, 4 comments）
- 提出キューが長く、タイムアウトが頻発
- **重要度**: 低

### Local validation vs public LB mismatch（680874, Alessio 475位, 2 votes, 1 comment）
- CV/LB乖離の報告。前処理の違いか、分布の違いか
- Wisdom Aduah (21位): 「dataset contaminationに注意。val setの選び方でリークが起きる」
- **重要度**: 中高（我々のexp016-exp023でも同様の乖離あり）

### how to create a Dependency Installation Script（681149, K.Dallash, 0 votes, 1 comment）
- 初心者質問
- **重要度**: 低

### deeppast.org/oare does not exist（680889, CPMP, 2 votes, 1 comment）
- OAREサイトがダウンしている報告
- **重要度**: 低

### (pdf)-Translation sometimes different from train.csv（680446, Marius Heuser, 2 votes, 1 comment）
- PDFテキストエディションとtrain.csvの翻訳が異なる箇所がある
- **重要度**: 中（publications.csvからの追加データ構築時に注意）

---

## 戦略的インサイト

### LB 25→38 のブリッジ戦略

| レベル | スコア帯 | 必要な施策 |
|--------|----------|-----------|
| Level 1 | ~26 | ByT5-small starter baseline（beam=4） |
| Level 2 | ~28-30 | ByT5-base + 双方向学習 + 文アライメント |
| Level 3 | ~30-32 | Host推奨前処理（gap統一、PN→gap、分数変換、fem./pl.除去、ローマ数字→整数、commodity置換） |
| Level 4 | ~33-35 | MBRデコーディング（beam4 + multi-temp sampling, chrF++選択）+ 充実した後処理 |
| Level 5 | ~35-37 | Cross-model MBR（2モデルのプールを統合）+ Model Soup + Weighted MBR |
| Level 6 | ~37-39 | 独自学習モデル + 追加データ + 前処理/後処理の完全マッチング |
| Level 7 | 39+ | 未知の独自手法（top 5チームの秘密） |

### 現在の我々の位置と必要なアクション

**現在**: exp023 LB=30.03（Level 3）

**Level 4到達に必要（LB 33-35）**:
1. **Cross-model MBR**: 2モデル(ByT5-base)の候補プール統合。公開notebook分析では+3-5ptの最大要因
2. **multi-temperature sampling**: beam4候補 + temp=[0.6, 0.8, 1.05]でsampling各2本 = 10候補/モデル
3. **後処理の充実**: 繰り返し除去（n-gramレベル）、禁止文字除去、スペース正規化

**Level 5到達に必要（LB 35-37）**:
1. **独自学習モデルの公開**: 自前でByT5-baseを学習し、Kaggle Modelsにアップロード
2. **Model Soup**: 複数foldまたは複数epochのチェックポイントを重み平均
3. **公開モデル(assiaben/byt5-akkadian-optimized-34x, mattiaangeli/byt5-akkadian-mbr-v2)の活用**: 推論パイプラインに組み込み

**Level 6到達に必要（LB 37+）**:
1. **公開モデルとは異なる学習データ/設定で訓練した独自モデル**: 多様性が鍵
2. **追加データ**: publications.csv OCR活用、eBL dictionary活用
3. **固有名詞処理**: onomasticonベースの後処理/制約デコーディング
4. **CV/LBの相関改善**: GroupKFoldでリーク排除、正確なCV指標

### 既存の戦略的インサイト

1. **固有名詞が最大のボトルネック**（Host直言）→ onomasticon + OA_Lexiconの活用が鍵
2. **データ前処理がスコアを左右**: gap統一・分数変換・fem./pl.除去・PN→`<gap>`等、Host推奨の前処理をtrainに適用すべき
3. **テストのフォーマットに合わせる**: ダイアクリティクス保持、下付き数字→整数、限定詞`{d}`形式
4. **Ḫ→H変換はOptional**: テストにḪがないとは断言されていない
5. **ByT5-largeのLoRAは効果薄**（18.0止まり）→ フルファインチューニングが必要
6. **スターターnotebookで約30**: ベースラインとして参考
7. **先行研究（Akkademia/PNAS）**: CNN fconvでBLEU4=37.47
8. **追加データ**: onomasticon.csv（人名リスト）とPDFテキストエディションが公開済み
9. **Unicode正規化（NFC）** が推奨される
10. **翻訳中の `/` による代替翻訳**: 一方を選択すべき（テストでは代替なし）
11. **Public LBはoverfitリスクが高い**: 34%のみで評価。公開notebookアンサンブルはprivate shake-upで落ちる可能性大
12. **dataset contamination注意**: val setの選び方でリークが起きる（GroupKFold推奨）
