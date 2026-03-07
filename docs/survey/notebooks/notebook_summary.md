# Notebook Survey Summary

**調査日**: 2026-03-06
**コンペ**: Deep Past Challenge - Translate Akkadian to English
**対象**: 公開ノートブック上位20件（Most Votesソート）

---

## 1. 全体傾向

- **全ノートブックがByT5（Byte-level T5）を使用** — バイトレベルトークナイザーがアッカド語transliterationのノイズ・特殊文字に有効
- 主要モデル: `byt5-small`（starter）, `byt5-akkadian-optimized-34x`（assiaben）, `byt5-akkadian-mbr` v5/v6（mattiaangeli）
- 推論手法の高度さがスコアに直結: Beam Search < MBR Decoding < Cross-model MBR
- 後処理（gap正規化、禁止文字除去、分数変換、繰り返し除去）がスコア向上に大きく寄与

---

## 2. 上位ノートブック一覧

| # | Notebook | Author | Votes | 主要手法 | 推定LB |
|---|----------|--------|-------|----------|--------|
| 1 | dpc-starter-infer | takamichitoda | 581 | ByT5-small, beam=4 | ~26 |
| 2 | dpc-starter-infer-add-sentencealign | qifeihhh666 | 581 | ByT5 + ルールベースTM | ~28 |
| 3 | dpc-starter-train | takamichitoda | 412 | ByT5-small, 文アライメント, 双方向学習 | — |
| 4 | byt-ensemble | jiexusheng20bz | 379 | **重み平均アンサンブル** (2モデル) | ~32 |
| 5 | deep-pasta-mbr | mattiaangeli | 378 | **MBRデコーディング** + 充実した後処理 | ~35 |
| 6 | akkadiam-exemple | lgregory | 378 | MBR (mattiaangeliモデル使用) | ~35 |
| 7 | byt-ensemble-script | anthonytherrien | 351 | Beam=8 + 後処理 | ~34 |
| 8 | adaptive-beams-test-v1 | prayagp1 | 237 | Adaptive beams | 35.1-35.2 |
| 9 | lb-35-2-ensemble | baidalinadilzhan | 237 | アンサンブル | 35.2 |
| 10 | dpc-infer-with-post-processing-by-llm | takamichitoda | 193 | **3モデルSoup + OA Lexicon + LLM後処理** | 32.6 |
| 11 | akkadian2eng-v1 | kkashyap14 | 190 | 3モデルModel Soup | — |
| 12 | akkadian-english-inference-byt5-optimized-34x | assiaben | 174 | Beam=8, シンプル | ~34 |
| 13 | akkadian-english-byt5-optimized-again | serariagomes | 159 | MBR系 | — |
| 14 | deep-past-challenge-byt5-optimized | yongsukprasertsuk | 156 | ByT5最適化 | — |
| 15 | deep-pasta-mbr-v2 | mattiaangeli | 143 | MBR v2 | — |
| 16 | hybrid-best-akkadian | meenalsinha | 142 | **Cross-model MBR** (2モデル) | ~35.3 |
| 17 | byt5-seq2seq-infer | jackcerion | 136 | Beam + ensemble | — |
| 18 | deep-past-challenge-byt5-base-inference | qifeihhh666 | 127 | Beam Search + 基本後処理 | — |
| 19 | score-35-3-byt5-mbr-pipeline | loopassembly | 124 | MBR pipeline | **35.3** |

---

## 3. 主要テクニック分析

### 3.1 モデル・学習

| テクニック | 使用NB | 詳細 |
|-----------|--------|------|
| **ByT5-small** | starter | バイトレベルT5、アッカド語のノイズに強い |
| **文アライメント** | starter, sentencealign | train=文書レベル→test=文レベルのミスマッチ解消。英語文数=アッカド語行数なら1:1分割 |
| **双方向学習** | starter | Akk→Eng + Eng→Akk で訓練データ2倍 |
| **Adafactor** | starter | AdamWより省メモリ |
| **label_smoothing=0.2** | starter | 正則化 |
| **FP32 / BF16** | 全般 | FP16はByT5でNaN頻発。BF16推奨 |

### 3.2 推論手法

| テクニック | スコア寄与 | 詳細 |
|-----------|-----------|------|
| **Beam Search** (基本) | ベースライン | num_beams=4-10, length_penalty=1.08-1.3 |
| **MBR Decoding** | **+3-5pt** | beam候補4本 + sampling候補2本、chrF++で最良候補選択。最大の差別化要因 |
| **Cross-model MBR** | **最高スコア** | 2モデルの候補をプール（最大12候補）してchrF++で選択 |
| **Model Soup** | +1-2pt | 複数チェックポイントのstate_dictを加重平均。推論コスト=単一モデル |
| **Adaptive Beams** | 微小改善 | 短文(100トークン未満)はbeam数を半減 |
| **repetition_penalty=1.2** | 繰り返し抑制 | MBR系ノートブックで使用 |

### 3.3 前処理・後処理

| テクニック | 詳細 |
|-----------|------|
| **Gap正規化** | `xx`→`<gap>`, `...`/`…`→`<big_gap>` |
| **禁止文字除去** | `!?()"——<>⌈⌋⌊[]+ʾ/;` を削除 |
| **分数変換** | 0.5→½, 0.25→¼, 0.75→¾, 1/3→⅓ |
| **繰り返し除去** | 単語レベル + n-gramレベルの重複除去 |
| **Translation Memory** | trainデータの完全一致をそのまま使用 |
| **OA Lexicon** | 固有名詞の正規化（発音ベースのスペル統一） |
| **LLM後処理** | Gemma-3-4b-itで最小限のpolish（安全装置付き: BLEU破壊防止） |
| **句読点修正** | スペース正規化、末尾ピリオド付加 |
| **ダイアクリティクス変換** | sz→ś, s,→ş 等のASCII→Unicode変換 |

---

## 4. LBスコアの推移（推定）

| 手法 | 推定LB |
|------|--------|
| ByT5-small baseline (beam=4) | ~26 |
| + 文アライメント + 双方向学習 | ~28-30 |
| + 後処理改善 (gap, 分数, 繰り返し) | ~32-33 |
| + Model Soup + OA Lexicon | ~32-33 |
| + MBR decoding (単一モデル) | ~35 |
| + Cross-model MBR | **~35.3** |

---

## 5. 我々の実験に取り入れるべきアイデア

### 堅実案（確実にスコアが上がる見込み）

1. **ByT5ベースのファインチューニング** — 全ノートブックの基盤。byt5-smallから始め、byt5-baseへスケールアップ
2. **文アライメント** — train/testのミスマッチ解消。データ量増加で確実にスコア向上
3. **双方向学習** — データ2倍、実装コスト低
4. **MBRデコーディング** — beam候補+sampling候補をchrF++で選択。最大のスコア向上要因（+3-5pt）
5. **後処理パイプライン** — gap正規化、禁止文字除去、分数変換、繰り返し除去。既存コード流用可
6. **Model Soup** — 複数チェックポイントの重み平均。推論コスト0でアンサンブル効果
7. **Translation Memory** — trainデータの完全一致利用

### 爆発案（失敗リスク高いが当たれば大きい）

1. **大規模モデルへのスケール** — byt5-large / byt5-xl。計算コスト大だがByT5の限界を突破できる可能性
2. **mBART / NLLB / mT5 との異種モデルアンサンブル** — ByT5一辺倒の状況で、異なるアーキテクチャの候補をMBRプールに追加
3. **外部コーパス活用** — ORACC, CDLI等のアッカド語テキストDBからの追加データ。ルールにデータ制約がなければ大幅改善
4. **Character-level Contrastive Learning** — transliterationの表記揺れに対するロバスト性向上
5. **LLM-as-Judge後処理** — 大規模LLM（GPT-4, Claude等）でMBR候補のリランキング。ドメイン知識を活用
6. **Curriculum Learning** — 短文→長文の段階的学習。アッカド語の文書長にバリエーションがある点を活用
7. **合成データ生成** — Eng→Akk方向のモデルでback-translation。ノイジーだが量で勝負

---

## 6. 主要モデル一覧（Kaggle Datasets）

| モデル名 | 作成者 | 用途 |
|----------|--------|------|
| byt5-akkadian-optimized-34x | assiaben | 汎用推論 |
| byt5-akkadian-mbr v5/v6 | mattiaangeli | MBR推論用 |
| byt5-akkadian-mbr-v2 | mattiaangeli | MBR改良版 |
| byt5-base-big-data2 | jeanjean111 | 大規模データ学習 |
| byt5-akkadian-model | — | 汎用 |
| byt5-akkadian-en-v5/v6/v7 | kkashyap14 | Model Soup用 |

---

## 7. 技術的メモ

- **FP16は使わない**: ByT5でNaN頻発。BF16またはFP32を使用
- **BetterTransformer**: optimumライブラリで20-50%高速化
- **Bucket Batching**: 入力長でバケット分け→パディング削減→推論高速化
- **Task Prefix**: `"translate Akkadian to English: "` が標準（小文字始まり）
- **max_length**: 入力512、出力256-512
- **Adafactor optimizer**: AdamWより省メモリで、ByT5との相性良好
