# Notebook Survey Summary

**調査日**: 2026-03-15 (前回: 2026-03-08)
**コンペ**: Deep Past Challenge - Translate Akkadian to English
**対象**: 公開ノートブック上位20件（Most Votesソート）

---

## 0. 前回(3/8)からの変更点

### 新規ノートブック (2件)
| Notebook | Author | Votes | Best Score | 特徴 |
|----------|--------|-------|------------|------|
| lb-35-9-with-regex-corrections-public-model | vitorhugobarbedo | 405 | **35.9** | 自前モデル(EMA) + assiaben、weighted MBR、multi-temp sampling |
| lb-35-9-ensembling-post-processing-baseline | giovannyrodrguez | 315 | **35.9** | 自前モデル×2、cross-model MBR、**training data非公開** |

### 投票数の上昇（主要）
- dpc-starter-train: 412→436
- byt-ensemble: 379→408

---

## 1. 全体傾向

- **全ノートブックがByT5（Byte-level T5）を使用** — バイトレベルトークナイザーがアッカド語transliterationのノイズ・特殊文字に有効
- 主要モデル: `byt5-small`（starter）, `byt5-akkadian-optimized-34x`（assiaben）, `byt5-akkadian-mbr` v2（mattiaangeli）
- 推論手法の高度さがスコアに直結: Beam Search < MBR Decoding < Cross-model MBR
- 後処理（gap正規化、禁止文字除去、分数変換、繰り返し除去）がスコア向上に大きく寄与
- **最新トレンド**: weighted MBR (chrF++ + BLEU + Jaccard)、multi-temperature sampling、EMAモデル

---

## 2. 上位ノートブック一覧（スコア付き）

| # | Notebook | Author | Votes | Best Score | 使用モデル | Training公開 |
|---|----------|--------|-------|------------|------------|:---:|
| 1 | dpc-starter-infer | takamichitoda | 589 | 26.9 | byt5-small (starter-train) | ✅ |
| 2 | dpc-starter-infer-add-sentencealign | qifeihhh666 | 589 | 27.2 | starter-train + TM | ✅ |
| 3 | dpc-starter-train | takamichitoda | 436 | — | **Training NB本体** | ✅ |
| 4 | byt-ensemble | jiexusheng20bz | 408 | 32.4 | assiaben + starter | ❌ |
| 5 | **lb-35-9-with-regex-corrections** | vitorhugobarbedo | 405 | **35.9** | assiaben + **自前EMA** | ⚠️部分的 |
| 6 | deep-pasta-mbr | mattiaangeli | 381 | 33.2 | mbr-v2 (自前) | ❌ |
| 7 | akkadiam-exemple | lgregory | 381 | 33.0 | mattiaangeli mbr v6 | ❌ |
| 8 | byt-ensemble-script | anthonytherrien | 357 | 33.5 | assiaben + starter | ❌ |
| 9 | **lb-35-9-ensembling-pp-baseline** | giovannyrodrguez | 315 | **35.9** | **自前×2 (Private)** | ❌ |
| 10 | hybrid-best-akkadian | meenalsinha | 261 | **35.3** | assiaben + mattiaangeli mbr-v2 | ❌ |
| 11 | adaptive-beams-test-v1 | prayagp1 | 249 | 33.5 | assiaben | ❌ |
| 12 | lb-35-2-ensemble | baidalinadilzhan | 249 | 33.5 | assiaben + mattiaangeli mbr v6 | ❌ |
| 13 | dpc-infer-with-post-processing-by-llm | takamichitoda | 199 | 32.0 | starter + big-data2 + gap model, Gemma-3 | ⚠️部分的 |
| 14 | akkadian2eng-v1 | kkashyap14 | 193 | 34.4 | 自前v5/v6/v7 Model Soup | ❌ |
| 15 | akkadian-english-byt5-optimized-34x | assiaben | 182 | 33.5 | 自前(34x) | ❌ |
| 16 | deep-past-challenge-byt5-optimized | yongsukprasertsuk | 166 | — | assiaben | ❌ |
| 17 | deep-pasta-mbr-v2 | mattiaangeli | 150 | 34.1 | mbr-v2 (自前) | ❌ |
| 18 | byt5-seq2seq-infer | jackcerion | 140 | 31.8 | 複数モデルensemble | ❌ |
| 19 | score-35-3-byt5-mbr-pipeline | loopassembly | 137 | 32.8 | mattiaangeli mbr v6 | ❌ |
| 20 | deep-past-challenge-byt5-base-inference | qifeihhh666 | — | 31.8 | 自前(byt5-base) | ❌ |

---

## 3. Training Code分析（公開Training Notebook全件調査）

Top 20以外も含め、Kaggle APIで全ノートブックを検索し、training関連を全件調査した。

### 3.1 モデル学習コード（完全公開）

#### ✅ takamichitoda/dpc-starter-train → dpc-starter-infer (LB 26.9)

**Training Code**: https://www.kaggle.com/code/takamichitoda/dpc-starter-train
**Votes**: 436

| 項目 | 内容 |
|------|------|
| ベースモデル | `google/byt5-small` |
| データ前処理 | simple_sentence_aligner (英文数=アッカド語行数なら1:1分割) |
| データ拡張 | なし（文アライメントのみ） |
| Optimizer | AdamW (デフォルト) |
| LR | 2e-4 |
| Epochs | 20 |
| Batch | 4 (gradient_accum=2 → effective 8) |
| FP16 | **False** (ByT5でNaN回避) |
| 評価指標 | chrF++ |
| Task Prefix | `"translate Akkadian to English: "` |
| MAX_LENGTH | 512 |

**特徴**: 最もシンプルなbaseline。文アライメントが唯一の工夫。

#### ✅ llkh0a/dpc-baseline-train-infer (Training + Inference一体型)

**Training Code**: https://www.kaggle.com/code/llkh0a/dpc-baseline-train-infer
**Votes**: (top 20圏外)

| 項目 | 内容 |
|------|------|
| ベースモデル | `google/byt5-base` (smallからスケールアップ) |
| Resume from | `jeanjean111/byt5-base-big-data2` → `llkh0a/byt5-akkadian-model` |
| データ前処理 | simple_sentence_aligner (starterと同じ) |
| **データ拡張** | **双方向学習** (Akk→Eng + Eng→Akk、データ2倍) |
| **Optimizer** | **Adafactor** (省メモリ) |
| **Label Smoothing** | **0.2** |
| LR | 1e-4 |
| Epochs | 1 (resume trainingなので少ない) |
| Batch | 2 (gradient_accum=8 → effective 16) |
| 評価指標 | **geo_mean(chrF++ × BLEU)** (コンペ指標に近い) |
| generation_num_beams | 2 (評価時) |
| load_best_model_at_end | True (metric_for_best_model="geo_mean") |

**特徴**:
- starterからの改良版。**byt5-base**への移行が最大の違い
- **双方向学習**でデータ2倍化
- **Adafactor + label_smoothing=0.2** はByT5の定石
- Resume training: `byt5-base-big-data2` → さらにfine-tune → upload

#### ✅ qifeihhh666/deep-past-challenge-byt5-base-training (64 votes)

**Training Code**: https://www.kaggle.com/code/qifeihhh666/deep-past-challenge-byt5-base-training

| 項目 | 内容 |
|------|------|
| ベースモデル | `google/byt5-base` |
| データ前処理 | simple_sentence_aligner + **gap-augmented data** (`output_gap_big_gap5.csv`) |
| Optimizer | AdamW (デフォルト) |
| LR | **2e-4** |
| Epochs | **10** |
| Batch | 2 (gradient_accum=2 → effective 4) |
| FP16 | **False** |
| LR Scheduler | **cosine** |
| 評価指標 | chrF++ (evaluate library) |
| Task Prefix | `"translate Akkadian to English: "` |

**特徴**:
- **gap処理済みデータ**を使用（`output_gap_big_gap5.csv`）— gap/big_gapの正規化がスコアに影響
- **cosine LR scheduler** — starterのlinearより良い可能性
- コメントアウトされた試行: extended corpus (old-assyrian-extended-corpus)、resume training、tablet-level data

#### ✅ adarsh2626/full-model-making (32 votes)

**Training Code**: https://www.kaggle.com/code/adarsh2626/full-model-making-datacleaning-training-inference

| 項目 | 内容 |
|------|------|
| ベースモデル | `google/byt5-small` |
| **データクリーニング** | clean_transliteration (記号除去、角括弧・半括弧除去、下付き数字変換) + gap正規化 |
| Optimizer | AdamW |
| LR | **1e-4** |
| Epochs | **20** |
| Batch | 4 (gradient_accum=2 → effective 8) |
| weight_decay | 0.02 |
| FP16 | **False** |
| Task Prefix | なし（PREFIX定義あるが未使用） |

**特徴**:
- **前処理に独自のクリーニング関数**:
  - `clean_transliteration`: `!?⁄:—` 除去、`<...>` `˹...˺` `[...]` 除去、subscript→通常数字
  - `replace_gaps`: `...` `[x]` `x` `…` → `<gap>` / `<big_gap>` トークンに正規化
- Full pipeline（クリーニング→学習→推論）が1ノートブックに収まっている
- 推論時: beam=4, repetition_penalty=1.2

### 3.2 データ拡張・文アライメントコード（学習データ改善）

#### 📊 zhangyue199/dpc-sentence-alignment-oare-firstword (32 votes)

**Code**: https://www.kaggle.com/code/zhangyue199/dpc-sentence-alignment-oare-firstword

**種別**: 文アライメント用データ前処理

| 項目 | 内容 |
|------|------|
| 入力 | train.csv + Sentences_Oare_FirstWord_LinNum.csv |
| アライメント手法 | **chrF++ベースのfuzzy matching** |
| 出力 | sentence_alignment.csv（文レベルの対訳ペア） |

**手法**:
- `Sentences_Oare_FirstWord_LinNum.csv`の`first_word_number`で文境界を特定
- 英語翻訳を句読点(`.;!?`)で分割してclause単位に
- 各アッカド語文に対し、全英語clauseとchrF++スコアを計算→best match
- **窓拡張**: best match + 次のclauseを結合してスコアが上がれば採用
- train.csvとSentences_OareをOARE_IDでjoin

**特徴**: starterのsimple_sentence_alignerより高度。chrF++ベースのfuzzy matchingで品質向上。

#### 📊 seraquevence/dpc-increase-the-train-data-v02 (18 votes)

**Code**: https://www.kaggle.com/code/seraquevence/dpc-increase-the-train-data-v02

**種別**: 手動PDF抽出によるtraining data拡張

| 項目 | 内容 |
|------|------|
| ソース1 | Larsen 2002 - The Aššur-nada Archive (PIHANS 96) → **13テキストの翻訳を手動転記** |
| ソース2 | Dercksen - Six Texts (Kt c/k series) → **6テキスト** |
| published_textsとの紐付け | aliases/excavation_noでjoin → transliteration取得 |
| 出力 | train_plus.csv（約18件の追加対訳ペア） |

**特徴**:
- **PDFから手動で翻訳を転記**するという地道なアプローチ
- published_texts.csvにtransliterationがあるがtranslationがないテキストに翻訳を追加
- 量は少ない(~18件)が、低リソース環境では貴重な追加データ

#### 📊 aaronbornstein/eng-akk-low-resource-sent-alignment (29 votes)

**Code**: https://www.kaggle.com/code/aaronbornstein/eng-akk-low-resource-sent-alignment

**種別**: 高度な文アライメントパイプライン

| 項目 | 内容 |
|------|------|
| 手法 | **IBM Model 1 (双方向)** + **Viterbi DP** |
| スコアリング | 翻訳確率 + IDF + Soundex(音韻照合) + 対数正規長さ比率 |
| アンカー検出 | 固有名詞(Soundex), gap markers, 数詞 |
| 追加辞書 | OA_Lexicon_eBL.csv (form→lexeme), eBL Dictionary |
| Sumerogram | DUMU→son, DAM→wife等のlogogram→英語マッピング |
| 品質フィルタリング | Normalized Regret Scoring + Monotonic Anchoring |

**手法詳細**:
1. **IBM Model 1**: EMアルゴリズムで双方向翻訳確率を学習
2. **AlignmentScorer**: 多特徴量スコアリング（翻訳確率、IDF重み、fuzzy/音韻マッチ、長さ比率prior、clause境界ヒューリスティクス）
3. **Viterbi DP**: 動的計画法で文全体の最適アライメントを求める（対角prior付き）
4. **アンカー**: 高信頼アライメント点（固有名詞のSoundex一致、gap marker、数詞）を先に固定
5. **品質フィルタリング**: 低品質ペアを除外

**特徴**: このコンペで最も高度なアライメント手法。NMTモデル学習用の高品質並列コーパス生成が目的。
独立代名詞(anāku, attā等)やclause接辞(-ma, umma等)のアッカド語知識を活用。

### 3.3 非Neural手法

#### 📋 leiwong/dps-baseline-extended-dataset (31 votes)

**Code**: https://www.kaggle.com/code/leiwong/dps-baseline-extended-dataset

**種別**: TF-IDF検索ベースの翻訳（Neural Trainingなし）

- 拡張データセット(7,953テキスト)からTF-IDFでtest入力に最も類似するtransliterationを検索
- 見つかった類似テキストのtranslationをそのまま使用
- Neural modelを一切使わないretrieval-based approach

### 3.4 Training Codeは非公開だが、使用モデルが公開されているもの

#### ⚠️ vitorhugobarbedo/lb-35-9 (LB 35.9)

- **Model A**: `assiaben/final-byt5/byt5-akkadian-optimized-34x` (公開)
- **Model B**: `vitorhugobarbedo/newmodelf0` (自前学習、**EMAモデル**、公開dataset)
- Training codeは非公開だが、model Bのパス名から: `byt5_finetune_v1/final_model_ema`
  - EMA (Exponential Moving Average) を使った学習
  - v1なので初期バージョンの可能性

**推論の差別化ポイント**:
- **Weighted MBR**: chrF++(0.55) + BLEU(0.25) + Jaccard(0.20) + Length bonus(0.10)
- **Multi-temperature sampling**: 3温度(0.60, 0.80, 1.05) × 2候補/温度 = 6 sampling候補
- Diverse beam search (準備されているがOFF)
- ホスト推奨の後処理修正7点（括弧保持、5/12shekel修正など）

#### ⚠️ giovannyrodrguez/lb-35-9 (LB 35.9)

- **Model A**: `giovannyrodrguez/modelofinalbyt5/byt5-akkadian-model_final` (**Private Dataset**)
- **Model B**: `giovannyrodrguez/nomarl36/byt5-akkadian-model_final` (**Private Dataset**)
- 両モデルともPrivateなので学習詳細不明
- 推論はgiovannyrodrguezオリジナルのcross-model MBR (chrF++のみ)

#### ⚠️ assiaben/byt5-akkadian-optimized-34x (多くのノートブックで使用)

- 学習コードは非公開
- モデルは`assiaben/final-byt5`としてKaggle Datasetsに公開
- 名前の「34x」は34倍のデータ拡張？または34エポック？
- 多くの高スコアノートブック(33.5-35.9)のベースモデルとして使用

#### ⚠️ mattiaangeli/byt5-akkadian-mbr v2/v5/v6 (MBRの定番モデル)

- 学習コードは非公開
- モデルはKaggle Modelsに公開
- MBRデコーディング用に最適化されたモデル
- meenalsinha(35.3)やlgregory(33.0)など多数が使用

---

## 4. モデル系譜図

```
google/byt5-small
  └─ takamichitoda/dpc-starter-train (LB 26.9)  ← 唯一の完全公開training code
      └─ qifeihhh666 (LB 27.2) [+TM, sentence alignment改良]

google/byt5-base
  └─ jeanjean111/byt5-base-big-data2 (大規模データ学習)
      └─ llkh0a/byt5-akkadian-model (LB ?) ← 公開training code (resume training)

assiaben/byt5-akkadian-optimized-34x (training非公開、モデル公開)
  ├─ assiaben (LB 33.5)
  ├─ anthonytherrien (LB 33.5)
  ├─ prayagp1 (LB 33.5)
  ├─ vitorhugobarbedo (LB 35.9) [+ 自前EMAモデル]
  └─ meenalsinha (LB 35.3) [+ mattiaangeli MBR]

mattiaangeli/byt5-akkadian-mbr-v2 (training非公開、モデル公開)
  ├─ mattiaangeli deep-pasta-mbr (LB 33.2)
  ├─ mattiaangeli deep-pasta-mbr-v2 (LB 34.1)
  ├─ lgregory (LB 33.0)
  └─ meenalsinha (LB 35.3) [+ assiaben]

kkashyap14/byt5-akkadian-en-v5/v6/v7 (training非公開)
  └─ kkashyap14 Model Soup (LB 34.4)

giovannyrodrguez/modelofinalbyt5 + nomarl36 (training非公開、Private Dataset)
  └─ giovannyrodrguez (LB 35.9)
```

---

## 5. LBスコアの推移

| 手法 | Best Score |
|------|-----------|
| ByT5-small baseline (beam=4) | 26.9 |
| + Translation Memory | 27.2 |
| ByT5-base + 双方向学習 + resume train | ~30-31 |
| + 3モデルensemble | 31.8-32.4 |
| + MBR decoding (単一モデル) | 33.0-33.5 |
| + MBR v2 | 34.1 |
| + Model Soup (v5/v6/v7) | 34.4 |
| + Cross-model MBR (2モデル) | **35.3** |
| + 自前モデル + weighted MBR + multi-temp | **35.9** |

---

## 6. Training手法まとめ（公開情報から推定）

### 確認済みのTraining設定（全公開ノートブック比較）

| 項目 | Starter (small) | llkh0a (base) | qifeihhh666 (base) | adarsh2626 (small) |
|------|----------------|---------------|--------------------|--------------------|
| ベースモデル | byt5-small | byt5-base | byt5-base | byt5-small |
| データ前処理 | sentence align | sentence align | **gap-augmented** | **独自cleaning** |
| データ拡張 | なし | **双方向学習** | なし | なし |
| Optimizer | AdamW | **Adafactor** | AdamW | AdamW |
| Label Smoothing | なし | **0.2** | なし | なし |
| LR | 2e-4 | 1e-4 | 2e-4 | 1e-4 |
| LR Scheduler | linear | linear | **cosine** | linear |
| Epochs | 20 | 1 (resume) | 10 | 20 |
| Effective Batch | 8 | 16 | 4 | 8 |
| FP16 | False | False | False | False |
| 評価指標 | chrF | **geo_mean** | chrF | なし |
| Task Prefix | ✅ | ✅ | ✅ | ✅(定義のみ) |

### データ改善手法の比較

| 手法 | 作者 | アプローチ | 品質 |
|------|------|-----------|------|
| simple_sentence_aligner | starter | 行数一致なら1:1分割 | 低（雑なマッチ） |
| chrF++ fuzzy matching | zhangyue199 | chrF++スコアで最適clause選択 | 中（fuzzyだが自動） |
| **Viterbi DP + IBM Model 1** | **aaronbornstein** | **双方向翻訳確率+音韻+IDF+DP** | **高（最も高度）** |
| PDF手動転記 | seraquevence | Larsen 2002等から翻訳を転記 | 高（手動だが少量） |
| gap-augmented data | qifeihhh666 | gap/big_gapの正規化処理 | 中（前処理改善） |
| transliteration cleaning | adarsh2626 | 記号・括弧除去+gap token化 | 中（ノイズ除去） |

### 非公開だが推定される手法
- **assiaben (34x)**: 大量のデータ拡張 or 長時間学習、byt5-small/base
- **mattiaangeli (mbr)**: MBR最適化のためのfine-tuning（sampling分布が広くなるよう学習？）
- **vitorhugobarbedo (EMA)**: EMA付きfine-tuning、augmented data
- **giovannyrodrguez (2モデル)**: 異なる設定で2モデル学習、両方Private

---

## 7. 我々の実験に取り入れるべきアイデア

### 堅実案（確実にスコアが上がる見込み）

**学習設定**:
1. **Adafactor + label_smoothing=0.2** — llkh0aで確認済み。定石
2. **双方向学習 (Akk→Eng + Eng→Akk)** — データ2倍、llkh0aで実装済み
3. **geo_mean(chrF × BLEU) で best model selection** — コンペ指標に近い
4. **Resume training** — 公開モデル(byt5-base-big-data2等)から継続学習
5. **EMAモデル** — vitorhugobarbedoがLB 35.9で使用。学習の安定化+汎化
6. **cosine LR scheduler** — qifeihhh666が使用。linear schedulerより良い可能性

**データ改善**:
7. **高度な文アライメント (Viterbi DP)** — aaronbornsteinの手法。IBM Model 1 + Soundex + IDF + DP
8. **gap/transliteration cleaning** — adarsh2626/qifeihhh666式の前処理。`<gap>`/`<big_gap>` token化
9. **PDF手動抽出データの追加** — seraquevenceの手法。Larsen 2002等から~18件追加

**推論**:
10. **Weighted MBR (chrF++ + BLEU + Jaccard)** — chrF++単体より良い結果(35.9)
11. **Multi-temperature sampling** — 候補の多様性向上。0.6/0.8/1.05の3温度
12. **ホスト推奨の後処理修正** — 括弧保持、5/12shekel修正、ḫ→h変換、stray mark除去

### 爆発案（失敗リスク高いが当たれば大きい）

1. **大規模モデルへのスケール** — byt5-large / byt5-xl。公開ノートブックでは誰も試していない
2. **MBR-aware training** — MBRデコーディングのchrF++をloss関数に組み込む
3. **Domain-Adaptive Pre-Training (DAPT)** — ORACC/CDLIコーパスでの事前学習
4. **Self-Training / Back-Translation** — Eng→Akkモデルで合成データ生成
5. **Contrastive Learning** — transliterationの表記揺れに対するロバスト性向上
6. **LLM-as-Reranker** — MBR候補をLLMでリランキング（Gemma-3等）
7. **異種モデルアンサンブル** — mBART/NLLB/mT5をMBRプールに追加
8. **aaronbornsteinアライメント → 全trainingデータ再構築** — 最高品質の文ペアで学習し直す

---

## 8. 主要モデル一覧（Kaggle Datasets/Models）

| モデル名 | 作成者 | 公開 | 用途 | 使用NB数 |
|----------|--------|:---:|------|---------|
| byt5-akkadian-optimized-34x | assiaben | ✅ | 汎用推論 | 6+ |
| byt5-akkadian-mbr v2 | mattiaangeli | ✅ | MBR推論用 | 5+ |
| byt5-akkadian-mbr v5/v6 | mattiaangeli | ✅ | MBR推論用 | 3+ |
| byt5-base-big-data2 | jeanjean111 | ✅ | 大規模データ学習 | 2 |
| byt5-akkadian-model | llkh0a | ✅ | 汎用 | 3 |
| newmodelf0 (EMA) | vitorhugobarbedo | ✅ | LB35.9用 | 1 |
| byt5-akkadian-en-v5/v6/v7 | kkashyap14 | ✅ | Model Soup | 1 |
| modelofinalbyt5, nomarl36 | giovannyrodrguez | ❌ | LB35.9用 | 1 |

---

## 9. 技術的メモ

- **FP16は使わない**: ByT5でNaN頻発。BF16またはFP32を使用
- **BetterTransformer**: optimumライブラリで20-50%高速化
- **Bucket Batching**: 入力長でバケット分け→パディング削減→推論高速化
- **Task Prefix**: `"translate Akkadian to English: "` が標準（小文字始まり）
- **max_length**: 入力512、出力256-384
- **Adafactor optimizer**: AdamWより省メモリで、ByT5との相性良好
- **generation_max_length=512**: ByT5はバイトレベルなので長めに設定必要
