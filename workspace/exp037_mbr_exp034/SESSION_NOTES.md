# SESSION_NOTES: exp037_mbr_exp034

## セッション情報
- **日付**: 2026-03-18
- **作業フォルダ**: workspace/exp037_mbr_exp034
- **目標**: exp034モデルでのデコーディング戦略最適化（MBR → 温度探索 → weighted MBR → kNN rerank → round-trip rerank）

## 仮説
- MBRで1モデルのスコアを下回るのは直感に反する
- 過去のMBR試行(notebook式, rep_penalty=1.2)はdoc-CV崩壊 → ペナルティなしでクリーンに検証
- beam候補は多様性が低い → sampling（多温度）で候補の多様性を確保
- MBRのchrF++ recallバイアスが繰り返しを選好 → weighted MBRやkNN rerankで解消できるか

## 試したアプローチと結果

### Phase 1: Beam MBR（3候補）

| アプローチ | 変更点 | sent-CV | doc-CV | rep%(sent) | 時間 |
|-----------|--------|---------|--------|------------|------|
| greedy | num_beams=1 | 37.38 | 24.70 | 13.9% | 339s |
| beam4 | num_beams=4 | 36.71 | 24.31 | 12.5% | 379s |
| MBR 3候補(beam) | beam4, n_ret=3, chrF++ | 37.40 | 24.71 | 13.3% | 699s |

### Phase 2: Sampling MBR（6/7候補）

| アプローチ | 変更点 | sent-CV | doc-CV | rep%(sent) | 時間 |
|-----------|--------|---------|--------|------------|------|
| sample_6 | temp(0.6/0.8/1.05)×2, chrF++ MBR | **38.48** | **26.01** | 17.3% | 1160s |
| greedy+sample_7 | greedy+上記6候補, chrF++ MBR | 38.27 | 25.71 | 17.0% | 1498s |

**LB検証: sample_6構成で提出 → LB 29.4（beam4 LB 31.7から-2.3pt悪化）**
CVでは+1.10ptなのにLBで大幅悪化。rep%増加（13.9→17.3%）がLBで拡大した可能性。

### Phase 3: 温度アブレーション（6温度×1候補, sent-CVのみ）

各温度で1候補生成し個別評価（seed=42固定）:

| temp | chrF++ | BLEU | geo | rep% |
|------|--------|------|-----|------|
| greedy (t=0) | 49.13 | 28.44 | 37.38 | 13.9% |
| t=0.2 | 48.95 | 28.21 | 37.16 | 13.9% |
| **t=0.4** | **49.21** | **28.62** | **37.53** | **12.7%** |
| t=0.6 | 48.76 | 28.22 | 37.09 | 13.5% |
| t=0.8 | 48.09 | 27.60 | 36.43 | 16.2% |
| t=1.05 | 46.74 | 26.40 | 35.13 | 13.3% |

**t=0.4が単体ベスト**（geo最高かつrep最低）。t=0.8以上は品質低下。

chrF++ MBR組み合わせTop5:

| 組み合わせ | geo | rep% |
|-----------|-----|------|
| t=0.2+0.4+0.8+1.05 | 38.59 | 17.0% |
| t=0.4+0.6+0.8+1.05 | 38.58 | 17.5% |
| greedy+0.4+0.6+0.8+1.05 | 38.56 | 17.7% |
| t=0.2+0.4+0.6 | 38.51 | 16.8% |
| t=0.2+0.4+1.05 | 38.51 | 16.4% |

**全MBR組み合わせでrep%が単体より悪化** — MBRの構造的問題を確認。

### Phase 4: Weighted MBR（5スコアリング手法 × 16組み合わせ）

beam4 top-3候補も追加し、MBRスコアリング手法を比較:

| 手法 | 重み配分 |
|------|---------|
| chrf_only | chrF++ 100% |
| weighted_full | chrF++(0.55) + BLEU(0.25) + Jaccard(0.20) + LenBonus(0.10) |
| weighted_no_len | chrF++(0.60) + BLEU(0.25) + Jaccard(0.15) |
| bleu_heavy | chrF++(0.40) + BLEU(0.40) + Jaccard(0.10) + LenBonus(0.10) |
| jaccard_heavy | chrF++(0.40) + BLEU(0.20) + Jaccard(0.30) + LenBonus(0.10) |

**2候補ではスコアリング手法の違いが出ない**（beam4_0+t=0.4は全手法でgeo=38.41, rep=13.9%）。
3候補以上でweighted MBRはrep%を1-3pt改善するが、geoも下がる:

| 構成 | chrf_only | jaccard_heavy |
|------|-----------|---------------|
| beam4_0+t=0.4+t=0.6 | geo=38.69, rep=16.2% | geo=38.13, **rep=14.3%** |
| beam4_0+beam4_1+t=0.4+t=0.6 | geo=38.25, rep=14.6% | geo=37.88, **rep=13.3%** |

### Phase 5: kNN pseudo-reference rerank

sentence_aligned.csvのtrain fold文をchar 3-6gram TF-IDFで検索し、top-5近傍翻訳を疑似参照として候補をスコアリング。MBRと直接比較:

| 候補 | MBR_chrf geo | MBR rep% | kNN_chrf geo | kNN rep% |
|------|-------------|----------|-------------|----------|
| beam4_0+t=0.4 | **38.41** | 13.9% | 37.72 | **12.3%** |
| t=0.2+t=0.4 | **38.40** | 14.8% | 38.06 | **12.9%** |
| t=0.2+t=0.4+t=0.6 | **38.51** | 16.8% | 38.36 | **13.9%** |
| beam4_0+t=0.4+t=0.6 | **38.69** | 16.2% | 37.94 | **13.3%** |
| beam4_0+5cand | **38.71** | 17.5% | 38.17 | **14.8%** |

kNN rerankの全手法比較（t=0.2+t=0.4+t=0.6）:

| 手法 | geo | rep% |
|------|-----|------|
| kNN_chrf | 38.36 | 13.9% |
| kNN_wchrf (類似度加重) | 38.32 | 13.9% |
| kNN_geo (geo_mean基準) | 38.00 | 12.7% |

**kNN rerankは単体以下のrep%を実現するケースあり**（beam4_0+t=0.4: 12.3% < 両単体）。
MBRとは逆で、疑似参照に繰り返しがないため繰り返し候補が不利になる。
ただしgeoはMBRより0.1-0.7pt低い。

### Phase 6: Round-trip reranking（逆翻訳によるリランキング）

候補の英語テキストを `"translate English to Akkadian: "` プレフィックスで逆翻訳し、元のアッカド語ソースとの類似度で候補を選択。
モデルは双方向学習済み（Akk→Eng / Eng→Akk）なので同一モデルで逆翻訳可能。

#### BT品質（逆翻訳 vs ソースの平均chrF++）

| 候補 | BT chrF++ |
|------|-----------|
| beam4_0 | 0.6511 (最高) |
| t=0.4 | 0.6421 |
| greedy | 0.6380 |
| t=0.2 | 0.6355 |
| t=0.8 | 0.6298 |
| t=0.6 | 0.6275 |
| t=1.05 | 0.6067 (最低) |

#### 選択手法（11手法）

| カテゴリ | 手法 | 説明 |
|---------|------|------|
| ベースライン | eng_MBR_chrf | 標準chrF++ MBR（英語側consensus） |
| Direct | rt_chrf | BT vs ソースのchrF++ |
| Direct | rt_bleu | BT vs ソースのBLEU |
| Direct | rt_geo | BT vs ソースのgeo(chrF++×BLEU) |
| Direct | rt_jaccard | BT vs ソースのword Jaccard |
| Direct | rt_weighted | chrF++(0.55)+BLEU(0.25)+Jaccard(0.20)+LenBonus(0.10) |
| MBR on BT | rt_mbr_chrf | BT同士のchrF++ consensus |
| MBR on BT | rt_mbr_weighted | BT同士のweighted consensus |
| Hybrid | rt_hybrid_chrf | 0.5*direct + 0.5*mbr_consensus（正規化混合） |
| Clustering | rt_cosine | TF-IDF char n-gram cosine（BT vs ソース） |
| Clustering | rt_medoid | BT同士の平均cosine最大（medoid） |

#### 結果: geo上位

| 構成 | 手法 | chrF++ | BLEU | geo | rep% |
|------|------|--------|------|-----|------|
| beam4_0+5cand | eng_MBR_chrf | 50.21 | 29.85 | **38.71** | 17.5% |
| beam4_0+t=0.4+t=0.6 | eng_MBR_chrf | 50.17 | 29.84 | **38.69** | 16.2% |
| t=0.2+0.4+0.6 | **rt_chrf** | 50.09 | 29.41 | **38.38** | **13.3%** |
| t=0.2+0.4+0.6 | **rt_weighted** | 50.07 | 29.32 | **38.31** | **12.5%** |
| t=0.2+0.4+0.6 | rt_hybrid_chrf | 49.77 | 29.49 | 38.31 | 14.8% |
| beam4_0+5cand | rt_hybrid_chrf | 49.78 | 29.40 | 38.26 | 15.2% |
| t=0.2+0.4+0.6 | **rt_geo** | 49.96 | 29.18 | **38.18** | **12.1%** |
| t=0.2+0.4 | rt_weighted | 49.87 | 29.20 | 38.16 | 13.1% |
| t=0.2+0.4+0.6 | **rt_bleu** | 49.86 | 29.12 | **38.10** | **11.9%** |
| t=0.2+0.4 | rt_chrf | 49.82 | 29.13 | 38.10 | 12.9% |

#### 結果: rep%最低

| 構成 | 手法 | geo | rep% |
|------|------|-----|------|
| t=0.2+0.4+0.6 | rt_bleu | 38.10 | **11.9%** |
| t=0.2+0.4+0.8+1.05 | rt_bleu | 37.64 | **11.9%** |
| beam4_0+5cand | rt_bleu | 37.68 | **11.9%** |
| t=0.2+0.4+0.6 | rt_geo | 38.18 | **12.1%** |
| beam4_0+5cand | rt_geo | 37.79 | 12.3% |
| beam4_0+5cand | rt_weighted | 37.96 | 12.3% |
| t=0.2+0.4+0.6 | rt_weighted | 38.31 | 12.5% |
| beam4_0 (single) | - | 36.71 | 12.5% |
| t=0.4 (single) | - | 37.53 | 12.7% |

#### Phase 5 (kNN) との比較（t=0.2+t=0.4+t=0.6構成）

| 手法 | geo | rep% | 備考 |
|------|-----|------|------|
| eng_MBR_chrf | 38.51 | 16.8% | MBR: geo最高だがrep最悪 |
| kNN_chrf | 38.36 | 13.9% | kNN: repを単体並みに抑制 |
| **rt_chrf** | **38.38** | **13.3%** | **RT: kNNと同等geo、rep改善** |
| **rt_weighted** | **38.31** | **12.5%** | **RT: kNNよりrep大幅改善** |
| **rt_bleu** | **38.10** | **11.9%** | **RT: rep最低だがgeo若干低** |
| rt_mbr_chrf | 38.15 | 15.2% | BT側MBR: MBRと同じrep問題 |

### Phase 8: 2モデルアンサンブル + Round-trip rerank

Model A (exp034 ByT5-base) と Model B (s1_exp007 ByT5-large) の beam4 1-best を候補とし、
各モデルで逆翻訳（beam4）→ rt_weighted 等で選択。モデルは1つずつロード→推論→保存→解放（メモリ効率）。

| config | sent-geo | sent-rep% | doc-geo | doc-rep% |
|--------|----------|-----------|---------|----------|
| model_A (exp034 base) 単体 | 36.68 | 12.5% | 24.31 | 64.6% |
| model_B (s1_exp007 large) 単体 | 37.98 | 10.8% | 26.05 | 65.0% |
| **A+B \| rt_weighted** | 37.90 | 12.1% | **26.32** | 68.7% |
| A+B \| rt_chrf | 38.04 | 12.3% | 26.30 | 67.7% |
| A+B \| rt_bleu | 37.77 | 12.3% | 26.18 | 67.3% |
| A+B \| eng_MBR_chrf | **39.00** | 13.9% | **26.47** | 70.4% |

- Pick比率: rt_weighted A=176, B=121 (59:41) / eng_MBR_chrf A=153, B=144 (52:48)
- doc-CVでは全アンサンブル手法がB単体(26.05)を上回る
- eng_MBR_chrfがdoc-geo最高(26.47)だがrep%が70.4%に悪化（LBで不利の可能性）
- 逆翻訳もbeam4に変更（Phase 6ではgreedy(num_beams=1)だった）

**LB提出結果: A+B | rt_weighted → LB 32.6（beam4単体 LB 31.7から+0.9pt改善）**

## 詳細スコア（Phase 1-2）

| 手法 | chrF++ | BLEU | geo | rep% |
|------|--------|------|-----|------|
| greedy sent-CV | 49.13 | 28.44 | 37.38 | 13.9% |
| beam4 sent-CV | 48.87 | 27.58 | 36.71 | 12.5% |
| MBR 3cand sent-CV | 49.33 | 28.35 | 37.40 | 13.3% |
| sample_6 sent-CV | 49.76 | 29.76 | 38.48 | 17.3% |
| greedy+sample_7 sent-CV | 49.75 | 29.43 | 38.27 | 17.0% |
| greedy doc-CV | 37.96 | 16.08 | 24.70 | 66.0% |
| beam4 doc-CV | 37.78 | 15.64 | 24.31 | 64.6% |
| MBR 3cand doc-CV | 38.16 | 16.00 | 24.71 | 63.6% |
| sample_6 doc-CV | 38.90 | 17.39 | 26.01 | 71.7% |
| greedy+sample_7 doc-CV | 38.81 | 17.03 | 25.71 | 67.3% |

## ファイル構成
- `src/eval_mbr.py` — MBR評価スクリプト（動的パディング+ソート）
- `src/temp_ablation.py` — 温度アブレーション + MBR組み合わせグリッドサーチ
- `src/weighted_mbr_ablation.py` — weighted MBR比較（beam候補含む）、候補pklキャッシュ
- `src/knn_rerank.py` — kNN pseudo-reference rerank
- `src/roundtrip_rerank.py` — Round-trip（逆翻訳）rerank（11手法比較）
- `results/mbr_eval_fold3.json` — Phase 1 beam MBR結果
- `results/sampling_mbr_fold3.json` — Phase 2 sampling MBR結果
- `results/temp_ablation_fold3.json` — Phase 3 温度アブレーション結果
- `results/weighted_mbr_fold3.json` — Phase 4 weighted MBR結果
- `results/knn_rerank_fold3.json` — Phase 5 kNN rerank結果
- `results/roundtrip_rerank_fold3.json` — Phase 6 round-trip rerank結果
- `results/eval_full_rerank_fold3.json` — Phase 7 eval_full結果（sent+doc）
- `results/ensemble_rt_fold3.json` — Phase 8 2モデルアンサンブルRT結果
- `results/ensemble_cache/` — Phase 8 モデル別推論キャッシュ（model_a/b_fold3.pkl）
- `src/eval_ensemble_rt.py` — Phase 8 2モデルアンサンブル評価スクリプト
- `src/eval_ensemble_mbr_notebook.py` — Phase 9 ノートブック式ensemble MBR評価
- `src/eval_postprocess_comparison.py` — Phase 10 後処理差分比較
- `results/ensemble_mbr_cache/` — Phase 9 multi-candidate推論キャッシュ（model_a/b_multi_fold3.pkl）
- `results/ensemble_mbr_notebook_fold3.json` — Phase 9 結果
- `results/postprocess_comparison_fold3.json` — Phase 10 結果
- `results/candidates_fold3.pkl` — sent候補キャッシュ（6温度+beam4 top-3）
- `results/backtrans_fold3.pkl` — sent逆翻訳キャッシュ
- `results/doc_candidates_fold3.pkl` — doc候補キャッシュ（beam4_0, t=0.2/0.4/0.6）
- `results/doc_backtrans_fold3.pkl` — doc逆翻訳キャッシュ
- `results/*.log` — 各実行ログ

## 重要な知見

### MBRの構造的問題: chrF++ recallバイアス → 繰り返し選好
- chrF++のrecallが繰り返し候補を優遇（繰り返しにより参照のn-gramを全て含む → recall高）
- MBRは候補同士を比較するため、繰り返し候補が「他の候補と最も類似」と判定される
- **候補数が増えるほどrep%が悪化**（繰り返し候補がプールに入る確率増）
- 2候補でもMBRは各候補の繰り返しサンプルを「拾い集める」ため、個別より悪化
- weighted MBR（Jaccard/BLEU追加）は3候補以上でrep%を改善するが、2候補では効果なし
- **CVでのMBR改善がLBで-2.3pt悪化** — rep%増加がLBで拡大

### kNN rerankはMBRのrep問題を解決するが、geoは劣る
- 疑似参照（trainの翻訳）には繰り返しがない → 繰り返し候補が不利になる
- rep%が単体以下になるケースあり（MBRでは不可能）
- ただしgeoはMBRより0.1-0.7pt低い（疑似参照のノイズ）
- kNN_chrfがバランス最良、kNN_geoはrep最低だがgeo低下大

### Round-trip rerank: MBR+kNNの上位互換
- **原理**: 候補を逆翻訳（Eng→Akk）し、元のソースとの類似度で選択。忠実度を直接測定
- **MBRのrep問題を完全回避**: 繰り返し英語→逆翻訳で元ソースと乖離→自然に淘汰される
- **kNNよりrep%がさらに低い**: rt_bleuで11.9%（kNN_chrf 13.9%、単体t=0.4 12.7%を下回る）
- **geoもkNNと同等以上**: t=0.2+0.4+0.6 | rt_chrf geo=38.38（kNN_chrf 38.36と同等）
- **ベスト構成**: t=0.2+0.4+0.6 | rt_chrf（geo=38.38, rep=13.3%）or rt_weighted（geo=38.31, rep=12.5%）
- **Direct系 > MBR on BT系**: BT同士のconsensus（rt_mbr_chrf）はMBRと同じrep問題を持つ（15.2%）
- **medoidは性能低い**: クラスタリング系は全体的に弱い（geo=37.16-37.33、ベースライン以下のケースも）
- **hybridは中間的**: direct+mbr混合で両者の中間のrep%（14.8%）。明確な優位性なし
- **BT品質**: beam4_0が最高（0.6511）。確定的デコーディングの方が逆翻訳精度は高い
- **提出時の追加コスト**: 候補数×1回の逆翻訳推論。6候補で約20分追加（Kaggle 9h制限内に余裕あり）
- **単モデル内RT rerankはLBで悪化**: rt_chrf LB 30.8, rt_weighted LB 31.2（beam4 LB 31.7以下）。CVとLBの乖離が大きい

### 2モデルアンサンブル + Round-trip rerank
- **LB 32.6（beam4単体 31.7 → +0.9pt）**: 2モデルの多様性がCVだけでなくLBでも有効
- **単モデルRT rerankはLB悪化するが、異モデルアンサンブル+RTはLB改善**: 単モデル候補（同一分布）からの選択はCV過適合しやすいが、異モデル候補は分布が異なりLBでも汎化
- **逆翻訳beam4**: Phase 6のgreedy BTからbeam4 BTに変更。BT品質向上で判定精度改善の可能性
- **rt_weighted A:B=59:41**: Aが多く選ばれるが、Bも4割選ばれており適度な多様性

### 温度の最適値
- **t=0.4が単体ベスト**（geo=37.53, rep=12.7%）。greedy(37.38)より+0.15pt
- t=0.8以上は品質低下が顕著（ByT5のバイトレベルsamplingでノイズ増大）
- beam4_0は確定的でrep最低(12.5%)だがgeoも最低(36.71)

### Phase 7: eval_full（sent-CV + doc-CV 統合評価）

有力6構成をdoc-CVでも評価（doc-level候補生成→逆翻訳→リランク）:

| 構成 | sent-geo | sent-rep% | doc-geo | doc-rep% |
|------|----------|-----------|---------|----------|
| beam4 (現行LB31.7) | 36.71 | 12.5% | 24.40 | 66.0% |
| t=0.4 単体 | 37.53 | 12.7% | 24.70 | 64.6% |
| **t=0.2+0.4+0.6 \| rt_chrf** | **38.38** | 13.3% | **25.67** | 67.3% |
| **t=0.2+0.4+0.6 \| rt_weighted** | 38.31 | **12.5%** | 25.61 | 67.7% |
| t=0.2+0.4+0.6 \| rt_bleu | 38.10 | **11.9%** | 25.46 | 66.7% |
| beam4_0+t=0.4 \| eng_MBR_chrf | 38.41 | 13.9% | 25.13 | 66.3% |

**doc-CVでもround-trip rerankがベスト**:
- rt_chrf: beam4比 sent +1.67pt, doc +1.27pt
- rt_weighted: beam4比 sent +1.60pt, doc +1.21pt
- eng_MBR_chrf: sent-geoは最高(38.41)だがdoc-geoでは3位以下(25.13)
- doc-CVのrep%はどの手法も64-68%で大差なし（doc-levelは元々高rep%）

### LB提出結果
- beam4 (現行提出, exp034単体): LB 31.7
- sample_6 MBR: LB 29.4 (-2.3pt) — rep%増加がLBで悪影響
- t=0.2+0.4+0.6 | rt_chrf: LB 30.8 (-0.9pt) — 単モデル内RT rerankはLB悪化
- t=0.2+0.4+0.6 | rt_weighted: LB 31.2 (-0.5pt) — 同上、weightedの方がマシだがbeam4以下
- **A(exp034)+B(s1_exp007) | rt_weighted: LB 32.6 (+0.9pt)** — 2モデルアンサンブルでLBベスト更新
- A(exp034)+B(s1_exp007) | eng_MBR_chrf: LB 31.4 (-0.3pt) — 2候補MBRはbeam4単体以下。選択力不足

### Phase 9: Notebook式 Ensemble MBR (beam4x4 + sampling2) × 2モデル

Kaggle公開ノートブック "LB 35.9 Ensembling & Post Processing Baseline" (waterjoe) の方式を再現。
各モデルからbeam8 top-4候補 + stochastic sampling 2候補 = 6候補/model、2モデルで最大12候補をchrF++ MBRで選択。

生成パラメータ（ノートブック準拠）:
- beam: num_beams=8, num_return_sequences=4, length_penalty=1.3, repetition_penalty=1.2
- sampling: top_p=0.92, temperature=0.75, repetition_penalty=1.2, num_return_sequences=2

| config | sent-geo | sent-rep% | doc-geo | doc-rep% |
|--------|----------|-----------|---------|----------|
| model_A (exp034 base) 単体 | 36.35 | 11.4% | 23.31 | 59.9% |
| model_B (s1_exp007 large) 単体 | 38.15 | 11.0% | 25.73 | 60.3% |
| A+B \| 1-best MBR (2 cands) | 38.39 | 12.7% | 25.92 | 63.6% |
| A+B \| beam4x4 MBR (8 cands) | 38.94 | 13.1% | 26.10 | 65.7% |
| **A+B \| full pool MBR (12 cands)** | **39.45** | 14.1% | **26.56** | 67.0% |
| A+B \| rt_weighted (Phase 8参考) | 37.90 | 12.1% | 26.32 | 68.7% |
| A+B \| eng_MBR_chrf (Phase 8参考) | 39.00 | 13.9% | 26.47 | 70.4% |

- full pool MBR (12候補) がsent-geo/doc-geo両方で最高
- ただし候補数が増えるほどrep%悪化（MBRの構造的問題、Phase 2-4と同じ傾向）
- CVでのMBR改善がLBで悪化するリスクは高い（Phase 2のsample_6 MBRでLB -2.3ptの前例）

#### Lexical fidelity (固有名詞辞書) の効果

ノートブックのMBRスコア = 0.8 × chrF++ consensus + 0.2 × lexical fidelity。
OA_Lexicon_eBL.csvからPN(13,424) + GN(334) = 13,758エントリの辞書を構築。

| config | sent-geo (なし/あり) | doc-geo (なし/あり) |
|--------|---------------------|---------------------|
| 1-best MBR (2 cands) | 38.39 / 38.43 | 25.92 / 25.90 |
| beam4x4 MBR (8 cands) | 38.94 / 38.88 | 26.10 / 26.01 |
| full pool MBR (12 cands) | 39.45 / 39.34 | 26.56 / 26.61 |

**lexical fidelityの効果はほぼなし〜微マイナス**。ソースtransliterationと辞書のオーバーラップが薄い。

### Phase 10: 後処理の差分比較

キャッシュ済み候補（full pool 12候補）に対し、MBR前/後にノートブック式 vs v8式の後処理を適用して比較。

| 後処理 | sent-geo | sent-rep% | doc-geo | doc-rep% |
|--------|----------|-----------|---------|----------|
| none (後処理なし) | **39.45** | 14.1% | **26.56** | 67.0% |
| notebook (MBR前) | 39.05 | **13.5%** | 26.30 | **66.7%** |
| v8 (MBR前) | **39.45** | 14.1% | **26.56** | 67.0% |
| notebook_after (MBR後) | 39.11 | 14.1% | 26.19 | 66.7% |
| v8_after (MBR後) | **39.45** | 14.1% | **26.56** | 67.0% |

- **v8後処理はnoneと完全一致** — 現在のモデル出力にv8のルールがマッチするケースがほぼない
- **ノートブック式後処理はむしろ悪化** (sent -0.40pt, doc -0.26pt) — forbidden chars除去がCVリファレンスに含まれるトークンまで消している
- ノートブックのLB 35.9は後処理ではなくモデルの強さが主因
- v8後処理はtest時にパターンが出る可能性があるので残しておく

## コマンド履歴
```bash
# Phase 1: Beam MBR評価
python workspace/exp037_mbr_exp034/src/eval_mbr.py --fold 3 2>&1 | tee workspace/exp037_mbr_exp034/results/eval_mbr.log

# Phase 2: Sampling MBR評価
python workspace/exp037_mbr_exp034/src/eval_mbr.py --fold 3 2>&1 | tee workspace/exp037_mbr_exp034/results/sampling_mbr.log

# Phase 3: 温度アブレーション
python workspace/exp037_mbr_exp034/src/temp_ablation.py --fold 3 2>&1 | tee workspace/exp037_mbr_exp034/results/temp_ablation.log

# Phase 4: Weighted MBR比較
python workspace/exp037_mbr_exp034/src/weighted_mbr_ablation.py --fold 3 2>&1 | tee workspace/exp037_mbr_exp034/results/weighted_mbr.log

# Phase 5: kNN pseudo-reference rerank
python workspace/exp037_mbr_exp034/src/knn_rerank.py --fold 3 --top_k 5 2>&1 | tee workspace/exp037_mbr_exp034/results/knn_rerank.log

# Phase 6: Round-trip rerank（逆翻訳リランキング）
python workspace/exp037_mbr_exp034/src/roundtrip_rerank.py --fold 3 2>&1 | tee workspace/exp037_mbr_exp034/results/roundtrip_rerank.log

# Phase 7: eval_full（sent-CV + doc-CV統合評価）
python workspace/exp037_mbr_exp034/src/eval_full_rerank.py --fold 3 2>&1 | tee workspace/exp037_mbr_exp034/results/eval_full_rerank.log

# Phase 9: Notebook式 Ensemble MBR (beam4x4 + sampling2)
python workspace/exp037_mbr_exp034/src/eval_ensemble_mbr_notebook.py --fold 3 --batch_size 4 2>&1 | tee workspace/exp037_mbr_exp034/results/ensemble_mbr_notebook.log

# Phase 10: 後処理の差分比較
python workspace/exp037_mbr_exp034/src/eval_postprocess_comparison.py --fold 3 2>&1 | tee workspace/exp037_mbr_exp034/results/postprocess_comparison.log
```

## 次のステップ
- **3モデル以上のアンサンブル**: exp038(BT augment)等を追加してrt_weightedで3候補選択
- **A+B | rt_chrf 提出**: rt_chrfはsent-CV最高(38.04)、LBでrt_weighted(32.6)を超えるか検証
- ~~A+B | eng_MBR_chrf 提出~~: LB 31.4で確認済み。2候補MBRはbeam4以下、不採用
- **モデル改善**: s1_exp007(large)のLB単体提出でbase vs largeのLB差を確認
