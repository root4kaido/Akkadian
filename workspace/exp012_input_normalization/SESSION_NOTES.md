# SESSION_NOTES: exp012_weighted_mbr

## セッション情報
- **日付**: 2026-03-09
- **作業フォルダ**: workspace/exp012_input_normalization
- **目標**: MBR/greedy推論バリエーションのグリッドサーチ。exp011モデル1つで候補生成・スコアリングの全組合せを試す。

## 仮説
- 候補プール拡張（multi-temp sampling）でMBRの質が向上する
- weighted MBR（chrF++ + BLEU + Jaccard）がchrF++単体を上回る
- repetition_penaltyの調整でgreedy精度が改善する

## グリッドサーチ結果（ランキング）

| 順位 | 手法 | geo_clean | geo_raw | 繰り返し率 | 平均長 | 推定時間 | 候補数 |
|------|------|-----------|---------|-----------|--------|---------|--------|
| **1** | **B3: beam4+mt×3=13候補, chrF++** | **34.47** | 34.47 | 31% | 141 | 547s | 13 |
| **2** | **B5: beam4+mt×4=16候補, chrF++** | **34.45** | 34.45 | 32% | 143 | 676s | 16 |
| 3 | B2: beam4+mt×2=10候補, chrF++ | 34.10 | 34.11 | 30% | 137 | 419s | 10 |
| 4 | C2: chrF+++BLEU, 10候補 | 33.61 | 33.62 | 28% | 132 | 421s | 10 |
| 5 | C3: chrF+++BLEU+Jaccard, 10候補 | 33.60 | 33.61 | 29% | 132 | 421s | 10 |
| 6 | greedy (rp=1.2) + cleanup | 33.45 | 33.13 | 38% | 162 | 116s | 1 |
| 7 | B1: beam4+samp2=6候補, chrF++ | 33.44 | 33.44 | 29% | 133 | 250s | 6 |
| 8 | B4: beam8+mt×2=14候補, chrF++ | 33.06 | 33.06 | 29% | 130 | 589s | 14 |
| 9 | greedy (rp=1.3) | 32.91 | 32.90 | 31% | 141 | 100s | 1 |
| 10 | greedy (rp=1.5) | 29.13 | 29.36 | 15% | 111 | 83s | 1 |

## ファイル構成
- `src/eval_weighted_mbr.py` — 初回weighted MBR評価
- `src/grid_search_mbr.py` — 全バリエーションのグリッドサーチ

## 重要な知見

### 候補プール拡張が最重要
- **beam4+mt×3=13候補 が最強 (34.47)**。exp011のgreedy_clean (33.45)から **+1.02pt**
- 候補数: 6→10→13と増やすと33.44→34.10→34.47と単調増加
- ただし16候補(34.45)で頭打ち。13候補がスイートスポット

### beam数を増やすのは逆効果
- beam8+mt×2=14候補 (33.06) < beam4+mt×2=10候補 (34.10)
- beam候補はお互いに似すぎて多様性が低い。samplingの方が有効

### weighted MBRは改善しない
- chrF++単体 (34.10) > chrF+++BLEU (33.61) > chrF+++BLEU+Jaccard (33.60)
- BLEU/Jaccardを追加すると出力が短くなりすぎる（132 vs ref 166）

### repetition_penalty上げはスコア低下
- rp=1.2 (33.45) > rp=1.3 (32.91) > rp=1.5 (29.13)
- 繰り返し減少と引き換えに内容の質も下がる

### repeat_cleanupの効果はMBRではほぼゼロ
- MBR系はraw≈clean（MBR自体が繰り返しを選ばない）
- greedyでのみ+0.32pt (33.13→33.45)

## Submission推奨ランキング

| 優先度 | 手法 | 推定スコア | 実行コスト | 備考 |
|--------|------|-----------|-----------|------|
| **1** | beam4+mt(0.6,0.8,1.05)×3=13候補, chrF++ MBR | **34.47** | 中(~9min/157件) | **最良。メインsub** |
| **sub** | greedy(rp=1.2) + repeat_cleanup | 33.45 | 低(~2min) | **サブsub。最速・安全策。MBR失敗時の保険** |

## 性能変化の記録

| 指標 | exp011ベスト | exp012ベスト | 差分 |
|------|------------|------------|------|
| greedy_clean | 33.45 | 33.45 | 0 (同一) |
| mbr (6候補) | 31.54 | 33.44 | +1.90 (候補プール変更) |
| **best (13候補MBR)** | - | **34.47** | **+1.02** (vs greedy_clean) |

## コマンド履歴
```bash
# 初回weighted MBR評価
python workspace/exp012_input_normalization/src/eval_weighted_mbr.py

# グリッドサーチ
python workspace/exp012_input_normalization/src/grid_search_mbr.py

# Greedy-MBRハイブリッド評価
python workspace/exp012_input_normalization/src/eval_hybrid.py
```

## Greedy-MBRハイブリッド結果

| 手法 | chrF++ | BLEU | geo_mean | 繰り返し率 | 平均長 | 時間 |
|------|--------|------|----------|-----------|--------|------|
| greedy_raw | 42.06 | 26.10 | 33.13 | 38.9% | 162 | 118s |
| greedy_clean | 42.23 | 26.49 | 33.45 | 38.2% | 159 | 118s |
| **hybrid_raw** | **42.88** | **26.21** | **33.52** | **23.6%** | 136 | 502s |
| full_mbr (13候補) | 43.61 | 26.96 | 34.29 | 35.7% | 141 | 940s |

### ハイブリッドの評価
- 繰り返し率は大幅改善（38.9%→23.6%）だがスコア改善は+0.07ptのみ
- full MBRには-0.77pt劣る。greedy出力は繰り返しがなくてもMBRより翻訳品質が低いケースが多い
- Hybrid vs full_mbr: wins=34, loses=39（MBRの方が勝つ）
- **結論: ハイブリッドは精度面では不採用。full MBR（13候補chrF++）が最良**

## 次のステップ
- 最良設定(13候補+chrF++ MBR)でsubmission生成
- 2モデルクロスモデルMBR（別チェックポイント）
- Model Soup（重み平均）
