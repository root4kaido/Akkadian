# SESSION_NOTES: exp030_decode_strategy

## セッション情報
- **日付**: 2026-03-13
- **作業フォルダ**: workspace/exp030_decode_strategy
- **目標**: デコード戦略の最適化。beam/MBR/repetition_penalty/length_penaltyの効果を測定

## 仮説
- Notebook調査で、上位ノートブックはMBR decoding (+3-5pt)が最大の差別化要因と判明
- 我々はbeam4+greedyのみで推論しており、ここに大きな伸びしろがある
- repetition_penalty=1.2, length_penalty=1.3が上位の共通パラメータ

## 試したアプローチと結果

| 設定 | chrF++ | BLEU | sent-CV geo | rep% | time |
|------|--------|------|-------------|------|------|
| greedy | 47.03 | 26.57 | 35.35 | 14.1% | 197s |
| beam4 | 46.23 | 25.07 | 34.05 | 12.5% | 217s |
| beam8 | 46.39 | 25.29 | 34.25 | 11.6% | 238s |
| beam4_rp1.2 | 46.41 | 25.34 | 34.29 | 12.3% | 218s |
| beam8_rp1.2 | 46.45 | 25.22 | 34.23 | 11.6% | 236s |
| beam4_lp1.3 | 46.38 | 25.18 | 34.17 | 12.9% | 219s |
| beam8_lp1.3_rp1.2 | 46.67 | 25.42 | 34.44 | 12.1% | 237s |
| mbr_b4s2_t0.7 | 47.02 | 25.97 | 34.94 | 12.5% | 815s |
| **mbr_b4s2_multi** | **47.42** | **26.59** | **35.51** | 14.3% | 1626s |

**ベスト: MBR multi-temp(0.6/0.8/1.05), geo=35.51** (beam4比 +1.46pt, greedy比 +0.16pt)

### 追加sweep: 13候補MBR（exp012ベスト再現）

| 設定 | chrF++ | BLEU | sent-CV geo | rep% | time |
|------|--------|------|-------------|------|------|
| mbr_b4s3_13cand (rp=1.2) | 47.86 | 27.01 | 35.95 | 14.6% | 1709s |
| mbr_b4s3_13cand_lp1.3 | 47.68 | 26.86 | 35.79 | 13.5% | 1711s |
| mbr_b8s3_17cand (beam8, 17候補) | 47.30 | 26.37 | 35.32 | 12.9% | 1738s |
| **mbr_b4s3_13cand_nopen** | **47.89** | **27.36** | **36.20** | 15.4% | 1702s |

**ベスト更新: 13候補ペナルティなし geo=36.20** (+0.69pt vs 10候補)

### バッチ化MBR評価（sent-CV + doc-CV）

| 指標 | chrF++ | BLEU | geo | rep% | time |
|------|--------|------|-----|------|------|
| **sent-CV (481文)** | 48.04 | 27.23 | **36.17** | 15.6% | **1031s** |
| doc-CV (297文) | 37.73 | 15.96 | 24.54 | 72.4% | 1543s |

**exp023 beam4との比較:**
- sent-CV: 35.44 → **36.17 (+0.73pt)**
- doc-CV: 25.52 → 24.54 (-0.98pt) — MBRはdoc-levelでは繰り返しが悪化

**バッチ化による速度改善:**
- sent-CV: 1702s → **1031s (39%高速化)**
- 4000文推定: ~8600s ≈ 2.4時間（Kaggle 9時間制限に余裕）

## ファイル構成
- `src/decode_sweep.py` — デコード戦略グリッドサーチ
- `src/eval_mbr.py` — バッチ化MBR評価（sent-CV + doc-CV）
- `results/decode_sweep_results.txt` — sweep1結果
- `results/decode_sweep_results2.txt` — sweep2結果（13候補）
- `results/mbr_eval_fold3.json` — バッチ化MBR評価結果

## 重要な知見
- **greedyがbeam search全設定を上回る** — sent-CV条件（短文翻訳）ではbeam searchの探索が逆効果
- **MBR 13候補ペナルティなしがベスト (geo=36.20)** — exp012と同じ結論
- **ペナルティは全て逆効果**: rp1.2で-0.25pt、lp1.3でさらに-0.16pt
- **beam8候補は逆効果**: beam4(35.95) > beam8(35.32)。beam候補は多様性が低い
- **MBRはsent-CVで+0.73pt改善だがdoc-CVで-0.98pt悪化** — 長文では繰り返し増加(72.4%)
- **バッチ化で39%高速化**: batch=1→batch=2/4で1702s→1031s
- MBR効果(beam4比+2.12pt)は上位NB報告の+3-5ptに届かない。モデル品質自体の差が大きい

## コマンド履歴
```bash
python workspace/exp030_decode_strategy/src/decode_sweep.py 2>&1 | tee workspace/exp030_decode_strategy/results/decode_sweep_output.log
python workspace/exp030_decode_strategy/src/eval_mbr.py --fold 3 2>&1 | tee workspace/exp030_decode_strategy/results/eval_mbr_fold3.log
```

## 次のステップ
- MBR 13候補nopenをsubmit.pyに組み込んでLB評価
- doc-CVの繰り返し問題: repeat_cleanupの強化 or MBR+greedy hybrid
