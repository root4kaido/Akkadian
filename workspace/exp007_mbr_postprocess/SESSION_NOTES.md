# SESSION_NOTES: exp007_mbr_postprocess

## セッション情報
- **日付**: 2026/03/08
- **作業フォルダ**: workspace/exp007_mbr_postprocess
- **目標**: MBRデコーディング + OA_Lexicon後処理 + repeated_removalで推論パイプラインを改善

## 仮説
- MBRデコーディング（beam+sampling候補からchrF++で最良選択）により+3-5pt改善（全トップNBで使用、最大差別化要因）
- OA_Lexicon後処理（固有名詞正規化）により+0.5-1pt改善（takamichitoda NB実証済み、Host「固有名詞が最大ボトルネック」）
- repeated_removal後処理により+0.7pt改善（eda005で定量確認済み）
- 学習不要、exp005のモデルをそのまま使用

## 試したアプローチと結果

| アプローチ | 変更点 | CV (sent) | CV (doc) | LB | 備考 |
|-----------|--------|-----------|----------|-----|------|
| exp005 baseline | greedy decoding | 27.03 | 18.28 | - | 比較基準 |
| **[sent推論]** | | | | | **入力も文レベルにカットして推論（テスト条件に忠実）** |
| greedy (raw) | sent入力greedy | **26.56** | - | - | ベースライン |
| MBR (raw) | sent入力 beam8→4+sampling2 | **27.31** | - | - | **+0.75pt** |
| greedy + post | sent入力 + OA+TM+repeated | 26.61 | - | - | +0.05pt（ほぼ中立） |
| MBR + post | sent入力 + OA+TM+repeated | 27.29 | - | - | +0.73pt（ほぼMBR raw同等） |
| **[doc推論→1文抽出]** | | | | | **参考: doc全体を推論→英語1文目を抽出** |
| greedy (raw) | doc入力greedy | 26.57 | 17.81 | - | sent推論とほぼ同じ |
| MBR (raw) | doc入力MBR→1文抽出 | 28.22 | 11.10 | - | 入力コンテキストが豊富で高スコアだがテスト条件と異なる |

**注**: CV (sent) = 文レベル入力→推論→chrF++/BLEU幾何平均。テストは文レベル入力なのでsent推論が正確な評価。

## ファイル構成
- `src/infer_mbr.py` — doc-level MBRデコーディング推論スクリプト（OA_Lexicon + TM + repeated_removal統合）
- `src/eval_sentence_level.py` — **sent-level推論+評価スクリプト（テスト条件に忠実）**
- `src/evaluate_results.py` — doc-level評価スクリプト
- `results/val_predictions.csv` — doc-level全4手法の予測結果（157サンプル）
- `results/val_predictions_sentence.csv` — **sent-level全4手法の予測結果（157サンプル）**
- `results/run2.log` — doc-level実行ログ
- `results/eval_sent.log` — sent-level実行ログ

## 重要な知見

1. **MBR decodingはsent-levelで+1.65pt改善**（26.57→28.22）。BLEUが特に改善（+2.24pt）
2. **MBRはdoc-levelでは悪化**（17.81→11.10）。max_output_length=512の制約で長文書が途中で切れる
3. **後処理（OA_Lexicon + TM + repeated_removal）の効果は限定的**。MBR上ではほぼ横ばい、greedy上では+1.35pt
4. **Translation Memory**: 24.8% hit rate（train splitのみ使用）。完全一致がある場合は正確な翻訳が得られる
5. **繰り返し問題は依然深刻**: greedy 80.9%、MBR 61.8%。repeated_removalでは根本解決しない
6. **repetition_penalty=1.2は不十分**: 繰り返しを完全には抑制できていない
7. **TMリーク注意**: 初回実行でval splitを含むtrain全体でTM構築→96.8%ヒット→リーク発覚→修正

### MBR設定詳細
- beam: num_beams=8, num_return_sequences=4
- sampling: top_p=0.9, temperature=0.7, 2候補
- 合計6候補からchrF++で最良選択
- repetition_penalty=1.2, length_penalty=1.0

## 性能変化の記録

| 実験 | 変更内容 | sent CV | doc CV | 改善幅(sent) |
|------|---------|---------|--------|-------------|
| greedy raw | ベースライン | 26.57 | 17.81 | - |
| MBR raw | beam8→4 + sampling2 | 28.22 | 11.10 | +1.65 |
| MBR + post | + OA_Lexicon + TM + repeated | 28.22 | 10.37 | ±0.00 |
| greedy + post | + OA_Lexicon + TM + repeated | 27.92 | 17.02 | +1.35 |

## コマンド履歴
```bash
# 推論実行（TMリーク修正後）
python workspace/exp007_mbr_postprocess/src/infer_mbr.py 2>&1 | tee workspace/exp007_mbr_postprocess/results/run2.log

# 評価
python workspace/exp007_mbr_postprocess/src/evaluate_results.py
```

## 次のステップ
- [ ] repetition_penaltyを上げる（1.5〜2.0）で繰り返しをさらに抑制
- [ ] max_output_lengthをdoc-levelで大きくする（1024〜2048）
- [ ] MBR候補数を増やす（sampling 4〜8個）で効果検証
- [ ] 後処理の個別効果を分離測定（repeated_removalだけ、OA_Lexiconだけ、TMだけ）
- [ ] サブミッション生成（MBR rawが最も安全な選択）
