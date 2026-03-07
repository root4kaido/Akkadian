# SESSION_NOTES: exp005_label_masking

## セッション情報
- **日付**: 2026-03-07
- **作業フォルダ**: workspace/exp005_label_masking
- **目標**: 英語ラベル文末マスキング + 逆方向encoder拡大で、入出力カバレッジ不一致を解消

## 仮説

trainデータはドキュメントレベル（平均490バイト）だが、max_length=512でtruncateすると：
- アッカド語（diacriticsで1文字2-3バイト）が先にtruncateされる
- 英語ラベルにはアッカド語入力に含まれない後半部分の翻訳が残る
- → ノイジーなラベルで学習品質が低下

**解決策**:
1. 順方向(Akk→Eng): 英語ラベルを512バイトでtruncate後、最後の文末（ピリオド）まで残して残りをマスク
2. 逆方向(Eng→Akk): 英語入力のmax_lengthを1024に拡大（ASCIIなので効率的）

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| ラベルマスキング + 逆方向encoder拡大 | 順方向512+文末マスク, 逆方向enc=1024 | greedy=18.28 | - | +0.41pt vs exp003 |

## ファイル構成

- `src/preprocess.py` — ラベルマスキング（truncate_to_sentence_boundary）
- `src/train.py` — 方向別max_length対応、wandb、load_best_model
- `src/infer.py` — greedy + beam4比較
- `src/postprocess.py` — exp004と同一

## 重要な知見

- ラベルマスキングで+0.41ptの改善（17.87→18.28）。仮説は正しかったが効果は限定的
- 530/1404件（37.7%）の順方向ラベルが文末境界でtruncateされた
- training eval(23.72)とinference greedy(18.28)の乖離はexp003(23.55 vs 17.87)と同程度 — truncation水増しはまだ残る（valはマスキングなしのため）
- beam4(12.84)はgreedy(18.28)より大幅に低い — exp003と同傾向
- 後処理で悪化（18.28→17.10） — 後処理の見直しが必要

### Inference CV詳細

| 設定 | chrF | BLEU | geo_mean |
|------|------|------|----------|
| greedy raw | 28.24 | 11.83 | **18.28** |
| greedy post | 27.42 | 10.66 | 17.10 |
| beam4 raw | 24.10 | 6.84 | 12.84 |
| beam4 post | 23.52 | 6.02 | 11.90 |

### Training Eval (epoch別)

| epoch | chrF | BLEU | geo_mean |
|-------|------|------|----------|
| 17 | 35.07 | 15.54 | 23.34 |
| 18 | 35.43 | 15.68 | 23.57 |
| 19 | 34.40 | 14.92 | 22.66 |
| 20 | 35.42 | 15.88 | **23.72** |

### 文レベルCV（テスト条件模擬）

valドキュメントの最初の文（英語ピリオド分割）+ アッカド語先頭200バイトで評価。

| 評価方式 | chrF | BLEU | geo_mean |
|---------|------|------|----------|
| doc-level raw | 28.24 | 11.83 | 18.28 |
| **sent-level raw** | 38.12 | 19.17 | **27.03** |
| **sent-level post** | 38.01 | 19.49 | **27.22** |

- 文レベルCVはドキュメントレベルより+8.75pt高い → テスト条件に近い評価
- 後処理は文レベルではわずかに改善（27.03→27.22）
- 今後は文レベルCVを標準指標とすべき

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp003 | ベースライン（双方向、max_length=512） | greedy=17.87 | - |
| exp004 | max_length拡大(enc1024/dec2048) | greedy=10.68 | -7.19 |
| exp005 | ラベルマスキング + 逆方向enc拡大 | greedy=18.28 | +0.41 |

## コマンド履歴
```bash
python workspace/exp005_label_masking/src/train.py
python workspace/exp005_label_masking/src/infer.py
```

## 次のステップ
- [x] 学習完了後、CVスコアを記録
- [x] inference.pyでgreedy/beam比較
- [ ] 後処理の改善（現在は後処理で悪化）
- [ ] モデルサイズアップ（byt5-base）
- [ ] 追加データ活用（Sentences_Oare, publications）
