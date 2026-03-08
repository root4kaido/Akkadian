# SESSION_NOTES: exp006_data_filtering

## セッション情報
- **日付**: 2026-03-08
- **作業フォルダ**: workspace/exp006_data_filtering
- **目標**: 外れ値データ除去 + 後処理最適化で品質向上

## 仮説

1. **外れ値除去**: eda004で特定したword_ratio外れ値53件(3.4%)は、翻訳の打ち切り・結合等でノイジー。除去により学習品質が向上する
2. **後処理最適化**: eda005のablationで、repeated_removalのみがsent-levelで+0.70pt有効。forbidden_charsとfraction_conversionは有害

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| exp006 | 外れ値除去(49train+4val) + 後処理最適化 | doc=18.61, sent_raw=25.17, sent_post=26.34 | - | doc微増、sent悪化 |

## ファイル構成

- `src/preprocess.py` — split後に外れ値除去
- `src/train.py` — exp005ベース（logging force=True修正）
- `src/infer.py` — exp005ベース
- `src/postprocess.py` — repeated_removalのみに最適化
- `src/eval_sentence_level.py` — 文レベルCV評価

## 重要な知見

- training eval geo_mean=23.64（exp005: 23.72とほぼ同等）
- doc-level raw: 18.61（exp005: 18.28 → +0.33pt改善）
- **sent-level raw: 25.17（exp005: 27.03 → -1.86pt悪化）**
- sent-level post: 26.34（exp005: 27.22 → -0.88pt悪化）
- 同一153件での公平比較でもBLEU -2.90pt悪化（chrFは+0.48改善）
- **予測長が183→216文字に増加**（参照169文字）→ BLEUのprecisionペナルティが主因
- 外れ値データはexp005のラベルマスキング＋双方向学習で正しい部分のみloss計算されるため有用
- 小規模データ(1404件)での3.5%削減は無視できない影響
- 後処理最適化（repeated_removalのみ）: sent_raw→sent_postで+1.17pt改善
- **結論: 外れ値除去は棄却。exp005ベースで次に進む**

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp005 | ベースライン | doc=18.28, sent_raw=27.03, sent_post=27.22 | - |
| exp006 | 外れ値除去 + 後処理最適化 | doc=18.61, sent_raw=25.17, sent_post=26.34 | doc+0.33, sent-1.86 |

## コマンド履歴
```bash
python workspace/exp006_data_filtering/src/train.py
python workspace/exp006_data_filtering/src/infer.py
python workspace/exp006_data_filtering/src/eval_sentence_level.py
```

## 次のステップ
- [x] 学習完了後、CVスコアを記録
- [x] inference + 文レベルCV評価
- [x] 結果分析: 予測長増加(+33文字)によるBLEU precision低下が主因
- [x] 公平比較: 同一153件でもBLEU -2.90pt（eda006で分析）
