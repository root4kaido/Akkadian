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
| - | - | - | - | - |

## ファイル構成

- `src/preprocess.py` — split後に外れ値除去
- `src/train.py` — exp005ベース（logging force=True修正）
- `src/infer.py` — exp005ベース
- `src/postprocess.py` — repeated_removalのみに最適化
- `src/eval_sentence_level.py` — 文レベルCV評価

## 重要な知見

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp005 | ベースライン | sent=27.03 | - |
| exp006 | 外れ値除去 + 後処理最適化 | - | - |

## コマンド履歴
```bash
python workspace/exp006_data_filtering/src/train.py
python workspace/exp006_data_filtering/src/infer.py
python workspace/exp006_data_filtering/src/eval_sentence_level.py
```

## 次のステップ
- [ ] 学習完了後、CVスコアを記録
- [ ] inference + 文レベルCV評価
