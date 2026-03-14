# SESSION_NOTES: exp029_label_smoothing

## セッション情報
- **日付**: 2026-03-13
- **作業フォルダ**: workspace/exp029_label_smoothing
- **目標**: label smoothingで汎化力向上、E層(geo 10-30)の底上げ

## 仮説
- eda024分析でE層(geo 10-30, 840件36.3%)が最大の改善ターゲット
- 未見AKTグループへの汎化力不足が本質的な問題
- label_smoothing_factor=0.1でsoft targetにすることで、特定パターンへの過信を抑え汎化を促進
- Transformer原論文でもlabel_smoothing=0.1を使用しておりNMTでは定番

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| label_smoothing=0.1 | Seq2SeqTrainingArgsに追加 | sent-geo=34.88, doc-geo=23.84 | - | **棄却**。rep=70.0% |

## ファイル構成
- `src/train_gkf.py` — exp023 train_gkf.py + label_smoothing_factor=0.1

## 重要な知見
- label smoothingはByT5のseq2seq生成でrepetitionを誘発する（rep 0%→70%）
- soft targetにより停止トークンの確信度が下がり、生成が止まらなくなる
- NMT定番テクニックだがByT5(byte-level)では逆効果

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| - | - | - | - |

## コマンド履歴
```bash
# 再現性のための記録
```

## 次のステップ
