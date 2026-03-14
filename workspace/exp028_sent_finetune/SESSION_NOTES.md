# SESSION_NOTES: exp028_sent_finetune

## セッション情報
- **日付**: 2026-03-13
- **作業フォルダ**: workspace/exp028_sent_finetune
- **目標**: exp023 fold3 last_modelからsentence-level fine-tuneでE層(geo 10-30, 840件36.3%)を底上げ

## 仮説
- eda024エラー分析で、E層(geo 10-30)が840件(36.3%)と最大ボリューム
- 短い入力(0-100 bytes)でgeo=29.45と低い → 短文での翻訳精度が課題
- document-levelで学習したモデルをsentence-levelでfine-tuneすることで、短セグメントの翻訳精度が向上する
- sentence_aligned.csv (2,311行) をfold3 val oare_id除外して使用

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| Stage2 sent fine-tune | exp023 fold3 last → sent_aligned全2311行, lr=5e-5, 5ep | sent-geo=35.40, doc-geo=19.37 | - | **棄却**。rep=56.2% |

## ファイル構成
- `src/train_stage2.py` — Stage2 fine-tuning script
- `results/fold3/` — 学習結果

## 重要な知見
- sentence-level fine-tuneはdoc-level生成能力を破壊する（catastrophic forgetting）
- sent-geoすら改善しない（-0.04pt）→ 同じデータの形を変えても新しい情報がない
- 2段階学習アプローチは本コンペでは効果なし

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| - | - | - | - |

## コマンド履歴
```bash
# 再現性のための記録
```

## 次のステップ
