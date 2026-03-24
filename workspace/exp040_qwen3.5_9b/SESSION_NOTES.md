# SESSION_NOTES: exp040_qwen3.5_9b

## セッション情報
- **日付**: 2026-03-19
- **作業フォルダ**: workspace/exp040_qwen3.5_9b
- **目標**: Qwen3.5-9Bでアッカド語→英語翻訳のゼロショット性能評価

## 仮説
- generated_english.csv(Qwen生成)の品質が高く、BT経由で有効だった
- Qwen3.5-9Bに直接翻訳させればbyt5-base fine-tuned(sent-geo 37.22)を超える可能性
- int4量子化でKaggle T4×2(13GB)にも載る
- ゼロショットで性能確認後、LoRA fine-tuneで更に改善を目指す

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| ゼロショット | Qwen3.5-9B, enable_thinking=False | sent-geo=10.63, doc-geo=8.26 | - | rep=1.0%/20.5%。byt5比大幅に低い |

## ファイル構成
- `src/eval_zeroshot.py` — ゼロショット推論+CV評価

## 重要な知見

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| - | - | - | - |

## コマンド履歴
```bash
# ゼロショット評価
# tmux send-keys -t dev0 'python workspace/exp040_qwen3.5_9b/src/eval_zeroshot.py --fold 3' Enter
```

## 次のステップ
- ゼロショット結果次第でLoRA fine-tune
- few-shotプロンプトの検討
