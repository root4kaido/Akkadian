# SESSION_NOTES: exp043_bolmo_1b

## セッション情報
- **日付**: 2026-03-21
- **作業フォルダ**: workspace/exp043_bolmo_1b
- **目標**: Bolmo-1B (byte-level decoder-only LM) がアッカド語翻訳で使えるか検証
- **結論**: **不採用** — ByT5-baseに大幅に劣る

## 仮説
- ByT5と同じbyte-levelだが、decoder-onlyの1Bパラメータモデル
- 現代のLLMはdecoder-onlyでも翻訳タスクで高い性能を出す
- 1Bパラメータはアッカド語のような低リソース言語には十分か？
- ByT5-base (580M, enc-dec) と比較してどうか

## 試したアプローチと結果

| アプローチ | 変更点 | sent-CV geo | doc-CV chrF++ | 備考 |
|-----------|--------|-------------|---------------|------|
| fold3 baseline | CausalLM fine-tune, real data only, 10ep | 4.27 | 3.14 | 壊滅的 |

## ファイル構成
- `src/train.py` — CausalLM fine-tuning script (custom BolmoTrainer)
- `src/eval_cv.py` — sent-CV / doc-CV 評価スクリプト (manual_generate)
- `src/diagnose.py` — モデル出力診断スクリプト
- `results/fold3/` — 評価結果

## 重要な知見

### Bolmoの技術的ハマりどころ
1. **BOS二重付与問題**: byte-levelトークナイザが毎回BOSを自動付与。promptとtargetを別々にトークナイズして連結すると、境界にBOSが重複 → labelsの最初の有効トークンがEOS(=BOS, id=1)になり、モデルが即座にEOS出力を学習。修正: targetのトークナイズに `add_special_tokens=False`
2. **xlstmは系列長が64の倍数必須**: CausalLMDataCollatorでパディング必要
3. **BolmoForCausalLM.forward()はlabelsを受け付けない**: custom Trainerでloss計算が必要
4. **Bolmoのgenerate()は非標準API**: manual_generateで回避

### なぜByT5に劣るか
- **Decoder-onlyはbyte-level翻訳に不利**: byte単位だと1文=数百トークン。encoder-decoderのcross-attentionなしではソース全体の保持が困難
- **xlstm (mLSTM) の長距離依存の限界**: Transformerのattentionと比べ、長いbyte列から意味を抽出する力が弱い
- **1Bパラメータでも不足**: GPT-4クラスなら翻訳可能だが、1Bのdecoder-onlyでは低リソース言語翻訳は荷が重い

## 性能変化の記録

| 実験 | sent-CV chrF++ | sent-CV BLEU | sent-CV geo | doc-CV chrF++ | doc-CV geo |
|------|---------------|-------------|-------------|---------------|------------|
| exp043 Bolmo-1B | 12.20 | 1.50 | 4.27 | 3.14 | 0.00 |
| (参考) exp023 ByT5-base | ~45 | ~25 | ~33 | ~30 | ~25 |

## コマンド履歴
```bash
# 依存関係インストール
pip install xlstm==2.0.4

# fold3 学習 (A100)
rm -rf workspace/exp043_bolmo_1b/results/fold3
python workspace/exp043_bolmo_1b/src/train.py --fold 3

# 評価 (A100)
python workspace/exp043_bolmo_1b/src/eval_cv.py workspace/exp043_bolmo_1b/results/fold3/last_model exp043_bolmo_1b_last --fold 3
```

## 次のステップ
- この実験はクローズ。ByT5-baseベースのアプローチを継続すべき
