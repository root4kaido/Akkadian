# SESSION_NOTES: s1_exp010_xl_lr5e5

## セッション情報
- **日付**: 2026-03-21
- **作業フォルダ**: workspace/s1_exp010_xl_lr5e5
- **目標**: byt5-xl(3.7B)でスケーリング効果を検証

## 仮説
- base→largeでLB +1.9pt改善（exp034 31.7→s1_exp007 31.9、s1_exp008 32.5）
- xl(3.7B)でさらにスケーリング効果が得られるか
- lr=5e-5（large比1/2、base→largeと同じスケーリング則）で過学習を抑制

## 親実験との差分
- モデル: google/byt5-large → google/byt5-xl
- lr: 1e-4 → 5e-5
- batch_size: 4 → 2（VRAM対策）
- gradient_accumulation: 2 → 4（effective batch=8を維持）
- gradient_checkpointing: 有効化（VRAM対策）

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| byt5-xl last | lr=5e-5, ep20, grad_ckpt | sent-geo=37.34, doc-geo=25.63 | - | largeより悪化（sent -0.67, doc -0.42） |

## ファイル構成
- `src/train_gkf.py` — 学習スクリプト（s1_exp007ベース）

## 重要な知見
<!-- セッション中の発見、避けるべきアプローチ、有効だったテクニック -->

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| s1_exp007 (large) | lr=1e-4, ep20 | sent-CV=38.01, LB=31.9 | ベースライン |
| s1_exp008 (large+st) | +pseudo pretrain | LB=32.5 | +0.6pt |
| s1_exp010 (xl) | byt5-xl, lr=5e-5 | - | - |

## コマンド履歴
```bash
# fold3で実行
python workspace/s1_exp010_xl_lr5e5/src/train_gkf.py --fold 3
```

## 次のステップ
- xlでスケーリング効果があれば → self-training 2段階(s1_exp008方式)との組合せ
- 効果なし/過学習ならlrをさらに下げる(3e-5)、またはLoRA化を検討
- xxlへの展開判断
