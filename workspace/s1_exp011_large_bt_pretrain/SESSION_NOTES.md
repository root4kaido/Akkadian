# SESSION_NOTES: s1_exp011_large_bt_pretrain

## セッション情報
- **日付**: 2026-03-22
- **作業フォルダ**: workspace/s1_exp011_large_bt_pretrain
- **目標**: byt5-large 2段階学習（大量pseudo data pretrain → real finetune）

## 仮説
- s1_exp008はs1_exp007のlast_modelからpretrainしていた（意図せず3段階学習になっていた）
- HF pretrained weightsから直接2段階学習することで、正しいベースラインを確立
- pseudo_labels_v2 (6K) + backtranslated_v3 (62K) の大量データでpretrainすることでドメイン適応効果を最大化

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| - | - | - | - | - |

## ファイル構成
- `src/pretrain.py` — Stage1: pseudo data pretrain (lr=5e-5, 5ep)
- `src/finetune.py` — Stage2: real data finetune (lr=1e-5, 5ep)

## 重要な知見
- s1_exp008は実質3段階（HF pretrained → s1_exp007 real 20ep → pseudo 5ep → real 5ep）だった

## コマンド履歴
```bash
# Stage1: pseudo pretrain
python workspace/s1_exp011_large_bt_pretrain/src/pretrain.py --fold 3

# Stage2: real finetune
python workspace/s1_exp011_large_bt_pretrain/src/finetune.py --fold 3
```

## 次のステップ
- CV/LB結果次第でs1_exp008との比較
- 効果があればxl版への展開
