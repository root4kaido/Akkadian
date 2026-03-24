# SESSION_NOTES: exp042_bt_mixed_v2

## セッション情報
- **日付**: 2026-03-20
- **作業フォルダ**: workspace/exp042_bt_mixed_v2
- **目標**: generate_v2 BT(23k) + real train 混合学習 10ep（exp038方式の拡大版）

## 仮説
- exp038ではBT 2k件混合でsent+1.78, doc+1.12と大幅改善
- BT 23k件に拡大することでさらなる改善を期待
- ただしreal:BT比が約1:16と偏るため、過学習パターンが変わる可能性

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| BT 23k + real混合 10ep | generate_v2 BT拡大 | sent-geo=39.89, doc-geo=28.52 | 30.9 | CV同等だがLBはexp041(33.4)に大きく劣る。混合学習は汎化しにくい |

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp023 (baseline) | - | sent-geo=35.44, doc-geo=25.52 | - |
| exp038 (BT 2k混合) | generated_english BT | sent-geo=37.22, doc-geo=26.64 | sent+1.78, doc+1.12 |
| **exp042 (BT 23k混合)** | generate_v2 BT拡大 10ep | sent-geo=39.89, doc-geo=28.52 | sent+4.45, doc+3.00 |

## コマンド履歴
```bash
# BTデータはexp041で生成済み（symlink）

# 学習: BT+real混合 10ep (dev1)
python workspace/exp042_bt_mixed_v2/src/train.py --fold 3 2>&1 | tee workspace/exp042_bt_mixed_v2/results/fold3/train.log
```

## 次のステップ
