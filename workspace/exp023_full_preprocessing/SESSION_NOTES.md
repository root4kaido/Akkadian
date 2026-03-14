# SESSION_NOTES: exp023_full_preprocessing

## セッション情報
- **日付**: 2026-03-12
- **作業フォルダ**: workspace/exp023_full_preprocessing
- **目標**: eda022で洗い出した全前処理改善をexp022に適用し、LB向上を狙う

## 仮説
exp022(LB=30.1)はHost推奨前処理の基本を実装したが、以下が未対応:
- 小数→分数の丸め誤差(213+237件)
- 月名→月番号変換(171行)
- ローマ数字月のregexバグ(34件)
- gap重複(2件)、(ki)→{ki}(test2件)、(m) determinative(5件)
これらを全て修正することでtrain-test alignmentが改善し、スコア向上を期待。

## 親実験: exp022_train_cleaning
- CV: 47.09, LB: 30.1
- 基本的なHost前処理（除去・置換・小数完全一致・ローマ数字基本）

## exp022からの変更点（eda022 report.md全項目）

### HIGH
1. **小数→分数の近似マッチ**: 完全一致→±0.002の最近傍マッチ。0.3332→⅓等
2. **Long float吸収**: 1.666699999999999等もまとめて近似マッチで変換
3. **月名→月番号**: OA月名辞書（Kuzallu→11, Allanātum→12等）

### MED
4. **ローマ数字月のregex修正**: `\bmonth\s+{roman}\b` → `\bmonth\s+{roman}(?=[\s,.:;!?\)]|$)`
5. **テスト(ki)→{ki}**: submit.pyの入力前処理

### LOW
6. **`<gap>` 重複**: `<gap> <gap>` → `<gap>`
7. **(m) determinative**: translation内の(m)→{m}

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| exp023 | eda022全改善 | training=48.50, sent-CV=46.85 | **30.03** | sent-CV=34.35, doc-CV=24.87 |

## ファイル構成
- `src/train.py` — exp022ベース + 全前処理改善
- `src/eval_cv.py` — 同上の前処理適用
- `src/submit.py` — (ki)→{ki}追加

## 重要な知見
- training eval: chrF++=60.21, BLEU=39.06, geo=48.50（exp022比+1.13pt）
- sent-CV(beam4): chrF++=55.21, BLEU=39.75, geo=46.85（exp022比-0.24pt）
- sent-level CV=34.35, doc-level CV=24.87（eval_full_doc.py）
- submit.pyにclean_translation後処理追加済み（月名、近似小数、ローマ数字月、(m)→{m}等）
- submit.pyにTurkish文字正規化追加（ş→ṣ, İ→I, ı→i）
- Kaggleモデルアップロード完了: nomorevotch/AkkadianModels/PyTorch/exp023_full_preprocessing

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| exp022 | 基本Host前処理 | CV=47.09, LB=30.1 | baseline |
| exp023 | 全前処理改善 | training=48.50, sent-CV=46.85, LB=30.03 | LB横ばい(exp022=30.1) |

## コマンド履歴
```bash
# 学習
tmux send-keys -t dev0 'python workspace/exp023_full_preprocessing/src/train.py' Enter
# eval_cv
tmux send-keys -t dev1 'python workspace/exp023_full_preprocessing/src/eval_cv.py' Enter
# eval_full_doc
tmux send-keys -t dev1 'python eda/eda020_sent_level_cv/eval_full_doc.py --exp_dir workspace/exp023_full_preprocessing' Enter
# モデルアップロード
bash tools/upload_model.sh workspace/exp023_full_preprocessing exp023_full_preprocessing
```

## 次のステップ
