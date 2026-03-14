# SESSION_NOTES: exp022_train_cleaning

## セッション情報
- **日付**: 2026-03-12
- **作業フォルダ**: workspace/exp022_train_cleaning
- **目標**: Host推奨のtrain前処理をexp016に適用してLB改善を狙う

## 仮説
- exp016はtrain.csvをそのまま使用（前処理なし）でLB=29.5
- Hostがディスカッション678899で明示した前処理をtrainのtranslationに適用
- テストラベルのフォーマットに合わせることで、フォーマット不一致によるスコアロスを解消
- CVは変化しないかもしれないが、LBは改善するはず

## Host推奨前処理（678899 "A Stitch in Time"）
### 翻訳から除去
- `fem.`, `sing.`, `pl.`, `plural`, `(?)`
- stray marks: `..`, 孤立`x`/`xx`, `<< >>`, `< >`（`<gap>`は残す）
- `/` による代替翻訳 → 一方を選択

### 翻訳で置換
- `PN` → `<gap>`
- `-gold` → `pašallum gold`
- `-tax` → `šadduātum tax`
- `-textiles` → `kutānum textiles`

### 小数→分数変換
- 0.5→½, 0.25→¼, 0.3333→⅓, 0.6666→⅔, 0.8333→⅚, 0.75→¾, 0.1666→⅙, 0.625→⅝

### ローマ数字→整数
- month V → month 5 等

### Transliteration（Optional）
- Ḫ→H, ḫ→h
- Unicode下付き数字→通常整数
- 小数→分数

## exp016からの変更点
| パラメータ | exp016 | exp022 |
|---|---|---|
| train前処理 | なし | **Host推奨前処理（translation + transliteration）** |
| その他 | 全てexp016と同一 | 同一 |

## 試したアプローチと結果

| アプローチ | 変更点 | sent-CV(geo) | doc-CV(geo) | LB | 備考 |
|-----------|--------|-------------|-------------|-----|------|
| Host前処理 | translation+transliteration前処理 | **47.09** | - | **30.1** | exp016(45.78/29.5)比CV+1.31pt,LB+0.6pt。rep=28.0% |

## ファイル構成
- src/train.py — 学習スクリプト（前処理関数追加）
- src/eval_cv.py — CV評価

## コマンド履歴
```bash
# dev0で学習実行（約80分）
python workspace/exp022_train_cleaning/src/train.py
# training eval: geo_mean=49.01 (exp016: 46.73, +2.28pt)

# CV評価
python workspace/exp022_train_cleaning/src/eval_cv.py
# beam4 sent clean: chrF++=55.22, BLEU=40.15, geo=47.09, rep=28.0%
```

## 課題
- **月名→月番号が未対応**: Kuzallu→11, Allanātum→12, Hubur→6 等がtranslationに多数残存（171行に`month`出現）
- **ローマ数字月の正規表現に漏れ**: `month II,eponymy` 等カンマ隣接パターン（34件残存）
- **小数→分数の丸め誤差吸収**: 0.3332(123件), 0.6665(51件)等。eda022のカウントにバグの可能性
- testにもtrain外のパターンが来る可能性あり

## 次のステップ
- 月名→月番号の変換マッピング追加
- ローマ数字月の正規表現修正（カンマ・コロン隣接対応）
- 小数→分数を近似マッチに改善
- 上記修正して再学習
