---
name: experiment
description: 命名規則に従って新しい実験フォルダを作成し、SESSION_NOTES.mdとconfig.yamlのテンプレートを自動生成する。実験を始めるとき、新しいexpフォルダが必要なときに使う。
argument-hint: "[実験名]"
---

# 新規実験フォルダ作成

## 命名規則

- `workspace/exp{3桁}_{実験名}` 例: `workspace/exp001_baseline`
- `workspace/` 内の既存フォルダを確認し、次の番号を自動決定する

## 手順

1. `workspace/` 内の既存フォルダを確認し、次の連番を決定
2. `$ARGUMENTS` から実験名を取得
3. フォルダを作成し、以下のサブディレクトリも作る:
   - `src/` — 実験コード
   - `results/` — 出力（学習ログ、学習曲線、CVスコア、submission.csv）
   - `dataset/` — 実験固有データ（特徴量キャッシュ等）
4. `config.yaml` を以下のテンプレートで作成:

```yaml
experiment:
  name: expXXX_name
  description: "実験の目的を簡潔に"

lineage:
  parent: expYYY       # 親実験（初回はnull）
  status: active
  diff_summary: "親実験からの変更点を記述"

# model:
#   name:
#   params: {}
#
# training:
#   seed: 42
#   n_folds: 5
#   epochs: 10
#   batch_size: 32
#   lr: 1e-3
#   amp: true
#   resume_from: null
```

5. `SESSION_NOTES.md` を以下のテンプレートで作成:

```markdown
# SESSION_NOTES: {フォルダ名}

## セッション情報
- **日付**: {今日の日付}
- **作業フォルダ**: {フォルダパス}
- **目標**: {実験名から推定 or 空欄}

## 仮説
<!-- 何を良くしたいのか -->

## 試したアプローチと結果

| アプローチ | 変更点 | CV | LB | 備考 |
|-----------|--------|-----|-----|------|
| - | - | - | - | - |

## ファイル構成
<!-- 作成したスクリプト、可視化結果、データファイル -->

## 重要な知見
<!-- セッション中の発見、避けるべきアプローチ、有効だったテクニック -->

## 性能変化の記録

| 実験 | 変更内容 | 結果 | 改善幅 |
|------|---------|------|--------|
| - | - | - | - |

## コマンド履歴
\```bash
# 再現性のための記録
\```

## 次のステップ
<!-- 結果を受けて、次の実験の候補や残課題など -->
```

6. `workspace/experiment_summary.md` の実験一覧表と系譜図に新実験を追加
7. 作成したフォルダのパスを報告する

## ハイパラ初期検証（コンペ序盤で必須）

**改善の積み上げはベースの学習設定が妥当である前提で成り立つ。土台がずれていると全部やり直しになる。**

- 公開ベースライン/Discussionがある場合: **ハイパラを正確に読み取ってから実験開始**（バージョン違いに注意）
- 公開ベースラインがない場合: **exp001-002の段階でlr sweepを実施**
  - lr候補: {1e-5, 5e-5, 1e-4, 5e-4, 1e-3} の5点
  - 2-3 epochだけ回してloss curveの形を確認し「学習が効いている範囲」を特定
  - best lrが決まってからLS, scheduler, batch size等を検討
- **モデルサイズを変えたらlrを再検証する**（small→baseで同じlrが最適とは限らない）

## 学習コードの鉄則

- **AMP (Mixed Precision) は常にON**
- **チェックポイント再開は必須**（PCが途中で止まることがある）
- **シード固定**（全実験で同じseedを使い、validation splitを統一する。splitが変わるとスコア比較ができない）
- ハイパーパラメータはすべてconfig.yamlで管理（ハードコーディング禁止）
- 学習ログ（loss, metric, lr）を記録し、`results/` に保存してください。pytorchの場合は、pytorch lightning(import lightning as L以降のバージョン)と、wandbを積極的に使ってください。ただし、言語モデルなどでtransformersのTrainerの方が適していると判断される場合、そちらを使ってください。。
- **validationの予測結果を `results/` に保存する**（後のアンサンブルや分析に必要）。ただし保存サイズが巨大になる場合（目安: 100GB超）は省略可
- **print文ではなくloggingを使い、ログをファイルに残す**（`results/` にログファイルを出力。後から全出力を参照できるようにする）

## Fold設計

**安易にランダムKFoldにしない。データの性質を先に確認する。**

- 時系列 → TimeSeriesSplit
- グループ構造あり → GroupKFold
- クラス不均衡 → StratifiedKFold
- グループ+不均衡 → StratifiedGroupKFold
- fold設計の理由と各foldの分布はSESSION_NOTES.mdに記録する
- CV/LBの相関を確認し、相関が弱ければfold設計を見直す

## その他の注意事項

- 実験フォルダ内で自己完結させ、外部に中間ファイルを散らかさないこと
