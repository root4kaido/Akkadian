---
name: survey-notebooks
description: Kaggleのノートブック（Code）を調査する。コンペのNotebook一覧をスクレイピングし、人気ノートブックの内容・手法を分析する。解法リサーチやアイデア探索に使う。
---

# ノートブック調査スキル

## 手順

1. **コンペURLの確認**: `CLAUDE.md` からコンペURLを取得し、Code（Notebook）ページのURLを構成
   - 例: `https://www.kaggle.com/competitions/{comp-slug}/code`

2. **環境確認**:
   - Playwright がインストールされているか確認（一覧取得に使用）
   - Kaggle API (`kaggle`) がインストール・認証済みか確認（詳細取得に使用）
   - なければ tmux 上で: `pip install playwright beautifulsoup4 lxml kaggle && playwright install chromium`

3. **スクレイピングスクリプトの確認/作成**:
   - `docs/survey/notebooks/` にスクレイピングスクリプトがあるか確認
   - なければ `scrape_notebooks.py` と `scrape_notebook_details.py` を作成

4. **データ取得**（tmux上で実行）:
   ```bash
   cd docs/survey/notebooks
   python scrape_notebooks.py
   python scrape_notebook_details.py
   ```

5. **ノートブック一覧スクレイピング** (`scrape_notebooks.py`):
   - コンペのCodeページを開く（ソート: Most Votes / Hotness）
   - スクロールして一覧を読み込む
   - 各ノートブックの情報を取得:
     - タイトル、URL、著者
     - 投票数（upvotes）
     - コメント数
     - 最終更新日
   - `snapshot_YYYYMMDD.json` に保存

6. **ノートブック詳細取得** (`scrape_notebook_details.py`):
   - `kaggle kernels pull` で投票数上位のノートブックを `.ipynb` としてダウンロード
   - `docs/survey/notebooks/downloaded/` にローカル保存
   - ipynbをパースしてコードセル・マークダウンセルを抽出
   - 自動解析する情報:
     - 使用ライブラリ（import文から抽出）
     - 使用モデル（LightGBM、pytorch等）
     - 前処理・特徴量エンジニアリングの手法
     - 独自のアイデア・テクニック
     - コード全文のサマリー（先頭15KB）
   - `snapshot_YYYYMMDD_details.json` に保存

7. **差分分析**（過去のスナップショットが存在する場合）:
   - 新規ノートブックの特定
   - 投票数の増加（注目度上昇）
   - 新しい手法・アプローチの出現

8. **結果の整理**:
   - `docs/survey/notebooks/snapshot_YYYYMMDD.json` にスナップショット保存
   - `docs/survey/notebooks/notebook_summary.md` に分析レポートを更新
     - 上位ノートブックの手法一覧
     - 使用されている特徴量・モデルの傾向
     - 独自のアイデア・テクニック
   - 重要な発見があれば `eda/eda_summary.md` の知見セクションにも追記

9. **レポート**: 注目ノートブックのサマリーを報告
   - 手法別の分類（特徴量系、モデル系、アンサンブル系等）
   - 我々の実験に取り入れるべきアイデアの提案（堅実案＋爆発案）
