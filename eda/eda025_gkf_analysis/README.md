# eda025: GroupKFold分析

## 目的
ランダム分割CVとLBの乖離原因を調査。AKTグループ（出版物別）でGroupKFoldを実施し、CV低下の原因を特定する。

## AKTグループ
train.csvの全1561 docsはpublished_texts.csvのaliasesから7つのAKTグループに分類される。

| グループ | 件数 | 備考 |
|---------|------|------|
| AKT 8 | 363 | Veenhof (2017) |
| AKT 6a | 300 | Larsen (2010) |
| AKT 6e | 251 | Larsen (2021) |
| AKT 6b | 220 | Larsen (2013) |
| AKT 6c | 200 | Larsen (2013) |
| AKT 6d | 135 | Larsen (2014) |
| AKT 5 | 77 | - |
| None | 15 | マッピング不可 |

## GroupKFold CV結果（exp023, fold0, last_model）

| split | sent-CV (geo) | doc-CV (geo) |
|-------|-------------|-------------|
| ランダム | 36.34 | 25.84 |
| GKF fold0 (val=AKT 8) | 25.06 | 22.74 |

sent-CVが11pt低下。ランダム分割にリークあり。

## 固有名詞分析（analyze_pn_gkf.py）

**仮説**: GKF CVの大幅低下は固有名詞（PN）の未見によるもの
**結論**: **棄却**。固有名詞はCV低下の主因ではない。

### 根拠
- AKT 8の固有名詞840種のうち457種（54.4%）がtrain未出現
- Train既知PNの予測一致率: 54.5%、未見PNの一致率: 17.1% → 未見PNは確かに予測困難
- しかし**スコアへの影響はない**:
  - 未見PN含む文: geo=25.17
  - 全PN既知 or PNなし: geo=24.91
  - 未見PNを含む文の方がむしろスコアが高い
- 固有名詞を含む文は定型的（「Witnessed by X, by Y」等）で、名前以外の部分で稼いでいるため

## AKTグループ間の文書特性分析（analyze_akt_differences.py）

**結論**: AKTグループ間の違いは固有名詞ではなく、**文書構造・構文パターン・語彙分布**の根本的な違い。

### 文書長の違い
| グループ | 翻訳word数 (mean) | 翻字bytes (mean) |
|---------|-------------------|-------------------|
| AKT 5 | 149.6 | 1093 |
| AKT 8 | 65.1 | 560 |
| AKT 6a | 52.3 | 440 |
| AKT 6b | 47.3 | 421 |
| AKT 6c | 40.3 | 363 |
| AKT 6d | 41.5 | 375 |
| AKT 6e | 30.9 | 289 |

AKT 5が圧倒的に長く、AKT 6eが最短。5倍近い差がある。

### 構文パターンの違い
- **手紙型** (AKT 6a-6d): "From X to Y:" 形式が31-44%出現
- **証人型** (AKT 5, 8): "Say to X, thus Y" 20-22%、witness/seal表現が多い
- **会計型** (AKT 6e): textile/caravan/tin等の商品語彙が特徴的、短文

### 語彙の類似度（bigram Jaccard）
- グループ間の類似度は**0.067〜0.152**と非常に低い
- 最も遠い組: AKT 8 ↔ AKT 6e (0.067)
- 最も近い組: AKT 6c ↔ AKT 6d (0.152)
- 同じ出版物シリーズ（AKT 6系）でも類似度は低い

### GKFで過学習する理由
- GKFのval setは完全に未見のドメイン（文書タイプ・構文が異なる）
- Lossは全トークンの確率を見るため、ドメイン固有の語彙・構文パターンへの過適合が顕著に現れる
- 一方、eval metric (GEO)はn-gram一致ベースで、定型表現の骨格が合っていればスコアが出る
- → Loss overfitting + eval metric improvement の乖離が発生

## ファイル構成
- `analyze_pn_gkf.py` — 固有名詞分析スクリプト
- `pn_analysis_result.txt` — 分析結果出力
- `analyze_akt_differences.py` — AKTグループ間の文書特性比較
- `akt_differences_result.txt` — 分析結果出力
- `check_gkf_folds.py` — 5fold分割の確認スクリプト
