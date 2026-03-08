"""
EDA006: 外れ値除去の影響分析
- 除去された49件(train)+4件(val)の中身を確認
- exp005 vs exp006の予測差分を分析（共通153件）
- chrFは改善しているのにBLEUが悪化した原因を特定
"""
import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))


def compute_word_ratio(df):
    akk_words = df["transliteration"].astype(str).str.split().str.len()
    eng_words = df["translation"].astype(str).str.split().str.len()
    return eng_words / akk_words.clip(lower=1)


def compute_sample_metrics(preds, refs):
    """サンプルごとのchrFとBLEUを計算"""
    metric_chrf = evaluate.load("chrf")
    metric_bleu = evaluate.load("sacrebleu")
    chrfs = []
    bleus = []
    for p, r in zip(preds, refs):
        c = metric_chrf.compute(predictions=[p], references=[r])["score"]
        b = metric_bleu.compute(predictions=[p], references=[[r]])["score"]
        chrfs.append(c)
        bleus.append(b)
    return chrfs, bleus


def main():
    # === 1. 除去されたサンプルの確認 ===
    print("=" * 70)
    print("=== 1. 除去されたサンプルの分析 ===")
    print("=" * 70)

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "datasets/raw/train.csv"))
    df = df[
        (df["transliteration"].astype(str).str.len() > 0)
        & (df["translation"].astype(str).str.len() > 0)
    ]

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    ratio_train = compute_word_ratio(train_df)
    ratio_val = compute_word_ratio(val_df)

    removed_train = train_df[(ratio_train < 0.3) | (ratio_train > 5.0)]
    removed_val = val_df[(ratio_val < 0.3) | (ratio_val > 5.0)]

    print(f"\n除去: train={len(removed_train)}件, val={len(removed_val)}件")

    # カテゴリ分類
    def categorize(row):
        trans = str(row["translation"])
        translit = str(row["transliteration"])
        gap_count = translit.count("<gap>") + translit.count("<big_gap>")
        if trans.lower().startswith("seal"):
            return "Seal文書"
        elif gap_count >= 3:
            return "gap大量"
        elif re.match(r'^[\d\s.]+', trans) and len(trans.split()) <= 5:
            return "数値のみ"
        elif len(trans.split()) <= 3:
            return "極短翻訳"
        else:
            return "その他"

    print("\n--- 除去trainサンプルのカテゴリ分布 ---")
    categories = removed_train.apply(categorize, axis=1)
    for cat, count in categories.value_counts().items():
        print(f"  {cat}: {count}件")

    print("\n--- 除去trainサンプル詳細 (先頭20件) ---")
    for i, (idx, row) in enumerate(removed_train.head(20).iterrows()):
        r = ratio_train.loc[idx]
        cat = categorize(row)
        akk = str(row["transliteration"])[:80]
        eng = str(row["translation"])[:80]
        print(f"\n[{i}] ratio={r:.3f} cat={cat}")
        print(f"  akk: {akk}")
        print(f"  eng: {eng}")

    print("\n--- 除去valサンプル詳細 (4件全部) ---")
    for i, (idx, row) in enumerate(removed_val.iterrows()):
        r = ratio_val.loc[idx]
        cat = categorize(row)
        akk = str(row["transliteration"])[:80]
        eng = str(row["translation"])[:80]
        print(f"\n[{i}] ratio={r:.3f} cat={cat}")
        print(f"  akk: {akk}")
        print(f"  eng: {eng}")

    # === 2. 予測差分分析（共通153件） ===
    print("\n" + "=" * 70)
    print("=== 2. exp005 vs exp006 予測差分分析 (共通153件) ===")
    print("=" * 70)

    exp005_csv = os.path.join(PROJECT_ROOT, "workspace/exp005_label_masking/results/val_predictions_sentence.csv")
    exp006_csv = os.path.join(PROJECT_ROOT, "workspace/exp006_data_filtering/results/val_predictions_sentence.csv")

    df5 = pd.read_csv(exp005_csv)
    df6 = pd.read_csv(exp006_csv)

    # 共通サンプルをreferenceでマッチ
    common_refs = set(df5["reference"]) & set(df6["reference"])
    df5_common = df5[df5["reference"].isin(common_refs)].sort_values("reference").reset_index(drop=True)
    df6_common = df6[df6["reference"].isin(common_refs)].sort_values("reference").reset_index(drop=True)
    print(f"共通サンプル: {len(df5_common)}件")

    # サンプルごとのメトリクス計算
    print("\nサンプルごとのメトリクス計算中...")
    chrf5, bleu5 = compute_sample_metrics(df5_common["prediction_raw"].tolist(), df5_common["reference"].tolist())
    chrf6, bleu6 = compute_sample_metrics(df6_common["prediction_raw"].tolist(), df6_common["reference"].tolist())

    df_compare = pd.DataFrame({
        "reference": df5_common["reference"],
        "pred5": df5_common["prediction_raw"],
        "pred6": df6_common["prediction_raw"],
        "chrf5": chrf5,
        "chrf6": chrf6,
        "bleu5": bleu5,
        "bleu6": bleu6,
        "chrf_diff": [c6 - c5 for c5, c6 in zip(chrf5, chrf6)],
        "bleu_diff": [b6 - b5 for b5, b6 in zip(bleu5, bleu6)],
    })

    # 全体統計
    print(f"\n--- 全体統計 ---")
    print(f"chrF: exp005 mean={np.mean(chrf5):.2f}, exp006 mean={np.mean(chrf6):.2f}, diff={np.mean(chrf6)-np.mean(chrf5):.2f}")
    print(f"BLEU: exp005 mean={np.mean(bleu5):.2f}, exp006 mean={np.mean(bleu6):.2f}, diff={np.mean(bleu6)-np.mean(bleu5):.2f}")

    # chrF改善 & BLEU悪化のサンプル
    paradox = df_compare[(df_compare["chrf_diff"] > 0) & (df_compare["bleu_diff"] < 0)]
    print(f"\nchrF改善 & BLEU悪化: {len(paradox)}件 / {len(df_compare)}件")

    # BLEU悪化Top10
    print("\n--- BLEU悪化Top10 ---")
    worst_bleu = df_compare.nsmallest(10, "bleu_diff")
    for _, row in worst_bleu.iterrows():
        print(f"\n  bleu: {row['bleu5']:.1f} → {row['bleu6']:.1f} ({row['bleu_diff']:+.1f})  chrf: {row['chrf5']:.1f} → {row['chrf6']:.1f} ({row['chrf_diff']:+.1f})")
        print(f"  ref:  {str(row['reference'])[:100]}")
        print(f"  p005: {str(row['pred5'])[:100]}")
        print(f"  p006: {str(row['pred6'])[:100]}")

    # BLEU改善Top10
    print("\n--- BLEU改善Top10 ---")
    best_bleu = df_compare.nlargest(10, "bleu_diff")
    for _, row in best_bleu.iterrows():
        print(f"\n  bleu: {row['bleu5']:.1f} → {row['bleu6']:.1f} ({row['bleu_diff']:+.1f})  chrf: {row['chrf5']:.1f} → {row['chrf6']:.1f} ({row['chrf_diff']:+.1f})")
        print(f"  ref:  {str(row['reference'])[:100]}")
        print(f"  p005: {str(row['pred5'])[:100]}")
        print(f"  p006: {str(row['pred6'])[:100]}")

    # === 3. 繰り返し問題の変化 ===
    print("\n" + "=" * 70)
    print("=== 3. 繰り返しパターンの変化 ===")
    print("=" * 70)

    def count_repeats(text):
        """繰り返しパターンの数をカウント"""
        matches = re.findall(r'\b(\w+)(?:\s+\1\b)+', str(text))
        return len(matches)

    repeats5 = [count_repeats(p) for p in df5_common["prediction_raw"]]
    repeats6 = [count_repeats(p) for p in df6_common["prediction_raw"]]
    print(f"繰り返し含むサンプル: exp005={sum(1 for r in repeats5 if r > 0)}件, exp006={sum(1 for r in repeats6 if r > 0)}件")
    print(f"繰り返し総数: exp005={sum(repeats5)}, exp006={sum(repeats6)}")

    # 予測長の変化
    lens5 = [len(str(p)) for p in df5_common["prediction_raw"]]
    lens6 = [len(str(p)) for p in df6_common["prediction_raw"]]
    ref_lens = [len(str(r)) for r in df5_common["reference"]]
    print(f"\n予測長: exp005 mean={np.mean(lens5):.0f}, exp006 mean={np.mean(lens6):.0f}")
    print(f"参照長: mean={np.mean(ref_lens):.0f}")

    print("\n完了")


if __name__ == "__main__":
    main()
