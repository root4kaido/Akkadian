"""
exp005 vs exp006: validation setの差分分析
- exp005にあってexp006にないサンプル（除去された4件）を特定
- 除去4件のexp005でのスコアを分析
- 4件を除外したexp005の153件でCV再計算 → exp006と公平比較
"""

import pandas as pd
import numpy as np
import evaluate


def compute_cv(preds, refs, label=""):
    metric_chrf = evaluate.load("chrf")
    metric_bleu = evaluate.load("sacrebleu")
    chrf = metric_chrf.compute(predictions=preds, references=refs)["score"]
    bleu = metric_bleu.compute(predictions=preds, references=[[x] for x in refs])["score"]
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    return {"chrf": round(chrf, 4), "bleu": round(bleu, 4), "geo_mean": round(geo_mean, 4)}


def main():
    # --- 1. データ読み込み ---
    exp005 = pd.read_csv(
        "/home/user/work/Akkadian/workspace/exp005_label_masking/results/val_predictions_sentence.csv"
    )
    exp006 = pd.read_csv(
        "/home/user/work/Akkadian/workspace/exp006_data_filtering/results/val_predictions_sentence.csv"
    )

    print(f"exp005: {len(exp005)} samples")
    print(f"exp006: {len(exp006)} samples")

    # --- 2. exp005にあってexp006にないreferenceを特定 ---
    refs_005 = set(exp005["reference"].tolist())
    refs_006 = set(exp006["reference"].tolist())
    removed_refs = refs_005 - refs_006
    print(f"\n除去されたサンプル数: {len(removed_refs)}")

    # --- 3. 除去された4件の詳細表示 ---
    removed_df = exp005[exp005["reference"].isin(removed_refs)].copy()
    print("\n" + "=" * 80)
    print("除去された4件のサンプル詳細 (exp005での予測)")
    print("=" * 80)

    for i, (_, row) in enumerate(removed_df.iterrows()):
        print(f"\n--- 除去サンプル {i+1} ---")
        print(f"  input:          {row['input'][:120]}...")
        print(f"  reference:      {row['reference'][:120]}...")
        print(f"  prediction_raw: {row['prediction_raw'][:120]}...")

        # 個別のchrF/BLEUスコア
        metric_chrf = evaluate.load("chrf")
        metric_bleu = evaluate.load("sacrebleu")
        chrf_i = metric_chrf.compute(
            predictions=[row["prediction_raw"]], references=[row["reference"]]
        )["score"]
        bleu_i = metric_bleu.compute(
            predictions=[row["prediction_raw"]], references=[[row["reference"]]]
        )["score"]
        geo_i = (chrf_i * bleu_i) ** 0.5 if chrf_i > 0 and bleu_i > 0 else 0.0
        print(f"  chrF={chrf_i:.2f}, BLEU={bleu_i:.2f}, geo_mean={geo_i:.2f}")

    # --- 4. exp005全体のCV (157件) ---
    print("\n" + "=" * 80)
    print("exp005 全体 (157件) — prediction_raw")
    cv_005_all_raw = compute_cv(
        exp005["prediction_raw"].tolist(), exp005["reference"].tolist()
    )
    print(f"  chrF={cv_005_all_raw['chrf']:.4f}, BLEU={cv_005_all_raw['bleu']:.4f}, geo_mean={cv_005_all_raw['geo_mean']:.4f}")

    print("\nexp005 全体 (157件) — prediction_post")
    cv_005_all_post = compute_cv(
        exp005["prediction_post"].tolist(), exp005["reference"].tolist()
    )
    print(f"  chrF={cv_005_all_post['chrf']:.4f}, BLEU={cv_005_all_post['bleu']:.4f}, geo_mean={cv_005_all_post['geo_mean']:.4f}")

    # --- 5. exp005から4件除外した153件でCV再計算 ---
    exp005_filtered = exp005[~exp005["reference"].isin(removed_refs)].copy()
    print(f"\n{'=' * 80}")
    print(f"exp005 除外後 ({len(exp005_filtered)}件) — prediction_raw")
    cv_005_filtered_raw = compute_cv(
        exp005_filtered["prediction_raw"].tolist(), exp005_filtered["reference"].tolist()
    )
    print(f"  chrF={cv_005_filtered_raw['chrf']:.4f}, BLEU={cv_005_filtered_raw['bleu']:.4f}, geo_mean={cv_005_filtered_raw['geo_mean']:.4f}")

    print(f"\nexp005 除外後 ({len(exp005_filtered)}件) — prediction_post")
    cv_005_filtered_post = compute_cv(
        exp005_filtered["prediction_post"].tolist(), exp005_filtered["reference"].tolist()
    )
    print(f"  chrF={cv_005_filtered_post['chrf']:.4f}, BLEU={cv_005_filtered_post['bleu']:.4f}, geo_mean={cv_005_filtered_post['geo_mean']:.4f}")

    # --- 6. exp006のCV (153件) ---
    print(f"\n{'=' * 80}")
    print(f"exp006 ({len(exp006)}件) — prediction_raw")
    cv_006_raw = compute_cv(
        exp006["prediction_raw"].tolist(), exp006["reference"].tolist()
    )
    print(f"  chrF={cv_006_raw['chrf']:.4f}, BLEU={cv_006_raw['bleu']:.4f}, geo_mean={cv_006_raw['geo_mean']:.4f}")

    print(f"\nexp006 ({len(exp006)}件) — prediction_post")
    cv_006_post = compute_cv(
        exp006["prediction_post"].tolist(), exp006["reference"].tolist()
    )
    print(f"  chrF={cv_006_post['chrf']:.4f}, BLEU={cv_006_post['bleu']:.4f}, geo_mean={cv_006_post['geo_mean']:.4f}")

    # --- 7. サマリー比較表 ---
    print(f"\n{'=' * 80}")
    print("公平比較サマリー (153件同一セット, prediction_raw)")
    print(f"  {'':>25s}  {'chrF':>8s}  {'BLEU':>8s}  {'geo_mean':>10s}")
    print(f"  {'exp005 (153件, raw)':>25s}  {cv_005_filtered_raw['chrf']:>8.2f}  {cv_005_filtered_raw['bleu']:>8.2f}  {cv_005_filtered_raw['geo_mean']:>10.2f}")
    print(f"  {'exp006 (153件, raw)':>25s}  {cv_006_raw['chrf']:>8.2f}  {cv_006_raw['bleu']:>8.2f}  {cv_006_raw['geo_mean']:>10.2f}")

    diff_chrf = cv_006_raw['chrf'] - cv_005_filtered_raw['chrf']
    diff_bleu = cv_006_raw['bleu'] - cv_005_filtered_raw['bleu']
    diff_geo = cv_006_raw['geo_mean'] - cv_005_filtered_raw['geo_mean']
    print(f"  {'差分 (006 - 005)':>25s}  {diff_chrf:>+8.2f}  {diff_bleu:>+8.2f}  {diff_geo:>+10.2f}")

    print(f"\n公平比較サマリー (153件同一セット, prediction_post)")
    print(f"  {'':>25s}  {'chrF':>8s}  {'BLEU':>8s}  {'geo_mean':>10s}")
    print(f"  {'exp005 (153件, post)':>25s}  {cv_005_filtered_post['chrf']:>8.2f}  {cv_005_filtered_post['bleu']:>8.2f}  {cv_005_filtered_post['geo_mean']:>10.2f}")
    print(f"  {'exp006 (153件, post)':>25s}  {cv_006_post['chrf']:>8.2f}  {cv_006_post['bleu']:>8.2f}  {cv_006_post['geo_mean']:>10.2f}")

    diff_chrf_p = cv_006_post['chrf'] - cv_005_filtered_post['chrf']
    diff_bleu_p = cv_006_post['bleu'] - cv_005_filtered_post['bleu']
    diff_geo_p = cv_006_post['geo_mean'] - cv_005_filtered_post['geo_mean']
    print(f"  {'差分 (006 - 005)':>25s}  {diff_chrf_p:>+8.2f}  {diff_bleu_p:>+8.2f}  {diff_geo_p:>+10.2f}")


if __name__ == "__main__":
    main()
