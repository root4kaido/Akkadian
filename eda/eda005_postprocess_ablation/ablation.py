"""
後処理ablationテスト: 各コンポーネントの影響を個別に計測
保存済み予測CSVを使用（再推論不要）
"""
import os
import sys
import re
import logging
import pandas as pd
import numpy as np
import evaluate
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# === 後処理コンポーネント（個別関数化） ===

SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SPECIAL_CHARS_TRANS = str.maketrans("ḫḪ", "hH")
FORBIDDEN_CHARS = '!?()"——<>⌈⌋⌊[]+ʾ/;'
FORBIDDEN_TRANS = str.maketrans("", "", FORBIDDEN_CHARS)

PATTERNS = {
    "gap": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
    "big_gap": re.compile(r"(\.{3,}|…|\[\.+\])"),
    "annotations": re.compile(
        r"\((fem|plur|pl|sing|singular|plural|\?|!)\.*\s*\w*\)", re.I
    ),
    "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
    "whitespace": re.compile(r"\s+"),
    "punct_space": re.compile(r"\s+([.,:])"),
    "repeated_punct": re.compile(r"([.,])\1+"),
}


def apply_special_chars(s: pd.Series) -> pd.Series:
    """Ḫ→H, 下付き数字→通常数字"""
    s = s.str.translate(SPECIAL_CHARS_TRANS)
    s = s.str.translate(SUBSCRIPT_TRANS)
    return s


def apply_gap_normalization(s: pd.Series) -> pd.Series:
    """ギャップ記号の正規化"""
    s = s.str.replace(PATTERNS["gap"], "<gap>", regex=True)
    s = s.str.replace(PATTERNS["big_gap"], "<big_gap>", regex=True)
    s = s.str.replace("<gap> <gap>", "<big_gap>", regex=False)
    s = s.str.replace("<big_gap> <big_gap>", "<big_gap>", regex=False)
    return s


def apply_annotation_removal(s: pd.Series) -> pd.Series:
    """アノテーション除去 (fem), (plur) etc."""
    s = s.str.replace(PATTERNS["annotations"], "", regex=True)
    return s


def apply_forbidden_chars(s: pd.Series) -> pd.Series:
    """禁止文字除去（ギャップ保護あり）"""
    s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
    s = s.str.replace("<big_gap>", "\x00BIG\x00", regex=False)
    s = s.str.translate(FORBIDDEN_TRANS)
    s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
    s = s.str.replace("\x00BIG\x00", " <big_gap> ", regex=False)
    return s


def apply_fraction_conversion(s: pd.Series) -> pd.Series:
    """分数変換 0.5→½ etc."""
    s = s.str.replace(r"(\d+)\.5\b", r"\1½", regex=True)
    s = s.str.replace(r"\b0\.5\b", "½", regex=True)
    s = s.str.replace(r"(\d+)\.25\b", r"\1¼", regex=True)
    s = s.str.replace(r"\b0\.25\b", "¼", regex=True)
    s = s.str.replace(r"(\d+)\.75\b", r"\1¾", regex=True)
    s = s.str.replace(r"\b0\.75\b", "¾", regex=True)
    return s


def apply_repeated_removal(s: pd.Series) -> pd.Series:
    """繰り返し単語・n-gram除去"""
    s = s.str.replace(PATTERNS["repeated_words"], r"\1", regex=True)
    for n in range(4, 1, -1):
        pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
        s = s.str.replace(pattern, r"\1", regex=True)
    return s


def apply_punctuation_fixes(s: pd.Series) -> pd.Series:
    """句読点修正"""
    s = s.str.replace(PATTERNS["punct_space"], r"\1", regex=True)
    s = s.str.replace(PATTERNS["repeated_punct"], r"\1", regex=True)
    return s


def apply_cleanup(s: pd.Series) -> pd.Series:
    """最終クリーニング（whitespace + strip + empty→broken text）"""
    s = s.str.replace(PATTERNS["whitespace"], " ", regex=True)
    s = s.str.strip().str.strip("-").str.strip()
    s = s.replace("", "broken text")
    s = s.where(s.str.len() > 0, "broken text")
    return s


def apply_all(s: pd.Series) -> pd.Series:
    """全後処理を適用（postprocess_batchと同等）"""
    s = s.fillna("").astype(str)
    s = apply_special_chars(s)
    s = s.str.replace(PATTERNS["whitespace"], " ", regex=True)
    s = s.str.strip()
    s = apply_gap_normalization(s)
    s = apply_annotation_removal(s)
    s = apply_forbidden_chars(s)
    s = apply_fraction_conversion(s)
    s = apply_repeated_removal(s)
    s = apply_punctuation_fixes(s)
    s = apply_cleanup(s)
    return s


# === 評価 ===

def compute_metrics(preds, refs):
    metric_chrf = evaluate.load("chrf")
    metric_bleu = evaluate.load("sacrebleu")
    chrf = metric_chrf.compute(predictions=preds, references=refs)["score"]
    bleu = metric_bleu.compute(predictions=preds, references=[[x] for x in refs])["score"]
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    return {"chrf": round(chrf, 2), "bleu": round(bleu, 2), "geo_mean": round(geo_mean, 2)}


def run_ablation(preds_raw: list, refs: list, label: str):
    """各コンポーネント単独適用 + 全適用 + 除外テスト"""
    s_raw = pd.Series(preds_raw).fillna("").astype(str)

    # コンポーネント定義
    components = {
        "special_chars": apply_special_chars,
        "gap_normalization": apply_gap_normalization,
        "annotation_removal": apply_annotation_removal,
        "forbidden_chars": apply_forbidden_chars,
        "fraction_conversion": apply_fraction_conversion,
        "repeated_removal": apply_repeated_removal,
        "punctuation_fixes": apply_punctuation_fixes,
    }

    results = {}

    # ベースライン: raw（クリーンアップのみ）
    baseline = apply_cleanup(s_raw.copy())
    m = compute_metrics(baseline.tolist(), refs)
    results["00_raw (cleanup only)"] = m
    logger.info(f"[{label}] raw (cleanup only): {m}")

    # 各コンポーネント単独適用
    for name, func in components.items():
        s = func(s_raw.copy())
        s = apply_cleanup(s)
        m = compute_metrics(s.tolist(), refs)
        results[f"01_only_{name}"] = m
        logger.info(f"[{label}] only {name}: {m}")

    # 全適用
    s_all = apply_all(s_raw.copy())
    m_all = compute_metrics(s_all.tolist(), refs)
    results["02_all"] = m_all
    logger.info(f"[{label}] all: {m_all}")

    # 各コンポーネント除外テスト（全適用から1つずつ抜く）
    for exclude_name, exclude_func in components.items():
        s = s_raw.copy().fillna("").astype(str)
        s = apply_special_chars(s) if exclude_name != "special_chars" else s
        s = s.str.replace(PATTERNS["whitespace"], " ", regex=True).str.strip()
        if exclude_name != "gap_normalization":
            s = apply_gap_normalization(s)
        if exclude_name != "annotation_removal":
            s = apply_annotation_removal(s)
        if exclude_name != "forbidden_chars":
            s = apply_forbidden_chars(s)
        if exclude_name != "fraction_conversion":
            s = apply_fraction_conversion(s)
        if exclude_name != "repeated_removal":
            s = apply_repeated_removal(s)
        if exclude_name != "punctuation_fixes":
            s = apply_punctuation_fixes(s)
        s = apply_cleanup(s)
        m = compute_metrics(s.tolist(), refs)
        results[f"03_without_{exclude_name}"] = m
        logger.info(f"[{label}] without {exclude_name}: {m}")

    return results


def plot_results(results_doc, results_sent, save_path):
    """ablation結果を可視化"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    for ax, (results, title) in zip(axes, [
        (results_doc, "Document-level CV"),
        (results_sent, "Sentence-level CV"),
    ]):
        names = list(results.keys())
        geo_means = [results[n]["geo_mean"] for n in names]
        baseline = results["00_raw (cleanup only)"]["geo_mean"]

        # 短縮名
        short_names = [n.split("_", 1)[1] if "_" in n else n for n in names]

        colors = []
        for n in names:
            if "00_raw" in n:
                colors.append("#4477AA")
            elif "02_all" in n:
                colors.append("#EE6677")
            elif "01_only" in n:
                colors.append("#66CCEE")
            else:
                colors.append("#CCBB44")

        bars = ax.barh(range(len(names)), geo_means, color=colors)
        ax.axvline(x=baseline, color="gray", linestyle="--", alpha=0.7, label=f"raw baseline={baseline:.2f}")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_xlabel("geo_mean")
        ax.set_title(title)
        ax.legend(fontsize=9)

        # 値ラベル
        for bar, val in zip(bars, geo_means):
            diff = val - baseline
            sign = "+" if diff >= 0 else ""
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f} ({sign}{diff:.2f})", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {save_path}")


def main():
    exp_dir = os.path.join(SCRIPT_DIR, "..", "..", "workspace", "exp005_label_masking")

    # doc-level predictions
    doc_csv = os.path.join(exp_dir, "results", "val_predictions.csv")
    df_doc = pd.read_csv(doc_csv)
    doc_preds = df_doc["prediction_raw"].tolist()
    doc_refs = df_doc["reference"].tolist()
    logger.info(f"Doc-level: {len(doc_preds)} samples")

    # sent-level predictions
    sent_csv = os.path.join(exp_dir, "results", "val_predictions_sentence.csv")
    df_sent = pd.read_csv(sent_csv)
    sent_preds = df_sent["prediction_raw"].tolist()
    sent_refs = df_sent["reference"].tolist()
    logger.info(f"Sent-level: {len(sent_preds)} samples")

    # Ablation実行
    logger.info("\n" + "=" * 60)
    logger.info("=== Document-level Ablation ===")
    logger.info("=" * 60)
    results_doc = run_ablation(doc_preds, doc_refs, "doc")

    logger.info("\n" + "=" * 60)
    logger.info("=== Sentence-level Ablation ===")
    logger.info("=" * 60)
    results_sent = run_ablation(sent_preds, sent_refs, "sent")

    # 可視化
    plot_path = os.path.join(SCRIPT_DIR, "figures", "ablation_results.png")
    plot_results(results_doc, results_sent, plot_path)

    # サマリーテーブル出力
    logger.info("\n" + "=" * 60)
    logger.info("=== Summary ===")
    logger.info("=" * 60)

    baseline_doc = results_doc["00_raw (cleanup only)"]["geo_mean"]
    baseline_sent = results_sent["00_raw (cleanup only)"]["geo_mean"]

    rows = []
    for key in results_doc.keys():
        doc_gm = results_doc[key]["geo_mean"]
        sent_gm = results_sent[key]["geo_mean"]
        rows.append({
            "component": key,
            "doc_geo_mean": doc_gm,
            "doc_diff": round(doc_gm - baseline_doc, 2),
            "sent_geo_mean": sent_gm,
            "sent_diff": round(sent_gm - baseline_sent, 2),
        })

    df_summary = pd.DataFrame(rows)
    summary_path = os.path.join(SCRIPT_DIR, "ablation_results.csv")
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"\n{df_summary.to_string(index=False)}")
    logger.info(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
