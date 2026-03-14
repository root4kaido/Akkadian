"""
eda024: exp023ベストモデルのエラー分析
- val_predictions.csvを使い、どこでミスしているかを多角的に分析
"""
import re
import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import sacrebleu

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
VAL_PRED_PATH = PROJECT_ROOT / "workspace" / "exp023_full_preprocessing" / "results" / "val_predictions.csv"
FORM_TAG_DICT_PATH = PROJECT_ROOT / "workspace" / "exp010_pn_gn_tagging" / "dataset" / "form_type_dict.json"
TRAIN_PATH = PROJECT_ROOT / "datasets" / "raw" / "train.csv"

print("=" * 70)
print("eda024: exp023 ベストモデル エラー分析")
print("=" * 70)

# ============================================================
# 1. データ読み込み
# ============================================================
df = pd.read_csv(VAL_PRED_PATH)
print(f"\nVal predictions: {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# inputからprefixを除去してtransliterationを取得
PREFIX = "translate Akkadian to English: "
df["transliteration"] = df["input"].str.replace(PREFIX, "", n=1)
df["pred"] = df["prediction_clean"].astype(str)
df["ref"] = df["reference"].astype(str)

# ============================================================
# 2. 文ごとのchrF++スコアを計算
# ============================================================
print("\n" + "=" * 70)
print("2. 文ごとのchrF++スコア分布")
print("=" * 70)

chrf_scores = []
bleu_scores = []
for _, row in df.iterrows():
    chrf = sacrebleu.sentence_chrf(row["pred"], [row["ref"]], word_order=2).score
    bleu = sacrebleu.sentence_bleu(row["pred"], [row["ref"]]).score
    chrf_scores.append(chrf)
    bleu_scores.append(bleu)

df["chrf"] = chrf_scores
df["bleu"] = bleu_scores
df["geo"] = np.where(
    (df["chrf"] > 0) & (df["bleu"] > 0),
    np.sqrt(df["chrf"] * df["bleu"]),
    0.0
)

print(f"chrF++ mean={df['chrf'].mean():.2f}, median={df['chrf'].median():.2f}")
print(f"BLEU   mean={df['bleu'].mean():.2f}, median={df['bleu'].median():.2f}")
print(f"geo    mean={df['geo'].mean():.2f}, median={df['geo'].median():.2f}")

# スコア分布
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
hist, _ = np.histogram(df["chrf"], bins=bins)
print(f"\nchrF++ distribution:")
for i in range(len(bins) - 1):
    bar = "█" * hist[i]
    print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {hist[i]:3d} {bar}")

# ============================================================
# 3. worst / best 予測の分析
# ============================================================
print("\n" + "=" * 70)
print("3. Worst 10 predictions (lowest chrF++)")
print("=" * 70)

worst = df.nsmallest(10, "chrf")
for i, (_, row) in enumerate(worst.iterrows()):
    print(f"\n  [{i+1}] chrF++={row['chrf']:.1f}, BLEU={row['bleu']:.1f}")
    print(f"    Input: {row['transliteration'][:120]}...")
    print(f"    Ref:   {row['ref'][:120]}...")
    print(f"    Pred:  {row['pred'][:120]}...")

print("\n" + "=" * 70)
print("4. Best 10 predictions (highest chrF++)")
print("=" * 70)

best = df.nlargest(10, "chrf")
for i, (_, row) in enumerate(best.iterrows()):
    print(f"\n  [{i+1}] chrF++={row['chrf']:.1f}, BLEU={row['bleu']:.1f}")
    print(f"    Ref:  {row['ref'][:120]}...")
    print(f"    Pred: {row['pred'][:120]}...")

# ============================================================
# 5. 入力長 vs スコア
# ============================================================
print("\n" + "=" * 70)
print("5. 入力長 vs スコア")
print("=" * 70)

df["input_bytes"] = df["transliteration"].apply(lambda x: len(str(x).encode("utf-8")))
df["ref_len"] = df["ref"].apply(lambda x: len(str(x).split()))
df["pred_len"] = df["pred"].apply(lambda x: len(str(x).split()))

# 入力長のbin別スコア
input_bins = [(0, 100), (100, 200), (200, 300), (300, 500), (500, 1000), (1000, 99999)]
print(f"\n{'Input bytes':>15s} {'Count':>6s} {'chrF++':>8s} {'BLEU':>8s} {'geo':>8s}")
for lo, hi in input_bins:
    mask = (df["input_bytes"] >= lo) & (df["input_bytes"] < hi)
    sub = df[mask]
    if len(sub) == 0:
        continue
    print(f"  {lo:>5d}-{hi:>5d}B   {len(sub):>5d}  {sub['chrf'].mean():>7.2f} {sub['bleu'].mean():>7.2f} {sub['geo'].mean():>7.2f}")

# ============================================================
# 6. 参照文長 vs スコア
# ============================================================
print("\n" + "=" * 70)
print("6. 参照文長(単語数) vs スコア")
print("=" * 70)

ref_bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 99999)]
print(f"\n{'Ref words':>15s} {'Count':>6s} {'chrF++':>8s} {'BLEU':>8s} {'geo':>8s}")
for lo, hi in ref_bins:
    mask = (df["ref_len"] >= lo) & (df["ref_len"] < hi)
    sub = df[mask]
    if len(sub) == 0:
        continue
    print(f"  {lo:>5d}-{hi:>5d}w   {len(sub):>5d}  {sub['chrf'].mean():>7.2f} {sub['bleu'].mean():>7.2f} {sub['geo'].mean():>7.2f}")

# ============================================================
# 7. 予測長 vs 参照長の比較（長すぎ/短すぎ分析）
# ============================================================
print("\n" + "=" * 70)
print("7. 予測長 vs 参照長")
print("=" * 70)

df["len_ratio"] = df["pred_len"] / df["ref_len"].clip(lower=1)
print(f"Length ratio (pred/ref): mean={df['len_ratio'].mean():.2f}, median={df['len_ratio'].median():.2f}")
print(f"  Too short (<0.5): {(df['len_ratio'] < 0.5).sum()} ({(df['len_ratio'] < 0.5).mean()*100:.1f}%)")
print(f"  About right (0.5-1.5): {((df['len_ratio'] >= 0.5) & (df['len_ratio'] <= 1.5)).sum()} ({((df['len_ratio'] >= 0.5) & (df['len_ratio'] <= 1.5)).mean()*100:.1f}%)")
print(f"  Too long (>1.5): {(df['len_ratio'] > 1.5).sum()} ({(df['len_ratio'] > 1.5).mean()*100:.1f}%)")

# 長さ異常のスコア
for label, mask in [("Too short (<0.5)", df['len_ratio'] < 0.5),
                     ("About right", (df['len_ratio'] >= 0.5) & (df['len_ratio'] <= 1.5)),
                     ("Too long (>1.5)", df['len_ratio'] > 1.5)]:
    sub = df[mask]
    if len(sub) > 0:
        print(f"  {label}: n={len(sub)}, chrF++={sub['chrf'].mean():.2f}, BLEU={sub['bleu'].mean():.2f}")

# ============================================================
# 8. 繰り返し分析
# ============================================================
print("\n" + "=" * 70)
print("8. 繰り返し(repetition)分析")
print("=" * 70)

def has_repetition(text, min_repeat=3):
    words = str(text).split()
    if len(words) < min_repeat * 2:
        return False
    for n in range(min_repeat, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return True
    return False

df["has_repetition"] = df["pred"].apply(has_repetition)
rep_count = df["has_repetition"].sum()
print(f"Predictions with repetition: {rep_count}/{len(df)} ({rep_count/len(df)*100:.1f}%)")

rep_df = df[df["has_repetition"]]
norep_df = df[~df["has_repetition"]]
print(f"  With repetition: chrF++={rep_df['chrf'].mean():.2f}, BLEU={rep_df['bleu'].mean():.2f}")
print(f"  Without:         chrF++={norep_df['chrf'].mean():.2f}, BLEU={norep_df['bleu'].mean():.2f}")

# ============================================================
# 9. 固有名詞ミス分析
# ============================================================
print("\n" + "=" * 70)
print("9. 固有名詞ミス分析")
print("=" * 70)

# 参照中の固有名詞を抽出（大文字始まり、ハイフン/アポストロフィ含む名前パターン）
STOP_WORDS = {
    "The", "A", "An", "He", "She", "It", "They", "We", "You", "I",
    "His", "Her", "Its", "Their", "My", "Your", "Our",
    "If", "When", "While", "After", "Before", "Since", "Until",
    "But", "And", "Or", "So", "Yet", "For", "Nor",
    "In", "On", "At", "To", "From", "With", "By", "Of",
    "Not", "No", "Let", "May", "Shall", "Will", "Can",
    "Say", "Said", "Thus", "Therefore", "Because", "However",
    "Here", "There", "Total", "Month", "Seal",
    "Concerning", "Regarding", "According",
}

def extract_names(text):
    """翻訳テキストから固有名詞を抽出"""
    names = []
    # 文分割
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    for sent in sentences:
        words = sent.split()
        for i, word in enumerate(words):
            # ハイフン付き名前（例: Šalim-Aššur）を検出
            clean = re.sub(r"[,.:;!?'\"\(\)\[\]{}]$", "", word)
            clean = re.sub(r"^['\"\(\)\[\]{}]", "", clean)
            if not clean:
                continue
            # 大文字始まりかつストップワードでない
            if clean[0].isupper() and clean not in STOP_WORDS:
                # 文頭は除外（ただしハイフン付きは含む）
                if i == 0 and "-" not in clean:
                    continue
                names.append(clean)
    return names

name_correct = 0
name_total = 0
name_missed_counter = Counter()
rows_with_names = 0
rows_name_perfect = 0

for _, row in df.iterrows():
    ref_names = extract_names(row["ref"])
    if not ref_names:
        continue
    rows_with_names += 1
    pred_text = row["pred"]
    all_found = True
    for name in ref_names:
        name_total += 1
        # 名前が予測に含まれるか（部分一致も含む）
        if name in pred_text or name.rstrip("'s") in pred_text:
            name_correct += 1
        else:
            all_found = False
            name_missed_counter[name] += 1
    if all_found:
        rows_name_perfect += 1

print(f"Rows with proper names: {rows_with_names}/{len(df)}")
print(f"Name accuracy: {name_correct}/{name_total} ({name_correct/name_total*100:.1f}%)")
print(f"Rows with all names correct: {rows_name_perfect}/{rows_with_names} ({rows_name_perfect/rows_with_names*100:.1f}%)")
print(f"Unique missed names: {len(name_missed_counter)}")

print(f"\nTop 30 most missed names:")
for name, count in name_missed_counter.most_common(30):
    print(f"  {name}: {count}")

# 固有名詞ミスのスコア影響
df["ref_names"] = df["ref"].apply(extract_names)
df["n_names"] = df["ref_names"].apply(len)

for label, mask in [("No names", df["n_names"] == 0),
                     ("1-3 names", (df["n_names"] >= 1) & (df["n_names"] <= 3)),
                     ("4-6 names", (df["n_names"] >= 4) & (df["n_names"] <= 6)),
                     ("7+ names", df["n_names"] >= 7)]:
    sub = df[mask]
    if len(sub) > 0:
        print(f"  {label}: n={len(sub)}, chrF++={sub['chrf'].mean():.2f}, geo={sub['geo'].mean():.2f}")

# ============================================================
# 10. エラーカテゴリ分類
# ============================================================
print("\n" + "=" * 70)
print("10. エラーカテゴリ分類")
print("=" * 70)

# 各予測のエラータイプを分類
error_categories = Counter()
for _, row in df.iterrows():
    if row["chrf"] >= 80:
        error_categories["Good (chrF++≥80)"] += 1
        continue

    errors = []

    # 繰り返し
    if row["has_repetition"]:
        errors.append("repetition")

    # 長さ異常
    if row["len_ratio"] < 0.5:
        errors.append("too_short")
    elif row["len_ratio"] > 1.5:
        errors.append("too_long")

    # 固有名詞ミス
    ref_names = extract_names(row["ref"])
    if ref_names:
        missed = sum(1 for n in ref_names if n not in row["pred"] and n.rstrip("'s") not in row["pred"])
        if missed > 0:
            errors.append("name_error")

    # 内容の大幅な不一致（chrF++ < 30）
    if row["chrf"] < 30:
        errors.append("severe_mismatch")

    if not errors:
        errors.append("moderate_error")

    for e in errors:
        error_categories[e] += 1

print(f"\nError category distribution:")
for cat, count in error_categories.most_common():
    print(f"  {cat:25s}: {count:4d} ({count/len(df)*100:.1f}%)")

# ============================================================
# 11. <gap>の影響
# ============================================================
print("\n" + "=" * 70)
print("11. <gap>の影響")
print("=" * 70)

df["has_gap_ref"] = df["ref"].str.contains("<gap>", regex=False)
df["n_gaps"] = df["ref"].str.count("<gap>")

gap_df = df[df["has_gap_ref"]]
nogap_df = df[~df["has_gap_ref"]]
print(f"Refs with <gap>: {len(gap_df)}/{len(df)} ({len(gap_df)/len(df)*100:.1f}%)")
print(f"  With <gap>:    chrF++={gap_df['chrf'].mean():.2f}, geo={gap_df['geo'].mean():.2f}")
print(f"  Without <gap>: chrF++={nogap_df['chrf'].mean():.2f}, geo={nogap_df['geo'].mean():.2f}")

# gap数別
for n in [0, 1, 2, 3]:
    sub = df[df["n_gaps"] == n]
    if len(sub) > 0:
        print(f"  {n} gaps: n={len(sub)}, chrF++={sub['chrf'].mean():.2f}")
sub = df[df["n_gaps"] >= 4]
if len(sub) > 0:
    print(f"  4+ gaps: n={len(sub)}, chrF++={sub['chrf'].mean():.2f}")

# ============================================================
# 12. 数値翻訳の正確性
# ============================================================
print("\n" + "=" * 70)
print("12. 数値翻訳の正確性")
print("=" * 70)

def extract_numbers(text):
    """テキストから数値を抽出"""
    # 整数、分数(½等)、小数
    nums = re.findall(r'\d+(?:\.\d+)?|[½¼⅓⅔⅚¾⅙⅝]', str(text))
    return nums

num_correct = 0
num_total = 0
num_rows = 0
for _, row in df.iterrows():
    ref_nums = extract_numbers(row["ref"])
    pred_nums = extract_numbers(row["pred"])
    if ref_nums:
        num_rows += 1
        ref_set = Counter(ref_nums)
        pred_set = Counter(pred_nums)
        for num, count in ref_set.items():
            num_total += count
            num_correct += min(count, pred_set.get(num, 0))

print(f"Rows with numbers: {num_rows}/{len(df)}")
print(f"Number accuracy: {num_correct}/{num_total} ({num_correct/num_total*100:.1f}%)" if num_total > 0 else "No numbers found")

# 数値ありvsなしのスコア
df["has_numbers"] = df["ref"].apply(lambda x: len(extract_numbers(x)) > 0)
num_df = df[df["has_numbers"]]
nonum_df = df[~df["has_numbers"]]
print(f"  With numbers:    n={len(num_df)}, chrF++={num_df['chrf'].mean():.2f}, geo={num_df['geo'].mean():.2f}")
print(f"  Without numbers: n={len(nonum_df)}, chrF++={nonum_df['chrf'].mean():.2f}, geo={nonum_df['geo'].mean():.2f}")

# ============================================================
# 13. サマリー
# ============================================================
print("\n" + "=" * 70)
print("13. サマリー: 改善の優先度")
print("=" * 70)

print(f"""
=== exp023エラー分析サマリー ===

■ 全体スコア
  chrF++={df['chrf'].mean():.2f}, BLEU={df['bleu'].mean():.2f}, geo={df['geo'].mean():.2f}
  N={len(df)} predictions

■ 主要エラー要因（改善インパクト順）
""")

# インパクト推定: 各カテゴリを完璧にしたときの改善量
# 繰り返し
if len(rep_df) > 0:
    rep_impact = (norep_df["chrf"].mean() - rep_df["chrf"].mean()) * len(rep_df) / len(df)
    print(f"  1. 繰り返し: {len(rep_df)}件 ({len(rep_df)/len(df)*100:.1f}%)")
    print(f"     平均chrF++: {rep_df['chrf'].mean():.2f} vs 非繰り返し {norep_df['chrf'].mean():.2f}")
    print(f"     推定インパクト: +{rep_impact:.2f}pt (全て解消した場合)")

# 長さ異常
short_df = df[df["len_ratio"] < 0.5]
long_df = df[df["len_ratio"] > 1.5]
right_df = df[(df["len_ratio"] >= 0.5) & (df["len_ratio"] <= 1.5)]
if len(short_df) > 0:
    short_impact = (right_df["chrf"].mean() - short_df["chrf"].mean()) * len(short_df) / len(df)
    print(f"\n  2. 予測が短すぎ: {len(short_df)}件 ({len(short_df)/len(df)*100:.1f}%)")
    print(f"     平均chrF++: {short_df['chrf'].mean():.2f}")
    print(f"     推定インパクト: +{short_impact:.2f}pt")
if len(long_df) > 0:
    long_impact = (right_df["chrf"].mean() - long_df["chrf"].mean()) * len(long_df) / len(df)
    print(f"\n  3. 予測が長すぎ: {len(long_df)}件 ({len(long_df)/len(df)*100:.1f}%)")
    print(f"     平均chrF++: {long_df['chrf'].mean():.2f}")
    print(f"     推定インパクト: +{long_impact:.2f}pt")

# 固有名詞
print(f"\n  4. 固有名詞ミス: {name_total - name_correct}/{name_total} ({(name_total-name_correct)/name_total*100:.1f}%)")

# 数値
if num_total > 0:
    print(f"\n  5. 数値ミス: {num_total - num_correct}/{num_total} ({(num_total-num_correct)/num_total*100:.1f}%)")

# 結果保存
df.to_csv(RESULTS_DIR / "val_predictions_scored.csv", index=False)
print(f"\nScored predictions saved to {RESULTS_DIR / 'val_predictions_scored.csv'}")
