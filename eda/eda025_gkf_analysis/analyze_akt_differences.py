"""
AKTグループ間の文書特性の違いを分析
- 文書タイプ(genre_label)分布
- 文書長・語彙の違い
- 定型表現の出現頻度
- N-gram overlap（グループ間の語彙共有度）
- 構文パターン（文頭・文末・定型フレーズ）
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

train_df = pd.read_csv(PROJECT_ROOT / "datasets" / "raw" / "train.csv")
pub_df = pd.read_csv(PROJECT_ROOT / "datasets" / "raw" / "published_texts.csv",
                      usecols=["oare_id", "genre_label", "aliases", "description"])
akt = pd.read_csv(PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv")

# マージ
train_df = train_df.merge(akt, on="oare_id", how="left")
train_df["akt_group"] = train_df["akt_group"].fillna("None")
train_df = train_df.merge(pub_df[["oare_id", "genre_label", "description"]], on="oare_id", how="left")

# ============================================================
# 1. 基本統計
# ============================================================
print("=" * 80)
print("1. AKTグループ別基本統計")
print("=" * 80)

for grp, gdf in train_df.groupby("akt_group"):
    trans = gdf["translation"].astype(str)
    translit = gdf["transliteration"].astype(str)
    trans_words = trans.apply(lambda x: len(x.split()))
    translit_bytes = translit.apply(lambda x: len(x.encode("utf-8")))
    print(f"\n{grp} ({len(gdf)}件):")
    print(f"  翻訳 word数: mean={trans_words.mean():.1f}, median={trans_words.median():.0f}, max={trans_words.max()}")
    print(f"  翻字 bytes: mean={translit_bytes.mean():.0f}, median={translit_bytes.median():.0f}, max={translit_bytes.max()}")

# ============================================================
# 2. genre_label分布
# ============================================================
print("\n" + "=" * 80)
print("2. AKTグループ別 genre_label分布")
print("=" * 80)

for grp, gdf in train_df.groupby("akt_group"):
    genres = gdf["genre_label"].value_counts()
    print(f"\n{grp}:")
    for genre, count in genres.items():
        print(f"  {genre}: {count} ({100*count/len(gdf):.1f}%)")

# ============================================================
# 3. 定型フレーズの出現率（グループ別）
# ============================================================
print("\n" + "=" * 80)
print("3. 定型フレーズの出現率（グループ別）")
print("=" * 80)

PATTERNS = {
    "Seal of": r"\bSeal of\b",
    "Witnessed by": r"\bWitnessed? by\b",
    "son of": r"\bson of\b",
    "daughter of": r"\bdaughter of\b",
    "shekels of silver": r"\bshekels? of silver\b",
    "minas of silver": r"\bminas? of silver\b",
    "he will pay": r"\bhe will pay\b",
    "If he has not paid": r"\bIf he has not paid\b",
    "interest": r"\binterest\b",
    "eponymy": r"\beponymy\b",
    "From X to Y": r"\bFrom .+ to .+:",
    "Say to": r"\bSay to\b",
    "thus": r"\bthus\b",
    "tablet": r"\btablet\b",
    "witness": r"\bwitness",
    "textile": r"\btextile",
    "caravan": r"\bcaravan\b",
    "donkey": r"\bdonkey\b",
    "gold": r"\bgold\b",
    "copper": r"\bcopper\b",
    "tin": r"\btin\b",
}

results = {}
for grp, gdf in train_df.groupby("akt_group"):
    trans = gdf["translation"].astype(str)
    all_text = " ".join(trans)
    rates = {}
    for name, pat in PATTERNS.items():
        count = sum(1 for t in trans if re.search(pat, t, re.IGNORECASE))
        rates[name] = 100 * count / len(gdf)
    results[grp] = rates

# 表形式で出力
groups = sorted(results.keys())
print(f"\n{'Pattern':<25}", end="")
for g in groups:
    print(f"{g:>10}", end="")
print()
print("-" * (25 + 10 * len(groups)))
for pat in PATTERNS:
    print(f"{pat:<25}", end="")
    for g in groups:
        print(f"{results[g][pat]:>9.1f}%", end="")
    print()

# ============================================================
# 4. 語彙overlap（bigram Jaccard）
# ============================================================
print("\n" + "=" * 80)
print("4. グループ間 bigram Jaccard類似度")
print("=" * 80)

def get_bigrams(texts):
    bigrams = set()
    for t in texts:
        words = str(t).lower().split()
        for i in range(len(words) - 1):
            bigrams.add((words[i], words[i+1]))
    return bigrams

group_bigrams = {}
for grp, gdf in train_df.groupby("akt_group"):
    group_bigrams[grp] = get_bigrams(gdf["translation"])

print(f"\n{'':>10}", end="")
for g in groups:
    print(f"{g:>10}", end="")
print()

for g1 in groups:
    print(f"{g1:>10}", end="")
    for g2 in groups:
        if g1 == g2:
            print(f"{'1.000':>10}", end="")
        else:
            inter = len(group_bigrams[g1] & group_bigrams[g2])
            union = len(group_bigrams[g1] | group_bigrams[g2])
            jaccard = inter / union if union > 0 else 0
            print(f"{jaccard:>10.3f}", end="")
    print()

# ============================================================
# 5. グループ固有の高頻度語（他グループに少ない語）
# ============================================================
print("\n" + "=" * 80)
print("5. グループ固有の高頻度語（TF-IDF的）")
print("=" * 80)

group_word_counts = {}
for grp, gdf in train_df.groupby("akt_group"):
    words = Counter()
    for t in gdf["translation"].astype(str):
        for w in t.lower().split():
            w = re.sub(r'[,.:;!?\(\)\[\]"\']+', '', w)
            if w and len(w) > 2:
                words[w] += 1
    group_word_counts[grp] = words

# 各グループの特徴語（そのグループでの頻度/全体頻度 が高いもの）
total_counts = Counter()
for counts in group_word_counts.values():
    total_counts += counts

for grp in groups:
    counts = group_word_counts[grp]
    n_docs = len(train_df[train_df["akt_group"] == grp])
    scored = []
    for word, cnt in counts.items():
        if cnt < 5:
            continue
        total = total_counts[word]
        specificity = (cnt / n_docs) / (total / len(train_df))
        scored.append((word, cnt, total, specificity))
    scored.sort(key=lambda x: -x[3])
    print(f"\n{grp} 特徴語 TOP10:")
    for word, cnt, total, spec in scored[:10]:
        print(f"  {word}: {cnt}回 (全体{total}回, 特異度{spec:.2f})")

# ============================================================
# 6. 文頭パターン（最初の3語）
# ============================================================
print("\n" + "=" * 80)
print("6. 文頭パターン（翻訳の最初の3語）")
print("=" * 80)

for grp, gdf in train_df.groupby("akt_group"):
    starts = Counter()
    for t in gdf["translation"].astype(str):
        words = t.split()[:3]
        if len(words) >= 3:
            starts[" ".join(words)] += 1
    print(f"\n{grp} TOP10文頭:")
    for pattern, cnt in starts.most_common(10):
        print(f"  {pattern}: {cnt}回 ({100*cnt/len(gdf):.1f}%)")
