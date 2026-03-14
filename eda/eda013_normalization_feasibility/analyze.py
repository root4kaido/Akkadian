"""
EDA013: 入力翻字の正規化対象の実態調査
LB35.5ノートブック(ngyzly)が実装する正規化が、我々のデータにどれだけ当てはまるかを定量的に確認する。

調査項目:
1. ASCII→Unicode変換対象 (sz→š, s,→ṣ, t,→ṭ, vowel+2/3)
2. ギャップマーカーの種類と頻度
3. 限定詞 (D), {d} の頻度
4. 小数/分数表記の頻度
5. 特殊文字 (ḫ→h, ʾ→削除 等)
6. train vs test vs additional_data での分布差
7. 正規化前後のユニークトークン数の変化
"""
import os
import re
import math
import sys
from collections import Counter, defaultdict

import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# ============================================================
# Load data
# ============================================================
train = pd.read_csv(os.path.join(PROJECT_ROOT, "datasets/raw/train.csv"))
test = pd.read_csv(os.path.join(PROJECT_ROOT, "datasets/raw/test.csv"))

add_path = os.path.join(PROJECT_ROOT, "workspace/exp011_additional_data/dataset/additional_train.csv")
additional = pd.read_csv(add_path) if os.path.exists(add_path) else pd.DataFrame()

print(f"=== データ件数 ===")
print(f"train: {len(train)}")
print(f"test: {len(test)}")
print(f"additional: {len(additional)}")

datasets = {
    "train": train["transliteration"].astype(str),
    "test": test["transliteration"].astype(str),
}
if len(additional) > 0:
    datasets["additional"] = additional["transliteration"].astype(str)

all_translits = pd.concat(datasets.values())

# ============================================================
# 1. ASCII → Unicode diacritics
# ============================================================
print(f"\n{'='*60}")
print(f"=== 1. ASCII → Unicode変換対象 ===")
print(f"{'='*60}")

ascii_patterns = {
    "sz/SZ → š": r"(?<![a-zA-Z])sz(?![a-zA-Z])|(?<![a-zA-Z])SZ(?![a-zA-Z])",
    "s, → ṣ": r"s,|S,",
    "t, → ṭ": r"t,|T,",
    "vowel+2 (ASCII)": r"[aAeEiIuU]2(?![0-9])",
    "vowel+3 (ASCII)": r"[aAeEiIuU]3(?![0-9])",
    "vowel+₂ (subscript)": r"[aAeEiIuU]₂",
    "vowel+₃ (subscript)": r"[aAeEiIuU]₃",
}

for label, pat in ascii_patterns.items():
    counts = {}
    for name, ser in datasets.items():
        counts[name] = ser.str.contains(pat, regex=True).sum()
    total = sum(counts.values())
    detail = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"  {label}: total={total} ({detail})")

# ============================================================
# 2. ギャップマーカー
# ============================================================
print(f"\n{'='*60}")
print(f"=== 2. ギャップマーカーの種類と頻度 ===")
print(f"{'='*60}")

# Collect all gap-like patterns
gap_counter = Counter()
gap_pattern = re.compile(
    r"<[^>]*>"           # any XML-like tag
    r"|\.{3,}"           # ...
    r"|…+"               # ellipsis char
    r"|\[\.\.\.\]"       # [...]
    r"|\[x\]"            # [x]
    r"|\(x\)"            # (x)
    r"|x{2,}"            # xx, xxx
    r"|\bx\s+x\b"       # x x
    , re.I
)

for text in all_translits:
    for m in gap_pattern.finditer(text):
        gap_counter[m.group()] += 1

print(f"  パターン別出現回数（上位20）:")
for pat, cnt in gap_counter.most_common(20):
    print(f"    {repr(pat):40s}: {cnt}")

# Per dataset breakdown for common gaps
print(f"\n  データセット別 ギャップ含有文書数:")
common_gap_re = re.compile(r"<[^>]*>|\.{3,}|…|\[x\]|\(x\)|x{2,}", re.I)
for name, ser in datasets.items():
    n = ser.str.contains(common_gap_re, regex=True).sum()
    print(f"    {name}: {n}/{len(ser)} ({100*n/len(ser):.1f}%)")

# ============================================================
# 3. 限定詞 (Determinatives)
# ============================================================
print(f"\n{'='*60}")
print(f"=== 3. 限定詞 ===")
print(f"{'='*60}")

det_upper_re = re.compile(r"\([A-ZŠṬṢḪ\u00C0-\u00DE]{1,6}\)")
det_lower_re = re.compile(r"\([a-zšṭṣḫ\u00E0-\u00FF]{1,4}\)")

det_upper_counter = Counter()
det_lower_counter = Counter()

for text in all_translits:
    for m in det_upper_re.finditer(text):
        det_upper_counter[m.group()] += 1
    for m in det_lower_re.finditer(text):
        det_lower_counter[m.group()] += 1

print(f"  大文字限定詞 (D): ユニーク種={len(det_upper_counter)}, 総出現={sum(det_upper_counter.values())}")
for pat, cnt in det_upper_counter.most_common(10):
    print(f"    {pat}: {cnt}")

print(f"  小文字限定詞 (d): ユニーク種={len(det_lower_counter)}, 総出現={sum(det_lower_counter.values())}")
for pat, cnt in det_lower_counter.most_common(10):
    print(f"    {pat}: {cnt}")

# Per dataset
for name, ser in datasets.items():
    n_upper = ser.str.contains(det_upper_re, regex=True).sum()
    n_lower = ser.str.contains(det_lower_re, regex=True).sum()
    print(f"  {name}: 大文字={n_upper}, 小文字={n_lower}")

# ============================================================
# 4. 小数/分数
# ============================================================
print(f"\n{'='*60}")
print(f"=== 4. 小数/分数表記 ===")
print(f"{'='*60}")

frac_re = re.compile(r"\d+\.\d{3,}")
frac_counter = Counter()

for text in all_translits:
    for m in frac_re.finditer(text):
        val = m.group()
        try:
            f = float(val)
            frac_part = f - int(f)
            # Classify
            if abs(frac_part - 1/6) < 0.01:
                frac_counter["1/6"] += 1
            elif abs(frac_part - 1/4) < 0.01:
                frac_counter["1/4"] += 1
            elif abs(frac_part - 1/3) < 0.01:
                frac_counter["1/3"] += 1
            elif abs(frac_part - 1/2) < 0.01:
                frac_counter["1/2"] += 1
            elif abs(frac_part - 2/3) < 0.01:
                frac_counter["2/3"] += 1
            elif abs(frac_part - 3/4) < 0.01:
                frac_counter["3/4"] += 1
            elif abs(frac_part - 5/6) < 0.01:
                frac_counter["5/6"] += 1
            else:
                frac_counter[f"other({val})"] += 1
        except ValueError:
            frac_counter[f"parse_error({val})"] += 1

for name, ser in datasets.items():
    n = ser.str.contains(frac_re, regex=True).sum()
    print(f"  {name}: {n}/{len(ser)} ({100*n/len(ser):.1f}%)")

print(f"\n  分数の種類:")
for frac, cnt in frac_counter.most_common(20):
    print(f"    {frac}: {cnt}")

# ============================================================
# 5. 特殊文字
# ============================================================
print(f"\n{'='*60}")
print(f"=== 5. 特殊文字 ===")
print(f"{'='*60}")

special_chars = {
    "ḫ/Ḫ": r"[ḫḪ]",
    "ʾ (aleph)": r"ʾ",
    "ₓ (subscript x)": r"ₓ",
    "subscript digits (₀-₉)": r"[₀₁₂₃₄₅₆₇₈₉]",
    "em/en dash": r"[—–]",
}

for label, pat in special_chars.items():
    for name, ser in datasets.items():
        n = ser.str.contains(pat, regex=True).sum()
        total_occ = ser.str.count(pat).sum()
        if n > 0 or name == "train":
            print(f"  {label} in {name}: {n} docs, {total_occ} occurrences")

# ============================================================
# 6. トークン正規化のインパクト推定
# ============================================================
print(f"\n{'='*60}")
print(f"=== 6. 正規化によるユニークトークン数の変化推定 ===")
print(f"{'='*60}")

# Current unique tokens
all_tokens = Counter()
for text in all_translits:
    all_tokens.update(text.split())

print(f"  正規化前ユニークトークン数: {len(all_tokens)}")

# Simulate normalization
_CHAR_TRANS = str.maketrans({
    "ḫ": "h", "Ḫ": "H", "ʾ": "",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "—": "-", "–": "-",
})

def normalize_token(tok):
    tok = tok.translate(_CHAR_TRANS)
    tok = tok.replace("ₓ", "")
    return tok

normalized_tokens = Counter()
for text in all_translits:
    for tok in text.split():
        normalized_tokens[normalize_token(tok)] += 1

print(f"  正規化後ユニークトークン数: {len(normalized_tokens)}")
print(f"  削減: {len(all_tokens) - len(normalized_tokens)} tokens ({100*(len(all_tokens)-len(normalized_tokens))/len(all_tokens):.1f}%)")

# Show tokens that merge
merge_map = defaultdict(set)
for text in all_translits:
    for tok in text.split():
        normed = normalize_token(tok)
        if normed != tok:
            merge_map[normed].add(tok)

n_merge_groups = sum(1 for v in merge_map.values() if len(v) > 1)
print(f"  正規化でマージされるグループ数: {n_merge_groups}")
for normed, originals in sorted(merge_map.items(), key=lambda x: -len(x[1]))[:10]:
    if len(originals) > 1:
        print(f"    {normed} ← {originals}")

# ============================================================
# 7. train vs test の正規化対象分布差
# ============================================================
print(f"\n{'='*60}")
print(f"=== 7. train vs test 分布差 ===")
print(f"{'='*60}")

# Check if test has patterns not in train
train_gap_types = set()
test_gap_types = set()
for text in datasets["train"]:
    for m in gap_pattern.finditer(text):
        train_gap_types.add(m.group())
for text in datasets["test"]:
    for m in gap_pattern.finditer(text):
        test_gap_types.add(m.group())

test_only_gaps = test_gap_types - train_gap_types
print(f"  train内ギャップ種類: {len(train_gap_types)}")
print(f"  test内ギャップ種類: {len(test_gap_types)}")
print(f"  test固有ギャップ種類: {len(test_only_gaps)}")
for g in test_only_gaps:
    print(f"    {repr(g)}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"=== サマリー ===")
print(f"{'='*60}")
print(f"1. ASCII→Unicode変換: 対象なし（既にUnicode）")

gap_docs = all_translits.str.contains(common_gap_re, regex=True).sum()
print(f"2. ギャップマーカー: {gap_docs}/{len(all_translits)} docs ({100*gap_docs/len(all_translits):.1f}%)")

frac_docs = all_translits.str.contains(frac_re, regex=True).sum()
print(f"3. 小数/分数: {frac_docs}/{len(all_translits)} docs ({100*frac_docs/len(all_translits):.1f}%)")

det_docs = all_translits.str.contains(det_upper_re, regex=True).sum() + all_translits.str.contains(det_lower_re, regex=True).sum()
print(f"4. 限定詞: {det_docs} docs")

special_docs = all_translits.str.contains(r"[ḫḪʾₓ₀₁₂₃₄₅₆₇₈₉—–]", regex=True).sum()
print(f"5. 特殊文字(ḫ,ʾ,下付き等): {special_docs}/{len(all_translits)} docs ({100*special_docs/len(all_translits):.1f}%)")
print(f"6. トークン正規化: {len(all_tokens)}→{len(normalized_tokens)} ({len(all_tokens)-len(normalized_tokens)} 削減)")
