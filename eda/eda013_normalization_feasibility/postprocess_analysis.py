"""
exp011の予測出力に対して、各種後処理がどの程度該当するかを分析する。
LB35.5ノートブックの後処理コンポーネントの適用可能性を検証。
"""
import re
import math
import os
import sys
from collections import Counter

import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# exp011の予測結果を読み込み
pred_path = os.path.join(PROJECT_ROOT, "workspace/exp011_additional_data/results/val_predictions_sentence.csv")
df = pd.read_csv(pred_path)
print(f"予測データ: {len(df)} rows")
print(f"カラム: {list(df.columns)}")

greedy = df["greedy_pred"].astype(str)
refs = df["reference"].astype(str)

# ============================================================
# 1. 各後処理パターンの該当件数
# ============================================================
print(f"\n{'='*60}")
print(f"=== greedy_pred に含まれるパターン分析 ===")
print(f"{'='*60}")

# Grammar annotations: (fem.), (pl.), (sing.) etc
gram_re = re.compile(r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)(?:\.\s*(?:plur|plural|sing|singular))?\.\?\s*[^)]*\)", re.I)
gram_simple_re = re.compile(r"\((?:fem|plur|pl|sing|singular|plural)\.?\)", re.I)
n_gram = greedy.str.contains(gram_simple_re, regex=True).sum()
print(f"文法注記 (fem)/(pl)等: {n_gram}/{len(greedy)} ({100*n_gram/len(greedy):.1f}%)")

# Uncertain markers (?)
uncertain_re = re.compile(r"\(\?\)")
n_uncertain = greedy.str.contains(uncertain_re, regex=True).sum()
print(f"不確実マーカー (?): {n_uncertain}/{len(greedy)} ({100*n_uncertain/len(greedy):.1f}%)")

# Forbidden chars
forbidden = set('()——<>⌈⌋⌊[]+ʾ;')
n_forbidden = 0
forbidden_counter = Counter()
for text in greedy:
    chars_found = set(text) & forbidden
    if chars_found:
        n_forbidden += 1
        for c in chars_found:
            forbidden_counter[c] += 1
print(f"禁止文字含有: {n_forbidden}/{len(greedy)} ({100*n_forbidden/len(greedy):.1f}%)")
for c, cnt in forbidden_counter.most_common():
    print(f"  {repr(c)}: {cnt} docs")

# Curly quotes
curly_re = re.compile("[\u201c\u201d\u2018\u2019]")
n_curly = greedy.str.contains(curly_re, regex=True).sum()
print(f"カーリークォート: {n_curly}/{len(greedy)} ({100*n_curly/len(greedy):.1f}%)")

# Month Roman numerals
month_re = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
n_month = greedy.str.contains(month_re, regex=True).sum()
print(f"月のローマ数字: {n_month}/{len(greedy)} ({100*n_month/len(greedy):.1f}%)")

# Fractions/decimals
frac_re = re.compile(r"\d+\.\d{3,}")
n_frac = greedy.str.contains(frac_re, regex=True).sum()
print(f"長い小数: {n_frac}/{len(greedy)} ({100*n_frac/len(greedy):.1f}%)")

# PN marker
pn_re = re.compile(r"\bPN\b")
n_pn = greedy.str.contains(pn_re, regex=True).sum()
print(f"PN マーカー: {n_pn}/{len(greedy)} ({100*n_pn/len(greedy):.1f}%)")

# Repeated words (consecutive)
repeat_word_re = re.compile(r"\b(\w+)\s+\1\b")
n_repeat_word = greedy.str.contains(repeat_word_re, regex=True).sum()
print(f"連続重複語: {n_repeat_word}/{len(greedy)} ({100*n_repeat_word/len(greedy):.1f}%)")

# Repeated phrases (2+ words)
repeat_phrase_re = re.compile(r"\b(\w+\s+\w+)\s+\1\b")
n_repeat_phrase = greedy.str.contains(repeat_phrase_re, regex=True).sum()
print(f"連続重複フレーズ(2語): {n_repeat_phrase}/{len(greedy)} ({100*n_repeat_phrase/len(greedy):.1f}%)")

# <gap> in output
gap_re = re.compile(r"<gap>", re.I)
n_gap = greedy.str.contains(gap_re, regex=True).sum()
print(f"<gap>含有: {n_gap}/{len(greedy)} ({100*n_gap/len(greedy):.1f}%)")

# Slash alternatives
slash_re = re.compile(r"(?<!\d)\s*/\s*(?!\d)\S+")
n_slash = greedy.str.contains(slash_re, regex=True).sum()
print(f"スラッシュ代替: {n_slash}/{len(greedy)} ({100*n_slash/len(greedy):.1f}%)")

# ============================================================
# 2. referenceに含まれるパターン（後処理で消すと不一致になる）
# ============================================================
print(f"\n{'='*60}")
print(f"=== reference に含まれるパターン ===")
print(f"{'='*60}")

n_ref_forbidden = 0
ref_forbidden_counter = Counter()
for text in refs:
    chars_found = set(text) & forbidden
    if chars_found:
        n_ref_forbidden += 1
        for c in chars_found:
            ref_forbidden_counter[c] += 1
print(f"禁止文字含有: {n_ref_forbidden}/{len(refs)} ({100*n_ref_forbidden/len(refs):.1f}%)")
for c, cnt in ref_forbidden_counter.most_common():
    print(f"  {repr(c)}: {cnt} docs")

n_ref_gap = refs.str.contains(gap_re, regex=True).sum()
print(f"<gap>含有: {n_ref_gap}/{len(refs)} ({100*n_ref_gap/len(refs):.1f}%)")

n_ref_frac = refs.str.contains(frac_re, regex=True).sum()
print(f"長い小数: {n_ref_frac}/{len(refs)} ({100*n_ref_frac/len(refs):.1f}%)")

n_ref_paren = refs.str.contains(r"[()]", regex=True).sum()
print(f"括弧含有: {n_ref_paren}/{len(refs)} ({100*n_ref_paren/len(refs):.1f}%)")

n_ref_curly = refs.str.contains(curly_re, regex=True).sum()
print(f"カーリークォート: {n_ref_curly}/{len(refs)} ({100*n_ref_curly/len(refs):.1f}%)")

# ============================================================
# 3. repeat_cleanup後もまだ残る繰り返し
# ============================================================
print(f"\n{'='*60}")
print(f"=== repeat_cleanup後の残存繰り返し ===")
print(f"{'='*60}")

greedy_clean = df["greedy_clean"].astype(str)
n_clean_repeat_word = greedy_clean.str.contains(repeat_word_re, regex=True).sum()
n_clean_repeat_phrase = greedy_clean.str.contains(repeat_phrase_re, regex=True).sum()
print(f"greedy_clean 連続重複語: {n_clean_repeat_word}/{len(greedy_clean)}")
print(f"greedy_clean 連続重複フレーズ: {n_clean_repeat_phrase}/{len(greedy_clean)}")

# Show examples of remaining repetitions in greedy_clean
print(f"\n残存繰り返しの例:")
for i, text in enumerate(greedy_clean):
    matches = list(repeat_word_re.finditer(text))
    if matches:
        print(f"  [{i}] repeated: {[m.group() for m in matches[:3]]}")
        print(f"       text: {text[:150]}...")
        if sum(1 for _ in repeat_word_re.finditer(text)) > 0:
            count = sum(1 for _ in repeat_word_re.finditer(text))
            if count > 2:
                print(f"       ({count} repetitions total)")

# ============================================================
# 4. LB35.5ノートブックの後処理をシミュレーション適用
# ============================================================
print(f"\n{'='*60}")
print(f"=== LB35.5後処理シミュレーション ===")
print(f"{'='*60}")

# Try enhanced repeat removal (from LB35.5)
def enhanced_repeat_cleanup(text):
    """LB35.5ノートブック風の強化版繰り返し除去"""
    # Single word repetitions
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text)
    # Multi-word phrase repetitions (2-4 words)
    for n in range(4, 1, -1):
        pat = r"\b((?:\w+\s+){" + str(n-1) + r"}\w+)(?:\s+\1\b)+"
        text = re.sub(pat, r"\1", text)
    return text

# Apply enhanced cleanup to greedy_pred (raw)
enhanced_preds = [enhanced_repeat_cleanup(p) for p in greedy]

# Count how many changed
n_changed = sum(1 for a, b in zip(greedy, enhanced_preds) if a != b)
print(f"enhanced_repeat_cleanup で変化: {n_changed}/{len(greedy)} ({100*n_changed/len(greedy):.1f}%)")

# Compare with current repeat_cleanup
n_changed_vs_clean = sum(1 for a, b in zip(greedy_clean, enhanced_preds) if a != b)
print(f"enhanced vs current repeat_cleanup で差異: {n_changed_vs_clean}/{len(greedy)}")

# Show differences
if n_changed_vs_clean > 0:
    print(f"\n差異の例:")
    shown = 0
    for i, (clean, enhanced) in enumerate(zip(greedy_clean, enhanced_preds)):
        if clean != enhanced and shown < 5:
            print(f"  [{i}]")
            print(f"    current:  {clean[:200]}")
            print(f"    enhanced: {enhanced[:200]}")
            shown += 1

# ============================================================
# 5. サマリー
# ============================================================
print(f"\n{'='*60}")
print(f"=== サマリー: 後処理の適用可能性 ===")
print(f"{'='*60}")

print(f"""
パターン               | pred該当 | ref該当 | 判定
文法注記(fem/pl)       | {n_gram:3d}      | -       | {'要検証' if n_gram > 0 else '対象なし'}
不確実マーカー(?)      | {n_uncertain:3d}      | -       | {'要検証' if n_uncertain > 0 else '対象なし'}
禁止文字               | {n_forbidden:3d}      | {n_ref_forbidden:3d}     | eda005で有害
カーリークォート       | {n_curly:3d}      | {n_ref_curly:3d}     | {'要検証' if n_curly > 0 else '対象なし'}
月ローマ数字           | {n_month:3d}      | -       | {'要検証' if n_month > 0 else '対象なし'}
小数→分数             | {n_frac:3d}      | {n_ref_frac:3d}     | eda005で有害
連続重複語             | {n_repeat_word:3d}      | -       | repeat_cleanupで対応済
連続重複フレーズ       | {n_repeat_phrase:3d}      | -       | repeat_cleanupで対応済
強化版repeat cleanup   | {n_changed:3d}変化  | -       | 要ablation
""")
