"""
eda022: Host推奨前処理の網羅性監査
- exp022で実装済みの前処理 vs Host推奨の全項目を照合
- 未対応パターンの件数・具体例を洗い出す
- train/testそれぞれで確認
"""
import pandas as pd
import re
import unicodedata
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
train_df = pd.read_csv(PROJECT_ROOT / "datasets" / "raw" / "train.csv")
test_df = pd.read_csv(PROJECT_ROOT / "datasets" / "raw" / "test.csv")

results = {}

def report(section, key, value, examples=None):
    if section not in results:
        results[section] = {}
    results[section][key] = {"count": value}
    if examples:
        results[section][key]["examples"] = examples[:5]
    print(f"  {key}: {value}" + (f"  ex: {examples[0][:80]}" if examples else ""))


# ============================================================
# 1. 小数→分数: 完全一致 vs 丸め誤差
# ============================================================
print("=" * 60)
print("1. 小数→分数変換の網羅性")
print("=" * 60)

KNOWN_FRACTIONS = {
    0.5: "½", 0.25: "¼", 0.3333: "⅓", 0.6666: "⅔",
    0.8333: "⅚", 0.75: "¾", 0.1666: "⅙", 0.625: "⅝",
}

# 近似マッチ用のしきい値
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}

for label, series in [("translation", train_df["translation"].astype(str)),
                       ("transliteration", train_df["transliteration"].astype(str)),
                       ("test_transliteration", test_df["transliteration"].astype(str))]:
    print(f"\n--- {label} ---")
    all_decimals = series.str.extractall(r'(\d+\.\d+)')[0]
    if len(all_decimals) == 0:
        print("  No decimals found")
        continue

    exact_match = 0
    approx_match = 0
    no_match = 0
    no_match_examples = []

    for dec_str in all_decimals:
        frac_part = float(dec_str) % 1
        frac_rounded = round(frac_part, 4)

        if frac_rounded in KNOWN_FRACTIONS or frac_rounded == 0.0:
            exact_match += 1
        else:
            # 近似マッチ: 最も近い分数とのdistance
            min_dist = min(abs(frac_part - target) for target in FRACTION_TARGETS)
            if min_dist < 0.002:  # 0.2%以内
                approx_match += 1
            else:
                no_match += 1
                no_match_examples.append(dec_str)

    report(f"decimals_{label}", "exact_match", exact_match)
    report(f"decimals_{label}", "approx_match_needed", approx_match)
    report(f"decimals_{label}", "no_match", no_match, no_match_examples)

    # 未知小数の分布
    if no_match_examples:
        frac_counts = Counter(str(round(float(x) % 1, 4)) for x in no_match_examples)
        print(f"  未知小数部分: {dict(frac_counts.most_common(10))}")


# ============================================================
# 2. Translation前処理の漏れチェック
# ============================================================
print("\n" + "=" * 60)
print("2. Translation前処理の漏れチェック")
print("=" * 60)

t = train_df["translation"].astype(str)

# exp022で実装済み
print("\n--- 実装済み（件数確認） ---")
for pat, name in [
    (r'\bfem\.\b', 'fem.'), (r'\bsing\.\b', 'sing.'), (r'\bpl\.\b', 'pl.'),
    (r'\bplural\b', 'plural'), (r'\(\?\)', '(?)'), (r'\bPN\b', 'PN'),
    (r' / ', '/ alt'), (r'\b-gold\b', '-gold'), (r'\b-tax\b', '-tax'),
    (r'\b-textiles\b', '-textiles'), (r'month [IVX]+\b', 'month Roman'),
]:
    cnt = t.str.contains(pat).sum()
    report("translation_implemented", name, cnt)

# stray marks
print("\n--- stray marks ---")
for pat, name in [
    (r'<<\s*>>', '<< >>'), (r'(?<!\.)\.\.(?!\.)', 'stray ..'),
    (r'\bxx?\b', 'stray x/xx'),
]:
    cnt = t.str.contains(pat).sum()
    report("translation_stray", name, cnt)

# ? analysis
print("\n--- ? の詳細分析 ---")
has_q = t.str.contains(r'\?')
has_paren_q = t.str.contains(r'\(\?\)')
only_meaningful_q = has_q & ~has_paren_q
report("translation_question", "? total rows", has_q.sum())
report("translation_question", "(?) rows", has_paren_q.sum())
report("translation_question", "meaningful ? rows", only_meaningful_q.sum())

# ?の位置分析
mid_q = t.str.contains(r'[a-zA-Z]\s*\?\s*[a-zA-Z]')  # 文中の?
end_q = t.str.contains(r'\?\s*$')  # 文末の?
report("translation_question", "mid-sentence ?", mid_q.sum())
report("translation_question", "end-of-text ?", end_q.sum())


# ============================================================
# 3. Transliteration前処理の漏れチェック
# ============================================================
print("\n" + "=" * 60)
print("3. Transliteration前処理の漏れチェック")
print("=" * 60)

tr = train_df["transliteration"].astype(str)
test_tr = test_df["transliteration"].astype(str)

for label, series in [("train", tr), ("test", test_tr)]:
    print(f"\n--- {label} ---")

    # Ḫ/ḫ
    report(f"translit_{label}", "Ḫ/ḫ rows", series.str.contains(r'[Ḫḫ]').sum())

    # subscript digits
    report(f"translit_{label}", "subscript digits", series.str.contains(r'[₀₁₂₃₄₅₆₇₈₉]').sum())

    # KÙ.BABBAR vs KÙ.B.
    report(f"translit_{label}", "KÙ.BABBAR", series.str.contains(re.escape('KÙ.BABBAR')).sum())
    kub_short = series.str.contains(r'KÙ\.B\.(?!ABBAR)').sum()
    report(f"translit_{label}", "KÙ.B.(not BABBAR)", kub_short)

    # (d)/(ki) vs {d}/{ki}
    report(f"translit_{label}", "(d)", series.str.contains(r'\(d\)').sum())
    report(f"translit_{label}", "(ki)", series.str.contains(r'\(ki\)').sum())
    report(f"translit_{label}", "{d}", series.str.contains(r'\{d\}').sum())
    report(f"translit_{label}", "{ki}", series.str.contains(r'\{ki\}').sum())

    # Unicode NFC
    nfc_diff = sum(1 for x in series if x != unicodedata.normalize("NFC", x))
    report(f"translit_{label}", "NFC diff", nfc_diff)


# ============================================================
# 4. テストデータ固有のパターン（trainにない文字・記号）
# ============================================================
print("\n" + "=" * 60)
print("4. テストデータ固有のパターン")
print("=" * 60)

# テストのtransliterationに含まれる文字セット
test_chars = set()
for text in test_tr:
    test_chars.update(text)

train_chars = set()
for text in tr:
    train_chars.update(text)

test_only = test_chars - train_chars
train_only = train_chars - test_chars

print(f"Test-only chars ({len(test_only)}): {sorted(test_only)}")
print(f"Train-only chars ({len(train_only)}): {sorted(train_only)[:30]}...")


# ============================================================
# 5. サマリー
# ============================================================
print("\n" + "=" * 60)
print("5. 改善優先度サマリー")
print("=" * 60)

print("""
優先度HIGH:
- 小数→分数の丸め誤差吸収: approx_matchが必要な件数を確認上記参照
  → 完全一致→近似マッチに変更（0.002以内で最近傍分数にマッチ）
  → train/submit/eval全てで対応必要

優先度LOW（対応不要と判断）:
- NFC正規化: 差分0件
- (d)/(ki): 残存0件（Host更新済み）
- KÙ.B.: 短縮形0件（既にKÙ.BABBAR形式）
- stray ?: 意味のある?が大半、除去はリスク
- shekel分数(N/12): 0件（Host更新済み）
""")

# Save results
with open(RESULTS_DIR / "audit_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nResults saved to {RESULTS_DIR / 'audit_results.json'}")
