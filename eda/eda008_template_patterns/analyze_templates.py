"""
eda008: OA商業文書の定型パターン分析
Gutherz 2023「定型ジャンルほどスコアが高い」を受けて、
train.csvの翻訳テキストから定型構造を抽出・分類する。
"""

import pandas as pd
import re
from collections import Counter, defaultdict
import json
import os

OUT_DIR = "eda/eda008_template_patterns"
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)

# --- Load data ---
train = pd.read_csv("datasets/raw/train.csv")
print(f"Train size: {len(train)}")
print(f"Columns: {train.columns.tolist()}")
print()

translations = train["translation"].dropna().tolist()

# =====================================================
# 1. 文頭パターン・定型句の抽出
# =====================================================
print("=" * 60)
print("1. 文頭パターン・定型句の抽出")
print("=" * 60)

# 文単位に分割
all_sentences = []
for t in translations:
    # ピリオド・疑問符・感嘆符で分割
    sents = re.split(r'(?<=[.!?])\s+', t.strip())
    for s in sents:
        s = s.strip()
        if len(s) > 5:
            all_sentences.append(s)

print(f"Total sentences: {len(all_sentences)}")

# 文頭Nワードの頻度
for n_words in [2, 3, 4, 5]:
    starts = []
    for s in all_sentences:
        words = s.split()[:n_words]
        if len(words) == n_words:
            starts.append(" ".join(words))
    counter = Counter(starts)
    print(f"\n--- Top 30 sentence starts ({n_words} words) ---")
    for pattern, count in counter.most_common(30):
        if count >= 3:
            print(f"  {count:4d}x  {pattern}")

# =====================================================
# 2. 文書タイプ別の分類
# =====================================================
print("\n" + "=" * 60)
print("2. 文書タイプ別の分類")
print("=" * 60)

# キーワードベースの文書タイプ分類
doc_type_patterns = {
    "Seal": [r"(?i)seal\s+of\b", r"(?i)^seal\b"],
    "Envelope/Tablet": [r"(?i)envelope", r"(?i)tablet"],
    "Witness": [r"(?i)witness(?:es)?:", r"(?i)before\s+witness"],
    "Contract/Agreement": [r"(?i)(?:shall|will)\s+(?:pay|give|deliver|return)", r"(?i)agreed", r"(?i)contract"],
    "Letter/Message": [r"(?i)say\s+to\b", r"(?i)speak\s+to\b", r"(?i)thus\s+says?\b", r"(?i)message\s+of\b"],
    "Debt/Loan": [r"(?i)\bowe[sd]?\b", r"(?i)\bdebt\b", r"(?i)\bloan\b", r"(?i)\bcredit(?:or)?\b", r"(?i)\bsilver\b.*\bpay\b"],
    "Legal/Court": [r"(?i)\bswear\b", r"(?i)\boath\b", r"(?i)\bcourt\b", r"(?i)\bjudge\b"],
    "Receipt": [r"(?i)\breceive[d]?\b", r"(?i)\breceipt\b"],
    "Shipping/Transport": [r"(?i)\bcaravan\b", r"(?i)\btransport\b", r"(?i)\bdonkey\b", r"(?i)\bload\b"],
}

doc_types = defaultdict(list)
doc_type_counts = Counter()

for idx, t in enumerate(translations):
    matched_types = []
    for dtype, patterns in doc_type_patterns.items():
        for p in patterns:
            if re.search(p, t):
                matched_types.append(dtype)
                break
    if not matched_types:
        matched_types = ["Other/Unknown"]
    for dtype in matched_types:
        doc_types[dtype].append(idx)
        doc_type_counts[dtype] += 1

print("\n文書タイプ分類結果:")
for dtype, count in doc_type_counts.most_common():
    pct = count / len(translations) * 100
    print(f"  {dtype:25s}: {count:5d} ({pct:5.1f}%)")

# 複数タイプに分類された文書
multi_type = sum(1 for idx in range(len(translations))
                 if sum(1 for d, idxs in doc_types.items() if idx in idxs) > 1)
print(f"\n複数タイプに分類: {multi_type} ({multi_type/len(translations)*100:.1f}%)")

# =====================================================
# 3. テンプレート+穴埋めパターンの特定
# =====================================================
print("\n" + "=" * 60)
print("3. テンプレート+穴埋めパターンの特定")
print("=" * 60)

# 固有名詞パターン（大文字始まり）
name_pattern = r'\b[A-Z][a-z]+(?:-[A-Za-z]+)*\b'

# Sealパターンの分析
seal_docs = [translations[i] for i in doc_types.get("Seal", [])]
print(f"\n--- Seal文書: {len(seal_docs)}件 ---")

seal_templates = []
for t in seal_docs[:50]:
    # Sealに関する行を抽出
    lines = t.split("\n") if "\n" in t else re.split(r'(?<=[.!?])\s+', t)
    for line in lines:
        if re.search(r"(?i)seal", line):
            # 固有名詞をPLACEHOLDERに置換
            template = re.sub(name_pattern, "{NAME}", line.strip())
            seal_templates.append(template)

seal_template_counter = Counter(seal_templates)
print("Sealテンプレート（上位20）:")
for tmpl, count in seal_template_counter.most_common(20):
    if count >= 2:
        print(f"  {count:3d}x  {tmpl}")

# Letterパターンの分析
letter_docs = [translations[i] for i in doc_types.get("Letter/Message", [])]
print(f"\n--- Letter/Message文書: {len(letter_docs)}件 ---")

letter_starts = []
for t in letter_docs:
    # 最初の文を取得
    first_sent = re.split(r'(?<=[.!?])\s+', t.strip())[0]
    template = re.sub(name_pattern, "{NAME}", first_sent)
    letter_starts.append(template)

letter_template_counter = Counter(letter_starts)
print("Letter冒頭テンプレート（上位20）:")
for tmpl, count in letter_template_counter.most_common(20):
    if count >= 2:
        print(f"  {count:3d}x  {tmpl}")

# Debt/Loanパターンの分析
debt_docs = [translations[i] for i in doc_types.get("Debt/Loan", [])]
print(f"\n--- Debt/Loan文書: {len(debt_docs)}件 ---")

debt_templates = []
for t in debt_docs[:100]:
    sents = re.split(r'(?<=[.!?])\s+', t.strip())
    for s in sents:
        if re.search(r"(?i)\b(?:owe|debt|loan|silver|pay|credit)\b", s):
            template = re.sub(name_pattern, "{NAME}", s)
            # 数値もPLACEHOLDERに
            template = re.sub(r'\b\d+(?:/\d+)?\b', "{NUM}", template)
            debt_templates.append(template)

debt_template_counter = Counter(debt_templates)
print("Debt/Loan定型句（上位20）:")
for tmpl, count in debt_template_counter.most_common(20):
    if count >= 2:
        print(f"  {count:3d}x  {tmpl}")

# =====================================================
# 4. 定型部分と自由記述部分の比率推定
# =====================================================
print("\n" + "=" * 60)
print("4. 定型部分と自由記述部分の比率推定")
print("=" * 60)

# 各文を「定型」「半定型」「自由記述」に分類
# 定型: テンプレート化できる（同一パターンが3回以上出現）
# 半定型: 構造は類似するが詳細が異なる
# 自由記述: ユニークな内容

# 5-wordプレフィックスで定型度を測定
prefix_5 = Counter()
for s in all_sentences:
    words = s.split()[:5]
    if len(words) == 5:
        prefix_5[" ".join(words)] += 1

# 各文の定型度スコア
formulaic_scores = []
for s in all_sentences:
    words = s.split()
    if len(words) < 5:
        formulaic_scores.append(("short", s))
        continue
    prefix = " ".join(words[:5])
    count = prefix_5.get(prefix, 0)
    if count >= 5:
        formulaic_scores.append(("formulaic", s))
    elif count >= 2:
        formulaic_scores.append(("semi-formulaic", s))
    else:
        formulaic_scores.append(("free", s))

cat_counts = Counter(cat for cat, _ in formulaic_scores)
total = len(formulaic_scores)
print("\n文の定型度分類（5-wordプレフィックスベース）:")
for cat in ["formulaic", "semi-formulaic", "free", "short"]:
    c = cat_counts.get(cat, 0)
    print(f"  {cat:20s}: {c:5d} ({c/total*100:5.1f}%)")

# 定型文の例
print("\n--- 定型文の例（formulaic, top 10） ---")
formulaic_examples = [(s, prefix_5[" ".join(s.split()[:5])])
                       for cat, s in formulaic_scores if cat == "formulaic"]
formulaic_examples.sort(key=lambda x: -x[1])
seen = set()
for s, count in formulaic_examples:
    prefix = " ".join(s.split()[:5])
    if prefix not in seen:
        print(f"  [{count}x] {s[:120]}")
        seen.add(prefix)
    if len(seen) >= 10:
        break

# =====================================================
# 5. アッカド語翻字側の定型パターン
# =====================================================
print("\n" + "=" * 60)
print("5. アッカド語翻字側の定型パターン")
print("=" * 60)

transliterations = train["transliteration"].dropna().tolist()

# 翻字の先頭パターン
akk_lines = []
for t in transliterations:
    lines = t.strip().split("\n") if "\n" in t else [t.strip()]
    for line in lines:
        line = line.strip()
        if len(line) > 3:
            akk_lines.append(line)

print(f"Total transliteration lines: {len(akk_lines)}")

for n_words in [2, 3]:
    starts = []
    for line in akk_lines:
        words = line.split()[:n_words]
        if len(words) == n_words:
            starts.append(" ".join(words).lower())
    counter = Counter(starts)
    print(f"\n--- Top 20 transliteration starts ({n_words} words) ---")
    for pattern, count in counter.most_common(20):
        if count >= 5:
            print(f"  {count:4d}x  {pattern}")

# =====================================================
# 6. Witness/Sealセクションの構造分析
# =====================================================
print("\n" + "=" * 60)
print("6. Witness/Sealセクションの構造分析")
print("=" * 60)

# 文書末尾の定型構造を分析
witness_patterns = []
for t in translations:
    # witnessを含む文を抽出
    sents = re.split(r'(?<=[.!?])\s+', t.strip())
    for i, s in enumerate(sents):
        if re.search(r"(?i)witness", s):
            # この文以降を定型セクションとみなす
            remaining = sents[i:]
            template = []
            for r in remaining:
                tmpl = re.sub(name_pattern, "{NAME}", r)
                tmpl = re.sub(r'\b\d+(?:/\d+)?\b', "{NUM}", tmpl)
                template.append(tmpl)
            witness_patterns.append(" | ".join(template))
            break

print(f"Witness含有文書: {len(witness_patterns)}/{len(translations)} ({len(witness_patterns)/len(translations)*100:.1f}%)")

witness_counter = Counter(witness_patterns)
print("\n上位Witnessセクション構造（上位10）:")
for pattern, count in witness_counter.most_common(10):
    if count >= 2:
        print(f"  {count:3d}x  {pattern[:150]}")

# =====================================================
# 7. 数値・単位の定型パターン
# =====================================================
print("\n" + "=" * 60)
print("7. 数値・単位の定型パターン")
print("=" * 60)

unit_patterns = Counter()
for t in translations:
    # 数値+単位パターン
    matches = re.findall(r'\b(\d+(?:/\d+)?)\s+(minas?|shekels?|talents?|kutanu|textiles?|tin|silver|gold|donkeys?|bags?)\b', t, re.IGNORECASE)
    for num, unit in matches:
        unit_patterns[unit.lower()] += 1

print("頻出単位:")
for unit, count in unit_patterns.most_common(20):
    print(f"  {unit:20s}: {count:4d}")

# =====================================================
# 8. 全体サマリー統計
# =====================================================
print("\n" + "=" * 60)
print("8. 全体サマリー")
print("=" * 60)

# 文書あたりの定型文比率
doc_formulaic_ratios = []
for t in translations:
    sents = re.split(r'(?<=[.!?])\s+', t.strip())
    total_s = 0
    formulaic_s = 0
    for s in sents:
        if len(s.strip()) <= 5:
            continue
        total_s += 1
        words = s.split()
        if len(words) >= 5:
            prefix = " ".join(words[:5])
            if prefix_5.get(prefix, 0) >= 2:
                formulaic_s += 1
    if total_s > 0:
        doc_formulaic_ratios.append(formulaic_s / total_s)

import statistics
print(f"\n文書あたりの定型文比率:")
print(f"  Mean:   {statistics.mean(doc_formulaic_ratios):.3f}")
print(f"  Median: {statistics.median(doc_formulaic_ratios):.3f}")
print(f"  Std:    {statistics.stdev(doc_formulaic_ratios):.3f}")
print(f"  >50%定型: {sum(1 for r in doc_formulaic_ratios if r > 0.5)}/{len(doc_formulaic_ratios)} ({sum(1 for r in doc_formulaic_ratios if r > 0.5)/len(doc_formulaic_ratios)*100:.1f}%)")
print(f"  >75%定型: {sum(1 for r in doc_formulaic_ratios if r > 0.75)}/{len(doc_formulaic_ratios)} ({sum(1 for r in doc_formulaic_ratios if r > 0.75)/len(doc_formulaic_ratios)*100:.1f}%)")
print(f"  100%定型: {sum(1 for r in doc_formulaic_ratios if r >= 1.0)}/{len(doc_formulaic_ratios)} ({sum(1 for r in doc_formulaic_ratios if r >= 1.0)/len(doc_formulaic_ratios)*100:.1f}%)")

# 定型パターンの網羅性（テンプレートでカバーできるトークン割合）
total_tokens = 0
covered_tokens = 0
for s in all_sentences:
    words = s.split()
    total_tokens += len(words)
    if len(words) >= 5:
        prefix = " ".join(words[:5])
        if prefix_5.get(prefix, 0) >= 2:
            covered_tokens += len(words)

print(f"\nトークンカバレッジ（定型文に属するトークン/全トークン）:")
print(f"  {covered_tokens}/{total_tokens} ({covered_tokens/total_tokens*100:.1f}%)")

print("\n--- Done ---")
