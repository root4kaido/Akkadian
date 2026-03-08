"""
eda009: 辞書入力付加の実現可能性調査
BabyFST(Sahala 2020)の知見「80%の語が多義」を受けて、
eBL_Dictionary / OA_Lexiconで翻字をlemma付きに拡張できるか調査。
後処理(exp007で中立)ではなく、前処理(入力拡張)としての辞書活用を検討。
"""

import pandas as pd
import re
from collections import Counter, defaultdict
import os

OUT_DIR = "eda/eda009_dictionary_augmentation"
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)

# --- Load data ---
train = pd.read_csv("datasets/raw/train.csv")
test = pd.read_csv("datasets/raw/test.csv")
ebl = pd.read_csv("datasets/raw/eBL_Dictionary.csv")
oa_lex = pd.read_csv("datasets/raw/OA_Lexicon_eBL.csv")

print("=" * 60)
print("データセットサイズ")
print("=" * 60)
print(f"train: {len(train)}")
print(f"eBL_Dictionary: {len(ebl)}")
print(f"OA_Lexicon: {len(oa_lex)}")

# =====================================================
# 1. OA_Lexiconの構造分析
# =====================================================
print("\n" + "=" * 60)
print("1. OA_Lexiconの構造分析")
print("=" * 60)

print(f"\nColumns: {oa_lex.columns.tolist()}")
print(f"\nType分布:")
print(oa_lex["type"].value_counts())

# form列: 翻字形式（マッチに使う）
# norm列: 正規化形式
# lexeme列: 辞書見出し語
# eBL列: eBLリンク

print(f"\n--- サンプル（word type） ---")
words = oa_lex[oa_lex["type"] == "word"]
print(words.head(10).to_string())

print(f"\n--- サンプル（PN type） ---")
pns = oa_lex[oa_lex["type"] == "PN"]
print(pns.head(10).to_string())

# type別件数
for t in oa_lex["type"].unique():
    subset = oa_lex[oa_lex["type"] == t]
    n_lexemes = subset["lexeme"].nunique()
    n_forms = subset["form"].nunique()
    print(f"  {t:10s}: {len(subset):6d} entries, {n_lexemes:5d} unique lexemes, {n_forms:5d} unique forms")

# =====================================================
# 2. eBL_Dictionaryの構造分析
# =====================================================
print("\n" + "=" * 60)
print("2. eBL_Dictionaryの構造分析")
print("=" * 60)

print(f"\nColumns: {ebl.columns.tolist()}")
print(f"\nSamples:")
print(ebl.head(10).to_string())

# 定義の平均長
ebl["def_len"] = ebl["definition"].fillna("").str.len()
print(f"\n定義文の長さ: mean={ebl['def_len'].mean():.1f}, median={ebl['def_len'].median():.1f}")

# 英語意味を抽出（引用符内）
def extract_meanings(definition):
    if pd.isna(definition):
        return []
    meanings = re.findall(r'"([^"]+)"', definition)
    return meanings

ebl["meanings"] = ebl["definition"].apply(extract_meanings)
ebl["n_meanings"] = ebl["meanings"].apply(len)
print(f"意味が抽出できたエントリ: {(ebl['n_meanings'] > 0).sum()}/{len(ebl)} ({(ebl['n_meanings'] > 0).mean()*100:.1f}%)")
print(f"意味数の分布: mean={ebl['n_meanings'].mean():.2f}, max={ebl['n_meanings'].max()}")

# =====================================================
# 3. 翻字トークンの辞書マッチ率（OA_Lexicon）
# =====================================================
print("\n" + "=" * 60)
print("3. 翻字トークンのOA_Lexiconマッチ率")
print("=" * 60)

# OA_Lexiconのform→lexeme辞書を構築
form_to_lexeme = defaultdict(set)
for _, row in oa_lex.iterrows():
    form = str(row["form"]).strip().lower()
    lexeme = str(row["lexeme"]).strip()
    typ = str(row["type"]).strip()
    form_to_lexeme[form].add((lexeme, typ))

print(f"OA_Lexicon辞書エントリ数（form→lexeme）: {len(form_to_lexeme)}")

# trainの翻字からトークンを抽出
def tokenize_transliteration(text):
    """翻字テキストをトークンに分割"""
    # スペースで分割
    tokens = text.strip().split()
    # 各トークンをさらにクリーニング
    cleaned = []
    for t in tokens:
        # 括弧等を除去
        t = re.sub(r'[<>\[\](){}]', '', t)
        # 空でなければ追加
        if t and t != 'gap':
            cleaned.append(t.lower())
    return cleaned

# 全trainトークンの辞書マッチ
all_tokens = []
for text in train["transliteration"]:
    all_tokens.extend(tokenize_transliteration(str(text)))

print(f"\nTrain翻字の総トークン数: {len(all_tokens)}")
print(f"ユニークトークン数: {len(set(all_tokens))}")

# 完全一致マッチ
exact_match = 0
partial_match = 0
no_match = 0
match_details = {"word": 0, "PN": 0, "GN": 0, "other": 0}
ambiguous = 0

token_counter = Counter(all_tokens)
unique_tokens = set(all_tokens)

matched_tokens = set()
unmatched_tokens = set()
ambiguous_tokens = {}

for token in unique_tokens:
    if token in form_to_lexeme:
        matched_tokens.add(token)
        entries = form_to_lexeme[token]
        if len(entries) > 1:
            ambiguous_tokens[token] = entries
        for lexeme, typ in entries:
            if typ in match_details:
                match_details[typ] += 1
            else:
                match_details["other"] += 1
    else:
        unmatched_tokens.add(token)

# トークン出現頻度ベースのマッチ率
matched_freq = sum(token_counter[t] for t in matched_tokens)
total_freq = sum(token_counter.values())

print(f"\n--- ユニークトークンベース ---")
print(f"  完全一致: {len(matched_tokens)}/{len(unique_tokens)} ({len(matched_tokens)/len(unique_tokens)*100:.1f}%)")
print(f"  未マッチ: {len(unmatched_tokens)}/{len(unique_tokens)} ({len(unmatched_tokens)/len(unique_tokens)*100:.1f}%)")

print(f"\n--- 出現頻度ベース ---")
print(f"  マッチトークン出現数: {matched_freq}/{total_freq} ({matched_freq/total_freq*100:.1f}%)")

print(f"\n--- マッチタイプ内訳 ---")
for typ, count in sorted(match_details.items(), key=lambda x: -x[1]):
    print(f"  {typ:10s}: {count:5d}")

# =====================================================
# 4. 多義性分析（1トークンに複数のlexeme）
# =====================================================
print("\n" + "=" * 60)
print("4. 多義性分析")
print("=" * 60)

print(f"多義的トークン（複数lexeme一致）: {len(ambiguous_tokens)}/{len(matched_tokens)} ({len(ambiguous_tokens)/max(len(matched_tokens),1)*100:.1f}%)")

# 多義性の分布
ambig_counts = Counter(len(v) for v in ambiguous_tokens.values())
print(f"\n多義性の分布（lexeme数）:")
for n, count in sorted(ambig_counts.items()):
    print(f"  {n} lexemes: {count} tokens")

# 高頻度の多義的トークンの例
print(f"\n--- 高頻度の多義的トークン（上位15） ---")
ambig_by_freq = [(t, token_counter[t], form_to_lexeme[t]) for t in ambiguous_tokens]
ambig_by_freq.sort(key=lambda x: -x[1])
for token, freq, entries in ambig_by_freq[:15]:
    lexemes = [f"{l}({t})" for l, t in entries]
    print(f"  {token:25s} freq={freq:4d}  →  {', '.join(lexemes[:5])}")

# =====================================================
# 5. eBL_Dictionaryのマッチ率
# =====================================================
print("\n" + "=" * 60)
print("5. eBL_Dictionaryのマッチ率")
print("=" * 60)

# eBLのword列をキーにする
ebl_words = set(ebl["word"].dropna().str.strip().str.lower())
print(f"eBL辞書エントリ数: {len(ebl_words)}")

# 直接マッチ
ebl_matched = unique_tokens & ebl_words
ebl_matched_freq = sum(token_counter[t] for t in ebl_matched)

print(f"\n--- ユニークトークンベース ---")
print(f"  完全一致: {len(ebl_matched)}/{len(unique_tokens)} ({len(ebl_matched)/len(unique_tokens)*100:.1f}%)")

print(f"\n--- 出現頻度ベース ---")
print(f"  マッチ: {ebl_matched_freq}/{total_freq} ({ebl_matched_freq/total_freq*100:.1f}%)")

# eBLにマッチしてOA_Lexiconにマッチしないトークン
ebl_only = ebl_matched - matched_tokens
print(f"\neBLのみにマッチ（OA_Lexiconにはなし）: {len(ebl_only)}")
if ebl_only:
    ebl_only_by_freq = sorted(ebl_only, key=lambda t: -token_counter[t])
    print("  上位10:")
    for t in ebl_only_by_freq[:10]:
        # 定義を取得
        defn = ebl[ebl["word"].str.strip().str.lower() == t]["definition"].values
        d = defn[0][:80] if len(defn) > 0 else "N/A"
        print(f"    {t:25s} freq={token_counter[t]:4d}  def={d}")

# =====================================================
# 6. 入力拡張フォーマットの設計・シミュレーション
# =====================================================
print("\n" + "=" * 60)
print("6. 入力拡張フォーマットのシミュレーション")
print("=" * 60)

# サンプル翻字テキストで入力拡張をシミュレーション
def augment_with_dictionary(text, form_to_lex, ebl_dict):
    """翻字テキストの各トークンに辞書情報を付加"""
    tokens = text.strip().split()
    augmented = []
    stats = {"matched_oa": 0, "matched_ebl": 0, "unmatched": 0}
    for t in tokens:
        t_lower = t.lower().strip()
        t_clean = re.sub(r'[<>\[\](){}]', '', t_lower)

        if t_clean in form_to_lex:
            entries = form_to_lex[t_clean]
            # 最初のlexemeを使う（多義性は無視）
            lexeme, typ = list(entries)[0]
            if typ == "PN":
                augmented.append(f"{t}[PN]")
            elif typ == "GN":
                augmented.append(f"{t}[GN]")
            else:
                augmented.append(f"{t}[{lexeme}]")
            stats["matched_oa"] += 1
        elif t_clean in ebl_dict:
            meanings = ebl_dict.get(t_clean, "")
            if meanings:
                augmented.append(f"{t}[{meanings}]")
            else:
                augmented.append(t)
            stats["matched_ebl"] += 1
        else:
            augmented.append(t)
            stats["unmatched"] += 1

    return " ".join(augmented), stats

# eBLの簡易辞書構築（word→最初のmeaning）
ebl_simple = {}
for _, row in ebl.iterrows():
    w = str(row["word"]).strip().lower()
    meanings = extract_meanings(str(row["definition"]))
    if meanings:
        ebl_simple[w] = meanings[0][:30]

# 5つのサンプルで試す
print("\n--- 入力拡張サンプル ---")
for i in range(5):
    original = train.iloc[i]["transliteration"]
    # 最初の100文字程度
    short = " ".join(original.split()[:15])
    augmented, stats = augment_with_dictionary(short, form_to_lexeme, ebl_simple)
    print(f"\n[Doc {i}] Original: {short}")
    print(f"         Augmented: {augmented}")
    print(f"         Stats: OA={stats['matched_oa']}, eBL={stats['matched_ebl']}, unmatched={stats['unmatched']}")

# 全trainで入力拡張した場合の統計
print("\n--- 全train入力拡張の統計 ---")
total_stats = {"matched_oa": 0, "matched_ebl": 0, "unmatched": 0}
length_ratios = []
for text in train["transliteration"]:
    _, stats = augment_with_dictionary(str(text), form_to_lexeme, ebl_simple)
    for k in total_stats:
        total_stats[k] += stats[k]

    orig_len = len(str(text).encode("utf-8"))
    aug_text, _ = augment_with_dictionary(str(text), form_to_lexeme, ebl_simple)
    aug_len = len(aug_text.encode("utf-8"))
    length_ratios.append(aug_len / orig_len if orig_len > 0 else 1.0)

total_all = sum(total_stats.values())
for k, v in total_stats.items():
    print(f"  {k:15s}: {v:6d} ({v/total_all*100:.1f}%)")

import statistics
print(f"\n入力長増加率:")
print(f"  Mean:   {statistics.mean(length_ratios):.2f}x")
print(f"  Median: {statistics.median(length_ratios):.2f}x")
print(f"  Max:    {max(length_ratios):.2f}x")

# ByT5のmax_length=512バイトでのtruncation影響
truncated_before = sum(1 for text in train["transliteration"] if len(str(text).encode("utf-8")) > 512)
truncated_after = sum(1 for text, ratio in zip(train["transliteration"], length_ratios)
                      if len(str(text).encode("utf-8")) * ratio > 512)
print(f"\n512バイトtruncation:")
print(f"  拡張前: {truncated_before}/{len(train)} ({truncated_before/len(train)*100:.1f}%)")
print(f"  拡張後: {truncated_after}/{len(train)} ({truncated_after/len(train)*100:.1f}%)")

# =====================================================
# 7. 高頻度未マッチトークンの分析
# =====================================================
print("\n" + "=" * 60)
print("7. 高頻度未マッチトークンの分析")
print("=" * 60)

# OA_LexiconにもeBLにもマッチしないトークン
fully_unmatched = unmatched_tokens - ebl_matched
unmatched_by_freq = sorted(fully_unmatched, key=lambda t: -token_counter[t])

print(f"完全未マッチトークン: {len(fully_unmatched)}/{len(unique_tokens)} ({len(fully_unmatched)/len(unique_tokens)*100:.1f}%)")
print(f"\n--- 高頻度未マッチトークン（上位30） ---")
for t in unmatched_by_freq[:30]:
    print(f"  {t:30s} freq={token_counter[t]:4d}")

# 未マッチトークンのパターン分析
numeric = sum(1 for t in fully_unmatched if re.match(r'^[\d.]+$', t))
with_special = sum(1 for t in fully_unmatched if re.search(r'[<>₂₄ₓ…]', t))
short_tokens = sum(1 for t in fully_unmatched if len(t) <= 2)
print(f"\n未マッチトークンの内訳:")
print(f"  数値: {numeric}")
print(f"  特殊文字含む: {with_special}")
print(f"  短い(≤2文字): {short_tokens}")

# =====================================================
# 8. testデータでのマッチ率
# =====================================================
print("\n" + "=" * 60)
print("8. testデータでのマッチ率")
print("=" * 60)

test_tokens = []
for text in test["transliteration"]:
    test_tokens.extend(tokenize_transliteration(str(text)))

test_unique = set(test_tokens)
test_counter = Counter(test_tokens)
test_oa_matched = test_unique & set(form_to_lexeme.keys())
test_ebl_matched = test_unique & ebl_words
test_matched_freq = sum(test_counter[t] for t in (test_oa_matched | test_ebl_matched))

print(f"Test翻字トークン数: {len(test_tokens)} (unique: {len(test_unique)})")
print(f"OA_Lexiconマッチ: {len(test_oa_matched)}/{len(test_unique)} ({len(test_oa_matched)/len(test_unique)*100:.1f}%)")
print(f"eBLマッチ: {len(test_ebl_matched)}/{len(test_unique)} ({len(test_ebl_matched)/len(test_unique)*100:.1f}%)")
print(f"いずれかマッチ（頻度ベース）: {test_matched_freq}/{len(test_tokens)} ({test_matched_freq/len(test_tokens)*100:.1f}%)")

print("\n--- Done ---")
