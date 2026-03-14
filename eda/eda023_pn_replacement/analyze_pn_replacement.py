"""
eda023: PN/GNルールベース置換の実現可能性調査

調査項目:
1. trainデータからPN Akkadian形 → 英語名の変換辞書を構築できるか
2. transliteration中のPN位置とtranslation中の対応名詞のアライメント
3. 複数PNがある場合の順序対応
4. プレースホルダ置換の実現可能性レポート
"""

import json
import re
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = PROJECT_ROOT / "datasets" / "raw" / "train.csv"
FORM_TYPE_DICT_PATH = PROJECT_ROOT / "workspace" / "exp010_pn_gn_tagging" / "dataset" / "form_type_dict.json"
OA_LEXICON_PATH = PROJECT_ROOT / "datasets" / "raw" / "OA_Lexicon_eBL.csv"
RESULTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. データ読み込み
# ============================================================
print("=" * 60)
print("eda023: PN/GNルールベース置換の実現可能性調査")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
print(f"\nTrain data: {len(train_df)} rows")

with open(FORM_TYPE_DICT_PATH) as f:
    form_tag_dict = json.load(f)
print(f"form_tag_dict: {len(form_tag_dict)} entries")

pn_forms = {k for k, v in form_tag_dict.items() if v == "PN"}
gn_forms = {k for k, v in form_tag_dict.items() if v == "GN"}
print(f"  PN forms: {len(pn_forms)}, GN forms: {len(gn_forms)}")

# ============================================================
# 2. trainデータでPNトークンの出現パターンを分析
# ============================================================
print("\n" + "=" * 60)
print("2. trainデータでのPNトークン出現分析")
print("=" * 60)

pn_rows = []  # (row_idx, transliteration, translation, pn_tokens, pn_positions)
for idx, row in train_df.iterrows():
    translit = str(row["transliteration"])
    transl = str(row["translation"])
    tokens = translit.split()
    pn_tokens_in_row = []
    pn_positions = []
    for i, token in enumerate(tokens):
        # 改行内のトークン位置も考慮
        clean_token = token.strip()
        if clean_token in pn_forms:
            pn_tokens_in_row.append(clean_token)
            pn_positions.append(i)
    if pn_tokens_in_row:
        pn_rows.append({
            "idx": idx,
            "transliteration": translit,
            "translation": transl,
            "pn_tokens": pn_tokens_in_row,
            "pn_positions": pn_positions,
            "n_pn": len(pn_tokens_in_row),
        })

print(f"Rows with PN tokens: {len(pn_rows)}/{len(train_df)} ({len(pn_rows)/len(train_df)*100:.1f}%)")

n_pn_dist = Counter(r["n_pn"] for r in pn_rows)
print("\nPN tokens per row distribution:")
for n, count in sorted(n_pn_dist.items()):
    print(f"  {n} PNs: {count} rows ({count/len(pn_rows)*100:.1f}%)")

# 全PNトークンの頻度
all_pn_tokens = []
for r in pn_rows:
    all_pn_tokens.extend(r["pn_tokens"])
pn_freq = Counter(all_pn_tokens)
print(f"\nTotal PN token occurrences: {len(all_pn_tokens)}")
print(f"Unique PN tokens in train: {len(pn_freq)}")
print(f"\nTop 20 most common PN tokens:")
for token, count in pn_freq.most_common(20):
    print(f"  {token}: {count}")

# ============================================================
# 3. PN Akkadian形 → 翻訳中の英語名 対応を抽出
# ============================================================
print("\n" + "=" * 60)
print("3. PN Akkadian形 → 翻訳中の英語名 対応抽出")
print("=" * 60)

# 戦略: PNが1つだけの行で、翻訳中の固有名詞（大文字始まり）を探す
# まず、翻訳中の固有名詞候補を抽出する関数

# 英語翻訳中から固有名詞候補を抽出（大文字始まりの単語で一般的な文頭語を除外）
COMMON_SENTENCE_STARTERS = {
    "The", "A", "An", "He", "She", "It", "They", "We", "You", "I",
    "His", "Her", "Its", "Their", "My", "Your", "Our",
    "This", "That", "These", "Those",
    "If", "When", "While", "After", "Before", "Since", "Until",
    "But", "And", "Or", "So", "Yet", "For", "Nor",
    "In", "On", "At", "To", "From", "With", "By", "Of",
    "Not", "No", "Let", "May", "Shall", "Will", "Can",
    "As", "About", "Into", "Over", "Under", "Between",
    "Also", "Even", "Still", "Just", "Only", "Then", "Now",
    "Here", "There", "Where", "Who", "What", "Which", "How",
    "All", "Each", "Every", "Some", "Any", "Many", "Much",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven",
    "Eight", "Nine", "Ten",
    "Seal", "Witness", "Total", "Month", "Year", "Day",
    "Said", "Thus", "Therefore", "Because", "However",
    "According", "Concerning", "Regarding",
}

# 一般的な名詞も除外（翻訳で大文字始まりだが固有名詞ではないもの）
COMMON_NOUNS_UPPER = {
    "City", "Gate", "God", "King", "Queen", "Palace", "Temple",
    "House", "Father", "Mother", "Brother", "Sister", "Son", "Daughter",
    "Lord", "Lady", "Merchant", "Slave", "Servant", "Witness",
    "Silver", "Gold", "Copper", "Tin", "Iron",
    "Tablet", "Letter", "Envelope", "Seal",
}


def extract_proper_names_from_translation(text):
    """翻訳テキストから固有名詞候補を抽出"""
    names = []
    # 文を句読点で分割して、文頭語を識別
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        words = sent.split()
        for i, word in enumerate(words):
            # 大文字始まりで、文頭でない、一般語でもない
            clean_word = re.sub(r"[,.:;!?'\"\(\)\[\]{}]", "", word)
            if not clean_word:
                continue
            if clean_word[0].isupper() and clean_word not in COMMON_SENTENCE_STARTERS and clean_word not in COMMON_NOUNS_UPPER:
                # 文頭の単語は除外（i==0）ただしハイフン付き名前は含む
                if i == 0 and "-" not in clean_word and "'" not in clean_word:
                    continue
                names.append(clean_word)
    return names


# 単一PN行での対応分析
single_pn_rows = [r for r in pn_rows if r["n_pn"] == 1]
print(f"\nSingle-PN rows: {len(single_pn_rows)}")

# Akkadian PN → 英語名の候補マッピング
akk_to_eng_candidates = defaultdict(list)
match_examples = []

for r in single_pn_rows:
    pn_token = r["pn_tokens"][0]
    eng_names = extract_proper_names_from_translation(r["translation"])
    if eng_names:
        for name in eng_names:
            akk_to_eng_candidates[pn_token].append(name)
        if len(match_examples) < 10:
            match_examples.append({
                "akk_pn": pn_token,
                "eng_candidates": eng_names,
                "transliteration": r["transliteration"][:100],
                "translation": r["translation"][:100],
            })

print(f"PN tokens with at least 1 English candidate: {len(akk_to_eng_candidates)}")

# 対応の一貫性を分析
consistent_count = 0
inconsistent_count = 0
no_candidate_count = 0
mapping_examples = []

for pn_token, candidates in akk_to_eng_candidates.items():
    unique_candidates = set(candidates)
    if len(unique_candidates) == 1:
        consistent_count += 1
        if len(mapping_examples) < 5:
            mapping_examples.append((pn_token, list(unique_candidates)[0], len(candidates)))
    else:
        inconsistent_count += 1
        if len(mapping_examples) < 20:
            mapping_examples.append((pn_token, unique_candidates, len(candidates)))

print(f"\nMapping consistency (single-PN rows):")
print(f"  Consistent (1 candidate): {consistent_count}")
print(f"  Inconsistent (multiple): {inconsistent_count}")

# ============================================================
# 4. より正確なアプローチ: 文字列類似度ベースのマッチング
# ============================================================
print("\n" + "=" * 60)
print("4. 文字列類似度ベースのPN→英語名マッチング")
print("=" * 60)

# Akkadianの翻字と英語名は音が似ている（音写なので）
# 例: a-šur → Aššur, pu-zu-ur → Puzur
# 最初の数文字の一致度で対応を推定

def normalize_akk(token):
    """Akkadian翻字トークンを正規化（ハイフン除去、小文字化）"""
    return token.replace("-", "").lower()

def normalize_eng(name):
    """英語名を正規化（小文字化）"""
    return name.lower().replace("-", "")

def phonetic_similarity(akk_token, eng_name):
    """Akkadian翻字と英語名の音韻的類似度（先頭一致）"""
    akk_norm = normalize_akk(akk_token)
    eng_norm = normalize_eng(eng_name)
    # 先頭からの最長一致
    match_len = 0
    for a, e in zip(akk_norm, eng_norm):
        # 音韻的に等価な文字も考慮
        equiv = {
            ('š', 's'), ('s', 'š'), ('ṣ', 's'), ('s', 'ṣ'),
            ('ṭ', 't'), ('t', 'ṭ'),
            ('ḫ', 'h'), ('h', 'ḫ'),
            ('ú', 'u'), ('u', 'ú'),
            ('á', 'a'), ('a', 'á'),
            ('í', 'i'), ('i', 'í'),
            ('é', 'e'), ('e', 'é'),
        }
        if a == e or (a, e) in equiv:
            match_len += 1
        else:
            break
    # 正規化: 短い方の長さで割る
    min_len = min(len(akk_norm), len(eng_norm))
    if min_len == 0:
        return 0.0
    return match_len / min_len

# 全PNトークンについて、翻訳中の最もマッチする英語名を探す
pn_eng_mapping = {}  # akk_pn → [(eng_name, similarity, count)]
pn_eng_best = {}     # akk_pn → best_eng_name

for r in pn_rows:
    eng_names = extract_proper_names_from_translation(r["translation"])
    for pn_token in r["pn_tokens"]:
        if not eng_names:
            continue
        # 各英語名との類似度を計算
        best_sim = 0
        best_name = None
        for name in eng_names:
            sim = phonetic_similarity(pn_token, name)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_name and best_sim > 0.3:  # 閾値
            if pn_token not in pn_eng_mapping:
                pn_eng_mapping[pn_token] = defaultdict(lambda: {"count": 0, "sim": 0})
            pn_eng_mapping[pn_token][best_name]["count"] += 1
            pn_eng_mapping[pn_token][best_name]["sim"] = best_sim

# マッピング結果の集計
print(f"PN tokens with phonetic match (sim>0.3): {len(pn_eng_mapping)}/{len(pn_freq)}")

# 安定なマッピング（最頻出の英語名が全体の80%以上を占める）
stable_mappings = {}
unstable_mappings = {}
for akk_pn, eng_dict in pn_eng_mapping.items():
    total = sum(v["count"] for v in eng_dict.values())
    best_eng = max(eng_dict.keys(), key=lambda k: eng_dict[k]["count"])
    best_count = eng_dict[best_eng]["count"]
    ratio = best_count / total
    if ratio >= 0.8 and total >= 2:
        stable_mappings[akk_pn] = {
            "eng": best_eng,
            "count": total,
            "ratio": ratio,
            "sim": eng_dict[best_eng]["sim"],
        }
    else:
        unstable_mappings[akk_pn] = {
            "candidates": {k: v["count"] for k, v in eng_dict.items()},
            "total": total,
        }

print(f"\nStable mappings (≥80% consistency, ≥2 occurrences): {len(stable_mappings)}")
print(f"Unstable mappings: {len(unstable_mappings)}")

# トークンカバレッジ
stable_token_coverage = sum(pn_freq[pn] for pn in stable_mappings if pn in pn_freq)
total_pn_occurrences = sum(pn_freq.values())
print(f"Token coverage: {stable_token_coverage}/{total_pn_occurrences} ({stable_token_coverage/total_pn_occurrences*100:.1f}%)")

# Top安定マッピング例
print(f"\nTop 30 stable PN→English mappings:")
for akk_pn, info in sorted(stable_mappings.items(), key=lambda x: -x[1]["count"])[:30]:
    print(f"  {akk_pn:30s} → {info['eng']:25s} (n={info['count']}, sim={info['sim']:.2f})")

# 不安定マッピング例
print(f"\nTop 10 unstable mappings:")
for akk_pn, info in sorted(unstable_mappings.items(), key=lambda x: -x[1]["total"])[:10]:
    print(f"  {akk_pn}: {info['candidates']}")

# ============================================================
# 5. GNについても同様の分析
# ============================================================
print("\n" + "=" * 60)
print("5. GNトークンの分析")
print("=" * 60)

gn_rows = []
for idx, row in train_df.iterrows():
    translit = str(row["transliteration"])
    tokens = translit.split()
    gn_tokens_in_row = []
    for token in tokens:
        clean_token = token.strip()
        if clean_token in gn_forms:
            gn_tokens_in_row.append(clean_token)
    if gn_tokens_in_row:
        gn_rows.append({
            "transliteration": translit,
            "translation": str(row["translation"]),
            "gn_tokens": gn_tokens_in_row,
        })

all_gn_tokens = []
for r in gn_rows:
    all_gn_tokens.extend(r["gn_tokens"])
gn_freq = Counter(all_gn_tokens)
print(f"Rows with GN tokens: {len(gn_rows)}/{len(train_df)}")
print(f"Total GN occurrences: {len(all_gn_tokens)}")
print(f"Unique GN tokens: {len(gn_freq)}")
print(f"\nTop 20 GN tokens:")
for token, count in gn_freq.most_common(20):
    print(f"  {token}: {count}")

# ============================================================
# 6. プレースホルダ置換のシミュレーション
# ============================================================
print("\n" + "=" * 60)
print("6. プレースホルダ置換シミュレーション")
print("=" * 60)

# 安定マッピングを使って、実際にプレースホルダ置換した場合の効果を推定
# (a) 入力長の削減効果
# (b) 翻訳中の固有名詞がカバーされる割合

import numpy as np

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def clean_transliteration(text):
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = re.sub(r'\(ki\)', '{ki}', text)
    return text

# プレースホルダ置換のシミュレーション
replaced_count = 0
total_docs = 0
bytes_saved_list = []
replacement_examples = []

for idx, row in train_df.iterrows():
    translit = clean_transliteration(str(row["transliteration"]))
    tokens = translit.split()
    new_tokens = []
    pn_counter = 0
    replacements = []
    for token in tokens:
        if token in stable_mappings:
            pn_counter += 1
            placeholder = f"<PN{pn_counter}>"
            new_tokens.append(placeholder)
            replacements.append((token, stable_mappings[token]["eng"], placeholder))
        elif token in gn_forms:
            pn_counter += 1
            placeholder = f"<GN{pn_counter}>"
            new_tokens.append(placeholder)
            replacements.append((token, "?", placeholder))
        else:
            new_tokens.append(token)

    if replacements:
        replaced_count += 1
        orig_bytes = len(translit.encode("utf-8"))
        new_text = " ".join(new_tokens)
        new_bytes = len(new_text.encode("utf-8"))
        bytes_saved_list.append(orig_bytes - new_bytes)
        if len(replacement_examples) < 5:
            replacement_examples.append({
                "original": translit[:150],
                "replaced": new_text[:150],
                "mappings": replacements[:5],
                "bytes_saved": orig_bytes - new_bytes,
            })
    total_docs += 1

print(f"Docs with replacements: {replaced_count}/{total_docs} ({replaced_count/total_docs*100:.1f}%)")
if bytes_saved_list:
    print(f"Bytes saved per doc (when replaced): mean={np.mean(bytes_saved_list):.1f}, median={np.median(bytes_saved_list):.1f}, max={np.max(bytes_saved_list)}")

print(f"\nReplacement examples:")
for i, ex in enumerate(replacement_examples):
    print(f"\n  Example {i+1}:")
    print(f"    Original:  {ex['original']}")
    print(f"    Replaced:  {ex['replaced']}")
    print(f"    Mappings:  {ex['mappings'][:3]}")
    print(f"    Saved:     {ex['bytes_saved']}B")

# ============================================================
# 7. 逆方向: 翻訳中の英語名 → プレースホルダ置換の課題
# ============================================================
print("\n" + "=" * 60)
print("7. 翻訳中の英語名 → プレースホルダ置換の課題")
print("=" * 60)

# 学習時に翻訳側もプレースホルダに置換する必要がある
# 英語名の出現位置を特定し、対応するPN番号で置換できるか

# 安定マッピングの逆引き辞書
eng_to_akk = defaultdict(list)
for akk_pn, info in stable_mappings.items():
    eng_to_akk[info["eng"]].append(akk_pn)

print(f"Unique English names in stable mappings: {len(eng_to_akk)}")
ambiguous_eng = {k: v for k, v in eng_to_akk.items() if len(v) > 1}
print(f"English names mapping to multiple Akkadian forms: {len(ambiguous_eng)}")
if ambiguous_eng:
    print("Examples:")
    for eng, akks in list(ambiguous_eng.items())[:10]:
        print(f"  {eng} ← {akks}")

# 翻訳側の置換成功率
success_count = 0
partial_count = 0
fail_count = 0
failure_examples = []

for r in pn_rows[:500]:  # 先頭500行でサンプル
    transl = r["translation"]
    pn_tokens = r["pn_tokens"]
    # 各PNに対応する英語名を取得
    expected_eng_names = []
    for pn in pn_tokens:
        if pn in stable_mappings:
            expected_eng_names.append(stable_mappings[pn]["eng"])

    if not expected_eng_names:
        continue

    # 翻訳中にこれらの英語名が実際に出現するか
    found = 0
    for eng_name in expected_eng_names:
        # 翻訳テキスト中に英語名が出現するか（大文字小文字考慮、部分一致も含む）
        if eng_name in transl or eng_name.replace("-", " ") in transl:
            found += 1

    if found == len(expected_eng_names):
        success_count += 1
    elif found > 0:
        partial_count += 1
    else:
        fail_count += 1
        if len(failure_examples) < 5:
            failure_examples.append({
                "pn_tokens": pn_tokens[:3],
                "expected_eng": expected_eng_names[:3],
                "translation": transl[:150],
            })

total_checked = success_count + partial_count + fail_count
print(f"\nTranslation-side replacement feasibility (first 500 PN rows):")
print(f"  Full match:    {success_count}/{total_checked} ({success_count/total_checked*100:.1f}%)")
print(f"  Partial match: {partial_count}/{total_checked} ({partial_count/total_checked*100:.1f}%)")
print(f"  No match:      {fail_count}/{total_checked} ({fail_count/total_checked*100:.1f}%)")

if failure_examples:
    print(f"\nFailure examples:")
    for ex in failure_examples:
        print(f"  PN: {ex['pn_tokens']} → Expected: {ex['expected_eng']}")
        print(f"  Translation: {ex['translation']}")

# ============================================================
# 8. サマリー
# ============================================================
print("\n" + "=" * 60)
print("8. サマリーと実現可能性評価")
print("=" * 60)

print(f"""
=== PN/GNルールベース置換 実現可能性レポート ===

■ PN辞書構築
  - form_tag_dict: {len(pn_forms)} PN forms
  - trainデータ中のPN出現: {len(all_pn_tokens)} tokens / {len(pn_freq)} unique
  - 安定な Akk→Eng マッピング: {len(stable_mappings)} forms ({stable_token_coverage}/{total_pn_occurrences} = {stable_token_coverage/total_pn_occurrences*100:.1f}% of occurrences)
  - 不安定マッピング: {len(unstable_mappings)} forms

■ プレースホルダ置換効果
  - 対象ドキュメント: {replaced_count}/{total_docs} ({replaced_count/total_docs*100:.1f}%)
  - 入力バイト削減: mean={np.mean(bytes_saved_list):.1f}B (置換ありのdocのみ)

■ 翻訳側置換の課題
  - 完全対応: {success_count}/{total_checked} ({success_count/total_checked*100:.1f}%)
  - 部分対応: {partial_count}/{total_checked} ({partial_count/total_checked*100:.1f}%)
  - 不一致: {fail_count}/{total_checked} ({fail_count/total_checked*100:.1f}%)

■ 課題
  1. 翻訳側の英語名が辞書と異なる形で出現する場合がある
  2. 同じ英語名が複数のAkkadian形に対応: {len(ambiguous_eng)} 件
  3. GNの英語対応は未構築
  4. テストデータに未知のPN形が出現する可能性
""")

# 結果をファイルに保存
with open(RESULTS_DIR / "stable_pn_mappings.json", "w") as f:
    json.dump(stable_mappings, f, ensure_ascii=False, indent=2)
print(f"Stable PN mappings saved to {RESULTS_DIR / 'stable_pn_mappings.json'}")
