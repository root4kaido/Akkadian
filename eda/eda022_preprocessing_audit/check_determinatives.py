import pandas as pd
import re
from collections import Counter

train = pd.read_csv("/home/user/work/Akkadian/datasets/raw/train.csv")
translit = train["transliteration"].astype(str)
translation = train["translation"].astype(str)

print("=== determinative出現数 ===")
for marker in ["{d}", "{m}", "{ki}", "{f}"]:
    rows = translit.str.contains(re.escape(marker), regex=True).sum()
    total = translit.str.count(re.escape(marker)).sum()
    print(f"  translit {marker}: {rows} rows, {total} occurrences")

print()

# {d}の後に続くトークン
print("=== {d} + 後続トークン TOP20 ===")
d_pattern = re.compile(r"\{d\}(\S+)")
d_names = Counter()
for text in translit:
    for m in d_pattern.finditer(text):
        d_names[m.group(1)] += 1
for name, cnt in d_names.most_common(20):
    print(f"  {{d}}{name}: {cnt}")
print(f"  unique: {len(d_names)}")

print()
print("=== {ki} + 前方トークン TOP20 ===")
ki_pattern = re.compile(r"(\S+)\{ki\}")
ki_names = Counter()
for text in translit:
    for m in ki_pattern.finditer(text):
        ki_names[m.group(1)] += 1
for name, cnt in ki_names.most_common(20):
    print(f"  {name}{{ki}}: {cnt}")
print(f"  unique: {len(ki_names)}")

# OA_Lexiconとのマッチ
print()
print("=== OA_Lexicon PNとの照合 ===")
lexicon = pd.read_csv("/home/user/work/Akkadian/datasets/raw/OA_Lexicon_eBL.csv")
pn_entries = lexicon[lexicon["type"] == "PN"]
gn_entries = lexicon[lexicon["type"] == "GN"]
print(f"OA_Lexicon PN entries: {len(pn_entries)}")
print(f"OA_Lexicon GN entries: {len(gn_entries)}")

# {d}付き名前がOA_Lexicon PNのformに対応するか
d_name_set = set(d_names.keys())
pn_form_set = set(pn_entries["form"].dropna().unique())
print(f"\n{{d}}付きユニーク名: {len(d_name_set)}")

d_in_lexicon = 0
d_not_in_lexicon = []
for name in d_names:
    full = "{{d}}" + name
    if full in pn_form_set or name in pn_form_set:
        d_in_lexicon += 1
    else:
        d_not_in_lexicon.append((name, d_names[name]))
print(f"OA_Lexicon PNにヒット: {d_in_lexicon}/{len(d_name_set)}")
print(f"ヒットしないTOP10:")
for name, cnt in sorted(d_not_in_lexicon, key=lambda x: -x[1])[:10]:
    print(f"  {{d}}{name}: {cnt}")

# translationでの対応する名前を確認
print()
print("=== {d}付き名前のtranslation側対応（サンプル） ===")
d_rows = train[translit.str.contains(r"\{d\}", regex=True)]
for _, row in d_rows.head(10).iterrows():
    tl = row["transliteration"]
    tr = row["translation"]
    d_tokens = re.findall(r"\{d\}(\S+)", tl)
    print(f"  translit {d_tokens} => translation: {tr[:120]}")

# {ki}についても同様
print()
print("=== {ki}付き名前のtranslation側対応（サンプル） ===")
ki_rows = train[translit.str.contains(r"\{ki\}", regex=True)]
for _, row in ki_rows.head(10).iterrows():
    tl = row["transliteration"]
    tr = row["translation"]
    ki_tokens = re.findall(r"(\S+)\{ki\}", tl)
    print(f"  translit {ki_tokens} => translation: {tr[:120]}")

# onomasticonとの照合
print()
print("=== onomasticonとの照合 ===")
ono = pd.read_csv("/home/user/work/Akkadian/datasets/raw/onomasticon/onomasticon.csv")
print(f"onomasticon entries: {len(ono)}")
# Spellings列を展開
ono_spellings = set()
for spells in ono["Spellings_semicolon_separated"].dropna():
    for s in spells.split(";"):
        ono_spellings.add(s.strip())
print(f"onomasticon unique spellings: {len(ono_spellings)}")

# {d}名がonomasticonのspellingsにマッチするか
d_in_ono = 0
for name in d_names:
    full = "{d}" + name
    if full in ono_spellings or name in ono_spellings:
        d_in_ono += 1
print(f"{{d}}名がonomasticonにヒット: {d_in_ono}/{len(d_name_set)}")
