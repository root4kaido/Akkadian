import pandas as pd, re, unicodedata

df = pd.read_csv("datasets/raw/train.csv")
t = df["translation"].astype(str)
tr = df["transliteration"].astype(str)

# stray ?
has_q = t.str.contains(r'\?').sum()
paren_q = t.str.contains(r'\(\?\)').sum()
print(f"? 含む行: {has_q}")
print(f"(?) 含む行: {paren_q}")
print(f"(?以外の? : {has_q - paren_q}")
# show examples of ? that are not (?)
mask = t.str.contains(r'\?') & ~t.str.contains(r'\(\?\)')
examples = t[mask].head(5)
for i, ex in examples.items():
    snippet = ex[:150]
    print(f"  [{i}] {snippet}")
print()

# 小数丸め誤差
all_text = pd.concat([t, tr])
decs = all_text.str.extractall(r'(\d+\.\d+)')[0]
frac = decs.apply(lambda x: round(float(x) % 1, 4))
known = {0.5, 0.25, 0.3333, 0.6666, 0.8333, 0.75, 0.1666, 0.625, 0.0}
unknown = frac[~frac.isin(known)]
print(f"既知パターン: {frac[frac.isin(known)].shape[0]}")
print(f"未知パターン: {unknown.shape[0]}")
print(unknown.value_counts().head(15).to_string())
print()

# shekel fractions in different formats
print("--- shekel関連 ---")
print(f"N/12: {t.str.contains(r'\\d+/12').sum()}")
print(f"grains: {t.str.contains(r'grain').sum()}")
print(f"shekel: {t.str.contains(r'shekel').sum()}")

# KU.B. variants
print()
print("--- KÙ.B. variants ---")
for pat in ["KÙ.B.", "KÙ.BABBAR", "kù.b.", "KU.B.", "KU.BABBAR"]:
    cnt = tr.str.contains(re.escape(pat)).sum()
    print(f"  {pat}: {cnt}")
# Check if KÙ.B. without BABBAR exists
kub_not_babbar = tr.str.contains(r'KÙ\.B\.(?!ABBAR)').sum()
print(f"  KÙ.B.(BABBARなし): {kub_not_babbar}")
