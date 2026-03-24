"""Analyze Akkadian transliteration format: ASCII vs Unicode diacritics."""
import csv
import re
from collections import Counter

csv.field_size_limit(10_000_000)

with open("/home/user/work/Akkadian/datasets/raw/train.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")
translit = [r["transliteration"] for r in rows]

# --- Unicode diacritics ---
uni_shin = sum(1 for t in translit if "š" in t or "Š" in t)
uni_s_dot = sum(1 for t in translit if "ṣ" in t or "Ṣ" in t)
uni_t_dot = sum(1 for t in translit if "ṭ" in t or "Ṭ" in t)
uni_h_arc = sum(1 for t in translit if "ḫ" in t or "Ḫ" in t)

# Accented vowels (acute = index 2, grave = index 3 in ATF)
uni_acute = sum(1 for t in translit if re.search(r"[áéíú]", t))
uni_grave = sum(1 for t in translit if re.search(r"[àèìù]", t))

# --- ASCII equivalents ---
ascii_sz = sum(1 for t in translit if "sz" in t.lower())
ascii_s_comma = sum(1 for t in translit if "s," in t or "S," in t)
ascii_t_comma = sum(1 for t in translit if "t," in t or "T," in t)

# vowel+number patterns (careful: only match transliteration-style, not subscripts)
ascii_v2 = sum(1 for t in translit if re.search(r"[aeiuAEIU]2(?!\d)", t))
ascii_v3 = sum(1 for t in translit if re.search(r"[aeiuAEIU]3(?!\d)", t))

# h with ASCII variants
ascii_h_variants = sum(1 for t in translit if re.search(r"h,|ḫ", t, re.IGNORECASE) == None and "h" in t.lower())

print("\n=== Unicode Diacritics ===")
print(f"  š (shin):       {uni_shin} rows")
print(f"  ṣ (s-dot):      {uni_s_dot} rows")
print(f"  ṭ (t-dot):      {uni_t_dot} rows")
print(f"  ḫ (h-arc):      {uni_h_arc} rows")
print(f"  Acute (áéíú):   {uni_acute} rows")
print(f"  Grave (àèìù):   {uni_grave} rows")

print("\n=== ASCII Equivalents ===")
print(f"  'sz':            {ascii_sz} rows")
print(f"  's,' or 'S,':    {ascii_s_comma} rows")
print(f"  't,' or 'T,':    {ascii_t_comma} rows")
print(f"  vowel+2:         {ascii_v2} rows")
print(f"  vowel+3:         {ascii_v3} rows")

# --- Subscript numbers (e.g. tur₄, qé) ---
uni_subscripts = sum(1 for t in translit if re.search(r"[₀₁₂₃₄₅₆₇₈₉]", t))
ascii_subscripts = sum(1 for t in translit if re.search(r"[a-zA-Z][0-9]", t))  # rough

print("\n=== Subscript digits ===")
print(f"  Unicode subscripts (₀-₉): {uni_subscripts} rows")
print(f"  ASCII digit after letter:  {ascii_subscripts} rows (rough, includes false positives)")

# --- Show examples ---
print("\n=== Sample rows with Unicode diacritics ===")
count = 0
for t in translit[:200]:
    if re.search(r"[šṣṭḫáéíúàèìù]", t):
        print(f"  {t[:150]}")
        count += 1
        if count >= 5:
            break

print("\n=== Sample rows with vowel+number (if any) ===")
count = 0
for t in translit:
    if re.search(r"[aeiuAEIU][23](?!\d)", t):
        print(f"  {t[:150]}")
        count += 1
        if count >= 5:
            break
if count == 0:
    print("  (none found)")

# --- Check for any 'sz' examples ---
print("\n=== Sample rows with 'sz' (if any) ===")
count = 0
for t in translit:
    if "sz" in t.lower():
        print(f"  {t[:150]}")
        count += 1
        if count >= 5:
            break
if count == 0:
    print("  (none found)")

# --- Unique special characters ---
all_chars = set()
for t in translit:
    all_chars.update(set(t))
special = sorted(c for c in all_chars if ord(c) > 127)
print(f"\n=== All non-ASCII characters in transliterations ({len(special)} unique) ===")
for c in special:
    freq = sum(1 for t in translit if c in t)
    print(f"  U+{ord(c):04X}  {c}  appears in {freq} rows")

print("\nDone.")
