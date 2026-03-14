"""
trainのtranslation/transliterationの文字セットを調査
- 前処理前後でどの文字が残っているか
- 前処理で消えた/変換された文字は何か
"""
import re
import pandas as pd
from collections import Counter

PROJECT_ROOT = "/home/user/work/Akkadian"
train_df = pd.read_csv(f"{PROJECT_ROOT}/datasets/raw/train.csv")

# ============================================================
# exp023の前処理を再現
# ============================================================
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002

def _decimal_to_fraction(match):
    dec_str = match.group(0)
    try:
        value = float(dec_str)
    except ValueError:
        return dec_str
    int_part = int(value)
    frac_part = value - int_part
    if frac_part < 0.001:
        return dec_str
    best_frac = None
    best_dist = float('inf')
    for target, symbol in FRACTION_TARGETS.items():
        dist = abs(frac_part - target)
        if dist < best_dist:
            best_dist = dist
            best_frac = symbol
    if best_dist <= APPROX_TOLERANCE:
        return best_frac if int_part == 0 else f"{int_part} {best_frac}"
    return dec_str

ROMAN_TO_INT = {
    "XII": "12", "XI": "11", "VIII": "8", "VII": "7",
    "VI": "6", "IV": "4", "IX": "9", "III": "3",
    "II": "2", "X": "10", "V": "5", "I": "1",
}

MONTH_NAMES_TRANSLATION = {
    r"B[eē]lat[\s-]ekall[ie]m": "1",
    r"[Šš]a[\s-]sarr[aā]tim": "2",
    r"[Kk]en[aā]tim": "3",
    r"[Šš]a[\s-]k[eē]n[aā]tim": "3",
    r"Ma[hḫ]h?ur[\s-]il[iī]": "4",
    r"Ab[\s-]?[šš]arr[aā]ni": "5",
    r"[Aa]b[sš]arrani": "5",
    r"[Hh]ubur": "6",
    r"[Ṣṣ]ip['\u2019]?um": "7",
    r"[Qq]arr[aā]['\u2019]?[aā]tum": "8",
    r"[Qq]arr[aā]tum": "8",
    r"[Kk]an[wm]arta": "9",
    r"[Tt]e['\u2019\u02BE]?in[aā]tum": "10",
    r"[Tt][eē]['\u2019\u02BE]?in[aā]tum": "10",
    r"[Kk]uzall?[iu]m?": "11",
    r"[Aa]llan[aā]tum": "12",
}

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def clean_translation(text):
    if not isinstance(text, str) or not text.strip():
        return str(text)
    text = re.sub(r'\bfem\.\s*', '', text)
    text = re.sub(r'\bsing\.\s*', '', text)
    text = re.sub(r'\bpl\.\s*', '', text)
    text = re.sub(r'\bplural\b\s*', '', text)
    text = text.replace('(?)', '')
    text = re.sub(r'<<\s*>>', '', text)
    text = re.sub(r'<\s+>', '', text)
    text = re.sub(r'(?<!\.)\.\.(?!\.)', '', text)
    text = re.sub(r'\bxx?\b', '', text)
    text = re.sub(r'\bPN\b', '<gap>', text)
    text = re.sub(r'\b-gold\b', 'pašallum gold', text)
    text = re.sub(r'\b-tax\b', 'šadduātum tax', text)
    text = re.sub(r'\b-textiles\b', 'kutānum textiles', text)
    text = re.sub(r'(\S+)\s*/\s*\S+', r'\1', text)
    text = re.sub(r'\(m\)', '{m}', text)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    for roman, integer in sorted(ROMAN_TO_INT.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf'\bmonth\s+{roman}(?=[\s,.:;!?\)]|$)', f'month {integer}', text)
    for pattern, number in MONTH_NAMES_TRANSLATION.items():
        text = re.sub(rf'\bmonth\s+{pattern}\b', f'month {number}', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_transliteration(text):
    if not isinstance(text, str) or not text.strip():
        return str(text)
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text

# ============================================================
# 1. Translation: 前処理前 → 前処理後 の文字セット変化
# ============================================================
print("=" * 70)
print("1. Translation 文字セット（前処理前）")
print("=" * 70)

raw_trans = train_df["translation"].astype(str)
raw_chars = Counter()
for text in raw_trans:
    raw_chars.update(text)

print(f"ユニーク文字数: {len(raw_chars)}")
print(f"全文字:")
for ch, cnt in sorted(raw_chars.items(), key=lambda x: -x[1]):
    if ch in ' \n\r\t':
        label = repr(ch)
    else:
        label = ch
    print(f"  {label:4s}  U+{ord(ch):04X}  count={cnt}")

print()
print("=" * 70)
print("2. Translation 文字セット（前処理後）")
print("=" * 70)

cleaned_trans = raw_trans.apply(clean_translation)
cleaned_chars = Counter()
for text in cleaned_trans:
    cleaned_chars.update(text)

print(f"ユニーク文字数: {len(cleaned_chars)}")

# 前処理で消えた文字
disappeared = set(raw_chars.keys()) - set(cleaned_chars.keys())
print(f"\n前処理で消えた文字 ({len(disappeared)}個):")
for ch in sorted(disappeared, key=lambda c: -raw_chars[c]):
    print(f"  {repr(ch):6s}  U+{ord(ch):04X}  was count={raw_chars[ch]}")

# 前処理で新たに出現した文字
appeared = set(cleaned_chars.keys()) - set(raw_chars.keys())
print(f"\n前処理で新たに出現した文字 ({len(appeared)}個):")
for ch in sorted(appeared, key=lambda c: -cleaned_chars[c]):
    print(f"  {repr(ch):6s}  U+{ord(ch):04X}  count={cleaned_chars[ch]}")

# 前処理後の全文字
print(f"\n前処理後の全文字:")
for ch, cnt in sorted(cleaned_chars.items(), key=lambda x: -x[1]):
    if ch in ' \n\r\t':
        label = repr(ch)
    else:
        label = ch
    print(f"  {label:4s}  U+{ord(ch):04X}  count={cnt}")

# ============================================================
# 3. Transliteration: 前処理前 → 前処理後
# ============================================================
print()
print("=" * 70)
print("3. Transliteration 文字セット（前処理前）")
print("=" * 70)

raw_translit = train_df["transliteration"].astype(str)
raw_tl_chars = Counter()
for text in raw_translit:
    raw_tl_chars.update(text)

print(f"ユニーク文字数: {len(raw_tl_chars)}")

print()
print("=" * 70)
print("4. Transliteration 文字セット（前処理後）")
print("=" * 70)

cleaned_translit = raw_translit.apply(clean_transliteration)
cleaned_tl_chars = Counter()
for text in cleaned_translit:
    cleaned_tl_chars.update(text)

print(f"ユニーク文字数: {len(cleaned_tl_chars)}")

disappeared_tl = set(raw_tl_chars.keys()) - set(cleaned_tl_chars.keys())
print(f"\n前処理で消えた文字 ({len(disappeared_tl)}個):")
for ch in sorted(disappeared_tl, key=lambda c: -raw_tl_chars[c]):
    print(f"  {repr(ch):6s}  U+{ord(ch):04X}  was count={raw_tl_chars[ch]}")

appeared_tl = set(cleaned_tl_chars.keys()) - set(raw_tl_chars.keys())
print(f"\n前処理で新たに出現した文字 ({len(appeared_tl)}個):")
for ch in sorted(appeared_tl, key=lambda c: -cleaned_tl_chars[c]):
    print(f"  {repr(ch):6s}  U+{ord(ch):04X}  count={cleaned_tl_chars[ch]}")

print(f"\n前処理後の全文字:")
for ch, cnt in sorted(cleaned_tl_chars.items(), key=lambda x: -x[1]):
    if ch in ' \n\r\t':
        label = repr(ch)
    else:
        label = ch
    print(f"  {label:4s}  U+{ord(ch):04X}  count={cnt}")
