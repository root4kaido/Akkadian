"""
trainのtranslation/transliterationの文字セットと、
Host提供のテスト文字リストを照合する
"""
import pandas as pd
from collections import Counter

PROJECT_ROOT = "/home/user/work/Akkadian"
train_df = pd.read_csv(f"{PROJECT_ROOT}/datasets/raw/train.csv")

# Host提供のテスト文字リスト (665209 discussion)
TEST_TRANSLIT_CHARS = set(
    '!+-.0123456789:<>ABDEGHIKLMNPQRSTUWZ_abdeghiklmnpqrstuwz{}¼½'
    'ÀÁÈÉÌÍÙÚàáèéìíùúİışŠšṢṣṬṭ…⅓⅔⅙⅚'
)
# スペースと改行は暗黙的に許可
TEST_TRANSLIT_CHARS.update(' \n\r\t')

TEST_TRANSLATION_CHARS = set(
    "!\"\\\'()+,-.0123456789:;<>?ABCDEFGHIJKLMNOPQRSTUWYZ[]_"
    "abcdefghijklmnopqrstuvwxyz¼½àâāēğīışŠšūṢṣṬṭ–—\u2018\u2019\u201c\u201d⅓⅔⅙⅚"
)
TEST_TRANSLATION_CHARS.update(' \n\r\t')

print("=" * 70)
print("1. Train TRANSLATION vs Test Translation charset")
print("=" * 70)

train_trans = train_df["translation"].astype(str)
train_trans_chars = Counter()
for text in train_trans:
    train_trans_chars.update(text)

# trainにあってtestにない文字
train_only_trans = {}
for ch, cnt in train_trans_chars.most_common():
    if ch not in TEST_TRANSLATION_CHARS:
        train_only_trans[ch] = cnt

print(f"\nTrain translation unique chars: {len(train_trans_chars)}")
print(f"Test translation charset size: {len(TEST_TRANSLATION_CHARS)}")
print(f"Train-only chars (not in test): {len(train_only_trans)}")
print()

if train_only_trans:
    print("Char | Unicode | Count | Example context")
    print("-----|---------|-------|----------------")
    for ch, cnt in sorted(train_only_trans.items(), key=lambda x: -x[1]):
        # Find an example
        example = ""
        for text in train_trans:
            idx = text.find(ch)
            if idx >= 0:
                start = max(0, idx - 15)
                end = min(len(text), idx + 15)
                example = text[start:end].replace('\n', ' ')
                break
        print(f"  {repr(ch):6s} | U+{ord(ch):04X}  | {cnt:6d} | ...{example}...")

print()

print("=" * 70)
print("2. Train TRANSLITERATION vs Test Transliteration charset")
print("=" * 70)

train_translit = train_df["transliteration"].astype(str)
train_translit_chars = Counter()
for text in train_translit:
    train_translit_chars.update(text)

train_only_translit = {}
for ch, cnt in train_translit_chars.most_common():
    if ch not in TEST_TRANSLIT_CHARS:
        train_only_translit[ch] = cnt

print(f"\nTrain transliteration unique chars: {len(train_translit_chars)}")
print(f"Test transliteration charset size: {len(TEST_TRANSLIT_CHARS)}")
print(f"Train-only chars (not in test): {len(train_only_translit)}")
print()

if train_only_translit:
    print("Char | Unicode | Count | Example context")
    print("-----|---------|-------|----------------")
    for ch, cnt in sorted(train_only_translit.items(), key=lambda x: -x[1]):
        example = ""
        for text in train_translit:
            idx = text.find(ch)
            if idx >= 0:
                start = max(0, idx - 15)
                end = min(len(text), idx + 15)
                example = text[start:end].replace('\n', ' ')
                break
        print(f"  {repr(ch):6s} | U+{ord(ch):04X}  | {cnt:6d} | ...{example}...")

print()

print("=" * 70)
print("3. テストにあってtrainにない文字")
print("=" * 70)

test_only_trans = TEST_TRANSLATION_CHARS - set(train_trans_chars.keys()) - {' ', '\n', '\r', '\t'}
test_only_translit = TEST_TRANSLIT_CHARS - set(train_translit_chars.keys()) - {' ', '\n', '\r', '\t'}

print(f"\nTest-only translation chars: {sorted([repr(c) for c in test_only_trans])}")
print(f"Test-only transliteration chars: {sorted([repr(c) for c in test_only_translit])}")

print()
print("=" * 70)
print("4. exp023前処理後のtranslation文字セット確認")
print("=" * 70)

# exp023の前処理を簡易的に適用してから再チェック
import re

FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}

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
    if best_dist <= 0.002:
        return best_frac if int_part == 0 else f"{int_part} {best_frac}"
    return dec_str

def clean_translation_simple(text):
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r'\bfem\.\s*', '', text)
    text = re.sub(r'\bsing\.\s*', '', text)
    text = re.sub(r'\bpl\.\s*', '', text)
    text = re.sub(r'\bplural\b\s*', '', text)
    text = text.replace('(?)', '')
    text = re.sub(r'<<\s*>>', '', text)
    text = re.sub(r'\bxx?\b', '', text)
    text = re.sub(r'\bPN\b', '<gap>', text)
    text = re.sub(r'\b-gold\b', 'pašallum gold', text)
    text = re.sub(r'\b-tax\b', 'šadduātum tax', text)
    text = re.sub(r'\b-textiles\b', 'kutānum textiles', text)
    text = re.sub(r'(\S+)\s*/\s*\S+', r'\1', text)
    text = re.sub(r'\(m\)', '{m}', text)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_trans = train_df["translation"].astype(str).apply(clean_translation_simple)
cleaned_chars = Counter()
for text in cleaned_trans:
    cleaned_chars.update(text)

cleaned_only = {}
for ch, cnt in cleaned_chars.most_common():
    if ch not in TEST_TRANSLATION_CHARS:
        cleaned_only[ch] = cnt

print(f"\nAfter exp023 preprocessing:")
print(f"Cleaned translation unique chars: {len(cleaned_chars)}")
print(f"Still outside test charset: {len(cleaned_only)}")
print()

if cleaned_only:
    print("Char | Unicode | Count | Example context")
    print("-----|---------|-------|----------------")
    for ch, cnt in sorted(cleaned_only.items(), key=lambda x: -x[1]):
        example = ""
        for text in cleaned_trans:
            idx = text.find(ch)
            if idx >= 0:
                start = max(0, idx - 15)
                end = min(len(text), idx + 15)
                example = text[start:end].replace('\n', ' ')
                break
        print(f"  {repr(ch):6s} | U+{ord(ch):04X}  | {cnt:6d} | ...{example}...")
