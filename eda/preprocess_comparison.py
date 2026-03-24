"""Compare our preprocessing vs notebook preprocessing on train data."""
import re, math, sys
import pandas as pd

# ============================================================
# Our preprocessing (from eval_ensemble_rt.py / exp023)
# ============================================================
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.02

def _decimal_to_fraction_approx(m):
    dec_str = m.group(0)
    try:
        val = float(dec_str)
    except ValueError:
        return dec_str
    int_part = int(val)
    frac_part = val - int_part
    if frac_part < 1e-6:
        return str(int_part) if int_part > 0 else dec_str
    best_frac, best_dist = None, float('inf')
    for target_val, symbol in FRACTION_TARGETS.items():
        dist = abs(frac_part - target_val)
        if dist < best_dist:
            best_dist, best_frac = dist, symbol
    if best_dist <= APPROX_TOLERANCE:
        return best_frac if int_part == 0 else f"{int_part} {best_frac}"
    return dec_str

def our_preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction_approx, text)
    return text

# ============================================================
# Notebook preprocessing (from lb-35-9 notebook)
# ============================================================
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})

def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s

_GAP_UNIFIED_RE = re.compile(
    r"\.\.\."
    r"|\[\s*\.\.\.\s*\]"
    r"|\[\.+\]"
    r"|\[[^\]]*(?:broken|missing|illegible|damaged|effaced|erased|lost|traces)[^\]]*\]"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I
)

_CHAR_TRANS = str.maketrans({
    "ḫ":"h","Ḫ":"H","ʾ":"",
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
    "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9",
    "—":"-","–":"-",
})
_SUB_X = "ₓ"

_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")
_PN_RE = re.compile(r"\bPN\b")
_KUBABBAR_RE = re.compile(r"KÙ\.B\.")

_ALLOWED_FRACS = [
    (1/6, "0.16666"), (1/4, "0.25"), (1/3, "0.33333"),
    (1/2, "0.5"), (2/3, "0.66666"), (3/4, "0.75"), (5/6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")

def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    for target, label in _ALLOWED_FRACS:
        if abs(frac - target) < _FRAC_TOL:
            return label if ip == 0 else f"{ip} {label}"
    return f"{ip}.{str(round(frac, 4)).lstrip('0.')}" if ip else f"0.{str(round(frac, 4)).lstrip('0.')}"

_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚", "0.6666": "⅔", "0.3333": "⅓", "0.1666": "⅙",
    "0.625": "⅝", "0.75": "¾", "0.25": "¼", "0.5": "½",
}
def _frac_repl(m):
    return _EXACT_FRAC_MAP[m.group(0)]

_WS_RE = re.compile(r"\s+")

def notebook_preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    s = text
    # 1. ASCII to diacritics (no-op for our data, but included for completeness)
    s = _ascii_to_diacritics(s)
    # 2. Determinatives
    s = re.sub(_DET_UPPER_RE, r"\1", s)
    s = re.sub(_DET_LOWER_RE, r"{\1}", s)
    # 3. Gap normalization
    s = _GAP_UNIFIED_RE.sub("<gap>", s)
    # 4. Character translation (ḫ→h, ʾ→'', subscripts→digits, dashes)
    s = s.translate(_CHAR_TRANS)
    s = s.replace(_SUB_X, "")
    # 5. KÙ.B. → KÙ.BABBAR
    s = _KUBABBAR_RE.sub("KÙ.BABBAR", s)
    # 6. Fraction normalization (exact match)
    s = _EXACT_FRAC_RE.sub(_frac_repl, s)
    # 7. Float normalization (long decimals)
    s = _FLOAT_RE.sub(lambda m: _canon_decimal(float(m.group(1))), s)
    # 8. Whitespace cleanup
    s = _WS_RE.sub(" ", s).strip()
    return s


# ============================================================
# Run comparison
# ============================================================
df = pd.read_csv("/home/user/work/Akkadian/datasets/raw/train.csv")

# Also check sentence-level alignment data
import json
alignment_path = "/home/user/work/Akkadian/datasets/processed/aligned_all.json"
try:
    with open(alignment_path) as f:
        alignment = json.load(f)
    akk_segments = []
    for oare_id, segs in alignment.items():
        for seg in segs:
            akk_segments.append(seg['akk_segment'])
    print(f"\n=== Sentence-level segments: {len(akk_segments)} ===")
except FileNotFoundError:
    akk_segments = []
    print("\nNo alignment file found")

# Compare on doc-level transliterations
print("=" * 60)
print("DOC-LEVEL COMPARISON (train.csv transliteration)")
print("=" * 60)

diff_count = 0
diffs = []
for idx, row in df.iterrows():
    text = str(row['transliteration'])
    ours = our_preprocess(text)
    theirs = notebook_preprocess(text)
    if ours != theirs:
        diff_count += 1
        diffs.append({
            'idx': idx,
            'oare_id': row.get('oare_id', ''),
            'original': text[:100],
            'ours': ours[:100],
            'theirs': theirs[:100],
        })

print(f"\nTotal rows: {len(df)}")
print(f"Rows with differences: {diff_count}")
print(f"Rows identical: {len(df) - diff_count}")

if diffs:
    print(f"\n--- First 20 differences ---")
    for d in diffs[:20]:
        print(f"\n[Row {d['idx']}] oare_id={d['oare_id']}")
        print(f"  ORIG:     {d['original']}")
        print(f"  OURS:     {d['ours']}")
        print(f"  NOTEBOOK: {d['theirs']}")

# Categorize differences
print("\n\n--- Difference categories ---")
cat_counts = {
    'gap_norm': 0,
    'det_upper': 0,
    'det_lower': 0,
    'dash_norm': 0,
    'aleph_removal': 0,
    'kubabbar': 0,
    'fraction': 0,
    'whitespace': 0,
    'other': 0,
}

for idx, row in df.iterrows():
    text = str(row['transliteration'])
    ours = our_preprocess(text)
    theirs = notebook_preprocess(text)
    if ours == theirs:
        continue

    # Check each category
    # Gap: notebook converts [...], ..., x, (break) etc to <gap>
    if '<gap>' in theirs and '<gap>' not in ours:
        cat_counts['gap_norm'] += 1

    # Det upper: notebook removes parens around upper determinatives
    if re.search(r'\([A-ZŠṬṢḪ]', text):
        t1 = our_preprocess(text)
        t2 = re.sub(_DET_UPPER_RE, r"\1", t1)
        if t1 != t2:
            cat_counts['det_upper'] += 1

    # Det lower: notebook converts (ki) etc
    # We do (ki) -> {ki}, notebook does general (lowercase) -> {lowercase}
    if re.search(r'\([a-z]', text):
        cat_counts['det_lower'] += 1

    # Dash normalization
    if '—' in text or '–' in text:
        cat_counts['dash_norm'] += 1

    # Aleph removal
    if 'ʾ' in text:
        cat_counts['aleph_removal'] += 1

    # KÙ.B.
    if 'KÙ.B.' in text:
        cat_counts['kubabbar'] += 1

    # Fraction
    if re.search(r'\d+\.\d+', text):
        cat_counts['fraction'] += 1

for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    if count > 0:
        print(f"  {cat}: {count}")

# Segment level too
if akk_segments:
    print("\n\n" + "=" * 60)
    print("SENTENCE-LEVEL COMPARISON (aligned segments)")
    print("=" * 60)
    seg_diff = 0
    seg_diffs = []
    for seg in akk_segments:
        ours = our_preprocess(seg)
        theirs = notebook_preprocess(seg)
        if ours != theirs:
            seg_diff += 1
            if len(seg_diffs) < 10:
                seg_diffs.append((seg[:80], ours[:80], theirs[:80]))

    print(f"Total segments: {len(akk_segments)}")
    print(f"Segments with differences: {seg_diff}")

    if seg_diffs:
        print(f"\n--- First 10 differences ---")
        for orig, o, t in seg_diffs:
            print(f"  ORIG:     {orig}")
            print(f"  OURS:     {o}")
            print(f"  NOTEBOOK: {t}")
            print()
