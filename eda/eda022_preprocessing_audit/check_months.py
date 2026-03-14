import pandas as pd
import re

df = pd.read_csv("datasets/raw/train.csv")
t = df["translation"].astype(str)
tr = df["transliteration"].astype(str)

# Old Assyrian month names (from Host table)
MONTH_NAMES = {
    "Bēlat ekallem": 1, "Bēlat-ekallim": 1, "be-el-tí-É.GAL-lim": 1,
    "ša-sarratim": 2, "ša sá-ra-tim": 2,
    "Kenātim": 3, "ša kēnātim": 3, "ke-na-tim": 3,
    "Mahur-ilī": 4, "Ma-hu-ur-DINGIR": 4, "ma-ḫu-ur-ì-lí": 4,
    "Abšarrani": 5, "ab šarrāni": 5, "abšarrani": 5, "áb-ša-ra-ni": 5, "áb ša-ra-ni": 5,
    "Hubur": 6, "Hu-bu-ur": 6,
    "Ṣip'um": 7, "ṣipum": 7, "ṣí-ip-im": 7,
    "Qarrātum": 8, "qá-ra-a-tí": 8, "qá-ra-a-tim": 8,
    "Kanwarta": 9, "Kanmarta": 9, "kán-bar-ta": 9,
    "Te'inātum": 10, "té-i-na-tim": 10,
    "Kuzallum": 11, "ku-zal-li": 11, "ku-zal-lu": 11,
    "Allanātum": 12, "a-lá-na-tum": 12, "a-lá-na-tim": 12,
}

print("=== Translation内の月名検索 ===")
for name, num in sorted(set((v, k) for k, v in MONTH_NAMES.items()), key=lambda x: x[0]):
    pass

# Search by month number groups
for month_num in range(1, 13):
    names = [k for k, v in MONTH_NAMES.items() if v == month_num]
    total = 0
    for name in names:
        cnt = t.str.contains(re.escape(name), case=False).sum()
        if cnt > 0:
            total += cnt
            print(f"  Month {month_num} '{name}': {cnt} (translation)")
    # Also check transliteration
    for name in names:
        cnt = tr.str.contains(re.escape(name), case=False).sum()
        if cnt > 0:
            print(f"  Month {month_num} '{name}': {cnt} (transliteration)")

print()

# Check "month" keyword context
print("=== 'month' を含む翻訳の全パターン ===")
month_rows = t[t.str.contains(r'\bmonth\b', case=False)]
print(f"Total rows with 'month': {len(month_rows)}")

# Extract "month X" patterns
month_patterns = t.str.extractall(r'month\s+(\S+(?:\s+\S+)?)', flags=re.IGNORECASE)
if len(month_patterns) > 0:
    print(f"\n'month X' patterns (top 30):")
    print(month_patterns[0].value_counts().head(30).to_string())

print()

# Check numeric month patterns already present
print("=== 数字月パターン ===")
for i in range(1, 13):
    cnt = t.str.contains(rf'\bmonth\s+{i}\b').sum()
    if cnt > 0:
        print(f"  'month {i}': {cnt}")

print()

# Check Roman numeral months still present
print("=== ローマ数字月パターン ===")
romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
for r in romans:
    cnt = t.str.contains(rf'\bmonth\s+{r}\b').sum()
    if cnt > 0:
        print(f"  'month {r}': {cnt}")

print()

# Show example rows with "month" to understand context
print("=== 'month' を含む行のサンプル ===")
for idx, row in month_rows.head(10).items():
    # Find the "month" context
    matches = re.findall(r'.{0,20}month\s+\S+.{0,30}', row, re.IGNORECASE)
    for m in matches:
        print(f"  [{idx}] ...{m.strip()}...")
