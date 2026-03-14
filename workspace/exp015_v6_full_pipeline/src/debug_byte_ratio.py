"""Akkadian vs English のバイト比率を分析し、適切な参照長を検討する"""
import os, re, json
import pandas as pd, numpy as np, yaml
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))

with open(os.path.join(EXP_DIR, "config.yaml")) as f:
    config = yaml.safe_load(f)

# Val split
df = pd.read_csv(os.path.join(EXP_DIR, config["data"]["train_path"]))
df = df[(df["transliteration"].astype(str).str.len() > 0) & (df["translation"].astype(str).str.len() > 0)]
_, val_split = train_test_split(df, test_size=config["training"]["val_ratio"], random_state=config["training"]["seed"])

# テストデータ
test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "datasets", "raw", "test.csv"))

# PN/GN tags
with open(os.path.join(EXP_DIR, "dataset", "form_type_dict.json")) as f:
    form_tag_dict = json.load(f)

def tag(text):
    return " ".join(f"{t}[{form_tag_dict[t]}]" if t in form_tag_dict else t for t in str(text).split())

def extract_first_sentence(text):
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()

# === 1. Val data: Akkadian vs English byte ratio ===
print("=== Val data: byte lengths ===")
val_akk_bytes = [len(str(row["transliteration"]).encode('utf-8')) for _, row in val_split.iterrows()]
val_akk_tagged_bytes = [len(tag(str(row["transliteration"])).encode('utf-8')) for _, row in val_split.iterrows()]
val_eng_bytes = [len(str(row["translation"]).encode('utf-8')) for _, row in val_split.iterrows()]
val_eng_first = [len(extract_first_sentence(str(row["translation"])).encode('utf-8')) for _, row in val_split.iterrows()]

print(f"  Akkadian raw:    mean={np.mean(val_akk_bytes):.0f}B, median={np.median(val_akk_bytes):.0f}B")
print(f"  Akkadian tagged: mean={np.mean(val_akk_tagged_bytes):.0f}B, median={np.median(val_akk_tagged_bytes):.0f}B")
print(f"  English full:    mean={np.mean(val_eng_bytes):.0f}B, median={np.median(val_eng_bytes):.0f}B")
print(f"  English 1st-sent:mean={np.mean(val_eng_first):.0f}B, median={np.median(val_eng_first):.0f}B")

ratios = [e / a if a > 0 else 0 for e, a in zip(val_eng_bytes, val_akk_bytes)]
print(f"\n  Eng/Akk ratio: mean={np.mean(ratios):.2f}, median={np.median(ratios):.2f}")

# === 2. テストデータ ===
print("\n=== Test data: byte lengths ===")
test_akk_bytes = [len(str(row["transliteration"]).encode('utf-8')) for _, row in test_df.iterrows()]
test_akk_tagged_bytes = [len(tag(str(row["transliteration"])).encode('utf-8')) for _, row in test_df.iterrows()]
print(f"  Akkadian raw:    {test_akk_bytes}")
print(f"  Akkadian tagged: {test_akk_tagged_bytes}")
print(f"  min={min(test_akk_bytes)}, max={max(test_akk_bytes)}")

# === 3. Val dataで200B以下のサンプルのみ抽出してratio確認 ===
print("\n=== Val samples where Akkadian ≤ 300B (test-like) ===")
short_indices = [i for i, b in enumerate(val_akk_bytes) if b <= 300]
print(f"  Count: {len(short_indices)} / {len(val_akk_bytes)}")
if short_indices:
    short_akk = [val_akk_bytes[i] for i in short_indices]
    short_eng = [val_eng_bytes[i] for i in short_indices]
    short_eng_first = [val_eng_first[i] for i in short_indices]
    short_ratios = [e / a if a > 0 else 0 for e, a in zip(short_eng, short_akk)]
    print(f"  Akkadian: mean={np.mean(short_akk):.0f}B")
    print(f"  English full: mean={np.mean(short_eng):.0f}B")
    print(f"  English 1st-sent: mean={np.mean(short_eng_first):.0f}B")
    print(f"  Eng/Akk ratio: mean={np.mean(short_ratios):.2f}, median={np.median(short_ratios):.2f}")
    print(f"  English full / Akkadian per sample:")
    for i in short_indices[:10]:
        akk = val_akk_bytes[i]
        eng = val_eng_bytes[i]
        eng1 = val_eng_first[i]
        r = eng / akk if akk > 0 else 0
        print(f"    Akk={akk:4d}B → Eng_full={eng:4d}B (ratio={r:.2f}), Eng_1st={eng1:4d}B")

# === 4. 200Bで切った時、対応する英語は何バイトか推定 ===
print("\n=== 200Bで切ったAkkadianに対応する英語の推定 ===")
# 200B / full_akk_bytes * full_eng_bytes で比例推定
for _, row in val_split.iterrows():
    akk_full = len(str(row["transliteration"]).encode('utf-8'))
    eng_full = len(str(row["translation"]).encode('utf-8'))

estimated_eng_for_200B = []
for i, (akk, eng) in enumerate(zip(val_akk_bytes, val_eng_bytes)):
    if akk > 0:
        proportion = min(200, akk) / akk  # 200Bでカバーする割合
        estimated_eng_for_200B.append(eng * proportion)

print(f"  200B Akkadian → estimated English: mean={np.mean(estimated_eng_for_200B):.0f}B, median={np.median(estimated_eng_for_200B):.0f}B")
print(f"  (比較) first-sent ref: mean={np.mean(val_eng_first):.0f}B")

# 200Bでどのくらいカバーされるか
coverage_200B = [min(200, a) / a if a > 0 else 1 for a in val_akk_bytes]
print(f"\n=== 200Bでのカバー率 ===")
print(f"  mean={np.mean(coverage_200B):.1%}, median={np.median(coverage_200B):.1%}")
print(f"  100%カバー: {sum(1 for c in coverage_200B if c >= 1)} / {len(coverage_200B)}")

# === 5. コンペの評価指標確認 ===
print("\n=== コンペ評価指標の確認 ===")
print("  コンペはchrF++ (word_order=2) を使用")
print("  training evalはchrF (word_order=0) を使用 ← ここがバグ")
