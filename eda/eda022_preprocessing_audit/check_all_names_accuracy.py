"""val predictions全体で、固有名詞（大文字始まりのダイアクリティクス付き語）の正答率を調査"""
import pandas as pd
import re
from collections import Counter

val_preds = pd.read_csv("/home/user/work/Akkadian/workspace/exp023_full_preprocessing/results/val_predictions.csv")

# 固有名詞パターン: 大文字始まり、ダイアクリティクスやハイフン含む、2文字以上
# 一般英語の大文字語（The, From, Say, If, etc.）を除外
COMMON_WORDS = {
    "The", "From", "Say", "To", "If", "In", "As", "Of", "He", "We", "My",
    "You", "They", "When", "Let", "Do", "Send", "Give", "Take", "Have",
    "Not", "His", "Her", "Its", "Your", "Our", "Their", "But", "And",
    "Or", "So", "For", "With", "At", "By", "On", "An", "No", "All",
    "This", "That", "What", "Who", "How", "Why", "Where", "Which",
    "Seal", "Month", "City", "House", "Silver", "Gold", "Copper", "Tin",
    "God", "Goddess", "King", "Queen", "Son", "Father", "Mother", "Brother",
    "Sister", "Daughter", "Man", "Woman", "Witness", "Witnessed",
    "Here", "There", "Before", "After", "Since", "Until", "About",
    "Concerning", "According", "Because", "May", "Shall", "Will",
    "Should", "Would", "Could", "Can", "Apart", "Separately",
    "Reckoning", "Tablet", "Envelope", "Sealed", "Contracted",
    "Gate", "Total", "Amount", "Remainder", "Balance", "Debit", "Credit",
    "Anatolian", "Assyrian", "Save",
}

def extract_proper_names(text):
    """翻訳テキストから固有名詞を抽出"""
    # 大文字始まりの語（ハイフン接続含む）
    candidates = re.findall(r"[A-ZŠĀĒĪŪṢṬḪ][a-zA-Zšāēīūṣṭḫàáâãäåèéêëìíîïòóôõöùúûüʾ''ū-]+", text)
    # 一般語を除外
    names = [n for n in candidates if n not in COMMON_WORDS and len(n) >= 3]
    return names

total_ref_names = 0
total_found_in_pred = 0
total_not_found = 0
missed_names = Counter()
found_names = Counter()

per_row_results = []

for _, row in val_preds.iterrows():
    ref = str(row["reference"])
    pred = str(row["prediction_clean"])

    ref_names = extract_proper_names(ref)
    if not ref_names:
        continue

    found = 0
    missed = 0
    missed_list = []
    for name in ref_names:
        # predに名前が含まれるか（大文字小文字区別、部分一致OK）
        if name in pred:
            found += 1
            found_names[name] += 1
        else:
            missed += 1
            missed_names[name] += 1
            missed_list.append(name)

    total_ref_names += len(ref_names)
    total_found_in_pred += found
    total_not_found += missed

    per_row_results.append({
        "ref_name_count": len(ref_names),
        "found": found,
        "missed": missed,
        "accuracy": found / len(ref_names) if ref_names else 1.0,
        "missed_names": missed_list,
        "ref": ref[:100],
        "pred": pred[:100],
    })

print(f"=== 固有名詞正答率（val全体） ===")
print(f"reference中の固有名詞総数: {total_ref_names}")
print(f"predictionに出現: {total_found_in_pred} ({100*total_found_in_pred/total_ref_names:.1f}%)")
print(f"predictionに不在: {total_not_found} ({100*total_not_found/total_ref_names:.1f}%)")

print(f"\n=== 最も外しやすい固有名詞 TOP20 ===")
for name, cnt in missed_names.most_common(20):
    total_in_ref = missed_names[name] + found_names.get(name, 0)
    print(f"  {name}: missed {cnt}/{total_in_ref} ({100*cnt/total_in_ref:.0f}%)")

print(f"\n=== 行単位の精度分布 ===")
df = pd.DataFrame(per_row_results)
for bucket in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
    if bucket == 1.0:
        count = (df["accuracy"] == 1.0).sum()
        print(f"  100%正解: {count}/{len(df)} ({100*count/len(df):.1f}%)")
    elif bucket == 0.0:
        count = (df["accuracy"] == 0.0).sum()
        print(f"  0%正解: {count}/{len(df)} ({100*count/len(df):.1f}%)")
    else:
        count = ((df["accuracy"] >= bucket) & (df["accuracy"] < bucket + 0.2)).sum()
        print(f"  {bucket*100:.0f}-{(bucket+0.2)*100:.0f}%: {count}/{len(df)}")

print(f"\n=== 固有名詞を最も外した行（TOP5） ===")
worst = df.sort_values("missed", ascending=False).head(5)
for _, r in worst.iterrows():
    print(f"  missed {r['missed']}/{r['ref_name_count']}: {r['missed_names']}")
    print(f"    ref:  {r['ref']}")
    print(f"    pred: {r['pred']}")
    print()
