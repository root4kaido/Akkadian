"""同じ{d}ロゴグラムがtranslation中でどう訳されるかの揺れを調査"""
import pandas as pd
import re
from collections import defaultdict

train = pd.read_csv("/home/user/work/Akkadian/datasets/raw/train.csv")

# {d}+ロゴグラムを含む行を抽出し、対応するtranslation中の固有名詞を見る
d_pattern = re.compile(r"\{d\}(\S+)")

# 各ロゴグラムについて、出現する行のtranslationをすべて収集
logo_to_translations = defaultdict(list)
for _, row in train.iterrows():
    tl = str(row["transliteration"])
    tr = str(row["translation"])
    for m in d_pattern.finditer(tl):
        logo = m.group(1)
        logo_to_translations[logo].append(tr)

# 頻出ロゴグラムについて、translation中の固有名詞パターンを分析
# 各ロゴグラムの既知の対応（手動）
known_mappings = {
    "UTU": "Šamaš",
    "EN.LÍL": "Illil",
    "IM": "Adad",
    "IŠKUR": "Adad",
    "MAR.TU": "Amurrum",
    "AB": "Ab",
}

print("=== ロゴグラム別: translation中で対応する名前の出現パターン ===\n")

for logo, translations in sorted(logo_to_translations.items(), key=lambda x: -len(x[1])):
    if len(translations) < 3:
        continue
    print(f"--- {'{d}'}{logo} ({len(translations)}件) ---")
    # translation中から固有名詞っぽいもの（大文字始まり+ダイアクリティクス）を抽出
    all_names = []
    for tr in translations:
        # 大文字始まりの単語（ハイフン接続含む）
        names = re.findall(r"[A-ZŠĀĒĪŪṢṬḪȚÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜŻŹĆŃĹŔŚŤĎĽŇ][a-zA-Zšāēīūṣṭḫàáâãäåèéêëìíîïòóôõöùúûüżźćńĺŕśťďľňšū'ʾ-]+", tr)
        all_names.extend(names)

    # 既知のマッピングがあればその名前を含むものをカウント
    if logo in known_mappings:
        target = known_mappings[logo]
        matching = [n for n in all_names if target.lower() in n.lower()]
        other_candidates = [n for n in all_names if target.lower() not in n.lower()]
        print(f"  既知対応 '{target}' を含む名前: {len(matching)}件")
        # ユニークな名前でカウント
        from collections import Counter
        match_counts = Counter(matching)
        for name, cnt in match_counts.most_common(10):
            print(f"    {name}: {cnt}")
        print(f"  その他の名前（上位10）:")
        other_counts = Counter(other_candidates)
        for name, cnt in other_counts.most_common(10):
            print(f"    {name}: {cnt}")
    else:
        from collections import Counter
        name_counts = Counter(all_names)
        print(f"  出現する名前（上位15）:")
        for name, cnt in name_counts.most_common(15):
            print(f"    {name}: {cnt}")
    print()
