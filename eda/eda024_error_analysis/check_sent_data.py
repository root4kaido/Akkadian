"""sentence alignerの分割統計と、短文データの有無を確認"""
import pandas as pd
import re

train_df = pd.read_csv("/home/user/work/Akkadian/datasets/raw/train.csv")
print(f"Total docs: {len(train_df)}")

split_count = 0
sent_count = 0
unsplit_count = 0
all_sents = []

for _, row in train_df.iterrows():
    src = str(row["transliteration"])
    tgt = str(row["translation"])
    oare_id = row["oare_id"]
    tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
    src_lines = [s.strip() for s in src.split("\n") if s.strip()]
    if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
        split_count += 1
        for s, t in zip(src_lines, tgt_sents):
            if len(s) > 3 and len(t) > 3:
                all_sents.append({"src": s, "tgt": t, "oare_id": oare_id, "type": "split"})
                sent_count += 1
    else:
        unsplit_count += 1
        all_sents.append({"src": src, "tgt": tgt, "oare_id": oare_id, "type": "unsplit"})

print(f"Split docs: {split_count} -> {sent_count} sents")
print(f"Unsplit docs: {unsplit_count}")
print(f"Total after expansion: {sent_count + unsplit_count}")

# 短文の統計
df = pd.DataFrame(all_sents)
df["src_words"] = df["src"].str.split().str.len()
df["tgt_words"] = df["tgt"].str.split().str.len()
df["src_bytes"] = df["src"].str.len()

print(f"\n--- 文長分布 ---")
for lo, hi in [(0, 50), (50, 100), (100, 200), (200, 500), (500, 9999)]:
    sub = df[(df["src_bytes"] >= lo) & (df["src_bytes"] < hi)]
    print(f"  src {lo:4d}-{hi:4d} bytes: {len(sub):4d} ({len(sub)/len(df)*100:.1f}%)")

print(f"\n--- split vs unsplit ---")
for t in ["split", "unsplit"]:
    sub = df[df["type"] == t]
    print(f"  {t}: {len(sub)}, src_bytes mean={sub['src_bytes'].mean():.0f}, tgt_words mean={sub['tgt_words'].mean():.1f}")

# sentence_aligned.csvの確認
print(f"\n--- sentence_aligned.csv ---")
sa = pd.read_csv("/home/user/work/Akkadian/datasets/processed/sentence_aligned.csv")
print(f"Total rows: {len(sa)}")
print(f"sent_idx >= 1: {(sa['sent_idx'] >= 1).sum()}")
print(f"Columns: {list(sa.columns)}")
print(sa.head(3).to_string())
