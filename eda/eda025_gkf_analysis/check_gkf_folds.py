import pandas as pd
from sklearn.model_selection import GroupKFold
from pathlib import Path
import re

PROJECT_ROOT = Path("/home/user/work/Akkadian")
train_df = pd.read_csv(PROJECT_ROOT / "datasets/raw/train.csv")
akt = pd.read_csv(PROJECT_ROOT / "datasets/processed/akt_groups.csv")
oare_to_group = dict(zip(akt["oare_id"], akt["akt_group"].fillna("None")))

aligned = []
for _, row in train_df.iterrows():
    src, tgt = str(row["transliteration"]), str(row["translation"])
    tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
    src_lines = [s.strip() for s in src.split("\n") if s.strip()]
    if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
        for s, t in zip(src_lines, tgt_sents):
            if len(s) > 3 and len(t) > 3:
                aligned.append({"oare_id": row["oare_id"]})
    else:
        aligned.append({"oare_id": row["oare_id"]})
df = pd.DataFrame(aligned)
df["akt_group"] = df["oare_id"].map(oare_to_group).fillna("None")

print(f"Total samples: {len(df)}")
print(f"Groups: {df['akt_group'].value_counts().to_dict()}\n")

gkf = GroupKFold(n_splits=5)
for i, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["akt_group"].values)):
    val = df.iloc[val_idx]
    groups = val["akt_group"].value_counts().to_dict()
    print(f"Fold {i}: train={len(train_idx):>5}, val={len(val_idx):>4}  val_groups={groups}")
