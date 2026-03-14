"""
eda024: GKF validation エラーパターン深掘り分析
- 翻字そのまま出力パターン
- 極端に短い予測パターン
- 短い入力での低スコアパターン
- fold別・AKTグループ別の傾向
"""
import pandas as pd
import numpy as np
import re
import json
from collections import Counter

OUT_DIR = "/home/user/work/Akkadian/eda/eda024_error_analysis"
df = pd.read_csv(f"{OUT_DIR}/all_folds_sent_with_scores.csv")
print(f"Total: {len(df)} samples, 5 folds")

# AKTグループ情報を追加
akt_groups = pd.read_csv("/home/user/work/Akkadian/datasets/processed/akt_groups.csv")
fold_to_groups = {
    0: ["AKT 8"],
    1: ["AKT 6a"],
    2: ["AKT 6e", "None"],
    3: ["AKT 6b", "AKT 5"],
    4: ["AKT 6c", "AKT 6d"],
}

# ============================================================
# 1. 翻字そのまま出力パターンの詳細分析
# ============================================================
print("\n" + "=" * 80)
print("1. 翻字そのまま出力パターン")
print("=" * 80)

# アッカド語翻字の特徴的なパターン
AKK_MARKERS = [
    r'\bDUMU\b', r'\bKÙ\b', r'\bIGI\b', r'\bGÍN\b', r'\bKIŠIB\b',
    r'\bma-na\b', r'-šu\b', r'-im\b', r'-a\b', r'\ba-na\b',
    r'\bša\b.*\bDUMU\b',  # "ša ... DUMU" pattern
]

def count_akk_markers(text):
    if not isinstance(text, str):
        return 0
    return sum(1 for p in AKK_MARKERS if re.search(p, text))

df["pred_akk_markers"] = df["prediction_raw"].apply(count_akk_markers)
df["ref_akk_markers"] = df["reference"].apply(count_akk_markers)

# 予測にアッカド語マーカーが2個以上 & 参照にはない → 翻字出力
akk_output = df[(df["pred_akk_markers"] >= 2) & (df["ref_akk_markers"] == 0)]
print(f"\n翻字出力ケース: {len(akk_output)} / {len(df)} ({len(akk_output)/len(df)*100:.1f}%)")
print(f"geo mean: {akk_output['geo'].mean():.2f}")

print(f"\nfold別:")
for fold in range(5):
    sub = akk_output[akk_output["fold"] == fold]
    total = len(df[df["fold"] == fold])
    print(f"  fold{fold}: {len(sub)}/{total} ({len(sub)/total*100:.1f}%)")

print(f"\n代表例:")
for _, row in akk_output.head(10).iterrows():
    inp_short = str(row["input"])[-80:]  # 末尾80文字(prefix除去)
    print(f"\n  fold{row['fold']} | geo={row['geo']:.1f}")
    print(f"  INPUT: ...{inp_short}")
    print(f"  REF:  {str(row['reference'])[:100]}")
    print(f"  PRED: {str(row['prediction_raw'])[:100]}")

# 翻字出力のinput特徴
if len(akk_output) > 0:
    print(f"\n翻字出力の入力特徴:")
    print(f"  入力バイト長 mean: {akk_output['input'].astype(str).str.len().mean():.0f} vs 全体: {df['input'].astype(str).str.len().mean():.0f}")
    print(f"  ref単語数 mean: {akk_output['ref_words'].mean():.1f} vs 全体: {df['ref_words'].mean():.1f}")

# ============================================================
# 2. 極端に短い予測パターン
# ============================================================
print("\n" + "=" * 80)
print("2. 極端に短い予測パターン (pred/ref < 0.3)")
print("=" * 80)

short_pred = df[df["len_ratio"] < 0.3].copy()
print(f"\n短予測ケース: {len(short_pred)} / {len(df)} ({len(short_pred)/len(df)*100:.1f}%)")
print(f"geo mean: {short_pred['geo'].mean():.2f}")

# 短予測の予測内容を分析
print(f"\n予測の単語数分布:")
short_pred["pred_words"] = short_pred["prediction_raw"].astype(str).str.split().str.len()
for nw in [1, 2, 3]:
    count = (short_pred["pred_words"] == nw).sum()
    print(f"  {nw}語: {count}")

# 1語予測の内容
one_word = short_pred[short_pred["pred_words"] == 1]
if len(one_word) > 0:
    word_counts = Counter(one_word["prediction_raw"].astype(str).tolist())
    print(f"\n1語予測の頻出パターン:")
    for word, count in word_counts.most_common(15):
        print(f"  '{word}': {count}回")

print(f"\nfold別:")
for fold in range(5):
    sub = short_pred[short_pred["fold"] == fold]
    total = len(df[df["fold"] == fold])
    print(f"  fold{fold}: {len(sub)}/{total} ({len(sub)/total*100:.1f}%)")

print(f"\n代表例:")
for _, row in short_pred.sort_values("geo").head(15).iterrows():
    print(f"\n  fold{row['fold']} | geo={row['geo']:.1f} | ref_words={row['ref_words']}")
    print(f"  REF:  {str(row['reference'])[:120]}")
    print(f"  PRED: {str(row['prediction_raw'])[:120]}")

# ============================================================
# 3. 短い入力での低スコア分析
# ============================================================
print("\n" + "=" * 80)
print("3. 短い入力 (0-100 bytes) の詳細分析")
print("=" * 80)

df["input_bytes"] = df["input"].astype(str).str.len()
short_input = df[df["input_bytes"] < 100].copy()
long_input = df[df["input_bytes"] >= 100].copy()

print(f"\n短入力: {len(short_input)} samples, geo mean={short_input['geo'].mean():.2f}")
print(f"長入力: {len(long_input)} samples, geo mean={long_input['geo'].mean():.2f}")

# prefix("translate Akkadian to English: ")のバイト数
PREFIX_LEN = len("translate Akkadian to English: ")
print(f"prefix長: {PREFIX_LEN} bytes")
print(f"→ 短入力の実質的なアッカド語テキスト: {short_input['input_bytes'].mean() - PREFIX_LEN:.0f} bytes平均")

# 短入力のref内容分析
print(f"\n短入力のref単語数分布:")
for lo, hi in [(1, 5), (5, 10), (10, 15), (15, 20), (20, 999)]:
    sub = short_input[(short_input["ref_words"] >= lo) & (short_input["ref_words"] < hi)]
    if len(sub) > 0:
        print(f"  {lo:2d}-{hi:3d} words: n={len(sub):3d}, geo={sub['geo'].mean():.2f}")

# 短入力で特にgeoが低いケース
print(f"\n短入力 & geo<10:")
bad_short = short_input[short_input["geo"] < 10]
print(f"  {len(bad_short)} / {len(short_input)} ({len(bad_short)/len(short_input)*100:.1f}%)")

# 短入力の中で翻字出力/短予測の割合
print(f"\n短入力内のエラータイプ:")
print(f"  翻字出力: {short_input['pred_akk_markers'].apply(lambda x: x >= 2).sum()} ({short_input['pred_akk_markers'].apply(lambda x: x >= 2).mean()*100:.1f}%)")
print(f"  短予測: {(short_input['len_ratio'] < 0.3).sum()} ({(short_input['len_ratio'] < 0.3).mean()*100:.1f}%)")

# ============================================================
# 4. geo=0のケース分析
# ============================================================
print("\n" + "=" * 80)
print("4. geo=0 のケース分析")
print("=" * 80)

zero_geo = df[df["geo"] == 0].copy()
print(f"\ngeo=0: {len(zero_geo)} samples")
print(f"fold別: {zero_geo['fold'].value_counts().sort_index().to_dict()}")

# BLEU=0の原因（1-gramも一致しない）
print(f"\nBLEU=0だがchrF>0: {((zero_geo['bleu'] == 0) & (zero_geo['chrf'] > 0)).sum()}")
print(f"chrF=0: {(zero_geo['chrf'] == 0).sum()}")

# geo=0の予測内容分類
def classify_zero(row):
    pred = str(row["prediction_raw"])
    ref = str(row["reference"])
    if count_akk_markers(pred) >= 2:
        return "transliteration_output"
    if len(pred.split()) <= 2 and len(ref.split()) > 5:
        return "too_short"
    if re.search(r'(\b\w+(?:\s+\w+){0,2}?)(?:\s+\1){2,}', pred):
        return "repetition"
    return "wrong_content"

zero_geo["error_type"] = zero_geo.apply(classify_zero, axis=1)
print(f"\nエラータイプ:")
for etype, count in zero_geo["error_type"].value_counts().items():
    print(f"  {etype}: {count}")

print(f"\n全ケース:")
for _, row in zero_geo.iterrows():
    print(f"\n  fold{row['fold']} | chrf={row['chrf']:.1f} | type={row['error_type']}")
    print(f"  REF:  {str(row['reference'])[:100]}")
    print(f"  PRED: {str(row['prediction_raw'])[:100]}")

# ============================================================
# 5. 会話文・命令文（says/said）の分析
# ============================================================
print("\n" + "=" * 80)
print("5. 会話文・命令文の分析")
print("=" * 80)

# 会話マーカー
conversation_markers = {
    "says/said/speak": r"\b(says?|said|speak|spoke|tell|told)\b",
    "imperative (send/give/bring)": r"\b(send|give|bring|take|come|go|let|must|should)\b",
    "question": r"\?",
    "quotation": r'["""]',
}

for name, pat in conversation_markers.items():
    mask = df["reference"].astype(str).str.contains(pat, case=False, regex=True)
    sub = df[mask]
    not_sub = df[~mask]
    if len(sub) > 0:
        print(f"\n{name}: n={len(sub)}, geo={sub['geo'].mean():.2f} (vs others: {not_sub['geo'].mean():.2f})")
        # fold別
        for fold in range(5):
            fsub = sub[sub["fold"] == fold]
            if len(fsub) > 0:
                print(f"  fold{fold}: n={len(fsub)}, geo={fsub['geo'].mean():.2f}")

# ============================================================
# 6. trainデータとの語彙重複分析
# ============================================================
print("\n" + "=" * 80)
print("6. 入力テキストの語彙 - train/valの重複")
print("=" * 80)

# train.csvを読んでfold別のtrain語彙を構築
train_df = pd.read_csv("/home/user/work/Akkadian/datasets/raw/train.csv")
akt_groups_df = pd.read_csv("/home/user/work/Akkadian/datasets/processed/akt_groups.csv")
oare_to_group = dict(zip(akt_groups_df["oare_id"], akt_groups_df["akt_group"].fillna("None")))
train_df["akt_group"] = train_df["oare_id"].map(oare_to_group).fillna("None")

from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
groups = train_df["akt_group"].values
splits = list(gkf.split(train_df, groups=groups))

for fold in range(5):
    train_idx, val_idx = splits[fold]
    train_texts = train_df.iloc[train_idx]["transliteration"].astype(str)
    val_texts = train_df.iloc[val_idx]["transliteration"].astype(str)

    # 単語レベルの語彙
    train_vocab = set()
    for t in train_texts:
        train_vocab.update(t.lower().split())

    val_vocab = set()
    for t in val_texts:
        val_vocab.update(t.lower().split())

    overlap = train_vocab & val_vocab
    val_only = val_vocab - train_vocab

    print(f"\nfold{fold} ({fold_to_groups[fold]}):")
    print(f"  train語彙: {len(train_vocab)}, val語彙: {len(val_vocab)}")
    print(f"  重複: {len(overlap)} ({len(overlap)/len(val_vocab)*100:.1f}%)")
    print(f"  val固有: {len(val_only)} ({len(val_only)/len(val_vocab)*100:.1f}%)")

# val予測のスコアとval固有語彙の関係
print("\n\n--- val固有語彙を含むサンプルのスコア ---")
for fold in range(5):
    train_idx, val_idx = splits[fold]
    train_texts = train_df.iloc[train_idx]["transliteration"].astype(str)
    train_vocab = set()
    for t in train_texts:
        train_vocab.update(t.lower().split())

    fold_df = df[df["fold"] == fold].copy()
    # 入力からprefixを除去してアッカド語部分を抽出
    fold_df["akk_input"] = fold_df["input"].astype(str).str.replace(r"translate Akkadian to English: ", "", regex=False)

    def unseen_ratio(text):
        words = str(text).lower().split()
        if not words:
            return 0
        unseen = sum(1 for w in words if w not in train_vocab)
        return unseen / len(words)

    fold_df["unseen_ratio"] = fold_df["akk_input"].apply(unseen_ratio)

    # unseen_ratio別スコア
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]:
        sub = fold_df[(fold_df["unseen_ratio"] >= lo) & (fold_df["unseen_ratio"] < hi)]
        if len(sub) > 0:
            print(f"  fold{fold} unseen {lo:.0%}-{hi:.0%}: n={len(sub):3d}, geo={sub['geo'].mean():.2f}")

# ============================================================
# 7. サマリー：改善可能なエラーの分類
# ============================================================
print("\n" + "=" * 80)
print("7. エラー分類サマリー")
print("=" * 80)

def classify_error(row):
    pred = str(row["prediction_raw"])
    ref = str(row["reference"])
    geo = row["geo"]

    if geo >= 50:
        return "good (geo>=50)"

    # 翻字出力
    if count_akk_markers(pred) >= 2 and row["ref_akk_markers"] == 0:
        return "A: transliteration_output"

    # 極端に短い
    if row["len_ratio"] < 0.3:
        return "B: too_short"

    # 極端に長い（繰り返し含む）
    if row["len_ratio"] > 3.0:
        return "C: too_long/repetition"

    if geo < 10:
        return "D: very_wrong (<10)"
    elif geo < 30:
        return "E: partially_wrong (10-30)"
    else:
        return "F: mediocre (30-50)"

df["error_class"] = df.apply(classify_error, axis=1)

print(f"\n{'error_class':<30} {'count':>6} {'%':>6} {'geo_mean':>10}")
print("-" * 55)
for cls in sorted(df["error_class"].unique()):
    sub = df[df["error_class"] == cls]
    print(f"{cls:<30} {len(sub):6d} {len(sub)/len(df)*100:5.1f}% {sub['geo'].mean():10.2f}")

# 改善可能なエラー（A+B+C）のインパクト
fixable = df[df["error_class"].isin(["A: transliteration_output", "B: too_short", "C: too_long/repetition"])]
print(f"\n改善可能なエラー(A+B+C): {len(fixable)} samples ({len(fixable)/len(df)*100:.1f}%)")
# もしこれらがmedian(geo=29.5)に改善されたら全体geoはどうなるか
current_mean = df["geo"].mean()
hypothetical = df["geo"].copy()
hypothetical[df["error_class"].isin(["A: transliteration_output", "B: too_short", "C: too_long/repetition"])] = df["geo"].median()
new_mean = hypothetical.mean()
print(f"現在の全体geo mean: {current_mean:.2f}")
print(f"A+B+Cがmedianに改善された場合: {new_mean:.2f} (+{new_mean - current_mean:.2f})")

print("\nDone.")
