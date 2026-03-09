"""
追加データとtrain全体のテキスト類似度チェック
- transliteration同士の類似度（SequenceMatcher）
- translation同士の類似度
- 高類似度ペアを報告
"""
import pandas as pd
from difflib import SequenceMatcher
from collections import defaultdict

DATA_DIR = "datasets/raw"

# Load
train = pd.read_csv(f"{DATA_DIR}/train.csv")
additional = pd.read_csv("workspace/exp011_additional_data/dataset/additional_train.csv")

print(f"train: {len(train)} rows")
print(f"additional: {len(additional)} rows")

# 1. oare_id重複チェック（念のため）
train_ids = set(train["oare_id"])
additional_ids = set(additional["oare_id"])
overlap_ids = train_ids & additional_ids
print(f"\noare_id重複: {len(overlap_ids)}")

# 2. transliteration完全一致チェック
train_translits = set(train["transliteration"].astype(str))
additional_translits = additional["transliteration"].astype(str)
exact_match_translit = sum(1 for t in additional_translits if t in train_translits)
print(f"transliteration完全一致: {exact_match_translit}/{len(additional)}")

# 3. translation完全一致チェック
train_translations = set(train["translation"].astype(str))
additional_translations = additional["translation"].astype(str)
exact_match_trans = sum(1 for t in additional_translations if t in train_translations)
print(f"translation完全一致: {exact_match_trans}/{len(additional)}")

# 4. 高類似度ペア検出（transliteration）
# 全ペア比較は O(n*m) で重いので、先頭50文字のプレフィックスでフィルタリング
print(f"\n=== transliteration類似度チェック ===")

# プレフィックスインデックス
prefix_len = 30
prefix_index = defaultdict(list)
for idx, row in train.iterrows():
    t = str(row["transliteration"])
    prefix = t[:prefix_len]
    prefix_index[prefix].append((idx, t))

high_sim_pairs = []
for add_idx, add_row in additional.iterrows():
    add_t = str(add_row["transliteration"])
    prefix = add_t[:prefix_len]

    # 同じプレフィックスを持つtrainエントリと比較
    candidates = prefix_index.get(prefix, [])
    for train_idx, train_t in candidates:
        ratio = SequenceMatcher(None, add_t, train_t).ratio()
        if ratio > 0.7:
            high_sim_pairs.append({
                "add_idx": add_idx,
                "train_idx": train_idx,
                "sim_translit": round(ratio, 3),
                "add_translit": add_t[:100],
                "train_translit": train_t[:100],
            })

print(f"プレフィックス{prefix_len}文字一致でフィルタ後、類似度>0.7: {len(high_sim_pairs)}件")

# 5. ブルートフォースで全ペアサンプリングチェック（ランダム100件）
import random
random.seed(42)

sample_add = additional.sample(min(100, len(additional)), random_state=42)
brute_high = 0
brute_max_sim = 0
brute_max_pair = None

for _, add_row in sample_add.iterrows():
    add_t = str(add_row["transliteration"])
    best_sim = 0
    best_train_t = ""
    for _, train_row in train.iterrows():
        train_t = str(train_row["transliteration"])
        ratio = SequenceMatcher(None, add_t, train_t).ratio()
        if ratio > best_sim:
            best_sim = ratio
            best_train_t = train_t
    if best_sim > 0.7:
        brute_high += 1
    if best_sim > brute_max_sim:
        brute_max_sim = best_sim
        brute_max_pair = (add_t[:100], best_train_t[:100])

print(f"\nランダム100件 × train全件 ブルートフォース:")
print(f"  transliteration類似度>0.7: {brute_high}/100")
print(f"  最大類似度: {brute_max_sim:.3f}")
if brute_max_pair:
    print(f"  最類似ペア:")
    print(f"    add: {brute_max_pair[0]}")
    print(f"    train: {brute_max_pair[1]}")

# 6. translation類似度もサンプルチェック
print(f"\n=== translation類似度チェック (サンプル100件) ===")
brute_high_trans = 0
brute_max_sim_trans = 0

for _, add_row in sample_add.iterrows():
    add_t = str(add_row["translation"])
    best_sim = 0
    for _, train_row in train.iterrows():
        train_t = str(train_row["translation"])
        ratio = SequenceMatcher(None, add_t, train_t).ratio()
        if ratio > best_sim:
            best_sim = ratio
    if best_sim > 0.7:
        brute_high_trans += 1
    if best_sim > brute_max_sim_trans:
        brute_max_sim_trans = best_sim

print(f"  translation類似度>0.7: {brute_high_trans}/100")
print(f"  最大類似度: {brute_max_sim_trans:.3f}")

# サマリー
print(f"\n{'='*60}")
print(f"=== サマリー ===")
print(f"{'='*60}")
print(f"oare_id重複: {len(overlap_ids)}")
print(f"transliteration完全一致: {exact_match_translit}")
print(f"translation完全一致: {exact_match_trans}")
print(f"transliteration高類似(>0.7, prefix filter): {len(high_sim_pairs)}")
print(f"transliteration高類似(>0.7, brute 100件): {brute_high}/100")
print(f"translation高類似(>0.7, brute 100件): {brute_high_trans}/100")
