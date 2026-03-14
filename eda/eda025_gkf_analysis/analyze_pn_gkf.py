"""
GKF fold0 vs ランダム分割で固有名詞の影響を分析
- 参照中の固有名詞（大文字始まりの単語）を抽出
- trainに出現する固有名詞 vs 未出現の固有名詞でスコアを比較
"""
import re
import math
import pandas as pd
import sacrebleu
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path(__file__).resolve().parent
EDA020_DIR = Path(__file__).resolve().parent.parent / "eda020_sent_level_cv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
# GKF fold0の予測読み込み
# ============================================================
sent_df = pd.read_csv(EDA020_DIR / "exp023_gkf_fold0_last_sent_predictions.csv")
print(f"GKF fold0 sent predictions: {len(sent_df)}")

# ============================================================
# train dataのtranslation（GKF fold0のtrainに含まれるもの）
# ============================================================
train_df = pd.read_csv(PROJECT_ROOT / "datasets" / "raw" / "train.csv")
akt = pd.read_csv(PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv")
oare_to_group = dict(zip(akt["oare_id"], akt["akt_group"].fillna("None")))
train_df["akt_group"] = train_df["oare_id"].map(oare_to_group).fillna("None")

# fold0のvalはAKT 8。trainは残り全部
train_texts = train_df[train_df["akt_group"] != "AKT 8"]["translation"].astype(str).tolist()
val_texts = train_df[train_df["akt_group"] == "AKT 8"]["translation"].astype(str).tolist()

# ============================================================
# 固有名詞抽出（大文字始まりの単語、一般的な文頭単語を除外）
# ============================================================
COMMON_STARTERS = {
    "The", "He", "She", "It", "They", "We", "I", "You", "A", "An",
    "This", "That", "These", "Those", "My", "His", "Her", "Its",
    "Our", "Your", "Their", "If", "When", "After", "Before", "Since",
    "From", "To", "In", "On", "At", "By", "For", "With", "Of",
    "Seal", "Tablet", "Witness", "Total", "Month", "Year",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
    "Nine", "Ten", "Reckoned", "Witnessed", "Day",
}

def extract_proper_nouns(text):
    """大文字始まりの単語で、一般的な文頭語でないもの = 固有名詞候補"""
    words = str(text).split()
    pns = set()
    for w in words:
        # 記号除去
        clean = re.sub(r'[,.:;!?\(\)\[\]"\']+', '', w)
        if not clean:
            continue
        if clean[0].isupper() and clean not in COMMON_STARTERS and len(clean) > 1:
            pns.add(clean)
    return pns

# trainに出現する固有名詞を収集
train_pns = Counter()
for text in train_texts:
    for pn in extract_proper_nouns(text):
        train_pns[pn] += 1

# val（AKT 8）に出現する固有名詞
val_pns = Counter()
for text in val_texts:
    for pn in extract_proper_nouns(text):
        val_pns[pn] += 1

# AKT 8固有の名前（trainに出現しない）
akt8_only = {pn for pn in val_pns if pn not in train_pns}
shared_pns = {pn for pn in val_pns if pn in train_pns}

print(f"\n=== 固有名詞統計 ===")
print(f"Train固有名詞: {len(train_pns)} unique")
print(f"Val(AKT 8)固有名詞: {len(val_pns)} unique")
print(f"  共通: {len(shared_pns)}")
print(f"  AKT 8のみ: {len(akt8_only)} ({100*len(akt8_only)/max(len(val_pns),1):.1f}%)")

print(f"\nAKT 8固有の名前TOP20:")
for pn in sorted(akt8_only, key=lambda x: -val_pns[x])[:20]:
    print(f"  {pn}: {val_pns[pn]}回")

# ============================================================
# 各sentの参照に含まれる固有名詞のtrain出現率とスコアの関係
# ============================================================
print(f"\n=== sent-CV予測での固有名詞分析 ===")

rows = []
for _, row in sent_df.iterrows():
    ref = str(row["reference"])
    pred = str(row["prediction_clean"])
    ref_pns = extract_proper_nouns(ref)
    if not ref_pns:
        rows.append({"has_unseen_pn": False, "n_pns": 0, "n_unseen": 0,
                      "ref": ref, "pred": pred})
        continue
    unseen = ref_pns & akt8_only
    rows.append({
        "has_unseen_pn": len(unseen) > 0,
        "n_pns": len(ref_pns),
        "n_unseen": len(unseen),
        "ref": ref, "pred": pred,
    })

ana_df = pd.DataFrame(rows)

# 各文のchrF++計算
ana_df["chrf"] = [
    sacrebleu.sentence_chrf(p, [r], word_order=2).score
    for p, r in zip(ana_df["pred"], ana_df["ref"])
]

# 未見固有名詞あり vs なし
for has_unseen, group in ana_df.groupby("has_unseen_pn"):
    label = "未見PN含む" if has_unseen else "全PN既知 or PNなし"
    corpus_chrf = sacrebleu.corpus_chrf(group["pred"].tolist(), [group["ref"].tolist()], word_order=2).score
    corpus_bleu = sacrebleu.corpus_bleu(group["pred"].tolist(), [group["ref"].tolist()]).score
    geo = math.sqrt(corpus_chrf * corpus_bleu) if corpus_chrf > 0 and corpus_bleu > 0 else 0
    print(f"\n{label}: {len(group)}件")
    print(f"  corpus chrF++={corpus_chrf:.2f}, BLEU={corpus_bleu:.2f}, geo={geo:.2f}")
    print(f"  sentence chrF++ mean={group['chrf'].mean():.2f}, median={group['chrf'].median():.2f}")

# 固有名詞の予測一致率
print(f"\n=== 固有名詞の予測内一致率 ===")

seen_match = 0
seen_total = 0
unseen_match = 0
unseen_total = 0

for _, row in sent_df.iterrows():
    ref = str(row["reference"])
    pred = str(row["prediction_clean"])
    ref_pns = extract_proper_nouns(ref)
    pred_pns = extract_proper_nouns(pred)
    for pn in ref_pns:
        if pn in akt8_only:
            unseen_total += 1
            if pn in pred_pns:
                unseen_match += 1
        else:
            seen_total += 1
            if pn in pred_pns:
                seen_match += 1

print(f"Train既知PN: {seen_match}/{seen_total} 一致 ({100*seen_match/max(seen_total,1):.1f}%)")
print(f"Train未見PN: {unseen_match}/{unseen_total} 一致 ({100*unseen_match/max(unseen_total,1):.1f}%)")

# 未見PNの具体例
print(f"\n=== 未見PN含む低スコア例 ===")
unseen_rows = ana_df[ana_df["has_unseen_pn"]].sort_values("chrf").head(10)
for _, row in unseen_rows.iterrows():
    print(f"  chrF++={row['chrf']:.1f} | unseen={row['n_unseen']}/{row['n_pns']} PNs")
    print(f"    Ref:  {row['ref'][:120]}")
    print(f"    Pred: {row['pred'][:120]}")
