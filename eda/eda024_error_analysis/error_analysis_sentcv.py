"""
eda024: sent-CV予測(exp023, geo=34.35)に対するエラー分析
eval_full_docのsent_level_predictions.csvを使用
"""
import re
import math
import pandas as pd
import sacrebleu
from pathlib import Path
from collections import Counter

PRED_PATH = Path(__file__).resolve().parent.parent / "eda020_sent_level_cv" / "sent_level_predictions.csv"
df = pd.read_csv(PRED_PATH)

df["pred"] = df["prediction_clean"].astype(str)
df["ref"] = df["reference"].astype(str)
df["inp"] = df["input"].astype(str)

print(f"Total predictions: {len(df)}")

# ============================================================
# 1. 全体スコア（コーパスレベル）
# ============================================================
chrf = sacrebleu.corpus_chrf(df["pred"].tolist(), [df["ref"].tolist()], word_order=2).score
bleu = sacrebleu.corpus_bleu(df["pred"].tolist(), [df["ref"].tolist()]).score
geo = math.sqrt(chrf * bleu) if chrf > 0 and bleu > 0 else 0
print(f"\nCorpus: chrF++={chrf:.2f}, BLEU={bleu:.2f}, geo={geo:.2f}")

# ============================================================
# 2. 各文のスコアと長さ情報
# ============================================================
scores = []
for _, row in df.iterrows():
    s_chrf = sacrebleu.sentence_chrf(row["pred"], [row["ref"]], word_order=2).score
    s_bleu = sacrebleu.sentence_bleu(row["pred"], [row["ref"]]).score
    pred_words = len(row["pred"].split())
    ref_words = len(row["ref"].split())
    inp_bytes = len(row["inp"].encode("utf-8"))
    ratio = pred_words / max(ref_words, 1)
    scores.append({
        "chrf": s_chrf,
        "bleu": s_bleu,
        "pred_words": pred_words,
        "ref_words": ref_words,
        "inp_bytes": inp_bytes,
        "ratio": ratio,
    })

sdf = pd.DataFrame(scores)
df = pd.concat([df.reset_index(drop=True), sdf], axis=1)

print(f"\nSentence-level stats:")
print(f"  chrF++ mean={df['chrf'].mean():.2f}, median={df['chrf'].median():.2f}")
print(f"  BLEU   mean={df['bleu'].mean():.2f}, median={df['bleu'].median():.2f}")
print(f"  pred words: mean={df['pred_words'].mean():.1f}, median={df['pred_words'].median():.1f}")
print(f"  ref  words: mean={df['ref_words'].mean():.1f}, median={df['ref_words'].median():.1f}")
print(f"  ratio (pred/ref): mean={df['ratio'].mean():.2f}, median={df['ratio'].median():.2f}")

# ============================================================
# 3. 長さカテゴリ別分析
# ============================================================
print("\n" + "=" * 70)
print("長さカテゴリ別")
print("=" * 70)

bins = [
    ("too_short (ratio<0.5)", df["ratio"] < 0.5),
    ("short (0.5<=ratio<0.8)", (df["ratio"] >= 0.5) & (df["ratio"] < 0.8),),
    ("normal (0.8<=ratio<=1.3)", (df["ratio"] >= 0.8) & (df["ratio"] <= 1.3)),
    ("long (1.3<ratio<=2.0)", (df["ratio"] > 1.3) & (df["ratio"] <= 2.0)),
    ("too_long (ratio>2.0)", df["ratio"] > 2.0),
]

for name, mask in bins:
    subset = df[mask]
    if len(subset) == 0:
        continue
    mean_chrf = subset["chrf"].mean()
    mean_bleu = subset["bleu"].mean()
    print(f"\n{name}: {len(subset)}件 ({100*len(subset)/len(df):.1f}%)")
    print(f"  chrF++={mean_chrf:.2f}, BLEU={mean_bleu:.2f}")
    print(f"  avg pred={subset['pred_words'].mean():.1f}w, avg ref={subset['ref_words'].mean():.1f}w")

# ============================================================
# 4. 繰り返し検出
# ============================================================
print("\n" + "=" * 70)
print("繰り返し検出")
print("=" * 70)

def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False

df["has_rep"] = df["pred"].apply(has_repetition)
rep_count = df["has_rep"].sum()
print(f"繰り返しあり: {rep_count}/{len(df)} ({100*rep_count/len(df):.1f}%)")
if rep_count > 0:
    rep_df = df[df["has_rep"]]
    print(f"  chrF++={rep_df['chrf'].mean():.2f} vs 全体{df['chrf'].mean():.2f}")

# ============================================================
# 5. 予測にピリオドが複数ある（複数文生成）
# ============================================================
print("\n" + "=" * 70)
print("予測の文数分析")
print("=" * 70)

def count_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', str(text))
    sents = [s for s in sents if s.strip()]
    return len(sents)

df["pred_sents"] = df["pred"].apply(count_sentences)
df["ref_sents"] = df["ref"].apply(count_sentences)

for n in sorted(df["pred_sents"].unique()):
    subset = df[df["pred_sents"] == n]
    print(f"  pred={n}文: {len(subset)}件, chrF++={subset['chrf'].mean():.2f}")

print(f"\n参照側:")
for n in sorted(df["ref_sents"].unique()):
    subset = df[df["ref_sents"] == n]
    print(f"  ref={n}文: {len(subset)}件")

# 予測が参照より文数が多いケース
multi = df[df["pred_sents"] > df["ref_sents"]]
print(f"\n予測文数 > 参照文数: {len(multi)}件 ({100*len(multi)/len(df):.1f}%)")
if len(multi) > 0:
    print(f"  chrF++={multi['chrf'].mean():.2f} vs 全体{df['chrf'].mean():.2f}")

# ============================================================
# 6. 入力長別分析
# ============================================================
print("\n" + "=" * 70)
print("入力長(bytes)別")
print("=" * 70)

byte_bins = [
    ("<100B", df["inp_bytes"] < 100),
    ("100-200B", (df["inp_bytes"] >= 100) & (df["inp_bytes"] < 200)),
    ("200-300B", (df["inp_bytes"] >= 200) & (df["inp_bytes"] < 300)),
    ("300-512B", (df["inp_bytes"] >= 300) & (df["inp_bytes"] < 512)),
    (">=512B", df["inp_bytes"] >= 512),
]

for name, mask in byte_bins:
    subset = df[mask]
    if len(subset) == 0:
        continue
    print(f"  {name}: {len(subset)}件, chrF++={subset['chrf'].mean():.2f}, BLEU={subset['bleu'].mean():.2f}")

# ============================================================
# 7. chrF++ < 30 の低スコアケース
# ============================================================
print("\n" + "=" * 70)
print("chrF++ < 30 の低スコアケース")
print("=" * 70)

low = df[df["chrf"] < 30].sort_values("chrf")
print(f"件数: {len(low)}/{len(df)}")

for i, (_, row) in enumerate(low.head(10).iterrows()):
    print(f"\n[{i}] chrF++={row['chrf']:.1f}, ratio={row['ratio']:.2f}, inp={row['inp_bytes']}B")
    print(f"  Ref:  {row['ref'][:150]}")
    print(f"  Pred: {row['pred'][:150]}")

# ============================================================
# 8. 空予測・極短予測
# ============================================================
print("\n" + "=" * 70)
print("空・極短予測")
print("=" * 70)

empty = df[df["pred_words"] <= 2]
print(f"2語以下: {len(empty)}件")
for _, row in empty.iterrows():
    print(f"  Pred: '{row['pred']}' | Ref: '{row['ref'][:80]}'")

# Save
df.to_csv(Path(__file__).resolve().parent / "sentcv_predictions_scored.csv", index=False)
print(f"\nSaved to sentcv_predictions_scored.csv")
