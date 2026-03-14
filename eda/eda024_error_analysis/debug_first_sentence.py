"""
first_sentence截断で悪化したケースを具体的に調べる
"""
import re
import pandas as pd
import sacrebleu
from pathlib import Path

PRED_PATH = Path(__file__).resolve().parent.parent / "eda020_sent_level_cv" / "sent_level_predictions.csv"
df = pd.read_csv(PRED_PATH)
df["pred"] = df["prediction_clean"].astype(str)
df["ref"] = df["reference"].astype(str)


def extract_first_sentence(text: str) -> str:
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()


# 各行でoriginal vs truncatedのスコアを比較
results = []
for _, row in df.iterrows():
    pred_orig = row["pred"]
    pred_trunc = extract_first_sentence(pred_orig)
    ref = row["ref"]

    chrf_orig = sacrebleu.sentence_chrf(pred_orig, [ref], word_order=2).score
    chrf_trunc = sacrebleu.sentence_chrf(pred_trunc, [ref], word_order=2).score

    changed = pred_orig != pred_trunc
    results.append({
        "ref": ref,
        "pred_orig": pred_orig,
        "pred_trunc": pred_trunc,
        "chrf_orig": chrf_orig,
        "chrf_trunc": chrf_trunc,
        "delta": chrf_trunc - chrf_orig,
        "changed": changed,
    })

rdf = pd.DataFrame(results)

print(f"Total: {len(rdf)}")
print(f"Changed by truncation: {rdf['changed'].sum()}")
print(f"Unchanged: {(~rdf['changed']).sum()}")

changed = rdf[rdf["changed"]]
print(f"\n--- Changed cases ({len(changed)}) ---")
print(f"Improved (delta>0): {(changed['delta'] > 0).sum()}")
print(f"Worsened (delta<0): {(changed['delta'] < 0).sum()}")
print(f"Mean delta: {changed['delta'].mean():.2f}")

# 悪化したケースを表示
worsened = changed[changed["delta"] < 0].sort_values("delta")
print(f"\n=== 悪化ケース TOP10 ===")
for i, (_, row) in enumerate(worsened.head(10).iterrows()):
    print(f"\n[{i}] delta={row['delta']:+.1f}")
    print(f"  Ref:   {row['ref'][:150]}")
    print(f"  Orig:  {row['pred_orig'][:150]}")
    print(f"  Trunc: {row['pred_trunc'][:150]}")

# 改善したケースも表示
improved = changed[changed["delta"] > 0].sort_values("delta", ascending=False)
print(f"\n=== 改善ケース TOP5 ===")
for i, (_, row) in enumerate(improved.head(5).iterrows()):
    print(f"\n[{i}] delta={row['delta']:+.1f}")
    print(f"  Ref:   {row['ref'][:150]}")
    print(f"  Orig:  {row['pred_orig'][:150]}")
    print(f"  Trunc: {row['pred_trunc'][:150]}")
