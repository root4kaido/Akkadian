import pandas as pd
import numpy as np
from sacrebleu.metrics import CHRF

chrf = CHRF(word_order=2)

large_sent = pd.read_csv("eda/eda020_sent_level_cv/s1_exp001_byt5_large_fold3_last_sent_predictions.csv")
base_sent = pd.read_csv("eda/eda020_sent_level_cv/exp036_last_sent_predictions.csv")

print(f"Large sent: {len(large_sent)}, Base sent: {len(base_sent)}")

# 予測長（バイト数）
large_sent["pred_bytes"] = large_sent["prediction_clean"].astype(str).apply(lambda x: len(x.encode("utf-8")))
large_sent["ref_bytes"] = large_sent["reference"].astype(str).apply(lambda x: len(x.encode("utf-8")))
base_sent["pred_bytes"] = base_sent["prediction_clean"].astype(str).apply(lambda x: len(x.encode("utf-8")))

print("\n=== 予測長(bytes) ===")
print(f"Large: mean={large_sent['pred_bytes'].mean():.1f}, median={large_sent['pred_bytes'].median():.0f}, max={large_sent['pred_bytes'].max()}")
print(f"Base:  mean={base_sent['pred_bytes'].mean():.1f}, median={base_sent['pred_bytes'].median():.0f}, max={base_sent['pred_bytes'].max()}")
print(f"Ref:   mean={large_sent['ref_bytes'].mean():.1f}, median={large_sent['ref_bytes'].median():.0f}, max={large_sent['ref_bytes'].max()}")

# 512バイト付近
print(f"\n=== >=500 bytes ===")
print(f"Large: {(large_sent['pred_bytes'] >= 500).sum()}")
print(f"Base:  {(base_sent['pred_bytes'] >= 500).sum()}")

# 完全一致
large_sent["exact"] = large_sent["prediction_clean"].astype(str) == large_sent["reference"].astype(str)
base_sent["exact"] = base_sent["prediction_clean"].astype(str) == base_sent["reference"].astype(str)
print(f"\n=== 完全一致 ===")
print(f"Large: {large_sent['exact'].sum()} ({large_sent['exact'].mean()*100:.1f}%)")
print(f"Base:  {base_sent['exact'].sum()} ({base_sent['exact'].mean()*100:.1f}%)")

# per-sample chrF++
ls, bs = [], []
for i in range(len(large_sent)):
    ref = str(large_sent.iloc[i]["reference"])
    lp = str(large_sent.iloc[i]["prediction_clean"])
    bp = str(base_sent.iloc[i]["prediction_clean"])
    ls.append(chrf.sentence_score(lp, [ref]).score)
    bs.append(chrf.sentence_score(bp, [ref]).score)

large_sent["chrf"] = ls
base_sent["chrf"] = bs
diff = np.array(ls) - np.array(bs)

print(f"\n=== chrF++ ===")
print(f"Large mean: {np.mean(ls):.2f}")
print(f"Base mean:  {np.mean(bs):.2f}")
print(f"Large wins: {(diff > 0).sum()}, Base wins: {(diff < 0).sum()}, Tie: {(diff == 0).sum()}")

# Largeが大幅に負けるケースTop10
large_sent["diff"] = diff
print("\n=== Largeが大幅に負けるTop10 ===")
worst = large_sent.nsmallest(10, "diff")
for idx, row in worst.iterrows():
    bi = base_sent.iloc[idx]
    print(f"\n  diff={row['diff']:.1f} (L={row['chrf']:.1f}, B={bi['chrf']:.1f})")
    print(f"  ref:   {str(row['reference'])[:150]}")
    print(f"  large: {str(row['prediction_clean'])[:150]}")
    print(f"  base:  {str(bi['prediction_clean'])[:150]}")
