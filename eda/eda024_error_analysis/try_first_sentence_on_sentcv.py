"""
eda024: sent-CV予測(exp023, geo=34.35)にfirst_sentence截断を適用した場合の効果
"""
import re
import pandas as pd
import sacrebleu
from pathlib import Path

# sent-CV予測結果を読み込み
PRED_PATH = Path(__file__).resolve().parent.parent / "eda020_sent_level_cv" / "sent_level_predictions.csv"
df = pd.read_csv(PRED_PATH)

df["pred"] = df["prediction_clean"].astype(str)
df["ref"] = df["reference"].astype(str)

print(f"Total predictions: {len(df)}")


def extract_first_sentence(text: str) -> str:
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()


def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return " ".join(words[:i+n])
    return text


def compute_metrics(preds, refs):
    """コーパスレベルのchrF++, BLEU, geoを計算"""
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    geo = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0
    return chrf, bleu, geo


strategies = {
    "original (prediction_clean)": df["pred"].tolist(),
    "repeat_cleanup only": [repeat_cleanup(p) for p in df["pred"]],
    "first_sentence only": [extract_first_sentence(p) for p in df["pred"]],
    "repeat_cleanup + first_sentence": [extract_first_sentence(repeat_cleanup(p)) for p in df["pred"]],
}

refs = df["ref"].tolist()

print("=" * 70)
print("sent-CV (exp023) に対する後処理戦略の比較")
print("=" * 70)

for name, preds in strategies.items():
    chrf, bleu, geo = compute_metrics(preds, refs)
    pred_lens = [len(p.split()) for p in preds]
    ref_lens = [len(r.split()) for r in refs]
    avg_pred = sum(pred_lens) / len(pred_lens)
    avg_ref = sum(ref_lens) / len(ref_lens)
    ratio = sum(p / max(r, 1) for p, r in zip(pred_lens, ref_lens)) / len(pred_lens)
    print(f"\n{name}:")
    print(f"  chrF++={chrf:.2f}, BLEU={bleu:.2f}, geo={geo:.2f}")
    print(f"  avg pred={avg_pred:.1f}w, avg ref={avg_ref:.1f}w, ratio={ratio:.2f}")

# 繰り返し率も確認
print("\n" + "=" * 70)
print("繰り返し率")
print("=" * 70)

for name, preds in strategies.items():
    rep_count = 0
    for p in preds:
        words = p.split()
        if len(words) >= 6:
            for n in range(3, len(words) // 2 + 1):
                for i in range(len(words) - 2 * n + 1):
                    if words[i:i+n] == words[i+n:i+2*n]:
                        rep_count += 1
                        break
                else:
                    continue
                break
    print(f"  {name}: {rep_count}/{len(preds)} ({100*rep_count/len(preds):.1f}%)")
