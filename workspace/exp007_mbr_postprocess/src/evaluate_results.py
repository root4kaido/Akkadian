"""評価スクリプト: val_predictions.csvからスコア算出"""
import re
import math
import pandas as pd
import sacrebleu

def extract_first_sentence(text: str) -> str:
    """英語テキストの最初の文を抽出"""
    text = str(text).strip()
    pattern = r'(?<=[.!?])\s+'
    parts = re.split(pattern, text, maxsplit=1)
    return parts[0].strip() if parts else text

def evaluate(preds, refs, label=""):
    """chrF++, BLEU, geometric meanを計算"""
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    print(f"{label:30s}  chrF++={chrf.score:.2f}  BLEU={bleu.score:.2f}  geo_mean={geo:.2f}")
    return chrf.score, bleu.score, geo

df = pd.read_csv("/home/user/work/Akkadian/workspace/exp007_mbr_postprocess/results/val_predictions.csv")

refs = df["translation"].astype(str).tolist()
cols = ["greedy_pred", "mbr_pred", "mbr_post_pred", "greedy_post_pred"]

print("=" * 80)
print("=== Doc-level CV ===")
print("=" * 80)
for col in cols:
    preds = df[col].astype(str).tolist()
    evaluate(preds, refs, col)

print()
print("=" * 80)
print("=== Sent-level CV (first sentence) ===")
print("=" * 80)
sent_refs = [extract_first_sentence(r) for r in refs]
for col in cols:
    sent_preds = [extract_first_sentence(str(p)) for p in df[col].tolist()]
    evaluate(sent_preds, sent_refs, col)

# Translation Memory stats
tm_exact = sum(1 for g, gp in zip(df["greedy_pred"].tolist(), df["greedy_post_pred"].tolist()) if g != gp)
print(f"\nTranslation Memory hits: {tm_exact}/{len(df)} ({100*tm_exact/len(df):.1f}%)")

# Repeated output check
def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words)-min_repeat):
        chunk = " ".join(words[i:i+min_repeat])
        rest = " ".join(words[i+min_repeat:])
        if chunk in rest:
            return True
    return False

for col in cols:
    n_rep = sum(1 for p in df[col].tolist() if has_repetition(str(p)))
    print(f"Repetitions in {col}: {n_rep}/{len(df)} ({100*n_rep/len(df):.1f}%)")
