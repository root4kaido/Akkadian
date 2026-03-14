"""
eda024: 予測を最初のピリオドで截断した場合のスコア変化
"""
import re
import pandas as pd
import sacrebleu
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent
df = pd.read_csv(RESULTS_DIR / "val_predictions_scored.csv")

PREFIX = "translate Akkadian to English: "
df["pred"] = df["prediction_clean"].astype(str)
df["ref"] = df["reference"].astype(str)

def truncate_at_first_period(text):
    """最初のピリオド（文末）で截断"""
    # ピリオド+空白 or ピリオド+文末 で切る
    m = re.search(r'[.!?](?:\s|$)', text)
    if m:
        return text[:m.end()].strip()
    return text.strip()

def truncate_at_first_sentence(text):
    """最初の文を抽出（参照側と同じextract_first_sentenceロジック）"""
    m = re.search(r'^(.*?[.!?])(?:\s|$)', text)
    return m.group(1).strip() if m else text.strip()

# 3パターンで比較
strategies = {
    "original": df["pred"],
    "first_period": df["pred"].apply(truncate_at_first_period),
    "first_sentence": df["pred"].apply(truncate_at_first_sentence),
}

print("=" * 70)
print("予測截断戦略の比較")
print("=" * 70)

for name, preds in strategies.items():
    chrf_scores = []
    bleu_scores = []
    for pred, ref in zip(preds, df["ref"]):
        chrf = sacrebleu.sentence_chrf(pred, [ref], word_order=2).score
        bleu = sacrebleu.sentence_bleu(pred, [ref]).score
        chrf_scores.append(chrf)
        bleu_scores.append(bleu)

    mean_chrf = sum(chrf_scores) / len(chrf_scores)
    mean_bleu = sum(bleu_scores) / len(bleu_scores)
    geo = (mean_chrf * mean_bleu) ** 0.5 if mean_chrf > 0 and mean_bleu > 0 else 0

    # 長さ比較
    pred_lens = preds.apply(lambda x: len(x.split()))
    ref_lens = df["ref"].apply(lambda x: len(x.split()))
    ratio = (pred_lens / ref_lens.clip(lower=1)).mean()

    print(f"\n{name}:")
    print(f"  chrF++={mean_chrf:.2f}, BLEU={mean_bleu:.2f}, geo={geo:.2f}")
    print(f"  avg pred len={pred_lens.mean():.1f}w, avg ref len={ref_lens.mean():.1f}w, ratio={ratio:.2f}")

# 個別ケースでの変化を見る（長すぎだったものがどう変わるか）
print("\n" + "=" * 70)
print("長すぎ(ratio>1.5)だったケースの変化")
print("=" * 70)

df["pred_trunc"] = strategies["first_sentence"]
df["pred_len_orig"] = df["pred"].apply(lambda x: len(x.split()))
df["pred_len_trunc"] = df["pred_trunc"].apply(lambda x: len(x.split()))
df["ref_len"] = df["ref"].apply(lambda x: len(x.split()))
df["ratio_orig"] = df["pred_len_orig"] / df["ref_len"].clip(lower=1)
df["ratio_trunc"] = df["pred_len_trunc"] / df["ref_len"].clip(lower=1)

long_orig = df[df["ratio_orig"] > 1.5]
print(f"\n元の長すぎ: {len(long_orig)}件")

improved = 0
worsened = 0
for _, row in long_orig.iterrows():
    chrf_orig = sacrebleu.sentence_chrf(row["pred"], [row["ref"]], word_order=2).score
    chrf_trunc = sacrebleu.sentence_chrf(row["pred_trunc"], [row["ref"]], word_order=2).score
    if chrf_trunc > chrf_orig + 1:
        improved += 1
    elif chrf_trunc < chrf_orig - 1:
        worsened += 1

print(f"  截断で改善(+1pt以上): {improved}件")
print(f"  截断で悪化(-1pt以上): {worsened}件")
print(f"  変化なし/微小: {len(long_orig) - improved - worsened}件")

# 具体例: 最も改善したケース
print("\n截断で最も改善した5ケース:")
improvements = []
for _, row in long_orig.iterrows():
    chrf_orig = sacrebleu.sentence_chrf(row["pred"], [row["ref"]], word_order=2).score
    chrf_trunc = sacrebleu.sentence_chrf(row["pred_trunc"], [row["ref"]], word_order=2).score
    improvements.append((chrf_trunc - chrf_orig, row))

improvements.sort(key=lambda x: -x[0])
for delta, row in improvements[:5]:
    print(f"\n  delta={delta:+.1f}pt")
    print(f"    Ref:   {row['ref'][:120]}")
    print(f"    Orig:  {row['pred'][:120]}")
    print(f"    Trunc: {row['pred_trunc'][:120]}")
