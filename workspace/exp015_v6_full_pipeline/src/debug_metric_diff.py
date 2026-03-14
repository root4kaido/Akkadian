"""training eval と eval_cv のメトリクス計算方法の差異を検証する"""
import os, sys, math
import pandas as pd, numpy as np
import sacrebleu, evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(EXP_DIR, "results")

# eval_cv の予測と参照を読み込む
preds_df = pd.read_csv(os.path.join(RESULTS_DIR, "val_predictions.csv"))
preds = preds_df["prediction_clean"].tolist()
refs = preds_df["reference"].tolist()

print(f"Samples: {len(preds)}")
print(f"Pred len: mean={np.mean([len(p.encode('utf-8')) for p in preds]):.0f}B")
print(f"Ref len:  mean={np.mean([len(r.encode('utf-8')) for r in refs]):.0f}B")

# === 方法1: eval_cv方式 (sacrebleu直接) ===
chrf_direct = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
bleu_direct = sacrebleu.corpus_bleu(preds, [refs])
geo_direct = math.sqrt(chrf_direct.score * bleu_direct.score) if chrf_direct.score > 0 and bleu_direct.score > 0 else 0
print(f"\n=== eval_cv方式 (sacrebleu直接, word_order=2) ===")
print(f"  chrF++={chrf_direct.score:.4f}, BLEU={bleu_direct.score:.4f}, geo={geo_direct:.4f}")

# === 方法2: training eval方式 (evaluate library) ===
metric_chrf = evaluate.load("chrf")
metric_bleu = evaluate.load("sacrebleu")

# training evalのcompute_metricsと同じ呼び方
chrf_eval = metric_chrf.compute(predictions=preds, references=refs)["score"]
bleu_eval = metric_bleu.compute(predictions=preds, references=[[r] for r in refs])["score"]
geo_eval = math.sqrt(chrf_eval * bleu_eval) if chrf_eval > 0 and bleu_eval > 0 else 0
print(f"\n=== training eval方式 (evaluate library, デフォルト) ===")
print(f"  chrF={chrf_eval:.4f}, BLEU={bleu_eval:.4f}, geo={geo_eval:.4f}")

# === 方法2b: evaluate + word_order=2を明示 ===
chrf_eval2 = metric_chrf.compute(predictions=preds, references=refs, word_order=2)["score"]
geo_eval2 = math.sqrt(chrf_eval2 * bleu_eval) if chrf_eval2 > 0 and bleu_eval > 0 else 0
print(f"\n=== training eval方式 + word_order=2 ===")
print(f"  chrF++={chrf_eval2:.4f}, BLEU={bleu_eval:.4f}, geo={geo_eval2:.4f}")

# === 方法3: sacrebleu直接でword_order=0 (chrF, not chrF++) ===
chrf_wo0 = sacrebleu.corpus_chrf(preds, [refs], word_order=0)
geo_wo0 = math.sqrt(chrf_wo0.score * bleu_direct.score) if chrf_wo0.score > 0 and bleu_direct.score > 0 else 0
print(f"\n=== sacrebleu word_order=0 (chrF, not chrF++) ===")
print(f"  chrF={chrf_wo0.score:.4f}, BLEU={bleu_direct.score:.4f}, geo={geo_wo0:.4f}")

# === 参照形式の違い ===
# evaluate libraryはreferences=list[str]を受け取る
# sacrebleuはreferences=list[list[str]]を受け取る
# training evalのcompute_metricsはreferences=decoded_labels (list[str])を渡している
# sacrebleu.corpus_bleuの方は [[r] for r in refs]
print(f"\n=== evaluate library chrf のデフォルトword_order確認 ===")
# 小さいサンプルで確認
test_preds = ["hello world"]
test_refs = ["hello world"]
c_default = metric_chrf.compute(predictions=test_preds, references=test_refs)["score"]
c_wo0 = metric_chrf.compute(predictions=test_preds, references=test_refs, word_order=0)["score"]
c_wo2 = metric_chrf.compute(predictions=test_preds, references=test_refs, word_order=2)["score"]
print(f"  default={c_default:.4f}, wo0={c_wo0:.4f}, wo2={c_wo2:.4f}")
