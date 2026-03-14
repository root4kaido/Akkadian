"""予測CSVからchrF++/BLEU/geoを再計算する"""
import sys
import pandas as pd
import evaluate
import re

metric_chrf = evaluate.load("chrf")
metric_bleu = evaluate.load("sacrebleu")

def repeat_cleanup(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'\b(\w+(?:\s+\w+){0,2}?)(?:\s+\1){2,}\b', r'\1', text)
    return text

def calc_metrics(preds, refs):
    preds_clean = [repeat_cleanup(str(p)) for p in preds]
    chrf = metric_chrf.compute(predictions=preds_clean, references=[str(r) for r in refs])["score"]
    bleu = metric_bleu.compute(predictions=preds_clean, references=[[str(r)] for r in refs])["score"]
    geo = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    return chrf, bleu, geo

exp_name = sys.argv[1] if len(sys.argv) > 1 else "exp023"
base_dir = "/home/user/work/Akkadian/eda/eda020_sent_level_cv"

print(f"{'fold':>5} | {'sent-chrF':>10} {'sent-BLEU':>10} {'sent-geo':>9} | {'doc-chrF':>10} {'doc-BLEU':>10} {'doc-geo':>9}")
print("-" * 80)

sent_geos = []
doc_geos = []

for fold in range(5):
    sent_path = f"{base_dir}/{exp_name}_gkf_fold{fold}_last_sent_predictions.csv"
    doc_path = f"{base_dir}/{exp_name}_gkf_fold{fold}_last_doc_predictions.csv"
    try:
        sent_df = pd.read_csv(sent_path)
        doc_df = pd.read_csv(doc_path)

        s_chrf, s_bleu, s_geo = calc_metrics(sent_df["prediction_raw"], sent_df["reference"])
        d_chrf, d_bleu, d_geo = calc_metrics(doc_df["prediction_raw"], doc_df["reference"])

        sent_geos.append(s_geo)
        doc_geos.append(d_geo)

        print(f"fold{fold} | {s_chrf:10.2f} {s_bleu:10.2f} {s_geo:9.2f} | {d_chrf:10.2f} {d_bleu:10.2f} {d_geo:9.2f}")
    except Exception as e:
        print(f"fold{fold} | ERROR: {e}")

if sent_geos:
    import numpy as np
    print("-" * 80)
    print(f"{'mean':>5} | {'':>10} {'':>10} {np.mean(sent_geos):9.2f} | {'':>10} {'':>10} {np.mean(doc_geos):9.2f}")
    print(f"{'std':>5} | {'':>10} {'':>10} {np.std(sent_geos):9.2f} | {'':>10} {'':>10} {np.std(doc_geos):9.2f}")
