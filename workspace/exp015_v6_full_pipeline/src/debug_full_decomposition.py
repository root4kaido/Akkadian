"""
training eval (geo=48) vs eval_cv (geo=43) の差を完全分解する。

因子:
1. 入力: full vs 200B truncated
2. デコーディング: greedy vs beam4
3. 参照: 512B-tokenizer-truncated vs first-sentence
4. メトリクス: evaluate library (chrF, word_order=0) vs sacrebleu直接 (chrF++, word_order=2)

Training eval条件: full input + greedy + 512B ref + evaluate library
eval_cv条件: 200B input + beam4 + first-sent ref + sacrebleu直接
"""
import os, sys, re, json, math, logging
import pandas as pd, numpy as np, torch, yaml
import sacrebleu, evaluate
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

with open(os.path.join(EXP_DIR, "config.yaml")) as f:
    config = yaml.safe_load(f)
MAX_LENGTH = config["model"]["max_length"]
SEED = config["training"]["seed"]
PREFIX = config["model"]["prefix"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Val split
df = pd.read_csv(os.path.join(EXP_DIR, config["data"]["train_path"]))
df = df[(df["transliteration"].astype(str).str.len() > 0) & (df["translation"].astype(str).str.len() > 0)]
_, val_split = train_test_split(df, test_size=config["training"]["val_ratio"], random_state=SEED)

# PN/GN tags
with open(os.path.join(EXP_DIR, "dataset", "form_type_dict.json")) as f:
    form_tag_dict = json.load(f)

def tag(text):
    return " ".join(f"{t}[{form_tag_dict[t]}]" if t in form_tag_dict else t for t in text.split())

def extract_first_sentence(text):
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()

def truncate_200B(text):
    enc = str(text).encode('utf-8')
    if len(enc) <= 200: return str(text)
    trunc = enc[:200].decode('utf-8', errors='ignore')
    last = trunc.rfind(' ')
    return trunc[:last].strip() if last > 0 else trunc.strip()

def has_repetition(text, mr=3):
    w = str(text).split()
    for i in range(len(w)-mr):
        if " ".join(w[i:i+mr]) in " ".join(w[i+mr:]): return True
    return False

# 入力2種: full, 200B
inputs_full = [PREFIX + tag(str(row["transliteration"])) for _, row in val_split.iterrows()]
inputs_200B = [PREFIX + tag(truncate_200B(str(row["transliteration"]))) for _, row in val_split.iterrows()]

# 参照3種
refs_first_sent = [extract_first_sentence(str(row["translation"])) for _, row in val_split.iterrows()]
refs_full = [str(row["translation"]) for _, row in val_split.iterrows()]

# 512B tokenizer truncation (training evalと同一: tokenizer(text, max_length=512, truncation=True)してdecode)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

refs_512B_tok = []
for _, row in val_split.iterrows():
    t = str(row["translation"])
    tok = tokenizer(t, max_length=MAX_LENGTH, truncation=True)
    decoded = tokenizer.decode(tok["input_ids"], skip_special_tokens=True)
    refs_512B_tok.append(decoded)

logger.info(f"Val samples: {len(val_split)}")
logger.info(f"Input full: mean={np.mean([len(i.encode('utf-8')) for i in inputs_full]):.0f}B")
logger.info(f"Input 200B: mean={np.mean([len(i.encode('utf-8')) for i in inputs_200B]):.0f}B")
logger.info(f"Ref first-sent: mean={np.mean([len(r.encode('utf-8')) for r in refs_first_sent]):.0f}B")
logger.info(f"Ref 512B-tok:   mean={np.mean([len(r.encode('utf-8')) for r in refs_512B_tok]):.0f}B")
logger.info(f"Ref full:       mean={np.mean([len(r.encode('utf-8')) for r in refs_full]):.0f}B")

# Inference
class DS(TorchDataset):
    def __init__(self, texts):
        self.enc = tokenizer(texts, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()}

def run_inference(inputs, num_beams, label):
    loader = DataLoader(DS(inputs), batch_size=4, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=label):
            out = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_length=MAX_LENGTH,
                num_beams=num_beams,
                early_stopping=True if num_beams > 1 else False,
            )
            preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    return preds

# メトリクス計算2方式
metric_chrf_eval = evaluate.load("chrf")
metric_bleu_eval = evaluate.load("sacrebleu")

def calc_metrics_training_eval(preds, refs, label):
    """training evalと同じ方式: evaluate library, chrF default (word_order=0)"""
    chrf = metric_chrf_eval.compute(predictions=preds, references=refs)["score"]
    bleu = metric_bleu_eval.compute(predictions=preds, references=[[r] for r in refs])["score"]
    geo = math.sqrt(chrf * bleu) if chrf > 0 and bleu > 0 else 0
    rep = 100 * sum(has_repetition(p) for p in preds) / len(preds)
    pred_len = np.mean([len(p.encode('utf-8')) for p in preds])
    ref_len = np.mean([len(r.encode('utf-8')) for r in refs])
    logger.info(f"  [train_eval方式] {label}: chrF={chrf:.2f}, BLEU={bleu:.2f}, geo={geo:.2f}, rep={rep:.1f}%, pred={pred_len:.0f}B, ref={ref_len:.0f}B")
    return geo

def calc_metrics_eval_cv(preds, refs, label):
    """eval_cvと同じ方式: sacrebleu直接, chrF++ (word_order=2)"""
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0
    rep = 100 * sum(has_repetition(p) for p in preds) / len(preds)
    pred_len = np.mean([len(p.encode('utf-8')) for p in preds])
    ref_len = np.mean([len(r.encode('utf-8')) for r in refs])
    logger.info(f"  [eval_cv方式  ] {label}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep:.1f}%, pred={pred_len:.0f}B, ref={ref_len:.0f}B")
    return geo

# === 4種の推論 ===
logger.info("\n" + "="*70)
logger.info("Generating predictions for 4 conditions...")

preds_greedy_full = run_inference(inputs_full, num_beams=1, label="greedy+full")
preds_beam4_full = run_inference(inputs_full, num_beams=4, label="beam4+full")
preds_greedy_200B = run_inference(inputs_200B, num_beams=1, label="greedy+200B")
preds_beam4_200B = run_inference(inputs_200B, num_beams=4, label="beam4+200B")

# === 全条件の評価 ===
all_preds = {
    "greedy+full": preds_greedy_full,
    "beam4+full": preds_beam4_full,
    "greedy+200B": preds_greedy_200B,
    "beam4+200B": preds_beam4_200B,
}
all_refs = {
    "512B-tok": refs_512B_tok,
    "first-sent": refs_first_sent,
}

logger.info("\n" + "="*70)
logger.info("=== 完全因子分解: 4入力×2参照×2メトリクス ===")

results = []
for pred_name, preds in all_preds.items():
    for ref_name, refs in all_refs.items():
        label = f"{pred_name} vs {ref_name}"
        logger.info(f"\n--- {label} ---")
        geo_te = calc_metrics_training_eval(preds, refs, label)
        geo_cv = calc_metrics_eval_cv(preds, refs, label)
        results.append({
            "pred": pred_name, "ref": ref_name,
            "geo_train_eval": round(geo_te, 2),
            "geo_eval_cv": round(geo_cv, 2),
            "diff": round(geo_te - geo_cv, 2),
        })

# サマリーテーブル
logger.info("\n" + "="*70)
logger.info("=== サマリー ===")
logger.info(f"{'pred':<16} {'ref':<12} {'geo(train_eval)':<16} {'geo(eval_cv)':<14} {'diff':<8}")
logger.info("-" * 70)
for r in results:
    logger.info(f"{r['pred']:<16} {r['ref']:<12} {r['geo_train_eval']:<16} {r['geo_eval_cv']:<14} {r['diff']:<8}")

logger.info("\n=== 因子分解 ===")
# Training eval条件: greedy+full, 512B-tok ref, train_eval方式
# eval_cv条件: beam4+200B, first-sent ref, eval_cv方式
te_geo = [r for r in results if r["pred"]=="greedy+full" and r["ref"]=="512B-tok"][0]["geo_train_eval"]
cv_geo = [r for r in results if r["pred"]=="beam4+200B" and r["ref"]=="first-sent"][0]["geo_eval_cv"]
logger.info(f"Training eval再現 (greedy+full, 512B-tok, train_eval方式): {te_geo}")
logger.info(f"eval_cv再現 (beam4+200B, first-sent, eval_cv方式): {cv_geo}")
logger.info(f"Total gap: {te_geo - cv_geo:.2f}")

# Step by step decomposition
# Start from eval_cv条件, change one factor at a time
step0 = cv_geo
step1 = [r for r in results if r["pred"]=="beam4+200B" and r["ref"]=="first-sent"][0]["geo_train_eval"]
step2 = [r for r in results if r["pred"]=="greedy+200B" and r["ref"]=="first-sent"][0]["geo_train_eval"]
step3 = [r for r in results if r["pred"]=="greedy+full" and r["ref"]=="first-sent"][0]["geo_train_eval"]
step4 = [r for r in results if r["pred"]=="greedy+full" and r["ref"]=="512B-tok"][0]["geo_train_eval"]

logger.info(f"\nStep 0 (eval_cv条件):                    geo={step0:.2f}")
logger.info(f"Step 1 (+メトリクス変更 chrF++→chrF):      geo={step1:.2f} (diff={step1-step0:+.2f})")
logger.info(f"Step 2 (+デコーディング beam4→greedy):     geo={step2:.2f} (diff={step2-step1:+.2f})")
logger.info(f"Step 3 (+入力 200B→full):                  geo={step3:.2f} (diff={step3-step2:+.2f})")
logger.info(f"Step 4 (+参照 first-sent→512B-tok):       geo={step4:.2f} (diff={step4-step3:+.2f})")
logger.info(f"Total: {step0:.2f} → {step4:.2f} (gap={step4-step0:+.2f})")
