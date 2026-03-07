"""推論時のパディング方式による差異を検証"""
import os
import sys
import torch
import yaml
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

os.chdir(EXP_DIR)
from preprocess import prepare_data

with open(os.path.join(EXP_DIR, "config.yaml")) as f:
    config = yaml.safe_load(f)

model_path = os.path.join(EXP_DIR, "results", "best_model")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")
model.eval()

_, val_df = prepare_data(config)
prefix = config["model"]["prefix"]
val_texts = val_df["input_text"].tolist()  # prefix already included
val_ref = val_df["target_text"].tolist()


def eval_one_by_one(label, gen_kwargs):
    """1サンプルずつ推論（パディング不要）"""
    preds = []
    for text in tqdm(val_texts, desc=label):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        preds.append(pred)
    chrf = evaluate.load("chrf").compute(predictions=preds, references=val_ref)["score"]
    bleu = evaluate.load("sacrebleu").compute(
        predictions=preds, references=[[x] for x in val_ref]
    )["score"]
    geo = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    print(f"{label}: chrF={chrf:.2f}, BLEU={bleu:.2f}, geo_mean={geo:.2f}")
    return preds


# パディングなし（1サンプルずつ） greedy
preds_greedy = eval_one_by_one("greedy_nopad", {"max_length": 512})

# パディングなし beam4
preds_beam = eval_one_by_one("beam4_nopad", {"max_length": 512, "num_beams": 4, "early_stopping": True})

# サンプル比較
print("\n--- Sample comparison ---")
for i in range(min(3, len(preds_greedy))):
    print(f"[{i}] Greedy: {preds_greedy[i][:150]}")
    print(f"[{i}] Beam4:  {preds_beam[i][:150]}")
    print(f"[{i}] Ref:    {val_ref[i][:150]}")
    print()
