"""異なる生成戦略の比較デバッグ"""
import os
import sys
import torch
import yaml
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
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
val_src = [t.replace(prefix, "", 1) for t in val_df["input_text"].tolist()]
val_ref = val_df["target_text"].tolist()


class InfDS(Dataset):
    def __init__(self, texts):
        self.texts = [prefix + str(t) for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(
            self.texts[idx], max_length=512, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


def run_eval(gen_kwargs, label):
    loader = DataLoader(InfDS(val_src), batch_size=16, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=label):
            out = model.generate(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                **gen_kwargs,
            )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            preds.extend([d.strip() for d in decoded])
    chrf = evaluate.load("chrf").compute(predictions=preds, references=val_ref)["score"]
    bleu = evaluate.load("sacrebleu").compute(
        predictions=preds, references=[[x] for x in val_ref]
    )["score"]
    geo = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    print(f"{label}: chrF={chrf:.2f}, BLEU={bleu:.2f}, geo_mean={geo:.2f}")


run_eval({"max_length": 512}, "greedy")
run_eval({"max_length": 512, "num_beams": 4, "early_stopping": True}, "beam4+ES")
run_eval({"max_length": 512, "num_beams": 4, "early_stopping": False}, "beam4-noES")
run_eval({"max_length": 512, "num_beams": 4, "early_stopping": True, "no_repeat_ngram_size": 3}, "beam4+NR3")
run_eval({"max_length": 512, "num_beams": 4, "early_stopping": True, "no_repeat_ngram_size": 3, "length_penalty": 1.0}, "beam4+NR3+LP1")
