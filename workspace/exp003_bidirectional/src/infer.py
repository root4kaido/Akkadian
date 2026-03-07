"""
ByT5-small 推論スクリプト (exp003)
- テスト推論 + submission.csv作成
- Validationセットでのビームサーチ推論 → 正確なCV算出
"""
import os
import sys
import logging
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)
from postprocess import postprocess_batch


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, prefix):
        self.texts = [prefix + str(t) for t in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


def run_inference(model, tokenizer, texts, config, prefix, device):
    """ビームサーチ推論を実行"""
    max_length = config["model"]["max_length"]
    num_beams = config["inference"]["num_beams"]
    batch_size = config["inference"]["batch_size"]

    dataset = InferenceDataset(texts, tokenizer, max_length, prefix)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=config["inference"]["early_stopping"],
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend([d.strip() for d in decoded])

    return postprocess_batch(all_preds)


def main():
    config_path = os.path.join(EXP_DIR, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = os.path.join(EXP_DIR, "results", "best_model")
    os.chdir(EXP_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    prefix = config["model"]["prefix"]

    # === 1. Validation CV評価（ビームサーチ） ===
    logger.info("=== Validation CV Evaluation (beam search) ===")
    from preprocess import prepare_data
    _, val_df = prepare_data(config)

    # val_dfからprefixなしのtransliterationを取得
    val_src = [t.replace(prefix, "", 1) for t in val_df["input_text"].tolist()]
    val_ref = val_df["target_text"].tolist()

    val_preds = run_inference(model, tokenizer, val_src, config, prefix, device)

    # CV計算
    metric_chrf = evaluate.load("chrf")
    metric_bleu = evaluate.load("sacrebleu")

    chrf = metric_chrf.compute(predictions=val_preds, references=val_ref)["score"]
    bleu = metric_bleu.compute(
        predictions=val_preds, references=[[x] for x in val_ref]
    )["score"]
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0

    logger.info(f"CV (beam search): chrF={chrf:.2f}, BLEU={bleu:.2f}, geo_mean={geo_mean:.2f}")

    # CV結果保存
    cv_results = {
        "chrf": round(chrf, 2),
        "bleu": round(bleu, 2),
        "geo_mean": round(geo_mean, 2),
        "num_beams": config["inference"]["num_beams"],
        "postprocess": True,
    }
    cv_path = os.path.join(EXP_DIR, "results", "cv_beam_metrics.yaml")
    with open(cv_path, "w") as f:
        yaml.dump(cv_results, f, default_flow_style=False)
    logger.info(f"CV metrics saved to {cv_path}")

    # Val予測結果保存
    val_results = pd.DataFrame({
        "source": val_src,
        "reference": val_ref,
        "prediction": val_preds,
    })
    val_results_path = os.path.join(EXP_DIR, "results", "val_predictions.csv")
    val_results.to_csv(val_results_path, index=False)
    logger.info(f"Val predictions saved to {val_results_path}")

    # === 2. テスト推論 ===
    logger.info("=== Test Inference ===")
    test_path = config["data"]["test_path"]
    test_df = pd.read_csv(test_path)
    logger.info(f"Test samples: {len(test_df)}")

    test_preds = run_inference(
        model, tokenizer,
        test_df["transliteration"].astype(str).tolist(),
        config, prefix, device,
    )

    submission = pd.DataFrame({
        "id": test_df["id"],
        "translation": test_preds,
    })

    assert len(submission) == len(test_df), f"Row count mismatch: {len(submission)} vs {len(test_df)}"
    assert list(submission.columns) == ["id", "translation"], f"Column mismatch: {list(submission.columns)}"
    assert submission["translation"].isna().sum() == 0, "Found NaN translations"

    output_path = os.path.join(EXP_DIR, "results", "submission.csv")
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved: {output_path} ({len(submission)} rows)")
    for i in range(min(3, len(submission))):
        logger.info(f"  [{i}] {submission.iloc[i]['translation'][:100]}")


if __name__ == "__main__":
    main()
