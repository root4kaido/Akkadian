"""
ByT5-small 推論スクリプト (exp004)
- Validationセットでgreedy推論 → 正確なCV算出
- beam比較もオプションで実行可能
- テスト推論 + submission.csv作成
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
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
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


def run_inference(model, tokenizer, texts, max_length, gen_kwargs, batch_size, device):
    """推論を実行（生成パラメータを外部から指定）"""
    dataset = InferenceDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend([d.strip() for d in decoded])

    return all_preds


def compute_cv(preds, refs, label=""):
    """CV指標を計算"""
    metric_chrf = evaluate.load("chrf")
    metric_bleu = evaluate.load("sacrebleu")

    chrf = metric_chrf.compute(predictions=preds, references=refs)["score"]
    bleu = metric_bleu.compute(
        predictions=preds, references=[[x] for x in refs]
    )["score"]
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0

    logger.info(f"CV {label}: chrF={chrf:.2f}, BLEU={bleu:.2f}, geo_mean={geo_mean:.2f}")
    return {"chrf": round(chrf, 2), "bleu": round(bleu, 2), "geo_mean": round(geo_mean, 2)}


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
    encoder_max_length = config["model"]["encoder_max_length"]
    decoder_max_length = config["model"]["decoder_max_length"]
    batch_size = config["inference"]["batch_size"]
    num_beams = config["inference"]["num_beams"]

    # === 1. Validation CV評価 ===
    logger.info("=== Validation CV Evaluation ===")
    from preprocess import prepare_data
    _, val_df = prepare_data(config)

    val_texts = val_df["input_text"].tolist()  # prefix already included
    val_ref = val_df["target_text"].tolist()

    # Greedy推論
    logger.info(f"Running greedy inference (num_beams={num_beams})...")
    greedy_kwargs = {"max_length": decoder_max_length, "num_beams": num_beams}
    if num_beams > 1:
        greedy_kwargs["early_stopping"] = config["inference"]["early_stopping"]

    raw_preds = run_inference(
        model, tokenizer, val_texts, encoder_max_length,
        greedy_kwargs, batch_size, device,
    )

    # raw（後処理なし）でCV評価
    cv_raw = compute_cv(raw_preds, val_ref, label=f"raw (beams={num_beams})")

    # 後処理ありでCV評価
    post_preds = postprocess_batch(raw_preds)
    cv_post = compute_cv(post_preds, val_ref, label=f"postprocessed (beams={num_beams})")

    # CV結果保存
    cv_results = {
        "raw": cv_raw,
        "postprocessed": cv_post,
        "num_beams": num_beams,
    }
    cv_path = os.path.join(EXP_DIR, "results", "cv_metrics.yaml")
    with open(cv_path, "w") as f:
        yaml.dump(cv_results, f, default_flow_style=False)
    logger.info(f"CV metrics saved to {cv_path}")

    # beam比較（num_beams=1の場合、beam4も試す）
    if num_beams == 1:
        logger.info("=== Beam4 comparison on val ===")
        beam_preds = run_inference(
            model, tokenizer, val_texts, encoder_max_length,
            {"max_length": decoder_max_length, "num_beams": 4, "early_stopping": True},
            batch_size, device,
        )
        cv_beam_raw = compute_cv(beam_preds, val_ref, label="raw (beams=4)")
        beam_post_preds = postprocess_batch(beam_preds)
        cv_beam_post = compute_cv(beam_post_preds, val_ref, label="postprocessed (beams=4)")

        cv_results["beam4_raw"] = cv_beam_raw
        cv_results["beam4_postprocessed"] = cv_beam_post
        with open(cv_path, "w") as f:
            yaml.dump(cv_results, f, default_flow_style=False)

    # Val予測結果保存
    val_results = pd.DataFrame({
        "source": val_texts,
        "reference": val_ref,
        "prediction_raw": raw_preds,
        "prediction_post": post_preds,
    })
    val_results_path = os.path.join(EXP_DIR, "results", "val_predictions.csv")
    val_results.to_csv(val_results_path, index=False)
    logger.info(f"Val predictions saved to {val_results_path}")

    # === 2. テスト推論 ===
    logger.info("=== Test Inference ===")
    test_path = config["data"]["test_path"]
    test_df = pd.read_csv(test_path)
    logger.info(f"Test samples: {len(test_df)}")

    test_texts = [prefix + str(t) for t in test_df["transliteration"].astype(str).tolist()]
    test_raw = run_inference(
        model, tokenizer, test_texts, encoder_max_length,
        {"max_length": decoder_max_length, "num_beams": num_beams},
        batch_size, device,
    )
    test_preds = postprocess_batch(test_raw)

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
