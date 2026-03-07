"""
ByT5-small 推論スクリプト (exp002)
修正点: normalize_textを使わず生テキストで推論、padding=max_lengthは推論時は必要
"""
import os
import sys
import logging
import yaml
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
        # 正規化なし — 生テキストそのまま
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


def main():
    config_path = os.path.join(EXP_DIR, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = os.path.join(EXP_DIR, "results", "best_model")
    os.chdir(EXP_DIR)
    test_path = config["data"]["test_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    test_df = pd.read_csv(test_path)
    logger.info(f"Test samples: {len(test_df)}")

    prefix = config["model"]["prefix"]
    max_length = config["model"]["max_length"]
    num_beams = config["inference"]["num_beams"]
    batch_size = config["inference"]["batch_size"]

    dataset = InferenceDataset(
        test_df["transliteration"].astype(str).tolist(),
        tokenizer,
        max_length,
        prefix,
    )
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

    all_preds = postprocess_batch(all_preds)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "translation": all_preds,
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
