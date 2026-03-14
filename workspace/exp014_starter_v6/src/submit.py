"""
exp014_starter_v6: Starter推論の完全再現
- takamichitoda/dpc-starter-infer と完全同一の推論ロジック
- beam4, max_length=512, early_stopping=True
- 後処理: strip + 空文字→"broken text" のみ
"""
import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ============================================================
# Paths (ローカル用 / Kaggle用はコメントで切り替え)
# ============================================================
# --- ローカル用 ---
MODEL_PATH = "workspace/exp014_starter_v6/results/last_model"
TEST_PATH = "datasets/raw/test.csv"
OUTPUT_PATH = "workspace/exp014_starter_v6/results/submission.csv"

# --- Kaggle用 ---
# MODEL_PATH = "/kaggle/input/akkadianmodels/pytorch/exp014_starter_v6/1/last_model"
# TEST_PATH = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
# OUTPUT_PATH = "submission.csv"

# ============================================================
# Config (starter-inferと完全同一)
# ============================================================
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_BEAMS = 4
PREFIX = "translate Akkadian to English: "

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# Model loading
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# ============================================================
# Dataset (starter-inferと完全同一)
# ============================================================
class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["transliteration"].astype(str).tolist()
        self.texts = [PREFIX + t for t in self.texts]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",  # starterと同一: 固定512パディング
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


# ============================================================
# Inference (starter-inferと完全同一)
# ============================================================
test_df = pd.read_csv(TEST_PATH)
print(f"Test samples: {len(test_df)}")

test_dataset = InferenceDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend([d.strip() for d in decoded])

# ============================================================
# Submission (starter-inferと完全同一)
# ============================================================
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": all_predictions,
})

submission["translation"] = submission["translation"].apply(
    lambda x: x if len(x) > 0 else "broken text"
)

submission.to_csv(OUTPUT_PATH, index=False)
print(f"Submission saved to {OUTPUT_PATH}")
print(f"Shape: {submission.shape}")
print(f"Empty translations: {(submission['translation'] == 'broken text').sum()}")
