"""
exp022_train_cleaning: Kaggle提出用推論
- beam4, max_length=512, early_stopping=True
- repeat_cleanup後処理
- テスト入力のtransliterationにclean_transliteration()を適用
"""
import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ============================================================
# Paths — Kaggle上ではON_KAGGLEをTrueに変更するだけ
# ============================================================
ON_KAGGLE = False

if ON_KAGGLE:
    MODEL_BASE = "/kaggle/input/akkadianmodels/pytorch/exp022_train_cleaning/1"
    MODEL_PATH = f"{MODEL_BASE}/best_model"
    TEST_PATH = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    OUTPUT_PATH = "submission.csv"
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXP_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
    MODEL_PATH = os.path.join(EXP_DIR, "results", "best_model")
    TEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "test.csv")
    OUTPUT_PATH = os.path.join(EXP_DIR, "results", "submission.csv")

MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_BEAMS = 4
PREFIX = "translate Akkadian to English: "

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# Host推奨前処理（transliterationのみ、推論時に適用）
# ============================================================
DECIMAL_TO_FRACTION = {
    "0.5": "½", "0.25": "¼", "0.3333": "⅓", "0.6666": "⅔",
    "0.8333": "⅚", "0.75": "¾", "0.1666": "⅙", "0.625": "⅝",
}

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def clean_transliteration(text: str) -> str:
    """Host推奨のtransliteration前処理（train時と同一）"""
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    for decimal, fraction in sorted(DECIMAL_TO_FRACTION.items(), key=lambda x: -len(x[0])):
        text = text.replace(decimal, fraction)
    return text


def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return " ".join(words[:i+n])
    return text


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()


class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["transliteration"].astype(str).apply(clean_transliteration).tolist()
        self.texts = [PREFIX + t for t in self.texts]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text, max_length=MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


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
            input_ids=input_ids, attention_mask=attention_mask,
            max_length=MAX_LENGTH, num_beams=NUM_BEAMS, early_stopping=True,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend([d.strip() for d in decoded])

all_predictions = [repeat_cleanup(p) for p in all_predictions]


def extract_first_sentence(text: str) -> str:
    """テストは文レベルなので、予測を最初の1文に截断"""
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()


all_predictions = [extract_first_sentence(p) for p in all_predictions]

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
