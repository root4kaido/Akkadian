"""
exp023: Logit平均アンサンブル推論
- 複数モデルのlogitsを平均してgreedy decodeする
- Kaggle上: ON_KAGGLE=True, MODEL_PATHS を調整
"""
import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
ON_KAGGLE = False

if ON_KAGGLE:
    MODEL_BASE = "/kaggle/input/akkadianmodels/pytorch"
    MODEL_PATHS = [
        f"{MODEL_BASE}/exp023_fold{i}/1/fold{i}/best_model" for i in range(5)
    ]
    TOKENIZER_PATH = MODEL_PATHS[0]
    TEST_PATH = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    OUTPUT_PATH = "submission.csv"
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXP_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
    MODEL_PATHS = [
        os.path.join(EXP_DIR, "results", f"fold{i}", "best_model") for i in range(5)
    ]
    TOKENIZER_PATH = MODEL_PATHS[0]
    TEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "test.csv")
    OUTPUT_PATH = os.path.join(EXP_DIR, "results", "submission_logit_ensemble.csv")

MAX_LENGTH = 512
BATCH_SIZE = 8  # メモリ節約のため小さめ
PREFIX = "translate Akkadian to English: "

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# 前処理（submit.pyと同一）
# ============================================================
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002


def _decimal_to_fraction(match):
    dec_str = match.group(0)
    try:
        value = float(dec_str)
    except ValueError:
        return dec_str
    int_part = int(value)
    frac_part = value - int_part
    if frac_part < 0.001:
        return dec_str
    best_frac = None
    best_dist = float('inf')
    for target, symbol in FRACTION_TARGETS.items():
        dist = abs(frac_part - target)
        if dist < best_dist:
            best_dist = dist
            best_frac = symbol
    if best_dist <= APPROX_TOLERANCE:
        if int_part == 0:
            return best_frac
        else:
            return f"{int_part} {best_frac}"
    return dec_str


SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def clean_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


# ============================================================
# 後処理（submit.pyと同一）
# ============================================================
ROMAN_TO_INT = {
    "XII": "12", "XI": "11", "VIII": "8", "VII": "7",
    "VI": "6", "IV": "4", "IX": "9", "III": "3",
    "II": "2", "X": "10", "V": "5", "I": "1",
}

MONTH_NAMES_TRANSLATION = {
    r"B[eē]lat[\s-]ekall[ie]m": "1",
    r"[Šš]a[\s-]sarr[aā]tim": "2",
    r"[Kk]en[aā]tim": "3",
    r"[Šš]a[\s-]k[eē]n[aā]tim": "3",
    r"Ma[hḫ]h?ur[\s-]il[iī]": "4",
    r"Ab[\s-]?[šš]arr[aā]ni": "5",
    r"[Aa]b[sš]arrani": "5",
    r"[Hh]ubur": "6",
    r"[Ṣṣ]ip['\u2019]?um": "7",
    r"[Qq]arr[aā]['\u2019]?[aā]tum": "8",
    r"[Qq]arr[aā]tum": "8",
    r"[Kk]an[wm]arta": "9",
    r"[Tt]e['\u2019\u02BE]?in[aā]tum": "10",
    r"[Tt][eē]['\u2019\u02BE]?in[aā]tum": "10",
    r"[Kk]uzall?[iu]m?": "11",
    r"[Aa]llan[aā]tum": "12",
}


def clean_translation(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = re.sub(r'\bfem\.\s*', '', text)
    text = re.sub(r'\bsing\.\s*', '', text)
    text = re.sub(r'\bpl\.\s*', '', text)
    text = re.sub(r'\bplural\b\s*', '', text)
    text = text.replace('(?)', '')
    text = re.sub(r'<<\s*>>', '', text)
    text = re.sub(r'<\s+>', '', text)
    text = re.sub(r'(?<!\.)\.\.(?!\.)', '', text)
    text = re.sub(r'\bxx?\b', '', text)
    text = re.sub(r'\bPN\b', '<gap>', text)
    text = re.sub(r'\b-gold\b', 'pašallum gold', text)
    text = re.sub(r'\b-tax\b', 'šadduātum tax', text)
    text = re.sub(r'\b-textiles\b', 'kutānum textiles', text)
    text = re.sub(r'(\S+)\s*/\s*\S+', r'\1', text)
    text = re.sub(r'\(m\)', '{m}', text)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    for roman, integer in sorted(ROMAN_TO_INT.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf'\bmonth\s+{roman}(?=[\s,.:;!?\)]|$)', f'month {integer}', text)
    for pattern, number in MONTH_NAMES_TRANSLATION.items():
        text = re.sub(rf'\bmonth\s+{pattern}\b', f'month {number}', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
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


def extract_first_sentence(text: str) -> str:
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()


# ============================================================
# Dataset
# ============================================================
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


# ============================================================
# Logit平均 greedy decode
# ============================================================
def logit_ensemble_generate(models, input_ids, attention_mask, tokenizer, max_length=512):
    """複数モデルのlogitsを平均してgreedy decodeする"""
    batch_size = input_ids.shape[0]
    # encoder出力を事前計算
    encoder_outputs_list = []
    for model in models:
        encoder_outputs_list.append(
            model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        )

    # decoder: <pad> (= decoder_start_token_id) から開始
    decoder_start_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    decoder_input_ids = torch.full(
        (batch_size, 1), decoder_start_id, dtype=torch.long, device=input_ids.device
    )

    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    generated_ids = decoder_input_ids

    for step in range(max_length - 1):
        # 各モデルのlogitsを取得して平均
        avg_logits = None
        for model, enc_out in zip(models, encoder_outputs_list):
            outputs = model(
                encoder_outputs=enc_out,
                attention_mask=attention_mask,
                decoder_input_ids=generated_ids,
            )
            logits = outputs.logits[:, -1, :]  # (batch, vocab)
            if avg_logits is None:
                avg_logits = logits
            else:
                avg_logits = avg_logits + logits
        avg_logits = avg_logits / len(models)

        # greedy: argmax
        next_token = avg_logits.argmax(dim=-1, keepdim=True)  # (batch, 1)

        # EOSを出したサンプルはpad埋め
        next_token[finished] = tokenizer.pad_token_id
        finished = finished | (next_token.squeeze(-1) == eos_id)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if finished.all():
            break

    return generated_ids


# ============================================================
# Main
# ============================================================
print(f"Loading {len(MODEL_PATHS)} models...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
models = []
for i, path in enumerate(MODEL_PATHS):
    print(f"  Loading model {i}: {path}")
    m = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
    m.eval()
    models.append(m)
print(f"All {len(models)} models loaded.")

test_df = pd.read_csv(TEST_PATH)
print(f"Test samples: {len(test_df)}")

test_dataset = InferenceDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Logit Ensemble Inference"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        output_ids = logit_ensemble_generate(
            models, input_ids, attention_mask, tokenizer, max_length=MAX_LENGTH
        )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_predictions.extend([d.strip() for d in decoded])

all_predictions = [repeat_cleanup(p) for p in all_predictions]
all_predictions = [clean_translation(p) for p in all_predictions]
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
