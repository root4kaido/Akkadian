"""
exp023_full_preprocessing: Kaggle提出用推論 v3 (Sampling MBR)
- 3温度(0.6/0.8/1.05) × 2候補 = 6候補 → chrF++ MBR consensus
- 動的パディング + 長さソートで高速化
- ペナルティなし（repetition_penalty/length_penalty指定なし）
- repeat_cleanup + clean_translation 後処理
"""
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

# ============================================================
# Paths — Kaggle上ではON_KAGGLEをTrueに変更するだけ
# ============================================================
ON_KAGGLE = True

if ON_KAGGLE:
    MODEL_BASE = "/kaggle/input/models/nomorevotch/akkadianmodels/pytorch/exp034_st_pretrain_last/1/fold3"
    MODEL_PATH = f"{MODEL_BASE}/last_model"
    TEST_PATH = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"
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
PREFIX = "translate Akkadian to English: "

# MBR config
TEMPERATURES = [0.6, 0.8, 1.05]
NUM_SAMPLE_PER_TEMP = 2  # 3temp × 2 = 6候補

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# 前処理（train.pyと同一のclean_transliteration）
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
    """train.pyと同一のtransliteration前処理 + Turkish文字正規化"""
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    # Turkish文字正規化（テストデータに存在）
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


# ============================================================
# 出力後処理: train.pyと同一のclean_translation
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
    """train.pyと同一のtranslation後処理"""
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


# ============================================================
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()


# ============================================================
# 動的パディング + 長さソート
# ============================================================
class InferenceDataset(Dataset):
    """トークナイズのみ行い、パディングはcollate_fnで実施"""
    def __init__(self, df, tokenizer):
        texts = df["transliteration"].astype(str).apply(clean_transliteration).tolist()
        texts = [PREFIX + t for t in texts]
        self.items = []
        for t in texts:
            enc = tokenizer(t, max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
            self.items.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def dynamic_collate_fn(batch):
    """バッチ内最長に合わせてパディング"""
    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_id = tokenizer.pad_token_id or 0
    input_ids, attention_mask = [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask)}


# ============================================================
# MBR
# ============================================================
chrfpp = sacrebleu.metrics.CHRF(word_order=2)


def mbr_pick(candidates):
    """chrF++ consensus: 候補間のpairwiseスコア平均が最大の候補を選択"""
    cands = list(dict.fromkeys(candidates))  # dedup keeping order
    n = len(cands)
    if n <= 1:
        return cands[0] if cands else ""
    scores = []
    for i in range(n):
        s = sum(float(chrfpp.sentence_score(cands[i], [cands[j]]).score) for j in range(n) if j != i)
        scores.append(s / (n - 1))
    return cands[int(np.argmax(scores))]


# ============================================================
# Inference: Sampling MBR
# ============================================================
test_df = pd.read_csv(TEST_PATH)
print(f"Test samples: {len(test_df)}")

test_dataset = InferenceDataset(test_df, tokenizer)

# 長さソート → パディング効率UP → 元の順序は後で復元
lengths = [test_dataset.items[i]["input_ids"].size(0) for i in range(len(test_dataset))]
sorted_indices = sorted(range(len(test_dataset)), key=lambda i: lengths[i])
idx_map = {new: old for new, old in enumerate(sorted_indices)}

sorted_ds = torch.utils.data.Subset(test_dataset, sorted_indices)
test_loader = DataLoader(sorted_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_fn)

n_samples = len(test_dataset)

# Collect candidates: 3 temperatures × 2 candidates each = 6 per sample
all_candidates = [[] for _ in range(n_samples)]

for temp in TEMPERATURES:
    print(f"Sampling temp={temp}, n_ret={NUM_SAMPLE_PER_TEMP}...")
    sorted_samp = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"t={temp}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=MAX_LENGTH,
                do_sample=True, num_beams=1, top_p=0.9, temperature=temp,
                num_return_sequences=NUM_SAMPLE_PER_TEMP,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            sorted_samp.extend([d.strip() for d in decoded])
    # Assign to original indices
    for new_idx in range(n_samples):
        orig_idx = idx_map[new_idx]
        cands = sorted_samp[new_idx * NUM_SAMPLE_PER_TEMP: (new_idx + 1) * NUM_SAMPLE_PER_TEMP]
        all_candidates[orig_idx].extend(cands)

# MBR selection
print("MBR consensus selection...")
all_predictions = [mbr_pick(cands) for cands in tqdm(all_candidates, desc="MBR")]

# Post-processing
all_predictions = [repeat_cleanup(p) for p in all_predictions]
all_predictions = [clean_translation(p) for p in all_predictions]

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
