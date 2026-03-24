"""
exp023: Kaggle提出用推論 v7 (2-model ensemble + eng_MBR_chrf)
- Model A: exp034_st_pretrain (ByT5-base) beam4 1-best
- Model B: s1_exp007_large_lr1e4 (ByT5-large) beam4 1-best
- 英語候補同士のpairwise chrF++ (MBR) で最良候補選択
- 逆翻訳不要（forward beam4のみ）
- モデルは1つずつロード→推論→保存→解放（メモリ効率）
- 動的パディング + 長さソート
- repeat_cleanup + clean_translation 後処理
"""
import os
import re
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

# ============================================================
# Paths
# ============================================================
ON_KAGGLE = True

if ON_KAGGLE:
    MODEL_A_BASE = "/kaggle/input/models/nomorevotch/akkadianmodels/pytorch/exp034_st_pretrain_last/1/fold3"
    MODEL_A_PATH = f"{MODEL_A_BASE}/last_model"
    MODEL_B_BASE = "/kaggle/input/models/nomorevotch/akkadianmodels/pytorch/s1_exp007_large_lr1e4_last/1/fold3"
    MODEL_B_PATH = f"{MODEL_B_BASE}/last_model"
    TEST_PATH = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"
    OUTPUT_PATH = "submission.csv"
    CACHE_DIR = "."
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXP_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
    MODEL_A_PATH = os.path.join(PROJECT_ROOT, "workspace", "exp034_st_pretrain", "results", "fold3", "last_model")
    MODEL_B_PATH = os.path.join(PROJECT_ROOT, "workspace", "s1_exp007_large_lr1e4", "results", "fold3", "last_model")
    TEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "test.csv")
    OUTPUT_PATH = os.path.join(EXP_DIR, "results", "submission_v7_ensemble_mbr.csv")
    CACHE_DIR = os.path.join(EXP_DIR, "results")

MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_BEAMS = 4
PREFIX_FWD = "translate Akkadian to English: "

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# 前処理
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
    best_frac, best_dist = None, float('inf')
    for target, symbol in FRACTION_TARGETS.items():
        dist = abs(frac_part - target)
        if dist < best_dist:
            best_dist, best_frac = dist, symbol
    if best_dist <= APPROX_TOLERANCE:
        return best_frac if int_part == 0 else f"{int_part} {best_frac}"
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
# 出力後処理
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


# ============================================================
# 動的パディング + 長さソート
# ============================================================
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer):
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


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id or 0
    def dynamic_collate_fn(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids, attention_mask = [], []
        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask)}
    return dynamic_collate_fn


def generate_sorted(dataset, tokenizer, model, n_samples, desc,
                    num_beams=4, batch_size=BATCH_SIZE):
    """Generate with length-sorted batching, return in original order."""
    lengths = [dataset.items[i]["input_ids"].size(0) for i in range(len(dataset))]
    sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(dataset, sorted_indices)
    collate_fn = make_collate_fn(tokenizer)
    loader = DataLoader(sorted_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=num_beams, early_stopping=True,
            )
            sorted_preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    preds = [""] * n_samples
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    return preds


# ============================================================
# eng_MBR_chrf: 英語候補同士のpairwise chrF++で選択
# ============================================================
chrfpp = sacrebleu.metrics.CHRF(word_order=2)


def eng_mbr_pick(candidates):
    """Pick candidate with highest average pairwise chrF++ vs other candidates."""
    # Deduplicate
    seen = {}
    unique_cands = []
    for c in candidates:
        if c not in seen:
            seen[c] = True
            unique_cands.append(c)
    if len(unique_cands) <= 1:
        return unique_cands[0] if unique_cands else ""
    nc = len(unique_cands)
    scores = []
    for i in range(nc):
        s = sum(float(chrfpp.sentence_score(unique_cands[i], [unique_cands[j]]).score) / 100.0
                for j in range(nc) if j != i)
        scores.append(s / (nc - 1))
    return unique_cands[int(np.argmax(scores))]


# ============================================================
# Main
# ============================================================
test_df = pd.read_csv(TEST_PATH)
n_samples = len(test_df)
print(f"Test samples: {n_samples}")

source_texts = test_df["transliteration"].astype(str).apply(clean_transliteration).tolist()
fwd_texts = [PREFIX_FWD + t for t in source_texts]

# ============================================================
# Step 1: Model A — forward translate only
# ============================================================
print("=" * 60)
print("=== Step 1: Model A (exp034 ByT5-base) ===")
print("=" * 60)

tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A_PATH)
model_a = AutoModelForSeq2SeqLM.from_pretrained(MODEL_A_PATH).to(DEVICE)
model_a.eval()
print(f"Model A loaded from {MODEL_A_PATH}")

# Forward translation (beam4)
fwd_dataset_a = InferenceDataset(fwd_texts, tokenizer_a)
preds_a = generate_sorted(fwd_dataset_a, tokenizer_a, model_a, n_samples, desc="A fwd beam4")
print(f"  A fwd[0]: {preds_a[0][:80]}...")

# Save to file
cache_a = {"predictions": preds_a}
with open(os.path.join(CACHE_DIR, "model_a_cache.json"), "w") as f:
    json.dump(cache_a, f, ensure_ascii=False)
print("Model A results cached.")

# Release memory
del model_a, tokenizer_a, fwd_dataset_a
torch.cuda.empty_cache()
print("Model A released from GPU.")

# ============================================================
# Step 2: Model B — forward translate only
# ============================================================
print("=" * 60)
print("=== Step 2: Model B (s1_exp007 ByT5-large) ===")
print("=" * 60)

tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B_PATH)
model_b = AutoModelForSeq2SeqLM.from_pretrained(MODEL_B_PATH).to(DEVICE)
model_b.eval()
print(f"Model B loaded from {MODEL_B_PATH}")

# Forward translation (beam4)
fwd_dataset_b = InferenceDataset(fwd_texts, tokenizer_b)
preds_b = generate_sorted(fwd_dataset_b, tokenizer_b, model_b, n_samples, desc="B fwd beam4")
print(f"  B fwd[0]: {preds_b[0][:80]}...")

# Save to file
cache_b = {"predictions": preds_b}
with open(os.path.join(CACHE_DIR, "model_b_cache.json"), "w") as f:
    json.dump(cache_b, f, ensure_ascii=False)
print("Model B results cached.")

# Release memory
del model_b, tokenizer_b, fwd_dataset_b
torch.cuda.empty_cache()
print("Model B released from GPU.")

# ============================================================
# Step 3: eng_MBR_chrf rerank (2 candidates)
# ============================================================
print("=" * 60)
print("=== Step 3: eng_MBR_chrf rerank ===")
print("=" * 60)

# Reload from cache (in case of restart)
with open(os.path.join(CACHE_DIR, "model_a_cache.json"), "r") as f:
    cache_a = json.load(f)
with open(os.path.join(CACHE_DIR, "model_b_cache.json"), "r") as f:
    cache_b = json.load(f)

preds_a = cache_a["predictions"]
preds_b = cache_b["predictions"]

all_predictions = []
pick_stats = {"A": 0, "B": 0}

for i in tqdm(range(n_samples), desc="eng_MBR rerank"):
    candidates = [preds_a[i], preds_b[i]]
    best = eng_mbr_pick(candidates)
    all_predictions.append(best)
    if best == preds_a[i]:
        pick_stats["A"] += 1
    else:
        pick_stats["B"] += 1

print(f"Pick stats: A={pick_stats['A']}, B={pick_stats['B']} "
      f"(A={pick_stats['A']/n_samples*100:.1f}%, B={pick_stats['B']/n_samples*100:.1f}%)")

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
