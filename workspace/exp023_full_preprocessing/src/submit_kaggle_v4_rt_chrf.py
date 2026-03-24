"""
exp023: Kaggle提出用推論 v4 (Round-trip rerank, rt_chrf)
- 3温度(0.2/0.4/0.6) × 1候補 = 3候補
- 各候補を逆翻訳 (Eng→Akk) し、元ソースとのchrF++が最大の候補を選択
- 動的パディング + 長さソート
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
# Paths
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
    MODEL_PATH = os.path.join(PROJECT_ROOT, "workspace", "exp034_st_pretrain", "results", "fold3", "last_model")
    TEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "test.csv")
    OUTPUT_PATH = os.path.join(EXP_DIR, "results", "submission_v4_rt_chrf.csv")

MAX_LENGTH = 512
BATCH_SIZE = 16
PREFIX_FWD = "translate Akkadian to English: "
PREFIX_REV = "translate English to Akkadian: "

# Round-trip config
TEMPERATURES = [0.2, 0.4, 0.6]

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
# Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()


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


def dynamic_collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_id = tokenizer.pad_token_id or 0
    input_ids, attention_mask = [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_mask)}


def generate_sorted(dataset, sorted_indices, idx_map, n_samples, desc,
                    num_beams=1, do_sample=False, temperature=None, top_p=0.9):
    """Generate with length-sorted batching, return in original order."""
    sorted_ds = torch.utils.data.Subset(dataset, sorted_indices)
    loader = DataLoader(sorted_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate_fn)
    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            gen_kwargs = dict(input_ids=ids, attention_mask=mask, max_length=MAX_LENGTH)
            if do_sample:
                gen_kwargs.update(num_beams=1, do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gen_kwargs.update(num_beams=num_beams)
            out = model.generate(**gen_kwargs)
            sorted_preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    # Restore original order
    preds = [""] * n_samples
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    return preds


# ============================================================
# Round-trip scoring: chrF++
# ============================================================
chrfpp = sacrebleu.metrics.CHRF(word_order=2)


def rt_pick_chrf(candidates, back_translations, source):
    """Pick candidate whose back-translation has highest chrF++ vs source."""
    # Deduplicate
    seen = {}
    unique_cands, unique_bts = [], []
    for c, bt in zip(candidates, back_translations):
        if c not in seen:
            seen[c] = True
            unique_cands.append(c)
            unique_bts.append(bt)
    if len(unique_cands) <= 1:
        return unique_cands[0] if unique_cands else ""
    scores = [float(chrfpp.sentence_score(bt, [source]).score) for bt in unique_bts]
    return unique_cands[int(np.argmax(scores))]


# ============================================================
# Main
# ============================================================
test_df = pd.read_csv(TEST_PATH)
print(f"Test samples: {len(test_df)}")

# Prepare source texts (for round-trip comparison)
source_texts = test_df["transliteration"].astype(str).apply(clean_transliteration).tolist()
fwd_texts = [PREFIX_FWD + t for t in source_texts]

n_samples = len(test_df)

# Build dataset for forward translation
fwd_dataset = InferenceDataset(fwd_texts, tokenizer)
fwd_lengths = [fwd_dataset.items[i]["input_ids"].size(0) for i in range(len(fwd_dataset))]
fwd_sorted = sorted(range(len(fwd_dataset)), key=lambda i: fwd_lengths[i])
fwd_idx_map = {new: old for new, old in enumerate(fwd_sorted)}

# Step 1: Generate 3 candidates (one per temperature)
print("=== Step 1: Generate candidates ===")
torch.manual_seed(42)
all_candidates = {temp: [] for temp in TEMPERATURES}

for temp in TEMPERATURES:
    all_candidates[temp] = generate_sorted(
        fwd_dataset, fwd_sorted, fwd_idx_map, n_samples,
        desc=f"fwd t={temp}", do_sample=True, temperature=temp, top_p=0.9,
    )
    print(f"  t={temp}: {all_candidates[temp][0][:80]}...")

# Step 2: Back-translate each candidate
print("=== Step 2: Back-translate candidates ===")
all_bt = {}

for temp in TEMPERATURES:
    rev_texts = [PREFIX_REV + c for c in all_candidates[temp]]
    rev_dataset = InferenceDataset(rev_texts, tokenizer)
    rev_lengths = [rev_dataset.items[i]["input_ids"].size(0) for i in range(len(rev_dataset))]
    rev_sorted = sorted(range(len(rev_dataset)), key=lambda i: rev_lengths[i])
    rev_idx_map = {new: old for new, old in enumerate(rev_sorted)}

    all_bt[temp] = generate_sorted(
        rev_dataset, rev_sorted, rev_idx_map, n_samples,
        desc=f"BT t={temp}", num_beams=1,
    )
    print(f"  t={temp} BT: {all_bt[temp][0][:80]}...")

# Step 3: Round-trip rerank (chrF++ of BT vs source)
print("=== Step 3: Round-trip rerank (rt_chrf) ===")
all_predictions = []
for i in tqdm(range(n_samples), desc="rerank"):
    cands = [all_candidates[temp][i] for temp in TEMPERATURES]
    bts = [all_bt[temp][i] for temp in TEMPERATURES]
    src = source_texts[i]
    best = rt_pick_chrf(cands, bts, src)
    all_predictions.append(best)

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
