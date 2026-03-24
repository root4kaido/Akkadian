"""
exp023: Kaggle提出用推論 v9 (N-model ensemble + unified reverse model for BT)
- MODELSリストの各モデルで順翻訳beam4のみ実行
- REVERSE_MODELで全候補の逆翻訳を一括実行
- rt_weighted で最良候補選択
- モデルは1つずつロード→推論→キャッシュ→解放（メモリ効率）
- fast_batch_decode: special token事前フィルタでデコード高速化
- 動的パディング + 長さソート
- repeat_cleanup + clean_translation 後処理
"""
import os
import re
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

# ============================================================
# Paths & Models
# ============================================================
ON_KAGGLE = True

if ON_KAGGLE:
    MODELS = [
        {
            "name": "exp034_base",
            "path": "/kaggle/input/models/nomorevotch/akkadianmodels/pytorch/exp034_st_pretrain_last/1/fold3/last_model",
        },
        {
            "name": "s1_exp007_large",
            "path": "/kaggle/input/models/nomorevotch/akkadianmodels/pytorch/s1_exp007_large_lr1e4_last/1/fold3/last_model",
        },
        {
            "name": "exp041_bt_v2",
            "path": "/kaggle/input/models/nomorevotch/akkadianmodels/pytorch/exp041_bt_augment_v2_last/1/fold3/pretrain_ft/last_model",
        },
    ]
    REVERSE_MODEL = {
        "name": "s1_exp007_large",
        "path": "/kaggle/input/models/nomorevotch/akkadianmodels/pytorch/s1_exp007_large_lr1e4_last/1/fold3/last_model",
    }
    TEST_PATH = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"
    OUTPUT_PATH = "submission.csv"
    CACHE_DIR = "."
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXP_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
    MODELS = [
        {
            "name": "exp034_base",
            "path": os.path.join(PROJECT_ROOT, "workspace", "exp034_st_pretrain", "results", "fold3", "last_model"),
        },
        {
            "name": "s1_exp007_large",
            "path": os.path.join(PROJECT_ROOT, "workspace", "s1_exp007_large_lr1e4", "results", "fold3", "last_model"),
        },
        {
            "name": "exp041_bt_v2",
            "path": os.path.join(PROJECT_ROOT, "workspace", "exp041_bt_augment_v2", "results", "fold3", "pretrain_ft", "last_model"),
        },
    ]
    REVERSE_MODEL = {
        "name": "s1_exp007_large",
        "path": os.path.join(PROJECT_ROOT, "workspace", "s1_exp007_large_lr1e4", "results", "fold3", "last_model"),
    }
    TEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "test.csv")
    OUTPUT_PATH = os.path.join(EXP_DIR, "results", "submission_v9_unified_bt.csv")
    CACHE_DIR = os.path.join(EXP_DIR, "results")

MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_BEAMS = 4
PREFIX_FWD = "translate Akkadian to English: "
PREFIX_REV = "translate English to Akkadian: "

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Number of forward models: {len(MODELS)}")
for i, m in enumerate(MODELS):
    print(f"  [{i}] {m['name']}: {m['path']}")
print(f"Reverse model: {REVERSE_MODEL['name']}: {REVERSE_MODEL['path']}")


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
# 動的パディング + 長さソート + fast_batch_decode
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


def make_fast_batch_decode(tokenizer):
    """Create a fast batch_decode that avoids O(n*m) special token property rebuild."""
    special_ids = set(tokenizer.all_special_ids)
    def _fast_batch_decode(ids_tensor):
        results = []
        for ids in ids_tensor:
            filtered = [int(i) for i in ids if int(i) not in special_ids]
            results.append(tokenizer.decode(filtered, skip_special_tokens=False).strip())
        return results
    return _fast_batch_decode


def generate_sorted(dataset, tokenizer, model, n_samples, desc,
                    num_beams=4, batch_size=BATCH_SIZE):
    """Generate with length-sorted batching, return in original order."""
    lengths = [dataset.items[i]["input_ids"].size(0) for i in range(len(dataset))]
    sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])
    idx_map = {new: old for new, old in enumerate(sorted_indices)}
    sorted_ds = torch.utils.data.Subset(dataset, sorted_indices)
    collate_fn = make_collate_fn(tokenizer)
    loader = DataLoader(sorted_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    fast_decode = make_fast_batch_decode(tokenizer)
    sorted_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, num_beams=num_beams, early_stopping=True,
            )
            sorted_preds.extend(fast_decode(out))
    preds = [""] * n_samples
    for new_idx, pred in enumerate(sorted_preds):
        preds[idx_map[new_idx]] = pred
    return preds


# ============================================================
# Round-trip scoring: rt_weighted
# ============================================================
chrfpp = sacrebleu.metrics.CHRF(word_order=2)
bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)


def score_jaccard(hyp, ref):
    h_set = set(hyp.lower().split())
    r_set = set(ref.lower().split())
    if not h_set and not r_set:
        return 1.0
    if not h_set or not r_set:
        return 0.0
    return len(h_set & r_set) / len(h_set | r_set)


def score_length_bonus(hyp, ref):
    len_h = len(hyp.split())
    len_r = len(ref.split())
    if len_h == 0 and len_r == 0:
        return 1.0
    if len_h == 0 or len_r == 0:
        return 0.0
    return 1.0 - abs(len_h - len_r) / max(len_h, len_r)


def rt_pick_weighted(candidates, back_translations, source,
                     w_chrf=0.55, w_bleu=0.25, w_jaccard=0.20, w_len=0.10):
    """Pick candidate whose back-translation has highest weighted score vs source."""
    seen = {}
    unique_cands, unique_bts = [], []
    for c, bt in zip(candidates, back_translations):
        if c not in seen:
            seen[c] = True
            unique_cands.append(c)
            unique_bts.append(bt)
    if len(unique_cands) <= 1:
        return unique_cands[0] if unique_cands else ""
    scores = []
    for bt in unique_bts:
        s = (w_chrf * float(chrfpp.sentence_score(bt, [source]).score) / 100.0
             + w_bleu * float(bleu_metric.sentence_score(bt, [source]).score) / 100.0
             + w_jaccard * score_jaccard(bt, source)
             + w_len * score_length_bonus(bt, source))
        scores.append(s)
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
# Step 1: Forward translate with each model (fwd only → cache → release)
# ============================================================
all_fwd_results = []  # list of {"name", "predictions"}

for idx, model_cfg in enumerate(MODELS):
    model_name = model_cfg["name"]
    model_path = model_cfg["path"]
    cache_path = os.path.join(CACHE_DIR, f"fwd_{idx}_{model_name}_cache.json")

    print("=" * 60)
    print(f"=== Step 1.{idx}: Forward — {model_name} ===")
    print("=" * 60)

    # Check cache
    if os.path.exists(cache_path):
        print(f"  Loading from cache: {cache_path}")
        with open(cache_path, "r") as f:
            cached = json.load(f)
        all_fwd_results.append({"name": model_name, "predictions": cached["predictions"]})
        print(f"  Loaded {len(cached['predictions'])} predictions from cache.")
        continue

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded ({n_params:.0f}M params) in {time.time()-t0:.1f}s")

    # Forward translation (beam4)
    fwd_dataset = InferenceDataset(fwd_texts, tokenizer)
    preds = generate_sorted(fwd_dataset, tokenizer, model, n_samples,
                            desc=f"[{idx}] fwd beam4")
    print(f"  fwd[0]: {preds[0][:80]}...")

    # Cache
    with open(cache_path, "w") as f:
        json.dump({"predictions": preds}, f, ensure_ascii=False)
    print(f"  Cached to {cache_path}")

    all_fwd_results.append({"name": model_name, "predictions": preds})

    # Release
    del model, tokenizer, fwd_dataset
    torch.cuda.empty_cache()
    print(f"  Released. Time: {time.time()-t0:.1f}s")

# ============================================================
# Step 2: Back-translate ALL candidates with unified reverse model
# ============================================================
print()
print("=" * 60)
print(f"=== Step 2: Unified BT — {REVERSE_MODEL['name']} ===")
print("=" * 60)

bt_cache_path = os.path.join(CACHE_DIR, f"bt_unified_{REVERSE_MODEL['name']}_cache.json")

if os.path.exists(bt_cache_path):
    print(f"  Loading from cache: {bt_cache_path}")
    with open(bt_cache_path, "r") as f:
        bt_cache = json.load(f)
    all_bts = bt_cache["back_translations"]  # list of lists, same order as all_fwd_results
    print(f"  Loaded BTs for {len(all_bts)} models from cache.")
else:
    t0 = time.time()
    rev_path = REVERSE_MODEL["path"]
    tokenizer = AutoTokenizer.from_pretrained(rev_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(rev_path).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded ({n_params:.0f}M params) in {time.time()-t0:.1f}s")

    all_bts = []
    for fwd_res in all_fwd_results:
        rev_texts = [PREFIX_REV + p for p in fwd_res["predictions"]]
        rev_dataset = InferenceDataset(rev_texts, tokenizer)
        bts = generate_sorted(rev_dataset, tokenizer, model, n_samples,
                              desc=f"BT({fwd_res['name']})")
        all_bts.append(bts)
        print(f"  BT({fwd_res['name']})[0]: {bts[0][:80]}...")
        del rev_dataset

    # Cache
    with open(bt_cache_path, "w") as f:
        json.dump({"back_translations": all_bts}, f, ensure_ascii=False)
    print(f"  Cached to {bt_cache_path}")

    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"  Released. Total BT time: {time.time()-t0:.1f}s")

# ============================================================
# Step 3: Round-trip weighted rerank (N candidates per sample)
# ============================================================
n_models = len(all_fwd_results)
print()
print("=" * 60)
print(f"=== Step 3: Round-trip weighted rerank ({n_models} candidates) ===")
print("=" * 60)

all_predictions = []
pick_stats = {m["name"]: 0 for m in all_fwd_results}

for i in tqdm(range(n_samples), desc="rerank"):
    candidates = [m["predictions"][i] for m in all_fwd_results]
    back_translations = [all_bts[j][i] for j in range(n_models)]
    src = source_texts[i]
    best = rt_pick_weighted(candidates, back_translations, src)
    all_predictions.append(best)
    for m in all_fwd_results:
        if best == m["predictions"][i]:
            pick_stats[m["name"]] += 1
            break

print("Pick stats:")
for name, count in pick_stats.items():
    print(f"  {name}: {count} ({count/n_samples*100:.1f}%)")

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
print(f"\nSubmission saved to {OUTPUT_PATH}")
print(f"Shape: {submission.shape}")
print(f"Empty translations: {(submission['translation'] == 'broken text').sum()}")
