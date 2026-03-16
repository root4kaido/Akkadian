"""
exp023: Cross-model MBR推論（2GPU並列版）
- GPU0とGPU1にfoldを分散し、threading で並列に候補生成
- GPU0: fold 0,2,4 / GPU1: fold 1,3
- 生成完了後マージしてMBR選択→提出ファイル作成
- 1GPU環境では自動で逐次フォールバック
"""
import os
import re
import time
import threading

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

# ============================================================
# Config
# ============================================================
ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

if ON_KAGGLE:
    MODEL_BASE = "/kaggle/input/akkadianmodels/pytorch"
    MODEL_PATHS = [
        f"{MODEL_BASE}/exp023_fold{i}/1/fold{i}/best_model" for i in range(5)
    ]
    TEST_PATH = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    OUTPUT_PATH = "submission.csv"
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXP_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
    MODEL_PATHS = [
        os.path.join(EXP_DIR, "results", f"fold{i}", "best_model") for i in range(5)
    ]
    TEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "test.csv")
    OUTPUT_PATH = os.path.join(EXP_DIR, "results", "submission_cross_model_mbr_v2.csv")

MAX_LENGTH = 512
BATCH_SIZE = 4
PREFIX = "translate Akkadian to English: "

# MBR候補生成設定
TEMPERATURES = [0.6, 0.8, 1.05]
NUM_SAMPLES_PER_TEMP = 1
REPETITION_PENALTY = 1.2

NUM_GPUS = torch.cuda.device_count()
print(f"Available GPUs: {NUM_GPUS}")

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
# 後処理
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
# MBR
# ============================================================
chrfpp = sacrebleu.metrics.CHRF(word_order=2)


def mbr_pick(candidates):
    """chrF++ consensusで最良候補を選択"""
    cands = list(dict.fromkeys(candidates))  # dedup keeping order
    cands = cands[:32]
    n = len(cands)
    if n <= 1:
        return cands[0] if cands else ""
    scores = []
    for i in range(n):
        s = sum(
            float(chrfpp.sentence_score(cands[i], [cands[j]]).score)
            for j in range(n) if j != i
        )
        scores.append(s / (n - 1))
    return cands[int(np.argmax(scores))]


# ============================================================
# Dataset
# ============================================================
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx], max_length=MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


# ============================================================
# Candidate generation (1モデル・指定device)
# ============================================================
def generate_candidates_for_model(model, tokenizer, texts, device):
    """1モデルからgreedy + multi-temp samplingで候補生成"""
    ds = InferenceDataset(texts, tokenizer)
    all_candidates = [[] for _ in range(len(texts))]

    # --- Greedy ---
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  [{device}] greedy"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            bs = ids.shape[0]
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_length=MAX_LENGTH, do_sample=False, num_beams=1,
            )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            for i, d in enumerate(decoded):
                all_candidates[idx + i].append(d.strip())
            idx += bs

    # --- Multi-temperature sampling ---
    for temp in TEMPERATURES:
        loader1 = DataLoader(ds, batch_size=1, shuffle=False)
        idx = 0
        with torch.no_grad():
            for batch in tqdm(loader1, desc=f"  [{device}] sample t={temp}"):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                out = model.generate(
                    input_ids=ids, attention_mask=mask,
                    max_length=MAX_LENGTH,
                    do_sample=True, num_beams=1,
                    top_p=0.9, temperature=temp,
                    num_return_sequences=NUM_SAMPLES_PER_TEMP,
                    repetition_penalty=REPETITION_PENALTY,
                )
                decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
                for d in decoded:
                    all_candidates[idx].append(d.strip())
                idx += 1

    return all_candidates


# ============================================================
# GPU worker: 割り当てられたfoldを逐次処理
# ============================================================
def gpu_worker(fold_indices, input_texts, device, results_dict):
    """指定GPUで指定foldのモデルを逐次ロード→推論→アンロード"""
    n = len(input_texts)
    combined = [[] for _ in range(n)]

    for fold_i in fold_indices:
        model_path = MODEL_PATHS[fold_i]
        print(f"[{device}] Loading fold{fold_i} from {model_path}")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        model.eval()

        fold_candidates = generate_candidates_for_model(model, tokenizer, input_texts, device)

        for i in range(n):
            combined[i].extend(fold_candidates[i])

        n_cands = len(fold_candidates[0]) if fold_candidates else 0
        elapsed = time.time() - t0
        print(f"[{device}] fold{fold_i}: {n_cands} candidates/sample, {elapsed:.0f}s")

        del model
        torch.cuda.empty_cache()

    results_dict[device] = combined


# ============================================================
# Main
# ============================================================
test_df = pd.read_csv(TEST_PATH)
print(f"Test samples: {len(test_df)}")

# Prepare input texts
input_texts = test_df["transliteration"].astype(str).apply(clean_transliteration).tolist()
input_texts = [PREFIX + t for t in input_texts]

t_total = time.time()
n_samples = len(input_texts)

if NUM_GPUS >= 2:
    # === 2GPU並列: threadingで各GPUにfoldを分配 ===
    folds_gpu0 = [0, 2, 4]  # 3 folds
    folds_gpu1 = [1, 3]     # 2 folds
    print(f"2GPU parallel mode: GPU0={folds_gpu0}, GPU1={folds_gpu1}")

    results_dict = {}
    threads = []
    for device_str, folds in [("cuda:0", folds_gpu0), ("cuda:1", folds_gpu1)]:
        t = threading.Thread(
            target=gpu_worker,
            args=(folds, input_texts, torch.device(device_str), results_dict),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Merge candidates from both GPUs
    all_candidates = [[] for _ in range(n_samples)]
    for device_str in ["cuda:0", "cuda:1"]:
        gpu_cands = results_dict[torch.device(device_str)]
        for i in range(n_samples):
            all_candidates[i].extend(gpu_cands[i])

else:
    # === 1GPU逐次 ===
    print("Single GPU mode: running all folds sequentially")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results_dict = {}
    gpu_worker(list(range(5)), input_texts, device, results_dict)
    all_candidates = results_dict[device]

# Pool stats
pool_sizes = [len(c) for c in all_candidates]
unique_sizes = [len(set(c)) for c in all_candidates]
print(f"\nCandidate pool: mean={np.mean(pool_sizes):.1f}, unique={np.mean(unique_sizes):.1f}")

# MBR selection
print("Running MBR selection...")
t0 = time.time()
all_predictions = [mbr_pick(cands) for cands in tqdm(all_candidates, desc="MBR")]
mbr_time = time.time() - t0
print(f"MBR selection: {mbr_time:.0f}s")

# Post-processing
all_predictions = [repeat_cleanup(p) for p in all_predictions]
all_predictions = [clean_translation(p) for p in all_predictions]
all_predictions = [extract_first_sentence(p) for p in all_predictions]

# Save
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
print(f"Total elapsed: {time.time() - t_total:.0f}s")
