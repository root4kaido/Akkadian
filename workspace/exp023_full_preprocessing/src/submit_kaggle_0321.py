"""
s1_exp007_large_lr1e4: Kaggle提出用推論
- exp023_full_preprocessing/submit_kaggle.pyベース
- モデル: byt5-large fold3/last_model (lr=1e-4)
- beam4, max_length=512, early_stopping=True
- repeat_cleanup後処理
- batch_decode → _fast_batch_decode (byt5-largeハング対策)
"""
import os
import re
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ============================================================
# Paths — Kaggle上ではON_KAGGLEをTrueに変更するだけ
# ============================================================
ON_KAGGLE = True

if ON_KAGGLE:
    MODEL_BASE = "/kaggle/input/models/nomorevotch/akkadianmodels/pytorch/exp041_bt_augment_v2_last/1/fold3/pretrain_ft"
    MODEL_PATH = f"{MODEL_BASE}/last_model"
    TEST_PATH = "/kaggle/input/competitions/deep-past-initiative-machine-translation/test.csv"
    OUTPUT_PATH = "submission.csv"
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXP_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
    MODEL_PATH = os.path.join(EXP_DIR, "results", "fold3", "last_model")
    TEST_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "test.csv")
    OUTPUT_PATH = os.path.join(EXP_DIR, "results", "submission.csv")

MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_BEAMS = 4
PREFIX = "translate Akkadian to English: "

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# 前処理（train.pyと同一のclean_transliteration + ロバスト化）
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

# --- ロバスト化: testデータの表記揺れ対策（ノートブック由来） ---
# ASCII→ダイアクリティクス（trainには不要だがtestにASCII形式が混入する可能性への保険）
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a":"á","e":"é","i":"í","u":"ú","A":"Á","E":"É","I":"Í","U":"Ú"})
_GRAVE = str.maketrans({"a":"à","e":"è","i":"ì","u":"ù","A":"À","E":"È","I":"Ì","U":"Ù"})

def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s

# Gap正規化（多様な欠損表記を統一）
_GAP_UNIFIED_RE = re.compile(
    r"\.\.\."
    r"|\[\s*\.\.\.\s*\]"
    r"|\[\.+\]"
    r"|\[[^\]]*(?:broken|missing|illegible|damaged|effaced|erased|lost|traces)[^\]]*\]"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I
)

# Determinative（大文字: 括弧除去、小文字: {}変換）
_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")

# 文字正規化
_CHAR_CLEANUP = str.maketrans({"ʾ": "", "—": "-", "–": "-"})


def clean_transliteration(text: str) -> str:
    """train.pyと同一の前処理 + testデータ向けロバスト化"""
    if not isinstance(text, str) or not text.strip():
        return text
    # --- ロバスト化（既存データにはno-op、testの表記揺れ対策） ---
    text = _ascii_to_diacritics(text)           # ASCII→ダイアクリティクス
    text = re.sub(_DET_UPPER_RE, r"\1", text)   # 大文字det: (DINGIR) → DINGIR
    text = re.sub(_DET_LOWER_RE, r"{\1}", text) # 小文字det: (ki) → {ki}, (d) → {d}
    text = _GAP_UNIFIED_RE.sub("<gap>", text)   # Gap正規化
    text = text.translate(_CHAR_CLEANUP)         # ʾ削除, ダッシュ正規化
    # --- train.pyと同一の処理 ---
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = text.replace('ş', 'ṣ').replace('İ', 'I').replace('ı', 'i')
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


# --- ロバスト化: 後処理追加パターン（ノートブック由来） ---
# 文法マーカー（括弧付き複合形も対応）
_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)", re.I
)

# Shekel分数変換
_SHEKEL_REPLS = [
    (re.compile(r'5\s+11\s*/\s*12\s+shekels?', re.I), '6 shekels less 15 grains'),
    (re.compile(r'5\s*/\s*12\s+shekels?', re.I), '⅔ shekel 15 grains'),
    (re.compile(r'7\s*/\s*12\s+shekels?', re.I), '½ shekel 15 grains'),
    (re.compile(r'1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?', re.I), '15 grains'),
]

# Curly quotes除去
_CURLY_QUOTES_RE = re.compile("[\u201c\u201d\u2018\u2019]")

# Stray marks（<gap>以外のタグ）
_STRAY_MARKS_RE = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')

# Forbidden chars（出力に残るべきでない文字）— <gap>を壊さないよう保護して適用
_FORBIDDEN_TRANS = str.maketrans("", "", '()——⌈⌋⌊[]+ʾ;')

# 句読点の整形
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")


def clean_translation(text: str) -> str:
    """train.pyと同一の後処理 + ロバスト化"""
    if not isinstance(text, str) or not text.strip():
        return text
    # --- 既存処理 ---
    # 文法マーカー除去（括弧付き複合形も対応）
    text = re.sub(_SOFT_GRAM_RE, ' ', text)
    text = re.sub(r'\bfem\.\s*', '', text)
    text = re.sub(r'\bsing\.\s*', '', text)
    text = re.sub(r'\bpl\.\s*', '', text)
    text = re.sub(r'\bplural\b\s*', '', text)
    text = text.replace('(?)', '')
    text = re.sub(r'<<\s*>>', '', text)
    text = re.sub(r'<\s+>', '', text)
    text = re.sub(r'(?<!\.)\.\.(?!\.)', '', text)
    text = re.sub(r'\bxx?\b', '', text)
    # Gap/PN統一
    text = _GAP_UNIFIED_RE.sub("<gap>", text)
    text = re.sub(r'\bPN\b', '<gap>', text)
    # Commodity正規化
    text = re.sub(r'\b-gold\b', 'pašallum gold', text)
    text = re.sub(r'\b-tax\b', 'šadduātum tax', text)
    text = re.sub(r'\b-textiles\b', 'kutānum textiles', text)
    # Shekel分数変換
    for pat, repl in _SHEKEL_REPLS:
        text = re.sub(pat, repl, text)
    # 分数変換
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    # スラッシュ代替表現
    text = re.sub(r'(\S+)\s*/\s*\S+', r'\1', text)
    text = re.sub(r'\(m\)', '{m}', text)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    # Month正規化
    for roman, integer in sorted(ROMAN_TO_INT.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf'\bmonth\s+{roman}(?=[\s,.:;!?\)]|$)', f'month {integer}', text)
    for pattern, number in MONTH_NAMES_TRANSLATION.items():
        text = re.sub(rf'\bmonth\s+{pattern}\b', f'month {number}', text, flags=re.IGNORECASE)
    # --- ロバスト化（ノートブック由来） ---
    text = re.sub(_STRAY_MARKS_RE, '', text)         # stray marks除去
    text = re.sub(_CURLY_QUOTES_RE, '', text)         # curly quotes除去
    # Forbidden chars除去（<gap>を保護）
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.translate(_FORBIDDEN_TRANS)
    text = text.replace("\x00GAP\x00", " <gap> ")
    # 句読点整形
    text = re.sub(_PUNCT_SPACE_RE, r"\1", text)
    text = re.sub(_REPEAT_PUNCT_RE, r"\1", text)
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


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# Pre-cache special token IDs to avoid slow repeated property lookups in batch_decode
_special_ids = set(tokenizer.all_special_ids)


def _fast_batch_decode(ids_tensor):
    """batch_decode without skip_special_tokens to avoid O(n*m) property rebuild."""
    results = []
    for ids in ids_tensor:
        filtered = [int(i) for i in ids if int(i) not in _special_ids]
        results.append(tokenizer.decode(filtered, skip_special_tokens=False))
    return results


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
        decoded = _fast_batch_decode(outputs)
        all_predictions.extend([d.strip() for d in decoded])

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
