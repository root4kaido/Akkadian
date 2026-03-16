"""
exp035_mrt: Minimum Risk Training (SCST) for ByT5 Akkadian→English translation.

Self-Critical Sequence Training:
  L_total = α * L_SCST + (1 - α) * L_MLE
  L_SCST = -mean( (r(y_sample) - r(y_greedy)) * log P(y_sample|x) )
  r(y) = √(BLEU × chrF++)  (competition metric, sentence-level)

Starting from exp034 fold3 last_model.
"""
import os
import gc
import re
import sys
import math
import json
import logging
import argparse
from pathlib import Path

os.environ["WANDB_PROJECT"] = "akkadian-translation"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
import wandb

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
cmd_args = parser.parse_args()
FOLD = cmd_args.fold

# ============================================================
# Paths
# ============================================================
EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parent.parent
RESULTS_DIR = EXP_DIR / "results" / f"fold{FOLD}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AKT_GROUPS_PATH = PROJECT_ROOT / "datasets" / "processed" / "akt_groups.csv"

# exp034のlast_modelをベースに
BASE_MODEL_DIR = str(PROJECT_ROOT / "workspace" / "exp034_st_pretrain" / "results" / f"fold{FOLD}" / "last_model")

log_file = str(RESULTS_DIR / "train_mrt.log")
log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
    force=True,
)
logger = logging.getLogger(__name__)

# ============================================================
# Config
# ============================================================
MAX_LENGTH = 512
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42
BATCH_SIZE = 1           # ByT5 sequences are long; keep small
GRAD_ACCUM = 8           # effective BS = 1 * 8 = 8
NUM_SAMPLES = 4          # K samples per input
LOG_PROB_CHUNK = 2       # process log_prob in chunks to avoid OOM
ALPHA = 0.7              # MRT loss weight (SCST-heavy)
TEMPERATURE = 1.0        # full temp for diversity (eval() mode during generate)
TOP_P = 0.9              # nucleus sampling
GRAD_CLIP = 1.0
EVAL_STEPS = 50          # evaluate every N optimizer steps

BEST_MODEL_DIR = str(RESULTS_DIR / "best_model")
LAST_MODEL_DIR = str(RESULTS_DIR / "last_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
logger.info(f"Fold: {FOLD}")
logger.info(f"Base model: {BASE_MODEL_DIR}")

if not Path(BASE_MODEL_DIR).exists():
    logger.error(f"Base model not found at {BASE_MODEL_DIR}. Run exp034 first.")
    sys.exit(1)


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# Preprocessing (exp023-identical)
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
        return best_frac if int_part == 0 else f"{int_part} {best_frac}"
    return dec_str

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

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


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


def clean_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


# ============================================================
# Reward function
# ============================================================
def repeat_cleanup(text):
    """Remove repeated phrases (same as eval_full_doc.py)."""
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return " ".join(words[:i+n])
    return text


def reward_fn(hyp: str, ref: str) -> float:
    """Sentence-level chrF++ as reward.

    Using chrF++ alone instead of √(BLEU×chrF++) because:
    - sentence-level BLEU is often 0 for short texts (no 4-gram matches)
    - chrF++ is character-based and always >0 for non-empty outputs
    - optimizing chrF++ also improves BLEU indirectly
    """
    hyp = repeat_cleanup(hyp)
    if not hyp.strip():
        return 0.0
    chrf = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score / 100.0
    return chrf


# ============================================================
# Data loading
# ============================================================
PREFIX = "translate Akkadian to English: "

train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
logger.info(f"Original Train Data: {len(train_df)} docs")

akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))

logger.info("Applying full preprocessing (exp023)...")
train_df["translation"] = train_df["translation"].astype(str).apply(clean_translation)
train_df["transliteration"] = train_df["transliteration"].astype(str).apply(clean_transliteration)
logger.info("Preprocessing complete.")


def simple_sentence_aligner(df):
    aligned_data = []
    for idx, row in df.iterrows():
        src = str(row["transliteration"])
        tgt = str(row["translation"])
        oare_id = row["oare_id"]
        tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split("\n") if s.strip()]
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_data.append({"transliteration": s, "translation": t, "oare_id": oare_id})
        else:
            aligned_data.append({"transliteration": src, "translation": tgt, "oare_id": oare_id})
    return pd.DataFrame(aligned_data)


train_expanded = simple_sentence_aligner(train_df)
logger.info(f"Expanded Train Data: {len(train_expanded)} sentences")

train_expanded["akt_group"] = train_expanded["oare_id"].map(oare_to_group).fillna("None")

groups = train_expanded["akt_group"].values
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_expanded, groups=groups))
train_idx, val_idx = splits[FOLD]

train_split = train_expanded.iloc[train_idx].copy().reset_index(drop=True)
val_split = train_expanded.iloc[val_idx].copy().reset_index(drop=True)

logger.info(f"Fold {FOLD}: train={len(train_split)}, val={len(val_split)}")


# ============================================================
# Dataset
# ============================================================
class MRTDataset(Dataset):
    """Returns source text and reference for MRT training (Akkadian→English only)."""
    def __init__(self, df, tokenizer, max_length):
        self.sources = [PREFIX + str(t) for t in df["transliteration"].tolist()]
        self.references = df["translation"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src_enc = self.tokenizer(
            self.sources[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        tgt_enc = self.tokenizer(
            self.references[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": src_enc["input_ids"],
            "attention_mask": src_enc["attention_mask"],
            "labels": tgt_enc["input_ids"],
            "reference_text": self.references[idx],
        }


def collate_fn(batch, tokenizer):
    """Dynamic padding collator that also keeps reference texts."""
    refs = [b.pop("reference_text") for b in batch]

    max_src = max(len(b["input_ids"]) for b in batch)
    max_tgt = max(len(b["labels"]) for b in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for b in batch:
        pad_src = max_src - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad_src)
        attention_mask.append(b["attention_mask"] + [0] * pad_src)
        pad_tgt = max_tgt - len(b["labels"])
        labels.append(b["labels"] + [-100] * pad_tgt)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "reference_texts": refs,
    }


# ============================================================
# Model & Tokenizer
# ============================================================
logger.info(f"Loading model from {BASE_MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR)
model.gradient_checkpointing_enable()  # save memory
model.to(device)
gc.collect()
torch.cuda.empty_cache()

train_dataset = MRTDataset(train_split, tokenizer, MAX_LENGTH)
val_dataset = MRTDataset(val_split, tokenizer, MAX_LENGTH)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer),
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, tokenizer),
    num_workers=4,
    pin_memory=True,
)

# ============================================================
# Optimizer & Scheduler
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

total_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM
logger.info(f"Total optimizer steps: {total_steps}")
logger.info(f"Hyperparams: lr={LEARNING_RATE}, alpha={ALPHA}, K={NUM_SAMPLES}, "
            f"temp={TEMPERATURE}, top_p={TOP_P}, bs={BATCH_SIZE}, grad_accum={GRAD_ACCUM}")


# ============================================================
# Log probability computation
# ============================================================
def compute_sequence_log_prob(model, input_ids, attention_mask, decoder_input_ids):
    """Compute log P(tgt|src) for given sequences. Requires grad."""
    # Shift decoder_input_ids to create labels (remove first token, add padding)
    labels = decoder_input_ids[:, 1:].contiguous()
    dec_input = decoder_input_ids[:, :-1].contiguous()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=dec_input,
    )
    logits = outputs.logits  # [B, T-1, V]
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # Mask padding (-100 in labels or pad_token_id)
    mask = (labels != tokenizer.pad_token_id) & (labels != -100)
    mask = mask.float()

    # Length-normalized log prob
    seq_log_prob = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    return seq_log_prob


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def evaluate_model(model, val_loader):
    """Evaluate with greedy generation + competition metric."""
    model.eval()
    all_preds = []
    all_refs = []

    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        refs = batch["reference_texts"]

        out = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_length=MAX_LENGTH,
            num_beams=1,  # greedy
        )
        preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds = [repeat_cleanup(p.strip()) for p in preds]

        all_preds.extend(preds)
        all_refs.extend(refs)

    # Corpus-level metrics
    chrf = sacrebleu.corpus_chrf(all_preds, [all_refs], word_order=2).score
    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs]).score
    geo = math.sqrt(chrf * bleu) if chrf > 0 and bleu > 0 else 0.0

    model.train()
    return {"chrf": round(chrf, 2), "bleu": round(bleu, 2), "geo_mean": round(geo, 2)}


# ============================================================
# Training loop
# ============================================================
wandb.init(project="akkadian-translation", name=f"exp035_mrt_fold{FOLD}",
           config={"lr": LEARNING_RATE, "alpha": ALPHA, "K": NUM_SAMPLES,
                   "temperature": TEMPERATURE, "top_p": TOP_P,
                   "batch_size": BATCH_SIZE,
                   "grad_accum": GRAD_ACCUM, "epochs": NUM_EPOCHS})

best_geo = 0.0
global_step = 0
opt_step = 0
metrics_log = {"train": [], "eval": []}

# Enable bf16
scaler = None
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
logger.info(f"Using bf16: {use_bf16}")

# Initial evaluation
logger.info("Initial evaluation...")
init_metrics = evaluate_model(model, val_loader)
logger.info(f"Initial val metrics: {init_metrics}")
best_geo = init_metrics["geo_mean"]
metrics_log["eval"].append({"step": 0, **init_metrics})

model.train()
optimizer.zero_grad()

for epoch in range(NUM_EPOCHS):
    logger.info(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
    epoch_mle_loss = 0.0
    epoch_scst_loss = 0.0
    epoch_total_loss = 0.0
    epoch_avg_reward = 0.0
    epoch_avg_greedy_reward = 0.0
    epoch_avg_advantage = 0.0
    epoch_steps = 0

    for batch_idx, batch in enumerate(train_loader):
        src_ids = batch["input_ids"].to(device)
        src_mask = batch["attention_mask"].to(device)
        tgt_labels = batch["labels"].to(device)
        refs = batch["reference_texts"]
        B = src_ids.shape[0]

        # --------------------------------------------------
        # 1. MLE loss (teacher-forcing)
        # --------------------------------------------------
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            mle_out = model(input_ids=src_ids, attention_mask=src_mask, labels=tgt_labels)
            mle_loss = mle_out.loss

        # --------------------------------------------------
        # 2. SCST loss
        # --------------------------------------------------
        # Switch to eval mode for generation (gradient checkpointing breaks generate in train mode)
        model.eval()

        # 2a. Greedy baseline (no grad)
        with torch.no_grad():
            greedy_ids = model.generate(
                input_ids=src_ids,
                attention_mask=src_mask,
                max_length=MAX_LENGTH,
                num_beams=1,
            )
            greedy_texts = tokenizer.batch_decode(greedy_ids, skip_special_tokens=True)
            greedy_rewards = torch.tensor(
                [reward_fn(g, r) for g, r in zip(greedy_texts, refs)],
                dtype=torch.float32, device=device,
            )

        # 2b. Sample K candidates
        with torch.no_grad():
            sample_ids = model.generate(
                input_ids=src_ids,
                attention_mask=src_mask,
                max_length=MAX_LENGTH,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=NUM_SAMPLES,
            )

        # Back to train mode for log_prob computation
        model.train()
        # sample_ids shape: [B*K, seq_len]
        sample_texts = tokenizer.batch_decode(sample_ids, skip_special_tokens=True)

        # Compute rewards for each sample
        sample_rewards_list = []
        for i in range(B):
            for k in range(NUM_SAMPLES):
                s_text = sample_texts[i * NUM_SAMPLES + k]
                r = reward_fn(s_text, refs[i])
                sample_rewards_list.append(r)
        sample_rewards = torch.tensor(sample_rewards_list, dtype=torch.float32, device=device)

        # Greedy rewards repeated K times for advantage
        greedy_rewards_repeated = greedy_rewards.repeat_interleave(NUM_SAMPLES)
        advantages = sample_rewards - greedy_rewards_repeated  # [B*K]

        # 2c. Compute log P(sample|src) with gradient — chunked to avoid OOM
        src_ids_exp = src_ids.repeat_interleave(NUM_SAMPLES, dim=0)  # [B*K, src_len]
        src_mask_exp = src_mask.repeat_interleave(NUM_SAMPLES, dim=0)

        dec_start = torch.full(
            (sample_ids.shape[0], 1),
            model.config.decoder_start_token_id,
            dtype=torch.long, device=device,
        )
        decoder_input = torch.cat([dec_start, sample_ids[:, :-1]], dim=1)

        total_samples = sample_ids.shape[0]  # B*K
        all_seq_log_probs = []

        for chunk_start in range(0, total_samples, LOG_PROB_CHUNK):
            chunk_end = min(chunk_start + LOG_PROB_CHUNK, total_samples)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                c_out = model(
                    input_ids=src_ids_exp[chunk_start:chunk_end],
                    attention_mask=src_mask_exp[chunk_start:chunk_end],
                    decoder_input_ids=decoder_input[chunk_start:chunk_end],
                )
                c_logits = c_out.logits
                c_log_probs = F.log_softmax(c_logits, dim=-1)
                c_sample_ids = sample_ids[chunk_start:chunk_end]
                c_token_lp = c_log_probs.gather(-1, c_sample_ids.unsqueeze(-1)).squeeze(-1)
                c_mask = (c_sample_ids != tokenizer.pad_token_id).float()
                c_seq_lp = (c_token_lp * c_mask).sum(dim=-1) / c_mask.sum(dim=-1).clamp(min=1.0)
                all_seq_log_probs.append(c_seq_lp)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            seq_log_probs = torch.cat(all_seq_log_probs, dim=0)  # [B*K]
            scst_loss = -(advantages.detach() * seq_log_probs).mean()

        # --------------------------------------------------
        # 3. Combined loss
        # --------------------------------------------------
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = ALPHA * scst_loss + (1 - ALPHA) * mle_loss
            loss = loss / GRAD_ACCUM

        loss.backward()

        # Debug: log first 3 batches
        if global_step < 3:
            logger.info(f"  [DEBUG batch {global_step}]")
            logger.info(f"    src: {tokenizer.decode(src_ids[0], skip_special_tokens=True)[:100]}")
            logger.info(f"    ref: {refs[0][:100]}")
            logger.info(f"    greedy: {greedy_texts[0][:100]}")
            logger.info(f"    greedy_reward: {greedy_rewards[0].item():.4f}")
            for k in range(min(2, NUM_SAMPLES)):
                logger.info(f"    sample[{k}]: {sample_texts[k][:100]}")
                logger.info(f"    sample_reward[{k}]: {sample_rewards[k].item():.4f}")

        # Track metrics
        epoch_mle_loss += mle_loss.item()
        epoch_scst_loss += scst_loss.item()
        epoch_total_loss += (ALPHA * scst_loss + (1 - ALPHA) * mle_loss).item()
        epoch_avg_reward += sample_rewards.mean().item()
        epoch_avg_greedy_reward += greedy_rewards.mean().item()
        epoch_avg_advantage += advantages.mean().item()
        epoch_steps += 1
        global_step += 1

        # Gradient accumulation step
        if (batch_idx + 1) % GRAD_ACCUM == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            opt_step += 1

            # Log
            if opt_step % 10 == 0:
                avg_mle = epoch_mle_loss / epoch_steps
                avg_scst = epoch_scst_loss / epoch_steps
                avg_total = epoch_total_loss / epoch_steps
                avg_reward = epoch_avg_reward / epoch_steps
                avg_greedy_r = epoch_avg_greedy_reward / epoch_steps
                avg_adv = epoch_avg_advantage / epoch_steps
                logger.info(
                    f"  step {opt_step} | mle={avg_mle:.4f} scst={avg_scst:.4f} "
                    f"total={avg_total:.4f} sample_r={avg_reward:.4f} "
                    f"greedy_r={avg_greedy_r:.4f} adv={avg_adv:.4f}"
                )
                wandb.log({
                    "train/mle_loss": avg_mle,
                    "train/scst_loss": avg_scst,
                    "train/total_loss": avg_total,
                    "train/avg_sample_reward": avg_reward,
                    "train/avg_greedy_reward": avg_greedy_r,
                    "train/avg_advantage": avg_adv,
                    "train/opt_step": opt_step,
                })
                entry = {"step": opt_step, "mle_loss": avg_mle, "scst_loss": avg_scst,
                         "total_loss": avg_total, "avg_reward": avg_reward,
                         "avg_greedy_reward": avg_greedy_r}
                metrics_log["train"].append(entry)

            # Evaluate
            if opt_step % EVAL_STEPS == 0:
                val_metrics = evaluate_model(model, val_loader)
                logger.info(f"  [eval step {opt_step}] {val_metrics}")
                wandb.log({"val/chrf": val_metrics["chrf"], "val/bleu": val_metrics["bleu"],
                           "val/geo_mean": val_metrics["geo_mean"], "train/opt_step": opt_step})
                metrics_log["eval"].append({"step": opt_step, **val_metrics})

                if val_metrics["geo_mean"] > best_geo:
                    best_geo = val_metrics["geo_mean"]
                    logger.info(f"  New best geo_mean: {best_geo} — saving model")
                    model.save_pretrained(BEST_MODEL_DIR)
                    tokenizer.save_pretrained(BEST_MODEL_DIR)

                model.train()

    # End-of-epoch eval
    val_metrics = evaluate_model(model, val_loader)
    logger.info(f"Epoch {epoch + 1} final: {val_metrics}")
    wandb.log({"val/chrf": val_metrics["chrf"], "val/bleu": val_metrics["bleu"],
               "val/geo_mean": val_metrics["geo_mean"], "epoch": epoch + 1})
    metrics_log["eval"].append({"step": opt_step, "epoch": epoch + 1, **val_metrics})

    if val_metrics["geo_mean"] > best_geo:
        best_geo = val_metrics["geo_mean"]
        logger.info(f"  New best geo_mean: {best_geo} — saving model")
        model.save_pretrained(BEST_MODEL_DIR)
        tokenizer.save_pretrained(BEST_MODEL_DIR)

# ============================================================
# Save last model & logs
# ============================================================
logger.info(f"Saving last model to {LAST_MODEL_DIR}")
model.save_pretrained(LAST_MODEL_DIR)
tokenizer.save_pretrained(LAST_MODEL_DIR)

with open(str(RESULTS_DIR / "metrics_log.json"), "w") as f:
    json.dump(metrics_log, f, indent=2, ensure_ascii=False)

logger.info(f"Training complete. Best geo_mean: {best_geo}")
logger.info(f"Best model: {BEST_MODEL_DIR}")
logger.info(f"Last model: {LAST_MODEL_DIR}")
wandb.finish()
