"""
exp040: Qwen3.5-9B LoRA SFT for Akkadian → English translation
- unsloth + SFTTrainer + train_on_responses_only
- Data: train.csv (GroupKFold fold3 train) + backtranslated.csv
- After training: sent-CV / doc-CV evaluation
- Ref: https://unsloth.ai/docs/models/qwen3.5/fine-tune
"""

import os
os.environ['UNSLOTH_MOE_DISABLE_AUTOTUNE'] = '1'

import re
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from datasets import Dataset
from tqdm import tqdm
import sacrebleu

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=3, help="Fold index (0-4)")
parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
parser.add_argument("--eval_only", action="store_true", help="Skip training, load saved model and evaluate")
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
SENT_ALIGNED_PATH = PROJECT_ROOT / "datasets" / "processed" / "sentence_aligned.csv"

log_file = str(RESULTS_DIR / "sft_finetune.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    force=True,
)
logger = logging.getLogger(__name__)

# ============================================================
# Preprocessing (exp023 compatible)
# ============================================================
FRACTION_TARGETS = {
    1/2: "½", 1/4: "¼", 1/3: "⅓", 2/3: "⅔",
    5/6: "⅚", 3/4: "¾", 1/6: "⅙", 5/8: "⅝",
}
APPROX_TOLERANCE = 0.002
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

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


def clean_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    text = text.translate(SUBSCRIPT_MAP)
    text = re.sub(r'\(ki\)', '{ki}', text)
    text = re.sub(r'\d+\.\d+', _decimal_to_fraction, text)
    return text


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


def has_repetition(text):
    words = str(text).split()
    if len(words) < 6:
        return False
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return True
    return False


# ============================================================
# Data loading
# ============================================================
SYSTEM_PROMPT = "You are an expert Assyriologist specializing in Old Assyrian texts. Translate the following Akkadian transliteration into English. Output ONLY the English translation, nothing else."

logger.info("=" * 60)
logger.info(f"exp040: Qwen3.5-9B LoRA SFT (fold{FOLD})")
logger.info("=" * 60)

# Load train data
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
train_df["translation"] = train_df["translation"].astype(str).apply(clean_translation)
train_df["transliteration"] = train_df["transliteration"].astype(str).apply(clean_transliteration)

# AKT groups
akt_groups = pd.read_csv(str(AKT_GROUPS_PATH))
oare_to_group = dict(zip(akt_groups["oare_id"], akt_groups["akt_group"].fillna("None")))
train_df["akt_group"] = train_df["oare_id"].map(oare_to_group).fillna("None")

# GroupKFold split
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_df, groups=train_df["akt_group"].values))
train_idx, val_idx = splits[FOLD]

train_split = train_df.iloc[train_idx].copy().reset_index(drop=True)
val_df = train_df.iloc[val_idx].copy().reset_index(drop=True)
logger.info(f"Fold {FOLD}: train={len(train_split)}, val={len(val_df)} docs")

# Train data (real only)
combined_rows = []
for _, row in train_split.iterrows():
    src = str(row["transliteration"]).strip()
    tgt = str(row["translation"]).strip()
    if len(tgt) >= 10:
        combined_rows.append({"transliteration": src, "translation": tgt})

logger.info(f"Training samples: {len(combined_rows)}")

# sent-CV data
sent_df = pd.read_csv(str(SENT_ALIGNED_PATH))
val_oare_ids = set(val_df["oare_id"].tolist())
sent_val = sent_df[sent_df["oare_id"].isin(val_oare_ids)].reset_index(drop=True)
doc_sent_counts = sent_val.groupby("oare_id").size()
short_docs = doc_sent_counts[doc_sent_counts <= 6].index
sent_val_short = sent_val[sent_val["oare_id"].isin(short_docs)].reset_index(drop=True)
sent_val_short["akk_segment"] = sent_val_short["akk_segment"].astype(str).apply(clean_transliteration)
sent_val_short["eng_sentence"] = sent_val_short["eng_sentence"].astype(str).apply(clean_translation)
logger.info(f"sent-CV: {len(sent_val_short)} sents from {sent_val_short['oare_id'].nunique()} docs")
logger.info(f"doc-CV: {len(val_df)} docs")

# ============================================================
# Model loading (unsloth)
# ============================================================
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen3.5-9B"
max_seq_length = cmd_args.max_seq_length

logger.info(f"Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

final_dir = RESULTS_DIR / "final_model"

if cmd_args.eval_only:
    # Load saved LoRA adapter onto base model (skip get_peft_model)
    logger.info(f"--eval_only: Loading saved model from {final_dir}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(final_dir))
    logger.info("LoRA adapter loaded")
else:
    # LoRA (training only)
    model = FastLanguageModel.get_peft_model(
        model,
        r=cmd_args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=cmd_args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_seq_length,
    )
    logger.info("LoRA applied")
    # ============================================================
    # Dataset preparation (chat template)
    # ============================================================
    conversations = []
    for row in combined_rows:
        conversations.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["transliteration"]},
            {"role": "assistant", "content": row["translation"]},
        ])

    dataset = Dataset.from_dict({"conversations": conversations})

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    logger.info(f"Dataset prepared: {len(dataset)} samples")

    # ============================================================
    # Training
    # ============================================================
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only

    OUTPUT_DIR = str(RESULTS_DIR / "sft_checkpoints")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=cmd_args.batch_size,
            gradient_accumulation_steps=cmd_args.grad_accum,
            num_train_epochs=cmd_args.epochs,
            learning_rate=cmd_args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            seed=42,
            report_to="none",
            optim="adamw_8bit",
            max_grad_norm=1.0,
            output_dir=OUTPUT_DIR,
            bf16=True,
            max_seq_length=max_seq_length,
        ),
    )

    # Mask system+user loss, train on assistant responses only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    logger.info("Starting training...")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Model saved to {final_dir}")

# ============================================================
# Evaluation
# ============================================================
logger.info("Starting evaluation...")
FastLanguageModel.for_inference(model)


def translate(text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
        )
    decoded = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return decoded


# --- sent-CV ---
logger.info("=== sent-CV evaluation ===")
sent_sources = sent_val_short["akk_segment"].tolist()
sent_references = sent_val_short["eng_sentence"].tolist()

sent_predictions = []
for src in tqdm(sent_sources, desc="sent-CV"):
    pred = translate(src)
    sent_predictions.append(clean_translation(pred))

chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
sent_chrf = chrf_metric.corpus_score(sent_predictions, [sent_references]).score
sent_bleu = sacrebleu.corpus_bleu(sent_predictions, [sent_references]).score
sent_geo = (sent_chrf * sent_bleu) ** 0.5 if sent_chrf > 0 and sent_bleu > 0 else 0.0
sent_rep = sum(has_repetition(p) for p in sent_predictions) / len(sent_predictions) * 100

logger.info(f"  sent-CV ({len(sent_predictions)} sents): chrF++={sent_chrf:.2f}, BLEU={sent_bleu:.2f}, geo={sent_geo:.2f}, rep={sent_rep:.1f}%")

# ============================================================
# Save results
# ============================================================
logger.info("=" * 60)
logger.info(f"=== exp040 Qwen3.5-9B LoRA SFT (fold{FOLD}) ===")
logger.info(f"  sent-CV: chrF++={sent_chrf:.2f}, BLEU={sent_bleu:.2f}, geo={sent_geo:.2f}, rep={sent_rep:.1f}%")

sent_out = pd.DataFrame({
    "source": sent_sources,
    "reference": sent_references,
    "prediction": sent_predictions,
})
sent_out.to_csv(str(RESULTS_DIR / "sent_predictions_sft.csv"), index=False)

metrics = {
    "model": MODEL_NAME,
    "fold": FOLD,
    "epochs": cmd_args.epochs,
    "lr": cmd_args.lr,
    "lora_r": cmd_args.lora_r,
    "sent_chrf": sent_chrf,
    "sent_bleu": sent_bleu,
    "sent_geo": sent_geo,
    "sent_rep": sent_rep,
}
with open(str(RESULTS_DIR / "metrics_sft.json"), "w") as f:
    json.dump(metrics, f, indent=2)

logger.info(f"Results saved to {RESULTS_DIR}")
logger.info("Done.")
