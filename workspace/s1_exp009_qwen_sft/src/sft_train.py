"""
s1_exp009: Qwen3.5-27B QLoRA SFT on Akkadian English translations.
Goal: Train a domain-style English text generator for data augmentation.
Uses unsloth following official Qwen3.5 notebook pattern.
"""

import os
os.environ['UNSLOTH_MOE_DISABLE_AUTOTUNE'] = '1'

import re
import logging
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig

# ============================================================
# Config
# ============================================================
MODEL_NAME = "unsloth/Qwen3.5-27B"
SEED = 42
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
LORA_R = 64
MAX_SEQ_LENGTH = 512

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXP_DIR / "results" / "sft_model"

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(EXP_DIR / "results" / "sft_train.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Preprocessing (exp023 compatible)
# ============================================================
ROMAN_TO_INT = {
    "XII": "12", "XIII": "13", "XIV": "14", "XI": "11",
    "VIII": "8", "VII": "7", "VI": "6", "IV": "4",
    "IX": "9", "III": "3", "II": "2", "I": "1",
    "X": "10", "V": "5",
}

MONTH_NAMES_TRANSLATION = {
    r"Kuzallu": "11", r"Allanaatum": "12", r"Allanat[uü]m": "12",
    r"Hubur": "6", r"Ab[- ]?sarranu?i?": "1",
    r"Sa[ -]?k[eē]n[aā]tim": "2", r"Mahhur[ -]?il[iī]": "5",
    r"Ša[ -]?šar[rn]?[aā]ni": "5",
    r"Kanwarta": "4", r"Tamhirtum": "7",
    r"Kalit": "8", r"T[eē]'in[aā]tim": "9",
    r"Narmak[ -]?Aššur": "3",
}

FRACTION_MAP = {
    "0.5": "\u00BD", "0.3333": "\u2153", "0.6666": "\u2154",
    "0.25": "\u00BC", "0.75": "\u00BE", "0.1666": "\u2159",
    "0.8333": "\u215A", "0.625": "\u215D",
}


def _decimal_to_fraction(m):
    val = m.group(0)
    for dec, frac in FRACTION_MAP.items():
        if abs(float(val) - float(dec)) < 0.002:
            return frac
    return val


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


# ============================================================
# Data
# ============================================================
def load_and_prepare_data(tokenizer):
    train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))
    logger.info(f"Loaded {len(train_df)} documents")

    train_df["translation"] = train_df["translation"].astype(str).apply(clean_translation)

    INSTRUCTION = "Generate an English translation of an ancient Akkadian text."

    conversations = []
    skipped = 0
    for _, row in train_df.iterrows():
        translation = str(row["translation"]).strip()
        if len(translation) < 10:
            skipped += 1
            continue
        conversations.append([
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": translation},
        ])

    logger.info(f"Prepared {len(conversations)} conversations (skipped {skipped})")

    dataset = Dataset.from_dict({"conversations": conversations})

    # Apply chat template
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
    return dataset


# ============================================================
# Main
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("s1_exp009: Qwen3.5-27B QLoRA SFT (unsloth)")
    logger.info("=" * 60)

    # Load model in bf16 (QLoRA 4-bit is not recommended for MoE)
    model, processor = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=False,
        dtype=torch.bfloat16,
    )
    tokenizer = processor.tokenizer

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_R * 2,
        use_gradient_checkpointing=True,
        random_state=SEED,
        bias="none",
    )

    # Prepare data
    dataset = load_and_prepare_data(tokenizer)

    # Training (following official pattern exactly)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            seed=SEED,
            report_to="none",
            optim="adamw_8bit",
            max_grad_norm=1.0,
            output_dir=str(OUTPUT_DIR),
            bf16=True,
        ),
    )

    # Train on responses only (mask instruction loss)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    logger.info("Starting training...")
    trainer.train()

    # Save LoRA adapter
    final_dir = EXP_DIR / "results" / "final_model"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
