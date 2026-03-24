"""
s1_exp009: Bulk generation of domain-style English text using SFT'd Qwen3.5-27B.
Generates 100k samples with high randomness, saving incrementally to CSV.
"""

import os
os.environ['UNSLOTH_MOE_DISABLE_AUTOTUNE'] = '1'

import csv
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import PeftModel

# ============================================================
# Config
# ============================================================
TOTAL_SAMPLES = 100_000
BATCH_SIZE = 8  # generate this many per forward pass
SAVE_EVERY = 100  # flush to CSV every N samples
MAX_NEW_TOKENS = 384

# High randomness settings
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.15

EXP_DIR = Path(__file__).resolve().parents[1]
ADAPTER_DIR = EXP_DIR / "results" / "final_model"
BASE_MODEL = "unsloth/Qwen3.5-27B"
MAX_SEQ_LENGTH = 512
OUTPUT_CSV = EXP_DIR / "results" / "generated_english_2.csv"

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(EXP_DIR / "results" / "generate_bulk.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

INSTRUCTION = "Generate an English translation of an ancient Akkadian text."


def main():
    logger.info("=" * 60)
    logger.info("Bulk generation: 100k domain English samples")
    logger.info("=" * 60)

    # --------------------------------------------------------
    # Resume: check how many already generated
    # --------------------------------------------------------
    existing_count = 0
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            existing_count = sum(1 for _ in reader)
        logger.info(f"Resuming from {existing_count} existing samples")

    remaining = TOTAL_SAMPLES - existing_count
    if remaining <= 0:
        logger.info("Already generated enough samples. Done.")
        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    logger.info("Loading model...")
    model, processor = FastLanguageModel.from_pretrained(
        BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=False,
        dtype=torch.bfloat16,
    )
    tokenizer = processor.tokenizer
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()

    # Prepare input template (same for all samples)
    messages = [{"role": "user", "content": INSTRUCTION}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(model.device)
    input_len = input_ids.shape[1]

    # Expand for batch
    def make_batch(bs):
        return input_ids.expand(bs, -1)

    # --------------------------------------------------------
    # Generate loop
    # --------------------------------------------------------
    write_mode = "a" if existing_count > 0 else "w"
    csv_file = open(OUTPUT_CSV, write_mode, newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if existing_count == 0:
        writer.writerow(["id", "generated_translation"])

    generated_count = 0
    buffer = []
    start_time = time.time()

    logger.info(f"Generating {remaining} samples (batch_size={BATCH_SIZE})...")
    num_batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(total=remaining, initial=0, desc="Generating", unit="samples")

    try:
        while generated_count < remaining:
            bs = min(BATCH_SIZE, remaining - generated_count)
            batch_input = make_batch(bs)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch_input,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    do_sample=True,
                    repetition_penalty=REPETITION_PENALTY,
                )

            for j in range(bs):
                text = tokenizer.decode(
                    outputs[j][input_len:],
                    skip_special_tokens=True,
                ).strip()

                if len(text) >= 10:  # skip empty/too short
                    global_id = existing_count + generated_count + 1
                    buffer.append([global_id, text])

                generated_count += 1

            pbar.update(bs)

            # Flush periodically
            if len(buffer) >= SAVE_EVERY:
                for row in buffer:
                    writer.writerow(row)
                csv_file.flush()
                buffer = []

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving remaining buffer...")
    finally:
        pbar.close()
        # Save any remaining buffer
        for row in buffer:
            writer.writerow(row)
        csv_file.flush()
        csv_file.close()
        total = existing_count + generated_count
        logger.info(f"Saved {total} total samples to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
