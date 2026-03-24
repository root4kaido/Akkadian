"""
s1_exp009: Test generation from SFT'd Qwen3.5-27B.
Generate a few examples to verify domain-style output.
"""

import os
os.environ['UNSLOTH_MOE_DISABLE_AUTOTUNE'] = '1'

from pathlib import Path
import torch
from unsloth import FastLanguageModel

# ============================================================
# Config
# ============================================================
EXP_DIR = Path(__file__).resolve().parents[1]
ADAPTER_DIR = EXP_DIR / "results" / "final_model"
BASE_MODEL = "unsloth/Qwen3.5-27B"
MAX_SEQ_LENGTH = 512

# ============================================================
# Load model + adapter
# ============================================================
print("Loading model...")
model, processor = FastLanguageModel.from_pretrained(
    BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    fast_inference=False,
    dtype=torch.bfloat16,
)
tokenizer = processor.tokenizer

# Load LoRA adapter
from peft import PeftModel
model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
model.eval()

# ============================================================
# Generate
# ============================================================
INSTRUCTION = "Generate an English translation of an ancient Akkadian text."

prompts = [
    INSTRUCTION,
    INSTRUCTION,
    INSTRUCTION,
    "Generate an English translation of an ancient Akkadian administrative tablet.",
    "Generate an English translation of an ancient Akkadian royal inscription.",
]

print("=" * 60)
print("Generation test")
print("=" * 60)

for i, prompt in enumerate(prompts):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print(f"\n--- Example {i+1} [{prompt[:60]}] ---")
    print(generated.strip())
    print()
