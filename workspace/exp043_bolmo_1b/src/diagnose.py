"""
Bolmo-1B 診断スクリプト: モデルの出力を詳しく調べる
Usage: python workspace/exp043_bolmo_1b/src/diagnose.py <model_path>
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "workspace/exp043_bolmo_1b/results/fold3/last_model"

print(f"=== Loading from {MODEL_PATH} ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Vocab size: {tokenizer.vocab_size}, len(tokenizer): {len(tokenizer)}")
print(f"EOS: '{tokenizer.eos_token}' id={tokenizer.eos_token_id}")
print(f"PAD: '{tokenizer.pad_token}' id={tokenizer.pad_token_id}")
print(f"BOS: '{tokenizer.bos_token}' id={tokenizer.bos_token_id}")
print(f"All special tokens: {tokenizer.all_special_tokens}")
print(f"Special tokens map: {tokenizer.special_tokens_map}")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Model embedding size: {model.get_input_embeddings().weight.shape}")
print(f"Model config pad_token_id: {model.config.pad_token_id}")
print(f"Model config eos_token_id: {getattr(model.config, 'eos_token_id', 'N/A')}")

# Test prompt
prompt = "Translate Akkadian to English.\nSource: a-na a-bi-a qi2-bi2-ma\nTarget: "
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
print(f"\n=== Prompt ===")
print(f"Text: {repr(prompt)}")
print(f"Token IDs (first 20): {input_ids[0][:20].tolist()}")
print(f"Token IDs (last 20): {input_ids[0][-20:].tolist()}")
print(f"Prompt length: {input_ids.shape[1]}")

# Pad to 64 multiple (right pad)
seq_len = input_ids.shape[1]
padded_len = ((seq_len + 63) // 64) * 64
pad_size = padded_len - seq_len
if pad_size > 0:
    padded_ids = torch.cat([
        input_ids,
        torch.full((1, pad_size), tokenizer.pad_token_id, dtype=input_ids.dtype, device=device),
    ], dim=1)
    attn_mask = torch.cat([
        torch.ones((1, seq_len), dtype=torch.long, device=device),
        torch.zeros((1, pad_size), dtype=torch.long, device=device),
    ], dim=1)
else:
    padded_ids = input_ids
    attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

print(f"Padded length: {padded_ids.shape[1]}")

with torch.no_grad():
    outputs = model(padded_ids, attention_mask=attn_mask)

logits = outputs.logits[0, seq_len - 1, :]  # Last real token
probs = torch.softmax(logits.float(), dim=-1)
top_k = torch.topk(probs, 20)

print(f"\n=== Top-20 predicted tokens (at position {seq_len - 1}) ===")
for i, (prob, idx) in enumerate(zip(top_k.values, top_k.indices)):
    token_id = idx.item()
    try:
        token_str = tokenizer.decode([token_id])
    except Exception:
        token_str = f"<decode_error>"
    is_eos = " <<<< EOS" if token_id == tokenizer.eos_token_id else ""
    is_pad = " <<<< PAD" if token_id == tokenizer.pad_token_id else ""
    print(f"  #{i+1}: id={token_id:6d}, prob={prob.item():.6f}, token='{token_str}'{is_eos}{is_pad}")

# Also check: what does the model predict WITHOUT padding (raw length)?
print(f"\n=== Without padding (raw seq_len={seq_len}) ===")
try:
    with torch.no_grad():
        outputs_raw = model(input_ids)
    logits_raw = outputs_raw.logits[0, -1, :]
    probs_raw = torch.softmax(logits_raw.float(), dim=-1)
    top_k_raw = torch.topk(probs_raw, 10)
    for i, (prob, idx) in enumerate(zip(top_k_raw.values, top_k_raw.indices)):
        token_id = idx.item()
        try:
            token_str = tokenizer.decode([token_id])
        except Exception:
            token_str = f"<decode_error>"
        is_eos = " <<<< EOS" if token_id == tokenizer.eos_token_id else ""
        print(f"  #{i+1}: id={token_id:6d}, prob={prob.item():.6f}, token='{token_str}'{is_eos}")
except Exception as e:
    print(f"  Error (expected if not multiple of 64): {e}")

# Check: pretrained model (not fine-tuned) for comparison
print(f"\n=== Compare with pretrained allenai/Bolmo-1B ===")
try:
    tok_orig = AutoTokenizer.from_pretrained("allenai/Bolmo-1B", trust_remote_code=True)
    print(f"Original vocab size: {tok_orig.vocab_size}, len: {len(tok_orig)}")
    print(f"Original EOS: '{tok_orig.eos_token}' id={tok_orig.eos_token_id}")
    print(f"Original PAD: '{tok_orig.pad_token}' id={tok_orig.pad_token_id}")
    print(f"Original special tokens: {tok_orig.all_special_tokens}")
except Exception as e:
    print(f"  Could not load original: {e}")

print("\n=== Done ===")
