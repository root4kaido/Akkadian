"""
exp023: Model Soup — fold0-4のbest_modelを均等平均して1つのモデルを生成
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from safetensors.torch import save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(EXP_DIR, "results")

FOLD_DIRS = [os.path.join(RESULTS_DIR, f"fold{i}", "best_model") for i in range(5)]
OUTPUT_DIR = os.path.join(RESULTS_DIR, "soup_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# fold0をベースとしてロード
print(f"Loading base model from {FOLD_DIRS[0]} ...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(FOLD_DIRS[0])
soup_state = {k: v.clone().float() for k, v in base_model.state_dict().items()}

# fold1-4を加算
for i in range(1, 5):
    print(f"Loading fold{i} from {FOLD_DIRS[i]} ...")
    m = AutoModelForSeq2SeqLM.from_pretrained(FOLD_DIRS[i])
    for k, v in m.state_dict().items():
        soup_state[k] += v.float()
    del m
    torch.cuda.empty_cache()

# 5で割って平均
for k in soup_state:
    soup_state[k] /= 5.0

# ベースモデルに平均重みをロード
base_model.load_state_dict({k: v.to(base_model.dtype) for k, v in soup_state.items()})

# 保存
base_model.save_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(FOLD_DIRS[0])
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Soup model saved to {OUTPUT_DIR}")
print(f"Files: {os.listdir(OUTPUT_DIR)}")
