"""Extract training code links, model references, and scores from downloaded notebooks."""
import json
import re
import os
from pathlib import Path

DOWNLOAD_DIR = Path("downloaded")

def extract_from_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    all_text = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            all_text.extend(src)
        else:
            all_text.append(src)

    full_text = "\n".join(all_text)

    # Find kaggle code links
    code_links = re.findall(r'https?://(?:www\.)?kaggle\.com/code/[^\s\)\]"\']+', full_text)

    # Find kaggle model/dataset references
    model_links = re.findall(r'https?://(?:www\.)?kaggle\.com/(?:models|datasets)/[^\s\)\]"\']+', full_text)

    # Find /kaggle/input/ paths (model references)
    input_paths = re.findall(r'/kaggle/input/[^\s\)\]"\']+', full_text)

    # Find model_path or model references
    model_paths = re.findall(r'(?:model_path|MODEL_PATH|model_dir|MODEL_DIR|model_name)\s*=\s*["\']([^"\']+)["\']', full_text)

    # Find training-related mentions
    train_mentions = []
    for line in full_text.split("\n"):
        if re.search(r'train(?:ing)?\s*(?:notebook|code|script)', line, re.I):
            train_mentions.append(line.strip()[:200])

    # Find score mentions
    score_mentions = []
    for line in full_text.split("\n"):
        if re.search(r'(?:LB|score|public|leaderboard)\s*[:=]?\s*\d+\.?\d*', line, re.I):
            score_mentions.append(line.strip()[:200])

    # Find linked training notebooks specifically
    train_links = [l for l in code_links if 'train' in l.lower()]

    return {
        "code_links": list(set(code_links)),
        "train_links": list(set(train_links)),
        "model_links": list(set(model_links)),
        "input_paths": list(set(input_paths)),
        "model_paths": list(set(model_paths)),
        "train_mentions": train_mentions[:5],
        "score_mentions": score_mentions[:5],
    }

results = {}
for nb_dir in sorted(DOWNLOAD_DIR.iterdir()):
    if not nb_dir.is_dir():
        continue
    for nb_file in nb_dir.glob("*.ipynb"):
        slug = nb_dir.name
        info = extract_from_notebook(nb_file)
        # Only report if there's something interesting
        has_content = any([info["code_links"], info["train_links"], info["model_links"],
                         info["input_paths"], info["model_paths"]])
        if has_content:
            results[slug] = info

print("=" * 80)
print("NOTEBOOK ANALYSIS: Training Links & Model References")
print("=" * 80)

for slug, info in results.items():
    print(f"\n{'─' * 60}")
    print(f"📓 {slug}")
    if info["train_links"]:
        print(f"  🎯 TRAINING LINKS: {info['train_links']}")
    if info["code_links"]:
        other = [l for l in info["code_links"] if l not in info["train_links"]]
        if other:
            print(f"  🔗 Other code links: {other}")
    if info["input_paths"]:
        print(f"  📂 Input paths: {info['input_paths']}")
    if info["model_paths"]:
        print(f"  🤖 Model paths: {info['model_paths']}")
    if info["model_links"]:
        print(f"  📦 Model/Dataset links: {info['model_links']}")
    if info["train_mentions"]:
        print(f"  📝 Training mentions:")
        for m in info["train_mentions"]:
            print(f"     {m}")
    if info["score_mentions"]:
        print(f"  🏆 Score mentions:")
        for m in info["score_mentions"]:
            print(f"     {m}")
