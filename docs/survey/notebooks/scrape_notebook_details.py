"""
Kaggle Notebook詳細取得スクリプト
kaggle kernels pullでノートブックをダウンロードし、内容を解析する。
"""
import json
import glob
import os
import subprocess
import re
from datetime import datetime


def get_latest_snapshot():
    """最新のスナップショットファイルを取得"""
    files = sorted(glob.glob("snapshot_*.json"))
    files = [f for f in files if "_details" not in f]
    if not files:
        raise FileNotFoundError("No snapshot files found. Run scrape_notebooks.py first.")
    return files[-1]


def deduplicate_notebooks(notebooks):
    """kaggle_refベースで重複除去し、最大votesを保持"""
    seen = {}
    for nb in notebooks:
        ref = nb["kaggle_ref"]
        if ref not in seen or nb.get("votes", 0) > seen[ref].get("votes", 0):
            seen[ref] = nb
    # slugからタイトルを生成（タイトルが不正な場合）
    for ref, nb in seen.items():
        title = nb["title"]
        if re.match(r'^\d+\s*comments?$', title) or title == nb["author"] or len(title) < 5:
            # slugをタイトル化
            nb["title"] = nb["slug"].replace("-", " ").title()
    return sorted(seen.values(), key=lambda x: x.get("votes", 0), reverse=True)


def download_notebook(kaggle_ref, output_dir="downloaded"):
    """kaggle kernels pullでノートブックをダウンロード"""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = kaggle_ref.replace("/", "__")
    output_path = os.path.join(output_dir, safe_name)
    os.makedirs(output_path, exist_ok=True)

    try:
        result = subprocess.run(
            ["kaggle", "kernels", "pull", kaggle_ref, "-p", output_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  kaggle pull failed: {result.stderr.strip()}")
            return None

        # .ipynbファイルを探す
        ipynb_files = glob.glob(os.path.join(output_path, "*.ipynb"))
        if ipynb_files:
            return ipynb_files[0]
        # .pyファイルも探す
        py_files = glob.glob(os.path.join(output_path, "*.py"))
        if py_files:
            return py_files[0]
        return None
    except Exception as e:
        print(f"  Download error: {e}")
        return None


def analyze_notebook(filepath):
    """ノートブックの内容を解析"""
    analysis = {
        "libraries": [],
        "models": [],
        "techniques": [],
        "code_summary": "",
        "markdown_summary": "",
    }

    if filepath.endswith(".ipynb"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                nb = json.load(f)
        except Exception as e:
            analysis["error"] = str(e)
            return analysis

        code_cells = []
        markdown_cells = []

        for cell in nb.get("cells", []):
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code":
                code_cells.append(source)
            elif cell.get("cell_type") == "markdown":
                markdown_cells.append(source)

        all_code = "\n".join(code_cells)
        all_markdown = "\n".join(markdown_cells)

    elif filepath.endswith(".py"):
        with open(filepath, "r", encoding="utf-8") as f:
            all_code = f.read()
        all_markdown = ""
    else:
        return analysis

    # ライブラリ抽出
    import_pattern = re.compile(r'^(?:from|import)\s+(\S+)', re.MULTILINE)
    imports = set()
    for match in import_pattern.finditer(all_code):
        lib = match.group(1).split(".")[0]
        if lib and lib not in ("__future__",):
            imports.add(lib)
    analysis["libraries"] = sorted(imports)

    # モデル検出
    model_keywords = {
        "transformers": ["AutoModelForSeq2SeqLM", "AutoModelForCausalLM", "T5", "mBART", "NLLB",
                         "MarianMT", "Helsinki", "facebook/nllb", "google/mt5", "Seq2Seq",
                         "BertModel", "GPT", "Llama", "Mistral", "Qwen", "ByT5", "byt5"],
        "pytorch": ["nn.Module", "nn.Linear", "nn.Transformer"],
        "tensorflow": ["tf.keras", "tensorflow"],
        "sklearn": ["sklearn", "RandomForest", "XGBoost", "LightGBM"],
    }
    detected_models = set()
    for category, keywords in model_keywords.items():
        for kw in keywords:
            if kw.lower() in all_code.lower():
                detected_models.add(kw)
    analysis["models"] = sorted(detected_models)

    # テクニック検出
    technique_keywords = [
        "back-translation", "backtranslation", "data augmentation", "augment",
        "beam search", "beam_search", "num_beams", "greedy", "ensemble",
        "bleu", "sacrebleu", "chrF",
        "tokenizer", "sentencepiece", "bpe", "subword",
        "fine-tune", "finetune", "fine_tune", "lora", "qlora", "peft",
        "rag", "retrieval", "few-shot", "prompt",
        "cuneiform", "akkadian", "transliteration",
        "curriculum learning", "knowledge distillation",
        "cross-validation", "kfold",
        "mbr", "minimum bayes risk", "post-processing", "postprocess",
        "sentence align", "sentencealign",
        "regex", "rule-based", "rule_based",
    ]
    detected_techniques = set()
    combined = (all_code + " " + all_markdown).lower()
    for tech in technique_keywords:
        if tech.lower() in combined:
            detected_techniques.add(tech)
    analysis["techniques"] = sorted(detected_techniques)

    # コードサマリー（先頭15KB）
    analysis["code_summary"] = all_code[:15000]
    analysis["markdown_summary"] = all_markdown[:5000]

    return analysis


def scrape_details(snapshot_file=None, top_n=20):
    """上位N件のノートブックの詳細を取得"""
    if snapshot_file is None:
        snapshot_file = get_latest_snapshot()

    with open(snapshot_file, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    notebooks = snapshot["notebooks"]

    # 重複除去・タイトル修正
    notebooks = deduplicate_notebooks(notebooks)
    print(f"After dedup: {len(notebooks)} unique notebooks")

    # 上位N件を処理
    target = notebooks[:top_n]
    print(f"Processing top {len(target)} notebooks from {snapshot_file}")

    for i, nb in enumerate(target):
        ref = nb["kaggle_ref"]
        print(f"\n[{i+1}/{len(target)}] {nb['title'][:60]} ({ref}) votes={nb.get('votes', '?')}")

        # ダウンロード
        filepath = download_notebook(ref)
        if filepath:
            print(f"  Downloaded: {filepath}")
            analysis = analyze_notebook(filepath)
            nb["analysis"] = analysis
            nb["downloaded"] = True
            print(f"  Libraries: {', '.join(analysis['libraries'][:10])}")
            print(f"  Models: {', '.join(analysis['models'][:5])}")
            print(f"  Techniques: {', '.join(analysis['techniques'][:5])}")
        else:
            nb["downloaded"] = False
            print(f"  Failed to download")

    # 保存
    output_file = snapshot_file.replace(".json", "_details.json")
    snapshot["notebooks"] = notebooks
    snapshot["details_scraped_at"] = datetime.now().isoformat()
    snapshot["top_n_analyzed"] = top_n

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"\nSaved details to {output_file}")
    return snapshot


if __name__ == "__main__":
    scrape_details(top_n=20)
