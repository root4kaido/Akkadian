"""
文レベルCV評価 (exp006)
- valドキュメントの最初の文（英語ピリオド分割）を参照
- アッカド語入力も先頭部分のみ（文レベル相当）
- 外れ値除去後のvalセットを使用
"""
import os
import sys
import re
import logging
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)
from postprocess import postprocess_batch


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


def run_inference(model, tokenizer, texts, max_length, gen_kwargs, batch_size, device):
    dataset = InferenceDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend([d.strip() for d in decoded])
    return all_preds


def compute_cv(preds, refs, label=""):
    metric_chrf = evaluate.load("chrf")
    metric_bleu = evaluate.load("sacrebleu")
    chrf = metric_chrf.compute(predictions=preds, references=refs)["score"]
    bleu = metric_bleu.compute(predictions=preds, references=[[x] for x in refs])["score"]
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    logger.info(f"CV {label}: chrF={chrf:.2f}, BLEU={bleu:.2f}, geo_mean={geo_mean:.2f}")
    return {"chrf": round(chrf, 2), "bleu": round(bleu, 2), "geo_mean": round(geo_mean, 2)}


def extract_first_sentence(text: str) -> str:
    """英語テキストの最初の文を抽出（ピリオド分割）"""
    m = re.search(r'^(.*?[.!?])(?:\s|$)', text)
    if m:
        return m.group(1).strip()
    return text.strip()


def truncate_akkadian_to_sentence(translit: str, max_bytes: int = 200) -> str:
    """アッカド語transliterationを先頭max_bytesバイトでカット（スペース境界）"""
    encoded = translit.encode('utf-8')
    if len(encoded) <= max_bytes:
        return translit
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space].strip()
    return truncated.strip()


def main():
    config_path = os.path.join(EXP_DIR, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.chdir(EXP_DIR)
    model_path = os.path.join(EXP_DIR, "results", "best_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    prefix = config["model"]["prefix"]
    max_length = config["model"]["max_length"]
    batch_size = config["inference"]["batch_size"]

    # Val分割（preprocess.pyで外れ値除去済み）
    from preprocess import prepare_data
    _, val_prepared = prepare_data(config)

    # val_preparedからoriginal transliteration/translationを復元
    # input_text = prefix + transliteration, target_text = translation
    val_translits = [t[len(prefix):] for t in val_prepared["input_text"].tolist()]
    val_translations = val_prepared["target_text"].tolist()

    # === 文レベルvalデータ作成 ===
    sent_inputs = []
    sent_refs = []
    for translit, translation in zip(val_translits, val_translations):
        first_sent_eng = extract_first_sentence(str(translation))
        first_sent_akk = truncate_akkadian_to_sentence(str(translit), max_bytes=200)

        if len(first_sent_eng.strip()) > 0 and len(first_sent_akk.strip()) > 0:
            sent_inputs.append(prefix + first_sent_akk)
            sent_refs.append(first_sent_eng)

    logger.info(f"Sentence-level val samples: {len(sent_inputs)}")
    logger.info(f"Sample input: {sent_inputs[0][:100]}")
    logger.info(f"Sample ref:   {sent_refs[0][:100]}")

    # === ドキュメントレベル（比較用） ===
    doc_inputs = val_prepared["input_text"].tolist()
    doc_refs = val_prepared["target_text"].tolist()

    # === 推論 ===
    gen_kwargs = {"max_length": max_length, "num_beams": 1}

    logger.info("=== Document-level CV (greedy) ===")
    doc_preds = run_inference(model, tokenizer, doc_inputs, max_length, gen_kwargs, batch_size, device)
    cv_doc = compute_cv(doc_preds, doc_refs, label="doc-level raw")

    logger.info("=== Sentence-level CV (greedy) ===")
    sent_preds = run_inference(model, tokenizer, sent_inputs, max_length, gen_kwargs, batch_size, device)
    cv_sent_raw = compute_cv(sent_preds, sent_refs, label="sent-level raw")

    sent_preds_post = postprocess_batch(sent_preds)
    cv_sent_post = compute_cv(sent_preds_post, sent_refs, label="sent-level post")

    # === 結果保存 ===
    results = {
        "doc_level": cv_doc,
        "sent_level_raw": cv_sent_raw,
        "sent_level_post": cv_sent_post,
        "n_samples": len(sent_inputs),
    }

    results_path = os.path.join(EXP_DIR, "results", "sentence_level_cv.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    logger.info(f"Results saved to {results_path}")

    # 予測結果CSV保存
    sent_results = pd.DataFrame({
        "input": sent_inputs,
        "reference": sent_refs,
        "prediction_raw": sent_preds,
        "prediction_post": sent_preds_post,
    })
    sent_csv_path = os.path.join(EXP_DIR, "results", "val_predictions_sentence.csv")
    sent_results.to_csv(sent_csv_path, index=False)
    logger.info(f"Sentence-level predictions saved to {sent_csv_path}")

    # サンプル出力
    logger.info("\n=== Sample predictions (sentence-level) ===")
    for i in range(min(5, len(sent_preds))):
        logger.info(f"[{i}] input: {sent_inputs[i][:80]}")
        logger.info(f"     ref:  {sent_refs[i][:80]}")
        logger.info(f"     pred: {sent_preds[i][:80]}")


if __name__ == "__main__":
    main()
