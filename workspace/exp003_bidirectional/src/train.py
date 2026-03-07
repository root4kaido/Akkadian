"""
ByT5-small 学習スクリプト (exp003)
Starter完全再現: 双方向学習 + save_total_limit=1 + 最終epochモデル使用
"""
import os
import sys
import json
import logging
import yaml
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_callback import TrainerCallback
import evaluate

# ログ設定
log_handlers = [logging.StreamHandler(sys.stdout)]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)

log_file = os.path.join(EXP_DIR, "results", "train.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
log_handlers.append(logging.FileHandler(log_file))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))


class MetricsLogger(TrainerCallback):
    """学習メトリクスをJSONLファイルに保存（可視化用）"""

    def __init__(self, output_path):
        self.output_path = output_path
        self.train_logs = []
        self.eval_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step, **logs}
        if "eval_loss" in logs:
            self.eval_logs.append(entry)
        elif "loss" in logs:
            self.train_logs.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        data = {"train": self.train_logs, "eval": self.eval_logs}
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics log saved to {self.output_path}")


def load_config():
    config_path = os.path.join(EXP_DIR, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def tokenize_fn(examples, tokenizer, max_length):
    """動的パディング: truncationのみ、paddingはDataCollatorに委任"""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["target_text"],
        max_length=max_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    config = load_config()
    seed = config["training"]["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(f"Config: {config['experiment']['name']}")
    logger.info(f"Model: {config['model']['name']}")

    # データ準備
    os.chdir(EXP_DIR)
    sys.path.insert(0, SCRIPT_DIR)
    from preprocess import prepare_data

    train_df, val_df = prepare_data(config)
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    # トークナイザー・モデル
    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # トークナイズ
    train_tokenized = train_dataset.map(
        lambda x: tokenize_fn(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize_fn(x, tokenizer, max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 評価指標
    metric_chrf = evaluate.load("chrf")
    metric_bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        if hasattr(preds, "ndim") and preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        preds = preds.astype(np.int64)
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        chrf = metric_chrf.compute(
            predictions=decoded_preds, references=decoded_labels
        )["score"]
        bleu = metric_bleu.compute(
            predictions=decoded_preds,
            references=[[x] for x in decoded_labels],
        )["score"]

        geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
        return {"chrf": round(chrf, 2), "bleu": round(bleu, 2), "geo_mean": round(geo_mean, 2)}

    # メトリクスログ用コールバック
    metrics_log_path = os.path.join(EXP_DIR, "results", "metrics_log.json")
    metrics_logger = MetricsLogger(metrics_log_path)

    # 学習引数 — Starterと同じ: save_total_limit=1, load_best_model_at_end無効
    output_dir = os.path.join(EXP_DIR, "results", "model")
    tc = config["training"]

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=tc["learning_rate"],
        optim=tc["optimizer"],
        label_smoothing_factor=tc["label_smoothing"],
        fp16=tc["fp16"],
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        weight_decay=tc["weight_decay"],
        save_total_limit=1,
        num_train_epochs=tc["epochs"],
        predict_with_generate=True,
        generation_max_length=config["model"]["max_length"],
        logging_steps=50,
        report_to="none",
        # load_best_model_at_end=False (default) — Starterと同じ、最終epochモデル使用
        seed=seed,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[metrics_logger],
    )

    logger.info("Starting training...")
    trainer.train()

    # 最終epochモデル保存（load_best_model_at_endがFalseなので最終epochが現在のモデル）
    best_dir = os.path.join(EXP_DIR, "results", "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info(f"Final epoch model saved to {best_dir}")

    # 最終評価
    metrics = trainer.evaluate()
    logger.info(f"Final eval metrics: {metrics}")

    # 結果をファイルに保存
    results_path = os.path.join(EXP_DIR, "results", "eval_metrics.yaml")
    with open(results_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    logger.info(f"Metrics saved to {results_path}")


if __name__ == "__main__":
    main()
