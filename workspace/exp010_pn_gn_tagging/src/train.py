"""
ByT5-base 学習スクリプト (exp010)
exp008と同一ロジック。preprocess.pyがPN/GNタグ付加済みデータを返す。
"""
import os

os.environ["WANDB_PROJECT"] = "akkadian-translation"

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
    force=True,
)
logger = logging.getLogger(__name__)

# transformersのloggerにもFileHandlerを追加（propagate=False & level=WARNINGのため）
_tf_logger = logging.getLogger("transformers")
_tf_logger.setLevel(logging.INFO)
_file_handler = logging.FileHandler(log_file)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_tf_logger.addHandler(_file_handler)

PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))


class MetricsLogger(TrainerCallback):
    """学習メトリクスをJSONファイルに保存"""

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


def tokenize_fn(examples, tokenizer, max_length, reverse_encoder_max_length):
    """方向別のmax_lengthでトークナイズ。
    - forward: encoder=max_length(512), decoder=max_length(512)
    - reverse: encoder=reverse_encoder_max_length(1024), decoder=max_length(512)
    """
    input_texts = examples["input_text"]
    target_texts = examples["target_text"]
    directions = examples["direction"]

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for inp, tgt, direction in zip(input_texts, target_texts, directions):
        enc_max = reverse_encoder_max_length if direction == "reverse" else max_length
        dec_max = max_length

        enc = tokenizer(inp, max_length=enc_max, truncation=True)
        dec = tokenizer(tgt, max_length=dec_max, truncation=True)

        all_input_ids.append(enc["input_ids"])
        all_attention_masks.append(enc["attention_mask"])
        all_labels.append(dec["input_ids"])

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


def main():
    config = load_config()
    seed = config["training"]["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    max_length = config["model"]["max_length"]
    reverse_encoder_max_length = config["model"]["reverse_encoder_max_length"]

    logger.info(f"Config: {config['experiment']['name']}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Forward: encoder={max_length}, decoder={max_length}")
    logger.info(f"Reverse: encoder={reverse_encoder_max_length}, decoder={max_length}")

    # データ準備
    os.chdir(EXP_DIR)
    sys.path.insert(0, SCRIPT_DIR)
    from preprocess import prepare_data

    train_df, val_df = prepare_data(config)
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    # トークナイザー・モデル
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # トークナイズ
    train_tokenized = train_dataset.map(
        lambda x: tokenize_fn(x, tokenizer, max_length, reverse_encoder_max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize_fn(x, tokenizer, max_length, reverse_encoder_max_length),
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

    # 学習引数
    output_dir = os.path.join(EXP_DIR, "results", "model")
    tc = config["training"]

    training_args = {
        "output_dir": output_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": tc["learning_rate"],
        "optim": tc["optimizer"],
        "label_smoothing_factor": tc["label_smoothing"],
        "fp16": tc.get("fp16", False),
        "bf16": tc.get("bf16", False),
        "per_device_train_batch_size": tc["batch_size"],
        "per_device_eval_batch_size": config["inference"]["batch_size"],
        "gradient_accumulation_steps": tc["gradient_accumulation_steps"],
        "weight_decay": tc["weight_decay"],
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "geo_mean",
        "greater_is_better": True,
        "num_train_epochs": tc["epochs"],
        "predict_with_generate": True,
        "generation_max_length": max_length,
        "logging_steps": 50,
        "report_to": "wandb",
        "run_name": config["experiment"]["name"],
        "seed": seed,
        "dataloader_num_workers": 2,
    }

    # Cosine scheduler
    scheduler = tc.get("scheduler", "linear")
    if scheduler == "cosine":
        training_args["lr_scheduler_type"] = "cosine"
        warmup_ratio = tc.get("warmup_ratio", 0.1)
        training_args["warmup_ratio"] = warmup_ratio
        logger.info(f"Scheduler: cosine, warmup_ratio={warmup_ratio}")

    args = Seq2SeqTrainingArguments(**training_args)

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

    # チェックポイント再開
    resume_from = tc.get("resume_from")
    if resume_from and os.path.isdir(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        # output_dir内に既存checkpointがあれば自動再開
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")] if os.path.isdir(output_dir) else []
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            ckpt_path = os.path.join(output_dir, latest)
            logger.info(f"Auto-resuming from latest checkpoint: {ckpt_path}")
            trainer.train(resume_from_checkpoint=ckpt_path)
        else:
            logger.info("Starting training from scratch...")
            trainer.train()

    # ベストモデル保存
    best_dir = os.path.join(EXP_DIR, "results", "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info(f"Best model saved to {best_dir}")

    # 最終評価
    metrics = trainer.evaluate()
    logger.info(f"Best model eval metrics: {metrics}")

    # 結果をファイルに保存
    results_path = os.path.join(EXP_DIR, "results", "eval_metrics.yaml")
    with open(results_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    logger.info(f"Metrics saved to {results_path}")


if __name__ == "__main__":
    main()
