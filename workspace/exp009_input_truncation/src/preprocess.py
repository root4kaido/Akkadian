"""
前処理モジュール (exp009): 確率的入力truncation augmentation
- exp008のロジックをベースに、学習時にAkkadian入力を確率的に先頭200バイトに切り詰め
- 静的データ準備(prepare_data)はexp008と同一（ラベルマスキング+双方向）
- 動的augmentationはAugmentedDatasetクラスで各__getitem__時に適用
"""
import re
import random
import pandas as pd
import torch
from torch.utils.data import Dataset


def truncate_to_sentence_boundary(text: str, max_bytes: int) -> str:
    """テキストをmax_bytesに収まるよう文末（ピリオド）境界でtruncateする。"""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text

    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')

    last_period = -1
    for m in re.finditer(r'\.\s', truncated):
        last_period = m.end()

    if truncated.rstrip().endswith('.'):
        last_period = max(last_period, len(truncated.rstrip()))

    if last_period > 0:
        return truncated[:last_period].strip()

    return truncated.strip()


def truncate_bytes_at_word_boundary(text: str, max_bytes: int) -> str:
    """テキストをmax_bytesに収まるよう単語境界（スペース）でtruncateする。"""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space].strip()
    return truncated.strip()


def extract_first_sentence(text: str) -> str:
    """英語テキストの最初の文を抽出"""
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    if m:
        return m.group(1).strip()
    return str(text).strip()


def prepare_data(config: dict) -> tuple:
    """
    静的データ準備（exp008と同一ロジック）
    Returns: (train_df, val_df)
    - train_dfには生テキスト列を保持（augmentationはDataset側で実施）
    """
    from sklearn.model_selection import train_test_split

    train_path = config["data"]["train_path"]
    df = pd.read_csv(train_path)
    print(f"Raw train data: {len(df)} rows")

    df = df[
        (df["transliteration"].astype(str).str.len() > 0)
        & (df["translation"].astype(str).str.len() > 0)
    ]
    print(f"After filtering empty: {len(df)} rows")

    seed = config["training"]["seed"]
    val_ratio = config["training"]["val_ratio"]
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed)

    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class AugmentedTrainDataset(Dataset):
    """
    学習用Dataset: 各__getitem__で確率的にinput truncation augmentationを適用。

    doc版（確率 1-p）:
      Forward:  prefix_fwd + Akk全文              → Eng 1文目（ラベルマスク）
      Reverse:  prefix_rev + Eng全文              → Akk全文

    sent版（確率 p）:
      Forward:  prefix_fwd + Akk先頭200B          → Eng 1文目（ラベルマスク）
      Reverse:  prefix_rev + Eng 1文目            → Akk先頭200B
    """

    def __init__(self, train_df, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config

        prefix_fwd = config["model"]["prefix"]
        prefix_rev = config["model"]["prefix_reverse"]
        use_bidirectional = config["data"].get("use_bidirectional", False)
        max_label_bytes = config["model"]["max_length"]

        aug_config = config.get("augmentation", {}).get("input_truncation", {})
        self.aug_enabled = aug_config.get("enabled", False)
        self.aug_prob = aug_config.get("prob", 0.5)
        self.aug_max_bytes = aug_config.get("max_bytes", 200)

        self.max_length = config["model"]["max_length"]
        self.reverse_encoder_max_length = config["model"]["reverse_encoder_max_length"]

        # 各サンプルの生データを保持
        self.samples = []
        for _, row in train_df.iterrows():
            akk = str(row["transliteration"])
            eng = str(row["translation"])
            eng_masked = truncate_to_sentence_boundary(eng, max_label_bytes)
            eng_first_sent = extract_first_sentence(eng)
            akk_truncated = truncate_bytes_at_word_boundary(akk, self.aug_max_bytes)

            # Forward sample
            self.samples.append({
                "direction": "forward",
                "prefix": prefix_fwd,
                "akk_full": akk,
                "akk_truncated": akk_truncated,
                "eng_full": eng,
                "eng_masked": eng_masked,
                "eng_first_sent": eng_first_sent,
            })

            # Reverse sample (bidirectional)
            if use_bidirectional:
                self.samples.append({
                    "direction": "reverse",
                    "prefix": prefix_rev,
                    "akk_full": akk,
                    "akk_truncated": akk_truncated,
                    "eng_full": eng,
                    "eng_masked": eng_masked,
                    "eng_first_sent": eng_first_sent,
                })

        # Shuffle with seed
        random.seed(config["training"]["seed"])
        random.shuffle(self.samples)

        # マスキング統計
        n_label_truncated = sum(
            1 for s in self.samples
            if s["direction"] == "forward" and s["eng_masked"] != s["eng_full"]
        )
        n_fwd = sum(1 for s in self.samples if s["direction"] == "forward")
        print(f"Forward label masking: {n_label_truncated}/{n_fwd} labels truncated")
        print(f"Total train samples: {len(self.samples)} (aug_prob={self.aug_prob}, aug_bytes={self.aug_max_bytes})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        use_sent = self.aug_enabled and random.random() < self.aug_prob

        if s["direction"] == "forward":
            if use_sent:
                input_text = s["prefix"] + s["akk_truncated"]
                target_text = s["eng_masked"]
            else:
                input_text = s["prefix"] + s["akk_full"]
                target_text = s["eng_masked"]
            enc_max = self.max_length
        else:  # reverse
            if use_sent:
                input_text = s["prefix"] + s["eng_first_sent"]
                target_text = s["akk_truncated"]
            else:
                input_text = s["prefix"] + s["eng_full"]
                target_text = s["akk_full"]
            enc_max = self.reverse_encoder_max_length

        enc = self.tokenizer(input_text, max_length=enc_max, truncation=True)
        dec = self.tokenizer(target_text, max_length=self.max_length, truncation=True)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": dec["input_ids"],
        }
