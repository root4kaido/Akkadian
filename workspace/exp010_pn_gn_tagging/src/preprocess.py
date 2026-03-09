"""
前処理モジュール (exp010): PN/GNタグ付加
- OA_Lexiconのform→type辞書を使い、翻字トークンにPN/GNタグを付加
- 例: "KIŠIB ma-nu-ba-lúm-a-šur DUMU" → "KIŠIB ma-nu-ba-lúm-a-šur[PN] DUMU"
- ラベルマスキング・双方向はexp008と同一
"""
import json
import os
import re

import pandas as pd


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


def load_form_tag_dict(dict_path: str) -> dict:
    """form→tag辞書をJSONから読み込む"""
    with open(dict_path) as f:
        return json.load(f)


def tag_transliteration(text: str, form_tag_dict: dict) -> str:
    """翻字テキストのトークンにPN/GNタグを付加する。

    スペース区切りの各トークンを辞書でルックアップし、
    マッチしたPN/GNトークンに[PN]や[GN]を付加。
    """
    tokens = text.split()
    tagged_tokens = []
    for token in tokens:
        tag = form_tag_dict.get(token)
        if tag:
            tagged_tokens.append(f"{token}[{tag}]")
        else:
            tagged_tokens.append(token)
    return " ".join(tagged_tokens)


def prepare_data(config: dict) -> tuple:
    """
    データ準備: PN/GNタグ付加 + ラベルマスキング + 双方向
    Returns: (train_df, val_df)
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

    # 辞書読み込み
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.dirname(script_dir)
    dict_path = os.path.join(exp_dir, "dataset", "form_type_dict.json")
    form_tag_dict = load_form_tag_dict(dict_path)
    print(f"Loaded form_tag_dict: {len(form_tag_dict)} entries")

    # タグ付加
    df["transliteration_tagged"] = df["transliteration"].astype(str).apply(
        lambda t: tag_transliteration(t, form_tag_dict)
    )

    # タグ付加統計
    orig_lens = df["transliteration"].astype(str).apply(lambda t: len(t.encode('utf-8')))
    tagged_lens = df["transliteration_tagged"].apply(lambda t: len(t.encode('utf-8')))
    increase_ratio = (tagged_lens / orig_lens).mean()
    n_tagged = (tagged_lens > orig_lens).sum()
    print(f"Tagging stats: {n_tagged}/{len(df)} docs modified, avg length increase: {increase_ratio:.3f}x")

    # Train/Val分割
    seed = config["training"]["seed"]
    val_ratio = config["training"]["val_ratio"]
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed)

    prefix_fwd = config["model"]["prefix"]
    prefix_rev = config["model"]["prefix_reverse"]
    use_bidirectional = config["data"].get("use_bidirectional", False)
    max_length = config["model"]["max_length"]

    # 順方向: タグ付き翻字 → 英語ラベルマスク
    translations_masked = train_df["translation"].astype(str).apply(
        lambda t: truncate_to_sentence_boundary(t, max_length)
    )

    n_truncated = (translations_masked.apply(lambda t: len(t.encode('utf-8')))
                   < train_df["translation"].astype(str).apply(lambda t: len(t.encode('utf-8')))).sum()
    print(f"Forward label masking: {n_truncated}/{len(train_df)} labels truncated")

    fwd = pd.DataFrame({
        "input_text": prefix_fwd + train_df["transliteration_tagged"].values,
        "target_text": translations_masked.values,
        "direction": "forward",
    })

    if use_bidirectional:
        # 逆方向: 英語 → タグ付き翻字（モデルがタグも含めて生成を学習）
        bwd = pd.DataFrame({
            "input_text": prefix_rev + train_df["translation"].astype(str).values,
            "target_text": train_df["transliteration_tagged"].values,
            "direction": "reverse",
        })
        train_prepared = pd.concat([fwd, bwd], ignore_index=True)
        train_prepared = train_prepared.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"Train (bidirectional): {len(train_prepared)} rows")
    else:
        train_prepared = fwd.reset_index(drop=True)
        print(f"Train (unidirectional): {len(train_prepared)} rows")

    # Val: タグ付き翻字で評価（マスキングなし）
    val_prepared = pd.DataFrame({
        "input_text": prefix_fwd + val_df["transliteration_tagged"].values,
        "target_text": val_df["translation"].astype(str).values,
        "direction": "forward",
    }).reset_index(drop=True)
    print(f"Val: {len(val_prepared)} rows")

    return train_prepared, val_prepared
