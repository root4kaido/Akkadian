"""
前処理モジュール (exp006): 外れ値データ除去 + ラベルマスキング
- exp005ベース
- split後にword_ratio外れ値を除去（train/val両方から）
- validationのsplit自体はexp005と同じ（seed/ratio固定）
"""
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


def compute_word_ratio(df: pd.DataFrame) -> pd.Series:
    """英語/アッカド語の単語数比率を計算"""
    akk_words = df["transliteration"].astype(str).str.split().str.len()
    eng_words = df["translation"].astype(str).str.split().str.len()
    # ゼロ除算回避
    return eng_words / akk_words.clip(lower=1)


def filter_outliers(df: pd.DataFrame, ratio_min: float, ratio_max: float) -> pd.DataFrame:
    """word_ratio外れ値を除去"""
    ratio = compute_word_ratio(df)
    mask = (ratio >= ratio_min) & (ratio <= ratio_max)
    n_removed = (~mask).sum()
    print(f"Outlier filtering: {n_removed}/{len(df)} removed (ratio < {ratio_min} or > {ratio_max})")
    return df[mask]


def prepare_data(config: dict) -> tuple:
    """
    データ準備: 外れ値除去 + ラベルマスキング
    手順:
    1. 空行除去
    2. train/val分割（seed固定 → exp005と同じ分割）
    3. train/val両方から外れ値除去
    4. 双方向データ作成 + ラベルマスキング
    """
    from sklearn.model_selection import train_test_split

    train_path = config["data"]["train_path"]
    df = pd.read_csv(train_path)
    print(f"Raw train data: {len(df)} rows")

    # 空行除去
    df = df[
        (df["transliteration"].astype(str).str.len() > 0)
        & (df["translation"].astype(str).str.len() > 0)
    ]
    print(f"After filtering empty: {len(df)} rows")

    # Train/Val分割（exp005と同じseed/ratio）
    seed = config["training"]["seed"]
    val_ratio = config["training"]["val_ratio"]
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed)
    print(f"After split: train={len(train_df)}, val={len(val_df)}")

    # 外れ値除去（split後に両方から除去）
    if config["data"].get("filter_outliers", False):
        ratio_min = config["data"]["word_ratio_min"]
        ratio_max = config["data"]["word_ratio_max"]
        print(f"--- Train filtering ---")
        train_df = filter_outliers(train_df, ratio_min, ratio_max)
        print(f"--- Val filtering ---")
        val_df = filter_outliers(val_df, ratio_min, ratio_max)

    prefix_fwd = config["model"]["prefix"]
    prefix_rev = config["model"]["prefix_reverse"]
    use_bidirectional = config["data"].get("use_bidirectional", False)
    max_length = config["model"]["max_length"]

    # 順方向: 英語ラベルを文末境界でtruncate
    translations_masked = train_df["translation"].astype(str).apply(
        lambda t: truncate_to_sentence_boundary(t, max_length)
    )

    # マスキング統計
    orig_lens = train_df["translation"].astype(str).apply(lambda t: len(t.encode('utf-8')))
    masked_lens = translations_masked.apply(lambda t: len(t.encode('utf-8')))
    n_truncated = (masked_lens < orig_lens).sum()
    print(f"Forward label masking: {n_truncated}/{len(train_df)} labels truncated at sentence boundary")

    fwd = pd.DataFrame({
        "input_text": prefix_fwd + train_df["transliteration"].astype(str),
        "target_text": translations_masked.values,
        "direction": "forward",
    })

    if use_bidirectional:
        bwd = pd.DataFrame({
            "input_text": prefix_rev + train_df["translation"].astype(str),
            "target_text": train_df["transliteration"].astype(str),
            "direction": "reverse",
        })
        train_prepared = pd.concat([fwd, bwd], ignore_index=True)
        train_prepared = train_prepared.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"Train (bidirectional): {len(train_prepared)} rows")
    else:
        train_prepared = fwd.reset_index(drop=True)
        print(f"Train (unidirectional): {len(train_prepared)} rows")

    # Val: 単方向のみ（マスキングなし）
    val_prepared = pd.DataFrame({
        "input_text": prefix_fwd + val_df["transliteration"].astype(str),
        "target_text": val_df["translation"].astype(str),
        "direction": "forward",
    }).reset_index(drop=True)
    print(f"Val: {len(val_prepared)} rows")

    return train_prepared, val_prepared
