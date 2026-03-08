"""
前処理モジュール (exp008): exp005と同一ロジック
- 正規化なし（生テキスト）
- 双方向データ作成（Akk→Eng + Eng→Akk）
- 順方向: 英語ラベルを512バイトで切り、最後の文末(ピリオド)まで残す
- 逆方向: 英語入力を1024バイトまで許容
- Validationは単方向のみ（マスキングなし）
"""
import re
import pandas as pd


def truncate_to_sentence_boundary(text: str, max_bytes: int) -> str:
    """テキストをmax_bytesに収まるよう文末（ピリオド）境界でtruncateする。

    - max_bytes以内なら全文返す
    - 超える場合、max_bytes以内の最後のピリオド+スペースまでを返す
    - ピリオドが見つからない場合はmax_bytesでそのままカット
    """
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


def prepare_data(config: dict) -> tuple:
    """
    データ準備: ラベルマスキング対応
    - Train順方向: 英語ラベルを文末境界でtruncate
    - Train逆方向: 英語入力のmax_lengthを1024に拡大
    - Val: 単方向（Akk→Engのみ、マスキングなし）
    Returns: (train_df, val_df)
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

    # Train/Val分割
    seed = config["training"]["seed"]
    val_ratio = config["training"]["val_ratio"]
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed)

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

    # Val: 単方向のみ（マスキングなし - 完全な翻訳で評価）
    val_prepared = pd.DataFrame({
        "input_text": prefix_fwd + val_df["transliteration"].astype(str),
        "target_text": val_df["translation"].astype(str),
        "direction": "forward",
    }).reset_index(drop=True)
    print(f"Val: {len(val_prepared)} rows")

    return train_prepared, val_prepared
