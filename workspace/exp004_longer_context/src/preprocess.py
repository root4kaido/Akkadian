"""
前処理モジュール (exp004): exp003と同一
- 正規化なし（生テキスト）
- 双方向データ作成（Akk→Eng + Eng→Akk）
- Validationは単方向のみ
"""
import pandas as pd


def prepare_data(config: dict) -> tuple:
    """
    データ準備: Starterと同じロジック
    - Train: 双方向（データ2倍）、シャッフル
    - Val: 単方向（Akk→Engのみ、評価用）
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

    # Train: 双方向データ作成
    fwd = pd.DataFrame({
        "input_text": prefix_fwd + train_df["transliteration"].astype(str),
        "target_text": train_df["translation"].astype(str),
    })

    if use_bidirectional:
        bwd = pd.DataFrame({
            "input_text": prefix_rev + train_df["translation"].astype(str),
            "target_text": train_df["transliteration"].astype(str),
        })
        train_prepared = pd.concat([fwd, bwd], ignore_index=True)
        train_prepared = train_prepared.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"Train (bidirectional): {len(train_prepared)} rows")
    else:
        train_prepared = fwd.reset_index(drop=True)
        print(f"Train (unidirectional): {len(train_prepared)} rows")

    # Val: 単方向のみ
    val_prepared = pd.DataFrame({
        "input_text": prefix_fwd + val_df["transliteration"].astype(str),
        "target_text": val_df["translation"].astype(str),
    }).reset_index(drop=True)
    print(f"Val: {len(val_prepared)} rows")

    return train_prepared, val_prepared
