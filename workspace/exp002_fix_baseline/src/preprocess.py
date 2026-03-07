"""
前処理モジュール (exp002): Starter準拠の最小限前処理
exp001の知見: normalize_text()による正規化は情報損失を招く → 生テキストで学習
"""
import pandas as pd


def prepare_data(config: dict) -> tuple:
    """
    データ準備: 正規化なし、Starter準拠
    Returns: (train_df, val_df) — 各dfは input_text, target_text カラム
    """
    from sklearn.model_selection import train_test_split

    # データ読み込み
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
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # prefix付与（正規化なし、生テキストそのまま）
    prefix = config["model"]["prefix"]

    train_prepared = pd.DataFrame({
        "input_text": prefix + train_df["transliteration"].astype(str),
        "target_text": train_df["translation"].astype(str),
    }).reset_index(drop=True)

    val_prepared = pd.DataFrame({
        "input_text": prefix + val_df["transliteration"].astype(str),
        "target_text": val_df["translation"].astype(str),
    }).reset_index(drop=True)

    return train_prepared, val_prepared
