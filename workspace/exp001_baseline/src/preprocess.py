"""
前処理モジュール: 正規化・文アライメント・双方向データ作成
"""
import re
import unicodedata
import pandas as pd


# === 文字変換テーブル ===
SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SPECIAL_CHARS_TRANS = str.maketrans("ḫḪ", "hH")


def normalize_text(text: str) -> str:
    """翻字テキストの正規化"""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Unicode NFC正規化
    text = unicodedata.normalize("NFC", text)
    # Ḫ/ḫ → H/h
    text = text.translate(SPECIAL_CHARS_TRANS)
    # 下付き数字 → 通常数字
    text = text.translate(SUBSCRIPT_TRANS)
    # 連続空白 → 単一空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_sentence_aligner(df: pd.DataFrame) -> pd.DataFrame:
    """
    文アライメント: 英語文数 == アッカド語行数のとき1:1分割
    一致しない場合は元のペアをそのまま使用
    """
    aligned = []
    for _, row in df.iterrows():
        src = str(row["transliteration"])
        tgt = str(row["translation"])

        # 英語を文末記号で分割
        tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
        # アッカド語を改行で分割
        src_lines = [s.strip() for s in src.split("\n") if s.strip()]

        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned.append({"transliteration": s, "translation": t})
        else:
            aligned.append({"transliteration": src, "translation": tgt})

    return pd.DataFrame(aligned)


def create_bidirectional_data(
    df: pd.DataFrame,
    prefix_fwd: str = "translate Akkadian to English: ",
    prefix_bwd: str = "translate English to Akkadian: ",
    seed: int = 42,
) -> pd.DataFrame:
    """双方向データ作成: Akk→Eng + Eng→Akk"""
    # 順方向
    fwd = pd.DataFrame({
        "input_text": prefix_fwd + df["transliteration"].astype(str),
        "target_text": df["translation"].astype(str),
    })
    # 逆方向
    bwd = pd.DataFrame({
        "input_text": prefix_bwd + df["translation"].astype(str),
        "target_text": df["transliteration"].astype(str),
    })
    combined = pd.concat([fwd, bwd], ignore_index=True)
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def create_unidirectional_data(
    df: pd.DataFrame,
    prefix: str = "translate Akkadian to English: ",
) -> pd.DataFrame:
    """単方向データ作成（validation用）: Akk→Engのみ"""
    return pd.DataFrame({
        "input_text": prefix + df["transliteration"].astype(str),
        "target_text": df["translation"].astype(str),
    })


def prepare_data(config: dict) -> tuple:
    """
    データ準備のメインエントリポイント
    Returns: (train_df, val_df) — 各dfは input_text, target_text カラム
    """
    import yaml
    from sklearn.model_selection import train_test_split

    # データ読み込み
    train_path = config["data"]["train_path"]
    df = pd.read_csv(train_path)
    print(f"Raw train data: {len(df)} rows")

    # 正規化
    df["transliteration"] = df["transliteration"].apply(normalize_text)
    df["translation"] = df["translation"].apply(normalize_text)

    # 空行除去
    df = df[(df["transliteration"].str.len() > 0) & (df["translation"].str.len() > 0)]
    print(f"After filtering empty: {len(df)} rows")

    # 文アライメント
    if config["data"].get("use_sentence_alignment", True):
        df = simple_sentence_aligner(df)
        print(f"After sentence alignment: {len(df)} rows")

    # Train/Val分割
    seed = config["training"]["seed"]
    val_ratio = config["training"]["val_ratio"]
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # 双方向データ作成（trainのみ）
    prefix_fwd = config["model"]["prefix"]
    prefix_bwd = config["model"]["prefix_reverse"]

    if config["data"].get("use_bidirectional", True):
        train_prepared = create_bidirectional_data(train_df, prefix_fwd, prefix_bwd, seed)
        print(f"Train with bidirectional: {len(train_prepared)} rows")
    else:
        train_prepared = create_unidirectional_data(train_df, prefix_fwd)

    val_prepared = create_unidirectional_data(val_df, prefix_fwd)

    return train_prepared, val_prepared
