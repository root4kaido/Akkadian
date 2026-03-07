"""
後処理モジュール: 翻訳出力のクリーンアップ
deep-pasta-mbrのVectorizedPostprocessorを参考に最小限実装
"""
import re
from typing import List

import pandas as pd


# プリコンパイル済みパターン
PATTERNS = {
    "gap": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
    "big_gap": re.compile(r"(\.{3,}|…|\[\.+\])"),
    "annotations": re.compile(
        r"\((fem|plur|pl|sing|singular|plural|\?|!)\.*\s*\w*\)", re.I
    ),
    "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
    "whitespace": re.compile(r"\s+"),
    "punct_space": re.compile(r"\s+([.,:])"),
    "repeated_punct": re.compile(r"([.,])\1+"),
}

# 文字変換
SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SPECIAL_CHARS_TRANS = str.maketrans("ḫḪ", "hH")
FORBIDDEN_CHARS = '!?()"——<>⌈⌋⌊[]+ʾ/;'
FORBIDDEN_TRANS = str.maketrans("", "", FORBIDDEN_CHARS)


def postprocess_batch(translations: List[str]) -> List[str]:
    """翻訳出力のバッチ後処理"""
    s = pd.Series(translations)

    # 基本クリーニング
    s = s.fillna("").astype(str)
    s = s.str.translate(SPECIAL_CHARS_TRANS)
    s = s.str.translate(SUBSCRIPT_TRANS)
    s = s.str.replace(PATTERNS["whitespace"], " ", regex=True)
    s = s.str.strip()

    # ギャップ正規化
    s = s.str.replace(PATTERNS["gap"], "<gap>", regex=True)
    s = s.str.replace(PATTERNS["big_gap"], "<big_gap>", regex=True)
    s = s.str.replace("<gap> <gap>", "<big_gap>", regex=False)
    s = s.str.replace("<big_gap> <big_gap>", "<big_gap>", regex=False)

    # アノテーション除去
    s = s.str.replace(PATTERNS["annotations"], "", regex=True)

    # ギャップ保護 → 禁止文字除去 → 復元
    s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
    s = s.str.replace("<big_gap>", "\x00BIG\x00", regex=False)
    s = s.str.translate(FORBIDDEN_TRANS)
    s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
    s = s.str.replace("\x00BIG\x00", " <big_gap> ", regex=False)

    # 分数変換
    s = s.str.replace(r"(\d+)\.5\b", r"\1½", regex=True)
    s = s.str.replace(r"\b0\.5\b", "½", regex=True)
    s = s.str.replace(r"(\d+)\.25\b", r"\1¼", regex=True)
    s = s.str.replace(r"\b0\.25\b", "¼", regex=True)
    s = s.str.replace(r"(\d+)\.75\b", r"\1¾", regex=True)
    s = s.str.replace(r"\b0\.75\b", "¾", regex=True)

    # 繰り返し除去
    s = s.str.replace(PATTERNS["repeated_words"], r"\1", regex=True)
    for n in range(4, 1, -1):
        pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
        s = s.str.replace(pattern, r"\1", regex=True)

    # 句読点修正
    s = s.str.replace(PATTERNS["punct_space"], r"\1", regex=True)
    s = s.str.replace(PATTERNS["repeated_punct"], r"\1", regex=True)

    # 最終クリーニング
    s = s.str.replace(PATTERNS["whitespace"], " ", regex=True)
    s = s.str.strip().str.strip("-").str.strip()

    # 空文字列 → "broken text"
    s = s.replace("", "broken text")
    s = s.where(s.str.len() > 0, "broken text")

    return s.tolist()
