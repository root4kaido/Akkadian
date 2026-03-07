"""
後処理モジュール (exp006): 最適化版
eda005のablationテスト結果に基づき、有効なコンポーネントのみ残す:
- repeated_removal: sent-levelで+0.70pt → 採用
- forbidden_chars: 両方で有害(-0.26/-0.23) → 除外
- fraction_conversion: 両方で有害(-0.15/-0.27) → 除外
"""
import re
from typing import List

import pandas as pd


PATTERNS = {
    "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
    "whitespace": re.compile(r"\s+"),
}


def postprocess_batch(translations: List[str]) -> List[str]:
    """翻訳出力の後処理（最適化版）"""
    s = pd.Series(translations)
    s = s.fillna("").astype(str)

    # whitespace正規化
    s = s.str.replace(PATTERNS["whitespace"], " ", regex=True)
    s = s.str.strip()

    # 繰り返し除去（唯一有効な後処理コンポーネント）
    s = s.str.replace(PATTERNS["repeated_words"], r"\1", regex=True)
    for n in range(4, 1, -1):
        pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
        s = s.str.replace(pattern, r"\1", regex=True)

    # 最終クリーニング
    s = s.str.replace(PATTERNS["whitespace"], " ", regex=True)
    s = s.str.strip().str.strip("-").str.strip()

    # 空文字列 → "broken text"
    s = s.replace("", "broken text")
    s = s.where(s.str.len() > 0, "broken text")

    return s.tolist()
