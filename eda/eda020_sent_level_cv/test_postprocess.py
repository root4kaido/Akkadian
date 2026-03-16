"""
exp034の既存予測結果に追加後処理を適用して再評価。
notebookの後処理（引用符除去、括弧除去、単語繰り返し除去等）の効果を測定。
"""
import re
import math
import pandas as pd
import sacrebleu


# ============================================================
# 既存の後処理（eval_full_doc.py と同一）
# ============================================================
def repeat_cleanup(text):
    words = text.split()
    if len(words) < 6:
        return text
    for n in range(3, len(words) // 2 + 1):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i+n] == words[i+n:i+2*n]:
                return " ".join(words[:i+n])
    return text


def has_repetition(text, min_repeat=3):
    words = str(text).split()
    for i in range(len(words) - min_repeat):
        chunk = " ".join(words[i:i + min_repeat])
        if chunk in " ".join(words[i + min_repeat:]):
            return True
    return False


# ============================================================
# notebook由来の追加後処理
# ============================================================
_QUOTES_RE = re.compile(r'["""'']')
_SOFT_GRAM_PARENS_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)",
    re.I,
)
_REPEATED_WORDS_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:;])")
_REPEATED_PUNCT_RE = re.compile(r"([.,:;])\1+")
_WS_RE = re.compile(r"\s+")

# 括弧・特殊文字（<gap>のangle bracketは保護）
FORBIDDEN_CHARS = "()""''—–⌈⌋⌊+ʾ"
FORBIDDEN_TRANS = str.maketrans("", "", FORBIDDEN_CHARS)


def extra_postprocess(text):
    """notebookの追加後処理"""
    if not isinstance(text, str) or not text.strip():
        return text

    # 文法タグ括弧の除去: (fem.), (plur.), (sing.), (?), (!) etc.
    text = _SOFT_GRAM_PARENS_RE.sub(" ", text)

    # 引用符除去
    text = _QUOTES_RE.sub("", text)

    # <gap>を保護してから括弧・特殊文字を除去
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.translate(FORBIDDEN_TRANS)
    text = text.replace("\x00GAP\x00", " <gap> ")

    # 単語レベルの繰り返し除去: "the the" → "the"
    text = _REPEATED_WORDS_RE.sub(r"\1", text)

    # フレーズレベルの繰り返し除去 (2-4語)
    for n in range(4, 1, -1):
        pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
        text = re.sub(pattern, r"\1", text)

    # 句読点クリーンアップ
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)

    text = _WS_RE.sub(" ", text).strip()
    return text


# ============================================================
# メトリクス計算
# ============================================================
def calc_metrics(preds, refs, label):
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep_rate = 100 * sum(has_repetition(p) for p in preds) / len(preds)
    print(f"  {label}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep_rate:.1f}%")
    return chrf.score, bleu.score, geo, rep_rate


# ============================================================
# Main
# ============================================================
def main():
    base = "eda/eda020_sent_level_cv"

    for model_tag in ["exp034_st_pretrain_last"]:
        print(f"\n{'='*60}")
        print(f"Model: {model_tag}")
        print(f"{'='*60}")

        for level in ["sent", "doc"]:
            csv_path = f"{base}/{model_tag}_{level}_predictions.csv"
            df = pd.read_csv(csv_path)

            refs = df["reference"].tolist()
            preds_raw = df["prediction_raw"].tolist()

            # 1. 既存の後処理のみ（現状のCV）
            preds_existing = [repeat_cleanup(p) for p in preds_raw]
            print(f"\n--- {level}-CV ---")
            calc_metrics(preds_existing, refs, "existing (repeat_cleanup only)")

            # 2. 既存 + 追加後処理
            preds_extra = [extra_postprocess(repeat_cleanup(p)) for p in preds_raw]
            calc_metrics(preds_extra, refs, "existing + extra_postprocess")

            # 3. 差分サンプル表示
            diffs = []
            for i, (e, x) in enumerate(zip(preds_existing, preds_extra)):
                if e != x:
                    diffs.append((i, e, x))

            print(f"  Changed: {len(diffs)}/{len(preds_existing)} predictions")
            for idx, old, new in diffs[:5]:
                print(f"  [{idx}] OLD: {old[:100]}")
                print(f"       NEW: {new[:100]}")
                print()


if __name__ == "__main__":
    main()
