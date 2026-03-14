"""
EDA022: Preprocessing Audit - Missing Items Check
report.mdに記載されていない前処理関連項目を検証する
"""

import pandas as pd
import re
from collections import Counter

TRAIN_PATH = "/home/user/work/Akkadian/datasets/raw/train.csv"
TEST_PATH = "/home/user/work/Akkadian/datasets/raw/test.csv"

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_examples(examples, max_show=5):
    for i, (idx, text) in enumerate(examples[:max_show]):
        print(f"  [{idx}] ...{text}...")
    if len(examples) > max_show:
        print(f"  ... and {len(examples) - max_show} more")

def check_gap_issues(train, test):
    """1. Gap統合の検証: <big_gap>残存、<gap> <gap>重複"""
    print_header("1. Gap統合の検証")

    for name, df in [("train", train), ("test", test)]:
        cols = ["transliteration", "translation"] if "translation" in df.columns else ["transliteration"]
        for col in cols:
            series = df[col].dropna().astype(str)

            # <big_gap> 残存
            big_gap = [(i, s) for i, s in zip(series.index, series) if "<big_gap>" in s]
            print(f"\n  [{name}/{col}] <big_gap> 残存: {len(big_gap)} 件")
            if big_gap:
                print_examples([(i, s[max(0, s.find("<big_gap>")-20):s.find("<big_gap>")+30]) for i, s in big_gap])

            # <gap> <gap> 重複 (連続する<gap>)
            dup_gap = [(i, s) for i, s in zip(series.index, series)
                       if re.search(r"<gap>\s*<gap>", s)]
            print(f"  [{name}/{col}] <gap> <gap> 重複: {len(dup_gap)} 件")
            if dup_gap:
                print_examples([(i, s[max(0, s.find("<gap>")-10):s.find("<gap>")+40]) for i, s in dup_gap])

def check_tug_parens(train, test):
    """2. (TÚG)→TÚG: (TÚG)がまだ残っていないか"""
    print_header("2. (TÚG) 残存チェック")

    # ついでに他の括弧付きパターンも確認
    patterns = [r"\(TÚG\)", r"\(d\)", r"\(ki\)", r"\(m\)", r"\(f\)"]
    for name, df in [("train", train), ("test", test)]:
        cols = ["transliteration", "translation"] if "translation" in df.columns else ["transliteration"]
        for col in cols:
            series = df[col].dropna().astype(str)
            print(f"\n  [{name}/{col}]")
            for pat in patterns:
                matches = [(i, s) for i, s in zip(series.index, series) if re.search(pat, s)]
                print(f"    {pat}: {len(matches)} 件")
                if matches and pat == r"\(TÚG\)":
                    print_examples([(i, s[max(0, m.start()-15):m.end()+15])
                                    for i, s in matches
                                    for m in [re.search(pat, s)]], max_show=3)

def check_long_floats(train, test):
    """3. Long float shortening: 小数点以下5桁以上の数値"""
    print_header("3. Long float (小数点以下5桁以上) チェック")

    # 小数点以下5桁以上のパターン
    float_pat = re.compile(r"\d+\.\d{5,}")

    for name, df in [("train", train), ("test", test)]:
        cols = ["transliteration", "translation"] if "translation" in df.columns else ["transliteration"]
        for col in cols:
            series = df[col].dropna().astype(str)
            results = []
            for i, s in zip(series.index, series):
                found = float_pat.findall(s)
                if found:
                    results.append((i, found))
            print(f"\n  [{name}/{col}] 長い小数: {len(results)} 件")
            if results:
                for i, floats in results[:5]:
                    print(f"    [{i}] {floats}")
                if len(results) > 5:
                    print(f"    ... and {len(results) - 5} more")

            # 全ての小数値も集計
            all_floats = []
            for i, s in zip(series.index, series):
                all_floats.extend(float_pat.findall(s))
            if all_floats:
                print(f"    ユニーク値: {Counter(all_floats).most_common(10)}")

def check_stray_angle_brackets(train, test):
    """4. < > (stray angle brackets, << >>とは別)"""
    print_header("4. Stray angle brackets < > チェック")

    # < > を検出するが、<gap>, <big_gap>, << >> は除外
    # まず単独の < または > を探す
    # パターン: < の後に既知タグ名以外が続く、または単独の < >
    stray_pat = re.compile(r"<\s+>")  # < > (空白入り)
    lone_lt = re.compile(r"(?<!<)<(?!<)(?!gap>|big_gap>|/)")  # 単独 < (<<除外, <gap>除外)
    lone_gt = re.compile(r"(?<!>)>(?!>)(?!})")  # 単独 > (>>除外)

    for name, df in [("train", train), ("test", test)]:
        cols = ["transliteration", "translation"] if "translation" in df.columns else ["transliteration"]
        for col in cols:
            series = df[col].dropna().astype(str)

            # < > パターン
            stray = [(i, s) for i, s in zip(series.index, series) if stray_pat.search(s)]
            print(f"\n  [{name}/{col}] '< >' (空白入り): {len(stray)} 件")
            if stray:
                print_examples([(i, s[max(0, stray_pat.search(s).start()-15):stray_pat.search(s).end()+15])
                                for i, s in stray], max_show=5)

            # 単独 < > を含む行（<gap>等を除外した上で）
            # より厳密: <gap>等を除去した後に < or > が残るか
            def has_stray_brackets(text):
                cleaned = re.sub(r"<big_gap>|<gap>|<<|>>", "", text)
                return "<" in cleaned or ">" in cleaned

            stray_any = [(i, s) for i, s in zip(series.index, series) if has_stray_brackets(s)]
            print(f"  [{name}/{col}] 既知タグ除外後に<>残存: {len(stray_any)} 件")
            if stray_any:
                # 残っている< >の文脈を表示
                for i, s in stray_any[:5]:
                    cleaned = re.sub(r"<big_gap>|<gap>|<<|>>", "", s)
                    brackets = [(m.start(), m.group()) for m in re.finditer(r"[<>]", cleaned)]
                    context_snippets = []
                    for pos, ch in brackets[:3]:
                        start = max(0, pos - 10)
                        end = min(len(cleaned), pos + 10)
                        context_snippets.append(cleaned[start:end])
                    print(f"    [{i}] {context_snippets}")
                if len(stray_any) > 5:
                    print(f"    ... and {len(stray_any) - 5} more")

def check_quotation_marks(train, test):
    """5. Quotation marks / apostrophes の存在確認"""
    print_header("5. Quotation marks / apostrophes 確認")

    # 各種引用符・アポストロフィ
    patterns = {
        'Double quote "': r'"',
        "Single quote '": r"'",
        "Left double quote \u201c": r"\u201c",
        "Right double quote \u201d": r"\u201d",
        "Left single quote \u2018": r"\u2018",
        "Right single quote \u2019": r"\u2019",
        "Backtick `": r"`",
    }

    for name, df in [("train", train), ("test", test)]:
        cols = ["transliteration", "translation"] if "translation" in df.columns else ["transliteration"]
        for col in cols:
            series = df[col].dropna().astype(str)
            print(f"\n  [{name}/{col}]")
            for desc, pat in patterns.items():
                matches = [(i, s) for i, s in zip(series.index, series) if re.search(pat, s)]
                count = len(matches)
                if count > 0:
                    print(f"    {desc}: {count} 件")
                    # 最初の例を表示
                    ex_s = matches[0][1]
                    m = re.search(pat, ex_s)
                    start = max(0, m.start() - 20)
                    end = min(len(ex_s), m.end() + 20)
                    print(f"      例: ...{ex_s[start:end]}...")
                else:
                    print(f"    {desc}: 0 件")

def check_exclamation_marks(train, test):
    """6. Exclamation marks ! の存在確認"""
    print_header("6. Exclamation marks '!' 確認")

    for name, df in [("train", train), ("test", test)]:
        cols = ["transliteration", "translation"] if "translation" in df.columns else ["transliteration"]
        for col in cols:
            series = df[col].dropna().astype(str)
            matches = [(i, s) for i, s in zip(series.index, series) if "!" in s]
            print(f"\n  [{name}/{col}] '!' 含む行: {len(matches)} 件")
            if matches:
                # 文脈を表示
                for i, s in matches[:5]:
                    pos = s.index("!")
                    start = max(0, pos - 25)
                    end = min(len(s), pos + 25)
                    print(f"    [{i}] ...{s[start:end]}...")
                if len(matches) > 5:
                    print(f"    ... and {len(matches) - 5} more")

                # ! の出現位置パターン分析
                positions = []
                for i, s in matches:
                    for m in re.finditer(r"!", s):
                        # 前後の文字を取得
                        ctx = s[max(0, m.start()-3):m.end()+3]
                        positions.append(ctx)
                print(f"    出現パターン top10: {Counter(positions).most_common(10)}")

def main():
    print("=" * 70)
    print("  EDA022: Preprocessing Audit - Missing Items Check")
    print("=" * 70)

    train, test = load_data()
    print(f"\nTrain: {len(train)} rows, Test: {len(test)} rows")
    print(f"Train columns: {list(train.columns)}")
    print(f"Test columns: {list(test.columns)}")

    check_gap_issues(train, test)
    check_tug_parens(train, test)
    check_long_floats(train, test)
    check_stray_angle_brackets(train, test)
    check_quotation_marks(train, test)
    check_exclamation_marks(train, test)

    print(f"\n{'='*70}")
    print("  完了")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
