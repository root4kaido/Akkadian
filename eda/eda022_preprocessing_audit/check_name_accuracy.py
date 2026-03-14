"""val predictions中の{d}付き行で、固有名詞がどれくらい正しく出力されているか"""
import pandas as pd
import re

train = pd.read_csv("/home/user/work/Akkadian/datasets/raw/train.csv")
val_preds = pd.read_csv("/home/user/work/Akkadian/workspace/exp023_full_preprocessing/results/val_predictions.csv")

# trainのvalidation行を特定（val_predictionsのinput列からoriginal transliterationを復元）
prefix = "translate Akkadian to English: "

# ロゴグラム→名前マッピング（trainから抽出した確実なもの）
LOGOGRAM_TO_NAME = {
    "UTU": "Šamaš",
    "EN.LÍL": "Illil",
    "IM": "Adad",
    "IŠKUR": "Adad",
    "MAR.TU": "Amurrum",
    "NIN.ŠUBUR": "Ilabrat",
    "AB": "Illil",  # AB-ba-ni → Illil-bāni
    "EN.ZU": "Suen",
    "IM.GAL": "Adad-rabi",
}

# trainのtransliteration中に{d}を含む行のインデックスを取得
d_rows = train[train["transliteration"].astype(str).str.contains(r"\{d\}", regex=True)]
print(f"train中 {{d}}含む行: {len(d_rows)}")

# val_predictions中で{d}を含む行を探す
d_val = val_preds[val_preds["input"].str.contains(r"\{d\}", regex=True)]
print(f"val predictions中 {{d}}含む行: {len(d_val)}")

# 各行について、{d}ロゴグラムから期待される名前がreferenceとpredictionに含まれるかチェック
d_pattern = re.compile(r"\{d\}(\S+)")

results = []
for _, row in d_val.iterrows():
    inp = row["input"].replace(prefix, "")
    ref = str(row["reference"])
    pred = str(row["prediction_clean"])

    logos = d_pattern.findall(inp)
    for logo_full in logos:
        # ロゴグラム部分を抽出（ハイフン前の部分）
        # {d}UTU-ba-ni → UTU, {d}EN.LÍL → EN.LÍL
        # まず完全一致を試し、なければベース部分で
        base = logo_full.split("-")[0] if "-" in logo_full else logo_full

        expected = LOGOGRAM_TO_NAME.get(logo_full) or LOGOGRAM_TO_NAME.get(base)
        if not expected:
            continue

        in_ref = expected.lower() in ref.lower()
        in_pred = expected.lower() in pred.lower()

        results.append({
            "logogram": f"{{d}}{logo_full}",
            "expected_name": expected,
            "in_ref": in_ref,
            "in_pred": in_pred,
            "correct": in_ref == in_pred or (not in_ref),  # refにないならスキップ
            "ref_snippet": ref[:80],
            "pred_snippet": pred[:80],
        })

df = pd.DataFrame(results)
print(f"\n解析対象: {len(df)}件（{{d}}ロゴグラムでマッピング可能なもの）")

# refに含まれるもの（本当にその名前が正解に含まれるケース）のみで精度を見る
df_with_ref = df[df["in_ref"]]
print(f"referenceに期待名あり: {len(df_with_ref)}件")

if len(df_with_ref) > 0:
    correct = df_with_ref["in_pred"].sum()
    total = len(df_with_ref)
    print(f"prediction正解: {correct}/{total} ({100*correct/total:.1f}%)")

    # 間違えたケースを表示
    wrong = df_with_ref[~df_with_ref["in_pred"]]
    print(f"\n=== 間違えたケース ({len(wrong)}件) ===")
    for _, r in wrong.iterrows():
        print(f"  {r['logogram']} → 期待: {r['expected_name']}")
        print(f"    ref:  {r['ref_snippet']}")
        print(f"    pred: {r['pred_snippet']}")
        print()

# referenceにない（他の名前の一部として出現）
df_no_ref = df[~df["in_ref"]]
print(f"\nreferenceに期待名なし: {len(df_no_ref)}件（名前が別の形で出現している可能性）")
for _, r in df_no_ref.head(5).iterrows():
    print(f"  {r['logogram']} → 期待: {r['expected_name']}")
    print(f"    ref:  {r['ref_snippet']}")
    print(f"    pred: {r['pred_snippet']}")
    print()
