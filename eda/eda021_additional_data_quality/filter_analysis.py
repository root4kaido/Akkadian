"""
additional_train.csvから完全な翻訳のみを抽出するフィルタリング分析
"""
import pandas as pd
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

additional_df = pd.read_csv(str(PROJECT_ROOT / "workspace" / "exp017_additional_data" / "dataset" / "additional_train.csv"))
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))

print(f"元のadditional: {len(additional_df)} rows")
print()

# ============================================================
# フィルタ1: 省略記号 "..." を含む翻訳を除外
# ============================================================
mask_ellipsis = additional_df['translation'].astype(str).str.contains(r'\.\.\.', regex=True)
print(f"フィルタ1 - 省略記号'...': {mask_ellipsis.sum()} rows 除外")

# ============================================================
# フィルタ2: ドイツ語検出（簡易: ドイツ語頻出語が多いものを除外）
# ============================================================
german_words = {'und', 'der', 'die', 'das', 'des', 'dem', 'den', 'ich', 'du', 'er',
                'sie', 'wir', 'ihr', 'ein', 'eine', 'einem', 'einen', 'einer',
                'nicht', 'ist', 'hat', 'mit', 'von', 'zu', 'für', 'auf', 'aus',
                'nach', 'über', 'aber', 'oder', 'wenn', 'wie', 'auch', 'noch',
                'silber', 'minen', 'schekel'}

def german_ratio(text):
    words = str(text).lower().split()
    if len(words) == 0:
        return 0
    german_count = sum(1 for w in words if w in german_words)
    return german_count / len(words)

additional_df['german_ratio'] = additional_df['translation'].apply(german_ratio)
mask_german = additional_df['german_ratio'] > 0.05  # 5%以上ドイツ語
print(f"フィルタ2 - ドイツ語(>5%): {mask_german.sum()} rows 除外")

# ドイツ語率の分布
print(f"  german_ratio: 0%={((additional_df['german_ratio']==0).sum())}, "
      f"0-5%={((additional_df['german_ratio']>0) & (additional_df['german_ratio']<=0.05)).sum()}, "
      f"5-10%={((additional_df['german_ratio']>0.05) & (additional_df['german_ratio']<=0.10)).sum()}, "
      f"10%+={((additional_df['german_ratio']>0.10).sum())}")

# サンプル: ドイツ語率高いもの
high_german = additional_df[additional_df['german_ratio'] > 0.10].head(3)
for _, row in high_german.iterrows():
    print(f"  [{row['german_ratio']:.2f}] {str(row['translation'])[:120]}")

# ============================================================
# フィルタ3: 極端な長さ比
# ============================================================
akk_wlen = additional_df['transliteration'].astype(str).str.split().str.len()
eng_wlen = additional_df['translation'].astype(str).str.split().str.len()
ratio = eng_wlen / akk_wlen.clip(lower=1)
mask_ratio = (ratio < 0.3) | (ratio > 5.0)
print(f"フィルタ3 - 極端な長さ比(<0.3 or >5.0): {mask_ratio.sum()} rows 除外")

# ============================================================
# フィルタ4: 短すぎる翻訳（英訳5語未満）
# ============================================================
mask_short = eng_wlen < 5
print(f"フィルタ4 - 短い英訳(<5 words): {mask_short.sum()} rows 除外")

# ============================================================
# 組み合わせ
# ============================================================
mask_any = mask_ellipsis | mask_german | mask_ratio | mask_short
print(f"\n--- いずれかに該当: {mask_any.sum()} rows 除外 ---")
print(f"残り: {(~mask_any).sum()} rows ({(~mask_any).sum()/len(additional_df)*100:.1f}%)")

# フィルタ間の重複を見る
print(f"\n重複分析:")
print(f"  ellipsis AND german: {(mask_ellipsis & mask_german).sum()}")
print(f"  ellipsis AND ratio: {(mask_ellipsis & mask_ratio).sum()}")
print(f"  german AND ratio: {(mask_german & mask_ratio).sum()}")
print(f"  ellipsis only: {(mask_ellipsis & ~mask_german & ~mask_ratio & ~mask_short).sum()}")
print(f"  german only: {(mask_german & ~mask_ellipsis & ~mask_ratio & ~mask_short).sum()}")
print(f"  ratio only: {(mask_ratio & ~mask_ellipsis & ~mask_german & ~mask_short).sum()}")
print(f"  short only: {(mask_short & ~mask_ellipsis & ~mask_german & ~mask_ratio).sum()}")

# ============================================================
# フィルタ後のデータの品質確認
# ============================================================
clean_df = additional_df[~mask_any].copy()
print(f"\n=== フィルタ後のデータ品質 ===")
print(f"件数: {len(clean_df)}")

clean_akk_wlen = clean_df['transliteration'].astype(str).str.split().str.len()
clean_eng_wlen = clean_df['translation'].astype(str).str.split().str.len()
clean_ratio = clean_eng_wlen / clean_akk_wlen.clip(lower=1)

print(f"Akk word len: mean={clean_akk_wlen.mean():.1f}, median={clean_akk_wlen.median():.1f}")
print(f"Eng word len: mean={clean_eng_wlen.mean():.1f}, median={clean_eng_wlen.median():.1f}")
print(f"Eng/Akk ratio: mean={clean_ratio.mean():.2f}, median={clean_ratio.median():.2f}")

# Train比較
train_ratio = train_df['translation'].astype(str).str.split().str.len() / train_df['transliteration'].astype(str).str.split().str.len().clip(lower=1)
print(f"\n(参考) Train Eng/Akk ratio: mean={train_ratio.mean():.2f}, median={train_ratio.median():.2f}")

# サンプル表示
print(f"\n=== フィルタ後サンプル5件 ===")
for _, row in clean_df.sample(5, random_state=42).iterrows():
    akk = str(row['transliteration'])[:100]
    eng = str(row['translation'])[:100]
    print(f"  AKK: {akk}")
    print(f"  ENG: {eng}")
    print()

# 除外サンプル表示
print(f"=== 除外サンプル5件 ===")
removed_df = additional_df[mask_any]
for _, row in removed_df.sample(min(5, len(removed_df)), random_state=42).iterrows():
    akk = str(row['transliteration'])[:80]
    eng = str(row['translation'])[:120]
    reasons = []
    idx = row.name
    if mask_ellipsis.loc[idx]: reasons.append("ellipsis")
    if mask_german.loc[idx]: reasons.append(f"german({row['german_ratio']:.2f})")
    if mask_ratio.loc[idx]: reasons.append(f"ratio({ratio.loc[idx]:.2f})")
    if mask_short.loc[idx]: reasons.append(f"short({eng_wlen.loc[idx]}w)")
    print(f"  [{', '.join(reasons)}]")
    print(f"  AKK: {akk}")
    print(f"  ENG: {eng}")
    print()

# フィルタ後データを保存
output_path = PROJECT_ROOT / "datasets" / "processed" / "additional_train_filtered.csv"
clean_df[['transliteration', 'translation']].to_csv(str(output_path), index=False)
print(f"フィルタ後データ保存: {output_path} ({len(clean_df)} rows)")
