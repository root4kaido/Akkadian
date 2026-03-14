"""
langdetectで追加データの言語判定
"""
import pandas as pd
from collections import Counter
from pathlib import Path
from langdetect import detect, DetectorFactory

# 再現性のためseed固定
DetectorFactory.seed = 0

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

additional_df = pd.read_csv(str(PROJECT_ROOT / "workspace" / "exp017_additional_data" / "dataset" / "additional_train.csv"))
train_df = pd.read_csv(str(PROJECT_ROOT / "datasets" / "raw" / "train.csv"))

def detect_lang(text):
    try:
        return detect(str(text))
    except:
        return "unknown"

print("=== Additional data (1165 rows) ===")
additional_df['lang'] = additional_df['translation'].apply(detect_lang)
lang_counts = additional_df['lang'].value_counts()
print(lang_counts)
print(f"\n非英語: {(additional_df['lang'] != 'en').sum()} rows ({(additional_df['lang'] != 'en').sum()/len(additional_df)*100:.1f}%)")

print("\n--- 言語別サンプル ---")
for lang in lang_counts.index:
    if lang == 'en':
        continue
    samples = additional_df[additional_df['lang'] == lang].head(2)
    print(f"\n[{lang}] ({lang_counts[lang]}件):")
    for _, row in samples.iterrows():
        print(f"  ENG: {str(row['translation'])[:120]}")

print("\n\n=== Train data (1561 rows) ===")
train_df['lang'] = train_df['translation'].apply(detect_lang)
train_lang_counts = train_df['lang'].value_counts()
print(train_lang_counts)
print(f"\n非英語: {(train_df['lang'] != 'en').sum()} rows ({(train_df['lang'] != 'en').sum()/len(train_df)*100:.1f}%)")

# キーワード手法との比較
print("\n\n=== キーワード手法との比較 ===")
german_words = {'und', 'der', 'die', 'das', 'des', 'dem', 'den', 'ich', 'du', 'er',
                'sie', 'wir', 'ihr', 'ein', 'eine', 'einem', 'einen', 'einer',
                'nicht', 'ist', 'hat', 'mit', 'von', 'zu', 'für', 'auf', 'aus',
                'nach', 'über', 'aber', 'oder', 'wenn', 'wie', 'auch', 'noch',
                'silber', 'minen', 'schekel'}

def german_ratio(text):
    words = str(text).lower().split()
    if len(words) == 0:
        return 0
    return sum(1 for w in words if w in german_words) / len(words)

additional_df['german_ratio'] = additional_df['translation'].apply(german_ratio)
keyword_german = additional_df['german_ratio'] > 0.05
langdetect_nonenglish = additional_df['lang'] != 'en'

print(f"キーワード手法（ドイツ語>5%）: {keyword_german.sum()} rows")
print(f"langdetect（非英語）: {langdetect_nonenglish.sum()} rows")
print(f"両方に該当: {(keyword_german & langdetect_nonenglish).sum()} rows")
print(f"キーワードのみ: {(keyword_german & ~langdetect_nonenglish).sum()} rows")
print(f"langdetectのみ: {(~keyword_german & langdetect_nonenglish).sum()} rows")

# langdetectのみで検出されたもののサンプル
langdetect_only = additional_df[~keyword_german & langdetect_nonenglish]
if len(langdetect_only) > 0:
    print(f"\n--- langdetectのみで検出 ({len(langdetect_only)}件) サンプル ---")
    for _, row in langdetect_only.head(5).iterrows():
        print(f"  [{row['lang']}] {str(row['translation'])[:120]}")
