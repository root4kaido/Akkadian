"""
eda018: Round-trip翻訳によるAkkadian文レベルアライメントの可能性調査

目的:
- exp016(ByT5-base)の逆翻訳(Eng→Akk)品質を確認
- 英語の文分割→逆翻訳で、Akkadian側のsentence boundaryを推定できるか検証
- 実用的なアライメント手法として使えるかを判断

手順:
1. train.csvから複数文を含むドキュメントを選択
2. 英語翻訳を文分割
3. 各英語文をEng→Akkに逆翻訳
4. 逆翻訳Akkと元Akkの対応関係を分析
"""
import os
import re
import sys
import json
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

# ============================================================
# Setup
# ============================================================
EDA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EDA_DIR.parent.parent
FIGURES_DIR = EDA_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MODEL_PATH = str(PROJECT_ROOT / "workspace" / "exp016_byt5_base" / "results" / "best_model")
TRAIN_PATH = str(PROJECT_ROOT / "datasets" / "raw" / "train.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(EDA_DIR / "roundtrip_analysis.log")),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# ============================================================
# Load model
# ============================================================
logger.info(f"Loading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
logger.info("Model loaded")

# ============================================================
# Load and filter data: 複数文を含むドキュメント
# ============================================================
df = pd.read_csv(TRAIN_PATH)
logger.info(f"Total train documents: {len(df)}")

def count_sentences(text):
    """英語テキストの文数を推定"""
    text = str(text).strip()
    # ピリオド、!、?の後にスペースまたは文末
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return len(sents)

def split_sentences(text):
    """英語テキストを文分割"""
    text = str(text).strip()
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sents

df['n_eng_sents'] = df['translation'].apply(count_sentences)
df['akk_bytes'] = df['transliteration'].astype(str).apply(lambda t: len(t.encode('utf-8')))
df['eng_bytes'] = df['translation'].astype(str).apply(lambda t: len(t.encode('utf-8')))

logger.info(f"\n文数分布:")
logger.info(f"{df['n_eng_sents'].value_counts().sort_index().to_string()}")

# 2文以上のドキュメントを対象に
multi_sent = df[df['n_eng_sents'] >= 2].copy()
logger.info(f"\n複数文ドキュメント: {len(multi_sent)} / {len(df)} ({100*len(multi_sent)/len(df):.1f}%)")

# ============================================================
# 逆翻訳: Eng → Akk
# ============================================================
def translate_batch(texts, prefix, batch_size=8, max_length=512, num_beams=4):
    """バッチ翻訳"""
    prefixed = [prefix + t for t in texts]
    results = []
    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i:i+batch_size]
        inputs = tokenizer(
            batch, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=max_length,
                num_beams=num_beams, early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend([d.strip() for d in decoded])
    return results

# サンプリング: 計算量の制約で50件に限定
np.random.seed(42)
sample_indices = np.random.choice(multi_sent.index, size=min(50, len(multi_sent)), replace=False)
sample_df = multi_sent.loc[sample_indices].copy()
logger.info(f"\n分析対象: {len(sample_df)}件")

# まず全ドキュメントのAkk→Eng (forward) も確認
PREFIX_FWD = "translate Akkadian to English: "
PREFIX_REV = "translate English to Akkadian: "

# Forward翻訳 (doc-level)
logger.info("\n=== Forward翻訳 (Akk→Eng) ===")
fwd_preds = translate_batch(
    sample_df['transliteration'].astype(str).tolist(),
    PREFIX_FWD, batch_size=4,
)
sample_df['fwd_pred'] = fwd_preds

# 各英語文を逆翻訳
logger.info("\n=== 逆翻訳 (Eng→Akk) ===")
all_results = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Round-trip"):
    akk_orig = str(row['transliteration'])
    eng_orig = str(row['translation'])
    eng_sents = split_sentences(eng_orig)

    # 各英語文を逆翻訳
    akk_preds = translate_batch(eng_sents, PREFIX_REV, batch_size=len(eng_sents))

    # 逆翻訳結果を連結
    akk_roundtrip = " ".join(akk_preds)

    # chrF++で元Akkとの類似度を計算
    chrf_whole = sacrebleu.sentence_chrf(akk_roundtrip, [akk_orig], word_order=2).score

    # 各文ごとの逆翻訳を個別に記録
    sent_details = []
    for j, (eng_s, akk_p) in enumerate(zip(eng_sents, akk_preds)):
        sent_details.append({
            'sent_idx': j,
            'eng_sent': eng_s,
            'akk_pred': akk_p,
            'eng_bytes': len(eng_s.encode('utf-8')),
            'akk_pred_bytes': len(akk_p.encode('utf-8')),
        })

    all_results.append({
        'oare_id': row['oare_id'],
        'akk_orig': akk_orig,
        'eng_orig': eng_orig,
        'fwd_pred': row['fwd_pred'],
        'n_sents': len(eng_sents),
        'akk_roundtrip': akk_roundtrip,
        'chrf_roundtrip': chrf_whole,
        'akk_orig_bytes': len(akk_orig.encode('utf-8')),
        'akk_roundtrip_bytes': len(akk_roundtrip.encode('utf-8')),
        'sent_details': sent_details,
    })

# ============================================================
# 分析結果
# ============================================================
results_df = pd.DataFrame(all_results)

logger.info("\n" + "=" * 60)
logger.info("=== Round-trip翻訳品質 ===")
logger.info(f"chrF++ (逆翻訳Akk vs 元Akk): mean={results_df['chrf_roundtrip'].mean():.2f}, median={results_df['chrf_roundtrip'].median():.2f}")
logger.info(f"バイト比 (roundtrip/orig): mean={( results_df['akk_roundtrip_bytes'] / results_df['akk_orig_bytes']).mean():.2f}")

logger.info(f"\nchrF++分布:")
for threshold in [70, 60, 50, 40, 30, 20, 10]:
    n = (results_df['chrf_roundtrip'] >= threshold).sum()
    logger.info(f"  chrF++ >= {threshold}: {n}/{len(results_df)} ({100*n/len(results_df):.0f}%)")

# ============================================================
# 具体例の表示
# ============================================================
results_sorted = results_df.sort_values('chrf_roundtrip', ascending=False)

logger.info("\n" + "=" * 60)
logger.info("=== 上位5件 (round-trip品質が高い) ===")
for _, row in results_sorted.head(5).iterrows():
    logger.info(f"\nchrF++={row['chrf_roundtrip']:.1f} | {row['n_sents']}文")
    logger.info(f"  AKK_ORIG:      {row['akk_orig'][:150]}")
    logger.info(f"  AKK_ROUNDTRIP: {row['akk_roundtrip'][:150]}")
    logger.info(f"  ENG_ORIG:      {row['eng_orig'][:150]}")
    for sd in row['sent_details']:
        logger.info(f"    sent[{sd['sent_idx']}] ENG: {sd['eng_sent'][:80]}")
        logger.info(f"    sent[{sd['sent_idx']}] AKK: {sd['akk_pred'][:80]}")

logger.info("\n" + "=" * 60)
logger.info("=== 下位5件 (round-trip品質が低い) ===")
for _, row in results_sorted.tail(5).iterrows():
    logger.info(f"\nchrF++={row['chrf_roundtrip']:.1f} | {row['n_sents']}文")
    logger.info(f"  AKK_ORIG:      {row['akk_orig'][:150]}")
    logger.info(f"  AKK_ROUNDTRIP: {row['akk_roundtrip'][:150]}")
    logger.info(f"  ENG_ORIG:      {row['eng_orig'][:150]}")

# ============================================================
# アライメント可能性の分析
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("=== アライメント可能性の分析 ===")

# 逆翻訳された各文のAkkが、元Akkのどの部分に対応するかをchrF++で評価
# 元Akkを均等分割して各セグメントとの類似度を測る
alignment_scores = []

for _, row in results_df.iterrows():
    akk_orig = row['akk_orig']
    sent_details = row['sent_details']
    n_sents = row['n_sents']

    if n_sents < 2:
        continue

    # 元Akkをスペースで分割し、均等にn_sents個のチャンクに分ける
    akk_tokens = akk_orig.split()
    chunk_size = max(1, len(akk_tokens) // n_sents)
    akk_chunks = []
    for i in range(n_sents):
        start = i * chunk_size
        end = start + chunk_size if i < n_sents - 1 else len(akk_tokens)
        akk_chunks.append(" ".join(akk_tokens[start:end]))

    # 各逆翻訳文と各チャンクのchrF++行列を計算
    for si, sd in enumerate(sent_details):
        akk_pred = sd['akk_pred']
        best_chunk_idx = -1
        best_score = -1
        for ci, chunk in enumerate(akk_chunks):
            score = sacrebleu.sentence_chrf(akk_pred, [chunk], word_order=2).score
            if score > best_score:
                best_score = score
                best_chunk_idx = ci

        alignment_scores.append({
            'sent_idx': si,
            'n_sents': n_sents,
            'best_chunk_idx': best_chunk_idx,
            'is_diagonal': si == best_chunk_idx,  # 順序が保たれているか
            'best_score': best_score,
        })

align_df = pd.DataFrame(alignment_scores)
if len(align_df) > 0:
    diagonal_rate = align_df['is_diagonal'].mean()
    logger.info(f"対角一致率 (順序保存): {diagonal_rate:.2%} ({align_df['is_diagonal'].sum()}/{len(align_df)})")
    logger.info(f"平均chrF++ (best chunk): {align_df['best_score'].mean():.2f}")

    # 文数別の対角一致率
    for ns in sorted(align_df['n_sents'].unique()):
        sub = align_df[align_df['n_sents'] == ns]
        logger.info(f"  {ns}文ドキュメント: 対角一致 {sub['is_diagonal'].mean():.2%} (n={len(sub)})")

# ============================================================
# サマリー統計を保存
# ============================================================
summary = {
    'n_multi_sent_docs': len(multi_sent),
    'n_analyzed': len(sample_df),
    'chrf_roundtrip_mean': float(results_df['chrf_roundtrip'].mean()),
    'chrf_roundtrip_median': float(results_df['chrf_roundtrip'].median()),
    'diagonal_alignment_rate': float(diagonal_rate) if len(align_df) > 0 else None,
    'alignment_chrf_mean': float(align_df['best_score'].mean()) if len(align_df) > 0 else None,
}

with open(EDA_DIR / "summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)

# 詳細結果をCSVに保存
results_df.drop(columns=['sent_details']).to_csv(EDA_DIR / "roundtrip_results.csv", index=False)
logger.info(f"\n結果保存完了: {EDA_DIR}")

logger.info("\n=== 完了 ===")
