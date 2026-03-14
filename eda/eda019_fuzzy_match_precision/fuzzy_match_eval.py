"""
eda019: 逆翻訳Akkフラグメントのfuzzy match精度検証

目的:
- 英語文を逆翻訳したAkkフラグメントが、元Akkテキスト内の正しい位置にマッチできるかを検証
- スライディングウィンドウ + chrF++ で最適位置を探索
- 文レベルアライメントの実用性を定量的に判断

手順:
1. train.csvから複数文ドキュメントを抽出
2. 英語翻訳を文分割→各文をEng→Akkに逆翻訳
3. 逆翻訳Akkを元Akkテキスト上でスライディングウィンドウ検索
4. マッチ位置の正しさを順序保存率で評価
5. 抽出されたセグメントの品質をchrF++で評価
"""
import os
import re
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        logging.FileHandler(str(EDA_DIR / "fuzzy_match_eval.log")),
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
# Load data
# ============================================================
df = pd.read_csv(TRAIN_PATH)
logger.info(f"Total train documents: {len(df)}")


def split_sentences(text):
    """英語テキストを文分割"""
    text = str(text).strip()
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sents


df['n_eng_sents'] = df['translation'].apply(lambda t: len(split_sentences(str(t))))
df['akk_bytes'] = df['transliteration'].astype(str).apply(lambda t: len(t.encode('utf-8')))

# 2文以上かつ512B超のドキュメントに注目（truncationが発生するケース）
multi_sent = df[df['n_eng_sents'] >= 2].copy()
long_docs = multi_sent[multi_sent['akk_bytes'] > 512].copy()
logger.info(f"複数文ドキュメント: {len(multi_sent)} / {len(df)}")
logger.info(f"  うち512B超: {len(long_docs)} ({100*len(long_docs)/len(df):.1f}%)")

# ============================================================
# Translation helper
# ============================================================
PREFIX_REV = "translate English to Akkadian: "


def translate_batch(texts, prefix, batch_size=8, max_length=512, num_beams=4):
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


# ============================================================
# Fuzzy match: sliding window on token level
# ============================================================
def find_best_match_sliding_window(query_akk, orig_akk, window_sizes=None):
    """
    query_akk（逆翻訳Akk）を orig_akk 上でスライディングウィンドウ検索。
    トークン（スペース区切り）レベルでウィンドウを動かす。

    Returns:
        best_start_tok: 最適ウィンドウの開始トークンindex
        best_end_tok: 最適ウィンドウの終了トークンindex
        best_score: chrF++スコア
        best_segment: マッチしたテキストセグメント
    """
    orig_tokens = orig_akk.split()
    query_tokens = query_akk.split()
    n_orig = len(orig_tokens)
    n_query = len(query_tokens)

    if n_orig == 0 or n_query == 0:
        return 0, 0, 0.0, ""

    # ウィンドウサイズ: クエリ長の0.5倍〜2.0倍の範囲
    if window_sizes is None:
        min_w = max(1, int(n_query * 0.5))
        max_w = min(n_orig, int(n_query * 2.0))
        # ステップを入れて効率化
        window_sizes = list(range(min_w, max_w + 1, max(1, (max_w - min_w) // 10)))
        # クエリ長そのものも必ず含める
        if n_query not in window_sizes and n_query <= n_orig:
            window_sizes.append(n_query)
        window_sizes = sorted(set(window_sizes))

    best_score = -1
    best_start = 0
    best_end = n_orig
    best_segment = orig_akk

    for w in window_sizes:
        if w > n_orig:
            continue
        for start in range(0, n_orig - w + 1):
            end = start + w
            segment = " ".join(orig_tokens[start:end])
            score = sacrebleu.sentence_chrf(query_akk, [segment], word_order=2).score
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end
                best_segment = segment

    return best_start, best_end, best_score, best_segment


# ============================================================
# Main analysis
# ============================================================
# サンプリング: 複数文ドキュメント全体から100件
np.random.seed(42)
sample_size = 100
sample_indices = np.random.choice(multi_sent.index, size=min(sample_size, len(multi_sent)), replace=False)
sample_df = multi_sent.loc[sample_indices].copy()
logger.info(f"\n分析対象: {len(sample_df)}件")
logger.info(f"  うち512B超: {(sample_df['akk_bytes'] > 512).sum()}件")

# 逆翻訳 + fuzzy match
all_doc_results = []
all_sent_results = []

for doc_idx, (idx, row) in enumerate(tqdm(sample_df.iterrows(), total=len(sample_df), desc="Fuzzy match")):
    akk_orig = str(row['transliteration'])
    eng_orig = str(row['translation'])
    eng_sents = split_sentences(eng_orig)
    n_sents = len(eng_sents)

    # 各英語文を逆翻訳
    akk_preds = translate_batch(eng_sents, PREFIX_REV, batch_size=len(eng_sents))

    # 各逆翻訳Akkについてfuzzy matchで最適位置を探す
    sent_matches = []
    for si, (eng_s, akk_pred) in enumerate(zip(eng_sents, akk_preds)):
        start_tok, end_tok, score, segment = find_best_match_sliding_window(akk_pred, akk_orig)

        sent_matches.append({
            'doc_idx': doc_idx,
            'oare_id': row['oare_id'],
            'sent_idx': si,
            'n_sents': n_sents,
            'akk_bytes': len(akk_orig.encode('utf-8')),
            'is_long': len(akk_orig.encode('utf-8')) > 512,
            'eng_sent': eng_s,
            'akk_pred': akk_pred,
            'match_start_tok': start_tok,
            'match_end_tok': end_tok,
            'match_score': score,
            'matched_segment': segment,
        })
        all_sent_results.append(sent_matches[-1])

    # 順序保存チェック: マッチ位置が文の順番通りか
    positions = [(m['match_start_tok'], m['match_end_tok']) for m in sent_matches]
    is_ordered = all(positions[i][0] <= positions[i+1][0] for i in range(len(positions)-1))
    # 重複チェック: マッチ領域の重なり
    has_overlap = any(
        positions[i][1] > positions[i+1][0]
        for i in range(len(positions)-1)
    ) if is_ordered else True

    # カバレッジ: マッチ領域が元テキストの何%をカバーしているか
    orig_n_tokens = len(akk_orig.split())
    covered_tokens = set()
    for m in sent_matches:
        for t in range(m['match_start_tok'], m['match_end_tok']):
            covered_tokens.add(t)
    coverage = len(covered_tokens) / orig_n_tokens if orig_n_tokens > 0 else 0

    all_doc_results.append({
        'oare_id': row['oare_id'],
        'n_sents': n_sents,
        'akk_bytes': len(akk_orig.encode('utf-8')),
        'is_long': len(akk_orig.encode('utf-8')) > 512,
        'is_ordered': is_ordered,
        'has_overlap': has_overlap,
        'coverage': coverage,
        'mean_match_score': np.mean([m['match_score'] for m in sent_matches]),
        'min_match_score': np.min([m['match_score'] for m in sent_matches]),
    })

doc_df = pd.DataFrame(all_doc_results)
sent_df = pd.DataFrame(all_sent_results)

# ============================================================
# Results
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("=== Fuzzy Match 精度サマリー ===")
logger.info(f"\n全{len(doc_df)}ドキュメント:")
logger.info(f"  順序保存率: {doc_df['is_ordered'].mean():.1%} ({doc_df['is_ordered'].sum()}/{len(doc_df)})")
logger.info(f"  重複なし率 (順序保存のうち): {(~doc_df['has_overlap'] & doc_df['is_ordered']).sum()}/{doc_df['is_ordered'].sum()}")
logger.info(f"  平均カバレッジ: {doc_df['coverage'].mean():.1%}")
logger.info(f"  平均match chrF++: {doc_df['mean_match_score'].mean():.2f}")
logger.info(f"  min match chrF++ mean: {doc_df['min_match_score'].mean():.2f}")

# 文数別
logger.info("\n--- 文数別 ---")
for ns in sorted(doc_df['n_sents'].unique()):
    sub = doc_df[doc_df['n_sents'] == ns]
    if len(sub) < 3:
        continue
    logger.info(f"  {ns}文 (n={len(sub)}): 順序保存={sub['is_ordered'].mean():.0%}, "
                f"カバレッジ={sub['coverage'].mean():.0%}, "
                f"mean chrF++={sub['mean_match_score'].mean():.1f}")

# 512B超 vs 以下
logger.info("\n--- 512B超 vs 以下 ---")
for is_long, label in [(True, "512B超"), (False, "512B以下")]:
    sub = doc_df[doc_df['is_long'] == is_long]
    if len(sub) == 0:
        continue
    logger.info(f"  {label} (n={len(sub)}): 順序保存={sub['is_ordered'].mean():.0%}, "
                f"カバレッジ={sub['coverage'].mean():.0%}, "
                f"mean chrF++={sub['mean_match_score'].mean():.1f}")

# 具体例: 順序保存かつスコア高い例
logger.info("\n--- 成功例 (順序保存 & mean chrF++ > 60) ---")
good = doc_df[doc_df['is_ordered'] & (doc_df['mean_match_score'] > 60)].sort_values('mean_match_score', ascending=False)
for _, drow in good.head(5).iterrows():
    oare = drow['oare_id']
    doc_sents = sent_df[sent_df['oare_id'] == oare].sort_values('sent_idx')
    logger.info(f"\n  [{oare}] {drow['n_sents']}文, {drow['akk_bytes']}B, chrF++={drow['mean_match_score']:.1f}")
    for _, s in doc_sents.iterrows():
        logger.info(f"    sent[{s['sent_idx']}] score={s['match_score']:.1f} "
                     f"tok[{s['match_start_tok']}:{s['match_end_tok']}]")
        logger.info(f"      ENG: {s['eng_sent'][:80]}")
        logger.info(f"      PRED: {s['akk_pred'][:80]}")
        logger.info(f"      ORIG: {s['matched_segment'][:80]}")

# 失敗例
logger.info("\n--- 失敗例 (順序が崩れた例) ---")
bad = doc_df[~doc_df['is_ordered']].sort_values('mean_match_score', ascending=False)
for _, drow in bad.head(3).iterrows():
    oare = drow['oare_id']
    doc_sents = sent_df[sent_df['oare_id'] == oare].sort_values('sent_idx')
    logger.info(f"\n  [{oare}] {drow['n_sents']}文, {drow['akk_bytes']}B, chrF++={drow['mean_match_score']:.1f}")
    for _, s in doc_sents.iterrows():
        logger.info(f"    sent[{s['sent_idx']}] score={s['match_score']:.1f} "
                     f"tok[{s['match_start_tok']}:{s['match_end_tok']}]")

# ============================================================
# 実用性評価: 512B超ドキュメントの文分割で得られるデータ量
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("=== 実用性評価: データ増加量の推定 ===")

long_multi = df[(df['n_eng_sents'] >= 2) & (df['akk_bytes'] > 512)]
logger.info(f"512B超の複数文ドキュメント: {len(long_multi)}件")
logger.info(f"  平均文数: {long_multi['n_eng_sents'].mean():.1f}")
logger.info(f"  平均Akkバイト数: {long_multi['akk_bytes'].mean():.0f}")

# 分割すると各文が512B以下になる可能性
estimated_sents = long_multi['n_eng_sents'].sum()
logger.info(f"  分割後の推定文数: {estimated_sents}")
logger.info(f"  現在truncationで失われるデータ: ~{long_multi['akk_bytes'].sum() - len(long_multi)*512:.0f} bytes")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Match score distribution
axes[0, 0].hist(sent_df['match_score'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('chrF++ Match Score')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Fuzzy Match Score Distribution (per sentence)')
axes[0, 0].axvline(x=60, color='r', linestyle='--', label='threshold=60')
axes[0, 0].legend()

# 2. 順序保存率 by n_sents
ns_groups = doc_df.groupby('n_sents').agg(
    order_rate=('is_ordered', 'mean'),
    count=('is_ordered', 'count')
).reset_index()
ns_groups = ns_groups[ns_groups['count'] >= 3]
axes[0, 1].bar(ns_groups['n_sents'].astype(str), ns_groups['order_rate'],
               edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Number of sentences')
axes[0, 1].set_ylabel('Order preservation rate')
axes[0, 1].set_title('Order Preservation Rate by Sentence Count')
axes[0, 1].set_ylim(0, 1.05)
for i, row in ns_groups.iterrows():
    axes[0, 1].text(str(row['n_sents']), row['order_rate'] + 0.02,
                     f"n={row['count']}", ha='center', fontsize=9)

# 3. Coverage distribution
axes[1, 0].hist(doc_df['coverage'], bins=20, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Token Coverage')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Coverage of Original Text by Matched Segments')

# 4. Match score by long/short
for is_long, label, color in [(False, '≤512B', 'steelblue'), (True, '>512B', 'coral')]:
    sub = sent_df[sent_df['is_long'] == is_long]
    if len(sub) > 0:
        axes[1, 1].hist(sub['match_score'], bins=20, alpha=0.5, label=label, color=color, edgecolor='black')
axes[1, 1].set_xlabel('chrF++ Match Score')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Match Score: Short vs Long Documents')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "fuzzy_match_summary.png"), dpi=150)
logger.info(f"\nFigure saved: {FIGURES_DIR / 'fuzzy_match_summary.png'}")

# ============================================================
# Save results
# ============================================================
summary = {
    'n_docs_analyzed': len(doc_df),
    'order_preservation_rate': float(doc_df['is_ordered'].mean()),
    'mean_coverage': float(doc_df['coverage'].mean()),
    'mean_match_chrf': float(doc_df['mean_match_score'].mean()),
    'n_long_docs_total': len(long_multi),
    'estimated_sents_from_long': int(estimated_sents),
}

with open(EDA_DIR / "summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)

sent_df.to_csv(EDA_DIR / "fuzzy_match_results.csv", index=False)
doc_df.to_csv(EDA_DIR / "doc_results.csv", index=False)
logger.info(f"\n結果保存完了: {EDA_DIR}")
logger.info("=== 完了 ===")
