"""
ByT5-large (s1_exp001) vs Base (exp036) prediction analysis
Investigate why CV is similar but LB is much worse for large model.
"""
import pandas as pd
import numpy as np
import re
from collections import Counter

# ============================================================
# Load data
# ============================================================
large_doc = pd.read_csv("s1_exp001_byt5_large_fold3_last_doc_predictions.csv")
large_sent = pd.read_csv("s1_exp001_byt5_large_fold3_last_sent_predictions.csv")
base_doc = pd.read_csv("exp036_last_doc_predictions.csv")
base_sent = pd.read_csv("exp036_last_sent_predictions.csv")

# Also load exp023 fold3 for another comparison
try:
    base023_doc = pd.read_csv("exp023_gkf_fold3_last_doc_predictions.csv")
    base023_sent = pd.read_csv("exp023_gkf_fold3_last_sent_predictions.csv")
    has_023 = True
except:
    has_023 = False

print("=" * 80)
print("DATASET SIZES")
print("=" * 80)
print(f"Large doc: {len(large_doc)} rows, Large sent: {len(large_sent)} rows")
print(f"Base036 doc: {len(base_doc)} rows, Base036 sent: {len(base_sent)} rows")
if has_023:
    print(f"Base023 doc: {len(base023_doc)} rows, Base023 sent: {len(base023_sent)} rows")

# ============================================================
# 1. Prediction length analysis
# ============================================================
print("\n" + "=" * 80)
print("1. PREDICTION LENGTH ANALYSIS")
print("=" * 80)

def length_stats(df, col='prediction_clean', name=''):
    lengths = df[col].astype(str).str.len()
    ref_lengths = df['reference'].astype(str).str.len()
    ratios = lengths / ref_lengths.clip(lower=1)
    print(f"\n--- {name} ---")
    print(f"  Pred length:  mean={lengths.mean():.1f}, med={lengths.median():.1f}, "
          f"min={lengths.min()}, max={lengths.max()}, std={lengths.std():.1f}")
    print(f"  Ref length:   mean={ref_lengths.mean():.1f}, med={ref_lengths.median():.1f}, "
          f"min={ref_lengths.min()}, max={ref_lengths.max()}, std={ref_lengths.std():.1f}")
    print(f"  Ratio (pred/ref): mean={ratios.mean():.3f}, med={ratios.median():.3f}, "
          f"min={ratios.min():.3f}, max={ratios.max():.3f}")
    # Check for very long or very short predictions
    very_short = (ratios < 0.5).sum()
    very_long = (ratios > 2.0).sum()
    print(f"  Very short (<0.5x ref): {very_short} ({100*very_short/len(df):.1f}%)")
    print(f"  Very long (>2x ref): {very_long} ({100*very_long/len(df):.1f}%)")
    return lengths, ref_lengths, ratios

l_doc_len, l_doc_ref, l_doc_ratio = length_stats(large_doc, name="Large DOC")
b_doc_len, b_doc_ref, b_doc_ratio = length_stats(base_doc, name="Base036 DOC")
l_sent_len, l_sent_ref, l_sent_ratio = length_stats(large_sent, name="Large SENT")
b_sent_len, b_sent_ref, b_sent_ratio = length_stats(base_sent, name="Base036 SENT")

# ============================================================
# 2. Truncation / cut-off analysis
# ============================================================
print("\n" + "=" * 80)
print("2. TRUNCATION / CUT-OFF ANALYSIS")
print("=" * 80)

def check_truncation(df, name=''):
    """Check for predictions that end abruptly (no proper sentence ending)"""
    preds = df['prediction_clean'].astype(str)
    # Check if prediction ends with proper punctuation
    ends_properly = preds.str.match(r'.*[.!?;"\')]$')
    no_proper_end = ~ends_properly
    print(f"\n--- {name} ---")
    print(f"  Total predictions: {len(df)}")
    print(f"  No proper ending (no .!?;\"'): {no_proper_end.sum()} ({100*no_proper_end.sum()/len(df):.1f}%)")

    # Check for specific truncation patterns
    pred_lengths = preds.str.len()

    # Find the max length predictions that might be truncated
    long_threshold = pred_lengths.quantile(0.95)
    long_preds = df[pred_lengths >= long_threshold]
    long_no_end = long_preds[~long_preds['prediction_clean'].astype(str).str.match(r'.*[.!?;"\')]$')]
    print(f"  Long predictions (>95th percentile, len>={long_threshold:.0f}): {len(long_preds)}")
    print(f"  Long + no proper ending: {len(long_no_end)} ({100*len(long_no_end)/max(len(long_preds),1):.1f}%)")

    # Show examples of truncated predictions
    if len(long_no_end) > 0:
        print(f"\n  Examples of potentially truncated predictions:")
        for idx, row in long_no_end.head(3).iterrows():
            pred = str(row['prediction_clean'])
            ref = str(row['reference'])
            print(f"    [{idx}] Pred (last 80 chars): ...{pred[-80:]}")
            print(f"         Ref  (last 80 chars): ...{ref[-80:]}")
            print(f"         Pred len: {len(pred)}, Ref len: {len(ref)}")

    return no_proper_end

l_doc_trunc = check_truncation(large_doc, name="Large DOC")
b_doc_trunc = check_truncation(base_doc, name="Base036 DOC")
l_sent_trunc = check_truncation(large_sent, name="Large SENT")
b_sent_trunc = check_truncation(base_sent, name="Base036 SENT")

# Also check max token length at specific cutoff points
print("\n--- Character length distribution at top end ---")
for name, df in [("Large DOC", large_doc), ("Base036 DOC", base_doc),
                  ("Large SENT", large_sent), ("Base036 SENT", base_sent)]:
    lens = df['prediction_clean'].astype(str).str.len()
    print(f"  {name}: 90th={lens.quantile(0.9):.0f}, 95th={lens.quantile(0.95):.0f}, "
          f"99th={lens.quantile(0.99):.0f}, max={lens.max()}")

# ============================================================
# 3. Repetition analysis
# ============================================================
print("\n" + "=" * 80)
print("3. REPETITION ANALYSIS")
print("=" * 80)

def detect_repetitions(text, min_ngram=3, min_repeats=2):
    """Detect repeated n-grams in text"""
    words = text.split()
    if len(words) < min_ngram * min_repeats:
        return 0, []

    repeated = []
    for n in range(min_ngram, min(8, len(words)//2 + 1)):
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        for ngram, count in counts.items():
            if count >= min_repeats:
                repeated.append((ngram, count))

    return len(repeated), repeated

def repetition_stats(df, name=''):
    preds = df['prediction_clean'].astype(str)
    rep_counts = []
    heavy_rep_examples = []

    for idx, pred in preds.items():
        n_reps, reps = detect_repetitions(pred)
        rep_counts.append(n_reps)
        if n_reps > 5:
            heavy_rep_examples.append((idx, pred[:200], reps[:3]))

    rep_counts = np.array(rep_counts)
    has_rep = (rep_counts > 0).sum()
    heavy_rep = (rep_counts > 5).sum()

    print(f"\n--- {name} ---")
    print(f"  Any repetition: {has_rep} ({100*has_rep/len(df):.1f}%)")
    print(f"  Heavy repetition (>5 patterns): {heavy_rep} ({100*heavy_rep/len(df):.1f}%)")
    print(f"  Mean rep count: {rep_counts.mean():.2f}")

    if heavy_rep_examples:
        print(f"\n  Examples of heavy repetition:")
        for idx, pred, reps in heavy_rep_examples[:3]:
            print(f"    [{idx}] {pred}...")
            for ngram, count in reps:
                print(f"         Repeated {count}x: '{ngram}'")

    return rep_counts

# Also check for consecutive word repetition (stuttering)
def check_consecutive_repeats(df, name=''):
    """Check for patterns like 'the the the' or repeated phrases"""
    preds = df['prediction_clean'].astype(str)
    stutter_count = 0
    stutter_examples = []

    for idx, pred in preds.items():
        # Check for 3+ consecutive repeated words
        match = re.search(r'\b(\w+)(?:\s+\1){2,}', pred)
        if match:
            stutter_count += 1
            if len(stutter_examples) < 3:
                stutter_examples.append((idx, pred[:200], match.group()))

    print(f"\n--- {name} Consecutive word repeats ---")
    print(f"  Stuttering (3+ same word in row): {stutter_count} ({100*stutter_count/len(df):.1f}%)")
    for idx, pred, match in stutter_examples:
        print(f"    [{idx}] Match: '{match}' in: {pred[:100]}...")

l_doc_rep = repetition_stats(large_doc, name="Large DOC")
b_doc_rep = repetition_stats(base_doc, name="Base036 DOC")
l_sent_rep = repetition_stats(large_sent, name="Large SENT")
b_sent_rep = repetition_stats(base_sent, name="Base036 SENT")

check_consecutive_repeats(large_doc, "Large DOC")
check_consecutive_repeats(base_doc, "Base036 DOC")
check_consecutive_repeats(large_sent, "Large SENT")
check_consecutive_repeats(base_sent, "Base036 SENT")

# ============================================================
# 4. Special character / encoding analysis
# ============================================================
print("\n" + "=" * 80)
print("4. SPECIAL CHARACTER / ENCODING ANALYSIS")
print("=" * 80)

def check_special_chars(df, name=''):
    preds = df['prediction_clean'].astype(str)

    # Check for non-ASCII characters (excluding common diacritics)
    has_unusual = 0
    unusual_examples = []

    for idx, pred in preds.items():
        # Find non-printable or unusual Unicode characters
        unusual = re.findall(r'[^\x20-\x7E\u00C0-\u024F\u1E00-\u1EFF]', pred)
        if unusual:
            has_unusual += 1
            if len(unusual_examples) < 5:
                unusual_chars = set(unusual)
                unusual_examples.append((idx, unusual_chars, pred[:150]))

    print(f"\n--- {name} ---")
    print(f"  Predictions with unusual characters: {has_unusual} ({100*has_unusual/len(df):.1f}%)")
    for idx, chars, pred in unusual_examples[:3]:
        char_info = [(c, hex(ord(c)), f'U+{ord(c):04X}') for c in list(chars)[:5]]
        print(f"    [{idx}] Chars: {char_info}")
        print(f"         Text: {pred[:100]}...")

    # Check for <gap> markers
    has_gap = preds.str.contains('<gap>', regex=False).sum()
    print(f"  Predictions with <gap>: {has_gap} ({100*has_gap/len(df):.1f}%)")

    # Check for empty predictions
    empty = (preds.str.strip() == '').sum()
    print(f"  Empty predictions: {empty}")

    # Check for prediction_raw vs prediction_clean differences
    raw = df['prediction_raw'].astype(str)
    differs = (raw != preds).sum()
    print(f"  Raw != Clean: {differs} ({100*differs/len(df):.1f}%)")

check_special_chars(large_doc, "Large DOC")
check_special_chars(base_doc, "Base036 DOC")
check_special_chars(large_sent, "Large SENT")
check_special_chars(base_sent, "Base036 SENT")

# ============================================================
# 5. Sent vs Doc comparison for large model
# ============================================================
print("\n" + "=" * 80)
print("5. SENT vs DOC LEVEL ANALYSIS (Large model)")
print("=" * 80)

# Compare sent-level and doc-level lengths
print(f"\nLarge model:")
print(f"  Sent pred length: mean={large_sent['prediction_clean'].str.len().mean():.1f}")
print(f"  Doc pred length:  mean={large_doc['prediction_clean'].str.len().mean():.1f}")
print(f"  Doc ref length:   mean={large_doc['reference'].str.len().mean():.1f}")

# Doc-level: check ratio of predicted doc length vs sum of sent predictions
# This helps detect if doc-level generation adds/removes content
print(f"\nBase036 model:")
print(f"  Sent pred length: mean={base_sent['prediction_clean'].str.len().mean():.1f}")
print(f"  Doc pred length:  mean={base_doc['prediction_clean'].str.len().mean():.1f}")
print(f"  Doc ref length:   mean={base_doc['reference'].str.len().mean():.1f}")

# ============================================================
# 6. Side-by-side comparison of specific examples
# ============================================================
print("\n" + "=" * 80)
print("6. SIDE-BY-SIDE COMPARISON (Large vs Base036)")
print("=" * 80)

# Match on input column
print("\n--- SENT-level comparison ---")
merged_sent = large_sent.merge(base_sent, on='input', suffixes=('_large', '_base'))
print(f"Matched sent predictions: {len(merged_sent)}")

# Find cases where large is much worse than base
if len(merged_sent) > 0:
    # Compare lengths
    merged_sent['large_len'] = merged_sent['prediction_clean_large'].str.len()
    merged_sent['base_len'] = merged_sent['prediction_clean_base'].str.len()
    merged_sent['ref_len'] = merged_sent['reference_large'].str.len()
    merged_sent['large_ratio'] = merged_sent['large_len'] / merged_sent['ref_len'].clip(lower=1)
    merged_sent['base_ratio'] = merged_sent['base_len'] / merged_sent['ref_len'].clip(lower=1)
    merged_sent['ratio_diff'] = merged_sent['large_ratio'] - merged_sent['base_ratio']

    print(f"\nLength ratio (pred/ref):")
    print(f"  Large: mean={merged_sent['large_ratio'].mean():.3f}")
    print(f"  Base:  mean={merged_sent['base_ratio'].mean():.3f}")

    # Show examples where large prediction is much longer or shorter than base
    print(f"\n--- Cases where Large is much LONGER than Base (top 5) ---")
    much_longer = merged_sent.nlargest(5, 'ratio_diff')
    for _, row in much_longer.iterrows():
        print(f"\n  Input: {str(row['input'])[:100]}...")
        print(f"  Ref:   {str(row['reference_large'])[:120]}...")
        print(f"  Large: {str(row['prediction_clean_large'])[:120]}... (len={row['large_len']}, ratio={row['large_ratio']:.2f})")
        print(f"  Base:  {str(row['prediction_clean_base'])[:120]}... (len={row['base_len']}, ratio={row['base_ratio']:.2f})")

    print(f"\n--- Cases where Large is much SHORTER than Base (top 5) ---")
    much_shorter = merged_sent.nsmallest(5, 'ratio_diff')
    for _, row in much_shorter.iterrows():
        print(f"\n  Input: {str(row['input'])[:100]}...")
        print(f"  Ref:   {str(row['reference_large'])[:120]}...")
        print(f"  Large: {str(row['prediction_clean_large'])[:120]}... (len={row['large_len']}, ratio={row['large_ratio']:.2f})")
        print(f"  Base:  {str(row['prediction_clean_base'])[:120]}... (len={row['base_len']}, ratio={row['base_ratio']:.2f})")

print("\n--- DOC-level comparison ---")
merged_doc = large_doc.merge(base_doc, on='input', suffixes=('_large', '_base'))
print(f"Matched doc predictions: {len(merged_doc)}")

if len(merged_doc) > 0:
    merged_doc['large_len'] = merged_doc['prediction_clean_large'].str.len()
    merged_doc['base_len'] = merged_doc['prediction_clean_base'].str.len()
    merged_doc['ref_len'] = merged_doc['reference_large'].str.len()
    merged_doc['large_ratio'] = merged_doc['large_len'] / merged_doc['ref_len'].clip(lower=1)
    merged_doc['base_ratio'] = merged_doc['base_len'] / merged_doc['ref_len'].clip(lower=1)

    print(f"\nDoc length ratio (pred/ref):")
    print(f"  Large: mean={merged_doc['large_ratio'].mean():.3f}")
    print(f"  Base:  mean={merged_doc['base_ratio'].mean():.3f}")

    # Show specific examples where large deviates significantly
    merged_doc['ratio_diff'] = (merged_doc['large_ratio'] - merged_doc['base_ratio']).abs()

    print(f"\n--- Most divergent DOC predictions (top 5) ---")
    divergent = merged_doc.nlargest(5, 'ratio_diff')
    for _, row in divergent.iterrows():
        print(f"\n  Input: {str(row['input'])[:100]}...")
        print(f"  Ref:   {str(row['reference_large'])[:150]}...")
        print(f"  Large: {str(row['prediction_clean_large'])[:150]}... (len={row['large_len']}, ratio={row['large_ratio']:.2f})")
        print(f"  Base:  {str(row['prediction_clean_base'])[:150]}... (len={row['base_len']}, ratio={row['base_ratio']:.2f})")

# ============================================================
# 7. Detailed truncation check at specific char lengths
# ============================================================
print("\n" + "=" * 80)
print("7. DETAILED TRUNCATION CHECK - Max length cutoff detection")
print("=" * 80)

for name, df in [("Large DOC", large_doc), ("Base036 DOC", base_doc),
                  ("Large SENT", large_sent), ("Base036 SENT", base_sent)]:
    lens = df['prediction_clean'].astype(str).str.len()
    # Check if many predictions cluster at a specific length (sign of truncation)
    max_len = lens.max()
    near_max = (lens >= max_len - 5).sum()
    print(f"\n{name}: max_len={max_len}, near_max (within 5 chars): {near_max}")

    # Show top 10 longest predictions and their endings
    top10_idx = lens.nlargest(10).index
    for idx in top10_idx[:5]:
        pred = str(df.loc[idx, 'prediction_clean'])
        ref = str(df.loc[idx, 'reference'])
        print(f"  [{idx}] len={len(pred)}, ref_len={len(ref)}, "
              f"ends: ...{pred[-50:]}")

# ============================================================
# 8. Word-level analysis
# ============================================================
print("\n" + "=" * 80)
print("8. WORD-LEVEL ANALYSIS")
print("=" * 80)

def word_stats(df, name=''):
    preds = df['prediction_clean'].astype(str)
    refs = df['reference'].astype(str)

    pred_words = preds.str.split().str.len()
    ref_words = refs.str.split().str.len()
    ratio = pred_words / ref_words.clip(lower=1)

    print(f"\n--- {name} ---")
    print(f"  Pred words: mean={pred_words.mean():.1f}, med={pred_words.median():.1f}")
    print(f"  Ref words:  mean={ref_words.mean():.1f}, med={ref_words.median():.1f}")
    print(f"  Word ratio: mean={ratio.mean():.3f}, med={ratio.median():.3f}")

    # Vocabulary analysis
    all_pred_words = ' '.join(preds).lower().split()
    all_ref_words = ' '.join(refs).lower().split()
    pred_vocab = set(all_pred_words)
    ref_vocab = set(all_ref_words)

    print(f"  Pred vocab size: {len(pred_vocab)}")
    print(f"  Ref vocab size: {len(ref_vocab)}")
    print(f"  Pred-only words: {len(pred_vocab - ref_vocab)}")
    print(f"  Ref-only words: {len(ref_vocab - pred_vocab)}")

    # Most common pred-only words (hallucinations?)
    pred_only = pred_vocab - ref_vocab
    pred_only_freq = Counter(w for w in all_pred_words if w in pred_only)
    print(f"  Top pred-only words: {pred_only_freq.most_common(10)}")

word_stats(large_sent, "Large SENT")
word_stats(base_sent, "Base036 SENT")
word_stats(large_doc, "Large DOC")
word_stats(base_doc, "Base036 DOC")

# ============================================================
# 9. prediction_raw vs prediction_clean difference analysis
# ============================================================
print("\n" + "=" * 80)
print("9. RAW vs CLEAN PREDICTION DIFFERENCES")
print("=" * 80)

for name, df in [("Large DOC", large_doc), ("Base036 DOC", base_doc),
                  ("Large SENT", large_sent), ("Base036 SENT", base_sent)]:
    raw = df['prediction_raw'].astype(str)
    clean = df['prediction_clean'].astype(str)
    differs = raw != clean
    print(f"\n--- {name} ---")
    print(f"  Differs: {differs.sum()} ({100*differs.sum()/len(df):.1f}%)")
    if differs.sum() > 0:
        diff_examples = df[differs].head(3)
        for idx, row in diff_examples.iterrows():
            r = str(row['prediction_raw'])
            c = str(row['prediction_clean'])
            print(f"    [{idx}] Raw:   {r[:120]}...")
            print(f"         Clean: {c[:120]}...")

# ============================================================
# 10. Distribution of prediction/reference length ratio
# ============================================================
print("\n" + "=" * 80)
print("10. LENGTH RATIO HISTOGRAM (pred_len / ref_len)")
print("=" * 80)

def ratio_histogram(df, name=''):
    preds = df['prediction_clean'].astype(str).str.len()
    refs = df['reference'].astype(str).str.len()
    ratios = preds / refs.clip(lower=1)

    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 100]
    labels = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.1',
              '1.1-1.3', '1.3-1.5', '1.5-2.0', '2.0-3.0', '>3.0']
    hist = pd.cut(ratios, bins=bins, labels=labels).value_counts().sort_index()

    print(f"\n--- {name} ---")
    for label, count in hist.items():
        pct = 100 * count / len(df)
        bar = '#' * int(pct)
        print(f"  {label:>8s}: {count:4d} ({pct:5.1f}%) {bar}")

ratio_histogram(large_sent, "Large SENT")
ratio_histogram(base_sent, "Base036 SENT")
ratio_histogram(large_doc, "Large DOC")
ratio_histogram(base_doc, "Base036 DOC")

# ============================================================
# 11. Sentence ending patterns (important for test which is sentence-level)
# ============================================================
print("\n" + "=" * 80)
print("11. SENTENCE ENDING PATTERNS (Critical for test set)")
print("=" * 80)

for name, df in [("Large SENT", large_sent), ("Base036 SENT", base_sent)]:
    preds = df['prediction_clean'].astype(str)

    # Last character distribution
    last_chars = preds.str[-1:]
    char_dist = last_chars.value_counts().head(10)

    print(f"\n--- {name} ---")
    print(f"  Last character distribution:")
    for char, count in char_dist.items():
        print(f"    '{char}': {count} ({100*count/len(df):.1f}%)")

    # Last 5 characters for pattern detection
    last5 = preds.str[-5:]
    print(f"\n  Last 5 chars (sample):")
    for v in last5.head(10):
        print(f"    ...{v}")

# ============================================================
# 12. Summary of key findings
# ============================================================
print("\n" + "=" * 80)
print("12. KEY FINDINGS SUMMARY")
print("=" * 80)

# Compute some summary stats
l_sent_ratio_mean = (large_sent['prediction_clean'].str.len() / large_sent['reference'].str.len().clip(lower=1)).mean()
b_sent_ratio_mean = (base_sent['prediction_clean'].str.len() / base_sent['reference'].str.len().clip(lower=1)).mean()
l_doc_ratio_mean = (large_doc['prediction_clean'].str.len() / large_doc['reference'].str.len().clip(lower=1)).mean()
b_doc_ratio_mean = (base_doc['prediction_clean'].str.len() / base_doc['reference'].str.len().clip(lower=1)).mean()

l_sent_trunc_pct = 100 * (~large_sent['prediction_clean'].astype(str).str.match(r'.*[.!?;"\')]$')).sum() / len(large_sent)
b_sent_trunc_pct = 100 * (~base_sent['prediction_clean'].astype(str).str.match(r'.*[.!?;"\')]$')).sum() / len(base_sent)
l_doc_trunc_pct = 100 * (~large_doc['prediction_clean'].astype(str).str.match(r'.*[.!?;"\')]$')).sum() / len(large_doc)
b_doc_trunc_pct = 100 * (~base_doc['prediction_clean'].astype(str).str.match(r'.*[.!?;"\')]$')).sum() / len(base_doc)

print(f"""
                    Large           Base036
SENT len ratio:     {l_sent_ratio_mean:.3f}          {b_sent_ratio_mean:.3f}
DOC len ratio:      {l_doc_ratio_mean:.3f}          {b_doc_ratio_mean:.3f}
SENT trunc%:        {l_sent_trunc_pct:.1f}%           {b_sent_trunc_pct:.1f}%
DOC trunc%:         {l_doc_trunc_pct:.1f}%           {b_doc_trunc_pct:.1f}%
""")

print("=== ANALYSIS COMPLETE ===")
