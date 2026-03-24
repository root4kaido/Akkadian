"""
Deep analysis: Why does ByT5-large have similar CV but much worse LB?
Focus on sentence-level predictions since test is sentence-level.
"""
import pandas as pd
import numpy as np
import re

large_sent = pd.read_csv("s1_exp001_byt5_large_fold3_last_sent_predictions.csv")
base_sent = pd.read_csv("exp036_last_sent_predictions.csv")

# ============================================================
# A. ByT5 byte-level: check if max_new_tokens is hitting byte limit
# ============================================================
print("=" * 80)
print("A. BYTE-LEVEL LENGTH ANALYSIS (ByT5 uses bytes, not tokens)")
print("=" * 80)

def byte_lengths(df, name=''):
    preds = df['prediction_clean'].astype(str)
    refs = df['reference'].astype(str)

    pred_bytes = preds.apply(lambda x: len(x.encode('utf-8')))
    ref_bytes = refs.apply(lambda x: len(x.encode('utf-8')))

    print(f"\n--- {name} ---")
    print(f"  Pred bytes: mean={pred_bytes.mean():.1f}, max={pred_bytes.max()}, "
          f"95th={pred_bytes.quantile(0.95):.0f}, 99th={pred_bytes.quantile(0.99):.0f}")
    print(f"  Ref bytes:  mean={ref_bytes.mean():.1f}, max={ref_bytes.max()}, "
          f"95th={ref_bytes.quantile(0.95):.0f}, 99th={ref_bytes.quantile(0.99):.0f}")

    # Check common max_new_tokens values: 256, 512, 1024
    for limit in [128, 256, 384, 512, 768, 1024]:
        at_limit = (pred_bytes >= limit - 5).sum()
        if at_limit > 0:
            print(f"  At byte limit ~{limit} (within 5): {at_limit}")

    return pred_bytes, ref_bytes

l_bytes, l_ref_bytes = byte_lengths(large_sent, "Large SENT")
b_bytes, b_ref_bytes = byte_lengths(base_sent, "Base036 SENT")

# Also raw prediction bytes (before cleaning)
print("\n--- Raw prediction bytes ---")
for name, df in [("Large SENT", large_sent), ("Base036 SENT", base_sent)]:
    raw_bytes = df['prediction_raw'].astype(str).apply(lambda x: len(x.encode('utf-8')))
    print(f"  {name}: max={raw_bytes.max()}, 95th={raw_bytes.quantile(0.95):.0f}, "
          f"99th={raw_bytes.quantile(0.99):.0f}")

# ============================================================
# B. Per-sample BLEU/chrF comparison
# ============================================================
print("\n" + "=" * 80)
print("B. PER-SAMPLE QUALITY COMPARISON")
print("=" * 80)

try:
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF

    merged = large_sent.merge(base_sent, on='input', suffixes=('_large', '_base'))
    print(f"Matched: {len(merged)} samples")

    bleu = BLEU(effective_order=True)
    chrf = CHRF(word_order=2)

    scores = []
    for _, row in merged.iterrows():
        ref = str(row['reference_large'])
        pred_l = str(row['prediction_clean_large'])
        pred_b = str(row['prediction_clean_base'])

        bl_l = bleu.sentence_score(pred_l, [ref]).score
        bl_b = bleu.sentence_score(pred_b, [ref]).score
        cf_l = chrf.sentence_score(pred_l, [ref]).score
        cf_b = chrf.sentence_score(pred_b, [ref]).score

        geo_l = np.sqrt(max(bl_l, 0) * max(cf_l, 0))
        geo_b = np.sqrt(max(bl_b, 0) * max(cf_b, 0))

        scores.append({
            'bleu_large': bl_l, 'bleu_base': bl_b,
            'chrf_large': cf_l, 'chrf_base': cf_b,
            'geo_large': geo_l, 'geo_base': geo_b,
            'geo_diff': geo_l - geo_b,
            'ref_len': len(ref),
            'pred_large_len': len(pred_l),
            'pred_base_len': len(pred_b),
        })

    scores_df = pd.DataFrame(scores)
    merged = pd.concat([merged.reset_index(drop=True), scores_df], axis=1)

    print(f"\nOverall sent-level scores:")
    print(f"  Large: BLEU={scores_df['bleu_large'].mean():.2f}, chrF++={scores_df['chrf_large'].mean():.2f}, "
          f"geo={scores_df['geo_large'].mean():.2f}")
    print(f"  Base:  BLEU={scores_df['bleu_base'].mean():.2f}, chrF++={scores_df['chrf_base'].mean():.2f}, "
          f"geo={scores_df['geo_base'].mean():.2f}")

    # Where is large much worse?
    large_worse = merged[merged['geo_diff'] < -10].sort_values('geo_diff')
    print(f"\n--- Cases where Large is MUCH WORSE (geo_diff < -10): {len(large_worse)} ---")
    for _, row in large_worse.head(10).iterrows():
        print(f"\n  geo_diff={row['geo_diff']:.1f} (L={row['geo_large']:.1f}, B={row['geo_base']:.1f})")
        print(f"  Input:  {str(row['input'])[:100]}...")
        print(f"  Ref:    {str(row['reference_large'])[:120]}...")
        print(f"  Large:  {str(row['prediction_clean_large'])[:120]}...")
        print(f"  Base:   {str(row['prediction_clean_base'])[:120]}...")

    # Where is large much better?
    large_better = merged[merged['geo_diff'] > 10].sort_values('geo_diff', ascending=False)
    print(f"\n--- Cases where Large is MUCH BETTER (geo_diff > 10): {len(large_better)} ---")
    for _, row in large_better.head(5).iterrows():
        print(f"\n  geo_diff={row['geo_diff']:.1f} (L={row['geo_large']:.1f}, B={row['geo_base']:.1f})")
        print(f"  Input:  {str(row['input'])[:100]}...")
        print(f"  Ref:    {str(row['reference_large'])[:120]}...")
        print(f"  Large:  {str(row['prediction_clean_large'])[:120]}...")
        print(f"  Base:   {str(row['prediction_clean_base'])[:120]}...")

    # Score distribution by reference length
    print("\n--- Score by reference length bucket ---")
    merged['ref_len_bucket'] = pd.cut(merged['ref_len'], bins=[0, 50, 100, 200, 500, 10000],
                                       labels=['<50', '50-100', '100-200', '200-500', '>500'])
    for bucket in ['<50', '50-100', '100-200', '200-500', '>500']:
        sub = merged[merged['ref_len_bucket'] == bucket]
        if len(sub) > 0:
            print(f"  {bucket:>10s} (n={len(sub):3d}): "
                  f"Large geo={sub['geo_large'].mean():.2f}, "
                  f"Base geo={sub['geo_base'].mean():.2f}, "
                  f"diff={sub['geo_diff'].mean():.2f}")

    # Score by input length
    merged['input_len'] = merged['input'].str.len()
    merged['input_len_bucket'] = pd.cut(merged['input_len'], bins=[0, 100, 200, 400, 800, 10000],
                                         labels=['<100', '100-200', '200-400', '400-800', '>800'])
    print("\n--- Score by INPUT length bucket ---")
    for bucket in ['<100', '100-200', '200-400', '400-800', '>800']:
        sub = merged[merged['input_len_bucket'] == bucket]
        if len(sub) > 0:
            print(f"  {bucket:>10s} (n={len(sub):3d}): "
                  f"Large geo={sub['geo_large'].mean():.2f}, "
                  f"Base geo={sub['geo_base'].mean():.2f}, "
                  f"diff={sub['geo_diff'].mean():.2f}")

except ImportError:
    print("sacrebleu not available, skipping per-sample scoring")

# ============================================================
# C. Hallucination / untranslated text detection
# ============================================================
print("\n" + "=" * 80)
print("C. HALLUCINATION / UNTRANSLATED TEXT DETECTION")
print("=" * 80)

def detect_akkadian_in_prediction(df, name=''):
    """Check if Akkadian transliteration leaks into English predictions"""
    preds = df['prediction_clean'].astype(str)

    # Common Akkadian syllabic patterns that shouldn't appear in English
    akkadian_patterns = [
        r'\b[a-z]+-[a-z]+-[a-z]+\b',  # Syllabic hyphenation (e.g., a-na-ku)
        r'[šṣṭḫ]',  # Akkadian-specific diacritics in non-name context
    ]

    # Check for untranslated Akkadian (hyphenated syllables)
    has_syllabic = 0
    syllabic_examples = []
    for idx, pred in preds.items():
        # Find sequences of 3+ hyphenated syllables (not proper names like Puzur-Aššur)
        matches = re.findall(r'\b[a-zšṣṭḫ]+-[a-zšṣṭḫ]+-[a-zšṣṭḫ]+(?:-[a-zšṣṭḫ]+)*\b', pred.lower())
        # Filter out common proper names
        non_name_matches = [m for m in matches if not any(n in m for n in ['puzur', 'ennam', 'aššur', 'ištar'])]
        if non_name_matches:
            has_syllabic += 1
            if len(syllabic_examples) < 5:
                syllabic_examples.append((idx, non_name_matches, pred[:150]))

    print(f"\n--- {name} ---")
    print(f"  Predictions with untranslated Akkadian syllables: {has_syllabic} ({100*has_syllabic/len(df):.1f}%)")
    for idx, matches, pred in syllabic_examples:
        print(f"    [{idx}] Matches: {matches[:3]}")
        print(f"         Text: {pred[:120]}...")

detect_akkadian_in_prediction(large_sent, "Large SENT")
detect_akkadian_in_prediction(base_sent, "Base036 SENT")

# ============================================================
# D. <gap> handling comparison
# ============================================================
print("\n" + "=" * 80)
print("D. <gap> HANDLING COMPARISON")
print("=" * 80)

merged = large_sent.merge(base_sent, on='input', suffixes=('_large', '_base'))

# Inputs with <gap>
has_gap_input = merged['input'].str.contains('<gap>', regex=False)
no_gap_input = ~has_gap_input

print(f"Inputs with <gap>: {has_gap_input.sum()}")
print(f"Inputs without <gap>: {no_gap_input.sum()}")

# Check how each model handles <gap> in output
for subset_name, mask in [("With <gap> input", has_gap_input), ("Without <gap> input", no_gap_input)]:
    sub = merged[mask]
    l_gap_out = sub['prediction_clean_large'].str.contains('<gap>', regex=False).sum()
    b_gap_out = sub['prediction_clean_base'].str.contains('<gap>', regex=False).sum()
    print(f"\n  {subset_name} ({len(sub)} samples):")
    print(f"    Large has <gap> in output: {l_gap_out} ({100*l_gap_out/len(sub):.1f}%)")
    print(f"    Base has <gap> in output:  {b_gap_out} ({100*b_gap_out/len(sub):.1f}%)")

# ============================================================
# E. Investigate if ByT5 large generates different style
# ============================================================
print("\n" + "=" * 80)
print("E. STYLE DIFFERENCE ANALYSIS")
print("=" * 80)

# Check punctuation patterns
for name, df in [("Large SENT", large_sent), ("Base036 SENT", base_sent)]:
    preds = df['prediction_clean'].astype(str)
    refs = df['reference'].astype(str)

    # Count semicolons (used in Akkadian translation style)
    pred_semicolons = preds.str.count(';').mean()
    ref_semicolons = refs.str.count(';').mean()

    # Count quotation marks
    pred_quotes = preds.str.count('"').mean()
    ref_quotes = refs.str.count('"').mean()

    # Count commas
    pred_commas = preds.str.count(',').mean()
    ref_commas = refs.str.count(',').mean()

    print(f"\n--- {name} ---")
    print(f"  Semicolons per pred: {pred_semicolons:.2f} (ref: {ref_semicolons:.2f})")
    print(f"  Quotes per pred: {pred_quotes:.2f} (ref: {ref_quotes:.2f})")
    print(f"  Commas per pred: {pred_commas:.2f} (ref: {ref_commas:.2f})")

# ============================================================
# F. Token-level analysis for ByT5 max_new_tokens issue
# ============================================================
print("\n" + "=" * 80)
print("F. MAX_NEW_TOKENS TRUNCATION CHECK (ByT5 byte-level)")
print("=" * 80)

# For ByT5, each character = ~1-4 bytes = 1-4 tokens
# Common max_new_tokens settings: 128, 256, 512
# If max_new_tokens=256 but prediction needs 500 bytes, it truncates!

for name, df in [("Large SENT", large_sent), ("Base036 SENT", base_sent)]:
    preds_raw = df['prediction_raw'].astype(str)
    pred_byte_lens = preds_raw.apply(lambda x: len(x.encode('utf-8')))

    print(f"\n--- {name} RAW prediction byte lengths ---")
    print(f"  mean={pred_byte_lens.mean():.1f}, max={pred_byte_lens.max()}, "
          f"std={pred_byte_lens.std():.1f}")

    # Check for clustering at specific byte lengths
    for limit in [128, 256, 384, 512, 768, 1024, 1536]:
        exact = (pred_byte_lens == limit).sum()
        near = ((pred_byte_lens >= limit - 3) & (pred_byte_lens <= limit + 3)).sum()
        if near > 2:
            print(f"  Near {limit} bytes (±3): {near} predictions")

    # Top 20 byte lengths
    top_lens = pred_byte_lens.nlargest(20)
    print(f"  Top 20 byte lengths: {sorted(top_lens.values, reverse=True)}")

# ============================================================
# G. Specifically check max_length used in generation
# ============================================================
print("\n" + "=" * 80)
print("G. MAX_LENGTH INVESTIGATION")
print("=" * 80)

# The max prediction_clean character length for Large is 500 for SENT, 511 for DOC
# For ByT5-large, one char can be 1-4 bytes/tokens
# Let's check the actual byte lengths at the max

for name, df in [("Large SENT", large_sent), ("Large DOC", pd.read_csv("s1_exp001_byt5_large_fold3_last_doc_predictions.csv"))]:
    preds = df['prediction_clean'].astype(str)
    byte_lens = preds.apply(lambda x: len(x.encode('utf-8')))
    char_lens = preds.str.len()

    # Top 10 longest by bytes
    top_idx = byte_lens.nlargest(10).index
    print(f"\n--- {name} - Top 10 longest predictions ---")
    for idx in top_idx:
        pred = str(df.loc[idx, 'prediction_clean'])
        ref = str(df.loc[idx, 'reference'])
        print(f"  [{idx}] chars={len(pred)}, bytes={len(pred.encode('utf-8'))}, "
              f"ref_chars={len(ref)}")
        print(f"       Ends with: ...{pred[-60:]}")
        ends_properly = bool(re.match(r'.*[.!?;"\')]$', pred))
        print(f"       Ends properly: {ends_properly}")

print("\n=== DEEP ANALYSIS COMPLETE ===")
