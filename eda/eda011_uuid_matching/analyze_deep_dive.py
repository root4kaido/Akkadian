"""
EDA011 Deep Dive:
- Why do UUID-matched texts have translation mismatches?
- What's different about the 1303 unmatched train texts vs the 253 matched?
- More aggressive text matching with fuzzy methods
"""
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

train = pd.read_csv('datasets/raw/train.csv')
sentences = pd.read_csv('datasets/raw/Sentences_Oare_FirstWord_LinNum.csv')
published = pd.read_csv('datasets/raw/published_texts.csv')

def normalize_text(t):
    if pd.isna(t):
        return ""
    return str(t).strip().lower()

train_ids = set(train['oare_id'].astype(str).str.strip())
sent_text_ids = set(sentences['text_uuid'].astype(str).str.strip())
match_train_sent = train_ids & sent_text_ids

# ============================================================
# A. Detailed analysis of UUID matches with translation comparison
# ============================================================
print("="*80)
print("A. DETAILED TRANSLATION COMPARISON FOR UUID MATCHES")
print("="*80)

train_lookup = dict(zip(train['oare_id'].astype(str).str.strip(), train['translation']))

# For all 253 matches, compute similarity
results = []
for mid in match_train_sent:
    train_t = normalize_text(train_lookup.get(mid, ''))
    sent_group = sentences[sentences['text_uuid'].astype(str).str.strip() == mid].sort_values('sentence_obj_in_text')
    sent_t = ' '.join(str(s) for s in sent_group['translation'] if pd.notna(s)).strip().lower()
    n_sents = len(sent_group)

    if not train_t or not sent_t:
        sim = 0.0
    else:
        # Quick similarity
        sim = SequenceMatcher(None, train_t[:500], sent_t[:500]).ratio()

    results.append({
        'text_uuid': mid,
        'n_sentences': n_sents,
        'train_len': len(train_t),
        'sent_len': len(sent_t),
        'similarity': sim,
        'train_start': train_t[:80],
        'sent_start': sent_t[:80],
    })

df_res = pd.DataFrame(results)
print(f"\nSimilarity distribution for {len(df_res)} UUID matches:")
print(f"  Mean: {df_res['similarity'].mean():.3f}")
print(f"  Median: {df_res['similarity'].median():.3f}")
print(f"  Min: {df_res['similarity'].min():.3f}")
print(f"  Max: {df_res['similarity'].max():.3f}")

# Buckets
for threshold in [0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]:
    count = (df_res['similarity'] >= threshold).sum()
    print(f"  >= {threshold}: {count}")

# Look at the mismatches
print("\n--- LOW similarity examples (sim < 0.3) ---")
low_sim = df_res[df_res['similarity'] < 0.3].sort_values('similarity')
for _, row in low_sim.head(5).iterrows():
    print(f"\n  UUID: {row['text_uuid']}")
    print(f"  Similarity: {row['similarity']:.3f}, n_sents: {row['n_sentences']}")
    print(f"  Train: {row['train_start']}...")
    print(f"  Sent:  {row['sent_start']}...")

# Check: are low-similarity ones mostly single-sentence (truncated)?
print(f"\n--- Sentence count vs similarity ---")
for n in sorted(df_res['n_sentences'].unique()):
    subset = df_res[df_res['n_sentences'] == n]
    print(f"  n_sentences={n}: count={len(subset)}, mean_sim={subset['similarity'].mean():.3f}")

# ============================================================
# B. What's in Sentences_Oare but NOT in train?
# ============================================================
print("\n" + "="*80)
print("B. SENTENCES_OARE TEXTS NOT IN TRAIN")
print("="*80)

sent_not_in_train = sent_text_ids - train_ids
print(f"Sentences text_uuids NOT in train: {len(sent_not_in_train)}")
print(f"Sentences text_uuids IN train: {len(match_train_sent)}")

# Check if these are in published_texts
pub_ids = set(published['oare_id'].astype(str).str.strip())
sent_not_in_train_in_pub = sent_not_in_train & pub_ids
print(f"  Of {len(sent_not_in_train)} not-in-train: {len(sent_not_in_train_in_pub)} in published_texts, {len(sent_not_in_train - pub_ids)} not in published_texts")

# ============================================================
# C. More aggressive fuzzy matching
# ============================================================
print("\n" + "="*80)
print("C. AGGRESSIVE FUZZY TEXT MATCHING")
print("="*80)

# Build sentence concat translations
sent_concat = sentences.groupby('text_uuid').agg({
    'translation': lambda x: ' '.join(str(s) for s in x if pd.notna(s)),
    'display_name': 'first',
    'sentence_uuid': 'count'
}).reset_index()
sent_concat.columns = ['text_uuid', 'concat_translation', 'display_name', 'n_sentences']
sent_concat['norm_trans'] = sent_concat['concat_translation'].apply(normalize_text)

train_trans = {}
for _, row in train.iterrows():
    t = normalize_text(row['translation'])
    if t and len(t) > 10:
        train_trans[row['oare_id']] = t

# Only try to match sentence texts NOT already UUID-matched
unmatched_sent_texts = sent_concat[~sent_concat['text_uuid'].isin(match_train_sent)]
unmatched_train_ids = train_ids - match_train_sent

print(f"Unmatched sentence texts to try: {len(unmatched_sent_texts)}")
print(f"Unmatched train IDs to try: {len(unmatched_train_ids)}")

# Strategy 1: First 30 chars match
matches_30 = {}
for _, srow in unmatched_sent_texts.iterrows():
    s_trans = srow['norm_trans']
    if not s_trans or len(s_trans) < 15:
        continue
    s_prefix = s_trans[:30]
    for tid in unmatched_train_ids:
        t_trans = train_trans.get(tid, '')
        if t_trans and t_trans[:30] == s_prefix:
            matches_30[srow['text_uuid']] = tid
            break

print(f"\nStrategy 1 (first 30 chars exact): {len(matches_30)} new matches")

# Strategy 2: First individual sentence match
matches_sent1 = {}
first_sents = sentences.sort_values('sentence_obj_in_text').groupby('text_uuid').first().reset_index()
first_sents_unmatched = first_sents[~first_sents['text_uuid'].isin(match_train_sent)]

for _, srow in first_sents_unmatched.iterrows():
    s_trans = normalize_text(srow['translation'])
    if not s_trans or len(s_trans) < 20:
        continue
    s_prefix = s_trans[:30]
    for tid in unmatched_train_ids:
        t_trans = train_trans.get(tid, '')
        if t_trans and s_prefix in t_trans[:60]:
            matches_sent1[srow['text_uuid']] = tid
            break

print(f"Strategy 2 (first sentence prefix in train): {len(matches_sent1)} new matches")

# Strategy 3: Use published_texts as bridge
# published_texts has both oare_id and transliteration
# If text_uuid in sentences matches oare_id in published, and the published transliteration matches train transliteration
print("\n--- Strategy 3: published_texts as bridge ---")

# Build train transliteration lookup
train_translit = dict(zip(
    train['oare_id'].astype(str).str.strip(),
    train['transliteration'].astype(str).str.strip().str.lower()
))

pub_translit = {}
for _, row in published.iterrows():
    oid = str(row['oare_id']).strip()
    tl = str(row.get('transliteration', '')).strip().lower()
    pub_translit[oid] = tl

# For sentence text_uuids NOT in train, find their published_texts entry
# Then see if the published transliteration matches any train transliteration
bridge_matches = {}
for text_uuid in sent_not_in_train:
    if text_uuid in pub_ids:
        pub_tl = pub_translit.get(text_uuid, '')
        if not pub_tl or pub_tl == 'nan' or len(pub_tl) < 20:
            continue
        pub_prefix = pub_tl[:50]
        for tid in unmatched_train_ids:
            train_tl = train_translit.get(tid, '')
            if train_tl and train_tl[:50] == pub_prefix:
                bridge_matches[text_uuid] = tid
                break

print(f"Bridge matches (via published transliteration): {len(bridge_matches)}")

# Show examples
for i, (tuuid, tid) in enumerate(list(bridge_matches.items())[:3]):
    print(f"\n  [{i}] text_uuid={tuuid} -> train_id={tid}")
    t_trans = train_trans.get(tid, '')[:80]
    s_group = sentences[sentences['text_uuid'] == tuuid]
    s_trans = ' '.join(str(s) for s in s_group['translation'] if pd.notna(s))[:80]
    print(f"      Train trans: {t_trans}")
    print(f"      Sent trans:  {s_trans}")

# ============================================================
# D. Summary of all matching approaches
# ============================================================
print("\n" + "="*80)
print("D. FINAL SUMMARY")
print("="*80)

all_matched_train = match_train_sent.copy()
all_uuid_mapping = {}  # text_uuid -> train_oare_id (for non-direct matches)

for tuuid, tid in {**matches_30, **matches_sent1, **bridge_matches}.items():
    all_matched_train.add(tid)
    all_uuid_mapping[tuuid] = tid

print(f"Direct UUID matches: {len(match_train_sent)}")
print(f"Content matching (prefix 30): +{len(matches_30)}")
print(f"Content matching (first sentence): +{len(matches_sent1)}")
print(f"Bridge via published_texts transliteration: +{len(bridge_matches)}")
print(f"Total unique train docs matchable: {len(all_matched_train)}")
print(f"Total train docs: {len(train_ids)}")
print(f"Coverage: {len(all_matched_train)/len(train_ids)*100:.1f}%")
print(f"Still unmatched: {len(train_ids) - len(all_matched_train)}")

# The core issue
print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)
print(f"""
Key findings:
1. UUID format is identical (all valid UUID v4, 36 chars) across all files
2. train.csv oare_ids are 100% in published_texts.csv (1561/1561)
3. Sentences_Oare text_uuids are 83.4% in published_texts (1417/1700)
4. But only 253 train oare_ids directly appear in Sentences_Oare text_uuids
5. The 1303 unmatched train docs ARE in published_texts but NOT in Sentences_Oare

ROOT CAUSE: Sentences_Oare_FirstWord_LinNum.csv only covers {len(sent_text_ids)} unique
texts out of the {len(pub_ids)} in published_texts. The train set draws from the full
published_texts pool, but Sentences_Oare only covers a subset.

Of the 1700 Sentences_Oare texts and 1561 train texts, the overlap is only 253
because these are largely DIFFERENT subsets of published_texts.

Even for the 253 UUID matches:
- Only 40/253 have exact translation match (first 100 chars)
- Many show truncation or paraphrasing differences
- Suggests different annotation/transcription pipelines

Content-based matching adds very few extra matches (~5), confirming that the
non-overlap is NOT due to ID mismatch but genuinely different document sets.

IMPLICATION: Sentences_Oare can only provide sentence-level splitting for ~16%
of training data. For the remaining 84%, we need alternative sentence-splitting
strategies (rule-based, punctuation-based, etc.).
""")

# ============================================================
# E. Detailed stats about the sentences data
# ============================================================
print("="*80)
print("E. WHAT DOES SENTENCES_OARE COVER?")
print("="*80)

# How many sentences per text?
sent_per_text = sentences.groupby('text_uuid').size()
print(f"\nSentences per text distribution:")
print(f"  Mean: {sent_per_text.mean():.1f}")
print(f"  Median: {sent_per_text.median():.1f}")
print(f"  Min: {sent_per_text.min()}")
print(f"  Max: {sent_per_text.max()}")
print(f"  1 sentence: {(sent_per_text == 1).sum()}")
print(f"  2-5 sentences: {((sent_per_text >= 2) & (sent_per_text <= 5)).sum()}")
print(f"  6-10 sentences: {((sent_per_text >= 6) & (sent_per_text <= 10)).sum()}")
print(f"  >10 sentences: {(sent_per_text > 10).sum()}")

# For the 253 matched: sentence count distribution
matched_sent_count = sent_per_text[sent_per_text.index.isin(match_train_sent)]
print(f"\nFor 253 UUID-matched texts:")
print(f"  Mean sentences: {matched_sent_count.mean():.1f}")
print(f"  Median: {matched_sent_count.median():.1f}")
print(f"  1 sentence (no splitting benefit): {(matched_sent_count == 1).sum()}")
print(f"  2+ sentences (splitting possible): {(matched_sent_count >= 2).sum()}")

# Quality check: for 2+ sentence matches with high similarity, these are usable
high_quality = df_res[(df_res['similarity'] >= 0.5) & (df_res['n_sentences'] >= 2)]
print(f"\nHigh-quality matches (sim>=0.5 AND 2+ sentences): {len(high_quality)}")

print("\nDone!")
