"""
EDA011: Investigate UUID matching between train.csv and Sentences_Oare_FirstWord_LinNum.csv
Why only 253/1561 train documents match?
"""
import pandas as pd
import numpy as np
from collections import Counter

# ============================================================
# 0. Load data
# ============================================================
train = pd.read_csv('datasets/raw/train.csv')
sentences = pd.read_csv('datasets/raw/Sentences_Oare_FirstWord_LinNum.csv')
published = pd.read_csv('datasets/raw/published_texts.csv')

print("="*80)
print("DATASET SHAPES")
print("="*80)
print(f"train.csv: {train.shape}")
print(f"Sentences_Oare: {sentences.shape}")
print(f"published_texts: {published.shape}")

print(f"\ntrain columns: {list(train.columns)}")
print(f"Sentences_Oare columns: {list(sentences.columns)}")
print(f"published_texts columns: {list(published.columns)}")

# ============================================================
# 1. UUID format comparison - 5 examples from each
# ============================================================
print("\n" + "="*80)
print("1. UUID FORMAT COMPARISON")
print("="*80)

print("\n--- train.csv oare_id (first 5) ---")
for i, uid in enumerate(train['oare_id'].head(5)):
    print(f"  [{i}] {uid}  (len={len(str(uid))})")

print("\n--- Sentences_Oare text_uuid (first 5 unique) ---")
for i, uid in enumerate(sentences['text_uuid'].unique()[:5]):
    print(f"  [{i}] {uid}  (len={len(str(uid))})")

print("\n--- Sentences_Oare sentence_uuid (first 5) ---")
for i, uid in enumerate(sentences['sentence_uuid'].head(5)):
    print(f"  [{i}] {uid}  (len={len(str(uid))})")

print("\n--- published_texts oare_id (first 5) ---")
for i, uid in enumerate(published['oare_id'].head(5)):
    print(f"  [{i}] {uid}  (len={len(str(uid))})")

# Check UUID format consistency
import re
uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

def check_uuid_format(series, name):
    valid = series.astype(str).apply(lambda x: bool(uuid_pattern.match(x.strip())))
    print(f"\n{name}: {valid.sum()}/{len(series)} valid UUIDs ({valid.mean()*100:.1f}%)")
    if not valid.all():
        print(f"  Invalid examples: {series[~valid].head(3).tolist()}")

check_uuid_format(train['oare_id'], "train.oare_id")
check_uuid_format(sentences['text_uuid'], "sentences.text_uuid")
check_uuid_format(sentences['sentence_uuid'], "sentences.sentence_uuid")
check_uuid_format(published['oare_id'], "published.oare_id")

# ============================================================
# 2. train.csv oare_ids vs published_texts.csv oare_ids
# ============================================================
print("\n" + "="*80)
print("2. TRAIN oare_id vs PUBLISHED_TEXTS oare_id")
print("="*80)

train_ids = set(train['oare_id'].astype(str).str.strip())
pub_ids = set(published['oare_id'].astype(str).str.strip())

match_train_pub = train_ids & pub_ids
print(f"train unique oare_ids: {len(train_ids)}")
print(f"published unique oare_ids: {len(pub_ids)}")
print(f"Intersection: {len(match_train_pub)}")
print(f"Match rate (train in published): {len(match_train_pub)/len(train_ids)*100:.1f}%")
print(f"Train IDs NOT in published: {len(train_ids - pub_ids)}")

# ============================================================
# 3. Sentences_Oare text_uuids vs published_texts.csv oare_ids
# ============================================================
print("\n" + "="*80)
print("3. SENTENCES_OARE text_uuid vs PUBLISHED_TEXTS oare_id")
print("="*80)

sent_text_ids = set(sentences['text_uuid'].astype(str).str.strip())
match_sent_pub = sent_text_ids & pub_ids
print(f"Sentences unique text_uuids: {len(sent_text_ids)}")
print(f"published unique oare_ids: {len(pub_ids)}")
print(f"Intersection: {len(match_sent_pub)}")
print(f"Match rate (sentences text_uuid in published): {len(match_sent_pub)/len(sent_text_ids)*100:.1f}%")

# Direct: train oare_id vs sentences text_uuid
match_train_sent = train_ids & sent_text_ids
print(f"\nDirect: train oare_id ∩ sentences text_uuid: {len(match_train_sent)}")
print(f"  -> This is the '253' number we're investigating")

# ============================================================
# 4. Check ALL columns in Sentences_Oare for potential matches to train oare_ids
# ============================================================
print("\n" + "="*80)
print("4. CHECK ALL SENTENCES_OARE COLUMNS FOR MATCHES TO TRAIN oare_ids")
print("="*80)

for col in sentences.columns:
    sent_vals = set(sentences[col].dropna().astype(str).str.strip())
    matches = train_ids & sent_vals
    if len(matches) > 0:
        print(f"  {col}: {len(matches)} matches to train oare_ids")
    else:
        print(f"  {col}: 0 matches")

# Also check sentence_uuid
sent_sentence_ids = set(sentences['sentence_uuid'].astype(str).str.strip())
match_train_sentuuid = train_ids & sent_sentence_ids
print(f"\nExplicit check: train oare_id ∩ sentences sentence_uuid: {len(match_train_sentuuid)}")

# ============================================================
# 5. Alternative matching strategies
# ============================================================
print("\n" + "="*80)
print("5. ALTERNATIVE MATCHING: display_name and translation text")
print("="*80)

# 5a. Check if published_texts has a column that links to Sentences text_uuid
print("\n--- published_texts columns with UUID-like values ---")
for col in published.columns:
    sample = published[col].dropna().astype(str).head(3).tolist()
    if any(uuid_pattern.match(str(s).strip()) for s in sample):
        pub_col_vals = set(published[col].dropna().astype(str).str.strip())
        match_to_sent = pub_col_vals & sent_text_ids
        match_to_train = pub_col_vals & train_ids
        print(f"  {col}: {len(match_to_sent)} match sentences text_uuid, {len(match_to_train)} match train oare_id")

# 5b. Translation text matching
print("\n--- Translation text matching ---")
# Normalize translations for comparison
def normalize_text(t):
    if pd.isna(t):
        return ""
    return str(t).strip().lower()

train_trans = train[['oare_id', 'translation']].copy()
train_trans['norm_trans'] = train_trans['translation'].apply(normalize_text)

sent_trans = sentences[['text_uuid', 'sentence_uuid', 'translation', 'display_name']].copy()
sent_trans['norm_trans'] = sent_trans['translation'].apply(normalize_text)

# For each train translation, check if it appears as a substring or exact match in any sentence translation
# First, build a map of sentence translations grouped by text_uuid
sent_by_text = sentences.groupby('text_uuid')

# Build concatenated translations per text_uuid from sentences
sent_concat = sentences.groupby('text_uuid').agg({
    'translation': lambda x: ' '.join(str(s) for s in x if pd.notna(s)),
    'display_name': 'first'
}).reset_index()
sent_concat['norm_concat_trans'] = sent_concat['translation'].apply(normalize_text)

# Build lookup: train translation -> check in sentences
# Strategy: check if the FIRST sentence of a text matches the beginning of the train translation
print(f"Unique text_uuids in sentences: {len(sent_concat)}")

# 5c. Match via first sentence text content
# For each train document, see if the train translation STARTS WITH any sentence's translation
train_trans_set = {}
for _, row in train_trans.iterrows():
    t = row['norm_trans']
    if t:
        train_trans_set[row['oare_id']] = t

# For sentences concatenated per text, check overlap with train
text_match_count = 0
text_matches = []
for _, srow in sent_concat.iterrows():
    s_trans = srow['norm_concat_trans']
    if not s_trans or s_trans == 'nan':
        continue
    for tid, t_trans in train_trans_set.items():
        # Check if substantial overlap exists
        # Use first 50 chars as a quick fingerprint
        if len(s_trans) > 20 and len(t_trans) > 20:
            if s_trans[:50] == t_trans[:50] or t_trans[:50] == s_trans[:50]:
                text_matches.append({
                    'train_oare_id': tid,
                    'sent_text_uuid': srow['text_uuid'],
                    'display_name': srow['display_name'],
                    'train_trans_start': t_trans[:80],
                    'sent_trans_start': s_trans[:80],
                })
                text_match_count += 1
                break  # one match per sentence group

print(f"\nMatches by concatenated translation text (first 50 chars): {text_match_count}")

# Also try: for each train doc, check if ANY individual sentence's translation is contained in it
print("\n--- Individual sentence translation containment ---")
# Build a set of train translations for fast lookup
sent_individual_matches = 0
matched_train_ids_via_text = set()
matched_text_uuids_via_text = set()

# Group sentences by text_uuid and check if ALL sentences' translations are in a train translation
for text_uuid, group in sent_by_text:
    sent_translations = [normalize_text(t) for t in group['translation'] if pd.notna(t) and normalize_text(t)]
    if not sent_translations:
        continue
    # Check first sentence against all train translations
    first_sent = sent_translations[0]
    if len(first_sent) < 10:
        continue
    for tid, t_trans in train_trans_set.items():
        if first_sent[:40] in t_trans:
            matched_train_ids_via_text.add(tid)
            matched_text_uuids_via_text.add(text_uuid)
            break

print(f"Train docs matchable via sentence text containment: {len(matched_train_ids_via_text)}")
print(f"Sentence text_uuids matched: {len(matched_text_uuids_via_text)}")

# ============================================================
# 6. Verify the 253 UUID matches
# ============================================================
print("\n" + "="*80)
print("6. VERIFY THE 253 UUID MATCHES - COMPARE TRANSLATIONS")
print("="*80)

matched_ids = list(match_train_sent)[:20]  # Check first 20

train_lookup = dict(zip(train['oare_id'].astype(str).str.strip(), train['translation']))

mismatches = 0
verified = 0
for mid in matched_ids:
    train_t = normalize_text(train_lookup.get(mid, ''))
    # Get concatenated sentence translation
    sent_group = sentences[sentences['text_uuid'].astype(str).str.strip() == mid]
    sent_t = ' '.join(str(s) for s in sent_group['translation'] if pd.notna(s)).strip().lower()

    if not train_t or not sent_t:
        continue

    # Check similarity
    overlap = train_t[:100] == sent_t[:100]
    if overlap:
        verified += 1
    else:
        mismatches += 1
        if mismatches <= 3:
            print(f"\n  MISMATCH for {mid}:")
            print(f"    Train: {train_t[:120]}...")
            print(f"    Sent:  {sent_t[:120]}...")

print(f"\nVerified (first 100 chars match): {verified}/{len(matched_ids)}")
print(f"Mismatches: {mismatches}/{len(matched_ids)}")

# Full verification on all 253
all_verified = 0
all_mismatches = 0
all_close = 0
for mid in match_train_sent:
    train_t = normalize_text(train_lookup.get(mid, ''))
    sent_group = sentences[sentences['text_uuid'].astype(str).str.strip() == mid]
    sent_t = ' '.join(str(s) for s in sent_group['translation'] if pd.notna(s)).strip().lower()

    if not train_t or not sent_t:
        continue

    if train_t[:100] == sent_t[:100]:
        all_verified += 1
    elif train_t[:30] == sent_t[:30]:
        all_close += 1
    else:
        all_mismatches += 1

print(f"\nFull verification (all {len(match_train_sent)} matches):")
print(f"  Exact (first 100): {all_verified}")
print(f"  Close (first 30):  {all_close}")
print(f"  Mismatch:          {all_mismatches}")

# ============================================================
# 7. Count potential matches via text content (not UUID)
# ============================================================
print("\n" + "="*80)
print("7. POTENTIAL MATCHES VIA TEXT CONTENT")
print("="*80)

# More thorough text matching using sentence-level containment
# For each unique text_uuid in sentences, concatenate translations and match to train
from difflib import SequenceMatcher

# Build index: first 40 chars of translation -> train oare_ids
train_prefix_index = {}
for tid, t_trans in train_trans_set.items():
    prefix = t_trans[:40]
    if prefix not in train_prefix_index:
        train_prefix_index[prefix] = []
    train_prefix_index[prefix].append(tid)

# Also build index of all train translations for substring matching
# Use first sentence of each text group
matched_by_content = {}  # text_uuid -> train_oare_id
already_uuid_matched = set()

for _, srow in sent_concat.iterrows():
    text_uuid = srow['text_uuid']
    s_trans = srow['norm_concat_trans']
    if not s_trans or s_trans == 'nan' or len(s_trans) < 10:
        continue

    # Try prefix match
    s_prefix = s_trans[:40]
    if s_prefix in train_prefix_index:
        matched_by_content[text_uuid] = train_prefix_index[s_prefix][0]
        if text_uuid in match_train_sent:
            already_uuid_matched.add(text_uuid)
        continue

    # Try matching with individual sentence translations
    first_sents = sentences[sentences['text_uuid'] == text_uuid].sort_values('sentence_obj_in_text')
    if len(first_sents) == 0:
        continue
    first_sent_trans = normalize_text(first_sents.iloc[0]['translation'])
    if len(first_sent_trans) < 15:
        continue

    for tid, t_trans in train_trans_set.items():
        if first_sent_trans[:40] in t_trans[:80]:
            matched_by_content[text_uuid] = tid
            if text_uuid in match_train_sent:
                already_uuid_matched.add(text_uuid)
            break

print(f"Total matches by content: {len(matched_by_content)}")
print(f"  Of which already UUID-matched: {len(already_uuid_matched)}")
print(f"  NEW matches (not UUID-matched): {len(matched_by_content) - len(already_uuid_matched)}")
print(f"  Total train docs matchable: {len(set(matched_by_content.values()))}")

# Show some NEW matches
new_matches = {k: v for k, v in matched_by_content.items() if k not in match_train_sent}
print(f"\n--- Sample NEW content-based matches ---")
for i, (text_uuid, train_id) in enumerate(list(new_matches.items())[:5]):
    train_t = normalize_text(train_lookup.get(train_id, ''))
    sent_group = sentences[sentences['text_uuid'] == text_uuid]
    sent_t = ' '.join(str(s) for s in sent_group['translation'] if pd.notna(s)).strip().lower()
    disp = sent_group.iloc[0]['display_name'] if len(sent_group) > 0 else 'N/A'
    print(f"\n  [{i}] text_uuid={text_uuid}, train_id={train_id}")
    print(f"      display_name: {disp}")
    print(f"      Train:  {train_t[:100]}...")
    print(f"      Sent:   {sent_t[:100]}...")

# ============================================================
# Summary stats
# ============================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Train documents total: {len(train_ids)}")
print(f"Sentences_Oare unique texts: {len(sent_text_ids)}")
print(f"Direct UUID match (train ∩ sentences): {len(match_train_sent)} ({len(match_train_sent)/len(train_ids)*100:.1f}%)")
print(f"Content-based matches: {len(matched_by_content)}")
print(f"  - Already UUID-matched: {len(already_uuid_matched)}")
print(f"  - New content matches: {len(matched_by_content) - len(already_uuid_matched)}")
total_matchable = len(match_train_sent | set(matched_by_content.values()))
print(f"Total train docs matchable (UUID + content): {total_matchable} ({total_matchable/len(train_ids)*100:.1f}%)")
print(f"Remaining unmatched train docs: {len(train_ids) - total_matchable}")

# Why don't more match?
print(f"\n--- Analysis of unmatched ---")
unmatched_train = train_ids - match_train_sent - set(matched_by_content.values())
print(f"Unmatched train IDs: {len(unmatched_train)}")

# Check if unmatched train IDs are in published_texts but not in sentences
unmatched_in_pub = unmatched_train & pub_ids
print(f"  In published_texts but not in sentences: {len(unmatched_in_pub)}")
print(f"  Not in published_texts either: {len(unmatched_train - pub_ids)}")

# Check sentence counts for matched vs unmatched
matched_train_list = list(match_train_sent)[:10]
print(f"\n--- Sentence counts for UUID-matched texts ---")
for mid in matched_train_list[:5]:
    cnt = len(sentences[sentences['text_uuid'].astype(str).str.strip() == mid])
    print(f"  {mid}: {cnt} sentences")

print("\nDone!")
