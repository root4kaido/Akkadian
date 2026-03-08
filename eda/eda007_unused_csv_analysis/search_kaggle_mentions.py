"""
EDA007 追加分析: Kaggleディスカッション・ノートブックでの未使用CSV関連情報の抽出
"""
import json
import os
import re

DISC_PATH = "/home/user/work/Akkadian/docs/survey/discussion/snapshot_20260308_details.json"
NB_PATH = "/home/user/work/Akkadian/docs/survey/notebooks/snapshot_20260308_details.json"
NB_DL_DIR = "/home/user/work/Akkadian/docs/survey/notebooks/downloaded"

# 検索キーワード（未使用CSV + 関連概念）
KEYWORDS = {
    'Sentences_Oare': ['sentences_oare', 'sentence_oare', 'firstword', 'first_word',
                       'sentence alignment', 'sentence-level', 'sentence level',
                       'sent align', 'line_number', 'sent-level data'],
    'OA_Lexicon': ['oa_lexicon', 'lexicon_ebl', 'oa lexicon', 'old assyrian lexicon',
                   'proper name', 'proper noun', 'name normalization', 'pn list',
                   'name correction', 'onomastic'],
    'eBL_Dictionary': ['ebl_dictionary', 'ebl dictionary', 'ebl_dict', 'ebl dict',
                       'dictionary lookup', 'word definition'],
    'published_texts': ['published_texts', 'published_text', 'published.csv',
                        'aicc_translation', 'aicc translation', 'machine translation data',
                        'additional transliteration', 'extra data', '6388', '7702'],
    'publications': ['publications.csv', 'ocr text', 'scholarly paper', 'has_akkadian'],
    'onomasticon': ['onomasticon', 'onomastic', 'name list', 'personal name list'],
    'supplemental': ['supplemental', 'supplementary', 'additional data', 'extra csv',
                     'unused data', 'other csv'],
}

print("=" * 70)
print("=== ディスカッションでの言及 ===")
print("=" * 70)

with open(DISC_PATH) as f:
    disc = json.load(f)

for d in disc['discussions']:
    title = d.get('title', '')
    body = d.get('body', '')
    full_text = (title + ' ' + body).lower()

    matches = {}
    for csv_name, kws in KEYWORDS.items():
        for kw in kws:
            if kw in full_text:
                if csv_name not in matches:
                    matches[csv_name] = []
                matches[csv_name].append(kw)

    if matches:
        print(f"\n{'='*60}")
        print(f"TOPIC: {title[:100]}")
        print(f"URL: {d.get('url', '')}")
        print(f"Comments: {d.get('comments', '')}")
        print(f"Matched: {matches}")
        print(f"{'='*60}")
        # 該当キーワード周辺のコンテキストを抽出
        for csv_name, kws in matches.items():
            for kw in kws:
                idx = full_text.find(kw)
                while idx >= 0:
                    start = max(0, idx - 150)
                    end = min(len(full_text), idx + len(kw) + 300)
                    context = full_text[start:end].replace('\n', ' ')
                    print(f"\n  [{csv_name}] keyword='{kw}' @{idx}:")
                    print(f"    ...{context}...")
                    idx = full_text.find(kw, idx + 1)
                    if idx > 0 and idx - start < 500:
                        break  # avoid too many repeats

print()
print("=" * 70)
print("=== ノートブックコードでの言及 ===")
print("=" * 70)

# ダウンロード済みノートブックを直接検索
for root, dirs, files in os.walk(NB_DL_DIR):
    for fname in files:
        if not fname.endswith('.ipynb'):
            continue
        fpath = os.path.join(root, fname)
        try:
            with open(fpath) as f:
                nb = json.load(f)
        except:
            continue

        for i, cell in enumerate(nb.get('cells', [])):
            src = ''.join(cell.get('source', []))
            src_lower = src.lower()

            for csv_name, kws in KEYWORDS.items():
                for kw in kws:
                    if kw in src_lower:
                        print(f"\n{'='*60}")
                        print(f"NB: {fname}")
                        print(f"CELL {i} ({cell.get('cell_type','?')})")
                        print(f"Matched: [{csv_name}] keyword='{kw}'")
                        print(f"{'='*60}")
                        # 該当部分を含む行を表示
                        lines = src.split('\n')
                        for j, line in enumerate(lines):
                            if kw in line.lower():
                                start = max(0, j - 2)
                                end = min(len(lines), j + 5)
                                for k in range(start, end):
                                    marker = ">>>" if k == j else "   "
                                    print(f"  {marker} L{k}: {lines[k][:200]}")
                                print()
                        break  # one match per keyword group per cell

print()
print("=" * 70)
print("=== 新規ディスカッション（前回3/6にはなかったもの） ===")
print("=" * 70)

# 前回のスナップショットと比較
OLD_DISC = "/home/user/work/Akkadian/docs/survey/discussion/snapshot_20260306_details.json"
if os.path.exists(OLD_DISC):
    with open(OLD_DISC) as f:
        old_disc = json.load(f)
    old_ids = set(d['id'] for d in old_disc.get('discussions', []))
    new_topics = [d for d in disc['discussions'] if d['id'] not in old_ids]
    print(f"\n新規トピック数: {len(new_topics)}")
    for d in new_topics:
        print(f"\n  TITLE: {d.get('title', '')[:100]}")
        print(f"  URL: {d.get('url', '')}")
        print(f"  Comments: {d.get('comments', '')}")
        # body抜粋
        body = d.get('body', '')
        # 主要コンテンツ部分を抽出
        content = body[:1500]
        print(f"  BODY (先頭1500文字):")
        print(f"    {content[:1500]}")
else:
    print("前回スナップショットなし")

print()
print("=" * 70)
print("=== 完了 ===")
print("=" * 70)
