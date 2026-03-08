"""
eda010: BPEトークン分析
Gutherz 2023ではBPE 1K/10KでBLEU4=37.47を達成。
ByT5のバイトレベルトークナイゼーションとの比較分析。
アッカド語翻字の音節構造とトークナイゼーションの相性を検討。
"""

import pandas as pd
import re
from collections import Counter
import os
import json

OUT_DIR = "eda/eda010_bpe_analysis"
os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)

train = pd.read_csv("datasets/raw/train.csv")
test = pd.read_csv("datasets/raw/test.csv")

# =====================================================
# 1. アッカド語翻字の音節構造分析
# =====================================================
print("=" * 60)
print("1. アッカド語翻字の音節構造分析")
print("=" * 60)

# 翻字はハイフン区切りの音節で構成される
# 例: "a-na a-lá-ḫi-im" → 音節: a, na, a, lá, ḫi, im
all_syllables = []
all_words = []
syllable_per_word = []

for text in train["transliteration"]:
    words = str(text).strip().split()
    for w in words:
        # ロゴグラム（大文字のみ）はスキップ
        if w == w.upper() and len(w) > 1:
            all_words.append(("logogram", w))
            continue
        # <gap>等はスキップ
        if w.startswith("<"):
            continue
        # ハイフンで分割
        syllables = w.split("-")
        syllable_per_word.append(len(syllables))
        all_words.append(("syllabic", w))
        for s in syllables:
            s = re.sub(r'[{}()]', '', s).strip()
            if s:
                all_syllables.append(s.lower())

print(f"総ワード数: {len(all_words)}")
print(f"  音節語: {sum(1 for t, _ in all_words if t == 'syllabic')}")
print(f"  ロゴグラム: {sum(1 for t, _ in all_words if t == 'logogram')}")
print(f"総音節数: {len(all_syllables)}")

# 音節数/ワードの分布
import statistics
print(f"\n音節数/ワード:")
print(f"  Mean:   {statistics.mean(syllable_per_word):.2f}")
print(f"  Median: {statistics.median(syllable_per_word):.1f}")
print(f"  Max:    {max(syllable_per_word)}")

# 音節の長さ分布
syl_lengths = [len(s) for s in all_syllables]
print(f"\n音節の長さ（文字数）:")
print(f"  Mean:   {statistics.mean(syl_lengths):.2f}")
print(f"  Median: {statistics.median(syl_lengths):.1f}")
print(f"  Max:    {max(syl_lengths)}")

syl_len_dist = Counter(syl_lengths)
print(f"\n音節長の分布:")
for l in sorted(syl_len_dist.keys()):
    print(f"  {l}文字: {syl_len_dist[l]:6d} ({syl_len_dist[l]/len(all_syllables)*100:.1f}%)")

# 頻出音節
syl_counter = Counter(all_syllables)
print(f"\nユニーク音節数: {len(syl_counter)}")
print(f"\n--- 頻出音節 (top 30) ---")
for syl, count in syl_counter.most_common(30):
    print(f"  {syl:15s} {count:5d}")

# =====================================================
# 2. ByT5バイトレベルの分析
# =====================================================
print("\n" + "=" * 60)
print("2. ByT5バイトレベルでのトークン数")
print("=" * 60)

# ByT5は各バイトが1トークン（+ 特殊トークン）
byte_lengths_src = []
byte_lengths_tgt = []

for text in train["transliteration"]:
    byte_lengths_src.append(len(str(text).encode("utf-8")))
for text in train["translation"]:
    byte_lengths_tgt.append(len(str(text).encode("utf-8")))

print(f"翻字（ソース）バイト数:")
print(f"  Mean:   {statistics.mean(byte_lengths_src):.0f}")
print(f"  Median: {statistics.median(byte_lengths_src):.0f}")
print(f"  Max:    {max(byte_lengths_src)}")
print(f"  >512:   {sum(1 for b in byte_lengths_src if b > 512)} ({sum(1 for b in byte_lengths_src if b > 512)/len(byte_lengths_src)*100:.1f}%)")

print(f"\n翻訳（ターゲット）バイト数:")
print(f"  Mean:   {statistics.mean(byte_lengths_tgt):.0f}")
print(f"  Median: {statistics.median(byte_lengths_tgt):.0f}")
print(f"  Max:    {max(byte_lengths_tgt)}")
print(f"  >512:   {sum(1 for b in byte_lengths_tgt if b > 512)} ({sum(1 for b in byte_lengths_tgt if b > 512)/len(byte_lengths_tgt)*100:.1f}%)")

# ByT5でのバイト効率（1バイト = 1トークン）
# アッカド語翻字のバイト効率
akk_bytes_per_char = []
for text in train["transliteration"]:
    t = str(text)
    akk_bytes_per_char.append(len(t.encode("utf-8")) / max(len(t), 1))

eng_bytes_per_char = []
for text in train["translation"]:
    t = str(text)
    eng_bytes_per_char.append(len(t.encode("utf-8")) / max(len(t), 1))

print(f"\nバイト/文字 効率:")
print(f"  アッカド語翻字: {statistics.mean(akk_bytes_per_char):.3f} byte/char")
print(f"  英語翻訳:       {statistics.mean(eng_bytes_per_char):.3f} byte/char")
print(f"  → アッカド語はdiacriticsのため1.2-1.4倍のバイトを消費")

# =====================================================
# 3. SentencePiece BPEシミュレーション
# =====================================================
print("\n" + "=" * 60)
print("3. SentencePiece BPEシミュレーション")
print("=" * 60)

try:
    import sentencepiece as spm
    HAS_SPM = True
except ImportError:
    HAS_SPM = False
    print("sentencepieceがインストールされていません。pip installが必要です。")

if HAS_SPM:
    # 翻字テキストをファイルに書き出し
    with open(f"{OUT_DIR}/akk_texts.txt", "w") as f:
        for text in train["transliteration"]:
            f.write(str(text).strip() + "\n")

    with open(f"{OUT_DIR}/eng_texts.txt", "w") as f:
        for text in train["translation"]:
            f.write(str(text).strip() + "\n")

    # vocab_size=1000でBPE学習（Gutherz 2023の設定）
    for name, vocab_size in [("akk_500", 500), ("akk_1k", 1000)]:
        spm.SentencePieceTrainer.train(
            input=f"{OUT_DIR}/akk_texts.txt",
            model_prefix=f"{OUT_DIR}/{name}",
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9999,
            pad_id=3,
        )
        sp = spm.SentencePieceProcessor()
        sp.load(f"{OUT_DIR}/{name}.model")

        # トークン数を計測
        bpe_lengths = []
        for text in train["transliteration"]:
            tokens = sp.encode(str(text), out_type=str)
            bpe_lengths.append(len(tokens))

        print(f"\n--- BPE vocab={vocab_size} (アッカド語翻字) ---")
        print(f"  トークン数 Mean:   {statistics.mean(bpe_lengths):.0f}")
        print(f"  トークン数 Median: {statistics.median(bpe_lengths):.0f}")
        print(f"  トークン数 Max:    {max(bpe_lengths)}")
        print(f"  圧縮率 (vs バイト): {statistics.mean(byte_lengths_src)/statistics.mean(bpe_lengths):.2f}x")

        # サンプルトークナイゼーション
        print(f"\n  サンプルトークナイゼーション:")
        for i in range(3):
            text = str(train.iloc[i]["transliteration"])[:100]
            tokens = sp.encode(text, out_type=str)
            print(f"    Original: {text}")
            print(f"    BPE({vocab_size}): {' | '.join(tokens[:20])}")

        # 音節境界との一致率を推定
        # BPEのトークン境界がハイフン位置と一致するか
        boundary_match = 0
        total_boundaries = 0
        for text in train["transliteration"][:200]:
            text = str(text)
            tokens = sp.encode(text, out_type=str)
            # 各トークンがハイフンで始まるか終わるかをチェック
            for t in tokens:
                t_stripped = t.replace("▁", "")
                if t_stripped.startswith("-") or t_stripped.endswith("-"):
                    total_boundaries += 1
                    boundary_match += 1
                elif "-" in t_stripped:
                    # トークン内にハイフン → 音節境界を跨いでいる
                    total_boundaries += t_stripped.count("-")

        print(f"\n  音節境界情報:")
        print(f"    BPEトークンがハイフンを含む（音節を跨ぐ）ケースの分析は上記サンプルを参照")

    # 英語側のBPE
    for name, vocab_size in [("eng_10k", 10000)]:
        spm.SentencePieceTrainer.train(
            input=f"{OUT_DIR}/eng_texts.txt",
            model_prefix=f"{OUT_DIR}/{name}",
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9999,
            pad_id=3,
        )
        sp = spm.SentencePieceProcessor()
        sp.load(f"{OUT_DIR}/{name}.model")

        bpe_lengths = []
        for text in train["translation"]:
            tokens = sp.encode(str(text), out_type=str)
            bpe_lengths.append(len(tokens))

        print(f"\n--- BPE vocab={vocab_size} (英語翻訳) ---")
        print(f"  トークン数 Mean:   {statistics.mean(bpe_lengths):.0f}")
        print(f"  トークン数 Median: {statistics.median(bpe_lengths):.0f}")
        print(f"  トークン数 Max:    {max(bpe_lengths)}")
        print(f"  圧縮率 (vs バイト): {statistics.mean(byte_lengths_tgt)/statistics.mean(bpe_lengths):.2f}x")

# =====================================================
# 4. ByT5 vs BPEの系列長比較
# =====================================================
print("\n" + "=" * 60)
print("4. ByT5 vs BPEの系列長比較（まとめ）")
print("=" * 60)

print(f"""
| 方式 | ソース(Akk) | ターゲット(Eng) | 備考 |
|------|-------------|----------------|------|
| ByT5 (byte) | Mean={statistics.mean(byte_lengths_src):.0f} | Mean={statistics.mean(byte_lengths_tgt):.0f} | 1byte=1token, diacritics分膨張 |""")

if HAS_SPM:
    # BPE 1Kの数値を再計算
    sp1k = spm.SentencePieceProcessor()
    sp1k.load(f"{OUT_DIR}/akk_1k.model")
    sp10k = spm.SentencePieceProcessor()
    sp10k.load(f"{OUT_DIR}/eng_10k.model")

    akk_bpe_1k = [len(sp1k.encode(str(t))) for t in train["transliteration"]]
    eng_bpe_10k = [len(sp10k.encode(str(t))) for t in train["translation"]]

    sp4k = spm.SentencePieceProcessor()
    sp4k.load(f"{OUT_DIR}/akk_500.model")
    akk_bpe_4k = [len(sp4k.encode(str(t))) for t in train["transliteration"]]

    print(f"| BPE 1K/10K | Mean={statistics.mean(akk_bpe_1k):.0f} | Mean={statistics.mean(eng_bpe_10k):.0f} | Gutherz 2023設定 |")
    print(f"| BPE 500/10K | Mean={statistics.mean(akk_bpe_4k):.0f} | Mean={statistics.mean(eng_bpe_10k):.0f} | vocab拡大版 |")

    # 512トークン制約でのtruncation率
    print(f"\n512トークン制約でのtruncation率 (ソース):")
    print(f"  ByT5:     {sum(1 for b in byte_lengths_src if b > 512)/len(byte_lengths_src)*100:.1f}%")
    print(f"  BPE 1K:   {sum(1 for b in akk_bpe_1k if b > 512)/len(akk_bpe_1k)*100:.1f}%")
    print(f"  BPE 500:   {sum(1 for b in akk_bpe_4k if b > 512)/len(akk_bpe_4k)*100:.1f}%")

    print(f"\n512トークン制約でのtruncation率 (ターゲット):")
    print(f"  ByT5:     {sum(1 for b in byte_lengths_tgt if b > 512)/len(byte_lengths_tgt)*100:.1f}%")
    print(f"  BPE 10K:  {sum(1 for b in eng_bpe_10k if b > 512)/len(eng_bpe_10k)*100:.1f}%")

# =====================================================
# 5. 翻字の構造とトークナイゼーションの相性分析
# =====================================================
print("\n" + "=" * 60)
print("5. 翻字の特殊文字とバイト消費")
print("=" * 60)

# diacritics文字の分析
diacritic_chars = Counter()
non_ascii_count = 0
total_chars = 0
for text in train["transliteration"]:
    for c in str(text):
        total_chars += 1
        if ord(c) > 127:
            non_ascii_count += 1
            diacritic_chars[c] += 1

print(f"非ASCII文字の割合: {non_ascii_count}/{total_chars} ({non_ascii_count/total_chars*100:.1f}%)")
print(f"\n--- 頻出diacritics (top 20) ---")
for c, count in diacritic_chars.most_common(20):
    byte_len = len(c.encode("utf-8"))
    print(f"  '{c}' (U+{ord(c):04X}): {count:5d} ({byte_len} bytes)")

# 各diacriticのバイトコスト
total_extra_bytes = sum(
    count * (len(c.encode("utf-8")) - 1) for c, count in diacritic_chars.items()
)
total_bytes = sum(len(str(t).encode("utf-8")) for t in train["transliteration"])
print(f"\ndiacriticsによる追加バイト: {total_extra_bytes}/{total_bytes} ({total_extra_bytes/total_bytes*100:.1f}%)")
print(f"→ ASCIIのみに正規化すれば{total_extra_bytes/total_bytes*100:.1f}%の系列長を削減可能")
print(f"  ただしexp002で正規化なしが正解と判明済み（音素区別の破壊）")

# =====================================================
# 6. 音節→BPEの対応分析
# =====================================================
if HAS_SPM:
    print("\n" + "=" * 60)
    print("6. BPEが音節構造をどう分割するか")
    print("=" * 60)

    sp = spm.SentencePieceProcessor()
    sp.load(f"{OUT_DIR}/akk_1k.model")

    # 典型的な音節語のBPE分割
    test_words = [
        "a-na", "ta-aq-bi-a-am", "kù.babbar", "qí-bi-ma",
        "ma-nu-ba-lúm-a-šur", "ša-lim-a-šùr", "i-dí-na-ku-um",
        "um-ma", "a-šùr", "KIŠIB", "DUMU"
    ]

    print("\n--- 音節語のBPE分割 (vocab=1K) ---")
    for w in test_words:
        tokens = sp.encode(w, out_type=str)
        syllables = w.split("-") if "-" in w else [w]
        preserves = "○" if all(any(s in t.replace("▁", "") for t in tokens) for s in syllables) else "×"
        print(f"  {w:30s} → {' | '.join(tokens):50s}  音節保持:{preserves}")

    sp4 = spm.SentencePieceProcessor()
    sp4.load(f"{OUT_DIR}/akk_500.model")

    print("\n--- 音節語のBPE分割 (vocab=500) ---")
    for w in test_words:
        tokens = sp4.encode(w, out_type=str)
        print(f"  {w:30s} → {' | '.join(tokens)}")

print("\n--- Done ---")
