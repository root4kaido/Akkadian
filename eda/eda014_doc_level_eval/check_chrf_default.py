import evaluate

m = evaluate.load("chrf")

preds = ["The king went to the temple and offered sacrifices"]
refs = [["The king traveled to the great temple and made offerings"]]

for wo in [0, 1, 2]:
    r = m.compute(predictions=preds, references=refs, word_order=wo)
    print(f"word_order={wo}: score={r['score']:.2f}")

# Default (no word_order specified)
r_default = m.compute(predictions=preds, references=refs)
print(f"default:       score={r_default['score']:.2f}")

# Check what the default actually is by inspecting
import sacrebleu
chrf_default = sacrebleu.metrics.CHRF()
print(f"\nsacrebleu CHRF default word_order = {chrf_default.word_order}")

chrf_pp = sacrebleu.metrics.CHRF(word_order=2)
print(f"sacrebleu CHRF++ word_order = {chrf_pp.word_order}")
