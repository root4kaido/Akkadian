"""Check public scores for key notebooks via Playwright."""
import json
import re
import time
from playwright.sync_api import sync_playwright

NOTEBOOKS = [
    "takamichitoda/dpc-starter-infer",
    "qifeihhh666/dpc-starter-infer-add-sentencealign",
    "jiexusheng20bz/byt-ensemble",
    "vitorhugobarbedo/lb-35-9-with-regex-corrections-public-model",
    "mattiaangeli/deep-pasta-mbr",
    "lgregory/akkadiam-exemple",
    "anthonytherrien/byt-ensemble-script",
    "giovannyrodrguez/lb-35-9-ensembling-post-processing-baseline",
    "prayagp1/adaptive-beams-test-v1",
    "baidalinadilzhan/lb-35-2-ensemble",
    "takamichitoda/dpc-infer-with-post-processing-by-llm",
    "meenalsinha/hybrid-best-akkadian",
    "assiaben/akkadian-english-inference-byt5-optimized-34x",
    "loopassembly/score-35-3-byt5-mbr-pipeline",
    "llkh0a/dpc-baseline-train-infer",
    "jackcerion/byt5-seq2seq-infer",
    "ngyzly/lb-35-1-ensembling-post-processing-baseline",
    "qifeihhh666/deep-past-challenge-byt5-base-inference",
    "mattiaangeli/deep-pasta-mbr-v2",
    "kkashyap14/akkadian2eng-v1",
]

results = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for nb_ref in NOTEBOOKS:
        url = f"https://www.kaggle.com/code/{nb_ref}"
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(2)

            text = page.evaluate("() => document.body.innerText.substring(0, 5000)")

            # Extract score
            score_match = re.search(r'(?:Public Score|Best Score)\s*\n?\s*([\d.]+)', text)
            best_match = re.search(r'Best Score\s*\n?\s*([\d.]+)', text)

            # Extract training links from the page
            links = page.evaluate("""() => {
                const anchors = document.querySelectorAll('a');
                return Array.from(anchors)
                    .map(a => a.href)
                    .filter(h => h.includes('/code/') && h.includes('train'));
            }""")

            # Check for model/dataset inputs
            input_info = re.findall(r'\[Private Dataset\]|(?:DATASETS|MODELS)\s+\S+', text)

            entry = {
                "ref": nb_ref,
                "public_score": score_match.group(1) if score_match else None,
                "best_score": best_match.group(1) if best_match else None,
                "train_links": links,
                "input_info": input_info[:5],
            }
            results.append(entry)
            print(f"  {nb_ref}: score={entry['best_score'] or entry['public_score'] or '?'}, train_links={len(links)}")

        except Exception as e:
            print(f"  {nb_ref}: ERROR - {e}")
            results.append({"ref": nb_ref, "error": str(e)})

    browser.close()

with open("scores_20260315.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(results)} results to scores_20260315.json")
