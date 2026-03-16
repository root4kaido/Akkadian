"""
Scrape ALL notebooks from the competition code page, scrolling until no more load.
Then filter for training-related notebooks.
"""
import json
import re
import time
from datetime import datetime
from playwright.sync_api import sync_playwright

COMP_SLUG = "deep-past-initiative-machine-translation"
CODE_URL = f"https://www.kaggle.com/competitions/{COMP_SLUG}/code?sortBy=voteCount"

def scrape_all_notebooks():
    notebooks = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 900})

        print(f"Navigating to {CODE_URL}")
        page.goto(CODE_URL, wait_until="networkidle", timeout=60000)
        time.sleep(3)

        # Scroll until no more notebooks load
        prev_count = 0
        for i in range(50):  # max 50 scrolls
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)

            links = page.query_selector_all('a[href*="/code/"]')
            curr_count = len(links)
            print(f"  Scroll {i+1}: {curr_count} links found")

            if curr_count == prev_count and i > 5:
                print("  No new content, stopping.")
                break
            prev_count = curr_count

        # Extract notebook links
        links = page.query_selector_all('a[href*="/code/"]')
        seen = set()

        for link in links:
            href = link.get_attribute("href")
            if not href or "/code/" not in href:
                continue
            if href.endswith("/code") or href.endswith("/code/"):
                continue
            if "#" in href:
                continue
            if COMP_SLUG in href and href.rstrip("/").endswith("/code"):
                continue

            match = re.search(r'/code/([^/]+)/([^/]+)', href)
            if not match:
                continue

            author = match.group(1)
            slug = match.group(2)
            ref = f"{author}/{slug}"

            if ref in seen:
                continue
            seen.add(ref)

            title = link.inner_text().strip()
            if not title or len(title) < 3:
                continue
            if title.lower() in ("code", "new notebook", "filters"):
                continue

            full_url = f"https://www.kaggle.com{href}" if href.startswith("/") else href

            notebooks.append({
                "title": title,
                "url": full_url,
                "author": author,
                "slug": slug,
                "ref": ref,
            })

        browser.close()

    print(f"\nTotal unique notebooks: {len(notebooks)}")

    # Filter for training-related
    train_nbs = [nb for nb in notebooks if re.search(r'train', nb['slug'], re.I)]
    print(f"Training-related notebooks: {len(train_nbs)}")
    for nb in train_nbs:
        print(f"  - {nb['ref']}: {nb['title']}")

    # Also show ALL notebooks for completeness
    print(f"\nAll {len(notebooks)} notebooks:")
    for nb in notebooks:
        marker = " *** TRAIN ***" if re.search(r'train', nb['slug'], re.I) else ""
        print(f"  {nb['ref']}{marker}")

    # Save full list
    with open("all_notebooks_20260315.json", "w") as f:
        json.dump(notebooks, f, ensure_ascii=False, indent=2)

    return notebooks

if __name__ == "__main__":
    scrape_all_notebooks()
