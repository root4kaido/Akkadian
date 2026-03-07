"""
Kaggle Notebook一覧スクレイピング (Playwright版)
コンペのCodeページからノートブック一覧を取得する（Most Votesソート）
"""

import json
import re
import time
from datetime import datetime
from playwright.sync_api import sync_playwright

COMP_SLUG = "deep-past-initiative-machine-translation"
CODE_URL = f"https://www.kaggle.com/competitions/{COMP_SLUG}/code?sortBy=voteCount"
OUTPUT_FILE = "snapshot_{}.json".format(datetime.now().strftime("%Y%m%d"))


def scrape_notebooks():
    notebooks = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 900})

        print(f"Navigating to {CODE_URL}")
        page.goto(CODE_URL, wait_until="networkidle", timeout=60000)
        time.sleep(3)

        # Scroll to load more notebooks
        for i in range(10):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)
            print(f"  Scroll {i+1}/10")

        # Extract notebook links
        links = page.query_selector_all('a[href*="/code/"]')
        seen_urls = set()

        for link in links:
            href = link.get_attribute("href")
            if not href or "/code/" not in href:
                continue
            if href.endswith("/code") or href.endswith("/code/"):
                continue
            if "#" in href:
                continue
            parts = href.strip("/").split("/")
            if len(parts) < 2:
                continue
            if COMP_SLUG in href and href.rstrip("/").endswith("/code"):
                continue

            full_url = f"https://www.kaggle.com{href}" if href.startswith("/") else href
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            title = link.inner_text().strip()
            if not title or len(title) < 3:
                continue
            if title.lower() in ("code", "new notebook", "filters"):
                continue

            # Extract author/slug from URL
            match = re.search(r'/code/([^/]+)/([^/]+)', href)
            author = match.group(1) if match else ""
            slug = match.group(2) if match else ""

            notebooks.append({
                "title": title,
                "url": full_url,
                "author": author,
                "slug": slug,
                "kaggle_ref": f"{author}/{slug}" if author and slug else "",
            })

        print(f"Found {len(notebooks)} raw notebook links")

        # Try to get vote info from parent containers
        for nb in notebooks:
            try:
                link_el = page.query_selector(f'a[href="{nb["url"].replace("https://www.kaggle.com", "")}"]')
                if link_el:
                    parent = link_el.evaluate_handle("el => el.closest('li') || el.closest('div[class*=\"list\"]') || el.parentElement.parentElement.parentElement")
                    if parent:
                        parent_text = parent.inner_text()
                        # Extract vote count
                        vote_match = re.search(r'arrow_drop_up\s*(\d+)', parent_text)
                        if vote_match:
                            nb["votes"] = int(vote_match.group(1))
                        else:
                            num_match = re.search(r'(\d+)\s*$', parent_text.split('\n')[0].strip())
                            if num_match:
                                nb["votes"] = int(num_match.group(1))

                        # Extract comment count
                        comment_match = re.search(r'comment\s*(\d+)', parent_text)
                        if comment_match:
                            nb["comments"] = int(comment_match.group(1))
            except Exception:
                pass

        # Sort by votes (descending), notebooks without votes at the end
        notebooks.sort(key=lambda x: x.get("votes", -1), reverse=True)

        browser.close()

    # Save snapshot
    snapshot = {
        "competition": COMP_SLUG,
        "scraped_at": datetime.now().isoformat(),
        "sort": "voteCount",
        "total_notebooks": len(notebooks),
        "notebooks": notebooks,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(notebooks)} notebooks to {OUTPUT_FILE}")

    # Print top 10
    print("\n=== Top Notebooks ===")
    for i, nb in enumerate(notebooks[:10]):
        votes = nb.get("votes", "?")
        print(f"  {i+1}. [{votes} votes] {nb['title'][:80]}")

    return snapshot


if __name__ == "__main__":
    scrape_notebooks()
