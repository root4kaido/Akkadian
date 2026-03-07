"""
Kaggle Discussion 詳細スクレイピング (Playwright版)
各ディスカッショントピックの詳細（コメント数、本文、投票数など）を取得する
"""

import json
import glob
import time
from datetime import datetime
from playwright.sync_api import sync_playwright


def get_latest_snapshot():
    """最新のスナップショットファイルを取得"""
    files = sorted(glob.glob("snapshot_*.json"))
    # _details.json を除外
    files = [f for f in files if "_details" not in f]
    if not files:
        raise FileNotFoundError("No snapshot files found. Run scrape_with_playwright.py first.")
    return files[-1]


def scrape_details(snapshot_file=None):
    if snapshot_file is None:
        snapshot_file = get_latest_snapshot()

    with open(snapshot_file, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    discussions = snapshot["discussions"]
    print(f"Scraping details for {len(discussions)} discussions from {snapshot_file}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 900})

        for i, disc in enumerate(discussions):
            url = disc["url"]
            print(f"[{i+1}/{len(discussions)}] {disc['title'][:60]}...")

            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                time.sleep(2)

                # Extract post body
                body_el = page.query_selector('[class*="discussion-detail"] .markdown-converter, [class*="ForumTopicBody"], div[class*="topic-body"]')
                if body_el:
                    disc["body"] = body_el.inner_text().strip()[:2000]
                else:
                    # Fallback: get main content area
                    main = page.query_selector('main, [role="main"]')
                    if main:
                        disc["body"] = main.inner_text().strip()[:2000]
                    else:
                        disc["body"] = ""

                # Extract comment count
                comment_elements = page.query_selector_all('[class*="comment"], [data-testid*="comment"]')
                disc["comment_count_approx"] = len(comment_elements)

                # Extract vote count if available
                vote_el = page.query_selector('[class*="vote"] span, [aria-label*="vote"], [class*="Vote"]')
                if vote_el:
                    vote_text = vote_el.inner_text().strip()
                    try:
                        disc["votes"] = int(vote_text)
                    except ValueError:
                        disc["votes"] = 0
                else:
                    disc["votes"] = 0

                # Extract author
                author_el = page.query_selector('[class*="author"], [data-testid*="author"], a[href*="/users/"]')
                if author_el:
                    disc["author"] = author_el.inner_text().strip()

                # Extract date
                time_el = page.query_selector('time, [class*="date"], [class*="time"]')
                if time_el:
                    disc["posted_date"] = time_el.get_attribute("datetime") or time_el.inner_text().strip()

                disc["detail_scraped"] = True

            except Exception as e:
                print(f"  Error: {e}")
                disc["detail_scraped"] = False
                disc["error"] = str(e)

            time.sleep(1)  # Rate limiting

        browser.close()

    # Save updated snapshot with details
    output_file = snapshot_file.replace(".json", "_details.json")
    snapshot["discussions"] = discussions
    snapshot["details_scraped_at"] = datetime.now().isoformat()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"Saved details to {output_file}")
    return snapshot


if __name__ == "__main__":
    scrape_details()
