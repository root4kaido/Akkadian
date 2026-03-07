"""
Kaggle Discussion一覧スクレイピング（Playwright使用）
コンペのディスカッションページから全トピック一覧を取得してJSONに保存する。
"""
import json
import asyncio
import re
from datetime import datetime
from playwright.async_api import async_playwright

COMP_URL = "https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion"
OUTPUT_FILE = f"snapshot_{datetime.now().strftime('%Y%m%d')}.json"
DETAILS_FILE = f"snapshot_{datetime.now().strftime('%Y%m%d')}_details.json"


async def scrape_discussions():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print(f"Navigating to {COMP_URL}...")
        await page.goto(COMP_URL, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(3000)

        # スクロールして全トピックを読み込む
        prev_count = 0
        for scroll_attempt in range(20):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)

            # 現在のトピック数を確認
            topics = await page.query_selector_all('li[class*="discussion-list"] a, ul[class*="List"] li a[href*="/discussion/"]')
            if len(topics) == 0:
                # 別のセレクタを試す
                topics = await page.query_selector_all('a[href*="/discussion/"]')

            current_count = len(topics)
            print(f"  Scroll {scroll_attempt + 1}: found {current_count} links")

            if current_count == prev_count and scroll_attempt > 2:
                break
            prev_count = current_count

        # ページのHTMLを取得して解析
        html = await page.content()

        # ディスカッションリンクを抽出
        discussion_links = await page.evaluate("""
            () => {
                const results = [];
                const seen = new Set();

                // ディスカッションリンクを検索
                const links = document.querySelectorAll('a[href*="/discussion/"]');
                for (const link of links) {
                    const href = link.getAttribute('href');
                    if (!href) continue;

                    // /discussion/数字 のパターンにマッチするもの
                    const match = href.match(/\\/discussion\\/(\\d+)/);
                    if (!match) continue;

                    const discussionId = match[1];
                    if (seen.has(discussionId)) continue;
                    seen.add(discussionId);

                    // タイトルテキストを取得
                    let title = link.textContent.trim();
                    if (!title || title.length < 3) continue;

                    // 親要素からコメント数などの情報を探す
                    let comments = '';
                    let votes = '';
                    let author = '';
                    let date = '';

                    // 近くの要素からメタ情報を取得
                    const listItem = link.closest('li') || link.closest('[class*="list"]') || link.parentElement?.parentElement;
                    if (listItem) {
                        const text = listItem.textContent;
                        const commentMatch = text.match(/(\\d+)\\s*(?:comment|reply|replies)/i);
                        if (commentMatch) comments = commentMatch[1];

                        const voteMatch = text.match(/(\\d+)\\s*(?:vote|upvote)/i);
                        if (voteMatch) votes = voteMatch[1];
                    }

                    results.push({
                        id: discussionId,
                        title: title.substring(0, 200),
                        url: 'https://www.kaggle.com' + href,
                        comments: comments,
                        votes: votes
                    });
                }
                return results;
            }
        """)

        print(f"\nFound {len(discussion_links)} discussion topics")

        # 保存（詳細スクリプトとキー名を統一: "discussions"）
        snapshot = {
            "date": datetime.now().isoformat(),
            "competition": "deep-past-initiative-machine-translation",
            "total_topics": len(discussion_links),
            "discussions": discussion_links
        }

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

        print(f"Saved to {OUTPUT_FILE}")

        # HTMLも保存（デバッグ用）
        with open("_last_page.html", 'w', encoding='utf-8') as f:
            f.write(html)

        await browser.close()
        return discussion_links


if __name__ == "__main__":
    results = asyncio.run(scrape_discussions())
    for i, r in enumerate(results[:10]):
        print(f"  {i+1}. [{r['id']}] {r['title'][:80]}")
