import os
import re
import requests
from bs4 import BeautifulSoup

# Directories for saving output
pages_dir = "scraped_pages_lvl1"
links_dir = "scraped_links_lvl1"
os.makedirs(pages_dir, exist_ok=True)
os.makedirs(links_dir, exist_ok=True)

# File to store discovered links not yet scraped
discovered_links_file = "discovered_links.txt"

# Function to fetch <main> tag content and links
def scrape_main_content_and_links(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"\u274c Failed to fetch {url} (Status {response.status_code})")
            return None, None

        soup = BeautifulSoup(response.text, 'html.parser')
        main_tag = soup.find('main')
        if not main_tag:
            print(f"\u274c No <main> tag found for {url}")
            return None, None

        # Clean unwanted tags
        for tag in main_tag(['script', 'style', 'noscript']):
            tag.decompose()

        # Extract text
        text = main_tag.get_text(separator='\n', strip=True)

        # Extract only absolute links
        links = [a['href'] for a in main_tag.find_all('a', href=True) if a['href'].startswith('http')]

        return text, links

    except Exception as e:
        print(f"\u274c Error processing {url}: {e}")
        return None, None

# ----------------------------------------------------------------------------------------------------------

# Safe filename generator
def url_to_filename(url):
    base = re.sub(r'^https?://', '', url)
    safe = re.sub(r'[^\w\-_.]', '_', base)
    return safe + ".txt"

# Read URLs
with open("luddy_complete_links.txt", "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

# Load scraped links
scraped_links_file = "scraped_links.txt"
if os.path.exists(scraped_links_file):
    with open(scraped_links_file, "r", encoding="utf-8") as f:
        scraped_links = set(line.strip() for line in f if line.strip())
else:
    scraped_links = set()

new_links_to_add = set()

# ------------------------------------------------------------------------------------------------------------------

# Process each URL
for url in urls:
    if url in scraped_links:
        print(f"🔁 Skipping (already scraped): {url}")
        continue

    print(f"🔍 Scraping: {url}")
    text, links = scrape_main_content_and_links(url)

    if text:
        text_filename = url_to_filename(url)
        text_path = os.path.join(pages_dir, text_filename)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(f"[Source URL]: {url}\n\n")
            f.write(text)
        print(f"✅ Saved main text to: {text_path}")

    if links:
        links_filename = "links_" + url_to_filename(url)
        links_path = os.path.join(links_dir, links_filename)
        with open(links_path, "w", encoding="utf-8") as f:
            f.write(f"[Source URL]: {url}\n\n")
            for link in links:
                f.write(link + "\n")
        print(f"✅ Saved links to: {links_path}")

        # Track links not in scraped set
        for link in links:
            if link not in scraped_links:
                new_links_to_add.add(link)

    # Mark current URL as scraped
    scraped_links.add(url)
    with open(scraped_links_file, "a", encoding="utf-8") as f:
        f.write(url + "\n")

    if not text and not links:
        print(f"⚠️ Skipping {url} due to error or no content.")

# ----------------------------------------------------------------------------------------------------------

# Save discovered (but not scraped) links
if new_links_to_add:
    with open(discovered_links_file, "a", encoding="utf-8") as f:
        for link in sorted(new_links_to_add):
            f.write(link + "\n")
    print(f"🆕 Saved {len(new_links_to_add)} new links to discovered_links.txt")
