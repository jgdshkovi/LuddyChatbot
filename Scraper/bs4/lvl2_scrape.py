import os
import re
import requests
from bs4 import BeautifulSoup

# Directories for Level 2
pages_dir = "scraped_pages_lvl2"
links_dir = "scraped_links_lvl2"
os.makedirs(pages_dir, exist_ok=True)
os.makedirs(links_dir, exist_ok=True)

# Load discovered links from Level 1
discovered_links_file = "discovered_links.txt"
with open(discovered_links_file, "r", encoding="utf-8") as f:
    lvl2_urls = [line.strip() for line in f if line.strip()]

# Load Scraped links
scraped_links_file = "scraped_links.txt"
if os.path.exists(scraped_links_file):
    with open(scraped_links_file, "r", encoding="utf-8") as f:
        scraped_links = set(line.strip() for line in f if line.strip())
else:
    scraped_links = set()

new_links_to_add = set()

# Utility to convert URL to safe filename
def url_to_filename(url):
    base = re.sub(r'^https?://', '', url)
    safe = re.sub(r'[^\w\-_.]', '_', base)
    return safe + ".txt"

# Scraping function
def scrape_main_content_and_links(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"❌ Failed to fetch {url} (Status {response.status_code})")
            return None, None

        soup = BeautifulSoup(response.text, 'html.parser')
        main_tag = soup.find('main')
        if not main_tag:
            print(f"❌ No <main> tag found for {url}")
            return None, None

        for tag in main_tag(['script', 'style', 'noscript']):
            tag.decompose()

        text = main_tag.get_text(separator='\n', strip=True)
        links = [a['href'] for a in main_tag.find_all('a', href=True) if a['href'].startswith('http')]
        return text, links

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None, None

# Scrape each URL
for url in lvl2_urls:
    if url in scraped_links:
        print(f"🔁 Already scraped: {url}")
        continue

    print(f"🔍 Scraping Level 2 URL: {url}")
    text, links = scrape_main_content_and_links(url)

    if text:
        text_filename = url_to_filename(url)
        text_path = os.path.join(pages_dir, text_filename)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(f"[Source URL]: {url}\n\n")
            f.write(text)
        print(f"✅ Text saved to: {text_path}")

    if links:
        links_filename = "links_" + url_to_filename(url)
        links_path = os.path.join(links_dir, links_filename)
        with open(links_path, "w", encoding="utf-8") as f:
            f.write(f"[Source URL]: {url}\n\n")
            for link in links:
                f.write(link + "\n")
        print(f"✅ Links saved to: {links_path}")

        for link in links:
            if link not in scraped_links:
                new_links_to_add.add(link)

    # Mark current URL as scraped
    scraped_links.add(url)

# Update scraped_links.txt
if new_links_to_add:
    with open(scraped_links_file, "a", encoding="utf-8") as f:
        for link in new_links_to_add:
            f.write(link + "\n")
    print(f"🆕 {len(new_links_to_add)} new links added to scraped_links.txt")

# Save discovered links for Level 3
if new_links_to_add:
    with open("discovered_links_lvl2.txt", "w", encoding="utf-8") as f:
        for link in sorted(new_links_to_add):
            f.write(link + "\n")
    print(f"📄 discovered_links_lvl2.txt created with new links.")
