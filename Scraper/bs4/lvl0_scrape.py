import requests
from bs4 import BeautifulSoup

def main_get_text_and_links(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        main_tag = soup.find('main')

        if main_tag:
            # Remove unwanted tags
            for tag in main_tag(['script', 'style', 'noscript']):
                tag.decompose()

            # Extract and save visible text
            text = main_tag.get_text(separator='\n', strip=True)
            with open("luddy_main_text.txt", "w", encoding="utf-8") as text_file:
                text_file.write(text)

            # Extract all <a> tags with href attributes
            links = main_tag.find_all('a', href=True)

            # Filter only absolute URLs (complete links)
            complete_links = [tag['href'] for tag in links if tag['href'].startswith('http')]

            # Save links to file
            with open("luddy_complete_links.txt", "w", encoding="utf-8") as link_file:
                for href in complete_links:
                    link_file.write(href + '\n')

            print(f"✅ Saved {len(complete_links)} links and main text.")
        else:
            print("❌ No <main> tag found.")
    else:
        print(f"❌ Failed to retrieve page. Status code: {response.status_code}")

# Run the function
url = "https://luddy.indiana.edu/academics/grad-programs/index.html"
main_get_text_and_links(url)
