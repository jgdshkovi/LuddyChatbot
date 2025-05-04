import requests
from bs4 import BeautifulSoup

def scrape(url1):
    response = requests.get(url1)

    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the title
        title = soup.title.text
        print(f"Page Title: {title}\n")
        
        # Extract main paragraph content
        main_content = soup.find('div').find_all('p')
        print("Main Content:")
        for paragraph in main_content:
            print(f"- {paragraph.text.strip()}")
        print()
        
        # Extract links
        links = soup.find_all('a')
        print("Links:")
        for link in links:
            print(f"- {link.text.strip()}: {link.get('href')}")
        print()
        
        # Optional: Extract CSS (but in a more structured way)
        style_tag = soup.find('style')
        if style_tag:
            print("CSS detected (not showing full content)")
            
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


def get_links(url2):
    response = requests.get(url2)

    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main paragraph content
        main_content = soup.find('div').find_all('p')
        print("Main Content:")
        for paragraph in main_content:
            print(f"- {paragraph.text.strip()}")
        print()
        
        # Extract links
        links = soup.find_all('a')
        print("Links:")
        for link in links:
            print(f"- {link.text.strip()}: {link.get('href')}")
        print()
        
        # Optional: Extract CSS (but in a more structured way)
        style_tag = soup.find('style')
        if style_tag:
            print("CSS detected (not showing full content)")
            
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")



url = "https://luddy.indiana.edu/academics/grad-programs/index.html"  # Replace with your target website

scrape(url)