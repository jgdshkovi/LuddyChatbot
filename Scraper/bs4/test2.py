import requests
from bs4 import BeautifulSoup
import time
import csv

def scrape_page(url):
    """Scrape a single page and return the soup object"""
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        else:
            print(f"Failed to retrieve {url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def extract_program_info(program_soup, program_url):
    """Extract relevant information from a program page"""
    program_info = {
        'url': program_url,
        'title': '',
        'description': '',
        'requirements': '',
        'contact': ''
    }
    
    # Extract the program title
    if program_soup.find('h1'):
        program_info['title'] = program_soup.find('h1').text.strip()
    
    # Extract program description (assuming it's in a div with a certain class)
    description_div = program_soup.find('div', class_='program-description')
    if description_div:
        program_info['description'] = description_div.text.strip()
    
    # Extract program requirements (this will vary by page structure)
    requirements_div = program_soup.find('div', class_='program-requirements')
    if requirements_div:
        program_info['requirements'] = requirements_div.text.strip()
    
    # Extract contact information
    contact_div = program_soup.find('div', class_='contact-info')
    if contact_div:
        program_info['contact'] = contact_div.text.strip()
    
    return program_info

def main():
    # Main graduate programs page
    base_url = 'https://luddy.indiana.edu'
    grad_programs_url = f'{base_url}/academics/grad-programs/index.html'
    
    # Scrape the main page
    main_soup = scrape_page(grad_programs_url)
    if not main_soup:
        print("Failed to scrape the main page. Exiting.")
        return
    
    # Find all program links
    # This selector may need adjustment based on the actual page structure
    program_links = []
    
    # Finding master's program links
    masters_section = main_soup.find('h2', string='Master\'s Programs')
    if masters_section:
        # Get the next relevant elements which contain the links
        program_elements = masters_section.find_next_siblings('div')
        for element in program_elements:
            links = element.find_all('a')
            for link in links:
                if link.get('href') and '.html' in link.get('href'):
                    program_links.append(link.get('href'))
    
    # Finding PhD program links
    phd_section = main_soup.find('h2', string='Doctoral Programs')
    if phd_section:
        program_elements = phd_section.find_next_siblings('div')
        for element in program_elements:
            links = element.find_all('a')
            for link in links:
                if link.get('href') and '.html' in link.get('href'):
                    program_links.append(link.get('href'))
    
    print(f"Found {len(program_links)} program links.")
    
    # Store program information
    all_programs = []
    
    # Visit each program page and extract information
    for i, link in enumerate(program_links[:5]):  # Limiting to 5 for testing
        full_url = link if link.startswith('http') else f"{base_url}{link}" if link.startswith('/') else f"{base_url}/{link}"
        print(f"Scraping program {i+1}/{min(5, len(program_links))}: {full_url}")
        
        program_soup = scrape_page(full_url)
        if program_soup:
            program_info = extract_program_info(program_soup, full_url)
            all_programs.append(program_info)
            print(f"  - Successfully scraped: {program_info['title'] or 'Unknown title'}")
        
        # Polite delay to avoid overwhelming the server
        time.sleep(2)
    
    # Save to CSV
    with open('luddy_graduate_programs.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'url', 'description', 'requirements', 'contact']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for program in all_programs:
            writer.writerow(program)
    
    print(f"Completed scraping. Data saved to luddy_graduate_programs.csv")

if __name__ == "__main__":
    main()