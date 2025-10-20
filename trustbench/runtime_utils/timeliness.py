import requests
import re
import whois
from datetime import datetime

def get_last_modified(url):
    """
    Retrieves the Last-Modified header from a given URL's HTTP response.

    Args:
        url (str): The URL of the website to check.

    Returns:
        str or None: The value of the Last-Modified header if present, otherwise None.
    """
    try:
        response = requests.head(url)  # Use HEAD request for efficiency
        if 'Last-Modified' in response.headers:
            return response.headers['Last-Modified']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return None

def date_from_domain(url, verbose=False):
    try:
        domain_info = whois.whois(url)
        
        if domain_info.updated_date:
            # The creation_date can be a single datetime object or a list of them
            if isinstance(domain_info.updated_date, list):
                creation_date = domain_info.updated_date[0].year
            else:
                creation_date = domain_info.updated_date.year
        
            current_date = datetime.now().year
            domain_age = current_date - creation_date
            if(verbose):
                print(f"Domain: {url}")
                print(f"Domain Age since last update ({domain_age} years)")
            return domain_age
        else:
            if(verbose):
                print(f"Could not find creation date for {url}")
            return None 
    
    except Exception as e:
        pass 

def extract_urls(text: str) -> list:
    """
    Extracts all URLs from a given string of text using regex.

    Args:
        text: The text to search for URLs.

    Returns:
        A list of URL strings found in the text.
    """
    # Updated regex to find URLs that may not start with http/https,
    # but include "www" or are a simple domain. Now requires TLDs
    # to be at least two characters to avoid matching "e.g.".
    url_pattern = r'(?:(?:https?://)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))'
    urls = re.findall(url_pattern, text)
    urls = [url.replace("?utm_source=chatgpt.com","") for url in urls]
    return urls