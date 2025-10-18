import requests
import re

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

def verify_link(url: str, verbose=False) -> bool:
    """
    Checks if a given URL is real and accessible by making an HTTP request.

    This function sends a HEAD request, which is more efficient than a GET
    request as it doesn't download the page content. It checks for a successful
    status code (less than 400).

    Args:
        url: The URL string to verify.
        verbose : If True, print logs. default: False

    Returns:
        True if the website is accessible and returns a success status code,
        False otherwise.
    """
    try:
        # Add a scheme (https://) if one is not present in the URL.
        # This is necessary for the requests library to work correctly.
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'https://' + url
            if(verbose):
                print(f"No scheme provided. Trying with: {url}")

        # Set a common user-agent to mimic a real browser request.
        # Some websites may block requests from scripts without a user-agent.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Make a HEAD request to the URL.
        # allow_redirects=True ensures that we follow any redirects.
        # A timeout is set to prevent the script from hanging indefinitely.
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=10)

        # Check the HTTP status code. Codes below 400 (e.g., 200 OK, 301 Redirect)
        # typically indicate that the link is valid and reachable.
        if response.status_code < 400:
            if(verbose):
                print(f"✅ Success! The link is real and accessible.")
                print(f"Final URL after redirects: {response.url}")
                print(f"Status Code: {response.status_code}")
            return True
        elif response.status_code == 405 or response.status_code == 403:
            if(verbose):
                print(f"✅ Success! The link is real but may not be accessible")
                print(f"Status Code: {response.status_code}")
            return True
        elif response.status_code == 402:
            if(verbose):
                print(f"✅ Success! The link is real but may not be accessible without payment")
                print(f"Status Code: {response.status_code}")
            return True
        elif response.status_code == 404:
            if(verbose):
                print(f"❌ Failed. The link is correctly formatted but not correct.")
                print(f"Status Code: {response.status_code}")
            return False
        else:
            
            if(verbose):
                print(f"❌ Failed. The link exists but returned an error.")
                print(f"Status Code: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        # Catch all exceptions from the requests library (e.g., connection error,
        # timeout, invalid URL) which indicate the link is not accessible.
        if(verbose):
            print(f"❌ Error: The link appears to be fake or is currently down.")
            print(f"Details: {e}")
        return False


if __name__ == "__main__":
    sample_text = """
    Here are some links to check:
    1. https://www.example.com
    2. http://nonexistent.website.fake
    3. www.github.com
    4. invalid-url
    5. example.org
    """

    urls = extract_urls(sample_text)
    print("Extracted URLs:", urls)

    for url in urls:
        print(f"\nVerifying URL: {url}")
        is_valid = verify_link(url, verbose=True)
        print(f"Is the URL valid? {is_valid}")