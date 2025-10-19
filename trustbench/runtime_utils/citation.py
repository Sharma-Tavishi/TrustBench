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
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=3)

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


def extract_references(text):
    """
    Extracts full, multi-line reference citations from a reference list.

    This function is designed to parse a block of text containing a
    numbered bibliography and extract each full reference. It identifies
    references that start with a number (e.g., "1. ") and captures the
    entire multi-line entry.

    It specifically avoids simple inline citations like '[1]' or '(Smith, 2021)'.

    Args:
        text (str): The text containing the reference list.

    Returns:
        list: A list of all found full reference strings.
    """

    citation_pattern = r'\((?P<title1>.*?),\s*(?P<date1>\d{4}),\s*(?P<venue1>.*?)\)|"(?P<title2>.*?)\.”\s*,\s*(?P<date2>\d{4}),\s*(?P<venue2>.*?)\.'
    academic_references = []
    for match in re.finditer(citation_pattern, text):
        # Check which named group was populated to determine the format
        if match.group('title1') is not None:
            # It's the first format: (title, date, venue)
            title = match.group('title1').strip()
            year = match.group('date1').strip()
            venue = match.group('venue1').strip()
        else:
            # It's the second format: "Title.", date, venue.
            title = match.group('title2').strip()
            year = match.group('date2').strip()
            venue = match.group('venue2').strip()
        academic_references.append({'title':title, 'year':year, 'venue':venue})
    return academic_references

class ReferenceScreener:
    def __init__(self, whitelist: list):
        """ Initializes the ReferenceScreener with a whitelist of allowed venues.

        Args:
            whitelist (list): List of allowed publication venues.
        """
        self.whitelist = whitelist
    
    def process_references(self, references: list):
        if(self.whitelist is None):
            for ref in references:
                ref['allowed'] = True
            return references
        else:
            for ref in references:
                if ref['venue'] in self.whitelist:
                    ref['allowed'] = True
                else:
                    ref['allowed'] = False
            return references

def extract_references_old(text):
    """
    Extracts full, multi-line reference citations from a reference list.

    This function is designed to parse a block of text containing a
    numbered bibliography and extract each full reference. It identifies
    references that start with a number (e.g., "1. ") and captures the
    entire multi-line entry.

    It specifically avoids simple inline citations like '[1]' or '(Smith, 2021)'.

    Args:
        text (str): The text containing the reference list.

    Returns:
        list: A list of all found full reference strings.
    """
    # Pre-process the text to strip leading whitespace from each line. This
    # makes the main regex more robust against indented reference lists.
    processed_text = re.sub(r'^\s+', '', text.strip(), flags=re.MULTILINE)

    # This regex finds entries in a numbered list. It starts by finding a line
    # beginning with digits and a period. It then non-greedily matches all
    # characters (including newlines) until it sees the start of the next
    # reference (another line starting with digits and a period) or the
    # end of the string.
    citation_pattern = re.compile(
        r"^\d+\..+?(?=\n^\d+\.|\Z)",
        re.MULTILINE | re.DOTALL
    )

    # Find all non-overlapping matches in the processed string
    citations = citation_pattern.findall(processed_text)
    # Strip leading/trailing whitespace from each found citation
    return [citation.strip() for citation in citations]

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