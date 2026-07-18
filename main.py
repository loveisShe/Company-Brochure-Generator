import os
import requests
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import display, Markdown, update_display
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

openai = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY")
)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}


# FIX 1: Corrected the typo in the function name (was: fectch_websiite_links)
def fetch_website_links(url: str) -> list[str]:
    """Fetch all unique, usable hyperlinks from a webpage."""
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    links = []
    for tag in soup.find_all("a"):
        href = tag.get("href")
        if href:
            href = urljoin(url, href)
            if not (
                href.startswith("mailto:")
                or href.startswith("tel:")
                or href.startswith("javascript:")
                or "#" in href
            ):
                links.append(href)

    return list(set(links))


def fetch_website_contents(url: str) -> str:
    """Fetch and clean the visible text content of a webpage."""
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

link_system_prompt = """
You are an expert website navigator.

Your task is to analyze a list of URLs extracted from a company's website and identify ONLY the pages that contain important information for generating a professional company brochure.

Return ONLY a valid JSON object.

Rules:
- Only include links that are likely to contain useful company information.
- Always return absolute URLs.
- Do not invent or modify URLs.
- Do not include duplicate URLs.
- Ignore anchors (#), mailto links, tel links, javascript links, PDFs, images, videos, login pages, cart pages, search pages, privacy/legal pages, cookie pages, terms pages, or social media links unless they are the company's primary contact page.
- If multiple URLs point to the same section, keep the best one.
- If a page belongs to more than one category, choose the most relevant category.
- If no useful links exist, return:
{
  "links": []
}

Output format:

{
  "links": [
    {
      "type": "<category>",
      "url": "<absolute_url>"
    }
  ]
}

Return ONLY the JSON object.
"""

brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""


# ──────────────────────────────────────────────
# Link selection
# ──────────────────────────────────────────────

def get_links_user_prompt(url: str) -> str:
    user_prompt = (
        f"Here is the list of links on the website {url} -\n"
        "Please decide which of these are relevant web links for a brochure about the company.\n"
        "Respond with the full https URL in JSON format.\n"
        "Do not include Terms of Service, Privacy, email links.\n"
        "Links (some might be relative links):\n"
    )
    # FIX 1 (continued): Using corrected function name
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt


def select_relevant_links(url: str) -> dict:
    """Use an LLM to pick the most brochure-relevant links from a page."""
    response = openai.chat.completions.create(
        # Use gemini-3.1-flash-lite for fast, structured JSON link selection
        model="gemini-3.1-flash-lite",
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"}
    )

    result = response.choices[0].message.content

    # FIX 3: Removed the dead-code markdown-fence stripping:
    #   result.replace("```json", "").replace("```", "")
    # When response_format={"type":"json_object"} is set, the API returns raw
    # JSON — never wrapped in markdown fences — so those replaces never matched.
    return json.loads(result)


# ──────────────────────────────────────────────
# Page aggregation
# ──────────────────────────────────────────────

def fetch_page_and_all_relevant_links(url: str) -> str:
    """Scrape the landing page plus all LLM-selected relevant sub-pages."""
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url)

    result = f"## Landing Page\n\n{contents}\n\n"
    result += "## Relevant Pages\n"

    for link in relevant_links["links"]:
        result += f"\n\n### {link['type']}\n"
        try:
            result += fetch_website_contents(link["url"])
        except Exception as e:
            # FIX 4: Log failures instead of silently swallowing them.
            # The original `except Exception: pass` gave zero visibility into
            # network errors, 403s, timeouts, etc.
            logger.warning("Could not fetch %s — %s: %s", link["url"], type(e).__name__, e)

    return result


# ──────────────────────────────────────────────
# Brochure generation
# ──────────────────────────────────────────────

def get_brochure_user_prompt(company_name: str, url: str) -> str:
    user_prompt = (
        f"You are looking at a company called {company_name}.\n\n"
        "Here are the contents of its landing page and other relevant pages.\n\n"
        "Create a professional brochure in Markdown.\n\n"
    )
    user_prompt += fetch_page_and_all_relevant_links(url)

    # FIX 5: Raised the hard character cap from 5,000 → 20,000.
    # At 5,000 chars almost all scraped content was silently discarded before
    # reaching the LLM, producing a very thin brochure. 20,000 chars ≈ ~5,000
    # tokens, well within gemini-2.5-flash's context window.
    return user_prompt[:20_000]


def create_brochure(company_name: str, url: str) -> None:
    """Generate and display a company brochure (non-streaming)."""
    response = openai.chat.completions.create(
        # Use gemini-3.5-flash for high quality brochure writing
        model="gemini-3.5-flash",
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ]
    )
    display(Markdown(response.choices[0].message.content))


def stream_brochure(company_name: str, url: str) -> None:
    """Generate and stream a company brochure to the notebook output in real time."""
    stream = openai.chat.completions.create(
        model="gemini-3.5-flash",
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
        stream=True
    )

    response = ""
    handle = display(Markdown(""), display_id=True)

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            response += delta
            update_display(Markdown(response), display_id=handle.display_id)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    stream_brochure(
        "HuggingFace",
        "https://huggingface.co"
    )
