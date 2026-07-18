# Company Brochure Generator

This is a simple Python pipeline built in a Jupyter notebook that generates a markdown brochure about a company by scraping its website.

It uses the **Gemini API** (accessed via the OpenAI library compatibility layer) to filter website links and write the brochure.

## How it works

1. **Scrapes the homepage:** Downloads the HTML of the main URL and extracts all hyperlinks.
2. **Filters links:** Sends the list of found links to Gemini to choose which pages are actually relevant (e.g. *About Us*, *Products*, *Careers*) and ignore junk links (e.g. *Privacy Policy*, *Cart*).
3. **Scrapes inner pages:** Downloads the text contents of the chosen relevant pages.
4. **Writes the brochure:** Passes all aggregated text to Gemini to compile into a structured Markdown brochure.

## Requirements

Before running the notebook, install the following python packages:

```bash
pip install requests beautifulsoup4 python-dotenv openai ipython
```

## Setup

1. Create a `.env` file in the same directory:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```
2. Open either `main_fixed.ipynb` or `ok.ipynb` in your Jupyter environment.
3. Run the cells step-by-step to fetch the links and generate the brochure.
