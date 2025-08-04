import requests
import pandas as pd
import os

API_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"
API_KEY = "7bf150c6982b04dce71e528e8f89ce31aef534df"
EMAIL = "92-berried.kilo@icloud.com"

def extract_data(query="corporation", max_cases=100):
    """
    Extract corporate litigation cases from CourtListener API (CAP data).
    Fetches up to `max_cases` results.
    """

    headers = { 
        "User-Agent": f"inst414-final-project ({EMAIL})",
        "Authorization": f"Token {API_KEY}"
    }

    params = {
        "search": query,
        # The API's limit per page is 20
        "page_size": 20 
    }

    all_cases = []
    url = API_URL

    print(f"Extracting cases from CourtListener API with query: {query}")

    while url and len(all_cases) < max_cases:
        response = requests.get(url, headers=headers, params=params if url == API_URL else None)
        if response.status_code != 200:
            print(response.text)
            raise Exception(f"API request failed with status code {response.status_code}")

        data = response.json()

        for case in data.get("results", []):
            all_cases.append({
                "case_id": case.get("id"),
                "case_name": case.get("case_name", ""),
                "court": case.get("court", ""),
                "date_created": case.get("date_created", ""),
                "citation": case.get("citation", ""),
                "plain_text_url": case.get("plain_text", "")
            })

        # Get "next" page URL if available
        url = data.get("next")

    df = pd.DataFrame(all_cases)
    os.makedirs("data/extracted", exist_ok=True)
    df.to_csv("data/extracted/raw_data.csv", index=False)
    print(f"Extracted {len(df)} cases to data/extracted/raw_data.csv")

    return df