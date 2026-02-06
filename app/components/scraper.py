"""
Scraper component for fetching the latest earthquake data from PHIVOLCS.

Extracts only the latest 100 rows of seismic data for real-time inference.
"""

from bs4 import BeautifulSoup
import requests
import urllib3
import pandas as pd
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"User-Agent": "earthquake-model/1.0 (+https://example.local)"}

# PHIVOLCS latest earthquakes page
LATEST_URL = "https://earthquake.phivolcs.dost.gov.ph"


def fetch_page(url: str) -> requests.Response:
    """Fetch a page with SSL fallback."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15, verify=True)
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        response.raise_for_status()
        return response


def parse_earthquake_data(html_content: str) -> list:
    """Parse earthquake data from PHIVOLCS HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    items = soup.select("div.auto-style94 table.MsoNormalTable tbody tr")

    data = []
    for item in items:
        td_list = item.select("td")

        if len(td_list) != 6:
            continue

        row_data = {
            "datetime_str": td_list[0].get_text(separator=' ', strip=True),
            "latitude": td_list[1].get_text(separator=' ', strip=True),
            "longitude": td_list[2].get_text(separator=' ', strip=True),
            "depth": td_list[3].get_text(separator=' ', strip=True),
            "magnitude": td_list[4].get_text(separator=' ', strip=True),
            "location": td_list[5].get_text(separator=' ', strip=True),
        }

        try:
            row_data["datetime"] = datetime.strptime(
                row_data["datetime_str"], "%d %B %Y - %I:%M %p"
            )
            data.append(row_data)
        except ValueError:
            continue

    return data

def scrape_latest(max_rows: int = 100) -> pd.DataFrame:
    """
    Scrape the latest earthquake data from PHIVOLCS.

    Fetches recent months and returns the most recent `max_rows` records.

    Parameters
    ----------
    max_rows : int
        Maximum number of rows to return (default 100)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: datetime, latitude, longitude, depth, magnitude, location
    """

    try:
        response = fetch_page(LATEST_URL)
        print(response)
        data = parse_earthquake_data(response.text)
        print(data)
    except requests.exceptions.RequestException as e:
        print(e)
        return pd.DataFrame(columns=["datetime", "latitude", "longitude", "depth", "magnitude", "location"])

    df = pd.DataFrame(data)
    df = df[["datetime", "latitude", "longitude", "depth", "magnitude", "location"]]

    # Convert numeric columns
    for col in ["latitude", "longitude", "depth", "magnitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "datetime", "magnitude"])
    df = df.sort_values("datetime", ascending=False).reset_index(drop=True)

    # Take only the latest max_rows
    df = df.head(max_rows).reset_index(drop=True)

    # Re-sort chronologically (oldest first) for feature engineering
    df = df.sort_values("datetime").reset_index(drop=True)

    return df
