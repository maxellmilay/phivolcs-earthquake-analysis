from bs4 import BeautifulSoup
import requests
import urllib3
import sys
from datetime import datetime
import pandas as pd
import os

# Suppress InsecureRequestWarning when we disable SSL verification for this host
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"User-Agent": "earthquake-model/1.0 (+https://example.local)"}


def build_batch_urls():
    """Build list of URLs for all months from 2018 to 2025."""
    years = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    
    urls = []
    for year in years:
        for month in months:
            urls.append(f"https://earthquake.phivolcs.dost.gov.ph/EQLatest-Monthly/{year}/{year}_{month}.html")
    return urls


def fetch_page(url):
    """Fetch a page with SSL fallback."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15, verify=True)
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        print(f"SSL verification failed for {url}. Retrying with verification disabled...", file=sys.stderr)
        response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        response.raise_for_status()
        return response


def parse_earthquake_data(html_content):
    """Parse earthquake data from HTML content."""
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
            row_data["datetime"] = datetime.strptime(row_data["datetime_str"], "%d %B %Y - %I:%M %p")
            data.append(row_data)
        except ValueError:
            # Skip rows with invalid datetime format
            continue
    
    return data


def scrape_all_urls(urls):
    """Scrape all URLs and consolidate earthquake data."""
    all_earthquake_data = []
    successful_urls = 0
    failed_urls = 0
    
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Fetching: {url}")
        
        try:
            response = fetch_page(url)
            data = parse_earthquake_data(response.text)
            all_earthquake_data.extend(data)
            successful_urls += 1
            print(f"  ✓ Found {len(data)} records")
        except requests.exceptions.RequestException as e:
            failed_urls += 1
            print(f"  ✗ Failed: {e}")
            continue
    
    print(f"\nScraping complete: {successful_urls} URLs succeeded, {failed_urls} URLs failed")
    return all_earthquake_data


def save_to_csv(earthquake_data):
    """Save earthquake data to CSV file."""
    if not earthquake_data:
        print("No valid earthquake data found to save.")
        return None
    
    df = pd.DataFrame(earthquake_data)
    
    # Reorder columns
    columns = ['datetime', 'latitude', 'longitude', 'depth', 'magnitude', 'location']
    df = df[columns]
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Save to CSV in data folder
    os.makedirs("data", exist_ok=True)
    csv_filename = f"data/earthquake_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"\nSuccessfully saved {len(earthquake_data)} earthquake records to {csv_filename}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return csv_filename


def main():
    """Main entry point."""
    print("Building batch URLs...")
    batch_urls = build_batch_urls()
    print(f"Total URLs to scrape: {len(batch_urls)}\n")
    
    print("Starting scraping process...")
    earthquake_data = scrape_all_urls(batch_urls)
    
    save_to_csv(earthquake_data)


if __name__ == "__main__":
    main()
