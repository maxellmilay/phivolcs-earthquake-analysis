from bs4 import BeautifulSoup
import requests
import certifi
import sys
import re
from datetime import datetime
import pandas as pd

url = "https://earthquake.phivolcs.dost.gov.ph"

headers = {"User-Agent": "earthquake-model/1.0 (+https://example.local)"}

try:
    response = requests.get(url, headers=headers, timeout=15, verify=certifi.where())
    response.raise_for_status()
except requests.exceptions.SSLError as ssl_error:
    print(
        f"SSL verification failed for {url}. Attempting HTTP fallback...\n{ssl_error}",
        file=sys.stderr,
    )
    insecure_url = url.replace("https://", "http://", 1)
    response = requests.get(insecure_url, headers=headers, timeout=15, verify=False)
    response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

items = soup.select("div.auto-style94 table.MsoNormalTable tbody tr")

print(f"Items found: {len(items)}")

# List to store all earthquake data
earthquake_data = []

for i, item in enumerate(items):
    print(f"Row {i+1} ", end="")
    td_list = item.select("td")

    if len(td_list) != 6:
        continue

    row_data = {}

    for j, td in enumerate(td_list):
        if j == 0:
            row_data["datetime_str"] = td.get_text(separator=' ', strip=True)
        elif j == 1:
            row_data["latitude"] = td.get_text(separator=' ', strip=True)
        elif j == 2:
            row_data["longitude"] = td.get_text(separator=' ', strip=True)
        elif j == 3:
            row_data["depth"] = td.get_text(separator=' ', strip=True)
        elif j == 4:
            row_data["magnitude"] = td.get_text(separator=' ', strip=True)
        elif j == 5:
            row_data["location"] = td.get_text(separator=' ', strip=True)

    try:
        raw_str = row_data["datetime_str"]
        converted_datetime_string = datetime.strptime(raw_str, "%d %B %Y - %I:%M %p")
        row_data["datetime"] = converted_datetime_string
        
        # Add the row data to our list
        earthquake_data.append(row_data)
        print("✓")  # Success indicator
        
    except ValueError:
        print(f"Invalid datetime string: {row_data['datetime_str']}")
        continue

# Create DataFrame and save to CSV
if earthquake_data:
    df = pd.DataFrame(earthquake_data)
    
    # Reorder columns to put datetime as the second column
    columns = ['datetime', 'latitude', 'longitude', 'depth', 'magnitude', 'location']
    df = df[columns]
    
    # Save to CSV
    csv_filename = f"earthquake_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"\nSuccessfully saved {len(earthquake_data)} earthquake records to {csv_filename}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
else:
    print("No valid earthquake data found to save.")


