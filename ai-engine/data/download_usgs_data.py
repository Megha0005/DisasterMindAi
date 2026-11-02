
from io import StringIO
from pathlib import Path
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1"

START_YEAR = 2000
END_YEAR = 2025
MIN_MAGNITUDE = 4.0
API_LIMIT = 20000

OUTPUT_FILE = Path(
    "data/raw/usgs_earthquakes_2000_2025_m4.csv"
)


def fetch_url(url):
    """Fetch data from USGS with a basic User-Agent."""
    request = Request(
        url,
        headers={"User-Agent": "DisasterMind-AI/1.0"}
    )

    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def get_event_count(start_date, end_date):
    """Get the number of matching earthquake events."""
    params = {
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": MIN_MAGNITUDE,
        "eventtype": "earthquake",
    }

    url = f"{BASE_URL}/count?{urlencode(params)}"
    return int(fetch_url(url).strip())


def download_year(year):
    """Download all matching events for one calendar year."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    print(f"\nDownloading year: {year}")

    total_events = get_event_count(start_date, end_date)
    print(f"Events available: {total_events}")

    if total_events == 0:
        return pd.DataFrame()

    yearly_data = []

    for offset in range(1, total_events + 1, API_LIMIT):
        params = {
            "format": "csv",
            "starttime": start_date,
            "endtime": end_date,
            "minmagnitude": MIN_MAGNITUDE,
            "eventtype": "earthquake",
            "orderby": "time-asc",
            "limit": API_LIMIT,
            "offset": offset,
        }

        url = f"{BASE_URL}/query?{urlencode(params)}"

        csv_text = fetch_url(url)
        batch = pd.read_csv(StringIO(csv_text))

        if not batch.empty:
            yearly_data.append(batch)

        print(
            f"Downloaded {min(offset + API_LIMIT - 1, total_events)}"
            f" / {total_events}"
        )

        sleep(0.2)

    if not yearly_data:
        return pd.DataFrame()

    return pd.concat(yearly_data, ignore_index=True)


def main():
    all_years = []

    for year in range(START_YEAR, END_YEAR + 1):
        year_data = download_year(year)

        if not year_data.empty:
            all_years.append(year_data)

        sleep(0.5)

    if not all_years:
        print("No earthquake records were downloaded.")
        return

    final_data = pd.concat(all_years, ignore_index=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_data.to_csv(OUTPUT_FILE, index=False)

    print("\n========== DOWNLOAD COMPLETE ==========")
    print(f"Total records: {len(final_data)}")
    print(f"Total columns: {len(final_data.columns)}")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")
    print("\nAvailable columns:")
    print(final_data.columns.tolist())


if __name__ == "__main__":
    main()
