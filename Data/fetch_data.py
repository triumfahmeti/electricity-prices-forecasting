from pathlib import Path
import csv
import requests
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
from Data import *
from data_fetching_utils import _add_months, load_config


def fetch_day_ahead_prices():
    """
    Fetches historical day-ahead prices between a start and end date provided in init file, from energy-charts API.
    Stores this data lcoally as a single csv file.
    """
    cfg = load_config()
    ec = cfg["ENERGY_CHARTS"]

    url = ec["URL"].rstrip("/") + ec["PRICE_ENDPOINT"]
    params = {"bzn": BIDDING_ZONE, "start": START_DATE, "end": END_DATE}

    out_dir = Path("data_csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    r = requests.get(url, params=params, timeout=ec.get("timeout_seconds", 30))
    r.raise_for_status()
    data = r.json()

    unix_seconds = data.get("unix_seconds", [])
    prices = data.get("price", [])

    if len(unix_seconds) != len(prices):
        raise ValueError(f"Length mismatch: unix_seconds={len(unix_seconds)} vs price={len(prices)}")

    ts_fetch = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%dT%H%M")
    out_path = out_dir / f"day_ahead_prices_{params['bzn']}_{ts_fetch}.csv"

    tz = ZoneInfo(TIMEZONE)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "price", "bzn"])  # no unix_seconds, no unit

        for t, p in zip(unix_seconds, prices):
            ts_local = datetime.fromtimestamp(int(t), tz=timezone.utc).astimezone(tz).isoformat()
            writer.writerow([ts_local, p, params["bzn"]])

    print(f"Saved CSV: {out_path}")

def fetch_public_power_for_all_production_types():
    """
    Fetches the public power production for all production types in 15-min frequency between a start and end date,
    provided in the init file from energy-charts API.
    Make sure that start and end data are a small chunk of data as it will fetch all production types at once in 15-min freq.
    Adjust it in the init file, when using this function specifically.
    """
    cfg = load_config()
    ec = cfg["ENERGY_CHARTS"]

    url = ec["URL"].rstrip("/") + ec["PUBLIC_POWER_ENDPOINT"]
    params = {
        "country": COUNTRY.lower(),
        "start": START_DATE,
        "end": END_DATE
    }

    out_dir = Path("data_csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    r = requests.get(url, params=params, timeout=ec.get("timeout_seconds", 30))
    r.raise_for_status()
    data = r.json()

    unix_seconds = data.get("unix_seconds", [])
    production_types = data.get("production_types", [])

    tz = ZoneInfo(TIMEZONE)  # e.g. "Europe/Berlin"
    ts_fetch = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%dT%H%M%z")
    out_path = out_dir / f"public_power_{params['country']}_{ts_fetch}.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "value", "production_type", "country"])

        # Write ALL production types returned by the API
        for pt in production_types:
            name = pt.get("name", "")
            values = pt.get("data", [])

            if len(values) != len(unix_seconds):
                raise ValueError(
                    f"Length mismatch for '{name}': unix_seconds={len(unix_seconds)} vs data={len(values)}"
                )

            for t, v in zip(unix_seconds, values):
                ts_local = datetime.fromtimestamp(int(t), tz=timezone.utc).astimezone(tz).isoformat()
                writer.writerow([ts_local, v, name, params["country"]])

    print(f"Saved CSV: {out_path}")

def fetch_public_power_for_wanted_production_types():
    """
    Fetches the public power production for all production types in 15-min frequency between a start and end date,
    provided in the init file from energy-charts API. Filters only wanted production types before storing the data.
    Make sure that start and end data are a small chunk of data as it will fetch all production types at once in 15-min freq.
    It handles the length of the start and end date by creating small chunks of 2-months from the start and end date
    interval provided. So that the user can also specify start and end date spanning years.
    """

    cfg = load_config()
    ec = cfg["ENERGY_CHARTS"]

    url = ec["URL"].rstrip("/") + ec["PUBLIC_POWER_ENDPOINT"]

    country = COUNTRY
    tz = ZoneInfo(TIMEZONE)
    wanted = list(PRODUCTION_TYPE)
    wanted_set = set(wanted)


    start_d = date.fromisoformat(START_DATE)
    end_d = date.fromisoformat(END_DATE)

    out_dir = Path("data_csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_fetch = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%dT%H%M%z")
    out_path = out_dir / f"public_power_for_chosen_production_types_{country}_{START_DATE}_to_{END_DATE}_{ts_fetch}.csv"

    header = ["timestamp", "country"] + wanted

    # Track what we've already written (avoid duplicate timestamps across chunks)
    written_ts = set()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        chunk_start = start_d
        while chunk_start <= end_d:
            # 2-month chunk: [chunk_start, chunk_end]
            chunk_end = _add_months(chunk_start, 2)  # same day-of-month, 2 months later
            chunk_end = min(chunk_end, end_d)

            params = {
                "country": country,
                "start": chunk_start.isoformat(),
                "end": chunk_end.isoformat(),
            }

            r = requests.get(url, params=params, timeout=ec.get("timeout_seconds", 30))
            if not r.ok:
                raise RuntimeError(f"Request failed: {r.status_code} {r.url}\n{r.text[:1000]}")
            data = r.json()

            unix_seconds = data.get("unix_seconds", [])
            production_types = data.get("production_types", [])

            # Build a mapping: type_name -> list of values
            series = {}
            for pt in production_types:
                name = pt.get("name", "")
                if name in wanted_set:
                    series[name] = pt.get("data", [])

            # Ensure we got all wanted types (warn, but still proceed)
            missing = [t for t in wanted if t not in series]
            if missing:
                print(f"Warning: missing types in {chunk_start}..{chunk_end}: {missing}")

            # Validate lengths for the types we did receive
            for name, values in series.items():
                if len(values) != len(unix_seconds):
                    raise ValueError(
                        f"Length mismatch for '{name}' in {chunk_start}..{chunk_end}: "
                        f"unix_seconds={len(unix_seconds)} vs data={len(values)}"
                    )

            # Write rows: one row per timestamp, with columns for each wanted type
            # Use blank value if a type is missing in this chunk.
            for i, t in enumerate(unix_seconds):
                ts_local = datetime.fromtimestamp(int(t), tz=timezone.utc).astimezone(tz).isoformat()

                # De-duplicate across chunks
                if ts_local in written_ts:
                    continue
                written_ts.add(ts_local)

                row = [ts_local, country]
                for name in wanted:
                    vals = series.get(name)
                    row.append(vals[i] if vals is not None else "")
                writer.writerow(row)

            print(f"Fetched {chunk_start} to {chunk_end} -> wrote {len(unix_seconds)} timestamps (before dedupe)")

            # Next chunk starts the day after chunk_end to avoid overlap
            chunk_start = chunk_end.fromordinal(chunk_end.toordinal() + 1)

    print(f"Saved merged CSV: {out_path}")


if __name__ == "__main__":
    fetch_day_ahead_prices()
    fetch_public_power_for_wanted_production_types()
    fetch_public_power_for_all_production_types()

