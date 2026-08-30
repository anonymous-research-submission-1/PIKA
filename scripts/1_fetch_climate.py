"""
Expand ERA5 climate data to ALL viable WGMS glaciers (not just the original 84).

Viable = 10+ years of annual mass balance data, max year >= 2018.
Fetches daily temperature (ERA5-Land) and precipitation (ERA5) from Open-Meteo.

Resumes from existing data — safe to interrupt and restart.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
DATASETS = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "era5_openmeteo_daily_expanded.csv"

URL = "https://archive-api.open-meteo.com/v1/archive"
TEMP_VARS = "temperature_2m_mean,temperature_2m_max,temperature_2m_min"
PRECIP_VARS = "precipitation_sum,rain_sum"

COLUMNS = ["time", "temp_c", "temp_max_c", "temp_min_c", "precipitation_sum",
           "rain_sum", "elev_land_m", "elev_era5_m", "glacier_id", "glacier_name",
           "role", "solid_precip_we_mm"]

MARGIN_YEARS = 1
FLOOR, CEIL = 1950, 2024
PAUSE = 3.0
PROBE_INTERVAL = 120
MAX_WAIT = 5400
MAX_RETRIES = 8


def _wait_for_quota():
    waited = 0
    while waited < MAX_WAIT:
        time.sleep(PROBE_INTERVAL)
        waited += PROBE_INTERVAL
        try:
            p = requests.get(URL, params={
                "latitude": 46.8, "longitude": 10.8,
                "start_date": "2020-01-01", "end_date": "2020-01-02",
                "daily": "temperature_2m_mean", "models": "era5",
                "timezone": "UTC"}, timeout=60)
            if p.status_code == 200:
                print(f"      quota cleared after {waited // 60} min", flush=True)
                return True
        except requests.RequestException:
            pass
        if waited % 600 == 0:
            print(f"      still rate-limited ({waited // 60} min)", flush=True)
    print(f"      gave up waiting after {MAX_WAIT // 60} min", flush=True)
    return False


def _get(params):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(URL, params=params, timeout=300)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                if not _wait_for_quota():
                    return None
                continue
            print(f"      HTTP {r.status_code}: {r.text[:150]}", flush=True)
            time.sleep(10 * (attempt + 1))
        except requests.RequestException as e:
            print(f"      {type(e).__name__}; retry", flush=True)
            time.sleep(10 * (attempt + 1))
    return None


def fetch_glacier(lat, lon, start, end):
    base = {"latitude": lat, "longitude": lon, "start_date": start,
            "end_date": end, "timezone": "UTC"}
    # Try combined call first
    combined = _get({**base, "daily": f"{TEMP_VARS},{PRECIP_VARS}",
                     "models": "era5_land,era5"})
    if combined and "daily" in combined:
        keys = combined["daily"].keys()
        if any(k.endswith("_era5_land") for k in keys) and any(k.endswith("_era5") for k in keys):
            return _from_combined(combined)

    # Fallback: two separate calls
    land = _get({**base, "daily": TEMP_VARS, "models": "era5_land"})
    time.sleep(PAUSE)
    era5 = _get({**base, "daily": PRECIP_VARS, "models": "era5"})
    if not land or not era5 or "daily" not in land or "daily" not in era5:
        return None
    a = pd.DataFrame(land["daily"]).rename(columns={
        "temperature_2m_mean": "temp_c", "temperature_2m_max": "temp_max_c",
        "temperature_2m_min": "temp_min_c"})
    b = pd.DataFrame(era5["daily"])
    df = a.merge(b, on="time", how="outer")
    df["elev_land_m"] = land.get("elevation")
    df["elev_era5_m"] = era5.get("elevation")
    return df


def _from_combined(js):
    d = pd.DataFrame(js["daily"])
    ren = {}
    for c in d.columns:
        if c == "time":
            continue
        stem = c.replace("_era5_land", "").replace("_era5", "")
        if c.endswith("_era5_land") and stem.startswith("temperature_2m"):
            ren[c] = {"temperature_2m_mean": "temp_c", "temperature_2m_max": "temp_max_c",
                      "temperature_2m_min": "temp_min_c"}[stem]
        elif c.endswith("_era5") and stem in ("precipitation_sum", "rain_sum"):
            ren[c] = stem
    df = d[["time"] + list(ren)].rename(columns=ren)
    elev = js.get("elevation")
    df["elev_land_m"] = elev
    df["elev_era5_m"] = elev
    return df


# ==========================================================================
# BUILD GLACIER LIST: All WGMS glaciers with 10+ years, max_year >= 2018
# ==========================================================================
print("Loading mass balance and glacier metadata...")
mb = pd.read_csv(DATASETS / "mass_balance.csv")
gl = pd.read_csv(DATASETS / "glacier.csv")

mb_valid = mb[mb["annual_balance"].notna()].copy()
per_glacier = mb_valid.groupby("glacier_id").agg(
    n_years=("year", "count"),
    min_year=("year", "min"),
    max_year=("year", "max"),
    glacier_name=("glacier_name", "first")
).reset_index()

# Viable: 10+ years, data extends to at least 2018
viable = per_glacier[(per_glacier["n_years"] >= 10) & (per_glacier["max_year"] >= 2018)].copy()
print(f"Viable glaciers (10+ years, max_year >= 2018): {len(viable)}")

# Get lat/lon from glacier.csv
gl_coords = gl[["id", "latitude", "longitude"]].rename(columns={"id": "glacier_id"})
viable = viable.merge(gl_coords, on="glacier_id", how="left")
viable = viable[viable["latitude"].notna() & viable["longitude"].notna()]
print(f"With valid coordinates: {len(viable)}")

# Already fetched (from existing parquet or the old CSV)?
existing_ids = set()
if (DATA / "era5_openmeteo_daily.parquet").exists():
    existing = pd.read_parquet(DATA / "era5_openmeteo_daily.parquet", columns=["glacier_id"])
    existing_ids |= set(existing["glacier_id"].unique())
    print(f"Already in parquet: {len(existing_ids)}")

# Also check if the expanded CSV already has some
if OUT.exists():
    done_df = pd.read_csv(OUT, usecols=["glacier_id"])
    existing_ids |= set(done_df["glacier_id"].unique())
    print(f"Already in expanded CSV: {done_df['glacier_id'].nunique()}")

# Also include any from the old GEE era5 file
era5_old = pd.read_csv(DATASETS / "era5_daily_all_glaciers (2).csv", usecols=["glacier_id"])
era5_old_ids = set(era5_old["glacier_id"].unique())

to_fetch = viable[~viable["glacier_id"].isin(existing_ids)].copy()
to_fetch = to_fetch.sort_values("n_years", ascending=False)  # longest records first (most valuable)
print(f"To fetch: {len(to_fetch)} glaciers")
print(f"Already covered (parquet + CSV + GEE): {len(viable) - len(to_fetch)}")

# ==========================================================================
# FETCH
# ==========================================================================
failures = []
for i, row in enumerate(to_fetch.itertuples(), 1):
    start = f"{max(FLOOR, int(row.min_year) - MARGIN_YEARS)}-01-01"
    end = f"{min(CEIL, int(row.max_year) + MARGIN_YEARS)}-12-31"

    df = fetch_glacier(row.latitude, row.longitude, start, end)
    if df is None:
        print(f"  [{i}/{len(to_fetch)}] {row.glacier_name}: FAILED", flush=True)
        failures.append(row.glacier_id)
        continue

    df["glacier_id"] = row.glacier_id
    df["glacier_name"] = row.glacier_name
    df["role"] = "training_population"
    df["solid_precip_we_mm"] = df["precipitation_sum"] - df["rain_sum"]

    df = df.reindex(columns=COLUMNS)
    header = not OUT.exists()
    df.to_csv(OUT, mode="a", header=header, index=False)
    n_days = len(df)
    print(f"  [{i}/{len(to_fetch)}] {row.glacier_name}: {n_days} days {start[:4]}-{end[:4]}",
          flush=True)
    time.sleep(PAUSE)

print(f"\nDone. Output: {OUT}")
if failures:
    print(f"FAILED ({len(failures)}): {failures}")
if OUT.exists():
    out_df = pd.read_csv(OUT)
    print(f"Expanded CSV: {out_df['glacier_id'].nunique()} glaciers, {len(out_df)} rows")
