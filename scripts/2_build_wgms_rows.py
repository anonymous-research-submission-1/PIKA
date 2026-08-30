"""Fetch remaining screened WGMS glaciers and write leakage-safe annual rows.

Does NOT modify master_glacier_data_fixed.csv or candidate_master_rows.csv.

Leakage rule (the Johnsons/Mera mistake): a glacier whose RGI region is already
used as OOD holdout, or is Himalaya/Antarctic, must not enter training.
Those rows are tagged external_holdout. Training is only regions that the
clean protocol already trains on, plus RGI-10 (Leviy Aktru) which is new and
does not overlap any current holdout.

Dokriani (3454) is dropped. The five glaciers already in candidate_master_rows
are skipped here (`assemble_expanded_master.py` assigns holdout roles). Extra
Alps (RGI-11) are skipped:
training already has 24, and they do not add a LORO region.

Daily climate: data/era5_openmeteo_expansion.csv (gitignored).
Annual table:  data/expansion_master_rows.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ALT = DATA / "alt_datasets"
DAILY = DATA / "era5_openmeteo_expansion.csv"
OUT = DATA / "expansion_master_rows.csv"

URL = "https://archive-api.open-meteo.com/v1/archive"
TEMP_VARS = "temperature_2m_mean,temperature_2m_max,temperature_2m_min"
PRECIP_VARS = "precipitation_sum,rain_sum"
COLUMNS = [
    "time", "temp_c", "temp_max_c", "temp_min_c", "precipitation_sum",
    "rain_sum", "elev_land_m", "elev_era5_m", "glacier_id", "glacier_name",
    "role", "solid_precip_we_mm",
]

# Already integrated (notebook routes 4 to holdout; Dokriani dropped).
SKIP_IDS = {3366, 3367, 3454, 3996, 3997}
DROP_IDS = {3454}

# Clean-protocol holdout regions + Himalaya (14) even if Parlung is 15.
# New glaciers in 16/17 would sit in Antizana / Martial Este's regions.
NEVER_TRAIN_REGIONS = {5, 14, 15, 16, 17, 18, 19}
# Already in training, or new and disjoint from holdouts (10 = North Asia).
TRAIN_OK_REGIONS = {1, 2, 3, 6, 7, 8, 10, 11, 12, 13}
SKIP_EXTRA_ALPS = True

MARGIN_YEARS = 1
FLOOR, CEIL = 1950, 2025
CHUNK_YEARS = 20
PAUSE = 4.0
PROBE_INTERVAL = 120
MAX_WAIT = 5400
MAX_RETRIES = 4


def hydrological_year_window(year: int, lat: float):
    if lat < 0:
        return pd.Timestamp(year - 1, 4, 1), pd.Timestamp(year, 3, 31)
    return pd.Timestamp(year - 1, 10, 1), pd.Timestamp(year, 9, 30)


def summer_months(lat: float):
    return [12, 1, 2] if lat < 0 else [6, 7, 8]


def first_rgi_id(raw) -> str | None:
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    return str(raw).split("|")[0].strip() or None


def rgi_region_int(rgi_id: str | None) -> int | None:
    if not rgi_id:
        return None
    m = pd.Series([rgi_id]).str.extract(r"RGI\d+-(\d+)\.", expand=False)
    if m.isna().all():
        return None
    return int(m.iloc[0])


def assign_role(region: int | None) -> str:
    if region is None:
        return "unresolved"
    if region in NEVER_TRAIN_REGIONS:
        return "external_holdout"
    if region in TRAIN_OK_REGIONS:
        return "training_population"
    return "unresolved"


def _wait_for_quota() -> bool:
    waited = 0
    while waited < MAX_WAIT:
        time.sleep(PROBE_INTERVAL)
        waited += PROBE_INTERVAL
        try:
            p = requests.get(
                URL,
                params={
                    "latitude": 46.8, "longitude": 10.8,
                    "start_date": "2020-01-01", "end_date": "2020-01-02",
                    "daily": "temperature_2m_mean", "models": "era5",
                    "timezone": "UTC",
                },
                timeout=60,
            )
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
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                if not _wait_for_quota():
                    return None
                continue
            print(f"      HTTP {r.status_code}: {r.text[:180]}", flush=True)
            time.sleep(10 * (attempt + 1))
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = 20 * (attempt + 1)
            print(f"      {type(e).__name__}; wait {wait}s ({attempt+1}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
        except requests.RequestException as e:
            print(f"      {type(e).__name__}; retry", flush=True)
            time.sleep(10 * (attempt + 1))
    return None


def _from_combined(js) -> pd.DataFrame | None:
    d = pd.DataFrame(js["daily"])
    ren = {}
    for c in d.columns:
        if c == "time":
            continue
        stem = c.replace("_era5_land", "").replace("_era5", "")
        if c.endswith("_era5_land") and stem.startswith("temperature_2m"):
            ren[c] = {
                "temperature_2m_mean": "temp_c",
                "temperature_2m_max": "temp_max_c",
                "temperature_2m_min": "temp_min_c",
            }[stem]
        elif c.endswith("_era5") and stem in ("precipitation_sum", "rain_sum"):
            ren[c] = stem
    need = ["temp_c", "precipitation_sum", "rain_sum"]
    if not all(v in ren.values() for v in need):
        return None
    df = d[["time"] + list(ren)].rename(columns=ren)
    elev = js.get("elevation")
    df["elev_land_m"] = elev
    df["elev_era5_m"] = elev
    return df


def fetch_span(lat, lon, start, end) -> pd.DataFrame | None:
    base = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end, "timezone": "UTC",
    }
    combined = _get({**base, "daily": f"{TEMP_VARS},{PRECIP_VARS}", "models": "era5_land,era5"})
    if combined and "daily" in combined:
        parsed = _from_combined(combined)
        if parsed is not None:
            return parsed
    print("      combined call missed; split land/era5", flush=True)
    land = _get({**base, "daily": TEMP_VARS, "models": "era5_land"})
    time.sleep(PAUSE)
    era5 = _get({**base, "daily": PRECIP_VARS, "models": "era5"})
    if not land or not era5 or "daily" not in land or "daily" not in era5:
        return None
    a = pd.DataFrame(land["daily"]).rename(columns={
        "temperature_2m_mean": "temp_c",
        "temperature_2m_max": "temp_max_c",
        "temperature_2m_min": "temp_min_c",
    })
    b = pd.DataFrame(era5["daily"])
    df = a.merge(b, on="time", how="outer")
    df["elev_land_m"] = land.get("elevation")
    df["elev_era5_m"] = era5.get("elevation")
    return df


def fetch_glacier(lat, lon, year_min: int, year_max: int) -> pd.DataFrame | None:
    start_y = max(FLOOR, int(year_min) - MARGIN_YEARS)
    end_y = min(CEIL, int(year_max) + MARGIN_YEARS)
    chunks = []
    y = start_y
    while y <= end_y:
        y1 = min(y + CHUNK_YEARS - 1, end_y)
        start = f"{y}-01-01"
        end = f"{y1}-12-31"
        part = fetch_span(lat, lon, start, end)
        if part is None:
            return None
        chunks.append(part)
        y = y1 + 1
        time.sleep(1.0)
    return pd.concat(chunks, ignore_index=True).drop_duplicates("time")


def load_targets() -> pd.DataFrame:
    cand = pd.read_csv(DATA / "candidate_glaciers.csv")
    meta = pd.read_csv(
        ALT / "glacier.csv",
        usecols=["id", "short_name", "rgi60_ids", "rgi50_ids", "latitude", "longitude"],
        low_memory=False,
    ).rename(columns={"id": "glacier_id"})
    df = cand.merge(meta, on="glacier_id", how="left")
    df["rgi_id"] = df["rgi60_ids"].map(first_rgi_id)
    missing = df["rgi_id"].isna() & df["rgi50_ids"].notna()
    df.loc[missing, "rgi_id"] = (
        df.loc[missing, "rgi50_ids"].map(first_rgi_id).str.replace("RGI50-", "RGI60-", n=1)
    )
    df["rgi_region"] = df["rgi_id"].map(rgi_region_int)
    df["role"] = df["rgi_region"].map(assign_role)
    df = df[~df["glacier_id"].isin(SKIP_IDS | DROP_IDS)].copy()
    if SKIP_EXTRA_ALPS:
        df = df[df["rgi_region"] != 11].copy()
    df = df[df["role"] != "unresolved"].copy()
    # Fetch holdouts and the new LORO region first; Alps (if any) last.
    df["_prio"] = np.where(df["role"] == "external_holdout", 0, 1)
    df["_prio"] = np.where(df["rgi_region"] == 10, -1, df["_prio"])
    return df.sort_values(["_prio", "n_obs"], ascending=[True, False]).drop(columns="_prio")


def fetch_all(targets: pd.DataFrame) -> None:
    done = set()
    if DAILY.exists():
        done = set(pd.read_csv(DAILY, usecols=["glacier_id"])["glacier_id"].unique())
        print(f"resuming: {len(done)} already fetched, {len(targets) - len(done)} remaining")
    failures = []
    for i, g in enumerate(targets.itertuples(), 1):
        if g.glacier_id in done:
            continue
        print(
            f"  [{i}/{len(targets)}] {g.glacier_name} (RGI-{g.rgi_region}, {g.role}) "
            f"{int(g.first_year)}-{int(g.last_year)} ...",
            flush=True,
        )
        df = fetch_glacier(g.lat, g.lon, int(g.first_year), int(g.last_year))
        if df is None:
            print("      FAILED", flush=True)
            failures.append(int(g.glacier_id))
            continue
        df["glacier_id"] = int(g.glacier_id)
        df["glacier_name"] = g.glacier_name
        df["role"] = g.role
        df["solid_precip_we_mm"] = df["precipitation_sum"] - df["rain_sum"]
        df = df.reindex(columns=COLUMNS)
        df.to_csv(DAILY, mode="a", header=not DAILY.exists(), index=False)
        print(f"      {len(df)} days", flush=True)
        time.sleep(PAUSE)
    if failures:
        print(f"FAILED glacier_ids: {failures}")


def build_annual(targets: pd.DataFrame) -> pd.DataFrame:
    if not DAILY.exists():
        raise SystemExit(f"{DAILY} missing — fetch did not write any climate")
    clim = pd.read_csv(DAILY, parse_dates=["time"])
    mb = pd.read_csv(ALT / "mass_balance.csv", low_memory=False)
    mb = mb[mb["glacier_id"].isin(targets["glacier_id"])].copy()
    mb = mb[mb["annual_balance"].notna()]
    for c in ("begin_date", "end_date"):
        mb[c] = pd.to_datetime(mb[c], errors="coerce")

    rgi = pd.read_csv(
        ALT / "rgi6_attributes.csv",
        usecols=["RGIId", "Zmed", "Area", "Slope", "Aspect"],
    ).set_index("RGIId")

    unc = pd.read_csv(DATA / "wgms_mass_balance_slim.csv")["annual_balance_unc"]
    fallback_unc = float(unc.median())
    print(f"uncertainty fallback: {fallback_unc:.3f}")

    have = set(clim["glacier_id"].unique())
    rows = []
    t = targets.set_index("glacier_id")
    for gid, meta in t.iterrows():
        if gid not in have:
            print(f"  skip {gid} {meta.glacier_name}: no climate")
            continue
        daily = clim[clim["glacier_id"] == gid].sort_values("time")
        lat, lon = float(meta.lat), float(meta.lon)
        rgi_id = meta.rgi_id
        static = {"elevation_med_m": np.nan, "area_km2": np.nan, "slope_deg": np.nan, "aspect_deg": np.nan}
        if isinstance(rgi_id, str) and rgi_id in rgi.index:
            rec = rgi.loc[rgi_id]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            static = {
                "elevation_med_m": rec["Zmed"],
                "area_km2": rec["Area"],
                "slope_deg": rec["Slope"],
                "aspect_deg": rec["Aspect"],
            }
        gmb = mb[mb["glacier_id"] == gid]
        for r in gmb.itertuples():
            begin, end = r.begin_date, r.end_date
            estimated = pd.isna(begin) or pd.isna(end)
            if estimated:
                begin, end = hydrological_year_window(int(r.year), lat)
            w = daily[(daily["time"] >= begin) & (daily["time"] <= end)]
            if w.empty:
                continue
            sw = w[w["time"].dt.month.isin(summer_months(lat))]
            year_unc = r.annual_balance_unc if pd.notna(getattr(r, "annual_balance_unc", np.nan)) else np.nan
            if pd.isna(year_unc):
                year_unc = fallback_unc
            rows.append({
                "glacier_id": int(gid),
                "glacier_name": meta.glacier_name,
                "year": int(r.year),
                "annual_balance": float(r.annual_balance),
                "begin_date": begin,
                "end_date": end,
                "pdd": float(w.loc[w["temp_c"] > 0, "temp_c"].sum()),
                "solid_precip_mm": float(w.loc[w["temp_c"] <= 1.0, "precipitation_sum"].sum()),
                "summer_temp_c": float(sw["temp_c"].mean()) if not sw.empty else np.nan,
                "solid_precip_v2": float(w["solid_precip_we_mm"].sum()),
                "rain_mm": float(w["rain_sum"].sum()),
                "total_precip_mm": float(w["precipitation_sum"].sum()),
                "n_days_in_window": int(len(w)),
                "window_estimated": bool(estimated),
                "role": meta.role,
                "lat": lat,
                "lon": lon,
                "rgi_id": rgi_id,
                "rgi_region": int(meta.rgi_region),
                **static,
                "annual_balance_unc": float(year_unc),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no expansion rows — climate windows missed every survey year")

    # Training glaciers with no geometry follow Hamaguri: do not impute; drop them.
    train = out["role"] == "training_population"
    null_geom = out["elevation_med_m"].isna() | out["area_km2"].isna()
    drop_train = out.loc[train & null_geom, "glacier_id"].unique()
    if len(drop_train):
        names = out.loc[out["glacier_id"].isin(drop_train), ["glacier_id", "glacier_name"]].drop_duplicates()
        print("Dropped training glaciers with no RGI geometry (no imputation):")
        print(names.to_string(index=False))
        out = out[~out["glacier_id"].isin(drop_train)].copy()

    out.to_csv(OUT, index=False)
    return out


def main() -> None:
    targets = load_targets()
    print("Leakage-safe expansion targets:")
    print(
        targets[["glacier_id", "glacier_name", "rgi_id", "rgi_region", "role", "n_obs", "first_year", "last_year"]]
        .to_string(index=False)
    )
    print(f"\n{len(targets)} glaciers  "
          f"train={int((targets.role=='training_population').sum())}  "
          f"holdout={int((targets.role=='external_holdout').sum())}")
    fetch_all(targets)
    out = build_annual(targets)
    print(f"\nwrote {OUT}: {len(out)} rows, {out['glacier_id'].nunique()} glaciers")
    summary = (
        out.drop_duplicates("glacier_id")
        .groupby(["role", "rgi_region"])
        .size()
        .rename("n_glaciers")
    )
    print(summary.to_string())
    print("\nDo not concat this into training for RGI 5/14/15/16/17/18/19.")
    print("Assemble with assemble_expanded_master.py, then quote RMSE only from")
    print("a notebook run on that table (see README.md).")


if __name__ == "__main__":
    main()
