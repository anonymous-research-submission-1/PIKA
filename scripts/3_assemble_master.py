"""Assemble the leakage-safe expanded master. Does not touch master_glacier_data_fixed.csv.

PIKA's forward pass never sees rgi_id / rgi_region. Those fields only group LORO
and holdout splits. The glacier number after the dot (e.g. .02059 vs .99999) is
unused. The first-order region (15, 19, …) *does* decide train vs holdout, so
new Himalaya / Antarctic / NZ / Greenland / holdout-Andes glaciers stay out of
training even though the network would ignore the ID.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ALT = DATA / "alt_datasets"
OUT = DATA / "master_glacier_data_expanded.csv"
OUT_FINAL = DATA / "master_glacier_data_expanded.csv"

DROP_IDS = [3454]  # Dokriani
ROUTE_TO_HOLDOUT = [3366, 3367, 3996, 3997]  # Johnsons, Hurd, Mera, Pokalde
# Regions that new glaciers must not enter as training (clean OOD + Himalaya).
# Original master already has 1 train glacier in 16 and 1 in 17; those stay.
NEW_NEVER_TRAIN = {5, 14, 15, 18, 19}

CANDIDATE_RGI = {
    3366: "RGI60-19.02059",  # Johnsons
    3367: "RGI60-19.02056",  # Hurd
    3996: "RGI60-15.03586",  # Mera
    3997: "RGI60-15.03416",  # Pokalde
}

COLS = [
    "glacier_id", "glacier_name", "year", "annual_balance",
    "begin_date", "end_date", "pdd", "solid_precip_mm", "summer_temp_c",
    "solid_precip_v2", "rain_mm", "total_precip_mm", "n_days_in_window",
    "window_estimated", "role", "lat", "lon", "rgi_id", "rgi_region",
    "elevation_med_m", "area_km2", "slope_deg", "aspect_deg", "annual_balance_unc",
]


def extract_rgi_region(rgi_id: pd.Series) -> pd.Series:
    return rgi_id.astype(str).str.extract(r"RGI\d+-(\d+)\.", expand=False).astype(float).astype("Int64")


def align(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "rgi_region" not in out.columns or out["rgi_region"].isna().all():
        out["rgi_region"] = extract_rgi_region(out["rgi_id"])
    else:
        missing = out["rgi_region"].isna() & out["rgi_id"].notna()
        out.loc[missing, "rgi_region"] = extract_rgi_region(out.loc[missing, "rgi_id"])
    for c in COLS:
        if c not in out.columns:
            out[c] = float("nan") if c not in ("glacier_name", "begin_date", "end_date", "role", "rgi_id") else pd.NA
    return out[COLS]


def fill_candidate_rgi(cand: pd.DataFrame) -> pd.DataFrame:
    rgi = pd.read_csv(ALT / "rgi6_attributes.csv", usecols=["RGIId", "Zmed", "Area", "Slope", "Aspect"])
    rgi = rgi.set_index("RGIId")
    cand = cand.copy()
    for gid, rid in CANDIDATE_RGI.items():
        m = cand["glacier_id"] == gid
        cand.loc[m, "rgi_id"] = rid
        if rid in rgi.index:
            rec = rgi.loc[rid]
            cand.loc[m, "elevation_med_m"] = rec["Zmed"]
            cand.loc[m, "area_km2"] = rec["Area"]
            cand.loc[m, "slope_deg"] = rec["Slope"]
            cand.loc[m, "aspect_deg"] = rec["Aspect"]
    return cand


def drop_train_null_geometry(df: pd.DataFrame) -> pd.DataFrame:
    static = ["elevation_med_m", "area_km2", "slope_deg", "aspect_deg"]
    train_ids = df.loc[df["role"] == "training_population", "glacier_id"].unique()
    dropped = []
    for gid in train_ids:
        g = df[df["glacier_id"] == gid]
        if g[static].isna().all().all():
            dropped.append(gid)
    if dropped:
        names = df.loc[df["glacier_id"].isin(dropped), ["glacier_id", "glacier_name"]].drop_duplicates()
        print("Dropped training glaciers with no RGI geometry:")
        print(names.to_string(index=False))
        df = df[~df["glacier_id"].isin(dropped)].copy()
    return df


def main() -> None:
    master = pd.read_csv(DATA / "master_glacier_data_fixed.csv")
    cand = pd.read_csv(DATA / "candidate_master_rows.csv")
    exp = pd.read_csv(DATA / "expansion_master_rows.csv")

    cand = cand[~cand["glacier_id"].isin(DROP_IDS)].copy()
    print("Dropped Dokriani (3454)")
    cand.loc[cand["glacier_id"].isin(ROUTE_TO_HOLDOUT), "role"] = "external_holdout"
    cand = fill_candidate_rgi(cand)

    # Expansion roles are already leakage-safe. Do not re-label master 16/17 trainers.
    df = pd.concat([align(master), align(cand), align(exp)], ignore_index=True)
    df = drop_train_null_geometry(df)

    train = df[df["role"] == "training_population"].drop_duplicates("glacier_id")
    hold = df[df["role"] == "external_holdout"].drop_duplicates("glacier_id")

    leak = train[train["rgi_region"].isin(NEW_NEVER_TRAIN)]
    if len(leak):
        raise SystemExit(
            "Leakage: training glaciers in holdout-only regions 5/14/15/18/19:\n"
            + leak[["glacier_id", "glacier_name", "rgi_region"]].to_string(index=False)
        )
    routed = set(ROUTE_TO_HOLDOUT)
    if routed & set(train["glacier_id"]):
        raise SystemExit("Leakage: Johnsons/Hurd/Mera/Pokalde still in training")
    if set(DROP_IDS) & set(df["glacier_id"]):
        raise SystemExit("Dokriani still present")

    df.to_csv(OUT, index=False)
    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FINAL, index=False)

    print(f"\nwrote {OUT}")
    print(f"wrote {OUT_FINAL}")
    print(f"rows={len(df)}  train={len(train)}  holdout={len(hold)}")
    print("\nTrain by RGI region:")
    print(train["rgi_region"].value_counts().sort_index().to_string())
    print("\nHoldout roster:")
    roster = hold[["glacier_id", "glacier_name", "rgi_region"]].sort_values(["rgi_region", "glacier_id"])
    print(roster.to_string(index=False))
    print("\nRGI id is not a model feature. Region number only affects splits.")
    print("Wrote data/master_glacier_data_expanded.csv")


if __name__ == "__main__":
    main()
