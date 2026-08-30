# Data

Three tables. `master_glacier_data_expanded.csv` is the only one the notebook
requires; the other two are inputs to the assembly step.

| File | Rows | What |
|---|---|---|
| `master_glacier_data_expanded.csv` | 5,033 | One row per glacier-year: annual mass balance, climate covariates, static attributes, and the train/holdout role. **This is what the notebook loads.** |
| `candidate_master_rows.csv` | 93 | Screened candidate glaciers, kept separate so the frozen split is auditable. |
| `wgms_mass_balance_slim.csv` | 8,944 | WGMS balance records with reported uncertainty, used for optional error-weighting checks. |

## Columns that matter

- `annual_balance` — surface mass balance in m w.e., the prediction target.
- `pdd` — positive degree-day sum over the glacier's reported survey window.
- `solid_precip_mm` — solid (snow) precipitation over the same window.
- `summer_temp_c` — mean summer temperature.
- `elevation_med_m`, `area_km2` — static attributes; used **only** by the
  training-time temperature-index term, never by the forecast path.
- `role` — `training_population` or `external_holdout`.
- `rgi_region` — first-order RGI region, used to define splits, never as a feature.
- `window_estimated` — flags glacier-years whose survey dates were inferred
  rather than reported. The strict test windows exclude these.

## Provenance

- **Mass balance** — World Glacier Monitoring Service (WGMS), glaciological method.
- **Static attributes** — Randolph Glacier Inventory (RGI) 6.0.
- **Climate** — ERA5-Land daily reanalysis, aggregated per glacier over each
  reported survey window and retrieved through the Open-Meteo archive API.

## Split rule

Training is restricted to regions that are not held out. No Antarctic (RGI 19),
Himalayan (14, 15), Greenland (5), or New Zealand (18) glacier appears in
training, and the newly added tropical and southern Andes glaciers are holdout.
This is enforced in `scripts/3_assemble_master.py`, not in the notebook, so the
split cannot drift between runs.

## Rebuilding

The committed table is sufficient to reproduce every number. To rebuild it
from source, run `scripts/1_fetch_climate.py` → `2_build_wgms_rows.py` →
`3_assemble_master.py`. Step 1 hits a public API and takes several hours.
