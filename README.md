# PIKA: A Physics-Informed Koopman Autoencoder for Glacier Mass Balance Forecasting

Code, data and evaluation for the paper. PIKA forecasts five years of annual
glacier surface mass balance from tabular climate and inventory records, with
no satellite imagery. The headline model has **5,852 parameters**.

Everything reported in the paper is reproduced by one command.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_experiments.py       # ~1.5-2 h on CPU
```

That executes `notebooks/pika.ipynb` end to end and refreshes `figures/`. The
notebook ships with its stored outputs, so every table can be read without
running anything.

---

## Layout

```
data/        annual glacier-year table + provenance (see data/README.md)
notebooks/   pika.ipynb  — model, baselines, evaluation, ablation, UQ
scripts/     1-3_*.py    — rebuild the dataset from source (optional)
             run_experiments.py       — execute the notebook, refresh figures
             residual_mlp_control.py  — ablate the Koopman operator (~90 s)
             plot_glacier_map.py, plot_model_comparison.py
results/     paired ablation and Koopman-control tables
figures/     figures used in the paper
```

---

## Data

121 glaciers: **101 training, 20 held out**, 5,033 glacier-years.

- **Mass balance** — WGMS, glaciological method
- **Static attributes** — RGI 6.0
- **Climate** — ERA5-Land daily reanalysis, aggregated over each glacier's
  reported survey window

Training uses years ≤ 2018 (2,259 overlapping windows). Evaluation is three
tiers: **IID temporal** (65 glaciers, last strict 2019+ window), **OOD spatial**
(15 of the 20 holdouts with a valid window), and **leave-one-region-out** over
10 RGI regions. No Antarctic, Himalayan, Greenland or New Zealand glacier
appears in training. The split is enforced in `scripts/3_assemble_master.py`,
not in the notebook, so it cannot drift between runs.

Forecasts are **conditional on observed climate**: future covariates come from
reanalysis, not from a climate prediction.

---

## Results

Pooled RMSE in m w.e. IID entries for the neural models are mean ± sd over five
seeds; the rest are single deterministic fits. OOD and LORO use seed 0.

| Model | IID RMSE | OOD RMSE | OOD WMAPE | LORO RMSE |
|---|---:|---:|---:|---:|
| PIKA (16/32) | 0.845 ± 0.046 | 0.901 | 69.5% | 0.80 |
| PIKA (48/96) | 0.837 ± 0.030 | **0.769** | **55.3%** | – |
| LSTM (51,265 params) | **0.822 ± 0.033** | 0.933 | 62.0% | **0.61** |
| XGBoost | 0.882 | 0.941 | 65.0% | – |
| Random forest | 1.008 | 1.184 | 79.1% | – |
| PDD/TI | 0.855 | 0.949 | 71.9% | – |
| Naive persistence | 1.403 | 0.949 | 71.9% | – |

**In distribution**, PIKA is statistically indistinguishable from an LSTM
8.8× its size: the paired 95% CI for the difference is [−0.085, +0.041] and
contains zero, though the LSTM has the lower mean and wins 4/5 seeds.

**Out of distribution**, PIKA has the lower pooled RMSE (0.901 vs 0.933) but the
higher WMAPE (69.5% vs 62.0%). The two metrics disagree because PIKA has the
smaller worst case (1.62 vs 2.55 m w.e. on the hardest holdout glacier) while
the LSTM is better on the typical one. Report the metric alongside the ranking.

**Transfer is not achieved.** Under leave-one-region-out the LSTM is stronger
(0.61 vs 0.80; bootstrap CI for the difference [−0.318, −0.078] excludes zero),
and PIKA wins only 1 of 10 regions. PIKA received 800 epochs per fold against
the LSTM's 600, so this is not a compute artefact.

### Uncertainty

One glacier-split 80% interval, calibrated on 39 in-distribution glaciers and
evaluated on 26. Raw quantile coverage is 59.2% at width 1.281 m w.e.;
split-conformal adjustment reaches 83.8% at width 2.447. Conformal coverage is
validated **in distribution only**.

### What the components contribute

Every variant retrained on all five seeds and differenced against the full
model at the same seed (`results/ablation_multiseed_summary.csv`):

| Component removed | mean Δ RMSE | 95% CI |
|---|---:|:---:|
| Residual parameterisation | +0.037 | [−0.016, +0.074] |
| Quantile head | −0.001 | [−0.035, +0.029] |
| Capacity (48/96 instead of 16/32) | −0.008 | [−0.035, +0.020] |
| Temperature-index prior | −0.005 | [−0.018, +0.005] |
| Spectral-floor term | 0.000 | [0.000, 0.000] |

**Every interval contains zero: no individual component has a detectable effect
on in-distribution accuracy.** The spectral-floor term is exactly zero because
the eigenvalue parameterisation already satisfies its constraint, so the penalty
never activates.

Replacing the Koopman operator with a plain residual MLP of similar size
(`scripts/residual_mlp_control.py`) gives Δ = −0.040, CI [−0.119, +0.018] — also
indistinguishable. We cannot show that the operator adds in-distribution
accuracy; what it adds is a readable spectrum.

### Learned dynamics

Eigenvalues of the one-year propagator are real and positive by construction,
so the spectrum is a set of decay rates with no oscillatory modes. The trained
spectral radius is 0.755, an e-folding time of ~3.6 years.

---

## Rebuilding the dataset

The committed table is sufficient to reproduce every number. To rebuild from
source, run `scripts/1_fetch_climate.py` → `2_build_wgms_rows.py` →
`3_assemble_master.py`. Step 1 queries a public API and takes several hours.

## Reproducibility

`requirements.txt` pins the versions used for the reported run. Neural results
drift by roughly 0.005 RMSE across library versions; the deterministic
baselines reproduce exactly.

## License

MIT — see `LICENSE`.
