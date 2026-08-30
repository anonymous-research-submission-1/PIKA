"""Residual-MLP / Persistence++ control vs PIKA 16/32, 5 paired seeds.

Same inputs, residual target, optimizer, epoch budget, quantile/physics
regularizers as full PIKA 16/32. The Koopman operator is removed: each
horizon is decoded independently from z0 + control(c_t).

Does not retrain PIKA. Pairs against Full PIKA rows in
results/ablation_multiseed_rows.csv (byte-identical 5-seed IID table).

Run from repo root:
    python -u scripts/residual_mlp_control.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "pika.ipynb"
OUT_DIR = ROOT / "results"
SETUP_CELLS = (1, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 16, 18)
SEEDS = (0, 1, 2, 3, 4)


def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def load_notebook_setup(ns: dict) -> None:
    os.chdir(ROOT)
    os.environ.setdefault("MPLBACKEND", "Agg")
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    for idx in SETUP_CELLS:
        src = _src(cells[idx])
        print(f"[setup] exec cell {idx}: {src.strip().splitlines()[0][:70]}", flush=True)
        exec(compile(src, f"nb_cell_{idx}", "exec"), ns)
    missing = [
        name
        for name in (
            "all_train_seqs",
            "iid_temporal_seqs",
            "ood_test_seqs",
            "region_lookup",
            "TrainConfig",
            "PIKAv2",
            "QuantileDecoder",
            "DifferentiableTemperatureIndex",
            "train_pikav2",
            "predict_pikav2",
            "compute_metrics",
            "seqs_to_tensors",
            "Scaler",
            "quantile_loss",
            "HISTORY_LEN",
            "FORECAST_LEN",
            "CLIMATE_DIM",
            "device",
        )
        if name not in ns
    ]
    if missing:
        raise RuntimeError(f"notebook setup missing: {missing}")


def define_residual_mlp(ns: dict) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HISTORY_LEN = ns["HISTORY_LEN"]
    FORECAST_LEN = ns["FORECAST_LEN"]
    CLIMATE_DIM = ns["CLIMATE_DIM"]
    QuantileDecoder = ns["QuantileDecoder"]
    DifferentiableTemperatureIndex = ns["DifferentiableTemperatureIndex"]

    class ResidualMLP(nn.Module):
        """Persistence++: ŷ = last_b + f(history, per-step climate). No K.

        Encoder / control / decoder / quantile / physics match PIKA 16/32.
        Recurrence is not used: z_t = z0 + control(c_t) independently.
        """

        def __init__(
            self,
            latent_dim: int = 16,
            hidden: int = 32,
            history_len: int = HISTORY_LEN,
            forecast_len: int = FORECAST_LEN,
            climate_dim: int = CLIMATE_DIM,
            use_quantiles: bool = True,
            use_physics: bool = True,
        ):
            super().__init__()
            self.latent_dim = latent_dim
            self.hidden = hidden
            self.history_len = history_len
            self.forecast_len = forecast_len
            self.climate_dim = climate_dim
            self.use_quantiles = use_quantiles
            self.use_physics = use_physics

            enc_input_dim = history_len + history_len * climate_dim
            self.encoder = nn.Sequential(
                nn.Linear(enc_input_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, latent_dim),
                nn.LayerNorm(latent_dim),
            )
            self.climate_control = nn.Sequential(
                nn.Linear(climate_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, latent_dim),
            )
            self.point_decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            self.climate_skip = nn.Linear(climate_dim, 1)
            nn.init.zeros_(self.climate_skip.weight)
            nn.init.zeros_(self.climate_skip.bias)
            if use_quantiles:
                self.quantile_decoder = QuantileDecoder(latent_dim, hidden, forecast_len)
            if use_physics:
                self.physics = DifferentiableTemperatureIndex(hidden=32)

        def forward(self, hist_b, hist_c, fut_c, elevation=None, area=None, lat=None, residual_shrink=None):
            B = hist_b.shape[0]
            z0 = self.encoder(torch.cat([hist_b, hist_c.reshape(B, -1)], dim=-1))
            q_steps, d_steps = [], []
            for t in range(self.forecast_len):
                c_t = fut_c[:, t, :]
                z_t = z0 + self.climate_control(c_t)
                point_t = self.point_decoder(z_t).squeeze(-1) + torch.tanh(self.climate_skip(c_t)).squeeze(-1)
                d_steps.append(point_t)
                if self.use_quantiles:
                    q_raw = self.quantile_decoder(z_t)
                    med = q_raw[:, QuantileDecoder.N_QUANTILES // 2 : QuantileDecoder.N_QUANTILES // 2 + 1]
                    q_t = q_raw - med + point_t.unsqueeze(-1)
                    q_steps.append(q_t)
            out = {"delta": torch.stack(d_steps, dim=1)}
            if self.use_quantiles:
                out["quantiles"] = torch.stack(q_steps, dim=1)
            if self.use_physics and elevation is not None and area is not None:
                finite = torch.isfinite(elevation) & torch.isfinite(area)
                if finite.any():
                    out["physics"] = self.physics(
                        fut_c,
                        torch.nan_to_num(elevation, 0.0),
                        torch.nan_to_num(area, 0.0),
                    )
            return out

        def lyapunov_loss(self):
            return torch.zeros((), device=self.climate_skip.weight.device)

        def spectral_info(self):
            return {
                "spectral_radius": float("nan"),
                "eig_min": float("nan"),
                "eig_max": float("nan"),
                "eig_mean": float("nan"),
                "eigenvalues": [],
            }

    ns["ResidualMLP"] = ResidualMLP

    pika = ns["PIKAv2"](latent_dim=16, hidden=32)
    mlp = ResidualMLP(latent_dim=16, hidden=32)
    ns["_n_params_pika"] = sum(p.numel() for p in pika.parameters())
    ns["_n_params_mlp"] = sum(p.numel() for p in mlp.parameters())
    print(
        f"Params  PIKA 16/32 = {ns['_n_params_pika']:,}  "
        f"ResidualMLP = {ns['_n_params_mlp']:,}  "
        f"(gap is OrthogonalLayer + log_eigs)",
        flush=True,
    )


def train_residual_mlp(ns: dict, train_seqs, cfg):
    """Same loop as train_pikav2, ResidualMLP instead of PIKAv2, no Lyapunov."""
    import numpy as np
    import torch
    import torch.nn as nn

    device = ns["device"]
    CLIMATE_DIM = ns["CLIMATE_DIM"]
    FORECAST_LEN = ns["FORECAST_LEN"]
    ResidualMLP = ns["ResidualMLP"]
    seqs_to_tensors = ns["seqs_to_tensors"]
    Scaler = ns["Scaler"]
    quantile_loss = ns["quantile_loss"]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.use_physics:
        train_seqs = [
            s for s in train_seqs if np.isfinite(s["elevation"]) and np.isfinite(s["area"])
        ]
        if not train_seqs:
            raise ValueError("No training sequences remain after dropping missing geometry")

    hist_b, hist_c, fut_c, target_b = seqs_to_tensors(train_seqs)
    b_scaler = Scaler(hist_b, scalar=True)
    c_all = torch.cat([hist_c, fut_c], dim=1)
    c_scaler = Scaler(c_all.reshape(-1, CLIMATE_DIM), scalar=False)

    hist_b_n = b_scaler.transform(hist_b).to(device)
    hist_c_n = c_scaler.transform(hist_c.reshape(-1, CLIMATE_DIM)).reshape(hist_c.shape).to(device)
    fut_c_n = c_scaler.transform(fut_c.reshape(-1, CLIMATE_DIM)).reshape(fut_c.shape).to(device)
    target_b_n = b_scaler.transform(target_b).to(device)
    hist_b_n = torch.nan_to_num(hist_b_n, 0.0)
    hist_c_n = torch.nan_to_num(hist_c_n, 0.0)
    fut_c_n = torch.nan_to_num(fut_c_n, 0.0)
    target_b_n = torch.nan_to_num(target_b_n, 0.0)
    last_b_n = hist_b_n[:, -1:].expand_as(target_b_n)
    target_delta_n = target_b_n - last_b_n
    b_scaler.to(device)
    c_scaler.to(device)

    model = ResidualMLP(
        latent_dim=cfg.latent_dim,
        hidden=cfg.hidden,
        use_quantiles=cfg.use_quantiles,
        use_physics=cfg.use_physics,
    ).to(device)

    elev_t, area_t = None, None
    if cfg.use_physics:
        elevations = np.array([s["elevation"] for s in train_seqs], dtype=np.float32)
        areas = np.array([s["area"] for s in train_seqs], dtype=np.float32)
        model.physics.set_normalization(
            float(elevations.mean()),
            float(max(elevations.std(), 1e-6)),
            float(areas.mean()),
            float(max(areas.std(), 1e-6)),
        )
        elev_t = torch.tensor(elevations, device=device)
        area_t = torch.tensor(areas, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    horizon_weights = torch.tensor(
        [cfg.horizon_discount ** i for i in range(FORECAST_LEN)],
        device=device,
    ).unsqueeze(0)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(hist_b_n, hist_c_n, fut_c_n, elev_t, area_t)
        delta_pred = out["delta"]
        main_loss = (((delta_pred - target_delta_n) ** 2) * horizon_weights).mean()
        if cfg.use_quantiles and "quantiles" in out:
            main_loss = main_loss + 0.3 * quantile_loss(out["quantiles"], target_delta_n)

        physics_loss = torch.tensor(0.0, device=device)
        warmup_frac = min(1.0, epoch / (cfg.epochs * 0.1)) if cfg.use_physics else 0.0
        if cfg.use_physics and "physics" in out:
            physics_loss_raw = ((out["delta"] - out["physics"]) ** 2).mean()
            physics_loss = (
                physics_loss_raw * warmup_frac
                if torch.isfinite(physics_loss_raw)
                else torch.tensor(0.0, device=device)
            )

        total_loss = main_loss + cfg.physics_weight * physics_loss
        if not torch.isfinite(total_loss):
            optimizer.zero_grad()
            scheduler.step()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            mse = ((out["delta"] - target_delta_n) ** 2).mean().item()
        rec = {
            "epoch": epoch,
            "loss": float(total_loss.item()) if torch.isfinite(total_loss) else float("nan"),
            "mse": mse,
            "physics": float(physics_loss.item()),
        }
        history.append(rec)
        if epoch % 200 == 0 or epoch == cfg.epochs - 1:
            print(
                f"  Epoch {epoch:4d} | loss={rec['loss']:.4f} mse={mse:.4f} phys={rec['physics']:.4f}",
                flush=True,
            )
    return model, (b_scaler, c_scaler), history


def paired_ci(d, n_boot=20000, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = d[rng.integers(0, n, n)].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def tail_slope(history, n_tail=100):
    import numpy as np

    mse = np.array([r["mse"] for r in history], dtype=float)
    tail = mse[-n_tail:]
    x = np.arange(len(tail), dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0])
    return float(tail[-1]), slope


def metric_pack(m: dict) -> dict:
    return {
        "pooled_rmse": m["pooled_rmse"],
        "region_balanced_rmse": m["region_balanced_rmse"],
        "pooled_wmape": m.get("pooled_wmape"),
        "pooled_mae": m.get("pooled_mae"),
        "n_glaciers": m["n_glaciers"],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import numpy as np
    import pandas as pd

    t0 = time.time()
    ns: dict = {"__name__": "__main__"}
    load_notebook_setup(ns)
    define_residual_mlp(ns)

    TrainConfig = ns["TrainConfig"]
    predict_pikav2 = ns["predict_pikav2"]
    compute_metrics = ns["compute_metrics"]
    all_train_seqs = ns["all_train_seqs"]
    iid_temporal_seqs = ns["iid_temporal_seqs"]
    ood_test_seqs = ns["ood_test_seqs"]
    region_lookup = ns["region_lookup"]

    pika_rows = pd.read_csv(OUT_DIR / "ablation_multiseed_rows.csv")
    pika_full = pika_rows[pika_rows["variant"] == "Full PIKA"].set_index("seed")
    if list(pika_full.index) != list(SEEDS):
        raise RuntimeError(f"unexpected PIKA seeds in ablation CSV: {list(pika_full.index)}")

    rows = []
    for seed in SEEDS:
        cfg = TrainConfig(
            epochs=1200,
            lr=1e-3,
            latent_dim=16,
            hidden=32,
            use_quantiles=True,
            use_physics=True,
            seed=seed,
        )
        print(f"\n=== ResidualMLP seed {seed} ===", flush=True)
        t_seed = time.time()
        model, scalers, history = train_residual_mlp(ns, all_train_seqs, cfg)
        iid = predict_pikav2(model, scalers, iid_temporal_seqs)
        ood = predict_pikav2(model, scalers, ood_test_seqs)
        m_iid = compute_metrics(iid, region_lookup)
        m_ood = compute_metrics(ood, region_lookup)
        final_mse, slope = tail_slope(history)
        pika_rmse = float(pika_full.loc[seed, "pooled_rmse"])
        row = {
            "seed": seed,
            "variant": "Residual MLP (no K)",
            "n_params": ns["_n_params_mlp"],
            "n_params_pika": ns["_n_params_pika"],
            **{f"iid_{k}": v for k, v in metric_pack(m_iid).items()},
            **{f"ood_{k}": v for k, v in metric_pack(m_ood).items()},
            "pika_iid_pooled_rmse": pika_rmse,
            "delta_vs_pika_iid": m_iid["pooled_rmse"] - pika_rmse,
            "final_mse": final_mse,
            "tail_slope_per_epoch": slope,
            "seconds": time.time() - t_seed,
        }
        rows.append(row)
        wmape = m_iid.get("pooled_wmape")
        wmape_s = f"{wmape:.1%}" if wmape is not None else "n/a"
        print(
            f"  IID RMSE={m_iid['pooled_rmse']:.4f}  PIKA={pika_rmse:.4f}  "
            f"Δ={row['delta_vs_pika_iid']:+.4f}  WMAPE={wmape_s}  "
            f"OOD RMSE={m_ood['pooled_rmse']:.4f}  slope={slope:.2e}/epoch",
            flush=True,
        )
        del model

    df = pd.DataFrame(rows)
    d = df["delta_vs_pika_iid"].to_numpy()
    lo, hi = paired_ci(d)
    summary = pd.DataFrame(
        [
            {
                "variant": "Residual MLP (no K)",
                "mean_delta": float(d.mean()),
                "sd_delta": float(d.std(ddof=1)),
                "ci_lo": lo,
                "ci_hi": hi,
                "resolved": bool(lo > 0 or hi < 0),
                "wins_vs_pika": int((d < 0).sum()),
                "n_seeds": len(d),
                "mean_iid_rmse": float(df["iid_pooled_rmse"].mean()),
                "mean_ood_rmse": float(df["ood_pooled_rmse"].mean()),
                "n_params": int(df["n_params"].iloc[0]),
                "n_params_pika": int(df["n_params_pika"].iloc[0]),
            }
        ]
    )

    rows_path = OUT_DIR / "residual_mlp_multiseed_rows.csv"
    sum_path = OUT_DIR / "residual_mlp_multiseed_summary.csv"
    df.to_csv(rows_path, index=False)
    summary.to_csv(sum_path, index=False)

    print("\n" + "=" * 70)
    print("Residual MLP vs Full PIKA (paired IID pooled RMSE)")
    print(f"  mean Δ (MLP − PIKA) = {d.mean():+.4f}  sd={d.std(ddof=1):.4f}")
    print(f"  95% paired CI       = [{lo:+.4f}, {hi:+.4f}]")
    print(f"  MLP lower RMSE on   {int((d < 0).sum())}/{len(d)} seeds")
    if lo <= 0 <= hi:
        print("  CI includes 0: Koopman has no detectable IID effect vs residual MLP.")
    elif hi < 0:
        print("  CI excludes 0: residual MLP is better (Koopman hurts).")
    else:
        print("  CI excludes 0: PIKA is better (Koopman helps).")
    print(f"  wrote {rows_path.name} and {sum_path.name}")
    print(f"  elapsed {time.time() - t0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
