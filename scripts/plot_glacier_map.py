"""World map of glaciers in the expanded PIKA protocol.

101 training / 20 holdout. Close pairs are spread on the *main* map only.
Roles come from master_glacier_data_expanded.csv (already leakage-safe).
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
CACHE = OUT / "_ne_110m_land.geojson"
OUT.mkdir(parents=True, exist_ok=True)

NE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_land.geojson"
)

HOLDOUT_KEY = [
    (1629, "Mittivakkat"),
    (3350, "Freya"),
    (2921, "Chhota Shigri"),
    (3987, "Parlung No. 94"),
    (3996, "Mera"),
    (3997, "Pokalde"),
    (1624, "Antizana"),
    (2721, "Conejeras"),
    (226, "Yanamarey"),
    (35232, "Artesonraju"),
    (2667, "Charquini Sur"),
    (2000, "Martial Este"),
    (3902, "Conconta Norte"),
    (3903, "Brown Superior"),
    (3972, "Mocho-Choshuenco SE"),
    (1597, "Brewster"),
    (1538, "Rolleston"),
    (2665, "Bahía del Diablo"),
    (3366, "Johnsons"),
    (3367, "Hurd"),
]

# Main-map display positions. True coords stay in the table / southern inset.
MAIN_XY = {
    1629: (-37.80, 65.70),
    3350: (-20.82, 74.38),
    2921: (77.52, 32.23),
    3987: (107.00, 34.00),
    3996: (86.88, 27.71),
    3997: (58.00, 18.00),
    1624: (-78.15, -0.47),
    2721: (-75.37, 8.50),
    226: (-77.27, -9.65),
    35232: (-90.00, -12.00),
    2667: (-68.11, -16.30),
    2000: (-68.40, -54.78),
    3902: (-69.64, -29.98),
    3903: (-54.00, -30.00),
    3972: (-72.02, -39.95),
    1597: (169.44, -44.07),
    1538: (178.00, -39.50),
    2665: (-38.00, -63.82),
    3366: (-60.35, -62.67),
    3367: (-110.00, -62.67),
}

NUM_XYTEXT = {gid: (8, 8) for gid, _ in HOLDOUT_KEY}
NUM_XYTEXT.update({
    1629: (7, 6),
    3350: (8, 8),
    3997: (8, -9),
    226: (8, 8),
    35232: (-10, -8),
    2667: (8, -8),
    2000: (9, 9),
    3903: (8, -8),
    1597: (7, -8),
    1538: (8, 8),
    2665: (8, -8),
    3367: (8, -9),
})

INSETS = [
    dict(title="Western N. America", lon=(-152, -113), lat=(46, 64)),
    dict(title="Alps", lon=(5.5, 14.5), lat=(44.2, 47.8)),
    dict(title="Scandinavia / Svalbard", lon=(4, 32), lat=(58, 81)),
    dict(title="S. Andes / Antarctic Pen.", lon=(-72, -54), lat=(-67, -52)),
]


def load_glaciers() -> pd.DataFrame:
    path = DATA / "master_glacier_data_expanded.csv"
    df = pd.read_csv(path)
    return (
        df.drop_duplicates("glacier_id")[["glacier_id", "glacier_name", "lat", "lon", "role"]]
        .dropna(subset=["lat", "lon"])
        .copy()
    )


def land_geojson() -> dict:
    if not CACHE.exists():
        urllib.request.urlretrieve(NE_URL, CACHE)
    return json.loads(CACHE.read_text(encoding="utf-8"))


def polygons_to_patches(gj: dict) -> list:
    patches = []
    for feat in gj["features"]:
        geom = shape(feat["geometry"])
        if geom.geom_type == "Polygon":
            geoms = [geom]
        elif geom.geom_type == "MultiPolygon":
            geoms = list(geom.geoms)
        else:
            continue
        for poly in geoms:
            coords = np.asarray(poly.exterior.coords)
            if np.nanmax(coords[:, 0]) - np.nanmin(coords[:, 0]) > 300:
                continue
            patches.append(mpatches.Polygon(coords, closed=True))
    return patches


def add_land(ax, land_patches) -> None:
    ax.set_facecolor("#F4F6F8")
    ax.add_collection(
        PatchCollection(
            land_patches, facecolor="#E6E2D8", edgecolor="#9A9488",
            linewidths=0.3, zorder=1,
        )
    )


def style_axes(ax, *, tick_fs: float, equal: bool = True) -> None:
    if equal:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.55)
    ax.tick_params(labelsize=tick_fs, colors="#444444", length=3)
    ax.grid(True, color="#D0D4D8", linewidth=0.3, zorder=0)


def hold_display(hold: pd.DataFrame, *, main: bool) -> pd.DataFrame:
    h = hold.copy()
    h["plot_lon"] = h["lon"]
    h["plot_lat"] = h["lat"]
    if main:
        for gid, (lon, lat) in MAIN_XY.items():
            m = h["glacier_id"] == gid
            h.loc[m, "plot_lon"] = lon
            h.loc[m, "plot_lat"] = lat
    else:
        # Southern inset: pull Hurd off Johnsons by ~0.9°.
        m = h["glacier_id"] == 3367
        h.loc[m, "plot_lon"] = h.loc[m, "lon"] - 1.8
        h.loc[m, "plot_lat"] = h.loc[m, "lat"] - 1.3
    return h


def scatter_train_hold(ax, train, hold_xy, *, train_s, hold_s) -> None:
    ax.scatter(
        train["lon"], train["lat"],
        s=train_s, c="#3D5A80", marker="o",
        edgecolors="white", linewidths=0.35, zorder=3,
    )
    ax.scatter(
        hold_xy["plot_lon"], hold_xy["plot_lat"],
        s=hold_s, c="#C1121F", marker="^",
        edgecolors="white", linewidths=0.45, zorder=4,
    )


def number_holdouts(ax, hold_xy: pd.DataFrame) -> None:
    id_to_n = {gid: i + 1 for i, (gid, _) in enumerate(HOLDOUT_KEY)}
    for _, row in hold_xy.iterrows():
        gid = int(row["glacier_id"])
        n = id_to_n[gid]
        dx, dy = NUM_XYTEXT[gid]
        ax.annotate(
            str(n),
            xy=(row["plot_lon"], row["plot_lat"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color="#6B1D22",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#C1121F", linewidth=0.5),
            zorder=6,
        )


def main() -> None:
    g = load_glaciers()
    train = g[g["role"] == "training_population"]
    hold_raw = g[g["role"] == "external_holdout"]
    hold_main = hold_display(hold_raw, main=True)
    hold_true = hold_display(hold_raw, main=False)
    land = polygons_to_patches(land_geojson())

    fig = plt.figure(figsize=(11.6, 7.8), dpi=150)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 4, height_ratios=[2.85, 1.55], width_ratios=[1, 1, 1, 1],
        hspace=0.38, wspace=0.28,
    )
    ax_main = fig.add_subplot(gs[0, :3])
    ax_key = fig.add_subplot(gs[0, 3])
    ax_ins = [fig.add_subplot(gs[1, i]) for i in range(4)]

    add_land(ax_main, land)
    scatter_train_hold(ax_main, train, hold_main, train_s=16, hold_s=50)
    number_holdouts(ax_main, hold_main)
    ax_main.set_xlim(-180, 180)
    ax_main.set_ylim(-75, 85)
    ax_main.set_xlabel("Longitude")
    ax_main.set_ylabel("Latitude")
    ax_main.set_xticks(range(-180, 181, 60))
    ax_main.set_yticks(range(-60, 81, 30))
    style_axes(ax_main, tick_fs=8, equal=False)

    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)
    ax_key.axis("off")
    ax_key.text(
        0.0, 1.0,
        f"Training  n={len(train)}   (blue circles)\n"
        f"Holdout   n={len(hold_raw)}   (red triangles)",
        transform=ax_key.transAxes, va="top", ha="left",
        fontsize=7.5, color="#333333", linespacing=1.45, fontweight="bold",
    )
    ax_key.text(
        0.0, 0.88, "Holdouts",
        transform=ax_key.transAxes, va="top", ha="left",
        fontsize=8.5, fontweight="bold", color="#333333",
    )
    col_a = HOLDOUT_KEY[:10]
    col_b = HOLDOUT_KEY[10:]
    lines_a = "\n".join(f"{i+1:>2d}  {n}" for i, (_, n) in enumerate(col_a))
    lines_b = "\n".join(f"{i+11:>2d}  {n}" for i, (_, n) in enumerate(col_b))
    ax_key.text(
        0.0, 0.83, lines_a,
        transform=ax_key.transAxes, va="top", ha="left",
        fontsize=6.6, family="DejaVu Sans", linespacing=1.48, color="#333333",
    )
    ax_key.text(
        0.52, 0.83, lines_b,
        transform=ax_key.transAxes, va="top", ha="left",
        fontsize=6.6, family="DejaVu Sans", linespacing=1.48, color="#333333",
    )
    ax_key.text(
        0.0, 0.04,
        "Close pairs are shifted on the\n"
        "world map only: Hurd (20),\n"
        "Pokalde (6), Brown Superior\n"
        "(14), Artesonraju (10),\n"
        "Rolleston (17). Southern inset\n"
        "is near-true (Hurd 1.8° off\n"
        "Johnsons). No Himalaya/\n"
        "Antarctic glacier in training.",
        transform=ax_key.transAxes, va="bottom", ha="left",
        fontsize=6.1, color="#555555",
    )

    for ax, spec in zip(ax_ins, INSETS):
        add_land(ax, land)
        use_hold = hold_true if "Antarctic" in spec["title"] else hold_true
        scatter_train_hold(ax, train, use_hold, train_s=28, hold_s=44)
        ax.set_xlim(*spec["lon"])
        ax.set_ylim(*spec["lat"])
        style_axes(ax, tick_fs=6)
        n_train = int((
            train["lon"].between(*spec["lon"]) & train["lat"].between(*spec["lat"])
        ).sum())
        n_hold = int((
            hold_raw["lon"].between(*spec["lon"]) & hold_raw["lat"].between(*spec["lat"])
        ).sum())
        extra = f", {n_hold} holdout" if n_hold else ""
        ax.set_title(f"{spec['title']}\n(n={n_train} training{extra})", fontsize=7.5, pad=4)
        ax.set_xlabel("Longitude", fontsize=7)
        ax.set_ylabel("Latitude", fontsize=7)

    fig.suptitle("Glaciers in the expanded PIKA protocol", fontsize=11, y=0.995)
    png = OUT / "fig_glacier_map.png"
    pdf = OUT / "fig_glacier_map.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"train={len(train)}  holdout={len(hold_raw)}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
