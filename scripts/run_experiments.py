"""Execute notebooks/pika.ipynb end to end and refresh figures/.

Runs every cell in place (outputs are written back into the notebook) and
copies the seven generated panels into figures/. The glacier map is produced
separately by plot_glacier_map.py and is not touched here.

    python scripts/run_experiments.py

CPU-only is fine; a full run takes roughly 1.5-2 hours.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "pika.ipynb"
FIG_DIR = ROOT / "figures"
FIG_NAMES = [
    "fig1_training_curves.png",
    "fig2_model_comparison.png",
    "fig3_per_horizon.png",
    "fig4_ablation.png",
    "fig5_uncertainty_fan.png",
    "fig6_loro.png",
    "fig7_spectral.png",
]


class StreamingClient(NotebookClient):
    """Mirror cell stdout to the terminal while the notebook runs."""

    def process_message(self, msg, cell, cell_index):
        msg_type = msg.get("msg_type")
        content = msg.get("content", {})
        if msg_type == "stream":
            print(content.get("text", ""), end="", flush=True)
        elif msg_type == "error":
            print("\n".join(content.get("traceback", [])), flush=True)
        return super().process_message(msg, cell, cell_index)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(ROOT)
    print(f"notebook={NB_PATH.relative_to(ROOT)}", flush=True)

    nb = nbformat.read(NB_PATH, as_version=4)
    total = len(nb.cells)
    client = StreamingClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    original = client.async_execute_cell

    async def async_execute_cell(cell, cell_index, **kwargs):
        head = "".join(cell.source)[:80].replace("\n", " ")
        head = head.encode("ascii", "replace").decode("ascii")
        print(f"\n=== cell {cell_index + 1}/{total}  {head} ===", flush=True)
        result = await original(cell, cell_index, **kwargs)
        nbformat.write(nb, NB_PATH)   # checkpoint after every cell
        return result

    client.async_execute_cell = async_execute_cell
    client.execute()
    nbformat.write(nb, NB_PATH)
    print(f"\nwrote {NB_PATH.relative_to(ROOT)}", flush=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in FIG_NAMES:
        produced = ROOT / name
        if produced.exists():
            shutil.move(str(produced), FIG_DIR / name)
            print(f"figures/{name}", flush=True)


if __name__ == "__main__":
    main()
