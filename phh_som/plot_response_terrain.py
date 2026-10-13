#!/usr/bin/env python3
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from pipeline import (ALL_CONTRASTS, CLIP_Z, CONTRASTS, DATA, FIGURES, MM, ROOT,
                      VIRUS_ORDER, VIRUSES, out_suffix, save_figure,
                      set_rcparams)
from plot_som_map import (Grid, grid_dims,
                                 overlay_grid, surface)

CONDS = ["untreated24", "untreated72", "Ruxo24", "Ruxo72"]
COND_TITLE = {"untreated24": "Untreated, 24 hpi",
              "untreated72": "Untreated, 72 hpi",
              "Ruxo24": "Ruxolitinib, 24 hpi",
              "Ruxo72": "Ruxolitinib, 72 hpi"}

FIG_W = 183
ROW_H, FIG_MARGINS = 35, 48

PAPER_ORDER = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]


def virus_colors():
    return json.loads((ROOT / "config" / "colors.json").read_text())["viruses"]


def load_codebook(data_dir, tag):
    cb = pd.read_csv(data_dir / f"som_codebook{out_suffix(tag)}.tsv",
                     sep="\t", index_col=0)
    if cb.shape[1] == len(CONTRASTS):
        return cb, VIRUSES, CONDS
    if cb.shape[1] == len(ALL_CONTRASTS):
        return cb, PAPER_ORDER, CONDS
    for hpi in ("24", "72"):
        conds = [f"untreated{hpi}", f"Ruxo{hpi}"]
        if list(cb.columns) == [f"{v}_{c}" for v in VIRUS_ORDER
                                for c in conds]:
            return cb, PAPER_ORDER, conds
    raise ValueError(
        f"the response terrain is defined for the {len(CONTRASTS)} WNV/YFV, "
        f"the {len(ALL_CONTRASTS)} all-virus or the 10 single-hpi all-virus "
        f"conditions; this codebook has {cb.shape[1]} columns")


def shared_scale(cb):
    z = min(float(np.nanmax(np.abs(cb.to_numpy()))), CLIP_Z)
    return TwoSlopeNorm(vmin=-z, vcenter=0, vmax=z), np.linspace(-z, z, 21)


def terrain_panel(fig, slot, cb, grid, virus, cond, norm, levels,
                  column_header, colors, first_cond=CONDS[0]):
    ax = fig.add_subplot(slot)
    surface(ax, cb[f"{virus}_{cond}"], grid, "RdBu_r", norm, levels)
    overlay_grid(ax, grid)
    grid.frame(ax)
    if column_header:
        ax.set_title(COND_TITLE[cond], fontsize=7, pad=2)
    if cond == first_cond:
        ax.text(grid.xlim[0] - 0.25, sum(grid.ylim) / 2, virus,
                rotation=90, ha="right", va="center", fontsize=8,
                fontweight="bold", color=colors[virus], clip_on=False)
    return ax


def main():
    ap = argparse.ArgumentParser(description='the response terrain - codebook-weight surfaces per condition.')
    ap.add_argument("--tag", default="isg",
                    help="gene-set tag of the trained map (8-condition "
                         "WNV/YFV, 20-condition all-virus or 10-condition "
                         "single-hpi all-virus codebook)")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=FIGURES / "plot_response_terrain")
    args = ap.parse_args()

    set_rcparams()
    cb, viruses, conds = load_codebook(args.data_dir, args.tag)
    colors = virus_colors()
    grid = Grid(*grid_dims(args.data_dir, args.tag))
    norm, levels = shared_scale(cb)

    fig_h = (FIG_MARGINS + ROW_H * len(viruses)) * MM
    fig = plt.figure(figsize=(FIG_W * MM, fig_h))
    gs = fig.add_gridspec(len(viruses), len(conds), left=0.055, right=0.995,
                          top=0.93, bottom=0.16, wspace=0.05, hspace=0.12)
    for r, virus in enumerate(viruses):
        for c, cond in enumerate(conds):
            terrain_panel(fig, gs[r, c], cb, grid, virus, cond, norm, levels,
                          column_header=(r == 0), colors=colors,
                          first_cond=conds[0])

    tick = np.floor(norm.vmax * 2) / 2
    saturated = norm.vmax >= CLIP_Z
    cax = fig.add_axes([0.28, 0.055, 0.44, 0.022])
    bar = mpl.colorbar.ColorbarBase(cax, cmap="RdBu_r", norm=norm,
                                    orientation="horizontal",
                                    ticks=[-tick, 0, tick],
                                    extend="both" if saturated else "neither")
    bar.ax.set_xticklabels([f"≤ {-tick:+.1f}" if saturated else f"{-tick:+.1f}",
                            "0",
                            f"≥ {tick:+.1f}" if saturated else f"{tick:+.1f}"])
    bar.ax.tick_params(labelsize=6, length=1.6, pad=1)
    bar.outline.set_linewidth(0.4)
    cax.set_title("codebook weight (z), shared scale"
                  + (", winsorized" if saturated else ""), fontsize=6, pad=2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"figs_som_{args.tag}_response_terrain.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print("wrote " + ", ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
