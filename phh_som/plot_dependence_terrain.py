#!/usr/bin/env python3
import argparse
import json
import os
import re

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.colors import TwoSlopeNorm

from pipeline import (DATA, FIGURES, MM, ROOT, VIRUS_ORDER,
                      out_suffix, save_figure, set_rcparams)
from plot_som_map import (DEP_BREAKS, Grid, MIN_SCORED, can_interpolate,
                                 discrete_map, grid_dims, low_n_outline,
                                 overlay_grid, surface)

NCOLS = 3
PANEL_W, PANEL_H = 61, 44


def virus_colors():
    return json.loads((ROOT / "config" / "colors.json").read_text())["viruses"]


def load_stats(data_dir, tag):
    stats = pd.read_csv(data_dir / f"som_neuron_stats{out_suffix(tag)}.csv",
                        index_col="neuron")
    missing = [v for v in VIRUS_ORDER
               if f"mean_jak_index_{v.lower()}" not in stats.columns]
    if missing:
        raise ValueError(
            f"som_neuron_stats{out_suffix(tag)}.csv has no per-virus columns "
            f"for {missing} - re-run neuron_stats.py (current version "
            f"scores all five viruses)")
    return stats


def draw_dependence_panels(fig, cells, stats, grid, colors, n_dep=None,
                           lev=None, viruses=None, fontsize=7.5,
                           title_fmt="JAK-STAT dependence — {virus}"):
    viruses = viruses or ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]
    n_dep = n_dep or TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    lev = np.linspace(-1, 1, 21) if lev is None else lev
    axes = []
    for cell, virus in zip(cells, viruses):
        ax = fig.add_subplot(cell)
        col = f"mean_jak_index_{virus.lower()}"
        if can_interpolate(stats, virus):
            surface(ax, stats[col].dropna(), grid, "RdBu_r", n_dep, lev,
                    contours=DEP_BREAKS)
            overlay_grid(ax, grid)
            if len(grid.centres) <= 100:
                low_n_outline(ax, stats, grid, [virus])
        else:
            discrete_map(ax, stats[col], stats, grid, plt.get_cmap("RdBu_r"),
                         n_dep, [virus])
        grid.frame(ax, pad_top=1.35)
        ax.set_title(title_fmt.format(virus=virus), color=colors[virus],
                     fontsize=fontsize)
        axes.append(ax)
    return axes


def parse_units(spec):
    if not spec:
        return []
    if spec.strip().lower() == "all":
        return "all"
    return [int(u) for u in re.split(r"[,\s]+", spec.strip()) if u]


def load_module_names(path, units):
    names = {}
    if path and Path(path).exists():
        table = pd.read_csv(path)
        if {"unit", "module"} <= set(table.columns):
            names = {int(r.unit): str(r.module) for r in table.itertuples()
                     if str(r.module).strip()}
    return {u: names.get(u, "") for u in units}


def _spread(ys, min_dy):
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        if out[j] - out[i] < min_dy:
            out[i] = out[j] - min_dy
    return out


def mark_units(ax, grid, units, fontsize=4.2, ring=True, leaders=True,
               min_dy=None, dx=2.6):
    if units == "all":
        units = list(grid.centres)
    units = [u for u in units if u in grid.centres]
    if not units:
        return
    xs = [grid.centres[u][0] for u in units]
    ys = [grid.centres[u][1] for u in units]
    if ring:
        for x, y in zip(xs, ys):
            ax.add_patch(RegularPolygon((x, y), 6, radius=0.98, orientation=0,
                                        fc="none", ec="#111417", lw=0.6, zorder=6))
    if not leaders:
        for u, x, y in zip(units, xs, ys):
            ax.text(x, y, str(u), ha="center", va="center", fontsize=fontsize,
                    color="#111417", zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.1, foreground="white")])
        return
    if min_dy is None:
        span = grid.ylim[1] - grid.ylim[0]
        min_dy = max(1.2, 0.92 * span / (len(units) + 2))
    lx = max(xs) + dx
    ly = _spread(ys, min_dy)
    for u, x, y, y2 in zip(units, xs, ys, ly):
        ax.plot([x + 1.0, lx - 0.35], [y, y2], color="#111417", lw=0.35,
                zorder=6, solid_capstyle="butt")
        ax.text(lx, y2, str(u), ha="left", va="center", fontsize=fontsize,
                color="#111417", zorder=7)
    ax.set_xlim(ax.get_xlim()[0], max(ax.get_xlim()[1], lx + 2.2))


def module_key(ax, names, fontsize=None, title="named modules", top=0.96):
    items = [f"{u} · {n}" if n else f"{u}" for u, n in names.items()]
    if not items:
        return
    if fontsize is None:
        fontsize = 5.0 if len(items) <= 8 else max(3.8, 5.0 - 0.12 * (len(items) - 8))
    ax.text(0.02, top, title, fontsize=fontsize + 0.6, fontweight="bold",
            va="top", ha="left", transform=ax.transAxes)
    ax.text(0.02, top - 0.09, "\n".join(items), fontsize=fontsize, va="top",
            ha="left", linespacing=1.28, transform=ax.transAxes)


def main():
    ap = argparse.ArgumentParser(description='JAK-STAT dependence terrain - one panel per virus, all five.')
    ap.add_argument("--tag", default="isg",
                    help="gene-set tag of the trained map (any - neuron_stats scores "
                         "all five viruses for every gene set)")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=FIGURES / "plot_dependence_terrain")
    ap.add_argument("--label-units", default="",
                    help="comma-separated neuron numbers to ring and number on "
                         "the terrain, or 'all' (default: none)")
    ap.add_argument("--label-panel", default="first",
                    choices=["first", "all"],
                    help="draw the neuron labels on the first panel only "
                         "(default) or on every virus panel")
    ap.add_argument("--callout-key", action="store_true",
                    help="list the labelled neurons and their module names "
                         "beside the colorbar")
    ap.add_argument("--callout-table", type=Path,
                    default=ROOT / "phh_som" / "data" / "som_degs_module_table.csv",
                    help="CSV with `unit` and `module` columns supplying the "
                         "names for --callout-key")
    ap.add_argument("--label-fontsize", type=float, default=4.6)
    ap.add_argument("--no-leaders", action="store_true",
                    help="draw the neuron number on the hexagon instead of "
                         "stacking it beside the map with a leader line")
    ap.add_argument("--suffix", default="",
                    help="appended to the output stem, so an annotated figure "
                         "does not overwrite the plain one")
    args = ap.parse_args()

    set_rcparams()
    stats = load_stats(args.data_dir, args.tag)
    colors = virus_colors()
    grid = Grid(*grid_dims(args.data_dir, args.tag))
    n_dep = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    lev = np.linspace(-1, 1, 21)

    nrows = int(np.ceil(len(VIRUS_ORDER) / NCOLS))
    fig = plt.figure(figsize=(NCOLS * PANEL_W * MM, nrows * PANEL_H * MM))
    gs = fig.add_gridspec(nrows, NCOLS, left=0.03, right=0.99, top=0.90,
                          bottom=0.05, wspace=0.08, hspace=0.30)
    cells = [gs[i // NCOLS, i % NCOLS] for i in range(len(VIRUS_ORDER))]
    axes = draw_dependence_panels(fig, cells, stats, grid, colors, n_dep, lev)

    units = parse_units(args.label_units)
    if units:
        targets = axes if args.label_panel == "all" else axes[:1]
        for ax in targets:
            mark_units(ax, grid, units, fontsize=args.label_fontsize,
                       leaders=not args.no_leaders)

    ax_cb = fig.add_subplot(gs[-1, -1])
    ax_cb.axis("off")
    want_key = args.callout_key and units and units != "all"
    cax = ax_cb.inset_axes([0.06, 0.88, 0.88, 0.075] if want_key
                           else [0.06, 0.42, 0.88, 0.10])
    bar = mpl.colorbar.ColorbarBase(cax, cmap="RdBu_r", norm=n_dep,
                                    orientation="horizontal",
                                    ticks=[-1, -0.5, 0, 0.5, 1])
    bar.ax.tick_params(labelsize=6, length=1.6, pad=1)
    bar.outline.set_linewidth(0.4)
    cax.set_title("mean dependence index (black: 0.2 / 0.5 class breaks)",
                  fontsize=6, pad=2)
    if len(grid.centres) <= 100:
        ax_cb.text(0.06, 0.60 if want_key else 0.22,
                   f"dotted ring = < {MIN_SCORED} scored genes",
                   fontsize=6, color="#666666", ha="left", va="center",
                   transform=ax_cb.transAxes)
    if want_key:
        module_key(ax_cb, load_module_names(args.callout_table, units), top=0.70)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"figs_som_{args.tag}_dependence_terrain{args.suffix}.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print("wrote " + ", ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
