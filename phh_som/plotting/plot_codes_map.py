#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from phh_som.utils.pipeline import (DATA, FIGURES, MM, PAPER_VIRUS_ORDER, save_figure,
                      set_rcparams, virus_colors)
from phh_som.plotting.plot_som_map import (CALLOUT_PAD_BOTTOM, COUNT_CMAP, Grid, NOTABLE_GENES,
                          callouts, count_norm, count_ticks, draw_callouts,
                          fan_scale, grid_dims)
from phh_som.utils.somdata import load_universe
from phh_som.utils.somgeom import fan_glyph, fan_key

MAX_LABELLED = 100
CALLOUT_TARGET = 8


def paper_conds(cols):
    order = {v: i for i, v in enumerate(PAPER_VIRUS_ORDER)}
    return sorted(cols, key=lambda c: order.get(c.split("_", 1)[0],
                                                len(order)))


def map_callouts(assign, stats, target=CALLOUT_TARGET):
    boxes = callouts(assign, stats)
    if len(boxes) >= target:
        return boxes
    hits = {}
    for n, grp in assign.groupby("neuron"):
        h = sorted(s for s in grp["symbol"].dropna() if s in NOTABLE_GENES)
        if h and int(n) not in boxes:
            hits[int(n)] = h
    ranked = sorted(hits, key=lambda n: (-len(hits[n]),
                                         -int(stats.loc[n, "n_genes"])))
    for n in ranked[:target - len(boxes)]:
        boxes[n] = hits[n]
    return boxes


def switch_neurons(cb, conds):
    ut = cb[[c for c in conds if "untreated" in c]].abs().max(axis=1)
    r72 = cb[[c for c in conds if "Ruxo72" in c]]
    return (list(cb.index[(r72.min(axis=1) >= 0.9) & (ut <= 0.35)]),
            list(cb.index[(r72.max(axis=1) <= -0.7) & (ut <= 0.35)]))


def panel_geometry(grid):
    labelled = grid.n <= MAX_LABELLED
    pad_top = 3.30 if labelled else 8.4
    pad_bottom = CALLOUT_PAD_BOTTOM if labelled else 0.2
    side_pad = 3.8 if labelled else 0.6
    return {
        "labelled": labelled,
        "m": 1.0 if labelled else 2.6,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "side_pad": side_pad,
        "x_range": grid.xlim[1] - grid.xlim[0] + 2 * side_pad,
        "y_range": grid.ylim[1] - grid.ylim[0] + pad_top + pad_bottom,
        "fig_w": 183 if labelled else 140,
    }


def draw_codes_map(ax, cb, assign, stats, grid, geom=None, legends=True):
    geom = geom or panel_geometry(grid)
    labelled, m = geom["labelled"], geom["m"]
    pad_top, pad_bottom = geom["pad_top"], geom["pad_bottom"]
    side_pad = geom["side_pad"]
    conds = paper_conds(cb.columns)
    colors = virus_colors()

    counts = stats["n_genes"]
    cnorm = count_norm(counts)
    k, lo, hi = fan_scale(cb)


    for n in grid.centres:
        fc = COUNT_CMAP(cnorm(counts[n]))
        x, y = grid.hexagon(ax, n, fc=fc, ec="0.55",
                            lw=0.5 if labelled else 0.15, zorder=1)
        fan_glyph(ax, x, y, cb.loc[n, conds].to_numpy(), k=k, conds=conds,
                  colors=colors, lw=0.35 if labelled else 0.05, ring=labelled)
        if labelled:
            ax.text(x, y - 0.74, str(n), ha="center", va="center",
                    fontsize=5.4, color="0.30", zorder=5)

    on, off = switch_neurons(cb, conds)
    if labelled:
        for n, mark in [(i, "^") for i in on] + [(i, "v") for i in off]:
            x, y = grid.centres[n]
            ax.scatter([x - 0.62], [y + 0.50], marker=mark, s=13, fc="#2F3338",
                       ec="none", zorder=6)

    grid.frame(ax, pad_top=pad_top, pad_bottom=pad_bottom)
    ax.set_xlim(grid.xlim[0] - side_pad, grid.xlim[1] + side_pad)
    if labelled:
        draw_callouts(ax, map_callouts(assign, stats), grid,
                      sizes=stats["n_genes"])

    if labelled:
        cb_x, key_x = grid.cx - 4.8, grid.cx + 5.7
        cb_y, key_y = 0.62, 1.78
    else:
        cb_x, key_x = grid.xlim[0] + 2.0, grid.xlim[1] - 10.5
        cb_y, key_y = pad_top / 2 - 0.9, pad_top / 2 + 0.7
    if not legends:
        return

    boundaries = cnorm.boundaries
    cax = ax.inset_axes([cb_x, grid.ylim[1] + cb_y, 8.0, 0.35 * m],
                        transform=ax.transData)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        cax.add_patch(Rectangle((left, 0), right - left, 1,
                                fc=COUNT_CMAP(cnorm((left + right) / 2)),
                                ec="0.3", lw=0.4))
    cax.set_xlim(boundaries[0], boundaries[-1])
    cax.set_ylim(0, 1)
    cax.axis("off")
    for v in count_ticks(counts):
        cax.text(v, -0.18 * m, str(v), ha="center", va="top", fontsize=5.5)
    cax.set_title("# genes per neuron", fontsize=5.5, pad=2)
    fan_key(ax, k, lo, hi, conds, colors,
            origin=(key_x, grid.ylim[1] + key_y), magnify=m)


def codes_map(cb, assign, stats, grid, tag, out_pdf):
    geom = panel_geometry(grid)
    fig_w, ratio = geom["fig_w"], geom["y_range"] / geom["x_range"]
    fig, ax = plt.subplots(figsize=(fig_w * MM, fig_w * ratio * MM))
    draw_codes_map(ax, cb, assign, stats, grid, geom)
    fig.tight_layout(pad=0.3)
    written = save_figure(fig, out_pdf)
    plt.close(fig)
    print("wrote " + ", ".join(str(p) for p in written))


def main():
    ap = argparse.ArgumentParser(description='the Fig-5E-style codes map for ANY gene set.')
    ap.add_argument("--tag", default="isg",
                    help="gene-set tag of the trained map (any train_isg_universes universe)")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=FIGURES / "plot_codes_map")
    args = ap.parse_args()

    set_rcparams()
    cb, assign, stats = load_universe(args.data_dir, args.tag)
    grid = Grid(*grid_dims(args.data_dir, args.tag))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    codes_map(cb, assign, stats, grid, args.tag,
              args.out_dir / f"figs_som_{args.tag}_codes_map.pdf")


if __name__ == "__main__":
    main()
