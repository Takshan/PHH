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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle, Wedge

from pipeline import (CLIP_Z, DATA, FIGURES, MM, ROOT, out_suffix,
                      save_figure, set_rcparams)
from plot_som_map import (CALLOUT_PAD_BOTTOM, COUNT_CMAP,
                                 FAN_R0, Grid, NOTABLE_GENES, callouts,
                                 count_norm, count_ticks, draw_callouts,
                                 fan_scale, grid_dims, lighten)

MAX_LABELLED = 100
CALLOUT_TARGET = 8

PAPER_VIRUS_ORDER = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]


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

COND_LABEL = {"untreated24": "UT 24", "untreated72": "UT 72",
              "Ruxo24": "Ruxo 24", "Ruxo72": "Ruxo 72"}
TIME_COLORS = {"24": "#2166ac", "72": "#b2182b"}


def cond_color(rest):
    base = TIME_COLORS["24" if rest.endswith("24") else "72"]
    return lighten(base, 0.50) if "Ruxo" in rest else base


def virus_colors():
    return json.loads((ROOT / "config" / "colors.json").read_text())["viruses"]


def fan_glyph(ax, x, y, weights, k, conds, colors, r0=FAN_R0, lw=0.35,
              ring=True):
    n = len(weights)
    for i, w in enumerate(weights):
        w = float(np.clip(w, -CLIP_Z, CLIP_Z))
        a2, a1 = 90 - i * 360 / n, 90 - (i + 1) * 360 / n
        virus, rest = conds[i].split("_", 1)
        fc = cond_color(rest)
        r = r0 + k * w
        if w >= 0:
            ax.add_patch(Wedge((x, y), r, a1, a2, width=max(r - r0, 1e-3),
                               fc=fc, ec="white", lw=lw, zorder=3))
        else:
            ax.add_patch(Wedge((x, y), r0, a1, a2, width=max(r0 - r, 1e-3),
                               fc=fc, ec="white", lw=lw, alpha=0.40, zorder=3,
                               hatch="////"))
    if ring:
        ax.add_patch(Circle((x, y), r0, fc="none", ec="0.30", lw=0.45,
                            ls=(0, (1.3, 1.3)), zorder=4))
        ax.plot([x, x], [y - r0 - 0.5 * k, y + r0 + 0.5 * k], color="0.30",
                lw=0.45, zorder=4)


def fan_key(ax, k, lo, hi, conds, colors, origin=(0.0, 0.0), magnify=1.0):
    ax.set_aspect("equal")
    ax.axis("off")
    r0 = FAN_R0 * magnify
    k = k * magnify
    R = r0 + k * hi
    ox, oy = origin
    n = len(conds)
    fan_glyph(ax, ox, oy, [hi] * n, k=k, conds=conds, colors=colors,
              r0=r0, lw=0.5)
    per_virus = [c.split("_")[0] for c in conds].count(conds[0].split("_")[0])
    if n <= 8:
        for i, cond in enumerate(conds):
            mid = np.deg2rad(90 - (i + 0.5) * 360 / n)
            virus, rest = cond.split("_", 1)
            ax.text(ox + (R + 0.09 * magnify) * np.cos(mid),
                    oy + (R + 0.09 * magnify) * np.sin(mid),
                    COND_LABEL[rest], fontsize=5.2, color="0.40",
                    va="center",
                    ha="left" if np.cos(mid) > 0.05 else
                       ("right" if np.cos(mid) < -0.05 else "center"))
    for vi in range(0, n, per_virus):
        virus = conds[vi].split("_")[0]
        mid = np.deg2rad(90 - (vi + per_virus / 2) * 360 / n)
        ax.text(ox + (R + 0.40 * magnify) * np.cos(mid),
                oy + (R + 0.40 * magnify) * np.sin(mid),
                virus, fontsize=6.5, fontweight="bold", color=colors[virus],
                ha="center", va="center")
    y = oy - (R + 0.30 * magnify)
    x0, x1 = ox + r0 + k * lo, ox + r0 + k * hi
    if n > 8:
        dx = (ox - 1.2 * magnify) - x1
        x0, x1 = x0 + dx, x1 + dx
    ax.plot([x0, x1], [y, y], color="0.35", lw=0.6)
    ends = [(lo, x0, "right", y - 0.035 * magnify, "top"),
            (hi, x1, "left", y - 0.035 * magnify, "top")]
    if n > 8:
        ends[1] = (hi, x1, "left", y + 0.10 * magnify, "bottom")
    for v, x, ha, ty, va in ends:
        ax.plot([x, x], [y, y + 0.085 * magnify], color="0.35", lw=0.5)
        sat = (v <= -CLIP_Z and ha == "right") or (v >= CLIP_Z and ha == "left")
        label = f"{v:+.1f}".replace("-", "−")
        ax.text(x, ty, ("≤ " if v < 0 else "≥ ") * sat + label, ha=ha,
                va=va, fontsize=4.8, color="0.35")
    cx_scale = (x0 + x1) / 2
    ax.text(cx_scale, y - 0.30 * magnify,
            "codebook weight (z)" + ", winsorized" * (lo <= -CLIP_Z or hi >= CLIP_Z), ha="center",
            va="top", fontsize=5.5, color="0.35")
    handles = [Patch(fc=cond_color(rest), ec="none", label=COND_LABEL[rest])
               for rest in ("untreated24", "Ruxo24", "untreated72", "Ruxo72")
               if any(c.split("_", 1)[1] == rest for c in conds)]
    handles += [
        Patch(fc="white", ec="0.40", lw=0.5, hatch="////",
              label="negative (inward)"),
        Line2D([], [], color="0.30", lw=0.8, ls=(0, (1.3, 1.3)),
               label="zero ring"),
    ]
    if n > 8:
        ax.legend(handles=handles, loc="center left", fontsize=5.2,
                  frameon=False, handlelength=1.1, handletextpad=0.5,
                  labelspacing=0.45, borderaxespad=0.0,
                  bbox_to_anchor=(ox + R + 0.9 * magnify, oy),
                  bbox_transform=ax.transData)
    else:
        ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=5.0,
                  frameon=False, handlelength=1.1, handletextpad=0.5,
                  labelspacing=0.3, columnspacing=0.9, borderaxespad=0.0,
                  bbox_to_anchor=(ox, y - 0.50 * magnify),
                  bbox_transform=ax.transData)


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
    suff = out_suffix(args.tag)
    cb = pd.read_csv(args.data_dir / f"som_codebook{suff}.tsv", sep="\t",
                     index_col=0)
    assign = pd.read_csv(args.data_dir / f"som_neuron_assignments{suff}.tsv",
                         sep="\t")
    stats = pd.read_csv(args.data_dir / f"som_neuron_stats{suff}.csv",
                        index_col="neuron")
    grid = Grid(*grid_dims(args.data_dir, args.tag))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    codes_map(cb, assign, stats, grid, args.tag,
              args.out_dir / f"figs_som_{args.tag}_codes_map.pdf")


if __name__ == "__main__":
    main()
