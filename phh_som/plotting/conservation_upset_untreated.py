#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

from phh_som.utils.pipeline import (FIGURES, MM, PAPER_VIRUS_ORDER, save_figure, save_svg,
                      set_rcparams)
from phh_som.plotting.conservation_programmes import VIRUS_COLORS, gene_sets, load, mm_axes
from phh_som.plotting.conservation_upset_multi import FS_MIN, _tint

VIRUSES = PAPER_VIRUS_ORDER
W = 120.0
X0, LAB_W, SET_W = 7.0, 14.0, 20.0
GRID_W = W - X0 - LAB_W - 3.0 - SET_W - 3.0
ROW_MM, BARS_H, BOX_H = 3.4, 44.0, 24.0
MAT_H = len(VIRUSES) * ROW_MM
H = 4 + BOX_H + 2 + MAT_H + 2 + BARS_H + 3
MIN_BOX = 5
UP_COL, DN_COL = "#A01915", "#1C8AA1"
BG_UP, BG_DN = _tint(UP_COL, 0.06), _tint(DN_COL, 0.07)
GRID_GREY = "0.80"
ROW_SHADE = "0.965"
REGION_CUTS = (4, 2)
REGION_LINE = dict(color="0.88", lw=0.4, zorder=1)


def assemble(dg):
    up, dn = gene_sets(dg)
    cu, cd = up.value_counts(), dn.value_counts()
    combos = sorted(set(cu.index) | set(cd.index),
                    key=lambda k: (-len(k), -cu.get(k, 0), -cd.get(k, 0),
                                   tuple(sorted(k))))
    ut = dg[(dg.treatment == "untreated") & dg.deg]
    by_up = ut[ut.log2fc > 0].assign(combo=lambda d: d.gene_id.map(up)).groupby("combo").log2fc
    by_dn = ut[ut.log2fc < 0].assign(combo=lambda d: d.gene_id.map(dn)).groupby("combo").log2fc
    vals_up = {k: g.to_numpy() for k, g in by_up}
    vals_dn = {k: g.to_numpy() for k, g in by_dn}
    return (cu.reindex(combos, fill_value=0), cd.reindex(combos, fill_value=0),
            {v: int(up.map(lambda s: v in s).sum()) for v in VIRUSES},
            {v: int(dn.map(lambda s: v in s).sum()) for v in VIRUSES},
            [vals_up.get(k, np.array([])) for k in combos],
            [vals_dn.get(k, np.array([])) for k in combos])


def draw(fig, cu, cd, tot_up, tot_dn, lfc_up, lfc_dn, x0=X0, y0=4.0, grid_w=GRID_W,
         set_w=SET_W, row_mm=ROW_MM, bars_h=BARS_H, box_h=BOX_H, noun="genes",
         tags_right=False):
    X0, GRID_W, SET_W, BARS_H, BOX_H = x0, grid_w, set_w, bars_h, box_h
    MAT_H = len(VIRUSES) * row_mm
    combos = list(cu.index)
    n = len(combos)
    row_y = {v: len(VIRUSES) - 1 - i for i, v in enumerate(VIRUSES)}
    y_box = y0
    y_mat = y_box + BOX_H + 2
    col_u = n / GRID_W
    CHIP_L, CHIP_W = -12.5 * col_u, 8.4 * col_u
    region = [sum(len(k) < c for c in REGION_CUTS) for k in combos]
    cuts = [i - 0.5 for i in range(1, n) if region[i] != region[i - 1]]

    ax = mm_axes(fig, X0 + LAB_W, y_mat + MAT_H + 2, GRID_W, BARS_H)
    u = float(max(cu.max(), cd.max()))
    lo, hi = -(cd.max() + 0.34 * u), cu.max() + 0.34 * u
    ax.add_patch(Rectangle((-0.5, 0), n, hi, fc=BG_UP, ec="none", zorder=0))
    ax.add_patch(Rectangle((-0.5, lo), n, -lo, fc=BG_DN, ec="none", zorder=0))
    for i, k in enumerate(combos):
        for val, sgn, fc in ((cu.iloc[i], 1, UP_COL), (cd.iloc[i], -1, DN_COL)):
            if val == 0:
                continue
            ax.add_patch(Rectangle((i - 0.35, 0), 0.7, sgn * val, fc=fc, ec="none"))
            ax.text(i, sgn * (val + 0.02 * u), f"{val:,}", rotation=90, ha="center",
                    va="bottom" if sgn > 0 else "top", fontsize=FS_MIN)
    tx, ha = (n - 0.7, "right") if tags_right else (-0.3, "left")
    ax.text(tx, hi, "induced", ha=ha, va="top", fontsize=6, color=UP_COL)
    ax.text(tx, lo, "repressed", ha=ha, va="bottom", fontsize=6, color=DN_COL)
    ax.axhline(0, color="0.3", lw=0.5)
    for x in cuts:
        ax.axvline(x, **REGION_LINE)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(lo, hi)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.set_ylabel(f"{noun} in intersection", fontsize=6)
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    mx = mm_axes(fig, X0 + LAB_W, y_mat, GRID_W, MAT_H)
    for v in VIRUSES[::2]:
        mx.add_patch(Rectangle((-0.5, row_y[v] - 0.5), n, 1, fc=ROW_SHADE,
                               ec="none", zorder=0))
    for v in VIRUSES:
        for i, k in enumerate(combos):
            if v in k:
                mx.plot(i, row_y[v], "o", ms=3.4, zorder=2, color=VIRUS_COLORS[v])
            else:
                mx.plot(i, row_y[v], "o", ms=2.0, mfc="white", mec="0.75", mew=0.4, zorder=2)
        mx.add_patch(Rectangle((CHIP_L, row_y[v] - 0.4), CHIP_W, 0.8,
                               fc=VIRUS_COLORS[v], ec="none", clip_on=False, zorder=4))
        mx.text(CHIP_L + CHIP_W / 2, row_y[v], v, ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold", zorder=5)
    for i, k in enumerate(combos):
        ys = sorted(row_y[v] for v in k)
        if len(ys) > 1:
            mx.plot([i, i], [ys[0], ys[-1]], color="0.35", lw=0.8, zorder=1)
    for x in cuts:
        mx.axvline(x, **REGION_LINE)
    mx.set_xlim(-0.5, n - 0.5)
    mx.set_ylim(-0.6, len(VIRUSES) - 0.4)
    mx.set_axis_off()

    sx = mm_axes(fig, X0 + LAB_W + GRID_W + 3.0, y_mat, SET_W, MAT_H)
    for v in VIRUSES:
        sx.add_patch(Rectangle((0, row_y[v]), tot_up[v], 0.36, fc=UP_COL, ec="none"))
        sx.add_patch(Rectangle((0, row_y[v] - 0.36), tot_dn[v], 0.36, fc=DN_COL,
                               ec="none"))
    short = min(VIRUSES, key=lambda v: tot_up[v])
    for tot, va, word, c in ((tot_up, "bottom", "induced", UP_COL),
                             (tot_dn, "top", "repressed", DN_COL)):
        sx.text(tot[short] + max(tot_up.values()) * 0.04, row_y[short], word,
                ha="left", va=va, fontsize=FS_MIN, color=c)
    sx.set_xlim(0, max(tot_up.values()) * 1.12)
    sx.set_ylim(-0.6, len(VIRUSES) - 0.4)
    sx.set_yticks([])
    sx.set_xlabel(f"{noun} per virus", fontsize=5.6, labelpad=1)
    sx.xaxis.set_label_position("top")
    sx.xaxis.tick_top()
    sx.xaxis.set_major_locator(MaxNLocator(3))
    sx.tick_params(labelsize=FS_MIN, length=2, pad=1)
    for sp in ("bottom", "right", "left"):
        sx.spines[sp].set_visible(False)

    bx = mm_axes(fig, X0 + LAB_W, y_box, GRID_W, BOX_H)
    vals = np.concatenate([a for a in lfc_up + lfc_dn if len(a)])
    pad = 0.08 * (vals.max() - vals.min())
    b_lo, b_hi = min(vals.min(), 0) - pad, max(vals.max(), 0) + pad
    bx.add_patch(Rectangle((-0.5, 0), n, b_hi, fc=BG_UP, ec="none", zorder=0))
    bx.add_patch(Rectangle((-0.5, b_lo), n, -b_lo, fc=BG_DN, ec="none", zorder=0))
    for arrs, fc in ((lfc_up, UP_COL), (lfc_dn, DN_COL)):
        pos = [i for i, a in enumerate(arrs) if len(a) >= MIN_BOX]
        if pos:
            bx.boxplot([arrs[i] for i in pos], positions=pos, widths=0.6,
                       patch_artist=True, manage_ticks=False, zorder=2,
                       boxprops=dict(fc=fc, ec=fc, lw=0.5),
                       medianprops=dict(color="white", lw=0.8),
                       whiskerprops=dict(color=fc, lw=0.5), capprops=dict(color=fc, lw=0.5),
                       flierprops=dict(marker="o", ms=1.0, mfc=fc, mec="none", alpha=0.6))
        for i, a in enumerate(arrs):
            if 0 < len(a) < MIN_BOX:
                bx.plot([i] * len(a), a, "o", ms=1.6, color=fc, mec="none", zorder=3)
    bx.axhline(0, color="0.3", lw=0.5)
    for x in cuts:
        bx.axvline(x, **REGION_LINE)
    bx.yaxis.grid(True, color=GRID_GREY, lw=0.4)
    bx.set_axisbelow(True)
    bx.set_xlim(-0.5, n - 0.5)
    bx.set_ylim(b_lo, b_hi)
    bx.set_xticks([])
    bx.set_ylabel("log₂FC", fontsize=6)
    bx.tick_params(labelsize=5.5, length=2, pad=1)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)


def main():
    set_rcparams()
    dg, sym, asg, cb = load()
    cu, cd, tot_up, tot_dn, lfc_up, lfc_dn = assemble(dg)
    fig = plt.figure(figsize=(W * MM, H * MM))
    draw(fig, cu, cd, tot_up, tot_dn, lfc_up, lfc_dn)
    out = FIGURES / "conservation_upset_untreated" / "fig_conservation_upset_untreated.pdf"
    written = save_figure(fig, out, bbox_inches=None)
    written += save_svg(fig, out.with_suffix(".svg"))
    plt.close(fig)

    for word, c in (("induced", cu), ("repressed", cd)):
        deg = c.groupby([len(k) for k in c.index]).sum()
        print(f"{word:9s}: {int((c > 0).sum())} intersections, {int(c.sum()):,} genes; "
              f"by number of viruses {deg.to_dict()}")
    print(f"all-five column, log2FC values: induced n = {len(lfc_up[0])} DE calls, "
          f"median {np.median(lfc_up[0]):.2f}; repressed n = {len(lfc_dn[0])}, "
          f"median {np.median(lfc_dn[0]):.2f}")
    print(f"columns {len(cu)}; page {W:.0f} x {H:.0f} mm")
    for q in written:
        print("wrote", q)


if __name__ == "__main__":
    main()
