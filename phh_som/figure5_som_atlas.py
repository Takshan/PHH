#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from pipeline import FIGURES, MM, save_figure, set_rcparams
import som_figure_context
import som_atlas_panels

OUT = FIGURES / "figure5_som_atlas" / "figure5.pdf"
FIG_W = 183.0
RESTRICTED_RED = "#B4231C"
V = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]
som_atlas_panels.VIRUS_ORDER = V


def panel_programme_matrix(ax, ctx):
    induced = ctx.n_up[ctx.n_up > 0]
    cls_order = list(som_figure_context.CLASSES)
    units = sorted(induced.index,
                   key=lambda n: (-induced[n], cls_order.index(ctx.cls(n)),
                                  -ctx.amp_up.loc[n].max()))
    Z = ctx.amp_up.loc[units, V].to_numpy(dtype=float).T
    norm = Normalize(vmin=0, vmax=4)
    ax.imshow(np.clip(Z, 0, 4), cmap="Reds", norm=norm, aspect="auto")
    iy, ix = np.nonzero(Z > 1)
    ax.scatter(ix, iy, s=2.2, c="#1F2328", lw=0, zorder=4)

    for j, n in enumerate(units):
        ax.add_patch(Rectangle((j - 0.5, -1.35), 1.0, 0.6,
                               fc=som_figure_context.CLASSES[ctx.cls(n)][1], ec="white",
                               lw=0.2, clip_on=False))
    n5 = int((induced == 5).sum())
    restricted = [j for j, n in enumerate(units) if induced[n] == 1]
    top = -1.6
    for lo, hi, text, col in ((0, n5 - 1, f"induced by all five · {n5}",
                               "#1F2328"),
                              (min(restricted), max(restricted),
                               f"one virus · {len(restricted)}",
                               RESTRICTED_RED)):
        ax.plot([lo - 0.4, lo - 0.4, hi + 0.4, hi + 0.4],
                [top + 0.25, top, top, top + 0.25], color=col, lw=0.5,
                clip_on=False)
        ax.text((lo + hi) / 2, top - 0.2, text, fontsize=4.3, color=col,
                ha="center", va="bottom", clip_on=False)
    for j in restricted:
        ax.add_patch(Rectangle((j - 0.5, -0.5), 1.0, len(V),
                               fc="none", ec=RESTRICTED_RED, lw=0.8,
                               zorder=5))
    ax.set_xticks(range(len(units)))
    ax.set_xticklabels([str(n) for n in units], fontsize=3.4, rotation=90)
    for lab, n in zip(ax.get_xticklabels(), units):
        if n in som_figure_context.RESTRICTED:
            lab.set_color(RESTRICTED_RED)
            lab.set_fontweight("bold")
    ax.set_yticks(range(len(V)))
    ax.set_yticklabels(V, fontsize=4.4)
    colors = som_figure_context.virus_colors()
    for lab, v in zip(ax.get_yticklabels(), V):
        lab.set_color(colors[v])
        lab.set_fontweight("bold")
    ax.tick_params(length=1.0, pad=1.2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.set_xlabel(f"SOM unit — the {len(units)} of {len(ctx.sup)} supported "
                  "programmes induced by ≥ 1 virus", fontsize=4.3,
                  labelpad=1.5)
    return norm


def class_key(fig, x, y, ctx, dy=0.0085, ncol=2, colw=0.2):
    shown = [k for k in som_figure_context.CLASSES if k != "tech"
             and any(ctx.cls(n) == k for n in ctx.n_up[ctx.n_up > 0].index)]
    for i, k in enumerate(shown):
        xx, yy = x + (i % ncol) * colw, y - (i // ncol) * dy
        fig.patches.append(Rectangle((xx, yy - 0.0022), 0.009, 0.0044,
                                     transform=fig.transFigure,
                                     fc=som_figure_context.CLASSES[k][1], ec="none"))
        fig.text(xx + 0.012, yy, som_atlas_panels.SHORT_CLASS.get(k, som_figure_context.CLASSES[k][0]),
                 fontsize=4.0, va="center")


def main():
    set_rcparams()
    ctx = som_figure_context.Context()
    fig = plt.figure(figsize=(FIG_W * MM, 247.0 * MM))

    ax_a = fig.add_axes([0.008, 0.40, 0.984, 0.58])
    bottom = som_atlas_panels.panel_atlas_neuron_boxes(ax_a, ctx, side_pad=18.0,
                                          right_pad=18.0, glyphs_out=True,
                                          plain_units=(56,), named_key=False,
                                          orphan640=True, top_stats=False,
                                          simple=True)
    xspan = ax_a.get_xlim()[1] - ax_a.get_xlim()[0]
    a_mm = (ax_a.get_ylim()[1] - bottom) * 0.984 * FIG_W / xspan

    low_mm = 58.0
    fig_h = 4.0 + a_mm + 4.0 + low_mm
    fig.set_size_inches(FIG_W * MM, fig_h * MM)
    f = 1.0 / fig_h
    a_frac = a_mm * f
    ax_a.set_position([0.008, 1.0 - 4.0 * f - a_frac, 0.984, a_frac])
    som_atlas_panels.label(fig, 0.004, 1.0 - 2.0 * f, "a")

    low_top = 1.0 - (8.0 + a_mm) * f

    b_top = low_top - 9.0 * f
    ax_b = fig.add_axes([12.0 / FIG_W, b_top - 18.0 * f, 92.0 / FIG_W, 18.0 * f])
    bnorm = panel_programme_matrix(ax_b, ctx)
    som_atlas_panels.label(fig, 0.004, low_top, "b")
    class_key(fig, 12.0 / FIG_W, b_top - 27.0 * f, ctx, dy=2.0 * f, ncol=2,
              colw=0.17)
    cax = fig.add_axes([80.0 / FIG_W, b_top - 34.0 * f, 22.0 / FIG_W, 1.2 * f])
    bar = mpl.colorbar.ColorbarBase(cax, cmap="Reds", norm=bnorm,
                                    orientation="horizontal",
                                    ticks=[0, 1, 2, 3, 4], extend="max")
    bar.ax.tick_params(labelsize=3.8, length=1.0, pad=0.8)
    bar.outline.set_linewidth(0.4)
    fig.text(91.0 / FIG_W, b_top - 32.4 * f,
             "peak untreated metagene z · dot = z > 1", fontsize=3.9,
             ha="center", va="bottom")

    cx = 118.0 / FIG_W
    som_atlas_panels.label(fig, cx - 9.0 / FIG_W, low_top, "c")
    gs_c = fig.add_gridspec(1, 1, left=(118.0 + 24.0) / FIG_W,
                            right=174.0 / FIG_W,
                            top=low_top - 14.0 * f, bottom=low_top - 39.0 * f)
    pnorm = som_atlas_panels.panel_phase_tiles(fig, gs_c[0], ctx)
    fig.axes[-1].set_title("")
    fig.axes[-1].tick_params(axis="y", labelsize=4.0)
    cax = fig.add_axes([175.5 / FIG_W, low_top - 39.0 * f, 1.5 / FIG_W,
                        25.0 * f])
    bar = mpl.colorbar.ColorbarBase(cax, cmap="RdBu_r", norm=pnorm,
                                    ticks=[-6, 0, 6], extend="both")
    cax.tick_params(labelsize=3.8, length=1.0, pad=0.8)
    bar.outline.set_linewidth(0.4)
    s24 = ctx.share24
    fig.text(cx, low_top - 47.5 * f,
             "tile = mean untreated log₂FC vs Mock of the phase's genes\n"
             f"already DE at 24 hpi: median {np.median(s24['induced']):.0%} "
             f"induced vs {np.median(s24['repressed']):.0%} repressed "
             "programme genes", fontsize=3.9, color="0.3", ha="left",
             va="top")

    written = save_figure(fig, OUT)
    plt.close(fig)
    print(f"figure5: {FIG_W:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
