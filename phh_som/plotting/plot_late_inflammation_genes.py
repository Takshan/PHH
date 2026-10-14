#!/usr/bin/env python3
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from phh_som.utils.pipeline import FIGURES, MM, PNG_DPI, set_rcparams
from phh_som.plotting.late_inflammation import (DEG, GREY, INK, LFC, V, VC, asg,
                                       gid, style)

OUT = FIGURES / "plot_late_inflammation_genes"

GROUPS = {
    "Immediate-early AP-1": ["FOS", "EGR1", "JUNB", "NR4A1"],
    "AP-1 programme DUSP5/FOSL1/MAFF/CCL20": ["DUSP5", "FOSL1", "MAFF", "CCL20"],
    "AP-1 programme ETS2/ZFP36": ["ETS2", "ZFP36"],
    "chr17q12 cassette with CD70/CH25H": ["CCL3", "CCL3L3", "CCL4", "CCL4L2",
                                          "CD70", "CH25H"],
    "IRF3-direct programme": ["CCL5"],
}
RATIO_GENES = ["FOS", "EGR1", "JUNB"]
HPI = [0, 24, 72]
ARMS = (("untreated", GREY, 0.9), ("Ruxo", None, 1.2))


def gene_row(fig, gs, i, s, bottom):
    g = gid[s]
    y = {(v, tr): [0.0, LFC.loc[g, (v, tr, 24)], LFC.loc[g, (v, tr, 72)]]
         for v in V for tr, _, _ in ARMS}
    lo = min(-0.5, min(min(t) for t in y.values()))
    hi = max(1.5, max(max(t) for t in y.values()))
    pad = 0.1 * (hi - lo)
    first = None
    for j, v in enumerate(V):
        ax = fig.add_subplot(gs[i, j], sharey=first)
        first = first or ax
        style(ax)
        ax.axhline(0, color="#D0D4D8", lw=0.4, zorder=0)
        ax.axhline(1, color=GREY, lw=0.4, ls=":", zorder=0)
        for tr, c, lw in ARMS:
            c = c or VC[v]
            ax.plot(HPI, y[(v, tr)], "-", color=c, lw=lw, zorder=2)
            for h, val in zip((24, 72), y[(v, tr)][1:]):
                deg = DEG.loc[g, (v, tr, h)]
                ax.plot(h, val, "o", ms=2.6, mfc=c if deg else "white", mec=c,
                        mew=0.6, zorder=3)
        ax.plot(0, 0, "o", ms=2.2, mfc="white", mec=GREY, mew=0.5, zorder=3)
        if s in RATIO_GENES:
            ut, rx = y[(v, "untreated")][2], y[(v, "Ruxo")][2]
            txt = f"×{rx / ut:.1f}" if ut > 0 else "–"
            strong = ut > 0 and rx / ut >= 2 and DEG.loc[g, (v, "untreated", 72)]
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=5.2,
                    va="top", color=INK if strong else GREY,
                    fontweight="bold" if strong else "normal")
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.set_xticks(HPI)
        ax.set_xlim(-6, 78)
        ax.tick_params(labelsize=5)
        if not bottom:
            ax.set_xticklabels([])
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            n = asg.neuron.get(g)
            unit = f"N{int(n)}" if n == n and n is not None else "not on map"
            ax.set_ylabel(f"$\\it{{{s}}}$\n{unit}", fontsize=5.5, rotation=0,
                          ha="right", va="center", labelpad=4)
    return first


def main():
    set_rcparams()
    layout = []
    for name, genes in GROUPS.items():
        layout.append(("group", name))
        layout += [("gene", s) for s in genes]
    heights = [0.5 if kind == "group" else 1.0 for kind, _ in layout]
    fig = plt.figure(figsize=(183 * MM, 235 * MM))
    gs = GridSpec(len(layout), len(V), figure=fig, height_ratios=heights,
                  hspace=0.3, wspace=0.12, left=0.15, right=0.98, top=0.925,
                  bottom=0.045)
    last_gene = max(i for i, (kind, _) in enumerate(layout) if kind == "gene")
    for i, (kind, item) in enumerate(layout):
        if kind == "group":
            ax = fig.add_subplot(gs[i, :])
            ax.axis("off")
            ax.text(-0.13, 0.05, item, transform=ax.transAxes, fontsize=6.5,
                    fontweight="bold", va="bottom")
            if i == 0:
                for j, v in enumerate(V):
                    ax.text((j + 0.5) / len(V), 1.05, v, transform=ax.transAxes,
                            ha="center", fontsize=7, color=VC[v],
                            fontweight="bold")
        else:
            gene_row(fig, gs, i, item, bottom=i == last_gene)
    fig.text(0.565, 0.018, "hpi", ha="center", fontsize=6)
    fig.text(0.008, 0.5, "log2FC vs mock", rotation=90, va="center", fontsize=6)
    handles = [
        plt.Line2D([], [], color=GREY, lw=0.9, label="untreated"),
        plt.Line2D([], [], color=INK, lw=1.2, label="JAKi (virus colour)"),
        plt.Line2D([], [], ls="", marker="o", ms=2.6, mfc=INK, mec=INK,
                   label="DEG"),
        plt.Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=INK,
                   label="not DEG"),
        plt.Line2D([], [], color=GREY, lw=0.6, ls=":", label="two-fold"),
        plt.Line2D([], [], ls="", marker="o", ms=2.2, mfc="white", mec=GREY,
                   label="0 hpi = mock baseline"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=5.5,
               frameon=False, bbox_to_anchor=(0.565, 0.995), handlelength=1.8)
    fig.text(0.565, 0.962, "×: JAKi / untreated log2FC at 72 hpi "
             "(bold where ≥ 2 and untreated is a DEG)", ha="center", fontsize=5.5,
             color=INK)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"fig_late_inflammation_genes.{ext}")
    fig.savefig(OUT / "fig_late_inflammation_genes.png", dpi=PNG_DPI)
    print("wrote", OUT / "fig_late_inflammation_genes.{pdf,svg,png}")


if __name__ == "__main__":
    main()
