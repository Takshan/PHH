#!/usr/bin/env python3
import textwrap

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phh_som.utils.pipeline import (DATA, FIGURES, MM, PNG_DPI, save_figure, save_svg,
                      set_rcparams)
from phh_som.plotting.conservation_programmes import FCMAP, FCNORM
from phh_som.plotting.plot_conservation_jaki import SUBCOLS
from phh_som.plotting.late_inflammation import (GREY, INK, LFC, V, VC,
                                       fc_key, gid, style, sym)

OUT = FIGURES / "plot_late_vulnerabilities"

W, H = 183.0, 200.0
X0 = 30.0
CELL, GAP = 2.8, 2.4
BW = 4 * CELL
STEP = BW + GAP
X1 = X0 + len(V) * BW + (len(V) - 1) * GAP
A_TOP, A_H, A_GAP = 13.0, 10.0, 2.4
C_TOP = 94.0
HPI = [0, 24, 72]
BCHIP = 104.0
PX0 = 116.0
PW = W - 5.0 - PX0
ROW_H, ROW_GAP = 9.0, 1.4
BG = "#E3E6EA"

MODULES = [
    ("anti-ferroptotic", ["GPX4", "FTH1", "FTL"], "#D55E00"),
    ("pro-ferroptotic", ["ACSL4"], "#E69F00"),
    ("iron homeostasis", ["STEAP3", "NCOA4", "CP", "SLC40A1"], "#8C4A00"),
    ("25-HC synthesis", ["CH25H"], "#0072B2"),
    ("25-HC clearance", ["CYP7B1"], "#56B4E9"),
    ("ER translocon / OST",
     ["SEC61A1", "SEC61B", "SEC61G", "SEC62", "SEC63", "RPN1", "RPN2",
      "STT3A", "STT3B", "DDOST", "SPCS1", "SPCS3", "SEL1L", "SYVN1"],
     "#009E73"),
]
A_ROWS = [n for n, _, _ in MODULES]

set_rcparams()
COL = {n: c for n, _, c in MODULES}
GENES = {n: g for n, g, _ in MODULES}


def ids(genes):
    return [gid[s] for s in genes if s in gid and gid[s] in LFC.index]


def mm_ax(fig, x, y_top, w, h):
    return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])


def chips(cv, y, x_of, width, h=3.2, fs=5):
    for i, v in enumerate(V):
        cv.add_patch(plt.Rectangle((x_of(i), y), width, h, fc=VC[v], ec="none"))
        cv.text(x_of(i) + width / 2, y + h / 2, v, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold")


def panel_a(fig, cv):
    chips(cv, A_TOP - 4.4, lambda i: X0 + i * STEP, BW)
    for r, name in enumerate(A_ROWS):
        mem = ids(GENES[name])
        m = LFC.loc[mem].median()
        lo = min(-0.5, np.floor(m.min()))
        hi = max(1.0, np.ceil(m.max()))
        tick = 1 if hi - lo <= 5 else (2 if hi - lo <= 12 else 4)
        for j, v in enumerate(V):
            ax = mm_ax(fig, X0 + j * STEP, A_TOP + r * (A_H + A_GAP), BW, A_H)
            style(ax)
            ax.axhline(0, color="#D0D4D8", lw=0.4, zorder=0)
            ax.axvspan(48, 76, color="#F4EFE6", lw=0, zorder=-1)
            for tr, c, lw in (("untreated", GREY, 0.9), ("Ruxo", VC[v], 1.2)):
                y = [0, m[(v, tr, 24)], m[(v, tr, 72)]]
                ax.plot(HPI, y, "-o", color=c, lw=lw, ms=1.8, mec="none")
            ax.set_ylim(lo, hi)
            ax.set_yticks(np.arange(int(np.ceil(lo / tick)) * tick,
                                    hi + 0.01, tick))
            ax.set_xticks(HPI)
            ax.set_xlim(-6, 78)
            ax.tick_params(labelsize=5, pad=1)
            if r < len(A_ROWS) - 1:
                ax.set_xticklabels([])
            if j:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(f"{textwrap.fill(name, 12)}\nn = {len(mem)}",
                              fontsize=5, rotation=0, ha="right", va="center",
                              labelpad=2, color=COL[name])
    bottom = A_TOP + len(A_ROWS) * (A_H + A_GAP) - A_GAP
    cv.text(X0 + (X1 - X0) / 2, bottom + 5.2, "hpi", ha="center", fontsize=5)
    cv.text(4.0, A_TOP + (bottom - A_TOP) / 2,
            "median module log$_2$FC vs mock", rotation=90, va="center",
            ha="center", fontsize=5)
    handles = [plt.Line2D([], [], color=GREY, lw=0.9, marker="o", ms=1.8,
                          label="untreated"),
               plt.Line2D([], [], color=INK, lw=1.2, marker="o", ms=1.8,
                          label="JAKi (virus colour)")]
    lax = mm_ax(fig, X0, 1.0, X1 - X0, 3.5)
    lax.axis("off")
    lax.legend(handles=handles, loc="center", ncol=2, fontsize=5, frameon=False,
               handlelength=1.6)
    cv.text(3.0, A_TOP - 4.4, "a", fontsize=8, fontweight="bold", va="top")


def panel_b(fig, cv):
    jd = pd.read_csv(DATA / "jak_dependence.csv").pivot_table(
        index="gene_id", columns="virus", values="jak_index")[V]
    lanes = np.linspace(0.82, 0.18, len(MODULES))
    xlim = (-1.05, 1.05)

    for (txt, c), x, ha in (
            (("← amplified under ruxolitinib", "#B5553A"), PX0, "left"),
            (("JAK-dependent →", "#3E6A98"), PX0 + PW, "right")):
        cv.text(x, A_TOP - 1.5, txt, ha=ha, va="bottom", fontsize=5, color=c)

    for r, v in enumerate(V):
        y0 = A_TOP + r * (ROW_H + ROW_GAP)
        cv.add_patch(plt.Rectangle((BCHIP, y0 + ROW_H / 2 - 1.6), 8.5, 3.2,
                                   fc=VC[v], ec="none"))
        cv.text(BCHIP + 4.25, y0 + ROW_H / 2, v, ha="center", va="center",
                fontsize=5, color="white", fontweight="bold")

        ax = mm_ax(fig, PX0, y0, PW, ROW_H)
        style(ax)
        ax.spines["left"].set_visible(False)
        bg = jd[v].dropna()
        cnt, edges = np.histogram(bg, bins=60, range=xlim)
        ax.fill_between((edges[:-1] + edges[1:]) / 2, 0,
                        0.95 * cnt / cnt.max(), step="mid", color=BG, lw=0,
                        zorder=0)
        ax.plot([bg.median()] * 2, [0, 0.95], color=GREY, lw=0.5,
                ls=(0, (2, 1)), zorder=1)
        ax.axvline(0, color=INK, lw=0.4, zorder=1)
        for (name, genes, col), ly in zip(MODULES, lanes):
            x = jd[v].reindex(ids(genes)).dropna()
            if x.empty:
                continue
            if len(x) > 1:
                ax.scatter(x, np.full(len(x), ly), s=1.8, color=col,
                           alpha=0.5, lw=0, zorder=2)
            ax.scatter(x.median(), ly, s=11, color=col, edgecolor="white",
                       lw=0.4, zorder=3)
        ax.set_xlim(xlim)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(labelsize=5, pad=1)
        if r < len(V) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("JAK-dependence index $J$ (24 + 72 hpi pooled)",
                          fontsize=5.5, labelpad=1.5)

    bottom = A_TOP + len(V) * (ROW_H + ROW_GAP) - ROW_GAP
    handles = [plt.Line2D([], [], ls="", marker="o", ms=2.4, mec="white",
                          mew=0.4, color=c, label=n) for n, _, c in MODULES]
    lax = mm_ax(fig, PX0, bottom + 5.0, PW, 8.0)
    lax.axis("off")
    lax.legend(handles=handles, loc="upper left", ncol=2, fontsize=5,
               frameon=False, handletextpad=0.25, labelspacing=0.25,
               columnspacing=0.6, borderpad=0)
    cv.text(BCHIP - 4.5, A_TOP - 4.4, "b", fontsize=8, fontweight="bold",
            va="top")


def panel_c(fig, cv):
    rows = [(n, g) for n, genes, _ in MODULES for g in ids(genes)]
    chips(cv, C_TOP, lambda i: X0 + i * STEP, BW)
    for i, v in enumerate(V):
        for c, (hpi, _, sign) in enumerate(SUBCOLS):
            cv.text(X0 + i * STEP + (c + 0.5) * CELL, C_TOP + 12.4,
                    f"{hpi} {sign}", ha="center", va="bottom", fontsize=5,
                    color="0.25", rotation=90)
    top = C_TOP + 13.2
    ys, y, prev = [], top, None
    for grp, _ in rows:
        if prev is not None and grp != prev:
            y += 1.4
        ys.append(y)
        prev, y = grp, y + CELL
    for (grp, g), y0 in zip(rows, ys):
        cv.text(X0 - 0.8, y0 + CELL / 2, sym.symbol[g], ha="right",
                va="center", fontsize=5, style="italic")
        for i, v in enumerate(V):
            for c, (hpi, tr, _) in enumerate(SUBCOLS):
                x0 = X0 + i * STEP + c * CELL
                cv.add_patch(plt.Rectangle(
                    (x0, y0), CELL, CELL,
                    fc=FCMAP(FCNORM(LFC.loc[g, (v, tr, hpi)])), ec="0.7",
                    lw=0.2))
    xb = X1 + 2.0
    for grp in dict.fromkeys(grp for grp, _ in rows):
        yy = [y0 for (gr, _), y0 in zip(rows, ys) if gr == grp]
        for i in range(len(V)):
            cv.add_patch(plt.Rectangle((X0 + i * STEP, yy[0]), BW,
                                       yy[-1] + CELL - yy[0], fill=False,
                                       ec="black", lw=0.6))
        cv.plot([xb, xb], [yy[0] + 0.3, yy[-1] + CELL - 0.3],
                color=COL[grp], lw=1.0)
        cv.text(xb + 1.0, (yy[0] + yy[-1] + CELL) / 2, textwrap.fill(grp, 16),
                fontsize=5, va="center", color=COL[grp])
    cv.text(3.0, C_TOP, "c", fontsize=8, fontweight="bold", va="top")
    return ys[-1] + CELL


def main():
    fig = plt.figure(figsize=(W * MM, H * MM))
    cv = fig.add_axes([0, 0, 1, 1])
    cv.set_xlim(0, W)
    cv.set_ylim(H, 0)
    cv.axis("off")

    panel_a(fig, cv)
    panel_b(fig, cv)
    bottom = panel_c(fig, cv)

    fc_key(fig, X0, H - bottom - 9.0,
           note="c: hpi, − untreated, + JAKi.   n = 3 donors per arm "
                "(YFV untreated 72 hpi, n = 2).")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig_late_vulnerabilities.pdf"
    save_figure(fig, stem, dpi=PNG_DPI)
    save_svg(fig, stem.with_suffix(".svg"))
    plt.close(fig)
    print("wrote", stem.with_suffix(".*"))


if __name__ == "__main__":
    main()
