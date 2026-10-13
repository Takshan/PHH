#!/usr/bin/env python3
import textwrap

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import FIGURES, MM, PNG_DPI, set_rcparams
from conservation_programmes import FCMAP, FCNORM
from plot_conservation_jaki import SUBCOLS
from late_inflammation import (DEG, GREY, INK, LFC,
                                       MODULE_ANCHORS, V, VC, asg, dg,
                                       fc_key, gid, style, sym)
from plot_late_inflammation_genes import GROUPS

OUT = FIGURES / "plot_late_inflammation"

W, H = 122.0, 190.0
X0 = 22.0
CELL, GAP = 2.8, 2.4
BW = 4 * CELL
STEP = BW + GAP
X1 = X0 + len(V) * BW + (len(V) - 1) * GAP
A_TOP, A_H, A_GAP = 14.0, 12.0, 2.5
C_TOP = 80.0
YMIN_TOP = 4
HPI = [0, 24, 72]


def mm_ax(fig, x, y_top, w, h):
    return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])


def chips(cv, y, x_of, width, h=3.2, fs=5):
    for i, v in enumerate(V):
        cv.add_patch(plt.Rectangle((x_of(i), y), width, h, fc=VC[v], ec="none"))
        cv.text(x_of(i) + width / 2, y + h / 2, v, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold")


def panel_a(fig, cv):
    chips(cv, A_TOP - 4.4, lambda i: X0 + i * STEP, BW)
    for r, (name, anchor) in enumerate(MODULE_ANCHORS.items()):
        n = int(asg.loc[gid[anchor], "neuron"])
        mem = asg.index[asg.neuron == n].intersection(LFC.index)
        m = LFC.loc[mem].median()
        ymax = max(YMIN_TOP, int(np.ceil(m.max())))
        tick = 1 if ymax <= YMIN_TOP else 2
        ymax += ymax % tick
        for j, v in enumerate(V):
            ax = mm_ax(fig, X0 + j * STEP, A_TOP + r * (A_H + A_GAP), BW, A_H)
            style(ax)
            ax.axvspan(48, 76, color="#F4EFE6", lw=0, zorder=-1)
            ax.axhline(0, color="#D0D4D8", lw=0.4, zorder=0)
            for tr, c, lw in (("untreated", GREY, 0.9), ("Ruxo", VC[v], 1.2)):
                y = [0, m[(v, tr, 24)], m[(v, tr, 72)]]
                ax.plot(HPI, y, "-o", color=c, lw=lw, ms=1.8, mec="none", clip_on=False)
            ax.set_ylim(-0.5, ymax)
            ax.set_yticks(range(0, ymax + 1, tick))
            ax.set_xticks(HPI)
            ax.set_xlim(-6, 78)
            ax.tick_params(labelsize=4.5, pad=1)
            if r < len(MODULE_ANCHORS) - 1:
                ax.set_xticklabels([])
            if j:
                ax.set_yticklabels([])
            else:
                label = textwrap.fill(name.split("\n")[0], 12)
                ax.set_ylabel(f"{label}\nN{n} · n = {len(mem)}",
                              fontsize=5, rotation=0, ha="right", va="center",
                              labelpad=2)
    bottom = A_TOP + len(MODULE_ANCHORS) * (A_H + A_GAP) - A_GAP
    cv.text(X0 + (X1 - X0) / 2, bottom + 5.2, "hpi", ha="center", fontsize=5)
    cv.text(2.0, A_TOP + (bottom - A_TOP) / 2, "median unit log$_2$FC vs mock",
            rotation=90, va="center", ha="center", fontsize=5)
    handles = [plt.Line2D([], [], color=GREY, lw=0.9, marker="o", ms=1.8,
                          label="untreated"),
               plt.Line2D([], [], color=INK, lw=1.2, marker="o", ms=1.8,
                          label="JAKi (virus colour)")]
    lax = mm_ax(fig, X0, 1.0, X1 - X0, 3.5)
    lax.axis("off")
    lax.legend(handles=handles, loc="center", ncol=2, fontsize=5, frameon=False,
               handlelength=1.6)
    cv.text(3.0, A_TOP - 4.4, "a", fontsize=8, fontweight="bold", va="top")


def heat_rows():
    rows = [(grp, gid[s]) for grp, genes in GROUPS.items() for s in genes]
    listed = {g for _, g in rows}
    nde = dg[dg.deg].groupby("gene_id").size()
    ck = sym[sym.symbol.fillna("").str.match(r"^(CCL|CXCL|CX3CL|XCL)\d")
             & (sym.gene_type == "protein_coding")].index
    ck = sorted((g for g in ck if nde.get(g, 0) >= 4 and g not in listed),
                key=lambda g: -nde[g])
    unit_group = {asg.neuron.get(g): grp for grp, g in rows}
    other = "Other responsive chemokines"
    for g in ck:
        grp = unit_group.get(asg.neuron.get(g), other)
        last = max(i for i, (gr, _) in enumerate(rows) if gr == grp) \
            if grp != other else len(rows) - 1
        rows.insert(last + 1, (grp, g))
    return rows


def panel_b(fig, cv):
    rows = heat_rows()
    chips(cv, C_TOP, lambda i: X0 + i * STEP, BW)
    for i in range(len(V)):
        for c, (hpi, _, sign) in enumerate(SUBCOLS):
            cv.text(X0 + i * STEP + (c + 0.5) * CELL, C_TOP + 10.6, f"{hpi} {sign}",
                    ha="center", va="bottom", fontsize=3.8, color="0.25",
                    rotation=90)
    top = C_TOP + 11.4
    ys, y, prev = [], top, None
    for grp, _ in rows:
        if prev is not None and grp != prev:
            y += 1.4
        ys.append(y)
        prev, y = grp, y + CELL
    for (grp, g), y0 in zip(rows, ys):
        cv.text(X0 - 0.8, y0 + CELL / 2, sym.symbol[g], ha="right", va="center",
                fontsize=5, style="italic")
        for i, v in enumerate(V):
            for c, (hpi, tr, _) in enumerate(SUBCOLS):
                x0 = X0 + i * STEP + c * CELL
                val = LFC.loc[g, (v, tr, hpi)]
                cv.add_patch(plt.Rectangle((x0, y0), CELL, CELL,
                                           fc=FCMAP(FCNORM(val)), ec="0.7", lw=0.2))
        n = asg.neuron.get(g)
        cv.text(X1 + 1.5, y0 + CELL / 2, f"N{int(n)}" if pd.notna(n) else "–",
                ha="left", va="center", fontsize=4.3, color="0.45")
    cv.text(X1 + 1.5, top - 0.6, "SOM\nunit", ha="left", va="bottom",
            fontsize=4.3, color="0.45")
    xb = X1 + 6.5
    for grp in dict.fromkeys(grp for grp, _ in rows):
        yy = [y0 for (gr, _), y0 in zip(rows, ys) if gr == grp]
        for i in range(len(V)):
            cv.add_patch(plt.Rectangle((X0 + i * STEP, yy[0]), BW,
                                       yy[-1] + CELL - yy[0], fill=False,
                                       ec="black", lw=0.6))
        cv.plot([xb, xb], [yy[0] + 0.3, yy[-1] + CELL - 0.3], color=INK, lw=0.5)
        cv.text(xb + 1.0, (yy[0] + yy[-1] + CELL) / 2,
                textwrap.fill(grp, 22), fontsize=4.8, va="center")
    cv.text(3.0, C_TOP, "b", fontsize=8, fontweight="bold", va="top")
    return ys[-1] + CELL


def main():
    set_rcparams()
    fig = plt.figure(figsize=(W * MM, H * MM))
    cv = fig.add_axes([0, 0, 1, 1])
    cv.set_xlim(0, W)
    cv.set_ylim(H, 0)
    cv.axis("off")
    panel_a(fig, cv)
    b_bottom = panel_b(fig, cv)
    fc_key(fig, X0 + 8, H - b_bottom - 8.5,
           note="b columns: hpi, − untreated, + JAKi;  "
                "per-contrast DEG calls in the supplementary table")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"fig_late_inflammation_v2.{ext}")
    fig.savefig(OUT / "fig_late_inflammation_v2.png", dpi=PNG_DPI)
    print("wrote", OUT / "fig_late_inflammation_v2.{pdf,svg,png}")


if __name__ == "__main__":
    main()
