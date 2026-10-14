#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats as st

pd.set_option("future.infer_string", False)

from phh_som.utils.pipeline import FIGURES, load_isg_symbols, set_rcparams, virus_colors
from phh_som.utils.jak import jak_by_hpi
from phh_som.plotting.conservation_programmes import (FCMAP, FCNORM, MISSING,
                                    programmes)
from phh_som.plotting.plot_conservation_jaki import SUBCOLS
from phh_som.plotting.jak_dependence_tiers import (MIN_AMP, MINUS, V, load,
                               supported)
from phh_som.plotting.conservation_upset_untreated import assemble
from phh_som.plotting.conservation_upset_untreated import draw as upset

W, H = 183.0, 247.0
FS, FS_SMALL = 5, 4.5
INK, GREY = "#1F2328", "#9AA0A6"
LATE_BAND = dict(color="#F4EFE6", lw=0, zorder=-1)
GROUP_COL = {"resistant": "#E69F00", "oas": "#56B4E9", "isgf3": "#009E73"}
ROWS = [756, 728, 726, 695, 722, 751, 724]
SHORT = {756: "IFN ligands", 728: "IRF3-direct", 726: "OAS1/3–RIGI–DDX60",
         695: "ISGF3 core", 722: "RNF213/APOL3", 751: "STAT2/PARP9",
         724: "BCL2L14/SIDT1"}
LABEL_WRAP = {726: "OAS1/3–RIGI–\nDDX60", 722: "RNF213/\nAPOL3"}
GROUP = {**dict.fromkeys([756, 728], "resistant"), 726: "oas",
         **dict.fromkeys([695, 722, 751, 724], "isgf3")}
CELL, VGAP, ROW_GAP = 1.58, 2.4, 1.2
BW = 4 * CELL
STEP = BW + VGAP
X0 = 15.0
PX0, LW, LGAP = 80.0, 16.0, 1.5
JX, JW = 169.5, 9.0
CHIP_H = 3.2
E_TOP = 96.0
B_BOT = 49.0
ROW2 = 51.0

OUT = FIGURES / "jak_dependence_figure"
VC = virus_colors()
set_rcparams()
mpl.rcParams.update({"font.size": 6, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def main():
    dg, sym, asg, cb, nstat, fc = load()
    sym_s = sym["symbol"]

    def members(n):
        return list(asg.index[asg.neuron == n])

    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    cv = fig.add_axes([0, 0, 1, 1])
    cv.set(xlim=(0, W), ylim=(H, 0))
    cv.axis("off")

    def axes(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    def head(x, y, letter):
        cv.text(x, y, letter, fontsize=8, fontweight="bold", va="top")

    J24g = jak_by_hpi(dg)["24"].pivot_table(index="gene_id", columns="virus",
                                            values="jak_index")
    j24 = {n: float(J24g.reindex(members(n)).stack().mean()) for n in ROWS}
    ret = {}
    for n in ROWS:
        r = []
        for v in V:
            den = fc.loc[members(n), f"{v}_untreated24"].mean()
            if abs(den) >= MIN_AMP:
                r.append(100 * fc.loc[members(n), f"{v}_Ruxo24"].mean() / den)
        ret[n] = (min(r), max(r))
    blocks = {}
    for n in ROWS:
        s = sym_s.reindex(members(n))
        genes = s[s.notna() & ~s.str.startswith("ENSG")].index
        ut24 = fc.loc[genes, [f"{v}_untreated24" for v in V]]
        genes = ut24.index[ut24.abs().max(axis=1) >= MIN_AMP]
        blocks[n] = list(ut24.loc[genes].mean(axis=1)
                         .sort_values(ascending=False).index)

    SUP, _ = supported(dg, asg)
    UT = [f"{v}_untreated{h}" for v in V for h in (24, 72)]
    S24 = SUP[cb.loc[SUP, UT].mean(axis=1) > 1]
    IND = programmes(dg, asg, cb)[0].index
    amp = cb.loc[IND, UT].max(axis=1)
    jall = pd.Series({n: float(J24g.reindex(members(n)).stack().mean())
                      for n in IND})
    rho, p = st.spearmanr(jall[S24], amp[S24])
    head(2, 2.5, "a")
    ax = axes(9, 9, 64, 30)
    other = [n for n in S24 if n not in GROUP]
    ax.scatter(amp[other], jall[other], s=3, c="#BDBDBD", lw=0)
    MARK = {756: "s", 728: "D"}
    for n in ROWS:
        ax.scatter(amp[n], jall[n], s=11, c=GROUP_COL[GROUP[n]], lw=0,
                   marker=MARK.get(n, "o"))
    key = [("IFN ligands", GROUP_COL["resistant"], "s", 3.0),
           ("ISGF3 programs", GROUP_COL["isgf3"], "o", 3.3),
           ("IRF3-direct", GROUP_COL["resistant"], "D", 2.6),
           ("other programs", "#BDBDBD", "o", 1.8),
           ("OAS1/3–RIGI–DDX60", GROUP_COL["oas"], "o", 3.3)]
    ax.legend(handles=[Line2D([], [], ls="none", marker=mk, ms=ms, mfc=c,
                              mec="none", label=lab) for lab, c, mk, ms in key],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=FS,
              handletextpad=0.2, columnspacing=0.9, labelspacing=0.2,
              borderaxespad=0)
    ax.axhline(0, color="black", lw=0.4)
    ax.annotate(f"Spearman ρ = {rho:.2f}, $P$ = {p:.2f}".translate(MINUS),
                (0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                fontsize=FS)
    ax.set(xlim=(0, None), ylim=(-0.1, 1.08),
           xlabel="peak untreated induction (SOM codebook $z$)",
           ylabel="JAK dependence $J$ at 24 hpi")
    ax.tick_params(labelsize=FS_SMALL)
    ax.xaxis.label.set_size(FS)
    ax.yaxis.label.set_size(FS)

    panel = load_isg_symbols() & set(dg.symbol)
    up = dg[(dg.treatment == "untreated") & dg.deg & (dg.log2fc > 0)]
    nvir = (up[up.symbol.isin(panel)].groupby("symbol").virus.nunique()
            .reindex(sorted(panel), fill_value=0))
    core = set(nvir.index[nvir == len(V)])
    LFC = dg[dg.symbol.isin(core)].pivot_table(index="gene_id",
                                               columns=["virus", "treatment", "hpi"],
                                               values="log2fc")
    jhp = jak_by_hpi(dg)
    JG = {h: (jhp[str(h)][jhp[str(h)].gene_id.isin(LFC.index)]
              .pivot_table(index="gene_id", columns="virus", values="jak_index")
              .reindex(columns=V)) for h in (24, 72)}
    mJ = {h: JG[h].mean() for h in (24, 72)}

    head(80, 2.5, "b")
    cu, cd, tot_up, tot_dn, lfc_up, lfc_dn = assemble(dg[dg.symbol.isin(panel)])
    upset(fig, cu, cd, tot_up, tot_dn, lfc_up, lfc_dn, x0=86, y0=H - B_BOT,
          grid_w=50, set_w=18, row_mm=2.6, bars_h=18, box_h=10, noun="ISGs",
          tags_right=True)
    assert cu[frozenset(V)] == len(core)

    head(2, ROW2, "c")
    ax = axes(14, ROW2 + 5, 33, 18)
    ax.axhline(0, color="0.75", lw=0.4, zorder=0)
    for v in V:
        ax.plot([24, 72], [mJ[24][v], mJ[72][v]], color=VC[v], lw=0.9,
                marker="o", ms=2.2, mec="none", clip_on=False, zorder=3)
    lab_y, prev = {}, None
    for v in sorted(V, key=lambda v: -mJ[72][v]):
        y = mJ[72][v] if prev is None else min(mJ[72][v], prev - 0.10)
        lab_y[v], prev = y, y
    for v in V:
        ax.plot([72, 75.5], [mJ[72][v], lab_y[v]], color=VC[v], lw=0.4,
                clip_on=False, zorder=2)
        ax.text(76.5, lab_y[v], v, fontsize=FS_SMALL, color=VC[v], va="center",
                ha="left")
    gm = [float(np.mean([mJ[h][v] for v in V])) for h in (24, 72)]
    ax.plot([24, 72], gm, color=INK, lw=1.8, marker="o", ms=3.0, mec="none",
            zorder=4, clip_on=False)
    for x, y, dy in ((24, gm[0], 0.05), (72, gm[1], -0.06)):
        ax.text(x - 1.5, y + dy, f"{y:.2f}", fontsize=FS, fontweight="bold",
                color=INK, ha="right", va="bottom" if dy > 0 else "top")
    ax.set(ylim=(-0.05, 0.9), yticks=[0, 0.25, 0.5, 0.75], xlim=(4, 88),
           xticks=[24, 72], xlabel="hpi",
           ylabel=f"core ISGs (n = {len(core)})\nmean dependence $J$")
    ax.xaxis.label.set_size(FS)
    ax.tick_params(labelsize=FS_SMALL)
    ax.yaxis.label.set_size(FS)

    head(56, ROW2, "d")
    ARMS = [("untreated", "-", "untreated"), ("Ruxo", (0, (2.2, 1.2)), "JAKi")]
    DX0, PW, DGAP = 66, 20.0, 3.0
    med = LFC.median()
    ys = [0] + [med[(v, arm, h)] for v in V for arm, _, _ in ARMS for h in (24, 72)]
    lo, hi = min(-1, min(ys)), max(1, max(ys))
    for i, v in enumerate(V):
        x = DX0 + i * (PW + DGAP)
        ax = axes(x, ROW2 + 5, PW, 18)
        ax.text(0.04, 0.98, v, transform=ax.transAxes, ha="left", va="top",
                fontsize=6, fontweight="bold", color=VC[v])
        ax.axvspan(48, 76, **LATE_BAND)
        ax.axhline(0, color="0.75", lw=0.4, zorder=0)
        for cut in (-1, 1):
            ax.axhline(cut, color="0.35", lw=0.4, ls=(0, (1, 1.5)), zorder=0)
        ut = [0, med[(v, "untreated", 24)], med[(v, "untreated", 72)]]
        rx = [0, med[(v, "Ruxo", 24)], med[(v, "Ruxo", 72)]]
        ax.fill_between([0, 24, 72], ut, rx, color=VC[v], alpha=0.18, lw=0,
                        zorder=1)
        for arm, ls, _ in ARMS:
            ax.plot([0, 24, 72], [0, med[(v, arm, 24)], med[(v, arm, 72)]],
                    color=VC[v], ls=ls, lw=0.9, marker="o", ms=1.6, mec="none",
                    clip_on=False)
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_yticks(range(0, int(np.floor(hi)) + 1, 2))
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(f"core ISGs (n = {len(core)})\nmedian log$_2$FC vs mock",
                          fontsize=FS, labelpad=1.5)
        ax.set(xlim=(-6, 78), xticks=[0, 24, 72])
        ax.tick_params(labelsize=FS_SMALL)
    cv.text(DX0 + (5 * PW + 4 * DGAP) / 2, ROW2 + 26.5, "hpi", ha="center",
            va="top", fontsize=FS)
    cv.legend(handles=[Line2D([], [], color="0.25", lw=0.8, ls=ls, label=lab)
                       for _, ls, lab in ARMS]
              + [plt.Rectangle((0, 0), 1, 1, fc="0.25", alpha=0.18, lw=0,
                               label="lost under JAKi")],
              loc="upper right",
              bbox_to_anchor=((DX0 + 5 * PW + 4 * DGAP) / W, 1 - (ROW2 - 0.3) / H),
              ncol=3, fontsize=FS, handlelength=2.2, columnspacing=1.2,
              borderaxespad=0)

    head(2, 82.5, "e")
    head(JX - 3.0, 82.5, "f")
    handles = [Line2D([], [], color=GREY, lw=0.9, marker="o", ms=1.8,
                      label="untreated"),
               Line2D([], [], color=INK, lw=1.2, marker="o", ms=1.8,
                      label="JAKi (virus colour)")]
    lax = axes(108, 82.0, 50, 4.2)
    lax.axis("off")
    lax.legend(handles=handles, loc="center", ncol=2, fontsize=FS,
               handlelength=1.6)

    for i, v in enumerate(V):
        cv.add_patch(plt.Rectangle((X0 + i * STEP, 86.5), BW, CHIP_H,
                                   fc=VC[v], ec="none"))
        cv.text(X0 + i * STEP + BW / 2, 86.5 + CHIP_H / 2, v, ha="center",
                va="center", fontsize=FS, color="white", fontweight="bold")
        for c, (hpi, _, sign) in enumerate(SUBCOLS):
            cv.text(X0 + i * STEP + (c + 0.5) * CELL, 90.7, f"{hpi} {sign}",
                    ha="center", va="top", fontsize=FS_SMALL, color="0.25",
                    rotation=90)
        cv.add_patch(plt.Rectangle((PX0 + i * (LW + LGAP), 86.5), LW, CHIP_H,
                                   fc=VC[v], ec="none"))
        cv.text(PX0 + i * (LW + LGAP) + LW / 2, 86.5 + CHIP_H / 2, v,
                ha="center", va="center", fontsize=FS, color="white",
                fontweight="bold")
    cv.text(JX + JW / 2 + 1.0, 86.5, "JAK\ndependence $J$", ha="center",
            va="top", fontsize=FS, color="0.25", linespacing=1.2)

    y = E_TOP
    block_span = {}
    for n in ROWS:
        genes = blocks[n]
        block_span[n] = (y, len(genes) * CELL)
        for r, g in enumerate(genes):
            cv.text(X0 - 0.8, y + r * CELL + CELL / 2, sym_s.get(g, g),
                    ha="right", va="center", fontsize=FS_SMALL, style="italic")
            for i, v in enumerate(V):
                for c, (hpi, tr, _) in enumerate(SUBCOLS):
                    val = fc.loc[g, f"{v}_{tr}{hpi}"]
                    face = MISSING if pd.isna(val) else FCMAP(FCNORM(val))
                    cv.add_patch(plt.Rectangle(
                        (X0 + i * STEP + c * CELL, y + r * CELL), CELL, CELL,
                        fc=face, ec="0.7", lw=0.2))
        for i in range(len(V)):
            cv.add_patch(plt.Rectangle((X0 + i * STEP, y), BW,
                                       len(genes) * CELL, fill=False,
                                       ec="black", lw=0.6, clip_on=False))
        y += len(genes) * CELL + ROW_GAP

    jh = {}
    for h in ("24", "72"):
        Jg = jak_by_hpi(dg)[h].pivot_table(index="gene_id", columns="virus",
                                           values="jak_index")
        for n in ROWS:
            for v in V:
                jh[n, v, h] = Jg.reindex(members(n))[v].mean()
    for n in ROWS:
        mem = members(n)
        m = fc.loc[mem, [f"{v}_{t}{h}" for v in V for t in ("untreated", "Ruxo")
                         for h in (24, 72)]].median()
        ymax = max(4, int(np.ceil(m.max())))
        tick = 1 if ymax <= 4 else 2 if ymax <= 8 else 4
        ymax = int(np.ceil(ymax / tick) * tick)
        ymin = min(-0.5, float(m.min()) - 0.15 * tick)
        y0, bh = block_span[n]
        for j, v in enumerate(V):
            ax = axes(PX0 + j * (LW + LGAP), y0, LW, bh)
            ax.axvspan(48, 76, **LATE_BAND)
            ax.axhline(0, color="#D0D4D8", lw=0.4, zorder=0)
            for tr, c, lw in (("untreated", GREY, 0.9), ("Ruxo", VC[v], 1.2)):
                yy = [0] + [m[f"{v}_{tr}{h}"] for h in (24, 72)]
                ax.plot([0, 24, 72], yy, "-o", color=c, lw=lw, ms=1.6,
                        mec="none", clip_on=False)
            ax.set(xlim=(-6, 78), ylim=(ymin, ymax), xticks=[0, 24, 72],
                   yticks=[t for t in range(-ymax, ymax + 1, tick)
                           if ymin <= t <= ymax])
            ax.tick_params(labelsize=FS_SMALL)
            if n != ROWS[-1]:
                ax.set_xticklabels([])
            if j:
                ax.set_yticklabels([])
            else:
                lo2, hi2 = ret[n]
                name = LABEL_WRAP.get(n, SHORT[n])
                ax.set_ylabel(
                    f"{name} ({n})\n"
                    f"J = {j24[n]:.2f} · {lo2:.0f}–{hi2:.0f}%"
                    .translate(MINUS),
                    fontsize=FS, rotation=0, ha="right", va="center",
                    color=GROUP_COL[GROUP[n]], linespacing=1.3, labelpad=2)
        ax = axes(JX, y0, JW, bh)
        ax.axhline(0, color="#D0D4D8", lw=0.4, zorder=0)
        ax.axhline(0.5, color="0.8", lw=0.4, ls=(0, (2, 2)), zorder=0)
        for v in V:
            pts = [(int(h), jh[n, v, h]) for h in ("24", "72")
                   if pd.notna(jh[n, v, h])]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                        color=VC[v], lw=1.0, ms=1.6, mec="none",
                        clip_on=False)
        jt = [0, 0.5, 1] if bh >= 8 else [0, 1]
        jmin = min([-0.08] + [jh[n, v, h] - 0.05 for v in V for h in ("24", "72")
                              if pd.notna(jh[n, v, h])])
        ax.set(xlim=(12, 84), ylim=(jmin, 1.05), xticks=[24, 72], yticks=jt,
               yticklabels=[f"{t:g}" for t in jt])
        ax.yaxis.tick_right()
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=FS_SMALL)
        if n != ROWS[-1]:
            ax.set_xticklabels([])

    bot = E_TOP + y - E_TOP - ROW_GAP
    cv.text(PX0 + (5 * LW + 4 * LGAP) / 2, bot + 3.2, "hpi", ha="center",
            va="top", fontsize=FS)

    kx, ky, kw = X0, bot + 3.0, 30.0
    for i in range(120):
        cv.add_patch(plt.Rectangle((kx + kw * i / 120, ky), kw / 120, 2.0,
                                   fc=FCMAP(FCNORM(-4 + 8 * i / 119)),
                                   ec="none", lw=0))
    for val, lab in [(-4, "<−4"), (-2, "−2"), (0, "0"), (2, "+2"), (4, ">+4")]:
        cv.text(kx + kw * (val + 4) / 8, ky + 2.6, lab, ha="center", va="top",
                fontsize=FS_SMALL)
    cv.text(kx + kw + 2.0, ky + 1.0, "log$_2$FC vs mock", ha="left",
            va="center", fontsize=FS)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"fig_jak_combined.{ext}", transparent=True)
    fig.savefig(OUT / "fig_jak_combined.png", dpi=300)
    print(f"wrote {OUT}/fig_jak_combined.{{pdf,svg,png}}  "
          f"({sum(len(g) for g in blocks.values())} genes, core {len(core)}, "
          f"rho {rho:.2f} P {p:.2f}, mean J {gm[0]:.2f}->{gm[1]:.2f}, "
          f"{W:.0f}x{H:.0f} mm)")


if __name__ == "__main__":
    main()
