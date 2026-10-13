#!/usr/bin/env python3
import json

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from pipeline import FIGURES, MM, PNG_DPI, ROOT, set_rcparams
from conservation_programmes import FCMAP, FCNORM, MISSING
from plot_conservation_jaki import SUBCOLS

V = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]
VC = json.loads((ROOT / "config" / "colors.json").read_text())["viruses"]
INK, GREY = "#1F2328", "#9AA0A6"
OUT = FIGURES / "late_inflammation"

MODULE_ANCHORS = {
    "AP-1 unit A\n(DUSP5/FOSL1/MAFF/CCL20)": "FOSL1",
    "AP-1 unit B\n(ETS2/ZFP36)": "ZFP36",
    "chr17q12 cassette\n(CCL3/4, CD70, CH25H)": "CCL4",
    "IRF3-direct\n(CCL5)": "CCL5",
}
AP1_GENES = ["FOS", "EGR1", "JUNB", "DUSP5", "NR4A1", "MAFF", "CCL20"]
LABEL_C = ["CCL4", "CCL5", "CCL3", "IFNB1", "IFNL1", "IFNL2", "FOS", "EGR1"]
ISG_CONTROLS = ["MX1", "IFIT1", "ISG15", "OAS1", "RSAD2"]

dg = pd.read_parquet("data/degs.parquet")
tpm = pd.read_parquet("data/tpm_long.parquet")
sym = dg.drop_duplicates("gene_id").set_index("gene_id")[["symbol", "gene_type"]]
gid = {s: g for g, s in sym.symbol.dropna().items()}
asg = pd.read_csv("expressed_28x27/data/som_neuron_assignments_expressed.tsv",
                  sep="\t").set_index("gene_id")
LFC = dg.pivot_table(index="gene_id", columns=["virus", "treatment", "hpi"],
                     values="log2fc")
DEG = dg.pivot_table(index="gene_id", columns=["virus", "treatment", "hpi"],
                     values="deg", aggfunc="max").astype(bool)


def lfc(s, v, tr, h):
    return float(LFC.loc[gid[s], (v, tr, h)])


def donor_lfc(symbols, hpi=72):
    t = tpm[tpm.gene_id.isin([gid[s] for s in symbols]) & (tpm.hpi == hpi)]
    mock = t[t.infection == "Mock"].set_index(["gene_id", "treatment", "donor"]).tpm
    t = t[t.infection != "Mock"].copy()
    t["lfc"] = np.log2((t.tpm.values + 1) /
                       (mock.reindex(pd.MultiIndex.from_frame(
                           t[["gene_id", "treatment", "donor"]])).values + 1))
    return t.groupby(["infection", "treatment", "donor"]).lfc.mean()


def tag(ax, letter, x=-0.02, y=1.0):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=0.5, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)


YMIN_TOP_A = 5


def panel_a(fig, spec):
    rows = list(MODULE_ANCHORS)
    gs = GridSpecFromSubplotSpec(len(rows), len(V), spec, wspace=0.12, hspace=0.25)
    axes = []
    for i, name in enumerate(rows):
        n = int(asg.loc[gid[MODULE_ANCHORS[name]], "neuron"])
        mem = asg.index[asg.neuron == n].intersection(LFC.index)
        m = LFC.loc[mem].median()
        ymax = max(YMIN_TOP_A, int(np.ceil(m.max())))
        tick = 1 if ymax <= YMIN_TOP_A else 2
        ymax += ymax % tick
        for j, v in enumerate(V):
            ax = fig.add_subplot(gs[i, j], sharey=axes[-1] if j else None)
            axes.append(ax)
            style(ax)
            for tr, c, lw in (("untreated", GREY, 0.9), ("Ruxo", VC[v], 1.2)):
                y = np.array([0, m[(v, tr, 24)], m[(v, tr, 72)]])
                ax.plot([0, 24, 72], y, "-o", color=c, lw=lw, ms=1.8, mec="none",
                        clip_on=False)
            ax.axhline(0, color="#D0D4D8", lw=0.4, zorder=0)
            ax.axvspan(48, 76, color="#F4EFE6", lw=0, zorder=-1)
            ax.set_xticks([0, 24, 72])
            ax.set_xlim(-6, 78)
            ax.set_ylim(-0.5, ymax)
            ax.set_yticks(range(0, ymax + 1, tick))
            ax.tick_params(labelsize=5)
            if i < len(rows) - 1:
                ax.set_xticklabels([])
            if j:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(f"{name}\nN{n}, n={len(mem)}", fontsize=5,
                              rotation=0, ha="right", va="center", labelpad=2)
            if i == 0:
                ax.set_title(v, fontsize=6, color=VC[v], fontweight="bold", pad=2)
    axes[-3].set_xlabel("hpi", fontsize=5, labelpad=1)
    tag(axes[0], "a", x=-1.25, y=1.12)
    fig.text(axes[0].get_position().x0, axes[0].get_position().y1 + 0.022,
             "Median member log2FC vs mock", fontsize=5, color=INK)
    h = [plt.Line2D([], [], color=GREY, lw=0.9, marker="o", ms=1.8, label="untreated"),
         plt.Line2D([], [], color=INK, lw=1.2, marker="o", ms=1.8,
                    label="ruxolitinib (virus colour)")]
    axes[len(V) - 1].legend(handles=h, fontsize=5, frameon=False, loc="lower right",
                            bbox_to_anchor=(1.05, 1.18), ncol=2, handlelength=1.5)


def panel_b(fig, spec):
    ax = fig.add_subplot(spec)
    style(ax)
    jd = pd.read_csv("data/jak_dependence.csv")
    jd["symbol"] = jd.gene_id.map(sym.symbol)
    jd = jd[jd.symbol.isin(AP1_GENES)]
    ax.axvspan(0, 1.05, color="#EEF3F8", lw=0, zorder=0)
    ax.axvspan(-1.05, 0, color="#FBEFEA", lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=0.5)
    off = dict(zip(V, np.linspace(-0.28, 0.28, len(V))))
    for k, g in enumerate(AP1_GENES):
        for _, r in jd[jd.symbol == g].iterrows():
            ax.scatter(r.jak_index, k + off[r.virus], s=10, color=VC[r.virus],
                       edgecolor="white", lw=0.3, zorder=3)
        ax.axhline(k + 0.5, color="#E3E6EA", lw=0.4)
    ax.set_yticks(range(len(AP1_GENES)), AP1_GENES, fontsize=5.5, style="italic")
    ax.set_ylim(len(AP1_GENES) - 0.5, -0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("JAK-dependence index J", fontsize=6)
    ax.tick_params(axis="x", labelsize=5)
    ax.text(0.5, -0.62, "JAK-dependent →", ha="left", fontsize=5, color="#3E6A98")
    ax.text(-0.5, -0.62, "← amplified by JAKi", ha="right", fontsize=5, color="#B5553A")
    for i, v in enumerate(V):
        ax.scatter([], [], s=10, color=VC[v], label=v)
    ax.legend(fontsize=5, frameon=False, loc="upper center", ncol=5,
              bbox_to_anchor=(0.5, -0.13), handletextpad=0.05, columnspacing=0.6)
    tag(ax, "b", x=-0.2)


def panel_c(fig, spec):
    gs = GridSpecFromSubplotSpec(2, 4, spec, wspace=0.35, hspace=0.45,
                                 width_ratios=[1, 1, 0.9, 0.9])
    expressed = dg[dg.hpi == 72].groupby("gene_id").maxgroupmeans.max()
    expressed = expressed.index[expressed >= 1].intersection(LFC.index)
    big = fig.add_subplot(gs[:, :2])
    small = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in (2, 3)]
    order = ["YFV"] + [v for v in V if v != "YFV"]
    lim = (-4, 13)
    for ax, v in zip([big] + small, order):
        style(ax)
        x = LFC.loc[expressed, (v, "untreated", 72)]
        y = LFC.loc[expressed, (v, "Ruxo", 72)]
        up_rx = DEG.loc[expressed, (v, "Ruxo", 72)] & (y > 1)
        up_ut = DEG.loc[expressed, (v, "untreated", 72)] & (x > 1)
        on = up_rx & ~up_ut & (y - x > 2)
        hyper = up_rx & up_ut & (y - x > 2)
        ax.fill_between(lim, [lim[0] + 2, lim[1] + 2], lim[1], color="#FBEFEA",
                        lw=0, zorder=0)
        ax.scatter(x[~(on | hyper)], y[~(on | hyper)], s=0.6, color="#C7CCD1",
                   lw=0, rasterized=True, zorder=1)
        ax.scatter(x[hyper], y[hyper], s=2.5, color="#B5553A", lw=0, zorder=2,
                   rasterized=True)
        ax.scatter(x[on], y[on], s=2.5, color="#6A4C93", lw=0, zorder=2,
                   rasterized=True)
        ax.plot(lim, lim, color=INK, lw=0.5)
        ax.plot(lim, [lim[0] + 2, lim[1] + 2], color=INK, lw=0.5, ls="--")
        ax.axvline(1, color=GREY, lw=0.4, ls=":")
        ax.axhline(1, color=GREY, lw=0.4, ls=":")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        ax.set_title(v, fontsize=6 if ax is not big else 7, color=VC[v],
                     fontweight="bold", pad=2)
        labels = LABEL_C if ax is big else ["CCL4", "CCL5", "IFNB1"]
        for s in labels:
            if gid.get(s) in x.index:
                ax.scatter(x[gid[s]], y[gid[s]], s=8 if ax is big else 4,
                           facecolor="none", edgecolor=INK, lw=0.5, zorder=4)
                ax.annotate(s, (x[gid[s]], y[gid[s]]), xytext=(2, 1.5),
                            textcoords="offset points", fontsize=5 if ax is big else 4.5,
                            style="italic", zorder=5)
    big.set_xlabel("Untreated log2FC vs mock, 72 hpi", fontsize=6)
    big.set_ylabel("Ruxolitinib log2FC vs mock, 72 hpi", fontsize=6)
    big.text(-3.6, -3.4, "shaded: treated − untreated > 2 log2", fontsize=5, color="#B5553A",
             va="bottom")
    big.text(12.6, 11.6, "y = x", fontsize=5, ha="right", color=INK)
    h = [plt.Line2D([], [], ls="", marker="o", ms=2.5, color="#6A4C93",
                    label="switched on (DEG only when treated)"),
         plt.Line2D([], [], ls="", marker="o", ms=2.5, color="#B5553A",
                    label="hyperinduced (DEG in both, Δ > 2)")]
    big.legend(handles=h, fontsize=5, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.06),
               handletextpad=0.1)
    tag(big, "c", x=-0.12)


def panel_d(fig, spec):
    gs = GridSpecFromSubplotSpec(1, 2, spec, wspace=0.3)
    first = None
    for k, g in enumerate(["CCL4", "CCL5"]):
        ax = fig.add_subplot(gs[k], sharey=first)
        first = first or ax
        style(ax)
        d = donor_lfc([g])
        for i, v in enumerate(V):
            u, r = d.loc[(v, "untreated")], d.loc[(v, "Ruxo")]
            x0, x1 = 2.2 * i, 2.2 * i + 1
            for dn in u.index:
                ax.plot([x0, x1], [u[dn], r[dn]], color=VC[v], lw=0.7, alpha=0.9)
            ax.scatter([x0] * len(u), u, s=5, color="white", edgecolor=VC[v], lw=0.6,
                       zorder=3)
            ax.scatter([x1] * len(r), r, s=5, color=VC[v], lw=0, zorder=3)
        ax.axhline(1, color=GREY, lw=0.4, ls=":")
        ax.axhline(0, color="#D0D4D8", lw=0.4)
        ax.set_xticks([2.2 * i + 0.5 for i in range(len(V))], V, fontsize=5,
                      rotation=90)
        ax.set_title(g, fontsize=6.5, style="italic", pad=2)
        ax.tick_params(axis="y", labelsize=5)
        if k == 0:
            ax.set_ylabel("Per-donor log2FC vs mock, 72 hpi", fontsize=6)
            tag(ax, "d", x=-0.28)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    ax.text(1.0, 1.06, "○ untreated  ● ruxolitinib", transform=ax.transAxes,
            fontsize=5, ha="right")


def panel_e(fig, spec):
    ax = fig.add_subplot(spec)
    d72 = dg[dg.hpi == 72]
    rx5 = (d72[(d72.treatment == "Ruxo") & d72.deg & (d72.log2fc > 0)]
           .groupby("gene_id").virus.nunique())
    delta = pd.DataFrame({v: LFC[(v, "Ruxo", 72)] - LFC[(v, "untreated", 72)]
                          for v in V})
    pc = sym.index[sym.gene_type == "protein_coding"]
    cand = delta.loc[delta.index.intersection(rx5.index[rx5 == 5]).intersection(pc)]
    hyper = cand[(cand > 1).all(axis=1)]
    hyper = hyper.loc[hyper.min(axis=1).sort_values(ascending=False).index]
    ctrl = delta.loc[[gid[s] for s in ISG_CONTROLS]]
    M = pd.concat([hyper, ctrl])
    names = list(sym.symbol.reindex(M.index))
    for (i, j), val in np.ndenumerate(M.values):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc=FCMAP(FCNORM(val)),
                                   ec="0.7", lw=0.2))
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=4.5,
                color="white" if abs(val) > 3 else INK)
        if i < len(hyper) and val > 2:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       ec=INK, lw=0.6))
    for top, n in ((0, len(hyper)), (len(hyper), len(ctrl))):
        ax.add_patch(plt.Rectangle((-0.5, top - 0.5), len(V), n, fill=False,
                                   ec="black", lw=0.7))
    for j, v in enumerate(V):
        ax.add_patch(plt.Rectangle((j - 0.45, -1.45), 0.9, 0.8, fc=VC[v], ec="none"))
        ax.text(j, -1.05, v, ha="center", va="center", fontsize=5, color="white",
                fontweight="bold")
    ax.set_xlim(-0.5, len(V) - 0.5)
    ax.set_ylim(len(M) - 0.5, -1.5)
    ax.set_yticks(range(len(M)), names, fontsize=5.5, style="italic")
    ax.set_xticks([])
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(len(V) - 0.3, len(hyper) / 2 - 0.5,
            "ruxolitinib-induced\nin all five,\nΔ > 1 in all five",
            fontsize=5, va="center")
    ax.text(len(V) - 0.3, len(hyper) + len(ctrl) / 2 - 0.5, "canonical\nISGs",
            fontsize=5, va="center")
    pos = ax.get_position()
    W, H = fig.get_size_inches() / MM
    fc_key(fig, pos.x0 * W + 8, pos.y0 * H - 7,
           "Δlog$_2$FC, + minus − (72 hpi)")
    tag(ax, "e", x=-0.25, y=1.06)


def fc_key(fig, x_mm, y_mm, label="log$_2$FC vs mock", note=None):
    kw, kh, bw = 21.5, 2.0, 2.5
    W, H = fig.get_size_inches() / MM
    kax = fig.add_axes([(x_mm - bw - 5) / W, y_mm / H, (kw + 2 * bw + 45) / W, kh / H])
    kax.set_axis_off()
    kax.set_xlim(-bw - 5, kw + bw + 40)
    kax.set_ylim(0, kh)
    for i in range(120):
        kax.add_patch(plt.Rectangle((i * kw / 120, 0), kw / 120 + 0.02, kh,
                                    fc=FCMAP(FCNORM(-4 + 8 * i / 119)), lw=0))
    kax.add_patch(plt.Rectangle((0, 0), kw, kh, fill=False, ec="0.2", lw=0.5))
    for tx in (0, kw / 4, kw / 2, 3 * kw / 4, kw):
        kax.plot([tx, tx], [0, 0.7], color="0.2", lw=0.5, solid_capstyle="butt")
        kax.plot([tx, tx], [kh - 0.7, kh], color="0.2", lw=0.5,
                 solid_capstyle="butt")
    for f, lb in ((0, "−4"), (kw / 4, "−2"), (kw / 2, "0"), (3 * kw / 4, "+2"),
                  (kw, "+4")):
        kax.text(f, -0.6, lb, fontsize=4.6, ha="center", va="top")
    for bx, v, lb, tx, ha in ((-bw - 0.8, -99, "<−4", -bw - 1.1, "right"),
                              (kw + 0.8, 99, ">4", kw + bw + 1.1, "left")):
        kax.add_patch(plt.Rectangle((bx, 0), bw, kh, fc=FCMAP(FCNORM(v)),
                                    ec="0.2", lw=0.5))
        kax.text(tx, kh / 2, lb, fontsize=4.6, ha=ha, va="center")
    kax.text(kw + bw + 5.5, kh / 2, label, fontsize=5.2, va="center")
    if note:
        kax.text(-bw - 5, -3.6, note, fontsize=4.8, va="top")


def panel_f(fig, spec):
    ax = fig.add_subplot(spec)
    unit_of = {int(asg.loc[gid[a], "neuron"]): nm.split("\n")[0]
               for nm, a in MODULE_ANCHORS.items()}
    nde = dg[dg.deg].groupby("gene_id").size()
    ck = sym[sym.symbol.fillna("").str.match(r"^(CCL|CXCL|CX3CL|XCL)\d")
             & (sym.gene_type == "protein_coding")].index
    ck = [g for g in ck if nde.get(g, 0) >= 4]

    def group(g):
        n = asg.neuron.get(g)
        return (unit_of[n], n) if n in unit_of else ("other", None)

    size = pd.Series([group(g) for g in ck]).value_counts()
    ck.sort(key=lambda g: (group(g)[0] == "other", -size[group(g)], group(g)[0],
                           -nde[g]))
    cols = [(v, tr, h) for v in V for h, tr, _ in SUBCOLS]
    ys, y, prev = [], 0.0, None
    for g in ck:
        if prev is not None and group(g) != prev:
            y += 0.5
        ys.append(y)
        prev, y = group(g), y + 1
    step = 4.6
    xs = [step * (k // 4) + k % 4 for k in range(len(cols))]
    for g, y0 in zip(ck, ys):
        for x0, c in zip(xs, cols):
            val = LFC.loc[g, c]
            ax.add_patch(plt.Rectangle(
                (x0, y0), 1, 1, ec="0.7", lw=0.2,
                fc=MISSING if pd.isna(val) else FCMAP(FCNORM(val))))
    groups = list(dict.fromkeys(group(g) for g in ck))
    xr = xs[-1] + 1.6
    for gk in groups:
        yy = [y0 for g, y0 in zip(ck, ys) if group(g) == gk]
        for i in range(len(V)):
            ax.add_patch(plt.Rectangle((step * i, yy[0]), 4, yy[-1] + 1 - yy[0],
                                       fill=False, ec="black", lw=0.7))
        name, n = gk
        lab = name if n is None else (f"{name} (N{n})" if len(yy) == 1
                                      else f"{name}\n(N{n})")
        ax.plot([xr, xr], [yy[0] + 0.1, yy[-1] + 0.9], color=INK, lw=0.5)
        ax.text(xr + 0.4, (yy[0] + yy[-1] + 1) / 2, lab, fontsize=4.8,
                va="center")
    for i, v in enumerate(V):
        ax.add_patch(plt.Rectangle((step * i, -3.5), 4, 1.4, fc=VC[v], ec="none"))
        ax.text(step * i + 2, -2.8, v, ha="center", va="center", fontsize=5,
                color="white", fontweight="bold")
        for c, (hpi, _, sign) in enumerate(SUBCOLS):
            ax.text(step * i + c + 0.5, -0.15, f"{hpi} {sign}", ha="center",
                    va="bottom", fontsize=3.8, color="0.25", rotation=90)
    ax.set_xlim(-0.1, xr + 9.5)
    ax.set_ylim(ys[-1] + 1.1, -3.6)
    ax.set_aspect("equal")
    ax.set_yticks([y0 + 0.5 for y0 in ys], list(sym.symbol.reindex(ck)),
                  fontsize=5, style="italic")
    ax.set_xticks([])
    ax.tick_params(length=0, pad=1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.apply_aspect()
    pos = ax.get_position()
    W, H = fig.get_size_inches() / MM
    fc_key(fig, pos.x0 * W + 8, pos.y0 * H - 7,
           note="− untreated, + ruxolitinib")
    tag(ax, "f", x=-0.14, y=1.0)


def main():
    set_rcparams()
    fig = plt.figure(figsize=(183 * MM, 225 * MM))
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.05, 0.85], hspace=0.32,
                     left=0.17, right=0.97, top=0.95, bottom=0.05)
    r1 = GridSpecFromSubplotSpec(1, 2, outer[0], width_ratios=[2.3, 1], wspace=0.35)
    r2 = GridSpecFromSubplotSpec(1, 2, outer[1], width_ratios=[1.9, 1], wspace=0.28)
    r3 = GridSpecFromSubplotSpec(1, 2, outer[2], width_ratios=[1, 1.5], wspace=0.75)
    panel_a(fig, r1[0])
    panel_b(fig, r1[1])
    panel_c(fig, r2[0])
    panel_d(fig, r2[1])
    panel_e(fig, r3[0])
    panel_f(fig, r3[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"fig_late_inflammation.{ext}")
    fig.savefig(OUT / "fig_late_inflammation.png", dpi=PNG_DPI)
    print("wrote", OUT / "fig_late_inflammation.{pdf,svg,png}")


if __name__ == "__main__":
    main()
