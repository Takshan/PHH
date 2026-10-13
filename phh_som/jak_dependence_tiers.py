#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pandas as pd
from plotnine import (aes, annotate, element_blank, element_line,
                      element_rect, element_text, geom_hline, geom_point,
                      geom_text, geom_tile, ggplot, guide_legend, guides, labs,
                      scale_color_identity, scale_color_manual, scale_fill_manual,
                      scale_shape_manual, scale_size_identity, scale_x_continuous,
                      scale_y_continuous, theme, theme_bw)
from plotnine._mpl.gridspec import p9GridSpec
from plotnine.composition import Stack
from scipy import stats as st

from pipeline import FIGURES, ROOT
from timepoint_map_compare import jak_by_hpi
from conservation_programmes import programmes

WIDTH_MM = 180
PANEL_MM = 70
KEY_ROW_MM = 3.3
KEY_COLS = 2
KEY_COL_STEP = 0.62
TEXT_SIZE_PT = 7
FONT = "Liberation Serif"
LINE_WIDTH = 0.4
LEGEND_KEY_PT = 8
LEGEND_KEY_GAP_PT = 1
POINT_SIZE = 2.6
CONTEXT_POINT_SIZE = 1.4
OPEN_STROKE = 0.45
SHAPE_SCALE = {"*": 1.35}
WONG = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7",
        "#D55E00", "#F0E442", "#0072B2", "#000000"]
CONTEXT_GREY = "#BDBDBD"
GRID_GREY = "#E5E5E5"
NC_GREY = "#EFEFEF"
CMAP = "viridis"
UNIT_SHAPES = ["s", "D", "*", "^", "v", "P", "X"]
CONTEXT_SHAPE = "o"
LABEL_SPAN = 7.6

V = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]
E = "expressed_28x27/data"
OUT = FIGURES / "jak_dependence_tiers"

DEP_CUT = 0.5
MIN_AMP = 1.0
N_GENES = 5
RESISTANT, OAS = [756, 728], [726]
GROUPS = ["JAK-resistant tier", "OAS1/3–RIGI–DDX60 unit", "four ISGF3 programs"]
CONTEXT = "other induced programs (mean $z$ > 1)"
WEAK = "weakly induced (mean $z$ ≤ 1), not in ρ"
MINUS = str.maketrans("-", "−")


class RatioStack(Stack):
    heights = None

    def _create_gridspec(self, figure, nest_into):
        self.gridspec = p9GridSpec(self.nrow, self.ncol, figure,
                                   nest_into=nest_into, height_ratios=self.heights)

mpl.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none",
                     "mathtext.fontset": "custom", "mathtext.rm": FONT,
                     "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
                     "mathtext.cal": FONT, "mathtext.sf": FONT})


def load():
    dg = pd.read_parquet("data/degs.parquet")
    sym = dg.drop_duplicates("gene_id").set_index("gene_id")[["symbol", "gene_type"]]
    asg = pd.read_csv(f"{E}/som_neuron_assignments_expressed.tsv",
                      sep="\t").set_index("gene_id")
    cb = pd.read_csv(f"{E}/som_codebook_expressed.tsv", sep="\t").set_index("neuron")
    nstat = pd.read_csv(f"{E}/som_neuron_stats_expressed.csv").set_index("neuron")
    fc = pd.read_parquet(f"{E}/expressed_matrix_masked_log2fc.parquet")
    return dg, sym, asg, cb, nstat, fc


def supported(dg, asg):
    g_max = dg.groupby("gene_id").agg(maxtpm=("maxgroupmeans", "max"))
    n_de = dg[dg.deg].groupby("gene_id").size().rename("nd")
    A = asg.join(g_max).join(n_de).fillna({"nd": 0})
    us = A.groupby("neuron").agg(n=("neuron", "size"), mt=("maxtpm", "median"),
                                 md=("nd", "median"))
    return us[(us.n >= 4) & (us.mt > 3) & (us.md >= 4)].index, us


def base_theme(height_mm):
    t = TEXT_SIZE_PT
    return theme_bw(base_size=t, base_family=FONT) + theme(
        text=element_text(size=t, family=FONT, color="black"),
        axis_text=element_text(size=t, color="black"),
        axis_title=element_text(size=t),
        legend_title=element_text(size=t, ha="left"),
        legend_text=element_text(size=t),
        plot_tag=element_text(size=t, weight="bold"),
        plot_title=element_blank(), plot_subtitle=element_blank(),
        panel_grid_major=element_line(color=GRID_GREY, size=LINE_WIDTH),
        panel_grid_minor=element_blank(),
        panel_border=element_rect(color="black", size=LINE_WIDTH),
        axis_ticks=element_line(color="black", size=LINE_WIDTH),
        axis_ticks_minor=element_blank(),
        panel_background=element_rect(fill="white"),
        plot_background=element_rect(fill="none", color="none"),
        legend_background=element_rect(fill="none", color="none"),
        legend_key=element_rect(fill="none", color="none"),
        legend_key_size=LEGEND_KEY_PT,
        legend_key_spacing_y=LEGEND_KEY_GAP_PT,
        figure_size=(WIDTH_MM / 25.4, height_mm / 25.4),
    )


def quantile_bands(vals):
    k = int(np.clip(len(vals) // 6, 3, 6))
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, k + 1)))
    labels = [f"{lo:.0f} to {hi:.0f}".translate(MINUS)
              for lo, hi in zip(edges[:-1], edges[1:])]
    cols = [mpl.colors.to_hex(c)
            for c in mpl.colormaps[CMAP](np.linspace(0, 1, len(labels)))]
    return edges, labels, cols


def main():
    dg, sym, asg, cb, nstat, fc = load()
    SUP, us = supported(dg, asg)
    UT = [f"{v}_untreated{h}" for v in V for h in (24, 72)]
    Jn = nstat[[f"mean_jak_index_{v.lower()}" for v in V]]
    Jn.columns = V
    S24 = SUP[cb.loc[SUP, UT].mean(axis=1) > 1]
    IND = programmes(dg, asg, cb)[0].index
    assert set(S24) <= set(IND)
    mj = Jn.loc[S24].mean(axis=1)
    amp = cb.loc[IND, UT].max(axis=1)
    module = pd.read_csv(ROOT / "phh_som" / "data" / "som_expressed_module_table.csv",
                         index_col="unit")["module"]
    maxtpm = dg.groupby("gene_id")["maxgroupmeans"].max()

    jhp = jak_by_hpi(dg)
    J24g = jhp["24"].pivot_table(index="gene_id", columns="virus",
                                 values="jak_index")

    def members(n):
        return list(asg.index[asg.neuron == n])

    def top_genes(n):
        s = sym.reindex(members(n)).assign(tpm=maxtpm).dropna(subset=["symbol"])
        s = s[~s.symbol.str.startswith("ENSG")].sort_values("tpm", ascending=False)
        return list(s.symbol[:N_GENES])

    j24 = pd.Series({n: float(J24g.reindex(members(n)).stack().mean()) for n in IND})
    assert j24[S24].notna().all()
    OPEN = [n for n in IND if n not in S24 and pd.notna(j24[n])]
    DEP = sorted(mj.index[mj >= DEP_CUT], key=lambda n: -j24[n])
    group = {n: g for g, units in enumerate([RESISTANT, OAS, DEP]) for n in units}
    rows = RESISTANT + OAS + DEP
    shape = dict(zip(rows, UNIT_SHAPES))
    gcol = dict(zip(GROUPS, WONG))
    gcol[CONTEXT] = gcol[WEAK] = CONTEXT_GREY
    gfill = {**gcol, WEAK: "white"}
    unit_lab = {n: f"{n}  {module[n]}" for n in rows}
    rho, p = st.spearmanr(j24[S24], amp[S24])
    rho_pool, p_pool = st.spearmanr(mj, amp[S24])
    shown = list(S24) + OPEN
    rho_all, p_all = st.spearmanr(j24[shown], amp[shown])
    print(f"{len(S24)} supported induced programmes; 24-hpi J vs amplitude "
          f"rho={rho:.3f} P={p:.3f} (pooled J: rho={rho_pool:.3f} P={p_pool:.3f}); "
          f"dependent units {DEP} (24-hpi J {j24[DEP].round(2).tolist()}); "
          f"others with 24-hpi J >= {j24[DEP].min():.2f}: "
          f"{sorted(n for n in S24 if n not in group and j24[n] >= j24[DEP].min())}")
    print(f"Fig 5D induced: {len(IND)}; open points {sorted(OPEN)}; no 24-hpi J "
          f"{sorted(set(IND) - set(shown))}; all {len(shown)} plotted: "
          f"rho={rho_all:.3f} P={p_all:.3f}")

    n_grey = sum(n not in group for n in shown)
    key_mm = int(np.ceil(n_grey / KEY_COLS)) * KEY_ROW_MM + 2
    height_mm = 2 * PANEL_MM + key_mm

    A = pd.DataFrame({"unit": shown, "amp": amp[shown].to_numpy(),
                      "j24": j24[shown].to_numpy()})
    named = A.unit.isin(rows)
    is_open = A.unit.isin(OPEN)
    A["group"] = [GROUPS[group[n]] if n in group else WEAK if n in OPEN else CONTEXT
                  for n in A.unit]
    A["key"] = [unit_lab[n] if n in group else CONTEXT for n in A.unit]
    A["size"] = [POINT_SIZE * SHAPE_SCALE.get(shape[n], 1) if n in group
                 else CONTEXT_POINT_SIZE for n in A.unit]
    A["label"] = [("" if n in group else str(n)) for n in A.unit]
    A["group"] = pd.Categorical(A.group, GROUPS + [CONTEXT, WEAK])
    A["key"] = pd.Categorical(A.key, [unit_lab[n] for n in rows] + [CONTEXT])
    stat = (f"Spearman ρ = {rho:.2f}, $P$ = {p:.2f}\n"
            f"$n$ = {len(S24)} programs (filled)").translate(MINUS)
    y_lo = min(-0.05, A.j24.min() - 0.05)
    y_hi = max(1.0, A.j24.max() + 0.05)
    pa = (ggplot(A, aes("amp", "j24"))
          + geom_hline(yintercept=0, color="black", size=LINE_WIDTH * 0.6)
          + geom_point(A[~named & ~is_open], aes(fill="group", color="group",
                                                 shape="key", size="size"), stroke=0.2)
          + geom_point(A[is_open], aes(fill="group", color="group", shape="key",
                                       size="size"), stroke=OPEN_STROKE)
          + geom_text(A, aes(label="label"), size=TEXT_SIZE_PT,
                      family=FONT, color="#616161",
                      adjust_text={"expand": (1.7, 2.0), "force_static": (0.6, 1.0),
                                   "arrowprops": {"arrowstyle": "-", "lw": 0.3,
                                                  "color": "#9E9E9E"}})
          + geom_point(A[named], aes(fill="group", color="group", shape="key",
                                     size="size"), stroke=0.2)
          + annotate("text", x=amp.max(), y=0.42, label=stat, ha="right",
                     size=TEXT_SIZE_PT, family=FONT)
          + scale_fill_manual(values=gfill, name="Group")
          + scale_color_manual(values=gcol, name="Group")
          + scale_shape_manual(values={**{unit_lab[n]: shape[n] for n in rows},
                                       CONTEXT: CONTEXT_SHAPE},
                               breaks=[unit_lab[n] for n in rows], name="Unit")
          + scale_size_identity()
          + scale_y_continuous(limits=(y_lo, y_hi), breaks=np.arange(0, 1.01, 0.2))
          + guides(fill=guide_legend(order=1, override_aes={"shape": "o", "size": 3,
                                                            "stroke": OPEN_STROKE}),
                   color=guide_legend(order=1),
                   shape=guide_legend(order=2, override_aes={
                       "fill": [gcol[GROUPS[group[n]]] for n in rows],
                       "color": [gcol[GROUPS[group[n]]] for n in rows],
                       "size": [POINT_SIZE * SHAPE_SCALE.get(shape[n], 1)
                                for n in rows]}))
          + labs(x="Peak untreated induction (SOM codebook $z$)",
                 y="JAK dependence $J$ at 24 hpi", tag="a")
          + base_theme(height_mm))

    K = pd.DataFrame({"unit": sorted(n for n in shown if n not in group)})
    n_rows = int(np.ceil(len(K) / KEY_COLS))
    K["x"] = (K.index // n_rows) * KEY_COL_STEP
    K["y"] = n_rows - K.index % n_rows
    K["text"] = [f"{n}  {module[n]}" for n in K.unit]
    k_open = K.unit.isin(OPEN)
    pk = (ggplot(K, aes("x", "y"))
          + geom_point(K[~k_open], shape=CONTEXT_SHAPE, fill=CONTEXT_GREY,
                       color=CONTEXT_GREY, size=CONTEXT_POINT_SIZE, stroke=0.2)
          + geom_point(K[k_open], shape=CONTEXT_SHAPE, fill="white",
                       color=CONTEXT_GREY, size=CONTEXT_POINT_SIZE, stroke=OPEN_STROKE)
          + geom_text(aes(x="x + 0.015", label="text"), ha="left",
                      size=TEXT_SIZE_PT, family=FONT)
          + scale_x_continuous(limits=(-0.01, 1), expand=(0, 0))
          + scale_y_continuous(limits=(0.4, n_rows + 0.6), expand=(0, 0))
          + labs(x="", y="")
          + base_theme(height_mm)
          + theme(panel_grid_major=element_blank(), panel_border=element_blank(),
                  axis_ticks=element_blank(), axis_text=element_blank()))

    ret = {}
    for n in rows:
        for v in V:
            den = fc.loc[members(n), f"{v}_untreated24"].mean()
            num = fc.loc[members(n), f"{v}_Ruxo24"].mean()
            ret[n, v] = 100 * num / den if abs(den) >= MIN_AMP else np.nan
    B = pd.Series(ret).rename("ret").rename_axis(["unit", "virus"]).reset_index()
    edges, bands, band_cols = quantile_bands(B.ret.dropna().to_numpy())
    B["band"] = pd.cut(B.ret, edges, labels=bands, include_lowest=True)
    B["band"] = B.band.cat.add_categories("n.c.").fillna("n.c.")
    B["x"] = B.virus.map({v: i + 1 for i, v in enumerate(V)})
    B["y"] = B.unit.map({n: len(rows) - i for i, n in enumerate(rows)})
    B["text"] = [("n.c." if np.isnan(r) else f"{r:.0f}").translate(MINUS)
                 for r in B.ret]
    dark = dict(zip(bands, [i < len(bands) / 2 for i in range(len(bands))]))
    B["tcol"] = ["white" if dark.get(b, False) else "black" for b in B.band]
    R = pd.DataFrame({"unit": rows, "y": [len(rows) - i for i in range(len(rows))]})
    R["genes"] = [", ".join(top_genes(n)) for n in rows]
    R["name"] = [f"{unit_lab[n]}  ·  " + f"J = {j24[n]:.2f}".translate(MINUS)
                 for n in rows]
    lab_x = -0.2
    pb = (ggplot(B, aes("x", "y"))
          + geom_tile(aes(fill="band"), width=0.94, height=0.88)
          + geom_text(aes(label="text", color="tcol"), size=TEXT_SIZE_PT, family=FONT)
          + geom_text(R, aes(x=lab_x, y="y + 0.17", label="name"), ha="right",
                      size=TEXT_SIZE_PT, family=FONT)
          + geom_text(R, aes(x=lab_x, y="y - 0.17", label="genes"), ha="right",
                      fontstyle="italic", size=TEXT_SIZE_PT, family=FONT)
          + scale_fill_manual(values={**dict(zip(bands, band_cols)), "n.c.": NC_GREY},
                              name="Retained at 24 hpi (%)")
          + scale_color_identity()
          + scale_x_continuous(breaks=list(range(1, len(V) + 1)), labels=V,
                               limits=(-LABEL_SPAN, len(V) + 0.55), expand=(0, 0))
          + scale_y_continuous(limits=(0.4, len(rows) + 0.6), expand=(0, 0))
          + labs(x="", y="", tag="b")
          + base_theme(height_mm)
          + theme(panel_grid_major=element_blank(), panel_border=element_blank(),
                  axis_ticks=element_blank(), axis_text_y=element_blank()))
    for n in rows:
        pb += geom_point(R[R.unit == n], aes(x=0.2, y="y"), shape=shape[n],
                         fill=gcol[GROUPS[group[n]]], color=gcol[GROUPS[group[n]]],
                         stroke=0.2,
                         size=POINT_SIZE * SHAPE_SCALE.get(shape[n], 1))

    for n in rows:
        r = [ret[n, v] for v in V if not np.isnan(ret[n, v])]
        print(f"  N{n} [{shape[n]}] retained {min(r):.0f}% to {max(r):.0f}% "
              f"({len(r)} of {len(V)} viruses), 24-hpi J {j24[n]:.2f}; "
              f"{', '.join(top_genes(n))}")
    print("  retention bands:", dict(zip(bands, band_cols)))

    stack = RatioStack([pa, pk, pb])
    stack.heights = [PANEL_MM, key_mm, PANEL_MM]
    fig = stack.draw()
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text() in set(K.text):
                t.set_clip_on(False)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"fig_jak_tiers.{ext}", transparent=True)
    fig.savefig(OUT / "fig_jak_tiers.png", dpi=300)
    print("wrote", OUT / "fig_jak_tiers.{pdf,svg,png}")


if __name__ == "__main__":
    main()
