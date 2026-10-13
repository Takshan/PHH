#!/usr/bin/env python3
import os
import re
import sys
import textwrap

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from pycirclize import Circos

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline
from timepoint_map_compare import jak_by_hpi

FONT_FAMILY = ["Times New Roman", "Nimbus Roman", "Liberation Serif"]
TEXT_SIZE_PT = 6.0

LINE_W_DATA = 0.3
LINE_W_STRUCT = 0.4
POINT_SIZE_PT = 1.2

WIDTH_MM, HEIGHT_MM = 180.0, 178.0

WONG = ["#E69F00",
        "#56B4E9",
        "#009E73", "#CC79A7", "#D55E00", "#F0E442", "#0072B2", "#000000"]
VIRUS_ORDER = ["WNV", "YFV"]

VIRIDIS_MAP = "viridis"

INK, STRUCT_GREY, SECTOR_EDGE = "#1A1A1A", "#8C8C8C", "white"
NA_GREY = "#CFCAC2"
RIBBON_ALPHA = 0.55

R_RIM = (92.0, 100.0)
R_MARK = (101.0, 103.5)
R_LINK = 90.5
R_LABEL_PROG = 105.0
R_LABEL_TAIL = 106.0
SPACE_SMALL, SPACE_BIG = 0.8, 5.0
MIN_UNITS = 1.2
LABEL_CAP, LABEL_WRAP = 30, 22
LEGEND_XY = (0.012, 0.988)
CBAR_BOUNDS = (0.770, 0.930, 0.200, 0.013)
PREVIEW_DPI = 400

E = "expressed_28x27/data"
V = list(pipeline.VIRUS_ORDER)
OUT = "figures/plot_jak_tier_chord/jak_tier_chord_circos"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MODULE_TABLE = "data/som_expressed_module_table.csv"
TAILS = [("WNV", "suppressed"), ("YFV", "suppressed"),
         ("YFV", "resistant"), ("WNV", "resistant")]
TAIL_FILE = {"suppressed": "up", "resistant": "down"}
VIRUS_COLOR = dict(zip(VIRUS_ORDER, WONG))


def resolve_font(candidates):
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((c for c in candidates if c in have), "serif")


FONT = resolve_font(FONT_FAMILY)


def derive():
    dg = pd.read_parquet("data/degs.parquet")
    asg = pd.read_csv(f"{E}/som_neuron_assignments_expressed.tsv", sep="\t").set_index("gene_id")
    fc = pd.read_parquet(f"{E}/expressed_matrix_masked_log2fc.parquet")

    g = dg.groupby("gene_id").agg(maxtpm=("maxgroupmeans", "max"))
    nd = dg[dg.deg].groupby("gene_id").size().rename("nd")
    A = asg.join(g).join(nd).fillna({"nd": 0})
    us = A.groupby("neuron").agg(n=("neuron", "size"), mt=("maxtpm", "median"), md=("nd", "median"))
    sup = us[(us.n >= 4) & (us.mt > 3) & (us.md >= 4)].index
    cent = fc.join(asg["neuron"]).groupby("neuron").mean()
    ut = [f"{v}_untreated{h}" for v in V for h in (24, 72)]
    induced = [n for n in sup if cent.loc[n, ut].mean() > 0]

    j24g = jak_by_hpi(dg)["24"].pivot_table(index="gene_id", columns="virus", values="jak_index")

    def jak24(n):
        x = j24g.reindex(asg.index[asg.neuron == n]).stack()
        return float(x.mean()) if len(x) else np.nan

    J24 = pd.Series({n: jak24(n) for n in induced}).dropna()

    enr = pd.read_csv(f"{E}/dependent_neuron_enrichment_expressed.csv")
    depset = {int(x) for x in enr.dependent_neurons.iloc[0].split()}
    names = pd.read_csv(MODULE_TABLE).set_index("unit")["module"].replace("NEEDS_NAME", np.nan)

    n_of = asg["neuron"]
    rows = []
    for virus, tail in TAILS:
        t = pd.read_csv(f"data/top50_isg_{TAIL_FILE[tail]}_{virus.lower()}.csv")
        for gid in t.gene_id:
            n = n_of.get(gid, np.nan)
            dest = int(n) if pd.notna(n) and int(n) in J24.index else None
            rows.append(dict(virus=virus, tail=tail, gene_id=gid, dest=dest))
    flows = pd.DataFrame(rows)
    got = flows["dest"].value_counts()

    ligand = next((int(n) for n in J24.index
                   if isinstance(names.get(int(n)), str)
                   and "interferon ligand" in names[int(n)].lower()), None)
    keep = sorted({int(k) for k in got.index} | ({ligand} if ligand else set()),
                  key=lambda n: J24[n])
    prog = [dict(neuron=n, J=float(J24[n]), n_genes=int(got.get(n, 0)),
                 dep=n in depset,
                 label=names.get(n) if isinstance(names.get(n), str) else f"unit {n}")
            for n in keep]
    return prog, flows, enr, depset


def short(name):
    if len(name) > LABEL_CAP and "(" in name:
        name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    return "\n".join(textwrap.wrap(name, LABEL_WRAP, break_long_words=False,
                                   break_on_hyphens=False)[:2])


def main():
    plt.rcParams.update({"font.family": FONT, "font.size": TEXT_SIZE_PT})
    prog, flows, enr, depset = derive()
    en = enr.set_index(["virus", "tail"])

    jvals = np.array([p["J"] for p in prog], dtype=float)
    nbin = int(np.clip(len(jvals) // 3, 4, 8))
    breaks = np.unique(np.quantile(jvals, np.linspace(0, 1, nbin + 1)))
    cmap = plt.get_cmap(VIRIDIS_MAP)
    norm = BoundaryNorm(breaks, cmap.N, extend="neither")
    jcolor = {p["neuron"]: cmap(norm(p["J"])) for p in prog}

    pname = {p["neuron"]: f"P{p['neuron']}" for p in prog}
    sectors = {pname[p["neuron"]]: max(p["n_genes"], MIN_UNITS) for p in prog}
    tname = {(v, t): f"{v} {t}" for v, t in TAILS}
    sectors.update({tname[(v, t)]: 50.0 for v, t in TAILS})

    spaces = ([SPACE_SMALL] * (len(prog) - 1) + [SPACE_BIG]
              + [SPACE_SMALL] * (len(TAILS) - 1) + [SPACE_BIG])
    unit_deg = (360.0 - sum(spaces)) / sum(sectors.values())
    prog_span = sum(max(p["n_genes"], MIN_UNITS) for p in prog) * unit_deg \
        + SPACE_SMALL * (len(prog) - 1)
    start = -prog_span / 2
    circos = Circos(sectors, start=start, end=start + 360, space=spaces)

    for p in prog:
        s = circos.get_sector(pname[p["neuron"]])
        s.add_track(R_RIM).axis(fc=jcolor[p["neuron"]], ec=SECTOR_EDGE, lw=LINE_W_DATA)
        if p["dep"]:
            s.add_track(R_MARK).axis(fc=INK, ec="none")
        s.text(short(p["label"]), r=R_LABEL_PROG, orientation="vertical",
               size=TEXT_SIZE_PT, color=INK, linespacing=1.15)

    flow_at_end = {tname[k]: i >= len(TAILS) // 2 for i, k in enumerate(TAILS)}
    for v, t in TAILS:
        s = circos.get_sector(tname[(v, t)])
        flow = int(flows[(flows["virus"] == v) & (flows["tail"] == t)]["dest"].notna().sum())
        track = s.add_track(R_RIM)
        track.axis(fc=NA_GREY, ec=SECTOR_EDGE, lw=LINE_W_DATA)
        lo, hi = (50.0 - flow, 50.0) if flow_at_end[tname[(v, t)]] else (0.0, flow)
        track.rect(lo, hi, fc=VIRUS_COLOR[v], ec=SECTOR_EDGE, lw=LINE_W_DATA)
        n_dep = int(en.loc[(v, t), "n_in_dependent"])
        s.text(f"{v}   50 most {t}\n{n_dep} of 50 JAK-dependent", r=R_LABEL_TAIL,
               orientation="horizontal", size=TEXT_SIZE_PT, color=INK, linespacing=1.35)

    order = {p["neuron"]: i for i, p in enumerate(prog)}
    cursor = {p["neuron"]: 0.0 for p in prog}
    for v, t in TAILS:
        sub = flows[(flows["virus"] == v) & (flows["tail"] == t)]
        counts = sub["dest"].value_counts()
        at_end = flow_at_end[tname[(v, t)]]
        x = 50.0 - int(counts.sum()) if at_end else 0.0
        for neuron in sorted(counts.index, key=lambda n: order[n] if at_end else -order[n]):
            n_genes = int(counts[neuron])
            y0 = cursor[neuron]
            y1 = y0 + n_genes
            cursor[neuron] = y1
            circos.link((tname[(v, t)], x, x + n_genes), (pname[neuron], y0, y1),
                        r1=R_LINK, r2=R_LINK, color=jcolor[neuron],
                        alpha=RIBBON_ALPHA, height_ratio=0.42, allow_twist=False)
            x += n_genes

    fig = circos.plotfig(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=PREVIEW_DPI)

    handles = ([Patch(fc=VIRUS_COLOR[v], ec=SECTOR_EDGE, lw=LINE_W_DATA, label=v)
                for v in VIRUS_ORDER]
               + [Patch(fc=NA_GREY, ec=SECTOR_EDGE, lw=LINE_W_DATA,
                        label="in no induced programme"),
                  Patch(fc=INK, ec="none", label="JAK-dependent set (pooled J ≥ 0.5)")])
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=LEGEND_XY,
               ncol=1, frameon=False, handlelength=1.1, handleheight=1.1,
               labelspacing=0.55, prop={"family": FONT, "size": TEXT_SIZE_PT})

    cax = fig.add_axes(CBAR_BOUNDS)
    jticks = [t for t in (0.0, 0.25, 0.5, 0.75, 1.0) if breaks[0] <= t <= breaks[-1]]
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=jticks, spacing="proportional")
    cb.set_label("dependence index J at 24 hpi", size=TEXT_SIZE_PT, family=FONT, color=INK)
    cb.ax.tick_params(labelsize=TEXT_SIZE_PT, length=1.5, width=LINE_W_DATA, color=INK)
    cb.ax.set_xticklabels([f"{t:g}" for t in jticks], family=FONT, color=INK)
    cb.outline.set_linewidth(LINE_W_DATA)

    fig.savefig(f"{OUT}.pdf", transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}.png", dpi=PREVIEW_DPI, facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"programmes on the rim : {len(prog)} "
          f"({sum(p['dep'] for p in prog)} in the dependent set), "
          f"J quantile breaks: {np.round(breaks, 2).tolist()}")
    for v, t in TAILS:
        sub = flows[(flows["virus"] == v) & (flows["tail"] == t)]
        nd = sum(1 for d in sub["dest"] if pd.notna(d) and int(d) in depset)
        print(f"  {v} {t:<10} {nd:2d}/50 into the dependent set, "
              f"{int(sub['dest'].isna().sum())} in no induced programme")
    print(f"wrote:\n    {OUT}.pdf\n    {OUT}.png")


if __name__ == "__main__":
    main()
