#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import blended_transform_factory

from pipeline import (CLIP_Z, MM, PUBLISH, VIRUS_ORDER, ROOT,
                      load_isg_symbols, save_figure, set_rcparams)
from plot_som_map import Grid, grid_dims, lighten, surface
from plot_response_terrain import COND_TITLE, virus_colors
from plot_som_map import fan_scale
from plot_codes_map import (draw_codes_map, fan_glyph,
                                fan_key, panel_geometry)

UNIVERSE = PUBLISH / "expressed_28x27"
DATA = UNIVERSE / "data"
FIGDIR = UNIVERSE / "figures" / "fig6_variants"
TAG = "expressed"
CONDS = ["untreated24", "untreated72", "Ruxo24", "Ruxo72"]

CLASSES = {
    "ifn":     ("Type I/III interferon ligands", "#CC79A7"),
    "sense":   ("Sensing & IRF3-direct effectors", "#D55E00"),
    "isgf3":   ("ISGF3 amplification & effectors", "#E69F00"),
    "antigen": ("Antigen presentation & immune interface", "#009E73"),
    "inflam":  ("Inflammatory, NF-κB/AP-1 & stress", "#0072B2"),
    "metab":   ("Hepatocyte metabolic repression", "#8C6D46"),
    "tech":    ("rRNA / MT-RNR (technical)", "#B8BCC2"),
}

FUNCTIONAL_CLASS = {
    756: "ifn",
    728: "sense", 726: "sense", 698: "sense",
    754: "isgf3", 753: "isgf3", 751: "isgf3", 727: "isgf3", 725: "isgf3",
    724: "isgf3", 723: "isgf3", 722: "isgf3", 721: "isgf3", 696: "isgf3",
    695: "isgf3",
    697: "antigen", 677: "antigen", 666: "antigen", 665: "antigen",
    637: "antigen",
    693: "inflam", 668: "inflam", 650: "inflam", 649: "inflam", 648: "inflam",
    638: "inflam", 622: "inflam", 594: "inflam", 593: "inflam", 582: "inflam",
    565: "inflam", 554: "inflam", 749: "inflam",
    113: "metab", 93: "metab", 85: "metab", 65: "metab", 62: "metab",
    61: "metab", 59: "metab", 58: "metab", 57: "metab", 34: "metab",
    33: "metab", 32: "metab", 31: "metab", 9: "metab", 6: "metab", 4: "metab",
    56: "tech",
}

CALLOUTS = {
    756: ("Interferon ligands", "IFNL1–4, IFNB1"),
    728: ("IRF3-direct effectors", "IFIT1/2, RSAD2, CXCL10/11, CCL5"),
    727: ("ISG effector core", "ISG15, MX1, IFI6, NRIR, MX1-AS1"),
    726: ("RLR sensing & OAS amplifier", "OAS1/3, RIGI, DDX60"),
    725: ("ISGF3 signalling & checkpoints", "STAT1, SAMHD1, CD274"),
    695: ("ISGF3 core & antigen processing", "HLA-B, TAP1, PSMB9, BST2, IRF7"),
    666: ("MHC-I antigen presentation", "B2M, HLA-A/C/E, TAP2, LY6E"),
    668: ("Secreted immunomodulators", "IL33, IL4I1, MDK, LAG3"),
    582: ("HIF / glycolytic", "HK2, STC1, DKK1"),
    56:  ("rRNA / MT-RNR", "technical"),
    648: ("Integrated stress response", "ATF3, BBC3, PPP1R15A"),
    650: ("IRF1 / NF-κB arm", "IRF1, SOCS3"),
    594: ("Immediate-early", "ETS2, ZFP36"),
    93:  ("Core hepatocyte identity", "CYP2E1, CYP3A5, CAT, SCD"),
    59:  ("Ketogenesis", "HMGCS2, AKR1D1, LECT2"),
    32:  ("Conjugation", "UGT2B15, SULT1E1, ACOT12"),
    34:  ("Detoxification", "GSTA2, ACOX1, UGT1A9"),
}
RESTRICTED = {648: "WNV", 32: "DENV"}

NAMED_UNSUPPORTED = {
    640: "orphan late wave",
    700: "CCL3/CCL4 cassette",
    621: "FOS",
    620: "EGR1/JUNB",
    19: "DENV acute phase",
    42: "DENV acute phase",
}

REGION_NAMES = {
    756: "Interferon, ISG and antigen-presentation core",
    648: "Inflammatory, stress and immediate-early",
    59: "Hepatocyte metabolic and secretory",
}

IDENTITY_GENES = ["HNF4A", "HNF1A", "FOXA1", "FOXA2", "NR5A2"]

CITED_IN_RESULTS = {756, 754, 753, 751, 728, 727, 726, 725, 724, 722, 698,
                    695, 668, 666, 648, 640, 621, 620, 700, 42, 32, 19}

KIN_COLORS = {
    ("induced", "early"): "#FBC3A7", ("induced", "sustained"): "#E4694A",
    ("induced", "late"): "#9E2A20",
    ("repressed", "early"): "#BBD4E8", ("repressed", "sustained"): "#5B9BD5",
    ("repressed", "late"): "#1F4E79",
}
TOP_STRIP = 3.4
BOTTOM_STRIP = 3.4
CONSERVATION_RAMP = ["#DCD3E8", "#B9A5D1", "#9477BA", "#6E4BA0", "#472A78"]
BLANK = "#F2F3F4"
HEX_EDGE = "#E3E5E7"


class Context:

    def __init__(self):
        self.grid = Grid(*grid_dims(DATA, TAG))
        self.cb = pd.read_csv(DATA / f"som_codebook_{TAG}.tsv", sep="\t",
                              index_col=0)
        self.assign = pd.read_csv(DATA / f"som_neuron_assignments_{TAG}.tsv",
                                  sep="\t")
        self.stats = pd.read_csv(DATA / f"som_neuron_stats_{TAG}.csv",
                                 index_col="neuron")
        self.kin = pd.read_csv(DATA / "s25_kinetic_classes.csv",
                               index_col="neuron")
        self.mod = pd.read_csv(ROOT / "phh_som" / "data" / "som_expressed_module_table.csv",
                               index_col="unit")
        degs = pd.read_parquet(PUBLISH / "data" / "degs.parquet")

        g = degs.groupby("gene_id").agg(maxtpm=("maxgroupmeans", "max"))
        nd = degs[degs.deg].groupby("gene_id").size().rename("nd")
        a = (self.assign.set_index("gene_id")[["neuron"]].join(g).join(nd)
             .fillna({"nd": 0}))
        us = a.groupby("neuron").agg(n=("neuron", "size"),
                                     mt=("maxtpm", "median"),
                                     md=("nd", "median"))
        self.sup = list(us[(us.n >= 4) & (us.mt > 3) & (us.md >= 4)].index)

        ut = {v: self.cb[[f"{v}_untreated24", f"{v}_untreated72"]] for v in VIRUS_ORDER}
        self.amp_up = pd.DataFrame({v: ut[v].max(axis=1) for v in VIRUS_ORDER})
        self.amp_dn = pd.DataFrame({v: ut[v].min(axis=1) for v in VIRUS_ORDER})
        self.n_up = (self.amp_up.loc[self.sup] > 1).sum(axis=1)
        self.n_dn = (self.amp_dn.loc[self.sup] < -1).sum(axis=1)
        self.n_shared = pd.Series(
            {n: (self.n_up[n] if self.n_up[n] >= self.n_dn[n] else self.n_dn[n])
             for n in self.sup})

        self.n_genes = us["n"]

        d = degs[(degs.treatment == "untreated") & degs.deg]
        self.gene_up = d[d.log2fc > 0].groupby("gene_id")["virus"].nunique()
        self.gene_dn = d[d.log2fc < 0].groupby("gene_id")["virus"].nunique()
        self.sym = degs.drop_duplicates("gene_id").set_index("gene_id")[
            ["symbol", "gene_type"]]
        self.isg_panel = load_isg_symbols()
        on = self.sym.symbol.isin(self.isg_panel)
        self.up_on = self.gene_up[on.reindex(self.gene_up.index).fillna(False)]
        self.up_lnc = self.gene_up[
            (self.sym.gene_type == "lncRNA").reindex(self.gene_up.index).fillna(False)]

        b = d[d.hpi == 72].groupby("virus").size()
        self.breadth = b.reindex(VIRUS_ORDER)

        self.identity = degs[degs.symbol.isin(IDENTITY_GENES)][
            ["symbol", "virus", "treatment", "hpi", "log2fc", "fdr"]]

        s24 = set(zip(d[d.hpi == 24].gene_id, d[d.hpi == 24].virus))
        s72 = set(zip(d[d.hpi == 72].gene_id, d[d.hpi == 72].virus))
        by_neuron = self.assign.groupby("neuron")["gene_id"].apply(list)
        fc = pd.read_parquet(DATA / f"{TAG}_matrix_masked_log2fc.parquet")
        cent = fc.join(self.assign.set_index("gene_id")["neuron"]).groupby("neuron").mean()
        ut_cols = [f"{v}_untreated{h}" for v in VIRUS_ORDER for h in (24, 72)]
        self.induced_units = [n for n in self.sup if cent.loc[n, ut_cols].mean() > 0]
        self.repressed_units = [n for n in self.sup if n not in self.induced_units]

        def share24(neurons):
            out = []
            for n in neurons:
                mem, tot, hit = by_neuron.get(n, []), 0, 0
                for v in VIRUS_ORDER:
                    m = [x for x in mem if (x, v) in s72]
                    tot += len(m)
                    hit += sum((x, v) in s24 for x in m)
                if tot:
                    out.append(hit / tot)
            return np.array(out)

        self.share24 = {"induced": share24(self.induced_units),
                        "repressed": share24(self.repressed_units)}

        neuron_of = self.assign.set_index("gene_id")["neuron"]
        gl = degs[degs.gene_id.isin(neuron_of.index)].copy()
        gl["neuron"] = gl.gene_id.map(neuron_of)
        gl["cls"] = gl.neuron.map({n: FUNCTIONAL_CLASS.get(int(n))
                                   for n in self.sup})
        self.gene_long = gl[gl.cls.notna()][
            ["gene_id", "symbol", "neuron", "cls", "virus", "treatment", "hpi",
             "log2fc", "deg"]]
        u = gl[gl.treatment == "untreated"]
        self.gene_ut = (u.pivot_table(index="gene_id", columns="hpi",
                                      values="log2fc", aggfunc="mean")
                        .rename(columns={24: "h24", 72: "h72"}))
        self.gene_ut["cls"] = self.gene_ut.index.map(
            lambda g: FUNCTIONAL_CLASS.get(int(neuron_of.get(g, -1)), None))
        self.gene_ut = self.gene_ut[self.gene_ut.cls.notna()]
        self.gene_maxfc = u.groupby("gene_id").log2fc.max()
        self.regions = spatial_regions(self)

    def cls(self, n):
        return FUNCTIONAL_CLASS.get(int(n))

    def members_of(self, key):
        return [n for n in self.sup if self.cls(n) == key]


def spatial_regions(ctx, threshold=5.2):
    from scipy.cluster.hierarchy import fcluster, linkage

    pts = np.array([ctx.grid.centres[int(n)] for n in ctx.sup])
    labels = fcluster(linkage(pts, method="single"), t=threshold,
                      criterion="distance")
    out = []
    for c in sorted(set(labels)):
        units = [int(n) for n, l in zip(ctx.sup, labels) if l == c]
        name = next((REGION_NAMES[s] for s in REGION_NAMES if s in units), None)
        if name is None or len(units) < 2:
            continue
        out.append({"name": name, "units": units,
                    "n_genes": int(ctx.n_genes.reindex(units).sum())})
    return out


def announce(ctx):
    print(f"  supported programmes            : {len(ctx.sup)}")
    print(f"  induced by >=1 virus            : {int((ctx.n_up >= 1).sum())}")
    print(f"  induced by all five             : {int((ctx.n_up == 5).sum())}")
    print(f"  induced by exactly one virus    : "
          f"{ {v: n for n in ctx.n_up.index[ctx.n_up == 1] for v in VIRUS_ORDER if ctx.amp_up.loc[n, v] > 1} }")
    print(f"  genes induced by >=4 viruses    : {int((ctx.gene_up >= 4).sum())}")
    print(f"  genes induced by all five       : {int((ctx.gene_up == 5).sum())}")
    print(f"  genes repressed by all five     : {int((ctx.gene_dn == 5).sum())}")
    missing = set(ctx.sup) - set(FUNCTIONAL_CLASS)
    if missing:
        print(f"  [warn] supported but unclassified: {sorted(missing)}")
    check_callout_genes(ctx)


def check_callout_genes(ctx):
    members = ctx.assign.groupby("neuron")["symbol"].apply(
        lambda s: set(s.dropna())).to_dict()
    where = {}
    for n, syms in members.items():
        for s in syms:
            where.setdefault(s, []).append(int(n))
    bad = []
    for n, (_, genes) in CALLOUTS.items():
        if genes == "technical":
            continue
        for token in (t.strip() for t in genes.split(",")):
            for gene in expand_gene_token(token):
                if gene not in members.get(n, set()):
                    bad.append((n, gene, where.get(gene, [])))
    if bad:
        lines = "\n".join(f"    label for neuron {n} names {g}, which is in "
                          f"{w or 'no neuron on this map'}" for n, g, w in bad)
        raise SystemExit("CALLOUTS name genes that are not in that neuron:\n"
                         + lines)
    print(f"  call-out gene check              : all "
          f"{sum(len(expand_gene_token(t)) for _, (_, g) in CALLOUTS.items() if g != 'technical' for t in g.split(','))}"
          f" named genes are in the neuron they label")


def expand_gene_token(token):
    import re

    token = token.strip()
    m = re.match(r"^(.*?)(\d)[–-](\d)$", token)
    if m:
        stem, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        return [f"{stem}{i}" for i in range(lo, hi + 1)]
    if "/" in token:
        head, *rest = token.split("/")
        if head[-1:].isdigit():
            stem = head.rstrip("0123456789")
        elif "-" in head:
            stem = head.rsplit("-", 1)[0] + "-"
        else:
            stem = ""
        return [head] + [stem + r for r in rest]
    return [token]


def hex_ring(x, y, r=1.0):
    ang = np.pi / 2 + np.arange(6) * np.pi / 3
    return [(x + r * np.cos(a), y + r * np.sin(a)) for a in ang]


def territory_path(neurons, grid, radius=1.0):
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union

    polys = [SPoly(hex_ring(*grid.centres[int(n)], r=radius)) for n in neurons
             if int(n) in grid.centres]
    if not polys:
        return None
    u = unary_union(polys)
    rings = []
    for geom in (u.geoms if u.geom_type == "MultiPolygon" else [u]):
        rings.append(np.asarray(geom.exterior.coords))
        rings += [np.asarray(i.coords) for i in geom.interiors]
    verts, codes = [], []
    for ring in rings:
        verts += list(ring)
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) \
            + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def draw_territory(ax, neurons, grid, ec, lw=0.8, ls="-", radius=1.0, z=6):
    path = territory_path(neurons, grid, radius)
    if path is not None:
        ax.add_patch(PathPatch(path, fc="none", ec=ec, lw=lw, ls=ls,
                               zorder=z, joinstyle="round"))


def hex_field(ax, grid, facecolors, edge=HEX_EDGE, lw=0.12, radius=0.98):
    for n, (x, y) in grid.centres.items():
        ax.add_patch(Polygon(hex_ring(x, y, radius), closed=True,
                             fc=facecolors.get(n, BLANK), ec=edge, lw=lw,
                             zorder=2))


def white_box(ax, x, y, w, h, z=7, alpha=0.90):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="none", alpha=alpha,
                           zorder=z))


def spread(ideal, lo, hi, gap):
    n = len(ideal)
    gap = min(gap, (hi - lo) / max(n - 1, 1))
    out = list(ideal)
    for i in range(1, n):
        out[i] = min(out[i], out[i - 1] - gap)
    for i in range(n - 1, -1, -1):
        out[i] = max(out[i], lo + (n - 1 - i) * gap)
    for i in range(1, n):
        out[i] = min(out[i], out[i - 1] - gap)
    if out[0] > hi:
        out = [v - (out[0] - hi) for v in out]
    return out


KEY_BAND = (-10.8, -31.5)


def leader_labels(ax, ctx, grid, side_pad, fontsize=4.7, gene_fontsize=4.0,
                  gap=2.7, avoid_left=None):
    x_l, x_r = grid.xlim[0] - side_pad + 0.6, grid.xlim[1] + side_pad - 0.6
    sides = {"L": [], "R": []}
    for n in CALLOUTS:
        x, y = grid.centres[n]
        sides["L" if x < grid.cx else "R"].append((n, y))

    lo_all, hi_all = grid.ylim[0] + 1.2, grid.ylim[1] - 0.6
    for side, items in sides.items():
        items.sort(key=lambda t: -t[1])
        if side == "L" and avoid_left:
            top, bottom = avoid_left
            mid = (top + bottom) / 2
            upper = [it for it in items if it[1] > mid]
            lower = [it for it in items if it[1] <= mid]
            ys = (spread([y for _, y in upper], top, hi_all, gap)
                  + spread([y for _, y in lower], lo_all, bottom, gap))
            items = upper + lower
        else:
            ys = spread([y for _, y in items], lo_all, hi_all, gap)
        tx = x_l if side == "L" else x_r
        ha = "left" if side == "L" else "right"
        for (n, _), ty in zip(items, ys):
            head, genes = CALLOUTS[n]
            colour = CLASSES[ctx.cls(n)][1]
            hx, hy = grid.centres[n]
            anchor_x = tx + (8.5 if side == "L" else -8.5)
            ax.annotate("", xy=(hx + (1.0 if side == "L" else -1.0), hy),
                        xytext=(anchor_x, ty),
                        arrowprops=dict(arrowstyle="-", lw=0.35, color=colour,
                                        shrinkA=0, shrinkB=1), zorder=8)
            tag = RESTRICTED.get(n)
            size = int(ctx.n_genes.get(n, 0))
            ax.text(tx, ty + 0.28, f"{n}  {head}  ({size} genes)", ha=ha,
                    va="bottom", fontsize=fontsize, color="#1F2328",
                    fontweight="bold", zorder=9)
            sub = genes if not tag else f"{genes}  — {tag}-restricted"
            ax.text(tx, ty - 0.28, sub, ha=ha, va="top",
                    fontsize=gene_fontsize,
                    color=colour if tag else "0.35",
                    fontstyle="italic", zorder=9)


def class_legend(ax, keys, x, y, dy=1.55, box=1.05, fontsize=4.9, title=None,
                 ncol=2, colw=23.0):
    rows = -(-len(keys) // ncol)
    if title:
        ax.text(x, y + dy, title, fontsize=4.9, fontweight="bold",
                color="#1F2328", ha="left", va="center", zorder=9)
    for i, key in enumerate(keys):
        label, colour = CLASSES[key]
        col, row = divmod(i, rows)
        xx, yy = x + col * colw, y - row * dy
        ax.add_patch(Rectangle((xx, yy - box / 2), box * 1.6, box, fc=colour,
                               ec="white", lw=0.3, zorder=9))
        ax.text(xx + box * 2.1, yy, label, fontsize=fontsize, va="center",
                ha="left", color="#1F2328", zorder=9)
    return rows


def panel_atlas(ax, ctx, side_pad=26.0, labels=True):
    grid = ctx.grid
    amp = ctx.mod["consensus_amp"].abs()
    faces = {}
    for n in ctx.sup:
        key = ctx.cls(n)
        if key is None:
            continue
        base = np.array(mpl.colors.to_rgb(CLASSES[key][1]))
        f = 0.40 + 0.60 * min(float(amp.get(n, 1.0)) / 4.0, 1.0)
        faces[n] = tuple(1 - f + f * base)
    hex_field(ax, grid, faces)

    for region in ctx.regions:
        draw_territory(ax, region["units"], grid, "#8A9099", lw=0.55,
                       ls=(0, (2.4, 1.8)), radius=1.75, z=3)
    for key in CLASSES:
        members = ctx.members_of(key)
        if members:
            draw_territory(ax, members, grid, CLASSES[key][1], lw=0.7,
                           radius=0.99)

    unsupported_rings(ax, grid)

    grid.frame(ax)
    ax.set_xlim(grid.xlim[0] - side_pad, grid.xlim[1] + side_pad)
    ax.set_ylim(grid.ylim[0] - BOTTOM_STRIP, grid.ylim[1] + TOP_STRIP)
    if labels:
        leader_labels(ax, ctx, grid, side_pad, avoid_left=KEY_BAND)
    region_labels(ax, ctx, grid)

    atlas_key(ax, ctx, grid, side_pad,
              depth_note="colour depth = programme amplitude")


def region_labels(ax, ctx, grid, top_strip=True):
    for region in ctx.regions:
        xs = [grid.centres[n][0] for n in region["units"]]
        ys = [grid.centres[n][1] for n in region["units"]]
        x, ha = float(np.mean(xs)), "center"
        if float(np.mean(ys)) < sum(grid.ylim) / 2:
            y, va = grid.ylim[0] - 0.7, "top"
        else:
            above = max(ys) + 2.6
            if above <= grid.ylim[1] - 0.5:
                y, va = above, "bottom"
            elif top_strip:
                y, va = grid.ylim[1] + 0.7, "bottom"
                x, ha = max(xs) + 1.0, "right"
            else:
                y, va = min(ys) - 2.6, "top"
                x, ha = min(max(xs) + 1.0, grid.xlim[1] - 0.8), "right"
        ax.text(x, y,
                f"{region['name']}\n{len(region['units'])} programmes · "
                f"{region['n_genes']:,} genes",
                fontsize=4.7, color="#4A5058", ha=ha, va=va,
                fontstyle="italic", zorder=9,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))


def atlas_key(ax, ctx, grid, side_pad, depth_note=""):
    x = grid.xlim[0] - side_pad + 0.6
    total = int(ctx.n_genes.reindex(ctx.sup).sum())
    ax.text(x, -11.5, f"{len(ctx.sup)} supported\nprogrammes · {total:,} genes",
            fontsize=5.4, fontweight="bold", color="#1F2328", ha="left",
            va="top", zorder=9)
    class_legend(ax, list(CLASSES), x, -15.0, ncol=1, dy=1.45, fontsize=4.4)
    note = [f"grey = below threshold "
            f"({len(grid.centres) - len(ctx.sup)} of {len(grid.centres)});",
            depth_note.replace("colour depth", "colour depth") if depth_note
            else "",
            "dashed hull = region; dashed hexagon =",
            "named in the Results, below threshold"]
    ax.text(x, -25.4, "\n".join(l for l in note if l), fontsize=4.3,
            color="0.40", va="top", ha="left", zorder=9)


def unsupported_rings(ax, grid):
    for i, n in enumerate(sorted(NAMED_UNSUPPORTED)):
        if n in grid.centres:
            x, y = grid.centres[n]
            ax.add_patch(Polygon(hex_ring(x, y, 1.02), closed=True, fc="none",
                                 ec="#3F4550", lw=0.7, ls=(0, (1.5, 1.2)),
                                 zorder=7))
            up = i % 2 == 0
            ax.text(x, y + (1.25 if up else -1.25), str(n), ha="center",
                    va="bottom" if up else "top", fontsize=4.0,
                    color="#3F4550", zorder=8)


def territory_note(ax, grid, text, colour, corner):
    right, top = corner[1] == "r", corner[0] == "t"
    x = grid.xlim[1] - 0.6 if right else grid.xlim[0] + 0.6
    y = grid.ylim[1] + 0.4 if top else grid.ylim[0] - 0.4
    ax.text(x, y, text, fontsize=4.8, ha="right" if right else "left",
            va="bottom" if top else "top", color=colour, fontstyle="italic",
            zorder=9, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))


def shared_terrain_scale(cb):
    z = min(float(np.nanmax(np.abs(cb.to_numpy()))), CLIP_Z)
    return TwoSlopeNorm(vmin=-z, vcenter=0, vmax=z), np.linspace(-z, z, 21)


def panel_terrain(fig, slot, ctx, outline=True, compact=False):
    grid, cb = ctx.grid, ctx.cb
    norm, levels = shared_terrain_scale(cb)
    colors = virus_colors()
    induced = [n for n in ctx.sup if ctx.mod.loc[n, "direction"] == "induced"]
    repressed = [n for n in ctx.sup if ctx.mod.loc[n, "direction"] == "repressed"]

    gs = slot.subgridspec(len(VIRUS_ORDER), len(CONDS), wspace=0.05,
                          hspace=0.10)
    for r, virus in enumerate(VIRUS_ORDER):
        for c, cond in enumerate(CONDS):
            ax = fig.add_subplot(gs[r, c])
            surface(ax, cb[f"{virus}_{cond}"], grid, "RdBu_r", norm, levels)
            if outline:
                draw_territory(ax, induced, grid, "#1F2328", lw=0.45, z=7)
                draw_territory(ax, repressed, grid, "#1F2328", lw=0.40,
                               ls=(0, (1.6, 1.2)), z=7)
            grid.frame(ax)
            if r == 0 and c == 0:
                terrain_region_tags(ax, grid)
            if r == 0:
                title = COND_TITLE[cond]
                ax.set_title(title.replace(", ", "\n") if compact else title,
                             fontsize=4.8 if compact else 5.6, pad=1.5)
            if c == 0:
                ax.text(grid.xlim[0] - 0.4, sum(grid.ylim) / 2, virus,
                        rotation=90, ha="right", va="center", fontsize=6.2,
                        fontweight="bold", color=colors[virus], clip_on=False)
    return norm, len(induced), len(repressed)


TERRAIN_TAGS = [("IFN / ISG", "right", "top"),
                ("stress", "left", "top"),
                ("metabolic", "left", "bottom")]


def terrain_region_tags(ax, grid):
    for text, ha, va in TERRAIN_TAGS:
        ax.text(grid.xlim[0] + 0.8 if ha == "left" else grid.xlim[1] - 0.8,
                grid.ylim[1] - 0.8 if va == "top" else grid.ylim[0] + 0.8,
                text, fontsize=3.6, ha=ha, va=va, color="#1F2328", zorder=9,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4))


def terrain_key(fig, rect, norm, n_induced=0, n_repressed=0):
    cax = fig.add_axes(rect)
    bar = mpl.colorbar.ColorbarBase(cax, cmap="RdBu_r", norm=norm,
                                    orientation="horizontal",
                                    ticks=[-norm.vmax, 0, norm.vmax],
                                    extend="both")
    bar.ax.set_xticklabels([f"≤ −{norm.vmax:.0f}", "0",
                            f"≥ +{norm.vmax:.0f}"])
    bar.ax.tick_params(labelsize=4.8, length=1.4, pad=1)
    bar.outline.set_linewidth(0.4)
    cax.set_title("codebook weight (z), shared scale, winsorized",
                  fontsize=4.8, pad=2)
    fig.text(rect[0] - 0.055, rect[1] - 0.014,
             f"outline: the {n_induced} induced (solid) and {n_repressed} "
             f"repressed (dashed) supported programmes of panel a\n"
             "geography, the same in all 20 panels: top-right = interferon / "
             "ISG core, top-left = inflammatory\nand stress, bottom-left = "
             "hepatocyte metabolic (named on the first panel)",
             fontsize=4.4, color="0.35", va="top", ha="left")


MAP_STRIP = 15.0


def panel_kinetic_map(ax, ctx):
    grid = ctx.grid
    faces = {}
    for n, row in ctx.kin.iterrows():
        key = (row["direction"], row["kinetic_class"])
        if key in KIN_COLORS:
            faces[int(n)] = KIN_COLORS[key]
    hex_field(ax, grid, faces, lw=0.10)
    grid.frame(ax)
    ax.set_ylim(grid.ylim[0] - MAP_STRIP, grid.ylim[1] + 1.6)

    counts = ctx.kin.groupby(["direction", "kinetic_class"]).size()
    order = [("induced", "early"), ("induced", "sustained"),
             ("induced", "late"), ("repressed", "early"),
             ("repressed", "sustained"), ("repressed", "late")]
    names = {"early": "24 hpi only", "sustained": "both", "late": "72 hpi only"}
    x0, y0 = grid.xlim[0] + 0.4, grid.ylim[0] - 7.4
    ax.text(x0, y0 + 3.6, "neurons by direction and kinetics", fontsize=4.9,
            fontweight="bold", ha="left", va="center")
    for i, key in enumerate(order):
        col, row_i = divmod(i, 3)
        x = x0 + col * 25.5
        y = y0 - row_i * 2.8
        ax.add_patch(Rectangle((x, y - 0.75), 1.7, 1.5, fc=KIN_COLORS[key],
                               ec="white", lw=0.3))
        ax.text(x + 2.4, y, f"{key[0]}, {names[key[1]]}  ({counts.get(key, 0)})",
                fontsize=4.5, va="center", ha="left")
    territory_note(ax, grid, "① interferon production, then ② effectors",
                   "#7A2A20", corner="tr")
    territory_note(ax, grid, "③ metabolic repression, late", "#1F4E79",
                   corner="bl")


def panel_succession(ax, ctx):
    rng = np.random.default_rng(143)
    for i, (key, colour) in enumerate([("induced", "#C0392B"),
                                       ("repressed", "#2E6DA4")]):
        vals = 100 * ctx.share24[key]
        ax.scatter(vals, np.full(len(vals), i) + rng.uniform(-0.16, 0.16, len(vals)),
                   s=6, fc=colour, ec="white", lw=0.25, alpha=0.85, zorder=3)
        med = float(np.median(vals))
        ax.plot([med, med], [i - 0.34, i + 0.34], color="#1F2328", lw=1.2,
                zorder=4)
        ax.text(med, i + 0.44, f"median {med:.0f}%", fontsize=4.4,
                ha="center", va="bottom", color="#1F2328")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"induced\n({len(ctx.induced_units)})",
                        f"repressed\n({len(ctx.repressed_units)})"],
                       fontsize=4.6)
    ax.set_ylim(-0.65, 1.75)
    ax.set_xlim(-4, 104)
    ax.set_xlabel("% of a programme's 72-hpi DEGs already DE at 24 hpi",
                  fontsize=4.8, labelpad=1.5)
    ax.tick_params(labelsize=4.6, length=1.3, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)


PHASE_ROWS = [("ifn", "①"), ("sense", "②"), ("isgf3", "②"),
              ("antigen", "②"), ("inflam", "②"), ("metab", "③")]
PHASE_NAMES = {"①": "interferon\nproduction",
               "②": "antiviral &\nimmune effector",
               "③": "metabolic\nrepression"}


def condition_heatmap(ax, ctx, M, row_labels, row_colors, value_fmt="{:+.1f}"):
    cols = [(h, v) for h in ("24", "72") for v in VIRUS_ORDER]
    norm, _ = shared_terrain_scale(ctx.cb)
    ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([v for _, v in cols], fontsize=4.8, rotation=90)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=4.9)
    for i, colour in enumerate(row_colors):
        ax.get_yticklabels()[i].set_color(colour)
    ax.tick_params(length=1.2, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    ax.axvline(4.5, color="white", lw=1.6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, value_fmt.format(M[i, j]), ha="center", va="center",
                    fontsize=3.9,
                    color="white" if abs(M[i, j]) > norm.vmax * 0.55
                    else "#23262B")
    ax.text(2.0, -0.95, "24 hpi", ha="center", va="bottom", fontsize=5.4,
            fontweight="bold")
    ax.text(7.0, -0.95, "72 hpi", ha="center", va="bottom", fontsize=5.4,
            fontweight="bold")
    ax.set_xlim(-0.5, len(cols) - 0.5)
    return norm


def panel_phase_heatmap(fig, slot, ctx):
    cols = [(h, v) for h in ("24", "72") for v in VIRUS_ORDER]
    M = np.array([[ctx.cb.loc[ctx.members_of(key), f"{v}_untreated{h}"].mean()
                   for h, v in cols] for key, _ in PHASE_ROWS])
    ax = fig.add_subplot(slot)
    condition_heatmap(ax, ctx, M, [CLASSES[k][0] for k, _ in PHASE_ROWS],
                      [CLASSES[k][1] for k, _ in PHASE_ROWS])
    ax.annotate("", xy=(9.4, -1.55), xytext=(-0.4, -1.55),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.35"))
    seen = {}
    for i, (_, phase) in enumerate(PHASE_ROWS):
        seen.setdefault(phase, []).append(i)
    for phase, idx in seen.items():
        lo, hi = min(idx) - 0.42, max(idx) + 0.42
        ax.plot([9.7, 9.9, 9.9, 9.7], [lo, lo, hi, hi], lw=0.6, color="0.35",
                clip_on=False)
        ax.text(10.2, (lo + hi) / 2, f"{phase} {PHASE_NAMES[phase]}",
                fontsize=4.9, va="center", ha="left", clip_on=False)
    return ax


TREATMENT_CONDS = [("untreated24", "UT\n24 hpi"), ("untreated72", "UT\n72 hpi"),
                   ("Ruxo24", "Ruxo\n24 hpi"), ("Ruxo72", "Ruxo\n72 hpi")]


def panel_treatment_grid(fig, slot, ctx):
    M = np.array([[ctx.cb.loc[ctx.members_of(key),
                              [f"{v}_{cond}" for v in VIRUS_ORDER]].to_numpy().mean()
                   for cond, _ in TREATMENT_CONDS] for key, _ in PHASE_ROWS])
    norm, _ = shared_terrain_scale(ctx.cb)

    ax = fig.add_subplot(slot)
    ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(TREATMENT_CONDS)))
    ax.set_xticklabels([lab for _, lab in TREATMENT_CONDS], fontsize=4.6)
    ax.set_yticks(range(len(PHASE_ROWS)))
    ax.set_yticklabels([f"{ph} {CLASSES[k][0]}" for k, ph in PHASE_ROWS],
                       fontsize=4.7)
    for i, (k, _) in enumerate(PHASE_ROWS):
        ax.get_yticklabels()[i].set_color(CLASSES[k][1])
    ax.tick_params(length=1.2, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.axvline(1.5, color="white", lw=1.8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center",
                    fontsize=4.2,
                    color="white" if abs(M[i, j]) > norm.vmax * 0.55
                    else "#23262B")
    ax.set_title("mean of the five viruses; JAK–STAT blocked in the two right "
                 "columns.\nRinged: the ligand tier is untouched by the drug, "
                 "the ISGF3 tier is abolished.", fontsize=4.4, pad=3)
    for i, key in enumerate(k for k, _ in PHASE_ROWS):
        if key in ("ifn", "isgf3"):
            ax.add_patch(Rectangle((-0.5, i - 0.5), M.shape[1], 1.0, fc="none",
                                   ec="#1F2328", lw=0.9, zorder=5))
    return ax


def panel_conservation(ax, ctx):
    grid = ctx.grid
    faces = {int(n): CONSERVATION_RAMP[int(k) - 1]
             for n, k in ctx.n_shared.items() if k >= 1}
    hex_field(ax, grid, faces, lw=0.12)
    for k in range(1, 6):
        members = [int(n) for n, v in ctx.n_shared.items() if v == k]
        if members:
            draw_territory(ax, members, grid, "#5A5F66", lw=0.35, radius=0.99)
    grid.frame(ax)
    ax.set_ylim(grid.ylim[0] - MAP_STRIP, grid.ylim[1] + 1.6)

    n_ind = int((ctx.n_up >= 1).sum())
    n_all = int((ctx.n_up == 5).sum())
    x0, base = grid.xlim[0] + 0.4, grid.ylim[0]
    ax.text(x0, base - 0.8,
            f"{n_all} of {n_ind} induced programmes are shared by all five\n"
            f"viruses; only two are restricted to a single virus",
            fontsize=4.9, ha="left", va="top")
    ax.text(x0, base - 5.2, "colour = viruses moving the programme in its own "
                            "direction", fontsize=4.4, ha="left", va="top")
    for i, k in enumerate(range(1, 6)):
        x = x0 + i * 4.0
        ax.add_patch(Rectangle((x, base - 9.0), 2.8, 1.5,
                               fc=CONSERVATION_RAMP[k - 1], ec="white", lw=0.3))
        ax.text(x + 1.4, base - 9.4, str(k), fontsize=4.6, ha="center",
                va="top")

    for n, virus in RESTRICTED.items():
        x, y = grid.centres[n]
        ax.add_patch(Polygon(hex_ring(x, y, 1.15), closed=True, fc="none",
                             ec="#B4231C", lw=0.9, zorder=8))
        name = ctx.mod.loc[n, "module"]
        top = n == 648
        ax.annotate(f"{n}  {virus}-restricted\n{name}",
                    xy=(x, y), xytext=(x + 7.0, y + (4.5 if top else 4.0)),
                    fontsize=4.6, color="#B4231C", ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#B4231C",
                                    shrinkA=0, shrinkB=2), zorder=10,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))


def panel_gene_conservation(ax, ctx):
    ks = range(1, 6)
    on = [int((ctx.up_on == k).sum()) for k in ks]
    total = [int((ctx.gene_up == k).sum()) for k in ks]
    off = [t - o for t, o in zip(total, on)]
    dn = [int((ctx.gene_dn == k).sum()) for k in ks]

    ax.bar(ks, on, color="#472A78", ec="white", lw=0.3, width=0.78,
           label="on the 382-ISG panel")
    ax.bar(ks, off, bottom=on, color="#B9A5D1", ec="white", lw=0.3, width=0.78,
           label="off the panel")
    ax.bar(ks, [-x for x in dn], color="#2E6DA4", ec="white", lw=0.3,
           width=0.78, label="repressed")
    for k, t in zip(ks, total):
        ax.text(k, t + 35, f"{t:,}", ha="center", va="bottom", fontsize=4.2)
    for k, v in zip(ks, dn):
        ax.text(k, -v - 35, f"{v:,}", ha="center", va="top", fontsize=4.2,
                color="#2E6DA4")
    ax.axhline(0, color="0.3", lw=0.5)
    ax.set_xticks(list(ks))
    ax.tick_params(labelsize=4.6, length=1.3, pad=1)
    ax.set_xlabel("viruses moving the gene", fontsize=4.8, labelpad=1.5)
    ax.set_ylabel("genes   ↑ induced  ↓ repressed", fontsize=4.8, labelpad=1.5)
    n4 = int((ctx.gene_up >= 4).sum())
    off4 = n4 - int((ctx.up_on >= 4).sum())
    lnc4 = int((ctx.up_lnc >= 4).sum())
    ax.set_title(f"{n4:,} genes induced by ≥ 4 viruses — {off4} off the ISG "
                 f"panel, {lnc4} lncRNA;\n{int((ctx.gene_dn == 5).sum())} genes "
                 f"repressed by all five", fontsize=4.8, pad=3)
    ax.legend(fontsize=4.3, frameon=False, loc="upper right", handlelength=0.9,
              handletextpad=0.4, labelspacing=0.25)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.20)


RESTRICTED_ROWS = [
    (756, "756  interferon ligands (all five)", "ifn"),
    (648, "648  stress response (WNV only)", "inflam"),
    (32, "32  conjugation (DENV only)", "metab"),
]


def panel_restricted(fig, slot, ctx):
    M = np.array([[ctx.cb.loc[n, f"{v}_untreated{h}"]
                   for h in ("24", "72") for v in VIRUS_ORDER]
                  for n, _, _ in RESTRICTED_ROWS])
    ax = fig.add_subplot(slot)
    condition_heatmap(ax, ctx, M, [t for _, t, _ in RESTRICTED_ROWS],
                      [CLASSES[k][1] for _, _, k in RESTRICTED_ROWS])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 1:
                ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94,
                                       fc="none", ec="#1F2328", lw=0.7,
                                       zorder=5))
    ax.text(9.9, 1.0, "ring: z > +1,\nthe threshold the\nrestriction is\ncalled on",
            fontsize=4.4, color="0.35", ha="left", va="center", clip_on=False)
    return ax


def panel_codes_map(ax, ctx, side_pad=26.0, pad_scale=2.6, top_strip=True):
    geom = dict(panel_geometry(ctx.grid))
    base_range = geom["x_range"]
    geom["side_pad"] = side_pad
    geom["x_range"] = ctx.grid.xlim[1] - ctx.grid.xlim[0] + 2 * side_pad
    geom["m"] *= geom["x_range"] / base_range
    geom["pad_top"] *= pad_scale
    draw_codes_map(ax, ctx.cb, ctx.assign, ctx.stats, ctx.grid, geom)
    for key in CLASSES:
        members = ctx.members_of(key)
        if members:
            draw_territory(ax, members, ctx.grid, CLASSES[key][1], lw=0.75,
                           radius=1.02, z=8)
    for region in ctx.regions:
        draw_territory(ax, region["units"], ctx.grid, "#8A9099", lw=0.55,
                       ls=(0, (2.4, 1.8)), radius=1.75, z=7)
    unsupported_rings(ax, ctx.grid)
    ax.set_ylim(ctx.grid.ylim[0] - BOTTOM_STRIP,
                ctx.grid.ylim[1] + geom["pad_top"])
    leader_labels(ax, ctx, ctx.grid, side_pad, avoid_left=KEY_BAND)
    region_labels(ax, ctx, ctx.grid, top_strip=top_strip)
    atlas_key(ax, ctx, ctx.grid, side_pad)


def label(fig, x, y, text):
    fig.text(x, y, text, fontsize=8, fontweight="bold", ha="left", va="top")


def compose_v1(ctx):
    fig_w, fig_h = 183.0, 208.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax = fig.add_axes([0.02, 0.545, 0.96, 0.425])
    panel_codes_map(ax, ctx)
    label(fig, 0.008, 0.988, "a")

    gs = fig.add_gridspec(1, 1, left=0.055, right=0.985, top=0.505,
                          bottom=0.085)
    norm, n_i, n_r = panel_terrain(fig, gs[0], ctx)
    label(fig, 0.008, 0.525, "b")
    terrain_key(fig, [0.20, 0.032, 0.30, 0.011], norm, n_i, n_r)

    out = FIGDIR / "fig6_v1_annotated.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v1 annotated : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v2(ctx):
    fig_w, fig_h = 183.0, 280.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.020, 0.700, 0.960, 0.295])
    panel_atlas(ax_a, ctx)
    label(fig, 0.008, 0.996, "a")

    gs_b = fig.add_gridspec(1, 1, left=0.060, right=0.470, top=0.676,
                            bottom=0.492)
    norm, n_i, n_r = panel_terrain(fig, gs_b[0], ctx, compact=True)
    label(fig, 0.008, 0.690, "b")
    terrain_key(fig, [0.100, 0.470, 0.180, 0.006], norm, n_i, n_r)

    ax_c = fig.add_axes([0.540, 0.492, 0.450, 0.184])
    panel_kinetic_map(ax_c, ctx)
    label(fig, 0.512, 0.690, "c")

    gs_d = fig.add_gridspec(1, 1, left=0.660, right=0.885, top=0.448,
                            bottom=0.376)
    panel_phase_heatmap(fig, gs_d[0], ctx)
    label(fig, 0.512, 0.468, "d")

    gs_e = fig.add_gridspec(1, 1, left=0.700, right=0.850, top=0.320,
                            bottom=0.240)
    panel_treatment_grid(fig, gs_e[0], ctx)
    label(fig, 0.512, 0.352, "e")

    ax_f = fig.add_axes([0.585, 0.196, 0.380, 0.036])
    panel_succession(ax_f, ctx)
    label(fig, 0.512, 0.234, "f")

    ax_g = fig.add_axes([0.020, 0.030, 0.450, 0.184])
    panel_conservation(ax_g, ctx)
    label(fig, 0.008, 0.222, "g")

    ax_h = fig.add_axes([0.600, 0.096, 0.360, 0.046])
    panel_gene_conservation(ax_h, ctx)
    label(fig, 0.512, 0.172, "h")

    gs_i = fig.add_gridspec(1, 1, left=0.660, right=0.880, top=0.056,
                            bottom=0.026)
    panel_restricted(fig, gs_i[0], ctx)
    label(fig, 0.512, 0.074, "i")

    out = FIGDIR / "fig6_v2_atlas.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v2 atlas     : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v3(ctx):
    fig_w, fig_h = 183.0, 232.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.020, 0.630, 0.960, 0.358])
    panel_atlas(ax_a, ctx)
    label(fig, 0.008, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.340, 0.440, 0.250])
    panel_kinetic_map(ax_b, ctx)
    label(fig, 0.008, 0.606, "b")

    gs_c = fig.add_gridspec(1, 1, left=0.660, right=0.880, top=0.582,
                            bottom=0.470)
    panel_phase_heatmap(fig, gs_c[0], ctx)
    label(fig, 0.512, 0.606, "c")

    gs_d = fig.add_gridspec(1, 1, left=0.700, right=0.850, top=0.408,
                            bottom=0.310)
    panel_treatment_grid(fig, gs_d[0], ctx)
    label(fig, 0.512, 0.440, "d")

    ax_e = fig.add_axes([0.585, 0.226, 0.380, 0.042])
    panel_succession(ax_e, ctx)
    label(fig, 0.512, 0.290, "e")

    ax_f = fig.add_axes([0.020, 0.038, 0.440, 0.250])
    panel_conservation(ax_f, ctx)
    label(fig, 0.008, 0.304, "f")

    ax_g = fig.add_axes([0.600, 0.114, 0.360, 0.062])
    panel_gene_conservation(ax_g, ctx)
    label(fig, 0.512, 0.200, "g")

    gs_h = fig.add_gridspec(1, 1, left=0.660, right=0.880, top=0.070,
                            bottom=0.040)
    panel_restricted(fig, gs_h[0], ctx)
    label(fig, 0.512, 0.092, "h")

    out = FIGDIR / "fig6_v3_narrative.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v3 narrative : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


MARKERS = [
    ("ifn", ["IFNL1", "IFNB1"]),
    ("sense", ["IFIT1", "RSAD2", "OAS1", "RIGI"]),
    ("isgf3", ["ISG15", "MX1", "IFI6", "STAT1"]),
    ("antigen", ["B2M", "HLA-E"]),
    ("inflam", ["ATF3", "SOCS3", "IL33", "SAA2"]),
    ("metab", ["HMGCS2", "CYP3A4", "ADH1B", "CYP2E1"]),
]


def region_of(ctx, neuron):
    for r in ctx.regions:
        if int(neuron) in r["units"]:
            return r["name"]
    return None


def panel_glyph_atlas(ax, ctx, side_pad=30.0):
    grid, cb = ctx.grid, ctx.cb
    conds = list(cb.columns)
    colors = virus_colors()
    k, lo, hi = fan_scale(cb)

    for n in grid.centres:
        supported = n in ctx.sup
        key = ctx.cls(n) if supported else None
        fc = (lighten(CLASSES[key][1], 0.68) if key else "#F4F5F6")
        x, y = grid.hexagon(ax, n, radius=0.98, fc=fc, ec="#DDE0E3", lw=0.12,
                            zorder=2)
        if supported:
            fan_glyph(ax, x, y, cb.loc[n].to_numpy(), k=k, conds=conds,
                      colors=colors, lw=0.10, ring=False)
    for r in ctx.regions:
        draw_territory(ax, r["units"], grid, "#8A9099", lw=0.6,
                       ls=(0, (2.4, 1.8)), radius=1.7, z=3)
    unsupported_rings(ax, grid)
    for n in CALLOUTS:
        x, y = grid.centres[n]
        ax.text(x, y - 1.35, str(n), ha="center", va="top", fontsize=3.8,
                fontweight="bold", color="#1F2328", zorder=9,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))

    grid.frame(ax)
    ax.set_xlim(grid.xlim[0] - side_pad, grid.xlim[1] + side_pad)
    ax.set_ylim(grid.ylim[0] - 1.0, grid.ylim[1] + 12.0)
    fan_key(ax, k, lo, hi, conds, colors,
            origin=(grid.cx, grid.ylim[1] + 6.4), magnify=3.4)
    atlas_margin_lists(ax, ctx, grid, side_pad)


def _block(ax, x, y, heading, entries, colour, dy=1.15, ha="left"):
    ax.text(x, y, heading[0], fontsize=4.6, fontweight="bold", color="#3A4048",
            ha=ha, va="top")
    ax.text(x, y - dy, heading[1], fontsize=4.2, color="0.45", ha=ha, va="top",
            fontstyle="italic")
    y -= dy * 2.3
    for n, head, genes in entries:
        ax.text(x, y, f"{n}  {head}", fontsize=4.3, fontweight="bold",
                color="#1F2328", ha=ha, va="top")
        ax.text(x, y - dy * 0.95, genes, fontsize=3.9, color=colour(n),
                fontstyle="italic", ha=ha, va="top")
        y -= dy * 2.1
    return y


def atlas_margin_lists(ax, ctx, grid, side_pad):
    def colour(n):
        return CLASSES[ctx.cls(n)][1] if ctx.cls(n) else "0.4"

    by_region = {}
    for n, (head, genes) in CALLOUTS.items():
        by_region.setdefault(region_of(ctx, n) or "other", []).append(
            (n, head, genes))
    for v in by_region.values():
        v.sort(key=lambda t: -grid.centres[t[0]][1])

    x_l = grid.xlim[0] - side_pad + 0.6
    x_r = grid.xlim[1] + side_pad - 0.6
    core = next(r["name"] for r in ctx.regions if 756 in r["units"])
    inflam = next(r["name"] for r in ctx.regions if 648 in r["units"])
    metab = next(r["name"] for r in ctx.regions if 59 in r["units"])
    tot = {r["name"]: (len(r["units"]), r["n_genes"]) for r in ctx.regions}

    y = grid.ylim[1] - 0.5
    ax.text(x_l, y, f"{len(ctx.sup)} supported programmes · "
                    f"{int(ctx.n_genes.reindex(ctx.sup).sum()):,} genes",
            fontsize=5.2, fontweight="bold", color="#1F2328", ha="left",
            va="top")
    y -= 2.6
    for key in CLASSES:
        ax.add_patch(Rectangle((x_l, y - 0.85), 1.5, 0.95,
                               fc=CLASSES[key][1], ec="white", lw=0.3))
        ax.text(x_l + 2.1, y - 0.38, CLASSES[key][0], fontsize=4.2,
                va="center", ha="left", color="#1F2328")
        y -= 1.25
    y -= 1.4
    y = _block(ax, x_l, y, (inflam, f"{tot[inflam][0]} programmes · "
                                    f"{tot[inflam][1]:,} genes"),
               by_region.get(inflam, []), colour)
    y -= 0.8
    _block(ax, x_l, y, (metab, f"{tot[metab][0]} programmes · "
                               f"{tot[metab][1]:,} genes"),
           by_region.get(metab, []), colour)

    y = _block(ax, x_r, grid.ylim[1] - 0.5,
               (core, f"{tot[core][0]} programmes · {tot[core][1]:,} genes"),
               by_region.get(core, []), colour, ha="right")
    for n, head, genes in by_region.get("other", []):
        ax.text(x_r, y, f"{n}  {head}", fontsize=4.3, fontweight="bold",
                color="#1F2328", ha="right", va="top")
        ax.text(x_r, y - 1.1, genes, fontsize=3.9, color="0.45",
                fontstyle="italic", ha="right", va="top")
        y -= 2.4
    ax.text(x_r, y - 0.6,
            "blank hexagon = below the support threshold\n"
            "dashed hexagon = named in the Results, below threshold\n"
            "dashed grey hull = spatial region",
            fontsize=4.0, color="0.45", ha="right", va="top")


RAW_CONDS = [("untreated", 24, "UT 24"), ("untreated", 72, "UT 72"),
             ("Ruxo", 24, "Rx 24"), ("Ruxo", 72, "Rx 72")]

SHORT_CLASS = {"ifn": "IFN ligands", "sense": "Sensing / IRF3",
               "isgf3": "ISGF3 effectors", "antigen": "Antigen pres.",
               "inflam": "Inflammatory", "metab": "Metabolic"}


def panel_class_distributions(ax, ctx):
    colors = virus_colors()
    gl = ctx.gene_long
    ypos, ylabels, ycolors = [], [], []
    data, medians = [], []
    for ci, (key, _) in enumerate(PHASE_ROWS):
        for j, (tr, hpi, lab) in enumerate(RAW_CONDS):
            sel = gl[(gl.cls == key) & (gl.treatment == tr) & (gl.hpi == hpi)]
            y = -(ci * 5.0 + j)
            ypos.append(y)
            ylabels.append(lab)
            ycolors.append(CLASSES[key][1])
            data.append(sel.log2fc.to_numpy())
            medians.append([(v, float(sel[sel.virus == v].log2fc.median()))
                            for v in VIRUS_ORDER])
    bp = ax.boxplot(data, positions=ypos, widths=0.72,
                    orientation="horizontal", showfliers=False,
                    patch_artist=True, manage_ticks=False)
    for patch, colour in zip(bp["boxes"], ycolors):
        patch.set_facecolor(lighten(colour, 0.55))
        patch.set_edgecolor(colour)
        patch.set_linewidth(0.5)
    for part in ("whiskers", "caps"):
        for ln in bp[part]:
            ln.set_linewidth(0.5)
            ln.set_color("0.45")
    for ln in bp["medians"]:
        ln.set_linewidth(0.9)
        ln.set_color("#1F2328")
    for y, meds in zip(ypos, medians):
        for v, med in meds:
            ax.scatter([med], [y], s=3.4, fc=colors[v], ec="white", lw=0.15,
                       zorder=5)

    ax.axvline(0, color="0.4", lw=0.5, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=4.0)
    for lab, colour in zip(ax.get_yticklabels(), ycolors):
        lab.set_color(colour)
    outside = blended_transform_factory(ax.transAxes, ax.transData)
    for ci, (key, ph) in enumerate(PHASE_ROWS):
        ax.text(-0.38, -(ci * 5.0 + 1.5), f"{ph} {SHORT_CLASS[key]}",
                transform=outside, fontsize=4.4, color=CLASSES[key][1],
                ha="left", va="center", fontweight="bold")
    ax.set_xlabel("log₂FC vs Mock, one box per condition over all member "
                  "genes × 5 viruses", fontsize=4.6, labelpad=1.5)
    ax.tick_params(labelsize=4.2, length=1.2, pad=1)
    ax.set_ylim(min(ypos) - 1.2, 2.6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, fc=colors[v], ec="none")
               for v in VIRUS_ORDER]
    ax.legend(handles, VIRUS_ORDER, fontsize=4.0, frameon=False, ncol=5,
              handlelength=0.7, handletextpad=0.3, columnspacing=0.7,
              loc="lower right", title="per-virus median",
              title_fontsize=4.0, borderaxespad=0.2)


def panel_trajectory(ax, ctx):
    g = ctx.gene_ut
    for key, _ in PHASE_ROWS:
        sel = g[g.cls == key]
        ax.scatter(sel.h24, sel.h72, s=2.6, fc=CLASSES[key][1], ec="none",
                   alpha=0.75, zorder=3, label=CLASSES[key][0])
    lim = (float(np.floor(min(g.h24.min(), g.h72.min()))),
           float(np.ceil(max(g.h24.max(), g.h72.max()))))
    ax.plot(lim, lim, color="0.45", lw=0.5, ls=(0, (2.4, 1.8)), zorder=2)
    ax.axhline(0, color="0.6", lw=0.4, zorder=1)
    ax.axvline(0, color="0.6", lw=0.4, zorder=1)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("log₂FC, 24 hpi", fontsize=4.6, labelpad=1.2)
    ax.set_ylabel("log₂FC, 72 hpi", fontsize=4.6, labelpad=1.2)
    ax.tick_params(labelsize=4.2, length=1.2, pad=1)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("1,035 programme genes, untreated,\n5-virus mean",
                 fontsize=4.5, pad=2)


def panel_conservation_effect(ax, ctx):
    on_panel = ctx.sym.symbol.isin(ctx.isg_panel)
    rng = np.random.default_rng(143)
    for k in range(1, 6):
        ids = ctx.gene_up.index[ctx.gene_up == k]
        fc = ctx.gene_maxfc.reindex(ids)
        on = on_panel.reindex(ids).fillna(False).to_numpy()
        for mask, dx, colour, lab in ((on, -0.18, "#472A78", "on the 382-ISG panel"),
                                      (~on, 0.18, "#B9A5D1", "off the panel")):
            vals = fc.to_numpy()[mask]
            if not len(vals):
                continue
            ax.scatter(k + dx + rng.uniform(-0.10, 0.10, len(vals)), vals,
                       s=1.6, fc=colour, ec="none", alpha=0.55, zorder=3,
                       label=lab if k == 1 else None)
            ax.plot([k + dx - 0.16, k + dx + 0.16],
                    [np.median(vals)] * 2, color="#1F2328", lw=0.9, zorder=5)
        ax.text(k, 1.02, f"{len(ids):,}", transform=ax.get_xaxis_transform(),
                fontsize=4.0, ha="center", va="bottom", clip_on=False)
    ax.axhline(1, color="0.55", lw=0.5, ls=(0, (2.2, 1.6)), zorder=1)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("viruses inducing the gene", fontsize=4.6, labelpad=1.2)
    ax.set_ylabel("largest untreated log₂FC", fontsize=4.6, labelpad=1.2)
    ax.tick_params(labelsize=4.2, length=1.2, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=4.0, frameon=False, loc="upper left", handlelength=0.7,
              handletextpad=0.3, labelspacing=0.25, markerscale=2.6,
              borderaxespad=0.1)


def panel_restricted_genes(ax, ctx):
    colors = virus_colors()
    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    gl = ctx.gene_long[ctx.gene_long.treatment == "untreated"]
    rng = np.random.default_rng(7)
    units = [(648, "648  WNV-restricted"), (32, "32  DENV-restricted")]
    for ui, (n, title) in enumerate(units):
        genes = list(neuron_of[neuron_of == n].index)
        for vi, v in enumerate(VIRUS_ORDER):
            x = ui * 6.0 + vi
            sel = gl[(gl.neuron == n) & (gl.virus == v)]
            for is_deg, kw in ((True, dict(fc=colors[v], ec="#1F2328", lw=0.2)),
                               (False, dict(fc="none", ec=colors[v], lw=0.4))):
                d = sel[sel.deg == is_deg]
                if len(d):
                    ax.scatter(x + rng.uniform(-0.22, 0.22, len(d)), d.log2fc,
                               s=4.5, zorder=3, **kw)
            ax.plot([x - 0.30, x + 0.30], [sel.log2fc.median()] * 2,
                    color="#1F2328", lw=0.9, zorder=5)
        ax.text(ui * 6.0 + 2, 1.04, f"{title}  ({len(genes)} genes)",
                transform=ax.get_xaxis_transform(), fontsize=4.3, ha="center",
                va="bottom", clip_on=False, color="#B4231C")
    ax.axhline(0, color="0.4", lw=0.5, zorder=1)
    for y in (1, -1):
        ax.axhline(y, color="0.6", lw=0.4, ls=(0, (2.2, 1.6)), zorder=1)
    ax.set_xticks([ui * 6.0 + vi for ui in (0, 1) for vi in range(5)])
    ax.set_xticklabels(VIRUS_ORDER * 2, fontsize=4.2, rotation=90)
    ax.set_ylabel("untreated log₂FC\n(both time points)", fontsize=4.6,
                  labelpad=1.2)
    ax.tick_params(labelsize=4.2, length=1.2, pad=1)
    ax.set_xlim(-0.8, 10.8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(10.6, 0.02, "filled = called DE",
            transform=ax.get_xaxis_transform(), fontsize=4.0, color="0.4",
            ha="right", va="bottom")


def panel_response_profile(ax, ctx):
    gl = ctx.gene_long
    ax.axvspan(1.5, 3.4, color="#EFF1F3", zorder=0, lw=0)
    ax.text(2.45, ax.get_ylim()[1], "JAK–STAT blocked", fontsize=4.3,
            color="0.45", ha="center", va="bottom")
    for key, ph in PHASE_ROWS:
        m, lo, hi = [], [], []
        for tr, hpi, _ in RAW_CONDS:
            v = gl[(gl.cls == key) & (gl.treatment == tr)
                   & (gl.hpi == hpi)].log2fc
            mu, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
            m.append(mu)
            lo.append(mu - 1.96 * se)
            hi.append(mu + 1.96 * se)
        c = CLASSES[key][1]
        ax.fill_between(range(4), lo, hi, color=c, alpha=0.20, lw=0, zorder=2)
        ax.plot(range(4), m, "-o", color=c, lw=1.0, ms=2.4, mec="white",
                mew=0.3, zorder=3)
        ax.text(3.12, m[-1], f"{ph} {SHORT_CLASS[key]}", fontsize=4.3,
                color=c, ha="left", va="center", fontweight="bold")
    ax.axhline(0, color="0.4", lw=0.5, zorder=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels([lab for _, _, lab in RAW_CONDS], fontsize=4.5)
    ax.set_xlim(-0.35, 3.1)
    ax.set_ylabel("mean log₂FC vs Mock\n(all member genes × 5 viruses, 95% CI)",
                  fontsize=4.6)
    ax.tick_params(labelsize=4.4, length=1.3, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)


def panel_marker_heatmap(fig, slot, ctx):
    sym = ctx.assign.set_index("symbol")["neuron"]
    dg = ctx.gene_long
    rows, rowcolors, rowlabels = [], [], []
    for key, genes in MARKERS:
        for g in genes:
            gid = ctx.assign.set_index("symbol").loc[g, "gene_id"]
            r = []
            for hpi in (24, 72):
                for v in VIRUS_ORDER:
                    m = dg[(dg.gene_id == gid) & (dg.virus == v)
                           & (dg.treatment == "untreated") & (dg.hpi == hpi)]
                    r.append(float(m.log2fc.iloc[0]) if len(m) else np.nan)
            rows.append(r)
            rowlabels.append(f"{g}   ({int(sym[g])})")
            rowcolors.append(CLASSES[key][1])
    M = np.array(rows)
    norm = TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6)
    ax = fig.add_subplot(slot)
    ax.imshow(np.clip(M, -6, 6), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(10))
    ax.set_xticklabels(VIRUS_ORDER * 2, fontsize=4.2, rotation=90)
    ax.set_yticks(range(len(rowlabels)))
    ax.set_yticklabels(rowlabels, fontsize=4.2, fontstyle="italic")
    for lab, c in zip(ax.get_yticklabels(), rowcolors):
        lab.set_color(c)
    ax.axvline(4.5, color="white", lw=1.6)
    ax.text(2.0, -1.1, "24 hpi", ha="center", va="bottom", fontsize=5.0,
            fontweight="bold")
    ax.text(7.0, -1.1, "72 hpi", ha="center", va="bottom", fontsize=5.0,
            fontweight="bold")
    ax.tick_params(length=1.2, pad=1.2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    cax = ax.inset_axes([1.04, 0.30, 0.035, 0.40])
    mpl.colorbar.ColorbarBase(cax, cmap="RdBu_r", norm=norm,
                              ticks=[-6, 0, 6], extend="both")
    cax.tick_params(labelsize=4.0, length=1.2, pad=1)
    cax.set_title("log₂FC", fontsize=4.2, pad=2)
    return ax


def panel_conservation_bars(ax, ctx):
    ks = list(range(1, 6))
    on = [int((ctx.up_on == k).sum()) for k in ks]
    tot = [int((ctx.gene_up == k).sum()) for k in ks]
    off = [t - o for t, o in zip(tot, on)]
    ax.bar(ks, on, color="#472A78", ec="white", lw=0.3, width=0.74,
           label="on the 382-ISG panel")
    ax.bar(ks, off, bottom=on, color="#B9A5D1", ec="white", lw=0.3, width=0.74,
           label="off the panel (missed by it)")
    for k, t, o in zip(ks, tot, on):
        ax.text(k, t + 22, f"{t:,}", ha="center", va="bottom", fontsize=4.3)
        if t - o:
            ax.text(k, o + (t - o) / 2, f"{t - o}", ha="center", va="center",
                    fontsize=4.0, color="#3A2E52")
    ax.set_xticks(ks)
    ax.set_xlabel("number of viruses inducing the gene", fontsize=4.7,
                  labelpad=1.5)
    ax.set_ylabel("genes", fontsize=4.7, labelpad=1.5)
    ax.tick_params(labelsize=4.4, length=1.3, pad=1)
    n4 = int((ctx.gene_up >= 4).sum())
    off4 = n4 - int((ctx.up_on >= 4).sum())
    ax.set_title(f"{n4:,} genes are induced by ≥ 4 of the 5 viruses;\n"
                 f"{off4} of them are not on the ISG panel",
                 fontsize=4.6, pad=3)
    ax.legend(fontsize=4.2, frameon=False, loc="upper right", handlelength=0.9,
              handletextpad=0.4, labelspacing=0.3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.20)


def panel_restricted_named(fig, slot, ctx, per_unit=8):
    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    gl = ctx.gene_long[ctx.gene_long.treatment == "untreated"]
    rows, labels, colours, seps = [], [], [], []
    for n, tag in ((648, "WNV"), (32, "DENV")):
        genes = list(neuron_of[neuron_of == n].index)
        sub = gl[gl.gene_id.isin(genes)]
        rank = sub.groupby("gene_id").log2fc.apply(lambda v: v.abs().max())
        for gid in rank.sort_values(ascending=False).head(per_unit).index:
            r = []
            for v in VIRUS_ORDER:
                d = sub[(sub.gene_id == gid) & (sub.virus == v)]
                r.append(float(d.log2fc.iloc[d.log2fc.abs().argmax()])
                         if len(d) else np.nan)
            rows.append(r)
            labels.append(str(sub[sub.gene_id == gid].symbol.iloc[0]))
            colours.append(CLASSES[ctx.cls(n)][1])
        seps.append(len(rows))
    M = np.array(rows)
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    ax = fig.add_subplot(slot)
    ax.imshow(np.clip(M, -4, 4), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.axhline(seps[0] - 0.5, color="white", lw=1.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(VIRUS_ORDER, fontsize=4.3, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=4.2, fontstyle="italic")
    for lab, c in zip(ax.get_yticklabels(), colours):
        lab.set_color(c)
    ax.tick_params(length=1.2, pad=1.2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.text(5.7, (seps[0] - 1) / 2, "648\nWNV-restricted", fontsize=4.3,
            color="#B4231C", va="center", ha="left")
    ax.text(5.7, (seps[0] + seps[1] - 1) / 2, "32\nDENV-restricted",
            fontsize=4.3, color="#B4231C", va="center", ha="left")
    ax.set_title(f"strongest {per_unit} members of each, untreated log₂FC",
                 fontsize=4.5, pad=3)
    return ax


def compose_v4(ctx):
    fig_w, fig_h = 183.0, 247.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.020, 0.598, 0.960, 0.392])
    panel_codes_map(ax_a, ctx, pad_scale=1.6, top_strip=False)
    label(fig, 0.008, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.320, 0.440, 0.238])
    panel_kinetic_map(ax_b, ctx)
    label(fig, 0.008, 0.574, "b")

    gs_c = fig.add_gridspec(1, 1, left=0.660, right=0.880, top=0.552,
                            bottom=0.446)
    panel_phase_heatmap(fig, gs_c[0], ctx)
    label(fig, 0.512, 0.574, "c")

    gs_d = fig.add_gridspec(1, 1, left=0.700, right=0.850, top=0.388,
                            bottom=0.296)
    panel_treatment_grid(fig, gs_d[0], ctx)
    label(fig, 0.512, 0.418, "d")

    ax_e = fig.add_axes([0.585, 0.216, 0.380, 0.040])
    panel_succession(ax_e, ctx)
    label(fig, 0.512, 0.276, "e")

    ax_f = fig.add_axes([0.020, 0.036, 0.440, 0.238])
    panel_conservation(ax_f, ctx)
    label(fig, 0.008, 0.290, "f")

    ax_g = fig.add_axes([0.600, 0.110, 0.360, 0.058])
    panel_gene_conservation(ax_g, ctx)
    label(fig, 0.512, 0.190, "g")

    gs_h = fig.add_gridspec(1, 1, left=0.660, right=0.880, top=0.068,
                            bottom=0.038)
    panel_restricted(fig, gs_h[0], ctx)
    label(fig, 0.512, 0.088, "h")

    out = FIGDIR / "fig6_v4_glyph_narrative.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v4 glyphs    : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v5(ctx):
    fig_w, fig_h = 183.0, 247.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.020, 0.600, 0.960, 0.390])
    panel_codes_map(ax_a, ctx, pad_scale=1.6, top_strip=False)
    label(fig, 0.008, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.318, 0.430, 0.240])
    panel_kinetic_map(ax_b, ctx)
    label(fig, 0.008, 0.576, "b")

    ax_c = fig.add_axes([0.610, 0.330, 0.365, 0.228])
    panel_class_distributions(ax_c, ctx)
    label(fig, 0.500, 0.576, "c")

    ax_d = fig.add_axes([0.020, 0.032, 0.430, 0.240])
    panel_conservation(ax_d, ctx)
    label(fig, 0.008, 0.288, "d")

    ax_e = fig.add_axes([0.573, 0.168, 0.165, 0.098])
    panel_trajectory(ax_e, ctx)
    label(fig, 0.500, 0.292, "e")

    ax_f = fig.add_axes([0.815, 0.168, 0.160, 0.098])
    panel_conservation_effect(ax_f, ctx)
    label(fig, 0.762, 0.292, "f")

    ax_g = fig.add_axes([0.585, 0.040, 0.390, 0.078])
    panel_restricted_genes(ax_g, ctx)
    label(fig, 0.500, 0.140, "g")

    out = FIGDIR / "fig6_v5_raw.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v5 raw       : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v6(ctx):
    fig_w, fig_h = 183.0, 244.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.015, 0.640, 0.970, 0.350])
    panel_glyph_atlas(ax_a, ctx)
    label(fig, 0.006, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.352, 0.420, 0.245])
    panel_kinetic_map(ax_b, ctx)
    label(fig, 0.006, 0.616, "b")

    ax_c = fig.add_axes([0.590, 0.420, 0.300, 0.170])
    panel_response_profile(ax_c, ctx)
    label(fig, 0.500, 0.616, "c")

    gs_d = fig.add_gridspec(1, 1, left=0.610, right=0.860, top=0.360,
                            bottom=0.075)
    panel_marker_heatmap(fig, gs_d[0], ctx)
    label(fig, 0.500, 0.386, "d")

    ax_e = fig.add_axes([0.020, 0.060, 0.420, 0.245])
    panel_conservation(ax_e, ctx)
    label(fig, 0.006, 0.322, "e")

    ax_f = fig.add_axes([0.075, 0.010, 0.150, 0.038])
    panel_conservation_bars(ax_f, ctx)
    label(fig, 0.006, 0.056, "f")

    gs_g = fig.add_gridspec(1, 1, left=0.320, right=0.400, top=0.048,
                            bottom=0.008)
    panel_restricted_named(fig, gs_g[0], ctx)
    label(fig, 0.270, 0.056, "g")

    out = FIGDIR / "fig6_v6_reader.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v6 reader    : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def panel_programme_index(ax, ctx, ncol=2):
    rows = -(-len(ctx.sup) // ncol)
    order = sorted(ctx.sup, key=lambda n: (ctx.mod.loc[n, "direction"] != "induced",
                                           -float(ctx.mod.loc[n, "consensus_amp"])))
    ax.set_xlim(0, ncol)
    ax.set_ylim(rows + 1.2, -1.6)
    ax.axis("off")
    for i, n in enumerate(order):
        col, row = divmod(i, rows)
        x = col + 0.012
        cited = n in CITED_IN_RESULTS
        ax.add_patch(Rectangle((x, row + 0.14), 0.022, 0.66,
                               fc=CLASSES[ctx.cls(n)][1], ec="none",
                               transform=ax.transData))
        name = str(ctx.mod.loc[n, "module"])
        ax.text(x + 0.032, row + 0.5, f"{n}", fontsize=4.3, va="center",
                ha="left", fontweight="bold" if cited else "normal",
                color="#1F2328" if cited else "0.42")
        ax.text(x + 0.072, row + 0.5,
                f"{name[:44]}  ({int(ctx.n_genes[n])})", fontsize=4.3,
                va="center", ha="left",
                color="#1F2328" if cited else "0.42")
        if cited:
            ax.text(x - 0.004, row + 0.5, "▸", fontsize=4.0, va="center",
                    ha="right", color="#B4231C")
    for c in range(ncol):
        ax.text(c + 0.012, -0.9,
                "unit · programme (genes)", fontsize=4.6, fontweight="bold",
                va="center", ha="left")
    ax.text(0.012, rows + 0.9,
            f"▸ bold = one of the {len(CITED_IN_RESULTS & set(ctx.sup))} supported "
            f"programmes the Results cites by unit number; the remaining "
            f"{len(ctx.sup) - len(CITED_IN_RESULTS & set(ctx.sup))} are shown on "
            f"the map but not discussed. Swatch = functional class, as in the atlas.",
            fontsize=4.4, color="0.35", va="center", ha="left")


def panel_identity(ax, ctx):
    colors = virus_colors()
    d = ctx.identity
    for i, gene in enumerate(IDENTITY_GENES):
        ut = d[(d.symbol == gene) & (d.treatment == "untreated")]
        ax.scatter(np.full(len(ut), i) + np.linspace(-0.17, 0.17, len(ut)),
                   ut.log2fc, s=6, fc="0.55", ec="white", lw=0.25, zorder=3,
                   label="untreated, all 10 contrasts" if i == 0 else None)
        for j, v in enumerate(("WNV", "YFV")):
            r = d[(d.symbol == gene) & (d.treatment != "untreated")
                  & (d.hpi == 72) & (d.virus == v)]
            if len(r):
                ax.scatter([i + (j - 0.5) * 0.30], [r.log2fc.iloc[0]], s=13,
                           marker="D", fc=colors[v], ec="#1F2328", lw=0.3,
                           zorder=4, label=f"{v} + ruxolitinib, 72 hpi"
                           if i == 0 else None)
    for y in (1, -1):
        ax.axhline(y, color="0.55", lw=0.5, ls=(0, (2.2, 1.6)), zorder=1)
    ax.axhline(0, color="0.3", lw=0.5, zorder=1)
    ax.set_xticks(range(len(IDENTITY_GENES)))
    ax.set_xticklabels(IDENTITY_GENES, fontsize=5.0, fontstyle="italic")
    ax.set_ylabel("log₂FC vs Mock", fontsize=5.2)
    ax.tick_params(labelsize=4.8, length=1.4, pad=1.5)
    ax.set_xlim(-0.6, len(IDENTITY_GENES) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    ax.text(-0.5, 1.08, "|log₂FC| = 1", fontsize=4.3, color="0.45",
            va="bottom", ha="left")
    ax.legend(fontsize=4.4, frameon=False, loc="upper right", handlelength=0.9,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)


def panel_breadth(ax, ctx):
    colors = virus_colors()
    vals = [int(ctx.breadth[v]) for v in VIRUS_ORDER]
    ax.bar(range(len(VIRUS_ORDER)), vals,
           color=[colors[v] for v in VIRUS_ORDER], ec="white", lw=0.3,
           width=0.72, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 40, f"{v:,}", ha="center", va="bottom", fontsize=4.6)
    ax.set_xticks(range(len(VIRUS_ORDER)))
    ax.set_xticklabels(VIRUS_ORDER, fontsize=5.0)
    ax.set_ylabel("differentially expressed genes\nuntreated, 72 hpi",
                  fontsize=5.2)
    ax.tick_params(labelsize=4.8, length=1.4, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    ax.margins(y=0.18)
    lo, hi = min(vals), max(vals)
    ax.set_title(f"{hi / lo:.0f}-fold range ({lo:,}–{hi:,}) — the limit on "
                 f"detecting\nvirus-private programmes in weak infections",
                 fontsize=4.8, pad=3)


def compose_evidence(ctx):
    fig_w, fig_h = 183.0, 148.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.030, 0.395, 0.945, 0.560])
    panel_programme_index(ax_a, ctx)
    label(fig, 0.008, 0.988, "a")

    ax_b = fig.add_axes([0.075, 0.085, 0.425, 0.235])
    panel_identity(ax_b, ctx)
    label(fig, 0.008, 0.355, "b")

    ax_c = fig.add_axes([0.630, 0.085, 0.340, 0.235])
    panel_breadth(ax_c, ctx)
    label(fig, 0.560, 0.355, "c")

    out = FIGDIR / "fig6_evidence.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"evidence     : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


README = """# Fig. 6 candidates

Three alternative compositions of Fig. 6 plus a companion evidence figure,
written by `phh_som/som_figure_context.py`. None of them is the figure of
record: the original composite script still writes
`../figure_som_expressed_fig1_composite.pdf`, unchanged.

**No legend or annotation block sits on a lattice.** The atlas key lives in the
free middle band of the left margin; the kinetic and conservation maps carry
their keys in a strip below the hexagons.

| file | panels | size |
|---|---|---|
| `fig6_v1_annotated` | **a** fan-glyph codes map + regions and call-outs; **b** 20-condition terrain | 183 x 208 mm |
| `fig6_v2_atlas` | **a** atlas; **b** terrain; **c** kinetic map; **d** phase heatmap; **e** treatment grid; **f** succession; **g** conservation; **h** gene conservation; **i** virus-restricted | 183 x **280 mm** |
| `fig6_v3_narrative` | **a** flat atlas; **b** kinetic map; **c** phase heatmap; **d** treatment grid; **e** succession; **f** conservation; **g** gene conservation; **h** virus-restricted | 183 x 232 mm |
| `fig6_v4_glyph_narrative` | v3's layout with v1's panel **a**: every neuron keeps its radial per-condition glyph, with the atlas annotation on top | 183 x 247 mm |
| `fig6_v5_raw` | v4's two maps, but every summary panel replaced by measured gene log2FC: **c** class x condition boxes, **e** 24-vs-72 scatter, **f** conservation vs effect size, **g** the restricted units gene by gene | 183 x 247 mm |
| `fig6_evidence` | **a** index of all 50 supported programmes; **b** identity factors; **c** response breadth | 183 x 148 mm |

`fig6_v4_glyph_narrative` is v3 with the detail of v1's panel a: the codes map
keeps each neuron's radial glyph (one wedge per condition - blue 24 hpi, red
72 hpi, pale = ruxolitinib, hatched inward = negative) and carries the same
class territories, region hulls, call-outs and left-margin key as the flat
atlas. It costs ~15 mm of height over v3 because the fan key needs a top pad,
and the region that reaches the top row is labelled below itself instead.

`fig6_v5_raw` answers the objection that v3/v4's grids are means of codebook
prototypes - two steps from the measurements. Panels c, e, f and g there read
`degs.parquet` directly and plot gene-level log2FC:

- **c** one box per functional class per condition over every member gene x
  five viruses (n = 40 to 2,150 per box), the five per-virus medians drawn on
  top so conservation is judged rather than asserted. It replaces both the
  phase heatmap and the treatment grid, and shows the same partition: the
  ISGF3 box collapses to the zero line under ruxolitinib at 24 hpi while the
  interferon-ligand box does not move.
- **e** all 1,035 programme genes at 24 vs 72 hpi, untreated, five-virus mean.
- **f** the conservation counts with the effect size behind them - every
  induced gene's largest log2FC, split by ISG-panel membership.
- **g** the two virus-restricted programmes gene by gene (23 and 8 genes),
  filled where the gene is called differentially expressed.

**v2 does not fit the 183 x 247 mm page.** It is the only variant that keeps all
20 terrain panels AND the grids that explain them. For submission either use v3
(232 mm) or cut the terrain's ruxolitinib columns, which illustrate the Fig. 7
section rather than this one.

Regenerate:

    cd phh_som && ../.venv/bin/python som_figure_context.py [--variant 1|2|3|4|5|evidence|all]

## How to read the terrain (v1, v2 panel b)

All 20 small maps are the SAME lattice; position never changes meaning, only
colour does. Row = virus. Column = condition: UT is infection vs Mock without
drug, Ruxo is infection + ruxolitinib vs Mock + ruxolitinib - so a Ruxo panel is
the infection response while JAK-STAT is blocked, not the drug's own effect.
Geography, named on the first panel: top-right = interferon / ISG core, top-left
= inflammatory and stress, bottom-left = hepatocyte metabolic.

The terrain's tiers sit within a few hexagons of one another in the top-right
corner, so at ~18 mm per panel the reader cannot see the ligands hold while the
ISGF3 tier collapses. The treatment grid states it instead (5-virus means):

| | UT 24 | UT 72 | Ruxo 24 | Ruxo 72 |
|---|---|---|---|---|
| interferon ligands | +12.7 | +7.9 | +13.6 | +9.3 |
| sensing & IRF3-direct | +4.8 | +4.5 | +2.7 | +3.1 |
| ISGF3 amplification | +3.3 | +3.8 | +0.1 | +1.6 |
| metabolic | -0.3 | -1.5 | -0.1 | -1.0 |

## Which panel carries which claim (v3 / v4 lettering - identical)

| claim in the Results | panel |
|---|---|
| 50 supported programmes, 1,035 genes | a, key + per-call-out gene counts |
| the three regions, incl. the 34 programmes the text never names | a, dashed hulls; full list in evidence a |
| all five viruses occupy the same territory | v1/v2 terrain panel b |
| 34 induced by >= 1 virus, 18 by all five, 2 virus-restricted | f |
| 333 genes by >= 4 viruses, 108 by all five, 11 repressed by all five | g |
| 214 of the 333 off the 382-ISG panel, 84 lncRNA | g |
| three temporal phases | b (map) and c (class x virus x time) |
| the JAK-STAT partition of those phases | d |
| median 50% already DE at 24 hpi vs 0% repressed | e |
| WNV- and DENV-restricted units | f (map) and h (per virus, per time point) |
| named units below threshold (620, 621, 640, 700, 19, 42) | a, dashed hexagons |
| identity factors unchanged untreated | evidence b |
| 15-fold range in untreated 72-hpi DEG breadth (199-3,036) | evidence c |

## Notes

- Every count is re-derived at runtime by the rules in the Methods, and
  printed on each run. Curated here are only the functional classes, the region
  names, the call-out wording, and which units the Results cites by number.
- `check_callout_genes` refuses to draw if a call-out names a gene that is not
  in the neuron it labels; it caught three such errors when it was added.
- The treatment grid (panel d/e) is Fig. 7's subject - the JAK-STAT partition.
  It is here because it is the only thing that makes the terrain's ruxolitinib
  columns readable; if the terrain goes, this panel belongs in Fig. 7.
- v1 and v2 carry the 20 interpolated terrains as vector contours, so their PDFs
  are several MB; rasterize those panels before submission.
- Two edits are still needed in `documents/kimi_publication_draft_expressed.md`
  and have NOT been made: the "shared core" gene list cites (Fig. 6b), which is
  the terrain and carries no gene names - it should cite the atlas panel; and
  the current caption's "neurons named in the Results are ringed and numbered"
  is true of only seven of them.
"""


def write_readme():
    out = FIGDIR / "README.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(README)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description='reader-facing variants of Fig. 6, built beside the original.')
    ap.add_argument("--variant", default="all",
                    choices=["1", "2", "3", "4", "5", "6", "evidence", "all"])
    args = ap.parse_args()

    set_rcparams()
    ctx = Context()
    announce(ctx)
    if args.variant in ("1", "all"):
        compose_v1(ctx)
    if args.variant in ("2", "all"):
        compose_v2(ctx)
    if args.variant in ("3", "all"):
        compose_v3(ctx)
    if args.variant in ("4", "all"):
        compose_v4(ctx)
    if args.variant in ("5", "all"):
        compose_v5(ctx)
    if args.variant in ("6", "all"):
        compose_v6(ctx)
    if args.variant in ("evidence", "all"):
        compose_evidence(ctx)
    write_readme()


if __name__ == "__main__":
    main()
