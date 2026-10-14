#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from phh_som.utils.deg import gene_annotation
from phh_som.utils.pipeline import (DATA as BASE_DATA, VIRUS_ORDER,
                      load_isg_symbols, universe, virus_colors)
from phh_som.plotting.plot_som_map import Grid, grid_dims, lighten, fan_scale
from phh_som.plotting.plot_codes_map import (draw_codes_map, fan_glyph,
                            fan_key, panel_geometry)
from phh_som.utils.somdata import supported_neuron_stats
from phh_som.utils.somgeom import hex_ring


UNIVERSE = universe("expressed_28x27")

DATA = UNIVERSE / "data"

TAG = "expressed"

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
        self.mod = pd.read_csv(BASE_DATA / "som_expressed_module_table.csv",
                               index_col="unit")
        degs = pd.read_parquet(BASE_DATA / "degs.parquet")

        self.sup, us = supported_neuron_stats(
            degs, self.assign.set_index("gene_id"))
        self.sup = list(self.sup)

        ut = {v: self.cb[[f"{v}_untreated24", f"{v}_untreated72"]] for v in VIRUS_ORDER}
        self.amp_up = pd.DataFrame({v: ut[v].max(axis=1) for v in VIRUS_ORDER})
        self.amp_dn = pd.DataFrame({v: ut[v].min(axis=1) for v in VIRUS_ORDER})
        self.n_up = (self.amp_up.loc[self.sup] > 1).sum(axis=1)
        self.n_dn = (self.amp_dn.loc[self.sup] < -1).sum(axis=1)
        self.n_shared = pd.Series(
            {n: (self.n_up[n] if self.n_up[n] >= self.n_dn[n] else self.n_dn[n])
             for n in self.sup})
        self.restricted = {int(n): v for n in self.n_up.index[self.n_up == 1]
                           for v in VIRUS_ORDER if self.amp_up.loc[n, v] > 1}

        self.n_genes = us["n"]

        d = degs[(degs.treatment == "untreated") & degs.deg]
        self.gene_up = d[d.log2fc > 0].groupby("gene_id")["virus"].nunique()
        self.gene_dn = d[d.log2fc < 0].groupby("gene_id")["virus"].nunique()
        self.sym = gene_annotation(degs)
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

def region_of(ctx, neuron):
    for r in ctx.regions:
        if int(neuron) in r["units"]:
            return r["name"]
    return None

SHORT_CLASS = {"ifn": "IFN ligands", "sense": "Sensing / IRF3",
               "isgf3": "ISGF3 effectors", "antigen": "Antigen pres.",
               "inflam": "Inflammatory", "metab": "Metabolic"}
