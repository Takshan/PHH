#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle

import pipeline
from pipeline import FIGURES, MM, save_figure, set_rcparams
from upsetplot import UpSet

V = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]
E = Path("expressed_28x27/data")
PANEL_GENES = Path("../config/fig_conservation_panel_genes.txt")

VIRUS_COLORS = json.loads(
    (pipeline.ROOT / "config" / "colors.json").read_text())["viruses"]
def _soften(hex_color, frac):
    r, g, b = mpl.colors.to_rgb(hex_color)
    return "#{:02x}{:02x}{:02x}".format(
        *(int(round(255 * (1 - frac) + 255 * c * frac)) for c in (r, g, b)))


FCMAP = LinearSegmentedColormap.from_list("fc_jaki_scheme", [
    (0.00, "#1C8AA1"),
    (0.05, _soften("#1C8AA1", 0.9)),
    (0.275, "#65AEBE"),
    (0.50, "#f7f7f7"),
    (0.725, _soften("#A01915", 0.68)),
    (0.95, _soften("#A01915", 0.9)),
    (1.00, "#A01915"),
])
FCNORM = Normalize(vmin=-4 / 0.9, vmax=4 / 0.9)
MISSING = "#DDDDDD"
HPI = (24, 72)
COLS = [(v, h) for v in V for h in HPI]


def load():
    dg = pd.read_parquet("data/degs.parquet")
    sym = dg.drop_duplicates("gene_id").set_index("gene_id")[["symbol", "gene_type"]]
    asg = pd.read_csv(E / "som_neuron_assignments_expressed.tsv", sep="\t").set_index("gene_id")
    cb = pd.read_csv(E / "som_codebook_expressed.tsv", sep="\t").set_index("neuron")
    return dg, sym, asg, cb


def gene_sets(dg, treatment="untreated"):
    d = dg[(dg.treatment == treatment) & dg.deg]
    up = d[d.log2fc > 0].groupby("gene_id")["virus"].apply(frozenset)
    dn = d[d.log2fc < 0].groupby("gene_id")["virus"].apply(frozenset)
    return up, dn


def programmes(dg, asg, cb):
    g_max = dg.groupby("gene_id").agg(maxtpm=("maxgroupmeans", "max"))
    n_de = dg[dg.deg].groupby("gene_id").size().rename("nd")
    a = asg.join(g_max).join(n_de).fillna({"nd": 0})
    us = a.groupby("neuron").agg(n=("neuron", "size"), mt=("maxtpm", "median"), md=("nd", "median"))
    sup = us[(us.n >= 4) & (us.mt > 3) & (us.md >= 4)].index
    peak = pd.DataFrame({v: cb.loc[sup, [f"{v}_untreated24", f"{v}_untreated72"]].max(axis=1)
                         for v in V})
    mask = peak > 1
    ind = mask.any(axis=1)
    return mask[ind], peak[ind]


def panel_genes(spec):
    if "," in spec or not Path(spec).exists():
        genes = [g.strip() for g in spec.split(",") if g.strip()]
        if not genes:
            raise SystemExit(f"--genes: no symbols in {spec!r} and no such file")
        return genes
    return [ln.split("#")[0].strip() for ln in Path(spec).read_text().splitlines()
            if ln.split("#")[0].strip()]


def auto_genes(dg, sym, asg, isg, prog, n):
    all5 = set(prog.index[prog.sum(axis=1) == 5])
    mem = set(asg.index[asg.neuron.isin(all5)])
    ut = dg[(dg.treatment == "untreated") & dg.gene_id.isin(mem)]
    weakest = ut.groupby(["gene_id", "virus"]).log2fc.max().groupby("gene_id").min()
    on_panel = weakest[sym.reindex(weakest.index).symbol.isin(isg).values]
    return [sym.loc[i, "symbol"] for i in on_panel.sort_values(ascending=False).head(n).index]


def fc_table(dg, gene_ids):
    ut = dg[(dg.treatment == "untreated") & dg.gene_id.isin(set(gene_ids))]
    return ut.pivot_table(index="gene_id", columns=["virus", "hpi"],
                          values="log2fc").reindex(gene_ids)


TPM_FLOOR = 0.001
TPM_COLS = [("Mock", 24), ("Mock", 72)] + COLS
TPM_X = list(range(12))


def tpm_table(gene_ids):
    t = pd.read_parquet("data/tpm_long.parquet")
    t = t[(t.treatment == "untreated") & t.gene_id.isin(set(gene_ids))]
    g = t.groupby(["gene_id", "infection", "hpi"]).tpm.mean() \
        .unstack(["infection", "hpi"])
    return np.log10(g.reindex(gene_ids).loc[:, TPM_COLS] + TPM_FLOOR)


def mm_axes(fig, x, y, w, h):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x * MM / fw, y * MM / fh, w * MM / fw, h * MM / fh])


def heat_block(ax, fc, labels, marks=(), frame=None):
    n = len(labels)
    for r, gid in enumerate(fc.index):
        for c, (v, h) in enumerate(COLS):
            val = fc.loc[gid, (v, h)] if (v, h) in fc.columns else np.nan
            face = MISSING if pd.isna(val) else FCMAP(FCNORM(val))
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, fc=face, ec="0.7", lw=0.3))
    ax.add_patch(Rectangle((0, 0), len(COLS), n, fill=False, ec="black",
                           lw=0.9, clip_on=False))
    if frame is not None:
        start, colour = frame
        ax.add_patch(Rectangle((start, -0.12), 2, n + 0.24, fill=False,
                               ec=colour, lw=1.2, clip_on=False))
    for r, lab in enumerate(labels):
        ax.text(-0.4, n - 1 - r + 0.5, lab + ("†" if lab in marks else ""),
                ha="right", va="center", fontsize=5.2,
                style="italic" if not lab.startswith("ENSG") else "normal")
    ax.set_xlim(0, len(COLS))
    ax.set_ylim(0, n)
    ax.set_axis_off()


def unit_profile(fig, x, y, w, h, cb, n, colour):
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([x * MM / fw, y * MM / fh, w * MM / fw, h * MM / fh])
    z = [cb.loc[n, f"{v}_untreated{hp}"] for v, hp in COLS]
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.axhline(1, color="0.75", lw=0.4)
    ax.plot(np.arange(len(z)) + 0.5, z, "-", color=colour, lw=0.7)
    for i, (v_hp, zi) in enumerate(zip(COLS, z)):
        above = zi > 1
        ax.plot(i + 0.5, zi, "o", ms=2.6 if above else 1.4,
                mfc=VIRUS_COLORS[v_hp[0]] if above else "0.75",
                mec="0.15" if above else "white", mew=0.3 if above else 0.25,
                clip_on=False)
    ax.set_xlim(0, len(COLS))
    lim = max(3.4, max(abs(min(z)), abs(max(z))) * 1.1)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([0])
    ax.set_yticklabels(["0"], fontsize=4.4)
    ax.text(len(COLS) + 0.15, 1, "z = 1", fontsize=4.2, color="0.5", va="center",
            ha="left", clip_on=False)
    ax.tick_params(length=1.5, pad=0.8)
    ax.set_ylabel("codebook z", fontsize=4.8, labelpad=1)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    return ax


def chip(ax, x, y, label, color, height):
    ax.add_patch(FancyBboxPatch((x - 0.8, y), 1.6, height,
                                boxstyle="round,pad=0.12,rounding_size=0.3",
                                fc=color, ec="none", clip_on=False))
    ax.text(x, y + 0.35, label, ha="center", va="bottom", rotation=90,
            fontsize=4.4, color="white", fontweight="bold")


def heat_header(ax, n_rows):
    for c, (v, h) in enumerate(COLS):
        ax.text(c + 0.5, n_rows + 0.15, str(h), ha="center", va="bottom", fontsize=4.6,
                color="0.3")
    for i, v in enumerate(V):
        chip(ax, 2 * i + 1, n_rows + 1.3, v, VIRUS_COLORS[v], 2.6)


def tpm_block(fig, x, y, w, h, tp, header=True):
    n = len(tp)
    ax = mm_axes(fig, x, y, w, h)
    norm = mpl.colors.Normalize(vmin=-2, vmax=2)
    cmap = plt.get_cmap("magma")
    for r in range(n):
        for c, xp in enumerate(TPM_X):
            ax.add_patch(Rectangle((xp, n - 1 - r), 1, 1,
                                   fc=cmap(norm(tp.iloc[r, c])), ec="0.7",
                                   lw=0.3))
    ax.add_patch(Rectangle((0, 0), TPM_X[-1] + 1, n, fill=False, ec="black",
                           lw=0.9, clip_on=False))
    if header:
        for xp, (_, hp) in zip(TPM_X, TPM_COLS):
            ax.text(xp + 0.5, n + 0.15, str(hp), ha="center", va="bottom",
                    fontsize=4.6, color="0.3")
        chip(ax, 1, n + 1.3, "Uninfected", "0.55", 5.4)
        for i, v in enumerate(V):
            chip(ax, 2 * i + 3, n + 1.3, v, VIRUS_COLORS[v], 2.6)
    ax.set_xlim(-0.2, TPM_X[-1] + 1.2)
    ax.set_ylim(0, n)
    ax.set_axis_off()


def panel_a(fig, up, x0, y0, w, h, set_label="genes induced", cond_label=None):
    combos = up.value_counts()
    idx = pd.MultiIndex.from_frame(pd.DataFrame(
        [[v in k for v in V] for k in combos.index], columns=V))
    counts = pd.Series(combos.values, index=idx)
    upset = UpSet(counts, sort_by="degree", sort_categories_by="cardinality",
                  facecolor="0.2")

    lab_w = 10.0
    set_w = 20.0
    grid_w = 100.0
    mat_h = 18.0
    bars_h = h - mat_h - 8.0

    names = list(upset.intersections.index.names)
    mems = [frozenset(v for v, b in zip(names, m) if b)
            for m in upset.intersections.index.tolist()]
    cats = list(upset.totals.index)

    ax = mm_axes(fig, x0 + lab_w, y0 + mat_h, grid_w, bars_h)
    upset.plot_intersections(ax=ax)
    n_cols = len(upset.intersections)
    ax.set_xlim(-0.5, n_cols - 0.5)
    for patch, k in zip(ax.patches, mems):
        patch.set_color("0.1" if len(k) >= 2 else VIRUS_COLORS[next(iter(k))])
        patch.set_x(patch.get_x() - 0.1)
        patch.set_width(0.7)
    ymax = float(upset.intersections.max())
    ax.set_ylim(0, ymax * 1.18)
    for i, n in enumerate(upset.intersections.to_numpy()):
        ax.text(i, n + ymax * 0.025, f"{int(n):,}", ha="center", va="bottom",
                fontsize=4.2, rotation=90)
    ax.set_xticks([])
    ax.grid(False)
    ax.set_ylabel("genes in intersection", fontsize=6)
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    mx = mm_axes(fig, x0 + lab_w, y0, grid_w, mat_h)
    row_y = {v: r for r, v in enumerate(reversed(cats))}
    for v, y in row_y.items():
        mx.text(-1.2, y, v, ha="right", va="center", fontsize=6,
                color=VIRUS_COLORS[v], fontweight="bold")
        for i, k in enumerate(mems):
            mx.plot(i, y, "o", ms=3.4,
                    color=VIRUS_COLORS[v] if v in k else "0.87", zorder=2)
    for i, k in enumerate(mems):
        ys = [row_y[v] for v in k]
        if len(ys) > 1:
            mx.plot([i, i], [min(ys), max(ys)], color="0.1", lw=1.1, zorder=1)
    mx.set_xlim(-0.5, n_cols - 0.5)
    mx.set_ylim(-0.7, len(cats) - 0.3)
    mx.set_axis_off()
    if cond_label:
        fw, fh = fig.get_size_inches()
        fig.text((x0 - 6.5) * MM / fw, (y0 + mat_h / 2) * MM / fh, cond_label,
                 rotation=90, fontsize=6.4, fontweight="bold", ha="center",
                 va="center")

    sx = mm_axes(fig, x0 + lab_w + grid_w + 3.0, y0, set_w, mat_h)
    upset.plot_totals(ax=sx)
    for patch, v in zip(sx.patches, cats):
        patch.set_color(VIRUS_COLORS[v])
    sx.set_xlim(0, max(upset.totals) * 1.25)
    sx.set_ylim(len(cats) - 0.4, -0.6)
    sx.set_xlabel(set_label, fontsize=5.6, labelpad=1)
    sx.xaxis.set_label_position("top")
    sx.xaxis.tick_top()
    sx.tick_params(labelsize=5, length=2, pad=1)
    for sp in ("bottom", "right", "left"):
        sx.spines[sp].set_visible(False)

    n_core = int(sum(n for k, n in combos.items() if len(k) >= 4))
    n_all5 = int(sum(n for k, n in combos.items() if len(k) == 5))
    tot = {v: int(sum(n for k, n in combos.items() if v in k)) for v in V}
    return {"n_core": n_core, "n_all5": n_all5, "sets": tot,
            "combos": {tuple(sorted(k)): int(n) for k, n in combos.items()}}


def draw_column(fig, dg, x, top, lab_w, hm_w, ch, blocks, cb=None, header=True,
                twin_w=None):
    fw, fh = fig.get_size_inches()
    y = top
    for i, (title, colour, ids, labels, marks, prof, frame) in enumerate(blocks):
        h = ch * len(ids)
        y -= h
        ax = mm_axes(fig, x + lab_w, y, hm_w, h)
        heat_block(ax, fc_table(dg, ids), labels, marks,
                   frame=None if frame is None else (2 * V.index(frame),
                                                     VIRUS_COLORS[frame]))
        hdr = (17.0 if twin_w is not None else 13.0) if (i == 0 and header) else 0.0
        if hdr:
            heat_header(ax, len(ids))
        if twin_w is not None:
            tpm_block(fig, x + lab_w + hm_w + 2.0, y, twin_w, h,
                      tpm_table(ids), header=i == 0 and header)
        if prof is not None:
            unit_profile(fig, x + lab_w, y + h + hdr + 1.0, hm_w, 6.0, cb, prof, colour)
        title_y = y + h + hdr + (7.5 if prof is not None else 0.0) + 1.2
        fig.text(x * MM / fw, title_y * MM / fh,
                 title, fontsize=5.8, ha="left", va="bottom", color=colour,
                 fontweight="bold", linespacing=1.5)
        y -= 6.0 + 6.0 + (7.5 if prof is not None else 0.0)
    return y + 6.0


def panel_bc(fig, dg, sym, asg, cb, prog, peak, dn, named, title_b, x_b, x_c,
             top, col_w, ch):
    sym2id = {s: i for i, s in sym.symbol.dropna().items()}
    n_prog = prog.sum(axis=1)
    all5 = set(n_prog.index[n_prog == 5])
    lab_w = 17.0

    unknown = [g for g in named if g not in sym2id]
    if unknown:
        raise SystemExit(f"not in the annotation: {', '.join(unknown)}")
    ids = [sym2id[s] for s in named]
    neuron = {s: int(asg.loc[sym2id[s], "neuron"]) for s in named
              if sym2id.get(s) in asg.index}
    off = {s for s in named if neuron.get(s) not in all5}
    rep = sorted(dn.index[dn.apply(len) == 5], key=lambda i: sym.loc[i, "symbol"])

    hm_b = 26.0
    bottom_b = draw_column(fig, dg, x_b, top, lab_w, hm_b, ch, [
        (title_b, "#0E5C57", ids,
         [sym.loc[i, "symbol"] for i in ids], off, None, None),
        ("repressed by all five", "#2166AC", rep,
         [sym.loc[i, "symbol"] for i in rep], (), None, None),
    ], twin_w=31.2)

    restricted = {v: int(n) for n in prog.index[n_prog == 1] for v in V if prog.loc[n, v]}
    zpeak = {}
    n_mem = {}
    blocks = []
    for v, n in sorted(restricted.items(),
                       key=lambda kv: -int((asg.neuron == kv[1]).sum())):
        mem = list(asg.index[asg.neuron == n])
        amp = dg[(dg.treatment == "untreated") & dg.gene_id.isin(set(mem))]
        amp = amp.assign(a=amp.log2fc.abs()).groupby("gene_id").a.max()
        mem = sorted(mem, key=lambda i: -amp.get(i, 0))
        z = float(peak.loc[n, v])
        zpeak[v] = z
        n_mem[v] = len(mem)
        zz = {(u, h): cb.loc[n, f"{u}_untreated{h}"] for u in V for h in (24, 72)}
        if all(abs(zz[(u, h)]) <= 1 for u in V if u != v for h in (24, 72)):
            title = f"neuron {n} · {v}-restricted"
        else:
            hp = max((24, 72), key=lambda h: zz[(v, h)])
            down = sorted((u for u in V if zz[(u, 72)] < -1),
                          key=["WNV", "YFV", "DENV", "TBEV", "ZIKV"].index)
            title = (f"neuron {n} · marginal in {v} at {hp} hpi\n"
                     f"repressed at 72 hpi in {', '.join(down)}")
        blocks.append((title, VIRUS_COLORS[v], mem,
                       [sym.loc[i, "symbol"] for i in mem], (), n, v))
    bottom_c = draw_column(fig, dg, x_c, top, lab_w, hm_b, ch, blocks, cb=cb,
                           twin_w=31.2)
    return {"named": named, "off_all5": off, "neuron": neuron,
            "restricted": restricted, "peak_z": zpeak, "n_members": n_mem,
            "bottom_b": bottom_b, "bottom_c": bottom_c,
            "repressed": [sym.loc[i, "symbol"] for i in rep]}


def programme_strip(fig, dg, sym, asg, prog, named, x0, y0, w):
    all5 = list(prog.index[prog.sum(axis=1) == 5])
    mod = pd.read_csv(pipeline.ROOT / "phh_som" / "data" / "som_expressed_module_table.csv",
                      index_col="unit")["module"]
    meanfc = dg[dg.treatment == "untreated"].groupby("gene_id").log2fc.mean()
    canon = set(named)
    units = sorted(all5, key=lambda n: -int((asg.neuron == n).sum()))
    row_h, cell, lab_w = 2.6, 1.5, 28.0
    ax = mm_axes(fig, x0, y0, w, 9 * row_h + 6)
    ax.set_xlim(0, w)
    ax.set_ylim(0, 9 * row_h + 6)
    ax.set_axis_off()
    ax.text(0, 9 * row_h + 3.5,
            "all-five programmes, all member genes — mean untreated "
            "log$_2$FC; outline = named in the Results",
            fontsize=5.4, va="center")
    for col in range(2):
        cx = col * (w / 2)
        for r, n in enumerate(units[col * 9:(col + 1) * 9]):
            y = (8 - r) * row_h
            name = str(mod.get(n, ""))
            if len(name) > 26:
                name = name[:24] + "…"
            ax.text(cx, y + row_h / 2, f"N{n} · {name}", fontsize=4.4,
                    va="center", ha="left")
            genes = list(asg.index[asg.neuron == n])
            genes.sort(key=lambda g: -meanfc.get(g, 0))
            for j, g in enumerate(genes):
                fc = meanfc.get(g, np.nan)
                face = MISSING if pd.isna(fc) else FCMAP(FCNORM(fc))
                xc = cx + lab_w + j * cell
                ax.add_patch(Rectangle((xc, y + 0.3), cell - 0.15, row_h - 0.6,
                                       fc=face, ec="0.7", lw=0.2))
                if sym.symbol.get(g) in canon:
                    ax.add_patch(Rectangle((xc, y + 0.3), cell - 0.15,
                                           row_h - 0.6, fill=False,
                                           ec="black", lw=0.6))
    return y0


def legend(fig, x0, y0, w):
    ax = mm_axes(fig, x0, y0, w, 21)
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(-1, 20)

    ax.text(0, 19.5, "upset: bar = genes induced in exactly the virus combination\n"
            "dotted below; a: untreated (top) vs + JAKi (bottom)",
            fontsize=5.4, va="top", linespacing=1.3)

    ax.add_patch(Rectangle((46, 16.4), 2.6, 2.0, fc="0.1", lw=0))
    ax.text(49.4, 18.2, "bar: shared by ≥ 2 viruses;", fontsize=4.6, va="center")
    ax.text(49.4, 16.2, "singleton bars in virus colours", fontsize=4.6,
            va="center")

    ax.text(0, 11.8, "log$_2$FC vs mock, untreated (b, left; c)", fontsize=5.4,
            va="bottom")
    for i in range(120):
        ax.add_patch(Rectangle((3.3 + i * 27 / 120, 9.0), 27 / 120 + 0.02, 2.0,
                               fc=FCMAP(FCNORM(-4 + 8 * i / 119)), lw=0))
    ax.add_patch(Rectangle((0, 9.0), 2.5, 2.0, fc="#1C8AA1", lw=0.5, ec="0.2"))
    ax.add_patch(Rectangle((31.1, 9.0), 2.5, 2.0, fc="#A01915", lw=0.5, ec="0.2"))
    for f, lb in ((3.3, "−4"), (10.05, "−2"), (16.8, "0"), (23.55, "+2"),
                  (30.3, "+4")):
        ax.text(f, 8.5, lb, fontsize=4.6, ha="center", va="top")
    ax.text(-0.3, 10.0, "<−4", fontsize=4.6, ha="right", va="center")
    ax.text(33.9, 10.0, ">4", fontsize=4.6, ha="left", va="center")
    ax.add_patch(Rectangle((36.6, 9.0), 2.6, 2.0, fc=MISSING, lw=0))
    ax.text(40.2, 10.0, "not tested", fontsize=4.8, va="center")

    for rx0, rx1, ry, tcks in (
            (3.3, 30.3, 9.0, [3.3, 10.05, 16.8, 23.55, 30.3]),
            (0.0, 30.0, 1.8, [0.0, 7.5, 15.0, 22.5, 30.0])):
        ax.add_patch(Rectangle((rx0, ry), rx1 - rx0, 2.0, fill=False,
                               ec="0.2", lw=0.5))
        for tx in tcks:
            ax.plot([tx, tx], [ry, ry + 0.7], color="0.2", lw=0.5,
                    solid_capstyle="butt")
            ax.plot([tx, tx], [ry + 1.3, ry + 2.0], color="0.2", lw=0.5,
                    solid_capstyle="butt")

    ax.text(0, 4.6, "log$_{10}$(TPM), group mean (b, right)",
            fontsize=5.4, va="bottom")
    cmap = plt.get_cmap("magma")
    for i in range(120):
        ax.add_patch(Rectangle((i * 30 / 120, 1.8), 30 / 120 + 0.02, 2.0,
                               fc=cmap(i / 119), lw=0))
    for f, lb in ((0, "−2"), (0.25, "−1"), (0.5, "0"), (0.75, "1"), (1, "2")):
        ax.text(f * 30, 1.3, lb, fontsize=4.6, ha="center", va="top")

    ax.text(41.6, 2.8, "†", fontsize=5, ha="center", va="center")
    ax.text(43.6, 2.8, "gene's programme is induced by four viruses, not five",
            fontsize=4.8, va="center")


def main(argv=None):
    ap = argparse.ArgumentParser(description='the conservation paragraph as one figure (UpSet + heat maps).')
    ap.add_argument("--genes", default=str(PANEL_GENES),
                    help="panel b rows: a file of HGNC symbols (one per line), a "
                         "comma-separated list, or 'auto' to take them from the data - the "
                         "ISG-panel members of the all-five programmes, weakest virus "
                         "first, as many rows as the default file has "
                         "(default: %(default)s)")
    args = ap.parse_args(argv)

    set_rcparams()
    dg, sym, asg, cb = load()
    up, dn = gene_sets(dg)
    up_rx, _ = gene_sets(dg, "Ruxo")
    prog, peak = programmes(dg, asg, cb)

    if args.genes == "auto":
        n = len(panel_genes(str(PANEL_GENES))) if PANEL_GENES.exists() else 18
        named = auto_genes(dg, sym, asg, pipeline.load_isg_symbols(), prog, n)
        title_b = "conserved core\nISG-panel genes in the all-five programmes"
    else:
        named = panel_genes(args.genes)
        title_b = "conserved core\ncanonical components named in the Results"
    CH = 2.6
    W = 180.0
    n_named = len(named)
    n_rep = int((dn.apply(len) == 5).sum())
    n_res = sorted((asg.neuron == n).sum() for n in prog.index[prog.sum(axis=1) == 1])
    left = CH * (n_named + n_rep) + 2 * 6 + 13 + 6
    right = CH * sum(n_res) + 2 * 6 + 13 + 2 * 7.5 + 6
    col = max(left, right + 4)
    A_BAND = 176.0
    H = A_BAND + col + 6 + 34

    fig = plt.figure(figsize=(W * MM, H * MM))
    top = H - A_BAND
    info_bc = panel_bc(fig, dg, sym, asg, cb, prog, peak, dn, named, title_b,
                       12, W / 2 + 4, top,
                       W / 2 - 16, CH)
    strip_y0 = 29.0
    programme_strip(fig, dg, sym, asg, prog, named, 12, strip_y0, W - 24)
    legend(fig, 12, 4.0, W / 2 - 16)
    info_a = panel_a(fig, up, 14, top + 102, W - 20, 66, cond_label="untreated")
    info_ar = panel_a(fig, up_rx, 14, top + 30, W - 20, 64,
                      cond_label="+ JAKi")

    for lab, (x, y) in (("a", (3, H - 4)), ("b", (3, top + 13)),
                        ("c", (W / 2 - 1, top + 21)),
                        ("d", (3, strip_y0 + 27.5))):
        fig.text(x * MM / (W * MM), y * MM / (H * MM), lab, fontsize=9,
                 fontweight="bold", va="top", ha="left")

    tag = {str(PANEL_GENES): "", "auto": "_auto"}.get(args.genes, "_custom")
    stem = f"figs_conservation_upset{tag}"
    written = save_figure(fig, FIGURES / "conservation_programmes" / f"{stem}.pdf", bbox_inches=None)
    plt.close(fig)

    n_prog = prog.sum(axis=1)
    print(f"programmes induced by >=1 virus : {len(prog)}"
          f"   all five: {int((n_prog == 5).sum())}"
          f"   four: {int((n_prog == 4).sum())}"
          f"   restricted: {int((n_prog == 1).sum())} {info_bc['restricted']}"
          f"   members {info_bc['n_members']}   peak z {info_bc['peak_z']}")
    print(f"genes induced by >=4 viruses    : {info_a['n_core']}   all five: {info_a['n_all5']}")
    print(f"  under + Ruxolitinib           : {info_ar['n_core']}   all five: {info_ar['n_all5']}")
    print(f"genes repressed by all five     : {len(info_bc['repressed'])} "
          f"({', '.join(info_bc['repressed'])})")
    print(f"panel b rows: {len(named)} from {args.genes}: {', '.join(named)}")
    print(f"rows off an all-five programme: {sorted(info_bc['off_all5'])} "
          f"-> { {g: info_bc['neuron'][g] for g in info_bc['off_all5']} }")
    print("set sizes:", info_a["sets"], f"   page {W:.0f} x {H:.0f} mm")
    for q in written:
        print("wrote", q)


if __name__ == "__main__":
    sys.exit(main())
