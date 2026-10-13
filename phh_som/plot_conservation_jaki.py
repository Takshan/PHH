#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pipeline
from pipeline import FIGURES, MM, save_figure, set_rcparams
from conservation_programmes import (FCMAP, FCNORM, MISSING,
                                    TPM_FLOOR, VIRUS_COLORS, gene_sets, load,
                                    mm_axes, panel_genes, programmes)

CELL = 1.7
LABEL_W = 11.5
BLOCK_GAP = 3.0
HEADER_H = 12.5
ZBAND_H = 7.0
TPM_GAP = 2.0
TPM_W = 24 * CELL
SUBCOLS = [(24, "untreated", "−"), (24, "Ruxo", "+"),
           (72, "untreated", "−"), (72, "Ruxo", "+")]
COL_BLOCKS = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]
TPM_BLOCKS = ["Mock"] + COL_BLOCKS
VIRUS_ROWS = COL_BLOCKS


def tblock(fig, x, y, ids, label_key, fc):
    n = len(ids)
    ax = mm_axes(fig, x, y, 4 * CELL, n * CELL)
    for r, g in enumerate(ids):
        for c, (hpi, treatment, sign) in enumerate(SUBCOLS):
            val = fc.get((g, label_key, treatment, hpi))
            face = MISSING if val is None or pd.isna(val) \
                else FCMAP(FCNORM(val))
            ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1, fc=face, ec="0.7",
                                       lw=0.2))
    ax.add_patch(plt.Rectangle((0, 0), 4, n, fill=False, ec="black", lw=0.7,
                               clip_on=False))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, n)
    ax.set_axis_off()


def col_header(fig, x, y, width):
    ax = mm_axes(fig, x, y, width, HEADER_H)
    ax.set_xlim(0, width)
    ax.set_ylim(0, HEADER_H)
    ax.set_axis_off()
    for i, v in enumerate(COL_BLOCKS):
        cx = (i * 4 + 2) * CELL
        ax.add_patch(plt.Rectangle((cx - 2.2, 4.8), 4.4, 7.2,
                                   fc=VIRUS_COLORS[v], ec="none",
                                   clip_on=False))
        ax.text(cx, 8.4, v, ha="center", va="center", fontsize=4.2,
                color="white", fontweight="bold", rotation=90)
        for c, (hpi, _, sign) in enumerate(SUBCOLS):
            ax.text((i * 4 + c + 0.5) * CELL, 0.2, f"{hpi} {sign}",
                    ha="center", va="bottom", fontsize=3.4, color="0.25",
                    rotation=90)


def tpm_twin(fig, x, y, ids, tp):
    n = len(ids)
    ax = mm_axes(fig, x, y, TPM_W, n * CELL)
    norm = mpl.colors.Normalize(vmin=-2, vmax=2)
    cmap = plt.get_cmap("magma")
    for r, g in enumerate(ids):
        for bi, name in enumerate(TPM_BLOCKS):
            for c, (hpi, treatment, _) in enumerate(SUBCOLS):
                c0 = (bi * 4 + c) * 3
                for d in (1, 2, 3):
                    val = tp.get((g, name, treatment, hpi, d))
                    face = MISSING if val is None else cmap(norm(val))
                    ax.add_patch(plt.Rectangle((c0 + d - 1, n - 1 - r), 1, 1,
                                               fc=face, ec="0.7", lw=0.1))
    ax.add_patch(plt.Rectangle((0, 0), 72, n, fill=False, ec="black", lw=0.7,
                               clip_on=False))
    for gi in range(24):
        for d in (1, 2, 3):
            ax.text(gi * 3 + d - 0.5, -0.25, f"d{d}", ha="center", va="top",
                    fontsize=2.6, color="0.25", rotation=90, clip_on=False)
    ax.set_xlim(0, 72)
    ax.set_ylim(0, n)
    ax.set_axis_off()


def tpm_header(fig, x, y):
    ax = mm_axes(fig, x, y, TPM_W, HEADER_H)
    ax.set_xlim(0, TPM_W)
    ax.set_ylim(0, HEADER_H)
    ax.set_axis_off()
    chips = [("Mock", "0.55")] + [(v, VIRUS_COLORS[v]) for v in COL_BLOCKS]
    for i, (name, color) in enumerate(chips):
        cx = (i * 4 + 2) * CELL
        ax.add_patch(plt.Rectangle((cx - 2.2, 4.8), 4.4, 7.2, fc=color,
                                   ec="none", clip_on=False))
        ax.text(cx, 8.4, name, ha="center", va="center", fontsize=4.2,
                color="white", fontweight="bold", rotation=90)
        for c, (hpi, _, sign) in enumerate(SUBCOLS):
            ax.text((i * 4 + c + 0.5) * CELL, 0.2, f"{hpi} {sign}",
                    ha="center", va="bottom", fontsize=3.4, color="0.25",
                    rotation=90)


def z_profile(fig, x, y, width, gv, gn, cb, annotate):
    ax = mm_axes(fig, x, y, width, ZBAND_H)
    z = [cb.loc[gn, f"{v}_untreated{hp}"]
         for v in VIRUS_ROWS for hp in (24, 72)]
    lim = max(3.4, max(abs(zi) for zi in z) * 1.1)
    xs, vs = [], []
    for i, v in enumerate(VIRUS_ROWS):
        for hp, c0 in ((24, 0), (72, 2)):
            xs.append((i * 4 + c0 + 0.5) * CELL)
            vs.append(v)
    ax.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.axhline(1, color="0.75", lw=0.4)
    ax.plot(xs, z, "-", color=VIRUS_COLORS[gv], lw=0.7, zorder=1)
    for xp, v, zi in zip(xs, vs, z):
        above = zi > 1
        ax.plot(xp, zi, "o", ms=2.2 if above else 1.2,
                mfc=VIRUS_COLORS[v] if above else "0.75",
                mec="0.15" if above else "white",
                mew=0.3 if above else 0.25, clip_on=False, zorder=2)
    ax.text(width + 0.8, 1, "z = 1", fontsize=3.8, color="0.5", va="center",
            ha="left", clip_on=False)
    ax.set_xlim(0, width)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([-2, 0, 2])
    ax.set_yticklabels(["−2", "0", "2"], fontsize=3.6)
    ax.tick_params(length=1.2, pad=0.6)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    if annotate:
        ax.set_ylabel("codebook z", fontsize=4.4, labelpad=1.5)


def main():
    set_rcparams()
    mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    dg, sym, asg, cb = load()
    prog, _ = programmes(dg, asg, cb)
    named = panel_genes("../config/fig_conservation_panel_genes.txt")
    sym2id = {s: i for i, s in sym.symbol.dropna().items()}
    core = [sym2id[s] for s in named]

    restricted = {v: int(n) for n in prog.index[prog.sum(axis=1) == 1]
                  for v in pipeline.VIRUS_ORDER if prog.loc[n, v]}
    ut = dg[dg.treatment.isin(["untreated", "Ruxo"])]
    amp = ut[ut.treatment == "untreated"]
    _, dn = gene_sets(dg)
    rep = sorted(dn.index[dn.apply(len) == 5], key=lambda i: sym.loc[i, "symbol"])
    strict = set()
    groups = [("conserved core", core, None, None),
              ("repressed by all five", rep, None, None)]
    for v, n in sorted(restricted.items(),
                       key=lambda kv: -int((asg.neuron == kv[1]).sum())):
        mem = list(asg.index[asg.neuron == n])
        strength = amp[amp.gene_id.isin(mem)].groupby("gene_id").log2fc.max()
        mem = sorted(mem, key=lambda g: -strength.get(g, 0))
        z = {(u, hp): cb.loc[n, f"{u}_untreated{hp}"]
             for u in COL_BLOCKS for hp in (24, 72)}
        if all(abs(z[(u, hp)]) <= 1 for u in COL_BLOCKS if u != v
               for hp in (24, 72)):
            strict.add(n)
            label = f"neuron {n} · {v}-restricted"
        else:
            hp = max((24, 72), key=lambda h: z[(v, h)])
            down = [u for u in COL_BLOCKS if z[(u, 72)] < -1]
            label = (f"neuron {n} · z > 1 only in {v} ({hp} hpi, z {z[(v, hp)]:.2f})\n"
                     f"repressed at 72 hpi in {', '.join(down)} (z < −1)")
        groups.append((label, mem, v, n))

    fc = dg.set_index(["gene_id", "virus", "treatment", "hpi"]).log2fc \
        .to_dict()
    tp = np.log10(pd.read_parquet("data/tpm_long.parquet")
                  .groupby(["gene_id", "infection", "treatment", "hpi",
                            "donor"])
                  .tpm.mean() + TPM_FLOOR).to_dict()

    grid_w = len(COL_BLOCKS) * 4 * CELL
    stride = LABEL_W + 0.8 + grid_w + TPM_GAP + TPM_W
    max_rows = max(len(ids) for _, ids, _, _ in groups)
    TOP_BAND = 6.0 + ZBAND_H + 1.0 + HEADER_H + 0.5
    BOTTOM = 11.0
    W = 4.0 + len(groups) * stride + (len(groups) - 1) * BLOCK_GAP + 7.0
    H = TOP_BAND + max_rows * CELL + BOTTOM
    grid_top = H - TOP_BAND

    fig = plt.figure(figsize=(W * MM, H * MM))
    annotated = False
    for gi, (gname, ids, gv, gn) in enumerate(groups):
        h = len(ids) * CELL
        gx = 4.0 + gi * (stride + BLOCK_GAP) + LABEL_W + 0.8
        y = grid_top - h
        for bi, v in enumerate(COL_BLOCKS):
            tblock(fig, gx + bi * 4 * CELL, y, ids, v, fc)
        col_header(fig, gx, grid_top + 0.5, grid_w)
        tpm_twin(fig, gx + grid_w + TPM_GAP, y, ids, tp)
        tpm_header(fig, gx + grid_w + TPM_GAP, grid_top + 0.5)
        fig.text((gx + (grid_w + TPM_GAP + TPM_W) / 2) * MM / (W * MM),
                 (grid_top + 0.5 + HEADER_H + 1.0 + ZBAND_H + 0.8)
                 * MM / (H * MM), gname, fontsize=5.6, ha="center",
                 va="bottom", linespacing=1.15,
                 color=VIRUS_COLORS[gv] if gn in strict else "black",
                 fontweight="bold" if gn in strict else "normal")
        if gn is not None:
            z_profile(fig, gx, grid_top + 0.5 + HEADER_H + 1.0, grid_w,
                      gv, gn, cb, annotate=not annotated)
            annotated = True
            ov = mm_axes(fig, gx, y, grid_w, h)
            ov.set_xlim(0, grid_w)
            ov.set_ylim(0, h)
            ov.set_axis_off()
            bi = COL_BLOCKS.index(gv)
            for hp in (24, 72):
                if cb.loc[gn, f"{gv}_untreated{hp}"] > 1:
                    c0 = 0 if hp == 24 else 2
                    ov.add_patch(plt.Rectangle(
                        (bi * 4 * CELL + c0 * CELL - 0.08, -0.08),
                        2 * CELL + 0.16, h + 0.16, fill=False,
                        ec=VIRUS_COLORS[gv], lw=1.1, clip_on=False))
        axl = mm_axes(fig, gx - 0.8 - LABEL_W, y, LABEL_W, h)
        axl.set_xlim(0, LABEL_W)
        axl.set_ylim(0, h)
        axl.set_axis_off()
        for r, g in enumerate(ids):
            symbol = sym.symbol.get(g)
            label = symbol if isinstance(symbol, str) else g
            axl.text(LABEL_W, h - r * CELL - CELL / 2, label,
                     ha="right", va="center", fontsize=4.0, color="0.15",
                     style="italic" if isinstance(symbol, str) else "normal")

    kx, ky, kw, kh, bw = 11.0, 3.5, 21.5, 2.0, 2.5
    kax = mm_axes(fig, kx - bw - 5, ky, kw + 2 * bw + 56, kh)
    kax.set_axis_off()
    kax.set_xlim(-bw - 5, kw + bw + 51)
    kax.set_ylim(0, kh)
    for i in range(120):
        kax.add_patch(plt.Rectangle((i * kw / 120, 0), kw / 120 + 0.02, kh,
                                    fc=FCMAP(FCNORM(-4 + 8 * i / 119)), lw=0))
    kax.add_patch(plt.Rectangle((0, 0), kw, kh, fill=False, ec="0.2", lw=0.5))
    for tx in (0, kw / 4, kw / 2, 3 * kw / 4, kw):
        kax.plot([tx, tx], [0, 0.7], color="0.2", lw=0.5,
                 solid_capstyle="butt")
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
    kax.text(kw + bw + 5.5, kh / 2, "log$_2$FC", fontsize=5.2, va="center")
    tx0, tw = kw + bw + 15.0, 12.0
    tnorm = mpl.colors.Normalize(vmin=-2, vmax=2)
    tcmap = plt.get_cmap("magma")
    for i in range(120):
        kax.add_patch(plt.Rectangle((tx0 + i * tw / 120, 0), tw / 120 + 0.02,
                                    kh, fc=tcmap(tnorm(-2 + 4 * i / 119)),
                                    lw=0))
    kax.add_patch(plt.Rectangle((tx0, 0), tw, kh, fill=False, ec="0.2",
                                lw=0.5))
    for f, lb in ((0, "−2"), (0.25, "−1"), (0.5, "0"), (0.75, "+1"),
                  (1, "+2")):
        tx = tx0 + f * tw
        kax.plot([tx, tx], [0, 0.7], color="0.2", lw=0.5,
                 solid_capstyle="butt")
        kax.plot([tx, tx], [kh - 0.7, kh], color="0.2", lw=0.5,
                 solid_capstyle="butt")
        kax.text(tx, -0.6, lb, fontsize=4.6, ha="center", va="top")
    kax.text(tx0 + tw + 1.5, kh / 2, "log$_{10}$(TPM), per donor", fontsize=5.2,
             va="center")

    out = save_figure(fig, FIGURES / "plot_conservation_jaki" / "figs_conservation_jaki.pdf",
                      bbox_inches=None)
    plt.close(fig)
    core_fc = pd.DataFrame(
        {(v, t, h): [fc.get((g, v, t, h), np.nan) for g in core]
         for v in VIRUS_ROWS for h, t, _ in SUBCOLS})
    for v in VIRUS_ROWS:
        u72 = core_fc[(v, "untreated", 72)].mean()
        r72 = core_fc[(v, "Ruxo", 72)].mean()
        print(f"  core mean log2FC 72h {v:<5}: untreated {u72:+.2f}  "
              f"Ruxo {r72:+.2f}  retained {100 * r72 / u72:.0f}%")
    rep_fc = pd.DataFrame(
        {(v, t, h): [fc.get((g, v, t, h), np.nan) for g in rep]
         for v in VIRUS_ROWS for h, t, _ in SUBCOLS})
    for v in VIRUS_ROWS:
        u72 = rep_fc[(v, "untreated", 72)].mean()
        r72 = rep_fc[(v, "Ruxo", 72)].mean()
        print(f"  repressed mean log2FC 72h {v:<5}: untreated {u72:+.2f}  "
              f"Ruxo {r72:+.2f}")
    for p in out:
        print("wrote", p)


if __name__ == "__main__":
    main()
