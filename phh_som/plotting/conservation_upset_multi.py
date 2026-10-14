#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from phh_som.utils import pipeline
from phh_som.utils.pipeline import FIGURES, MM, save_figure, save_svg, set_rcparams
from phh_som.plotting.conservation_programmes import V, VIRUS_COLORS, load, mm_axes

W, H = 183.0, 105.0
MIN_GENES = 10
ARMS = ("untreated", "Ruxo")
HPIS = (24, 72)
ARM_LABEL = {"untreated": "ut", "Ruxo": "+R"}
FS_MIN = 5.0
PAL = json.loads((pipeline.ROOT / "config" / "colors.json").read_text())
BAR_UT, BAR_RX, BAR_MIX = "0.15", PAL["padjust_gradient"][0][1], \
    PAL["padjust_gradient"][3][1]

CLASSES = ["untreated · ≥4 viruses", "untreated · 2–3 viruses",
           "untreated · virus-private", "mixed arms", "+Ruxo only (emergent)"]


def _tint(colour, strength):
    r, g, b = mpl.colors.to_rgb(colour)
    return "#{:02x}{:02x}{:02x}".format(
        *(int(round(255 * (1 - strength) + 255 * c * strength))
          for c in (r, g, b)))


BAND = {0: _tint(BAR_UT, 0.10), 1: _tint(BAR_UT, 0.10),
        2: _tint(BAR_UT, 0.10), 3: _tint(BAR_MIX, 0.22),
        4: _tint(BAR_RX, 0.16)}


def column_class(combo):
    n_ut = len({c[0] for c in combo if c[1] == "untreated"})
    n_rx = len({c[0] for c in combo if c[1] == "Ruxo"})
    if n_rx == 0:
        return 0 if n_ut >= 4 else (1 if n_ut >= 2 else 2)
    return 4 if n_ut == 0 else 3


def is_core(combo):
    return len({c[0] for c in combo if c[1] == "untreated"}) == 5


def bar_colour(combo):
    arms = {c[1] for c in combo}
    if arms == {"untreated"}:
        return BAR_UT
    if arms == {"Ruxo"}:
        return BAR_RX
    return BAR_MIX


def assemble(dg):
    d = dg[dg.deg & (dg.log2fc > 0)]
    va = d.assign(va=list(zip(d.virus, d.treatment)))
    sets = va.groupby("gene_id")["va"].apply(frozenset)
    counts = sets.value_counts()
    cond_genes = {c: set(g.gene_id)
                  for c, g in d.groupby(["virus", "treatment", "hpi"])}
    cond_tot = {c: len(g) for c, g in cond_genes.items()}
    viruses = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]
    shown = counts[counts >= MIN_GENES]
    shown = shown[sorted(shown.index,
                         key=lambda k: (column_class(k), -shown[k],
                                        tuple(sorted(k))))]
    col_genes = [set(sets.index[sets == k]) for k in shown.index]
    return shown, counts, cond_genes, cond_tot, viruses, col_genes


def draw(fig, shown, counts, cond_genes, cond_tot, viruses, col_genes,
         dy=0.0):
    conds = [(v, arm, hpi) for v in viruses for hpi in HPIS for arm in ARMS]
    row_y = {c: len(conds) - 1 - i for i, c in enumerate(conds)}
    combos = list(shown.index)
    n = len(combos)
    cells = [[len(g & cond_genes.get(c, ())) for c in conds]
             for g in col_genes]

    x0, lab_w, grid_w, set_w = 7.0, 14.0, 126.0, 28.0
    mat_h, bars_h = 56.0, 40.0
    y_mat = 4 + dy
    col_u = n / grid_w
    HPI_X, JAKI_X = -3.2 * col_u, -5.2 * col_u
    CHIP_L, CHIP_W = -15.0 * col_u, 8.4 * col_u

    classes = [column_class(k) for k in combos]
    bounds = [i for i in range(1, n) if classes[i] != classes[i - 1]]
    spans = [(s, e - 1, classes[s]) for s, e in
             zip([0] + bounds, bounds + [n])]

    ax = mm_axes(fig, x0 + lab_w, y_mat + mat_h + 2, grid_w, bars_h)
    ymax = float(shown.max()) * 1.34
    for s, e, cidx in spans:
        ax.add_patch(Rectangle((s - 0.5, 0), e - s + 1, ymax,
                               fc=BAND[cidx], ec="none", zorder=0))
    for i, k in enumerate(combos):
        ax.add_patch(Rectangle((i - 0.35, 0), 0.7, shown.iloc[i],
                               fc=bar_colour(k), ec="none"))
        ax.text(i, shown.iloc[i] + ymax * 0.02, f"{shown.iloc[i]:,}",
                rotation=90, ha="center", va="bottom", fontsize=FS_MIN)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, ymax)
    ax.set_xticks([])
    ax.set_ylabel("genes in intersection", fontsize=6)
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    mx = mm_axes(fig, x0 + lab_w, y_mat, grid_w, mat_h)
    for s, e, cidx in spans:
        mx.add_patch(Rectangle((s - 0.5, -0.7), e - s + 1, len(conds) + 0.4,
                               fc=BAND[cidx], ec="none", zorder=0))
    for i, k in enumerate(combos):
        if is_core(k):
            mx.add_patch(Rectangle((i - 0.45, -0.65), 0.9, len(conds) - 0.35,
                                   fill=False, ec="black", lw=0.7,
                                   clip_on=False, zorder=3))
    for ci, c in enumerate(conds):
        v, arm, hpi = c
        y = row_y[c]
        mx.text(HPI_X, y, str(hpi), ha="center", va="center",
                fontsize=FS_MIN, color="0.25")
        mx.text(JAKI_X, y, "−" if arm == "untreated" else "+",
                ha="center", va="center", fontsize=6, color="0.25")
        for i, k in enumerate(combos):
            if (v, arm) not in k or cells[i][ci] == 0:
                mx.plot(i, y, "o", ms=2.0, mfc="white", mec="0.75",
                        mew=0.4, zorder=2)
            else:
                mx.plot(i, y, "o", ms=3.4, zorder=2, color=VIRUS_COLORS[v])
    mx.text((HPI_X + JAKI_X) / 2, -1.6, "JAKi  hpi", ha="center",
            va="center", fontsize=FS_MIN, color="0.25")
    for i, k in enumerate(combos):
        ys = sorted(row_y[c] for ci, c in enumerate(conds)
                    if (c[0], c[1]) in k and cells[i][ci] > 0)
        if len(ys) > 1:
            mx.plot([i, i], [ys[0], ys[-1]], color="0.1", lw=0.9, zorder=1)
    for v in viruses:
        ys = [row_y[(v, arm, hpi)] for arm in ARMS for hpi in HPIS]
        mx.add_patch(Rectangle((CHIP_L, min(ys) - 0.38), CHIP_W,
                               max(ys) - min(ys) + 0.76,
                               fc=VIRUS_COLORS[v], ec="none",
                               clip_on=False, zorder=4))
        mx.text(CHIP_L + CHIP_W / 2, sum(ys) / 4, v, ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold", zorder=5)
    mx.set_xlim(-0.5, n - 0.5)
    mx.set_ylim(-0.7, len(conds) - 0.3)
    mx.set_axis_off()

    sx = mm_axes(fig, x0 + lab_w + grid_w + 3.0, y_mat, set_w, mat_h)
    for c in conds:
        v = c[0]
        sx.add_patch(Rectangle((0, row_y[c] - 0.32), cond_tot.get(c, 0), 0.64,
                               fc=VIRUS_COLORS[v], ec="none"))
    sx.set_xlim(0, max(cond_tot.values()) * 1.12)
    sx.set_ylim(-0.7, len(conds) - 0.3)
    sx.set_yticks([])
    sx.set_xlabel("genes induced", fontsize=5.6, labelpad=1)
    sx.xaxis.set_label_position("top")
    sx.xaxis.tick_top()
    sx.tick_params(labelsize=FS_MIN, length=2, pad=1)
    for sp in ("bottom", "right", "left"):
        sx.spines[sp].set_visible(False)


def main():
    set_rcparams()
    dg, sym, asg, cb = load()
    shown, counts, cond_genes, cond_tot, viruses, col_genes = assemble(dg)

    fig = plt.figure(figsize=(W * MM, H * MM))
    draw(fig, shown, counts, cond_genes, cond_tot, viruses, col_genes)
    written = save_figure(fig, FIGURES / "conservation_upset_multi" / "figs_multi_upset.pdf",
                          bbox_inches=None)
    svg = FIGURES / "conservation_upset_multi" / "figs_multi_upset.svg"
    written += save_svg(fig, svg)
    plt.close(fig)

    n_arm = {a: int(dg[dg.deg & (dg.log2fc > 0) & (dg.treatment == a)]
                    .gene_id.nunique()) for a in ARMS}
    print(f"sets                         : 10 (virus x arm, hpi pooled); "
          f"rows = 20 conditions (viruses ordered {', '.join(viruses)})")
    print(f"genes induced anywhere       : {int(counts.sum())} "
          f"(untreated arm {n_arm['untreated']}, +Ruxo arm {n_arm['Ruxo']})")
    print(f"non-empty intersections      : {len(counts)}; shown "
          f"{len(shown)} covering {int(shown.sum())} genes "
          f"({100 * shown.sum() / counts.sum():.0f}%)")
    tail = counts[counts < MIN_GENES]
    print(f"not shown (caption)          : {int(tail.sum())} genes in "
          f"{len(tail)} intersections (< {MIN_GENES} genes)")
    print("column classes               :")
    for cidx, name in enumerate(CLASSES):
        sub = shown[[column_class(k) == cidx for k in shown.index]]
        print(f"   {name:28s} {len(sub):3d} columns, {int(sub.sum()):5d} genes")
    core = shown[[is_core(k) for k in shown.index]]
    print(f"framed core columns          : {len(core)} columns, "
          f"{int(core.sum())} genes untreated in all five viruses")
    print("largest intersections        :")
    for k, val in counts.head(5).items():
        print(f"   {val:5d}  "
              + ", ".join(f"{c[0]} {ARM_LABEL[c[1]]}" for c in sorted(k)))
    print(f"page {W:.0f} x {H:.0f} mm")
    for q in written:
        print("wrote", q)


if __name__ == "__main__":
    main()
