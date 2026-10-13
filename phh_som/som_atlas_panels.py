#!/usr/bin/env python3
import argparse
import os
from functools import lru_cache

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import (Circle, FancyBboxPatch, PathPatch,
                                Polygon, Rectangle, Wedge)

from pipeline import MM, VIRUS_ORDER, save_figure, set_rcparams
from plot_som_map import (COUNT_CMAP, FAN_R0, SQ3,
                                 count_norm, count_ticks)
from plot_codes_map import paper_conds
import som_figure_context

FIGDIR = som_figure_context.FIGDIR
CLASSES = som_figure_context.CLASSES
CALLOUTS = som_figure_context.CALLOUTS
PHASE_ROWS = som_figure_context.PHASE_ROWS
SHORT_CLASS = som_figure_context.SHORT_CLASS

REGION_TINTS = {
    "Interferon, ISG and antigen-presentation core": "#8E7CC3",
    "Inflammatory, stress and immediate-early": "#6FA8DC",
    "Hepatocyte metabolic and secretory": "#C2A15C",
}
REGION_SHORT = {
    "Interferon, ISG and antigen-presentation core": "Interferon / ISG / antigen core",
    "Inflammatory, stress and immediate-early": "Inflammatory, stress & immediate-early",
    "Hepatocyte metabolic and secretory": "Hepatocyte metabolic & secretory",
}

PHASES = [
    ("①", "Interferon production", ["ifn"], "#CC79A7",
     "IFNL1–4, IFNB1 · unit 756"),
    ("②", "Antiviral & immune effectors", ["sense", "isgf3", "antigen",
     "inflam"], "#E69F00",
     "ISG core, antigen presentation · 728 / 726 / 725 / 727 / 668"),
    ("③", "Metabolic repression", ["metab"], "#8C6D46",
     "ketogenesis, CYP450, conjugation · 59 / 32 / 34 region"),
]

ORPHAN_PHASE = ("④", "Orphan late wave", 640, "#009E73",
                "uncharacterised transcripts · unit 640")

DOT_PT2 = 0.7

COND_DOTS = [("untreated", 24, "untreated 24 hpi", "#2166AC"),
             ("untreated", 72, "untreated 72 hpi", "#B2182B"),
             ("Ruxo", 24, "ruxolitinib 24 hpi", "#92C5DE"),
             ("Ruxo", 72, "ruxolitinib 72 hpi", "#F4A582")]

PHASE_GENES = [("①", ["IFNL1", "IFNB1"]),
               ("②", ["ISG15", "MX1", "STAT1"]),
               ("③", ["HMGCS2", "CYP3A4", "ADH1B"]),
               ("④", ["ENSG00000235862", "ENSG00000260914",
                      "ENSG00000288894", "ENSG00000295170"])]


def label(fig, x, y, text):
    fig.text(x, y, text, fontsize=8, fontweight="bold", ha="left", va="top")


def panel_clarity_atlas(ax, ctx, side_pad=30.0, fan_origin=None):
    grid, cb = ctx.grid, ctx.cb
    conds = list(cb.columns)
    colors = som_figure_context.virus_colors()
    k, lo, hi = som_figure_context.fan_scale(cb)

    for r in ctx.regions:
        path = som_figure_context.territory_path(r["units"], grid, radius=1.75)
        tint = REGION_TINTS[r["name"]]
        ax.add_patch(PathPatch(path, fc=tint, ec="none", alpha=0.14, zorder=1,
                               joinstyle="round"))
        ax.add_patch(PathPatch(path, fc="none", ec=tint, lw=0.7, alpha=0.85,
                               zorder=3, joinstyle="round"))

    for n in grid.centres:
        supported = n in ctx.sup
        key = ctx.cls(n) if supported else None
        fc = som_figure_context.lighten(CLASSES[key][1], 0.68) if key else "#F4F5F6"
        x, y = grid.hexagon(ax, n, radius=0.98, fc=fc, ec="#DDE0E3", lw=0.12,
                            zorder=2)
        if supported:
            som_figure_context.fan_glyph(ax, x, y, cb.loc[n].to_numpy(), k=k, conds=conds,
                          colors=colors, lw=0.10, ring=False)

    unsupported_rings_spread(ax, grid)
    by_row = {}
    for n in CALLOUTS:
        x, y = grid.centres[n]
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.05), closed=True, fc="none",
                             ec="#1F2328", lw=0.55, zorder=8))
        by_row.setdefault(round(y / 1.5), []).append((x, n))
    for members in by_row.values():
        members.sort()
        for i, (x, n) in enumerate(members):
            y = grid.centres[n][1]
            up = i % 2 == 1
            ax.text(x, y + (1.55 if up else -1.35), str(n), ha="center",
                    va="bottom" if up else "top", fontsize=3.8,
                    fontweight="bold", color="#1F2328", zorder=9,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))

    grid.frame(ax)
    ax.set_xlim(grid.xlim[0] - side_pad, grid.xlim[1] + side_pad)
    ax.set_ylim(grid.ylim[0] - 1.0, grid.ylim[1] + 12.0)
    if fan_origin is None:
        fan_origin = (grid.cx, grid.ylim[1] + 6.4)
    som_figure_context.fan_key(ax, k, lo, hi, conds, colors,
                origin=fan_origin, magnify=3.4)
    atlas_left_key(ax, ctx, grid, side_pad)
    atlas_right_list(ax, ctx, grid, side_pad)


def unsupported_rings_spread(ax, grid):
    for i, n in enumerate(sorted(som_figure_context.NAMED_UNSUPPORTED)):
        if n not in grid.centres:
            continue
        x, y = grid.centres[n]
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.02), closed=True, fc="none",
                             ec="#3F4550", lw=0.7, ls=(0, (1.5, 1.2)),
                             zorder=7))
        up = i % 2 == 0
        for m in CALLOUTS:
            mx, my = grid.centres[m]
            if abs(mx - x) < 1.8 and 0.4 < my - y < 2.6:
                up = False
            if abs(mx - x) < 1.8 and -2.6 < my - y < -0.4:
                up = True
        ax.text(x, y + (1.25 if up else -1.25), str(n), ha="center",
                va="bottom" if up else "top", fontsize=4.0, color="#3F4550",
                zorder=8)


def atlas_left_key(ax, ctx, grid, side_pad):
    x = grid.xlim[0] - side_pad + 0.6
    y = grid.ylim[1] - 0.5
    total = int(ctx.n_genes.reindex(ctx.sup).sum())
    n_ind = int((ctx.n_up >= 1).sum())
    n_all = int((ctx.n_up == 5).sum())
    n_one = int((ctx.n_up == 1).sum())
    ax.text(x, y, f"{len(ctx.sup)} supported programmes · {total:,} genes",
            fontsize=5.2, fontweight="bold", color="#1F2328", ha="left",
            va="top")
    y -= 2.1
    ax.text(x, y, f"{n_ind} induced by ≥ 1 virus · {n_all} by all five · "
                  f"{n_one} virus-restricted", fontsize=4.2, color="0.35",
            ha="left", va="top")
    y -= 2.3
    for key in CLASSES:
        ax.add_patch(Rectangle((x, y - 0.85), 1.5, 0.95, fc=CLASSES[key][1],
                               ec="white", lw=0.3))
        ax.text(x + 2.1, y - 0.38, CLASSES[key][0], fontsize=4.2, va="center",
                ha="left", color="#1F2328")
        y -= 1.25
    y -= 2.0
    ax.text(x, y, "Regions (tinted on the map)", fontsize=4.6,
            fontweight="bold", color="#3A4048", ha="left", va="top")
    y -= 1.9
    for r in ctx.regions:
        tint = REGION_TINTS[r["name"]]
        ax.add_patch(Rectangle((x, y - 0.95), 1.5, 1.0, fc=tint, ec=tint,
                               lw=0.5, alpha=0.45))
        ax.text(x + 2.1, y - 0.1, REGION_SHORT[r["name"]], fontsize=4.2,
                va="top", ha="left", color="#1F2328")
        ax.text(x + 2.1, y - 1.15, f"{len(r['units'])} programmes · "
                                   f"{r['n_genes']:,} genes", fontsize=3.9,
                va="top", ha="left", color="0.40", fontstyle="italic")
        y -= 2.9


def atlas_right_list(ax, ctx, grid, side_pad):
    import textwrap

    x_num = grid.xlim[1] + 2.4
    x_txt = x_num + 3.6
    width = 52
    by_region = {}
    for n, (head, genes) in CALLOUTS.items():
        by_region.setdefault(som_figure_context.region_of(ctx, n) or "other", []).append(
            (n, head, genes))
    for v in by_region.values():
        v.sort(key=lambda t: -grid.centres[t[0]][1])

    order = [r["name"] for r in ctx.regions] + ["other"]
    y = grid.ylim[1] + 6.0
    for name in order:
        entries = by_region.get(name, [])
        if not entries:
            continue
        if name != "other":
            ax.text(x_num, y, REGION_SHORT[name], fontsize=4.4,
                    fontweight="bold", color=REGION_TINTS[name], ha="left",
                    va="top")
            y -= 1.7
        for n, head, genes in entries:
            tag = som_figure_context.RESTRICTED.get(n)
            sub = genes if not tag else f"{genes} · {tag}-restricted"
            colour = "0.25" if not tag else "#B4231C"
            ax.text(x_num, y, str(n), fontsize=4.1, fontweight="bold",
                    color="#1F2328", ha="left", va="top")
            lines = textwrap.wrap(f"{head} — {sub}", width=width)
            for li, line in enumerate(lines):
                ax.text(x_txt if li == 0 else x_txt + 1.6, y, line,
                        fontsize=3.9, color=colour, ha="left", va="top")
                y -= 1.3
            y -= 0.25
        y -= 0.7
    below = "  ·  ".join(f"{n} {t}" for n, t in
                         sorted(som_figure_context.NAMED_UNSUPPORTED.items()))
    for li, line in enumerate(textwrap.wrap(
            "ringed, below the support threshold:  " + below, width=width + 4)):
        ax.text(x_num if li == 0 else x_num + 1.6, y, line, fontsize=3.8,
                color="0.40", ha="left", va="top")
        y -= 1.25
    y -= 0.5
    ax.text(x_num, y,
            "blank hexagon = below the support threshold\n"
            "tint + solid boundary = spatial region",
            fontsize=3.8, color="0.40", ha="left", va="top")


def phase_trajectory(ctx, keys):
    units = [n for key in keys for n in ctx.members_of(key)]
    return unit_set_trajectory(ctx, units)


def unit_set_trajectory(ctx, units):
    z24 = ctx.cb.loc[units, [f"{v}_untreated24"
                             for v in VIRUS_ORDER]].to_numpy().mean()
    z72 = ctx.cb.loc[units, [f"{v}_untreated72"
                             for v in VIRUS_ORDER]].to_numpy().mean()
    return float(z24), float(z72)


def panel_phases(ax, ctx, fontsize=4.9, label_bbox=False, compact=False):
    xs = np.linspace(0, 72, 200)
    curves = [(sym, colour, phase_trajectory(ctx, keys))
              for sym, _, keys, colour, _ in PHASES]
    o_sym, _, o_unit, o_colour, _ = ORPHAN_PHASE
    curves.append((o_sym, o_colour, unit_set_trajectory(ctx, [o_unit])))
    for sym, colour, (z24, z72) in curves:
        fit = np.polyfit([0, 24, 72], [0.0, z24, z72], 2)
        ys = np.polyval(fit, xs)
        ax.fill_between(xs, 0, ys, color=colour, alpha=0.18, lw=0, zorder=2)
        ax.plot(xs, ys, color=colour, lw=1.4, zorder=3)
        ax.plot([24, 72], [z24, z72], "o", color=colour, ms=3.0, mec="white",
                mew=0.4, zorder=4)
    for x in (24, 72):
        ax.axvline(x, color="0.75", lw=0.5, ls=(0, (2.0, 1.6)), zorder=1)
    ax.axhline(0, color="0.4", lw=0.6, zorder=1)

    ax.set_ylim(-6.5, 16.5)
    lab_box = dict(fc="white", ec="none", alpha=0.85, pad=0.8) \
        if label_bbox else None
    ax.text(15, 13.9, "① Interferon production\nIFNL1–4, IFNB1 · "
            "unit 756", fontsize=fontsize, color=PHASES[0][3], ha="center",
            va="bottom", fontweight="bold", bbox=lab_box)
    if compact:
        ax.text(24, 4.2, "② Antiviral & immune effectors\n"
                "728 / 726 / 725 / 727 / 668",
                fontsize=fontsize - 0.4, color=PHASES[1][3], ha="center",
                va="bottom", fontweight="bold", bbox=lab_box)
    else:
        ax.text(25, 6.0, "② Antiviral & immune effector programmes\n"
                "ISG core, antigen presentation\nunits 728 / 726 / 725 / 727 / 668",
                fontsize=fontsize - 0.4, color=PHASES[1][3], ha="center",
                va="bottom", fontweight="bold", bbox=lab_box)
    o_z24, o_z72 = curves[-1][2]
    o_text = (f"④ Orphan late wave · unit {o_unit}" if compact else
              f"④ Orphan late wave\nuncharacterised transcripts · unit {o_unit}")
    ax.text(70, o_z72 + 0.7, o_text, fontsize=fontsize - 0.4,
            color=o_colour, ha="right", va="bottom", fontweight="bold",
            bbox=lab_box)
    ax.text(70, -1.9, "③ Hepatocyte metabolic repression\n"
            "ketogenesis, CYP450, conjugation\n59 / 32 / 34 region — late",
            fontsize=fontsize - 0.4, color=PHASES[2][3], ha="right",
            va="top", fontweight="bold", bbox=lab_box)

    med_i = 100 * float(np.median(ctx.share24["induced"]))
    med_r = 100 * float(np.median(ctx.share24["repressed"]))
    ax.text(0.02, 0.03, f"median {med_i:.0f}% of induced-programme genes "
            f"already DE at 24 hpi\nvs {med_r:.0f}% for repressed programmes",
            transform=ax.transAxes, fontsize=fontsize - 0.3, color="#1F2328",
            ha="left", va="bottom",
            bbox=dict(fc="white", ec="0.8", lw=0.4, pad=2.0))

    ax.set_xlim(-4, 78)
    ax.set_xticks([0, 24, 72])
    ax.set_xticklabels(["Mock", "24 hpi", "72 hpi"], fontsize=5.0)
    ax.set_ylabel("mean codebook weight (z)\nuntreated, 5-virus mean",
                  fontsize=4.8)
    ax.tick_params(labelsize=4.6, length=1.3, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)


def retention_stats(ctx):
    r_ifn = [float(ctx.mod.loc[756, f"retain24_{v}"]) for v in VIRUS_ORDER]
    r_isg = [float(ctx.mod.loc[n, f"retain24_{v}"])
             for n in ctx.members_of("isgf3") for v in VIRUS_ORDER
             if ctx.cb.loc[n, f"{v}_untreated24"] > 1]
    return min(r_ifn), max(r_ifn), float(np.median(r_isg))


def panel_class_condition_dots(ax, ctx):
    gl = ctx.gene_long
    rows = [(key, ph) for key, ph in PHASE_ROWS]
    offsets = np.linspace(-0.27, 0.27, len(COND_DOTS))
    for i, (key, ph) in enumerate(rows):
        y = len(rows) - 1 - i
        for (tr, hpi, lab, colour), dy in zip(COND_DOTS, offsets):
            sel = gl[(gl.cls == key) & (gl.treatment == tr) & (gl.hpi == hpi)]
            for v in VIRUS_ORDER:
                vm = float(sel[sel.virus == v].log2fc.mean())
                ax.plot([vm, vm], [y + dy - 0.075, y + dy + 0.075],
                        color="0.78", lw=0.4, zorder=2)
            mu = float(sel.log2fc.mean())
            se = float(sel.log2fc.std(ddof=1) / np.sqrt(len(sel)))
            ax.errorbar([mu], [y + dy], xerr=[[1.96 * se], [1.96 * se]],
                        fmt="o", color=colour, ms=3.0, mec="white", mew=0.3,
                        elinewidth=0.6, capsize=0, zorder=3)
        ax.text(-0.28, y, f"{ph} {SHORT_CLASS[key]}", fontsize=4.6,
                color=CLASSES[key][1], ha="right", va="center",
                fontweight="bold", transform=ax.get_yaxis_transform())
    ax.axvline(0, color="0.35", lw=0.6, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("mean log₂FC vs Mock, member genes × 5 viruses "
                  "(dot = mean, bar = 95% CI; tick = one virus)",
                  fontsize=4.6, labelpad=1.5)
    ax.tick_params(labelsize=4.4, length=1.3, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec="none")
               for _, _, _, c in COND_DOTS]
    ax.legend(handles, [lab for _, _, lab, _ in COND_DOTS], fontsize=4.0,
              frameon=False, ncol=2, handlelength=0.8, handletextpad=0.4,
              columnspacing=0.8, loc="lower right", borderaxespad=0.1)

    ifn_lo, ifn_hi, isg_med = retention_stats(ctx)
    ax.set_title(f"IFN ligands retain {100*ifn_lo:.0f}–{100*ifn_hi:.0f}% of "
                 f"induction with JAK–STAT blocked;\nthe ISGF3 tier is "
                 f"abolished (median {100*isg_med:.0f}% retained)",
                 fontsize=4.6, pad=3)


def panel_conservation_hbars(ax, ctx):
    ks = list(range(1, 6))
    on = [int((ctx.up_on == k).sum()) for k in ks]
    tot = [int((ctx.gene_up == k).sum()) for k in ks]
    off = [t - o for t, o in zip(tot, on)]
    dn5 = int((ctx.gene_dn == 5).sum())

    ys = [6 - k for k in ks]
    ax.barh(ys, on, color="#472A78", ec="white", lw=0.3, height=0.72,
            label="on the 382-ISG panel")
    ax.barh(ys, off, left=on, color="#B9A5D1", ec="white", lw=0.3, height=0.72,
            label="off the panel")
    ax.barh([5.6], [dn5], color="#2E6DA4", ec="white", lw=0.3, height=0.72)
    for y, t, o in zip(ys, tot, on):
        ax.text(t + 18, y, f"{t:,}", ha="left", va="center", fontsize=4.3)
        if t - o:
            ax.text(o + (t - o) / 2, y, f"{t - o}", ha="center", va="center",
                    fontsize=3.8, color="#3A2E52")
    ax.text(dn5 + 18, 5.6, f"{dn5}", ha="left", va="center", fontsize=4.3,
            color="#2E6DA4")
    ax.set_yticks([5.6] + ys)
    ax.set_yticklabels(["repressed\nby all five", "1 virus", "2 viruses",
                        "3 viruses", "4 viruses", "5 viruses"], fontsize=4.4)
    ax.set_xlabel("genes (untreated)", fontsize=4.7, labelpad=1.5)
    ax.tick_params(labelsize=4.4, length=1.3, pad=1)
    ax.set_xlim(0, max(tot) * 1.16)
    ax.set_ylim(-0.6, 6.4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=4.1, frameon=False, loc="lower right", handlelength=0.9,
              handletextpad=0.4, labelspacing=0.3)


def share24_per_neuron(ctx):
    ut = ctx.gene_long[ctx.gene_long.treatment == "untreated"]
    s24 = set(zip(ut[(ut.hpi == 24) & ut.deg].gene_id,
                  ut[(ut.hpi == 24) & ut.deg].virus))
    s72 = set(zip(ut[(ut.hpi == 72) & ut.deg].gene_id,
                  ut[(ut.hpi == 72) & ut.deg].virus))
    by_neuron = ctx.assign.groupby("neuron")["gene_id"].apply(list)
    out = {}
    for n in ctx.sup:
        mem, tot, hit = by_neuron.get(n, []), 0, 0
        for v in VIRUS_ORDER:
            m = [x for x in mem if (x, v) in s72]
            tot += len(m)
            hit += sum((x, v) in s24 for x in m)
        out[n] = hit / tot if tot else np.nan
    return out


def panel_early_late(ax, ctx, shares):
    rows = [(key, ph) for key, ph in PHASE_ROWS]
    rng = np.random.default_rng(143)
    for i, (key, ph) in enumerate(rows):
        vals = np.array([shares[n] for n in ctx.members_of(key)
                         if not np.isnan(shares[n])])
        med = float(np.median(vals))
        ax.bar([i], [100 * med], color=CLASSES[key][1], ec="white", lw=0.3,
               width=0.70, zorder=3)
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.18, 0.18, len(vals)),
                   100 * vals, s=3.0, fc="white", ec=CLASSES[key][1], lw=0.4,
                   zorder=4)
        ax.text(i, 100 * med + 3, f"{100 * med:.0f}%", ha="center",
                va="bottom", fontsize=4.4, color="#1F2328")
    med_i = 100 * float(np.median(ctx.share24["induced"]))
    med_r = 100 * float(np.median(ctx.share24["repressed"]))
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{ph} {SHORT_CLASS[k]}" for k, ph in rows],
                       fontsize=4.0, rotation=28, ha="right")
    for tick, (key, _) in zip(ax.get_xticklabels(), rows):
        tick.set_color(CLASSES[key][1])
    ax.set_ylabel("% of a programme's 72-hpi\nDEGs already DE at 24 hpi",
                  fontsize=4.6)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=4.4, length=1.3, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"induction is early, repression is late\nprogramme median "
                 f"{med_i:.0f}% induced vs {med_r:.0f}% repressed",
                 fontsize=4.5, pad=3)
    ax.text(0.97, 0.97, "dots = individual programmes",
            transform=ax.transAxes, fontsize=3.8, color="0.45", ha="right",
            va="top")


def panel_gene_conservation_terrain(ax, ctx, compact=False):
    grid = ctx.grid
    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    up4 = ctx.gene_up[ctx.gene_up >= 4]
    up5 = ctx.gene_up[ctx.gene_up == 5]
    counts = neuron_of.reindex(up4.index).value_counts()
    marked = set(neuron_of.reindex(up5.index).dropna().astype(int))

    n4, n5 = int(len(up4)), int(len(up5))
    off4 = n4 - int((ctx.up_on >= 4).sum())
    lnc4 = int((ctx.up_lnc >= 4).sum())
    dn5 = int((ctx.gene_dn == 5).sum())
    top = int(counts.max())
    sup = set(ctx.sup)
    n_ind = int((ctx.n_up >= 1).sum())
    in_ind = [n for n in counts.index if n in sup and ctx.n_up.get(n, 0) >= 1]
    n_below = len(set(counts.index) - sup)

    cmap = mpl.colormaps["BuGn"]
    norm = mpl.colors.Normalize(vmin=0, vmax=top)
    for n, (x, y) in grid.centres.items():
        c = int(counts.get(n, 0))
        fc = cmap(norm(c)) if c else "#F2F3F4"
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 0.98), closed=True, fc=fc,
                             ec="#DDE0E3", lw=0.12, zorder=2))
    for n in marked:
        x, y = grid.centres[n]
        ax.plot(x, y, "o", ms=1.8, mfc="white", mec="#1F2328", mew=0.35,
                zorder=5)
    for r in ctx.regions:
        som_figure_context.draw_territory(ax, r["units"], grid, REGION_TINTS[r["name"]],
                           lw=0.8, radius=1.75, z=6)
    grid.frame(ax)
    ax.set_ylim(grid.ylim[0] - (9.0 if compact else 12.0),
                grid.ylim[1] + (1.6 if compact else 2.6))
    if not compact:
        ax.text(grid.xlim[1] - 0.5, grid.ylim[1] + 0.7,
                "the IFN / ISG core holds the conserved genes",
                fontsize=4.4, ha="right", va="bottom", fontstyle="italic",
                color=REGION_TINTS[
                    "Interferon, ISG and antigen-presentation core"],
                zorder=9, bbox=dict(fc="white", ec="none", alpha=0.85,
                                    pad=0.6))

    x0, y = grid.xlim[0] + 0.4, grid.ylim[0] - 2.0
    if compact:
        lines = [f"{n4} genes ≥ 4/5 viruses · ◯ = {n5} by all five",
                 f"{off4} off the ISG panel · {lnc4} lncRNA · "
                 f"{dn5} repressed by all five"]
    else:
        lines = [f"{n4} genes induced by ≥ 4 of 5 viruses · {n5} by all "
                 f"five (white dot)",
                 f"{off4} of the {n4} off the 382-ISG panel · {lnc4} lncRNA · "
                 f"{dn5} repressed by all five",
                 f"spread over {len(in_ind)} of the {n_ind} induced programmes"
                 + (f" (+ {n_below} neurons below threshold)"
                    if n_below else "")]
    for line in lines:
        ax.text(x0, y, line, fontsize=4.2, ha="left", va="top",
                color="#1F2328")
        y -= 1.55
    cax = ax.inset_axes([0.30, 0.016 if compact else 0.048, 0.44, 0.038])
    bar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                    orientation="horizontal",
                                    ticks=[0, top])
    bar.ax.tick_params(labelsize=4.0, length=1.2, pad=1)
    bar.outline.set_linewidth(0.4)
    cax.set_xlabel("member genes induced by ≥ 4 of 5 viruses, per neuron",
                   fontsize=4.2, labelpad=1.5)
    ax.set_title(f"where the {n4} conserved genes live — gene level",
                 fontsize=4.8, pad=3)


def panel_hybrid_codes_map(ax, ctx, side_pad=30.0, pad_scale=1.6):
    grid = ctx.grid
    geom = dict(som_figure_context.panel_geometry(grid))
    base_range = geom["x_range"]
    geom["side_pad"] = side_pad
    geom["x_range"] = grid.xlim[1] - grid.xlim[0] + 2 * side_pad
    geom["m"] *= geom["x_range"] / base_range
    geom["pad_top"] *= pad_scale

    for r in ctx.regions:
        path = som_figure_context.territory_path(r["units"], grid, radius=1.75)
        ax.add_patch(PathPatch(path, fc=REGION_TINTS[r["name"]], ec="none",
                               alpha=0.14, zorder=0.5, joinstyle="round"))

    som_figure_context.draw_codes_map(ax, ctx.cb, ctx.assign, ctx.stats, grid, geom,
                       legends=False)
    counts = ctx.stats["n_genes"]
    cnorm = count_norm(counts)
    b = cnorm.boundaries
    x0 = grid.xlim[0] - side_pad + 0.6
    y_key = grid.ylim[1] + 8.5
    ax.text(x0, y_key + 1.9, "# genes per neuron", fontsize=4.4, ha="left",
            va="center")
    w = 2.0
    for i in range(len(b) - 1):
        ax.add_patch(Rectangle((x0 + i * w, y_key - 0.75), w, 1.5,
                               fc=COUNT_CMAP(cnorm((b[i] + b[i + 1]) / 2)),
                               ec="0.3", lw=0.3))
    for v in count_ticks(counts):
        fx = (v - b[0]) / (b[-1] - b[0])
        ax.text(x0 + fx * (len(b) - 1) * w, y_key - 1.7, str(v), ha="center",
                va="top", fontsize=4.0)
    conds = list(ctx.cb.columns)
    colors = som_figure_context.virus_colors()
    k, lo, hi = som_figure_context.fan_scale(ctx.cb)
    som_figure_context.fan_key(ax, k, lo, hi, conds, colors,
                origin=(grid.cx, grid.ylim[1] + 6.4), magnify=3.4)

    for r in ctx.regions:
        path = som_figure_context.territory_path(r["units"], grid, radius=1.75)
        ax.add_patch(PathPatch(path, fc="none", ec=REGION_TINTS[r["name"]],
                               lw=0.8, alpha=0.9, zorder=7, joinstyle="round"))
    for key in CLASSES:
        members = ctx.members_of(key)
        if members:
            som_figure_context.draw_territory(ax, members, grid, CLASSES[key][1], lw=0.6,
                               radius=1.0, z=7)

    unsupported_rings_spread(ax, grid)
    by_row = {}
    for n in CALLOUTS:
        x, y = grid.centres[n]
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.05), closed=True, fc="none",
                             ec="#1F2328", lw=0.55, zorder=8))
        by_row.setdefault(round(y / 1.5), []).append((x, n))
    for members in by_row.values():
        members.sort()
        for i, (x, n) in enumerate(members):
            y = grid.centres[n][1]
            up = i % 2 == 1
            ax.text(x, y + (1.55 if up else -1.35), str(n), ha="center",
                    va="bottom" if up else "top", fontsize=3.8,
                    fontweight="bold", color="#1F2328", zorder=9,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))

    ax.set_ylim(grid.ylim[0] - 1.0, grid.ylim[1] + geom["pad_top"])
    atlas_left_key(ax, ctx, grid, side_pad)
    atlas_right_list(ax, ctx, grid, side_pad)


def umatrix_values(ctx):
    ids = sorted(ctx.grid.centres)
    pts = np.array([ctx.grid.centres[n] for n in ids])
    Z = ctx.cb.loc[ids].to_numpy(dtype=float)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    nbr = (d2 > 0) & (d2 < 1.9 ** 2)
    out = {}
    for i, n in enumerate(ids):
        m = nbr[i]
        dz = np.sqrt(((Z[i] - Z[m]) ** 2).sum(-1))
        out[n] = float(dz.mean()) if m.any() else np.nan
    return out


UMATRIX_LANDMARKS = [(756, "interferon ligands"), (648, "stress response"),
                     (59, "ketogenesis")]


def panel_umatrix(ax, ctx):
    grid = ctx.grid
    u = umatrix_values(ctx)
    vals = np.array(list(u.values()))
    norm = mpl.colors.Normalize(vmin=float(np.nanpercentile(vals, 2)),
                                vmax=float(np.nanpercentile(vals, 98)))
    cmap = mpl.colormaps["magma"]
    for n, (x, y) in grid.centres.items():
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 0.98), closed=True,
                             fc=cmap(norm(u[n])), ec="white", lw=0.05,
                             zorder=2))
    for r in ctx.regions:
        som_figure_context.draw_territory(ax, r["units"], grid, REGION_TINTS[r["name"]],
                           lw=0.8, radius=1.75, z=6)
    for n, _ in UMATRIX_LANDMARKS:
        x, y = grid.centres[n]
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.05), closed=True, fc="none",
                             ec="#1F2328", lw=0.6, zorder=7))
        ax.text(x, y - 1.35, str(n), ha="center", va="top", fontsize=4.0,
                fontweight="bold", color="#1F2328", zorder=8,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
    grid.frame(ax)
    ax.set_ylim(grid.ylim[0] - 9.5, grid.ylim[1] + 1.6)

    x0, base = grid.xlim[0] + 0.4, grid.ylim[0] - 2.2
    marks = "  ·  ".join(f"{n} {t}" for n, t in UMATRIX_LANDMARKS)
    ax.text(x0, base, "rings: " + marks, fontsize=4.4, ha="left", va="top")
    region_key = {"Interferon, ISG and antigen-presentation core":
                  "IFN / ISG / antigen core",
                  "Inflammatory, stress and immediate-early":
                  "inflammatory & stress",
                  "Hepatocyte metabolic and secretory":
                  "metabolic & secretory"}
    y = base - 2.0
    for r in ctx.regions:
        ax.text(x0, y, f"— {region_key[r['name']]}", fontsize=4.2,
                color=REGION_TINTS[r["name"]], ha="left", va="top",
                fontweight="bold")
        y -= 1.55
    cax = ax.inset_axes([0.64, 0.030, 0.34, 0.045])
    bar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                    orientation="horizontal",
                                    ticks=[norm.vmin, norm.vmax])
    bar.ax.set_xticklabels(["valley\n(similar)", "ridge\n(different)"],
                           fontsize=3.9)
    bar.ax.tick_params(length=1.2, pad=1)
    bar.outline.set_linewidth(0.4)
    cax.set_title("mean distance to neighbouring neurons", fontsize=4.2, pad=2)


CITED_RESTRICTED = {648: ["ATF3", "BBC3", "PPP1R15A"], 32: []}


def restricted_heatmap(ax, ctx, neuron, per_unit, norm, header_size=4.6,
                       isg=False):
    gl = ctx.gene_long[ctx.gene_long.treatment == "untreated"]
    genes = list(ctx.assign[ctx.assign.neuron == neuron].gene_id)
    sub = gl[gl.gene_id.isin(genes)]
    rank = sub.groupby("gene_id").log2fc.apply(lambda v: v.abs().max())
    cited = ctx.sym.symbol[ctx.sym.symbol.isin(CITED_RESTRICTED[neuron])]
    cited = [g for g in cited.index if g in rank.index]
    rest = rank.drop(index=cited).sort_values(ascending=False)
    keep = cited + list(rest.head(per_unit - len(cited)).index)
    rows, labels, dots = [], [], []
    for gid in keep:
        r, d = [], []
        for hpi in (24, 72):
            for v in VIRUS_ORDER:
                m = sub[(sub.gene_id == gid) & (sub.virus == v)
                        & (sub.hpi == hpi)]
                r.append(float(m.log2fc.iloc[0]) if len(m) else np.nan)
                d.append(bool(m.deg.iloc[0]) if len(m) else False)
        rows.append(r)
        dots.append(d)
        labels.append(str(sub[sub.gene_id == gid].symbol.iloc[0]))
    M = np.array(rows)
    ax.imshow(np.clip(M, -4, 4), cmap="RdBu_r", norm=norm, aspect="auto")
    for i, d in enumerate(dots):
        for j, hit in enumerate(d):
            if hit:
                ax.plot(j, i, ".", color="#1F2328", ms=1.1, zorder=5)
    ax.axvline(4.5, color="white", lw=1.4)
    ax.set_xticks(range(10))
    ax.set_xticklabels(VIRUS_ORDER * 2, fontsize=3.9, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=4.0, fontstyle="italic")
    if isg:
        for lab, gid in zip(ax.get_yticklabels(), keep):
            sym = ctx.sym["symbol"].get(gid)
            on = bool(sym in ctx.isg_panel) if isinstance(sym, str) else False
            lab.set_color("#472A78" if on else "0.55")
    ax.tick_params(length=1.0, pad=1.2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.text(2.0, -0.85, "24 hpi", ha="center", va="bottom",
            fontsize=header_size, fontweight="bold")
    ax.text(7.0, -0.85, "72 hpi", ha="center", va="bottom",
            fontsize=header_size, fontweight="bold")
    return len(genes)


def panel_restricted_pair(fig, slots, ctx, cbar_rect=None, title_size=4.4,
                          short=False, per_unit=12, isg=False,
                          title_counts=True):
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    heads = [(648, "WNV-restricted · integrated stress response"),
             (32, "DENV-restricted · hepatic metabolic")]
    if short:
        heads = [(648, "WNV-restricted · stress (648)"),
                 (32, "DENV-restricted · metabolic (32)")]
    for slot, (n, head) in zip(slots, heads):
        ax = fig.add_subplot(slot)
        size = restricted_heatmap(ax, ctx, n, per_unit if n == 648 else 8,
                                  norm, header_size=title_size + 0.2,
                                  isg=isg)
        shown = min(size, per_unit if n == 648 else 8)
        if short:
            if title_counts:
                head = f"{head} · {size} genes"
            ax.set_title(head, fontsize=title_size, color="#B4231C", pad=13)
        else:
            ax.set_title(f"{head}\nunit {n}, {size} genes"
                         + (f" — Results-named genes + strongest, {shown} shown"
                            if shown < size else ""),
                         fontsize=title_size, color="#B4231C", pad=13)
    if cbar_rect is not None:
        cax = fig.add_axes(cbar_rect)
        bar = mpl.colorbar.ColorbarBase(cax, cmap="RdBu_r", norm=norm,
                                        ticks=[-4, 0, 4], extend="both")
        cax.tick_params(labelsize=4.0, length=1.2, pad=1)
        bar.outline.set_linewidth(0.4)
        fig.text(cbar_rect[0] - 0.006, cbar_rect[1] + cbar_rect[3] / 2,
                 "untreated log₂FC vs Mock · dot = DE call", fontsize=4.0,
                 color="0.25", rotation=90, ha="center", va="center")
    return norm


def panel_phase_examples(fig, slot, ctx):
    dg = pd.read_parquet(som_figure_context.PUBLISH / "data" / "degs.parquet")
    sym2gid = ctx.assign.set_index("symbol")["gene_id"]
    colour_of = {sym: colour for sym, _, _, colour, _ in PHASES}
    colour_of[ORPHAN_PHASE[0]] = ORPHAN_PHASE[3]
    rows, labels, colours, seps = [], [], [], []
    for sym, genes in PHASE_GENES:
        for g in genes:
            gid = sym2gid.loc[g]
            gid = gid.iloc[0] if hasattr(gid, "iloc") else gid
            r = []
            for hpi in (24, 72):
                for v in VIRUS_ORDER:
                    m = dg[(dg.gene_id == gid) & (dg.virus == v)
                           & (dg.treatment == "untreated") & (dg.hpi == hpi)]
                    r.append(float(m.log2fc.iloc[0]) if len(m) else np.nan)
            rows.append(r)
            labels.append(g)
            colours.append(colour_of[sym])
        seps.append(len(rows))
    M = np.array(rows)
    norm = TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6)
    ax = fig.add_subplot(slot)
    ax.imshow(np.clip(M, -6, 6), cmap="RdBu_r", norm=norm, aspect="auto")
    for s in seps[:-1]:
        ax.axhline(s - 0.5, color="white", lw=1.8)
    ax.axvline(4.5, color="white", lw=1.6)
    ax.set_xticks(range(10))
    ax.set_xticklabels(VIRUS_ORDER * 2, fontsize=4.2, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=4.6, fontstyle="italic")
    for lab, c in zip(ax.get_yticklabels(), colours):
        lab.set_color(c)
    ax.text(2.0, -0.9, "24 hpi", ha="center", va="bottom", fontsize=5.2,
            fontweight="bold")
    ax.text(7.0, -0.9, "72 hpi", ha="center", va="bottom", fontsize=5.2,
            fontweight="bold")
    ax.tick_params(length=1.2, pad=1.2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    for i, (sym, _) in enumerate(PHASE_GENES):
        lo = seps[i - 1] if i else 0
        trans = som_figure_context.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(-0.36, (lo + seps[i] - 1) / 2, sym, fontsize=7,
                color=colour_of[sym], ha="center", va="center",
                fontweight="bold", transform=trans)
    ax.set_title("untreated log₂FC vs Mock", fontsize=4.8, pad=3)
    return ax


def compose_v7(ctx, shares):
    fig_w, fig_h = 183.0, 244.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.015, 0.640, 0.970, 0.350])
    panel_clarity_atlas(ax_a, ctx)
    label(fig, 0.006, 0.996, "a")

    ax_b = fig.add_axes([0.055, 0.415, 0.400, 0.175])
    panel_phases(ax_b, ctx, compact=True)
    label(fig, 0.006, 0.616, "b")

    ax_c = fig.add_axes([0.610, 0.415, 0.370, 0.175])
    panel_class_condition_dots(ax_c, ctx)
    label(fig, 0.500, 0.616, "c")

    ax_d = fig.add_axes([0.070, 0.065, 0.260, 0.235])
    panel_conservation_hbars(ax_d, ctx)
    label(fig, 0.006, 0.360, "d")

    ax_e = fig.add_axes([0.430, 0.065, 0.200, 0.235])
    panel_early_late(ax_e, ctx, shares)
    label(fig, 0.360, 0.360, "e")

    gs_f = fig.add_gridspec(1, 2, left=0.700, right=0.985, top=0.280,
                            bottom=0.060, wspace=1.35)
    panel_restricted_pair(fig, [gs_f[0], gs_f[1]], ctx)
    label(fig, 0.640, 0.360, "f")

    out = FIGDIR / "fig6_v7_clarity.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v7 clarity   : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v8(ctx, shares):
    fig_w, fig_h = 183.0, 244.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.015, 0.645, 0.970, 0.345])
    panel_clarity_atlas(ax_a, ctx)
    label(fig, 0.006, 0.996, "a")

    ax_b = fig.add_axes([0.065, 0.445, 0.880, 0.165])
    panel_phases(ax_b, ctx, fontsize=5.4)
    label(fig, 0.006, 0.628, "b")

    gs_c = fig.add_gridspec(1, 1, left=0.105, right=0.430, top=0.390,
                            bottom=0.050)
    panel_phase_examples(fig, gs_c[0], ctx)
    label(fig, 0.006, 0.408, "c")

    ax_d = fig.add_axes([0.600, 0.260, 0.370, 0.125])
    panel_conservation_hbars(ax_d, ctx)
    label(fig, 0.500, 0.408, "d")

    gs_e = fig.add_gridspec(1, 2, left=0.590, right=0.970, top=0.170,
                            bottom=0.025, wspace=1.35)
    panel_restricted_pair(fig, [gs_e[0], gs_e[1]], ctx)
    label(fig, 0.500, 0.205, "e")

    out = FIGDIR / "fig6_v8_phase.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v8 phase     : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v9(ctx):
    fig_w, fig_h = 183.0, 247.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.015, 0.628, 0.970, 0.362])
    panel_hybrid_codes_map(ax_a, ctx)
    label(fig, 0.006, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.350, 0.430, 0.235])
    som_figure_context.panel_kinetic_map(ax_b, ctx)
    label(fig, 0.006, 0.604, "b")

    ax_c = fig.add_axes([0.545, 0.350, 0.430, 0.235])
    panel_umatrix(ax_c, ctx)
    label(fig, 0.510, 0.604, "c")

    ax_d = fig.add_axes([0.025, 0.035, 0.290, 0.270])
    panel_gene_conservation_terrain(ax_d, ctx)
    label(fig, 0.006, 0.325, "d")

    ax_e = fig.add_axes([0.395, 0.020, 0.255, 0.275])
    som_figure_context.panel_conservation(ax_e, ctx)
    label(fig, 0.355, 0.325, "e")

    gs_f = fig.add_gridspec(1, 2, left=0.695, right=0.940, top=0.250,
                            bottom=0.075, wspace=2.1)
    panel_restricted_pair(fig, [gs_f[0], gs_f[1]], ctx,
                          cbar_rect=[0.952, 0.095, 0.009, 0.105],
                          title_size=4.0)
    label(fig, 0.660, 0.325, "f")

    out = FIGDIR / "fig6_v9_hybrid.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v9 hybrid    : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


CLASS_SHORT = {"ifn": "IFN ligands", "sense": "Sensing / IRF3",
               "isgf3": "ISGF3 effectors", "antigen": "Antigen pres.",
               "inflam": "Inflammatory", "metab": "Metabolic",
               "tech": "rRNA/MT-RNR (tech.)"}
REGION_SHORT_V10 = {
    "Interferon, ISG and antigen-presentation core": "IFN / ISG core",
    "Inflammatory, stress and immediate-early": "inflammatory & stress",
    "Hepatocyte metabolic and secretory": "metabolic & secretory",
}
RING_RESTRICTED = "#B4231C"
RING_IFN = "#CC79A7"


def darken(hexcolour, f=0.72):
    r, g, b = mpl.colors.to_rgb(hexcolour)
    return (r * f, g * f, b * f)


def hex_edge_path(units, grid, eps=0.01):
    from matplotlib.path import Path as MplPath
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union

    u = unary_union([SPoly(som_figure_context.hex_ring(*grid.centres[int(n)], r=1.0))
                     .buffer(eps, join_style="mitre") for n in units
                     if int(n) in grid.centres]
                    ).buffer(-eps, join_style="mitre")
    verts, codes = [], []
    for g in (u.geoms if u.geom_type == "MultiPolygon" else [u]):
        for ring in [g.exterior, *g.interiors]:
            xy = np.asarray(ring.coords)
            verts += list(xy)
            codes += ([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2)
                      + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)


def draw_region_outline(ax, units, grid, colour, halo, halo_lw, core_lw, z):
    path = som_figure_context.territory_path(units, grid, radius=1.75)
    if path is None:
        return
    ax.add_patch(PathPatch(path, fc="none", ec=halo, lw=halo_lw, alpha=0.6,
                           zorder=z, joinstyle="round", capstyle="round"))
    ax.add_patch(PathPatch(path, fc="none", ec=colour, lw=core_lw,
                           alpha=0.95, zorder=z + 0.5, joinstyle="round",
                           capstyle="round"))


def panel_hybrid_codes_map_v10(ax, ctx, side_pad=30.0, right_pad=7.0,
                               pad_scale=1.6, region_halo=False,
                               left_key=None, fan_origin=None,
                               drop_rings=(), thin_regions=False,
                               class_outlines=True, marks=True):
    grid = ctx.grid
    geom = dict(som_figure_context.panel_geometry(grid))
    base_range = geom["x_range"]
    geom["side_pad"] = side_pad
    geom["x_range"] = grid.xlim[1] - grid.xlim[0] + 2 * side_pad
    geom["m"] *= geom["x_range"] / base_range
    geom["pad_top"] *= pad_scale

    for r in ctx.regions if not thin_regions else ():
        path = som_figure_context.territory_path(r["units"], grid, radius=1.75)
        ax.add_patch(PathPatch(path, fc=REGION_TINTS[r["name"]], ec="none",
                               alpha=0.22 if region_halo else 0.14,
                               zorder=0.5, joinstyle="round"))

    som_figure_context.draw_codes_map(ax, ctx.cb, ctx.assign, ctx.stats, grid, geom,
                       legends=False)
    counts = ctx.stats["n_genes"]
    cnorm = count_norm(counts)
    b = cnorm.boundaries
    x0 = grid.xlim[0] - side_pad + 0.6
    y_key = grid.ylim[1] + 8.5
    ax.text(x0, y_key + 1.9, "# genes per neuron", fontsize=4.4, ha="left",
            va="center")
    w = 2.0
    for i in range(len(b) - 1):
        ax.add_patch(Rectangle((x0 + i * w, y_key - 0.75), w, 1.5,
                               fc=COUNT_CMAP(cnorm((b[i] + b[i + 1]) / 2)),
                               ec="0.3", lw=0.3))
    for v in count_ticks(counts):
        fx = (v - b[0]) / (b[-1] - b[0])
        ax.text(x0 + fx * (len(b) - 1) * w, y_key - 1.7, str(v), ha="center",
                va="top", fontsize=4.0)
    conds = paper_conds(ctx.cb.columns)
    colors = som_figure_context.virus_colors()
    k, lo, hi = som_figure_context.fan_scale(ctx.cb)
    som_figure_context.fan_key(ax, k, lo, hi, conds, colors,
                origin=(grid.cx, grid.ylim[1] + 6.4), magnify=3.4)

    for r in ctx.regions:
        tint = REGION_TINTS[r["name"]]
        if thin_regions:
            path = hex_edge_path(r["units"], grid)
            clip = PathPatch(path, transform=ax.transData)
            for lw, a in ((6.0, 0.14), (3.6, 0.22), (2.0, 0.32)):
                glow = PathPatch(path, fc="none", ec=tint, lw=lw, alpha=a,
                                 zorder=6.4, joinstyle="round")
                ax.add_patch(glow)
                glow.set_clip_path(clip)
            ax.add_patch(PathPatch(path, fc="none", ec=darken(tint), lw=1.2,
                                   alpha=0.95, zorder=6.5,
                                   joinstyle="round"))
        elif region_halo:
            draw_region_outline(ax, r["units"], grid, darken(tint), halo=tint,
                                halo_lw=2.8, core_lw=1.0, z=6.5)
        else:
            path = som_figure_context.territory_path(r["units"], grid, radius=1.75)
            ax.add_patch(PathPatch(path, fc="none", ec=tint, lw=0.8,
                                   alpha=0.9, zorder=7, joinstyle="round"))
    for key in CLASSES if class_outlines else ():
        members = ctx.members_of(key)
        if members:
            som_figure_context.draw_territory(ax, members, grid, CLASSES[key][1], lw=0.6,
                               radius=1.0, z=7)

    if not marks:
        for n in CALLOUTS:
            if n in drop_rings:
                continue
            x, y = grid.centres[n]
            ax.text(x, y, str(n), ha="center", va="center", fontsize=3.8,
                    fontweight="bold", color="#1F2328", zorder=9,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
    else:
        unsupported_rings_spread(ax, grid)
    by_row = {}
    for n in CALLOUTS if marks else ():
        if n in drop_rings:
            continue
        x, y = grid.centres[n]
        ring = (RING_RESTRICTED if n in som_figure_context.RESTRICTED
                else RING_IFN if n == 756 else "#1F2328")
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.05), closed=True, fc="none",
                             ec=ring, lw=0.7 if ring != "#1F2328" else 0.55,
                             zorder=8))
        by_row.setdefault(round(y / 1.5), []).append((x, n))
    for members in by_row.values():
        members.sort()
        for i, (x, n) in enumerate(members):
            y = grid.centres[n][1]
            up = i % 2 == 1
            ax.text(x, y + (1.55 if up else -1.35), str(n), ha="center",
                    va="bottom" if up else "top", fontsize=3.8,
                    fontweight="bold", color="#1F2328", zorder=9,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))

    ax.set_xlim(grid.xlim[0] - side_pad, grid.xlim[1] + right_pad)
    ax.set_ylim(grid.ylim[0] - 1.0, grid.ylim[1] + geom["pad_top"])
    if left_key is None:
        atlas_left_key_v10(ax, ctx, grid, side_pad)
    elif left_key == "compact":
        atlas_left_key_compact(ax, ctx, grid, side_pad)


def atlas_left_key_compact(ax, ctx, grid, side_pad):
    x = grid.xlim[0] - side_pad + 0.6
    y = grid.ylim[1] - 0.5
    total = int(ctx.n_genes.reindex(ctx.sup).sum())
    n_ind = int((ctx.n_up >= 1).sum())
    n_all = int((ctx.n_up == 5).sum())
    n_one = int((ctx.n_up == 1).sum())
    ax.text(x, y, f"{len(ctx.sup)} programmes · {total:,} genes",
            fontsize=5.0, fontweight="bold", color="#1F2328", ha="left",
            va="top")
    y -= 1.9
    ax.text(x, y, f"{n_ind} induced · {n_all} by all five · {n_one} "
                  f"virus-restricted", fontsize=4.0, color="0.35", ha="left",
            va="top")
    y -= 2.1
    short = {"ifn": "IFN ligands", "sense": "Sensing/IRF3",
             "isgf3": "ISGF3", "antigen": "Antigen", "inflam": "Inflammatory",
             "metab": "Metabolic", "tech": "rRNA/MT (tech.)"}
    for key in CLASSES:
        ax.add_patch(Rectangle((x, y - 0.7), 1.4, 0.85, fc=CLASSES[key][1],
                               ec="white", lw=0.3))
        ax.text(x + 2.0, y - 0.28, short[key], fontsize=4.0, va="center",
                ha="left", color="#1F2328")
        y -= 1.05
    y -= 1.4
    for r in ctx.regions:
        tint = REGION_TINTS[r["name"]]
        ax.add_patch(Rectangle((x, y - 0.8), 1.4, 0.9, fc=tint, ec=tint,
                               lw=0.5, alpha=0.45))
        ax.text(x + 2.0, y - 0.35, REGION_SHORT_V10[r["name"]], fontsize=4.0,
                va="center", ha="left", color="#1F2328")
        y -= 1.3
    y -= 1.2
    ax.text(x, y, "numbered ring: named in the Results (Table SX);\n"
                  "red ring: virus-restricted; magenta: IFN ligands;\n"
                  "dashed: named, below the support threshold",
            fontsize=3.8, color="0.40", ha="left", va="top")


def atlas_left_key_v10(ax, ctx, grid, side_pad):
    x = grid.xlim[0] - side_pad + 0.6
    y = grid.ylim[1] - 0.5
    total = int(ctx.n_genes.reindex(ctx.sup).sum())
    n_ind = int((ctx.n_up >= 1).sum())
    n_all = int((ctx.n_up == 5).sum())
    n_one = int((ctx.n_up == 1).sum())
    ax.text(x, y, f"{len(ctx.sup)} programmes · {total:,} genes",
            fontsize=5.2, fontweight="bold", color="#1F2328", ha="left",
            va="top")
    y -= 2.1
    ax.text(x, y, f"{n_ind} induced · {n_all} by all five · {n_one} "
                  f"virus-restricted", fontsize=4.2, color="0.35", ha="left",
            va="top")
    y -= 2.4
    for key in CLASSES:
        ax.add_patch(Rectangle((x, y - 0.85), 1.5, 0.95, fc=CLASSES[key][1],
                               ec="white", lw=0.3))
        ax.text(x + 2.1, y - 0.38, CLASS_SHORT[key], fontsize=4.2,
                va="center", ha="left", color="#1F2328")
        y -= 1.25
    y -= 1.8
    for r in ctx.regions:
        tint = REGION_TINTS[r["name"]]
        ax.add_patch(Rectangle((x, y - 0.95), 1.5, 1.0, fc=tint, ec=tint,
                               lw=0.5, alpha=0.45))
        ax.text(x + 2.1, y - 0.45, REGION_SHORT_V10[r["name"]], fontsize=4.2,
                va="center", ha="left", color="#1F2328")
        y -= 1.5
    y -= 1.6
    ax.text(x, y, "numbered ring: named in the Results (Table SX)\n"
                  "red ring: virus-restricted (648 WNV · 32 DENV)\n"
                  "magenta ring: interferon ligands (756)\n"
                  "dashed: named, below the support threshold",
            fontsize=4.0, color="0.40", ha="left", va="top")


def panel_umatrix_v10(ax, ctx, region_halo=False):
    grid = ctx.grid
    u = umatrix_values(ctx)
    vals = np.array(list(u.values()))
    norm = mpl.colors.Normalize(vmin=float(np.nanpercentile(vals, 2)),
                                vmax=float(np.nanpercentile(vals, 98)))
    cmap = mpl.colormaps["magma"]
    for n, (x, y) in grid.centres.items():
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 0.98), closed=True,
                             fc=cmap(norm(u[n])), ec="white", lw=0.05,
                             zorder=2))
    for r in ctx.regions:
        tint = REGION_TINTS[r["name"]]
        if region_halo:
            draw_region_outline(ax, r["units"], grid, darken(tint, 0.8),
                                halo=tint, halo_lw=2.4, core_lw=1.0, z=6.0)
        else:
            som_figure_context.draw_territory(ax, r["units"], grid, tint, lw=0.8,
                               radius=1.75, z=6)
    for n, _ in UMATRIX_LANDMARKS:
        x, y = grid.centres[n]
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.05), closed=True, fc="none",
                             ec="#1F2328", lw=0.6, zorder=7))
    grid.frame(ax)
    ax.set_ylim(grid.ylim[0] - 11.0, grid.ylim[1] + 1.6)

    x0 = grid.xlim[0] + 0.4
    y = grid.ylim[0] - 2.4
    for r in ctx.regions:
        tint = REGION_TINTS[r["name"]]
        ax.add_patch(Rectangle((x0, y - 0.6), 1.4, 1.2, fc="none", ec=tint,
                               lw=0.9))
        ax.text(x0 + 1.9, y, REGION_SHORT_V10[r["name"]], fontsize=4.2,
                va="center", ha="left", color="#1F2328")
        y -= 1.6
    ax.text(x0, y - 1.6, "ringed = named in the Results\n(756 · 648 · 59)",
            fontsize=4.2, color="0.30", ha="left", va="center")
    cax = ax.inset_axes([0.55, 0.030, 0.40, 0.040])
    bar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                    orientation="horizontal",
                                    ticks=[norm.vmin, norm.vmax])
    bar.ax.set_xticklabels(["valley\n(similar)", "ridge\n(different)"],
                           fontsize=3.9)
    bar.ax.tick_params(length=1.2, pad=1)
    bar.outline.set_linewidth(0.4)
    cax.set_title("mean distance to neighbouring neurons", fontsize=4.2, pad=2)


def panel_conservation_combined(ax, ctx, regions=False):
    grid = ctx.grid
    faces = {int(n): som_figure_context.CONSERVATION_RAMP[int(k) - 1]
             for n, k in ctx.n_shared.items() if k >= 1}
    som_figure_context.hex_field(ax, grid, faces, lw=0.12)
    for k in range(1, 6):
        members = [int(n) for n, v in ctx.n_shared.items() if v == k]
        if members:
            som_figure_context.draw_territory(ax, members, grid, "#5A5F66", lw=0.35,
                               radius=0.99)

    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    up4 = ctx.gene_up[ctx.gene_up >= 4]
    up5 = ctx.gene_up[ctx.gene_up == 5]
    counts = neuron_of.reindex(up4.index).value_counts()
    marked = set(neuron_of.reindex(up5.index).dropna().astype(int))
    for n, c in counts.items():
        x, y = grid.centres[int(n)]
        if int(n) in marked:
            ax.scatter([x], [y], s=int(c) * DOT_PT2, fc="white",
                       ec="#1F2328", lw=0.4, zorder=5)
        else:
            ax.scatter([x], [y], s=int(c) * DOT_PT2, fc="#1F2328",
                       ec="white", lw=0.25, zorder=5)

    for n in som_figure_context.RESTRICTED:
        x, y = grid.centres[n]
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.15), closed=True, fc="none",
                             ec=RING_RESTRICTED, lw=1.0, zorder=8))

    if regions:
        for r in ctx.regions:
            draw_region_outline(ax, r["units"], grid,
                                REGION_TINTS[r["name"]], halo="white",
                                halo_lw=1.8, core_lw=0.8, z=5.6)

    grid.frame(ax)
    ax.set_ylim(grid.ylim[0] - (16.0 if regions else 13.5),
                grid.ylim[1] + 1.6)

    n_ind = int((ctx.n_up >= 1).sum())
    n_all = int((ctx.n_up == 5).sum())
    n4, n5 = int(len(up4)), int(len(up5))
    off4 = n4 - int((ctx.up_on >= 4).sum())
    lnc4 = int((ctx.up_lnc >= 4).sum())
    dn5 = int((ctx.gene_dn == 5).sum())
    ax.set_title(f"{n_all} of {n_ind} induced programmes shared by all five",
                 fontsize=4.8, pad=3)

    x0 = grid.xlim[0] + 0.4
    x1 = x0 + 27.0
    y1 = grid.ylim[0] - 2.2
    ax.text(x0, y1, "programmes: viruses agreeing", fontsize=4.4,
            fontweight="bold", ha="left", va="center", color="#1F2328")
    ax.text(x1, y1, "genes ≥ 4/5 viruses, per neuron", fontsize=4.4,
            fontweight="bold", ha="left", va="center", color="#1F2328")
    y2 = y1 - 2.2
    x = x0 + 1.0
    for k in range(1, 6):
        ax.add_patch(Rectangle((x, y2 - 0.9), 2.4, 1.8,
                               fc=som_figure_context.CONSERVATION_RAMP[k - 1], ec="white",
                               lw=0.3))
        ax.text(x + 1.2, y2 - 1.9, str(k), fontsize=4.2, ha="center",
                va="top")
        x += 3.4
    for dx, c in ((2.5, 1), (7.5, 10), (13.5, 30)):
        ax.scatter([x1 + dx], [y2], s=c * DOT_PT2, fc="0.35", ec="none",
                   zorder=5)
        ax.text(x1 + dx, y2 - 2.0, str(c), fontsize=4.2, ha="center",
                va="top", color="#1F2328")
    y3 = y2 - 4.6
    ax.add_patch(Polygon(som_figure_context.hex_ring(x0 + 1.0, y3, 1.0), closed=True,
                         fc="none", ec=RING_RESTRICTED, lw=0.9, zorder=8))
    ax.text(x0 + 2.6, y3, "ringed = virus-restricted", fontsize=4.2,
            ha="left", va="center", color="#1F2328")
    ax.scatter([x1 + 2.5], [y3], s=10 * DOT_PT2, fc="white", ec="#1F2328",
               lw=0.4, zorder=5)
    ax.text(x1 + 4.4, y3, "white = all five viruses", fontsize=4.2,
            ha="left", va="center", color="#1F2328")
    y_num = y3 - 2.4
    if regions:
        x = x0
        for r in ctx.regions:
            tint = REGION_TINTS[r["name"]]
            ax.add_patch(Rectangle((x, y_num - 0.6), 1.4, 1.2, fc="none",
                                   ec=tint, lw=0.9))
            x += 1.9
            ax.text(x, y_num, REGION_SHORT_V10[r["name"]], fontsize=4.0,
                    va="center", ha="left", color="#1F2328")
            x += len(REGION_SHORT_V10[r["name"]]) * 0.66 + 2.6
        y_num -= 2.0
    ax.text(x0, y_num, f"{n4} genes ≥ 4/5 viruses · {off4} off the ISG "
                       f"panel · {lnc4} lncRNA", fontsize=4.2, ha="left",
            va="center", color="#1F2328")
    ax.text(x0, y_num - 1.6, f"{n5} by all five (white dot) · {dn5} repressed "
                             f"by all five", fontsize=4.2, ha="left",
            va="center", color="#1F2328")


DOT_RAMP = ["#C7E9E3", "#8FD5CB", "#4EBEB0", "#1F9086", "#085B54"]

AGRE_RAMP = ["#FCAE61", "#FD8D3C", "#F03B20", "#BD0026", "#7A0019"]

PIE_COLORS = ("#472A78", "0.72", "#1F9086")

CONS_LEGEND_MM = 36.0
COMP_LEGEND_MM = 13.2

V19_LEGEND_DEPTH_MM = 8.0


def _v19_legend_scale(ax, grid):
    rect_h_mm = (ax.get_position().height
                 * ax.figure.get_size_inches()[1] * 25.4)
    map_span = (grid.ylim[1] + 1.6) - grid.ylim[0]
    return (rect_h_mm - V19_LEGEND_DEPTH_MM) / map_span


def _dot_pt2_for_scale(scale):
    diameter_mm = 0.85 * SQ3 * 0.98 * scale
    radius_pt = diameter_mm / 2 * 72 / 25.4
    return np.pi * radius_pt ** 2 / 30


def _v19_dot_pt2(ax, grid):
    return _dot_pt2_for_scale(_v19_legend_scale(ax, grid))

TRUNC_MAGMA = mpl.colors.LinearSegmentedColormap.from_list(
    "magma_trunc", mpl.colormaps["magma"](np.linspace(0.15, 0.85, 256)))

TRUNC_GREYS = mpl.colors.LinearSegmentedColormap.from_list(
    "greys_trunc", mpl.colormaps["Greys"](np.linspace(0.10, 0.80, 256)))


def _conservation_legend_v19(ax, ctx, grid, dot_pt2):
    step_counts = {k: int((ctx.n_up == k).sum()) for k in range(1, 6)}
    scale = _v19_legend_scale(ax, grid)

    def u(mmv):
        return mmv / scale

    def yv(mmv):
        return grid.ylim[0] - mmv / scale

    x0 = grid.xlim[0] + 0.4

    ax.text(x0, yv(1.72), "viruses agreeing", fontsize=4.4,
            fontweight="bold", ha="left", va="center", color="#1F2328",
            zorder=11)
    x = x0 + u(18.0)
    for k in range(1, 6):
        ax.text(x + 1.2, yv(1.72), str(k), fontsize=4.2, ha="center",
                va="center", color="#1F2328", zorder=11)
        ax.add_patch(Rectangle((x, yv(2.54) - u(1.15)), 2.4, u(1.15),
                               fc=DOT_RAMP[k - 1], ec="white", lw=0.3,
                               zorder=5))
        ax.text(x + 1.2, yv(3.99), str(step_counts[k]), fontsize=4.2,
                ha="center", va="top", color="#1F2328", zorder=11)
        x += 3.4

    y2 = yv(6.32)
    ax.text(x0, y2, "genes ≥ 4/5", fontsize=4.4, fontweight="bold",
            ha="left", va="center", color="#1F2328", zorder=11)
    for dx, c in ((13.5, 1), (16.8, 10), (20.4, 30)):
        ax.scatter([x0 + u(dx)], [y2], s=max(c * dot_pt2, 0.3), fc="0.35",
                   ec="#1F2328", lw=0.3, zorder=5)
        ax.text(x0 + u(dx + 0.9), y2, str(c), fontsize=4.2, ha="left",
                va="center", color="#1F2328", zorder=11)
    ax.scatter([x0 + u(24.2)], [y2], s=max(10 * dot_pt2, 0.3), fc="white",
               ec="#1F2328", lw=0.5, zorder=5)
    ax.text(x0 + u(25.2), y2, "all five", fontsize=4.2, ha="left",
            va="center", color="#1F2328", zorder=11)
    ax.text(grid.xlim[1] - 0.4, y2, "colours as in b", fontsize=4.2,
            ha="right", va="center", color="0.35", zorder=11)
    return grid.ylim[0] - u(V19_LEGEND_DEPTH_MM)


def panel_structure_conservation(ax, ctx, grey=False, locators=None,
                                 regions=True, restricted=True,
                                 legend_v19=False, dot_pt2=0.7):
    grid = ctx.grid
    cmap = TRUNC_GREYS if grey else TRUNC_MAGMA
    dot_edge = "#1F2328" if grey else "white"
    if legend_v19:
        faces = {int(n): som_figure_context.KIN_COLORS[(r["direction"], r["kinetic_class"])]
                 for n, r in ctx.kin.iterrows()
                 if (r["direction"], r["kinetic_class"]) in som_figure_context.KIN_COLORS}
        som_figure_context.hex_field(ax, grid, faces, lw=0.10)
        dot_pt2 = _v19_dot_pt2(ax, grid)
    else:
        u = umatrix_values(ctx)
        vals = np.array(list(u.values()))
        norm = mpl.colors.Normalize(vmin=float(np.nanpercentile(vals, 2)),
                                    vmax=float(np.nanpercentile(vals, 98)))
        for n, (x, y) in grid.centres.items():
            ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 0.98), closed=True,
                                 fc=cmap(norm(u[n])), ec="#C4C9CE" if grey
                                 else "white", lw=0.05, zorder=2))
    if regions:
        for r in ctx.regions:
            tint = REGION_TINTS[r["name"]]
            draw_region_outline(ax, r["units"], grid, darken(tint, 0.8),
                                halo=tint, halo_lw=2.4, core_lw=1.0, z=3.5)

    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    up4 = ctx.gene_up[ctx.gene_up >= 4]
    up5 = ctx.gene_up[ctx.gene_up == 5]
    counts = neuron_of.reindex(up4.index).value_counts()
    marked = set(neuron_of.reindex(up5.index).dropna().astype(int))
    sup = set(ctx.sup)
    for n, c in counts.items():
        n = int(n)
        if n not in sup:
            continue
        x, y = grid.centres[n]
        share = int(ctx.n_shared.get(n, 1))
        s = max(int(c) * dot_pt2, 0.3)
        if n in marked:
            ax.scatter([x], [y], s=s, fc="white",
                       ec="#1F2328", lw=0.4, zorder=5)
        else:
            ax.scatter([x], [y], s=s, fc=DOT_RAMP[share - 1],
                       ec=dot_edge, lw=0.25, zorder=5)
    if restricted:
        for n in som_figure_context.RESTRICTED:
            x, y = grid.centres[n]
            ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.15), closed=True,
                                 fc="none", ec=RING_RESTRICTED, lw=1.0,
                                 zorder=8))

    if locators:
        for (x0, x1, yy0, yy1), tint, sym in locators:
            ax.add_patch(Rectangle((x0, yy0), x1 - x0, yy1 - yy0, fc="none",
                                   ec=darken(tint, 0.7), lw=1.0,
                                   ls=(0, (3.0, 2.0)), zorder=9))
            ax.text(x0 + 0.4, yy1 + 0.4, sym, fontsize=5.4, fontweight="bold",
                    color=darken(tint, 0.7), ha="left", va="bottom", zorder=9)

    grid.frame(ax)
    ax.set_ylim(grid.ylim[0] - (22.0 if regions else 18.5),
                grid.ylim[1] + 1.6)

    n_ind = int((ctx.n_up >= 1).sum())
    n_all = int((ctx.n_up == 5).sum())
    n4, n5 = int(len(up4)), int(len(up5))
    off4 = n4 - int((ctx.up_on >= 4).sum())
    lnc4 = int((ctx.up_lnc >= 4).sum())
    dn5 = int((ctx.gene_dn == 5).sum())
    ax.set_title(f"{n_all} of {n_ind} induced programmes shared by all five",
                 fontsize=4.8, pad=3)

    if legend_v19:
        bottom = _conservation_legend_v19(ax, ctx, grid, dot_pt2)
        ax.set_ylim(bottom, ax.get_ylim()[1])
        return

    x0 = grid.xlim[0] + 0.4
    x1 = x0 + 29.0
    y1 = grid.ylim[0] - 2.2
    ax.text(x0, y1, "colour: viruses agreeing", fontsize=4.0,
            fontweight="bold", ha="left", va="center", color="#1F2328")
    ax.text(x1, y1, "size: genes ≥ 4/5", fontsize=4.0, fontweight="bold",
            ha="left", va="center", color="#1F2328")
    y2 = y1 - 2.4
    x = x0 + 1.0
    for k in range(1, 6):
        ax.add_patch(Rectangle((x, y2 - 1.1), 2.8, 2.2, fc=DOT_RAMP[k - 1],
                               ec="white", lw=0.4))
        ax.text(x + 1.4, y2 - 2.3, str(k), fontsize=4.4, ha="center",
                va="top")
        x += 3.6
    for dx, c in ((2.5, 1), (7.5, 10), (13.5, 30)):
        ax.scatter([x1 + dx], [y2], s=c * DOT_PT2, fc="0.35", ec="#1F2328",
                   lw=0.3, zorder=5)
        ax.text(x1 + dx, y2 - 2.2, str(c), fontsize=4.4, ha="center",
                va="top", color="#1F2328")
    y3 = y2 - 4.8
    if restricted:
        ax.add_patch(Polygon(som_figure_context.hex_ring(x0 + 1.1, y3, 1.1), closed=True,
                             fc="none", ec=RING_RESTRICTED, lw=1.2, zorder=8))
        ax.text(x0 + 2.9, y3, "ringed = virus-restricted", fontsize=4.4,
                ha="left", va="center", color="#1F2328")
    ax.scatter([x1 + 2.5], [y3], s=10 * DOT_PT2, fc="white", ec="#1F2328",
               lw=0.5, zorder=5)
    ax.text(x1 + 4.6, y3, "white = all five viruses", fontsize=4.4,
            ha="left", va="center", color="#1F2328")

    y5 = y3 - 2.8
    for i, line in enumerate([f"{n4} genes ≥ 4/5 viruses",
                              f"{off4} off ISG panel",
                              f"{lnc4} lncRNA · {n5} all five ○",
                              f"{dn5} repressed by all five"]):
        ax.text(x0, y5 - 1.7 * i, line, fontsize=4.2, ha="left", va="center",
                color="#1F2328")
    ax.text(x1, y5 + 0.8, "fill = mean distance to", fontsize=4.2, ha="left",
            va="center", color="#1F2328")
    ax.text(x1, y5 - 0.9, "neighbouring neurons", fontsize=4.2,
            ha="left", va="center", color="#1F2328")
    bw, bh = 20.0, 1.8
    for i in range(40):
        ax.add_patch(Rectangle((x1 + i * bw / 40, y5 - 3.3 - bh / 2),
                               bw / 40, bh, fc=cmap(i / 39), ec="none"))
    ax.add_patch(Rectangle((x1, y5 - 3.3 - bh / 2), bw, bh, fc="none",
                           ec="0.3", lw=0.5))
    ax.text(x1, y5 - 3.3 - 1.9, "valley", fontsize=4.2, ha="left", va="top",
            color="#1F2328")
    ax.text(x1 + bw, y5 - 3.3 - 1.9, "ridge", fontsize=4.2, ha="right",
            va="top", color="#1F2328")
    if regions:
        y8 = y5 - 8.6
        x = x0
        for r in ctx.regions:
            tint = REGION_TINTS[r["name"]]
            ax.add_patch(Rectangle((x, y8 - 0.7), 1.7, 1.4, fc="none",
                                   ec=tint, lw=1.1))
            x += 2.1
            ax.text(x, y8, REGION_SHORT_V10[r["name"]], fontsize=4.1,
                    va="center", ha="left", color="#1F2328")
            x += len(REGION_SHORT_V10[r["name"]]) * 0.68 + 2.4


def compose_v10(ctx):
    fig_w, fig_h = 183.0, 247.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.010, 0.628, 0.980, 0.362])
    panel_hybrid_codes_map_v10(ax_a, ctx)
    label(fig, 0.006, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.350, 0.430, 0.235])
    som_figure_context.panel_kinetic_map(ax_b, ctx)
    label(fig, 0.006, 0.604, "b")

    ax_c = fig.add_axes([0.545, 0.350, 0.430, 0.235])
    panel_umatrix_v10(ax_c, ctx)
    label(fig, 0.510, 0.604, "c")

    ax_d = fig.add_axes([0.030, 0.030, 0.330, 0.270])
    panel_conservation_combined(ax_d, ctx)
    label(fig, 0.006, 0.325, "d")

    gs_e = fig.add_gridspec(1, 2, left=0.645, right=0.935, top=0.250,
                            bottom=0.065, wspace=1.6)
    panel_restricted_pair(fig, [gs_e[0], gs_e[1]], ctx,
                          cbar_rect=[0.947, 0.095, 0.009, 0.105],
                          title_size=4.0, short=True, per_unit=23)
    label(fig, 0.610, 0.325, "e")

    out = FIGDIR / "fig6_v10_legend.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v10 legend   : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v11(ctx):
    fig_w, fig_h = 183.0, 247.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.010, 0.628, 0.980, 0.362])
    panel_hybrid_codes_map_v10(ax_a, ctx, region_halo=True)
    label(fig, 0.006, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.355, 0.250, 0.245])
    som_figure_context.panel_kinetic_map(ax_b, ctx)
    label(fig, 0.006, 0.618, "b")

    ax_c = fig.add_axes([0.310, 0.355, 0.370, 0.245])
    panel_structure_conservation(ax_c, ctx)
    label(fig, 0.290, 0.618, "c")

    ax_d = fig.add_axes([0.055, 0.075, 0.330, 0.225])
    panel_phases(ax_d, ctx, fontsize=4.6, label_bbox=True)
    label(fig, 0.006, 0.325, "d")

    gs_e = fig.add_gridspec(1, 2, left=0.640, right=0.935, top=0.255,
                            bottom=0.095, wspace=1.6)
    panel_restricted_pair(fig, [gs_e[0], gs_e[1]], ctx,
                          cbar_rect=[0.947, 0.115, 0.009, 0.095],
                          title_size=4.0, short=True, per_unit=23, isg=True,
                          title_counts=False)
    label(fig, 0.600, 0.325, "e")
    fig.text(0.640, 0.056, "columns: ZIKV WNV TBEV DENV YFV at 24 hpi | "
                           "72 hpi", fontsize=4.2, color="0.30", ha="left",
             va="top")
    fig.text(0.640, 0.042, "all genes shown (23 + 8)", fontsize=4.2,
             color="0.30", ha="left", va="top")
    fig.text(0.640, 0.028, "purple gene label = on the 382-ISG panel",
             fontsize=4.2, color="#472A78", ha="left", va="top")
    fig.text(0.640, 0.014, "grey gene label = off the panel",
             fontsize=4.2, color="0.55", ha="left", va="top")

    out = FIGDIR / "fig6_v11_nature.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v11 nature   : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v12(ctx):
    fig_w, fig_h = 183.0, 235.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.010, 0.618, 0.980, 0.372])
    panel_hybrid_codes_map_v10(ax_a, ctx, region_halo=True)
    label(fig, 0.006, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.040, 0.440, 0.540])
    som_figure_context.panel_kinetic_map(ax_b, ctx)
    label(fig, 0.006, 0.596, "b")

    ax_c = fig.add_axes([0.510, 0.040, 0.470, 0.540])
    panel_structure_conservation(ax_c, ctx, grey=True)
    label(fig, 0.500, 0.596, "c")

    out = FIGDIR / "fig6_v12_atlas.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v12 atlas    : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


PHASE_DB = [("①", "interferon production", "IFNL1–4, IFNB1 · unit 756",
             "#CC79A7"),
            ("②", "antiviral & immune effectors", "units 728 · 726 · 725 · 727",
             "#E69F00"),
            ("③", "metabolic repression", "ketogenesis · CYP450 · conjugation",
             "#8C6D46")]


def phase_means_measured(ctx):
    gl = ctx.gene_long[ctx.gene_long.treatment == "untreated"]
    units = {"①": [756], "②": [728, 726, 725, 727],
             "③": ctx.members_of("metab")}
    return {sym: gl[gl.neuron.isin(us)].groupby(["virus", "hpi"])
            .log2fc.mean().unstack() for sym, us in units.items()}


def panel_phase_dumbbells(fig, slot, ctx):
    colors = som_figure_context.virus_colors()
    means = phase_means_measured(ctx)
    lo = min(float(df.to_numpy().min()) for df in means.values())
    hi = max(float(df.to_numpy().max()) for df in means.values())
    pad = 0.06 * (hi - lo)
    gs = slot.subgridspec(1, 3, wspace=0.55)
    for pi, (sym, name, caption, colour) in enumerate(PHASE_DB):
        ax = fig.add_subplot(gs[pi])
        df = means[sym]
        for vi, v in enumerate(VIRUS_ORDER):
            v24, v72 = float(df.loc[v, 24]), float(df.loc[v, 72])
            ax.plot([v24, v72], [vi, vi], color=colors[v], lw=0.7, zorder=2)
            ax.scatter([v24], [vi], s=13, fc="white", ec=colors[v], lw=0.7,
                       zorder=3)
            ax.scatter([v72], [vi], s=13, fc=colors[v], ec="white", lw=0.3,
                       zorder=3)
        m24, m72 = float(df[24].mean()), float(df[72].mean())
        ym = len(VIRUS_ORDER) + 0.4
        ax.plot([m24, m72], [ym, ym], color="#1F2328", lw=1.1, zorder=2)
        ax.scatter([m24], [ym], s=17, fc="white", ec="#1F2328", lw=0.9,
                   zorder=3)
        ax.scatter([m72], [ym], s=17, fc="#1F2328", ec="white", lw=0.3,
                   zorder=3)
        ax.axvline(0, color="0.4", lw=0.5, zorder=1)
        ax.set_yticks(list(range(len(VIRUS_ORDER))) + [ym])
        ax.set_yticklabels(list(VIRUS_ORDER) + ["mean"], fontsize=4.5)
        ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.set_ylim(-0.6, ym + 0.7)
        ax.set_xlim(lo - pad, hi + pad)
        ax.tick_params(axis="x", labelsize=4.2, length=1.2, pad=1)
        ax.tick_params(axis="y", length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(f"{sym} {name}\n{caption}", fontsize=4.4, color=colour,
                     pad=3)
        if pi == 1:
            ax.set_xlabel("mean untreated log₂FC vs Mock, phase member genes\n"
                          "○ 24 hpi    ● 72 hpi", fontsize=4.4, labelpad=2)
        else:
            ax.set_xlabel("log₂FC vs Mock", fontsize=4.2, labelpad=2)


def compose_v13(ctx):
    fig_w, fig_h = 183.0, 247.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.010, 0.632, 0.980, 0.358])
    panel_hybrid_codes_map_v10(ax_a, ctx, region_halo=True)
    label(fig, 0.006, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.340, 0.430, 0.270])
    som_figure_context.panel_kinetic_map(ax_b, ctx)
    label(fig, 0.006, 0.625, "b")

    ax_c = fig.add_axes([0.510, 0.340, 0.470, 0.270])
    panel_structure_conservation(ax_c, ctx, grey=True)
    label(fig, 0.490, 0.625, "c")

    gs_d = fig.add_gridspec(1, 1, left=0.055, right=0.975, top=0.270,
                            bottom=0.080)
    panel_phase_dumbbells(fig, gs_d[0], ctx)
    label(fig, 0.006, 0.300, "d")
    med_i = 100 * float(np.median(ctx.share24["induced"]))
    med_r = 100 * float(np.median(ctx.share24["repressed"]))
    fig.text(0.055, 0.052, f"median {med_i:.0f}% of induced-programme genes "
                           f"already DE at 24 hpi · {med_r:.0f}% for "
                           f"repressed programmes", fontsize=4.4,
             color="#1F2328", ha="left", va="top")

    out = FIGDIR / "fig6_v13_measured.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v13 measured : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


BADGE_PT2 = 0.14
BADGE_DY = -1.08


def callout_box(ax, x, y, w, title, lines, colour):
    h = 3.0 + 1.55 * len(lines)
    ax.add_patch(Rectangle((x, y - h), w, h, fc="white", ec=colour, lw=0.8,
                           zorder=9))
    ax.text(x + 1.0, y - 1.2, title, fontsize=4.5, fontweight="bold",
            color=darken(colour, 0.8), ha="left", va="center", zorder=10)
    for i, line in enumerate(lines):
        ax.text(x + 1.0, y - 2.8 - 1.55 * i, line, fontsize=4.2,
                color="#1F2328", ha="left", va="center", zorder=10)
    return y - h


def panel_super_atlas(ax, ctx, side_pad=30.0, right_pad=22.0, pad_scale=1.6):
    grid = ctx.grid
    geom = dict(som_figure_context.panel_geometry(grid))
    base_range = geom["x_range"]
    geom["side_pad"] = side_pad
    geom["x_range"] = grid.xlim[1] - grid.xlim[0] + 2 * side_pad
    geom["m"] *= geom["x_range"] / base_range
    geom["pad_top"] *= pad_scale

    for r in ctx.regions:
        path = som_figure_context.territory_path(r["units"], grid, radius=1.75)
        tint = REGION_TINTS[r["name"]]
        ax.add_patch(PathPatch(path, fc=tint, ec="none", alpha=0.22,
                               zorder=0.5, joinstyle="round"))

    som_figure_context.draw_codes_map(ax, ctx.cb, ctx.assign, ctx.stats, grid, geom,
                       legends=False)
    counts = ctx.stats["n_genes"]
    cnorm = count_norm(counts)
    b = cnorm.boundaries
    x0c = grid.xlim[0] - side_pad + 0.6
    y_key = grid.ylim[1] + 8.5
    ax.text(x0c, y_key + 1.9, "# genes per neuron", fontsize=4.4, ha="left",
            va="center")
    w = 2.0
    for i in range(len(b) - 1):
        ax.add_patch(Rectangle((x0c + i * w, y_key - 0.75), w, 1.5,
                               fc=COUNT_CMAP(cnorm((b[i] + b[i + 1]) / 2)),
                               ec="0.3", lw=0.3))
    for v in count_ticks(counts):
        fx = (v - b[0]) / (b[-1] - b[0])
        ax.text(x0c + fx * (len(b) - 1) * w, y_key - 1.7, str(v), ha="center",
                va="top", fontsize=4.0)
    conds = list(ctx.cb.columns)
    colors = som_figure_context.virus_colors()
    k, lo, hi = som_figure_context.fan_scale(ctx.cb)
    som_figure_context.fan_key(ax, k, lo, hi, conds, colors,
                origin=(grid.cx + 4.0, grid.ylim[1] + 6.4), magnify=3.4)

    for r in ctx.regions:
        tint = REGION_TINTS[r["name"]]
        draw_region_outline(ax, r["units"], grid, darken(tint), halo=tint,
                            halo_lw=2.8, core_lw=1.0, z=6.5)
    for key in CLASSES:
        members = ctx.members_of(key)
        if members:
            som_figure_context.draw_territory(ax, members, grid, CLASSES[key][1], lw=0.6,
                               radius=1.0, z=7)

    unsupported_rings_spread(ax, grid)
    by_row = {}
    for n in CALLOUTS:
        x, y = grid.centres[n]
        ring = (RING_RESTRICTED if n in som_figure_context.RESTRICTED
                else RING_IFN if n == 756 else "#1F2328")
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.05), closed=True, fc="none",
                             ec=ring, lw=0.7 if ring != "#1F2328" else 0.55,
                             zorder=8))
        by_row.setdefault(round(y / 1.5), []).append((x, n))
    for members in by_row.values():
        members.sort()
        for i, (x, n) in enumerate(members):
            y = grid.centres[n][1]
            up = i % 2 == 1
            ax.text(x, y + (1.55 if up else -1.35), str(n), ha="center",
                    va="bottom" if up else "top", fontsize=3.8,
                    fontweight="bold", color="#1F2328", zorder=9,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))

    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    up4 = ctx.gene_up[ctx.gene_up >= 4]
    up5 = ctx.gene_up[ctx.gene_up == 5]
    bcounts = neuron_of.reindex(up4.index).value_counts()
    bmarked = set(neuron_of.reindex(up5.index).dropna().astype(int))
    n5 = int(len(up5))
    for n, c in bcounts.items():
        n = int(n)
        if n not in set(ctx.sup):
            continue
        x, y = grid.centres[n]
        share = int(ctx.n_shared.get(n, 1))
        if n in bmarked:
            ax.scatter([x], [y + BADGE_DY], s=0.6 + BADGE_PT2 * int(c),
                       fc="white", ec="#1F2328", lw=0.35, zorder=6)
        else:
            ax.scatter([x], [y + BADGE_DY], s=0.6 + BADGE_PT2 * int(c),
                       fc=DOT_RAMP[share - 1], ec="#1F2328", lw=0.25,
                       zorder=6)

    ax.set_xlim(grid.xlim[0] - side_pad, grid.xlim[1] + right_pad)
    ax.set_ylim(grid.ylim[0] - 13.0, grid.ylim[1] + geom["pad_top"])
    atlas_left_key_v14(ax, ctx, grid, side_pad)

    n4 = int(len(up4))
    dn5 = int((ctx.gene_dn == 5).sum())

    def region_facts(r):
        genes = set(ctx.assign[ctx.assign.neuron.isin(r["units"])].gene_id)
        return (len(set(ctx.gene_up[ctx.gene_up == 5].index) & genes),
                len(set(ctx.gene_up[ctx.gene_up >= 4].index) & genes),
                len(set(ctx.gene_dn[ctx.gene_dn == 5].index) & genes))

    core = next(r for r in ctx.regions if 756 in r["units"])
    inflam = next(r for r in ctx.regions if 648 in r["units"])
    metab = next(r for r in ctx.regions if 59 in r["units"])
    xr = grid.xlim[1] + 2.0
    yl0 = grid.ylim[0]

    a5, a4, _ = region_facts(core)
    callout_box(ax, xr, 1.0, 19.0,
                "Interferon, ISG & antigen core",
                [f"{len(core['units'])} programmes · {core['n_genes']:,} genes",
                 "phases ①–② · early IFN & effectors",
                 f"{a5} of {n5} all-five genes here"],
                REGION_TINTS[core["name"]])
    ax.plot([xr - 0.3, 48.0], [-0.8, -0.1], color=REGION_TINTS[core["name"]],
            lw=0.6, alpha=0.8, zorder=8)
    callout_box(ax, xr, -9.0, 19.0,
                "virus-restricted units",
                ["648 WNV · stress response", "ATF3 · BBC3 · PPP1R15A",
                 "32 DENV · hepatic metabolic", "ACOT12 · ANXA13 · SULT1E1 · "
                 "UGT2B15"],
                RING_RESTRICTED)

    _, a4i, _ = region_facts(inflam)
    bot = callout_box(ax, grid.xlim[0] + 1.5, grid.ylim[1] + 9.9, 19.5,
                      "Inflammatory & stress",
                      ["immediate-early",
                       f"{len(inflam['units'])} programmes · "
                       f"{inflam['n_genes']} genes",
                       f"phase ② · {a4i} of {n4} ≥4-virus genes"],
                      REGION_TINTS[inflam["name"]])
    ax.plot([8.5, 6.4], [bot - 0.2, -3.2], color=REGION_TINTS[inflam["name"]],
            lw=0.6, alpha=0.8, zorder=8)

    _, _, d5m = region_facts(metab)
    bot = callout_box(ax, grid.xlim[0] + 1.0, yl0 - 2.6, 17.0,
                      "Hepatocyte metabolic &",
                      ["secretory",
                       f"{len(metab['units'])} programmes · "
                       f"{metab['n_genes']} genes",
                       f"phase ③ · {d5m} of {dn5} all-five-",
                       "repressed genes here"],
                      REGION_TINTS[metab["name"]])
    ax.plot([3.5, 3.5], [yl0 - 2.8, -37.2], color=REGION_TINTS[metab["name"]],
            lw=0.6, alpha=0.8, zorder=8)

    xb = grid.xlim[0] + 21.0
    ax.text(xb, yl0 - 4.0, "hexagon badge (bottom vertex)", fontsize=4.4,
            fontweight="bold", ha="left", va="center", color="#1F2328")
    ax.text(xb, yl0 - 6.0, "colour: viruses agreeing", fontsize=4.2, ha="left",
            va="center", color="#1F2328")
    x = xb + 18.5
    for k5 in range(1, 6):
        ax.add_patch(Rectangle((x, yl0 - 6.7), 2.2, 1.4, fc=DOT_RAMP[k5 - 1],
                               ec="white", lw=0.3))
        ax.text(x + 1.1, yl0 - 7.6, str(k5), fontsize=4.0, ha="center",
                va="top")
        x += 3.0
    ax.text(xb, yl0 - 9.6, "size: genes ≥ 4/5", fontsize=4.2, ha="left",
            va="center", color="#1F2328")
    for dx, c in ((12.0, 1), (16.5, 10), (22.0, 30)):
        ax.scatter([xb + dx], [yl0 - 9.6], s=0.6 + BADGE_PT2 * c, fc="0.35",
                   ec="#1F2328", lw=0.3, zorder=5)
        ax.text(xb + dx, yl0 - 10.7, str(c), fontsize=4.0, ha="center",
                va="top", color="#1F2328")
    ax.scatter([xb + 27.0], [yl0 - 9.6], s=0.6 + BADGE_PT2 * 10, fc="white",
               ec="#1F2328", lw=0.4, zorder=5)
    ax.text(xb + 28.6, yl0 - 9.6, "= all five", fontsize=4.2, ha="left",
            va="center", color="#1F2328")


def atlas_left_key_v14(ax, ctx, grid, side_pad):
    x = grid.xlim[0] - side_pad + 0.6
    y = grid.ylim[1] - 0.5
    total = int(ctx.n_genes.reindex(ctx.sup).sum())
    n_ind = int((ctx.n_up >= 1).sum())
    n_all = int((ctx.n_up == 5).sum())
    n_one = int((ctx.n_up == 1).sum())
    ax.text(x, y, f"{len(ctx.sup)} programmes · {total:,} genes",
            fontsize=5.2, fontweight="bold", color="#1F2328", ha="left",
            va="top")
    y -= 2.1
    ax.text(x, y, f"{n_ind} induced · {n_all} by all five · {n_one} "
                  f"virus-restricted", fontsize=4.2, color="0.35", ha="left",
            va="top")
    y -= 2.4
    for key in CLASSES:
        ax.add_patch(Rectangle((x, y - 0.85), 1.5, 0.95, fc=CLASSES[key][1],
                               ec="white", lw=0.3))
        ax.text(x + 2.1, y - 0.38, CLASS_SHORT[key], fontsize=4.2,
                va="center", ha="left", color="#1F2328")
        y -= 1.25
    y -= 1.8
    ax.text(x, y, "numbered ring: named in the Results (Table SX)\n"
                  "red ring: virus-restricted (648 WNV · 32 DENV)\n"
                  "magenta ring: interferon ligands (756)\n"
                  "dashed: named, below the support threshold",
            fontsize=4.0, color="0.40", ha="left", va="top")


def panel_phase_tiles(fig, slot, ctx):
    means = phase_means_measured(ctx)
    M = np.array([[float(means[sym].loc[v, h]) for h in (24, 72)
                   for v in VIRUS_ORDER] for sym, _, _, _ in PHASE_DB])
    norm = TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6)
    ax = fig.add_subplot(slot)
    ax.imshow(np.clip(M, -6, 6), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.axvline(4.5, color="white", lw=1.6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center",
                    fontsize=4.4,
                    color="white" if abs(M[i, j]) > 3.3 else "#23262B")
    ax.set_xticks(range(10))
    ax.set_xticklabels(VIRUS_ORDER * 2, fontsize=4.2, rotation=90)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{sym} {name}\n{cap}" for sym, name, cap, _ in
                        PHASE_DB], fontsize=4.4)
    for lab, (_, _, _, colour) in zip(ax.get_yticklabels(), PHASE_DB):
        lab.set_color(colour)
    ax.text(2.0, -0.85, "24 hpi", ha="center", va="bottom", fontsize=5.0,
            fontweight="bold")
    ax.text(7.0, -0.85, "72 hpi", ha="center", va="bottom", fontsize=5.0,
            fontweight="bold")
    ax.tick_params(length=1.2, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.set_title("the three phases, measured", fontsize=5.0, pad=4)
    return norm


def compose_v14(ctx):
    fig_w, fig_h = 183.0, 247.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.008, 0.505, 0.984, 0.485])
    panel_super_atlas(ax_a, ctx)
    label(fig, 0.004, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.025, 0.430, 0.440])
    som_figure_context.panel_kinetic_map(ax_b, ctx)
    label(fig, 0.006, 0.478, "b")

    gs_c = fig.add_gridspec(1, 1, left=0.600, right=0.855, top=0.420,
                            bottom=0.130)
    norm = panel_phase_tiles(fig, gs_c[0], ctx)
    label(fig, 0.490, 0.478, "c")
    cax = fig.add_axes([0.872, 0.130, 0.009, 0.230])
    bar = mpl.colorbar.ColorbarBase(cax, cmap="RdBu_r", norm=norm,
                                    ticks=[-6, 0, 6], extend="both")
    cax.tick_params(labelsize=4.0, length=1.2, pad=1)
    bar.outline.set_linewidth(0.4)
    fig.text(0.864, 0.245, "mean untreated log₂FC vs Mock", fontsize=4.0,
             color="0.25", rotation=90, ha="center", va="center")
    fig.text(0.520, 0.095, "tiles: mean untreated log₂FC vs Mock of the "
                           "phase's member genes", fontsize=4.2, color="0.30",
             ha="left", va="top")
    med_i = 100 * float(np.median(ctx.share24["induced"]))
    med_r = 100 * float(np.median(ctx.share24["repressed"]))
    fig.text(0.520, 0.080, f"median {med_i:.0f}% of induced-programme genes "
                           f"already DE at 24 hpi · {med_r:.0f}% for "
                           f"repressed programmes", fontsize=4.2,
             color="#1F2328", ha="left", va="top")

    out = FIGDIR / "fig6_v14_merged_atlas.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v14 merged   : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def zoom_windows(ctx, margin=2.5):
    out = []
    for sym, sentinel in (("i", 756), ("ii", 648), ("iii", 59)):
        r = next(r for r in ctx.regions if sentinel in r["units"])
        xs = [ctx.grid.centres[n][0] for n in r["units"]]
        ys = [ctx.grid.centres[n][1] for n in r["units"]]
        out.append((sym, r, (min(xs) - margin, max(xs) + margin,
                             min(ys) - margin, max(ys) + margin)))
    return out


def panel_structure_zoom(ax, ctx, sym, region, window):
    import textwrap

    grid = ctx.grid
    x0, x1, y0, y1 = window
    tint = REGION_TINTS[region["name"]]
    u = umatrix_values(ctx)
    vals = np.array(list(u.values()))
    norm = mpl.colors.Normalize(vmin=float(np.nanpercentile(vals, 2)),
                                vmax=float(np.nanpercentile(vals, 98)))
    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    up4 = ctx.gene_up[ctx.gene_up >= 4]
    up5 = ctx.gene_up[ctx.gene_up == 5]
    counts = neuron_of.reindex(up4.index).value_counts()
    marked = set(neuron_of.reindex(up5.index).dropna().astype(int))
    sup = set(ctx.sup)

    for n, (x, y) in grid.centres.items():
        if x0 - 1.1 < x < x1 + 1.1 and y0 - 1.1 < y < y1 + 1.1:
            ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 0.98), closed=True,
                                 fc=TRUNC_GREYS(norm(u[n])), ec="#C4C9CE",
                                 lw=0.15, zorder=2))
    draw_region_outline(ax, region["units"], grid, darken(tint, 0.8),
                        halo=tint, halo_lw=2.4, core_lw=1.0, z=3.5)
    for n, c in counts.items():
        n = int(n)
        if n not in sup:
            continue
        x, y = grid.centres[n]
        if not (x0 < x < x1 and y0 < y < y1):
            continue
        share = int(ctx.n_shared.get(n, 1))
        kw = dict(fc="white", ec="#1F2328", lw=0.5) if n in marked else \
            dict(fc=DOT_RAMP[share - 1], ec="#1F2328", lw=0.3)
        ax.scatter([x], [y], s=2.0 + 5.0 * int(c), zorder=5, **kw)
    for n in som_figure_context.RESTRICTED:
        x, y = grid.centres[n]
        if x0 < x < x1 and y0 < y < y1:
            ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.12), closed=True,
                                 fc="none", ec=RING_RESTRICTED, lw=1.2,
                                 zorder=8))
    key_units = []
    for n in sorted(sup):
        if n not in CALLOUTS:
            continue
        x, y = grid.centres[n]
        if x0 < x < x1 and y0 < y < y1:
            ax.text(x, y - 0.62, str(n), ha="center", va="center",
                    fontsize=4.4, fontweight="bold", color="#1F2328",
                    zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.15))
            key_units.append(n)

    parts = []
    for n in key_units:
        genes = ", ".join(CALLOUTS[n][1].split(", ")[:3])
        parts.append(f"{n}: {genes}")
    key = "  ·  ".join(parts)
    width_chars = max(24, int((x1 - x0) * 3.0 / 0.85))
    lines = textwrap.wrap(key, width=width_chars)
    key_h = len(lines) * 1.7 + 1.0
    for i, line in enumerate(lines):
        ax.text(x0 + 0.3, y0 - 1.0 - 1.7 * i, line, fontsize=4.3,
                fontstyle="italic", color="#1F2328", ha="left", va="top",
                zorder=9)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0 - key_h, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    return key_units


def compose_v15(ctx):
    fig_w, fig_h = 183.0, 247.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.010, 0.632, 0.980, 0.358])
    panel_hybrid_codes_map_v10(ax_a, ctx, region_halo=True)
    label(fig, 0.006, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.345, 0.430, 0.265])
    som_figure_context.panel_kinetic_map(ax_b, ctx)
    label(fig, 0.006, 0.625, "b")

    zooms = zoom_windows(ctx)
    ax_c = fig.add_axes([0.510, 0.345, 0.470, 0.270])
    panel_structure_conservation(
        ax_c, ctx, grey=True,
        locators=[(win, REGION_TINTS[r["name"]], sym)
                  for sym, r, win in zooms])
    label(fig, 0.490, 0.625, "c")

    n5 = int((ctx.gene_up == 5).sum())
    facts = {}
    for sym, r, win in zooms:
        genes = set(ctx.assign[ctx.assign.neuron.isin(r["units"])].gene_id)
        a5 = len(set(ctx.gene_up[ctx.gene_up == 5].index) & genes)
        facts[sym] = a5
    titles = {
        "i": f"i · IFN / ISG core — {facts['i']} of {n5} all-five-conserved "
             f"genes",
        "ii": "ii · inflammatory & stress — WNV-restricted stress unit 648",
        "iii": "iii · metabolic & secretory — DENV-restricted unit 32 · "
               "phase ③",
    }
    gs_d = fig.add_gridspec(1, 3, left=0.030, right=0.970, top=0.270,
                            bottom=0.030, wspace=0.30,
                            width_ratios=[win[1] - win[0] for _, _, win in
                                          zooms])
    for j, (sym, r, win) in enumerate(zooms):
        ax = fig.add_subplot(gs_d[j])
        panel_structure_zoom(ax, ctx, sym, r, win)
        ax.set_title(titles[sym], fontsize=4.8, color=darken(
            REGION_TINTS[r["name"]], 0.8), pad=3)
    label(fig, 0.006, 0.310, "d")

    out = FIGDIR / "fig6_v15_zoom.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v15 zoom     : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))
    for sym, r, win in zooms:
        print(f"  zoom {sym}: {r['name'][:40]:42} window x "
              f"{win[0]:.1f}..{win[1]:.1f}  y {win[2]:.1f}..{win[3]:.1f}")


def zoom_key_entries(ctx, window):
    grid = ctx.grid
    x0, x1, y0, y1 = window
    out = []
    for n in sorted(ctx.sup):
        if n not in CALLOUTS:
            continue
        x, y = grid.centres[n]
        if x0 < x < x1 and y0 < y < y1:
            genes = ", ".join(CALLOUTS[n][1].split(", ")[:2])
            out.append((n, genes))
    return out


def zoom_callout_panel(ax_z, ctx, region, window, key_entries, key_w=8.0):
    grid, cb = ctx.grid, ctx.cb
    x0, x1, y0, y1 = window
    counts_stat = ctx.stats["n_genes"]
    cnorm = count_norm(counts_stat)
    conds = list(cb.columns)
    colors = som_figure_context.virus_colors()
    k, lo, hi = som_figure_context.fan_scale(cb)
    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    up4 = ctx.gene_up[ctx.gene_up >= 4]
    up5 = ctx.gene_up[ctx.gene_up == 5]
    bcounts = neuron_of.reindex(up4.index).value_counts()
    bmarked = set(neuron_of.reindex(up5.index).dropna().astype(int))
    sup = set(ctx.sup)

    for n, (x, y) in grid.centres.items():
        if x0 - 1.1 < x < x1 + 1.1 and y0 - 1.1 < y < y1 + 1.1:
            fc = COUNT_CMAP(cnorm(counts_stat[n]))
            grid.hexagon(ax_z, n, fc=fc, ec="0.55", lw=0.15, zorder=1)
            som_figure_context.fan_glyph(ax_z, x, y, cb.loc[n].to_numpy(), k=k, conds=conds,
                          colors=colors, lw=0.12, ring=False)
    for n, c in bcounts.items():
        n = int(n)
        if n not in sup:
            continue
        x, y = grid.centres[n]
        if not (x0 < x < x1 and y0 < y < y1):
            continue
        share = int(ctx.n_shared.get(n, 1))
        kw = dict(fc="white", ec="#1F2328", lw=0.45) if n in bmarked else \
            dict(fc=DOT_RAMP[share - 1], ec="#1F2328", lw=0.3)
        ax_z.scatter([x], [y - 1.08], s=1.0 + 0.3 * int(c), zorder=5, **kw)
    for n in som_figure_context.RESTRICTED:
        x, y = grid.centres[n]
        if x0 < x < x1 and y0 < y < y1:
            ax_z.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.12), closed=True,
                                   fc="none", ec=RING_RESTRICTED, lw=1.2,
                                   zorder=8))
    for n in sorted(sup):
        x, y = grid.centres[n]
        if x0 < x < x1 and y0 < y < y1:
            ax_z.text(x, y - 1.62, str(n), ha="center", va="top",
                      fontsize=4.0, fontweight="bold", color="#1F2328",
                      zorder=6,
                      bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.15))
    ky = y1 + 0.2
    for n, genes in key_entries:
        ax_z.text(x1 + 0.8, ky, f"{n}:", fontsize=4.2, fontweight="bold",
                  color="#1F2328", ha="left", va="top")
        ax_z.text(x1 + 3.9, ky, genes, fontsize=4.0, fontstyle="italic",
                  color="0.30", ha="left", va="top")
        ky -= 1.6
    ax_z.set_xlim(x0 - 0.4, x1 + key_w)
    ax_z.set_ylim(y0 - 0.4, y1 + 0.4)
    ax_z.set_aspect("equal")
    ax_z.set_xticks([])
    ax_z.set_yticks([])
    for sp in ax_z.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(darken(REGION_TINTS[region["name"]], 0.8))
        sp.set_linewidth(1.1)


def panel_atlas_locators(ax, ctx):
    panel_hybrid_codes_map_v10(ax, ctx, region_halo=True)
    for sym, r, (x0, x1, y0, y1) in zoom_windows(ctx):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none",
                               ec=darken(REGION_TINTS[r["name"]], 0.7),
                               lw=1.0, ls=(0, (3.0, 2.0)), zorder=9))
        ax.text(x0 + 0.4, y1 + 0.4, sym, fontsize=5.4, fontweight="bold",
                color=darken(REGION_TINTS[r["name"]], 0.7), ha="left",
                va="bottom", zorder=9)


def panel_atlas_callouts(ax, ctx):
    panel_atlas_locators(ax, ctx)
    grid = ctx.grid
    zooms = zoom_windows(ctx)

    def nearest_region(n):
        x, y = grid.centres[n]
        d = [min((x - grid.centres[m][0]) ** 2 + (y - grid.centres[m][1]) ** 2
                 for m in r["units"]) for _, r, _ in zooms]
        return zooms[int(np.argmin(d))][0]

    per_box = {sym: [] for sym, _, _ in zooms}
    for n in som_figure_context.NAMED_UNSUPPORTED:
        if n in grid.centres:
            per_box[nearest_region(n)].append(n)

    char_w = 0.42
    strip_top = grid.ylim[0] - 3.3
    x_left = ax.get_xlim()[0] + 1.0

    def draw_box(sym, r, x_box, y_top):
        tint = REGION_TINTS[r["name"]]
        named = [n for n in CALLOUTS if n in set(r["units"])]
        named.sort(key=lambda n: -grid.centres[n][1])
        entries = []
        for n in named:
            head, genes = CALLOUTS[n]
            if genes == "technical":
                genes_full = ""
            else:
                genes_full = ", ".join(g for tok in genes.split(",")
                                       for g in som_figure_context.expand_gene_token(tok))
            entries.append((str(n), f"{head} ({int(ctx.n_genes[n])} genes)",
                            genes_full, bool(som_figure_context.RESTRICTED.get(n)), True))
        for n in sorted(per_box[sym], key=lambda n: -grid.centres[n][1]):
            entries.append((f"{n}†", f"{som_figure_context.NAMED_UNSUPPORTED[n]} "
                            f"({int(ctx.n_genes[n])} genes)", "", False,
                            False))
        longest = max(1.0 + 5.4 + len(mid) * char_w + 1.7
                      + len(genes) * 0.40 for _, mid, genes, _, _ in entries)
        longest = max(longest, len(f"{sym} · {REGION_SHORT[r['name']]}")
                      * 0.46 + 2.0)
        box_w = longest + 2.2
        n_lines = 2 + len(entries) + (1 if per_box[sym] else 0)
        box_h = 2.6 + 1.55 * n_lines
        ax.add_patch(Rectangle((x_box, y_top - box_h), box_w, box_h,
                               fc="white", ec=darken(tint, 0.8), lw=1.0,
                               zorder=10))
        ax.text(x_box + 1.0, y_top - 1.3,
                f"{sym} · {REGION_SHORT[r['name']]}", fontsize=4.6,
                fontweight="bold", color=darken(tint, 0.8), ha="left",
                va="center", zorder=11)
        ax.text(x_box + 1.0, y_top - 2.9,
                f"{len(r['units'])} programmes · {r['n_genes']:,} genes",
                fontsize=4.0, color="0.40", fontstyle="italic", ha="left",
                va="center", zorder=11)
        y = y_top - 4.5
        for num, mid, genes, restricted, supported in entries:
            num_colour = "#1F2328" if supported else "0.45"
            ax.text(x_box + 1.0, y, num, fontsize=4.2, fontweight="bold",
                    color=num_colour, ha="left", va="center", zorder=11)
            ax.text(x_box + 5.4, y, mid, fontsize=4.2, color="#1F2328",
                    ha="left", va="center", zorder=11)
            if genes:
                ax.text(x_box + 5.4 + len(mid) * char_w + 1.7, y,
                        "— " + genes, fontsize=4.0, fontstyle="italic",
                        color="#B4231C" if restricted else "0.35", ha="left",
                        va="center", zorder=11)
            y -= 1.55
        if per_box[sym]:
            ax.text(x_box + 1.0, y, "† below the support threshold",
                    fontsize=3.9, color="0.45", ha="left", va="center",
                    zorder=11)
        return box_w, box_h

    w_i, h_i = draw_box("i", zooms[0][1], x_left, strip_top)
    x_right = x_left + w_i + 2.5
    w_ii, h_ii = draw_box("ii", zooms[1][1], x_right, strip_top)
    w_iii, h_iii = draw_box("iii", zooms[2][1], x_right,
                            strip_top - h_ii - 2.0)
    box_bottoms = [strip_top - h_i, strip_top - h_ii,
                   strip_top - h_ii - 2.0 - h_iii]
    if 56 in CALLOUTS and all(56 not in set(r["units"]) for _, r, _ in zooms):
        ax.text(x_right, strip_top - h_ii - 2.0 - h_iii - 1.6,
                f"56  rRNA / MT-RNR ({int(ctx.n_genes[56])} genes) - "
                f"technical; sits outside the three regions",
                fontsize=3.9, color="0.45", ha="left", va="top", zorder=11)
    ax.set_ylim(min(box_bottoms) - 2.0, ax.get_ylim()[1])


def compose_v17(ctx):
    fig_w, fig_h = 183.0, 247.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.008, 0.300, 0.984, 0.690])
    panel_atlas_callouts(ax_a, ctx)
    label(fig, 0.004, 0.996, "a")

    ax_b = fig.add_axes([0.020, 0.020, 0.430, 0.255])
    som_figure_context.panel_kinetic_map(ax_b, ctx)
    label(fig, 0.006, 0.288, "b")

    ax_c = fig.add_axes([0.510, 0.020, 0.470, 0.255])
    panel_structure_conservation(ax_c, ctx, grey=True, regions=False)
    label(fig, 0.490, 0.288, "c")

    out = FIGDIR / "fig6_v17_region_callouts.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v17 callouts : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))
    for sym, r, _ in zoom_windows(ctx):
        named = [n for n in CALLOUTS if n in set(r["units"])]
        print(f"  box {sym}: {REGION_SHORT[r['name']]:28} named units "
              f"{sorted(named)}")


BOX_PASTELS = {
    "Interferon, ISG and antigen-presentation core":
        (som_figure_context.lighten("#8E7CC3", 0.55), "#8E7CC3"),
    "Inflammatory, stress and immediate-early":
        (som_figure_context.lighten("#6FA8DC", 0.55), "#6FA8DC"),
    "Hepatocyte metabolic and secretory":
        (som_figure_context.lighten("#C2A15C", 0.55), "#C2A15C"),
}


def _box_fill(ctx, n, restricted_edge=True):
    rname = som_figure_context.region_of(ctx, n)
    fc, ec = BOX_PASTELS.get(rname, (som_figure_context.lighten("#B8BCC2", 0.55), "#B8BCC2"))
    if restricted_edge and n in som_figure_context.RESTRICTED:
        ec = RING_RESTRICTED
    return fc, ec


@lru_cache(maxsize=None)
def _max_group_tpm(ctx):
    degs = pd.read_parquet(som_figure_context.PUBLISH / "data" / "degs.parquet")
    return degs.groupby("gene_id")["maxgroupmeans"].max()


def _box_lines(ctx, n, per_box):
    tpm = _max_group_tpm(ctx)
    sym = ctx.sym.symbol
    named = [g for g in ctx.assign.loc[ctx.assign.neuron == n, "gene_id"]
             if isinstance(sym.get(g), str) and sym.get(g) != g]
    ranked = tpm.reindex(named).dropna().sort_values(ascending=False)
    shown = [sym[g] for g in ranked.index[:per_box]]
    if int(ctx.n_genes[n]) > len(shown):
        shown = shown + ["…"]
    return shown


def _box_depth(ctx, n, per_box):
    return 3.0 + len(_box_lines(ctx, n, per_box)) * 1.35


def _orphan640_lines(ctx):
    degs = pd.read_parquet(som_figure_context.PUBLISH / "data" / "degs.parquet")
    mem = ctx.assign[ctx.assign.neuron == 640].gene_id
    d = degs[degs.gene_id.isin(mem)]
    n_genes = len(mem)
    n_de24 = int(d[(d.hpi == 24) & d.deg].shape[0])
    up72 = d[(d.hpi == 72) & (d.treatment == "untreated") & (d.log2fc > 1)]
    n72_all5 = int((up72.groupby("gene_id")["virus"].nunique() == 5).sum())
    unnamed = int(sum(s == g for s, g in
                      zip(ctx.sym.symbol.reindex(mem), mem)))
    print(f"  v21 orphan 640: {n_genes} members ({unnamed} unnamed), "
          f"DE rows at 24 hpi: {n_de24}, induced at 72 hpi in all five: "
          f"{n72_all5} of {n_genes}")
    return ["orphan late wave"]


def neuron_boxes_at_units(ax, ctx, units, tx, ha, per_box=4, gap=0.3,
                          glyphs=False, glyphs_out=False, marks=True):
    from matplotlib.transforms import Bbox

    if glyphs or glyphs_out:
        cb = ctx.cb
        conds = paper_conds(cb.columns)
        colors = som_figure_context.virus_colors()
        k, _, hi = som_figure_context.fan_scale(cb)
        mag = 1.8
        r_glyph = mag * (FAN_R0 + k * hi)

    drawn = []
    cursor = None
    for n in sorted(units, key=lambda m: -ctx.grid.centres[m][1]):
        uy = ctx.grid.centres[n][1]
        top = uy + 1.0 if cursor is None else min(uy + 1.0, cursor)
        if n == 640:
            lines = _orphan640_lines(ctx)
        else:
            lines = _box_lines(ctx, n, per_box)
        depth = 0.9 + len(lines) * 0.8 + 0.5
        header = "Neuron 640†" if n == 640 else f"Neuron {n}"
        head_t = ax.text(tx, top, header, fontsize=4.0,
                         fontweight="bold", color="#1F2328", ha=ha,
                         va="top", zorder=11)
        body_t = ax.text(tx, top - 0.9, "\n".join(lines), fontsize=3.6,
                         fontstyle="italic", color="0.25", ha=ha, va="top",
                         linespacing=1.3, zorder=11)
        drawn.append((n, head_t, body_t))
        cursor = top - depth - gap
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    for n, head_t, body_t in drawn:
        union = Bbox.union([head_t.get_window_extent(),
                            body_t.get_window_extent()])
        (x0, y0), (x1, y1) = inv.transform(union)
        if glyphs:
            gx = x1 + 0.5 + r_glyph if ha == "left" else x0 - 0.5 - r_glyph
            gy = (y0 + y1) / 2
            x0, x1 = min(x0, gx - r_glyph), max(x1, gx + r_glyph)
            y0, y1 = min(y0, gy - r_glyph), max(y1, gy + r_glyph)
        fc, ec = _box_fill(ctx, n, restricted_edge=marks)
        ls = "solid"
        if n == 640:
            fc = som_figure_context.lighten("#B8BCC2", 0.55)
            ec, ls = "#3F4550", (0, (1.5, 1.2))
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.3,"
                                             "rounding_size=0.5",
                                    fc=fc, ec=ec, lw=0.55, ls=ls,
                                    zorder=10, clip_on=False))
        if glyphs_out:
            gx = x1 + 0.8 + r_glyph if ha == "left" else x0 - 0.8 - r_glyph
            gy = (y0 + y1) / 2
        if glyphs or glyphs_out:
            npatches, nlines = len(ax.patches), len(ax.lines)
            som_figure_context.fan_glyph(ax, gx, gy, cb.loc[n, conds].to_numpy(), k=k * mag,
                          conds=conds, colors=colors, r0=FAN_R0 * mag,
                          lw=0.12, ring=True)
            for p in ax.patches[npatches:]:
                p.set_zorder(11)
            for ln in ax.lines[nlines:]:
                ln.set_zorder(11)
    return cursor


def panel_atlas_neuron_boxes(ax, ctx, side_pad=14.0, right_pad=14.0,
                             glyphs=False, glyphs_out=False, plain_units=(),
                             named_key=True, orphan640=False,
                             top_stats=True, simple=False):
    grid = ctx.grid
    panel_hybrid_codes_map_v10(
        ax, ctx, side_pad=side_pad, right_pad=right_pad, region_halo=True,
        left_key="none",
        fan_origin=(grid.xlim[1] - 9.0, grid.ylim[1] + 6.4),
        drop_rings=plain_units, thin_regions=simple,
        class_outlines=not simple, marks=not simple)

    if top_stats:
        total = int(ctx.n_genes.reindex(ctx.sup).sum())
        n_ind = int((ctx.n_up >= 1).sum())
        n_all = int((ctx.n_up == 5).sum())
        n_one = int((ctx.n_up == 1).sum())
        ax.text(grid.xlim[0] - side_pad + 0.6, grid.ylim[1] + 12.2,
                f"{len(ctx.sup)} programmes · {total:,} genes", fontsize=4.8,
                fontweight="bold", color="#1F2328", ha="left", va="center",
                zorder=11)
        ax.text(grid.xlim[0] - side_pad + 0.6 + 16.5, grid.ylim[1] + 12.2,
                f"{n_ind} induced · {n_all} by all five · {n_one} "
                f"virus-restricted", fontsize=4.0, color="0.35", ha="left",
                va="center", zorder=11)

    yb = grid.ylim[0] - 2.3
    short = {"ifn": "IFN ligands", "sense": "Sensing/IRF3",
             "isgf3": "ISGF3", "antigen": "Antigen", "inflam": "Inflammatory",
             "metab": "Metabolic", "tech": "rRNA/MT"}
    if not simple:
        ax.text(0.5, yb, "class", fontsize=4.0, fontweight="bold",
                color="#1F2328", ha="left", va="center", zorder=11)
        x = 5.5
        for key in CLASSES:
            ax.add_patch(Rectangle((x, yb - 0.45), 1.3, 0.9,
                                   fc=CLASSES[key][1], ec="white", lw=0.3))
            ax.text(x + 1.8, yb, short[key], fontsize=3.8, va="center",
                    ha="left", color="#1F2328", zorder=11)
            x += 1.8 + (len(short[key]) * 0.58 + 1.0)
        yb -= 1.9
    ax.text(0.5, yb, "regions", fontsize=4.0, fontweight="bold",
            color="#1F2328", ha="left", va="center", zorder=11)
    x = 7.0
    for r in ctx.regions:
        tint = REGION_TINTS[r["name"]]
        if simple:
            ax.add_patch(Polygon(som_figure_context.hex_ring(x + 0.65, yb, 0.75),
                                 closed=True, fc=tint, alpha=0.3, ec="none",
                                 zorder=11))
            ax.add_patch(Polygon(som_figure_context.hex_ring(x + 0.65, yb, 0.75),
                                 closed=True, fc="none", ec=darken(tint),
                                 lw=1.2, zorder=11))
        else:
            ax.add_patch(Rectangle((x, yb - 0.45), 1.3, 0.9, fc=tint,
                                   ec=tint, lw=0.5, alpha=0.45))
        ax.text(x + 1.8, yb, REGION_SHORT_V10[r["name"]], fontsize=3.8,
                va="center", ha="left", color="#1F2328", zorder=11)
        x += 1.8 + (len(REGION_SHORT_V10[r["name"]]) * 0.58 + 2.6)
    ring_rows = [
        [entry for entry in
         [("#1F2328", "-", "named in Results (Table SX)"),
          (RING_RESTRICTED, "-", "virus-restricted")]
         if named_key or "named" not in entry[2]],
        [(RING_IFN, "-", "interferon ligands"),
         ("#3F4550", (0, (1.5, 1.2)), "below threshold")],
    ]
    yb -= 1.9
    for entries in ring_rows if not simple else ():
        x = 0.5
        for colour, ls, lab in entries:
            ax.add_patch(Polygon(som_figure_context.hex_ring(x + 0.6, yb, 0.75), closed=True,
                                 fc="none", ec=colour, lw=0.8, ls=ls,
                                 zorder=11))
            ax.text(x + 1.6, yb, lab, fontsize=3.8, va="center", ha="left",
                    color="#1F2328", zorder=11)
            x += 1.6 + len(lab) * 0.58 + 3.0
        yb -= 1.6

    b1 = neuron_boxes_at_units(ax, ctx, [648, 650, 594],
                               grid.xlim[0] - 1.0, "right", per_box=4,
                               glyphs=glyphs, glyphs_out=glyphs_out,
                               marks=not simple)
    b1m = neuron_boxes_at_units(ax, ctx, [93, 59, 32, 34],
                                grid.xlim[0] - 1.0, "right", per_box=3,
                                glyphs=glyphs, glyphs_out=glyphs_out,
                                marks=not simple)
    b2 = neuron_boxes_at_units(
        ax, ctx, [n for n in [756, 728, 727, 726, 725, 695, 666, 668, 582,
                              56] if n not in plain_units]
                 + ([640] if orphan640 and not simple else []),
        grid.xlim[1] + 1.0, "left", per_box=4,
        glyphs=glyphs, glyphs_out=glyphs_out, marks=not simple)
    bottom = min(b1, b1m, b2, yb - 1.0) - 1.2
    ax.set_ylim(bottom, ax.get_ylim()[1])
    return bottom


def compose_v18(ctx):
    fig_w = 183.0
    fig = plt.figure(figsize=(fig_w * MM, 247.0 * MM))

    ax_a = fig.add_axes([0.008, 0.300, 0.984, 0.690])
    bottom = panel_atlas_neuron_boxes(ax_a, ctx)

    ax_b = fig.add_axes([0.020, 0.020, 0.430, 0.255])
    som_figure_context.panel_kinetic_map(ax_b, ctx)
    ax_c = fig.add_axes([0.510, 0.020, 0.470, 0.255])
    panel_structure_conservation(ax_c, ctx, grey=True, regions=False)

    xspan = ax_a.get_xlim()[1] - ax_a.get_xlim()[0]
    scale = 0.984 * fig_w / xspan
    span = ax_a.get_ylim()[1] - bottom
    a_mm = span * scale
    row2_mm = min(75.0, 247.0 - 12.0 - a_mm)
    fig_h = a_mm + 7.0 + row2_mm + 5.0
    fig.set_size_inches(fig_w * MM, fig_h * MM)
    a_frac = a_mm / fig_h
    a_bottom = 1.0 - 0.030 - a_frac
    ax_a.set_position([0.008, a_bottom, 0.984, a_frac])
    r2_h = row2_mm / fig_h
    r2_y = a_bottom - 0.025 - r2_h
    ax_b.set_position([0.020, r2_y, 0.430, r2_h])
    ax_c.set_position([0.510, r2_y, 0.470, r2_h])
    label(fig, 0.004, 0.995, "a")
    label(fig, 0.006, a_bottom - 0.012, "b")
    label(fig, 0.490, a_bottom - 0.012, "c")

    out = FIGDIR / "fig6_v18_neuron_boxes.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v18 boxes    : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def panel_virus_strip(fig, ctx, top_mm, map_scale=0.34,
                      title="all five viruses occupy the same territory — "
                            "untreated infection vs Mock"):
    grid = ctx.grid
    norm, _ = som_figure_context.shared_terrain_scale(ctx.cb)
    cmap = mpl.colormaps["RdBu_r"]
    colors = som_figure_context.virus_colors()
    fig_w, fig_h = fig.get_size_inches()
    fig_w /= MM
    fig_h /= MM
    fx = lambda mm: mm / fig_w
    fy = lambda mm: 1.0 - mm / fig_h

    def rect(top, height, left, width):
        return [fx(left), 1.0 - (top + height) / fig_h, fx(width),
                height / fig_h]

    span_x = grid.xlim[1] - grid.xlim[0] + 2.0
    span_y = grid.ylim[1] - grid.ylim[0] + 2.0
    map_w = span_x * map_scale
    map_h = span_y * map_scale

    if title:
        fig.text(0.5, fy(top_mm + 1.6), title, fontsize=4.6, ha="center",
                 va="center", color="#1F2328")
    label_w, gap, cbar_w = 6.0, 1.6, 12.0
    strip_w = label_w + 5 * map_w + 4 * gap + cbar_w
    x0 = (fig_w - strip_w) / 2
    row_h = map_h + 2.5
    for r, hpi in enumerate((24, 72)):
        for c, v in enumerate(VIRUS_ORDER):
            axm = fig.add_axes(rect(top_mm + 6.0 + r * row_h, map_h,
                                    x0 + label_w + c * (map_w + gap), map_w))
            for n, (x, y) in grid.centres.items():
                z = ctx.cb.loc[n, f"{v}_untreated{hpi}"]
                axm.add_patch(Polygon(som_figure_context.hex_ring(x, y, 0.98), closed=True,
                                      fc=cmap(norm(z)), ec="white", lw=0.05,
                                      zorder=2))
            som_figure_context.draw_territory(axm, ctx.induced_units, grid, "#3F4550",
                               lw=0.3, z=6)
            axm.set_xlim(grid.xlim[0] - 1.0, grid.xlim[1] + 1.0)
            axm.set_ylim(grid.ylim[0] - 1.0, grid.ylim[1] + 1.0)
            axm.set_aspect("equal")
            axm.axis("off")
            if r == 0:
                axm.set_title(v, fontsize=4.2, fontweight="bold",
                              color=colors[v], pad=1.5)
        fig.text(fx(x0 + 1.0), fy(top_mm + 6.0 + r * row_h + map_h / 2),
                 f"{hpi} hpi", fontsize=4.2, ha="center", va="center",
                 rotation=90, color="#1F2328")

    cax = fig.add_axes(rect(top_mm + 6.0, 2 * map_h + 2.5,
                            x0 + label_w + 5 * map_w + 4 * gap + 3.0, 1.6))
    bar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                    ticks=[-6, 0, 6], extend="both")
    cax.tick_params(labelsize=3.6, length=1.2, pad=1)
    bar.outline.set_linewidth(0.4)
    cax.set_title("codebook weight (z),\nwinsorized ±6", fontsize=3.7, pad=2)
    return 6.0 + 2 * row_h + 2.0


def panel_kinetic_map_clean(ax, ctx, title=True):
    grid = ctx.grid
    faces = {}
    for n, row in ctx.kin.iterrows():
        key = (row["direction"], row["kinetic_class"])
        if key in som_figure_context.KIN_COLORS:
            faces[int(n)] = som_figure_context.KIN_COLORS[key]
    som_figure_context.hex_field(ax, grid, faces, lw=0.10)
    grid.frame(ax)
    ax.set_ylim(grid.ylim[0] - som_figure_context.MAP_STRIP, grid.ylim[1] + 1.6)

    counts = ctx.kin.groupby(["direction", "kinetic_class"]).size()
    order = [("induced", "early"), ("induced", "sustained"),
             ("induced", "late"), ("repressed", "early"),
             ("repressed", "sustained"), ("repressed", "late")]
    names = {"early": "24 hpi only", "sustained": "both",
             "late": "72 hpi only"}
    x0 = grid.xlim[0] + 0.4
    y0 = grid.ylim[0] - (7.4 if title else 4.0)
    if title:
        ax.text(x0, y0 + 3.6, "neurons by direction and kinetics",
                fontsize=4.9, fontweight="bold", ha="left", va="center")
    for i, key in enumerate(order):
        col, row_i = divmod(i, 3)
        x = x0 + col * 25.5
        y = y0 - row_i * 2.8
        ax.add_patch(Rectangle((x, y - 0.75), 1.7, 1.5,
                               fc=som_figure_context.KIN_COLORS[key], ec="white", lw=0.3))
        ax.text(x + 2.4, y, f"{key[0]}, {names[key[1]]}  "
                f"({counts.get(key, 0)})", fontsize=4.5, va="center",
                ha="left")


def compose_v19(ctx):
    fig_w = 183.0
    fig = plt.figure(figsize=(fig_w * MM, 247.0 * MM))

    ax_a = fig.add_axes([0.008, 0.300, 0.984, 0.690])
    bottom = panel_atlas_neuron_boxes(ax_a, ctx, side_pad=18.0,
                                      right_pad=18.0)

    xspan = ax_a.get_xlim()[1] - ax_a.get_xlim()[0]
    scale = 0.984 * fig_w / xspan
    span = ax_a.get_ylim()[1] - bottom
    a_mm = span * scale

    strip_mm = 6.0 + 2 * ((41.3 + 2.0) * 0.28 + 2.5) + 2.0
    row2_mm = min(53.0, 247.0 - 18.0 - a_mm - strip_mm)
    fig_h = a_mm + 7.0 + row2_mm + 6.0 + strip_mm + 5.0
    fig.set_size_inches(fig_w * MM, fig_h * MM)
    a_frac = a_mm / fig_h
    a_bottom = 1.0 - 0.030 - a_frac
    ax_a.set_position([0.008, a_bottom, 0.984, a_frac])

    r2_h = row2_mm / fig_h
    r2_y = a_bottom - 0.025 - r2_h
    ax_b = fig.add_axes([0.020, r2_y, 0.430, r2_h])
    panel_kinetic_map_clean(ax_b, ctx)
    label(fig, 0.006, a_bottom - 0.012, "b")
    ax_c = fig.add_axes([0.510, r2_y, 0.470, r2_h])
    panel_structure_conservation(ax_c, ctx, grey=True, regions=False,
                                 restricted=False, legend_v19=True)
    label(fig, 0.490, a_bottom - 0.012, "c")

    strip_top_mm = (1.0 - r2_y) * fig_h + 2.5
    panel_virus_strip(fig, ctx, strip_top_mm, map_scale=0.28)
    label(fig, 0.006, 1.0 - (strip_top_mm + 3.0) / fig_h, "d")

    out = FIGDIR / "fig6_v19_virus_strip.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v19 strip    : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def panel_kinetic_conservation_v20(ax, ctx):
    grid = ctx.grid
    map_span = (grid.ylim[1] + 1.6) - grid.ylim[0]
    rect_h_mm = (ax.get_position().height
                 * ax.figure.get_size_inches()[1] * 25.4)
    scale = (rect_h_mm - 1.0) / map_span

    def u(mmv):
        return mmv / scale

    def yt(mmv):
        return grid.ylim[1] - mmv / scale

    faces = {int(n): som_figure_context.KIN_COLORS[(r["direction"], r["kinetic_class"])]
             for n, r in ctx.kin.iterrows()
             if (r["direction"], r["kinetic_class"]) in som_figure_context.KIN_COLORS}
    som_figure_context.hex_field(ax, grid, faces, lw=0.10)

    all5 = sorted(int(n) for n in ctx.sup if int(ctx.n_up[n]) == 5)
    assert len(all5) == 18, f"expected 18 all-five programmes, got {len(all5)}"
    print(f"  v20 all-five programmes ({len(all5)}): {all5}")
    for n in all5:
        x, y = grid.centres[n]
        for r in (1.05, 1.22):
            ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, r), closed=True,
                                 fc="none", ec="#1F2328", lw=0.8,
                                 zorder=8))

    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    up4 = ctx.gene_up[ctx.gene_up >= 4]
    up5 = ctx.gene_up[ctx.gene_up == 5]
    counts = neuron_of.reindex(up4.index).value_counts()
    marked = set(neuron_of.reindex(up5.index).dropna().astype(int))
    dot_pt2 = _dot_pt2_for_scale(scale)
    sup = set(ctx.sup)
    for n, c in counts.items():
        n = int(n)
        if n not in sup:
            continue
        x, y = grid.centres[n]
        share = int(ctx.n_shared.get(n, 1))
        s = max(int(c) * dot_pt2, 0.3)
        if n in marked:
            ax.scatter([x], [y], s=s, fc="white", ec="#1F2328", lw=0.4,
                       zorder=5)
        else:
            ax.scatter([x], [y], s=s, fc=DOT_RAMP[share - 1],
                       ec="#1F2328", lw=0.25, zorder=5)

    grid.frame(ax)
    ax.set_xlim(grid.xlim[0], grid.xlim[1] + u(62.0))
    ax.set_ylim(grid.ylim[0] - u(1.0), grid.ylim[1] + 1.6)
    n_ind = int((ctx.n_up >= 1).sum())
    n_all = int((ctx.n_up == 5).sum())
    ax.set_title(f"kinetics and conservation — {n_all} of {n_ind} induced "
                 f"programmes shared by all five", fontsize=4.8, pad=3)

    lx = grid.xlim[1] + u(5.0)
    bold = dict(fontsize=4.4, fontweight="bold", ha="left", va="center",
                color="#1F2328", zorder=11)
    plain = dict(fontsize=4.2, ha="left", va="center", color="#1F2328",
                 zorder=11)

    kin_counts = ctx.kin.groupby(["direction", "kinetic_class"]).size()
    names = {"early": "24 hpi only", "sustained": "both",
             "late": "72 hpi only"}
    ax.text(lx, yt(2.2), "neurons by direction and kinetics", **bold)
    for i, cls in enumerate(("early", "sustained", "late")):
        for j, direction in enumerate(("induced", "repressed")):
            x = lx + j * u(28.0)
            y = yt(4.9 + i * 2.6)
            ax.add_patch(Rectangle((x, y - u(0.75)), u(1.7), u(1.5),
                                   fc=som_figure_context.KIN_COLORS[(direction, cls)],
                                   ec="white", lw=0.3, zorder=5))
            ax.text(x + u(2.4), y,
                    f"{direction}, {names[cls]}  "
                    f"({kin_counts.get((direction, cls), 0)})", **plain)

    step_counts = {k: int((ctx.n_up == k).sum()) for k in range(1, 6)}
    ax.text(lx, yt(14.2), "viruses agreeing", **bold)
    x = lx + u(17.5)
    for k in range(1, 6):
        ax.text(x + 1.2, yt(14.2), str(k), fontsize=4.2, ha="center",
                va="center", color="#1F2328", zorder=11)
        ax.add_patch(Rectangle((x, yt(15.0) - u(1.15)), 2.4, u(1.15),
                               fc=DOT_RAMP[k - 1], ec="white", lw=0.3,
                               zorder=5))
        if k == 5:
            for pad in (0.35, 0.7):
                ax.add_patch(Rectangle(
                    (x - pad, yt(15.0) - u(1.15) - pad), 2.4 + 2 * pad,
                    u(1.15) + 2 * pad, fc="none", ec="#1F2328", lw=0.7,
                    zorder=6))
        ax.text(x + 1.2, yt(16.5), str(step_counts[k]), fontsize=4.2,
                ha="center", va="top", color="#1F2328", zorder=11)
        x += 3.4

    y2 = yt(19.6)
    ax.text(lx, y2, "genes ≥ 4/5", **bold)
    for dx, c in ((13.5, 1), (16.8, 10), (20.4, 30)):
        s_c = max(c * dot_pt2, 0.3)
        ax.scatter([lx + u(dx)], [y2], s=s_c, fc="0.35",
                   ec="#1F2328", lw=0.3, zorder=5)
        r_mm = (s_c / np.pi) ** 0.5 * 25.4 / 72
        ax.text(lx + u(dx + r_mm + 0.35), y2, str(c), **plain)
    ax.scatter([lx + u(24.2)], [y2], s=max(10 * dot_pt2, 0.3), fc="white",
               ec="#1F2328", lw=0.5, zorder=5)
    ax.text(lx + u(25.2), y2, "all five", **plain)

    y3 = yt(23.0)
    for r in (0.75, 0.95):
        ax.add_patch(Polygon(som_figure_context.hex_ring(lx + u(1.1), y3, r), closed=True,
                             fc="white" if r < 0.8 else "none",
                             ec="#1F2328", lw=0.7, zorder=5))
    ax.text(lx + u(2.6), y3, f"induced by all five viruses ({len(all5)})",
            **plain)


def compose_v20(ctx):
    fig_w = 183.0
    fig = plt.figure(figsize=(fig_w * MM, 247.0 * MM))

    ax_a = fig.add_axes([0.008, 0.300, 0.984, 0.690])
    bottom = panel_atlas_neuron_boxes(ax_a, ctx, side_pad=18.0,
                                      right_pad=18.0, glyphs=True)

    xspan = ax_a.get_xlim()[1] - ax_a.get_xlim()[0]
    scale = 0.984 * fig_w / xspan
    span = ax_a.get_ylim()[1] - bottom
    a_mm = span * scale

    strip_mm = 6.0 + 2 * ((41.3 + 2.0) * 0.28 + 2.5) + 2.0
    b_mm = min(53.0, 247.0 - 18.0 - a_mm - strip_mm)
    fig_h = a_mm + 7.0 + b_mm + 6.0 + strip_mm + 5.0
    fig.set_size_inches(fig_w * MM, fig_h * MM)
    a_frac = a_mm / fig_h
    a_bottom = 1.0 - 0.030 - a_frac
    ax_a.set_position([0.008, a_bottom, 0.984, a_frac])

    b_h = b_mm / fig_h
    b_y = a_bottom - 0.025 - b_h
    ax_b = fig.add_axes([0.008, b_y, 0.984, b_h])
    panel_kinetic_conservation_v20(ax_b, ctx)
    label(fig, 0.006, a_bottom - 0.012, "b")

    strip_top_mm = (1.0 - b_y) * fig_h + 2.5
    panel_virus_strip(fig, ctx, strip_top_mm, map_scale=0.28)
    label(fig, 0.006, 1.0 - (strip_top_mm + 3.0) / fig_h, "c")

    out = FIGDIR / "fig6_v20_unified.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v20 unified  : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def panel_conservation_v21(ax, ctx):
    grid = ctx.grid
    legend_w_mm = CONS_LEGEND_MM
    map_span = (grid.ylim[1] + 1.6) - grid.ylim[0]
    pos = ax.get_position()
    rect_h_mm = pos.height * ax.figure.get_size_inches()[1] * 25.4
    rect_w_mm = pos.width * ax.figure.get_size_inches()[0] * 25.4
    scale = min(rect_h_mm / map_span,
                (rect_w_mm - legend_w_mm)
                / (grid.xlim[1] - grid.xlim[0]))

    def u(mmv):
        return mmv / scale

    def yt(mmv):
        return grid.ylim[1] - mmv / scale

    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    up4 = ctx.gene_up[ctx.gene_up >= 4]
    up5 = ctx.gene_up[ctx.gene_up == 5]
    counts = neuron_of.reindex(up4.index).value_counts()
    marked = set(neuron_of.reindex(up5.index).dropna().astype(int))
    sup = set(ctx.sup)

    cmap = mpl.colormaps["BuGn"]
    top = int(counts.max())
    norm = mpl.colors.Normalize(vmin=0, vmax=top)
    faces = {int(n): cmap(norm(int(c))) for n, c in counts.items()
             if int(n) in sup}
    som_figure_context.hex_field(ax, grid, faces, lw=0.10)

    for n in sup:
        share = int(ctx.n_up[n])
        if share < 1:
            continue
        x, y = grid.centres[n]
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.0), closed=True,
                             fc="none", ec=AGRE_RAMP[share - 1], lw=0.6,
                             zorder=6))

    all5 = sorted(int(n) for n in ctx.sup if int(ctx.n_up[n]) == 5)
    assert len(all5) == 18, f"expected 18 all-five programmes, got {len(all5)}"
    for n in all5:
        x, y = grid.centres[n]
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.18), closed=True,
                             fc="none", ec="#4A4F57", lw=0.55, zorder=8))

    for n in marked:
        if n not in sup:
            continue
        x, y = grid.centres[n]
        ax.plot([x], [y], "o", ms=1.8, mfc="white", mec="#1F2328",
                mew=0.35, zorder=7)

    dn5_genes = ctx.gene_dn[ctx.gene_dn == 5]
    assert len(dn5_genes) == 11, f"expected 11, got {len(dn5_genes)}"
    dn_units = neuron_of.reindex(dn5_genes.index).dropna().astype(int)
    dn_set = sorted(set(dn_units))
    print(f"  v21 repressed-all-five: {len(dn5_genes)} genes in units "
          f"{dn_set} ({dn_units.value_counts().to_dict()}), supported: "
          f"{sorted(set(dn_set) & sup)}")
    for n in dn_set:
        x, y = grid.centres[n]
        if n in sup:
            ax.plot([x], [y], marker="v", ms=2.2, mfc="#1F4E79",
                    mec="#1F2328", mew=0.3, ls="none", zorder=7)
        else:
            ax.plot([x], [y], marker="v", ms=2.2, mfc="none",
                    mec="#1F4E79", mew=0.5, ls="none", zorder=7)

    grid.frame(ax)
    ax.set_xlim(grid.xlim[0], grid.xlim[1] + u(legend_w_mm))
    ax.set_ylim(grid.ylim[0], grid.ylim[1] + 1.6)

    lx = grid.xlim[1] + u(4.0)
    bold = dict(fontsize=4.4, fontweight="bold", ha="left", va="center",
                color="#1F2328", zorder=11)
    plain = dict(fontsize=4.2, ha="left", va="center", color="#1F2328",
                 zorder=11)

    ax.text(lx, yt(2.2), "member genes induced", **bold)
    ax.text(lx, yt(3.8), "in ≥ 4 of 5 viruses", **bold)
    bw = u(16.0)
    for i in range(40):
        ax.add_patch(Rectangle((lx + i * bw / 40, yt(5.9) - u(1.2)),
                               bw / 40, u(1.2), fc=cmap(i / 39), ec="none",
                               zorder=5))
    ax.add_patch(Rectangle((lx, yt(5.9) - u(1.2)), bw, u(1.2), fc="none",
                           ec="0.3", lw=0.5, zorder=5))
    for t in sorted({t for t in (0, 10, 20, 30) if t < top} | {top}):
        ax.text(lx + bw * t / top, yt(7.5), str(t), fontsize=4.0,
                ha="center", va="top", color="#1F2328", zorder=11)

    step_counts = {k: int((ctx.n_up == k).sum()) for k in range(1, 6)}
    ax.text(lx, yt(10.8), "programme induced in N of 5 viruses", **bold)
    x = lx
    for k in range(1, 6):
        ax.text(x + 1.2, yt(12.5), str(k), fontsize=4.2, ha="center",
                va="center", color="#1F2328", zorder=11)
        ax.add_patch(Polygon(som_figure_context.hex_ring(x + 1.2, yt(14.2), 0.75),
                             closed=True, fc="white", ec=AGRE_RAMP[k - 1],
                             lw=0.6, zorder=5))
        if k == 5:
            ax.add_patch(Polygon(
                som_figure_context.hex_ring(x + 1.2, yt(14.2), 0.95), closed=True,
                fc="none", ec="#4A4F57", lw=0.55, zorder=5))
        ax.text(x + 1.2, yt(15.4), str(step_counts[k]), fontsize=4.2,
                ha="center", va="top", color="#1F2328", zorder=11)
        x += 3.4

    y3 = yt(18.2)
    ax.plot([lx + u(1.1)], [y3], "o", ms=1.8, mfc="white", mec="#1F2328",
            mew=0.35, zorder=5)
    ax.text(lx + u(2.6), y3, "gene induced by all five viruses", **plain)

    y4 = yt(20.6)
    ax.add_patch(Polygon(som_figure_context.hex_ring(lx + u(1.1), y4, 0.85), closed=True,
                         fc="none", ec="#4A4F57", lw=0.55, zorder=5))
    ax.text(lx + u(2.6), y4, f"programme induced by all five viruses "
            f"({len(all5)})", **plain)

    y5 = yt(23.2)
    ax.plot([lx + u(1.1)], [y5], marker="v", ms=2.2, mfc="#1F4E79",
            mec="#1F2328", mew=0.3, ls="none", zorder=5)
    ax.text(lx + u(2.6), y5, f"genes repressed by all five viruses "
            f"({len(dn5_genes)})", **plain)
    ax.text(lx + u(2.6), yt(24.9), "open = below threshold", fontsize=4.0,
            ha="left", va="center", color="0.35", zorder=11)

    n_ind = int((ctx.n_up >= 1).sum())
    print(f"  v21 conservation : {len(up4)} genes induced by >= 4/5 viruses, "
          f"{len(up5)} by all five, {int((ctx.gene_dn == 5).sum())} repressed "
          f"by all five; {len(all5)} of {n_ind} induced programmes shared by "
          f"all five (caption numbers, no longer drawn)")


def panel_composition_clone_v21(ax, ctx):
    grid = ctx.grid
    pos = ax.get_position()
    rect_h_mm = pos.height * ax.figure.get_size_inches()[1] * 25.4
    rect_w_mm = pos.width * ax.figure.get_size_inches()[0] * 25.4
    legend_mm = COMP_LEGEND_MM
    scale = min((rect_h_mm - legend_mm)
                / ((grid.ylim[1] + 1.6) - grid.ylim[0]),
                rect_w_mm / (grid.xlim[1] - grid.xlim[0]))

    def u(mmv):
        return mmv / scale

    def yb(mmv):
        return grid.ylim[0] - mmv / scale

    som_figure_context.hex_field(ax, grid, {}, lw=0.10)

    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    up4 = ctx.gene_up[ctx.gene_up >= 4]
    counts = neuron_of.reindex(up4.index).value_counts()
    sup = set(ctx.sup)
    gene_cls = pd.Series(
        np.where(ctx.sym.symbol.isin(ctx.isg_panel)
                 .reindex(up4.index).fillna(False), 0,
                 np.where((ctx.sym.gene_type == "lncRNA")
                          .reindex(up4.index).fillna(False), 2, 1)),
        index=up4.index)
    comp = (pd.DataFrame({"neuron": neuron_of.reindex(up4.index),
                          "cls": gene_cls})
            .dropna().groupby(["neuron", "cls"]).size()
            .unstack(fill_value=0))
    dot_pt2 = _dot_pt2_for_scale(scale)
    for n, c in counts.items():
        n = int(n)
        if n not in sup:
            continue
        x, y = grid.centres[n]
        c = int(c)
        s = max(c * dot_pt2, 0.3)
        r = (s / np.pi) ** 0.5 * 25.4 / 72 / scale
        parts = [int(comp.loc[n].get(k, 0)) if n in comp.index else 0
                 for k in (0, 1, 2)]
        lo = 0.0
        for cnt, fc in zip(parts, PIE_COLORS):
            if cnt:
                hi = lo + 360.0 * cnt / c
                ax.add_patch(Wedge((x, y), r, 90 - hi, 90 - lo, fc=fc,
                                   ec="none", zorder=5))
                lo = hi
        ax.add_patch(Circle((x, y), r, fc="none", ec="#1F2328", lw=0.25,
                            zorder=5.2))

    grid.frame(ax)
    ax.set_ylim(grid.ylim[0] - u(legend_mm), grid.ylim[1] + 1.6)
    n4 = int(len(up4))
    on4 = int((ctx.up_on >= 4).sum())
    lnc4 = int((ctx.up_lnc >= 4).sum())
    other4 = n4 - on4 - lnc4
    print(f"  v21 composition  : {n4} genes >= 4/5 = {on4} on the 382-ISG "
          f"panel + {other4} off-panel other + {lnc4} off-panel lncRNA")
    bold = dict(fontsize=4.4, fontweight="bold", ha="left", va="center",
                color="#1F2328", zorder=11)
    plain = dict(fontsize=4.2, ha="left", va="center", color="#1F2328",
                 zorder=11)
    x0 = grid.xlim[0] + 0.4
    y1 = yb(2.4)
    ax.text(x0, y1, "genes induced in ≥ 4 of 5 viruses", **bold)
    for dx, c in ((1.5, 1), (5.0, 10), (9.5, 30)):
        s_c = max(c * dot_pt2, 0.3)
        ax.scatter([x0 + u(dx)], [yb(4.8)], s=s_c, fc="0.35", ec="#1F2328",
                   lw=0.3, zorder=5)
        r_mm = (s_c / np.pi) ** 0.5 * 25.4 / 72
        ax.text(x0 + u(dx + r_mm + 0.35), yb(4.8), str(c), **plain)
    for i, (fc, lab) in enumerate(zip(
            PIE_COLORS,
            (f"on the 382-ISG panel ({on4})",
             f"off the panel ({other4})",
             f"off-panel lncRNA ({lnc4})"))):
        y = yb(7.4 + i * 2.1)
        ax.add_patch(Rectangle((x0, y - u(0.65)), u(1.5), u(1.3), fc=fc,
                               ec="white", lw=0.3, zorder=5))
        ax.text(x0 + u(2.2), y, lab, **plain)


def compose_v21(ctx, virus_strip=True, out_name="fig6_v21_split",
                tag="v21 split", equal_panels=False):
    fig_w = 183.0
    fig = plt.figure(figsize=(fig_w * MM, 247.0 * MM))

    ax_a = fig.add_axes([0.008, 0.300, 0.984, 0.690])
    bottom = panel_atlas_neuron_boxes(ax_a, ctx, side_pad=18.0,
                                      right_pad=18.0, glyphs_out=True,
                                      plain_units=(56,), named_key=False,
                                      orphan640=True, top_stats=False)

    xspan = ax_a.get_xlim()[1] - ax_a.get_xlim()[0]
    scale = 0.984 * fig_w / xspan
    span = ax_a.get_ylim()[1] - bottom
    a_mm = span * scale

    strip_mm = (6.0 + 2 * ((41.3 + 2.0) * 0.28 + 2.5) + 2.0
                if virus_strip else 0.0)
    row2_mm = min(53.0, 247.0 - 18.0 - a_mm - strip_mm)
    fig_h = (a_mm + 7.0 + row2_mm + 5.0
             + (6.0 + strip_mm if virus_strip else 0.0))
    fig.set_size_inches(fig_w * MM, fig_h * MM)
    a_frac = a_mm / fig_h
    a_bottom = 1.0 - 0.030 - a_frac
    ax_a.set_position([0.008, a_bottom, 0.984, a_frac])

    r2_h = row2_mm / fig_h
    r2_y = a_bottom - 0.025 - r2_h
    if equal_panels:
        gx = ctx.grid.xlim[1] - ctx.grid.xlim[0]
        gy = (ctx.grid.ylim[1] + 1.6) - ctx.grid.ylim[0]
        margin, gap = 4.0, 3.0
        scale = (fig_w - 2 * margin - 2 * gap - CONS_LEGEND_MM) / (3 * gx)
        scale = min(scale,
                    row2_mm / (gy + som_figure_context.MAP_STRIP),
                    (row2_mm - COMP_LEGEND_MM) / gy)
        w = scale * gx
        rects, x = [], margin
        for extra, hh in zip((0.0, CONS_LEGEND_MM, 0.0),
                             (scale * (gy + som_figure_context.MAP_STRIP), scale * gy,
                              scale * gy + COMP_LEGEND_MM)):
            rects.append((x / fig_w, (w + extra) / fig_w, hh / fig_h))
            x += w + extra + gap
    else:
        rects = [(0.015, 0.290, r2_h), (0.310, 0.430, r2_h),
                 (0.755, 0.240, r2_h)]
    r2_top = r2_y + r2_h
    (bx, bw, bh), (cx, cw, ch), (dx, dw, dh) = rects
    ax_b = fig.add_axes([bx, r2_top - bh, bw, bh])
    panel_kinetic_map_clean(ax_b, ctx, title=False)
    label(fig, bx - 2.0 / fig_w, a_bottom - 0.012, "b")
    ax_c = fig.add_axes([cx, r2_top - ch, cw, ch])
    panel_conservation_v21(ax_c, ctx)
    label(fig, cx - 2.0 / fig_w, a_bottom - 0.012, "c")
    ax_d = fig.add_axes([dx, r2_top - dh, dw, dh])
    panel_composition_clone_v21(ax_d, ctx)
    label(fig, dx - 2.0 / fig_w, a_bottom - 0.012, "d")

    if virus_strip:
        strip_top_mm = (1.0 - r2_y) * fig_h + 2.5
        panel_virus_strip(fig, ctx, strip_top_mm, map_scale=0.28,
                          title="untreated infection vs Mock")
        label(fig, 0.006, 1.0 - (strip_top_mm + 3.0) / fig_h, "e")

    out = FIGDIR / f"{out_name}.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"{tag:<13}: {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v25(ctx):
    compose_v21(ctx, virus_strip=False, out_name="fig6_v25_four_panel",
                tag="v25 four-panel", equal_panels=True)


def compose_v26(ctx, simple=False, out_name="fig6_v26_simplified",
                tag="v26 atlas"):
    fig_w = 183.0
    fig = plt.figure(figsize=(fig_w * MM, 247.0 * MM))

    ax = fig.add_axes([0.008, 0.040, 0.984, 0.920])
    bottom = panel_atlas_neuron_boxes(ax, ctx, side_pad=18.0, right_pad=18.0,
                                      glyphs_out=True, plain_units=(56,),
                                      named_key=False, orphan640=True,
                                      top_stats=False, simple=simple)

    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
    scale = 0.984 * fig_w / xspan
    a_mm = (ax.get_ylim()[1] - bottom) * scale
    fig_h = a_mm + 8.0
    fig.set_size_inches(fig_w * MM, fig_h * MM)
    a_frac = a_mm / fig_h
    ax.set_position([0.008, 1.0 - 4.0 / fig_h - a_frac, 0.984, a_frac])

    out = FIGDIR / f"{out_name}.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"{tag:<13}: {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v27(ctx):
    compose_v26(ctx, simple=True, out_name="fig6_v27_clean",
                tag="v27 clean")


def coherence_stats_v22(ctx):
    import itertools

    Z = {r["name"]: ctx.cb.loc[r["units"]].to_numpy(dtype=float)
         for r in som_figure_context.spatial_regions(ctx)}

    def corr_dist(A, B):
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)
        An = A / np.linalg.norm(A, axis=1, keepdims=True)
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
        return 1 - An @ Bn.T

    within = {name: float(corr_dist(Za, Za)[np.triu_indices(len(Za), 1)]
                          .mean())
              for name, Za in Z.items()}
    between = {(a, b): float(corr_dist(Z[a], Z[b]).mean())
               for a, b in itertools.combinations(Z, 2)}
    w = float(np.mean(list(within.values())))
    b = float(np.mean(list(between.values())))
    print(f"  v22 coherence    : within "
          f"{ {REGION_SHORT_V10[k]: round(v, 2) for k, v in within.items()} }, "
          f"between { {(REGION_SHORT_V10[a], REGION_SHORT_V10[c]): round(v, 2)
                       for (a, c), v in between.items()} }, "
          f"ratio {b / w:.2f}x")
    return within, between, w, b


def panel_umatrix_supp_v22(ax, ctx):
    grid = ctx.grid
    legend_w_mm = 52.0
    map_span = (grid.ylim[1] + 1.6) - grid.ylim[0]
    pos = ax.get_position()
    rect_h_mm = pos.height * ax.figure.get_size_inches()[1] * 25.4
    rect_w_mm = pos.width * ax.figure.get_size_inches()[0] * 25.4
    scale = min((rect_h_mm - 1.0) / map_span,
                (rect_w_mm - 1.0 - legend_w_mm)
                / (grid.xlim[1] - grid.xlim[0]))

    def u(mmv):
        return mmv / scale

    def yt(mmv):
        return grid.ylim[1] - mmv / scale

    u_mat = umatrix_values(ctx)
    vals = np.array(list(u_mat.values()))
    norm = mpl.colors.Normalize(vmin=float(np.nanpercentile(vals, 2)),
                                vmax=float(np.nanpercentile(vals, 98)))
    som_figure_context.hex_field(ax, grid, {n: TRUNC_GREYS(norm(u_mat[n]))
                             for n in grid.centres}, lw=0.10)

    for r in ctx.regions:
        tint = REGION_TINTS[r["name"]]
        draw_region_outline(ax, r["units"], grid, darken(tint, 0.8),
                            halo=tint, halo_lw=2.4, core_lw=1.0, z=3.5)

    landmarks = [(756, "interferon ligands", RING_IFN),
                 (648, "WNV-restricted stress", RING_RESTRICTED),
                 (32, "DENV-restricted metabolic", RING_RESTRICTED),
                 (59, "ketogenesis", "#1F2328")]
    for n, _, colour in landmarks:
        x, y = grid.centres[n]
        ax.add_patch(Polygon(som_figure_context.hex_ring(x, y, 1.05), closed=True,
                             fc="none", ec=colour, lw=0.9, zorder=8))

    grid.frame(ax)
    ax.set_xlim(grid.xlim[0], grid.xlim[1] + u(legend_w_mm))
    ax.set_ylim(grid.ylim[0] - u(1.0), grid.ylim[1] + 1.6)
    ax.set_title("U-matrix — the map's own cluster structure",
                 fontsize=5.2, pad=3)

    lx = grid.xlim[1] + u(4.0)
    bold = dict(fontsize=4.4, fontweight="bold", ha="left", va="center",
                color="#1F2328", zorder=11)
    plain = dict(fontsize=4.2, ha="left", va="center", color="#1F2328",
                 zorder=11)

    ax.text(lx, yt(2.4), "fill = mean distance to", **bold)
    ax.text(lx, yt(4.1), "neighbouring neurons", **bold)
    bw = u(20.0)
    for i in range(40):
        ax.add_patch(Rectangle((lx + i * bw / 40, yt(6.2) - u(1.2)),
                               bw / 40, u(1.2), fc=TRUNC_GREYS(i / 39),
                               ec="none", zorder=5))
    ax.add_patch(Rectangle((lx, yt(6.2) - u(1.2)), bw, u(1.2), fc="none",
                           ec="0.3", lw=0.5, zorder=5))
    ax.text(lx, yt(7.8), "valley: similar responses · ridge: boundaries",
            fontsize=4.0, ha="left", va="center", color="0.35", zorder=11)

    ax.text(lx, yt(11.2), "regions", **bold)
    for i, r in enumerate(ctx.regions):
        y = yt(13.2 + i * 2.4)
        tint = REGION_TINTS[r["name"]]
        ax.add_patch(Rectangle((lx, y - u(0.75)), u(1.7), u(1.5), fc=tint,
                               ec=darken(tint, 0.8), lw=0.5, alpha=0.45,
                               zorder=5))
        ax.text(lx + u(2.4), y, REGION_SHORT_V10[r["name"]], **plain)

    ax.text(lx, yt(21.6), "landmark units", **bold)
    for i, (n, name, colour) in enumerate(landmarks):
        y = yt(23.6 + i * 2.4)
        ax.add_patch(Polygon(som_figure_context.hex_ring(lx + u(1.1), y, 0.85), closed=True,
                             fc="none", ec=colour, lw=0.9, zorder=5))
        ax.text(lx + u(2.6), y, f"{n} — {name}", **plain)

    within, between, w_mean, b_mean = coherence_stats_v22(ctx)
    ax.text(lx, yt(34.4), "region coherence", **bold)
    ax.text(lx, yt(36.2), "correlation distance (1 − Pearson r):",
            fontsize=4.0, ha="left", va="center", color="0.35", zorder=11)
    ax.text(lx, yt(38.0),
            f"within {min(within.values()):.2f}–{max(within.values()):.2f}"
            f" · between {min(between.values()):.2f}–"
            f"{max(between.values()):.2f}", **plain)
    ax.text(lx, yt(39.8), f"coherence ratio ≈ {b_mean / w_mean:.1f}×",
            **plain)


def compose_v22(ctx):
    fig_w, fig_h = 150.0, 105.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))
    ax = fig.add_axes([0.030, 0.050, 0.940, 0.920])
    panel_umatrix_supp_v22(ax, ctx)

    out = FIGDIR / "fig6_v22_umatrix_supp.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v22 umatrix  : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


SCH_VIOLET, SCH_BLUE, SCH_OCHRE = "#8E7CC3", "#6FA8DC", "#C2A15C"
SCH_RED = "#b2182b"
SCH_DARKBLUE = "#2166ac"


def sch_box(ax, x, y, w, h, lines, fc="white", ec="0.55", lw=0.8, pad=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,"
                                "rounding_size=1.2", fc=fc, ec=ec, lw=lw,
                                zorder=3))
    style = {"head": dict(fontsize=5.6, fontweight="bold", color="#1F2328"),
             "body": dict(fontsize=5.0, color="#1F2328"),
             "tag": dict(fontsize=4.6, fontstyle="italic", color="0.30"),
             "alert": dict(fontsize=4.6, fontstyle="italic", color=SCH_RED)}
    lead = {"head": 2.6, "body": 2.3, "tag": 2.1, "alert": 2.1}
    yy = y + h - pad
    for text, kind in lines:
        ax.text(x + w / 2, yy, text, ha="center", va="top", zorder=4,
                **style[kind])
        yy -= lead[kind]


def sch_arrow(ax, p0, p1, dashed=False):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.0,
                                ls=(0, (2.5, 2.0)) if dashed else "solid",
                                shrinkA=1, shrinkB=1), zorder=2)


def compose_v23(ctx):
    fig_w, fig_h = 180.0, 145.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(fig_w / 2, 140.5,
            "A conserved, three-phase transcriptional architecture, "
            "partitioned by JAK–STAT", fontsize=7.2, fontweight="bold",
            ha="center", va="center", color="#1F2328")

    t_mock, t_24, t_72 = 15.0, 60.0, 145.0
    for t in (t_24, t_72):
        ax.plot([t, t], [97, 133], color="0.80", lw=0.6, ls=(0, (2, 2)),
                zorder=1)
    sch_box(ax, 15, 124, 80, 9, [
        ("① interferon production", "head"),
        ("IFNL1–4 · IFNB1 (unit 756) · peaks 24 hpi", "body")],
        fc=som_figure_context.lighten(SCH_VIOLET, 0.72), ec=SCH_VIOLET)
    sch_box(ax, 40, 110, 120, 9, [
        ("② antiviral & immune effectors — ISGF3 core + IRF3-direct "
         "(units 728 · 727 · 726 · 725)", "head"),
        ("coding + non-coding: 84 lncRNA (LINC00487 · MX1-AS1 · NRIR)",
         "body")], fc=som_figure_context.lighten(SCH_BLUE, 0.72), ec=SCH_BLUE)
    sch_box(ax, 85, 96, 75, 9, [
        ("③ late metabolic repression", "head"),
        ("ketogenesis · CYP450 · conjugation · 72 hpi", "body")],
        fc=som_figure_context.lighten(SCH_OCHRE, 0.72), ec=SCH_OCHRE)
    ax.plot([t_mock, 160], [92.5, 92.5], color="0.25", lw=1.0, zorder=2)
    for t, lab in ((t_mock, "Mock"), (t_24, "24 hpi"), (t_72, "72 hpi")):
        ax.plot([t, t], [92.5, 94.0], color="0.25", lw=1.0, zorder=2)
        ax.text(t, 91.3, lab, fontsize=4.6, ha="center", va="top",
                color="#1F2328", zorder=4)
    ax.text(fig_w / 2, 86.5, "① and ② overlap: a median 50% of the 72-hpi "
            "induced genes are already DE at 24 hpi", fontsize=4.6,
            fontstyle="italic", ha="center", va="center", color="0.35",
            zorder=4)

    sch_box(ax, 6, 60, 20, 8, [("viral RNA", "head")])
    sch_box(ax, 34, 58, 26, 10, [("sensing", "head"),
                                 ("RIG-I · IFIH1", "body")])
    sch_box(ax, 70, 70, 44, 12, [("IRF3-direct arm", "head"),
                                 ("IFIT · RSAD2 · CXCL10/11", "body"),
                                 ("47–91% retained under JAKi", "tag")])
    sch_box(ax, 70, 52, 44, 12, [("IFN production", "head"),
                                 ("IFNL1–4 · IFNB1", "body"),
                                 ("91–105% retained under JAKi", "tag")])
    sch_box(ax, 122, 54, 22, 8, [("IFN receptor", "head")])
    sch_box(ax, 152, 54, 22, 8, [("JAK–STAT", "head")])
    sch_box(ax, 148, 34, 28, 12, [("ISGF3 amplification", "head"),
                                  ("& effector programmes", "body"),
                                  ("J = 0.73–0.90 — abolished", "alert")])
    sch_box(ax, 102, 34, 40, 12, [("OAS1/3–RIGI–DDX60 (unit 726)", "head"),
                                  ("sensing/amplification module", "body"),
                                  ("falls with the ISGF3 tier · J = 0.72",
                                   "alert")])
    sch_arrow(ax, (26, 64), (34, 63))
    sch_arrow(ax, (60, 66), (70, 74))
    sch_arrow(ax, (60, 60), (70, 58))
    sch_arrow(ax, (114, 58), (122, 58))
    sch_arrow(ax, (144, 58), (152, 58))
    sch_arrow(ax, (163, 54), (163, 50.5))
    ax.plot([157, 169], [49.5, 49.5], color=SCH_RED, lw=2.2, zorder=4,
            solid_capstyle="butt")
    ax.text(156, 49.5, "ruxolitinib (JAKi)", fontsize=4.6, ha="right",
            va="center", color=SCH_RED, zorder=4)
    sch_arrow(ax, (148, 40), (142, 40))
    sch_arrow(ax, (55, 58), (115, 46), dashed=True)
    ax.text(fig_w / 2, 28.5, "JAK dependence tracks ISGF3-target status, "
            "not position in the sensing-to-effector chain", fontsize=4.6,
            fontstyle="italic", ha="center", va="center", color="0.35",
            zorder=4)

    ax.text(fig_w / 2, 23.5, "JAKi re-partitioning", fontsize=5.6,
            fontweight="bold", ha="center", va="center", color="#1F2328",
            zorder=4)
    sch_box(ax, 15, 4, 75, 15, [
        ("JAK–STAT-dependent", "head"),
        ("ISGF3 amplification & effector programmes", "body"),
        ("(incl. the OAS1/3 sensing module)", "tag"),
        ("abolished under JAKi", "alert")],
        fc=som_figure_context.lighten(SCH_RED, 0.88), ec=SCH_RED)
    sch_box(ax, 95, 4, 77, 15, [
        ("JAK–STAT-independent", "head"),
        ("IFN production · IRF3-direct arm", "body"),
        ("NF-κB/AP-1 late regime", "body"),
        ("retained or amplified", "tag")],
        fc=som_figure_context.lighten(SCH_DARKBLUE, 0.88), ec=SCH_DARKBLUE)

    out = FIGDIR / "fig6_v23_schematic.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v23 schematic: {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


def compose_v16(ctx):
    fig_w, fig_h = 183.0, 247.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    ax_a = fig.add_axes([0.008, 0.568, 0.984, 0.422])
    panel_atlas_locators(ax_a, ctx)
    label(fig, 0.004, 0.996, "a")

    zoom_mm = 2.2
    key_w = 8.0
    zooms = zoom_windows(ctx)
    n5 = int((ctx.gene_up == 5).sum())
    facts = {}
    for sym, r, win in zooms:
        genes = set(ctx.assign[ctx.assign.neuron.isin(r["units"])].gene_id)
        facts[sym] = len(set(ctx.gene_up[ctx.gene_up == 5].index) & genes)
    captions = {
        "i": f"i · IFN / ISG core — {facts['i']} of {n5} all-five genes",
        "ii": "ii · inflammatory & stress — 648 WNV-restricted",
        "iii": "iii · metabolic & secretory — 32 DENV-restricted · ③",
    }
    x_mm = 4.0
    top = 0.545
    for sym, r, win in zooms:
        entries = zoom_key_entries(ctx, win)
        w_mm = ((win[1] - win[0]) + 0.4 + key_w) * zoom_mm
        h_mm = ((win[3] - win[2]) + 0.8) * zoom_mm
        ax_z = fig.add_axes([x_mm / fig_w, top - h_mm / fig_h,
                             w_mm / fig_w, h_mm / fig_h])
        zoom_callout_panel(ax_z, ctx, r, win, entries, key_w)
        ax_z.set_title(captions[sym], fontsize=4.5,
                       color=darken(REGION_TINTS[r["name"]], 0.8),
                       loc="left", pad=2.5)
        x_mm += w_mm + 5.0

    lx = fig.add_axes([0.022, 0.318, 0.940, 0.028])
    lx.set_xlim(0, 100)
    lx.set_ylim(0, 3)
    lx.axis("off")
    lx.text(0, 1.5, "dots in the boxes: colour = viruses agreeing",
            fontsize=4.2, ha="left", va="center", color="#1F2328")
    x = 27.0
    for k5 in range(1, 6):
        lx.add_patch(Rectangle((x, 0.8), 2.6, 1.5, fc=DOT_RAMP[k5 - 1],
                               ec="white", lw=0.3))
        lx.text(x + 1.3, -0.4, str(k5), fontsize=4.0, ha="center", va="top",
                clip_on=False)
        x += 3.4
    lx.text(46.5, 1.5, "size = genes ≥ 4/5:", fontsize=4.2, ha="left",
            va="center", color="#1F2328")
    x = 61.0
    for c in (1, 10, 30):
        lx.scatter([x], [1.5], s=1.0 + 0.3 * c, fc="0.35", ec="#1F2328",
                   lw=0.3)
        lx.text(x, -0.4, str(c), fontsize=4.0, ha="center", va="top",
                clip_on=False)
        x += 4.6
    lx.scatter([x], [1.5], s=1.0 + 0.3 * 10, fc="white", ec="#1F2328",
               lw=0.45)
    lx.text(x + 2.2, 1.5, "= all five", fontsize=4.2, ha="left", va="center",
            color="#1F2328")

    ax_b = fig.add_axes([0.020, 0.020, 0.430, 0.255])
    som_figure_context.panel_kinetic_map(ax_b, ctx)
    label(fig, 0.006, 0.288, "b")

    ax_c = fig.add_axes([0.510, 0.020, 0.470, 0.255])
    panel_structure_conservation(ax_c, ctx, grey=True, regions=False)
    label(fig, 0.490, 0.288, "c")

    out = FIGDIR / "fig6_v16_callout_zoom.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print(f"v16 callout  : {fig_w:.0f} x {fig_h:.0f} mm")
    print("  wrote " + ", ".join(str(p) for p in written))


README_V78 = """# Fig. 6 candidates v7–v23 (clarity series)

Further variants written by `phh_som/som_atlas_panels.py`, built on top
of som_figure_context's machinery (`som_figure_context.Context`, the atlas geometry, the curation dicts) in
response to the v5 review: no captions or leader lines over the lattice, and
every summary panel readable without the figure legend. som_figure_context and its README.md
are untouched; every number below is re-derived at runtime by the same rules
som_figure_context uses.

| file | panels | size |
|---|---|---|
| `fig6_v7_clarity` | **a** clarity atlas — tinted spatial territories (soft fill + thin solid boundary), class-coloured fan glyphs for the 50 supported programmes, named units ringed + numbered, class/region key in the left margin, numbered call-out list grouped by region in the right margin; **b** three temporal phases plus the orphan late wave — per-phase mean untreated codebook trajectories Mock → 24 → 72 hpi (② spans units 728/726/725/727 and 668; ④ unit 640 overlaid from its own codebook row, below the support threshold) with the 50%-vs-0% succession statistic; **c** class × condition dot plot — mean log2FC ± 95% CI per functional class for UT 24 / UT 72 / Ruxo 24 / Ruxo 72, per-virus means as faint ticks, with the ruxolitinib retention ranges stated; **d** gene-level conservation — horizontal stacked bars, genes induced by 1–5 viruses split by 382-ISG-panel membership, plus the genes repressed by all five; **e** early vs late — median share of a programme's 72-hpi DEGs already DE at 24 hpi, per class, programmes as dots; **f** the two virus-restricted programmes as gene × virus heatmaps (untreated log2FC, dot = DE call) | 183 × 244 mm |
| `fig6_v8_phase` | phase-narrative layout: **a** the same clarity atlas; **b** the phase schematic larger and central (phases ①②③ plus the ④ orphan late wave at unit 640); **c** per-phase example genes (IFNL1/IFNB1 — ISG15/MX1/STAT1 — HMGCS2/CYP3A4/ADH1B — unit 640's four unnamed Ensembl transcripts) as untreated log2FC heatmaps; **d** the conservation bars; **e** the two virus-restricted heatmaps | 183 × 244 mm |
| `fig6_v9_hybrid` | the reviewer's hybrid: **a** v1's fan-glyph codes map (every neuron keeps its radial per-condition glyph) with v7's tinted region territories, thin solid boundaries, ringed + numbered named units and the two margin lists; **b** v6's kinetic map (neurons by direction × kinetic class); **c** U-matrix — mean codebook-space distance to lattice neighbours, region boundaries overlaid, so the regions read as valleys/ridges in the data (landmarks 756 / 648 / 59 ringed); **d** gene-conservation terrain — per-neuron count of member genes induced by ≥ 4 of 5 viruses (BuGn, deliberately not panel e's purple), neurons holding an all-five-conserved gene marked with a white dot, region boundaries overlaid, headline counts (333 / 108 / 214 off-panel / 84 lncRNA / 11 repressed) in the caption strip; **e** v6's programme-conservation map ("18 of 34 induced programmes shared by all five" — programme level, direction of agreement); **f** the two virus-restricted heatmaps with a shared labelled colorbar (−4…+4 log2FC vs Mock, dot = DE call) | 183 × 247 mm |
| `fig6_v10_legend` | v9's panels carried by legends, swatches, colorbars and symbols instead of prose, with no words on any lattice: **a** hybrid codes map without the right-margin call-out list — left margin is a pure key (counts line, class swatches, region swatches, ring-symbol key), on the map only rings + numbers (red ring = virus-restricted, magenta = IFN ligands); **b** kinetic map; **c** U-matrix with region/symbol legend below (rings unlabelled on the map); **d** combined conservation map — programme fill (viruses agreeing, discrete 1–5 ramp) + per-neuron dots sized by member genes induced by ≥ 4/5 viruses (white = contains an all-five gene), red rings for the two virus-restricted units, two headed legend blocks plus a two-line numbers strip (333 · 214 off-panel · 84 lncRNA / 108 all five · 11 repressed); **e** the two virus-restricted heatmaps with ALL genes (23 + 8), one-line titles and the shared colorbar | 183 × 247 mm |
| `fig6_v11_nature` | v10 pushed toward publication quality, with ONE shared region language (violet = IFN/ISG core, blue = inflammatory & stress, ochre = metabolic & secretory, halo + core outlines) and the U-matrix and conservation map MERGED into one structure × conservation panel: **a** hybrid codes map — saturated region washes + thick tinted-halo outlines, no words on the lattice; **b** kinetic map; **c** merged structure × conservation — U-matrix fill (magma truncated to 0.15–0.85: plum valley, warm ridge, never black), dots per supported programme (size = member genes induced by ≥ 4/5 viruses, teal 1–5 ramp = viruses agreeing, white fill = all five), red rings for the virus-restricted units, region outlines, headed legend blocks (dot colour, dot size, ring, white dot, region swatches, numbers strip, framed U-matrix scale bar); **d** three-temporal-phase schematic (per-phase mean untreated codebook-z trajectories Mock → 24 → 72 hpi, ① 756 · ② 728/726/725/727 · ③ metabolic, median-50%-vs-0% stat); **e** restricted heatmaps with all genes (23 + 8), ISG-panel membership colours (purple = on, grey = off), full legend lines, shared colorbar | 183 × 247 mm |
| `fig6_v12_atlas` | three panels only: **a** hybrid codes map; **b** kinetic map; **c** the merged structure × conservation map with a MONOCHROME U-matrix fill (Greys truncated to 0.10–0.80: light-grey valley → dark-slate ridge) so the violet/blue/ochre region outlines are the sole hue on the lattice; teal/white conservation dots unchanged (dark edges in grey mode); b and c enlarged into the space v11's d/e freed | 183 × 235 mm |
| `fig6_v13_measured` | v12's three map panels (a atlas, b kinetic, c merged structure × conservation, grey U-matrix) plus **d** the three phases from measurements only — one sub-panel per phase, one dumbbell per virus: mean untreated log2FC vs Mock of the phase's member genes at 24 hpi (open dot) and 72 hpi (filled dot) with a straight connector, black dumbbell = 5-virus mean, zero line marked; no curves or interpolation anywhere; succession stat (median 50% vs 0%) as one line below | 183 × 247 mm |
| `fig6_v14_merged_atlas` | the super-atlas: **a** hybrid codes map with panel c's conservation layer absorbed — badges at each supported neuron's bottom vertex (teal 1–5 = viruses agreeing, size = genes ≥ 4/5, white = all five), region territories drawn once with halo outlines, and one callout box per region plus the virus-restricted pair, each in a hexagon-free zone with a single short connector (the standalone U-matrix map retires to supplementary grade); **b** kinetic map; **c** the three phases as a measured tile grid — 3 phases × (24 hpi | 72 hpi × 5 viruses), tile = mean untreated log2FC of phase member genes, value printed in the tile, RdBu_r shared scale | 183 × 247 mm |
| `fig6_v15_zoom` | v13's maps (a hybrid codes map, b kinetic, c merged structure × conservation with grey U-matrix) with dashed locator windows on c, plus **d** magnified zoom insets of the three territories (~3×): same grey fill, conservation dots, red rings and region outlines, plus unit numbers inside hexagons and a gene key per zoom (genes from the validated CALLOUTS); titles carry the per-region fact (103 of 108 all-five genes in the core; WNV-restricted 648; DENV-restricted 32, phase ③) | 183 × 247 mm |
| `fig6_v16_callout_zoom` | the zoom view moved INTO panel a: **a** hybrid codes map + three zoomed callout boxes in a strip below the lattice (one per territory; each a magnified cut-out of panel a itself — count shading, fan glyphs, unit numbers — with the conservation dots parked in the bottom-vertex gaps: teal 1–5 = viruses agreeing, size = genes ≥ 4/5, white = all five; gene key beside each mini-map; dashed locator windows on the lattice with matching frames and numerals i–iii); **b** kinetic map; **c** merged structure × conservation map WITHOUT region outlines (territories live on panel a only) — grey U-matrix, dots, rings, its dot/ring/numbers/valley–ridge legends | 183 × 247 mm |\n| `fig6_v17_region_callouts` | v16's layout with the boxes as TEXT call-outs in the original grammar — one framed box per region under the atlas, listing every named unit as 'number · programme name (N genes) — genes' (shorthand expanded, still guarded by check_callout_genes), below-threshold Results units dagger-marked; dashed i/ii/iii locator windows tie boxes to territories; no mini-maps, no dots in the boxes; **b** kinetic; **c** deduplicated structure × conservation (as v16) | 183 × 247 mm |\n| `fig6_v18_neuron_boxes` | the 382-ISG reference figure's callout grammar on the Fig. 6 atlas: **a** hybrid codes map (lattice ~111 mm wide) with one rounded pastel box per Results-named unit in a single column per margin, each box roughly at its unit's height and tight against the lattice (bold 'Neuron N' header + italic genes ≤ 4 + … for unshown members; genes read from the neuron's own membership — its most expressed named members, no gene list in the script), box fill/edge = pastel of the unit's region hue (violet core / blue inflammatory / ochre metabolic; red edge on the two virus-restricted units); count strip top-left, fan key top-right with Neuron 756 beside it, class/region/ring keys in the free band below the lattice; **b** kinetic; **c** deduplicated structure × conservation (as v16/v17) | 183 × 245 mm (height fitted to content) |\n| `fig6_v19_virus_strip` | v18's panel **a** (per-neuron pastel boxes, ring swatch key), **b** annotation-free kinetic map (pure data + legend, no ①/②/③ territory notes), **c** the SAME kinetic-class fill as b (red = induced, blue = repressed, lightness = 24-only / both / 72-only; tagged "colours as in b") carrying the conservation dots over a minimal two-line legend strip — "viruses agreeing": teal ramp with the induced-programme count under each step 1–5 (2 / 0 / 0 / 14 / 18, summing to the title's 34, step 5 = the 18), "genes ≥ 4/5": dot sizes 1/10/30 (largest dot sized to 0.85 of its hexagon), white dot = all five; no frame, no prose lines, and the grey U-matrix retires to supplementary — and **d** the per-virus response strip at the bottom: 5 viruses × 2 rows (untreated 24 / 72 hpi, no ruxolitinib), each ~14 × 12 mm SOM map on the shared winsorized ±6 RdBu_r scale, one thin neutral outline of the induced territory, virus column headers in virus colours, shared colorbar — direct proof that all five viruses occupy the same territory, replacing v1/v2's 20-panel terrain | 183 × 247 mm |\n| `fig6_v20_unified` | v19 with panels b and c MERGED (they carried the identical kinetic fill): **a** v19's atlas, each callout box now carrying its neuron's fan glyph inside the box's outer side (~4.6 mm, same winsorized scale and UT/Ruxo colour grammar as the lattice glyphs, dashed zero ring; boxes grow into the margins, lattice unchanged); **b** the unified kinetics + conservation map — kinetic-class fill (red = induced, blue = repressed, lightness = 24-only / both / 72-only) with the conservation dots on top (teal = viruses agreeing, size = member genes ≥ 4/5, white = all five, largest dot 0.85 of a hexagon), enlarged into the freed width, its legend a three-group column right of the lattice: the 6 kinetic swatches with counts (42 / 27 / 61 induced × 24-only / both / 72-only, 21 / 16 / 59 repressed), the viruses-agreeing ramp with per-step programme counts (2 / 0 / 0 / 14 / 18, summing to the title's 34, step 5 = the 18), and the dot sizes 1/10/30 + white-dot symbol; the 18 all-five programmes are marked ON the lattice with a bold dark double ring (units 668, 677, 695–698, 722–728, 749, 751, 753, 754, 756 — 17 in the top-right IFN/ISG core, 677 top-left; legend swatch "induced by all five viruses (18)", and the ramp's step-5 swatch carries the same outline); **c** the per-virus response strip from v19's d, moved up to the bottom band | 183 × 247 mm |\n| `fig6_v21_split` | v20 split back apart (the unified panel was too complex) and trimmed for publication - no panel titles, no summary or stat lines, only scale, axis and key labels on the page (every count and claim moves to the caption): **a** the atlas with each fan glyph OUTSIDE its callout box — adjacent to the box's outer edge (~1 mm gap), vertically centred, one uniform scale for all 16 boxes (box genes read from the membership table at draw time, ~4.6 mm, box fill wraps the text alone, lattice unchanged; the technical unit 56 gets neither box nor lattice ring, and the 'named in Results' swatch is dropped from the ring key; the programme/gene summary line above the lattice is dropped; unit 640's orphan note joins the right stack as a dashed grey box in the below-threshold grammar carrying its label alone — 'Neuron 640† · orphan late wave' — with its fan glyph like the other boxes, its claims still re-verified at runtime (4 members, all bare Ensembl ids, 0 DE rows at 24 hpi, 3 of 4 with untreated log2fc > 1 in all five at 72 hpi) and stated in the caption); **b** v19's clean kinetic map (fill + 6-swatch legend with counts, no title, no dots/rings); **c** the split-out conservation map in v9 panel d's two-channel grammar: FILL = per-neuron count of member genes induced by ≥ 4/5 viruses on a continuous BuGn scale (zero and non-supported blank) with a labelled colour bar (0/10/20/30; heading "member genes induced in ≥ 4 of 5 viruses", wrapped to two lines so it clears the legend column), and "programme induced in N of 5 viruses" 1–5 as a discrete warm outline per induced programme (#FCAE61 → #7A0019, colourblind-safe, distinct from the fill — steps 1–3 now visible even where no ≥4/5 gene exists, e.g. units 32/648 at step 1), legend swatches as outline hexagons with the per-step counts 2 / 0 / 0 / 14 / 18; white dot = neuron contains a gene induced by all five viruses; soft dark-grey ring (#4A4F57) = the 18 all-five programmes (swatch "programme induced by all five viruses (18)", step-5 swatch ringed to match); steel downward triangles (#1F4E79) = units holding genes repressed by all five (11 genes in units 1, 30, 58, 294 — 58 supported, in the metabolic territory as expected for phase ③; 30/1/294 below threshold, drawn open; legend "genes repressed by all five viruses (11)" + "open = below threshold"), legend as a column right of the lattice; no title and no stat line — the counts (333 genes induced by ≥ 4/5 viruses · 108 by all five · 11 repressed by all five; 18 of 34 induced programmes shared by all five) print at runtime for the caption; **d** a small COMPOSITION CLONE map beside c (blank lattice, no fill encoding, no rings): one dot per neuron holding ≥ 4/5-induced genes, sized by count, pie-sectored by ISG-panel membership (purple #472A78 on the 382-ISG panel · grey 0.72 off-panel · teal #1F9086 off-panel lncRNA; solid dots for single-class neurons) — no title; legend = three class swatches with the runtime totals (119 on the ISG panel · 130 off-panel · 84 lncRNA) + a dot-size sample under the heading "genes induced in ≥ 4 of 5 viruses" (samples on their own row, all rows left-aligned); **e** the per-virus response strip, bottom band, labelled with the contrast alone ("untreated infection vs Mock"; the "same territory" reading is the caption's) | 183 × 247 mm |\n| `fig6_v22_umatrix_supp` | STANDALONE supplementary U-matrix (not a page composition): the 28×27 lattice filled by the U-matrix (mean Euclidean distance to immediate lattice neighbours over the 20 codebook columns) on v12/v13's truncated-Greys scale (light valley = similar responses, dark ridge = boundaries) with a framed valley↔ridge bar, the three spatial regions as halo + core outlines with swatches, four landmark units ringed in panel a's grammar (756 interferon ligands, magenta; 648 WNV-restricted stress and 32 DENV-restricted metabolic, red; 59 ketogenesis, dark) with a ring → unit/name legend, and a stat block with the region-coherence numbers re-derived at runtime (correlation distance 1 − Pearson r between codebook vectors, mean pairwise within vs between regions: within 0.26–0.64, between 0.70–1.42, coherence ratio ≈ 2.7×) | 150 × 105 mm |\n| `fig6_v23_schematic` | the CORRECTED graphical abstract (pure schematic, no data): top — the three phases on a real Mock / 24 / 72 hpi time axis, overlapping (① interferon production, violet, IFNL1–4 · IFNB1, unit 756, peaks 24 hpi; ② antiviral & immune effectors, blue, ISGF3 core + IRF3-direct, units 728/727/726/725, coding + non-coding (84 lncRNA: LINC00487 · MX1-AS1 · NRIR); ③ late metabolic repression, ochre, 72 hpi; ① and ② both active at 24 hpi, median 50% already DE); middle — the signalling chain viral RNA → sensing (RIG-I · IFIH1) → IRF3-direct arm (47–91% retained) / IFN production (91–105% retained) → IFN receptor → JAK–STAT, the ruxolitinib T-bar AT the JAK–STAT step, ISGF3 amplification & effectors abolished (J = 0.73–0.90), and the OAS1/3–RIGI–DDX60 module (726) on the DEPENDENT side (falls with the ISGF3 tier, J = 0.72) — correcting the earlier sketch's 'viral sensing is JAK-independent'; dependence tracks ISGF3-target status, not chain position; bottom — JAKi re-partitioning as two columns (dependent: ISGF3 tier abolished / independent: IFN production · IRF3-direct · NF-κB/AP-1 late regime, retained or amplified) | 180 × 145 mm |
| `fig6_v25_four_panel` | v21 without panel **e**: the atlas (**a**), the clean kinetic map (**b**), the conservation map (**c**) and the composition clone (**d**) only. The per-virus response strip retires to supplementary — its "same territory" reading is already quantified in **c** (18 of 34 induced programmes induced by all five), and pooling its five viruses into one map per time point would only redraw **b** (pooling the five untreated columns at z > 1 reproduces b's 42 / 27 / 61 induced early / sustained / late exactly). The page is width-bound, so the freed 37 mm mostly shortens the figure; b, c and d are then sized FROM one shared map scale and top-aligned — each axes gets its map's width plus whatever its key needs beside it (c a legend column, d none) and exactly its own content's height, so the aspect fit has no slack to centre away and all three lattices render 43.5 mm wide (v21: 40.9 / 41.6 / 43.1 mm). b's two legend columns stop colliding at the larger size | 183 × 209 mm |
| `fig6_v26_simplified` | v21's panel a as a STANDALONE single-panel figure — the plain atlas with the class + region grammar and nothing else: the count-shaded lattice with the '# genes per neuron' strip (uniform hexagon size), the three region territories as tint + halo outlines with swatches, one rounded pastel callout box per Results-named unit stacked in the margins (bold 'Neuron N' header + the unit's own most expressed named members in italic, ≤ 4 + …, read from the membership table at draw time, guarded by check_callout_genes) with each fan glyph OUTSIDE its box at one uniform size, unit 640's dashed grey orphan note (claims re-verified at runtime), the technical unit 56 unboxed, the winsorized codebook-weight fan key with the UT/Ruxo virus colour legend, the class and ring-semantics keys, and the unit numbers on named neurons. Every conservation layer tried in earlier v26 drafts (BuGn fill + colour bar, composition pie dots, white all-five dots, grey rings, repression triangles, the agreement-outline ramp, the hexagon-size encoding) is retired — that content stays in v21/v25's panels c and d; the kinetic map and the per-virus response strip are likewise supplementary. One axes, no panel letter, height fitted to content | 183 × 152 mm (height fitted to content) |
| `fig6_v27_clean` | v26 reduced to the regions: each region territory is a 1.2 pt outline traced along its member hexagons' own edges over a soft glow clipped to the territory (inward only, never over a neighbouring hexagon); no functional-class outlines or class key, no rings of any kind (named, virus-restricted, IFN ligands, dashed below-threshold), no ring key, no red box edges, no unit 640 note; named units keep their number centred on the hexagon to tie them to their boxes; region key = outlined-hexagon swatches; everything else as v26 | 183 mm wide (height fitted to content) |

v7 is the balanced replacement for v5; v8 walks the reader through the three
phases and drops the class × condition dot plot and the early-vs-late bars;
v9 combines the v6 panels the reviewer liked (a, b, e) with the clarity series'
region tinting, margin lists and colour-scaled heatmaps, and adds two
SOM-intrinsic panels: the U-matrix (c) and the gene-conservation terrain (d),
which answers "where do the 333 conserved genes live" on the map itself.
v10 is v9 with the prose removed - every panel carries its information by
legends, swatches, colorbars and symbols (reviewer: "too many text").
v11 is v10 at publication quality: the three-phase schematic returns as
panel f, and the restricted heatmaps (e) mark 382-ISG-panel membership and
spell out every symbol in legend lines. v12 strips back to the three map
panels (atlas, kinetic, merged structure x conservation) with a monochrome
U-matrix so the region colours carry the lattice alone. v13 replaces the
interpolated phase curves with measured-only dumbbells (reviewer: "making
curves like that is not scientific"). v14 merges the conservation layers
into the atlas itself (badges + region callout boxes) and turns the phases
into a measured tile grid; the standalone U-matrix is supplementary-grade
from v14 on. v15 keeps v13's maps and replaces the phase panel with
magnified zoom insets of the three territories (locator windows on c).
v16 moves the zooms into panel a as callout boxes carrying the
conservation-dot encoding, and panel c sheds the region outlines. v17 keeps
that layout but the boxes are the classic text call-outs - boxed per region,
original number + name + genes grammar. v18 switches the call-outs to the
382-ISG reference figure's per-neuron rounded pastel boxes ('Neuron N' +
genes + …, stacked in the margins; the genes are the unit's own most
expressed named members, read from the membership table at draw time, so no
gene list lives in this script). v19 moves the per-virus response strip
(5 viruses × 2 time points, shared ±6 scale) to the bottom as panel d, makes
panel b an annotation-free kinetic map, and puts panel c on the same kinetic
fill as b with a minimal two-line legend strip (ramp with per-step programme
counts, dot sizes, white-dot symbol; the U-matrix is supplementary-grade).
v20 merges v19's panels b and c into one unified kinetics + conservation
map (they had shared the same fill), enlarged into the freed width with a
three-group legend column, and re-letters the per-virus strip as panel c.
v21 splits them apart again (b clean kinetic, c conservation with the 18
double-ringed on the lattice) and moves panel a's fan glyphs outside the
callout boxes at one uniform size; it is also trimmed for publication -
panel titles, summary and stat lines and the orphan-640 prose are dropped,
leaving only scale, axis and key labels on the page (the numbers still
print at runtime, for the caption). v22 is the standalone supplementary
U-matrix figure (lattice filled by the U-matrix on truncated Greys, region
outlines, landmark rings, coherence stats re-derived at runtime). v23 is
the corrected graphical abstract (pure schematic: overlapping three-phase
timing band, signalling chain with the ruxolitinib gate at JAK-STAT, OAS
module on the dependent side, two-column re-partitioning). v25 is v21 with
panel e dropped (183 x 209 mm): one SOM underlies every panel, so the strip's
ten small maps were ten columns of the one codebook, and a pooled two-time-
point version of them is panel b exactly - what the strip alone carried was
the unpooled per-virus terrain, which moves to supplementary. v26 is v21's
panel a as a standalone single-panel figure: the plain atlas with the
class + region grammar, every conservation layer (fill, composition dots,
all-five marks, triangles, outlines, size encoding) retired to v21/v25's
panels c and d.

Regenerate:

    cd phh_som && ../.venv/bin/python som_atlas_panels.py [--variant 7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|25|26|27|all]
"""


def write_readme():
    out = FIGDIR / "README_v7_v8.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(README_V78)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description='clarity variants of Fig. 6 (v7, v8), built on top of som_figure_context.')
    ap.add_argument("--variant", default="all",
                    choices=["7", "8", "9", "10", "11", "12", "13", "14", "15",
                             "16", "17", "18", "19", "20", "21", "22", "23",
                             "25", "26", "27", "all"])
    args = ap.parse_args()

    set_rcparams()
    ctx = som_figure_context.Context()
    som_figure_context.announce(ctx)

    n4 = int((ctx.gene_up >= 4).sum())
    print(f"  genes induced by 1..5 viruses   : "
          f"{[int((ctx.gene_up == k).sum()) for k in range(1, 6)]}")
    print(f"  >=4: {n4}, all five: {int((ctx.gene_up == 5).sum())}, "
          f"off-panel >=4: {n4 - int((ctx.up_on >= 4).sum())}, "
          f"lncRNA >=4: {int((ctx.up_lnc >= 4).sum())}, "
          f"repressed by all five: {int((ctx.gene_dn == 5).sum())}")
    med_i = float(np.median(ctx.share24['induced']))
    med_r = float(np.median(ctx.share24['repressed']))
    print(f"  share already DE at 24 hpi      : induced median "
          f"{100 * med_i:.0f}%, repressed median {100 * med_r:.0f}%")
    for sym, name, keys, _, _ in PHASES:
        z24, z72 = phase_trajectory(ctx, keys)
        print(f"  phase {sym} {name:<32}: mean z 24 hpi {z24:+.1f}, "
              f"72 hpi {z72:+.1f}")
    o_sym, o_name, o_unit, _, _ = ORPHAN_PHASE
    z24, z72 = unit_set_trajectory(ctx, [o_unit])
    print(f"  phase {o_sym} {o_name:<32}: mean z 24 hpi {z24:+.1f}, "
          f"72 hpi {z72:+.1f}")
    ifn_lo, ifn_hi, isg_med = retention_stats(ctx)
    print(f"  retained under ruxolitinib, 24h : IFN ligands "
          f"{100*ifn_lo:.0f}-{100*ifn_hi:.0f}%, ISGF3 tier median "
          f"{100*isg_med:.0f}% (cases with untreated z > 1)")
    neuron_of = ctx.assign.set_index("gene_id")["neuron"]
    up4n = neuron_of.reindex(ctx.gene_up[ctx.gene_up >= 4].index) \
        .value_counts()
    sup = set(ctx.sup)
    in_ind = [n for n in up4n.index if n in sup and ctx.n_up.get(n, 0) >= 1]
    print(f"  the {n4} conserved genes sit in : {len(in_ind)} of the "
          f"{int((ctx.n_up >= 1).sum())} induced programmes, max "
          f"{int(up4n.max())} in one neuron (unit {int(up4n.idxmax())}), "
          f"{len(set(up4n.index) - sup)} below-threshold neurons carry one")

    shares = share24_per_neuron(ctx)
    if args.variant in ("13", "all"):
        pm = phase_means_measured(ctx)
        for sym, name, _, _ in PHASE_DB:
            df = pm[sym]
            row = "  ".join(f"{v} {df.loc[v, 24]:+.2f}/{df.loc[v, 72]:+.2f}"
                            for v in VIRUS_ORDER)
            print(f"  phase {sym} measured log2FC 24/72 : {row}  "
                  f"(mean {df[24].mean():+.2f}/{df[72].mean():+.2f})")
    if args.variant in ("7", "all"):
        compose_v7(ctx, shares)
    if args.variant in ("8", "all"):
        compose_v8(ctx, shares)
    if args.variant in ("9", "all"):
        compose_v9(ctx)
    if args.variant in ("10", "all"):
        compose_v10(ctx)
    if args.variant in ("11", "all"):
        compose_v11(ctx)
    if args.variant in ("12", "all"):
        compose_v12(ctx)
    if args.variant in ("13", "all"):
        compose_v13(ctx)
    if args.variant in ("14", "all"):
        compose_v14(ctx)
    if args.variant in ("15", "all"):
        compose_v15(ctx)
    if args.variant in ("16", "all"):
        compose_v16(ctx)
    if args.variant in ("17", "all"):
        compose_v17(ctx)
    if args.variant in ("18", "all"):
        compose_v18(ctx)
    if args.variant in ("19", "all"):
        compose_v19(ctx)
    if args.variant in ("20", "all"):
        compose_v20(ctx)
    if args.variant in ("21", "all"):
        compose_v21(ctx)
    if args.variant in ("22", "all"):
        compose_v22(ctx)
    if args.variant in ("23", "all"):
        compose_v23(ctx)
    if args.variant in ("25", "all"):
        compose_v25(ctx)
    if args.variant in ("26", "all"):
        compose_v26(ctx)
    if args.variant in ("27", "all"):
        compose_v27(ctx)
    write_readme()


if __name__ == "__main__":
    main()
