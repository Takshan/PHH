#!/usr/bin/env python3
import os
from functools import lru_cache

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import (FancyBboxPatch, PathPatch, Polygon, Rectangle)

from phh_som.utils.pipeline import VIRUS_ORDER
from phh_som.plotting.plot_som_map import COUNT_CMAP, FAN_R0, count_norm, count_ticks
from phh_som.plotting.plot_codes_map import paper_conds
from phh_som.plotting import som_figure_context


CLASSES = som_figure_context.CLASSES

CALLOUTS = som_figure_context.CALLOUTS

SHORT_CLASS = som_figure_context.SHORT_CLASS

REGION_TINTS = {
    "Interferon, ISG and antigen-presentation core": "#8E7CC3",
    "Inflammatory, stress and immediate-early": "#6FA8DC",
    "Hepatocyte metabolic and secretory": "#C2A15C",
}

def label(fig, x, y, text):
    fig.text(x, y, text, fontsize=8, fontweight="bold", ha="left", va="top")

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
        ring = (RING_RESTRICTED if n in ctx.restricted
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
    if restricted_edge and n in ctx.restricted:
        ec = RING_RESTRICTED
    return fc, ec

@lru_cache(maxsize=None)
def _max_group_tpm(ctx):
    degs = pd.read_parquet(som_figure_context.BASE_DATA / "degs.parquet")
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

def _orphan640_lines(ctx):
    degs = pd.read_parquet(som_figure_context.BASE_DATA / "degs.parquet")
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
