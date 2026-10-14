#!/usr/bin/env python3
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import (LinearSegmentedColormap, TwoSlopeNorm,
                               to_rgb)
from matplotlib.patches import (FancyBboxPatch, Polygon, Rectangle,
                                RegularPolygon)
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from scipy.interpolate import RBFInterpolator

from phh_som.utils.pipeline import (CLIP_Z, CONTRASTS, DATA, FIGURES, GRID_X, GRID_Y, MM,
                      VIRUSES, out_suffix, save_figure, set_rcparams,
                      virus_colors)
from phh_som.utils.somdata import load_universe
from phh_som.utils.somgeom import COND_LABEL, FAN_R0, fan_glyph as _fan_glyph, hex_ring, lighten

VIRUS_COLORS = {v: virus_colors()[v] for v in VIRUSES}
COUNT_CMAP = LinearSegmentedColormap.from_list("hexcount", ["#FFFFFF", "#A8C6E8"])

DEP_BREAKS = (0.2, 0.5)
MIN_SCORED = 20
MIN_FOR_SURFACE = (5, 8)
CALLOUT_PAD_BOTTOM = 1.55
CALLOUT_PER_BOX = 4
SQ3 = np.sqrt(3)

NOTABLE_GENES = {
    "STAT1", "STAT2", "IRF9", "IRF7", "RIGI", "IFIH1", "MAVS", "EIF2AK2",
    "IFIT1", "IFIT2", "IFIT3", "IFIT5", "ISG15", "MX1", "MX2",
    "OAS1", "OAS2", "OAS3", "OASL", "RSAD2", "BST2", "HERC5",
    "IFI6", "IFI27", "IFI35", "IFI44", "IFI44L", "IFI16",
    "IFITM1", "IFITM2", "IFITM3", "XAF1", "RTP4",
    "TRIM25", "ZC3HAV1", "LY6E", "RTCB", "POLI", "PARP11", "PARP10", "PARP12",
    "GLS", "LGALS9B", "LILRA1", "SP110", "SAMHD1", "ADAR", "APOBEC3A",
}

CALLOUT_GENES = {
    16: ["IFIT1", "IFIT2", "RSAD2", "CXCL10"],
    17: ["IFIT3", "ISG15", "OAS2"],
    18: ["IFIH1", "DHX58", "DDX60", "ISG20"],
    11: ["OASL", "CCL5", "CXCL9", "GBP5"],
    25: ["RIGI", "OAS1", "OAS3", "USP18"],
    24: ["IRF7", "STAT1", "TAP1", "PSMB9"],
    20: ["STAT2", "BST2", "EIF2AK2", "CD74"],
    23: ["IFI6", "IFITM1", "MX1"],
}


def rc(n, x_dim, y_dim):
    return y_dim - 1 - (n - 1) // x_dim, (n - 1) % x_dim


def centre(n, x_dim, y_dim):
    row, col = rc(n, x_dim, y_dim)
    return col * SQ3 + (row % 2) * SQ3 / 2, -row * 1.5


class Grid:

    def __init__(self, x_dim, y_dim):
        self.x, self.y = x_dim, y_dim
        self.n = x_dim * y_dim
        self.centres = {i: centre(i, x_dim, y_dim) for i in range(1, self.n + 1)}
        cxs = [x for x, _ in self.centres.values()]
        self.xlim = (min(cxs) - SQ3 / 2 - 0.5, max(cxs) + SQ3 / 2 + 0.5)
        self.ylim = (-(y_dim - 1) * 1.5 - 1.15, 1.15)
        self.cx = sum(self.xlim) / 2

    def hexagon(self, ax, n, radius=1.0, **kw):
        x, y = self.centres[n]
        ax.add_patch(RegularPolygon((x, y), 6, radius=radius, orientation=0, **kw))
        return x, y

    def frame(self, ax, pad_top=0.0, pad_bottom=0.0):
        ax.set_xlim(*self.xlim)
        ax.set_ylim(self.ylim[0] - pad_bottom, self.ylim[1] + pad_top)
        ax.set_aspect("equal")
        ax.axis("off")

    def outline_patch(self):
        verts, codes = [], []
        for n in self.centres:
            x, y = self.centres[n]
            pts = hex_ring(x, y)
            verts += pts + [pts[0]]
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * 5 + [MplPath.CLOSEPOLY]
        return mpl.patches.PathPatch(MplPath(verts, codes), fc="none", ec="none")


BOX_FC = lighten("#A8C6E8", 0.55)
BOX_EC = "#A8C6E8"


def ink(fc):
    r, g, b = to_rgb(fc)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else "#23262B"


COUNT_BIN = 5
COUNT_MAX_BINS = 24


def count_bin(counts):
    lo = COUNT_BIN * int(np.nanmin(counts) // COUNT_BIN)
    span = float(np.nanmax(counts)) - lo
    step, ladder, i = COUNT_BIN, (2, 2.5, 2), 0
    while span / step > COUNT_MAX_BINS:
        step *= ladder[i % len(ladder)]
        i += 1
    return int(step)


def count_norm(counts):
    step = count_bin(counts)
    lo = step * int(np.nanmin(counts) // step)
    hi = step * int(np.ceil(np.nanmax(counts) / step))
    return mpl.colors.BoundaryNorm(np.arange(lo, hi + step, step),
                                   COUNT_CMAP.N)


def count_ticks(counts):
    lo, hi = int(np.nanmin(counts)), int(np.nanmax(counts))
    return [lo, round(lo + (hi - lo) / 3), round(lo + 2 * (hi - lo) / 3), hi]


def dep_class(v):
    if not np.isfinite(v):
        return None
    return 0 if v < DEP_BREAKS[0] else (1 if v < DEP_BREAKS[1] else 2)


def fan_scale(cb, r0=FAN_R0, r_max=0.86, r_min=0.06):
    hi = min(float(np.nanmax(cb.to_numpy())), CLIP_Z)
    lo = max(float(np.nanmin(cb.to_numpy())), -CLIP_Z)
    return (min((r_max - r0) / max(hi, 1e-6), (r0 - r_min) / max(-lo, 1e-6)),
            lo, hi)


def _wedge_fc(i, virus, rest):
    base = VIRUS_COLORS[virus]
    return lighten(base, 0.50) if "Ruxo" in rest else base


def fan_glyph(ax, x, y, weights, k, r0=FAN_R0, lw=0.35, ring=True):
    return _fan_glyph(ax, x, y, weights, k, CONTRASTS, VIRUS_COLORS, r0=r0,
                      lw=lw, ring=ring, fc_fn=_wedge_fc)


def fan_key(ax, k, lo, hi, annotate, origin=(0.0, 0.0), frame=True):
    ax.set_aspect("equal")
    ax.axis("off")
    r0 = FAN_R0
    R = r0 + k * hi
    ox, oy = origin
    fan_glyph(ax, ox, oy, [hi] * 8, k=k, lw=0.5)
    for i, cond in enumerate(CONTRASTS):
        mid = np.deg2rad(90 - (i + 0.5) * 45)
        virus, rest = cond.split("_", 1)
        ax.text(ox + (R + 0.09) * np.cos(mid), oy + (R + 0.09) * np.sin(mid),
                COND_LABEL[rest], fontsize=5.2, color=VIRUS_COLORS[virus],
                va="center",
                ha="left" if np.cos(mid) > 0.05 else
                   ("right" if np.cos(mid) < -0.05 else "center"))
    for sign, v, ha in ((1, "WNV", "left"), (-1, "YFV", "right")):
        ax.text(ox + sign * R * 0.10, oy + R + 0.30, v, fontsize=6.5,
                fontweight="bold", color=VIRUS_COLORS[v], ha=ha, va="bottom")
    y = oy - (R + 0.30)
    ax.plot([ox + r0 + k * lo, ox + r0 + k * hi], [y, y], color="0.35", lw=0.6)
    for v, ha in ((lo, "right"), (hi, "left")):
        x = ox + r0 + k * v
        ax.plot([x, x], [y, y + 0.085], color="0.35", lw=0.5)
        sat = (v <= -CLIP_Z and ha == "right") or (v >= CLIP_Z and ha == "left")
        label = f"{v:+.1f}".replace("-", "−")
        ax.text(x, y - 0.035, ("≤ " if v < 0 else "≥ ") * sat + label, ha=ha,
                va="top", fontsize=4.8, color="0.35")
    ax.text(ox, y - 0.30, "codebook weight (z)" + ", winsorized" * (lo <= -CLIP_Z or hi >= CLIP_Z), ha="center", va="top",
            fontsize=5.5, color="0.35")
    bottom = y - 0.45
    if annotate:
        ax.text(ox, y - 0.52, "dashed ring = 0 · solid = untreated\n"
                "pale hatched inward = negative", ha="center", va="top",
                fontsize=5.2, color="0.45", linespacing=1.4)
        bottom = y - 0.92
    if frame:
        ax.set_xlim(ox - (R + 0.75), ox + R + 0.75)
        ax.set_ylim(bottom, oy + R + 0.62)


ISG_ALL_CALLOUT_GENES = {
    21: ["RSAD2", "CXCL10", "CXCL11", "IFIT1"],
    16: ["MX2", "IFI44L", "ZBP1"],
    6: ["ISG15", "IFIT3", "MX1", "OAS2"],
    1: ["OAS1", "OAS3", "USP18"],
}

def curated_callouts(assign, curated=None):
    curated = CALLOUT_GENES if curated is None else curated
    members = assign.groupby("neuron")["symbol"].agg(set)
    for n, genes in curated.items():
        if not set(genes) <= members.get(n, set()):
            return None
    return {n: list(genes) for n, genes in curated.items()}


def callouts(assign, stats, max_neurons=10):
    for table in (CALLOUT_GENES, ISG_ALL_CALLOUT_GENES):
        curated = curated_callouts(assign, table)
        if curated is not None:
            return curated
    hits = {}
    for n, grp in assign.groupby("neuron"):
        h = sorted(s for s in grp["symbol"].dropna() if s in NOTABLE_GENES)
        if h:
            hits[int(n)] = h
    ranked = sorted(hits, key=lambda n: (-len(hits[n]),
                                         -int(stats.loc[n, "n_genes"])))
    return {n: hits[n] for n in ranked[:max_neurons]}


def draw_callouts(ax, boxes, grid, sizes=None, per_box=CALLOUT_PER_BOX,
                  fontsize=5.4):
    BOX_PAD = 0.16
    MIN_GAP = 0.18
    sides = {"left": [], "right": []}
    for n in boxes:
        sides["left" if rc(n, grid.x, grid.y)[1] < grid.x / 2
              else "right"].append(n)
    while len(sides["left"]) - len(sides["right"]) > 1:
        sides["right"].append(sides["left"].pop(max(
            range(len(sides["left"])),
            key=lambda k: rc(sides["left"][k], grid.x, grid.y)[1])))
    while len(sides["right"]) - len(sides["left"]) > 1:
        sides["left"].append(sides["right"].pop(min(
            range(len(sides["right"])),
            key=lambda k: rc(sides["right"][k], grid.x, grid.y)[1])))
    drawn = []
    for side, members in sides.items():
        members = sorted(members, key=lambda n: grid.centres[n][1], reverse=True)
        if not members:
            continue
        slots = np.linspace(0.3, -(grid.y - 1) * 1.5 - 0.3, len(members))
        ha = "right" if side == "left" else "left"
        tx = (-2.1 if side == "left"
              else (grid.x - 1) * SQ3 + SQ3 / 2 + 2.1)
        for n, ty in zip(members, slots):
            shown = boxes[n][:per_box]
            genes = "\n".join(shown)
            rest = (int(sizes[n]) if sizes is not None else len(boxes[n])) - len(shown)
            if rest > 0:
                genes += "\n…"
            head = ax.text(tx, ty, f"$\\mathbf{{Neuron\\ {n}}}$", ha=ha,
                           va="bottom", fontsize=fontsize + 1.0, color="0.15",
                           zorder=8)
            body = ax.text(tx, ty - 0.08, genes, ha=ha, va="top",
                           fontsize=fontsize, color="0.20", linespacing=1.25,
                           zorder=8)
            drawn.append((side, head, body))
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    extents = {}
    for side, head, body in drawn:
        (x0, y0), (x1, y1) = inv.transform(
            Bbox.union([head.get_window_extent(), body.get_window_extent()]))
        extents[(side, head)] = (x0, y0, x1, y1)

    y_top, y_bot = 0.9, ax.get_ylim()[0] + BOX_PAD
    gaps = []
    for side in sides:
        hs = [extents[(side, h)][3] - extents[(side, h)][1] + 2 * BOX_PAD
              for s, h, b in drawn if s == side]
        if hs:
            gaps.append((y_top - y_bot - sum(hs)) / (len(hs) + 1))
    gap = max(min(gaps), MIN_GAP) if gaps else MIN_GAP
    for side in sides:
        cursor = y_top - gap
        for s, head, body in [d for d in drawn if d[0] == side]:
            x0, y0, x1, y1 = extents[(side, head)]
            dy = (cursor - BOX_PAD) - y1
            head.set_position((head.get_position()[0],
                               head.get_position()[1] + dy))
            body.set_position((body.get_position()[0],
                               body.get_position()[1] + dy))
            cursor -= (y1 - y0) + 2 * BOX_PAD + gap
    ax.figure.canvas.draw()

    for side, head, body in drawn:
        u = Bbox.union([head.get_window_extent(), body.get_window_extent()])
        (x0, y0), (x1, y1) = inv.transform(u)
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle=f"round,pad={BOX_PAD},"
                                             "rounding_size=0.10",
                                    fc=BOX_FC, ec=BOX_EC, lw=0.5, zorder=7.5,
                                    clip_on=False))


def switch_neurons(cb):
    ut = cb[[c for c in CONTRASTS if "untreated" in c]].abs().max(axis=1)
    r72 = cb[[f"{v}_Ruxo72" for v in VIRUSES]]
    return (list(cb.index[(r72.min(axis=1) >= 0.9) & (ut <= 0.35)]),
            list(cb.index[(r72.max(axis=1) <= -0.7) & (ut <= 0.35)]))


def codes_panel(fig, ax, cb, assign, stats, grid, annotate,
                side_pad=3.8, unit_label="genes"):
    counts = stats["n_genes"]
    cnorm = count_norm(counts)
    k, lo, hi = fan_scale(cb)
    for n in grid.centres:
        fc = COUNT_CMAP(cnorm(counts[n]))
        x, y = grid.hexagon(ax, n, fc=fc, ec="0.55", lw=0.5, zorder=1)
        fan_glyph(ax, x, y, cb.loc[n].to_numpy(), k=k)
        ax.text(x, y - 0.74, str(n), ha="center", va="center", fontsize=5.4,
                color="0.30", zorder=5)
    on, off = switch_neurons(cb)
    for n, mark in [(i, "^") for i in on] + [(i, "v") for i in off]:
        x, y = grid.centres[n]
        ax.scatter([x - 0.62], [y + 0.50], marker=mark, s=13, fc="#2F3338",
                   ec="none", zorder=6)
    grid.frame(ax, pad_top=3.30, pad_bottom=CALLOUT_PAD_BOTTOM)
    ax.set_xlim(grid.xlim[0] - side_pad, grid.xlim[1] + side_pad)
    draw_callouts(ax, callouts(assign, stats), grid,
                  sizes=stats["n_genes"])

    boundaries = cnorm.boundaries
    cax = ax.inset_axes([grid.cx - 4.8, grid.ylim[1] + 0.62, 8.0, 0.35],
                        transform=ax.transData)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        cax.add_patch(Rectangle((left, 0), right - left, 1,
                                fc=COUNT_CMAP(cnorm((left + right) / 2)),
                                ec="0.3", lw=0.4))
    cax.set_xlim(boundaries[0], boundaries[-1])
    cax.set_ylim(0, 1)
    cax.axis("off")
    for v in count_ticks(counts):
        cax.text(v, -0.18, str(v), ha="center", va="top", fontsize=5.5)
    cax.set_title(f"# of {unit_label} per neuron", fontsize=5.5, pad=2)
    fan_key(ax, k, lo, hi, annotate,
            origin=(grid.cx + 5.7, grid.ylim[1] + 1.78), frame=False)

    if annotate:
        notes = []
        if on:
            notes.append("▲ silent untreated, on only under Ruxo at 72 hpi "
                         f"({', '.join(map(str, on))})")
        if off:
            notes.append(f"▼ repressed specifically there ({', '.join(map(str, off))})")
        if notes:
            ax.text(grid.xlim[0] + 0.30, grid.ylim[0] - 0.42, "\n".join(notes),
                    fontsize=5.8, color="#3A3A3A", ha="left", va="center")


def surface(ax, values, grid, cmap, norm, levels, contours=()):
    pts = np.array([grid.centres[n] for n in values.index])
    rbf = RBFInterpolator(pts, values.to_numpy(), kernel="linear", smoothing=0.0)
    gx = np.linspace(*grid.xlim, 320)
    gy = np.linspace(*grid.ylim, 240)
    GX, GY = np.meshgrid(gx, gy)
    Z = np.clip(rbf(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape),
                levels[0], levels[-1])
    clip = grid.outline_patch()
    ax.add_patch(clip)
    cs = ax.contourf(GX, GY, Z, levels=levels, cmap=cmap, norm=norm)
    cs.set_clip_path(clip)
    cl = ax.contour(GX, GY, Z, levels=levels[::3], colors="white",
                    linewidths=0.35, alpha=0.55)
    cl.set_clip_path(clip)
    if contours:
        cc = ax.contour(GX, GY, Z, levels=list(contours), colors="#1F2328",
                        linewidths=0.8)
        cc.set_clip_path(clip)
        for txt in ax.clabel(cc, fmt="%.1f", fontsize=4.8, inline=True,
                             inline_spacing=2):
            if not clip.get_path().contains_point(txt.get_position()):
                txt.remove()


def overlay_grid(ax, grid, max_labels=100):
    labelled = len(grid.centres) <= max_labels
    lw = 0.7 if labelled else 0.2
    for n in grid.centres:
        x, y = grid.centres[n]
        ax.add_patch(Polygon(hex_ring(x, y), closed=True, fc="none",
                             ec="white", lw=lw, alpha=0.75, zorder=4))
        if labelled:
            ax.text(x, y, str(n), ha="center", va="center", fontsize=4.8,
                    color="#1F2328", zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="none",
                              alpha=0.62))


def low_n_outline(ax, stats, grid, viruses, cut=MIN_SCORED):
    flagged = []
    for n in grid.centres:
        if min(stats.loc[n, f"n_jak_{v.lower()}"] for v in viruses) < cut:
            flagged.append(n)
            grid.hexagon(ax, n, radius=0.93, fc="none", ec="#2F3338", lw=0.7,
                         ls=(0, (1.4, 1.4)), zorder=7)
    return flagged


def discrete_map(ax, values, stats, grid, cmap, norm, viruses,
                 cut=MIN_SCORED):
    for n in grid.centres:
        v = values.get(n, np.nan)
        solid = (np.isfinite(v)
                 and min(stats.loc[n, f"n_jak_{x.lower()}"] for x in viruses) >= cut)
        fc = "#EDEDED" if not np.isfinite(v) else cmap(norm(v))
        x, y = grid.hexagon(ax, n, fc=fc, ec="white", lw=1.4,
                            alpha=1.0 if solid else 0.42)
        if not solid and np.isfinite(v):
            grid.hexagon(ax, n, radius=0.93, fc="none", ec="#2F3338", lw=0.7,
                         ls=(0, (1.4, 1.4)))
        ax.text(x, y + 0.20, str(n), ha="center", va="center", fontsize=5.4,
                color=ink(fc) if solid else "0.30")
        ax.text(x, y - 0.34, "—" if not np.isfinite(v) else f"{v:+.2f}",
                ha="center", va="center", fontsize=4.8,
                color=ink(fc) if solid else "0.40")


def can_interpolate(stats, virus, min_genes=MIN_FOR_SURFACE[0],
                    min_neurons=MIN_FOR_SURFACE[1]):
    return int((stats[f"n_jak_{virus.lower()}"] >= min_genes).sum()) >= min_neurons


def inset_cbar(fig, ax, cmap, norm, ticks, title, x0=0.10):
    cax = ax.inset_axes([x0, -0.135, 0.46, 0.050])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       orientation="horizontal", ticks=ticks)
    bar.ax.tick_params(labelsize=5, length=1.4, pad=1)
    bar.outline.set_linewidth(0.4)
    bar.ax.set_title(title, fontsize=5.5, pad=2, loc="left")


def load_inputs(data_dir, tag):
    return load_universe(data_dir, tag)


def grid_dims(data_dir, tag):
    path = data_dir / f"som_summary{out_suffix(tag)}.json"
    if not path.exists():
        return GRID_X, GRID_Y
    x, y = json.loads(path.read_text())["grid"].split("x")
    return int(x), int(y)


def unit_label(tag):
    return "ISG" if tag == "isg" else "genes"


def draw_hybrid(fig, gs_slot, cb, assign, stats, annotate=False, letters=None,
                top_ratio=1.35, side_pad=3.8, sub_hspace=0.22,
                diff=True, unit_label="genes", grid_xy=None):
    if cb.shape[1] != len(CONTRASTS):
        raise ValueError(
            f"the hybrid figure is defined for the {len(CONTRASTS)} WNV/YFV "
            f"conditions; this codebook has {cb.shape[1]} columns")
    grid = Grid(*(grid_xy or (GRID_X, GRID_Y)))
    if grid.n < len(stats):
        raise ValueError(f"{len(stats)} neurons trained but the lattice is "
                         f"{grid.x}x{grid.y} = {grid.n}")
    l_codes, l_wnv, l_yfv, l_diff = letters or (None, None, None, None)

    outer = gs_slot.subgridspec(2, 1, height_ratios=[top_ratio, 1.0],
                                hspace=sub_hspace)
    gs_bot = outer[1].subgridspec(1, 3 if diff else 2, wspace=0.05)

    ax_a = fig.add_subplot(outer[0])
    codes_panel(fig, ax_a, cb, assign, stats, grid, annotate,
                side_pad=side_pad, unit_label=unit_label)
    if l_codes:
        ax_a.set_title(f"$\\mathbf{{{l_codes}}}$", loc="left", fontsize=7.5)

    n_dep = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    lev = np.linspace(-1, 1, 21)
    for i, (virus, l_virus) in enumerate(zip(VIRUSES, (l_wnv, l_yfv))):
        ax = fig.add_subplot(gs_bot[0, i])
        col = f"mean_jak_index_{virus.lower()}"
        if can_interpolate(stats, virus):
            surface(ax, stats[col].dropna(), grid, "RdBu_r", n_dep, lev,
                    contours=DEP_BREAKS)
            overlay_grid(ax, grid)
            low_n_outline(ax, stats, grid, [virus])
        else:
            discrete_map(ax, stats[col], stats, grid, plt.get_cmap("RdBu_r"),
                         n_dep, [virus])
        grid.frame(ax, pad_top=1.35)
        if l_virus:
            ax.set_title(f"$\\mathbf{{{l_virus}}}$  JAK-STAT dependence — {virus}",
                         loc="left", fontsize=7.5)
        else:
            ax.set_title(f"JAK-STAT dependence — {virus}" if annotate else virus,
                         color=VIRUS_COLORS[virus], fontsize=7.5)
        if i == 0:
            inset_cbar(fig, ax, "RdBu_r", n_dep, [-1, -0.5, 0, 0.5, 1],
                       "mean dependence index", x0=0.10)

    if diff:
        ax_d = fig.add_subplot(gs_bot[0, 2])
        residual = stats["mean_jak_index_wnv"] - stats["mean_jak_index_yfv"]
        n_dif = TwoSlopeNorm(vmin=-0.9, vcenter=0, vmax=0.9)
        discrete_map(ax_d, residual, stats, grid, plt.get_cmap("PRGn"), n_dif,
                     VIRUSES)
        grid.frame(ax_d, pad_top=1.35)
        if l_diff:
            ax_d.set_title(f"$\\mathbf{{{l_diff}}}$  WNV − YFV", loc="left",
                           fontsize=7.5)
        else:
            ax_d.set_title("WNV − YFV: what is left over" if annotate
                           else "WNV − YFV", fontsize=7.5)
        inset_cbar(fig, ax_d, "PRGn", n_dif, [-0.5, 0, 0.5],
                   "difference in mean dependence index", x0=0.10)
    return ax_a


STANDALONE_W, STANDALONE_H, REF_GRID_Y = 183, 170, 4


def lattice_height(y_dim):
    return (y_dim - 1) * 1.5 + 2.3


def hybrid_figure(cb, assign, stats, annotate=False, unit_label="genes",
                  grid_xy=None):
    y_dim = (grid_xy or (GRID_X, GRID_Y))[1]
    height = STANDALONE_H * lattice_height(y_dim) / lattice_height(REF_GRID_Y)
    fig = plt.figure(figsize=(STANDALONE_W * MM, height * MM))
    bottom = (0.10 if annotate else 0.02) + 0.045 * max(0, y_dim - REF_GRID_Y)
    outer = fig.add_gridspec(1, 1, left=0.012, right=0.982,
                             top=0.86 if annotate else 0.99,
                             bottom=bottom)
    draw_hybrid(fig, outer[0], cb, assign, stats, annotate,
                unit_label=unit_label, grid_xy=grid_xy)

    if annotate:
        fig.text(0.012, 0.962, "Supplementary Fig. S10 · design F | "
                 "Kohonen codes plot + interpolated JAK-dependence terrain",
                 fontsize=8, fontweight="bold", va="bottom")
        for i, line in enumerate((
                "a is discrete — one glyph per neuron, nothing drawn between them.",
                "b and c interpolate the dependence index between the neuron "
                "centres (linear RBF); black lines are the 0.2 / 0.5 class contours.",
                "d returns to discrete neurons: the residual between the two "
                "viruses is carried by the sparsely scored ones.")):
            fig.text(0.012, 0.936 - i * 0.021, line, fontsize=6, color="0.35",
                     va="bottom")
    return fig


def main():
    ap = argparse.ArgumentParser(description='the WNV/YFV-focused SOM figure - codes plot + dependence terrain.')
    ap.add_argument("--tag", default="isg",
                    help="gene-set tag of the trained map (isg or wnvyfv - the "
                         "figure needs the 8 WNV/YFV conditions)")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=FIGURES / "plot_som_map")
    ap.add_argument("--annotate", action="store_true",
                    help="stand-alone version with titles and explanatory notes")
    args = ap.parse_args()

    set_rcparams()
    cb, assign, stats = load_inputs(args.data_dir, args.tag)
    fig = hybrid_figure(cb, assign, stats, args.annotate,
                        unit_label=unit_label(args.tag),
                        grid_xy=grid_dims(args.data_dir, args.tag))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"figs_som_{args.tag}_hybrid.pdf"
    written = save_figure(fig, out)
    plt.close(fig)
    print("wrote " + ", ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
