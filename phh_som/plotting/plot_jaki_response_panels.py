#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import (LinearSegmentedColormap, Normalize)

from phh_som.utils.pipeline import (DATA, FIGURES, MM, VIRUSES, save_figure,
                      set_rcparams, virus_colors)
from phh_som.utils.jak import TOP_N

VIRUS_COLORS = {v: virus_colors()[v] for v in VIRUSES}
UP_REGULATED = "#A01915"
DOWN_REGULATED = "#1C8AA1"

SHARED_COLOR = "#3C4348"

GENE_SET = "ISG"


def fmt_p(p):
    if p < 1e-3:
        mantissa, exponent = f"{p:.1e}".split("e")
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        return f"p = {mantissa}×10" + str(int(exponent)).translate(sup)
    return f"p = {p:.3f}"


def load_utrux(data_dir=DATA):
    return {v: pd.read_csv(data_dir / f"utrux_stats_{v.lower()}.csv")
            .set_index("gene_id") for v in VIRUSES}


def heatmap_gene_sets(data_dir=DATA, gene_set="ISG"):
    orders = {}
    for virus in ("wnv", "yfv"):
        if gene_set == "DEG":
            tab = pd.read_csv(data_dir / f"utrux_stats_{virus}.csv")
            deg = tab.loc[tab["deg"]].dropna(subset=["log2fc"])
            orders[f"up_{virus}"] = deg.nlargest(50, "log2fc")["gene_id"].tolist()
            orders[f"resist_{virus}"] = deg.nsmallest(50, "log2fc")["gene_id"].tolist()
        else:
            orders[f"up_{virus}"] = pd.read_csv(
                data_dir / f"top50_isg_up_{virus}.csv")["gene_id"].tolist()
            orders[f"resist_{virus}"] = pd.read_csv(
                data_dir / f"top50_isg_down_{virus}.csv")["gene_id"].tolist()
    return orders


def heatmap_panels(utrux, orders, gene_set=GENE_SET):
    panels = {}
    for virus, up_key, down_key in (("WNV", "g", "i"), ("YFV", "h", "j")):
        symbols = utrux[virus]["symbol"]
        tab = utrux[virus]
        other = "YFV" if virus == "WNV" else "WNV"
        for key, tail, direction in ((up_key, "up", "up"), (down_key, "resist", "down")):
            genes = sorted(orders[f"{tail}_{virus.lower()}"],
                           key=lambda g: str(symbols[g]).upper())
            shared = (set(orders[f"{tail}_{virus.lower()}"])
                      & set(orders[f"{tail}_{other.lower()}"]))
            panels[key] = dict(
                title=f"top-50 {'up' if tail == 'up' else 'down'}-regulated "
                      f"{gene_set}s, {virus} - log₂ FC vs JAKi, 72 hpi",
                genes=genes,
                row_values=[pd.Series(0, index=tab.index), tab["log2fc"]],
                row_labels=[f"{virus} + JAKi", virus],
                row_colors=["black", "black"],
                symbols=symbols,
                direction=direction,
                shared=shared,
                shared_with=other,
                virus=virus,
            )
    return panels


def isg_population(utrux):
    tab = utrux[VIRUSES[0]]
    return int((tab["is_isg"] & tab["is_reliable"]
                & tab["log2fc"].notna()).sum())


def venn_counts(orders, tail):
    w, y = set(orders[f"{tail}_wnv"]), set(orders[f"{tail}_yfv"])
    return len(w - y), len(w & y), len(y - w)


def ranked_isgs(utrux):
    def sel(tab):
        return tab.loc[tab["is_isg"] & tab["is_reliable"]].dropna(
            subset=["log2fc"])["log2fc"]

    w, y = sel(utrux["WNV"]), sel(utrux["YFV"])
    common = w.index.intersection(y.index)
    return w[common], y[common]


def draw_utrux_scatter(ax, utrux, orders, title=""):
    from scipy.stats import spearmanr

    w, y = ranked_isgs(utrux)
    rho, pval = spearmanr(w, y)
    shared = {t: (set(orders[f"{t}_wnv"]) & set(orders[f"{t}_yfv"])
                  & set(w.index)) for t in ("up", "resist")}
    rest = [g for g in w.index if not any(g in v for v in shared.values())]

    lim = float(max(w.abs().max(), y.abs().max())) * 1.06
    ax.axline((0, 0), slope=1, color="0.80", lw=0.7, ls=(0, (4, 3)), zorder=1)
    for f in (ax.axhline, ax.axvline):
        f(0, color="0.85", lw=0.6, zorder=1)
    ax.scatter(w[rest], y[rest], s=4, c="0.78", lw=0, zorder=2, rasterized=True)
    for tail, color, lbl in (("up", UP_REGULATED, "shared JAKi-suppressed"),
                             ("resist", DOWN_REGULATED, "shared JAKi-resistant")):
        g = sorted(shared[tail])
        ax.scatter(w[g], y[g], s=6, c=color, lw=0, zorder=3,
                   label=f"{lbl} ({len(g)})")
    ax.text(0.03, 0.97,
            f"Spearman $\\rho$ = {rho:.2f}\n{fmt_p(pval).replace('p =', 'P =')}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
            color="0.25")
    ax.legend(loc="lower right", frameon=False, fontsize=5.2,
              handletextpad=0.2, borderpad=0.0, labelspacing=0.25,
              markerscale=1.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("WNV, untreated − JAKi (log₂ FC)", fontsize=6)
    ax.set_ylabel("YFV, untreated − JAKi (log₂ FC)", fontsize=6)
    ax.tick_params(labelsize=5.5, length=2, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if title:
        ax.set_title(title, loc="center", fontsize=7)
    return ax


def draw_topn_venn(ax, orders, utrux, tail="up", title="", top_n=TOP_N):
    from scipy.stats import hypergeom

    only_w, shared, only_y = venn_counts(orders, tail)
    n_pop = isg_population(utrux)
    pval = hypergeom.sf(shared - 1, n_pop, top_n, top_n)

    r, dx = 1.0, 0.58
    for cx, virus in ((-dx, "WNV"), (dx, "YFV")):
        ax.add_patch(mpl.patches.Circle((cx, 0), r, fc=VIRUS_COLORS[virus],
                                        ec="none", alpha=0.55, zorder=2))
        ax.add_patch(mpl.patches.Circle((cx, 0), r, fc="none", ec="0.35",
                                        lw=0.8, zorder=4))
        ax.text(cx + (-1 if cx < 0 else 1) * 0.62, r + 0.16, virus,
                ha="center", va="bottom", fontsize=6.5, color="0.15",
                fontweight="bold")
    for x, n in ((-dx - 0.52, only_w), (0.0, shared), (dx + 0.52, only_y)):
        ax.text(x, 0, str(n), ha="center", va="center", fontsize=8,
                color="0.10", zorder=5)
    ax.text(0, -r - 0.30, "hypergeometric " + fmt_p(pval).replace("p =", "P ="),
            ha="center", va="top", fontsize=5.8, color="0.30")
    if title:
        ax.set_title(title, loc="center", fontsize=7)
    ax.set_xlim(-dx - r - 0.75, dx + r + 0.75)
    ax.set_ylim(-r - 0.85, r + 0.55)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def draw_isg_volcano(ax, tab, virus, letter="", gene_set="DEG"):
    tab = tab.dropna(subset=["log2fc", "fdr"])
    tab = tab.assign(neglog10fdr=-np.log10(tab["fdr"].clip(lower=1e-300)).clip(upper=200))
    color = VIRUS_COLORS[virus]
    isg = tab.loc[tab["is_isg"]]
    if gene_set == "DEG":
        background = tab.loc[~tab["is_isg"]]
        sc_deg = ax.scatter(background["log2fc"], background["neglog10fdr"],
                            s=5, c="0.75", linewidths=0, zorder=2,
                            rasterized=True)
    sc_isg = ax.scatter(isg["log2fc"], isg["neglog10fdr"], s=12,
                        c=color, linewidths=0, zorder=3, rasterized=True)
    if gene_set == "DEG":
        ax.legend([sc_isg, sc_deg], ["ISG", "Non-ISG"], loc="upper left",
                  bbox_to_anchor=(0.02, 0.98), fontsize=6, frameon=False,
                  handletextpad=0.3, borderaxespad=0, markerscale=1.2)
    ax.axvline(1, color="0.4", lw=0.6, ls="--")
    ax.axvline(-1, color="0.4", lw=0.6, ls="--")
    ax.set_xlim(-15, 15)
    ax.set_ylim(0, 205)
    ax.set_yticks(range(0, 201, 50))
    ax.set_xlabel("Log$_2$ fold change", fontsize=6.5)
    ax.set_ylabel("-Log$_{10}$ FDR p-value", fontsize=6.5)
    ax.tick_params(labelsize=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(f"$\\mathbf{{{letter}}}$  UT vs JAKi - {virus} 72 hpi"
                 if letter else f"UT vs JAKi - {virus} 72 hpi",
                 loc="left", fontsize=7.5)


def draw_isg_waterfall(ax, tab, virus, letter="", gene_set=None):
    gene_set = gene_set or GENE_SET
    tab = tab.dropna(subset=["log2fc"])
    n_isg_total = int(tab["is_isg"].sum())
    if gene_set == "ISG":
        tab = tab.loc[tab["is_isg"]]
    else:
        tab = tab.loc[tab["deg"]]
    ranked = tab.sort_values("log2fc", ascending=False).reset_index(drop=True)
    x = np.arange(len(ranked))
    color = VIRUS_COLORS[virus]
    if gene_set == "ISG":
        fill = mpl.colors.to_hex(
            tuple(1 - 0.25 * (1 - c) for c in mpl.colors.to_rgb(color)))
        ax.scatter(x, ranked["log2fc"], s=14, facecolors=fill,
                   edgecolors="black", linewidths=0.4, zorder=2)
        ax.text(0.03 * len(ranked), 9.3, virus, fontsize=7, fontweight="bold",
                color="black", ha="left", va="center",
                bbox=dict(fc=color, ec="none", alpha=0.45, pad=2.5))
        ax.text(0.45, 0.97, f"n = {len(ranked)} ISGs", fontsize=6.5,
                color="black", ha="left", va="top", transform=ax.transAxes)
    else:
        ax.scatter(x, ranked["log2fc"], s=2, c="0.75", linewidths=0, zorder=2,
                   rasterized=True)
        isg = ranked["is_isg"].to_numpy()
        ax.scatter(x[isg], ranked["log2fc"].to_numpy()[isg], s=4, c=color,
                   linewidths=0, zorder=3, rasterized=True)
        ax.text(0.02, 0.97, f"{int(isg.sum())}/{n_isg_total} ISGs highlighted",
                fontsize=5.5, color=color, ha="left", va="top",
                transform=ax.transAxes)
        ax.text(0.98, 0.85, f"{virus} (n = {len(ranked):,} DEGs)", fontsize=6.5,
                color=color, ha="right", va="top", transform=ax.transAxes)
    ax.axhline(0, color="0.5", lw=0.6, ls="--")
    ax.set_xlim(-0.02 * len(ranked), (len(ranked) - 1) * 1.01)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(-10, 10)
    ax.set_yticks(range(-10, 11, 5))
    ax.set_ylabel("UT vs JAKi 72 hpi\n(log$_2$ FC)", fontsize=6.5)
    ax.tick_params(labelsize=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if letter:
        ax.set_title(f"$\\mathbf{{{letter}}}$", loc="left", fontsize=7.5)


def _draw_membership_track(fig, ax, genes, shared, virus, n_rows,
                           gap=0.30, height=0.55):
    y0 = n_rows - 0.5 + gap
    specific = VIRUS_COLORS.get(virus, "#C7CDD1")
    for col, g in enumerate(genes):
        ax.add_patch(mpl.patches.Rectangle(
            (col - 0.4, y0), 0.8, height, clip_on=False, ec="none", zorder=4,
            fc=SHARED_COLOR if g in shared else specific))
    fig.canvas.draw()
    unit_px = abs(ax.transData.transform((0, 1))[1]
                  - ax.transData.transform((0, 0))[1])
    ax.tick_params(axis="x",
                   pad=(gap + height) * unit_px * 72 / fig.dpi + 1)


def draw_isg_heatmap(
    fig,
    gs_slot,
    title,
    genes,
    row_values,
    row_labels,
    row_colors,
    symbols,
    direction="up",
    shared=None,
    shared_with=None,
    virus=None,
    track_legend=False,
    standalone=False,
    cap_intensity=0.75,
    legend=True,
):

    def _soften(hex_color, frac):
        r, g, b = mpl.colors.to_rgb(hex_color)
        return "#{:02x}{:02x}{:02x}".format(
            *(int(round(255 * (1 - frac) + 255 * c * frac)) for c in (r, g, b))
        )

    if not 0 < cap_intensity <= 1:
        raise ValueError("cap_intensity must be in (0, 1]")
    vmax = 6 / cap_intensity
    if direction == "up":
        full = UP_REGULATED
        cap = _soften(full, cap_intensity)
        cmap = LinearSegmentedColormap.from_list(
            "up_cmap",
            [
                (0.0, "#f7f7f7"),
                (cap_intensity * 5 / 18, _soften(full, 0.357)),
                (cap_intensity * 5 / 9, _soften(full, 0.678)),
                (cap_intensity, cap),
                (1.0, full),
            ],
        )
        extend = "max"
    else:
        full = DOWN_REGULATED
        cap = _soften(full, cap_intensity)
        cmap = LinearSegmentedColormap.from_list(
            "down_cmap",
            [
                (0.0, full),
                (1 - cap_intensity, cap),
                (1 - cap_intensity * 5 / 9, "#65AEBE"),
                (1 - cap_intensity * 5 / 18, "#AED3DA"),
                (1.0, "#f7f7f7"),
            ],
        )
        extend = "min"

    if direction == "up":
        norm = Normalize(vmin=0, vmax=vmax)
    else:
        norm = Normalize(vmin=-vmax, vmax=0)

    mat = pd.concat(row_values, axis=1).loc[genes].T.to_numpy()
    if standalone:
        ax = fig.add_subplot(gs_slot)
        aspect = "equal"
    else:
        ax = fig.add_subplot(gs_slot)
        aspect = "auto"
    im = ax.imshow(
        mat,
        cmap=cmap,
        norm=norm,
        aspect=aspect,
        interpolation="nearest",
    )
    for r in range(mat.shape[0] + 1):
        ax.axhline(r - 0.5, color="0.75", lw=0.5, zorder=3)
    for col in range(mat.shape[1] + 1):
        ax.axvline(col - 0.5, color="0.75", lw=0.5, zorder=3)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels([symbols[g] for g in genes], rotation=90, fontsize=5.5)
    if shared:
        _draw_membership_track(fig, ax, genes, shared, virus, mat.shape[0])
        if track_legend:
            ax.legend(
                handles=[mpl.patches.Rectangle((0, 0), 1, 1, fc=SHARED_COLOR,
                                               ec="none")],
                labels=["shared between viruses"],
                loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
                fontsize=5.5, handlelength=0.9, handleheight=0.9,
                borderpad=0.0, handletextpad=0.4)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=8)
    if standalone:
        if legend:
            legend_slot = gs_slot.get_gridspec()[0, 1]
            _draw_custom_heatmap_legend(
                fig, legend_slot, direction=direction, cap_intensity=cap_intensity
            )
    else:
        cw, cfs, tfs, tpad, cy, ch = 0.008, 5.5, 5, 2, 0.15, 0.70
        cax = ax.inset_axes([1.006, cy, cw, ch], transform=ax.transAxes)
        cb = fig.colorbar(im, cax=cax, extend=extend)
        cb.ax.tick_params(labelsize=tfs, length=1.5, pad=tpad)
        cb.ax.yaxis.set_ticks_position("right")
        cb.ax.text(
            -0.5,
            0.5,
            "Log₂ FC\nversus JAKi",
            transform=cb.ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=cfs,
        )
        if direction == "up":
            cb.set_ticks([0, 2, 4, 6])
            cb.ax.set_yticklabels(["0", "2", "4", "6"])
            cb.ax.text(
                1.12,
                1.0,
                ">6",
                transform=cb.ax.transAxes,
                ha="left",
                va="center",
                fontsize=tfs,
            )
        elif direction == "down":
            cb.set_ticks([-6, -4, -2, 0])
            cb.ax.set_yticklabels(["-6", "-4", "-2", " 0"])
            cb.ax.text(
                1.12,
                0.0,
                "< -6",
                transform=cb.ax.transAxes,
                ha="left",
                va="center",
                fontsize=tfs,
            )
    return ax


def _draw_custom_heatmap_legend(
    fig, legend_slot, direction="up", cap_intensity=0.9, scale=1.0
):

    def _soften(hex_color, frac):
        r, g, b = mpl.colors.to_rgb(hex_color)
        return "#{:02x}{:02x}{:02x}".format(
            *(int(round(255 * (1 - frac) + 255 * c * frac)) for c in (r, g, b))
        )

    def _fx(mm):
        return mm / 25.4 / fig_w

    def _fy(mm):
        return mm / 25.4 / fig_h

    full = UP_REGULATED if direction == "up" else DOWN_REGULATED
    cap = _soften(full, cap_intensity)
    fs = 7 * scale

    fig_w, fig_h = fig.get_size_inches()
    pos = legend_slot.get_position(fig)
    bar_w = _fx(2.6 * scale)
    x0 = pos.x0 + (pos.width - bar_w) / 2
    content_mm = (3.2 + 1.8 + 13.0 + 2.0 + 5.0) * scale
    slot_mm = pos.height * fig_h * 25.4
    y_top = pos.y1 - _fy(max((slot_mm - content_mm) / 2, 1.0 * scale))
    box_h = _fy(3.2 * scale)
    gap = _fy(1.8 * scale)
    grad_h = _fy(13.0 * scale)
    label_gap = _fy(2.0 * scale)

    if direction == "up":
        box_y = y_top - box_h
        grad_y = box_y - gap - grad_h
        main_norm = Normalize(vmin=0, vmax=6)
        main_ticks = [0, 2, 4, 6]
        over_label = ">6"
        main_cmap = LinearSegmentedColormap.from_list(
            "up_legend_cmap",
            [(0.0, "#f7f7f7"), (0.5, _soften(full, 0.678)), (1.0, cap)]
        )
    else:
        grad_y = y_top - grad_h
        box_y = grad_y - gap - box_h
        main_norm = Normalize(vmin=-6, vmax=0)
        main_ticks = [-6, -4, -2, 0]
        over_label = "< -6"
        main_cmap = LinearSegmentedColormap.from_list(
            "down_legend_cmap", [(0.0, cap), (0.5, "#65AEBE"), (1.0, "#f7f7f7")]
        )

    sm_main = mpl.cm.ScalarMappable(norm=main_norm, cmap=main_cmap)
    sm_main.set_array([])
    cax_main = fig.add_axes([x0, grad_y, bar_w, grad_h])
    cbar_main = fig.colorbar(
        sm_main, cax=cax_main, orientation="vertical", ticks=main_ticks
    )
    cbar_main.ax.tick_params(length=3 * scale, width=1, labelsize=fs)
    cbar_main.outline.set_linewidth(1.0)

    over_cmap = mpl.colors.ListedColormap([full])
    sm_over = mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=over_cmap)
    sm_over.set_array([])
    cax_over = fig.add_axes([x0, box_y, bar_w, box_h])
    cbar_over = fig.colorbar(sm_over, cax=cax_over, orientation="vertical")
    cbar_over.set_ticks([])
    cbar_over.outline.set_linewidth(1.0)
    cax_over.text(
        1.3,
        0.5,
        over_label,
        transform=cax_over.transAxes,
        ha="left",
        va="center",
        fontsize=fs,
    )

    fig.text(
        pos.x0 + pos.width / 2,
        min(grad_y, box_y) - label_gap,
        "Log₂ FC\nversus JAKi",
        ha="center",
        va="top",
        fontsize=6.5 * scale,
        linespacing=0.85,
    )


def panel_figures(panels, utrux, orders, out_dir):
    specs = {
        "c": ((70 * MM, 80 * MM), "WNV", draw_isg_volcano),
        "d": ((180 * MM, 40 * MM), "WNV", draw_isg_waterfall),
        "e": ((70 * MM, 80 * MM), "YFV", draw_isg_volcano),
        "f": ((180 * MM, 40 * MM), "YFV", draw_isg_waterfall),
    }
    for letter, (size, virus, draw) in specs.items():
        fig = plt.figure(figsize=size)
        draw(fig.add_subplot(111), utrux[virus], virus)
        fig.tight_layout()
        out = out_dir / f"figure_som_wnvyfv_isg_jaki_panel_{letter}.pdf"
        written = save_figure(fig, out, dpi=600)
        plt.close(fig)
        print("wrote " + ", ".join(str(p) for p in written))
    fig = plt.figure(figsize=(180 * MM, 56 * MM))
    gs_v = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.30)
    draw_utrux_scatter(fig.add_subplot(gs_v[0, 0]), utrux, orders,
                       title=f"all {GENE_SET}s, both viruses")
    for col, (tail, lbl) in enumerate((("up", "JAKi-suppressed"),
                                       ("resist", "JAKi-resistant")), start=1):
        draw_topn_venn(fig.add_subplot(gs_v[0, col]), orders, utrux, tail=tail,
                       title=f"top-50 {lbl}")
    out = out_dir / "figure_som_wnvyfv_isg_jaki_panel_k.pdf"
    written = save_figure(fig, out, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote " + ", ".join(str(p) for p in written))

    for letter in ("g", "h", "i", "j"):
        fig = plt.figure(figsize=(200 * MM, 28 * MM))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])
        draw_isg_heatmap(fig, gs[0, 0], **panels[letter], standalone=True,
                         track_legend=True)
        out = out_dir / f"figure_som_wnvyfv_isg_jaki_panel_{letter}.pdf"
        written = save_figure(fig, out, dpi=600, bbox_inches="tight",
                              pad_inches=0.02)
        plt.close(fig)
        print("wrote " + ", ".join(str(p) for p in written))


def main():
    ap = argparse.ArgumentParser(description='standalone UT-vs-JAKi panels - volcanoes, waterfalls, heatmaps.')
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=FIGURES / "plot_jaki_response_panels")
    args = ap.parse_args()

    set_rcparams()
    utrux = load_utrux(args.data_dir)
    orders = heatmap_gene_sets(args.data_dir)
    panels = heatmap_panels(utrux, orders)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_figures(panels, utrux, orders, args.out_dir)


if __name__ == "__main__":
    main()
