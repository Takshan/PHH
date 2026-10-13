#!/usr/bin/env python3
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import pipeline

VIRUSES = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]
COLORS = json.loads((pipeline.ROOT / "config" / "colors.json").read_text())["viruses"]

EXPRESSED = pipeline.PUBLISH / "expressed_28x27" / "data"


def supported_neurons(induced_only=False):
    asg = pd.read_csv(EXPRESSED / "som_neuron_assignments_expressed.tsv", sep="\t")
    fc = pd.read_parquet(EXPRESSED / "expressed_matrix_masked_log2fc.parquet")
    degs = pd.read_parquet(pipeline.DATA / "degs.parquet")
    per_gene = degs.groupby("gene_id").agg(maxtpm=("maxgroupmeans", "max"))
    n_deg = degs[degs.deg].groupby("gene_id").size().rename("n_deg")
    genes = per_gene.join(n_deg).fillna({"n_deg": 0})
    a = asg.set_index("gene_id").join(genes)
    stats_ = a.groupby("neuron").agg(
        n=("neuron", "size"), med_tpm=("maxtpm", "median"), med_ndeg=("n_deg", "median")
    )
    centroid = fc.join(asg.set_index("gene_id")["neuron"]).groupby("neuron").mean()
    untreated = [f"{v}_untreated{h}" for v in VIRUSES for h in (24, 72)]
    amp = pd.DataFrame(
        {
            v: np.where(
                centroid[f"{v}_untreated24"].abs() >= centroid[f"{v}_untreated72"].abs(),
                centroid[f"{v}_untreated24"],
                centroid[f"{v}_untreated72"],
            )
            for v in VIRUSES
        },
        index=centroid.index,
    )
    supported = stats_[(stats_.n >= 4) & (stats_.med_tpm > 3) & (stats_.med_ndeg >= 4)]
    keep = supported.index
    if induced_only:
        keep = keep[centroid.loc[keep, untreated].mean(axis=1) > 0]
    return amp.loc[keep]


def gain_fits(amp=None):
    amp = supported_neurons() if amp is None else amp
    consensus = amp.mean(axis=1)
    return amp, consensus, {v: stats.linregress(consensus, amp[v]) for v in VIRUSES}


def onset_percentages():
    degs = pd.read_parquet(pipeline.DATA / "degs.parquet")
    d = degs[(degs.treatment == "untreated") & degs.deg]
    counts = d.groupby(["virus", "hpi"]).gene_id.nunique().unstack()
    return (100 * counts[24] / counts[72]).reindex(VIRUSES)


def draw_gain(ax, amp=None, legend_fontsize=5.6):
    amp, consensus, fits = gain_fits(amp)
    grid = np.linspace(consensus.min() - 0.3, consensus.max() + 0.3, 50)
    for v in VIRUSES:
        ax.scatter(consensus, amp[v], s=9, alpha=0.75, color=COLORS[v],
                   edgecolors="none", zorder=3)
        ax.plot(grid, fits[v].slope * grid + fits[v].intercept, color=COLORS[v],
                lw=1.2, zorder=4, label=v)
    ax.plot(grid, grid, color="0.35", lw=0.8, ls=(0, (4, 3)), zorder=2, label="unity")
    ax.set_xlabel("five-virus consensus module amplitude (log$_2$FC)")
    ax.set_ylabel("per-virus module amplitude (log$_2$FC)")
    ax.legend(frameon=False, fontsize=legend_fontsize, loc="upper left",
              handlelength=1.4, borderpad=0.2, labelspacing=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return fits


def draw_onset(ax, onset=None):
    onset = onset_percentages() if onset is None else onset
    bars = ax.bar(range(len(VIRUSES)), onset.values,
                  color=[COLORS[v] for v in VIRUSES], width=0.62,
                  edgecolor="0.25", linewidth=0.4)
    for rect, value in zip(bars, onset.values):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 1.2, f"{value:.0f}%",
                ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(VIRUSES)))
    ax.set_xticklabels(VIRUSES)
    ax.set_ylabel("24 hpi DEGs as % of 72 hpi")
    ax.set_ylim(0, max(onset.values) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    return onset


def panel_gain_timing(out_dir):
    amp = supported_neurons()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(180 * pipeline.MM, 62 * pipeline.MM))
    fits = draw_gain(ax1, amp=amp)
    onset = draw_onset(ax2)
    fig.tight_layout()
    print(f"  gain fit over {len(amp)} supported neurons (induced and repressed)")
    for v in VIRUSES:
        print(f"  gain {v:<5}: slope {fits[v].slope:.2f}  "
              f"R^2 {fits[v].rvalue**2:.2f}   onset {onset[v]:.0f}%")
    return pipeline.save_figure(fig, Path(out_dir) / "figure_panflavi_gain_timing.pdf",
                                bbox_inches="tight")


def main():
    pipeline.set_rcparams()
    out_dir = pipeline.FIGURES / "plot_panflavi_panels"
    print("wrote", ", ".join(str(p) for p in panel_gain_timing(out_dir)))


if __name__ == "__main__":
    main()
