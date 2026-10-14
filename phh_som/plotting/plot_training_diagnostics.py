#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from phh_som.utils.pipeline import (DATA, FIGURES, MM, MARGINAL_MIN, out_suffix, save_figure,
                      set_rcparams, tag_from_matrix)

GENE_SET_LABELS = {"isg": "Schoggins ISGs", "wnvyfv": "WNV/YFV DEGs",
                   "degs": "DEGs", "isg_all": "Schoggins ISGs (all viruses)",
                   "all": "genes (whole transcriptome)"}


def gene_set_label(tag):
    return GENE_SET_LABELS.get(tag, tag)


def plot_sweep_figure(sweep, choice, n_genes, label, out_pdf):
    k_prod = choice["production_grid"]["n_neurons"]
    below = sweep["marginal_pct_per_neuron"] < MARGINAL_MIN
    k_first = int(sweep.loc[below, "n_neurons"].iloc[0]) if below.any() else None

    fig, (ax_qe, ax_marg) = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM))

    ax_qe.plot(sweep["n_neurons"], sweep["qe"], "o-", color="black",
               ms=3, lw=0.8)
    qe_row = sweep.loc[sweep["n_neurons"] == k_prod, "qe"]
    if qe_row.empty:
        ax_qe.axvline(k_prod, color="0.4", ls="--", lw=0.7)
        ax_qe.text(k_prod, 1.0, f" production {k_prod} (not swept) ",
                   rotation=90, transform=ax_qe.get_xaxis_transform(),
                   va="top", ha="left", fontsize=6, color="0.35")
    else:
        qe_prod = float(qe_row.iloc[0])
        ax_qe.annotate(f"QE {qe_prod:.5f}", xy=(k_prod, qe_prod),
                       xytext=(14, 10), textcoords="offset points", fontsize=6,
                       color="0.35",
                       arrowprops=dict(arrowstyle="-", lw=0.6, color="0.6"))
    ax_qe.set_xlabel("neurons in grid")
    ax_qe.set_ylabel("quantization error")
    ax_qe.set_title("a  converged QE vs grid size (R kohonen)", loc="left")

    ax_marg.plot(sweep["n_neurons"], sweep["marginal_pct_per_neuron"], "s-",
                 color="black", ms=3, lw=0.8)
    ax_marg.axhline(MARGINAL_MIN, color="0.4", ls=":", lw=0.8)
    ax_marg.text(sweep["n_neurons"].max() * 0.55, MARGINAL_MIN + 0.15,
                 f"{MARGINAL_MIN}%/neuron", color="0.35", fontsize=6)
    ax_marg.set_xlabel("neurons in grid")
    ax_marg.set_ylabel("marginal QE gain (% per added neuron)")
    ax_marg.set_title(f"b  marginal QE gain per added neuron\n"
                      f"(dotted line: {MARGINAL_MIN}%/neuron reference)",
                      loc="left", fontsize=6.5)

    markers = []
    if k_first is not None:
        markers.append((k_first, f"{k_first} (first < {MARGINAL_MIN}%)",
                        "#d95f02", "bottom", "left"))
    for ax in (ax_qe, ax_marg):
        for k, label_txt, color, va, ha in markers:
            ax.axvline(k, color=color, ls="--", lw=0.7)
            ax.text(k, 1.0 if va == "top" else 0.0, f" {label_txt} ", rotation=90,
                    transform=ax.get_xaxis_transform(),
                    va=va, ha=ha, fontsize=6, color=color)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"SOM grid-size selection (R kohonen) - {n_genes} {label}",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    written = save_figure(fig, out_pdf)
    plt.close(fig)
    print("wrote " + ", ".join(str(p) for p in written))


def plot_changes_figure(changes, summary, out_pdf):
    plt.figure(figsize=(70 * MM, 45 * MM))
    plt.plot(changes["iter"], changes["qe"], color="black", lw=0.8)
    qe = summary["converged_qe"]
    plt.axhline(qe, color="0.55", ls="--", lw=0.7)
    ax = plt.gca()
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.5)
    ax.tick_params(axis="both", width=1.5)
    ax.text(changes["iter"].min(), qe, f" converged {qe:.5f} ",
            ha="left", va="bottom", fontsize=5.5, color="0.4")
    plt.xlabel("Training iteration")
    plt.ylabel("Quantization error")
    plt.title(f"training curve (R kohonen, {summary['grid']} = "
              f"{summary['n_neurons']} neurons,\nrlen {summary['rlen']}, "
              f"seed {summary['seed']})", fontsize=6)
    plt.tight_layout()
    written = save_figure(plt.gcf(), out_pdf)
    plt.close()
    print("wrote " + ", ".join(str(p) for p in written))


def main():
    ap = argparse.ArgumentParser(description='publication figures for the SOM "bestsize" analysis.')
    ap.add_argument("--matrix", type=Path, default=DATA / "isg_matrix_z.parquet")
    ap.add_argument("--tag", default=None,
                    help="gene-set tag; default derived from --matrix")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=FIGURES / "plot_training_diagnostics")
    args = ap.parse_args()
    tag = args.tag or tag_from_matrix(args.matrix)
    suff = out_suffix(tag)

    set_rcparams()
    z = pd.read_parquet(args.matrix)
    changes = pd.read_csv(args.data_dir / f"som_changes{suff}.csv")
    summary = json.loads((args.data_dir / f"som_summary{suff}.json").read_text())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = args.data_dir / f"som_sweep{suff}.csv"
    if sweep_csv.exists():
        sweep = pd.read_csv(sweep_csv)
        choice = json.loads(
            (args.data_dir / f"som_grid_choice{suff}.json").read_text())
        plot_sweep_figure(sweep, choice, z.shape[0], gene_set_label(tag),
                          args.out_dir / f"figs_som_{tag}_bestsize_kohonen.pdf")
    else:
        print(f"[skip] sweep figure (no {sweep_csv.name})")
    plot_changes_figure(changes, summary,
                        args.out_dir / f"figs_som_{tag}_kohonen_changes.pdf")


if __name__ == "__main__":
    main()
