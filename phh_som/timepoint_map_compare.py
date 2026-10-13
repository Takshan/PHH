#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

from pipeline import (DATA, FDR_CUT, FC_CUT, MM, PUBLISH, TPM_CUT,
                      VIRUS_ORDER, save_figure, set_rcparams)
from plot_som_map import lighten
from plot_response_terrain import virus_colors

ACTIVE_Z = 1.0

ALLTIME = PUBLISH / "expressed_28x27"
EPOCH_MAPS = {"24h": ("expressed_24h_28x27", "expressed_24h", "24"),
              "72h": ("expressed_72h_28x27", "expressed_72h", "72")}
OUT_DATA = ALLTIME / "data"
OUT_FIG = ALLTIME / "figures"

MODULES = {"IFN-ligand (N1)": [1],
           "IRF3-direct effector (N28)": [28],
           "antigen-presentation (N84)": [84],
           "DENV acute-phase (N43+42+44)": [43, 42, 44]}

FIG_W, FIG_H = 120, 62


def load_universe(data_dir, tag):
    cb = pd.read_csv(data_dir / f"som_codebook_{tag}.tsv", sep="\t",
                     index_col=0)
    assign = pd.read_csv(data_dir / f"som_neuron_assignments_{tag}.tsv",
                         sep="\t")
    return cb, assign


def load_maps():
    cb_all, assign_all = load_universe(ALLTIME / "data", "expressed")
    epochs = {}
    for name, (dirname, tag, hpi) in EPOCH_MAPS.items():
        cb, assign = load_universe(PUBLISH / dirname / "data", tag)
        epochs[name] = {"cb": cb, "assign": assign, "hpi": hpi}
    return cb_all, assign_all, epochs


def breadth_table(cb_all, epochs, degs):
    rows = []
    for v in VIRUS_ORDER:
        row = {"virus": v,
               "n_induced_alltime_ut24": int((cb_all[f"{v}_untreated24"]
                                              > ACTIVE_Z).sum()),
               "n_induced_alltime_ut72": int((cb_all[f"{v}_untreated72"]
                                              > ACTIVE_Z).sum())}
        for name, m in epochs.items():
            col = f"{v}_untreated{m['hpi']}"
            row[f"n_induced_{name}map"] = int((m["cb"][col] > ACTIVE_Z).sum())
            row[f"amp_{name}map"] = float(m["cb"][col].clip(lower=0).sum())
            row[f"n_deg_untreated{m['hpi']}"] = int(
                degs.loc[v, int(m["hpi"])])
        rows.append(row)
    return pd.DataFrame(rows)


def module_tracking(assign_all, epochs):
    rows = []
    for module, neurons in MODULES.items():
        members = set(assign_all.loc[assign_all.neuron.isin(neurons),
                                     "gene_id"])
        for name, m in epochs.items():
            landed = m["assign"][m["assign"].gene_id.isin(members)]
            counts = landed.neuron.value_counts()
            best = int(counts.idxmax())
            best_genes = set(m["assign"].loc[m["assign"].neuron == best,
                                             "gene_id"])
            rows.append({
                "module": module, "n_members": len(members), "epoch": name,
                "best_neuron": best, "n_in_best": int(counts.max()),
                "frac_in_best": counts.max() / len(members),
                "jaccard_best": (len(members & best_genes)
                                 / len(members | best_genes)),
                "n_neurons_occupied": int(counts.size)})
    return pd.DataFrame(rows)


def jak_by_hpi(de):
    de = de.copy()
    de["contrast_short"] = de["virus"] + "_" + de["treatment"] + de["hpi"].astype(str)
    de["deg"] = ((de["fdr"] < FDR_CUT) & (de["fc"].abs() > FC_CUT)
                 & (de["maxgroupmeans"] > TPM_CUT))
    lfc = de.pivot(index="gene_id", columns="contrast_short", values="log2fc")
    deg = de.pivot(index="gene_id", columns="contrast_short",
                   values="deg").fillna(False)
    out = {}
    for hpi in ("24", "72"):
        rows = []
        for v in VIRUS_ORDER:
            ut = lfc[f"{v}_untreated{hpi}"].abs()
            rx = lfc[f"{v}_Ruxo{hpi}"].abs()
            idx = (ut - rx) / (ut + rx).replace(0, np.nan)
            induced = deg[f"{v}_untreated{hpi}"]
            rows.append(pd.DataFrame({"virus": v, "hpi": hpi,
                                      "gene_id": lfc.index[induced],
                                      "jak_index": idx[induced].to_numpy()}))
        out[hpi] = pd.concat(rows, ignore_index=True).dropna()
    return out


def jak_tables(jak, epochs):
    neuron_rows, summary_rows = [], []
    for name, m in epochs.items():
        cb, assign, hpi = m["cb"], m["assign"], m["hpi"]
        j = jak[hpi]
        for v in VIRUS_ORDER:
            scored = j[j.virus == v].set_index("gene_id").jak_index
            vals = scored.reindex(assign.gene_id).groupby(
                assign.neuron.to_numpy()).mean()
            induced = cb.index[cb[f"{v}_untreated{hpi}"] > ACTIVE_Z]
            vals = vals.loc[vals.index.intersection(induced)].dropna()
            for neuron, value in vals.items():
                neuron_rows.append({"virus": v, "epoch": name,
                                    "neuron": int(neuron),
                                    "neuron_mean_jak": value})
            summary_rows.append({"virus": v, "epoch": name,
                                 "n_induced_neurons": len(induced),
                                 "n_scored_neurons": int(vals.size),
                                 "mean_neuron_jak": vals.mean(),
                                 "median_neuron_jak": vals.median()})
    neurons = pd.DataFrame(neuron_rows)
    wide = (pd.DataFrame(summary_rows)
            .pivot(index="virus",
                   columns="epoch",
                   values=["n_induced_neurons", "n_scored_neurons",
                           "mean_neuron_jak", "median_neuron_jak"]))
    wide.columns = [f"{stat}_{epoch}" for stat, epoch in wide.columns]
    wide = wide.reindex(VIRUS_ORDER).reset_index()
    wide["delta_mean_jak_24h_minus_72h"] = (wide["mean_neuron_jak_24h"]
                                            - wide["mean_neuron_jak_72h"])
    return neurons, wide


def jak_figure(neurons):
    colors = virus_colors()
    paper = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]
    set_rcparams()
    fig, ax = plt.subplots(figsize=(FIG_W * MM, FIG_H * MM))
    width = 0.32
    for i, v in enumerate(paper):
        for k, (name, fc) in enumerate([("24h", colors[v]),
                                        ("72h", lighten(colors[v], 0.55))]):
            vals = neurons[(neurons.virus == v)
                           & (neurons.epoch == name)].neuron_mean_jak
            pos = i + (k - 0.5) * width * 1.15
            ax.boxplot(vals, positions=[pos], widths=width,
                       showfliers=False, patch_artist=True,
                       boxprops=dict(fc=fc, ec="#2F3338", lw=0.6),
                       medianprops=dict(color="#2F3338", lw=0.8),
                       whiskerprops=dict(color="#2F3338", lw=0.6),
                       capprops=dict(color="#2F3338", lw=0.6))
            ax.scatter(np.full(len(vals), pos), vals, s=2.5, fc=fc,
                       ec="none", alpha=0.35, zorder=3,
                       rasterized=len(vals) > 200)
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_xticks(range(len(paper)))
    ax.set_xticklabels(paper)
    for i, v in enumerate(paper):
        ax.get_xticklabels()[i].set_color(colors[v])
        ax.get_xticklabels()[i].set_fontweight("bold")
    ax.set_ylabel("neuron-mean JAK-STAT dependence index")
    handles = [mpl.patches.Patch(fc="0.25", label="J_24, induced neurons of the 24h map"),
               mpl.patches.Patch(fc="0.65", label="J_72, induced neurons of the 72h map")]
    ax.legend(handles=handles, fontsize=6, frameon=False, ncols=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    written = save_figure(fig, OUT_FIG / "figs_s24_c_jak_timepoint.pdf")
    plt.close(fig)
    print("wrote " + ", ".join(str(p) for p in written))


def main():
    cb_all, assign_all, epochs = load_maps()

    de = pd.read_parquet(DATA / "de_long.parquet")
    deg_counts = de[de.treatment == "untreated"].assign(
        deg=lambda d: (d["fdr"] < FDR_CUT) & (d["fc"].abs() > FC_CUT)
                      & (d["maxgroupmeans"] > TPM_CUT))
    deg_counts = deg_counts.groupby(["virus", "hpi"])["deg"].sum()

    breadth = breadth_table(cb_all, epochs, deg_counts)
    breadth.to_csv(OUT_DATA / "s24_a_response_breadth.csv", index=False)
    print(breadth.round(3).to_string(index=False))

    tracking = module_tracking(assign_all, epochs)
    tracking.to_csv(OUT_DATA / "s24_b_module_tracking.csv", index=False)
    print(tracking.round(3).to_string(index=False))

    neurons, wide = jak_tables(jak_by_hpi(de), epochs)
    neurons.to_csv(OUT_DATA / "s24_c_jak_neuron_values.csv", index=False)
    wide.to_csv(OUT_DATA / "s24_c_jak_by_timepoint.csv", index=False)
    print(wide.round(3).to_string(index=False))
    jak_figure(neurons)


if __name__ == "__main__":
    main()
