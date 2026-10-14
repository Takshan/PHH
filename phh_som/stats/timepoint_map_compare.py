#!/usr/bin/env python3
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from phh_som.utils.deg import is_deg
from phh_som.utils.jak import jak_by_hpi
from phh_som.utils.pipeline import DATA, RAW_DIR, VIRUS_ORDER, universe
from phh_som.utils.somdata import load_universe as _load_universe

ACTIVE_Z = 1.0

ALLTIME = universe("expressed_28x27")
EPOCH_MAPS = {"24h": ("expressed_24h_28x27", "expressed_24h", "24"),
              "72h": ("expressed_72h_28x27", "expressed_72h", "72")}
OUT_DATA = ALLTIME / "data"

MODULES = {"IFN-ligand (N1)": [1],
           "IRF3-direct effector (N28)": [28],
           "antigen-presentation (N84)": [84],
           "DENV acute-phase (N43+42+44)": [43, 42, 44]}



def load_universe(data_dir, tag):
    cb, assign, _ = _load_universe(data_dir, tag)
    return cb, assign


def load_maps():
    cb_all, assign_all = load_universe(ALLTIME / "data", "expressed")
    epochs = {}
    for name, (dirname, tag, hpi) in EPOCH_MAPS.items():
        cb, assign = load_universe(universe(dirname) / "data", tag)
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


def main():
    cb_all, assign_all, epochs = load_maps()

    de = pd.read_parquet(RAW_DIR / "de_long.parquet")
    deg_counts = de[de.treatment == "untreated"].assign(deg=is_deg)
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


if __name__ == "__main__":
    main()
