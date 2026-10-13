#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from pipeline import DATA, VIRUSES
from plot_som_map import (CALLOUT_PER_BOX, MIN_FOR_SURFACE, callouts,
                                 dep_class)

CLASS_NAME = {0: "ruxolitinib-resistant", 1: "intermediate",
              2: "JAK-STAT-dependent"}
MIN_SCORED_CLASS = MIN_FOR_SURFACE[0]


def membership_table(assign, stats, core_ids, named=None):
    named = callouts(assign, stats) if named is None else named
    rows = []
    for neuron in sorted(stats.index):
        genes = assign.loc[assign["neuron"] == neuron, ["gene_id", "symbol"]]
        symbols = sorted(genes["symbol"].dropna())
        row = {
            "neuron": neuron,
            "n_isgs": int(stats.loc[neuron, "n_genes"]),
            "n_core": int(genes["gene_id"].isin(core_ids).sum()),
            "frac_core": stats.loc[neuron, "frac_core"],
        }
        for v in VIRUSES:
            k = v.lower()
            idx, n = (stats.loc[neuron, f"mean_jak_index_{k}"],
                      int(stats.loc[neuron, f"n_jak_{k}"]))
            classed = np.isfinite(idx) and n >= MIN_SCORED_CLASS
            row[f"mean_jak_index_{k}"] = idx
            row[f"n_scored_{k}"] = n
            row[f"class_{k}"] = (CLASS_NAME[dep_class(idx)] if classed
                                 else "not classed")
        row["named_in_fig5e"] = "; ".join(named.get(neuron, [])[:CALLOUT_PER_BOX])
        row["members"] = "; ".join(symbols)
        rows.append(row)
    return pd.DataFrame(rows)


def dependent_neuron_enrichment(assign, stats, top50):
    dependent = [n for n in stats.index
                 if any(stats.loc[n, f"mean_jak_index_{v.lower()}"] >= 0.5
                        and stats.loc[n, f"n_jak_{v.lower()}"] >= MIN_SCORED_CLASS
                        for v in VIRUSES)]
    members = set(assign.loc[assign["neuron"].isin(dependent), "gene_id"])
    n_pop, n_dep = len(assign), len(members)
    rows = []
    for (virus, tail), genes in sorted(top50.items()):
        k, draw = len(members & set(genes)), len(genes)
        expected = draw * n_dep / n_pop
        rows.append({
            "virus": virus, "tail": tail, "n_drawn": draw,
            "n_in_dependent": k, "n_expected": round(expected, 1),
            "fold": round(k / expected, 2) if expected else np.nan,
            "pvalue": (hypergeom.sf(k - 1, n_pop, n_dep, draw) if tail == "suppressed"
                       else hypergeom.cdf(k, n_pop, n_dep, draw)),
            "dependent_neurons": " ".join(str(n) for n in dependent),
            "n_dependent": n_dep, "n_panel": n_pop,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='the full per-neuron ISG membership, as a supplementary table.')
    ap.add_argument("--assignments", type=Path,
                    default=DATA / "som_neuron_assignments.tsv")
    ap.add_argument("--stats", type=Path, default=None,
                    help="default: som_neuron_stats<suff>.csv beside --assignments")
    ap.add_argument("--core", type=Path, default=DATA / "core_class.csv")
    ap.add_argument("--top50-dir", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()

    suff = args.assignments.stem.replace("som_neuron_assignments", "")
    stats_path = args.stats or DATA / f"som_neuron_stats{suff}.csv"

    assign = pd.read_csv(args.assignments, sep="\t")
    stats = pd.read_csv(stats_path, index_col="neuron")
    core = pd.read_csv(args.core)
    core_ids = set(core.loc[core["class"] == "core", "gene_id"])

    tab = membership_table(assign, stats, core_ids)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"som_neuron_membership{suff}.csv"
    tab.to_csv(out, index=False)
    named = int((tab["named_in_fig5e"] != "").sum())
    print(f"som_neuron_membership{suff}: {len(tab)} neurons, "
          f"{int(tab['n_isgs'].sum()):,} members "
          f"({named} neurons named on the figure) -> {out.name}")

    tails = {(v, tail): args.top50_dir / f"top50_isg_{f}_{v.lower()}.csv"
             for v in VIRUSES
             for f, tail in (("up", "suppressed"), ("down", "resistant"))}
    if not all(p.exists() for p in tails.values()):
        print("dependent_neuron_enrichment: skipped, no UT-vs-JAKi top-50 tails")
        return
    top50 = {k: pd.read_csv(p)["gene_id"] for k, p in tails.items()}
    enr = dependent_neuron_enrichment(assign, stats, top50)
    out_e = args.out_dir / f"dependent_neuron_enrichment{suff}.csv"
    enr.to_csv(out_e, index=False)
    for r in enr.itertuples():
        print(f"dependent_neuron_enrichment{suff}: {r.virus} {r.tail:10s} tail "
              f"{r.n_in_dependent:2d}/{r.n_drawn} in the dependent neurons "
              f"(expected {r.n_expected}, P = {r.pvalue:.1g})")


if __name__ == "__main__":
    main()
