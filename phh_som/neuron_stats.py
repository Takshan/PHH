#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ALL_CONTRASTS, DATA, VIRUS_ORDER

CORE_MIN_VIRUSES = 4


def jak_dependence(de):
    lfc = de.pivot(index="gene_id", columns="contrast_short", values="log2fc")
    deg = de.pivot(index="gene_id", columns="contrast_short",
                   values="deg").fillna(False)
    lfc = lfc[ALL_CONTRASTS].fillna(0.0)
    rows = []
    for v in VIRUS_ORDER:
        ut = lfc[[f"{v}_untreated24", f"{v}_untreated72"]].abs().mean(axis=1)
        rx = lfc[[f"{v}_Ruxo24", f"{v}_Ruxo72"]].abs().mean(axis=1)
        idx = (ut - rx) / (ut + rx).replace(0, np.nan)
        induced = deg[[f"{v}_untreated24", f"{v}_untreated72"]].any(axis=1)
        rows.append(pd.DataFrame({"virus": v, "gene_id": lfc.index[induced],
                                  "jak_index": idx[induced].to_numpy()}))
    return pd.concat(rows, ignore_index=True).dropna()


def core_class(de):
    ut = de[de["treatment"] == "untreated"]
    induced = ut.groupby(["gene_id", "virus"])["deg"].any().reset_index()
    induced = induced[induced["deg"]]
    n = induced.groupby("gene_id")["virus"].nunique().rename("n_viruses")
    cls = np.where(n >= CORE_MIN_VIRUSES, "core",
                   np.where(n == 1, "specific", "intermediate"))
    return pd.DataFrame({"gene_id": n.index, "n_viruses": n.to_numpy(),
                         "class": cls})


def neuron_stats(assign, jak, core_ids, n_neurons):
    wide = jak.pivot_table(index="gene_id", columns="virus", values="jak_index")
    rows = []
    for neuron in range(1, n_neurons + 1):
        genes = assign.loc[assign["neuron"] == neuron, "gene_id"]
        row = {"neuron": neuron, "n_genes": len(genes),
               "frac_core": (float(genes.isin(core_ids).mean())
                             if len(genes) else np.nan)}
        for v in VIRUS_ORDER:
            vals = (wide[v].reindex(genes).dropna() if v in wide else
                    pd.Series(dtype=float))
            row[f"mean_jak_index_{v.lower()}"] = (float(vals.mean())
                                                  if len(vals) else np.nan)
            row[f"n_jak_{v.lower()}"] = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='per-gene JAK-STAT dependence and per-neuron summary statistics.')
    ap.add_argument("--degs", type=Path, default=DATA / "degs.parquet")
    ap.add_argument("--assignments", type=Path,
                    default=DATA / "som_neuron_assignments.tsv")
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()

    de = pd.read_parquet(args.degs)
    jak = jak_dependence(de)
    core = core_class(de)
    jak.to_csv(args.out_dir / "jak_dependence.csv", index=False)
    core.to_csv(args.out_dir / "core_class.csv", index=False)
    print(f"jak_dependence: {len(jak):,} gene x virus indices "
          f"({jak['virus'].nunique()} viruses)")
    print(f"core_class    : {(core['class'] == 'core').sum():,} core of "
          f"{len(core):,} genes")

    assign = pd.read_csv(args.assignments, sep="\t")
    core_ids = set(core.loc[core["class"] == "core", "gene_id"])
    stats = neuron_stats(assign, jak, core_ids, int(assign["neuron"].max()))
    suff = args.assignments.stem.replace("som_neuron_assignments", "")
    out = args.out_dir / f"som_neuron_stats{suff}.csv"
    stats.to_csv(out, index=False)
    print(f"som_neuron_stats{suff}: {len(stats)} neurons, "
          f"{int(stats['n_genes'].sum()):,} genes -> {out.name}")


if __name__ == "__main__":
    main()
