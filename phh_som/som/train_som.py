#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from phh_som.utils.deg import symbol_table
from phh_som.utils.pipeline import (DATA, GRID_X, GRID_Y, RLEN, SEED, TRAIN_R,
                      out_suffix, tag_from_matrix)

QE_RE = re.compile(r"converged QE:\s*([0-9.]+)")


def run_som_isg(z, x_dim=GRID_X, y_dim=GRID_Y, seed=SEED, rlen=RLEN):
    with tempfile.TemporaryDirectory() as tmp:
        mat = Path(tmp) / "matrix.csv"
        z.to_csv(mat)
        result = subprocess.run(
            ["Rscript", str(TRAIN_R), str(mat), str(x_dim), str(y_dim),
             str(rlen), str(seed), str(Path(tmp) / "som")],
            check=True, capture_output=True, text=True)
        print(result.stdout.strip())
        unit = pd.read_csv(Path(tmp) / "som_assign.csv", index_col=0)["unit"]
        codes = pd.read_csv(Path(tmp) / "som_codebook.csv", index_col=0)
        changes = pd.read_csv(Path(tmp) / "som_changes.csv")
    qe = float(QE_RE.search(result.stdout).group(1))
    unit = unit.reindex(z.index).to_numpy(dtype=int)
    winners = np.column_stack([(y_dim - 1) - (unit - 1) // x_dim,
                               (unit - 1) % x_dim])
    neuron = pd.Series(unit, index=z.index, name="neuron")
    flat = ((y_dim - 1) - (codes.index.to_numpy() - 1) // x_dim) * x_dim \
        + (codes.index.to_numpy() - 1) % x_dim
    codebook_flat = np.zeros((y_dim * x_dim, z.shape[1]))
    codebook_flat[flat] = codes[z.columns].to_numpy()
    return winners, neuron, codebook_flat, changes, qe


def neuron_gene_table(neuron, symbols, winners):
    tab = neuron.to_frame()
    tab["symbol"] = tab.index.map(symbols)
    tab["som_row"] = winners[:, 0]
    tab["som_col"] = winners[:, 1]
    return tab.reset_index().rename(columns={"index": "gene_id"})


def main():
    ap = argparse.ArgumentParser(description='train the production ISG SOM with the kohonen R package.')
    ap.add_argument("--matrix", type=Path, default=DATA / "isg_matrix_z.parquet")
    ap.add_argument("--degs", type=Path, default=DATA / "degs.parquet")
    ap.add_argument("--out-dir", type=Path, default=DATA)
    ap.add_argument("--grid-x", type=int, default=GRID_X)
    ap.add_argument("--grid-y", type=int, default=GRID_Y)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rlen", type=int, default=RLEN)
    ap.add_argument("--tag", default=None,
                    help="gene-set tag for output names; default derived from "
                         "--matrix (isg keeps the original unsuffixed names)")
    args = ap.parse_args()
    tag = args.tag or tag_from_matrix(args.matrix)
    suff = out_suffix(tag)
    grid_x, grid_y = args.grid_x, args.grid_y

    z = pd.read_parquet(args.matrix)
    de = pd.read_parquet(args.degs)
    print(f"[som-{tag}] production grid {grid_x}x{grid_y} = "
          f"{grid_x * grid_y} neurons, seed {args.seed}, "
          f"rlen {args.rlen}", flush=True)

    winners, neuron, codebook_flat, changes, qe = run_som_isg(
        z, x_dim=grid_x, y_dim=grid_y, seed=args.seed, rlen=args.rlen)

    symbols = symbol_table(de)
    assign = neuron_gene_table(neuron, symbols, winners)

    codebook = pd.DataFrame(
        codebook_flat,
        index=[(grid_y - 1 - i // grid_x) * grid_x + i % grid_x + 1
               for i in range(grid_y * grid_x)],
        columns=z.columns)
    codebook.index.name = "neuron"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    assign.to_csv(args.out_dir / f"som_neuron_assignments{suff}.tsv", sep="\t", index=False)
    codebook.to_csv(args.out_dir / f"som_codebook{suff}.tsv", sep="\t")
    changes.to_csv(args.out_dir / f"som_changes{suff}.csv", index=False)

    summary = {
        "grid": f"{grid_x}x{grid_y}",
        "n_neurons": grid_x * grid_y,
        "seed": args.seed,
        "rlen": args.rlen,
        "converged_qe": qe,
    }
    (args.out_dir / f"som_summary{suff}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
