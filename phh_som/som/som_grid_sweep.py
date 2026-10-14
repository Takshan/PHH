#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from phh_som.utils.pipeline import (DATA, GRID_X, GRID_Y, MARGINAL_MIN, RLEN, SEED, SWEEP_R,
                      out_suffix, tag_from_matrix)

ELBOW_CUTOFFS = (1.0, 0.5, 0.25)


def run_sweep(z, rlen=RLEN, seed=SEED, grids=None):
    with tempfile.TemporaryDirectory() as tmp:
        mat = Path(tmp) / "matrix.csv"
        z.to_csv(mat)
        cmd = ["Rscript", str(SWEEP_R), str(mat), str(rlen), str(seed),
               str(Path(tmp) / "som")]
        if grids is not None:
            cmd.append(str(grids))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout.strip())
        sweep = pd.read_csv(Path(tmp) / "som_sweep.csv")
    return sweep.sort_values("n_neurons").reset_index(drop=True)


def add_marginals(sweep):
    sweep = sweep.copy()
    qe = sweep["qe"].to_numpy()
    k = sweep["n_neurons"].to_numpy()
    growth = 100 * (qe[:-1] - qe[1:]) / qe[:-1]
    marginal = growth / np.diff(k)
    sweep["growth_pct"] = np.nan
    sweep.loc[1:, "growth_pct"] = growth
    sweep["marginal_pct_per_neuron"] = np.nan
    sweep.loc[1:, "marginal_pct_per_neuron"] = marginal
    return sweep


def elbow_for(sweep, cutoff):
    ok = sweep.dropna(subset=["marginal_pct_per_neuron"]).reset_index(drop=True)
    ks = ok["n_neurons"].to_numpy()
    below = (ok["marginal_pct_per_neuron"] < cutoff).to_numpy()
    pairs = np.where(below[:-1] & below[1:])[0]
    return int(ks[max(pairs[0] - 1, 0)]) if len(pairs) else int(ks[-1])


def select_grid(sweep, cutoff=MARGINAL_MIN):
    below = sweep["marginal_pct_per_neuron"] < cutoff
    if not below.any():
        return None
    row = sweep.loc[below].iloc[0]
    return int(row["x_dim"]), int(row["y_dim"])


def main():
    ap = argparse.ArgumentParser(description='kohonen grid-size sweep + grid selection.')
    ap.add_argument("--matrix", type=Path, default=DATA / "isg_matrix_z.parquet")
    ap.add_argument("--out-dir", type=Path, default=DATA)
    ap.add_argument("--grid-x", type=int, default=GRID_X)
    ap.add_argument("--grid-y", type=int, default=GRID_Y)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rlen", type=int, default=RLEN)
    ap.add_argument("--grids", type=Path, default=None,
                    help="CSV with x_dim,y_dim columns overriding the R "
                         "driver's default 19 grids (3x3 -> 19x18 = 342 "
                         "neurons); needed when the production grid is larger")
    ap.add_argument("--tag", default=None,
                    help="gene-set tag for output names; default derived from "
                         "--matrix (isg keeps the original unsuffixed names)")
    args = ap.parse_args()
    tag = args.tag or tag_from_matrix(args.matrix)
    suff = out_suffix(tag)

    z = pd.read_parquet(args.matrix)
    sweep = add_marginals(run_sweep(z, rlen=args.rlen, seed=args.seed,
                                    grids=args.grids))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.out_dir / f"som_sweep{suff}.csv", index=False)

    elbows = {f"{c}%/neuron": elbow_for(sweep, c) for c in ELBOW_CUTOFFS}
    grid_x, grid_y = args.grid_x, args.grid_y
    rule_grid = select_grid(sweep)
    n_prod = grid_x * grid_y
    row_prod = sweep.loc[sweep["n_neurons"] == n_prod]
    if row_prod.empty:
        print(f"WARNING: production grid {grid_x}x{grid_y} not in the sweep "
              "grids - qe_at_production_grid recorded as null")
        qe_prod = None
    else:
        qe_prod = float(row_prod.iloc[0]["qe"])

    choice = {
        "production_grid": {"x_dim": grid_x, "y_dim": grid_y,
                            "n_neurons": n_prod},
    }
    (args.out_dir / f"som_grid_choice{suff}.json").write_text(json.dumps(choice, indent=2))

    print(f"\nElbow vs cutoff:")
    for label, k in elbows.items():
        marker = "  <-- marginal cutoff" if label == f"{MARGINAL_MIN}%/neuron" else ""
        print(f"  {label} -> {k} neurons{marker}")
    qe_txt = "not swept" if qe_prod is None else f"QE={qe_prod:.5f}"
    print(f"\nProduction grid (manual): {grid_x}x{grid_y} = {n_prod} "
          f"neurons ({z.shape[0] / n_prod:.1f} genes/neuron), {qe_txt}")
    print(f"  diagnostic, first grid < {MARGINAL_MIN}%/neuron: "
          f"{'none' if rule_grid is None else rule_grid[0] * rule_grid[1]} "
          f"neurons (seed-dependent, not used)")


if __name__ == "__main__":
    main()
