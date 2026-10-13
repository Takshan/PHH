#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

import pandas as pd

PUBLISH = Path(__file__).resolve().parent
sys.path.insert(0, str(PUBLISH))

from pipeline import GRID_X, GRID_Y, RLEN, SEED, out_suffix

PY = PUBLISH.parent / ".venv" / "bin" / "python"

UNIVERSES = {
    "retrain_ISGs_allviruses": {"gene_set": "isg_all", "grid": (GRID_X, GRID_Y),
                                "top50_dir": PUBLISH / "data", "hybrid": False},
}


def run(script, *args):
    cmd = [str(PY), str(PUBLISH / script), *(str(a) for a in args)]
    print(f"\n+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=PUBLISH)


def skip_if_done(outputs, label):
    if all(Path(o).exists() for o in outputs):
        print(f"[skip] {label} (outputs exist)")
        return True
    return False


def sweep_and_grid(gene_set, data_dir, fixed_grid):
    suff = out_suffix(gene_set)
    if not skip_if_done([data_dir / f"som_sweep{suff}.csv",
                         data_dir / f"som_grid_choice{suff}.json"],
                        f"som_grid_sweep sweep ({gene_set})"):
        run("som_grid_sweep.py", "--matrix",
            data_dir / f"{gene_set}_matrix_z.parquet", "--out-dir", data_dir,
            "--tag", gene_set, "--grid-x", fixed_grid[0],
            "--grid-y", fixed_grid[1], "--seed", SEED, "--rlen", RLEN)
    return fixed_grid


def run_universe(name, spec):
    gene_set = spec["gene_set"]
    suff = out_suffix(gene_set)
    universe_dir = PUBLISH / name
    data_dir = universe_dir / "data"
    figures_dir = universe_dir / "figures"
    matrix = data_dir / f"{gene_set}_matrix_z.parquet"
    print(f"\n=== {name}: gene set '{gene_set}' ===", flush=True)

    if not skip_if_done([matrix], f"build_feature_matrix matrix ({gene_set})"):
        run("build_feature_matrix.py", "--gene-set", gene_set,
            "--out-dir", data_dir)

    grid_x, grid_y = sweep_and_grid(gene_set, data_dir, spec["grid"])

    if not skip_if_done([data_dir / f"som_summary{suff}.json"],
                        f"train_som train ({gene_set})"):
        run("train_som.py", "--matrix", matrix, "--out-dir", data_dir,
            "--tag", gene_set, "--grid-x", grid_x, "--grid-y", grid_y,
            "--seed", SEED, "--rlen", RLEN)

    fig_outs = [figures_dir / f"figs_som_{gene_set}_bestsize_kohonen.pdf",
                figures_dir / f"figs_som_{gene_set}_kohonen_changes.pdf"]
    if not skip_if_done(fig_outs, f"plot_training_diagnostics figures ({gene_set})"):
        run("plot_training_diagnostics.py", "--matrix", matrix, "--tag", gene_set,
            "--data-dir", data_dir, "--out-dir", figures_dir)

    assignments = data_dir / f"som_neuron_assignments{suff}.tsv"
    stats = data_dir / f"som_neuron_stats{suff}.csv"
    stats_stale = stats.exists() and "mean_jak_index_zikv" not in \
        pd.read_csv(stats, nrows=0).columns
    if stats_stale or not skip_if_done([stats], f"neuron_stats neuron stats ({gene_set})"):
        run("neuron_stats.py", "--assignments", assignments,
            "--out-dir", data_dir)

    membership = data_dir / f"som_neuron_membership{suff}.csv"
    if not skip_if_done([membership], f"neuron_membership_table membership ({gene_set})"):
        cmd = ["neuron_membership_table.py", "--assignments", assignments,
               "--stats", stats, "--core", data_dir / "core_class.csv",
               "--out-dir", data_dir]
        if spec["top50_dir"] is not None:
            cmd += ["--top50-dir", spec["top50_dir"]]
        run(*cmd)

    if spec["hybrid"]:
        hybrid = figures_dir / f"figs_som_{gene_set}_hybrid.pdf"
        if not skip_if_done([hybrid], f"plot_som_map hybrid ({gene_set})"):
            run("plot_som_map.py", "--tag", gene_set,
                "--data-dir", data_dir, "--out-dir", figures_dir)

    terrain = figures_dir / f"figs_som_{gene_set}_response_terrain.pdf"
    if not skip_if_done([terrain], f"plot_response_terrain response terrain ({gene_set})"):
        run("plot_response_terrain.py", "--tag", gene_set,
            "--data-dir", data_dir, "--out-dir", figures_dir)

    dep_terrain = figures_dir / f"figs_som_{gene_set}_dependence_terrain.pdf"
    if not skip_if_done([dep_terrain], f"plot_dependence_terrain dependence terrain ({gene_set})"):
        run("plot_dependence_terrain.py", "--tag", gene_set,
            "--data-dir", data_dir, "--out-dir", figures_dir)

    codes = figures_dir / f"figs_som_{gene_set}_codes_map.pdf"
    if not skip_if_done([codes], f"plot_codes_map codes map ({gene_set})"):
        run("plot_codes_map.py", "--tag", gene_set,
            "--data-dir", data_dir, "--out-dir", figures_dir)


def main():
    for name, spec in UNIVERSES.items():
        run_universe(name, spec)
    print("\nDone. Outputs in:")
    for name in UNIVERSES:
        print(f"  {PUBLISH / name}/{{data,figures}}")


if __name__ == "__main__":
    main()
