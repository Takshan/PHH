#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=../.venv/bin/python
SEED=143
RLEN=1000

write_choice() {
    "$PY" - "$1" "$2" "$3" "$4" "$5" <<'EOF'
import json
import sys
from pathlib import Path

import pandas as pd

data_dir, suff = Path(sys.argv[1]), sys.argv[2]
grid_x, grid_y, note = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
n_prod = grid_x * grid_y
qe_prod = None
sweep_csv = data_dir / f"som_sweep{suff}.csv"
if sweep_csv.exists():
    rows = pd.read_csv(sweep_csv).query("n_neurons == @n_prod")
    if not rows.empty:
        qe_prod = float(rows.iloc[0]["qe"])
choice = {
    "grid_source": note,
    "production_grid": {"x_dim": grid_x, "y_dim": grid_y, "n_neurons": n_prod},
    "qe_at_production_grid": qe_prod,
    "qe_definition": ("sqrt(mean(fit$distances) / (n_genes * n_conditions)); "
                      "kohonen's own error scale, as plotted"),
    "rlen": 1000,
    "seed": 143,
}
(data_dir / f"som_grid_choice{suff}.json").write_text(json.dumps(choice, indent=2))
EOF
}

run_universe() {
    local name=$1 gene_set=$2 grid_x=$3 grid_y=$4 note=$5
    shift 5
    local data=$name/data fig=$name/figures suff=_$gene_set
    local matrix=$data/${gene_set}_matrix_z.parquet
    local matrix_gene_set=$gene_set
    case $gene_set in
        *_24h|*_72h) matrix_gene_set=${gene_set%_*};;
    esac
    echo "=== $name: gene set '$gene_set', grid ${grid_x}x${grid_y} ==="
    mkdir -p "$data" "$fig"
    printf 'grid: %sx%s = %s neurons (fixed, seed %s, rlen %s)\ngene set: %s\ndriver: run_som_universes.sh\ndetails: data/som_grid_choice%s.json\n' \
        "$grid_x" "$grid_y" "$((grid_x * grid_y))" "$SEED" "$RLEN" \
        "$gene_set" "$suff" > "$name/grid.txt"

    local sweep=$data/som_sweep$suff.csv

    if [[ -f $matrix ]]; then
        echo "[skip] build_feature_matrix matrix ($gene_set)"
    else
        "$PY" build_feature_matrix.py --gene-set "$matrix_gene_set" \
            --out-dir "$data" "$@"
    fi

    local choice=$data/som_grid_choice$suff.json
    if [[ -f $choice ]]; then
        echo "[skip] grid choice ($gene_set)"
    else
        write_choice "$data" "$suff" "$grid_x" "$grid_y" "$note"
    fi

    local assignments=$data/som_neuron_assignments$suff.tsv
    local stats=$data/som_neuron_stats$suff.csv
    if [[ -f $data/som_summary$suff.json ]]; then
        echo "[skip] train_som train ($gene_set)"
    else
        "$PY" train_som.py --matrix "$matrix" --out-dir "$data" \
            --tag "$gene_set" --grid-x "$grid_x" --grid-y "$grid_y" \
            --seed "$SEED" --rlen "$RLEN"
    fi

    local changes_fig=$fig/figs_som_${gene_set}_kohonen_changes.pdf
    local sweep_fig=$fig/figs_som_${gene_set}_bestsize_kohonen.pdf
    if [[ -f $changes_fig && ( ! -f $sweep || -f $sweep_fig ) ]]; then
        echo "[skip] plot_training_diagnostics figures ($gene_set)"
    else
        "$PY" plot_training_diagnostics.py --matrix "$matrix" --tag "$gene_set" \
            --data-dir "$data" --out-dir "$fig"
    fi

    if [[ -f $stats ]]; then
        echo "[skip] neuron_stats neuron stats ($gene_set)"
    else
        "$PY" neuron_stats.py --assignments "$assignments" --out-dir "$data"
    fi

    if [[ -f $data/som_neuron_membership$suff.csv ]]; then
        echo "[skip] neuron_membership_table membership ($gene_set)"
    else
        "$PY" neuron_membership_table.py --assignments "$assignments" \
            --stats "$stats" --core "$data/core_class.csv" --out-dir "$data"
    fi

    if [[ -f $fig/figs_som_${gene_set}_response_terrain.pdf ]]; then
        echo "[skip] plot_response_terrain response terrain ($gene_set)"
    else
        "$PY" plot_response_terrain.py --tag "$gene_set" \
            --data-dir "$data" --out-dir "$fig"
    fi

    if [[ -f $fig/figs_som_${gene_set}_dependence_terrain.pdf ]]; then
        echo "[skip] plot_dependence_terrain dependence terrain ($gene_set)"
    else
        "$PY" plot_dependence_terrain.py --tag "$gene_set" \
            --data-dir "$data" --out-dir "$fig"
    fi

    if [[ -f $fig/figs_som_${gene_set}_codes_map.pdf ]]; then
        echo "[skip] plot_codes_map codes map ($gene_set)"
    else
        "$PY" plot_codes_map.py --tag "$gene_set" \
            --data-dir "$data" --out-dir "$fig"
    fi
}


run_universe expressed_28x27 expressed 28 27 \
    "Vesanto 5*sqrt(N) at 23,038 loci with max group mean >= 1 TPM in >= 1 condition (=759), fixed 28x27 = 756 neurons; no sweep run. build_feature_matrix blanks the ADAD2-AS1 division artefact for every unmasked gene set."

run_universe expressed_24h_28x27 expressed_24h 28 27 \
    "fixed 28x27 = 756 neurons (same grid as expressed_28x27, deliberately - comparability, not density, drives the choice); trained on the 10 untreated/Ruxo 24 hpi contrasts only (build_feature_matrix --hpi 24)" \
    --hpi 24

run_universe expressed_72h_28x27 expressed_72h 28 27 \
    "fixed 28x27 = 756 neurons (same grid as expressed_28x27, deliberately - comparability, not density, drives the choice); trained on the 10 untreated/Ruxo 72 hpi contrasts only (build_feature_matrix --hpi 72)" \
    --hpi 72
