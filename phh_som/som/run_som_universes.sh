#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python
SEED=143
RLEN=1000

write_choice() {
    "$PY" - "$1" "$2" "$3" "$4" <<'EOF'
import json
import sys
from pathlib import Path

data_dir, suff = Path(sys.argv[1]), sys.argv[2]
grid_x, grid_y = int(sys.argv[3]), int(sys.argv[4])
choice = {
    "production_grid": {"x_dim": grid_x, "y_dim": grid_y,
                        "n_neurons": grid_x * grid_y},
}
(data_dir / f"som_grid_choice{suff}.json").write_text(json.dumps(choice, indent=2))
EOF
}

run_universe() {
    local name=$1 gene_set=$2 grid_x=$3 grid_y=$4
    shift 4
    local data=${PHH_SOM_OUT:-results}/$name/data fig=${PHH_SOM_OUT:-results}/$name/figures suff=_$gene_set
    local matrix=$data/${gene_set}_matrix_z.parquet
    local matrix_gene_set=$gene_set
    case $gene_set in
        *_24h|*_72h) matrix_gene_set=${gene_set%_*};;
    esac
    echo "=== $name: gene set '$gene_set', grid ${grid_x}x${grid_y} ==="
    mkdir -p "$data" "$fig"

    local sweep=$data/som_sweep$suff.csv

    if [[ -f $matrix ]]; then
        echo "[skip] build_feature_matrix matrix ($gene_set)"
    else
        "$PY" -m phh_som.som.build_feature_matrix --gene-set "$matrix_gene_set" \
            --out-dir "$data" "$@"
    fi

    local choice=$data/som_grid_choice$suff.json
    if [[ -f $choice ]]; then
        echo "[skip] grid choice ($gene_set)"
    else
        write_choice "$data" "$suff" "$grid_x" "$grid_y"
    fi

    local assignments=$data/som_neuron_assignments$suff.tsv
    local stats=$data/som_neuron_stats$suff.csv
    if [[ -f $data/som_summary$suff.json ]]; then
        echo "[skip] train_som train ($gene_set)"
    else
        "$PY" -m phh_som.som.train_som --matrix "$matrix" --out-dir "$data" \
            --tag "$gene_set" --grid-x "$grid_x" --grid-y "$grid_y" \
            --seed "$SEED" --rlen "$RLEN"
    fi

    local changes_fig=$fig/figs_som_${gene_set}_kohonen_changes.pdf
    local sweep_fig=$fig/figs_som_${gene_set}_bestsize_kohonen.pdf
    if [[ -f $changes_fig && ( ! -f $sweep || -f $sweep_fig ) ]]; then
        echo "[skip] plot_training_diagnostics figures ($gene_set)"
    else
        "$PY" -m phh_som.plotting.plot_training_diagnostics --matrix "$matrix" --tag "$gene_set" \
            --data-dir "$data" --out-dir "$fig"
    fi

    if [[ -f $stats ]]; then
        echo "[skip] neuron_stats neuron stats ($gene_set)"
    else
        "$PY" -m phh_som.som.neuron_stats --assignments "$assignments" --out-dir "$data"
    fi

    if [[ -f $data/som_neuron_membership$suff.csv ]]; then
        echo "[skip] neuron_membership_table membership ($gene_set)"
    else
        "$PY" -m phh_som.som.neuron_membership_table --assignments "$assignments" \
            --stats "$stats" --core "$data/core_class.csv" --out-dir "$data"
    fi

    if [[ -f $fig/figs_som_${gene_set}_response_terrain.pdf ]]; then
        echo "[skip] plot_response_terrain response terrain ($gene_set)"
    else
        "$PY" -m phh_som.plotting.plot_response_terrain --tag "$gene_set" \
            --data-dir "$data" --out-dir "$fig"
    fi

    if [[ -f $fig/figs_som_${gene_set}_dependence_terrain.pdf ]]; then
        echo "[skip] plot_dependence_terrain dependence terrain ($gene_set)"
    else
        "$PY" -m phh_som.plotting.plot_dependence_terrain --tag "$gene_set" \
            --data-dir "$data" --out-dir "$fig"
    fi

    if [[ -f $fig/figs_som_${gene_set}_codes_map.pdf ]]; then
        echo "[skip] plot_codes_map codes map ($gene_set)"
    else
        "$PY" -m phh_som.plotting.plot_codes_map --tag "$gene_set" \
            --data-dir "$data" --out-dir "$fig"
    fi
}


run_universe expressed_28x27 expressed 28 27

run_universe expressed_24h_28x27 expressed_24h 28 27 --hpi 24

run_universe expressed_72h_28x27 expressed_72h 28 27 --hpi 72
