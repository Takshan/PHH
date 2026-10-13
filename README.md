# PHH orthoflavivirus SOM analysis

Code and derived data for the self-organizing map (SOM) analysis and the
figures for WNV, YFV, DENV, TBEV, ZIKV ± the JAK inhibitor
ruxolitinib, 24 and 72 hpi.

## Requirements

Python (3.12+; developed on 3.14):

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

R (>= 4) with the `kohonen` package installed and on the default library
path (`install.packages("kohonen")`)

# Layout

```
Makefile                       runs the whole pipeline and every figure script
requirements.txt               pinned Python dependencies
config/                        colours, panel gene list
phh_som/
  pipeline.py                  shared constants/paths (imported by all steps)
  temporal_phases_figure.py    temporal-phases figure script
  data/                        base derived tables (small ones only, see below)
  figures/<script_stem>/       generated figures, one directory per script
  expressed_28x27/data/        the 756-neuron expressed-gene SOM (map of record)
  expressed_24h_28x27/data/    the 24-hpi single-timepoint SOM (same loci/grid)
  expressed_72h_28x27/data/    the 72-hpi single-timepoint SOM (same loci/grid)
  retrain_ISGs_allviruses/data/  the 25-neuron 382-ISG SOM, all five viruses
```

## Analysis

Run the whole pipeline and every figure script with:

```
make all        # = make pipeline + make figures
```

or step by step:

```
make pipeline   # identify_degs -> run_som_universes.sh -> train_isg_universes.py
                # -> kinetic_classes -> build_jaki_response_tables
                # -> timepoint_map_compare
make figures    # every figure script (see the table below)
```

`make pipeline` expands to:

```
cd phh_som
../.venv/bin/python identify_degs.py         # -> data/degs.parquet
bash run_som_universes.sh                    # expressed universes -> expressed{,_24h,_72h}_28x27/
../.venv/bin/python train_isg_universes.py   # ISG universes -> retrain_ISGs_allviruses/
../.venv/bin/python kinetic_classes.py       # -> expressed_28x27/data/s25_kinetic_classes.csv
../.venv/bin/python build_jaki_response_tables.py  # -> data/utrux_stats_*.csv, top50_isg_*.csv
../.venv/bin/python timepoint_map_compare.py # -> expressed_28x27/data/s24_*.csv
```

SOM training is deterministic (kohonen, hexagonal grid, seed 143, rlen 1000,
learning rate 0.25 -> 0.01). Grids are fixed by hand — 28×27 = 756 units for
the expressed genes (same grid for the single-timepoint maps), 5×5 =
25 units for the 382 ISGs (`GRID_X`/`GRID_Y` in `pipeline.py`); the sweep
(`som_grid_sweep.py`) is a diagnostic only.

## Figures

`make figures` runs every script below. Run individual commands from
`phh_som/`. Every script writes PDF + PNG (+ SVG where used for assembly)
into its own directory under `phh_som/figures/<script_stem>/`.

### Fig — SOM atlas and conserved architecture

```
../.venv/bin/python figure5_som_atlas.py
../.venv/bin/python conservation_upset_untreated.py
../.venv/bin/python temporal_phases_figure.py
```

`figure5_som_atlas.py` draws the atlas, the programme × virus dot plot and
the temporal-phase tiles (its panels a, b, c) as
`figures/figure5_som_atlas/figure5.pdf`. `conservation_upset_untreated.py`
is panel C (untreated UpSet). `temporal_phases_figure.py` provides panels
E and F (unit trajectories, gene heat map; its panels c and b) alongside
two Extended Data variants in `figures/temporal_phases/`.

### Fig — JAK-STAT dependence of the ISG response

```
../.venv/bin/python plot_jaki_response_panels.py
../.venv/bin/python plot_codes_map.py --tag isg_all --data-dir retrain_ISGs_allviruses/data
../.venv/bin/python plot_response_terrain.py --tag isg_all --data-dir retrain_ISGs_allviruses/data
../.venv/bin/python plot_jak_tier_chord.py
../.venv/bin/python plot_dependence_terrain.py --tag isg_all --data-dir retrain_ISGs_allviruses/data
```

`plot_jaki_response_panels.py` draws panels A–D (volcanoes, waterfalls,
top-50 heat maps) from the shipped tables (rebuild them with
`build_jaki_response_tables.py`). The three `--tag isg_all` scripts draw
panel E (ISG SOM codes map), panel F (portraits, 5 viruses × 4
conditions) and panel H (mean J per unit) from the ISG universe.
`plot_jak_tier_chord.py` draws panel G (tier chord diagram).

### Fig — virus-restricted programmes, late inflammation, vulnerabilities

```
../.venv/bin/python plot_conservation_jaki.py
../.venv/bin/python plot_late_inflammation.py
../.venv/bin/python plot_late_inflammation_genes.py
../.venv/bin/python plot_late_vulnerabilities.py
```

`plot_conservation_jaki.py` draws panels A–C (restricted units, shared
repression ± JAKi). `plot_late_inflammation.py` draws panels D and E (its
panels a and b; AP-1 / chemokine programmes), with
`plot_late_inflammation_genes.py` as the gene-level companion.
`plot_late_vulnerabilities.py` draws panels G and H (its panels a and c;
ferroptosis / oxysterol / ER modules).

### Fig — JAK-STAT dependence and its 24 → 72 hpi decline

```
../.venv/bin/python jak_dependence_figure.py
```

Single script, whole page: `figures/jak_dependence_figure/fig_jak_combined.pdf`.

### Additional panels

```
../.venv/bin/python conservation_programmes.py
../.venv/bin/python plot_panflavi_panels.py
```

The full conservation UpSet across arms and time, and the pan-flavivirus
gain/timing panel.

## Results

See [`phh_som/figures/`](phh_som/figures/README.md) for the figures
