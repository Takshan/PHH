# PHH-SOM

**Self-organizing map analysis of the primary human hepatocyte (PHH) response
to orthoflavivirus infection** — WNV, YFV, DENV, TBEV and ZIKV, ± the JAK
inhibitor ruxolitinib, at 24 and 72 hpi.

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557c?logo=matplotlib&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.18-8CAAE6?logo=scipy&logoColor=white)
![plotnine](https://img.shields.io/badge/plotnine-0.15.8-4C72B0)
![UpSetPlot](https://img.shields.io/badge/UpSetPlot-0.9-8E44AD)
![pycirclize](https://img.shields.io/badge/pycirclize-1.10-C0392B)
![R](https://img.shields.io/badge/R-%E2%89%A54-276DC3?logo=r&logoColor=white)
![kohonen](https://img.shields.io/badge/kohonen-SOM-276DC3)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)



## Installation

```
python -m venv .venv
.venv/bin/pip install .
```

This installs the pinned dependencies and the `phh-som` command.

R (>= 4) with the `kohonen` package (`install.packages("kohonen")`) is needed
for SOM **training only**; the trained SOMs ship with this repository, so
re-running the pipeline and all figures is pure Python.

## Usage

Everything runs through the package CLI, from the repo root:

```
phh-som --help        
phh-som pipeline     
phh-som figures       
phh-som plot-response-terrain --tag isg_all --data-dir results/retrain_ISGs_allviruses/data
```

(Without installing, `.venv/bin/python -m phh_som ...` works the same way.)

## Layout

```
Makefile                       runs the whole pipeline and every figure script
pyproject.toml                 package metadata + pinned dependencies
config/                        colours, panel gene list, curated SOM module names
phh_som/                       the code package
  utils/                       shared constants, paths and helpers
    pipeline.py                  shared constants/paths (imported by all steps)
    deg.py  jak.py  somdata.py  somgeom.py
  som/                         SOM universes: feature matrices, training,
                               per-neuron tables (drivers run_som_universes.sh,
                               train_isg_universes.py; R: som_kohonen_*.R)
  stats/                       DEG calls, JAKi response tables, map comparison,
                               module table
  plotting/                    every figure script
sources/raw_data/              input base tables + ISG list (from the data
                               deposition, not regenerated; gitignored)
results/
  data/                        regenerated derived tables (degs.parquet,
                               som_expressed_module_table.csv,
                               utrux_stats_*.csv, top50_isg_*.csv, ...)
  figures/<script_stem>/       generated figures, one directory per script
  expressed_28x27/data/        the 756-neuron expressed-gene SOM (map of record)
  expressed_24h_28x27/data/    the 24-hpi single-timepoint SOM (same loci/grid)
  expressed_72h_28x27/data/    the 72-hpi single-timepoint SOM (same loci/grid)
  retrain_ISGs_allviruses/data/  the 25-neuron 382-ISG SOM, all five viruses
```

## License

This project is released under the [MIT License](LICENSE).
