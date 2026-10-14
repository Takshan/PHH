PY ?= .venv/bin/python
SOM := phh_som
PYS := ../$(PY)

ISG_ARGS := --tag isg_all --data-dir results/retrain_ISGs_allviruses/data

.PHONY: all pipeline figures clean clean-figures check-inputs

BASE_TABLES := sources/raw_data/de_long.parquet \
	sources/raw_data/tpm_long.parquet sources/raw_data/utrux_long.parquet \
	sources/raw_data/SchogginsISGlist_UPDATED.csv

all: pipeline figures

check-inputs:
	@missing=0; for f in $(BASE_TABLES); do \
		if [ ! -f "$$f" ]; then echo "missing: $$f"; missing=1; fi; \
	done; \
	if [ $$missing -eq 1 ]; then \
		echo; \
		echo "Place the combined base tables and the ISG list from the data"; \
		echo "deposition into sources/raw_data/ (see README)."; \
		exit 1; \
	fi

pipeline: check-inputs
	$(PY) -m phh_som identify-degs
	bash $(SOM)/som/run_som_universes.sh
	$(PY) -m phh_som train-isg-universes
	$(PY) -m phh_som neuron-stats --assignments results/retrain_ISGs_allviruses/data/som_neuron_assignments_isg_all.tsv --out-dir results/data
	$(PY) -m phh_som module-table
	$(PY) -m phh_som kinetic-classes
	$(PY) -m phh_som jaki-response-tables
	$(PY) -m phh_som timepoint-compare

figures: check-inputs
	bash $(SOM)/som/run_som_universes.sh
	$(PY) -m phh_som train-isg-universes
	$(PY) -m phh_som figure5-som-atlas
	$(PY) -m phh_som conservation-programmes
	$(PY) -m phh_som conservation-upset-untreated
	$(PY) -m phh_som plot-jaki-response-panels
	$(PY) -m phh_som plot-codes-map $(ISG_ARGS)
	$(PY) -m phh_som plot-response-terrain $(ISG_ARGS)
	$(PY) -m phh_som plot-dependence-terrain $(ISG_ARGS)
	$(PY) -m phh_som plot-jak-tier-chord
	$(PY) -m phh_som plot-conservation-jaki
	$(PY) -m phh_som plot-late-inflammation
	$(PY) -m phh_som plot-late-inflammation-genes
	$(PY) -m phh_som plot-late-vulnerabilities
	$(PY) -m phh_som plot-panflavi-panels
	$(PY) -m phh_som jak-dependence-figure
	$(PY) -m phh_som temporal-phases-figure

clean:
	rm -rf results/data
	find results/figures -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

clean-figures:
	find results/figures -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
	rm -rf results/expressed_28x27/figures results/expressed_24h_28x27/figures \
		results/expressed_72h_28x27/figures results/retrain_ISGs_allviruses/figures
