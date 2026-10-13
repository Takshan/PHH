PY ?= .venv/bin/python
SOM := phh_som
PYS := ../$(PY)

ISG_ARGS := --tag isg_all --data-dir retrain_ISGs_allviruses/data

.PHONY: all pipeline figures clean-figures check-inputs

BASE_TABLES := $(SOM)/data/de_long.parquet $(SOM)/data/tpm_long.parquet \
	$(SOM)/data/utrux_long.parquet sources/raw_data/SchogginsISGlist_UPDATED.csv

all: pipeline figures

check-inputs:
	@missing=0; for f in $(BASE_TABLES); do \
		if [ ! -f "$$f" ]; then echo "missing: $$f"; missing=1; fi; \
	done; \
	if [ $$missing -eq 1 ]; then \
		echo; \
		echo "Place the combined base tables from the data deposition into"; \
		echo "$(SOM)/data/ and the ISG list into sources/raw_data/ (see README)."; \
		exit 1; \
	fi

pipeline: check-inputs
	cd $(SOM) && $(PYS) identify_degs.py
	cd $(SOM) && bash run_som_universes.sh
	cd $(SOM) && $(PYS) train_isg_universes.py
	cd $(SOM) && $(PYS) kinetic_classes.py
	cd $(SOM) && $(PYS) build_jaki_response_tables.py
	cd $(SOM) && $(PYS) timepoint_map_compare.py

figures: check-inputs
	cd $(SOM) && $(PYS) figure5_som_atlas.py
	cd $(SOM) && $(PYS) conservation_programmes.py
	cd $(SOM) && $(PYS) conservation_upset_untreated.py
	cd $(SOM) && $(PYS) plot_jaki_response_panels.py
	cd $(SOM) && $(PYS) plot_codes_map.py $(ISG_ARGS)
	cd $(SOM) && $(PYS) plot_response_terrain.py $(ISG_ARGS)
	cd $(SOM) && $(PYS) plot_dependence_terrain.py $(ISG_ARGS)
	cd $(SOM) && $(PYS) plot_jak_tier_chord.py
	cd $(SOM) && $(PYS) plot_conservation_jaki.py
	cd $(SOM) && $(PYS) plot_late_inflammation.py
	cd $(SOM) && $(PYS) plot_late_inflammation_genes.py
	cd $(SOM) && $(PYS) plot_late_vulnerabilities.py
	cd $(SOM) && $(PYS) plot_panflavi_panels.py
	cd $(SOM) && $(PYS) jak_dependence_figure.py
	cd $(SOM) && $(PYS) temporal_phases_figure.py

clean-figures:
	find $(SOM)/figures -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
	rm -rf $(SOM)/expressed_28x27/figures $(SOM)/expressed_24h_28x27/figures \
		$(SOM)/expressed_72h_28x27/figures $(SOM)/retrain_ISGs_allviruses/figures
