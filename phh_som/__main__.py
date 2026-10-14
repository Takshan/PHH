import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = {
    "identify-degs": "phh_som.stats.identify_degs",
    "train-isg-universes": "phh_som.som.train_isg_universes",
    "kinetic-classes": "phh_som.som.kinetic_classes",
    "jaki-response-tables": "phh_som.stats.build_jaki_response_tables",
    "timepoint-compare": "phh_som.stats.timepoint_map_compare",
    "neuron-stats": "phh_som.som.neuron_stats",
    "module-table": "phh_som.stats.build_module_table",
}

DRIVERS = {
    "run-universes": ROOT / "phh_som" / "som" / "run_som_universes.sh",
}

FIGURE_MODULES = [
    "conservation_programmes", "conservation_upset_multi",
    "conservation_upset_untreated", "figure5_som_atlas",
    "jak_dependence_figure", "jak_dependence_tiers", "late_inflammation",
    "plot_codes_map", "plot_conservation_jaki", "plot_dependence_terrain",
    "plot_jaki_response_panels", "plot_jak_tier_chord",
    "plot_late_inflammation_genes", "plot_late_inflammation",
    "plot_late_vulnerabilities", "plot_panflavi_panels",
    "plot_response_terrain", "plot_som_map", "plot_training_diagnostics",
    "som_atlas_panels", "som_figure_context", "temporal_phases_figure",
]
FIGURES = {m.replace("_", "-"): f"phh_som.plotting.{m}" for m in FIGURE_MODULES}

COMMANDS = {**STEPS, **DRIVERS, **FIGURES}

ISG_ARGS = ["--tag", "isg_all", "--data-dir", "results/retrain_ISGs_allviruses/data"]

GROUPS = {
    "pipeline": [(step, []) for step in
                 ["identify-degs", "run-universes", "train-isg-universes"]]
               + [("neuron-stats", ["--assignments",
                                    "results/retrain_ISGs_allviruses/data/som_neuron_assignments_isg_all.tsv",
                                    "--out-dir", "results/data"]),
                  ("module-table", [])]
               + [(step, []) for step in
                  ["kinetic-classes", "jaki-response-tables",
                   "timepoint-compare"]],
    "figures": [(cmd, []) for cmd in ["run-universes", "train-isg-universes",
                                      "figure5-som-atlas",
                                      "conservation-programmes",
                                      "conservation-upset-untreated",
                                      "plot-jaki-response-panels"]]
               + [(cmd, ISG_ARGS) for cmd in ["plot-codes-map",
                                              "plot-response-terrain",
                                              "plot-dependence-terrain"]]
               + [(cmd, []) for cmd in ["plot-jak-tier-chord",
                                        "plot-conservation-jaki",
                                        "plot-late-inflammation",
                                        "plot-late-inflammation-genes",
                                        "plot-late-vulnerabilities",
                                        "plot-panflavi-panels",
                                        "jak-dependence-figure",
                                        "temporal-phases-figure"]],
}

HELP = """usage: phh-som <command> [args]

groups:
  pipeline            run the seven pipeline steps (Makefile order)
  figures             run the universe drivers and every figure script

pipeline steps:
  identify-degs       flag DEGs in the long DE table -> data/degs.parquet
  run-universes       expressed SOM universes (bash driver)
  train-isg-universes ISG SOM universe (retrain_ISGs_allviruses)
  kinetic-classes     early/late/sustained neuron classes
  jaki-response-tables UT-vs-JAKi tables, top-50 ISG tails
  timepoint-compare   all-time vs 24h/72h map comparison
  neuron-stats        JAK dependence + core classes from degs.parquet
  module-table        supported-unit module table (names from config)

figure scripts:
  {figures}

extra args are passed through to the underlying script, e.g.
  python -m phh_som plot-response-terrain --tag isg_all --data-dir results/retrain_ISGs_allviruses/data
"""


def run(command, args):
    if command in DRIVERS:
        cmd = ["bash", str(DRIVERS[command]), *args]
    else:
        cmd = [sys.executable, "-m", COMMANDS[command], *args]
    print(f"\n+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode:
        sys.exit(result.returncode)


def main():
    argv = sys.argv[1:]
    help_text = HELP.replace("{figures}", "\n  ".join(sorted(FIGURES)))
    if not argv:
        sys.stderr.write(help_text)
        sys.exit(2)
    if argv[0] in ("-h", "--help"):
        print(help_text)
        return
    command, args = argv[0], argv[1:]
    if command in GROUPS:
        for step, group_args in GROUPS[command]:
            run(step, [*group_args, *args])
        return
    if command not in COMMANDS:
        sys.stderr.write(f"python -m phh_som: error: unknown command "
                         f"'{command}'\n\n{help_text}")
        sys.exit(2)
    run(command, args)


if __name__ == "__main__":
    main()
