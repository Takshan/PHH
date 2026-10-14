import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLISH = Path(__file__).resolve().parents[1]


def out_root():
    env = os.environ.get("PHH_SOM_OUT")
    if env:
        return Path(env)
    cfg = ROOT / "config" / "paths.json"
    if cfg.exists():
        out = json.loads(cfg.read_text()).get("out_dir")
        if out:
            return Path(out)
    return ROOT / "results"


OUT = out_root()
DATA = OUT / "data"
FIGURES = OUT / "figures"


def universe(name):
    return OUT / name

RAW_DIR = ROOT / "sources" / "raw_data"
ISG_LIST_CSV = RAW_DIR / "SchogginsISGlist_UPDATED.csv"

SWEEP_R = PUBLISH / "som" / "som_kohonen_sweep.R"
TRAIN_R = PUBLISH / "som" / "som_kohonen_train.R"

FDR_CUT, FC_CUT, TPM_CUT = 0.05, 2.0, 1.0

GRID_X, GRID_Y = 5, 5
SEED = 143
RLEN = 1000
MARGINAL_MIN = 0.5

VIRUSES = ["WNV", "YFV"]
CONTRASTS = [f"{v}_{t}{h}" for v in VIRUSES
             for t in ("untreated", "Ruxo") for h in (24, 72)]

VIRUS_ORDER = ["ZIKV", "WNV", "TBEV", "DENV", "YFV"]
ALL_CONTRASTS = [f"{v}_{t}{h}" for v in VIRUS_ORDER
                 for t in ("untreated", "Ruxo") for h in (24, 72)]

PAPER_VIRUS_ORDER = ["WNV", "YFV", "DENV", "TBEV", "ZIKV"]

MM = 1 / 25.4

CLIP_Z = 6.0

PNG_DPI = 200

ISG_EXCLUDED_SYMBOLS = {"CMAHP"}


def load_isg_symbols(isg_list_csv=ISG_LIST_CSV):
    import pandas as pd

    symbols = set(pd.read_csv(isg_list_csv, header=None)[0])
    return symbols - ISG_EXCLUDED_SYMBOLS


def virus_colors():
    import json

    return json.loads((ROOT / "config" / "colors.json").read_text())["viruses"]


def tag_from_matrix(matrix_path):
    stem = Path(matrix_path).stem
    return stem[: -len("_matrix_z")] if stem.endswith("_matrix_z") else stem


def out_suffix(tag):
    return "" if tag == "isg" else f"_{tag}"


def save_figure(fig, pdf_path, png=True, dpi=None, **savefig_kw):
    from pathlib import Path as _Path

    pdf_path = _Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, **savefig_kw)
    written = [pdf_path]
    if png:
        png_path = pdf_path.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi or PNG_DPI, **savefig_kw)
        written.append(png_path)
    return written


def save_svg(fig, svg_path):
    import re
    from pathlib import Path as _Path

    svg_path = _Path(svg_path)
    fig.savefig(svg_path)
    svg_path.write_text(re.sub(r"font-size:\s*([0-9.]+)px",
                               r"font-size: \1pt", svg_path.read_text()))
    return [svg_path]


def set_rcparams():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
