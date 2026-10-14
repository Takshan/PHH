from pathlib import Path

import numpy as np
import pandas as pd

from phh_som.utils.deg import add_contrast_short, is_deg, pivot_de
from phh_som.utils.pipeline import ALL_CONTRASTS, VIRUS_ORDER

TOP_N = 50

TAIL_FILE = {"suppressed": "up", "resistant": "down"}


def jak_index(ut, rx):
    return (ut - rx) / (ut + rx).replace(0, np.nan)


def jak_dependence(de):
    lfc, deg = pivot_de(de, fillna_lfc=True)
    lfc = lfc[ALL_CONTRASTS]
    rows = []
    for v in VIRUS_ORDER:
        ut = lfc[[f"{v}_untreated24", f"{v}_untreated72"]].abs().mean(axis=1)
        rx = lfc[[f"{v}_Ruxo24", f"{v}_Ruxo72"]].abs().mean(axis=1)
        idx = jak_index(ut, rx)
        induced = deg[[f"{v}_untreated24", f"{v}_untreated72"]].any(axis=1)
        rows.append(pd.DataFrame({"virus": v, "gene_id": lfc.index[induced],
                                  "jak_index": idx[induced].to_numpy()}))
    return pd.concat(rows, ignore_index=True).dropna()


def jak_by_hpi(de):
    de = add_contrast_short(de.copy())
    de["deg"] = is_deg(de)
    lfc, deg = pivot_de(de)
    out = {}
    for hpi in ("24", "72"):
        rows = []
        for v in VIRUS_ORDER:
            ut = lfc[f"{v}_untreated{hpi}"].abs()
            rx = lfc[f"{v}_Ruxo{hpi}"].abs()
            idx = jak_index(ut, rx)
            induced = deg[f"{v}_untreated{hpi}"]
            rows.append(pd.DataFrame({"virus": v, "hpi": hpi,
                                      "gene_id": lfc.index[induced],
                                      "jak_index": idx[induced].to_numpy()}))
        out[hpi] = pd.concat(rows, ignore_index=True).dropna()
    return out


def top50_tails(tab, top_n=TOP_N):
    isg_tab = tab.loc[tab["is_isg"] & tab["is_reliable"]].dropna(subset=["log2fc"])
    ranked = isg_tab.sort_values("log2fc", ascending=False)
    return ranked.head(top_n), ranked.tail(top_n).sort_values("log2fc")


def top50_path(data_dir, virus, tail):
    return Path(data_dir) / f"top50_isg_{TAIL_FILE[tail]}_{virus.lower()}.csv"
