#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import pandas as pd

from pipeline import DATA, FC_CUT, FDR_CUT, ISG_LIST_CSV, TPM_CUT, VIRUSES, load_isg_symbols

TOP_N = 50

UNRELIABLE_GENE_TYPES = {"misc_RNA", "snRNA", "snoRNA", "rRNA", "rRNA_pseudogene", "scaRNA"}


def is_deg(df):
    return (df["fdr"] < FDR_CUT) & (df["fc"].abs() > FC_CUT) & (df["maxgroupmeans"] > TPM_CUT)


def gene_flags(degs, isg_list_csv=ISG_LIST_CSV):
    ann = degs.drop_duplicates("gene_id").set_index("gene_id")
    isg_symbols = load_isg_symbols(isg_list_csv)
    flags = pd.DataFrame({
        "is_reliable": (ann["chromosome"] != "Y")
                       & (~ann["gene_type"].isin(UNRELIABLE_GENE_TYPES)),
        "is_isg": ann["symbol"].isin(isg_symbols),
    })
    return flags


def build_utrux_table(utrux, virus, flags):
    tab = utrux.loc[utrux["virus"] == virus,
                    ["gene_id", "symbol", "log2fc", "fc", "fdr",
                     "maxgroupmeans", "deg"]].copy()
    tab["is_reliable"] = tab["gene_id"].map(flags["is_reliable"]).fillna(True)
    tab["is_isg"] = tab["gene_id"].map(flags["is_isg"]).fillna(False)
    return tab


def top50_tails(tab, top_n=TOP_N):
    isg_tab = tab.loc[tab["is_isg"] & tab["is_reliable"]].dropna(subset=["log2fc"])
    ranked = isg_tab.sort_values("log2fc", ascending=False)
    return ranked.head(top_n), ranked.tail(top_n).sort_values("log2fc")


def main():
    ap = argparse.ArgumentParser(description='UT-vs-JAKi (untreated vs ruxolitinib, 72 hpi) tables for the')
    ap.add_argument("--utrux", type=Path, default=DATA / "utrux_long.parquet")
    ap.add_argument("--degs", type=Path, default=DATA / "degs.parquet")
    ap.add_argument("--isg-list", type=Path, default=ISG_LIST_CSV)
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()

    utrux = pd.read_parquet(args.utrux)
    utrux["deg"] = is_deg(utrux)
    flags = gene_flags(pd.read_parquet(args.degs), args.isg_list)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for virus in VIRUSES:
        v = virus.lower()
        tab = build_utrux_table(utrux, virus, flags)
        up, down = top50_tails(tab)
        tab.to_csv(args.out_dir / f"utrux_stats_{v}.csv", index=False)
        up.to_csv(args.out_dir / f"top50_isg_up_{v}.csv", index=False)
        down.to_csv(args.out_dir / f"top50_isg_down_{v}.csv", index=False)
        n_isg = int(tab["is_isg"].sum())
        print(f"[{virus}] {int(tab['deg'].sum()):,} genome-wide UT-vs-JAKi DEGs; "
              f"{int(tab.loc[tab['is_isg'], 'deg'].sum())}/{n_isg} ISGs; "
              f"top-50 up/down tails written "
              f"({up['log2fc'].min():.2f}..{up['log2fc'].max():.2f} / "
              f"{down['log2fc'].min():.2f}..{down['log2fc'].max():.2f})")


if __name__ == "__main__":
    main()
