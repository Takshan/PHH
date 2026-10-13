#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from pipeline import DATA, FC_CUT, FDR_CUT, TPM_CUT


def is_deg(df):
    return (df["fdr"] < FDR_CUT) & (df["fc"].abs() > FC_CUT) & (df["maxgroupmeans"] > TPM_CUT)


def load_de(path):
    de = pd.read_parquet(path)
    de["deg"] = is_deg(de)
    de["contrast_short"] = de["virus"] + "_" + de["treatment"] + de["hpi"].astype(str)
    return de


def main():
    ap = argparse.ArgumentParser(description='flag differentially expressed genes in the long DE table.')
    ap.add_argument("--de-long", type=Path, default=DATA / "de_long.parquet")
    ap.add_argument("--out", type=Path, default=DATA / "degs.parquet")
    args = ap.parse_args()

    de = load_de(args.de_long)
    de.to_parquet(args.out, index=False)

    counts = de.groupby(["virus", "treatment", "hpi"], sort=True)["deg"].sum()
    print(f"DEG counts (fdr<{FDR_CUT} & |fc|>{FC_CUT} & maxgroupmeans>{TPM_CUT}):")
    print(counts.rename("n_deg").to_string())
    print(f"\ntotal rows: {len(de)}, contrasts: {de['contrast_short'].nunique()}, "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
