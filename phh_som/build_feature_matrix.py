#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ALL_CONTRASTS, CONTRASTS, DATA, ISG_LIST_CSV, load_isg_symbols

MAX_ABS_LOG2FC = 20.0
MIN_MAX_GROUP_MEAN = 1.0
MIN_CPM = 1.0
MIN_SAMPLES = 3


def cpm_passing_genes(tpm_long, min_cpm=MIN_CPM, min_samples=MIN_SAMPLES):
    lib = tpm_long.groupby("sample_id")["total_counts"].sum()
    cpm = (tpm_long["total_counts"]
           / tpm_long["sample_id"].map(lib) * 1e6)
    n_pass = cpm[cpm >= min_cpm].groupby(tpm_long["gene_id"]).size()
    return n_pass[n_pass >= min_samples].index


def sanitize_log2fc(de):
    bad = de["log2fc"].abs() > MAX_ABS_LOG2FC
    if bad.any():
        rows = de.loc[bad, ["gene_id", "contrast_short", "log2fc"]]
        for _, r in rows.iterrows():
            print(f"  [sanitize] {r.gene_id} @ {r.contrast_short}: "
                  f"log2fc={r.log2fc:,.1f} -> NaN (division artefact)")
        de = de.copy()
        de.loc[bad, "log2fc"] = np.nan
    return de


def load_isg_ids(de, isg_list_csv=ISG_LIST_CSV):
    isg_symbols = load_isg_symbols(isg_list_csv)
    symbols = de.drop_duplicates("gene_id").set_index("gene_id")["symbol"]
    isg_ids = symbols[symbols.isin(isg_symbols)].index
    return isg_ids, len(isg_symbols)


def build_feature_matrix(de, isg_ids):
    lfc = de.pivot(index="gene_id", columns="contrast_short", values="log2fc")
    deg = de.pivot(index="gene_id", columns="contrast_short", values="deg").fillna(False)
    lfc = lfc.reindex(isg_ids)[CONTRASTS]
    deg = deg.reindex(isg_ids).fillna(False)[CONTRASTS]
    masked = lfc.fillna(0.0)
    
    z = (masked - masked.mean()) / masked.std(ddof=0).replace(0, np.nan)
    return z.fillna(0.0), masked, deg


def build_isg_all_feature_matrix(de, isg_ids):
    lfc = de.pivot(index="gene_id", columns="contrast_short", values="log2fc")
    deg = de.pivot(index="gene_id", columns="contrast_short", values="deg").fillna(False)
    lfc = lfc.reindex(isg_ids)[ALL_CONTRASTS]
    deg = deg.reindex(isg_ids).fillna(False)[ALL_CONTRASTS]
    masked = lfc.fillna(0.0)
    z = (masked - masked.mean()) / masked.std(ddof=0).replace(0, np.nan)
    return z.fillna(0.0), masked, deg


def build_all_feature_matrix(de):
    lfc = de.pivot(index="gene_id", columns="contrast_short", values="log2fc")
    deg = de.pivot(index="gene_id", columns="contrast_short", values="deg").fillna(False)
    lfc, deg = lfc[ALL_CONTRASTS], deg[ALL_CONTRASTS]
    masked = lfc.fillna(0.0)
    masked = masked.loc[masked.abs().sum(axis=1) > 0]
    z = (masked - masked.mean()) / masked.std(ddof=0).replace(0, np.nan)
    return z.fillna(0.0), masked, deg.loc[masked.index]


def build_expressed_feature_matrix(de, min_expr=MIN_MAX_GROUP_MEAN):
    lfc = de.pivot(index="gene_id", columns="contrast_short", values="log2fc")
    deg = de.pivot(index="gene_id", columns="contrast_short",
                   values="deg").fillna(False)
    expressed = de.groupby("gene_id")["maxgroupmeans"].max() > min_expr
    keep = expressed[expressed].index
    lfc, deg = lfc.loc[lfc.index.intersection(keep), ALL_CONTRASTS], deg[ALL_CONTRASTS]
    masked = lfc.fillna(0.0)
    masked = masked.loc[masked.abs().sum(axis=1) > 0]
    z = (masked - masked.mean()) / masked.std(ddof=0).replace(0, np.nan)
    return z.fillna(0.0), masked, deg.loc[masked.index]


def build_expressed1cpm_feature_matrix(de, tpm_long):
    keep = cpm_passing_genes(tpm_long)
    lfc = de.pivot(index="gene_id", columns="contrast_short", values="log2fc")
    deg = de.pivot(index="gene_id", columns="contrast_short",
                   values="deg").fillna(False)
    lfc, deg = lfc.loc[lfc.index.intersection(keep), ALL_CONTRASTS], deg[ALL_CONTRASTS]
    masked = lfc.fillna(0.0)
    masked = masked.loc[masked.abs().sum(axis=1) > 0]
    z = (masked - masked.mean()) / masked.std(ddof=0).replace(0, np.nan)
    return z.fillna(0.0), masked, deg.loc[masked.index]


def build_expressed_pc_feature_matrix(de, min_expr=MIN_MAX_GROUP_MEAN):
    return build_expressed_feature_matrix(
        de[de["gene_type"] == "protein_coding"], min_expr)


def build_wnvyfv_feature_matrix(de):
    lfc = de.pivot(index="gene_id", columns="contrast_short", values="log2fc")
    deg = de.pivot(index="gene_id", columns="contrast_short", values="deg").fillna(False)
    masked = lfc[CONTRASTS].where(deg[CONTRASTS], 0.0).fillna(0.0)
    keep = deg[CONTRASTS].sum(axis=1) >= 1
    masked = masked.loc[keep]
    masked = masked.loc[masked.abs().sum(axis=1) > 0]
    z = (masked - masked.mean()) / masked.std(ddof=0).replace(0, np.nan)
    return z.fillna(0.0), masked, deg.loc[masked.index, CONTRASTS]


def build_degs_feature_matrix(de):
    lfc = de.pivot(index="gene_id", columns="contrast_short", values="log2fc")
    deg = de.pivot(index="gene_id", columns="contrast_short", values="deg").fillna(False)
    lfc = lfc[ALL_CONTRASTS]
    deg = deg[ALL_CONTRASTS]
    unmasked = lfc.fillna(0.0)
    keep = deg.sum(axis=1) >= 1
    unmasked = unmasked.loc[keep]
    unmasked = unmasked.loc[unmasked.abs().sum(axis=1) > 0]
    z = (unmasked - unmasked.mean()) / unmasked.std(ddof=0).replace(0, np.nan)
    return z.fillna(0.0), unmasked, deg.loc[unmasked.index]


def main():
    ap = argparse.ArgumentParser(description='build the SOM feature matrix for a gene set (ISGs or all DEGs).')
    ap.add_argument("--gene-set",
                    choices=["isg", "isg_all", "wnvyfv", "degs", "all",
                             "expressed", "expressed_pc", "expressed1cpm"],
                    default="isg")
    ap.add_argument("--hpi", choices=["all", "24", "72"], default="all",
                    help="keep only this time point's 10 contrasts (20-contrast "
                         "gene sets only); tag gains the _24h/_72h suffix")
    ap.add_argument("--degs", type=Path, default=DATA / "degs.parquet")
    ap.add_argument("--tpm", type=Path, default=DATA / "tpm_long.parquet",
                    help="per-sample TPM/counts table (expressed1cpm only)")
    ap.add_argument("--isg-list", type=Path, default=ISG_LIST_CSV)
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()

    de = sanitize_log2fc(pd.read_parquet(args.degs))
    if args.gene_set == "isg":
        isg_ids, n_isg_list = load_isg_ids(de, args.isg_list)
        z, masked, deg = build_feature_matrix(de, isg_ids)
        print(f"{n_isg_list} Schoggins ISGs -> {z.shape[0]} matched "
              f"x {z.shape[1]} conditions ({int(deg.any(axis=1).sum())} DEG in >=1 condition)")
        assert 0 < z.shape[0] <= n_isg_list
        assert z.shape[1] == len(CONTRASTS)
    elif args.gene_set == "isg_all":
        isg_ids, n_isg_list = load_isg_ids(de, args.isg_list)
        z, masked, deg = build_isg_all_feature_matrix(de, isg_ids)
        print(f"{n_isg_list} Schoggins ISGs -> {z.shape[0]} matched "
              f"x {z.shape[1]} conditions (all viruses; "
              f"{int(deg.any(axis=1).sum())} DEG in >=1 condition)")
        assert 0 < z.shape[0] <= n_isg_list
        assert z.shape[1] == len(ALL_CONTRASTS)
    elif args.gene_set == "wnvyfv":
        z, masked, deg = build_wnvyfv_feature_matrix(de)
        print(f"WNV/YFV-responding matrix: {z.shape[0]} genes x {z.shape[1]} "
              f"conditions (DEG in >= 1 of the 8 WNV/YFV conditions)")
        assert z.shape[1] == len(CONTRASTS)
    elif args.gene_set == "expressed":
        z, masked, deg = build_expressed_feature_matrix(de)
        print(f"expressed-transcriptome matrix: {z.shape[0]} genes x "
              f"{z.shape[1]} conditions (max group mean > "
              f"{MIN_MAX_GROUP_MEAN} TPM in >= 1 condition, unmasked; "
              f"{int(deg.any(axis=1).sum())} DEG in >=1 condition)")
        assert z.shape[1] == len(ALL_CONTRASTS)
    elif args.gene_set == "expressed_pc":
        z, masked, deg = build_expressed_pc_feature_matrix(de)
        print(f"expressed protein-coding matrix: {z.shape[0]} genes x "
              f"{z.shape[1]} conditions (protein_coding AND max group mean > "
              f"{MIN_MAX_GROUP_MEAN} TPM in >= 1 condition, unmasked; "
              f"{int(deg.any(axis=1).sum())} DEG in >=1 condition)")
        assert z.shape[1] == len(ALL_CONTRASTS)
    elif args.gene_set == "expressed1cpm":
        z, masked, deg = build_expressed1cpm_feature_matrix(
            de, pd.read_parquet(args.tpm))
        print(f"expressed >= {MIN_CPM:g} CPM matrix: {z.shape[0]} genes x "
              f"{z.shape[1]} conditions (CPM >= {MIN_CPM:g} in >= "
              f"{MIN_SAMPLES} samples, unmasked; "
              f"{int(deg.any(axis=1).sum())} DEG in >=1 condition)")
        assert z.shape[1] == len(ALL_CONTRASTS)
    else:
        if args.gene_set == "all":
            z, masked, deg = build_all_feature_matrix(de)
            print(f"whole-transcriptome matrix: {z.shape[0]} genes x "
                  f"{z.shape[1]} conditions (every gene in the combined DE tables, "
                  f"unmasked; {int(deg.any(axis=1).sum())} DEG in >=1 condition)")
            assert z.shape[1] == len(ALL_CONTRASTS)
        else:
            z, masked, deg = build_degs_feature_matrix(de)
            print(f"all-DEG matrix: {z.shape[0]} genes x {z.shape[1]} conditions "
                  f"({int(deg.any(axis=1).sum())} DEG in >=1 condition)")

    tag = args.gene_set
    if args.hpi != "all":
        if z.shape[1] != len(ALL_CONTRASTS):
            raise SystemExit(f"--hpi applies only to the {len(ALL_CONTRASTS)}-contrast "
                             f"gene sets; '{args.gene_set}' has {z.shape[1]} columns")
        keep = [c for c in ALL_CONTRASTS if c.endswith(args.hpi)]
        z, masked, deg = z[keep], masked[keep], deg[keep]
        tag = f"{args.gene_set}_{args.hpi}h"
        print(f"--hpi {args.hpi}: kept {len(keep)} contrasts "
              f"({z.shape[0]} genes unchanged), tag '{tag}'")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    z.to_parquet(args.out_dir / f"{tag}_matrix_z.parquet")
    masked.to_parquet(args.out_dir / f"{tag}_matrix_masked_log2fc.parquet")
    deg.to_parquet(args.out_dir / f"{tag}_deg_mask.parquet")
    print(f"wrote {tag}_matrix_z / {tag}_matrix_masked_log2fc / "
          f"{tag}_deg_mask to {args.out_dir}")


if __name__ == "__main__":
    main()
