import numpy as np
import pandas as pd

from phh_som.utils.pipeline import FC_CUT, FDR_CUT, ISG_LIST_CSV, TPM_CUT, load_isg_symbols

UNRELIABLE_GENE_TYPES = {"misc_RNA", "snRNA", "snoRNA", "rRNA",
                         "rRNA_pseudogene", "scaRNA"}


def is_deg(df):
    return (df["fdr"] < FDR_CUT) & (df["fc"].abs() > FC_CUT) & (df["maxgroupmeans"] > TPM_CUT)


def add_contrast_short(de):
    de["contrast_short"] = de["virus"] + "_" + de["treatment"] + de["hpi"].astype(str)
    return de


def pivot_de(de, fillna_lfc=False):
    lfc = de.pivot(index="gene_id", columns="contrast_short", values="log2fc")
    deg = de.pivot(index="gene_id", columns="contrast_short",
                   values="deg").fillna(False)
    if fillna_lfc:
        lfc = lfc.fillna(0.0)
    return lfc, deg


def zscore_columns(masked):
    z = (masked - masked.mean()) / masked.std(ddof=0).replace(0, np.nan)
    return z.fillna(0.0)


def gene_flags(degs, isg_list_csv=ISG_LIST_CSV):
    ann = degs.drop_duplicates("gene_id").set_index("gene_id")
    isg_symbols = load_isg_symbols(isg_list_csv)
    flags = pd.DataFrame({
        "is_reliable": (ann["chromosome"] != "Y")
                       & (~ann["gene_type"].isin(UNRELIABLE_GENE_TYPES)),
        "is_isg": ann["symbol"].isin(isg_symbols),
    })
    return flags


def gene_annotation(degs):
    return degs.drop_duplicates("gene_id").set_index("gene_id")[["symbol", "gene_type"]]


def symbol_table(degs):
    return degs.drop_duplicates("gene_id").set_index("gene_id")["symbol"]
