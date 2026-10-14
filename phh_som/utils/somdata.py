import pandas as pd

from phh_som.utils.pipeline import out_suffix

SUP_MIN_N = 4
SUP_MIN_MED_TPM = 3
SUP_MIN_MED_NDEG = 4


def load_universe(data_dir, tag):
    suff = out_suffix(tag)
    cb = pd.read_csv(data_dir / f"som_codebook{suff}.tsv", sep="\t", index_col=0)
    assign = pd.read_csv(data_dir / f"som_neuron_assignments{suff}.tsv", sep="\t")
    stats = pd.read_csv(data_dir / f"som_neuron_stats{suff}.csv",
                        index_col="neuron")
    return cb, assign, stats


def supported_neuron_stats(dg, asg):
    g_max = dg.groupby("gene_id").agg(maxtpm=("maxgroupmeans", "max"))
    n_de = dg[dg.deg].groupby("gene_id").size().rename("nd")
    a = asg.join(g_max).join(n_de).fillna({"nd": 0})
    us = a.groupby("neuron").agg(n=("neuron", "size"), mt=("maxtpm", "median"),
                                 md=("nd", "median"))
    keep = us[(us.n >= SUP_MIN_N) & (us.mt > SUP_MIN_MED_TPM)
              & (us.md >= SUP_MIN_MED_NDEG)].index
    return keep, us
