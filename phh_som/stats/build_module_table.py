#!/usr/bin/env python3
import numpy as np
import pandas as pd

from phh_som.utils.pipeline import DATA, ROOT, VIRUS_ORDER, universe

NAMES_CSV = ROOT / "config" / "som_module_names.csv"


def main():
    degs = pd.read_parquet(DATA / "degs.parquet")
    meta = (degs.groupby("gene_id").agg(maxtpm=("maxgroupmeans", "max"))
            .join(degs[degs.deg].groupby("gene_id").size().rename("n_deg"))
            .fillna({"n_deg": 0}))
    symbol = degs.drop_duplicates("gene_id").set_index("gene_id")["symbol"]

    uni = universe("expressed_28x27") / "data"
    asg = pd.read_csv(uni / "som_neuron_assignments_expressed.tsv",
                      sep="\t").set_index("gene_id")["neuron"]
    fc = pd.read_parquet(uni / "expressed_matrix_masked_log2fc.parquet")
    isg_panel = set(pd.read_csv(universe("retrain_ISGs_allviruses") / "data"
                                / "som_neuron_membership_isg_all.csv")
                    .members.str.split("; ").explode().str.strip())

    names = pd.read_csv(NAMES_CSV, dtype={"merged_from": str}) \
              .set_index("unit")

    a = asg.to_frame("neuron").join(meta)
    st = a.groupby("neuron").agg(n=("neuron", "size"), med_tpm=("maxtpm", "median"),
                                 med_ndeg=("n_deg", "median"))
    sup = st[(st.n >= 4) & (st.med_tpm > 3) & (st.med_ndeg >= 4)]
    cen = fc.join(asg).groupby("neuron").mean()

    rows = []
    for n in sup.index:
        members = asg[asg == n].index
        ut = {v: max(cen.loc[n, f"{v}_untreated24"], cen.loc[n, f"{v}_untreated72"], key=abs)
              for v in VIRUS_ORDER}
        consensus = float(np.mean(list(ut.values())))
        row = dict(
            unit=int(n),
            module=names.module.get(n, "NEEDS_NAME"),
            merged_from=names.merged_from.get(n, str(int(n))),
            direction="induced" if consensus > 0 else "repressed",
            n_genes=int(sup.loc[n, "n"]),
            pct_ISG=round(100 * np.mean([symbol.get(g) in isg_panel for g in members]), 1),
            median_maxTPM=round(float(sup.loc[n, "med_tpm"]), 2),
        )
        for v in VIRUS_ORDER:
            row[f"amp_{v}"] = round(ut[v], 3)
        row["consensus_amp"] = round(consensus, 3)
        row["n_viruses_|amp|>=1"] = int(sum(abs(x) >= 1 for x in ut.values()))
        for v in VIRUS_ORDER:
            d = cen.loc[n, f"{v}_untreated24"]
            row[f"retain24_{v}"] = round(cen.loc[n, f"{v}_Ruxo24"] / d, 3) if abs(d) > 1e-9 else np.nan
        row["top_members"] = "; ".join(
            symbol.reindex(meta.maxtpm.reindex(members).sort_values(ascending=False)
                           .head(12).index).dropna())
        rows.append(row)

    T = pd.DataFrame(rows).sort_values(["direction", "consensus_amp"],
                                       ascending=[True, False])
    DATA.mkdir(parents=True, exist_ok=True)
    T.to_csv(DATA / "som_expressed_module_table.csv", index=False)

    named = (T.module != "NEEDS_NAME").sum()
    print(f"supported neurons: {len(T)} "
          f"({int((T.direction == 'induced').sum())} induced, "
          f"{int((T.direction == 'repressed').sum())} repressed); "
          f"{named} named, {len(T) - named} NEEDS_NAME")
    print(f"wrote {DATA / 'som_expressed_module_table.csv'}")


if __name__ == "__main__":
    main()
