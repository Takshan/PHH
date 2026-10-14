#!/usr/bin/env python3
import pandas as pd

from phh_som.utils.pipeline import universe

DATA = universe("expressed_28x27") / "data"
VIRUSES = ["ZIKV", "WNV", "TBEV", "DENV", "YFV"]
Z_THRESHOLD = 1.0


def load_untreated_means(codebook):
    cb = pd.read_csv(codebook, sep="\t", index_col=0)
    return (cb[[f"{v}_untreated24" for v in VIRUSES]].mean(axis=1),
            cb[[f"{v}_untreated72" for v in VIRUSES]].mean(axis=1))


def classify(z24, z72, threshold=Z_THRESHOLD):
    up24, up72 = z24 > threshold, z72 > threshold
    dn24, dn72 = z24 < -threshold, z72 < -threshold

    direction = pd.Series("unresponsive", index=z24.index)
    direction[up24 | up72] = "induced"
    direction[dn24 | dn72] = "repressed"
    direction[(up24 | up72) & (dn24 | dn72)] = "discordant"

    active24 = up24 | dn24
    active72 = up72 | dn72
    kinetics = pd.Series("none", index=z24.index)
    kinetics[active24 & ~active72] = "early"
    kinetics[~active24 & active72] = "late"
    kinetics[active24 & active72] = "sustained"

    return pd.DataFrame({"mean_z_untreated24": z24, "mean_z_untreated72": z72,
                         "direction": direction, "kinetic_class": kinetics})


def summarize(calls):
    responsive = calls[calls["direction"] != "unresponsive"]
    table = (responsive.groupby(["direction", "kinetic_class"]).size()
             .unstack(fill_value=0)
             .reindex(columns=["early", "late", "sustained"], fill_value=0))
    table["total"] = table.sum(axis=1)
    return table


def main():
    z24, z72 = load_untreated_means(DATA / "som_codebook_expressed.tsv")
    calls = classify(z24, z72)
    calls.to_csv(DATA / "s25_kinetic_classes.csv", index_label="neuron")

    table = summarize(calls)
    print(f"z threshold {Z_THRESHOLD:+.1f}, {len(calls)} neurons on the map")
    print(table.to_string(), "\n")
    responsive = int(table["total"].sum())
    print(f"responsive neurons: {responsive}")
    for direction in ("induced", "repressed"):
        if direction not in table.index:
            continue
        row = table.loc[direction]
        pct = (row[["early", "late", "sustained"]] / row["total"] * 100).round(0)
        print(f"  {direction:<10} early {row['early']:>3} ({pct['early']:.0f}%)  "
              f"late {row['late']:>3} ({pct['late']:.0f}%)  "
              f"sustained {row['sustained']:>3} ({pct['sustained']:.0f}%)")

    print("\nthreshold sensitivity (late share of each direction):")
    for threshold in (0.5, 0.75, 1.0, 1.5, 2.0):
        alt = summarize(classify(z24, z72, threshold))
        parts = []
        for direction in ("induced", "repressed"):
            if direction in alt.index:
                row = alt.loc[direction]
                parts.append(f"{direction} {row['late'] / row['total']:.0%} "
                             f"(n={row['total']})")
        print(f"  z > {threshold:<4} " + "   ".join(parts))


if __name__ == "__main__":
    main()
