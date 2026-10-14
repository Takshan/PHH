#!/usr/bin/env python3
import re
import textwrap

import pandas as pd
pd.set_option("future.infer_string", False)

from phh_som.utils import pipeline
from phh_som.utils.deg import gene_annotation
from phh_som.plotting.plot_panflavi_panels import gain_fits, onset_percentages
from phh_som.plotting.som_figure_context import Context
from phh_som.plotting.conservation_programmes import FCMAP, FCNORM
from phh_som.plotting.plot_conservation_jaki import SUBCOLS
from phh_som.plotting.late_inflammation import GREY, INK, fc_key

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

TP_DIR = pipeline.FIGURES / "temporal_phases"
TP_DIR.mkdir(parents=True, exist_ok=True)
OUT = TP_DIR / "figure_temporal_phases"
V = pipeline.PAPER_VIRUS_ORDER
DY = 36.0
W_MM, H_MM = 183, 247 - DY
Y_OFF = 0.0

C_P1, C_P2, C_LATE, C_P3, C_ID = "#0072B2", "#009E73", "#CC79A7", "#D55E00", "#222222"
PHASE_COLOUR = {"p1": C_P1, "p2": C_P2, "late": C_LATE, "p3": C_P3, "id": C_ID}
LATE_BAND = dict(color="#F4EFE6", lw=0, zorder=-1)
C_CUT, DOT = "0.35", (0, (1, 1.5))
LOG2FC_CUT = float(np.log2(pipeline.FC_CUT))

TEXT_UNITS = [
    ("p1", 756, "756", ["IFNL1", "IFNL2", "IFNL3", "IFNL4", "IFNB1"], False),
    ("p2", 728, "728", ["IFIT1", "IFIT2", "RSAD2", "CXCL10", "CXCL11"], False),
    ("p2", 726, "726", ["OAS1", "OAS3", "RIGI", "DDX60"], True),
    ("p2", 725, "725", ["STAT1", "SAMHD1", "CD274"], False),
    ("p2", 727, "727", ["ISG15", "MX1", "IFI6", "EPSTI1",
                        "MX1-AS1", "NRIR", "LINC00487"], True),
    ("p2", 668, "668", ["IL33", "IL4I1", "MDK", "LAG3"], False),
    ("late", 640, "640", [], True),
]
IFN_LIGANDS = TEXT_UNITS[0][3]
IDENTITY = ["HNF4A", "HNF1A", "FOXA1", "FOXA2", "NR5A2"]
METAB_PATTERNS = {
    "ketogenesis": r"HMGCS2|HMGCL|BDH1|ACAT1|OXCT1",
    "alcohol DH": r"ADH\d\w*",
    "CYP450": r"CYP\d\w*",
    "conjugation": r"UGT\d\w*|SULT\d\w*|GST[AMPT]\d\w*|NAT\d",
    "lipid handling": (r"ACS[LMS]\d|ACOT\d+|ACOX\d|ACADS?B?|EHHADH|HADH|SCD|GPAM"
                       r"|PLIN\d|LPCAT\d|FADS\d|PNPLA\d|THRSP|SPTLC\d|APO\w+"),
}
SELECTIVE = [("WNV-restricted stress response", 648, ["ATF3", "BBC3", "PPP1R15A"]),
             ("accompanying inflammatory and stress genes", None,
              ["CCR1", "IL15", "IL15RA", "CGAS", "IFNGR1", "TNFAIP3"]),
             ("hepatic-metabolic unit", 32, ["ACOT12", "ANXA13", "SULT1E1", "UGT2B15"])]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 6,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})
VIRUS_COLORS = pipeline.virus_colors()

E = pipeline.universe("expressed_28x27") / "data"
fc = pd.read_parquet(E / "expressed_matrix_masked_log2fc.parquet")
unit = pd.read_csv(E / "som_neuron_assignments_expressed.tsv",
                   sep="\t").set_index("gene_id")["neuron"]
degs = pd.read_parquet(pipeline.DATA / "degs.parquet")
sym = gene_annotation(degs)
tpm = pd.read_parquet(pipeline.RAW_DIR / "tpm_long.parquet",
                      columns=["gene_id", "infection", "treatment", "hpi", "tpm"])
ctx = Context()


def gene_id(symbol):
    hits = sym.index[(sym.symbol == symbol) & sym.index.isin(fc.index)]
    assert len(hits) == 1, f"{symbol}: {len(hits)} loci in the expressed matrix"
    return hits[0]


def is_unnamed(g):
    s = sym.symbol.get(g)
    return not isinstance(s, str) or bool(re.fullmatch(r"ENSG\d+", s))


def unnamed(n):
    return [g for g in unit.index[unit == n] if is_unnamed(g)]


for _, n, _, genes, _ in TEXT_UNITS:
    for s in genes:
        assert unit[gene_id(s)] == n, f"{s} sits in unit {unit[gene_id(s)]}, text says {n}"
for _, n, genes in SELECTIVE:
    for s in genes:
        assert n is None or unit[gene_id(s)] == n, f"{s} not in unit {n}"

d_ut = degs[(degs.treatment == "untreated") & degs.deg]
n_up = d_ut[d_ut.log2fc > 0].groupby("gene_id").virus.nunique()
conserved = [g for g in n_up.index[n_up >= 4] if g in fc.index]
conserved_lnc = [g for g in conserved if sym.gene_type.get(g) == "lncRNA"]

rep_members = unit.index[unit.isin(ctx.repressed_units)]
metab = {cat: [g for g in rep_members if not is_unnamed(g)
               and re.fullmatch(p, sym.symbol[g])]
         for cat, p in METAB_PATTERNS.items()}
metab_all = sorted(set().union(*metab.values()))

PHASE_SETS = {"p1": [gene_id(s) for s in IFN_LIGANDS], "p2": conserved,
              "late": unnamed(640), "p3": metab_all}

fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))


def ax_mm(x, y, w, h):
    y -= Y_OFF
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM])


def letter(x, y, s, title):
    y -= Y_OFF
    fig.text(x / W_MM, 1 - y / H_MM, s, fontsize=8, fontweight="bold",
             ha="left", va="top")
    fig.text((x + 4) / W_MM, 1 - (y + 0.4) / H_MM, title, fontsize=6.5,
             ha="left", va="top")


def mm_canvas(f, w, h):
    c = f.add_axes([0, 0, 1, 1], zorder=-1)
    c.set_xlim(0, w)
    c.set_ylim(h, 0)
    c.axis("off")
    return c


cv = mm_canvas(fig, W_MM, H_MM)


def virus_chips(canvas, y, x_of, width, h=3.2, fs=5):
    for i, v in enumerate(V):
        canvas.add_patch(plt.Rectangle((x_of(i), y), width, h,
                                       fc=VIRUS_COLORS[v], ec="none"))
        canvas.text(x_of(i) + width / 2, y + h / 2, v, ha="center", va="center",
                    fontsize=fs, color="white", fontweight="bold")


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


TPM_FLOOR = 1.0
FLOOR_FADE = 0.38
XS = [0, 24, 72]
_gm_wide = (tpm[tpm.treatment == "untreated"]
            .groupby(["gene_id", "infection", "hpi"]).tpm.mean()
            .unstack(["infection", "hpi"]).reindex(fc.index).fillna(0.0))


def above_floor(ids, v, h):
    return float(np.median(np.fmax(_gm_wide.loc[ids, (v, h)].to_numpy(),
                                   _gm_wide.loc[ids, ("Mock", h)].to_numpy()))
                 ) >= TPM_FLOOR


named_p2 = [s for ph, _, _, gs, _ in TEXT_UNITS if ph == "p2" for s in gs]
A_SETS = [
    ("p1", f"Phase 1 · IFN ligands ({len(PHASE_SETS['p1'])})"),
    ("p2", f"Phase 2 · induced by ≥ 4 viruses ({len(conserved)})"),
    ("late", f"late orphan · N640 ({len(PHASE_SETS['late'])})"),
    ("p3", f"Phase 3 · metabolic genes ({len(metab_all)})"),
]
med = {(k, v): (fc.loc[PHASE_SETS[k], f"{v}_untreated24"].median(),
                fc.loc[PHASE_SETS[k], f"{v}_untreated72"].median())
       for k, _ in A_SETS for v in V}
floor_ok = {(k, v, h): above_floor(PHASE_SETS[k], v, h)
            for k, _ in A_SETS for v in V for h in (24, 72)}
AX0, AGAP = 14.0, 8.0
AW = (W_MM - 3.0 - AX0 - (len(A_SETS) - 1) * AGAP) / len(A_SETS)
ATOP, AH = 16.0, 24.0
for j, (k, title) in enumerate(A_SETS):
    axa = ax_mm(AX0 + j * (AW + AGAP), ATOP, AW, AH)
    axa.axvspan(48, 76, **LATE_BAND)
    axa.axhline(0, color="0.75", lw=0.4, zorder=0)
    for cut in (-LOG2FC_CUT, LOG2FC_CUT):
        axa.axhline(cut, color=C_CUT, lw=0.45, ls=DOT, zorder=0)
    for v in V:
        ys = [0, *med[(k, v)]]
        okp = [True, floor_ok[(k, v, 24)], floor_ok[(k, v, 72)]]
        for s in range(2):
            axa.plot(XS[s:s + 2], ys[s:s + 2], color=VIRUS_COLORS[v], lw=1.0,
                     alpha=1.0 if okp[s] and okp[s + 1] else FLOOR_FADE, zorder=2)
        for x, y, o in zip(XS, ys, okp):
            axa.plot(x, y, "o", ms=2.2, mew=0.5, mec=VIRUS_COLORS[v], zorder=3,
                     mfc=VIRUS_COLORS[v] if o else "white")
    lo_a = min(-LOG2FC_CUT, min(min(med[(k, v)]) for v in V))
    hi_a = max(LOG2FC_CUT, max(max(med[(k, v)]) for v in V))
    pad = 0.06 * (hi_a - lo_a)
    axa.set_ylim(lo_a - pad, hi_a + pad)
    axa.set_xlim(-6, 78)
    axa.set_xticks([0, 24, 72])
    axa.tick_params(labelsize=5, length=1.5, pad=1)
    axa.set_title(title, fontsize=5.5, color=PHASE_COLOUR[k], pad=3)
    if j == 0:
        axa.set_ylabel("median log$_2$FC", fontsize=5.3)
    despine(axa)
fig.legend(handles=[mpl.lines.Line2D([], [], color=VIRUS_COLORS[v], lw=1.0,
                                     marker="o", ms=2.2, mec="none", label=v)
                    for v in V]
           + [mpl.lines.Line2D([], [], color=C_CUT, lw=0.7, ls=DOT,
                               label="two-fold (|log$_2$FC| = 1)"),
              mpl.lines.Line2D([], [], color="0.62", lw=0.9, marker="o", ms=2.2,
                               mfc="white", mec="0.35", mew=0.6,
                               label="median member < 1 TPM")],
           loc="center left", ncol=7, frameon=False, fontsize=5.3,
           handlelength=1.8, columnspacing=1.4, borderaxespad=0,
           bbox_to_anchor=(AX0 / W_MM, 1 - 8.6 / H_MM))
letter(1, 3, "a", "Phases per virus: trend and strength")

ifn_syms = sorted(s for s in sym.symbol.dropna().unique()
                  if re.fullmatch(r"IFNL\d|IFNB1|IFNA\d+|IFNG", s))
ifn_ids = sym.index[sym.symbol.isin(ifn_syms)]
t = tpm[tpm.gene_id.isin(ifn_ids) & (tpm.infection != "Mock")]
gm = (t.groupby(["gene_id", "infection", "treatment", "hpi"]).tpm.mean()
      .groupby(["gene_id", "infection", "treatment"]).max().reset_index())
gm["symbol"] = gm.gene_id.map(sym.symbol)
gm["gene"] = gm.symbol.where(~gm.symbol.str.match(r"IFNA\d+$"), "IFNA")
peak = gm.groupby(["gene", "infection", "treatment"]).tpm.max()

Y_OFF = DY
cv.set_ylim(H_MM + DY, DY)

UT_COLS = [(24, "untreated", "24"), (72, "untreated", "72")]
RX_COLS = [(h, tr, f"{h} {sign}") for h, tr, sign in SUBCOLS]
FULL = dict(cell=2.3, gap=1.8, group_gap=0.9, chip_h=3.2, chip_fs=3.9,
            head_fs=4.3, head_up=7.4, label_fs=5.0, unnamed_fs=4.3, unit_fs=4.3,
            unit_dx=1.2, bracket_dx=7.0, name_fs=4.8, name_wrap=21, mono=False)
COMPACT = dict(cell=1.7, gap=1.0, group_gap=0.7, chip_h=2.4, chip_fs=3.6,
               head_fs=3.0, head_up=3.2, label_fs=4.0, unnamed_fs=3.5,
               unit_fs=3.3, unit_dx=0.8, bracket_dx=5.2, name_fs=4.0,
               name_wrap=26, mono=True)
C_MAIN = dict(COMPACT, cell=1.8, chip_h=2.6, chip_fs=3.8, label_fs=4.5,
              unnamed_fs=3.8, unit_fs=3.6, name_fs=4.6, name_wrap=24,
              mono=False)


def heat_grid(canvas, x0, top, groups, subcols=UT_COLS, st=FULL):
    cell, gap = st["cell"], st["gap"]
    bw = len(subcols) * cell
    step = bw + gap
    x1 = x0 + len(V) * bw + (len(V) - 1) * gap
    rotated = len(subcols) > 2
    virus_chips(canvas, top, lambda i: x0 + i * step, bw, h=st["chip_h"],
          fs=st["chip_fs"] if not rotated or st["mono"] else 5)
    head_top = top + st["chip_h"] + 0.4
    for i in range(len(V)):
        for c, (_, _, head) in enumerate(subcols):
            xc = x0 + i * step + (c + 0.5) * cell
            if rotated:
                canvas.text(xc, head_top + st["head_up"], head, ha="center",
                            va="bottom", fontsize=st["head_fs"], color="0.25",
                            rotation=90)
            else:
                canvas.text(xc, head_top, head, ha="center", va="top",
                            fontsize=st["head_fs"], color="0.25")
    y = head_top + (st["head_up"] + 0.7 if rotated else 2.9)
    canvas.text(x1 + st["unit_dx"], y - 0.4, "SOM\nunit", ha="left",
                va="bottom", fontsize=st["unit_fs"], color="0.45")
    xb = x1 + st["bracket_dx"]
    for k, (name, colour, rows) in enumerate(groups):
        if k:
            y += st["group_gap"]
        first = y
        for label, ids, kind in rows:
            canvas.text(x0 - 0.6, y + cell / 2, label, ha="right", va="center",
                        fontsize=st["unnamed_fs"] if kind == "unnamed"
                        else st["label_fs"],
                        style="italic" if kind == "gene" else "normal",
                        color=INK if st["mono"] or kind != "unnamed" else "0.4")
            for i, v in enumerate(V):
                for c, (h, tr, _) in enumerate(subcols):
                    val = fc.loc[ids, f"{v}_{tr}{h}"].median()
                    canvas.add_patch(plt.Rectangle(
                        (x0 + i * step + c * cell, y), cell, cell,
                        fc=FCMAP(FCNORM(val)), ec="0.7", lw=0.2))
            units = {int(unit[g]) for g in ids}
            canvas.text(x1 + st["unit_dx"], y + cell / 2,
                        f"N{units.pop()}" if len(units) == 1 else "",
                        ha="left", va="center", fontsize=st["unit_fs"],
                        color="0.45")
            y += cell
        for i in range(len(V)):
            canvas.add_patch(plt.Rectangle((x0 + i * step, first), bw,
                                           y - first, fill=False, ec="black",
                                           lw=0.6 if not st["mono"] else 0.5))
        canvas.plot([xb, xb], [first + 0.3, y - 0.3], color=colour, lw=0.6)
        canvas.text(xb + 1.0, (first + y) / 2, textwrap.fill(name, st["name_wrap"]),
                    fontsize=st["name_fs"], va="center", color=colour,
                    linespacing=1.05)
    return y


PHASE_GROUP = {
    756: ("Phase 1 · interferon ligands", C_P1),
    728: ("Phase 2 · IRF3-direct IFIT–RSAD2–CXCL10/11", C_P2),
    726: ("OAS1/3–RIGI–DDX60 sensing / amplification", C_P2),
    725: ("STAT1/SAMHD1/CD274", C_P2),
    727: ("ISG15/MX1/IFI6/EPSTI1 with lncRNAs", C_P2),
    668: ("secreted immunomodulators", C_P2),
    640: ("late orphan programme", C_LATE),
}
c_groups = []
for _, n, _, genes, show in TEXT_UNITS:
    rows = [(s, [gene_id(s)], "gene") for s in genes]
    if show:
        rows += [(g, [g], "unnamed") for g in unnamed(n)]
    c_groups.append((*PHASE_GROUP[n], rows))
dn72 = d_ut[(d_ut.hpi == 72) & (d_ut.log2fc < 0)].groupby("gene_id").virus.nunique()
max_tpm = degs.groupby("gene_id").maxgroupmeans.max()
CAT_NAME = {"ketogenesis": "Phase 3 · ketogenesis", "alcohol DH": "alcohol DH",
            "CYP450": "CYP450s", "conjugation": "conjugation", "lipid handling": "lipid handling"}
for cat, ids in metab.items():
    top = sorted(ids, key=lambda g: (-dn72.get(g, 0), -max_tpm.get(g, 0)))[:2]
    c_groups.append((CAT_NAME[cat], C_P3, [(sym.symbol[g], [g], "gene") for g in top]))
c_groups.append(("hepatocyte identity", C_ID,
                 [(s, [gene_id(s)], "gene") for s in IDENTITY]))
for name, _, genes in SELECTIVE[:2]:
    c_groups.append((name, INK, [(s, [gene_id(s)], "gene") for s in genes]))
C_TOP = 92.5
c_bottom = heat_grid(cv, 20.0, C_TOP, c_groups, st=C_MAIN)
letter(1, 86, "b", "The phases gene by gene")
fc_key(fig, 28.0, H_MM - (c_bottom - Y_OFF) - 8.0,
       note="b: untreated infection; columns 24 and 72 hpi")

D_ROWS = [(f"N{n} · n = {(unit == n).sum()}\n{name}", list(unit.index[unit == n]),
           C_P1 if n == 756 else C_LATE if n == 640 else C_P2)
          for n, name in ((756, "IFN ligands"), (728, "IRF3-direct"),
                          (726, "OAS1/3–RIGI"), (725, "STAT1/CD274"),
                          (727, "ISG15/MX1/IFI6"), (668, "secreted"),
                          (640, "unnamed transcripts"))]
D_ROWS += [(f"Phase 3 · n = {len(metab_all)}\nmetabolic genes", metab_all, C_P3)]
onset = onset_percentages()
DX0, DW, DG = 104.0, 14.0, 1.5
DTOP, DH, DRG = 97.5, 5.4, 1.3
virus_chips(cv, 90.6, lambda i: DX0 + i * (DW + DG), DW, h=2.6, fs=4.2)
cv.text(DX0 - 1.0, 94.8, "DEGs 24/72 h", ha="right", va="center", fontsize=4.2,
        color="0.4")
for i, v in enumerate(V):
    cv.text(DX0 + i * (DW + DG) + DW / 2, 94.8, f"{onset[v]:.0f}%", ha="center",
            va="center", fontsize=4.4, color="0.4")
D_ARMS = (("untreated", GREY, 0.9), ("Ruxo", None, 1.2))


def trend_grid(rows, top, arms, h=DH, rgap=DRG, x0=DX0, w=DW, gap=DG,
               axf=None, canvas=None):
    axf, canvas = axf or ax_mm, canvas or cv
    for r, (lab, ids, colour) in enumerate(rows):
        m = {(v, tr, hh): fc.loc[ids, f"{v}_{tr}{hh}"].median()
             for v in V for tr, *_ in arms for hh in (24, 72)}
        lo, hi = min(0, *m.values()), max(0, *m.values())
        pad = 0.15 * max(hi - lo, 1)
        span = hi - lo
        ticks = sorted({0} | ({int(np.floor(hi))} if hi >= 1 and hi >= 0.25 * span else set())
                       | ({int(np.ceil(lo))} if lo <= -1 and -lo >= 0.25 * span else set()))
        y0 = top + r * (h + rgap)
        for i, v in enumerate(V):
            axd = axf(x0 + i * (w + gap), y0, w, h)
            axd.axvspan(48, 76, **LATE_BAND)
            axd.axhline(0, color="#D0D4D8", lw=0.4, zorder=0)
            for tr, c, lw, *rest in arms:
                ls = rest[0] if rest else "-"
                axd.plot([0, 24, 72], [0, m[(v, tr, 24)], m[(v, tr, 72)]],
                         marker="o", ls=ls, color=c or VIRUS_COLORS.get(v, INK),
                         lw=lw, ms=1.8, mec="none")
            axd.set_xlim(-6, 78)
            axd.set_ylim(lo - pad, hi + pad)
            axd.set_xticks([0, 24, 72])
            axd.set_yticks(ticks)
            axd.tick_params(labelsize=4, length=1.5, pad=1)
            if r < len(rows) - 1:
                axd.set_xticklabels([])
            if i:
                axd.set_yticklabels([])
            despine(axd)
        canvas.text(x0 - 4.2, y0 + h / 2, lab, ha="right", va="center",
                    fontsize=4.4, color=colour, linespacing=1.0)
    bottom = top + len(rows) * (h + rgap) - rgap
    canvas.text(x0 + (len(V) * w + (len(V) - 1) * gap) / 2, bottom + 3.6,
                "hpi (0 = mock)", ha="center", va="top", fontsize=4.6)
    return bottom


d_bottom = trend_grid(D_ROWS, DTOP, D_ARMS)
letter(86, 86, "c", "Unit trajectories: median log$_2$FC vs mock")
fig.legend(handles=[plt.Line2D([], [], color=GREY, lw=0.9, marker="o", ms=1.8,
                               label="untreated"),
                    plt.Line2D([], [], color=INK, lw=1.2, marker="o", ms=1.8,
                               label="JAKi (virus colour)")],
           loc="upper right", bbox_to_anchor=(182 / W_MM, 1 - (85.6 - Y_OFF) / H_MM),
           ncol=2, frameon=False, fontsize=5, handlelength=1.6,
           columnspacing=1.0, borderaxespad=0)

amp, consensus, fits = gain_fits()
axe = ax_mm(97, 161, 44, 18)
grid = np.linspace(consensus.min(), consensus.max(), 50)
for v in sorted(V, key=lambda v: -fits[v].slope):
    c = VIRUS_COLORS.get(v, "0.3")
    axe.scatter(consensus, amp[v], s=2.5, color=c, lw=0, alpha=0.7, zorder=2)
    axe.plot(grid, fits[v].slope * grid + fits[v].intercept, color=c, lw=0.9,
             zorder=3)
axe.text(1.05, 0.99, "gain g (R²)", transform=axe.transAxes, fontsize=4.8,
         va="top", color="0.3")
for i, v in enumerate(sorted(V, key=lambda v: -fits[v].slope)):
    axe.text(1.05, 0.86 - i * 0.13, f"{v}  {fits[v].slope:.2f} "
             f"({fits[v].rvalue ** 2:.2f})", transform=axe.transAxes,
             color=VIRUS_COLORS.get(v, "black"), fontsize=4.8, va="top",
             fontweight="bold")
axe.axhline(0, color="0.7", lw=0.4, zorder=1)
axe.axvline(0, color="0.7", lw=0.4, zorder=1)
axe.set_xlabel(f"Consensus programme amplitude (log$_2$FC, n = {len(amp)})",
               fontsize=5.3)
axe.set_ylabel("Virus programme\namplitude", fontsize=5.3)
despine(axe)
letter(86, 155.5, "d", "One gain per virus")

def unit_row(n, genes):
    ids = list(unit.index[unit == n])
    more = ", …" if len(genes) < len(ids) else ""
    return (f"N{n} · n = {len(ids)}\n" + textwrap.fill(", ".join(genes) + more, 26),
            ids, INK)


acc = SELECTIVE[1][2]
acc_units = list(dict.fromkeys(int(unit[gene_id(s)]) for s in acc))
F_ROWS = ([unit_row(SELECTIVE[0][1], SELECTIVE[0][2])]
          + [unit_row(n, [s for s in acc if unit[gene_id(s)] == n]) for n in acc_units]
          + [unit_row(SELECTIVE[2][1], SELECTIVE[2][2])])
virus_chips(cv, 195.5, lambda i: DX0 + i * (DW + DG), DW, h=2.6, fs=4.2)
trend_grid(F_ROWS, 200.5, (("untreated", GREY, 1.0),), h=4.6, rgap=1.0)
letter(86, 190, "e", "Virus-selective units: whole SOM units, untreated")

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=450)
fig.savefig(OUT.with_suffix(".svg"))
print(f"wrote {OUT}.pdf/.png/.svg")

OUT_RX = TP_DIR / "figure_temporal_phases_ruxolitinib"
RW, RH = 84, 138
fig3 = plt.figure(figsize=(RW / 25.4, RH / 25.4))
cv3 = mm_canvas(fig3, RW, RH)
rx_bottom = heat_grid(cv3, 16.0, 2.0, c_groups, RX_COLS, st=COMPACT)
fc_key(fig3, 14.0, RH - rx_bottom - 6.0,
       note="columns: hpi, − untreated, + JAKi; each against its "
            "treatment-matched mock")
fig3.savefig(OUT_RX.with_suffix(".pdf"))
fig3.savefig(OUT_RX.with_suffix(".png"), dpi=450)
fig3.savefig(OUT_RX.with_suffix(".svg"))
print(f"wrote {OUT_RX}.pdf/.png/.svg  (grid bottom {rx_bottom:.0f} mm of {RH})")

OUT_P3 = TP_DIR / "figure_temporal_phases_phase3_units"
metab_set = set(metab_all)
p3, dropped = [], []
for n in ctx.repressed_units:
    mem = list(unit.index[unit == n])
    if sym.gene_type.reindex(mem).astype(str).str.contains("rRNA").any():
        dropped.append(n)
        continue
    named = [g for g in sorted(mem, key=lambda g: -max_tpm.get(g, 0))
             if not is_unnamed(g)][:3]
    p3.append((sum(g in metab_set for g in mem), n, mem, named))
p3.sort(key=lambda t: (-t[0], t[1]))
P3_ROWS = [(f"N{n} · n = {len(mem)} ({k} metabolic)\n"
            + ", ".join(sym.symbol[g] for g in named) + (", …" if len(named) < len(mem) else ""),
            mem, C_P3 if k else "0.45") for k, n, mem, named in p3]
PW, PRH, PRG = 183, 6.0, 1.4
PH = 16 + len(P3_ROWS) * (PRH + PRG) + 8
fig4 = plt.figure(figsize=(PW / 25.4, PH / 25.4))
cv4 = mm_canvas(fig4, PW, PH)


def ax4_mm(x, y, w, h):
    return fig4.add_axes([x / PW, 1 - (y + h) / PH, w / PW, h / PH])


PX0, PCW, PCG = 36.0, 26.5, 2.2
virus_chips(cv4, 8.5, lambda i: PX0 + i * (PCW + PCG), PCW, h=2.8, fs=5)
cv4.text(PX0 - 4.2, 13.6, "unit · n = members (metabolic-category genes)", ha="right",
         va="center", fontsize=4.4, color="0.4")
p3_bottom = trend_grid(P3_ROWS, 16.0, D_ARMS, h=PRH, rgap=PRG, x0=PX0, w=PCW,
                       gap=PCG, axf=ax4_mm, canvas=cv4)
if dropped:
    cv4.text(PW - 2, p3_bottom + 3.6, "not shown: " + ", ".join(f"N{n}" for n in dropped)
             + " (rRNA genes; residual rRNA in poly(A) libraries)", ha="right",
             va="top", fontsize=4.6, color="0.4")
fig4.legend(handles=[plt.Line2D([], [], color=GREY, lw=0.9, marker="o", ms=1.8,
                                label="untreated"),
                     plt.Line2D([], [], color=INK, lw=1.2, marker="o", ms=1.8,
                                label="JAKi (virus colour)")],
            loc="upper left", bbox_to_anchor=(2 / PW, 1 - 2 / PH), ncol=2,
            frameon=False, fontsize=5, handlelength=1.6, columnspacing=1.0,
            borderaxespad=0)
cv4.text(PW - 2, 3.0, "median member log$_2$FC vs mock, all members of each "
         "repressed programme", ha="right", va="center", fontsize=5, color="0.3")
fig4.savefig(OUT_P3.with_suffix(".pdf"))
fig4.savefig(OUT_P3.with_suffix(".png"), dpi=450)
fig4.savefig(OUT_P3.with_suffix(".svg"))
print(f"wrote {OUT_P3}.pdf/.png/.svg  ({len(P3_ROWS)} programmes)")

ifnl = [peak[(g, v, "untreated")] for g in ("IFNL1", "IFNL2", "IFNL3") for v in V]
print(f"IFNL1-3 untreated per-gene maxima: "
      f"{[round(max(peak[(g, v, 'untreated')] for v in V), 1) for g in ('IFNL1', 'IFNL2', 'IFNL3')]}"
      f"  IFNB1 {max(peak[('IFNB1', v, 'untreated')] for v in V):.1f}")
print(f"IFNA/IFNG max: {max(peak[(g, v, tr)] for g in ('IFNA', 'IFNG') for v in V for tr in ('untreated', 'Ruxo')):.2f} TPM")
print(f"conserved {len(conserved)} (lncRNA {len(conserved_lnc)}); metabolic genes "
      f"{ {k: len(x) for k, x in metab.items()} }; unnamed 726/727/640: "
      f"{len(unnamed(726))}/{len(unnamed(727))}/{len(unnamed(640))}")
print("gain: " + ", ".join(f"{v} {fits[v].slope:.2f} (R² {fits[v].rvalue ** 2:.2f})" for v in V))
below = [f"{k} {v} {h}h" for k, _ in A_SETS for v in V for h in (24, 72)
         if not floor_ok[(k, v, h)]]
print(f"below the {TPM_FLOOR:g} TPM floor (open marker, faded line): "
      + (", ".join(below) if below else "none"))
print("onset %: " + ", ".join(f"{v} {onset[v]:.1f}" for v in V))
