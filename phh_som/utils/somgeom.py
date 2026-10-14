import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Wedge

from phh_som.utils.pipeline import CLIP_Z

FAN_R0 = 0.335

TIME_COLORS = {"24": "#2166ac", "72": "#b2182b"}

COND_LABEL = {"untreated24": "UT 24", "untreated72": "UT 72",
              "Ruxo24": "Ruxo 24", "Ruxo72": "Ruxo 72"}


def lighten(color, f):
    r, g, b = to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def hex_ring(x, y, r=1.0):
    ang = np.pi / 2 + np.arange(6) * np.pi / 3
    return [(x + r * np.cos(a), y + r * np.sin(a)) for a in ang]


def cond_color(rest):
    base = TIME_COLORS["24" if rest.endswith("24") else "72"]
    return lighten(base, 0.50) if "Ruxo" in rest else base


def fan_glyph(ax, x, y, weights, k, conds, colors, r0=FAN_R0, lw=0.35,
              ring=True, fc_fn=None):
    if fc_fn is None:
        def fc_fn(i, virus, rest):
            return cond_color(rest)
    n = len(weights)
    for i, w in enumerate(weights):
        w = float(np.clip(w, -CLIP_Z, CLIP_Z))
        a2, a1 = 90 - i * 360 / n, 90 - (i + 1) * 360 / n
        virus, rest = conds[i].split("_", 1)
        fc = fc_fn(i, virus, rest)
        r = r0 + k * w
        if w >= 0:
            ax.add_patch(Wedge((x, y), r, a1, a2, width=max(r - r0, 1e-3),
                               fc=fc, ec="white", lw=lw, zorder=3))
        else:
            ax.add_patch(Wedge((x, y), r0, a1, a2, width=max(r0 - r, 1e-3),
                               fc=fc, ec="white", lw=lw, alpha=0.40, zorder=3,
                               hatch="////"))
    if ring:
        ax.add_patch(Circle((x, y), r0, fc="none", ec="0.30", lw=0.45,
                            ls=(0, (1.3, 1.3)), zorder=4))
        ax.plot([x, x], [y - r0 - 0.5 * k, y + r0 + 0.5 * k], color="0.30",
                lw=0.45, zorder=4)


def fan_key(ax, k, lo, hi, conds, colors, origin=(0.0, 0.0), magnify=1.0):
    ax.set_aspect("equal")
    ax.axis("off")
    r0 = FAN_R0 * magnify
    k = k * magnify
    R = r0 + k * hi
    ox, oy = origin
    n = len(conds)
    fan_glyph(ax, ox, oy, [hi] * n, k=k, conds=conds, colors=colors,
              r0=r0, lw=0.5)
    per_virus = [c.split("_")[0] for c in conds].count(conds[0].split("_")[0])
    if n <= 8:
        for i, cond in enumerate(conds):
            mid = np.deg2rad(90 - (i + 0.5) * 360 / n)
            virus, rest = cond.split("_", 1)
            ax.text(ox + (R + 0.09 * magnify) * np.cos(mid),
                    oy + (R + 0.09 * magnify) * np.sin(mid),
                    COND_LABEL[rest], fontsize=5.2, color="0.40",
                    va="center",
                    ha="left" if np.cos(mid) > 0.05 else
                       ("right" if np.cos(mid) < -0.05 else "center"))
    for vi in range(0, n, per_virus):
        virus = conds[vi].split("_")[0]
        mid = np.deg2rad(90 - (vi + per_virus / 2) * 360 / n)
        ax.text(ox + (R + 0.40 * magnify) * np.cos(mid),
                oy + (R + 0.40 * magnify) * np.sin(mid),
                virus, fontsize=6.5, fontweight="bold", color=colors[virus],
                ha="center", va="center")
    y = oy - (R + 0.30 * magnify)
    x0, x1 = ox + r0 + k * lo, ox + r0 + k * hi
    if n > 8:
        dx = (ox - 1.2 * magnify) - x1
        x0, x1 = x0 + dx, x1 + dx
    ax.plot([x0, x1], [y, y], color="0.35", lw=0.6)
    ends = [(lo, x0, "right", y - 0.035 * magnify, "top"),
            (hi, x1, "left", y - 0.035 * magnify, "top")]
    if n > 8:
        ends[1] = (hi, x1, "left", y + 0.10 * magnify, "bottom")
    for v, x, ha, ty, va in ends:
        ax.plot([x, x], [y, y + 0.085 * magnify], color="0.35", lw=0.5)
        sat = (v <= -CLIP_Z and ha == "right") or (v >= CLIP_Z and ha == "left")
        label = f"{v:+.1f}".replace("-", "−")
        ax.text(x, ty, ("≤ " if v < 0 else "≥ ") * sat + label, ha=ha,
                va=va, fontsize=4.8, color="0.35")
    cx_scale = (x0 + x1) / 2
    ax.text(cx_scale, y - 0.30 * magnify,
            "codebook weight (z)" + ", winsorized" * (lo <= -CLIP_Z or hi >= CLIP_Z), ha="center",
            va="top", fontsize=5.5, color="0.35")
    handles = [Patch(fc=cond_color(rest), ec="none", label=COND_LABEL[rest])
               for rest in ("untreated24", "Ruxo24", "untreated72", "Ruxo72")
               if any(c.split("_", 1)[1] == rest for c in conds)]
    handles += [
        Patch(fc="white", ec="0.40", lw=0.5, hatch="////",
              label="negative (inward)"),
        Line2D([], [], color="0.30", lw=0.8, ls=(0, (1.3, 1.3)),
               label="zero ring"),
    ]
    if n > 8:
        ax.legend(handles=handles, loc="center left", fontsize=5.2,
                  frameon=False, handlelength=1.1, handletextpad=0.5,
                  labelspacing=0.45, borderaxespad=0.0,
                  bbox_to_anchor=(ox + R + 0.9 * magnify, oy),
                  bbox_transform=ax.transData)
    else:
        ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=5.0,
                  frameon=False, handlelength=1.1, handletextpad=0.5,
                  labelspacing=0.3, columnspacing=0.9, borderaxespad=0.0,
                  bbox_to_anchor=(ox, y - 0.50 * magnify),
                  bbox_transform=ax.transData)
