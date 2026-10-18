"""Component-mass plane with KDE marginals, shared by Fig. 2 and Fig. S2.

Scatter of the samples with seaborn KDE marginals, grey lines of constant chirp
mass, a grey m_2 > m_1 region and a yellow marker at the injected masses. Drawn
at single-column size in the shared style of utils/plotting.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from . import config, plotting

FIGSIZE = (plotting.COLUMN_WIDTH, 3.1)  # in
# matplotlib/seaborn settings of the scatter, the marginal KDEs and the injection marker
SCATTER = dict(s=1.0, alpha=0.5, edgecolors="none", rasterized=True, zorder=2)
KDE = dict(fill=False, lw=1.0, bw_adjust=0.9, cut=3.0, legend=False)
INJECTION = dict(s=40, color="yellow", edgecolors="black", linewidths=0.8, zorder=3)
INJECTED_COMPONENT_MASS = config.TOTAL_MASS / 2.0


def use_style():
    """Apply the shared figure style (same as plotting.use_style)."""
    plotting.use_style()


def chirp_mass(m1, m2):
    """Chirp mass (m_1 m_2)^(3/5) / (m_1 + m_2)^(1/5)."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def m2_at_chirp_mass(m1, mc):
    """m_2 <= m_1 with chirp mass mc along a grid of m_1 (NaN where impossible)."""
    m1 = np.asarray(m1, dtype=float)
    # chirp mass grows monotonically with m_2, so bisect in m_2 in (0, m_1)
    lo, hi = np.zeros_like(m1), m1.copy()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        below = chirp_mass(m1, mid) < mc
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    m2 = 0.5 * (lo + hi)
    m2[chirp_mass(m1, m1) <= mc] = np.nan
    return m2


def padded_limits(values, pad_fraction=0.08):
    """Axis limits spanning values, widened by pad_fraction of the range on each side."""
    values = np.asarray(values, dtype=float)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    pad = pad_fraction * (vmax - vmin)
    return vmin - pad, vmax + pad


def nice_ticks(vmin, vmax, nbins=6):
    """Round tick positions that lie inside [vmin, vmax]."""
    ticks = MaxNLocator(nbins=nbins, min_n_ticks=5).tick_values(vmin, vmax)
    return ticks[(ticks >= vmin) & (ticks <= vmax)]


def mass_plane(samples, chirp_levels, legend_extra=(), legend_loc="upper right"):
    """Joint m_1-m_2 scatter with marginal KDEs.

    samples:      list of (label, colour, m1, m2), drawn in order (later on top).
    chirp_levels: chirp masses [Msun] at which to draw grey reference lines.
    legend_extra: additional legend handles, placed before the injection marker.
    legend_loc:   matplotlib legend location in the joint panel.
    Returns (fig, ax_joint, ax_top, ax_right).
    """
    all_m1 = np.concatenate([s[2] for s in samples] + [[INJECTED_COMPONENT_MASS]])
    all_m2 = np.concatenate([s[3] for s in samples] + [[INJECTED_COMPONENT_MASS]])
    xlim, ylim = padded_limits(all_m1), padded_limits(all_m2)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 2, width_ratios=(4.2, 1.3), height_ratios=(1.3, 4.2), wspace=0.03, hspace=0.03)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    # excluded m_2 > m_1 region, the m_1 = m_2 line and lines of constant chirp mass
    diag = min(xlim[1], ylim[1])
    ax.fill([xlim[0], xlim[0], diag], [xlim[0], diag, diag], color="0.75", zorder=0)
    ax.plot([xlim[0], diag], [xlim[0], diag], color="black", lw=0.8, zorder=1)
    m1_grid = np.linspace(*xlim, 1200)
    for mc in chirp_levels:
        ax.plot(m1_grid, m2_at_chirp_mass(m1_grid, mc), color="0.50", lw=0.6, zorder=1.1)

    # samples and their marginal KDEs
    for _, color, m1, m2 in samples:
        ax.scatter(m1, m2, color=color, **SCATTER)
        sns.kdeplot(x=np.asarray(m1), color=color, ax=ax_top, **KDE)
        sns.kdeplot(y=np.asarray(m2), color=color, ax=ax_right, **KDE)
    ax.scatter([INJECTED_COMPONENT_MASS], [INJECTED_COMPONENT_MASS], **INJECTION)

    handles = [Line2D([0], [0], marker="o", ls="None", markersize=4, markerfacecolor=c, markeredgecolor=c,
                      label=label) for label, c, _, _ in samples]
    handles += list(legend_extra)
    handles.append(Line2D([0], [0], marker="o", ls="None", markersize=5, markerfacecolor="yellow",
                          markeredgecolor="black", markeredgewidth=0.8, label="Injected value"))
    ax.legend(handles=handles, loc=legend_loc, frameon=True, framealpha=0.95)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(nice_ticks(*xlim))
    ax.set_yticks(nice_ticks(*ylim))
    ax.set_xlabel(r"$m_1\ [M_\odot]$")
    ax.set_ylabel(r"$m_2\ [M_\odot]$")

    ax_top.set_xlim(xlim)
    ax_top.set_ylim(bottom=0.0)
    ax_top.tick_params(axis="both", bottom=False, labelbottom=False, left=False, labelleft=False)
    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_right.set_xlim(left=0.0)
    ax_right.set_xlabel("")
    ax_right.set_ylabel("")
    ax_right.tick_params(axis="both", bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.grid(**plotting.GRID)
    ax_top.grid(axis="x", **plotting.GRID)    # marginals: grid along the shared mass axis only
    ax_right.grid(axis="y", **plotting.GRID)
    return fig, ax, ax_top, ax_right
