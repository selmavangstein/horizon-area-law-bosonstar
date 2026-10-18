"""Fig. 4: the area-law test.

Left:  posteriors on the total initial horizon area A_i (inspiral, solid) and the
       final area A_f (ringdown, dashed).
Right: posteriors on the fractional area change Delta A = (A_f - A_i) / A_f for
       all six inspiral-ringdown combinations. Delta A = 1 is the true value for
       the boson-star binary, whose initial horizon area is zero.
"""
import matplotlib.pyplot as plt
import numpy as np

from utils import area, config, plotting

AREA_SCALE = 1e5   # plot areas in units of 10^5 Msun^2
LINESTYLES = ("-", "--", ":")      # one per ringdown configuration in the right panel
SHADES = (0.55, 0.72, 0.88)        # colormap positions, one per ringdown configuration
CMAPS = {40: "Blues", 90: "Oranges"}
LEGEND_HEADROOM = 1.2   # top of each panel, in units of the highest curve
LEGEND_OVERLAP = 0.12   # axes fraction by which the legend may overlap the curves
LEGEND = dict(labelspacing=0.3, borderpad=0.4, handlelength=1.8, handletextpad=0.5)


def plot_kde(ax, draws, **kwargs):
    """Gaussian KDE with Silverman's bandwidth, evaluated over the 0.5-99.5% range plus 3 bandwidths."""
    draws = np.asarray(draws, dtype=float)
    bandwidth = 1.06 * np.std(draws, ddof=1) * draws.size ** (-1.0 / 5.0)
    lo, hi = np.percentile(draws, [0.5, 99.5])
    grid = np.linspace(lo - 3.0 * bandwidth, hi + 3.0 * bandwidth, 400)
    density = np.zeros_like(grid)
    for chunk in np.array_split(draws, max(1, draws.size // 20_000)):  # bounded memory
        density += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bandwidth) ** 2).sum(axis=1)
    density /= draws.size * bandwidth * np.sqrt(2.0 * np.pi)
    ax.plot(grid, density, linewidth=1.0, **kwargs)


def make_room_for_legend(ax, legend, side):
    """Widen the x-range on the legend's side until the curves at the legend's height overlap it
    by at most LEGEND_OVERLAP."""
    ax.figure.canvas.draw()
    box = legend.get_window_extent().transformed(ax.transAxes.inverted())
    y_bottom = ax.transData.inverted().transform(ax.transAxes.transform((0.0, box.y0)))[1]
    xs = np.concatenate([line.get_xdata()[line.get_ydata() >= y_bottom] for line in ax.get_lines()
                         if len(line.get_xdata()) > 2])
    x0, x1 = ax.get_xlim()
    if side == "right":
        frac = box.x0 + LEGEND_OVERLAP
        ax.set_xlim(x0, max(x1, x0 + (xs.max() - x0) / frac))
    else:
        frac = box.x1 - LEGEND_OVERLAP
        ax.set_xlim(min(x0, (xs.min() - frac * x1) / (1.0 - frac)), x1)


def main():
    """Draw the area and Delta A posteriors and write figures/fig4_area_law.pdf."""
    plotting.use_style()
    a_insp = {cut: area.inspiral_area(cut) for cut in config.INSPIRAL_CUTS_TM}
    a_ring = {(nm, st): area.ringdown_area(nm, st) for nm, st, _ in config.RINGDOWN_REFERENCE_RUNS}
    cmap = plt.get_cmap("tab10")

    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=(plotting.TEXT_WIDTH, 2.6))

    # left: areas
    for idx, (cut, a) in enumerate(a_insp.items()):
        plot_kde(ax_a, a / AREA_SCALE, linestyle="-", color=cmap(idx),
                 label=f"{plotting.INSPIRAL_SHORT[cut]} (inspiral)")
    for idx, (nm, st, label) in enumerate(config.RINGDOWN_REFERENCE_RUNS):
        plot_kde(ax_a, a_ring[(nm, st)] / AREA_SCALE, linestyle="--", color=cmap(idx + len(a_insp)),
                 label=f"{label} (ringdown)")
    ax_a.set_xlabel(r"Horizon area [$M_\odot^2 \times 10^5$]")
    legend_a = ax_a.legend(loc="upper right", **LEGEND)

    # right: fractional change
    for cut, rd in area.combinations():
        nm, st, label = rd
        idx = config.RINGDOWN_REFERENCE_RUNS.index(rd)
        draws = area.fractional_change_draws(a_insp[cut], a_ring[(nm, st)])
        plot_kde(ax_d, draws, linestyle=LINESTYLES[idx], color=plt.get_cmap(CMAPS[cut])(SHADES[idx]),
                 label=rf"{cut} $\rightarrow$ {st:g}, {nm} mode{'s' if nm > 1 else ''}")
    ax_d.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax_d.axvline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax_d.annotate("True value (BS binary)", xy=(1.0, 0.5), xycoords=("data", "axes fraction"),
                  xytext=(-3, 0), textcoords="offset points", ha="right", va="center", rotation=90,
                  fontsize=plotting.LEGEND_SIZE)
    ax_d.set_xlabel(r"$\Delta A = (A_{\mathrm{f}} - A_{\mathrm{i}})/(A_{\mathrm{f}})$")
    legend_d = ax_d.legend(loc="upper left", title=r"$t_<\,[T_M] \rightarrow t_>\,[T_{M_f}]$",
                           title_fontsize=plotting.LEGEND_SIZE, **LEGEND)

    for ax in (ax_a, ax_d):
        ax.set_ylabel("Posterior density")
        peak = max(line.get_ydata().max() for line in ax.get_lines() if len(line.get_ydata()) > 2)
        ax.set_ylim(0.0, LEGEND_HEADROOM * peak)  # room for the legend above the curves
        ax.grid(**plotting.GRID)
    fig.tight_layout()
    make_room_for_legend(ax_a, legend_a, "right")
    make_room_for_legend(ax_d, legend_d, "left")
    plotting.save(fig, "fig4_area_law.pdf")


if __name__ == "__main__":
    main()
