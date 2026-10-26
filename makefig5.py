"""Fig. S1: full-signal (inspiral-merger-ringdown) waveform reconstruction in L1.

Red: conditioned zero-noise boson-star injection. Blue: posterior median NRSur7dq4
reconstruction from the full-signal analysis, with the 90% (5-95%) band. The inset
zooms in on the merger.

Reads data/derived/full_signal_reconstruction.npz, which is produced by
pipeline/full_signal_reconstruction.py.
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils import config, plotting

IFO = "L1"
SCALE = 1e-21  # strain is plotted in units of 10^-21
XLIM_MS = (-350.0, 40.0)  # earlier times are affected by the taper applied at injection
ZOOM_MS = (-15.0, 8.0)    # time range of the inset
INSET = dict(width="25%", height="35%", loc="upper left")
INSET_CLEARANCE = 0.1   # axes fraction kept free below the inset for its tick labels
COLOR = plotting.COLORS["bh"]
INJECTION_COLOR = plotting.COLORS["bs"]


def draw(ax, r, lw_scale=1.0, label=True):
    """Draw the clean injection, the posterior median and the 90% (5-95%) band on ax."""
    t = 1e3 * r[f"{IFO}_t_rel"]
    ax.plot(t, r[f"{IFO}_clean"] / SCALE, lw=1.0 * lw_scale, color=INJECTION_COLOR,
            label="Clean injection" if label else None)
    ax.fill_between(t, r[f"{IFO}_q05"] / SCALE, r[f"{IFO}_q95"] / SCALE, color=COLOR, alpha=0.3, lw=0, zorder=1,
                    label="Posterior 90%" if label else None)
    ax.plot(t, r[f"{IFO}_q50"] / SCALE, lw=0.7 * lw_scale, color=COLOR, zorder=2,
            label="Posterior median" if label else None)


def visible_range(r, xlim):
    """Smallest and largest plotted strain (in units of SCALE) within xlim [ms]."""
    t = 1e3 * r[f"{IFO}_t_rel"]
    sel = (t >= xlim[0]) & (t <= xlim[1])
    values = np.concatenate([r[f"{IFO}_{k}"][sel] for k in ("clean", "q05", "q50", "q95")]) / SCALE
    return values.min(), values.max()


def main():
    """Draw the reconstruction with a zoomed inset and write figures/figS1_full_signal_reconstruction.pdf."""
    plotting.use_style()
    config.require(config.FULL_SIGNAL_RECONSTRUCTION, "run pipeline/full_signal_reconstruction.py first")
    r = np.load(config.FULL_SIGNAL_RECONSTRUCTION)

    fig, ax = plt.subplots(figsize=(plotting.COLUMN_WIDTH, 2.0))
    draw(ax, r)
    ax.set_xlim(*XLIM_MS)
    ax.set_xlabel(rf"$t - t_{{\mathrm{{peak}}}}^{{{IFO}}}$ [ms]")
    ax.set_ylabel(r"Strain [$10^{-21}$]")
    ax.grid(**plotting.GRID)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.17),
              columnspacing=1.0, handlelength=1.5)

    inset = inset_axes(ax, bbox_to_anchor=(0.03, 0.0, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=1.1,
                       **INSET)
    draw(inset, r, lw_scale=0.85, label=False)
    inset.set_xlim(*ZOOM_MS)
    zmin, zmax = visible_range(r, ZOOM_MS)
    pad = 0.08 * (zmax - zmin)
    inset.set_ylim(zmin - pad, zmax + pad)
    inset.tick_params(labelsize=5, length=1.5, pad=1.5)
    inset.grid(**plotting.GRID)

    # raise the top of the main axes until the waveform passes below the inset
    fig.canvas.draw()
    box = inset.get_position().transformed(fig.transFigure).transformed(ax.transAxes.inverted())
    under_lo, under_hi = visible_range(r, ax.transData.inverted().transform(
        ax.transAxes.transform([[box.x0 - 0.02, 0], [box.x1 + 0.02, 0]]))[:, 0])
    ymin, ymax = visible_range(r, XLIM_MS)
    bottom = ymin - 0.12 * (ymax - ymin)
    clear_below = box.y0 - INSET_CLEARANCE  # axes fraction the waveform under the inset must stay below
    top = max(ymax + 0.12 * (ymax - ymin), bottom + (under_hi - bottom) / clear_below)
    ax.set_ylim(bottom, top)

    plotting.save(fig, "figS1_full_signal_reconstruction.pdf")


if __name__ == "__main__":
    main()
