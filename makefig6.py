"""Fig. S2: component-mass posterior from the full-signal (inspiral-merger-ringdown) analysis.

Drawn like Fig. 2: component masses (scatter) with KDE marginals, lines of
constant chirp mass (grey, 24-30 Msun) and the lower total-mass prior bound
M = 58 Msun (dashed). The posterior is pushed against that bound.

Component masses are derived from the sampled total mass M and mass ratio
q = m_2/m_1 <= 1 as m_1 = M/(1+q), m_2 = qM (detector frame; see utils/data.py).
"""
import numpy as np
from matplotlib.lines import Line2D

from utils import data, massplane, plotting

TOTAL_MASS_PRIOR_MIN = 58.0   # Msun, lower prior bound of the TDInf runs (see run_tdinf.sh)
CHIRP_MASS_LEVELS = np.linspace(24.0, 30.0, 6)  # Msun, grey reference lines
BOUND = dict(color="black", lw=0.8, ls="--")
LEGEND_SPACE = 3.0  # Msun added below the m_2 range, for the legend


def main():
    """Plot the full-signal posterior in the m_1-m_2 plane and write figures/figS2_full_signal_masses.pdf."""
    massplane.use_style()
    s = data.full_signal_posterior()
    bound = Line2D([0], [0], label=rf"Prior bound $M = {TOTAL_MASS_PRIOR_MIN:g}\,M_\odot$", **BOUND)
    fig, ax, *_ = massplane.mass_plane(
        [("Full signal", plotting.COLORS["full"], s["mass_1"].to_numpy(), s["mass_2"].to_numpy())],
        CHIRP_MASS_LEVELS, legend_extra=[bound],
        legend_loc="lower left")  # upper right would hide the injection
    # the prior bound M = 58 Msun is the line m_2 = 58 - m_1
    m1 = np.linspace(TOTAL_MASS_PRIOR_MIN / 2, ax.get_xlim()[1], 200)  # m_2 <= m_1 only
    ax.plot(m1, TOTAL_MASS_PRIOR_MIN - m1, zorder=1.2, **BOUND)
    ax.set_ylim(bottom=ax.get_ylim()[0] - LEGEND_SPACE)
    plotting.save(fig, "figS2_full_signal_masses.pdf")


if __name__ == "__main__":
    main()
