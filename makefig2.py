"""Fig. 2: component-mass posteriors of the inspiral analyses, t_< = 40 T_M and 90 T_M.

The posteriors are scatter plots (not 2D KDEs), so their fine structure is not
smoothed away: they form narrow stripes close to lines of constant chirp mass
(grey; 26-32 Msun in steps of 1.2 Msun). The one-dimensional marginals are KDEs.
The grey region (m_2 > m_1) is excluded by convention, and the yellow marker is
the injected equal-mass boson-star binary.
"""
import numpy as np

from utils import config, data, massplane, plotting

CHIRP_MASS_LEVELS = np.linspace(26.0, 32.0, 6)  # Msun, grey reference lines


def main():
    """Plot both inspiral posteriors in the m_1-m_2 plane and write figures/fig2_inspiral_masses.pdf."""
    massplane.use_style()
    samples = []
    for cut in config.INSPIRAL_CUTS_TM:
        s = data.inspiral_posterior(cut)
        samples.append((plotting.INSPIRAL_SHORT[cut], plotting.COLORS[f"insp{cut}"],
                        s["mass_1"].to_numpy(), s["mass_2"].to_numpy()))
    fig, *_ = massplane.mass_plane(samples, CHIRP_MASS_LEVELS)
    plotting.save(fig, "fig2_inspiral_masses.pdf")


if __name__ == "__main__":
    main()
