"""Kerr horizon areas and the area-law test statistics."""
from statistics import NormalDist

import numpy as np

from . import config, data

_MAX_SPIN = 0.999999  # clip |chi| just below 1 so the square root stays real


def kerr_area(mass, chi):
    """Horizon area A = 8 pi M^2 (1 + sqrt(1 - chi^2)) in units of Msun^2 (G = c = 1)."""
    chi = np.clip(np.asarray(chi, dtype=float), -_MAX_SPIN, _MAX_SPIN)
    return 8.0 * np.pi * np.asarray(mass, dtype=float) ** 2 * (1.0 + np.sqrt(1.0 - chi**2))


def inspiral_area(cut_tm):
    """Total initial area A_1 + A_2 from the inspiral posterior with t_< = cut_tm T_M."""
    s = data.inspiral_posterior(cut_tm)
    return kerr_area(s["mass_1"], s["spin1_magnitude"]) + kerr_area(s["mass_2"], s["spin2_magnitude"])


def ringdown_area(num_modes, start_tmf):
    """Final area A_f from the ringdown fit with num_modes QNMs starting at t_> = start_tmf T_Mf."""
    post = data.ringdown_posterior(num_modes, start_tmf)
    return kerr_area(post["m"], post["chi"])


def fractional_change_draws(a_i, a_f, n_draws=200_000, seed=1234):
    """Monte-Carlo draws of Delta A = (A_f - A_i) / A_f from independent posteriors.

    Inspiral and ringdown samples are drawn with replacement and paired at random.
    Used for the Delta A densities and percentiles; P(Delta A > 0) itself is exact
    (see prob_increase).
    """
    rng = np.random.default_rng(seed)
    a_i = np.asarray(a_i)[rng.integers(0, len(a_i), n_draws)]
    a_f = np.asarray(a_f)[rng.integers(0, len(a_f), n_draws)]
    return (a_f - a_i) / a_f


def prob_increase(a_i, a_f):
    """P(A_f > A_i), evaluated exactly over all pairs of inspiral and ringdown samples."""
    # for each A_f, count the A_i below it by binary search in the sorted A_i
    a_i = np.sort(np.asarray(a_i, dtype=float))
    counts = np.searchsorted(a_i, np.asarray(a_f, dtype=float), side="left")
    return float(counts.sum()) / (a_i.size * len(a_f))


def sigma_from_prob(p):
    """One-sided Gaussian-equivalent significance (in sigma) of a probability p."""
    return NormalDist().inv_cdf(float(np.clip(p, 1e-12, 1 - 1e-12)))


def combinations():
    """All (inspiral cut, ringdown config) pairs used in the area-law test."""
    return [(cut, rd) for cut in config.INSPIRAL_CUTS_TM for rd in config.RINGDOWN_REFERENCE_RUNS]
