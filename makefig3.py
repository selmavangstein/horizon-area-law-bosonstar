"""Fig. 3: QNM amplitudes vs ringdown start time t_>.

Top:    fundamental-mode amplitude A_220 for the 220 model and the 220+221 model.
Bottom: first-overtone amplitude A_221 (220+221 model only).
Points show the posterior median, thick bars the 25-75% and thin bars (with caps)
the 5-95% credible intervals. The grey bands show the amplitude expected from
exponential decay of a reference fit (25-75% and 5-95%), using each posterior
sample's damping rate. The reference fits (t_> = 10.5 T_Mf for 220, 6 T_Mf for 221)
are marked by the dashed vertical lines.

Start times are plotted in ms (bottom axis) with T_Mf on the top axis.
"""
import matplotlib.pyplot as plt
import numpy as np

from utils import config, data, plotting

SCALE = 1e-21  # amplitudes are plotted in units of 10^-21
LABELS = {1: "220 model", 2: "220+221 model"}
COLORS = {1: plotting.COLORS["mode1"], 2: plotting.COLORS["mode2"]}
# reference fit for the expected decay of each mode: (num_modes, t_> in T_Mf)
DECAY_REFERENCE = {0: (1, 10.5), 1: (2, 6.0)}
REFERENCE_LABEL = {0: ("220 ref.", COLORS[1]), 1: ("221 ref.", COLORS[2])}


def to_ms(t_tmf):
    """Convert times in units of T_Mf to milliseconds."""
    return np.asarray(t_tmf) * config.T_MF * 1e3


def amplitude_summary(num_modes, mode_index):
    """Percentiles (5, 25, 50, 75, 95) of one mode amplitude for every start time of a model.

    Returns an array with one row per start time: [start, p05, p25, p50, p75, p95].
    """
    rows = []
    for nm, start in data.ringdown_runs():
        if nm != num_modes:
            continue
        a = data.ringdown_posterior(nm, start)["a"][:, mode_index] / SCALE
        rows.append([start, *np.percentile(a, [5, 25, 50, 75, 95])])
    return np.array(rows)


def expected_decay(mode_index, t_tmf):
    """Percentiles (5, 25, 75, 95) of the amplitude expected at times t_tmf.

    Each sample of the reference fit is propagated with its own damping rate,
    A(t) = A_0 exp(-gamma (t - t_ref)), and the percentiles are taken over samples.
    """
    num_modes, start = DECAY_REFERENCE[mode_index]
    post = data.ringdown_posterior(num_modes, start)
    a0 = post["a"][:, mode_index] / SCALE
    gamma = post["g"][:, mode_index]  # damping rate [1/s]
    curves = a0[:, None] * np.exp(-gamma[:, None] * (t_tmf[None, :] - start) * config.T_MF)
    return np.percentile(curves, [5, 25, 75, 95], axis=0)


def draw(ax, num_modes, mode_index):
    """Draw the median and the 25-75% and 5-95% intervals of one amplitude; return the start times."""
    s = amplitude_summary(num_modes, mode_index)
    starts, lo, p25, med, p75, hi = s.T
    c = COLORS[num_modes]
    ax.errorbar(to_ms(starts), med, yerr=[med - lo, hi - med], fmt="o", color=c, label=LABELS[num_modes],
                linewidth=0.7, markersize=3, capsize=1.5, capthick=0.7, alpha=0.7)
    for x, y0, y1 in zip(to_ms(starts), p25, p75):
        ax.plot([x, x], [y0, y1], color=c, linewidth=2, alpha=0.8)
    return starts


def draw_expected_decay(ax, mode_index, starts, label):
    """Shade the expected-decay band of one mode across the range of start times."""
    t = np.linspace(starts.min(), starts.max(), 100)
    p05, p25, p75, p95 = expected_decay(mode_index, t)
    ax.fill_between(to_ms(t), p05, p95, color="grey", alpha=0.2)
    ax.fill_between(to_ms(t), p25, p75, color="darkgrey", alpha=0.2, label=label)


def mark_reference(ax, mode_index):
    """Mark and label the start time of the reference fit with a dashed vertical line."""
    start = DECAY_REFERENCE[mode_index][1]
    text, color = REFERENCE_LABEL[mode_index]
    ax.axvline(to_ms(start), color=color, ls="--", lw=0.8, alpha=0.5, zorder=0)
    ax.annotate(text, xy=(to_ms(start), 0.62), xycoords=("data", "axes fraction"), ha="left", va="bottom",
                rotation=90, rotation_mode="anchor", color=color, fontsize=plotting.LEGEND_SIZE)


def main():
    """Draw both panels and write figures/fig3_amplitude_vs_start_time.pdf."""
    plotting.use_style()
    fig, axs = plt.subplots(2, 1, figsize=(plotting.TEXT_WIDTH, 3.4), sharex=True)

    # top: A_220 from both models
    ax = axs[0]
    starts = draw(ax, 1, 0)
    draw_expected_decay(ax, 0, starts, "Expected decay (220)")
    draw(ax, 2, 0)
    ax.set_ylabel(r"$A_{220}$ [$10^{-21}$]")
    mark_reference(ax, 0)
    top = ax.secondary_xaxis("top", functions=(lambda ms: ms * 1e-3 / config.T_MF,
                                               lambda tm: tm * config.T_MF * 1e3))
    top.set_xlabel(r"$t_> = t - t_{\mathrm{peak}}$ [$T_{M_f}$]")

    # bottom: A_221 from the two-mode model
    ax = axs[1]
    starts = draw(ax, 2, 1)
    draw_expected_decay(ax, 1, starts, "Expected decay (221)")
    ax.set_ylabel(r"$A_{221}$ [$10^{-21}$]")
    ax.set_xlabel(r"$t_> = t - t_{\mathrm{peak}}$ [ms]")
    mark_reference(ax, 1)

    for ax in axs:
        ax.legend(loc="upper right")
        ax.grid(**plotting.GRID)
        ax.set_ylim(0, None)

    fig.tight_layout(h_pad=0.4)
    plotting.save(fig, "fig3_amplitude_vs_start_time.pdf")


if __name__ == "__main__":
    main()
