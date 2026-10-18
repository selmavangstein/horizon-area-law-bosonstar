"""Fig. 1: the boson-star injection.

Top:    source-frame h_+ of the boson-star (GRChombo A17) and binary-BH (NRSur7dq4,
        same total mass, equal mass, non-spinning) waveforms, aligned at peak
        amplitude, at the injected inclination and distance.
Bottom: whitened L1 data (grey) with the whitened noiseless boson-star signal on
        top, and the inspiral end times t_< and ringdown start times t_>.

Both panels share a time axis relative to the peak: milliseconds (bottom) and
units of the remnant time scale T_Mf (top).

Needs external/: GRChombo waveform, SEOBNRv4ROM_v3.0.hdf5, NRSur7dq4_v1.0.h5.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils import config, data, plotting, waveforms

# Plot settings
IFO = "L1"                 # the louder detector
WINDOW_MS = (-170.0, 30.0)  # ~0.2 s of data
WHITEN_BAND = (20.0, 512.0)  # Hz, band kept after whitening
NOISE_ONLY_S = (-2.3, -1.0)  # s relative to peak; before the (tapered) signal starts
LEGEND_HEADROOM = 1.6       # top of the bottom panel, in units of the largest |data|; room for the legend
TOP_HEADROOM = 1.35         # top of the top panel, in units of the largest |h_+|


def to_ms(t):
    """Convert seconds to milliseconds."""
    return 1e3 * t


def main():
    """Draw both panels and write figures/fig1_injection.pdf."""
    plotting.use_style()
    meta = data.metadata()
    inclination = float(meta["inclination"])

    # --- top: source-frame waveforms -------------------------------------------
    t_bs, hp_bs, hc_bs = waveforms.boson_star_hphc(inclination)
    t_bh, hp_bh, hc_bh = waveforms.nrsur_hphc(inclination)
    # rotate the BH waveform so both have the same GW phase at peak amplitude
    z_bs, z_bh = hp_bs - 1j * hc_bs, hp_bh - 1j * hc_bh
    dphi = np.angle(z_bs[np.argmin(np.abs(t_bs))]) - np.angle(z_bh[np.argmin(np.abs(t_bh))])
    hp_bh = np.real(z_bh * np.exp(1j * dphi))

    # --- bottom: whitened detector data ------------------------------------------
    fs = float(meta["sampling_frequency"])
    t_gps, d = data.strain(IFO)
    _, s = data.strain(IFO, clean=True)
    f_psd, psd = data.psd(IFO)
    w_data = waveforms.whiten(d, f_psd, psd, fs, WHITEN_BAND)
    w_signal = waveforms.whiten(s, f_psd, psd, fs, WHITEN_BAND)
    t_rel = t_gps - float(meta[f"{IFO.lower()}_arrival_time"])  # 0 = geocentre peak time at L1
    # express both in units of the whitened noise standard deviation, measured in a
    # signal-free stretch of the zero-padded segment
    sigma = np.std(w_data[(t_rel > NOISE_ONLY_S[0]) & (t_rel < NOISE_ONLY_S[1])])
    w_data, w_signal = w_data / sigma, w_signal / sigma

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(plotting.COLUMN_WIDTH, 2.9), sharex=True,
                                   gridspec_kw={"hspace": 0.06, "height_ratios": [1, 1.15]})

    # top panel: source-frame h_+ of both binaries
    scale = 1e-21
    ax1.plot(to_ms(t_bh), hp_bh / scale, color=plotting.COLORS["bh"], lw=0.8, label="Binary BH (NRSur7dq4)")
    ax1.plot(to_ms(t_bs), hp_bs / scale, color=plotting.COLORS["bs"], lw=0.8, label="Binary BS (GRChombo)")
    ax1.set_ylabel(r"$h_+$ [$\times 10^{-21}$]")
    ax1.legend(loc="upper left")
    hmax = max(np.max(np.abs(hp_bh)), np.max(np.abs(hp_bs))) / scale
    ax1.set_ylim(-1.1 * hmax, TOP_HEADROOM * hmax)

    # bottom panel: whitened data and signal, with the analysis cut times
    sel = (to_ms(t_rel) >= WINDOW_MS[0] - 5) & (to_ms(t_rel) <= WINDOW_MS[1] + 5)
    ax2.plot(to_ms(t_rel[sel]), w_data[sel], color=plotting.COLORS["data"], lw=0.5, label=f"Whitened {IFO} data")
    ax2.plot(to_ms(t_rel[sel]), w_signal[sel], color=plotting.COLORS["bs"], lw=0.8, label="Whitened BS signal")

    for cut in config.INSPIRAL_CUTS_TM:
        ax2.axvline(-to_ms(cut * config.T_M), color=plotting.COLORS["tlt"], ls="--", lw=0.8)
    for start in config.RINGDOWN_STARTS_TMF:
        ax2.axvline(to_ms(start * config.T_MF), color=plotting.COLORS["tgt"], ls=":", lw=1.0)
    handles, _ = ax2.get_legend_handles_labels()
    handles += [
        Line2D([], [], color=plotting.COLORS["tlt"], ls="--", lw=0.8,
               label=r"$t_<$ ($%s\,T_M$)" % r",\,".join(f"-{c}" for c in config.INSPIRAL_CUTS_TM)),
        Line2D([], [], color=plotting.COLORS["tgt"], ls=":", lw=1.0,
               label=r"$t_>$ ($%s\,T_{M_f}$)" % r",\,".join(f"{s:g}" for s in config.RINGDOWN_STARTS_TMF)),
    ]
    ax2.legend(handles=handles, loc="upper left", ncol=2, columnspacing=1.0)
    ax2.set_ylabel(r"Whitened strain [$\sigma$]")
    ax2.set_xlabel(r"$t - t_{\mathrm{peak}}$ [ms]")
    ax2.set_xlim(*WINDOW_MS)
    ylim = np.max(np.abs(w_data[(to_ms(t_rel) >= WINDOW_MS[0]) & (to_ms(t_rel) <= WINDOW_MS[1])]))
    ax2.set_ylim(-1.15 * ylim, LEGEND_HEADROOM * ylim)

    # shared time axis: ms below, T_Mf on top
    top = ax1.secondary_xaxis("top", functions=(lambda ms: ms * 1e-3 / config.T_MF,
                                                lambda tm: tm * config.T_MF * 1e3))
    top.set_xlabel(r"$t - t_{\mathrm{peak}}$ [$T_{M_f}$]")
    for ax in (ax1, ax2):
        ax.grid(**plotting.GRID)

    plotting.save(fig, "fig1_injection.pdf")


if __name__ == "__main__":
    main()
