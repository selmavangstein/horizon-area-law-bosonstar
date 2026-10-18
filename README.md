# On the Interpretation of Tests of the Horizon Area Law

Code and data for

> S. Vangstein and C. J. Moore, *On the Interpretation of Tests of the Horizon Area Law* (2026), submitted to Phys. Rev. Lett.

## About the paper

Gravitational-wave tests of Hawking's horizon area law, most recently with GW250114,
infer horizon areas from models that assume the objects are Kerr black holes in
vacuum GR. We test how much a passed area-law test actually tells us.

To do this we inject the numerical-relativity signal of a merging **boson-star**
binary into simulated LIGO noise at GW250114-like parameters. Boson stars are
horizonless, so the initial horizon area is zero. The injection uses the GRChombo
`A17` simulation, scaled to M = 71.5 M☉ at 425 Mpc, giving a network SNR of 80.

We then repeat the GW250114 area-law analysis on this signal:
- **Inspiral:** a TDInf / NRSur7dq4 analysis of the data before t_< = t_peak − 40 or 90 T_M.
- **Ringdown:** a Kerr QNM analysis with `ringdown`, starting at t_> = t_peak + 6 or 10.5 T_Mf.

Despite there being no horizons before merger, the signal passes the test, with
P(ΔA > 0) up to ≈ 0.92. A full inspiral–merger–ringdown analysis is shown in the supplement.

## Repository layout

```
makefig1.py … makefig6.py   reproduce Figs. 1–4 and S1–S2 (outputs in figures/)
                            drawn at printed size: include with width=\columnwidth
                            (Figs. 1, 2, S1, S2) or width=\textwidth (Figs. 3, 4)
figures/                    the figures, as PDFs
data/                       injection, posterior samples, ringdown posteriors (see data/README.md)
utils/                      shared paths and constants, loaders, waveforms, area-law maths, plot style
pipeline/                   the full analysis, stage by stage (see below)
external/                   separately downloaded files (see external/README.md)
```

## Installation

You need Python 3.11. Then either use conda:

```bash
conda env create -f environment.yml
conda activate horizon-area-law
```

or pip:

```bash
pip install -r requirements.txt
```

Both install the paper's versions of `lalsuite`, `bilby`, `ringdown`, `tdinf`,
`jax`/`numpyro` and the scientific Python stack. `ringdown` and `tdinf` are pinned
to the exact git commits used.

## Reproducing the figures

All posterior samples are included in `data/`, so the figures can be remade without
rerunning any of the analyses.

| Figure | Command | Inputs | Runtime |
|---|---|---|---|
| Fig. 1 (injection: h₊ for BS vs BH; whitened data with t_< and t_>) | `python makefig1.py` | `data/injection/` + **external files** (see below) | ~1 min |
| Fig. 2 (inspiral component-mass posteriors) | `python makefig2.py` | `data/posteriors/` | seconds |
| Fig. 3 (QNM amplitudes vs ringdown start time) | `python makefig3.py` | `data/ringdown/` | seconds |
| Fig. 4 (horizon areas and ΔA) | `python makefig4.py` | `data/posteriors/`, `data/ringdown/` | ~10 s |
| Fig. S1 (full-signal waveform reconstruction) | `python makefig5.py` | `data/derived/full_signal_reconstruction.npz` | seconds |
| Fig. S2 (full-signal component masses) | `python makefig6.py` | `data/posteriors/` | seconds |

The area-law significances quoted in the paper come from

```bash
python pipeline/area_law_test.py
```

This writes `data/derived/area_law_significance.csv`.

Component masses are not sampled directly. Every analysis samples the detector-frame
total mass M and the mass ratio q = m₂/m₁ ≤ 1, and the figures use
m₁ = M/(1+q) and m₂ = qM (`utils/data.py`).

### External downloads (Fig. 1 and the full pipeline)

> **To run the full pipeline (and `makefig1.py`), download the SEOBNRv4 ROM data file
> `SEOBNRv4ROM_v3.0.hdf5` and the NRSur7dq4 data file `NRSur7dq4_v1.0.h5` from the
> [LALSuite waveform data Zenodo record (10.5281/zenodo.14999310)](https://zenodo.org/records/14999310)
> and place them in `external/`.**

```bash
wget -O external/SEOBNRv4ROM_v3.0.hdf5 "https://zenodo.org/records/14999310/files/SEOBNRv4ROM_v3.0.hdf5?download=1"
wget -O external/NRSur7dq4_v1.0.h5    "https://zenodo.org/records/14999310/files/NRSur7dq4_v1.0.h5?download=1"
```

The two files are needed for different reasons:
- **SEOBNRv4ROM** is only needed because LALSimulation's `NR_hdf5` approximant, used to
  generate the boson-star waveform, calls it internally.
- **NRSur7dq4** is the binary-BH model used in all the inspiral and full-signal analyses.

The boson-star waveform `GRChombo_BBSsol02_A17A17q100d17p000_Res40.h5` comes from the
simulation catalogue of [Evstafyeva et al. (2024)](https://doi.org/10.1103/PhysRevLett.133.131401)
and can be found at the accompanying [GitHub respository](https://github.com/tamaraevst/Boson-star-waveforms)

## The analysis pipeline

The scripts in `pipeline/` regenerate everything in `data/` from scratch. **None of
this is needed to make the figures.** The expensive stages, the TDInf runs and the
ringdown sweep, are included so the analysis can be audited and rerun, but their
outputs are already shipped.

| Stage | Command | Output | Cost |
|---|---|---|---|
| 1. Injection | `python pipeline/injection.py` | `output/injection/` | ~1 min |
| 2. Inspiral and full-signal PE (TDInf + emcee, NRSur7dq4) | `bash pipeline/inspiral/run_tdinf.sh {inspiral_40,inspiral_90,full_signal}` | `output/tdinf/<run>_results.h5` | **days per run** (64 walkers × 6×10⁵ steps) |
| 2a. Thin the chains | `python pipeline/inspiral/extract_samples.py output/tdinf/<run>_results.h5 data/posteriors/raw/<run>_samples_raw.txt` | raw samples | seconds |
| 2b. Convert to physical parameters | `python pipeline/decode_tdinf_samples.py` | `data/posteriors/*_samples.dat` | seconds |
| 3. Ringdown start-time sweep | `python pipeline/ringdown_sweep.py`, then `python pipeline/export_ringdown_posteriors.py output/ringdown` | `data/ringdown/ringdown_posteriors.h5` | ~3 h |
| 4. Area-law test | `python pipeline/area_law_test.py` | `data/derived/area_law_significance.csv` | seconds |
| S. Full-signal reconstruction | `python pipeline/full_signal_reconstruction.py` | `data/derived/full_signal_reconstruction.npz` | ~5 min |

Notes on reproducibility:
- **Injection:** `pipeline/injection.py` reproduces the shipped zero-noise strain exactly.
  The noise realisation used in the paper was drawn from bilby's unseeded generator and
  cannot be regenerated, so the shipped `data/injection/` files are the canonical dataset.
  Rerunning stage 1 gives a statistically equivalent injection with different noise.
- **TDInf settings:** `run_tdinf.sh` reproduces the settings of the paper's runs exactly.
  The original scripts are in `pipeline/inspiral/original_run_scripts/`.
- **TDInf prior bounds:** these must match between `run_tdinf.sh` and
  `decode_tdinf_samples.py`, because the raw samples are logistic transforms of the prior ranges.
  The bounds are:
  - M ∈ (58, 82) M☉
  - q ∈ (0.25, 1]
  - D_L ∈ (200, 700) Mpc
  - ι ∈ (0, π)
  - |χ| ∈ [0, 0.9)
- **Ringdown fits** use 4 NumPyro chains × 2000 samples.
  - Priors: M_f ∈ [50, 85] M☉, χ_f ∈ [0.255, 0.95].
  - Each fit uses 0.2 s of data from t_>.
  - T_Mf = 0.337 ms and T_M = 0.354 ms, following Abac et al. (2025).

The full emcee chains and `ringdown` result files are archived on Zenodo:
**DOI 10.5281/zenodo.XXXXXXX (to be added)**.

## Citation

If you use this code or data, please cite the paper (BibTeX to be added on publication).
Please also cite the software it builds on:
- [bilby](https://github.com/bilby-dev/bilby)
- [TDInf](https://github.com/simonajmiller/tdinf)
- [NRSur7dq4](https://doi.org/10.1103/PhysRevResearch.1.033015)
- [emcee](https://github.com/dfm/emcee)
- [ringdown](https://github.com/maxisi/ringdown)
- [qnm](https://github.com/duetosymmetry/qnm)
- [NumPyro](https://github.com/pyro-ppl/numpyro)
- [LALSuite](https://doi.org/10.7935/GT1W-FZ16)
- the GRChombo boson-star simulations of Evstafyeva et al. (2024)
