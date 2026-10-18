#!/usr/bin/env bash
# Stage 2: TDInf + emcee analyses with NRSur7dq4 (inspiral and full signal).
#
#   bash pipeline/inspiral/run_tdinf.sh inspiral_40   # t_< = t_peak - 40 T_M
#   bash pipeline/inspiral/run_tdinf.sh inspiral_90   # t_< = t_peak - 90 T_M
#   bash pipeline/inspiral/run_tdinf.sh full_signal   # cut after the end of the signal
#
# Settings are exactly those of the runs used in the paper (the original scripts are in
# original_run_scripts/): 64 walkers x 6e5 steps. The first 2e4
# steps are a burn-in that is not saved, after which the worst 30% of walkers are
# reseeded. Expect several days per run on ~8 cores. The resulting samples are already
# shipped in data/posteriors/, so this is only needed to redo the analysis.
# Writes output/tdinf/<run>_results.h5 (emcee backend). Then:
#   python pipeline/inspiral/extract_samples.py output/tdinf/<run>_results.h5 data/posteriors/raw/<run>_samples_raw.txt
#   python pipeline/decode_tdinf_samples.py
#
# Needs tdinf on PYTHONPATH (or pip-installed) and external/NRSur7dq4_v1.0.h5.
#
# Settings you can change (all options are listed by `python pipeline/inspiral/main.py --help`):
#   NCPU                      environment variable, number of worker processes (default 8)
#   --Tcut-tm, --nsteps       cutoff time [T_M] and number of steps, set per run in the case block below
#   --*-prior-bounds          prior ranges. If you change them, change PRIOR_BOUNDS in
#                             pipeline/decode_tdinf_samples.py (and PRIOR_ARGS in
#                             pipeline/full_signal_reconstruction.py) to match
#   --nwalkers                number of emcee walkers
#   --walkerinit              how walkers start: flat (prior), reference (around the injection),
#                             posterior (from an earlier run, with --init-from-backend <file.h5>),
#                             or resume (continue an interrupted run in --output-h5)
#   --sampling-rate           rate [Hz] the data are downsampled to
#   --max-padding-fraction    largest fraction of the analysed data that may be zero padding
#   --no-spins                fix both spins to zero
#   --verbose                 print more about the data conditioning and set-up
# The burn-in, reseed and walker-initialisation knobs that are not command-line options
# are constants at the top of main.py.
set -euo pipefail

run="${1:?usage: run_tdinf.sh inspiral_40|inspiral_90|full_signal}"
case "${run}" in
    inspiral_40) tcut=-40; nsteps=600000 ;;
    inspiral_90) tcut=-90; nsteps=600000 ;;
    full_signal) tcut=115; nsteps=600000 ;;
    *) echo "unknown run ${run}" >&2; exit 1 ;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../.." && pwd)"
inj="${repo}/data/injection"
out="${repo}/output/tdinf"
mkdir -p "${out}"
export LAL_DATA_PATH="${repo}/external:${LAL_DATA_PATH:-}"
PI=$(python -c "import math; print(math.pi)")

# Paper settings

python "${here}/main.py" \
    --output-h5 "${out}/${run}_results.h5" \
    --mode pre \
    --Tcut-tm "${tcut}" \
    --ifos H1 L1 \
    --prefix "${run}" \
    --metadata "${inj}/injection_metadata.csv" \
    --h1-data "${inj}/H1_injection_data.h5" \
    --l1-data "${inj}/L1_injection_data.h5" \
    --h1-psd "${inj}/H1_PSD.dat" \
    --l1-psd "${inj}/L1_PSD.dat" \
    --data-dir "${repo}/data" \
    --total-mass-prior-bounds 58 82 \
    --mass-ratio-prior-bounds 0.25 1.0 \
    --inclination-prior-bounds 0 "${PI}" \
    --luminosity-distance-prior-bounds 200 700 \
    --spin-magnitude-prior-bounds 0.0 0.9 \
    --approx NRSur7dq4 \
    --ncpu "${NCPU:-8}" \
    --nwalkers 64 \
    --nsteps "${nsteps}" \
    --sampling-rate 2048 \
    --data-sampling-rate 4096.0 \
    --max-padding-fraction 0.2 \
    --walkerinit flat \
    --run-emcee
