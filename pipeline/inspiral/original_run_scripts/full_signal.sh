script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${script_dir}/.." && pwd)"
shared="$(cd "${repo}/.." && pwd)"
main_py="${script_dir}/main.py"

inprefix="skypos_ascnode_taper_fix"
posterior_prefix="golden_nonoise"
#outprefix="newval_noise_initflat"
outprefix="r_newval_noise_initflat"

datadir="${repo}/data_injections"
model_dir="${repo}/data"
bilby_dir="${datadir}/data"
output_dir="${datadir}/tdinf_output"

mkdir -p "${output_dir}"
export LAL_DATA_PATH="${model_dir}:${LAL_DATA_PATH:-}"

PI=$(python3 -c "import math; print(math.pi)")
HALF_PI=$(python3 -c "import math; print(math.pi/2)")
THIRD_PI=$(python3 -c "import math; print(math.pi/3)")

python3 "${main_py}" \
--output-h5 "${output_dir}/${outprefix}_results.h5" \
--mode pre \
--Tcut-tm +115 \
--ifos H1 L1 \
--prefix "${outprefix}" \
--metadata "${bilby_dir}/${inprefix}_injection_metadata.csv" \
--h1-data "${bilby_dir}/${inprefix}_H1_injection_data.h5" \
--l1-data "${bilby_dir}/${inprefix}_L1_injection_data.h5" \
--h1-psd "${bilby_dir}/H1_PSD.dat" \
--l1-psd "${bilby_dir}/L1_PSD.dat" \
--data-dir "${datadir}" \
--total-mass-prior-bounds 58.0 82.0 \
--mass-ratio-prior-bounds 0.25 1.0 \
--inclination-prior-bounds 0.0 ${PI} \
--luminosity-distance-prior-bounds 200.0 700.0 \
--spin-magnitude-prior-bounds 0.0 0.9 \
--approx NRSur7dq4 \
--ncpu 8 \
--nwalkers 64 \
--nsteps 600000 \
--sampling-rate 2048 \
--data-sampling-rate 4096.0 \
--max-padding-fraction 0.2 \
--walkerinit flat \
--run-emcee \
## For posterior mode, set walkerinit=posterior and provide init backend:
## --walkerinit posterior --init-from-backend "${output_dir}/${posterior_prefix}_results.h5" \
