#!/bin/bash -l
#SBATCH -J 2DKP-Beasley-opt
#SBATCH --array=0-209
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint="XEON_E5_2637v3"

set -euo pipefail
source /home/mcaspar/3DPCPTP/bin/activate
export GRB_LICENSE_FILE=/software/gurobi/gurobi.lic
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

cd "${SLURM_SUBMIT_DIR}"
if [ "${BEASLEY_SUMMARIZE:-0}" = "1" ]; then
    python3 beasley2d_opt.py --summarize --results-dir "${BEASLEY_RESULT_DIR}"
    exit
fi

export BEASLEY_RESULT_DIR="${SLURM_SUBMIT_DIR}/results_Beasley2D_opt/${SLURM_ARRAY_JOB_ID}"
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    sbatch --dependency="afterany:${SLURM_ARRAY_JOB_ID}" \
        --job-name=2DKP-Beasley-summary --array=0 \
        --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:10:00 \
        --export=ALL,BEASLEY_SUMMARIZE=1 \
        "${SLURM_SUBMIT_DIR}/slurm_beasley2d_opt.sbatch" \
        || echo "Summary submission failed; run: python3 beasley2d_opt.py --summarize --results-dir \"${BEASLEY_RESULT_DIR}\"" >&2
fi

python3 beasley2d_opt.py --id "${SLURM_ARRAY_TASK_ID}" \
    --threads "${SLURM_CPUS_PER_TASK}" --time-limit 10800 \
    --results-dir "${BEASLEY_RESULT_DIR}"
