#!/bin/bash -l
#SBATCH -J 2DKP-EG2D-opt
#SBATCH --array=0-79
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
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
export LC_ALL=C
instances=()
for n in 30 50 100 200; do
    for class in D S T U W; do
        for kind in C R; do
            for percent in 25 75; do
                path="Egeblad/ep2/ep2-${n}-${class}-${kind}-${percent}.2kp"
                if [ ! -f "${path}" ]; then
                    echo "Missing instance: ${path}" >&2
                    exit 1
                fi
                instances+=("${path}")
            done
        done
    done
done
if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || \
   [ "${SLURM_ARRAY_TASK_ID}" -ge "${#instances[@]}" ]; then
    echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

instance="${instances[${SLURM_ARRAY_TASK_ID}]}"
name="$(basename "${instance}" .2kp)"
python3 eg2d_opt.py "${instance}" \
    --threads "${SLURM_CPUS_PER_TASK}" --time-limit 85800 \
    --output "${SLURM_SUBMIT_DIR}/results_EG2D_opt/${name}.json"
