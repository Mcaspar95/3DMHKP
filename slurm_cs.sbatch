#!/bin/bash -l
#SBATCH -J 3DMHKP-CS
#SBATCH --array=0-116
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail
source /home/mcaspar/3DPCPTP/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"
instances=(CS1-CS4/CS*.in)
if [ "${#instances[@]}" -ne 117 ]; then
    echo "Expected 117 CS instances, found ${#instances[@]}" >&2
    exit 1
fi

python3 CS-SA.py --instance "${instances[${SLURM_ARRAY_TASK_ID}]}" \
    --time-limit 1000 --results-dir results_CS_SA
