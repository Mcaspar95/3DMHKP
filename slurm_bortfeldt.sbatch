#!/bin/bash -l
#SBATCH -J 3DMHKP-Bortfeldt
# 64 Bortfeldt (2000) destination-restricted Ivancic instances, one per array
# task. Must match main_slurm_bortfeldt.py's grid:
#   python3 main_slurm_bortfeldt.py --print-grid
#SBATCH --array=0-63
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
# SA is pure Python on at most ~180 boxes (2-5 box types); the 31G of the
# Gurobi runs is not needed and would make 64 concurrent tasks needlessly
# hard to schedule.
#SBATCH --mem=4G
# Per array task, not for the array as a whole: one instance at 90 s
# (--time-limit below), plus margin for T0 calibration and I/O.
#SBATCH --time=00:15:00
##SBATCH -o %x-%A-%a.out
##SBATCH -e %x-%A-%a.err
#SBATCH --mail-type=END,FAIL
##SBATCH --constraint="XEON_E5_2637v3"
##SBATCH --constraint="XEON_E5_2640v3"
##SBATCH --constraint="EPYC_9135"
##EPYC_7262
##module load python/3.13
##module load gcc/latest
##module load gurobipy/12.0.3
source /home/mcaspar/3DPCPTP/bin/activate
export GRB_LICENSE_FILE=/software/gurobi/gurobi.lic
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# No single array task sees more than its own instance, so the class average
# is collected by a small follow-up job that waits for the whole array
# (afterany: also runs if some task failed, and reports what is missing).
# Submitted from task 0 only, before the solve, so it is queued even if this
# task is later preempted.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    SUMMARY_CMD="source /home/mcaspar/3DPCPTP/bin/activate; python3 ${SLURM_SUBMIT_DIR}/main_slurm_bortfeldt.py --summarize"
    sbatch --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
           --job-name=3DMHKP-Bortfeldt-summary \
           --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:10:00 \
           --output=3DMHKP-Bortfeldt-summary-%j.out \
           --wrap="${SUMMARY_CMD}" \
        || echo "summary job not submitted; once the array is done run: python3 main_slurm_bortfeldt.py --summarize"
fi

python3 main_slurm_bortfeldt.py --id ${SLURM_ARRAY_TASK_ID} --num_cpu ${SLURM_CPUS_PER_TASK} --time-limit 90
