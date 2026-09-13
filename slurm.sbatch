#!/bin/bash -l
#SBATCH -J 3DMHKP-BR0
# 100 Bischoff & Ratcliff BR0 instances, one per array task.
# Must match main_slurm.py's grid: python3 main_slurm.py --print-grid
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
# SA is pure Python on ~120 boxes; the 31G of the Gurobi runs is not needed
# and would make 100 concurrent tasks needlessly hard to schedule.
#SBATCH --mem=4G
# Per array task, not for the array as a whole: one instance at the 300 s
# default. Raise it together with --time-limit below.
#SBATCH --time=04:00:00
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
    SUMMARY_CMD="source /home/mcaspar/3DPCPTP/bin/activate; python3 ${SLURM_SUBMIT_DIR}/main_slurm.py --summarize"
    sbatch --dependency=afterany:${SLURM_ARRAY_JOB_ID} \
           --job-name=3DMHKP-BR0-summary \
           --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:10:00 \
           --output=3DMHKP-BR0-summary-%j.out \
           --wrap="${SUMMARY_CMD}" \
        || echo "summary job not submitted; once the array is done run: python3 main_slurm.py --summarize"
fi

python3 main_slurm.py --id ${SLURM_ARRAY_TASK_ID} --num_cpu ${SLURM_CPUS_PER_TASK} --time-limit 900
