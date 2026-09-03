#!/bin/bash -l
#SBATCH -J 3DMHKP
# 2 methods (D, SA) x 16 instances = 32 jobs.
# Must match main_slurm.py's TOTAL_JOBS: python3 main_slurm.py --print-grid
#SBATCH --array=0-31
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=31G
#SBATCH --time=1-23:00:00
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
python3 main_slurm.py --id ${SLURM_ARRAY_TASK_ID} --num_cpu ${SLURM_CPUS_PER_TASK}