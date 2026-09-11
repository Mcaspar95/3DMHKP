#!/bin/bash -l
#SBATCH -J 3DMHKP-ep3
# 60 Egeblad ep3 instances x 1 method (SA) = 60 jobs.
# Must match main_slurm_ep3.py's grid: python3 main_slurm_ep3.py --print-grid
#SBATCH --array=0-59
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# 300 s of SA per task plus interpreter start-up and report writing; 30 min is
# a wide margin. Short jobs like these also schedule sooner than a 2-day ask.
#SBATCH --time=00:30:00
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
# SA is pure Python and needs no Gurobi licence; the export is harmless and
# kept so this file stays interchangeable with slurm.sbatch.
export GRB_LICENSE_FILE=/software/gurobi/gurobi.lic
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python3 main_slurm_ep3.py --id ${SLURM_ARRAY_TASK_ID} --num_cpu ${SLURM_CPUS_PER_TASK} --time-limit 300
