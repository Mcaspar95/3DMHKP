#!/bin/bash -l
#SBATCH -J BR0-pattern-bounds
#SBATCH --array=0-99%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=BR0-pattern-%A-%a.out
#SBATCH --error=BR0-pattern-%A-%a.err

set -euo pipefail

# The current line-pattern implementation only supports homogeneous BR0.
# Defaults can be overridden with sbatch --export=ALL,NAME=value,... .
VENV_DIR="${VENV_DIR:-/home/mcaspar/3DPCPTP}"
cd "${SLURM_SUBMIT_DIR:?Submit from the 3DMHKP project directory}"

if [[ ! -f bound_br0_patterns.py || ! -f verify_br0_pattern_certificate.py || ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Missing pattern-bound scripts or $VENV_DIR/bin/activate; check the submission directory and VENV_DIR." >&2
    exit 2
fi
source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

task_id="${SLURM_ARRAY_TASK_ID:?This script requires a SLURM array task ID}"
if [[ ! "$task_id" =~ ^[0-9]+$ ]] || (( 10#$task_id > 99 )); then
    echo "Invalid array task ID: $task_id (expected 0 to 99)" >&2
    exit 2
fi
printf -v INSTANCE 'BR0-%03d' "$((10#$task_id + 1))"
INSTANCE_FILE="BischoffRatcliff/BR0/$INSTANCE.txt"
if [[ ! -f "$INSTANCE_FILE" ]]; then
    echo "Missing instance: $INSTANCE_FILE" >&2
    exit 2
fi

# A distinct output directory per array job avoids reuse of old results.
OUTPUT_DIR="${OUTPUT_DIR:-$SLURM_SUBMIT_DIR/results_bounds_BR0_patterns_${SLURM_ARRAY_JOB_ID:?Missing array job ID}}"
RESULT_FILE="$OUTPUT_DIR/$INSTANCE-patterns.json"
echo "Instance: $INSTANCE; output: $OUTPUT_DIR"
echo "The bound uses free rotations; more runtime alone does not strengthen it."
python3 bound_br0_patterns.py --instances "$INSTANCE" --output-dir "$OUTPUT_DIR"
python3 verify_br0_pattern_certificate.py "$RESULT_FILE"