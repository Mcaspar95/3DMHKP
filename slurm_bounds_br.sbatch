#!/bin/bash -l
#SBATCH -J packing-bounds-BR0-15
# One instance per task: 0 -> BR0-001, 99 -> BR0-100,
# 100 -> BR1-001, ..., 1599 -> BR15-100. At most ten run at once.
#SBATCH --array=0-1599%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=packing-bounds-%A-%a.out
#SBATCH --error=packing-bounds-%A-%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Defaults may be overridden using sbatch --export=ALL,NAME=value,... .
# The environment must have Python >=3.10, numpy and scipy; highspy is optional.
# The packing_bounds source is provided in this project, not installed by jobs.
VENV_DIR="${VENV_DIR:-/home/mcaspar/3DPCPTP}"
SOLVER_SECONDS="${SOLVER_SECONDS:-3600}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
# Includes construction and solver time; leave 30 minutes for process startup,
# checking saved placements and SLURM cleanup within the four-hour job limit.
WALL_SECONDS="${WALL_SECONDS:-12600}"

if [[ -n "${INSTANCE_CLASS:-}" ]]; then
    echo "INSTANCE_CLASS is no longer supported: array task IDs select BR0-BR15. Unset it before submitting." >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:?Submit this job from the 3DMHKP directory}"
if [[ ! -f bound_br15.py || ! -f packing_bounds/__init__.py || ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Missing bound_br15.py, packing_bounds/ or $VENV_DIR/bin/python; copy the project and check VENV_DIR." >&2
    exit 2
fi
PYTHON="$VENV_DIR/bin/python"
if ! "$PYTHON" -c 'import numpy, scipy, packing_bounds; print("Python packages:", packing_bounds.__file__, "NumPy", numpy.__version__, "SciPy", scipy.__version__)'; then
    echo "Cannot import local packing_bounds and its dependencies with $PYTHON; install NumPy and SciPy in this environment." >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

task_id="${SLURM_ARRAY_TASK_ID:?This script requires an array task ID}"
if [[ ! "$task_id" =~ ^[0-9]+$ || ! "$ARRAY_OFFSET" =~ ^[0-9]+$ ]]; then
    echo "Array task ID and ARRAY_OFFSET must be nonnegative integers." >&2
    exit 2
fi
task_id=$((10#$task_id + 10#$ARRAY_OFFSET))
if (( task_id > 1599 )); then
    echo "Array task ID plus ARRAY_OFFSET must be between 0 and 1599." >&2
    exit 2
fi
INSTANCE_CLASS="BR$((task_id / 100))"
printf -v INSTANCE '%s-%03d' "$INSTANCE_CLASS" "$((task_id % 100 + 1))"
INSTANCE_FILE="BischoffRatcliff/$INSTANCE_CLASS/$INSTANCE.txt"
if [[ ! -f "$INSTANCE_FILE" ]]; then
    echo "Missing instance: $INSTANCE_FILE" >&2
    exit 2
fi

# A distinct directory per array job prevents overwriting earlier runs and
# ensures an old timeout fallback cannot be mistaken for a new full run.
OUTPUT_DIR="${OUTPUT_DIR:-$SLURM_SUBMIT_DIR/results_bounds_${INSTANCE_CLASS}_full_${SLURM_ARRAY_JOB_ID}}"
echo "Instance: $INSTANCE; output: $OUTPUT_DIR"
echo "Full IP solver limit: $SOLVER_SECONDS s; build+solve wall limit: $WALL_SECONDS s"
"$PYTHON" bound_br15.py --class "$INSTANCE_CLASS" --instances "$INSTANCE" \
    --stage full --mode ip --time-limit "$SOLVER_SECONDS" \
    --wall-time-limit "$WALL_SECONDS" --output-dir "$OUTPUT_DIR"