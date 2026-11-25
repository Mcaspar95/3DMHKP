#!/bin/bash -l
#SBATCH -J 3DMHKP-EG3D-bounds
#SBATCH --array=0-59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --constraint="XEON_E5_2637v3"
#SBATCH --mail-type=END,FAIL

set -euo pipefail
source /home/mcaspar/3DPCPTP/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

cd "${SLURM_SUBMIT_DIR}"
shopt -s nullglob
export LC_ALL=C
instances=(Egeblad/ep3-*.txt)
if [ "${#instances[@]}" -ne 60 ]; then
    echo "Expected 60 Egeblad ep3 instances, found ${#instances[@]}" >&2
    exit 1
fi

instance="${SLURM_SUBMIT_DIR}/${instances[${SLURM_ARRAY_TASK_ID}]}"
name="$(basename "${instance}" .txt)"
results_dir="${SLURM_SUBMIT_DIR}/results_EG3D_bounds"
mkdir -p "${results_dir}"
input="$(mktemp "${TMPDIR:-/tmp}/eg3d-${SLURM_ARRAY_TASK_ID}-XXXXXX.json")"
trap 'rm -f -- "${input}"' EXIT

# Reuse the SA reader, retaining original per-box profits and multiplicities.
python3 - "${instance}" "${input}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("sa", "3DMHKP-SA.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)
path = Path(sys.argv[1])
boxes, containers = sa.parse_instance(path)
if len(containers) != 1 or containers[0]["count"] != 1:
    raise ValueError(f"{path}: expected exactly one container")
instance = {
    "name": path.stem,
    "container": containers[0]["dims"],
    "allow_rotations": True,
    "items": [
        {"id": box["type"], "dimensions": box["dims"],
         "profit": box["coef"], "copies": box["count"]}
        for box in boxes
    ],
}
Path(sys.argv[2]).write_text(json.dumps(instance), encoding="utf-8")
PY

# The wall limit includes model construction; on timeout the CLI records a
# fractional-volume upper bound.
python3 -m packing_bounds "${input}" --stage full --mode ip \
    --time-limit 900 --wall-time-limit 1200 \
    --output "${results_dir}/${name}.json"
