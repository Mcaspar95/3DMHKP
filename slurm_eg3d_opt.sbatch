#!/bin/bash -l
#SBATCH -J 3DMHKP-EG3D-opt
#SBATCH --array=0-19
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
shopt -s nullglob
export LC_ALL=C
instances=(Egeblad/ep3-20-*.txt)
if [ "${#instances[@]}" -ne 20 ]; then
    echo "Expected 20 Egeblad ep3-20 instances, found ${#instances[@]}" >&2
    exit 1
fi
if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || \
   [ "${SLURM_ARRAY_TASK_ID}" -ge "${#instances[@]}" ]; then
    echo "Invalid array task ID: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

instance="${instances[${SLURM_ARRAY_TASK_ID}]}"
name="$(basename "${instance}" .txt)"
results_dir="${SLURM_SUBMIT_DIR}/results_EG3D_opt"
mkdir -p "${results_dir}"

python3 - "${instance}" "${results_dir}/${name}.json" "${SLURM_CPUS_PER_TASK}" <<'PY'
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

from gurobipy import GRB

spec = importlib.util.spec_from_file_location("exact", "3DMHKP.py")
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)

path = Path(sys.argv[1])
output = Path(sys.argv[2])
threads = int(sys.argv[3])
started = time.time()
instance = exact.build_instance(path, value_mode="flat")
if len(instance["boxes"]) != 20 or len(instance["containers"]) != 1:
    raise ValueError(f"{path}: expected 20 boxes and one container")

compat = exact.compatibility(instance)
warm_start = exact.greedy_pack(instance, compat, time_budget=30)
errors = exact.verify_placement(instance, warm_start)
if errors:
    raise ValueError(f"Invalid greedy packing: {errors[0]}")

model = exact.build_model(instance, compat, warm_start=warm_start, verbose=True)
model.Params.Threads = threads
model.Params.TimeLimit = 85800
model.Params.MIPGap = 0
model.Params.MIPGapAbs = 0
model.Params.Symmetry = 2
model.optimize()

placement = exact.extract_placement(model, instance, compat)
errors = exact.verify_placement(instance, placement)
if errors:
    raise ValueError(f"Invalid solver packing: {errors[0]}")
if model.SolCount:
    value = exact.placement_value(instance, placement)
    if not math.isclose(value, model.ObjVal, rel_tol=0, abs_tol=1e-5):
        raise ValueError("Extracted packing value does not match the solver objective")
else:
    value = None

dual_bound = model.ObjBound
result = {
    "instance": instance["name"],
    "value_mode": "flat",
    "allow_rotations": True,
    "status_code": model.Status,
    "proven_optimal": model.Status == GRB.OPTIMAL and model.SolCount > 0,
    "value": value,
    "dual_bound": dual_bound if math.isfinite(dual_bound) else None,
    "runtime_seconds": time.time() - started,
    "solver_seconds": model.Runtime,
    "placements": [
        {
            "box": j,
            "type": instance["boxes"][j]["type"],
            "container": k,
            "position": pos,
            "dimensions": instance["boxes"][j]["orientations"][r],
        }
        for j, (k, r, pos) in sorted(placement.items())
    ],
}
output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(f"Result written to {output}", flush=True)
if not result["proven_optimal"]:
    raise SystemExit(f"Optimality not proven (Gurobi status {model.Status})")
PY
