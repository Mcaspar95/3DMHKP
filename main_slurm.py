#!/usr/bin/env python3
"""SLURM array-job dispatcher for the 3DMHKP solvers.

Each array task selects one solver and one Mohanty instance. The solver
modules are imported lazily because importing the exact MILP also initializes
Gurobi.

Examples:
    python main_slurm.py --print-grid
    python main_slurm.py --method D --instance 1 --time-limit 1800
    python main_slurm.py --method SA --instance 1 --time-limit 1800
    python main_slurm.py --id 0
"""

import argparse
import importlib.util
import sys
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INSTANCE_DIR = ROOT / "Mohanty"
INSTANCE_COUNT = 16

# Keep these names aligned with the solver filenames in this project.
module_names = {
    #"D": "3DMHKP",
    "SA": "3DMHKP-SA",
}

_SOLVER_CACHE = {}


def _get_solver(method):
    """Import and cache the solver selected by ``method``."""
    if method not in _SOLVER_CACHE:
        module_name = module_names[method]
        module_path = ROOT / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load solver module {module_path}")
        solver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solver)
        _SOLVER_CACHE[method] = solver
    return _SOLVER_CACHE[method]


METHODS = sorted(module_names)
INSTANCES = list(range(1, INSTANCE_COUNT + 1))
GRID = list(product(METHODS, INSTANCES))
TOTAL_JOBS = len(GRID)


def _resolve_id(task_id):
    """Map a zero-based SLURM task id to ``(method, instance)``."""
    if task_id < 0 or task_id >= TOTAL_JOBS:
        raise ValueError(
            f"--id {task_id} out of range [0, {TOTAL_JOBS - 1}]. "
            f"Grid has {TOTAL_JOBS} combinations."
        )
    return GRID[task_id]


def _solver_arguments(args, method, instance):
    """Build the solver-native argument list for one grid entry."""
    solver_args = [
        "--instances", str(instance),
        "--instance-dir", str(args.instance_dir),
        "--time-limit", str(args.time_limit),
    ]

    if args.value_mode:
        solver_args.extend(["--value-mode", args.value_mode])
    if args.no_report:
        solver_args.append("--no-report")
    if args.greedy_only:
        solver_args.append("--greedy-only")
    if args.seed is not None:
        solver_args.extend(["--seed", str(args.seed)])

    if method == "D":
        solver_args.extend(["--threads", str(args.num_cpu)])
        if args.verbose:
            solver_args.append("--verbose")
    elif args.quiet:
        solver_args.append("--quiet")

    return solver_args


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SLURM dispatcher for the 3DMHKP exact and SA solvers."
    )
    parser.add_argument(
        "--id", type=int, default=None,
        help=f"SLURM_ARRAY_TASK_ID (0..{TOTAL_JOBS - 1}).",
    )
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--instance", type=int, choices=INSTANCES, default=None)
    parser.add_argument(
        "--instance-dir", type=Path, default=INSTANCE_DIR,
        help=f"directory containing instanceNN.txt files (default: {INSTANCE_DIR})",
    )
    parser.add_argument(
        "--time-limit", "--timelimit", dest="time_limit", type=float,
        default=1800.0, help="seconds per instance (default: 900)",
    )
    parser.add_argument(
        "--num_cpu", type=int, default=1,
        help="CPUs requested from SLURM; used as the D solver thread count.",
    )
    parser.add_argument("--value-mode", choices=("volume", "flat"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--greedy-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--print-grid", action="store_true",
        help="print the active grid and matching SBATCH array range, then exit",
    )
    args = parser.parse_args(argv)

    if args.print_grid:
        print(f"{len(METHODS)} methods x {len(INSTANCES)} instances = {TOTAL_JOBS} jobs")
        for method in METHODS:
            print(f"{method}: {module_names[method]}.py")
        print(f"\n#SBATCH --array=0-{TOTAL_JOBS - 1}")
        return 0

    if args.id is not None:
        method, instance = _resolve_id(args.id)
    elif args.method is not None and args.instance is not None:
        method, instance = args.method, args.instance
    else:
        parser.error("provide either --id or both --method and --instance")

    instance_path = args.instance_dir / f"instance{instance:02d}.txt"
    if not instance_path.exists():
        parser.error(f"missing instance file: {instance_path}")

    print("=" * 60)
    print(f"  Method:   {method} ({module_names[method]}.py)")
    print(f"  Instance: {instance_path}")
    print(f"  Timelimit: {args.time_limit}s")
    print(f"  CPUs:     {args.num_cpu}")
    print("=" * 60, flush=True)

    solver = _get_solver(method)
    result = solver.main(_solver_arguments(args, method, instance))
    return 0 if result is None else result


if __name__ == "__main__":
    sys.exit(main())
