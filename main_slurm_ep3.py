#!/usr/bin/env python3
"""SLURM array-job dispatcher for the SA solver over the Egeblad ep3 set.

The companion of main_slurm.py, for the 60 converted Egeblad & Pisinger ep3
instances in Egeblad/ rather than the 16 Mohanty instances. One array task
solves one instance, so the grid is 60 jobs:

    #SBATCH --array=0-59

Two differences from main_slurm.py drive this being a separate file:

  * ep3 instances are addressed by FILENAME (ep3-20-C-C-50.txt), not by the
    instanceNN.txt numbering that main_slurm.py's integer --instance assumes.
  * They carry explicit per-box profits in the coefficient column, so they
    must run with --value-mode flat. Under the default volume mode the
    objective silently becomes profit * volume, which is a different problem
    and yields plausible-looking but wrong numbers.

Examples:
    python3 main_slurm_ep3.py --print-grid
    python3 main_slurm_ep3.py --id 0
    python3 main_slurm_ep3.py --instance ep3-20-C-C-50
    python3 main_slurm_ep3.py --id 0 --time-limit 1800
"""

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INSTANCE_DIR = ROOT / "Egeblad"
RESULTS_DIR = ROOT / "results_3DMHKP_ep3"

# Only SA: the exact solver does not close these within a sensible array-job
# budget (900 s leaves a 24.5% gap on ep3-20-C-C-90).
MODULE_NAME = "3DMHKP-SA"

# The ep3 profits are not proportional to volume, so value = coefficient.
VALUE_MODE = "flat"

# 300 s per instance. On the 20-box instances SA converges in milliseconds;
# the budget is there for the 60-box ones, which manage far fewer iterations
# per second because each decode packs three times as many boxes.
TIME_LIMIT = 300.0

_SOLVER = None


def _get_solver():
    """Import and cache the SA solver module."""
    global _SOLVER
    if _SOLVER is None:
        path = ROOT / f"{MODULE_NAME}.py"
        spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load solver module {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SOLVER = module
    return _SOLVER


def discover_instances(instance_dir):
    """The ep3 instance stems, sorted so a task id always maps to the same one.

    Sorted by (n_boxes, name) rather than lexically, so the grid runs the cheap
    20-box instances first and an array that is cut short still leaves a
    balanced set of results. The stems embed the size: ep3-<n>-<shape>-...
    """
    files = sorted(instance_dir.glob("ep3-*.txt"))

    def key(p):
        parts = p.stem.split("-")
        try:
            return (int(parts[1]), p.stem)
        except (IndexError, ValueError):
            return (1 << 30, p.stem)

    return [p.stem for p in sorted(files, key=key)]


def build_grid(instance_dir):
    return discover_instances(instance_dir)


def _solver_arguments(args, stem):
    """Build the SA solver's native argument list for one instance."""
    solver_args = [
        "--instance-files", str(args.instance_dir / f"{stem}.txt"),
        "--value-mode", VALUE_MODE,
        "--time-limit", str(args.time_limit),
        "--results-dir", str(args.results_dir),
    ]

    # The SA report filename carries alpha, the reheat fraction and the time
    # limit, so a sweep over any of them keeps its runs apart. Forward them, or
    # every array task of a sweep writes the same filename.
    if args.alpha is not None:
        solver_args.extend(["--alpha", str(args.alpha)])
    if args.reheat_fraction is not None:
        solver_args.extend(["--reheat-fraction", str(args.reheat_fraction)])
    if args.seed is not None:
        solver_args.extend(["--seed", str(args.seed)])
    if args.tag:
        solver_args.extend(["--tag", args.tag])
    if args.greedy_only:
        solver_args.append("--greedy-only")
    if args.no_report:
        solver_args.append("--no-report")
    if args.quiet:
        solver_args.append("--quiet")

    return solver_args


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SLURM dispatcher for the SA solver over the Egeblad ep3 set."
    )
    parser.add_argument("--id", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID, zero-based.")
    parser.add_argument("--instance", default=None, metavar="STEM",
                        help="instance stem, e.g. ep3-20-C-C-50 (with or "
                             "without the .txt / .3kp suffix).")
    parser.add_argument("--instance-dir", type=Path, default=INSTANCE_DIR,
                        help=f"directory of converted ep3 instances "
                             f"(default: {INSTANCE_DIR})")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"directory for the reports "
                             f"(default: {RESULTS_DIR.name})")
    parser.add_argument("--time-limit", "--timelimit", dest="time_limit",
                        type=float, default=TIME_LIMIT,
                        help=f"SA seconds per instance (default: {TIME_LIMIT:g})")
    parser.add_argument("--num_cpu", type=int, default=1,
                        help="CPUs requested from SLURM. SA is single-threaded, "
                             "so this is accepted for symmetry with "
                             "main_slurm.py and otherwise unused.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="SA geometric cooling factor; solver default when "
                             "omitted.")
    parser.add_argument("--reheat-fraction", type=float, default=None,
                        help="SA reheat factor; solver default when omitted.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default="",
                        help="extra suffix for the report filenames.")
    parser.add_argument("--greedy-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--print-grid", action="store_true",
                        help="print the grid and the matching SBATCH array "
                             "range, then exit")
    args = parser.parse_args(argv)

    if not args.instance_dir.is_dir():
        parser.error(f"missing instance directory: {args.instance_dir}\n"
                     f"Run convert_ep3.py to create it.")

    grid = build_grid(args.instance_dir)
    if not grid:
        parser.error(f"no ep3-*.txt instances in {args.instance_dir}")

    if args.print_grid:
        print(f"{len(grid)} ep3 instances, solver {MODULE_NAME}.py, "
              f"value-mode {VALUE_MODE}, {args.time_limit:g}s each")
        for i, stem in enumerate(grid):
            print(f"  {i:3d}  {stem}")
        print(f"\n#SBATCH --array=0-{len(grid) - 1}")
        return 0

    if args.id is not None:
        if not 0 <= args.id < len(grid):
            parser.error(f"--id {args.id} out of range [0, {len(grid) - 1}]. "
                         f"Grid has {len(grid)} instances.")
        stem = grid[args.id]
    elif args.instance is not None:
        # Accept ep3-20-C-C-50, ep3-20-C-C-50.txt and ep3-20-C-C-50.3kp alike.
        stem = args.instance
        for suffix in (".txt", ".3kp"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem not in grid:
            parser.error(f"unknown instance {args.instance!r}. "
                         f"Use --print-grid to list the {len(grid)} available.")
    else:
        parser.error("provide either --id or --instance")

    path = args.instance_dir / f"{stem}.txt"
    if not path.exists():
        parser.error(f"missing instance file: {path}")

    print("=" * 60)
    print(f"  Solver:     {MODULE_NAME}.py")
    print(f"  Instance:   {path}")
    print(f"  Value mode: {VALUE_MODE}")
    print(f"  Timelimit:  {args.time_limit:g}s")
    print(f"  Reports:    {args.results_dir}")
    print(f"  CPUs:       {args.num_cpu}")
    print("=" * 60, flush=True)

    result = _get_solver().main(_solver_arguments(args, stem))
    return 0 if result is None else result


if __name__ == "__main__":
    sys.exit(main())
