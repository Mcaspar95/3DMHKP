#!/usr/bin/env python3
"""SLURM array-job dispatcher for the SA solver over the Bischoff & Ratcliff
BR0 set (100 instances in BischoffRatcliff/BR0/).

One array task solves one instance, so the grid is 100 jobs:

    #SBATCH --array=0-99

Because each task only ever sees its own instance, the class average has to be
collected afterwards from the written reports - that is what --summarize does:

    python3 main_slurm.py --summarize

Reported figure is the mean % occupation (packed volume / container volume),
the standard metric for the BR container-loading benchmark.

Three differences from the earlier Mohanty dispatcher (recoverable with
`git show HEAD~1:main_slurm.py`) drive this file's shape:

  * BR instances are addressed by FILENAME (BR0-001.txt), not by the
    instanceNN.txt numbering that the integer --instance assumed.
  * They run with --value-mode volume: the converter set every coefficient to
    1, so v_i = volume_i and the packed value IS the packed volume.
  * The quantity of interest is occupation, not profit, and it is a class
    average over 100 instances rather than a per-instance number.

Examples:
    python3 main_slurm.py --print-grid
    python3 main_slurm.py --id 0
    python3 main_slurm.py --instance BR0-007
    python3 main_slurm.py --all --time-limit 60      # all 100 in one process
    python3 main_slurm.py --summarize
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INSTANCE_DIR = ROOT / "BischoffRatcliff" / "BR0"
RESULTS_DIR = ROOT / "results_3DMHKP_BR0"

# Only SA: the exact solver does not close instances of this size within a
# sensible array-job budget.
MODULE_NAME = "3DMHKP-SA"

# The BR converter writes value_coefficient = 1 for every box type, so under
# volume mode v_i = volume_i and maximizing value maximizes packed volume -
# exactly the CLP objective the BR tables report.
VALUE_MODE = "volume"

# 300 s per instance, the budget the ep3 runs used. BR0 instances hold ~120
# boxes of a single type; T0 calibration alone eats a few seconds, so much
# below ~60 s the SA loop barely runs at all.
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


def build_grid(instance_dir):
    """The instance stems, ordered so a task id always maps to the same one.

    The class prefix (BR0, BR15, ...) is taken from the instance directory's
    own name, so this works unchanged for any BR{k} class - not just BR0.
    Sorted by the trailing number rather than lexically, so task 9 is
    {prefix}-010 and not {prefix}-100.
    """
    files = instance_dir.glob(f"{instance_dir.name}-*.txt")

    def key(p):
        m = re.search(r"(\d+)$", p.stem)
        return (int(m.group(1)) if m else 1 << 30, p.stem)

    return [p.stem for p in sorted(files, key=key)]


def _solver_arguments(args, stems):
    """Build the SA solver's native argument list for one or more instances."""
    solver_args = [
        "--instance-files",
        *[str(args.instance_dir / f"{stem}.txt") for stem in stems],
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


# =========================================================================
# Summary over the written reports
# =========================================================================
_INSTANCE_RE = re.compile(r"^Instance:\s*(\S+)", re.M)
# "Volume utilization: 26101440 / 30089620 = 86.7%" - the two integers are
# parsed rather than the printed percentage, which is rounded to one decimal.
_UTIL_RE = re.compile(r"^Volume utilization:\s*(\d+)\s*/\s*(\d+)", re.M)


def read_report(path):
    """(instance, occupation) from one SA report, or None if it has neither."""
    text = path.read_text()
    inst = _INSTANCE_RE.search(text)
    util = _UTIL_RE.search(text)
    if inst is None or util is None:
        return None
    packed, total = int(util.group(1)), int(util.group(2))
    if total == 0:
        return None
    return inst.group(1), packed / total


def summarize(results_dir, grid, pattern="*.txt", verbose=True):
    """Print per-instance and mean % occupation over the reports found.

    Several reports can exist per instance - a time-limit sweep writes
    BR0-001-...-t300s.txt next to ...-t900s.txt - so the best occupation per
    instance is taken, which is the figure the BR tables report.
    """
    if not results_dir.is_dir():
        print(f"no results directory {results_dir}", file=sys.stderr)
        return 1

    best = {}
    n_reports = 0
    for path in sorted(results_dir.glob(pattern)):
        parsed = read_report(path)
        if parsed is None:
            continue
        n_reports += 1
        inst, occ = parsed
        if occ > best.get(inst, -1.0):
            best[inst] = occ

    if not best:
        print(f"no parsable reports in {results_dir} matching {pattern!r}",
              file=sys.stderr)
        return 1

    if verbose:
        print(f"{'instance':<12} {'occupation':>11}")
        print("-" * 24)
        for inst in sorted(best):
            print(f"{inst:<12} {best[inst]:>10.1%}")
        print("-" * 24)

    mean = sum(best.values()) / len(best)
    # Only hold the reports against the grid when they are in fact this grid's:
    # --results-dir can point at another set's reports, and then "99 missing"
    # would be noise rather than a warning.
    on_grid = grid and any(inst in best for inst in grid)
    missing = [s for s in grid if s not in best] if on_grid else []

    print(f"instances reported: {len(best)}"
          + (f" / {len(grid)}" if on_grid else "")
          + f"  ({n_reports} report files)")
    if missing:
        head = ", ".join(missing[:8])
        more = f", ... (+{len(missing) - 8})" if len(missing) > 8 else ""
        print(f"missing: {head}{more}")
    print(f"average % occupation: {mean:.2%}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SLURM dispatcher for the SA solver over the "
                    "Bischoff & Ratcliff BR0 set."
    )
    parser.add_argument("--id", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID, zero-based.")
    parser.add_argument("--instance", default=None, metavar="STEM",
                        help="instance stem, e.g. BR0-007 (with or without the "
                             ".txt suffix).")
    parser.add_argument("--instance-dir", type=Path, default=INSTANCE_DIR,
                        help=f"directory of converted BR instances, e.g. "
                             f"BischoffRatcliff/BR15 (default: {INSTANCE_DIR})")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="directory for the reports (default: "
                             f"results_3DMHKP_<instance-dir-name>, e.g. "
                             f"{RESULTS_DIR.name} for the default instance dir)")
    parser.add_argument("--time-limit", "--timelimit", dest="time_limit",
                        type=float, default=TIME_LIMIT,
                        help=f"SA seconds per instance (default: {TIME_LIMIT:g})")
    parser.add_argument("--num_cpu", type=int, default=1,
                        help="CPUs requested from SLURM. SA is single-threaded, "
                             "so this is accepted for symmetry with the other "
                             "dispatchers and otherwise unused.")
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
    parser.add_argument("--all", action="store_true",
                        help="solve all 100 instances sequentially in this one "
                             "process and print the class average at the end, "
                             "instead of one instance per array task.")
    parser.add_argument("--summarize", action="store_true",
                        help="do not solve: read the reports already in "
                             "--results-dir and print the average % occupation.")
    parser.add_argument("--summary-pattern", default="*.txt", metavar="GLOB",
                        help="which reports --summarize reads (default: *.txt). "
                             "Narrow it to one setting of a sweep, e.g. "
                             "'*-t300s.txt'.")
    parser.add_argument("--print-grid", action="store_true",
                        help="print the grid and the matching SBATCH array "
                             "range, then exit")
    args = parser.parse_args(argv)

    # Default results dir tracks the instance dir's class (BR0 -> BR0, BR15 ->
    # BR15, ...) so switching --instance-dir alone never mixes classes into
    # one results folder.
    if args.results_dir is None:
        args.results_dir = ROOT / f"results_3DMHKP_{args.instance_dir.name}"

    if args.summarize:
        grid = build_grid(args.instance_dir) if args.instance_dir.is_dir() else []
        return summarize(args.results_dir, grid, args.summary_pattern)

    if not args.instance_dir.is_dir():
        parser.error(f"missing instance directory: {args.instance_dir}\n"
                     f"Run convert_br.py to create it.")

    grid = build_grid(args.instance_dir)
    if not grid:
        parser.error(f"no {args.instance_dir.name}-*.txt instances in "
                     f"{args.instance_dir}")

    if args.print_grid:
        print(f"{len(grid)} {args.instance_dir.name} instances, solver {MODULE_NAME}.py, "
              f"value-mode {VALUE_MODE}, {args.time_limit:g}s each")
        for i, stem in enumerate(grid):
            print(f"  {i:3d}  {stem}")
        print(f"\n#SBATCH --array=0-{len(grid) - 1}")
        return 0

    if args.all:
        stems = grid
    elif args.id is not None:
        if not 0 <= args.id < len(grid):
            parser.error(f"--id {args.id} out of range [0, {len(grid) - 1}]. "
                         f"Grid has {len(grid)} instances.")
        stems = [grid[args.id]]
    elif args.instance is not None:
        stem = args.instance[:-4] if args.instance.endswith(".txt") else args.instance
        if stem not in grid:
            parser.error(f"unknown instance {args.instance!r}. "
                         f"Use --print-grid to list the {len(grid)} available.")
        stems = [stem]
    else:
        parser.error("provide --id, --instance, --all or --summarize")

    for stem in stems:
        path = args.instance_dir / f"{stem}.txt"
        if not path.exists():
            parser.error(f"missing instance file: {path}")

    print("=" * 60)
    print(f"  Solver:     {MODULE_NAME}.py")
    if len(stems) == 1:
        print(f"  Instance:   {args.instance_dir / (stems[0] + '.txt')}")
    else:
        print(f"  Instances:  {len(stems)} ({stems[0]} .. {stems[-1]})")
    print(f"  Value mode: {VALUE_MODE}")
    print(f"  Timelimit:  {args.time_limit:g}s per instance")
    print(f"  Reports:    {args.results_dir}")
    print(f"  CPUs:       {args.num_cpu}")
    print("=" * 60, flush=True)

    result = _get_solver().main(_solver_arguments(args, stems))

    # A single array task cannot know the class average - only the --all run,
    # which has just solved every instance, can close with it here. For the
    # array, the average comes from a --summarize pass once all tasks are done.
    if args.all and not args.no_report:
        print()
        summarize(args.results_dir, grid, args.summary_pattern, verbose=False)

    return 0 if result is None else result


if __name__ == "__main__":
    sys.exit(main())
