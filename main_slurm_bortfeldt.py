#!/usr/bin/env python3
"""SLURM array-job dispatcher for Bortfeldt-SA.py over the 64 Ivancic et al.
(1989) / Bortfeldt (2000) instances.

Unlike the earlier version of this dispatcher, no conversion step is needed:
Bortfeldt-SA.py reads the Ivancic/Bortfeldt file format directly and solves
Bortfeldt's ACTUAL problem (unlimited containers per type, every box packed,
minimise total container cost) - see Bortfeldt-SA.py's own docstring for the
full rationale and the validation against his published Table 1 figures.

Which folder you point --instance-dir at selects the problem variant:

    Ivancic/    (default) - the original, UNRESTRICTED problems. This is
                what Bortfeldt (2000) Table 1 reports per instance.
    Bortfeldt/            - his Sec. 6/7 destination-restricted extension
                (one box type per container). Bortfeldt-SA.py auto-detects
                the destination column and enforces it; only an aggregate
                figure (67.5%, std. dev. 14.9%) is published for this
                variant, not per-instance numbers.

One array task solves one instance, so the grid is 64 jobs:

    #SBATCH --array=0-63

Because each task only ever sees its own instance, the class average has to
be collected afterwards from the written reports - that is what --summarize
does:

    python3 main_slurm_bortfeldt.py --summarize

Examples:
    python3 main_slurm_bortfeldt.py --print-grid
    python3 main_slurm_bortfeldt.py --id 0
    python3 main_slurm_bortfeldt.py --instance problem16a
    python3 main_slurm_bortfeldt.py --instance-dir Bortfeldt --id 0   # destination-restricted
    python3 main_slurm_bortfeldt.py --all --time-limit 90   # all 64 in one process
    python3 main_slurm_bortfeldt.py --summarize
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INSTANCE_DIR = ROOT / "Ivancic"
RESULTS_DIR = ROOT / "results_Bortfeldt_SA"

MODULE_NAME = "Bortfeldt-SA"

# 90 s per instance: Bortfeldt reports MCL's mean runtime as 73.3 s on this
# same 64-instance set (Table 1) - not an arbitrary budget, a like-for-like
# one.
TIME_LIMIT = 90.0

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
    """The problemNNx instance stems, ordered so a task id always maps to the
    same one: numerically by group (01..17), then alphabetically by the a/b/
    c/d suffix within a group (so task 0 is problem01a, not problem10a).
    """
    files = instance_dir.glob("problem*.txt")

    def key(p):
        m = re.match(r"problem(\d+)([a-z]?)", p.stem)
        if not m:
            return (1 << 30, p.stem)
        return (int(m.group(1)), m.group(2))

    return [p.stem for p in sorted(files, key=key)]


def _solver_arguments(args, stems):
    """Build Bortfeldt-SA.py's native argument list for one or more instances."""
    solver_args = [
        "--instance-files",
        *[str(args.instance_dir / f"{stem}.txt") for stem in stems],
        "--time-limit", str(args.time_limit),
        "--results-dir", str(args.results_dir),
    ]

    if args.max_iterations is not None:
        solver_args.extend(["--max-iterations", str(args.max_iterations)])
    if args.ep_limit is not None:
        solver_args.extend(["--ep-limit", str(args.ep_limit)])
    if args.reheat_threshold is not None:
        solver_args.extend(["--reheat-threshold", str(args.reheat_threshold)])
    if args.reheat_fraction is not None:
        solver_args.extend(["--reheat-fraction", str(args.reheat_fraction)])
    if args.cost_mode is not None:
        solver_args.extend(["--cost-mode", args.cost_mode])
    if args.ignore_destinations:
        solver_args.append("--ignore-destinations")
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
# "Volume utilization: 17920 / 24000 = 74.7%" - the two integers are parsed
# rather than the printed percentage, which is rounded to one decimal.
_UTIL_RE = re.compile(r"^\s*Volume utilization:\s*(\d+)\s*/\s*(\d+)", re.M)
_CONTAINERS_RE = re.compile(r"^\s*Containers used:\s*(\d+)", re.M)
_COST_RE = re.compile(r"^\s*Total container cost:\s*([\d.]+)", re.M)


def read_report(path):
    """(instance, occupation, containers, cost) from one report, or None."""
    text = path.read_text()
    inst = _INSTANCE_RE.search(text)
    util = _UTIL_RE.search(text)
    if inst is None or util is None:
        return None
    packed, total = int(util.group(1)), int(util.group(2))
    if total == 0:
        return None
    n_cont = _CONTAINERS_RE.search(text)
    cost = _COST_RE.search(text)
    return (inst.group(1), packed / total,
            int(n_cont.group(1)) if n_cont else None,
            float(cost.group(1)) if cost else None)


def summarize(results_dir, grid, pattern="*.txt", verbose=True):
    """Print per-instance and mean % utilization over the reports found.

    Several reports can exist per instance - a sweep writes
    problem01a-...-t90s.txt next to ...-t900s.txt - so the BEST (lowest-cost,
    i.e. highest-utilization) report per instance is taken, matching how
    Bortfeldt's own Table 1 reports the better of his two MCL variants.
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
        inst, occ, n_cont, cost = parsed
        if occ > best.get(inst, (-1.0,))[0]:
            best[inst] = (occ, n_cont, cost)

    if not best:
        print(f"no parsable reports in {results_dir} matching {pattern!r}",
              file=sys.stderr)
        return 1

    if verbose:
        print(f"{'instance':<12} {'containers':>10} {'cost':>10} "
              f"{'utilization':>12}")
        print("-" * 48)
        for inst in sorted(best):
            occ, n_cont, cost = best[inst]
            cont_s = f"{n_cont}" if n_cont is not None else "?"
            cost_s = f"{cost:.1f}" if cost is not None else "?"
            print(f"{inst:<12} {cont_s:>10} {cost_s:>10} {occ:>11.1%}")
        print("-" * 48)

    mean = sum(v[0] for v in best.values()) / len(best)
    on_grid = grid and any(inst in best for inst in grid)
    missing = [s for s in grid if s not in best] if on_grid else []

    print(f"instances reported: {len(best)}"
          + (f" / {len(grid)}" if on_grid else "")
          + f"  ({n_reports} report files)")
    if missing:
        head = ", ".join(missing[:8])
        more = f", ... (+{len(missing) - 8})" if len(missing) > 8 else ""
        print(f"missing: {head}{more}")
    print(f"average % volume utilization: {mean:.2%}")
    print("(Bortfeldt's MCL averages 82.4% on this same 64-instance set, "
          "Table 1)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SLURM dispatcher for Bortfeldt-SA.py over the 64 "
                    "Ivancic et al. / Bortfeldt (2000) instances."
    )
    parser.add_argument("--id", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID, zero-based.")
    parser.add_argument("--instance", default=None, metavar="STEM",
                        help="instance stem, e.g. problem16a (with or "
                             "without the .txt suffix).")
    parser.add_argument("--instance-dir", type=Path, default=INSTANCE_DIR,
                        help=f"Ivancic (unrestricted, default) or Bortfeldt "
                             f"(destination-restricted) instance directory "
                             f"(default: {INSTANCE_DIR})")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"directory for the reports "
                             f"(default: {RESULTS_DIR.name})")
    parser.add_argument("--time-limit", "--timelimit", dest="time_limit",
                        type=float, default=TIME_LIMIT,
                        help=f"SA seconds per instance (default: {TIME_LIMIT:g}; "
                             f"MCL averaged 73.3s on this set)")
    parser.add_argument("--num_cpu", type=int, default=1,
                        help="CPUs requested from SLURM. SA is single-threaded, "
                             "so this is accepted for symmetry with the other "
                             "dispatchers and otherwise unused.")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="solver default when omitted.")
    parser.add_argument("--ep-limit", type=int, default=None,
                        help="max extreme points scanned per box; solver "
                             "default (unlimited) when omitted.")
    parser.add_argument("--reheat-threshold", type=int, default=None,
                        help="solver default when omitted.")
    parser.add_argument("--reheat-fraction", type=float, default=None,
                        help="solver default when omitted.")
    parser.add_argument("--cost-mode", choices=("file", "volume"),
                        default=None,
                        help="'file' (solver default) uses the instance's own "
                             "cost column; 'volume' derives cost = volume/10 "
                             "regardless of what the file says.")
    parser.add_argument("--ignore-destinations", action="store_true",
                        help="ignore the destination column even on "
                             "Bortfeldt/ files, solving the unrestricted "
                             "problem instead.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default="",
                        help="extra suffix for the report filenames.")
    parser.add_argument("--greedy-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="solve all 64 instances sequentially in this one "
                             "process and print the class average at the end, "
                             "instead of one instance per array task.")
    parser.add_argument("--summarize", action="store_true",
                        help="do not solve: read the reports already in "
                             "--results-dir and print the average % "
                             "utilization.")
    parser.add_argument("--summary-pattern", default="*.txt", metavar="GLOB",
                        help="which reports --summarize reads (default: *.txt). "
                             "Narrow it to one setting of a sweep, e.g. "
                             "'*-t90s.txt', or to one problem variant, e.g. "
                             "'*-dest-*' for destination-restricted runs.")
    parser.add_argument("--print-grid", action="store_true",
                        help="print the grid and the matching SBATCH array "
                             "range, then exit")
    args = parser.parse_args(argv)

    if args.summarize:
        grid = build_grid(args.instance_dir) if args.instance_dir.is_dir() else []
        return summarize(args.results_dir, grid, args.summary_pattern)

    if not args.instance_dir.is_dir():
        parser.error(f"missing instance directory: {args.instance_dir}\n"
                     f"Use --instance-dir Ivancic or --instance-dir Bortfeldt.")

    grid = build_grid(args.instance_dir)
    if not grid:
        parser.error(f"no problem*.txt instances in {args.instance_dir}")

    if args.print_grid:
        restricted = args.instance_dir.name == "Bortfeldt"
        print(f"{len(grid)} instances from {args.instance_dir.name}/ "
              f"({'destination-restricted' if restricted else 'unrestricted'}), "
              f"solver {MODULE_NAME}.py, {args.time_limit:g}s each")
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
    print(f"  Solver:     {MODULE_NAME}.py (Bortfeldt MCLP: minimise "
          f"container cost, every box packed)")
    if len(stems) == 1:
        print(f"  Instance:   {args.instance_dir / (stems[0] + '.txt')}")
    else:
        print(f"  Instances:  {len(stems)} ({stems[0]} .. {stems[-1]})")
    print(f"  Timelimit:  {args.time_limit:g}s per instance")
    print(f"  Reports:    {args.results_dir}")
    print(f"  CPUs:       {args.num_cpu}")
    print("=" * 60, flush=True)

    result = _get_solver().main(_solver_arguments(args, stems))

    if args.all and not args.no_report:
        print()
        summarize(args.results_dir, grid, args.summary_pattern, verbose=False)

    return 0 if result is None else result


if __name__ == "__main__":
    sys.exit(main())
