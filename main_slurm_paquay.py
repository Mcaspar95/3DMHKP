#!/usr/bin/env python3
"""SLURM array-job dispatcher for Paquay-SA.py over the 300 final instances of
Paquay, Limbourg & Schyns (2018), EJOR 267:52-64.

The instances under Paquay/ are the authors' own files (see Paquay/README.md
for provenance); Paquay-SA.py solves their actual problem -- every box packed
into ULDs of six types, minimise selected ULD volume -- and with --cg also
enforces their two centre-of-gravity constraints. Run WITH --cg: that is the
like-for-like setting, and it is the default here.

GRID
====
One array task solves one instance, so the grid is 300 jobs:

    #SBATCH --array=0-299

Task id maps to (sample size, sample index) in a fixed order -- sizes
10, 20, ..., 100, and within each size samples 0..29 -- so a given id always
means the same instance:

    id   0.. 29  ->  n=10,  sample 0..29
    id  30.. 59  ->  n=20,  sample 0..29
    ...
    id 270..299  ->  n=100, sample 0..29

Each task writes one JSON report per instance into --results-dir. No task sees
more than its own instance, so the comparison against the paper is assembled
afterwards from those reports by --summarize.

SUMMARY
=======
--summarize reproduces the comparison table against Fig. 11 of the paper: ULDs
used per sample size against the paper's own counts, and the five-number
summary of the per-ULD filling rate against the values printed in that figure.
The reference values live in compare_paquay_fig11.py and are imported here so
the two cannot drift apart.

Examples:
    python3 main_slurm_paquay.py --print-grid
    python3 main_slurm_paquay.py --id 0
    python3 main_slurm_paquay.py --instance n10sample7
    python3 main_slurm_paquay.py --n 10 --all         # one size in this process
    python3 main_slurm_paquay.py --all                # all 300, sequentially
    python3 main_slurm_paquay.py --summarize
    python3 main_slurm_paquay.py --summarize --n 10   # just the n=10 row
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_Paquay_SA_cg"

MODULE_NAME = "Paquay-SA"

# 12 s per instance: Paquay et al. report their heuristic's MAXIMUM runtime as
# "does not exceed 12 seconds for any instance" (Sec. 6.3.3), so this matches
# their worst case rather than being an arbitrary budget. Note the asymmetry
# that must be stated in any write-up: theirs is one deterministic
# construction, ours is an anytime metaheuristic that uses the whole budget.
TIME_LIMIT = 12.0

SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SAMPLES_PER_SIZE = 30

_SOLVER = None
_FIG11 = None


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_solver():
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = _load_module(MODULE_NAME, ROOT / f"{MODULE_NAME}.py")
    return _SOLVER


def _get_fig11():
    """compare_paquay_fig11.py, for the paper's reference values and the
    quantile convention. Imported rather than duplicated so the numbers in the
    summary below cannot drift from the ones in the comparison script."""
    global _FIG11
    if _FIG11 is None:
        _FIG11 = _load_module("compare_paquay_fig11",
                              ROOT / "compare_paquay_fig11.py")
    return _FIG11


def build_grid():
    """(n, sample) for every task id, in a fixed order."""
    return [(n, k) for n in SIZES for k in range(SAMPLES_PER_SIZE)]


def stem_of(n, k):
    return f"n{n}sample{k}"


def parse_stem(text):
    m = re.fullmatch(r"n(\d+)sample(\d+)", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# =========================================================================
# Solving
# =========================================================================

def solve_one(args, n, k):
    """Solve one instance and write its JSON report. Returns the row dict."""
    sa = _get_solver()
    from load_paquay import load_boxes, load_ulds, instance_path

    ulds = [sa.Uld(u) for u in load_ulds()]
    path = instance_path(n, k, args.training)
    if not Path(path).exists():
        raise SystemExit(f"missing instance file: {path}")
    boxes = load_boxes(path)

    res = sa.solve_instance(boxes, ulds, args.time_limit,
                            seed=args.seed + k, min_frac=args.support,
                            greedy=args.greedy, cg=not args.no_cg)
    if res is None:
        print(f"{stem_of(n, k)}: INFEASIBLE decode", file=sys.stderr)
        return None

    bins = res["bins"]
    row = {
        "n": n,
        "sample": k,
        "ulds": len(bins),
        "obj": res["obj"],
        "fills": [b.filling_rate() for b in bins],
        "seconds": res["seconds"],
        "iters": res["iters"],
        "types": [b.uld.code for b in bins],
        "cg_enforced": not args.no_cg,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"{stem_of(n, k)}{args.tag}.json"
    out.write_text(json.dumps(row, indent=1))
    return row


# =========================================================================
# Summary over the written reports
# =========================================================================

def read_reports(results_dir, pattern="*.json"):
    """Every parsable report in results_dir, best (fewest ULDs) per instance.

    A sweep can leave several reports for one instance -- a 12 s run next to a
    60 s one -- so the best is taken per (n, sample), where best means fewest
    ULDs and, on a tie, smaller selected volume. That mirrors the objective.
    """
    best = {}
    n_files = 0
    for path in sorted(results_dir.glob(pattern)):
        try:
            row = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if "n" not in row or "sample" not in row or "ulds" not in row:
            continue
        n_files += 1
        key = (row["n"], row["sample"])
        prev = best.get(key)
        if prev is None or (row["ulds"], row["obj"]) < (prev["ulds"], prev["obj"]):
            best[key] = row
    return best, n_files


def summarize(results_dir, only_n=None, pattern="*.json"):
    """Print the comparison against Fig. 11 of the paper."""
    if not results_dir.is_dir():
        print(f"no results directory {results_dir}", file=sys.stderr)
        return 1

    fig = _get_fig11()
    best, n_files = read_reports(results_dir, pattern)
    if not best:
        print(f"no parsable reports in {results_dir} matching {pattern!r}",
              file=sys.stderr)
        return 1

    sizes = [only_n] if only_n else SIZES
    rows = {n: [r for (nn, _k), r in best.items() if nn == n] for n in sizes}
    covered = [n for n in sizes if rows[n]]
    if not covered:
        print(f"no reports for size(s) {sizes}", file=sys.stderr)
        return 1

    enforced = {r.get("cg_enforced") for rs in rows.values() for r in rs}
    cg_note = ("CG ENFORCED" if enforced == {True}
               else "CG NOT enforced" if enforced == {False}
               else "MIXED cg settings -- not a single comparable run")

    print(f"Paquay et al. (2018) final data sets -- {cg_note}")
    print(f"reports: {n_files} file(s), {len(best)} distinct instances\n")

    print("ULDs USED  (fewer is better)")
    print("  %5s %8s %10s %10s %9s" % ("n", "inst", "Paquay", "ours", "delta"))
    t_paper = t_ours = 0
    incomplete = []
    for n in covered:
        got = len(rows[n])
        if got != SAMPLES_PER_SIZE:
            incomplete.append((n, got))
        ours = sum(r["ulds"] for r in rows[n])
        pap = fig.PAPER[n][0]
        t_paper += pap
        t_ours += ours
        print("  %5d %8d %10d %10d %+9d" % (n, got, pap, ours, ours - pap))
    print("  " + "-" * 46)
    print("  %5s %8d %10d %10d %+9d  (%+.1f%%)"
          % ("total", sum(len(rows[n]) for n in covered), t_paper, t_ours,
             t_ours - t_paper, 100.0 * (t_ours - t_paper) / t_paper))
    if incomplete:
        print("  WARNING: incomplete sizes %s -- the paper's counts pool all "
              "30\n           instances, so these rows are not comparable."
              % ", ".join("n=%d (%d/30)" % t for t in incomplete))

    print("\nFILLING RATE PER USED ULD  (higher is better)")
    print("  %5s %8s %8s %8s %8s %8s %8s"
          % ("n", "who", "min", "Q1", "median", "Q3", "max"))
    for n in covered:
        fills = [f for r in rows[n] for f in r["fills"]]
        ours = fig.five(fills)
        pap = fig.PAPER[n][1:]
        print("  %5d %8s %8.2f %8.2f %8.2f %8.2f %8.2f" % ((n, "Paquay") + pap))
        print("  %5s %8s %8.2f %8.2f %8.2f %8.2f %8.2f" % (("", "ours") + ours))
        print("  %5s %8s %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f"
              % (("", "delta") + tuple(ours[i] - pap[i] for i in range(5))))

    secs = [r["seconds"] for rs in rows.values() for r in rs]
    print("\n  runtime: %.1f s max per instance (paper: < 12 s)" % max(secs))
    print("\n  Fig. 11 reference values are transcribed in "
          "compare_paquay_fig11.py;")
    print("  run `python3 compare_paquay_fig11.py --selftest` to check them.")
    if enforced == {False}:
        print("\n  CAVEAT: these results do NOT enforce the paper's "
              "centre-of-gravity\n  constraints, so they are not like-for-like. "
              "Re-run without --no-cg.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SLURM dispatcher for Paquay-SA.py over the 300 Paquay "
                    "et al. (2018) final instances.")
    parser.add_argument("--id", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID, zero-based (0..299).")
    parser.add_argument("--instance", default=None, metavar="STEM",
                        help="instance stem, e.g. n10sample7.")
    parser.add_argument("--n", type=int, default=None, choices=SIZES,
                        help="restrict to one sample size (with --all, or "
                             "with --summarize).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"directory for the JSON reports "
                             f"(default: {RESULTS_DIR.name})")
    parser.add_argument("--time-limit", "--timelimit", dest="time_limit",
                        type=float, default=TIME_LIMIT,
                        help=f"SA seconds per instance (default: "
                             f"{TIME_LIMIT:g}; the paper's reported maximum)")
    parser.add_argument("--num_cpu", type=int, default=1,
                        help="CPUs requested from SLURM. SA is single-threaded, "
                             "so this is accepted for symmetry with the other "
                             "dispatchers and otherwise unused.")
    parser.add_argument("--no-cg", action="store_true",
                        help="do NOT enforce the centre-of-gravity "
                             "constraints. Off by default: leaving them off "
                             "makes the results not comparable to the paper.")
    parser.add_argument("--support", type=float, default=1.0,
                        help="required support fraction under a box "
                             "(default 1.0, i.e. fully supported).")
    parser.add_argument("--training", action="store_true",
                        help="use the 150 training instances instead of the "
                             "300 final ones (no published figures to compare "
                             "against).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="",
                        help="extra suffix for the report filenames, to keep "
                             "a sweep's runs apart.")
    parser.add_argument("--greedy", action="store_true",
                        help="decoder only, no SA (the constructive baseline).")
    parser.add_argument("--all", action="store_true",
                        help="solve every instance of the grid sequentially in "
                             "this one process (restrict with --n), instead of "
                             "one instance per array task.")
    parser.add_argument("--summarize", action="store_true",
                        help="do not solve: read the reports already in "
                             "--results-dir and print the comparison against "
                             "Fig. 11 of the paper.")
    parser.add_argument("--summary-pattern", default="*.json", metavar="GLOB",
                        help="which reports --summarize reads "
                             "(default: *.json).")
    parser.add_argument("--print-grid", action="store_true",
                        help="print the grid and the matching SBATCH array "
                             "range, then exit.")
    args = parser.parse_args(argv)

    if args.summarize:
        return summarize(args.results_dir, args.n, args.summary_pattern)

    grid = build_grid()

    if args.print_grid:
        print(f"{len(grid)} instances from Paquay/"
              f"{'training_data_sets' if args.training else 'Final_instances'}"
              f", solver {MODULE_NAME}.py, {args.time_limit:g}s each, "
              f"CG {'OFF' if args.no_cg else 'ENFORCED'}")
        for i, (n, k) in enumerate(grid):
            print(f"  {i:3d}  {stem_of(n, k)}")
        print(f"\n#SBATCH --array=0-{len(grid) - 1}")
        return 0

    if args.all:
        todo = [(n, k) for (n, k) in grid if args.n is None or n == args.n]
    elif args.id is not None:
        if not 0 <= args.id < len(grid):
            parser.error(f"--id {args.id} out of range [0, {len(grid) - 1}]. "
                         f"Grid has {len(grid)} instances.")
        todo = [grid[args.id]]
    elif args.instance is not None:
        parsed = parse_stem(args.instance)
        if parsed is None or parsed not in grid:
            parser.error(f"unknown instance {args.instance!r}. "
                         f"Use --print-grid to list the {len(grid)} available.")
        todo = [parsed]
    else:
        parser.error("provide --id, --instance, --all or --summarize")

    print("=" * 62)
    print(f"  Solver:     {MODULE_NAME}.py (3D MBSBPP: every box packed, "
          f"minimise selected ULD volume)")
    if len(todo) == 1:
        print(f"  Instance:   {stem_of(*todo[0])}")
    else:
        print(f"  Instances:  {len(todo)} "
              f"({stem_of(*todo[0])} .. {stem_of(*todo[-1])})")
    print(f"  CG:         {'NOT enforced' if args.no_cg else 'enforced'}")
    print(f"  Timelimit:  {args.time_limit:g}s per instance")
    print(f"  Reports:    {args.results_dir}")
    print(f"  CPUs:       {args.num_cpu}")
    print("=" * 62, flush=True)

    failed = 0
    for n, k in todo:
        row = solve_one(args, n, k)
        if row is None:
            failed += 1
            continue
        if len(todo) > 1:
            print("  %-14s ULDs %2d  %-24s vol %7.2f m3  "
                  "mean fill %5.2f%%  %5.1fs"
                  % (stem_of(n, k), row["ulds"], ",".join(row["types"])[:24],
                     row["obj"] / 1e9,
                     sum(row["fills"]) / len(row["fills"]), row["seconds"]),
                  flush=True)

    if len(todo) > 1:
        print()
        summarize(args.results_dir, args.n, args.summary_pattern)
    if failed:
        print(f"\n{failed} instance(s) had no feasible decode", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
