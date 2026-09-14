#!/usr/bin/env python3
"""Comparison against Paquay, Limbourg & Schyns (2018), EJOR 267:52-64.

    python compare_paquay.py                          # results_Paquay_SA
    python compare_paquay.py --dir results_Paquay_SA_t30s
    python compare_paquay.py --dir A --baseline B     # two of our own runs

WHAT CAN AND CANNOT BE COMPARED
===============================
This is the awkward part of this benchmark, and the reason this script exists
separately from the solver.

Paquay et al. report their filling rates as BOXPLOTS (Fig. 11), one per sample
size, not as a table of numbers. There is therefore NO published per-size mean
filling rate to difference against. Anyone quoting "Paquay et al. achieve x%"
per size is reading pixels off a figure.

What the paper does state numerically, and what this script therefore uses:

  * 1010 ULDs used in total across the 300 final instances (Sec. 6.3.2, given
    as the denominator of the CG deviation counts: "among the 1010 used ULDs").
  * Maximum runtime below 12 s per instance, even at 100 boxes (Sec. 6.3.3).
  * At 100 boxes, half the ULDs reach at least 50% filling, and the single best
    ULD reaches 72.78% (Sec. 6.3.1).
  * 75.35% of ULDs unbalanced before their jump & shift repair, 12.87% after
    (Table 3).

Total ULD count is the one clean, unambiguous, like-for-like number. It is also
the right one: the objective is to minimise selected ULD volume, and fewer ULDs
for the same boxes is strictly better.

CAVEAT that must accompany any comparison made with this script: our solver
does NOT enforce the CG constraints (height limit, allowable lateral area).
Paquay et al. do. Packing without those constraints is easier, so a favourable
ULD count here is not yet proof of a better method. The CG diagnostics printed
by Paquay-SA.py --report-cg indicate how much of our advantage would be at risk
if they were enforced. State this whenever the numbers are used.
"""

import argparse
import json
import os
import sys

# Paper reference points, Paquay et al. (2018).
PAPER_TOTAL_ULDS = 1010          # Sec. 6.3.2, "among the 1010 used ULDs"
PAPER_MAX_SECONDS = 12           # Sec. 6.3.3
PAPER_BEST_FILL_100 = 72.78      # Sec. 6.3.1, best single ULD at n=100
PAPER_MEDIAN_FILL_100 = 50.0     # Sec. 6.3.1, "half ... at least 50%"
SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load_run(d):
    rows = []
    if not os.path.isdir(d):
        sys.exit("no such directory: %s" % d)
    for fn in os.listdir(d):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as fh:
                rows.append(json.load(fh))
    if not rows:
        sys.exit("no result files in %s" % d)
    return rows


def stats(rows):
    per_size = {}
    for n in SIZES:
        grp = [r for r in rows if r["n"] == n]
        if not grp:
            continue
        fills = [f for r in grp for f in r["fills"]]
        per_size[n] = {
            "instances": len(grp),
            "ulds": sum(r["ulds"] for r in grp),
            "fills": sorted(fills),
            "seconds": [r["seconds"] for r in grp],
            "vol": sum(r["obj"] for r in grp) / 1e9,
        }
    return per_size


def q(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def report(name, per_size):
    print("\n%s" % name)
    print("  %4s %6s %7s %10s %8s %8s %8s %8s %7s" %
          ("n", "inst", "ULDs", "vol[m3]", "Q1", "median", "Q3", "max", "max s"))
    tot_u = 0
    allf = []
    maxs = 0.0
    for n, s in sorted(per_size.items()):
        f = s["fills"]
        tot_u += s["ulds"]
        allf.extend(f)
        maxs = max(maxs, max(s["seconds"]))
        print("  %4d %6d %7d %10.2f %8.2f %8.2f %8.2f %8.2f %7.2f"
              % (n, s["instances"], s["ulds"], s["vol"] / s["instances"],
                 q(f, .25), q(f, .5), q(f, .75), max(f), max(s["seconds"])))
    allf.sort()
    print("  " + "-" * 76)
    print("  TOTAL ULDs %d | mean fill %.2f%% | median %.2f%% | max %.2f%% | "
          "max %.1f s" % (tot_u, sum(allf) / len(allf), q(allf, .5),
                          max(allf), maxs))
    return tot_u, allf, maxs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results_Paquay_SA")
    ap.add_argument("--baseline", help="a second run of ours to compare against")
    args = ap.parse_args()

    rows = load_run(args.dir)
    per = stats(rows)
    tot, allf, maxs = report("OUR SA  [%s]" % args.dir, per)

    if args.baseline:
        brows = load_run(args.baseline)
        btot, ballf, bmaxs = report("BASELINE  [%s]" % args.baseline,
                                    stats(brows))
        print("\n  delta ULDs: %+d (%.1f%%)" % (tot - btot,
                                                100.0 * (tot - btot) / btot))

    n_inst = sum(s["instances"] for s in per.values())
    print("\nPAQUAY ET AL. (2018) reference points")
    print("  total ULDs over the 300 final instances : %d" % PAPER_TOTAL_ULDS)
    print("  max runtime per instance                : < %d s" % PAPER_MAX_SECONDS)
    print("  n=100, median ULD filling rate          : ~%.0f%%" % PAPER_MEDIAN_FILL_100)
    print("  n=100, best single ULD filling rate     : %.2f%%" % PAPER_BEST_FILL_100)

    if n_inst == 300:
        d = tot - PAPER_TOTAL_ULDS
        verdict = "fewer" if d < 0 else ("more" if d > 0 else "equal")
        print("\n  ULD count: ours %d vs paper %d  -> %+d (%s, %.1f%%)"
              % (tot, PAPER_TOTAL_ULDS, d, verdict,
                 100.0 * abs(d) / PAPER_TOTAL_ULDS))
        if 100 in per:
            f100 = per[100]["fills"]
            print("  n=100 median fill: ours %.2f%% vs paper ~%.0f%%"
                  % (q(f100, .5), PAPER_MEDIAN_FILL_100))
            print("  n=100 best ULD   : ours %.2f%% vs paper %.2f%%"
                  % (max(f100), PAPER_BEST_FILL_100))
        print("  runtime          : ours %.1f s max vs paper < %d s"
              % (maxs, PAPER_MAX_SECONDS))
    else:
        print("\n  (partial run: %d of 300 instances -- the 1010-ULD total is"
              % n_inst)
        print("   only comparable on the complete final data set)")

    print("\nCAVEAT: our solver does not enforce the CG height/balance")
    print("constraints that Paquay et al. enforce. Any ULD-count advantage is")
    print("therefore obtained on a weaker problem. See the module docstring.")


if __name__ == "__main__":
    main()
