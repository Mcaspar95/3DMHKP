#!/usr/bin/env python3
"""1:1 comparison against Fig. 11 of Paquay, Limbourg & Schyns (2018).

    python compare_paquay_fig11.py --dir results_Paquay_SA

WHERE THE PAPER'S NUMBERS COME FROM
===================================
Fig. 11 of the paper is a set of ten horizontal boxplots, one per sample size,
of the filling rate of EVERY ULD used across the 30 instances of that size. The
figure is drawn as vector graphics and every boxplot is annotated in the figure
itself with its five summary values. Those printed annotations -- not pixel
measurements -- are transcribed into PAPER below.

The y-axis label of each row is "<n> (<k>)" where k is the number of ULDs used
for all 30 instances of that size. Those ten counts sum to exactly 1010, which
is the total the paper states in Sec. 6.3.2 ("among the 1010 used ULDs"). That
agreement is a check on the transcription: it is reproduced by the --selftest
option below.

WHAT IS AND IS NOT 1:1
======================
Comparable, on identical instances and an identical statistic:
  * the number of ULDs used per sample size, and in total,
  * the five-number summary of the per-ULD filling rate.

Our filling rate is computed exactly as theirs: packed box volume divided by
the ULD's own volume, one observation per USED ULD, pooled over the 30
instances of a size. Note this makes filling rate and ULD count interlinked --
using more ULDs for the same boxes mechanically lowers the mean filling rate,
so the two columns must be read together, never separately.

Pass --cg-dir to show a run made with Paquay-SA.py --cg alongside the
unconstrained one. That column enforces the paper's two centre-of-gravity
constraints (maximum CG height, and the allowable lateral area their jump &
shift operators repair) and is the genuinely like-for-like comparison; the
plain column is the same solver on the relaxed problem, kept so the cost of
the constraints is visible.

NOT comparable without qualification, even with --cg-dir:
  * The numeric CG tolerances are inferred from columns 6-8 of uld.csv, not
    quoted from the paper, which defers them to Paquay et al. (2016) and a
    Boeing manual. See the note in Paquay-SA.py.
  * Their heuristic is a single deterministic construction; ours is an
    anytime metaheuristic given 12 s, which is their reported worst case.
"""

import argparse
import json
import os
import sys

# Transcribed from the value annotations printed in Fig. 11 (p. 62).
# size -> (ulds, min, Q1, median, Q3, max)
PAPER = {
    10:  (47,  6.99, 24.59, 33.17, 40.71, 61.10),
    20:  (54,  7.13, 28.50, 34.58, 42.74, 62.88),
    30:  (72, 14.58, 27.48, 36.56, 43.74, 63.87),
    40:  (83,  5.09, 29.57, 37.85, 48.50, 65.01),
    50:  (90,  0.90, 26.69, 42.15, 50.61, 63.37),
    60:  (108, 1.31, 29.00, 41.45, 50.01, 65.02),
    70:  (126, 5.09, 29.55, 45.74, 54.41, 67.76),
    80:  (129, 2.27, 32.39, 46.98, 55.09, 71.18),
    90:  (141, 6.19, 35.01, 47.48, 56.23, 73.44),
    100: (160, 5.09, 33.51, 50.17, 58.95, 72.78),
}
PAPER_TOTAL = 1010          # Sec. 6.3.2, and the sum of the counts above
SIZES = sorted(PAPER)


def quantile(sorted_vals, p):
    """Linear-interpolation quantile, the usual boxplot convention."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    h = (len(sorted_vals) - 1) * p
    lo = int(h)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (h - lo) * (sorted_vals[hi] - sorted_vals[lo])


def five(vals):
    s = sorted(vals)
    return (min(s), quantile(s, .25), quantile(s, .5), quantile(s, .75), max(s))


def load(d):
    if not os.path.isdir(d):
        sys.exit("no such directory: %s" % d)
    rows = []
    for fn in os.listdir(d):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as fh:
                rows.append(json.load(fh))
    if not rows:
        sys.exit("no result files in %s" % d)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results_Paquay_SA")
    ap.add_argument("--cg-dir", help="a second run with CG enforced, shown "
                                     "alongside as the like-for-like column")
    ap.add_argument("--selftest", action="store_true",
                    help="check the transcribed ULD counts sum to 1010")
    args = ap.parse_args()

    if args.selftest:
        tot = sum(v[0] for v in PAPER.values())
        print("transcribed ULD counts sum to %d, paper states %d -> %s"
              % (tot, PAPER_TOTAL, "OK" if tot == PAPER_TOTAL else "MISMATCH"))
        return 0 if tot == PAPER_TOTAL else 1

    rows = load(args.dir)
    have = {n: [r for r in rows if r["n"] == n] for n in SIZES}
    cgrows = load(args.cg_dir) if args.cg_dir else None
    cghave = ({n: [r for r in cgrows if r["n"] == n] for n in SIZES}
              if cgrows else None)
    missing = [n for n in SIZES if len(have[n]) != 30]
    if missing:
        print("WARNING: incomplete sizes %s -- per-size rows for those are not"
              % missing)
        print("         comparable to the paper, which pools all 30 instances.\n")

    print("ULDs USED  (fewer is better; same 30 instances per row)")
    if cghave:
        print("  %5s %10s %12s %12s" % ("n", "Paquay", "ours (no CG)",
                                        "ours (CG)"))
    else:
        print("  %5s %10s %10s %10s" % ("n", "Paquay", "ours", "delta"))
    t_paper = t_ours = t_cg = t_paper_cg = 0
    for n in SIZES:
        if not have[n]:
            continue
        ours = sum(r["ulds"] for r in have[n])
        pap = PAPER[n][0]
        t_paper += pap
        t_ours += ours
        if cghave:
            c = sum(r["ulds"] for r in cghave[n]) if cghave[n] else 0
            t_cg += c
            if cghave[n]:
                t_paper_cg += pap      # only sizes the CG run actually covers
            print("  %5d %10d %12s %12s"
                  % (n, pap, "%d (%+d)" % (ours, ours - pap),
                     "%d (%+d)" % (c, c - pap) if cghave[n] else "-"))
        else:
            print("  %5d %10d %10d %+10d" % (n, pap, ours, ours - pap))
    print("  " + "-" * (46 if cghave else 38))
    if cghave:
        print("  %5s %10d %12s %12s"
              % ("total", t_paper,
                 "%d (%+.1f%%)" % (t_ours, 100.0 * (t_ours - t_paper) / t_paper),
                 "-" if not t_paper_cg else
                 "%d (%+.1f%%)" % (t_cg, 100.0 * (t_cg - t_paper_cg) / t_paper_cg)))
        if t_paper_cg and t_paper_cg != t_paper:
            print("  NOTE: the CG column covers only the sizes it has results "
                  "for (paper total %d\n        for those sizes), so its %% is "
                  "not over all 300 instances." % t_paper_cg)
    else:
        print("  %5s %10d %10d %+10d   (%.1f%%)"
              % ("total", t_paper, t_ours, t_ours - t_paper,
                 100.0 * (t_ours - t_paper) / t_paper))

    print("\nFILLING RATE PER USED ULD  (higher is better)")
    print("  %5s %7s %8s %8s %8s %8s %8s"
          % ("n", "who", "min", "Q1", "median", "Q3", "max"))
    for n in SIZES:
        if not have[n]:
            continue
        fills = [f for r in have[n] for f in r["fills"]]
        o = five(fills)
        p = PAPER[n][1:]
        print("  %5d %7s %8.2f %8.2f %8.2f %8.2f %8.2f" % ((n, "Paquay") + p))
        print("  %5s %7s %8.2f %8.2f %8.2f %8.2f %8.2f" % (("", "ours") + o))
        if cghave and cghave[n]:
            cf = [f for r in cghave[n] for f in r["fills"]]
            co = five(cf)
            print("  %5s %7s %8.2f %8.2f %8.2f %8.2f %8.2f"
                  % (("", "ours+CG") + co))
            print("  %5s %7s %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f"
                  % (("", "d(CG)") + tuple(co[i] - p[i] for i in range(5))))
        else:
            print("  %5s %7s %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f"
                  % (("", "delta") + tuple(o[i] - p[i] for i in range(5))))
    # pooled
    allf = [f for n in SIZES for r in have[n] for f in r["fills"]]
    if allf:
        o = five(allf)
        print("  " + "-" * 60)
        print("  %5s %7s %8.2f %8.2f %8.2f %8.2f %8.2f"
              % (("all", "ours") + o))
        print("  (no pooled figure is published, so this row has no counterpart)")

    wins = sum(1 for n in SIZES if have[n]
               and sum(r["ulds"] for r in have[n]) < PAPER[n][0])
    med_better = sum(1 for n in SIZES if have[n] and
                     five([f for r in have[n] for f in r["fills"]])[2] > PAPER[n][3 - 1])
    print("\nSUMMARY")
    print("  sizes where we use fewer ULDs   : %d of %d" % (wins, len(SIZES)))
    print("  sizes with higher median filling: %d of %d" % (med_better, len(SIZES)))
    print("\n  CAVEAT: CG height and lateral-balance constraints are enforced by")
    print("  Paquay et al. and NOT by our solver, so the comparison above is")
    print("  favourable to us by an unquantified margin. See module docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
