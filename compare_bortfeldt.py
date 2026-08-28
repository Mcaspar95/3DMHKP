# Comparison against Bortfeldt (2000), OR Spektrum 22:239-261, Table 2 -
# the 16 value-maximizing Mohanty, Mathur & Ivancic (1994) problems.
# =========================================================================
# Bortfeldt's Table 2 reports, per problem, an upper bound on the stowable
# value together with the values reached by his heuristic (MCL) and by the
# procedure of Mohanty et al. The comparable figure is therefore
#
#       value / upper bound
#
# and NOT raw value, since the value scale differs by three orders of
# magnitude across the 16 instances. This script puts four methods side by
# side on that scale:
#
#       Mohanty  - Mohanty et al. (1994), best result, as cited by Bortfeldt
#       MCL      - Bortfeldt's own heuristic
#       MILP     - the exact model in 3DMHKP.py (incumbent after ~120 s)
#       SA       - the simulated annealing in 3DMHKP-SA.py (60 s)
#
# The upper-bound column printed here is NOT transcribed from the paper: it is
# recomputed by continuous_knapsack_bound() and read back out of our own report
# files. Bortfeldt's published bounds are transcribed into BORTFELDT_BOUNDS
# purely so that main() can assert the two agree - the percentage columns
# compare like with like only if we divide by his divisor. They agree exactly
# on all 16 instances.
#
# The Mohanty and MCL percentages are recomputed here from the transcribed
# absolute values rather than transcribed themselves, so they can differ from
# the printed percentages in the paper by 0.1 through rounding.
#
# CAVEAT on comparability: Bortfeldt (p. 240) additionally requires every box
# to be at least partially supported from below. Neither our SA decoder nor the
# MILP in 3DMHKP.py enforces that, so both solve a slight relaxation of his
# problem. Measured on the SA results, 2 of 996 packed boxes are unsupported,
# both in instance 15 (280.0 of its 37315.5 value); everywhere else the
# comparison is like for like.
#
#   python compare_bortfeldt.py                  # the untagged SA run
#   python compare_bortfeldt.py --sa-tag 30s     # a run written with --tag 30s
#   python compare_bortfeldt.py --sa-tag "" 30s  # both, side by side
#
import argparse
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results_3DMHKP"

# ---------------------------------------------------------------------------
# Transcribed from Bortfeldt (2000), Table 2, "Verstauter Kistenwert / absolut"
# ---------------------------------------------------------------------------
BORTFELDT_VALUES = {
    1: 8640.0,     2: 85120.0,    3: 53262.5,    4: 2333440.0,
    5: 581250.0,   6: 139584.0,   7: 17409.0,    8: 68645.6,
    9: 128952.0,  10: 15360.0,   11: 53202.8,   12: 24235.2,
    13: 36556.8,  14: 65316.8,   15: 39727.2,   16: 595770.0,
}
MOHANTY_VALUES = {
    1: 8640.0,     2: 83494.4,    3: 53262.5,    4: 2333440.0,
    5: 495500.0,   6: 138240.0,   7: 16668.0,    8: 65741.6,
    9: 119772.0,  10: 15360.0,   11: 49995.0,   12: 23529.0,
    13: 36556.8,  14: 56492.8,   15: 37558.8,   16: 556458.0,
}

# ---------------------------------------------------------------------------
# Transcribed from Kurpel, Scarpin, Pecora, Schenekemberg & Coelho (2020),
# EJOR 284:87-107, Table 12 ("This paper" column) - an EXACT 0-1 model over
# discretized placement points, 3600 s per instance. This is the strongest
# published result set on these 16 instances.
# ---------------------------------------------------------------------------
KURPEL_VALUES = {
    1: 9216.0,     2: 85555.2,    3: 53262.5,    4: 1354752.0,
    5: 583750.0,   6: 142464.0,   7: 17664.0,    8: 71972.4,
    9: 98748.0,   10: 15360.0,   11: 54761.0,   12: 24076.8,
    13: 36556.8,  14: 68723.2,   15: 40807.8,   16: 632274.0,
}

# The instances Kurpel et al. solved to PROVEN optimality (bold in their
# Table 12, gap 0.00). For these the true optimum is known, which is a far
# more informative yardstick than the loose continuous-knapsack bound.
KURPEL_OPTIMAL = {1, 2, 3, 4, 5, 10, 11, 13, 15}

# Instance 4 is NOT comparable and is excluded from every aggregate below.
# Kurpel et al. flag their own value with an asterisk and argue in their
# Appendix F that the values reported by Mohanty, Bortfeldt, Takahara and Ren
# are unattainable. They are, on the standard instance - the discrepancy is in
# the DATA. Their Appendix F describes instance 4 as ten containers,
# 5 x (60,40,72) and 5 x (40,36,52), totalling 1,238,400 volume units. The
# instance in Mohanty/instance04.txt has a third container type, 5 x (60,52,64),
# for 2,236,800 units in total - exactly their 1,238,400 plus the 998,400 of
# the missing type. Two independent checks say the 15-container version is the
# one the earlier literature used: continuous_knapsack_bound() on it returns
# 2,720,640, reproducing Bortfeldt's published Table 2 bound exactly (his bound
# is unreachable with only 1,238,400 units, whose value ceiling is 1,733,760),
# and our own SA attains 2,333,440 with a packing that passes
# verify_placement(). So Kurpel et al. solved a truncated instance 4 optimally.
KURPEL_INCOMPARABLE = {4}

# Kurpel et al., Table 12, "Time (second)". Their run used a 2.77 GHz Xeon with
# up to 120 GB RAM and Gurobi 8.1.0, with a 3600 s limit per instance. Summing
# these reproduces the 1593.66 s average printed in their table exactly, which
# is the check that the transcription is right.
KURPEL_TIMES = {
    1: 20.14,    2: 8.75,     3: 2.73,     4: 0.07,
    5: 0.43,     6: 3600.0,   7: 3600.0,   8: 3600.0,
    9: 3600.0,  10: 0.58,    11: 170.85,  12: 3600.0,
    13: 71.59,  14: 3600.0,  15: 23.46,   16: 3600.0,
}
KURPEL_TIME_LIMIT = 3600.0

# Bortfeldt's "Obere Schranke des verstaubaren Kistenwertes" column, kept only
# to assert that our recomputed bound really is his (see main()).
BORTFELDT_BOUNDS = {
    1: 11112.0,    2: 86016.0,    3: 53500.0,    4: 2720640.0,
    5: 653750.0,   6: 143424.0,   7: 20203.2,    8: 77986.8,
    9: 139356.0,  10: 15360.0,   11: 68353.2,   12: 24964.0,
    13: 36556.8,  14: 71552.0,   15: 42922.8,   16: 666829.6,
}

# The means Bortfeldt prints in the last row of Table 2, for cross-checking our
# recomputed ones.
PAPER_MEANS = {"Mohanty": 87.7, "MCL": 91.4}


def read_report(path):
    """Pull the headline figures out of one of our report files."""
    if not path.exists():
        return None
    out = {}
    for line in path.open():
        if line.startswith("Packed value"):
            out["value"] = float(line.split(":")[1])
        elif line.startswith("Continuous-knapsack bound"):
            out["bound"] = float(line.split(":")[1])
        elif line.startswith("Volume utilization"):
            out["util"] = float(line.rsplit("=", 1)[1].strip().rstrip("%"))
        elif line.startswith("Runtime"):
            out["runtime"] = float(line.split(":")[1].strip().split()[0])
    return out or None


def head_to_head(name_a, a, name_b, b):
    """Per-instance win/tie/loss of method a against method b."""
    win = sorted(n for n in a if n in b and a[n] > b[n] + 1e-6)
    tie = sorted(n for n in a if n in b and abs(a[n] - b[n]) <= 1e-6)
    loss = sorted(n for n in a if n in b and a[n] < b[n] - 1e-6)
    print(f"  {name_a} vs {name_b}: better on {len(win)} {win}, "
          f"tied on {len(tie)} {tie}, worse on {len(loss)} {loss}")


def sa_label(tag):
    return f"SA {tag}" if tag else "SA"


def main():
    parser = argparse.ArgumentParser(
        description="Compare our solvers against Bortfeldt (2000) Table 2.")
    parser.add_argument("--sa-tag", nargs="*", default=[""], metavar="TAG",
                        help="which SA runs to show, by the --tag they were "
                             "written with; empty string is the untagged run. "
                             "Pass several to get one column each.")
    args = parser.parse_args()
    sa_tags = args.sa_tag or [""]

    # sa_val[tag][instance]; the first tag listed is the primary run, i.e. the
    # one whose absolute values and utilization get their own columns.
    sa_val = {tag: {} for tag in sa_tags}
    milp_val, bounds, util, runtime = {}, {}, {}, {}
    for n in range(1, 17):
        for tag in sa_tags:
            suffix = f"-{tag}" if tag else ""
            rep = read_report(RESULTS_DIR / f"instance{n:02d}-3DMHKP-SA{suffix}.txt")
            if rep is None:
                print(f"missing {sa_label(tag)} report for instance {n:02d} - run "
                      f"3DMHKP-SA.py{f' --tag {tag}' if tag else ''} first",
                      file=sys.stderr)
                return 1
            sa_val[tag][n] = rep["value"]
            if tag == sa_tags[0]:
                bounds[n] = rep["bound"]
                util[n] = rep["util"]
                runtime[n] = {"SA": rep["runtime"]}

        milp = read_report(RESULTS_DIR / f"instance{n:02d}-3DMHKP.txt")
        if milp:
            milp_val[n] = milp["value"]
            runtime[n]["MILP"] = milp["runtime"]

    # The percentage columns only mean anything if our recomputed bound is the
    # divisor Bortfeldt's percentages are taken over.
    mismatched = [n for n in bounds if abs(bounds[n] - BORTFELDT_BOUNDS[n]) > 0.05]
    if mismatched:
        print(f"WARNING: recomputed bound differs from Bortfeldt's Table 2 on "
              f"instances {mismatched} - the %-columns are not comparable",
              file=sys.stderr)
    else:
        print("(recomputed upper bounds match Bortfeldt's Table 2 exactly "
              "on all 16 instances)")

    def pct(values, n):
        return 100.0 * values[n] / bounds[n] if n in values else None

    def cell(p, width=13):
        return f"{p:>{width - 1}.1f}%" if p is not None else f"{'-':>{width}}"

    method_cols = [("Mohanty", MOHANTY_VALUES), ("MCL", BORTFELDT_VALUES),
                   ("KUR", KURPEL_VALUES), ("MILP", milp_val)]
    method_cols += [(sa_label(tag), sa_val[tag]) for tag in sa_tags]
    primary = sa_label(sa_tags[0])
    width = 34 + 13 * len(method_cols) + 24

    print("=" * width)
    print("Stowed value as a percentage of the upper bound - "
          "Bortfeldt (2000) Table 2 vs. our solvers")
    print("=" * width)
    header = f"{'inst':<6}{'upper bound':>14}"
    for name, _ in method_cols:
        header += f"{name:>13}"
    print(header + f"{primary + ' value':>15}{'util':>9}")
    print("-" * width)

    sums = {name: [] for name, _ in method_cols}
    for n in range(1, 17):
        line = f"{n:<6}{bounds[n]:>14.1f}"
        for name, values in method_cols:
            if name == "KUR" and n in KURPEL_INCOMPARABLE:
                line += f"{'n/c':>13}"      # different instance data, see above
                continue
            p = pct(values, n)
            if p is not None:
                sums[name].append(p)
            line += cell(p)
        print(line + f"{sa_val[sa_tags[0]][n]:>15.1f}{util[n]:>8.1f}%")

    print("-" * width)
    means = {k: (sum(v) / len(v) if v else 0.0) for k, v in sums.items()}
    line = f"{'mean':<6}{'':>14}"
    for name, _ in method_cols:
        line += cell(means[name])
    print(line + f"{'':>15}{sum(util.values()) / len(util):>8.1f}%")

    print()
    for name, paper in PAPER_MEANS.items():
        if abs(means[name] - paper) > 0.15:
            print(f"  NOTE: recomputed {name} mean {means[name]:.1f}% differs from "
                  f"the {paper}% printed in Table 2 by more than rounding")
    print(f"  (Table 2 prints {PAPER_MEANS['Mohanty']}% and {PAPER_MEANS['MCL']}% "
          f"for Mohanty and MCL; the small deltas are rounding)")

    # ---- Against the proven optima -------------------------------------
    # Kurpel et al. closed 9 of these instances, so on those the true optimum
    # is known. Measuring against it is far more meaningful than measuring
    # against the continuous-knapsack bound, which is loose by construction.
    known = sorted(KURPEL_OPTIMAL - KURPEL_INCOMPARABLE)
    print(f"\nAgainst the {len(known)} comparable instances Kurpel et al. proved "
          f"optimal (instance 4 excluded, see the header):")
    print(f"  {'inst':<6}{'optimum':>14}{'SA value':>14}{'SA % of opt':>13}")
    ratios = []
    for n in known:
        opt = KURPEL_VALUES[n]
        v = sa_val[sa_tags[0]][n]
        r = 100.0 * v / opt
        ratios.append(r)
        flag = "  <- optimal" if abs(v - opt) < 1e-6 else ""
        print(f"  {n:<6}{opt:>14.1f}{v:>14.1f}{r:>12.1f}%{flag}")
    print(f"  {'mean':<6}{'':>14}{'':>14}{sum(ratios) / len(ratios):>12.1f}%")
    hit = sum(1 for n in known if abs(sa_val[sa_tags[0]][n] - KURPEL_VALUES[n]) < 1e-6)
    print(f"  {primary} reaches the proven optimum on {hit} of {len(known)}")

    print("\nHead to head, by instance:")
    for tag in sa_tags:
        label, values = sa_label(tag), sa_val[tag]
        head_to_head(label, values, "MCL", BORTFELDT_VALUES)
        head_to_head(label, values, "Mohanty", MOHANTY_VALUES)
        head_to_head(label, values, "MILP", milp_val)
        head_to_head(label, values, "KUR",
                     {n: v for n, v in KURPEL_VALUES.items()
                      if n not in KURPEL_INCOMPARABLE})
    head_to_head("MILP", milp_val, "MCL", BORTFELDT_VALUES)
    head_to_head("MILP", milp_val, "Mohanty", MOHANTY_VALUES)
    for a, b in zip(sa_tags, sa_tags[1:]):
        head_to_head(sa_label(a), sa_val[a], sa_label(b), sa_val[b])

    sa_t = sum(r["SA"] for r in runtime.values()) / len(runtime)
    milp_r = [r["MILP"] for r in runtime.values() if "MILP" in r]
    kur_mean = sum(KURPEL_TIMES.values()) / len(KURPEL_TIMES)
    kur_hit = sorted(n for n, s in KURPEL_TIMES.items() if s >= KURPEL_TIME_LIMIT)
    kur_solved = [KURPEL_TIMES[n] for n in sorted(KURPEL_OPTIMAL)]

    print(f"\nMean runtime per instance: {primary} {sa_t:.0f} s"
          + (f", MILP {sum(milp_r) / len(milp_r):.0f} s" if milp_r else "")
          + f", KUR {kur_mean:.0f} s, MCL 89.5 s.")
    print(f"  KUR is strongly bimodal: it closes {len(kur_solved)} instances in "
          f"{sum(kur_solved) / len(kur_solved):.0f} s on average "
          f"(median {sorted(kur_solved)[len(kur_solved) // 2]:.1f} s), and spends "
          f"the full {KURPEL_TIME_LIMIT:.0f} s limit on the other {len(kur_hit)} "
          f"{kur_hit} without proving optimality -")
    print(f"  those {len(kur_hit)} account for "
          f"{100 * sum(KURPEL_TIMES[n] for n in kur_hit) / sum(KURPEL_TIMES.values()):.0f}%"
          f" of their total {sum(KURPEL_TIMES.values()) / 3600:.1f} h.")
    print("  Hardware differs across all four (KUR: 2.77 GHz Xeon, Gurobi 8.1.0; "
          "MCL: 200 MHz Pentium), so wall-clock is indicative, not a fair "
          "head-to-head.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
