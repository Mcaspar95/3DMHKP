# Head-to-head of one SA run against Kurpel et al. (2020), EJOR 284:87-107,
# Table 12 ("This paper") - the strongest published result set on the 16
# Mohanty, Mathur & Ivancic (1994) instances.
# =========================================================================
# Unlike compare_bortfeldt.py, which puts five methods on the shared
# upper-bound scale, this script answers one question: how does a given SA run
# compare, instance by instance, against the best published values, and how
# does it compare against another of our own runs.
#
# The tables and constants come from compare_bortfeldt.py, so the two scripts
# cannot drift apart. Instance 4 is excluded from every aggregate: Kurpel et
# al. solved a 10-container truncation of it (see the header of
# compare_bortfeldt.py).
#
#   python compare_kurpel.py --sa-dir results_3DMHKP_Sa900 --sa-tag t7200s
#   python compare_kurpel.py --sa-dir results_3DMHKP_Sa900 --sa-tag t7200s \
#                            --baseline t1800s
#
import argparse
import sys
from pathlib import Path

from compare_bortfeldt import (KURPEL_VALUES, KURPEL_OPTIMAL, KURPEL_TIMES,
                               KURPEL_INCOMPARABLE, KURPEL_TIME_LIMIT,
                               find_reports, read_report)


def load(sa_dir, selector):
    """Read the 16 reports the selector picks out of sa_dir."""
    out = {}
    for n in range(1, 17):
        found = find_reports(sa_dir, n, selector)
        if len(found) != 1:
            what = ("no report" if not found
                    else f"{len(found)} reports ({', '.join(q.name for q in found)})")
            print(f"instance {n:02d}: {what} match '{selector}' in {sa_dir}",
                  file=sys.stderr)
            return None
        rep = read_report(found[0])
        if rep is None:
            print(f"unreadable report {found[0]}", file=sys.stderr)
            return None
        rep["path"] = found[0]
        out[n] = rep
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare one SA run against Kurpel et al. (2020) Table 12.")
    parser.add_argument("--sa-dir", type=Path,
                        default=Path(__file__).parent / "results_3DMHKP",
                        help="directory holding the SA reports")
    parser.add_argument("--sa-tag", default="", metavar="SELECTOR",
                        help="which run in that directory; any substring of "
                             "the report filename, e.g. t7200s")
    parser.add_argument("--baseline", default=None, metavar="SELECTOR",
                        help="a second run to diff against, same syntax")
    args = parser.parse_args(argv)

    sa = load(args.sa_dir, args.sa_tag)
    if sa is None:
        return 1
    base = None
    if args.baseline is not None:
        base = load(args.sa_dir, args.baseline)
        if base is None:
            return 1

    label = args.sa_tag or "SA"
    limits = {r["time_limit"] for r in sa.values() if "time_limit" in r}
    budget = (f"{limits.pop():.0f} s" if len(limits) == 1 else
              f"{sum(r['runtime'] for r in sa.values()) / 16:.0f} s elapsed")

    width = 104
    print(f"SA ({args.sa_dir}, selector '{label}', {budget} per instance)")
    print("vs Kurpel et al. (2020), EJOR 284:87-107, Table 12 'This paper' - "
          "exact 0-1 model, 3600 s")
    print("=" * width)
    print(f"{'inst':<5}{'KUR value':>13}{'KUR t':>8}{'opt':>5}{'SA value':>13}"
          f"{'delta':>13}{'delta%':>9}{'SA util':>9}   verdict")
    print("-" * width)

    d_pct, wins, ties, losses = [], [], [], []
    for n in range(1, 17):
        k, v = KURPEL_VALUES[n], sa[n]["value"]
        opt = "*" if n in KURPEL_OPTIMAL else ""
        t = KURPEL_TIMES[n]
        tcell = f"{t:.0f}s" if t < KURPEL_TIME_LIMIT else "lim"
        if n in KURPEL_INCOMPARABLE:
            print(f"{n:<5}{k:>13.1f}{tcell:>8}{opt:>5}{v:>13.1f}{'n/c':>13}"
                  f"{'':>9}{sa[n]['util']:>8.1f}%   not comparable "
                  f"(Kurpel solved a 10-container truncation)")
            continue
        d = v - k
        dp = 100.0 * d / k
        d_pct.append(dp)
        if d > 1e-6:
            verdict, bucket = "SA better", wins
        elif d < -1e-6:
            verdict, bucket = ("SA worse" + (" (below proven optimum)"
                                             if n in KURPEL_OPTIMAL else ""), losses)
        else:
            verdict, bucket = ("equal" + (" (OPTIMAL)" if n in KURPEL_OPTIMAL
                                          else ""), ties)
        bucket.append(n)
        print(f"{n:<5}{k:>13.1f}{tcell:>8}{opt:>5}{v:>13.1f}{d:>+13.1f}"
              f"{dp:>+8.1f}%{sa[n]['util']:>8.1f}%   {verdict}")

    print("-" * width)
    mean_d = sum(d_pct) / len(d_pct)
    median_d = sorted(d_pct)[len(d_pct) // 2]
    print(f"{'mean':<5}{'':>13}{'':>8}{'':>5}{'':>13}{'':>13}{mean_d:>+8.1f}%"
          f"{sum(r['util'] for r in sa.values()) / 16:>8.1f}%")
    print("-" * width)
    print("* = Kurpel proved optimality (their gap 0.00); 'lim' = they hit the "
          "3600 s limit.")
    print(f"Median delta {median_d:+.1f}% - report it next to the mean, which a "
          f"single large win distorts.")
    print()
    print(f"Over the 15 comparable instances: SA better on {len(wins)} {wins}, "
          f"equal on {len(ties)} {ties}, worse on {len(losses)} {losses}.")
    comparable = [n for n in KURPEL_VALUES if n not in KURPEL_INCOMPARABLE]
    sa_sum = sum(sa[n]["value"] for n in comparable)
    kur_sum = sum(KURPEL_VALUES[n] for n in comparable)
    print(f"Aggregate value over those 15: SA {sa_sum:.1f} vs KUR {kur_sum:.1f} "
          f"({100 * sa_sum / kur_sum - 100:+.1f}%).")

    known = sorted(KURPEL_OPTIMAL - KURPEL_INCOMPARABLE)
    hit = sum(1 for n in known if abs(sa[n]["value"] - KURPEL_VALUES[n]) < 1e-6)
    ratios = [100 * sa[n]["value"] / KURPEL_VALUES[n] for n in known]
    worst = min(known, key=lambda n: sa[n]["value"] / KURPEL_VALUES[n])
    print(f"On the {len(known)} instances with a PROVEN optimum {known}: SA "
          f"reaches it on {hit}, mean {sum(ratios) / len(ratios):.1f}% of "
          f"optimum, worst instance {worst} at {min(ratios):.1f}%.")
    open_i = sorted(n for n in KURPEL_VALUES
                    if n not in KURPEL_OPTIMAL and n not in KURPEL_INCOMPARABLE)
    print(f"On the {len(open_i)} instances Kurpel left OPEN {open_i} (they "
          f"burned the full 3600 s on each): SA better on "
          f"{sorted(n for n in open_i if sa[n]['value'] > KURPEL_VALUES[n] + 1e-6)}"
          f", worse on "
          f"{sorted(n for n in open_i if sa[n]['value'] < KURPEL_VALUES[n] - 1e-6)}.")

    # ---- against another of our own runs --------------------------------
    if base is not None:
        print()
        print("=" * width)
        print(f"'{label}' against our own '{args.baseline}' run")
        print("=" * width)
        print(f"{'inst':<5}{args.baseline:>15}{label:>15}{'delta':>13}"
              f"{'delta%':>9}   seeds differ")
        print("-" * width)
        b_better, b_worse, b_equal = [], [], []
        for n in range(1, 17):
            a, c = base[n]["value"], sa[n]["value"]
            d = c - a
            same_seed = base[n].get("seed") == sa[n].get("seed")
            if d > 1e-6:
                b_better.append(n)
            elif d < -1e-6:
                b_worse.append(n)
            else:
                b_equal.append(n)
            print(f"{n:<5}{a:>15.1f}{c:>15.1f}{d:>+13.1f}"
                  f"{100 * d / a if a else 0:>+8.1f}%   "
                  f"{'same seed' if same_seed else 'yes'}")
        print("-" * width)
        print(f"'{label}' better on {len(b_better)} {b_better}, equal on "
              f"{len(b_equal)} {b_equal}, worse on {len(b_worse)} {b_worse}.")
        bs = sum(base[n]["value"] for n in comparable)
        print(f"Aggregate over the 15 comparable instances: "
              f"{args.baseline} {bs:.1f} -> {label} {sa_sum:.1f} "
              f"({100 * sa_sum / bs - 100:+.1f}%).")
        if all(base[n].get("seed") != sa[n].get("seed") for n in range(1, 17)):
            print("CAVEAT: the two runs use different seeds throughout, so a "
                  "per-instance difference mixes the effect of the longer "
                  "budget with ordinary seed-to-seed SA variance. Only a "
                  "same-seed pair isolates the budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
