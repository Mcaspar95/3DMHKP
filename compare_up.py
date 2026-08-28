# Compare 3DMHKP.py (free 6-orientation rotation) against 3DMHKPup.py
# (Bischoff & Ratcliff "this way up" - rotation about the vertical axis only)
# on the same 16 Mohanty/Mathur/Ivancic instances, both run as exact MILP
# models with Gurobi under the same time limit.
#
# The "up" model is a restriction of the free model (a subset of orientations
# is always feasible for the free model too), so for a given time limit its
# incumbent can only be equal to or worse than the free model's, whether or
# not either run proved optimality. This script reports, per instance, the
# value lost to the upright constraint and whether that loss is a genuine
# packing-quality gap or just a slower/incomplete search (compare the dual
# bounds and status, not only the incumbent values).
#
#   python compare_up.py
#
import sys
from pathlib import Path

FREE_DIR = Path(__file__).parent / "results_3DMHKP"
UP_DIR = Path(__file__).parent / "results_3DMHKPup"


def read_report(path):
    """Pull the headline figures out of one of our report files."""
    if not path.exists():
        return None
    out = {}
    for line in path.open():
        if line.startswith("Status:"):
            out["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("Packed value"):
            out["value"] = float(line.split(":")[1])
        elif line.startswith("Dual bound"):
            out["dual"] = float(line.split(":")[1])
        elif line.startswith("Continuous-knapsack bound"):
            out["bound"] = float(line.split(":")[1])
        elif line.startswith("Optimality gap"):
            out["gap"] = float(line.split(":")[1].strip().rstrip("%"))
        elif line.startswith("Volume utilization"):
            out["util"] = float(line.rsplit("=", 1)[1].strip().rstrip("%"))
        elif line.startswith("Runtime"):
            out["runtime"] = float(line.split(":")[1].strip().split()[0])
    return out or None


def status_flag(status):
    return "*" if status == "optimal" else " "


def main():
    free, up = {}, {}
    for n in range(1, 17):
        f = read_report(FREE_DIR / f"instance{n:02d}-3DMHKP.txt")
        u = read_report(UP_DIR / f"instance{n:02d}-3DMHKPup.txt")
        if f is None:
            print(f"missing free-rotation report for instance {n:02d} - run "
                  f"3DMHKP.py first", file=sys.stderr)
            return 1
        if u is None:
            print(f"missing this-way-up report for instance {n:02d} - run "
                  f"3DMHKPup.py first", file=sys.stderr)
            return 1
        free[n], up[n] = f, u

    width = 118
    print("=" * width)
    print("3DMHKP (free rotation) vs 3DMHKPup (this way up) - MILP incumbents, "
          "same instances and time limit")
    print("* marks a status of 'optimal'; otherwise the run hit the time limit")
    print("=" * width)
    header = (f"{'inst':<6}{'free value':>13}{'up value':>13}{'loss':>10}"
              f"{'loss %':>9}{'free dual':>12}{'up dual':>12}"
              f"{'free gap%':>11}{'up gap%':>10}{'free util':>11}{'up util':>10}")
    print(header)
    print("-" * width)

    loss_pct, util_delta = [], []
    ties = 0
    for n in range(1, 17):
        f, u = free[n], up[n]
        loss = f["value"] - u["value"]
        lp = 100.0 * loss / f["value"] if f["value"] else 0.0
        loss_pct.append(lp)
        util_delta.append(f["util"] - u["util"])
        if loss <= 1e-6:
            ties += 1
        fstat, ustat = status_flag(f["status"]), status_flag(u["status"])
        print(f"{n:<6}{f['value']:>12.1f}{fstat}{u['value']:>12.1f}{ustat}"
              f"{loss:>10.1f}{lp:>8.1f}%{f['dual']:>12.1f}{u['dual']:>12.1f}"
              f"{f['gap']:>10.1f}%{u['gap']:>9.1f}%{f['util']:>10.1f}%{u['util']:>9.1f}%")

    print("-" * width)
    print(f"{'mean':<6}{'':>13}{'':>13}{'':>10}{sum(loss_pct)/16:>8.1f}%"
          f"{'':>12}{'':>12}{'':>11}{'':>10}"
          f"{sum(f['util'] for f in free.values())/16:>10.1f}%"
          f"{sum(u['util'] for u in up.values())/16:>9.1f}%")

    print(f"\nThe upright constraint costs value on {16 - ties} of 16 instances "
          f"(tied - no measurable loss - on {ties}); mean loss "
          f"{sum(loss_pct)/16:.1f}% of the free-rotation value, "
          f"worst case instance {max(range(1, 17), key=lambda n: loss_pct[n-1])} "
          f"at {max(loss_pct):.1f}%.")
    print(f"Mean volume utilization: free {sum(f['util'] for f in free.values())/16:.1f}%, "
          f"up {sum(u['util'] for u in up.values())/16:.1f}% "
          f"(delta {sum(util_delta)/16:+.1f} pts).")

    free_opt = sorted(n for n in free if free[n]["status"] == "optimal")
    up_opt = sorted(n for n in up if up[n]["status"] == "optimal")
    print(f"Proved optimal: free rotation on {free_opt}, this-way-up on {up_opt}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
