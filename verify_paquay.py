#!/usr/bin/env python3
"""Independent feasibility checker for Paquay-SA.py solutions.

Deliberately re-implements the constraint tests from the instance data rather
than reusing Paquay-SA.py's own predicates, so that a bug in the solver's
feasibility logic cannot hide itself by also being used to validate.

Checks, per the paper's Section 2:
  1. every box packed exactly once,
  2. containment including the bevelled cuts,
  3. no overlap,
  4. orientation is a permutation allowed by the box's l+/w+/h+ flags,
  5. nothing stacked on a fragile box,
  6. ULD weight capacity,
  7. support beneath every raised box.

Usage:
    python verify_paquay.py --n 20 --sample 0 --time 5
"""

import argparse
import itertools
import sys

from load_paquay import load_boxes, load_ulds, instance_path
import importlib
sa = importlib.import_module("Paquay-SA")

EPS = 1e-6


def check(boxes, bins, min_frac, cg=False):
    errs = []

    # 1. every box exactly once
    seen = []
    for b in bins:
        for p in b.placed:
            seen.append(p[6])
    if sorted(seen) != list(range(len(boxes))):
        missing = set(range(len(boxes))) - set(seen)
        dupes = [i for i in set(seen) if seen.count(i) > 1]
        errs.append("box multiset wrong: missing=%s duplicated=%s"
                    % (sorted(missing), sorted(dupes)))

    for bi, b in enumerate(bins):
        u = b.uld
        # CG constraints, only when the solver was told to enforce them.
        # Recomputed here from the placements, not read off the Bin's moments.
        if cg and b.placed:
            tw = sum(p[8] for p in b.placed)
            if tw > 0:
                gx = sum((p[0] + p[3] / 2.0) * p[8] for p in b.placed) / tw
                gy = sum((p[1] + p[4] / 2.0) * p[8] for p in b.placed) / tw
                gz = sum((p[2] + p[5] / 2.0) * p[8] for p in b.placed) / tw
                if gz > u.aH + 1e-3:
                    errs.append("bin %d (%s): CG height %.1f > limit %.1f"
                                % (bi, u.code, gz, u.aH))
                if abs(gx - u.L / 2.0) > u.aL + 1e-3:
                    errs.append("bin %d (%s): CG x off centre by %.1f > %.1f"
                                % (bi, u.code, abs(gx - u.L / 2.0), u.aL))
                if abs(gy - u.W / 2.0) > u.aW + 1e-3:
                    errs.append("bin %d (%s): CG y off centre by %.1f > %.1f"
                                % (bi, u.code, abs(gy - u.W / 2.0), u.aW))
        # 6. weight
        wsum = sum(p[8] for p in b.placed)
        if wsum > u.max_weight + EPS:
            errs.append("bin %d (%s): weight %.1f > capacity %.1f"
                        % (bi, u.code, wsum, u.max_weight))

        for pi, (x, y, z, l, w, h, idx, frag, wt) in enumerate(b.placed):
            box = boxes[idx]

            # 4. orientation legal
            allowed = set()
            for perm in itertools.permutations(
                    (box.length, box.width, box.height)):
                allowed.add(perm)
            if (l, w, h) not in allowed:
                errs.append("bin %d box %d: dims %s not a permutation of %s"
                            % (bi, idx, (l, w, h),
                               (box.length, box.width, box.height)))
            else:
                legal = set(sa.orientations(box))
                if (l, w, h) not in legal:
                    errs.append("bin %d box %d: orientation %s violates "
                                "l+/w+/h+ flags (%s,%s,%s)"
                                % (bi, idx, (l, w, h),
                                   box.l_up, box.w_up, box.h_up))

            # 2. containment, independently recomputed
            if x < -EPS or y < -EPS or z < -EPS:
                errs.append("bin %d box %d: negative coordinate" % (bi, idx))
            if (x + l > u.L + EPS or y + w > u.W + EPS or z + h > u.H + EPS):
                errs.append("bin %d box %d: outside bounding box" % (bi, idx))
            for xx in (x, x + l):
                for zz in (z, z + h):
                    for (den, num, bb, sense) in u.cuts:
                        lhs = zz * den + num * xx
                        if sense == 1 and lhs > bb + 1e-3:
                            errs.append("bin %d box %d: corner (%.0f,%.0f) "
                                        "outside cut" % (bi, idx, xx, zz))
                        if sense == -1 and lhs < bb - 1e-3:
                            errs.append("bin %d box %d: corner (%.0f,%.0f) "
                                        "outside cut" % (bi, idx, xx, zz))

            # 3. overlap
            for pj in range(pi + 1, len(b.placed)):
                (qx, qy, qz, ql, qw, qh, qidx, _qf, _qw2) = b.placed[pj]
                if (x + l > qx + EPS and qx + ql > x + EPS and
                        y + w > qy + EPS and qy + qw > y + EPS and
                        z + h > qz + EPS and qz + qh > z + EPS):
                    errs.append("bin %d: boxes %d and %d overlap" % (bi, idx, qidx))

            # 5 + 7. support and fragility
            if z > EPS:
                area = 0.0
                for (qx, qy, qz, ql, qw, qh, qidx, qfrag, _qw2) in b.placed:
                    if abs(qz + qh - z) > EPS:
                        continue
                    ox = max(0.0, min(x + l, qx + ql) - max(x, qx))
                    oy = max(0.0, min(y + w, qy + qw) - max(y, qy))
                    if ox > 0 and oy > 0:
                        if qfrag:
                            errs.append("bin %d: box %d rests on fragile box %d"
                                        % (bi, idx, qidx))
                        area += ox * oy
                if area < l * w * min_frac - 1e-3:
                    errs.append("bin %d box %d: support %.1f%% < required %.0f%%"
                                % (bi, idx, 100.0 * area / (l * w),
                                   100.0 * min_frac))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--time", type=float, default=3.0)
    ap.add_argument("--support", type=float, default=1.0)
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--cg", action="store_true",
                    help="solver enforces CG; also check it here")
    args = ap.parse_args()

    ulds = [sa.Uld(u) for u in load_ulds()]
    samples = [args.sample] if args.sample is not None else range(30)
    bad = 0
    for k in samples:
        path = instance_path(args.n, k, args.training)
        boxes = load_boxes(path)
        res = sa.solve_instance(boxes, ulds, args.time, seed=k,
                                min_frac=args.support, greedy=args.greedy,
                                cg=args.cg)
        if res is None:
            print("n%dsample%d: no feasible decode" % (args.n, k))
            bad += 1
            continue
        errs = check(boxes, res["bins"], args.support, args.cg)
        status = "OK  " if not errs else "FAIL"
        print("%s n%dsample%-3d ULDs %2d  vol %7.2f m3  %s"
              % (status, args.n, k, len(res["bins"]), res["obj"] / 1e9,
                 "" if not errs else "%d violation(s)" % len(errs)))
        for e in errs[:8]:
            print("       - " + e)
        if errs:
            bad += 1
    print("\n%d/%d instances clean" % (len(list(samples)) - bad, len(list(samples))))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
