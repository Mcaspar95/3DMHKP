#!/usr/bin/env python3
"""Exact-certificate line-pattern bounds for free-rotation homogeneous boxes.

For each axis, a line through the container intersects a feasible 1D pattern.
Integrating any valid weighted pattern inequality over its perpendicular plane
gives a linear inequality in the numbers of boxes of each orientation. The LP
only suggests multipliers: the reported bound is evaluated using exact integer
and rational arithmetic, including a correction for any dual infeasibility.
"""
import argparse
from fractions import Fraction
from itertools import permutations
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

import bound_br15 as br


def pattern_points(lengths, capacity, copies):
    """Return maximal patterns sufficient for nonnegative weight maximization."""
    if len(lengths) == 1:
        return [(min(copies, capacity // lengths[0]),)]
    if len(lengths) == 2:
        a, b = lengths
        return [(i, min(copies - i, (capacity - a * i) // b))
                for i in range(min(copies, capacity // a) + 1)]
    a, b, c = lengths
    return [(i, j, min(copies - i - j, (capacity - a * i - b * j) // c))
            for i in range(min(copies, capacity // a) + 1)
            for j in range(min(copies - i, (capacity - a * i) // b) + 1)]


def integer_weights(weights):
    """Round a candidate normal; validity comes from recomputing its RHS."""
    parts = [Fraction(max(0.0, float(w))).limit_denominator(10000) for w in weights]
    den = math.lcm(*(x.denominator for x in parts))
    ints = [x.numerator * (den // x.denominator) for x in parts]
    divisor = math.gcd(*ints)
    return tuple(x // divisor for x in ints) if divisor else None


def pattern_weights(points, dimensions):
    # Simple valid directions remain available even if hull construction fails.
    weights = {tuple(int(i == j) for i in range(dimensions)) for j in range(dimensions)}
    weights.add((1,) * dimensions)
    if dimensions > 1:
        try:
            hull = ConvexHull(np.asarray([(0,) * dimensions, *points], dtype=float))
            for equation in hull.equations:
                normal = equation[:-1]
                if min(normal) >= -1e-8 and max(normal) > 1e-8:
                    weight = integer_weights(normal)
                    if weight:
                        weights.add(weight)
        except QhullError:
            pass
    return sorted(weights)


def build_rows(container, box, copies):
    if len(container) != 3 or len(box) != 3 or any(v <= 0 or int(v) != v for v in (*container, *box)):
        raise ValueError("container and box dimensions must be positive integers")
    if copies < 0 or int(copies) != copies:
        raise ValueError("copies must be a nonnegative integer")
    container, box, copies = tuple(map(int, container)), tuple(map(int, box)), int(copies)
    orientations = sorted(p for p in set(permutations(box))
                          if all(p[i] <= container[i] for i in range(3)))
    if not orientations or not copies:
        return orientations, []
    rows = []
    for axis in range(3):
        lengths = sorted({o[axis] for o in orientations})
        points = pattern_points(lengths, container[axis], copies)
        for weight in pattern_weights(points, len(lengths)):
            maximum = max(sum(w * n for w, n in zip(weight, p)) for p in points)
            if not maximum:
                continue
            by_length = dict(zip(lengths, weight))
            coefficients = [by_length[o[axis]] * math.prod(o[i] for i in range(3) if i != axis)
                            for o in orientations]
            rhs = maximum * math.prod(container[i] for i in range(3) if i != axis)
            rows.append({"axis": axis, "lengths": lengths, "weights": weight,
                         "pattern_max": maximum, "coefficients": coefficients, "rhs": rhs})
    return orientations, rows


def calculate_bound(container, box, copies):
    orientations, rows = build_rows(container, box, copies)
    volume = math.prod(box)
    if not orientations or not copies:
        return {"upper_count": 0, "upper_bound": 0, "orientations": orientations,
                "certificate": {"rows": [], "correction": "0", "rational_count_bound": "0"}}
    # Normalize rows for numerical stability. Nonnegative duals need not be
    # feasible: the exact correction below accounts for any shortfall.
    a = np.array([[x / row["rhs"] for x in row["coefficients"]] for row in rows])
    result = linprog(-np.ones(len(orientations)), A_ub=a, b_ub=np.ones(len(rows)),
                     bounds=(0, copies), method="highs")
    if not result.success:
        raise RuntimeError(f"line-pattern LP failed: {result.message}")
    multipliers = [Fraction(max(0.0, float(-dual))).limit_denominator(10**9) / row["rhs"]
                   for dual, row in zip(result.ineqlin.marginals, rows)]
    coverage = [sum((mult * row["coefficients"][j] for mult, row in zip(multipliers, rows)), Fraction())
                for j in range(len(orientations))]
    correction = copies * sum((max(Fraction(), 1 - c) for c in coverage), Fraction())
    rational = sum((mult * row["rhs"] for mult, row in zip(multipliers, rows)), Fraction()) + correction
    # Independently valid exact capacity and supply bounds.
    count = min(copies, math.prod(container) // volume, rational.numerator // rational.denominator)
    active = [{**row, "multiplier": str(mult)} for row, mult in zip(rows, multipliers) if mult]
    return {"upper_count": count, "upper_bound": count * volume, "orientations": orientations,
            "certificate": {"rows": active, "correction": str(correction),
                            "rational_count_bound": str(rational)},
            "lp_count_estimate": -result.fun, "generated_rows": len(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", help="instance stems, e.g. BR0-001")
    parser.add_argument("--output-dir", type=Path, default=br.ROOT / "results_bounds_BR0_patterns")
    args = parser.parse_args(argv)
    instance_dir = br.ROOT / "BischoffRatcliff/BR0"
    paths = ([instance_dir / (name + ".txt") for name in args.instances] if args.instances
             else sorted(instance_dir.glob("BR0-*.txt")))
    if not paths:
        parser.error("no instances found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for path in paths:
        try:
            inst = br.sa.build_instance(path, value_mode="volume")
            payload = br.bound_instance(inst)
            if len(payload["items"]) != 1:
                raise ValueError("expected one homogeneous box type")
            item = payload["items"][0]
            result = calculate_bound(payload["container"], item["dimensions"], item["copies"])
            best = br.best_report(inst, br.ROOT / "results_3DMHKP_BR0")
            if best and best[0] > result["upper_bound"]:
                raise ValueError("verified packing exceeds exact upper bound")
            record = {"instance": inst["name"], "container": payload["container"],
                      "box": item["dimensions"], "copies": item["copies"],
                      "rotation_policy": "all axis permutations",
                      "lower_bound": best[0] if best else None,
                      "lower_bound_report": str(best[1]) if best else None, **result}
            (args.output_dir / f"{path.stem}-patterns.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"{path.stem}: lower={record['lower_bound']} upper={result['upper_bound']} "
                  f"rows={result.get('generated_rows', 0)}", flush=True)
        except (ValueError, RuntimeError, OSError) as exc:
            failures.append(path.stem)
            print(f"{path.stem}: ERROR: {exc}", file=sys.stderr, flush=True)
    if failures:
        print(f"Failed ({len(failures)}): {', '.join(failures)}", file=sys.stderr)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())