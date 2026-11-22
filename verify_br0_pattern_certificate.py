#!/usr/bin/env python3
"""Independently verify stored homogeneous-box certificates using knapsack DP.

This verifies the upper bound against the dimensions in each JSON file. It does
not authenticate the source instance or verify the saved packing/lower bound.
"""
import argparse
from fractions import Fraction
from itertools import permutations
import json
import math
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def knapsack_max(lengths, weights, capacity, copies):
    # At most copies segments; state is exact number and total length.
    current = {0: 0}
    best = 0
    for _ in range(copies):
        following = {}
        for used, value in current.items():
            for length, weight in zip(lengths, weights):
                new_used = used + length
                if new_used <= capacity:
                    following[new_used] = max(following.get(new_used, -1), value + weight)
        if not following:
            break
        current = following
        best = max(best, max(current.values()))
    return best


def verify(record):
    container, box, copies = record["container"], record["box"], record["copies"]
    require(len(container) == len(box) == 3 and all(type(x) is int and x > 0 for x in container + box),
            "dimensions must be positive integers")
    require(type(copies) is int and copies >= 0, "invalid copy count")
    orientations = sorted(set(p for p in permutations(box) if all(p[i] <= container[i] for i in range(3))))
    require(record["orientations"] == [list(p) for p in orientations], "incorrect orientations")
    certificate = record["certificate"]
    coverage = [Fraction() for _ in orientations]
    total = Fraction()
    for row in certificate["rows"]:
        axis = row["axis"]
        require(type(axis) is int and 0 <= axis < 3, "invalid axis")
        lengths, weights = row["lengths"], row["weights"]
        require(lengths == sorted({o[axis] for o in orientations}), "invalid lengths")
        require(len(weights) == len(lengths) and all(type(w) is int and w >= 0 for w in weights),
                "invalid weights")
        maximum = knapsack_max(lengths, weights, container[axis], copies)
        require(row["pattern_max"] == maximum, "incorrect pattern maximum")
        rhs = maximum * math.prod(container[i] for i in range(3) if i != axis)
        require(row["rhs"] == rhs, "incorrect row right-hand side")
        coefficients = [weights[lengths.index(o[axis])] * math.prod(o[i] for i in range(3) if i != axis)
                        for o in orientations]
        require(row["coefficients"] == coefficients, "incorrect row coefficients")
        multiplier = Fraction(row["multiplier"])
        require(multiplier >= 0, "negative multiplier")
        total += multiplier * rhs
        coverage = [c + multiplier * a for c, a in zip(coverage, coefficients)]
    correction = copies * sum((max(Fraction(), 1 - c) for c in coverage), Fraction())
    require(Fraction(certificate["correction"]) == correction, "incorrect correction")
    total += correction
    require(Fraction(certificate["rational_count_bound"]) == total, "incorrect rational bound")
    count = min(copies, math.prod(container) // math.prod(box), total.numerator // total.denominator)
    require(record["upper_count"] == count and record["upper_bound"] == count * math.prod(box),
            "incorrect final bound")
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failures = 0
    for path in args.files:
        try:
            count = verify(json.loads(path.read_text()))
            print(f"{path}: verified upper count {count}")
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as exc:
            failures += 1
            print(f"{path}: INVALID: {exc}", file=sys.stderr)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())