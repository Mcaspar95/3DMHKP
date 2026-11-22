from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
import math
import unittest

import bound_br0_patterns as bounds
import verify_br0_pattern_certificate as verifier


def integer_grid_optimum(container, orientations, copies):
    """Exhaustive integer-grid packing for small test containers."""
    placements = [[] for _ in range(math.prod(container))]
    for dims in orientations:
        for origin in product(*(range(container[i] - dims[i] + 1) for i in range(3))):
            cells = [x + container[0] * (y + container[1] * z)
                     for x in range(origin[0], origin[0] + dims[0])
                     for y in range(origin[1], origin[1] + dims[1])
                     for z in range(origin[2], origin[2] + dims[2])]
            mask = sum(1 << cell for cell in cells)
            for cell in cells:
                placements[cell].append(mask)

    @lru_cache(None)
    def search(available, remaining):
        if not available or not remaining:
            return 0
        first = (available & -available).bit_length() - 1
        best = search(available & ~(1 << first), remaining)
        for mask in placements[first]:
            if mask & available == mask:
                best = max(best, 1 + search(available ^ mask, remaining - 1))
        return best

    return search((1 << math.prod(container)) - 1, copies)


class PatternBoundTests(unittest.TestCase):
    def test_patterns_and_integrated_rows_exhaustively(self):
        for container, box, copies in [((5, 6, 7), (1, 2, 3), 9),
                                      ((4, 5, 6), (2, 2, 3), 6),
                                      ((3, 4, 5), (1, 1, 1), 2)]:
            orientations, rows = bounds.build_rows(container, box, copies)
            for row in rows:
                lengths = row["lengths"]
                feasible = (p for p in product(*(range(copies + 1) for _ in lengths))
                            if sum(p) <= copies and sum(x * y for x, y in zip(p, lengths)) <= container[row["axis"]])
                self.assertEqual(row["pattern_max"], max(sum(w * n for w, n in zip(row["weights"], p)) for p in feasible))
                for orient, coefficient in zip(orientations, row["coefficients"]):
                    weight = row["weights"][lengths.index(orient[row["axis"]])]
                    self.assertEqual(coefficient, weight * math.prod(orient[i] for i in range(3) if i != row["axis"]))

    def test_against_exhaustive_grid_packings(self):
        for container, box, copies in [((2, 2, 3), (1, 1, 2), 8),
                                      ((3, 3, 2), (1, 2, 2), 5),
                                      ((3, 3, 2), (1, 1, 3), 8),
                                      ((2, 2, 2), (1, 1, 1), 4),
                                      ((2, 2, 2), (2, 2, 3), 5)]:
            result = bounds.calculate_bound(container, box, copies)
            optimum = integer_grid_optimum(container, result["orientations"], copies)
            self.assertGreaterEqual(result["upper_count"], optimum)

    def test_exact_certificate_reconstruction(self):
        container, box, copies = (587, 220, 233), (106, 55, 38), 135
        result = bounds.calculate_bound(container, box, copies)
        rows = result["certificate"]["rows"]
        coverage = [sum((Fraction(row["multiplier"]) * row["coefficients"][i] for row in rows), Fraction())
                    for i in range(len(result["orientations"]))]
        correction = copies * sum((max(Fraction(), 1 - c) for c in coverage), Fraction())
        rational = sum((Fraction(row["multiplier"]) * row["rhs"] for row in rows), Fraction()) + correction
        self.assertEqual(str(correction), result["certificate"]["correction"])
        self.assertEqual(str(rational), result["certificate"]["rational_count_bound"])
        self.assertEqual(result["upper_count"], 132)
        self.assertEqual(result["upper_bound"], 29243280)
        record = json.loads(json.dumps({"container": container, "box": box, "copies": copies, **result}))
        self.assertEqual(verifier.verify(record), 132)
        record["certificate"]["rows"][0]["pattern_max"] -= 1
        with self.assertRaisesRegex(ValueError, "pattern maximum"):
            verifier.verify(record)

    def test_zero_and_unfit(self):
        self.assertEqual(bounds.calculate_bound((3, 3, 3), (1, 1, 1), 0)["upper_count"], 0)
        self.assertEqual(bounds.calculate_bound((3, 3, 3), (4, 4, 4), 4)["upper_count"], 0)
        result = bounds.calculate_bound((2, 3, 4), (2, 3, 4), 2)
        self.assertEqual(result["orientations"], [(2, 3, 4)])
        self.assertEqual(result["upper_count"], 1)

    def test_invalid_inputs(self):
        for container, box, copies in [((1, 2, 0), (1, 1, 1), 3),
                                      ((1, 2, 3), (1, 1, 1), -1),
                                      ((1, 2, 3), (1, 1, 1), 1.5)]:
            with self.assertRaises(ValueError):
                bounds.calculate_bound(container, box, copies)


if __name__ == "__main__":
    unittest.main()