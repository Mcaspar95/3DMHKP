import random
import tempfile
import unittest
from pathlib import Path

import generate_fekete07 as gen


class GeneratorTests(unittest.TestCase):
    def test_instance_properties(self):
        for cls in gen.WEIGHTS:
            for m, multiplicity in gen.SETTINGS:
                rows = gen.generate_instance(random.Random(2007), cls, m, multiplicity)
                self.assertEqual(len(rows), m)
                for type_id, count, x, y, z, profit in rows:
                    self.assertTrue(1 <= type_id <= m)
                    self.assertEqual(count, multiplicity)
                    self.assertTrue(any(all(lo <= d <= hi for d, (lo, hi) in zip((x, y, z), ranges))
                                        for ranges in gen.RANGES))
                    self.assertIn(profit, (x * y * z, 2 * x * y * z, 3 * x * y * z))

    def test_complete_reproducible_set_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            paths_a = gen.generate_all(Path(a), 2007)
            paths_b = gen.generate_all(Path(b), 2007)
            self.assertEqual(len(paths_a), 150)
            self.assertEqual([p.name for p in paths_a], [p.name for p in paths_b])
            self.assertEqual([p.read_bytes() for p in paths_a],
                             [p.read_bytes() for p in paths_b])
            with self.assertRaises(FileExistsError):
                gen.generate_all(Path(a), 2007)


if __name__ == "__main__":
    unittest.main()