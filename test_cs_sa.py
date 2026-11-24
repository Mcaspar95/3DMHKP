import importlib.util
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cs_sa", ROOT / "CS-SA.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


class CSTests(unittest.TestCase):
    def test_all_instances_parse(self):
        paths = sorted((ROOT / "CS1-CS4").glob("CS*.in"))
        self.assertEqual(len(paths), 117)
        for path in paths:
            with self.subTest(path=path.name):
                containers, boxes = cs.parse_instance(path)
                self.assertTrue(containers)
                self.assertTrue(boxes)

    def test_support_load_propagates_through_stack(self):
        containers = [{"type": "c", "dims": (2, 2, 3), "capacity": 10, "cost": 1}]
        boxes = [{"type": "b", "orientations": [((2, 2, 1), 0.25)],
                  "weight": 1, "client": 0, "cost": 1} for _ in range(3)]
        placements = cs.decode(containers, boxes, [0, 1, 2], [0, 0, 0],
                               time.monotonic() + 10)
        self.assertEqual(len(placements), 2)
        cs.verify(containers, boxes, placements)
        invalid = placements + [{"box": 2, "container": 0, "position": [0, 0, 2],
                                 "dimensions": [2, 2, 1], "bearing": 0.25}]
        with self.assertRaisesRegex(ValueError, "load-bearing"):
            cs.verify(containers, boxes, invalid)

    def test_destination_bands_and_container_weight(self):
        containers = [{"type": "c", "dims": (2, 4, 2), "capacity": 2, "cost": 1}]
        boxes = [{"type": str(i), "orientations": [((2, 2, 1), 10)],
                  "weight": 1, "client": i, "cost": 1} for i in range(3)]
        placements = cs.decode(containers, boxes, [0, 1, 2], [0, 0, 0],
                               time.monotonic() + 10)
        self.assertEqual(len(placements), 2)
        cs.verify(containers, boxes, placements)
        self.assertEqual(cs.objective(containers, boxes, placements), (1, 1, 2, 4))


if __name__ == "__main__":
    unittest.main()
