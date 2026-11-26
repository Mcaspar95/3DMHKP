import tempfile
import unittest
from pathlib import Path

from gurobipy import GRB

import eg2d_opt


ROOT = Path(__file__).parent


class EG2DTests(unittest.TestCase):
    def test_paper_instances_parse(self):
        directory = ROOT / "Egeblad" / "ep2"
        paths = [
            directory / f"ep2-{n}-{cls}-{kind}-{percent}.2kp"
            for n in (30, 50, 100, 200)
            for cls in "DSTUW"
            for kind in "CR"
            for percent in (25, 75)
        ]
        for path in paths:
            with self.subTest(path=path.name):
                width, height, items = eg2d_opt.read_instance(path)
                self.assertEqual(len(items), int(path.stem.split("-")[1]))
                self.assertGreater(width * height, 0)

    def test_parser_respects_copies_and_profit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instance.2kp"
            path.write_text("dim, 2, 3\nrect, 17, 3, 2, 7, 2\n", encoding="utf-8")
            width, height, items = eg2d_opt.read_instance(path)
        self.assertEqual((width, height), (2, 3))
        self.assertEqual([(i["id"], i["copy"], i["profit"]) for i in items],
                         [(17, 1, 7), (17, 2, 7)])

    def test_rotation_and_nonoverlap_in_exact_model(self):
        items = [
            {"id": 0, "copy": 1, "width": 3, "height": 2, "profit": 7},
            {"id": 1, "copy": 1, "width": 2, "height": 2, "profit": 5},
        ]
        model, selected, rotated, _, _ = eg2d_opt.build_model(2, 3, items)
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = 0
        model.optimize()
        self.assertEqual(model.Status, GRB.OPTIMAL)
        self.assertEqual(model.ObjVal, 7)
        self.assertGreater(selected[0].X, 0.5)
        self.assertGreater(rotated[0].X, 0.5)
        self.assertLess(selected[1].X, 0.5)

    def test_verifier_rejects_overlap(self):
        items = [{"width": 2, "height": 2}, {"width": 2, "height": 2}]
        placements = [
            {"index": 0, "rotated": False, "position": [0, 0], "dimensions": [2, 2]},
            {"index": 1, "rotated": False, "position": [1, 1], "dimensions": [2, 2]},
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            eg2d_opt.verify_placements(4, 4, items, placements)


if __name__ == "__main__":
    unittest.main()
