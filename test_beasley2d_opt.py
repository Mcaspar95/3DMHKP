import json
import tempfile
import unittest
from pathlib import Path

import beasley2d_opt
import eg2d_opt


class Beasley2DTests(unittest.TestCase):
    def test_all_instances_parse_with_original_counts(self):
        instances = beasley2d_opt.read_instances(beasley2d_opt.INSTANCE_FILE)
        self.assertEqual(len(instances), 210)
        self.assertEqual(len(instances[0][2]), 40)
        self.assertEqual(max(len(items) for _, _, items in instances), 4000)

    def test_infeasible_1d_choice_is_cut_without_rotation(self):
        items = [
            {"id": 0, "copy": 1, "width": 3, "height": 2, "profit": 10},
            {"id": 1, "copy": 1, "width": 2, "height": 3, "profit": 9},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            beasley2d_opt.solve(1, (2, 3, items), output, 1, 10)
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(result["proven_optimal"])
        self.assertEqual(result["one_dimensional_optimum"], 10)
        self.assertEqual(result["value"], 9)
        self.assertEqual(result["one_dimensional_deviation_percent"], 10)
        self.assertGreaterEqual(result["infeasible_selection_cuts"], 1)
        self.assertFalse(result["placements"][0]["rotated"])
        self.assertEqual(result["placements"][0]["id"], 1)

    def test_infeasible_duplicate_selection_does_not_remove_feasible_subsets(self):
        items = [
            {"id": 0, "copy": 1, "width": 3, "height": 2, "profit": 7},
            {"id": 0, "copy": 2, "width": 3, "height": 2, "profit": 7},
            {"id": 1, "copy": 1, "width": 1, "height": 3, "profit": 4},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            beasley2d_opt.solve(1, (4, 3, items), output, 1, 10)
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(result["proven_optimal"])
        self.assertEqual(result["one_dimensional_optimum"], 14)
        self.assertEqual(result["value"], 11)
        self.assertGreaterEqual(result["infeasible_selection_cuts"], 1)

    def test_timeout_does_not_claim_a_1d_or_packing_proof(self):
        items = [{"id": 0, "copy": 1, "width": 1, "height": 1, "profit": 5}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with self.assertRaises(SystemExit):
                beasley2d_opt.solve(1, (1, 1, items), output, 1, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(result["proven_optimal"])
        self.assertIsNone(result["one_dimensional_optimum"])
        self.assertIsNone(result["one_dimensional_deviation_percent"])
        self.assertIsNotNone(result["best_found_seconds"])
        self.assertEqual(result["value"], 5)

    def test_verifier_rejects_rotation(self):
        items = [{"width": 3, "height": 2}]
        placement = {"index": 0, "rotated": True, "position": [0, 0],
                     "dimensions": [2, 3]}
        with self.assertRaisesRegex(ValueError, "rotated"):
            eg2d_opt.verify_placements(2, 3, items, [placement], allow_rotations=False)

    def test_summary_reports_missing_and_incomplete_results(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            result = {"instance": 1, "proven_optimal": True,
                      "one_dimensional_deviation_percent": 10.0,
                      "runtime_seconds": 4.0, "best_found_seconds": 3.0}
            (results_dir / "ngcutfs1-001.json").write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaises(SystemExit):
                beasley2d_opt.summarize(results_dir)
            summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["results_present"], 1)
        self.assertEqual(len(summary["missing_instances"]), 209)
        self.assertEqual(summary["average_1d_deviation_percent"], 10)
        self.assertEqual(summary["average_time_to_best_seconds"], 3)


if __name__ == "__main__":
    unittest.main()
