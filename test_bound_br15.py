import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bound_br15 as br


class BR15BoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = br.INSTANCE_DIR / "BR15-001.txt"
        cls.inst = br.sa.build_instance(cls.path, value_mode="volume")
        cls.report = next(br.REPORT_DIR.glob("BR15-001-*.txt"))

    def test_conversion_and_checked_packing(self):
        payload = br.bound_instance(self.inst)
        self.assertEqual(payload["container"], [587, 220, 233])
        self.assertTrue(payload["allow_rotations"])
        self.assertEqual(sum(i["copies"] for i in payload["items"]), 119)
        self.assertEqual(sum(i["copies"] * i["profit"] for i in payload["items"]), 30069252)
        self.assertEqual(br.checked_report(self.inst, self.report), 25705619)

    def test_rejects_invalid_report(self):
        text = self.report.read_text().replace("pos (359, 70, 0)", "pos (9999, 70, 0)", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "sticks out"):
                br.checked_report(self.inst, path)

    def test_relaxation_incumbent_is_not_lower_bound(self):
        result = {"upper_bound": 30069252.0, "model_value": 30069252.0}
        with patch.object(br, "solve_isolated", return_value=result):
            answer = br.run_one(self.path, br.REPORT_DIR, "full", "ip", 1, 2)
        self.assertEqual(answer["lower_bound"], 25705619)
        self.assertEqual(answer["bound"]["model_value"], 30069252.0)
        self.assertAlmostEqual(answer["relative_gap"], (30069252 - 25705619) / 30069252)

    def test_rejects_bound_below_verified_packing(self):
        with patch.object(br, "solve_isolated", return_value={"upper_bound": 1.0}):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                br.run_one(self.path, br.REPORT_DIR, "full", "ip", 1, 2)


class BR0BoundTests(unittest.TestCase):
    def test_conversion_and_best_report(self):
        instance = br.sa.build_instance(br.ROOT / "BischoffRatcliff/BR0/BR0-001.txt")
        payload = br.bound_instance(instance)
        self.assertEqual(payload["container"], [587, 220, 233])
        self.assertEqual(payload["items"], [{"id": "1", "dimensions": [108, 76, 30],
                                             "copies": 122, "profit": 246240}])
        report_dir = br.ROOT / "results_3DMHKP_BR0"
        values = [br.checked_report(instance, p) for p in report_dir.glob("BR0-001-*.txt")]
        self.assertGreaterEqual(len(values), 2)
        self.assertEqual(br.best_report(instance, report_dir)[0], max(values))

    def test_class_selects_expected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bound = {"upper_bound": 30089620.0, "status": "test", "bound_source": "test"}
            with patch.object(br, "solve_isolated", return_value=bound):
                br.main(["--class", "BR0", "--instances", "BR0-001", "--output-dir", str(output)])
            self.assertTrue((output / "BR0-001-full-ip.json").exists())

    def test_batch_continues_after_solver_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bound = {"upper_bound": 30089620.0, "status": "test", "bound_source": "test"}
            with patch.object(br, "solve_isolated", side_effect=[RuntimeError("bad bound"), bound]):
                result = br.main(["--class", "BR0", "--instances", "BR0-001", "BR0-002",
                                  "--output-dir", str(output)])
            self.assertEqual(result, 1)
            self.assertFalse((output / "BR0-001-full-ip.json").exists())
            self.assertTrue((output / "BR0-002-full-ip.json").exists())


class BRMixedClassTests(unittest.TestCase):
    def test_selects_br1_and_accepts_missing_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bound = {"upper_bound": 30089620.0, "status": "test", "bound_source": "test"}
            with patch.object(br, "solve_isolated", return_value=bound) as solve:
                result = br.main(["--class", "BR1", "--instances", "BR1-001",
                                  "--report-dir", str(output / "no-reports"), "--output-dir", str(output)])
            self.assertEqual(result, 0)
            self.assertTrue((output / "BR1-001-full-ip.json").exists())
            self.assertGreater(len(solve.call_args.args[0]["items"]), 1)
            import json
            data = json.loads((output / "BR1-001-full-ip.json").read_text())
            self.assertIsNone(data["lower_bound"])


if __name__ == "__main__":
    unittest.main()