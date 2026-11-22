#!/usr/bin/env python3
"""Bound converted free-rotation BR0–BR15 instances and verify SA placements.

The packing-bounds IP is a relaxation, not a geometric packing solver. Its
model incumbent must never be treated as a feasible packing. Each result is
stored separately so long runs can be resumed with --skip-existing.
"""
import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

from packing_bounds import solve_isolated

ROOT = Path(__file__).resolve().parent
INSTANCE_DIR = ROOT / "BischoffRatcliff" / "BR15"
REPORT_DIR = ROOT / "results_3DMHKP_BR15"
OUTPUT_DIR = ROOT / "results_bounds_BR15"

spec = importlib.util.spec_from_file_location("br15_sa", ROOT / "3DMHKP-SA.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)

PLACEMENT = re.compile(
    r"^\s*box\s+(\d+) type (\d+) -> container (\d+) pos "
    r"\((-?\d+),\s*(-?\d+),\s*(-?\d+)\) dims "
    r"\((\d+),\s*(\d+),\s*(\d+)\)\s*$", re.M
)


def bound_instance(inst):
    if len(inst["containers"]) != 1:
        raise ValueError("packing-bounds requires exactly one container")
    if any(box["coef"] != 1 for box in inst["box_types"]):
        raise ValueError("BR box value coefficients must be 1")
    return {
        "name": inst["name"],
        "container": list(inst["containers"][0]["dims"]),
        "allow_rotations": True,
        "items": [
            {"id": str(box["type"]), "dimensions": list(box["dims"]),
             "copies": box["count"], "profit": math.prod(box["dims"])}
            for box in inst["box_types"]
        ],
    }


def checked_report(inst, path):
    """Reconstruct and verify the actual placement, not just its report flag."""
    text = path.read_text()
    name = re.search(r"^Instance:\s*(\S+)\s*$", text, re.M)
    claimed = re.search(r"^Packed value \(SA incumbent / lower bound\):\s*([\d.]+)\s*$", text, re.M)
    if not name or name.group(1) != inst["name"] or not claimed:
        raise ValueError(f"{path}: missing or mismatched instance/value")
    placement = {}
    for m in PLACEMENT.finditer(text):
        j, box_type, k, *coordinates = map(int, m.groups())
        if j in placement or not 0 <= j < len(inst["boxes"]) or not 0 <= k < len(inst["containers"]):
            raise ValueError(f"{path}: duplicate or invalid box/container index")
        box = inst["boxes"][j]
        dims = tuple(coordinates[3:])
        if box["type"] != box_type or dims not in box["orientations"]:
            raise ValueError(f"{path}: invalid type or orientation for box {j}")
        placement[j] = (k, box["orientations"].index(dims), tuple(coordinates[:3]))
    value = sum(inst["boxes"][j]["vol"] for j in placement)
    if not placement or float(claimed.group(1)) != value:
        raise ValueError(f"{path}: missing placements or packed value mismatch")
    errors = sa.verify_placement(inst, placement)
    if errors:
        raise ValueError(f"{path}: {errors[0]}")
    return value


def best_report(inst, report_dir):
    best = None
    for path in sorted(report_dir.glob(inst["name"] + "-*.txt")):
        value = checked_report(inst, path)
        if best is None or value > best[0]:
            best = (value, path)
    return best


def run_one(path, report_dir, stage, mode, time_limit, wall_time_limit):
    inst = sa.build_instance(path, value_mode="volume")
    payload = bound_instance(inst)
    best = best_report(inst, report_dir)
    bound = solve_isolated(payload, stage=stage, mode=mode,
                           time_limit=time_limit, wall_time_limit=wall_time_limit)
    upper = bound["upper_bound"]
    lower = best[0] if best else None
    if lower is not None and lower > upper + max(1e-5, 64 * math.ulp(upper)):
        raise ValueError(f"{inst['name']}: verified packing exceeds numerical upper bound")
    return {
        "instance": inst["name"],
        "objective": "packed volume", "rotation_policy": "all axis permutations",
        "warning": "Converted instances omit original vertical-orientation restrictions.",
        "container_volume": math.prod(payload["container"]),
        "lower_bound": lower, "lower_bound_report": str(best[1]) if best else None,
        "relative_gap": (upper - lower) / upper if lower is not None and upper else None,
        "bound": bound,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="instance_class", choices=tuple(f"BR{i}" for i in range(16)), default="BR15")
    parser.add_argument("--instance-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--instances", nargs="+", help="instance stems, e.g. BR15-001")
    parser.add_argument("--stage", default="full", choices=("volume", "fragments", "shapes", "baseline", "floor_mixed", "rounding_mixed", "data_mixed", "full"))
    parser.add_argument("--mode", default="ip", choices=("ip", "lp"))
    parser.add_argument("--time-limit", type=float, default=30)
    parser.add_argument("--wall-time-limit", type=float, default=60)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    args.instance_dir = args.instance_dir or ROOT / "BischoffRatcliff" / args.instance_class
    args.report_dir = args.report_dir or ROOT / f"results_3DMHKP_{args.instance_class}"
    args.output_dir = args.output_dir or ROOT / f"results_bounds_{args.instance_class}"
    paths = ([args.instance_dir / (name + ".txt") for name in args.instances]
             if args.instances else sorted(args.instance_dir.glob(f"{args.instance_class}-*.txt")))
    if not paths:
        parser.error("no instances found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for path in paths:
        output = args.output_dir / f"{path.stem}-{args.stage}-{args.mode}.json"
        if args.skip_existing and output.exists():
            print(f"{path.stem}: skipped {output}", flush=True)
            continue
        try:
            result = run_one(path, args.report_dir, args.stage, args.mode,
                             args.time_limit, args.wall_time_limit)
        except (ValueError, RuntimeError) as exc:
            failures.append(path.stem)
            print(f"{path.stem}: ERROR: {exc}", file=sys.stderr, flush=True)
            continue
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        bound = result["bound"]
        print(f"{path.stem}: lower={result['lower_bound']} upper={bound['upper_bound']} "
              f"status={bound['status']} source={bound['bound_source']}", flush=True)
    if failures:
        print(f"Failed instances ({len(failures)}): {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())