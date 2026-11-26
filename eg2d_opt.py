"""Exact single-bin 2D knapsack solver for Egeblad .2kp instances."""

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


def read_instance(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file, skipinitialspace=True))
    if not rows or len(rows[0]) != 3 or rows[0][0].strip() != "dim":
        raise ValueError(f"{path}: expected dim,width,height header")
    width, height = (int(value) for value in rows[0][1:])
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: bin dimensions must be positive")

    items = []
    seen_ids = set()
    for line, row in enumerate(rows[1:], 2):
        if len(row) != 6 or row[0].strip() != "rect":
            raise ValueError(f"{path}:{line}: expected rect,id,width,height,profit,copies")
        item_id, w, h, profit, copies = (int(value) for value in row[1:])
        if item_id in seen_ids or min(w, h, copies) <= 0 or profit < 0:
            raise ValueError(f"{path}:{line}: duplicate ID or invalid item values")
        seen_ids.add(item_id)
        for copy in range(copies):
            items.append({"id": item_id, "copy": copy + 1,
                          "width": w, "height": h, "profit": profit})
    if not items:
        raise ValueError(f"{path}: no rectangles")
    return width, height, items


def build_model(width, height, items):
    model = gp.Model("2DKP")
    n = len(items)
    selected = model.addVars(n, vtype=GRB.BINARY, name="selected")
    rotated = model.addVars(n, vtype=GRB.BINARY, name="rotated")
    x = model.addVars(n, lb=0, ub=width, name="x")
    y = model.addVars(n, lb=0, ub=height, name="y")
    w = {}
    h = {}

    for i, item in enumerate(items):
        wi, hi = item["width"], item["height"]
        model.addConstr(rotated[i] <= selected[i])
        w[i] = wi * selected[i] + (hi - wi) * rotated[i]
        h[i] = hi * selected[i] + (wi - hi) * rotated[i]
        model.addConstr(x[i] + w[i] <= width * selected[i])
        model.addConstr(y[i] + h[i] <= height * selected[i])
        if wi == hi:
            model.addConstr(rotated[i] == 0)
        selected[i].Start = 0
        rotated[i].Start = 0
        x[i].Start = 0
        y[i].Start = 0

    for i in range(n):
        for j in range(i + 1, n):
            left = model.addVar(vtype=GRB.BINARY)
            right = model.addVar(vtype=GRB.BINARY)
            below = model.addVar(vtype=GRB.BINARY)
            above = model.addVar(vtype=GRB.BINARY)
            separation = left + right + below + above
            model.addConstr(separation >= selected[i] + selected[j] - 1)
            model.addConstr(separation <= selected[i])
            model.addConstr(separation <= selected[j])
            model.addConstr(x[i] + w[i] <= x[j] + width * (1 - left))
            model.addConstr(x[j] + w[j] <= x[i] + width * (1 - right))
            model.addConstr(y[i] + h[i] <= y[j] + height * (1 - below))
            model.addConstr(y[j] + h[j] <= y[i] + height * (1 - above))

    model.addConstr(gp.quicksum(item["width"] * item["height"] * selected[i]
                                for i, item in enumerate(items)) <= width * height)
    identical = defaultdict(list)
    for i, item in enumerate(items):
        identical[item["width"], item["height"], item["profit"]].append(i)
    for indices in identical.values():
        for first, second in zip(indices, indices[1:]):
            model.addConstr(selected[first] >= selected[second])

    model.setObjective(gp.quicksum(item["profit"] * selected[i]
                                   for i, item in enumerate(items)), GRB.MAXIMIZE)
    return model, selected, rotated, x, y


def verify_placements(width, height, items, placements, tolerance=1e-5):
    seen = set()
    for placement in placements:
        i = placement["index"]
        if i in seen or not 0 <= i < len(items):
            raise ValueError(f"Duplicate or invalid rectangle index: {i}")
        seen.add(i)
        item = items[i]
        dims = ((item["height"], item["width"]) if placement["rotated"]
                else (item["width"], item["height"]))
        px, py = placement["position"]
        w, h = placement["dimensions"]
        if (w, h) != dims or min(px, py) < -tolerance \
                or px + w > width + tolerance or py + h > height + tolerance:
            raise ValueError(f"Rectangle {i} has invalid dimensions or position")
    for a, first in enumerate(placements):
        x1, y1 = first["position"]
        w1, h1 = first["dimensions"]
        for second in placements[a + 1:]:
            x2, y2 = second["position"]
            w2, h2 = second["dimensions"]
            if (x1 + w1 > x2 + tolerance and x2 + w2 > x1 + tolerance
                    and y1 + h1 > y2 + tolerance and y2 + h2 > y1 + tolerance):
                raise ValueError(f"Rectangles {first['index']} and {second['index']} overlap")


def solve(path, output, threads, time_limit):
    started = time.monotonic()
    width, height, items = read_instance(path)
    model, selected, rotated, x, y = build_model(width, height, items)
    model.Params.Threads = threads
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0
    model.Params.MIPGapAbs = 0
    model.Params.Symmetry = 2
    model.Params.NodefileStart = 0.5
    model.optimize()

    placements = []
    if model.SolCount:
        for i, item in enumerate(items):
            if selected[i].X < 0.5:
                continue
            is_rotated = rotated[i].X > 0.5
            dims = ([item["height"], item["width"]] if is_rotated
                    else [item["width"], item["height"]])
            placements.append({"index": i, "id": item["id"], "copy": item["copy"],
                               "position": [x[i].X, y[i].X], "dimensions": dims,
                               "rotated": is_rotated, "profit": item["profit"]})
        verify_placements(width, height, items, placements)
        value = sum(items[p["index"]]["profit"] for p in placements)
        if not math.isclose(value, model.ObjVal, rel_tol=0, abs_tol=1e-5):
            raise ValueError("Extracted packing value does not match the solver objective")
    else:
        value = None

    bound = model.ObjBound
    optimal = model.Status == GRB.OPTIMAL and model.SolCount > 0
    result = {
        "instance": path.stem,
        "container": [width, height],
        "item_count": len(items),
        "allow_rotations": True,
        "status_code": model.Status,
        "proven_optimal": optimal,
        "value": value,
        "dual_bound": bound if math.isfinite(bound) else None,
        "runtime_seconds": time.monotonic() - started,
        "solver_seconds": model.Runtime,
        "placements": placements,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Result written to {output}", flush=True)
    if not optimal:
        raise SystemExit(f"Optimality not proven (Gurobi status {model.Status})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=85800)
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.time_limit) or args.time_limit < 0:
        parser.error("threads must be positive and time limit must be finite and nonnegative")
    solve(args.instance, args.output, args.threads, args.time_limit)


if __name__ == "__main__":
    main()
