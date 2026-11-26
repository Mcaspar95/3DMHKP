"""Solve the Beasley ngcutfs1 instances by exact knapsack decomposition."""

import argparse
import json
import math
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import eg2d_opt


INSTANCE_FILE = Path(__file__).parent / "Beasley" / "ngcutfs1.txt"
INSTANCE_COUNT = 210


def read_instances(path):
    with path.open(encoding="utf-8") as file:
        lines = iter(enumerate(file, 1))

        def read_row(expected):
            try:
                line_number, line = next(lines)
            except StopIteration:
                raise ValueError(f"{path}: unexpected end of file") from None
            values = line.split()
            if len(values) != expected:
                raise ValueError(f"{path}:{line_number}: expected {expected} integers")
            return [int(value) for value in values]

        count, = read_row(1)
        if count != INSTANCE_COUNT:
            raise ValueError(f"{path}: expected {INSTANCE_COUNT} instances, found {count}")
        instances = []
        for instance_index in range(count):
            types, = read_row(1)
            width, height = read_row(2)
            if min(types, width, height) <= 0:
                raise ValueError(f"{path}: invalid header for instance {instance_index + 1}")
            items = []
            for item_id in range(types):
                w, h, minimum, copies, profit = read_row(5)
                if min(w, h, copies) <= 0 or profit < 0 or minimum != 0:
                    raise ValueError(f"{path}: unsupported item in instance {instance_index + 1}")
                items.extend({"id": item_id, "copy": copy + 1, "width": w,
                              "height": h, "profit": profit} for copy in range(copies))
            instances.append((width, height, items))
        if any(line.strip() for _, line in lines):
            raise ValueError(f"{path}: unexpected data after last instance")
    return instances


def make_master(width, height, items, threads):
    model = gp.Model("beasley_area_knapsack")
    chosen = model.addVars(len(items), vtype=GRB.BINARY, name="chosen")
    model.addConstr(gp.quicksum(item["width"] * item["height"] * chosen[i]
                                for i, item in enumerate(items)) <= width * height)
    for i in range(1, len(items)):
        if items[i]["id"] == items[i - 1]["id"]:
            model.addConstr(chosen[i - 1] >= chosen[i])
    model.setObjective(gp.quicksum(item["profit"] * chosen[i]
                                   for i, item in enumerate(items)), GRB.MAXIMIZE)
    model.Params.Threads = threads
    model.Params.MIPGap = 0
    model.Params.MIPGapAbs = 0
    model.Params.NodefileStart = 0.5
    return model, chosen


def extract_placements(width, height, items, selected_indices, model_vars):
    model, selected, rotated, x, y = model_vars
    placements = []
    for local_index, index in enumerate(selected_indices):
        if selected[local_index].X < 0.5:
            raise ValueError("Feasibility model omitted a required rectangle")
        item = items[index]
        placements.append({
            "index": index, "id": item["id"], "copy": item["copy"],
            "position": [x[local_index].X, y[local_index].X],
            "dimensions": [item["width"], item["height"]],
            "rotated": rotated[local_index].X > 0.5, "profit": item["profit"],
        })
    eg2d_opt.verify_placements(width, height, items, placements, allow_rotations=False)
    value = sum(items[p["index"]]["profit"] for p in placements)
    if not math.isclose(value, model.ObjVal, rel_tol=0, abs_tol=1e-5):
        raise ValueError("Extracted packing value does not match the solver objective")
    return placements, value


def greedy_pack(width, height, items):
    points = {(0, 0)}
    placements = []
    order = sorted(range(len(items)),
                   key=lambda i: (-items[i]["profit"] / (items[i]["width"] * items[i]["height"]),
                                  -items[i]["profit"], i))
    for i in order:
        item = items[i]
        w, h = item["width"], item["height"]
        for x, y in sorted(points, key=lambda p: (p[1], p[0])):
            if x + w > width or y + h > height:
                continue
            if any(x < p["position"][0] + p["dimensions"][0]
                   and x + w > p["position"][0]
                   and y < p["position"][1] + p["dimensions"][1]
                   and y + h > p["position"][1] for p in placements):
                continue
            placements.append({"index": i, "id": item["id"], "copy": item["copy"],
                               "position": [x, y], "dimensions": [w, h],
                               "rotated": False, "profit": item["profit"]})
            points.remove((x, y))
            points.update(((x + w, y), (x, y + h)))
            break
    eg2d_opt.verify_placements(width, height, items, placements, allow_rotations=False)
    return placements


def solve(instance_number, instance, output, threads, time_limit):
    started = time.monotonic()
    deadline = started + time_limit
    width, height, items = instance
    master, chosen = make_master(width, height, items, threads)
    one_d = None
    upper_bound = None
    placements = greedy_pack(width, height, items)
    value = sum(p["profit"] for p in placements)
    best_found_seconds = time.monotonic() - started if placements else None
    greedy_indices = {p["index"] for p in placements}
    for i in range(len(items)):
        chosen[i].Start = 1 if i in greedy_indices else 0
    cuts = 0
    status = "not_proven"
    master_status = None
    packing_status = None

    while time.monotonic() < deadline:
        master.Params.TimeLimit = max(0, deadline - time.monotonic())
        master.optimize()
        master_status = master.Status
        bound = master.ObjBound
        if math.isfinite(bound):
            upper_bound = bound if upper_bound is None else min(upper_bound, bound)
        if cuts == 0 and master.Status == GRB.OPTIMAL:
            one_d = int(round(master.ObjVal))
        if one_d is not None and value == one_d:
            status = "optimal"
            break
        if master.SolCount == 0 or time.monotonic() >= deadline:
            break
        indices = [i for i in range(len(items)) if chosen[i].X > 0.5]
        if not indices:
            if master.Status == GRB.OPTIMAL:
                status = "optimal"
                best_found_seconds = time.monotonic() - started
            break

        subset = [items[i] for i in indices]
        model_vars = eg2d_opt.build_model(width, height, subset, allow_rotations=False)
        packing, selected, _, _, _ = model_vars
        packing.Params.Threads = threads
        packing.Params.TimeLimit = max(0, deadline - time.monotonic())
        packing.Params.MIPGap = 0
        packing.Params.MIPGapAbs = 0
        for i in range(len(subset)):
            packing.addConstr(selected[i] == 1)
        packing.optimize()
        packing_status = packing.Status

        if packing.SolCount:
            candidate, candidate_value = extract_placements(width, height, items, indices, model_vars)
            if candidate_value > value:
                placements, value = candidate, candidate_value
                best_found_seconds = time.monotonic() - started
            if master.Status == GRB.OPTIMAL and candidate_value < value:
                raise ValueError("Feasible master optimum is worse than the greedy packing")
            if master.Status == GRB.OPTIMAL:
                status = "optimal"
            packing.dispose()
            break
        if packing.Status != GRB.INFEASIBLE:
            packing.dispose()
            break
        packing.dispose()
        master.addConstr(gp.quicksum(chosen[i] for i in indices) <= len(indices) - 1)
        cuts += 1

    elapsed = time.monotonic() - started
    deviation = (100 * (one_d - value) / one_d if one_d else
                 0.0 if one_d == 0 else None)
    result = {
        "instance": instance_number,
        "container": [width, height],
        "item_count": len(items),
        "allow_rotations": False,
        "status": status,
        "proven_optimal": status == "optimal",
        "value": value,
        "one_dimensional_optimum": one_d,
        "one_dimensional_deviation_percent": deviation,
        "dual_bound": upper_bound,
        "runtime_seconds": elapsed,
        "best_found_seconds": best_found_seconds,
        "infeasible_selection_cuts": cuts,
        "master_status_code": master_status,
        "packing_status_code": packing_status,
        "placements": placements,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    master.dispose()
    print(f"Result written to {output}", flush=True)
    if status != "optimal":
        raise SystemExit("Optimality not proven; see solver status codes in result")


def summarize(results_dir):
    results = []
    missing = []
    for i in range(1, INSTANCE_COUNT + 1):
        path = results_dir / f"ngcutfs1-{i:03}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["instance"] != i:
                raise ValueError(f"{path}: incorrect instance number")
            results.append(result)
        else:
            missing.append(i)
    comparable = [r for r in results if r["one_dimensional_deviation_percent"] is not None]
    proven = [r for r in comparable if r["proven_optimal"]]
    timed = [r for r in results if r["best_found_seconds"] is not None]
    summary = {
        "instances_expected": INSTANCE_COUNT,
        "results_present": len(results),
        "missing_instances": missing,
        "proven_optimal": sum(r["proven_optimal"] for r in results),
        "exact_1d_bound_available": len(comparable),
        "average_1d_deviation_percent": (
            sum(r["one_dimensional_deviation_percent"] for r in comparable) / len(comparable)
            if comparable else None),
        "average_1d_deviation_proven_optimal_percent": (
            sum(r["one_dimensional_deviation_percent"] for r in proven) / len(proven)
            if proven else None),
        "average_runtime_seconds": (
            sum(r["runtime_seconds"] for r in results) / len(results) if results else None),
        "time_to_best_count": len(timed),
        "average_time_to_best_seconds": (
            sum(r["best_found_seconds"] for r in timed) / len(timed) if timed else None),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if missing:
        raise SystemExit("Some instance results are missing; summary is incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--id", type=int, help="zero-based instance index")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=85800)
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.time_limit) or args.time_limit < 0:
        parser.error("threads must be positive and time limit must be finite and nonnegative")
    if args.summarize:
        summarize(args.results_dir)
        return
    if not 0 <= args.id < INSTANCE_COUNT:
        parser.error(f"id must be between 0 and {INSTANCE_COUNT - 1}")
    instances = read_instances(INSTANCE_FILE)
    solve(args.id + 1, instances[args.id],
          args.results_dir / f"ngcutfs1-{args.id + 1:03}.json",
          args.threads, args.time_limit)


if __name__ == "__main__":
    main()
