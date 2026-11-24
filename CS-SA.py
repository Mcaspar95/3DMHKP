#!/usr/bin/env python3
"""Sequence-decoded simulated annealing for the CS multi-drop instances.

The decoder uses conservative single-box support and non-overlapping
destination bands. Its objective is lexicographic: unloaded box cost, used
container cost, destination stops, then occupied container length. These are
not the unpublished weights used in Ceschia and Schaerf's experiments.
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path


CONTAINER_RE = re.compile(
    r"(\S+)\s+(\d+)\s+\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)"
    r"\s+(\S+)\s+(\S+)$"
)
BOX_RE = re.compile(
    r"(\S+)\s+(\d+)\s+\(\s*(\d+)\s+([01])\s+(\S+),\s*"
    r"(\d+)\s+([01])\s+(\S+),\s*(\d+)\s+([01])\s+(\S+)\s*\)"
    r"\s+(\S+)\s+(\d+)\s+(\S+)$"
)


def parse_instance(path):
    containers, boxes = [], []
    section = None
    counts = {}
    with path.open() as stream:
        for line_no, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("ContainerTypes:"):
                section = "containers"
                continue
            if line.startswith("BoxTypes:"):
                section = "boxes"
                continue
            if line.startswith("Total volume containers "):
                section = "footer"
                continue
            if section == "footer":
                raise ValueError(f"{path}:{line_no}: unexpected data after footer")
            if section is None:
                key, sep, value = line.partition(":")
                if not sep:
                    raise ValueError(f"{path}:{line_no}: invalid header")
                if key in ("NumContainerTypes", "NumBoxTypes", "NumClients"):
                    counts[key] = int(value.strip())
                continue
            match = (CONTAINER_RE if section == "containers" else BOX_RE).fullmatch(line)
            if match is None:
                raise ValueError(f"{path}:{line_no}: invalid {section} row: {line}")
            fields = match.groups()
            if section == "containers":
                name, quantity, length, depth, height, capacity, cost = fields
                entry = {"type": name, "dims": (int(length), int(depth), int(height)),
                         "capacity": float(capacity), "cost": float(cost)}
                containers.extend(entry.copy() for _ in range(int(quantity)))
                counts["parsed_containers"] = counts.get("parsed_containers", 0) + 1
            else:
                name, quantity = fields[:2]
                dims = tuple(int(fields[i]) for i in (2, 5, 8))
                flags = tuple(int(fields[i]) for i in (3, 6, 9))
                bearing = tuple(float(fields[i]) for i in (4, 7, 10))
                weight, client, cost = float(fields[11]), int(fields[12]), float(fields[13])
                orientations = []
                for vertical in range(3):
                    if flags[vertical]:
                        a, b = [axis for axis in range(3) if axis != vertical]
                        for x, y in ((a, b), (b, a)):
                            orientation = ((dims[x], dims[y], dims[vertical]),
                                           bearing[vertical])
                            if orientation not in orientations:
                                orientations.append(orientation)
                if not orientations:
                    raise ValueError(f"{path}:{line_no}: box has no allowed orientation")
                entry = {"type": name, "orientations": orientations, "weight": weight,
                         "client": client, "cost": cost}
                boxes.extend(entry.copy() for _ in range(int(quantity)))
                counts["parsed_boxes"] = counts.get("parsed_boxes", 0) + 1
    if counts.get("parsed_containers") != counts.get("NumContainerTypes"):
        raise ValueError(f"{path}: container type count mismatch")
    if counts.get("parsed_boxes") != counts.get("NumBoxTypes"):
        raise ValueError(f"{path}: box type count mismatch")
    if not containers or not boxes:
        raise ValueError(f"{path}: no containers or boxes")
    if any(b["client"] >= counts.get("NumClients", 0) for b in boxes):
        raise ValueError(f"{path}: client outside declared range")
    return containers, boxes


def overlap(a, b):
    return all(a[i] < b[i] + b[i + 3] and b[i] < a[i] + a[i + 3]
               for i in range(3))


def decode(containers, boxes, order, preferences, deadline):
    """Pack in client bands; each elevated box rests on exactly one box."""
    remaining = set(order)
    placements = []
    container_order = sorted(range(len(containers)),
                             key=lambda i: (containers[i]["cost"], -math.prod(containers[i]["dims"])))
    client_order = sorted({box["client"] for box in boxes})
    for ci in container_order:
        if not remaining or time.monotonic() >= deadline:
            break
        container = containers[ci]
        limit_x, limit_y, limit_z = container["dims"]
        placed = []
        points = {(0, 0, 0)}
        weight = 0.0
        band_start = 0
        for client in client_order:
            band_end = band_start
            failed = set()
            for bi in order:
                if time.monotonic() >= deadline:
                    return placements
                if bi not in remaining or boxes[bi]["client"] != client:
                    continue
                box = boxes[bi]
                if box["type"] in failed:
                    continue
                if weight + box["weight"] > container["capacity"]:
                    continue
                choices = box["orientations"]
                start = preferences[bi] % len(choices)
                hit = None
                for x, y, z in sorted(points, key=lambda p: (p[1], p[2], p[0])):
                    if y < band_start:
                        continue
                    for offset in range(len(choices)):
                        dims, bearing = choices[(start + offset) % len(choices)]
                        dx, dy, dz = dims
                        if x + dx > limit_x or y + dy > limit_y or z + dz > limit_z:
                            continue
                        rect = (x, y, z, dx, dy, dz)
                        if any(overlap(rect, p["rect"]) for p in placed):
                            continue
                        support = None
                        if z:
                            for pi, p in enumerate(placed):
                                px, py, pz, pdx, pdy, pdz = p["rect"]
                                if (pz + pdz == z and px <= x and py <= y
                                        and x + dx <= px + pdx and y + dy <= py + pdy):
                                    support = pi
                                    break
                            if support is None:
                                continue
                            ancestor = support
                            while ancestor is not None:
                                p = placed[ancestor]
                                if p["load"] + box["weight"] > p["capacity"] + 1e-9:
                                    break
                                ancestor = p["support"]
                            if ancestor is not None:
                                continue
                        hit = (rect, bearing, support)
                        break
                    if hit is not None:
                        break
                if hit is None:
                    failed.add(box["type"])
                    continue
                rect, bearing, support = hit
                ancestor = support
                while ancestor is not None:
                    placed[ancestor]["load"] += box["weight"]
                    ancestor = placed[ancestor]["support"]
                x, y, z, dx, dy, dz = rect
                placed.append({"rect": rect, "support": support, "load": 0.0,
                               "capacity": bearing * dx * dy})
                placements.append({"box": bi, "container": ci, "position": [x, y, z],
                                   "box_type": box["type"], "client": client,
                                   "container_type": container["type"],
                                   "dimensions": [dx, dy, dz], "bearing": bearing})
                remaining.remove(bi)
                weight += box["weight"]
                band_end = max(band_end, y + dy)
                points = {point for point in points
                          if not (x <= point[0] < x + dx and y <= point[1] < y + dy
                                  and z <= point[2] < z + dz)}
                for point in ((x + dx, y, z), (x, y + dy, z), (x, y, z + dz)):
                    if (point[0] < limit_x and point[1] < limit_y and point[2] < limit_z
                            and not any(px <= point[0] < px + pdx
                                        and py <= point[1] < py + pdy
                                        and pz <= point[2] < pz + pdz
                                        for px, py, pz, pdx, pdy, pdz in
                                        (p["rect"] for p in placed))):
                        points.add(point)
            if band_end > band_start:
                band_start = band_end
                if band_start < limit_y:
                    points.add((0, band_start, 0))
    return placements


def objective(containers, boxes, placements):
    packed = {p["box"] for p in placements}
    by_container = {}
    for p in placements:
        by_container.setdefault(p["container"], []).append(p)
    unloaded = sum(b["cost"] for i, b in enumerate(boxes) if i not in packed)
    container_cost = sum(containers[i]["cost"] for i in by_container)
    stops = sum(len({boxes[p["box"]]["client"] for p in ps})
                for ps in by_container.values())
    length = sum(max(p["position"][1] + p["dimensions"][1] for p in ps)
                 for ps in by_container.values())
    return unloaded, container_cost, stops, length


def verify(containers, boxes, placements):
    seen = set()
    groups = {}
    for p in placements:
        bi, ci = p["box"], p["container"]
        if bi in seen or not 0 <= bi < len(boxes) or not 0 <= ci < len(containers):
            raise ValueError("invalid or duplicate placement index")
        seen.add(bi)
        box, container = boxes[bi], containers[ci]
        dims, pos = tuple(p["dimensions"]), tuple(p["position"])
        if (dims, p["bearing"]) not in box["orientations"]:
            raise ValueError(f"box {bi}: invalid orientation")
        if any(pos[i] < 0 or pos[i] + dims[i] > container["dims"][i]
               for i in range(3)):
            raise ValueError(f"box {bi}: outside container")
        groups.setdefault(ci, []).append(p)
    for ci, group in groups.items():
        if sum(boxes[p["box"]]["weight"] for p in group) > containers[ci]["capacity"] + 1e-9:
            raise ValueError(f"container {ci}: weight limit exceeded")
        support = {}
        for i, p in enumerate(group):
            rect = tuple(p["position"]) + tuple(p["dimensions"])
            for q in group[:i]:
                other = tuple(q["position"]) + tuple(q["dimensions"])
                if overlap(rect, other):
                    raise ValueError(f"container {ci}: overlapping boxes")
                if boxes[p["box"]]["client"] != boxes[q["box"]]["client"]:
                    if not (rect[1] + rect[4] <= other[1]
                            or other[1] + other[4] <= rect[1]):
                        raise ValueError(f"container {ci}: overlapping destination bands")
            x, y, z, dx, dy, _ = rect
            if z:
                supporters = [j for j, q in enumerate(group)
                              if q["position"][2] + q["dimensions"][2] == z
                              and q["position"][0] <= x and q["position"][1] <= y
                              and x + dx <= q["position"][0] + q["dimensions"][0]
                              and y + dy <= q["position"][1] + q["dimensions"][1]]
                if not supporters:
                    raise ValueError(f"box {p['box']}: insufficient support")
                support[i] = supporters[0]
        loads = [0.0] * len(group)
        for i, p in enumerate(group):
            ancestor = support.get(i)
            while ancestor is not None:
                loads[ancestor] += boxes[p["box"]]["weight"]
                ancestor = support.get(ancestor)
        for i, p in enumerate(group):
            limit = p["bearing"] * p["dimensions"][0] * p["dimensions"][1]
            if loads[i] > limit + 1e-9:
                raise ValueError(f"box {p['box']}: load-bearing limit exceeded")


def solve(containers, boxes, seconds, seed):
    rng = random.Random(seed)
    start = time.monotonic()
    deadline = start + seconds
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i]["cost"])
    preferences = [0] * len(boxes)
    current = decode(containers, boxes, order, preferences, deadline)
    current_score = objective(containers, boxes, current)
    best, best_score = current, current_score
    iterations = 0
    scales = (max(1.0, sum(b["cost"] for b in boxes)),
              max(1.0, sum(c["cost"] for c in containers)),
              max(1, len(boxes)),
              max(1, sum(c["dims"][1] for c in containers)))
    while time.monotonic() < deadline and len(boxes) > 1:
        trial_order = order.copy()
        trial_pref = preferences.copy()
        move = rng.randrange(3)
        i, j = rng.sample(range(len(boxes)), 2)
        if move == 0:
            trial_order[i], trial_order[j] = trial_order[j], trial_order[i]
        elif move == 1:
            trial_order.insert(j, trial_order.pop(i))
        else:
            trial_pref[i] = rng.randrange(len(boxes[i]["orientations"]))
        trial = decode(containers, boxes, trial_order, trial_pref, deadline)
        if time.monotonic() >= deadline:
            break
        score = objective(containers, boxes, trial)
        iterations += 1
        if score < best_score:
            best, best_score = trial, score
        delta = next(((a - b) / scale for a, b, scale in
                      zip(score, current_score, scales) if a != b), 0.0)
        progress = (time.monotonic() - start) / seconds
        temperature = 0.1 * (0.001 ** progress)
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            order, preferences, current_score = trial_order, trial_pref, score
    return best, best_score, iterations, time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("results_CS_SA"))
    args = parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--time-limit must be finite and positive")
    containers, boxes = parse_instance(args.instance)
    placements, score, iterations, elapsed = solve(
        containers, boxes, args.time_limit, args.seed)
    verify(containers, boxes, placements)
    used_containers = {p["container"] for p in placements}
    result = {
        "instance": args.instance.name,
        "seed": args.seed,
        "time_limit_seconds": args.time_limit,
        "elapsed_seconds": elapsed,
        "iterations": iterations,
        "objective": {"unloaded_box_cost": score[0], "container_cost": score[1],
                      "destination_stops": score[2], "occupied_length": score[3],
                      "unused_length": sum(containers[i]["dims"][1]
                                           for i in used_containers) - score[3]},
        "packed_boxes": len(placements),
        "total_boxes": len(boxes),
        "placements": placements,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / f"{args.instance.stem}-SA-{args.time_limit:g}s.json"
    with output.open("w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"{args.instance.name}: packed {len(placements)}/{len(boxes)}, "
          f"objective {score}, {iterations} iterations, {elapsed:.2f}s")
    print(f"report -> {output}")


if __name__ == "__main__":
    main()
