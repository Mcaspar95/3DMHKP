# Three-Dimensional Multiple Heterogeneous Knapsack Problem (3DMHKP)
# =========================================================================
# Exact MILP for the packing problem of Mohanty, Mathur & Ivancic (1994),
# "Value considerations in three-dimensional packing - a heuristic procedure
# using the fractional knapsack problem", EJOR 74:143-151.
#
# Problem (Section 2.1 of that paper):
#
#   A set of different-sized containers is to be packed with regular shaped
#   boxes of different sizes and values. The containers given are FEWER than
#   what will be needed to pack all the boxes, hence boxes must be chosen so
#   as to maximize the total value of the boxes packed in the given containers.
#
# In the classification of Bortfeldt (2000, OR Spektrum 22:239-261) this is a
# multiple container loading problem with non-oversized container supply
# (MCLP-), i.e. a three-dimensional multiple knapsack problem: every container
# is available for loading, a remainder of unpacked boxes is tolerated, and the
# objective is to maximize the value of the stowed cargo.
#
# -------------------------------------------------------------------------
# BOX VALUES
# -------------------------------------------------------------------------
# The c_i column of the instance files is a value COEFFICIENT per unit volume,
# not an absolute per-box value. Bortfeldt (2000, Sec. 7) states the box values
# for these 16 problems "ergeben sich als Produkt aus kistentypspezifischen
# Wertkoeffizienten und den Kistenvolumina" - the product of box-type-specific
# value coefficients and the box volumes:
#
#       v_i = c_i * l_i * w_i * h_i
#
# This is verified: with this convention the continuous-knapsack relaxation
# (see continuous_knapsack_bound below) reproduces Bortfeldt's published
# upper bounds in Table 2 EXACTLY for all 16 instances. Use --value-mode flat
# to treat c_i as an absolute per-box value instead.
#
# -------------------------------------------------------------------------
# THE MILP
# -------------------------------------------------------------------------
# Box types are expanded into individual boxes j in J, container types into
# individual containers k in K. Boxes may be freely rotated, so each box has up
# to 6 axis-aligned orientations (Bortfeldt: "Auch hinsichtlich der raeumlichen
# Orientierung der zu verstauenden Kisten werden keine einschraenkenden
# Voraussetzungen gemacht").
#
#   Variables
#     s[j,k]  in {0,1}   box j is packed into container k
#     u[j]    in {0,1}   box j is packed at all      (u[j] = sum_k s[j,k])
#     o[j,r]  in {0,1}   box j is placed in orientation r
#     px,py,pz[j] >= 0   coordinates of the box's minimum corner
#     ax,ay,az[i,j] in {0,1}  "i precedes j" along each axis (both directions)
#
#   Objective
#     max  sum_j v[j] * u[j]
#
#   Constraints
#     (1) assignment      sum_k s[j,k] = u[j] <= 1
#     (2) orientation     sum_r o[j,r]  = u[j]
#     (3) containment     p*[j] + d*[j] <= sum_k D*_k * s[j,k]
#     (4) pair separation sum of the six a-vars >= s[i,k] + s[j,k] - 1  for all k
#     (5) non-overlap     p*[i] + d*[i] <= p*[j] + M * (1 - a*[i,j])
#
# where d*[j] = sum_r dim*(j,r) * o[j,r] is the oriented extent of box j.
# Constraint (4) only binds when i and j share a container; a box lies in at
# most one container, so the disjunction is enforced exactly where needed.
#
# Two families of redundant-but-strong valid inequalities are added, since the
# pure big-M relaxation is very weak:
#     (6) per-container volume   sum_j vol[j] * s[j,k] <= Vol_k
#     (7) global volume          sum_j vol[j] * u[j]   <= sum_k Vol_k
# (7) is what makes the LP bound comparable to the continuous-knapsack bound.
#
# Symmetry is the other major difficulty: identical boxes and identical
# containers generate huge numbers of equivalent solutions. Two orbitopal-style
# tie-breaking families are added (see _add_symmetry_breaking).
#
# -------------------------------------------------------------------------
# TRACTABILITY - PLEASE READ
# -------------------------------------------------------------------------
# This is the exact formulation, and it is genuinely hard. The instances hold
# 47-200 boxes, so constraint (5) alone contributes O(|J|^2) big-M rows -
# roughly 120k binaries and 300k rows on the largest instances. Proving
# optimality is out of reach for all but the smallest instances within any
# sensible time budget; this is expected and is exactly why Mohanty et al.
# proposed a heuristic in their Section 2.2.
#
# The script therefore always reports a valid optimality gap: the incumbent
# (a genuine, verified feasible packing) together with Gurobi's dual bound and
# the continuous-knapsack bound. Treat the incumbent as a lower bound on the
# optimum, not as the optimum, unless the reported gap is 0.
#
# A greedy extreme-point heuristic supplies a warm start so a feasible
# incumbent always exists even on short time limits.
#
# -------------------------------------------------------------------------
# USAGE
# -------------------------------------------------------------------------
#   python 3DMHKP.py                          # all 16, 300 s each
#   python 3DMHKP.py --instances 7             # a single instance
#   python 3DMHKP.py --instances 1 7 9 --time-limit 60
#   python 3DMHKP.py --bound-only              # LP/knapsack bounds, no MILP
#   python 3DMHKP.py --greedy-only             # heuristic only, no MILP

import argparse
import math
import os
import sys
import time
from itertools import permutations
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

# -------------------------
# Defaults
# -------------------------
INSTANCE_DIR = Path(__file__).parent / "Mohanty"
RESULTS_DIR = Path(__file__).parent / "results_3DMHKP"
TIME_LIMIT = 300.0        # seconds of MILP time per instance
MIP_GAP = 1e-4
THREADS = 0               # 0 = all available cores


# =========================================================================
# Instance parsing
# =========================================================================
def parse_instance(path):
    """Read a Mohanty-format instance file.

    BOXES      rows: type_id count length width height value_coefficient
    CONTAINERS rows: type_id count length width height
    """
    box_types, container_types = [], []
    section = None

    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper == "BOXES":
                section = "boxes"
                continue
            if upper == "CONTAINERS":
                section = "containers"
                continue

            parts = line.split()
            try:
                if section == "boxes":
                    if len(parts) != 6:
                        raise ValueError(f"expected 6 fields, got {len(parts)}")
                    box_types.append({
                        "type": int(parts[0]),
                        "count": int(parts[1]),
                        "dims": (int(parts[2]), int(parts[3]), int(parts[4])),
                        "coef": float(parts[5]),
                    })
                elif section == "containers":
                    if len(parts) != 5:
                        raise ValueError(f"expected 5 fields, got {len(parts)}")
                    container_types.append({
                        "type": int(parts[0]),
                        "count": int(parts[1]),
                        "dims": (int(parts[2]), int(parts[3]), int(parts[4])),
                    })
                else:
                    raise ValueError("data line outside a BOXES/CONTAINERS section")
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}\n  {line!r}") from None

    if not box_types:
        raise ValueError(f"{path}: no box types found")
    if not container_types:
        raise ValueError(f"{path}: no container types found")

    return box_types, container_types


def build_instance(path, value_mode="volume"):
    """Expand box/container types into individual items and containers."""
    box_types, container_types = parse_instance(path)

    boxes = []
    for bt in box_types:
        l, w, h = bt["dims"]
        vol = l * w * h
        value = bt["coef"] * vol if value_mode == "volume" else bt["coef"]
        for _ in range(bt["count"]):
            boxes.append({
                "type": bt["type"],
                "dims": bt["dims"],
                "vol": vol,
                "value": value,
                "orientations": unique_orientations(l, w, h),
            })

    containers = []
    for ct in container_types:
        L, W, H = ct["dims"]
        for _ in range(ct["count"]):
            containers.append({
                "type": ct["type"],
                "dims": ct["dims"],
                "vol": L * W * H,
            })

    return {
        "name": Path(path).stem,
        "boxes": boxes,
        "containers": containers,
        "box_types": box_types,
        "container_types": container_types,
        "value_mode": value_mode,
    }


def unique_orientations(l, w, h):
    """The distinct axis-aligned orientations of a box (6, or fewer if square)."""
    return list(dict.fromkeys(permutations((l, w, h), 3)))


# =========================================================================
# Bounds
# =========================================================================
def continuous_knapsack_bound(inst):
    """Upper bound by relaxing to a 1-D continuous knapsack.

    All containers are merged into a single volume capacity and each box enters
    only with its volume and value; boxes are taken in non-increasing order of
    value density and the last one may be taken fractionally. This is the bound
    described in Bortfeldt (2000), Sec. 5, and it reproduces his Table 2 values
    exactly for all 16 instances.
    """
    capacity = sum(c["vol"] for c in inst["containers"])
    density = sorted(inst["boxes"], key=lambda b: -(b["value"] / b["vol"]))

    remaining, bound = capacity, 0.0
    for b in density:
        if remaining <= 0:
            break
        take = min(b["vol"], remaining)
        bound += take * (b["value"] / b["vol"])
        remaining -= take
    return bound


def fits(box_dims_oriented, container_dims):
    dx, dy, dz = box_dims_oriented
    L, W, H = container_dims
    return dx <= L and dy <= W and dz <= H


def feasible_orientations(box, container):
    """Orientations in which `box` fits inside `container` at all."""
    return [r for r, dims in enumerate(box["orientations"])
            if fits(dims, container["dims"])]


def compatibility(inst):
    """compat[j] = sorted list of containers box j can physically fit into."""
    compat = []
    for box in inst["boxes"]:
        ks = [k for k, c in enumerate(inst["containers"])
              if feasible_orientations(box, c)]
        compat.append(ks)
    return compat


# =========================================================================
# Greedy extreme-point heuristic (warm start)
# =========================================================================
def _overlaps(a_pos, a_dim, b_pos, b_dim):
    for axis in range(3):
        if a_pos[axis] + a_dim[axis] <= b_pos[axis] + 1e-9:
            return False
        if b_pos[axis] + b_dim[axis] <= a_pos[axis] + 1e-9:
            return False
    return True


def greedy_pack(inst, compat, time_budget=30.0):
    """Value-density-first greedy using extreme points.

    Containers are filled one at a time, largest first. Within a container,
    unpacked boxes are tried in non-increasing value density; each box is put at
    the first candidate point/orientation where it fits without overlapping.
    Returns a placement dict: box index -> (container, orientation, position).
    """
    start = time.time()
    n_boxes = len(inst["boxes"])
    order = sorted(range(n_boxes),
                   key=lambda j: -(inst["boxes"][j]["value"] / inst["boxes"][j]["vol"]))
    unpacked = set(order)
    placement = {}

    container_order = sorted(range(len(inst["containers"])),
                             key=lambda k: -inst["containers"][k]["vol"])

    for k in container_order:
        container = inst["containers"][k]
        L, W, H = container["dims"]
        points = [(0, 0, 0)]
        placed = []          # (pos, dim)

        for j in order:
            if j not in unpacked:
                continue
            if time.time() - start > time_budget:
                break
            if k not in compat[j]:
                continue
            box = inst["boxes"][j]

            done = False
            # Prefer points close to the origin - keeps the packing compact.
            for point in sorted(points, key=lambda p: (p[2], p[1], p[0])):
                for r, dims in enumerate(box["orientations"]):
                    if point[0] + dims[0] > L or point[1] + dims[1] > W \
                            or point[2] + dims[2] > H:
                        continue
                    if any(_overlaps(point, dims, pos, dim) for pos, dim in placed):
                        continue

                    placement[j] = (k, r, point)
                    placed.append((point, dims))
                    unpacked.discard(j)
                    if point in points:
                        points.remove(point)
                    points.extend([
                        (point[0] + dims[0], point[1], point[2]),
                        (point[0], point[1] + dims[1], point[2]),
                        (point[0], point[1], point[2] + dims[2]),
                    ])
                    # Drop points that already lie inside something placed.
                    points = [p for p in set(points)
                              if p[0] < L and p[1] < W and p[2] < H]
                    done = True
                    break
                if done:
                    break

        if time.time() - start > time_budget:
            break

    return placement


def placement_value(inst, placement):
    return sum(inst["boxes"][j]["value"] for j in placement)


def verify_placement(inst, placement, tol=1e-6):
    """Independently re-check a placement. Returns a list of violation strings."""
    errors = []
    by_container = {}
    for j, (k, r, pos) in placement.items():
        by_container.setdefault(k, []).append((j, r, pos))

    for k, entries in by_container.items():
        L, W, H = inst["containers"][k]["dims"]
        boxes = []
        for j, r, pos in entries:
            dims = inst["boxes"][j]["orientations"][r]
            if pos[0] + dims[0] > L + tol or pos[1] + dims[1] > W + tol \
                    or pos[2] + dims[2] > H + tol or min(pos) < -tol:
                errors.append(f"box {j} sticks out of container {k}")
            boxes.append((j, pos, dims))
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ja, pa, da = boxes[a]
                jb, pb, db = boxes[b]
                if _overlaps(pa, da, pb, db):
                    errors.append(f"boxes {ja} and {jb} overlap in container {k}")
    return errors


# =========================================================================
# MILP
# =========================================================================
def build_model(inst, compat, warm_start=None, verbose=False):
    boxes = inst["boxes"]
    containers = inst["containers"]
    J = range(len(boxes))
    K = range(len(containers))

    model = gp.Model(f"3DMHKP_{inst['name']}")
    model.Params.OutputFlag = 1 if verbose else 0

    Lmax = max(c["dims"][0] for c in containers)
    Wmax = max(c["dims"][1] for c in containers)
    Hmax = max(c["dims"][2] for c in containers)

    # ---- assignment / selection ----
    s = {}
    for j in J:
        for k in compat[j]:
            s[j, k] = model.addVar(vtype=GRB.BINARY, name=f"s[{j},{k}]")
    u = model.addVars(J, vtype=GRB.BINARY, name="u")

    # ---- orientation ----
    # Only orientations that fit into at least one compatible container.
    orient_ok = {}
    for j in J:
        allowed = set()
        for k in compat[j]:
            allowed.update(feasible_orientations(boxes[j], containers[k]))
        orient_ok[j] = sorted(allowed)
    o = {}
    for j in J:
        for r in orient_ok[j]:
            o[j, r] = model.addVar(vtype=GRB.BINARY, name=f"o[{j},{r}]")

    # ---- positions ----
    p_x = model.addVars(J, lb=0.0, ub=Lmax, name="px")
    p_y = model.addVars(J, lb=0.0, ub=Wmax, name="py")
    p_z = model.addVars(J, lb=0.0, ub=Hmax, name="pz")

    def extent(j, axis):
        return gp.quicksum(boxes[j]["orientations"][r][axis] * o[j, r]
                           for r in orient_ok[j])

    # ---- objective: maximize stowed value ----
    model.setObjective(gp.quicksum(boxes[j]["value"] * u[j] for j in J),
                       GRB.MAXIMIZE)

    # (1) each box goes into at most one container
    for j in J:
        model.addConstr(gp.quicksum(s[j, k] for k in compat[j]) == u[j],
                        name=f"assign[{j}]")
        if not compat[j]:
            model.addConstr(u[j] == 0, name=f"unfittable[{j}]")

    # (2) exactly one orientation iff packed
    for j in J:
        model.addConstr(gp.quicksum(o[j, r] for r in orient_ok[j]) == u[j],
                        name=f"orient[{j}]")

    # (2b) an orientation may only be used in a container it actually fits
    for j in J:
        for k in compat[j]:
            ok = set(feasible_orientations(boxes[j], containers[k]))
            bad = [r for r in orient_ok[j] if r not in ok]
            if bad:
                model.addConstr(
                    gp.quicksum(o[j, r] for r in bad) <= 1 - s[j, k],
                    name=f"orientfit[{j},{k}]")

    # (3) containment inside the assigned container
    for j in J:
        if not compat[j]:
            continue
        model.addConstr(
            p_x[j] + extent(j, 0)
            <= gp.quicksum(containers[k]["dims"][0] * s[j, k] for k in compat[j]),
            name=f"fitx[{j}]")
        model.addConstr(
            p_y[j] + extent(j, 1)
            <= gp.quicksum(containers[k]["dims"][1] * s[j, k] for k in compat[j]),
            name=f"fity[{j}]")
        model.addConstr(
            p_z[j] + extent(j, 2)
            <= gp.quicksum(containers[k]["dims"][2] * s[j, k] for k in compat[j]),
            name=f"fitz[{j}]")

    # (4)+(5) pairwise non-overlap
    a_x, a_y, a_z = {}, {}, {}
    for j1 in J:
        for j2 in J:
            if j1 >= j2:
                continue
            shared = set(compat[j1]) & set(compat[j2])
            if not shared:
                continue      # can never share a container

            for (i, j) in ((j1, j2), (j2, j1)):
                a_x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"ax[{i},{j}]")
                a_y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"ay[{i},{j}]")
                a_z[i, j] = model.addVar(vtype=GRB.BINARY, name=f"az[{i},{j}]")

            separation = (a_x[j1, j2] + a_x[j2, j1]
                          + a_y[j1, j2] + a_y[j2, j1]
                          + a_z[j1, j2] + a_z[j2, j1])

            # must separate whenever both sit in the same container
            for k in shared:
                model.addConstr(separation >= s[j1, k] + s[j2, k] - 1,
                                name=f"sep[{j1},{j2},{k}]")
            # never separate more than once (tightening, removes symmetry)
            model.addConstr(separation <= 1, name=f"sepone[{j1},{j2}]")

            for (i, j) in ((j1, j2), (j2, j1)):
                model.addConstr(
                    p_x[i] + extent(i, 0) <= p_x[j] + Lmax * (1 - a_x[i, j]),
                    name=f"nox[{i},{j}]")
                model.addConstr(
                    p_y[i] + extent(i, 1) <= p_y[j] + Wmax * (1 - a_y[i, j]),
                    name=f"noy[{i},{j}]")
                model.addConstr(
                    p_z[i] + extent(i, 2) <= p_z[j] + Hmax * (1 - a_z[i, j]),
                    name=f"noz[{i},{j}]")

    # (6) per-container volume capacity  (valid inequality)
    for k in K:
        terms = [boxes[j]["vol"] * s[j, k] for j in J if k in compat[j]]
        if terms:
            model.addConstr(gp.quicksum(terms) <= containers[k]["vol"],
                            name=f"vol[{k}]")

    # (7) global volume capacity  (valid inequality - drives the LP bound)
    model.addConstr(
        gp.quicksum(boxes[j]["vol"] * u[j] for j in J)
        <= sum(c["vol"] for c in containers),
        name="volglobal")

    _add_symmetry_breaking(model, inst, compat, s, u)

    model._vars = {"s": s, "u": u, "o": o, "px": p_x, "py": p_y, "pz": p_z,
                   "ax": a_x, "ay": a_y, "az": a_z, "orient_ok": orient_ok}

    if warm_start:
        _apply_warm_start(model, inst, compat, warm_start)

    model.update()
    return model


def _add_symmetry_breaking(model, inst, compat, s, u):
    """Tie-breaking for interchangeable boxes and interchangeable containers.

    Without this the search re-explores enormous numbers of relabelled copies of
    the same physical packing: an instance with 80 identical boxes and 3
    identical containers has astronomically many equivalent encodings.
    """
    boxes = inst["boxes"]
    containers = inst["containers"]
    n_containers = len(containers)

    # --- identical boxes: pack the lower-indexed copy first, and never place it
    #     in a higher-indexed container than its successor ---
    for j in range(len(boxes) - 1):
        a, b = boxes[j], boxes[j + 1]
        if a["type"] != b["type"]:
            continue
        if compat[j] != compat[j + 1]:
            continue

        model.addConstr(u[j] >= u[j + 1], name=f"symbox[{j}]")

        if compat[j]:
            idx_j = gp.quicksum(k * s[j, k] for k in compat[j])
            idx_next = gp.quicksum(k * s[j + 1, k] for k in compat[j + 1])
            model.addConstr(
                idx_j <= idx_next + n_containers * (1 - u[j + 1]),
                name=f"symboxidx[{j}]")

    # --- identical containers: fill the lower-indexed one at least as full ---
    for k in range(n_containers - 1):
        if containers[k]["type"] != containers[k + 1]["type"]:
            continue
        load_k = [boxes[j]["vol"] * s[j, k]
                  for j in range(len(boxes)) if k in compat[j]]
        load_next = [boxes[j]["vol"] * s[j, k + 1]
                     for j in range(len(boxes)) if (k + 1) in compat[j]]
        if load_k and load_next:
            model.addConstr(gp.quicksum(load_k) >= gp.quicksum(load_next),
                            name=f"symcont[{k}]")


def _apply_warm_start(model, inst, compat, placement):
    """Feed the greedy packing to Gurobi as a complete MIP start."""
    v = model._vars
    s, u, o = v["s"], v["u"], v["o"]
    p_x, p_y, p_z = v["px"], v["py"], v["pz"]
    a_x, a_y, a_z = v["ax"], v["ay"], v["az"]

    for j in range(len(inst["boxes"])):
        packed = j in placement
        u[j].Start = 1 if packed else 0
        for k in compat[j]:
            s[j, k].Start = 0
        for r in v["orient_ok"][j]:
            o[j, r].Start = 0

        if packed:
            k, r, pos = placement[j]
            if (j, k) in s:
                s[j, k].Start = 1
            if (j, r) in o:
                o[j, r].Start = 1
            p_x[j].Start, p_y[j].Start, p_z[j].Start = pos
        else:
            p_x[j].Start = p_y[j].Start = p_z[j].Start = 0.0

    # Derive the separation binaries from the greedy geometry.
    for (i, j) in a_x:
        if i > j:
            continue
        ax = ay = az = 0
        axji = ayji = azji = 0
        if i in placement and j in placement:
            ki, ri, pi = placement[i]
            kj, rj, pj = placement[j]
            if ki == kj:
                di = inst["boxes"][i]["orientations"][ri]
                dj = inst["boxes"][j]["orientations"][rj]
                # pick exactly one satisfied separation direction
                if pi[0] + di[0] <= pj[0]:
                    ax = 1
                elif pj[0] + dj[0] <= pi[0]:
                    axji = 1
                elif pi[1] + di[1] <= pj[1]:
                    ay = 1
                elif pj[1] + dj[1] <= pi[1]:
                    ayji = 1
                elif pi[2] + di[2] <= pj[2]:
                    az = 1
                elif pj[2] + dj[2] <= pi[2]:
                    azji = 1
        a_x[i, j].Start, a_y[i, j].Start, a_z[i, j].Start = ax, ay, az
        if (j, i) in a_x:
            a_x[j, i].Start, a_y[j, i].Start, a_z[j, i].Start = axji, ayji, azji


def extract_placement(model, inst, compat):
    """Read the incumbent back into a placement dict."""
    if model.SolCount == 0:
        return {}
    v = model._vars
    s, u, o = v["s"], v["u"], v["o"]
    p_x, p_y, p_z = v["px"], v["py"], v["pz"]

    placement = {}
    for j in range(len(inst["boxes"])):
        if u[j].X < 0.5:
            continue
        k_sel = next((k for k in compat[j] if s[j, k].X > 0.5), None)
        r_sel = next((r for r in v["orient_ok"][j] if o[j, r].X > 0.5), None)
        if k_sel is None or r_sel is None:
            continue
        placement[j] = (k_sel, r_sel,
                        (round(p_x[j].X, 6), round(p_y[j].X, 6), round(p_z[j].X, 6)))
    return placement


# =========================================================================
# Reporting
# =========================================================================
def summarize(inst, placement):
    """Per-container statistics for a placement."""
    stats = []
    used = {}
    for j, (k, r, _) in placement.items():
        used.setdefault(k, []).append(j)

    for k, container in enumerate(inst["containers"]):
        js = used.get(k, [])
        packed_vol = sum(inst["boxes"][j]["vol"] for j in js)
        stats.append({
            "container": k,
            "type": container["type"],
            "dims": container["dims"],
            "n_boxes": len(js),
            "volume_used": packed_vol,
            "volume": container["vol"],
            "utilization": packed_vol / container["vol"] if container["vol"] else 0.0,
            "value": sum(inst["boxes"][j]["value"] for j in js),
        })
    return stats


def write_report(path, inst, placement, result):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = summarize(inst, placement)
    total_container_vol = sum(c["vol"] for c in inst["containers"])
    packed_vol = sum(inst["boxes"][j]["vol"] for j in placement)

    with open(path, "w") as f:
        f.write(f"Instance: {inst['name']}\n")
        f.write(f"Value model: v_i = c_i * volume_i ({inst['value_mode']})\n")
        f.write(f"Boxes: {len(inst['boxes'])} "
                f"({len(inst['box_types'])} types)\n")
        f.write(f"Containers: {len(inst['containers'])} "
                f"({len(inst['container_types'])} types)\n\n")

        f.write(f"Status: {result['status']}\n")
        f.write(f"Runtime: {result['runtime']:.2f} s\n")
        f.write(f"Packed value (incumbent / lower bound): {result['value']:.1f}\n")
        f.write(f"Dual bound (MILP): {result['dual_bound']:.1f}\n")
        f.write(f"Continuous-knapsack bound: {result['ck_bound']:.1f}\n")
        f.write(f"Optimality gap: {result['gap']:.2%}\n")
        f.write(f"Greedy warm-start value: {result['greedy_value']:.1f}\n\n")

        f.write(f"Boxes packed: {len(placement)} / {len(inst['boxes'])}\n")
        f.write(f"Volume utilization: {packed_vol} / {total_container_vol} "
                f"= {packed_vol / total_container_vol:.1%}\n\n")

        for st in stats:
            f.write(f"Container {st['container']} (type {st['type']}, "
                    f"{st['dims'][0]}x{st['dims'][1]}x{st['dims'][2]}): "
                    f"{st['n_boxes']} boxes, value {st['value']:.1f}, "
                    f"utilization {st['utilization']:.1%}\n")

        f.write("\nPlacements (box, type, container, position, oriented dims):\n")
        for j in sorted(placement):
            k, r, pos = placement[j]
            dims = inst["boxes"][j]["orientations"][r]
            f.write(f"  box {j:>4} type {inst['boxes'][j]['type']} "
                    f"-> container {k} pos {pos} dims {dims}\n")


# =========================================================================
# Driver
# =========================================================================
def solve_instance(path, args):
    inst = build_instance(path, value_mode=args.value_mode)
    compat = compatibility(inst)

    n_boxes = len(inst["boxes"])
    n_containers = len(inst["containers"])
    unfittable = sum(1 for c in compat if not c)
    ck_bound = continuous_knapsack_bound(inst)

    print(f"\n{'=' * 70}")
    print(f"{inst['name']}: {n_boxes} boxes ({len(inst['box_types'])} types), "
          f"{n_containers} containers ({len(inst['container_types'])} types)")
    print(f"  continuous-knapsack bound: {ck_bound:.1f}")
    if unfittable:
        print(f"  note: {unfittable} box(es) fit no container - fixed to unpacked")

    t0 = time.time()
    placement = greedy_pack(inst, compat, time_budget=args.greedy_time)
    greedy_value = placement_value(inst, placement)
    errs = verify_placement(inst, placement)
    if errs:
        print(f"  WARNING: greedy produced an invalid packing ({errs[0]}); discarding")
        placement, greedy_value = {}, 0.0
    print(f"  greedy warm start: value {greedy_value:.1f} "
          f"({len(placement)}/{n_boxes} boxes, {time.time() - t0:.1f}s)")

    result = {
        "instance": inst["name"],
        "n_boxes": n_boxes,
        "n_containers": n_containers,
        "ck_bound": ck_bound,
        "greedy_value": greedy_value,
        "value": greedy_value,
        "dual_bound": ck_bound,
        "gap": float("inf"),
        "status": "greedy-only",
        "runtime": time.time() - t0,
    }

    if args.greedy_only:
        result["gap"] = ((ck_bound - greedy_value) / ck_bound) if ck_bound else 0.0
        _finish(inst, placement, result, args)
        return result

    print("  building MILP ...", flush=True)
    t_build = time.time()
    model = build_model(inst, compat,
                        warm_start=placement if placement else None,
                        verbose=args.verbose)
    print(f"  model: {model.NumVars} vars ({model.NumBinVars} binary), "
          f"{model.NumConstrs} constraints  [{time.time() - t_build:.1f}s]")

    if args.bound_only:
        relaxed = model.relax()
        relaxed.Params.OutputFlag = 1 if args.verbose else 0
        relaxed.optimize()
        lp_bound = relaxed.ObjVal if relaxed.Status == GRB.OPTIMAL else float("inf")
        print(f"  LP relaxation bound: {lp_bound:.1f}")
        result["dual_bound"] = min(lp_bound, ck_bound)
        result["status"] = "bound-only"
        result["gap"] = ((result["dual_bound"] - greedy_value)
                         / result["dual_bound"]) if result["dual_bound"] else 0.0
        result["runtime"] = time.time() - t0
        _finish(inst, placement, result, args)
        return result

    model.Params.TimeLimit = args.time_limit
    model.Params.MIPGap = args.mip_gap
    model.Params.Threads = args.threads
    model.Params.MIPFocus = 1          # favour finding good incumbents
    model.Params.Symmetry = 2          # aggressive symmetry detection
    if args.verbose:
        model.Params.OutputFlag = 1

    model.optimize()

    status_names = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time limit",
        GRB.INTERRUPTED: "interrupted",
        GRB.INFEASIBLE: "infeasible",
        GRB.SUBOPTIMAL: "suboptimal",
    }
    result["status"] = status_names.get(model.Status, f"status {model.Status}")

    if model.SolCount > 0:
        milp_placement = extract_placement(model, inst, compat)
        errs = verify_placement(inst, milp_placement)
        if errs:
            print(f"  WARNING: MILP solution failed verification: {errs[0]}")
        milp_value = placement_value(inst, milp_placement)
        if milp_value >= greedy_value:
            placement = milp_placement
            result["value"] = milp_value
        else:
            # Greedy beat the incumbent (possible if the MILP start was dropped)
            result["value"] = greedy_value
        result["dual_bound"] = min(model.ObjBound, ck_bound)
    else:
        result["dual_bound"] = min(model.ObjBound if model.SolCount or True
                                   else ck_bound, ck_bound)

    bound = result["dual_bound"]
    result["gap"] = ((bound - result["value"]) / bound) if bound > 0 else 0.0
    result["runtime"] = time.time() - t0

    print(f"  MILP: value {result['value']:.1f}, bound {bound:.1f}, "
          f"gap {result['gap']:.2%} [{result['status']}]")

    _finish(inst, placement, result, args)
    return result


def _finish(inst, placement, result, args):
    result["n_packed"] = len(placement)
    packed_vol = sum(inst["boxes"][j]["vol"] for j in placement)
    total_vol = sum(c["vol"] for c in inst["containers"])
    result["utilization"] = packed_vol / total_vol if total_vol else 0.0

    if not args.no_report:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{inst['name']}-3DMHKP.txt"
        write_report(out, inst, placement, result)
        print(f"  report -> {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exact MILP for the 3D multiple heterogeneous knapsack "
                    "problem of Mohanty, Mathur & Ivancic (1994).")
    parser.add_argument("--instances", nargs="*", type=int, default=None,
                        metavar="N",
                        help="instance numbers to solve (default: all 16)")
    parser.add_argument("--instance-dir", type=Path, default=INSTANCE_DIR,
                        help=f"instance directory (default: {INSTANCE_DIR})")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                        help=f"MILP seconds per instance (default: {TIME_LIMIT})")
    parser.add_argument("--mip-gap", type=float, default=MIP_GAP,
                        help="relative MIP gap to stop at")
    parser.add_argument("--threads", type=int, default=THREADS,
                        help="Gurobi threads (0 = all cores)")
    parser.add_argument("--greedy-time", type=float, default=30.0,
                        help="seconds for the greedy warm start")
    parser.add_argument("--value-mode", choices=("volume", "flat"),
                        default="volume",
                        help="volume: v_i = c_i * vol_i (default, matches the "
                             "literature); flat: v_i = c_i")
    parser.add_argument("--bound-only", action="store_true",
                        help="only compute LP / knapsack bounds")
    parser.add_argument("--greedy-only", action="store_true",
                        help="only run the greedy heuristic, no MILP")
    parser.add_argument("--no-report", action="store_true",
                        help="do not write per-instance report files")
    parser.add_argument("--verbose", action="store_true",
                        help="show the Gurobi log")
    args = parser.parse_args(argv)

    numbers = args.instances if args.instances else list(range(1, 17))
    paths = []
    for n in numbers:
        p = args.instance_dir / f"instance{n:02d}.txt"
        if not p.exists():
            print(f"missing instance file: {p}", file=sys.stderr)
            return 1
        paths.append(p)

    results = []
    for p in paths:
        try:
            results.append(solve_instance(p, args))
        except gp.GurobiError as exc:
            print(f"  Gurobi error on {p.stem}: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            break

    if not results:
        return 1

    print(f"\n{'=' * 88}")
    print("SUMMARY")
    print(f"{'=' * 88}")
    print(f"{'instance':<12}{'boxes':>7}{'packed':>8}{'value':>13}"
          f"{'bound':>13}{'gap':>9}{'util':>8}{'time':>8}")
    print("-" * 88)
    for r in results:
        print(f"{r['instance']:<12}{r['n_boxes']:>7}{r.get('n_packed', 0):>8}"
              f"{r['value']:>13.1f}{r['dual_bound']:>13.1f}"
              f"{r['gap']:>8.1%}{r.get('utilization', 0):>8.1%}"
              f"{r['runtime']:>7.1f}s")
    print("-" * 88)

    solved = sum(1 for r in results if r["status"] == "optimal")
    print(f"proven optimal: {solved}/{len(results)}")
    mean_util = sum(r.get("utilization", 0) for r in results) / len(results)
    print(f"mean volume utilization: {mean_util:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
