# Three-Dimensional Multiple Heterogeneous Knapsack Problem WITH LOAD STABILITY
# =========================================================================
# The packing problem of Mohanty, Mathur & Ivancic (1994), EJOR 74:143-151,
# as solved in 3DMHKP.py, but with the practical constraint of VERTICAL LOAD
# STABILITY imposed: no box may float. Every packed box rests either on the
# floor of its container or on the top faces of boxes below it, and at least a
# fraction alpha of its base area must be supported.
#
# This is the cargo stability constraint of Kurpel, Scarpin, Pecora Junior,
# Schenekemberg & Coelho (2020), "The exact solutions of several types of
# container loading problems", EJOR 284:87-107, Section 3.3.2, itself adapted
# from Junqueira, Morabito & Yamashita (2012), C&OR 39:74-85.
#
# -------------------------------------------------------------------------
# WHY THIS FILE DOES NOT EXTEND THE MILP IN 3DMHKP.py
# -------------------------------------------------------------------------
# 3DMHKP.py (and 3DMHKPup.py) place boxes with CONTINUOUS coordinates px, py,
# pz and resolve overlaps with big-M disjunctions. Kurpel's stability
# constraint cannot be written linearly over such a model. Their constraint
# (19) requires the supported base area to reach alpha times the box base:
#
#   sum_i sum_{g in Omega_i : r'-h_ig >= 0}
#     sum_{p in X_igk : p'-l_ig+1 <= p <= p'+l_la-1}
#       sum_{q in Y_igk : q'-w_ig+1 <= q <= q'+w_la-1}
#         L_il * W_il * x_ig^{jkpq(r'-h_ig)}   >=   alpha * l_la * w_la
#                                                     * x_la^{jkp'q'r'}   (19)
#   L_il = min(p + l_ig, p' + l_la) - max(p, p')                          (20)
#   W_il = min(q + w_ig, q' + w_la) - max(q, q')                          (21)
#
# L_il and W_il are the x- and y-overlaps between the supporting box and the
# supported one. They are CONSTANT COEFFICIENTS only because Kurpel's model
# enumerates positions: p and p' are indices of the summation, known when the
# model is built. Over continuous px the same overlap is
# min(px_i+dx_i, px_j+dx_j) - max(px_i, px_j), so the contact AREA is a product
# of two variable overlap lengths - bilinear, and with no compact exact
# linearization. Kurpel say as much (p. 98): normal patterns lose no optimal
# solution precisely when alpha = 1.
#
# This file therefore adopts Kurpel's own formulation - their 0-1 model over
# discretized placement points, (12)-(15) for output maximization - which makes
# the stability constraint exact AND linear. That change is not a concession:
# on these 16 instances the discretized model is far smaller than the
# continuous one (~60k binaries against ~120k, ~8k rows against ~300k) and it
# carries no symmetry between identical boxes at all, because the variables are
# indexed by box TYPE and position rather than by individual box.
#
# Pass --alpha 0 to drop stability while keeping everything else identical.
# That, not 3DMHKP.py, is the right baseline for costing stability: it isolates
# the constraint from the change of model. See "MEASURING THE COST" below.
#
# -------------------------------------------------------------------------
# THE MODEL - Kurpel et al. (2020), Section 3.1-3.2
# -------------------------------------------------------------------------
# Box types i in {1..m} with dimensions (l_i, w_i, h_i), value v_i and
# availability b_i; containers k with dimensions (L_k, W_k, H_k). Omega_i is
# the set of distinct axis-aligned orientations of box type i (up to 6; fewer
# when the box has equal sides, which is Kurpel's symmetrical-rotation
# elimination, p. 89).
#
#   x_ig^{kpqr} = 1  if a box of type i, in orientation g, has its
#                    back-bottom-left vertex at point (p, q, r) of container k
#
#   max  sum v_i x_ig^{kpqr}                                              (12)
#
#   (13) conflict     for every discretization point (s,t,u) of container k,
#                     the boxes covering it sum to at most 1
#   (14) availability sum over everything of x_ig^{kpqr} <= b_i
#   (15) x binary
#
# Positions are drawn from the NORMAL PATTERNS of Herz (1972) and Christofides
# & Whitlock (1977), Kurpel's Appendix A.1:
#
#   X_k = {p in Z | p = sum_i beta_i l_ig, 0 <= p <= L_k - min_i(l_ig),
#          0 <= beta_i <= b_i}                                           (A.1)
#
# and likewise Y_k, Z_k. Kurpel evaluate four discretizations (NP, RRP, RNP,
# MiM); only NP is implemented here, because it is the only one they found safe
# under stability - RRP and MiM lose optimal solutions (their Appendix B gives a
# counterexample), and "in all tests performed in this paper considering the
# stability of the load, we adopted the NP for the discretization of the
# container" (p. 105).
#
# The discretization is verified against the paper: for the 16 Mohanty
# instances, normal_patterns() reproduces the variable counts of Kurpel's
# Table 2 (MHLOPP, NP column) exactly - min 1493 (instance 3), max 264270
# (instance 9), sum 950161. The sum matches only if instance 4 is taken with
# 10 containers of 2 types rather than the 15 of 3 types in Mohanty/instance04
# .txt, which independently re-confirms what compare_bortfeldt.py found by a
# different route: Kurpel solved a truncated instance 4. Run --self-test.
#
# -------------------------------------------------------------------------
# THE STABILITY CONSTRAINT
# -------------------------------------------------------------------------
# Two equivalent-at-alpha=1 encodings are implemented; --stability-form picks.
#
# "area" is Kurpel's (19)-(21) transcribed, and handles any alpha in [0,1].
#
# "column" is an exact reformulation available only at alpha = 1, and is the
# default there. For each container, each column (s,t) and each level r' > 0:
#
#       A(s,t,r')  <=  B(s,t,r')
#
#   A = sum of x whose BASE is at r' and whose footprint covers column (s,t)
#   B = sum of x whose TOP  is at r' and whose footprint covers column (s,t)
#
# i.e. wherever cargo starts, cargo must end. Constraint (13) already forbids
# two boxes from starting - or two from ending - at the same level in the same
# column, so A and B are each at most 1 and the row is a clean implication.
#
# This is exactly full base support. Any uncovered part of a supported base is
# a rectangle whose edges are box edges, hence lie on the normal-pattern
# lattice, so testing the columns {v in reach : v < L_k} x {v in reach : v < W_k}
# misses nothing. It is much cheaper than (19): summed over the 16 instances,
# (19) needs 8.7e8 nonzeros against 1.3e8 for the column form, and on instance 8
# it is the difference between a model that builds and one that does not.
#
# alpha defaults to 1. Kurpel use alpha = 1 throughout - "requiring 100% of
# support to the bottom of the boxes, the most restricted approach to static
# stability" (p. 102).
#
# Note their own caveat (p. 92): alpha controls the bearing area, and does not
# by itself guarantee stability under acceleration and vehicle oscillation
# (Ramos, Oliveira & Lopes, 2016, on static mechanical equilibrium).
#
# -------------------------------------------------------------------------
# MEASURING THE COST OF STABILITY
# -------------------------------------------------------------------------
#   python 3DMHKPst.py --alpha 0 --tag nostab     # same model, no stability
#   python 3DMHKPst.py                            # same model, alpha = 1
#
# Both write to results_3DMHKPst/, so the two runs differ in exactly one
# constraint family. Comparing either against results_3DMHKP/ instead mixes the
# cost of stability with the change of formulation.
#
# KURPEL_STABILITY below transcribes their Table E1 ("With load stability"),
# which is the published reference for this exact problem, and is printed
# alongside our own values in the run summary.
#
# -------------------------------------------------------------------------
# TRACTABILITY - PLEASE READ
# -------------------------------------------------------------------------
# Constraint (13) is dense: each variable appears in one row per discretization
# point its box covers, which on the larger instances is several hundred. The
# model is comfortably buildable for 13 of the 16 instances, and is not for
# instances 8, 9 and 16 (76M, 201M and 248M nonzeros in (13) alone). Those are
# skipped by default; --max-nnz raises the ceiling. Kurpel ran on a machine
# with 120 GB of RAM and still reported no dual bound for instances 7, 8, 9 and
# 16 under stability (Table E1).
#
# As in 3DMHKP.py, a greedy warm start - here stability-aware - guarantees a
# feasible incumbent, every reported packing is re-verified independently by
# verify_placement(), and the incumbent should be read as a lower bound unless
# the reported gap is 0.
#
# -------------------------------------------------------------------------
# USAGE
# -------------------------------------------------------------------------
#   python 3DMHKPst.py                          # all 16, alpha = 1, 300 s each
#   python 3DMHKPst.py --instances 1 3 10
#   python 3DMHKPst.py --alpha 0.75             # partial support, eq. (19)
#   python 3DMHKPst.py --alpha 0 --tag nostab   # stability off
#   python 3DMHKPst.py --self-test              # check against Kurpel Table 2

import argparse
import sys
import time
from bisect import bisect_left, bisect_right
from itertools import permutations
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

# -------------------------
# Defaults
# -------------------------
INSTANCE_DIR = Path(__file__).parent / "Mohanty"
RESULTS_DIR = Path(__file__).parent / "results_3DMHKPst"
TIME_LIMIT = 300.0
MIP_GAP = 1e-4
THREADS = 0
ALPHA = 1.0
MAX_NNZ = 25_000_000
TOL = 1e-6

# ---------------------------------------------------------------------------
# Transcribed from Kurpel et al. (2020), Table E1, "With load stability".
# Their run: 3600 s per instance, Gurobi 8.1.0, 2.77 GHz Xeon, up to 120 GB.
# None = the solver returned no dual bound (their "-").
# ---------------------------------------------------------------------------
KURPEL_STABILITY = {
    1: (8640.00, 1.11), 2: (85376.00, 0.18), 3: (53262.50, 0.00),
    4: (1354752.00, 0.00), 5: (583750.00, 0.00), 6: (142464.00, 0.67),
    7: (10908.00, None), 8: (40556.20, None), 9: (92052.00, None),
    10: (15360.00, 0.00), 11: (54761.00, 0.00), 12: (18504.00, 33.75),
    13: (36556.80, 0.00), 14: (68723.20, 0.62), 15: (40807.80, 0.00),
    16: (469266.00, None),
}
# Instances Kurpel proved optimal UNDER STABILITY (bold in their Table E1).
KURPEL_STABILITY_OPTIMAL = {3, 4, 5, 10, 11, 13, 15}
# Instance 4 is not comparable - they solved a 10-container truncation of it.
# See the header of compare_bortfeldt.py, and --self-test below.
KURPEL_INCOMPARABLE = {4}

# Kurpel Table 2, MHLOPP, NP column - what --self-test checks against.
KURPEL_TABLE2_NP = {"min": 1493, "max": 264270, "sum": 950161}


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


def unique_orientations(l, w, h):
    """The distinct axis-aligned orientations of a box (6, or fewer if square).

    Dropping duplicates is Kurpel's elimination of symmetrical rotations
    (p. 89): a box with l = w has orientation 1 equivalent to 3, 2 to 4 and
    5 to 6, so only Omega_i = {1, 2, 5} need be considered.
    """
    return list(dict.fromkeys(permutations((l, w, h), 3)))


def build_instance(path, value_mode="volume"):
    """Box TYPES stay types here - the model is indexed by type and position.

    Containers are still expanded into individual containers, since Kurpel's
    (12)-(15) indexes them individually (their j = 1..C_k).
    """
    box_types, container_types = parse_instance(path)

    types = []
    for bt in box_types:
        l, w, h = bt["dims"]
        vol = l * w * h
        types.append({
            "type": bt["type"],
            "dims": bt["dims"],
            "count": bt["count"],
            "vol": vol,
            "value": bt["coef"] * vol if value_mode == "volume" else bt["coef"],
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
        "types": types,
        "containers": containers,
        "box_types": box_types,
        "container_types": container_types,
        "value_mode": value_mode,
        "n_boxes": sum(t["count"] for t in types),
    }


# =========================================================================
# Bounds
# =========================================================================
def continuous_knapsack_bound(inst):
    """Upper bound by relaxing to a 1-D continuous knapsack.

    Identical to the bound in 3DMHKP.py - all containers merged into one volume
    capacity, boxes taken in non-increasing value density, the last one
    fractionally. Reproduces Bortfeldt (2000) Table 2 exactly on all 16.
    """
    capacity = sum(c["vol"] for c in inst["containers"])
    order = sorted(inst["types"], key=lambda t: -(t["value"] / t["vol"]))

    remaining, bound = capacity, 0.0
    for t in order:
        if remaining <= 0:
            break
        take = min(t["vol"] * t["count"], remaining)
        bound += take * (t["value"] / t["vol"])
        remaining -= take
    return bound


# =========================================================================
# Discretization - Kurpel Appendix A.1 (normal patterns)
# =========================================================================
def normal_patterns(types, limit):
    """Reachable sums of box extents, bounded by availability - eq. (A.1).

    A coordinate is usable iff it is sum_i beta_i * d, where d is any dimension
    of type i (any dimension, since orientations permute the three) and at most
    b_i boxes of type i contribute. Returned sorted, capped at `limit`.
    """
    reach = {0}
    for t in types:
        dims = sorted(set(t["dims"]))
        avail = t["count"]
        # fewest boxes of THIS type needed to reach v; edges all have d > 0,
        # so one increasing sweep relaxes every path.
        need = [None] * (limit + 1)
        need[0] = 0
        for v in range(limit + 1):
            base = need[v]
            if base is None or base >= avail:
                continue
            for d in dims:
                if v + d <= limit and (need[v + d] is None or base + 1 < need[v + d]):
                    need[v + d] = base + 1
        singles = [v for v in range(limit + 1) if need[v] is not None]
        reach = {a + b for a in reach for b in singles if a + b <= limit}
    return sorted(reach)


def discretize(inst):
    """Per-container normal-pattern point sets.

    X/Y/Z are the placement points of eq. (1)-(3): capped at L_k - min extent,
    since no box can start beyond that. Xf/Yf are the wider column sets used by
    the "column" stability form, which must see every possible box EDGE, not
    only every possible box origin.
    """
    types = inst["types"]
    containers = inst["containers"]
    min_ext = min(min(t["dims"]) for t in types)
    limit = max(max(c["dims"]) for c in containers)
    reach = normal_patterns(types, limit)

    disc = []
    for c in containers:
        L, W, H = c["dims"]
        disc.append({
            "X": [v for v in reach if v <= L - min_ext],
            "Y": [v for v in reach if v <= W - min_ext],
            "Z": [v for v in reach if v <= H - min_ext],
            "Xf": [v for v in reach if v < L],
            "Yf": [v for v in reach if v < W],
        })
    return {"reach": reach, "min_ext": min_ext, "per_container": disc}


def placements(inst, disc, i, g, c):
    """Points at which type i in orientation g fits into container c.

    X_igk of eq. (4)-(6): the container's own points, further capped so the box
    ends inside it. Returns (px, py, pz), or None if the orientation does not
    fit the container at all.
    """
    l, w, h = inst["types"][i]["orientations"][g]
    L, W, H = inst["containers"][c]["dims"]
    if l > L or w > W or h > H:
        return None
    d = disc["per_container"][c]
    return ([p for p in d["X"] if p <= L - l],
            [q for q in d["Y"] if q <= W - w],
            [r for r in d["Z"] if r <= H - h])


def _span(points, lo, hi):
    """Index range of `points` lying in [lo, hi]."""
    return bisect_left(points, lo), bisect_right(points, hi)


def estimate_size(inst, disc, alpha, form):
    """Variable and nonzero counts, without building anything.

    Both counts factor across the three axes, so this is cheap even where the
    model itself would be far too large to construct.
    """
    n_vars = nnz_conflict = nnz_stab = 0
    for c, cont in enumerate(inst["containers"]):
        d = disc["per_container"][c]
        X, Y, Z, Xf, Yf = d["X"], d["Y"], d["Z"], d["Xf"], d["Yf"]
        zset = set(Z)

        here = {}
        for i, t in enumerate(inst["types"]):
            for g in range(len(t["orientations"])):
                pts = placements(inst, disc, i, g, c)
                if pts is not None and all(pts):
                    here[i, g] = pts

        for (i, g), (px, py, pz) in here.items():
            l, w, h = inst["types"][i]["orientations"][g]
            n_vars += len(px) * len(py) * len(pz)

            cov_x = sum(len(range(*_span(X, p, p + l - 1))) for p in px)
            cov_y = sum(len(range(*_span(Y, q, q + w - 1))) for q in py)
            cov_z = sum(len(range(*_span(Z, r, r + h - 1))) for r in pz)
            nnz_conflict += cov_x * cov_y * cov_z

            if alpha <= 0:
                continue
            if form == "column":
                fx = sum(len(range(*_span(Xf, p, p + l - 1))) for p in px)
                fy = sum(len(range(*_span(Yf, q, q + w - 1))) for q in py)
                n_base = sum(1 for r in pz if r > 0)
                n_top = sum(1 for r in pz if r + h in zset)
                nnz_stab += fx * fy * (n_base + n_top)
            else:
                # eq. (19): one row per non-floor variable, summing over every
                # flush supporter overlapping its base. The (p, q) counts do not
                # depend on r2, so they are counted once per supporter.
                for (i2, g2), (qx, qy, _) in here.items():
                    l2, w2, h2 = inst["types"][i2]["orientations"][g2]
                    # the (p, q) overlap counts factor across the two axes and
                    # do not depend on the level, so count them once per
                    # supporter and multiply by the levels it can sit at
                    a = sum(len(range(*_span(qx, p - l2 + 1, p + l - 1))) for p in px)
                    b = sum(len(range(*_span(qy, q - w2 + 1, q + w - 1))) for q in py)
                    hits = sum(1 for r in pz
                               if r > 0 and 0 <= r - h2 <= cont["dims"][2] - h2
                               and r - h2 in zset)
                    nnz_stab += a * b * hits
    return n_vars, nnz_conflict, nnz_stab


# =========================================================================
# Geometry helpers
# =========================================================================
def _overlaps(a_pos, a_dim, b_pos, b_dim):
    for axis in range(3):
        if a_pos[axis] + a_dim[axis] <= b_pos[axis] + 1e-9:
            return False
        if b_pos[axis] + b_dim[axis] <= a_pos[axis] + 1e-9:
            return False
    return True


def _contact(pos_a, dim_a, pos_b, dim_b):
    """Base-area overlap of a sitting on b, or 0 if b's top is not flush."""
    if pos_b[2] + dim_b[2] != pos_a[2]:
        return 0
    ox = min(pos_a[0] + dim_a[0], pos_b[0] + dim_b[0]) - max(pos_a[0], pos_b[0])
    oy = min(pos_a[1] + dim_a[1], pos_b[1] + dim_b[1]) - max(pos_a[1], pos_b[1])
    return max(ox, 0) * max(oy, 0)


def support_fraction(pos, dims, placed):
    """Fraction of a box's base carried by the floor or by flush boxes below.

    `placed` is a list of (pos, dims) already in the same container. Exact -
    positions are known here, so this is the real area, not a surrogate.
    """
    if pos[2] == 0:
        return 1.0
    area = dims[0] * dims[1]
    got = sum(_contact(pos, dims, p, d) for p, d in placed)
    return got / area if area else 1.0


# =========================================================================
# Greedy extreme-point heuristic (stability-aware warm start)
# =========================================================================
def greedy_pack(inst, disc, alpha, time_budget=30.0):
    """Value-density-first greedy over normal-pattern points.

    Same shape as the heuristic in 3DMHKP.py, with two changes: candidate
    points are filtered to the discretization (so the packing maps onto the
    model's variables), and a placement is rejected unless it is alpha-stable.
    Returns a list of {type, orient, container, pos}.
    """
    start = time.time()
    remaining = [t["count"] for t in inst["types"]]
    order = sorted(range(len(inst["types"])),
                   key=lambda i: -(inst["types"][i]["value"] / inst["types"][i]["vol"]))
    placement = []

    container_order = sorted(range(len(inst["containers"])),
                             key=lambda c: -inst["containers"][c]["vol"])

    for c in container_order:
        if time.time() - start > time_budget:
            break
        L, W, H = inst["containers"][c]["dims"]
        d = disc["per_container"][c]
        xs, ys, zs = set(d["X"]), set(d["Y"]), set(d["Z"])
        points = [(0, 0, 0)]
        placed = []

        progress = True
        while progress:
            progress = False
            if time.time() - start > time_budget:
                break
            for i in order:
                if remaining[i] == 0:
                    continue
                done = False
                # Low, then near the origin - keeps the packing compact and
                # keeps new boxes resting on what is already there.
                for point in sorted(points, key=lambda p: (p[2], p[1], p[0])):
                    if point[0] not in xs or point[1] not in ys or point[2] not in zs:
                        continue
                    for g, dims in enumerate(inst["types"][i]["orientations"]):
                        if point[0] + dims[0] > L or point[1] + dims[1] > W \
                                or point[2] + dims[2] > H:
                            continue
                        if any(_overlaps(point, dims, pos, dim) for pos, dim in placed):
                            continue
                        if alpha > 0 and \
                                support_fraction(point, dims, placed) < alpha - TOL:
                            continue

                        placement.append({"type": i, "orient": g,
                                          "container": c, "pos": point})
                        placed.append((point, dims))
                        remaining[i] -= 1
                        if point in points:
                            points.remove(point)
                        points.extend([
                            (point[0] + dims[0], point[1], point[2]),
                            (point[0], point[1] + dims[1], point[2]),
                            (point[0], point[1], point[2] + dims[2]),
                        ])
                        points = [p for p in set(points)
                                  if p[0] < L and p[1] < W and p[2] < H]
                        done = progress = True
                        break
                    if done:
                        break

    return placement


def placement_value(inst, placement):
    return sum(inst["types"][e["type"]]["value"] for e in placement)


def placement_volume(inst, placement):
    return sum(inst["types"][e["type"]]["vol"] for e in placement)


def verify_placement(inst, placement, alpha, tol=1e-6):
    """Independently re-check a packing. Returns (errors, min support fraction).

    Checks containment, availability, pairwise non-overlap and - exactly, from
    the geometry rather than from the model - alpha-support of every box.
    """
    errors = []
    used = [0] * len(inst["types"])
    by_container = {}
    for e in placement:
        used[e["type"]] += 1
        by_container.setdefault(e["container"], []).append(e)

    for i, t in enumerate(inst["types"]):
        if used[i] > t["count"]:
            errors.append(f"type {t['type']}: {used[i]} placed, only "
                          f"{t['count']} available")

    worst = 1.0
    for c, entries in by_container.items():
        L, W, H = inst["containers"][c]["dims"]
        boxes = []
        for e in entries:
            dims = inst["types"][e["type"]]["orientations"][e["orient"]]
            pos = e["pos"]
            if pos[0] + dims[0] > L + tol or pos[1] + dims[1] > W + tol \
                    or pos[2] + dims[2] > H + tol or min(pos) < -tol:
                errors.append(f"a box sticks out of container {c}")
            boxes.append((pos, dims))

        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                if _overlaps(boxes[a][0], boxes[a][1], boxes[b][0], boxes[b][1]):
                    errors.append(f"two boxes overlap in container {c}")

        for a, (pos, dims) in enumerate(boxes):
            others = [boxes[b] for b in range(len(boxes)) if b != a]
            frac = support_fraction(pos, dims, others)
            worst = min(worst, frac)
            if alpha > 0 and frac < alpha - tol:
                errors.append(f"a box in container {c} is only {frac:.1%} "
                              f"supported, alpha = {alpha:.2f}")

    return errors, worst


# =========================================================================
# MILP - Kurpel et al. (2020), (12)-(15) + (19)
# =========================================================================
def build_model(inst, disc, alpha, form, warm_start=None, verbose=False):
    types = inst["types"]
    containers = inst["containers"]

    model = gp.Model(f"3DMHKPst_{inst['name']}")
    model.Params.OutputFlag = 1 if verbose else 0

    # ---- variables (15): one per (type, orientation, container, point) ----
    x = {}
    grid = {}          # (i, g, c) -> (px, py, pz)
    for c in range(len(containers)):
        for i, t in enumerate(types):
            for g in range(len(t["orientations"])):
                pts = placements(inst, disc, i, g, c)
                if pts is None:
                    continue
                px, py, pz = pts
                if not (px and py and pz):
                    continue
                grid[i, g, c] = pts
                for p in px:
                    for q in py:
                        for r in pz:
                            x[i, g, c, p, q, r] = model.addVar(
                                vtype=GRB.BINARY, name=f"x[{i},{g},{c},{p},{q},{r}]")

    # ---- objective (12): maximize stowed value ----
    model.setObjective(
        gp.quicksum(types[k[0]]["value"] * v for k, v in x.items()), GRB.MAXIMIZE)

    # ---- (13) at most one box over each discretization point ----
    for c in range(len(containers)):
        d = disc["per_container"][c]
        X, Y, Z = d["X"], d["Y"], d["Z"]
        nx, ny, nz = len(X), len(Y), len(Z)
        rows = [[] for _ in range(nx * ny * nz)]
        for (i, g, cc), (px, py, pz) in grid.items():
            if cc != c:
                continue
            l, w, h = types[i]["orientations"][g]
            sx = {p: range(*_span(X, p, p + l - 1)) for p in px}
            sy = {q: range(*_span(Y, q, q + w - 1)) for q in py}
            sz = {r: range(*_span(Z, r, r + h - 1)) for r in pz}
            for p in px:
                for q in py:
                    for r in pz:
                        var = x[i, g, c, p, q, r]
                        for si in sx[p]:
                            off_s = si * ny
                            for ti in sy[q]:
                                off_t = (off_s + ti) * nz
                                for ui in sz[r]:
                                    rows[off_t + ui].append(var)
        for terms in rows:
            if len(terms) > 1:
                model.addConstr(gp.quicksum(terms) <= 1)

    # ---- (14) availability ----
    for i, t in enumerate(types):
        terms = [v for k, v in x.items() if k[0] == i]
        if terms:
            model.addConstr(gp.quicksum(terms) <= t["count"], name=f"avail[{i}]")

    # ---- stability ----
    if alpha > 0:
        if form == "column":
            _add_stability_column(model, inst, disc, x, grid)
        else:
            _add_stability_area(model, inst, disc, x, grid, alpha)

    _add_valid_inequalities(model, inst, x)
    _add_symmetry_breaking(model, inst, x)

    model._vars = {"x": x, "grid": grid}
    if warm_start is not None:
        _apply_warm_start(model, x, warm_start)

    model.update()
    return model


def _add_stability_column(model, inst, disc, x, grid):
    """Full base support (alpha = 1), exactly - see the header.

    For every container, column (s,t) and level r' > 0: the box starting there
    is matched by a box ending there.
    """
    types = inst["types"]
    for c in range(len(inst["containers"])):
        d = disc["per_container"][c]
        Xf, Yf, Z = d["Xf"], d["Yf"], d["Z"]
        zidx = {r: n for n, r in enumerate(Z)}
        nx, ny, nz = len(Xf), len(Yf), len(Z)
        starts = [[] for _ in range(nx * ny * nz)]
        ends = [[] for _ in range(nx * ny * nz)]

        for (i, g, cc), (px, py, pz) in grid.items():
            if cc != c:
                continue
            l, w, h = types[i]["orientations"][g]
            sx = {p: range(*_span(Xf, p, p + l - 1)) for p in px}
            sy = {q: range(*_span(Yf, q, q + w - 1)) for q in py}
            for p in px:
                for q in py:
                    for r in pz:
                        var = x[i, g, c, p, q, r]
                        # a box based at r > 0 needs support at level r
                        base = zidx[r] if r > 0 else None
                        # a box topping out at r + h offers support there
                        top = zidx.get(r + h)
                        if base is None and top is None:
                            continue
                        for si in sx[p]:
                            off_s = si * ny
                            for ti in sy[q]:
                                off_t = (off_s + ti) * nz
                                if base is not None:
                                    starts[off_t + base].append(var)
                                if top is not None:
                                    ends[off_t + top].append(var)

        for n, above in enumerate(starts):
            if above:
                model.addConstr(gp.quicksum(above) <= gp.quicksum(ends[n]))


def _add_stability_area(model, inst, disc, x, grid, alpha):
    """Kurpel et al. (2020), eq. (19)-(21), transcribed.

    One row per non-floor variable: the contact areas of the flush supporters
    overlapping its base must reach alpha times that base.
    """
    types = inst["types"]
    for c, cont in enumerate(inst["containers"]):
        zset = set(disc["per_container"][c]["Z"])
        here = [(i, g, pts) for (i, g, cc), pts in grid.items() if cc == c]
        for i, g, (px, py, pz) in here:
            l, w, _ = types[i]["orientations"][g]
            base_area = l * w
            for r2 in pz:
                if r2 == 0:
                    continue                    # the container floor supports it
                # supporters: every orientation whose top lands exactly on r2
                cands = []
                for i2, g2, (qx, qy, qz) in here:
                    l2, w2, h2 = types[i2]["orientations"][g2]
                    rs = r2 - h2
                    if rs < 0 or rs not in zset or rs > cont["dims"][2] - h2:
                        continue
                    cands.append((i2, g2, l2, w2, rs, qx, qy))
                for p2 in px:
                    for q2 in py:
                        terms = []
                        for i2, g2, l2, w2, rs, qx, qy in cands:
                            lo, hi = _span(qx, p2 - l2 + 1, p2 + l - 1)
                            lo2, hi2 = _span(qy, q2 - w2 + 1, q2 + w - 1)
                            for p in qx[lo:hi]:
                                # (20) x-overlap of supporter with the base
                                ox = min(p + l2, p2 + l) - max(p, p2)
                                for q in qy[lo2:hi2]:
                                    # (21) y-overlap
                                    oy = min(q + w2, q2 + w) - max(q, q2)
                                    terms.append(ox * oy * x[i2, g2, c, p, q, rs])
                        var = x[i, g, c, p2, q2, r2]
                        if terms:
                            model.addConstr(gp.quicksum(terms)
                                            >= alpha * base_area * var)
                        else:
                            model.addConstr(var == 0)


def _add_valid_inequalities(model, inst, x):
    """Volume capacity, redundant but strong - as in 3DMHKP.py (6) and (7).

    The global one is what pulls the LP bound down towards the
    continuous-knapsack bound.
    """
    types = inst["types"]
    per_container = {}
    for k, v in x.items():
        per_container.setdefault(k[2], []).append(types[k[0]]["vol"] * v)
    for c, terms in per_container.items():
        model.addConstr(gp.quicksum(terms) <= inst["containers"][c]["vol"],
                        name=f"vol[{c}]")
    if x:
        model.addConstr(
            gp.quicksum(types[k[0]]["vol"] * v for k, v in x.items())
            <= sum(c["vol"] for c in inst["containers"]),
            name="volglobal")


def _add_symmetry_breaking(model, inst, x):
    """Fill the lower-indexed of two identical containers at least as full.

    Identical BOXES need no such treatment here: they are not distinguishable
    in this formulation to begin with, which is one of the reasons the
    discretized model behaves so much better than the continuous one.
    """
    types = inst["types"]
    containers = inst["containers"]
    load = {}
    for k, v in x.items():
        load.setdefault(k[2], []).append(types[k[0]]["vol"] * v)
    for c in range(len(containers) - 1):
        if containers[c]["type"] != containers[c + 1]["type"]:
            continue
        if c in load and (c + 1) in load:
            model.addConstr(gp.quicksum(load[c]) >= gp.quicksum(load[c + 1]),
                            name=f"symcont[{c}]")


def _apply_warm_start(model, x, placement):
    """Feed the greedy packing to Gurobi as a complete MIP start."""
    chosen = set()
    for e in placement:
        key = (e["type"], e["orient"], e["container"],
               e["pos"][0], e["pos"][1], e["pos"][2])
        if key in x:
            chosen.add(key)
    keys = list(x)
    model.setAttr("Start", [x[k] for k in keys],
                  [1.0 if k in chosen else 0.0 for k in keys])


def extract_placement(model, inst):
    """Read the incumbent back into a placement list."""
    if model.SolCount == 0:
        return []
    out = []
    for k, v in model._vars["x"].items():
        if v.X > 0.5:
            i, g, c, p, q, r = k
            out.append({"type": i, "orient": g, "container": c, "pos": (p, q, r)})
    return out


# =========================================================================
# Reporting
# =========================================================================
def summarize(inst, placement):
    """Per-container statistics for a placement."""
    used = {}
    for e in placement:
        used.setdefault(e["container"], []).append(e)

    stats = []
    for c, container in enumerate(inst["containers"]):
        es = used.get(c, [])
        packed_vol = sum(inst["types"][e["type"]]["vol"] for e in es)
        stats.append({
            "container": c,
            "type": container["type"],
            "dims": container["dims"],
            "n_boxes": len(es),
            "volume_used": packed_vol,
            "volume": container["vol"],
            "utilization": packed_vol / container["vol"] if container["vol"] else 0.0,
            "value": sum(inst["types"][e["type"]]["value"] for e in es),
        })
    return stats


def write_report(path, inst, placement, result):
    """Same headline lines as 3DMHKP.py, so the compare_*.py readers work."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = summarize(inst, placement)
    total_container_vol = sum(c["vol"] for c in inst["containers"])
    packed_vol = placement_volume(inst, placement)

    with open(path, "w") as f:
        f.write(f"Instance: {inst['name']}\n")
        if result["alpha"] > 0:
            f.write(f"Load stability: alpha = {result['alpha']:.2f} "
                    f"({'full' if result['alpha'] >= 1 else 'partial'} base "
                    f"support), form '{result['form']}'\n")
        else:
            f.write("Load stability: NOT imposed (alpha = 0)\n")
        f.write(f"Value model: v_i = c_i * volume_i ({inst['value_mode']})\n")
        f.write(f"Boxes: {inst['n_boxes']} ({len(inst['box_types'])} types)\n")
        f.write(f"Containers: {len(inst['containers'])} "
                f"({len(inst['container_types'])} types)\n\n")

        f.write(f"Status: {result['status']}\n")
        f.write(f"Runtime: {result['runtime']:.2f} s\n")
        f.write(f"Packed value (incumbent / lower bound): {result['value']:.1f}\n")
        f.write(f"Dual bound (MILP): {result['dual_bound']:.1f}\n")
        f.write(f"Continuous-knapsack bound: {result['ck_bound']:.1f}\n")
        f.write(f"Optimality gap: {result['gap']:.2%}\n")
        f.write(f"Greedy warm-start value: {result['greedy_value']:.1f}\n")
        f.write(f"Minimum base support achieved: {result['min_support']:.1%}\n\n")

        f.write(f"Boxes packed: {len(placement)} / {inst['n_boxes']}\n")
        f.write(f"Volume utilization: {packed_vol} / {total_container_vol} "
                f"= {packed_vol / total_container_vol:.1%}\n\n")

        for st in stats:
            f.write(f"Container {st['container']} (type {st['type']}, "
                    f"{st['dims'][0]}x{st['dims'][1]}x{st['dims'][2]}): "
                    f"{st['n_boxes']} boxes, value {st['value']:.1f}, "
                    f"utilization {st['utilization']:.1%}\n")

        f.write("\nPlacements (box, type, container, position, oriented dims):\n")
        ordered = sorted(placement, key=lambda e: (e["container"], e["pos"][2],
                                                   e["pos"][1], e["pos"][0]))
        for n, e in enumerate(ordered):
            dims = inst["types"][e["type"]]["orientations"][e["orient"]]
            f.write(f"  box {n:>4} type {inst['types'][e['type']]['type']} "
                    f"-> container {e['container']} pos {e['pos']} dims {dims}\n")


# =========================================================================
# Self-test against Kurpel Table 2
# =========================================================================
def self_test(instance_dir):
    """Check normal_patterns() against the variable counts Kurpel published."""
    counts = {}
    for n in range(1, 17):
        inst = build_instance(instance_dir / f"instance{n:02d}.txt")
        disc = discretize(inst)
        counts[n], _, _ = estimate_size(inst, disc, alpha=0.0, form="column")

    ok = True
    lo, hi = min(counts.values()), max(counts.values())
    for label, got, want in (("min", lo, KURPEL_TABLE2_NP["min"]),
                             ("max", hi, KURPEL_TABLE2_NP["max"])):
        mark = "ok" if got == want else "MISMATCH"
        ok &= got == want
        print(f"  Table 2 MHLOPP/NP {label}: ours {got:>8,}  paper {want:>8,}  {mark}")

    # Their sum matches only with the 10-container instance 4 (see the header).
    inst4 = build_instance(instance_dir / "instance04.txt")
    keep = {t["type"] for t in inst4["container_types"][:2]}
    inst4["containers"] = [c for c in inst4["containers"] if c["type"] in keep]
    inst4["container_types"] = inst4["container_types"][:2]
    trunc, _, _ = estimate_size(inst4, discretize(inst4), 0.0, "column")

    full_sum = sum(counts.values())
    adj_sum = full_sum - counts[4] + trunc
    mark = "ok" if adj_sum == KURPEL_TABLE2_NP["sum"] else "MISMATCH"
    ok &= adj_sum == KURPEL_TABLE2_NP["sum"]
    print(f"  Table 2 MHLOPP/NP sum: ours {adj_sum:>8,}  paper "
          f"{KURPEL_TABLE2_NP['sum']:>8,}  {mark}")
    print(f"    (instance 4 counted with {len(inst4['containers'])} containers / "
          f"2 types = {trunc:,} vars, not the {counts[4]:,} of the 15-container "
          f"file; the published sum is only reproduced that way, which\n"
          f"     re-confirms independently that Kurpel solved a truncated "
          f"instance 4 - cf. the header of compare_bortfeldt.py)")
    return 0 if ok else 1


# =========================================================================
# Driver
# =========================================================================
def solve_instance(path, args):
    inst = build_instance(path, value_mode=args.value_mode)
    disc = discretize(inst)
    form = args.stability_form
    if form == "auto":
        form = "column" if args.alpha >= 1.0 else "area"
    if form == "column" and 0 < args.alpha < 1.0:
        print("  note: the column form is exact only at alpha = 1; "
              "using eq. (19) instead")
        form = "area"

    ck_bound = continuous_knapsack_bound(inst)
    n_vars, nnz_conf, nnz_stab = estimate_size(inst, disc, args.alpha, form)

    print(f"\n{'=' * 70}")
    print(f"{inst['name']}: {inst['n_boxes']} boxes "
          f"({len(inst['box_types'])} types), {len(inst['containers'])} "
          f"containers ({len(inst['container_types'])} types)")
    print(f"  continuous-knapsack bound: {ck_bound:.1f}")
    print(f"  discretization: {n_vars:,} variables, "
          f"{nnz_conf:,} nonzeros in (13)"
          + (f", {nnz_stab:,} in stability" if args.alpha > 0 else ""))

    t0 = time.time()
    placement = greedy_pack(inst, disc, args.alpha, time_budget=args.greedy_time)
    errs, min_support = verify_placement(inst, placement, args.alpha)
    if errs:
        print(f"  WARNING: greedy produced an invalid packing ({errs[0]}); discarding")
        placement, min_support = [], 1.0
    greedy_value = placement_value(inst, placement)
    print(f"  greedy warm start: value {greedy_value:.1f} "
          f"({len(placement)}/{inst['n_boxes']} boxes, {time.time() - t0:.1f}s)")

    result = {
        "instance": inst["name"],
        "n_boxes": inst["n_boxes"],
        "ck_bound": ck_bound,
        "greedy_value": greedy_value,
        "value": greedy_value,
        "dual_bound": ck_bound,
        "gap": float("inf"),
        "status": "greedy-only",
        "runtime": time.time() - t0,
        "alpha": args.alpha,
        "form": form,
        "min_support": min_support,
    }

    total_nnz = nnz_conf + nnz_stab
    if args.greedy_only:
        result["gap"] = ((ck_bound - greedy_value) / ck_bound) if ck_bound else 0.0
        _finish(inst, placement, result, args)
        return result
    if total_nnz > args.max_nnz:
        print(f"  SKIPPED: {total_nnz:,} nonzeros exceeds --max-nnz "
              f"({args.max_nnz:,}); keeping the greedy packing")
        result["status"] = "skipped (too large)"
        result["gap"] = ((ck_bound - greedy_value) / ck_bound) if ck_bound else 0.0
        _finish(inst, placement, result, args)
        return result

    print("  building MILP ...", flush=True)
    t_build = time.time()
    model = build_model(inst, disc, args.alpha, form,
                        warm_start=placement if placement else None,
                        verbose=args.verbose)
    print(f"  model: {model.NumVars:,} vars ({model.NumBinVars:,} binary), "
          f"{model.NumConstrs:,} constraints, {model.NumNZs:,} nonzeros "
          f"[{time.time() - t_build:.1f}s]")

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
    model.Params.MIPFocus = 1
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
        milp_placement = extract_placement(model, inst)
        errs, support = verify_placement(inst, milp_placement, args.alpha)
        if errs:
            print(f"  WARNING: MILP solution failed verification: {errs[0]}")
        milp_value = placement_value(inst, milp_placement)
        if not errs and milp_value >= greedy_value:
            placement = milp_placement
            result["value"] = milp_value
            result["min_support"] = support
        else:
            result["value"] = greedy_value
    result["dual_bound"] = min(model.ObjBound, ck_bound)

    bound = result["dual_bound"]
    result["gap"] = ((bound - result["value"]) / bound) if bound > 0 else 0.0
    result["runtime"] = time.time() - t0

    print(f"  MILP: value {result['value']:.1f}, bound {bound:.1f}, "
          f"gap {result['gap']:.2%} [{result['status']}]")

    _finish(inst, placement, result, args)
    return result


def _finish(inst, placement, result, args):
    result["n_packed"] = len(placement)
    packed_vol = placement_volume(inst, placement)
    total_vol = sum(c["vol"] for c in inst["containers"])
    result["utilization"] = packed_vol / total_vol if total_vol else 0.0

    if not args.no_report:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"-{args.tag}" if args.tag else ""
        out = RESULTS_DIR / f"{inst['name']}-3DMHKPst{suffix}.txt"
        write_report(out, inst, placement, result)
        print(f"  report -> {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exact 0-1 model for the 3D multiple heterogeneous knapsack "
                    "problem of Mohanty, Mathur & Ivancic (1994) with the load "
                    "stability constraint of Kurpel et al. (2020).")
    parser.add_argument("--instances", nargs="*", type=int, default=None,
                        metavar="N",
                        help="instance numbers to solve (default: all 16)")
    parser.add_argument("--instance-dir", type=Path, default=INSTANCE_DIR,
                        help=f"instance directory (default: {INSTANCE_DIR})")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="stability coefficient: the minimum fraction of a "
                             "box base that must be supported. 1 = full support "
                             "(Kurpel's choice, the default); 0 = no stability")
    parser.add_argument("--stability-form", choices=("auto", "column", "area"),
                        default="auto",
                        help="auto (default) uses the exact column form at "
                             "alpha=1 and eq. (19) otherwise; area forces "
                             "eq. (19)")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                        help=f"MILP seconds per instance (default: {TIME_LIMIT})")
    parser.add_argument("--mip-gap", type=float, default=MIP_GAP,
                        help="relative MIP gap to stop at")
    parser.add_argument("--threads", type=int, default=THREADS,
                        help="Gurobi threads (0 = all cores)")
    parser.add_argument("--greedy-time", type=float, default=30.0,
                        help="seconds for the greedy warm start")
    parser.add_argument("--max-nnz", type=int, default=MAX_NNZ,
                        help=f"skip instances whose model would exceed this "
                             f"many nonzeros (default: {MAX_NNZ:,})")
    parser.add_argument("--value-mode", choices=("volume", "flat"),
                        default="volume",
                        help="volume: v_i = c_i * vol_i (default, matches the "
                             "literature); flat: v_i = c_i")
    parser.add_argument("--tag", default="",
                        help="suffix for the report filenames, e.g. --tag nostab")
    parser.add_argument("--bound-only", action="store_true",
                        help="only compute LP / knapsack bounds")
    parser.add_argument("--greedy-only", action="store_true",
                        help="only run the greedy heuristic, no MILP")
    parser.add_argument("--self-test", action="store_true",
                        help="check the discretization against Kurpel Table 2")
    parser.add_argument("--no-report", action="store_true",
                        help="do not write per-instance report files")
    parser.add_argument("--verbose", action="store_true",
                        help="show the Gurobi log")
    args = parser.parse_args(argv)

    if not 0.0 <= args.alpha <= 1.0:
        print("--alpha must lie in [0, 1]", file=sys.stderr)
        return 1

    if args.self_test:
        print("Checking the normal-pattern discretization against Kurpel et al. "
              "(2020), Table 2 (MHLOPP, NP):")
        return self_test(args.instance_dir)

    numbers = args.instances if args.instances else list(range(1, 17))
    paths = []
    for n in numbers:
        p = args.instance_dir / f"instance{n:02d}.txt"
        if not p.exists():
            print(f"missing instance file: {p}", file=sys.stderr)
            return 1
        paths.append((n, p))

    results = []
    for n, p in paths:
        try:
            r = solve_instance(p, args)
            r["number"] = n
            results.append(r)
        except gp.GurobiError as exc:
            print(f"  Gurobi error on {p.stem}: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            break

    if not results:
        return 1

    width = 102
    print(f"\n{'=' * width}")
    print(f"SUMMARY  (alpha = {args.alpha:.2f})")
    print(f"{'=' * width}")
    print(f"{'instance':<12}{'packed':>8}{'value':>14}{'bound':>14}{'gap':>9}"
          f"{'util':>8}{'supp':>8}{'time':>8}{'KUR-st':>15}")
    print("-" * width)
    for r in results:
        ref = KURPEL_STABILITY.get(r.get("number"))
        if ref is None or r.get("number") in KURPEL_INCOMPARABLE:
            ref_s = "n/c" if r.get("number") in KURPEL_INCOMPARABLE else "-"
        else:
            ref_s = f"{ref[0]:.1f}"
        print(f"{r['instance']:<12}{r.get('n_packed', 0):>8}{r['value']:>14.1f}"
              f"{r['dual_bound']:>14.1f}{r['gap']:>8.1%}"
              f"{r.get('utilization', 0):>8.1%}{r['min_support']:>8.0%}"
              f"{r['runtime']:>7.1f}s{ref_s:>15}")
    print("-" * width)

    solved = sum(1 for r in results if r["status"] == "optimal")
    print(f"proven optimal: {solved}/{len(results)}")
    mean_util = sum(r.get("utilization", 0) for r in results) / len(results)
    print(f"mean volume utilization: {mean_util:.1%}")
    if args.alpha > 0:
        worst = min(r["min_support"] for r in results)
        print(f"minimum base support over all packed boxes: {worst:.1%} "
              f"(required {args.alpha:.0%})")
    print("KUR-st is Kurpel et al. (2020) Table E1, 'With load stability', "
          "3600 s per instance on a 2.77 GHz Xeon with Gurobi 8.1.0;")
    print("  n/c marks instance 4, which they solved in a 10-container "
          "truncation (see --self-test).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
