# Simulated Annealing for the 3DMHKP — sequence-decoder variant
# =========================================================================
# Heuristic counterpart to the exact MILP in 3DMHKP.py for the packing
# problem of Mohanty, Mathur & Ivancic (1994), "Value considerations in
# three-dimensional packing - a heuristic procedure using the fractional
# knapsack problem", EJOR 74:143-151:
#
#   Fewer containers are given than are needed to hold all boxes, so a
#   subset of boxes must be chosen and placed such that the total value
#   of the stowed cargo is maximal. Boxes may be freely rotated (up to 6
#   axis-aligned orientations), box values are v_i = c_i * l_i*w_i*h_i.
#
# The SA engine is the one from 3dp-cptp-SA.py, transplanted:
#
#     1. pick ONE move operator at random (weighted by MOVE_WEIGHTS)
#     2. sample ONE random neighbour from that operator
#     3. accept or reject that single neighbour by the Metropolis criterion
#     4. cool down once per iteration, regardless of the outcome
#     5. reheat + diversify after prolonged stagnation
#
#   - Improving moves are always accepted
#   - Worsening moves are accepted with probability exp(delta / T)
#   - Temperature decreases geometrically: T *= ALPHA each iteration
#
# -------------------------------------------------------------------------
# WHAT CHANGES RELATIVE TO 3dp-cptp-SA.py
# -------------------------------------------------------------------------
# In the 3DP-CPTP the packing was a CONSTRAINT: SA searched over routes and a
# Gurobi sub-model answered "does this load fit?" for the routes a move
# touched. In the 3DMHKP — there are no routes, and
# calling Gurobi once per iteration would cost more than the MILP in 3DMHKP.py
# is worth. So the Gurobi sub-model is replaced by a deterministic constructive
# DECODER, and SA searches over the decoder's input:
#
#     solution = (priority order of the boxes,
#                 orientation preference per box,
#                 order in which the containers are filled)
#
#     decode(solution) -> a genuine, verified feasible packing + its value
#
# Every solution therefore maps to a feasible packing by construction: there
# are no infeasible neighbours to reject, and this file needs no Gurobi at all.
# The price is that the objective is no longer an analytic delta — the decoder
# has to run before the Metropolis test can be applied. Hence steps 2 and 3
# above are inverted with respect to 3dp-cptp-SA.py: there the cheap analytic
# delta was tested first and only an accepted move paid for a packing check;
# here every drawn neighbour pays for exactly one decode.
#
# The decoder is a deepest-bottom-left extreme-point packer: containers are
# filled one at a time in the container order, and within a container the
# unpacked boxes are offered in the priority order, each going to the first
# (lowest, then front, then left) extreme point and orientation where it fits.
# That is the same construction as greedy_pack() in 3DMHKP.py, but driven by
# the order SA controls rather than by a fixed value-density sort — and the
# initial solution IS that value-density sort, so SA starts exactly from the
# MILP's greedy warm start and can only improve on it.
#
# -------------------------------------------------------------------------
# THE MOVES (7 operators, same sampling scheme as the routing move set)
# -------------------------------------------------------------------------
#   relocate  — move one box to another position in the priority order
#   swap      — exchange the priority-order positions of two boxes
#   2opt      — reverse a segment of the priority order
#   promote   — pull a currently UNPACKED box forward (the "insert" analogue)
#   demote    — push a currently PACKED box back (the "remove" analogue)
#   orient    — change the orientation preference of one box
#   container — exchange two containers in the filling order
#
# The instances hold 47-200 boxes but only 2-6 box TYPES, so the priority order
# is highly degenerate: permuting two boxes of the same type with the same
# orientation preference decodes to an identical packing. Every sampler
# therefore rejects such no-op draws (see _key and _is_noop_span) — without
# that filter the majority of iterations would re-decode the same solution.
#
# -------------------------------------------------------------------------
# USAGE
# -------------------------------------------------------------------------
#   python 3DMHKP-SA.py                            # all 16, 300 s each
#   python 3DMHKP-SA.py --instances 7              # a single instance
#   python 3DMHKP-SA.py --instances 1 7 9 --time-limit 60
#   python 3DMHKP-SA.py --alpha 0.9995 --seed 1    # SA parameter variants
#   python 3DMHKP-SA.py --greedy-only              # decode the initial order only

import argparse
import math
import random
import shlex
import statistics
import sys
import time
from bisect import insort
from datetime import datetime
from math import gcd
from itertools import permutations
from pathlib import Path

# -------------------------
# Parameters
# -------------------------
INSTANCE_DIR = Path(__file__).parent / "Mohanty"
RESULTS_DIR = Path(__file__).parent / "results_3DMHKP"

T_INIT = None              # initial temperature; None = T_INIT_FRACTION * f(S_0)
T_INIT_FRACTION = 0.05     # T_INIT as a fraction of the initial objective value
ALPHA = 0.999              # geometric cooling factor (T *= ALPHA each iteration)
T_MIN = 1e-3               # minimum temperature
MAX_ITERATIONS = 500000000  # maximum SA iterations
TIME_LIMIT = 1800.0         # total wall-clock seconds per instance
EP_LIMIT = 0               # max extreme points scanned per box, 0 = unlimited

# Why T_INIT is derived from the objective rather than fixed at 100 as in
# 3dp-cptp-SA.py: there the objective was revenue minus travel cost, of order
# 1e2. Here the objective is stowed VALUE, which spans more than two orders of
# magnitude across the 16 instances (greedy 7.7e3 to 2.1e6) because it scales
# with the box volumes, and the deltas scale with it. A fixed T would be
# scalding on one instance and frozen on the next: at T=100, instance01 (mean
# box value 370) accepts a worsening move with probability 2.5%, instance04
# (mean box value 29358) with probability 1e-128.
#
# An earlier version estimated T_INIT by sampling 40 random neighbours of the
# initial solution and inverting the Metropolis criterion on the mean
# worsening delta. That estimate never left the band 1.7%-10.6% of the initial
# objective across the instances where it found any worsening draw at all (on
# two of sixteen it found none and fell back to a flat fraction anyway), so it
# is replaced by that fraction directly. Being off by a factor of three is
# harmless: geometric cooling absorbs it in ln(3)/ln(ALPHA) ~ 1100 iterations
# of a ~11.5k-iteration sweep, against the 1e5 iterations a run performs.
# Pass --t-init to override with a fixed value.

REHEAT_FRACTION = 0.25     # T is reset to REHEAT_FRACTION * T_0 on a reheat.
                           # Same rationale as in 3dp-cptp-SA.py: after a long
                           # stagnation T has annealed close to T_MIN, so a reheat
                           # has to RESET the temperature, not scale it.

# Iterations of one full geometric sweep T_INIT -> T_MIN: ~11.5k at ALPHA=0.999,
# ~115k at ALPHA=0.9999. One iteration here is one decode, far more expensive
# than one sampled route move in 3dp-cptp-SA.py, so ALPHA is correspondingly
# less aggressive — a 300 s run does on the order of 1e4 iterations, i.e. about
# one sweep. REHEAT_THRESHOLD is derived from the sweep length for the reason
# documented in 3dp-cptp-SA.py: an absolute threshold far below the sweep resets
# T before it can ever cool, which pins acceptance near 100% and degrades SA
# into a random walk.
REHEAT_THRESHOLD = None    # None = max(200, 0.5 * sweep length at ALPHA)

# ---- Time-based schedule (SCHEDULE = "time") -----------------------------
# The iteration-based geometric schedule above ties the annealing curve to
# iteration COUNT, but decode cost varies ~8x across the ep3 instances (6.4M
# iterations on ep3-20-C-C-50 against 824k on ep3-40-C-R-90 in the same 300 s).
# The same ALPHA therefore anneals one instance fully and another barely at all.
# Under SCHEDULE="time" temperature follows wall-clock progress instead,
#   T(p) = T0 * (T_MIN/T0) ** p,  p = elapsed / time_limit,
# so every instance traverses the identical T0 -> T_MIN curve regardless of how
# many decodes it fits into the budget, and ALPHA stops mattering.
SCHEDULE = "time"          # "time" (wall-clock) or "iter" (legacy geometric)

# T0 calibrated from the move-delta distribution rather than from f(S_0).
# T_INIT_FRACTION * f(S_0) was mis-scaled: the objective is a SUM over packed
# boxes while a single move perturbs only one or two of them, so 5% of the total
# dwarfed the typical |delta| and exp(delta/T) sat near 1. Measured acceptance
# was 60-77% for the whole run (a well-tuned SA ends at 1-5%) — a random walk.
# Sampling worsening deltas and inverting Metropolis targets a real acceptance
# rate: T0 = -mean|delta| / ln(TARGET_ACCEPT_INITIAL).
CALIBRATE_T_INIT = True
CALIBRATE_SAMPLES = 1000       # random moves drawn to estimate mean |delta|
# 0.35 rather than the textbook 0.8: roughly half of all sampled moves decode to
# an unchanged value on these instances (the decoder cannot distinguish many
# reorderings) and those are admitted at any temperature, so a T0 targeting 80%
# on the worsening moves alone leaves the walk far hotter than intended.
TARGET_ACCEPT_INITIAL = 0.35   # acceptance the calibrated T0 aims for
TARGET_ACCEPT_FINAL = 0.01     # acceptance T_MIN aims for, sets the floor

# Each reheat returns to a LOWER ceiling than the last, so the search still
# converges over a run with many reheats. Previously every reheat reset T to the
# same 0.25*T0 — with 1113 reheats in a 300 s run (measured) T never fell below
# that ceiling for the entire budget, which is why 13 of 24 runs found their
# best solution in the first ~3 s and then improved nothing for 297 s.
REHEAT_DECAY = 0.90        # ceiling multiplier per reheat: 0.25*T0 * DECAY**k

# Relative probability of drawing each move operator per iteration. Set an entry
# to 0 to disable that operator. These are weights, not probabilities — they are
# normalised.
MOVE_WEIGHTS = {
    "relocate":  0.0,
    "swap":      1.0,
    "2opt":      0.0,
    "promote":   1.5,   # slightly favoured: these two are the operators that
    "demote":    1.5,   # actually exchange packed against unpacked cargo
    "orient":    1.0,
    "container": 0.5,   # only meaningful on heterogeneous container fleets
}

_SAMPLE_TRIES = 12         # retries per sampler before it gives up this iteration


def cooling_sweep_iters(alpha, t_min=T_MIN, t_init=100.0):
    """Iterations of one full geometric sweep t_init -> t_min at this alpha."""
    return math.log(t_min / t_init) / math.log(alpha)


def default_reheat_threshold(alpha):
    return max(200, int(0.5 * cooling_sweep_iters(alpha)))


def calibrate_t_init(sol, inst, samples=CALIBRATE_SAMPLES, ep_limit=EP_LIMIT,
                     target_accept=TARGET_ACCEPT_INITIAL):
    """Estimate T0 from the objective deltas this instance actually produces.

    Draws random neighbours of `sol`, keeps the worsening ones, and inverts the
    Metropolis criterion on their mean magnitude:

        accept = exp(-mean|delta| / T0) = target_accept
        =>  T0 = -mean|delta| / ln(target_accept)

    The MEDIAN worsening delta is used, not the mean: the distribution is
    right-skewed (on ep3-20-U-C-50, median 3672 against mean 4903 with a max of
    9919), and a mean dragged up by rare large drops sets T0 high enough to
    accept the common small ones almost always.

    Returns (t_init, t_floor, n_worsening). t_floor is the same inversion at
    TARGET_ACCEPT_FINAL and is used as the end-of-schedule temperature, so the
    run spans a meaningful acceptance range instead of an arbitrary one.
    Returns (None, None, 0) when no worsening draw is found, leaving the caller
    to fall back to the f(S_0) fraction.
    """
    probe = copy_solution(sol)
    base = sol["value"]
    deltas = []
    for _ in range(samples):
        move = sample_random_move(probe, inst)
        if move is None:
            continue
        trial = copy_solution(probe)
        apply_move(trial, move, inst)
        delta = evaluate(trial, inst, ep_limit=ep_limit) - base
        if delta < 0:
            deltas.append(-delta)

    if not deltas:
        return None, None, 0

    ref = statistics.median(deltas)
    t_init = -ref / math.log(target_accept)
    t_floor = -ref / math.log(TARGET_ACCEPT_FINAL)
    return t_init, t_floor, len(deltas)


# =========================================================================
# Instance parsing  (identical format and value model to 3DMHKP.py)
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
    """The distinct axis-aligned orientations of a box (6, or fewer if square)."""
    return list(dict.fromkeys(permutations((l, w, h), 3)))


def fits(box_dims_oriented, container_dims):
    dx, dy, dz = box_dims_oriented
    L, W, H = container_dims
    return dx <= L and dy <= W and dz <= H


def _warn_on_value_mode(path, value_mode):
    """Warn when an instance file declares a value mode other than the one used.

    Converted Egeblad ep3 instances carry explicit profits in the coefficient
    column and so must run under --value-mode flat; under the default volume
    mode the objective silently becomes profit * volume, which is a different
    problem. Such files mark themselves with a "#! value-mode: flat" line.
    """
    declared = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("#!") and "value-mode:" in line:
                declared = line.split("value-mode:", 1)[1].strip()
                break
            if line and not line.startswith("#"):
                break
    if declared and declared != value_mode:
        print(f"  WARNING: {Path(path).name} declares value-mode {declared!r} "
              f"but is being solved with {value_mode!r}; pass "
              f"--value-mode {declared} for this instance's intended objective.",
              file=sys.stderr)


def build_instance(path, value_mode="volume"):
    """Expand box/container types into individual items and containers.

    The value model is the one verified in 3DMHKP.py: v_i = c_i * vol_i
    reproduces Bortfeldt's (2000) published upper bounds exactly.
    """
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

    inst = {
        "name": Path(path).stem,
        "boxes": boxes,
        "containers": containers,
        "box_types": box_types,
        "container_types": container_types,
        "value_mode": value_mode,
    }
    _precompute_fit_tables(inst)
    return inst


def _precompute_fit_tables(inst):
    """Cache, per (container, box type), the orientations that fit at all.

    Boxes of one type are interchangeable, so the table is keyed by type rather
    than by box index. compat[j] is the set of containers box j fits into.
    """
    boxes, containers = inst["boxes"], inst["containers"]
    types = {}
    for b in boxes:
        types.setdefault(b["type"], b)

    orient_by_kt = {}
    for k, c in enumerate(containers):
        for t, b in types.items():
            orient_by_kt[k, t] = tuple(d for d in b["orientations"]
                                       if fits(d, c["dims"]))

    inst["orient_by_kt"] = orient_by_kt
    inst["compat"] = [set(k for k in range(len(containers))
                          if orient_by_kt[k, b["type"]]) for b in boxes]


# =========================================================================
# Bound
# =========================================================================
def continuous_knapsack_bound(inst):
    """Upper bound by relaxing to a 1-D continuous knapsack.

    All containers are merged into a single volume capacity and each box enters
    only with its volume and value. This is the bound of Bortfeldt (2000),
    Sec. 5; see 3DMHKP.py for the verification against his Table 2.
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


def knapsack01_bound(inst):
    """Upper bound by the INTEGRAL 0-1 knapsack on volume.

    This is the "1D" bound of Egeblad & Pisinger (2009), Eq. (9): boxes may not
    be split, so it is tighter than continuous_knapsack_bound() above, and it
    reproduces their Table 8 "1D" column exactly on the ep3 instances.

    It is the bound to quote whenever rotation is allowed. The conservative-
    scales bound the paper reports alongside it is NOT valid under rotation
    (their Sec. 6.1), and a rotating solver can legitimately exceed it - on
    these instances six of our results do, which makes a gap against it
    meaningless rather than impressive.

    Volumes are divided through by their gcd to keep the DP table small; on the
    ep3 instances that is the difference between ~4e6 and ~4e3 states.
    """
    capacity = sum(c["vol"] for c in inst["containers"])
    boxes = inst["boxes"]
    if not boxes or capacity <= 0:
        return 0.0

    divisor = capacity
    for b in boxes:
        divisor = gcd(divisor, b["vol"])
    cap = capacity // divisor

    dp = [0.0] * (cap + 1)
    for b in boxes:
        w = b["vol"] // divisor
        v = b["value"]
        for c in range(cap, w - 1, -1):
            cand = dp[c - w] + v
            if cand > dp[c]:
                dp[c] = cand
    return max(dp)


# =========================================================================
# Decoder: priority order -> feasible packing
# =========================================================================
def decode(sol, inst, ep_limit=EP_LIMIT):
    """Deepest-bottom-left extreme-point packer driven by the SA solution.

    Containers are filled one at a time in sol["cont_order"]. Within a
    container the still-unpacked boxes are offered in sol["order"]; each is
    placed at the first extreme point (sorted by z, then y, then x) and the
    first orientation - starting from its preference sol["orient"][j] - where it
    fits inside the container without overlapping an already placed box.

    Returns (placement, value), placement being box -> (container, orientation
    index into box["orientations"], (x, y, z)).

    The `failed` set is what keeps this near-linear on the large instances:
    within one container pass the free space only ever shrinks, so once a box
    type has failed everywhere, every later box of that type must fail too and
    is skipped in O(1). It is only recorded when the point scan was exhaustive —
    under an ep_limit cap a failure is not conclusive.
    """
    boxes = inst["boxes"]
    containers = inst["containers"]
    compat = inst["compat"]
    orient_by_kt = inst["orient_by_kt"]
    order = sol["order"]
    orient_pref = sol["orient"]

    placement = {}
    value = 0.0

    for k in sol["cont_order"]:
        container = containers[k]
        L, W, H = container["dims"]
        free_vol = container["vol"]
        points = [(0, 0, 0)]        # extreme points as (z, y, x), kept sorted
        placed = []                 # (x, y, z, dx, dy, dz)
        failed = set()              # box types that no longer fit this container

        for j in order:
            if j in placement:
                continue
            box = boxes[j]
            t = box["type"]
            if t in failed:
                continue
            if k not in compat[j]:
                failed.add(t)
                continue
            if box["vol"] > free_vol:
                continue

            orients = orient_by_kt[k, t]
            nr = len(orients)
            start = orient_pref[j] % nr
            hit = None
            truncated = False

            for scanned, (z, y, x) in enumerate(points):
                if ep_limit and scanned >= ep_limit:
                    truncated = True
                    break
                for i in range(nr):
                    dims = orients[(start + i) % nr]
                    dx, dy, dz = dims
                    if x + dx > L or y + dy > W or z + dz > H:
                        continue
                    ok = True
                    for qx, qy, qz, qdx, qdy, qdz in placed:
                        if (x < qx + qdx and qx < x + dx
                                and y < qy + qdy and qy < y + dy
                                and z < qz + qdz and qz < z + dz):
                            ok = False
                            break
                    if ok:
                        hit = (x, y, z, dx, dy, dz,
                               box["orientations"].index(dims))
                        break
                if hit is not None:
                    break

            if hit is None:
                if not truncated:
                    failed.add(t)
                continue

            x, y, z, dx, dy, dz, r = hit
            placement[j] = (k, r, (x, y, z))
            value += box["value"]
            free_vol -= box["vol"]
            placed.append((x, y, z, dx, dy, dz))
            points.remove((z, y, x))

            # The three extreme points this box opens up. A point strictly
            # inside an already placed box can never host anything, so it is
            # dropped right away instead of being rescanned for every later box.
            for nx, ny, nz in ((x + dx, y, z), (x, y + dy, z), (x, y, z + dz)):
                if nx >= L or ny >= W or nz >= H:
                    continue
                p = (nz, ny, nx)
                if p in points:
                    continue
                inside = False
                for qx, qy, qz, qdx, qdy, qdz in placed:
                    if (qx <= nx < qx + qdx and qy <= ny < qy + qdy
                            and qz <= nz < qz + qdz):
                        inside = True
                        break
                if not inside:
                    insort(points, p)

    return placement, value


def evaluate(sol, inst, ep_limit=EP_LIMIT):
    """Decode a solution, store the packing on it, and return its value."""
    placement, value = decode(sol, inst, ep_limit=ep_limit)
    sol["placement"] = placement
    sol["packed"] = set(placement)
    sol["value"] = value
    return value


# =========================================================================
# Independent verification (same check as 3DMHKP.py)
# =========================================================================
def _overlaps(a_pos, a_dim, b_pos, b_dim):
    for axis in range(3):
        if a_pos[axis] + a_dim[axis] <= b_pos[axis] + 1e-9:
            return False
        if b_pos[axis] + b_dim[axis] <= a_pos[axis] + 1e-9:
            return False
    return True


def verify_placement(inst, placement, tol=1e-6):
    """Independently re-check a placement. Returns a list of violation strings."""
    errors = []
    by_container = {}
    for j, (k, r, pos) in placement.items():
        by_container.setdefault(k, []).append((j, r, pos))

    for k, entries in by_container.items():
        L, W, H = inst["containers"][k]["dims"]
        packed_boxes = []
        for j, r, pos in entries:
            dims = inst["boxes"][j]["orientations"][r]
            if pos[0] + dims[0] > L + tol or pos[1] + dims[1] > W + tol \
                    or pos[2] + dims[2] > H + tol or min(pos) < -tol:
                errors.append(f"box {j} sticks out of container {k}")
            packed_boxes.append((j, pos, dims))
        for a in range(len(packed_boxes)):
            for b in range(a + 1, len(packed_boxes)):
                ja, pa, da = packed_boxes[a]
                jb, pb, db = packed_boxes[b]
                if _overlaps(pa, da, pb, db):
                    errors.append(f"boxes {ja} and {jb} overlap in container {k}")
    return errors


# =========================================================================
# Solution representation
# =========================================================================
def copy_solution(sol):
    return {
        "order": list(sol["order"]),
        "orient": list(sol["orient"]),
        "cont_order": list(sol["cont_order"]),
        "placement": dict(sol["placement"]),
        "packed": set(sol["packed"]),
        "value": sol["value"],
    }


def build_initial_solution(inst, ep_limit=EP_LIMIT):
    """Value-density priority order, containers largest first.

    This is exactly greedy_pack() of 3DMHKP.py, so the initial decode
    reproduces the MILP's greedy warm-start value and SA starts from there.
    """
    boxes = inst["boxes"]
    order = sorted(range(len(boxes)),
                   key=lambda j: -(boxes[j]["value"] / boxes[j]["vol"]))
    cont_order = sorted(range(len(inst["containers"])),
                        key=lambda k: -inst["containers"][k]["vol"])
    sol = {
        "order": order,
        "orient": [0] * len(boxes),
        "cont_order": cont_order,
        "placement": {},
        "packed": set(),
        "value": 0.0,
    }
    evaluate(sol, inst, ep_limit=ep_limit)
    return sol


# =========================================================================
# Random single-move sampling
# =========================================================================
# Each sampler returns ONE randomly drawn move, or None when no move of that
# type can change the current solution. Two boxes of the same type with the same
# orientation preference are indistinguishable to the decoder, so every sampler
# filters draws that would decode to an identical packing; without that filter
# most iterations would be wasted re-decoding the solution SA already has.

def _key(sol, inst, pos):
    """What the decoder can actually tell apart at a position in the order."""
    j = sol["order"][pos]
    box = inst["boxes"][j]
    return (box["type"], sol["orient"][j] % len(box["orientations"]))


def _is_noop_span(sol, inst, src, dst):
    """True if moving the box at `src` to `dst` cannot change the packing.

    It cannot when every box it jumps over is indistinguishable from it.
    """
    if src == dst:
        return True
    key = _key(sol, inst, src)
    lo, hi = (src + 1, dst) if dst > src else (dst, src - 1)
    return all(_key(sol, inst, p) == key for p in range(lo, hi + 1))


def sample_relocate_move(sol, inst):
    """Move one box to another position in the priority order."""
    n = len(sol["order"])
    if n < 2:
        return None
    for _ in range(_SAMPLE_TRIES):
        src = random.randrange(n)
        dst = random.randrange(n)
        if _is_noop_span(sol, inst, src, dst):
            continue
        return {"type": "relocate", "src": src, "dst": dst}
    return None


def sample_swap_move(sol, inst):
    """Exchange the priority-order positions of two boxes."""
    n = len(sol["order"])
    if n < 2:
        return None
    for _ in range(_SAMPLE_TRIES):
        a = random.randrange(n)
        b = random.randrange(n)
        if a == b or _key(sol, inst, a) == _key(sol, inst, b):
            continue
        return {"type": "swap", "a": a, "b": b}
    return None


def sample_2opt_move(sol, inst):
    """Reverse a segment of the priority order.

    The routing 2-opt of 3dp-cptp-SA.py reversed a tour segment to save travel
    cost; there is no tour here, but reversing a segment of the priority order
    is the natural large-step perturbation: it flips the loading precedence of a
    whole block of boxes at once.
    """
    n = len(sol["order"])
    if n < 3:
        return None
    for _ in range(_SAMPLE_TRIES):
        i = random.randrange(n - 1)
        j = random.randrange(i + 1, n)
        keys = {_key(sol, inst, p) for p in range(i, j + 1)}
        if len(keys) < 2:            # homogeneous block — reversing changes nothing
            continue
        return {"type": "2opt", "i": i, "j": j}
    return None


def sample_promote_move(sol, inst):
    """Pull a currently UNPACKED box forward — the "insert" analogue.

    Moving it ahead of boxes that are currently packed is what lets it displace
    them; what the trade is worth is whatever the decoder makes of it.
    """
    order = sol["order"]
    unpacked = [p for p, j in enumerate(order) if j not in sol["packed"]]
    if not unpacked:
        return None
    random.shuffle(unpacked)
    for src in unpacked[:_SAMPLE_TRIES]:
        if src == 0:
            continue
        dst = random.randrange(src)
        if _is_noop_span(sol, inst, src, dst):
            continue
        return {"type": "promote", "src": src, "dst": dst}
    return None


def sample_demote_move(sol, inst):
    """Push a currently PACKED box back — the "remove" analogue."""
    order = sol["order"]
    n = len(order)
    packed = [p for p, j in enumerate(order) if j in sol["packed"]]
    if not packed:
        return None
    random.shuffle(packed)
    for src in packed[:_SAMPLE_TRIES]:
        if src == n - 1:
            continue
        dst = random.randrange(src + 1, n)
        if _is_noop_span(sol, inst, src, dst):
            continue
        return {"type": "demote", "src": src, "dst": dst}
    return None


def sample_orient_move(sol, inst):
    """Change the orientation preference of one box.

    This is the operator the routing SA has no counterpart for: it does not
    touch the priority order at all, only how the decoder first tries to lay the
    box down, which decides what shape of free space is left behind it.
    """
    boxes = inst["boxes"]
    n = len(boxes)
    for _ in range(_SAMPLE_TRIES):
        j = random.randrange(n)
        nr = len(boxes[j]["orientations"])
        if nr < 2:
            continue
        new = random.randrange(nr)
        if new == sol["orient"][j] % nr:
            continue
        return {"type": "orient", "box": j, "pref": new}
    return None


def sample_container_move(sol, inst):
    """Exchange two containers in the filling order.

    Only containers of different types are worth exchanging — filling two
    identical containers in the other order decodes to the same packing.
    """
    cont_order = sol["cont_order"]
    containers = inst["containers"]
    if len(cont_order) < 2:
        return None
    for _ in range(_SAMPLE_TRIES):
        a = random.randrange(len(cont_order))
        b = random.randrange(len(cont_order))
        if a == b:
            continue
        if containers[cont_order[a]]["type"] == containers[cont_order[b]]["type"]:
            continue
        return {"type": "container", "a": a, "b": b}
    return None


_SAMPLERS = {
    "relocate":  sample_relocate_move,
    "swap":      sample_swap_move,
    "2opt":      sample_2opt_move,
    "promote":   sample_promote_move,
    "demote":    sample_demote_move,
    "orient":    sample_orient_move,
    "container": sample_container_move,
}


def sample_random_move(sol, inst, move_weights=None):
    """Draw ONE random neighbour: pick a random operator, then a random move.

    Operators are drawn without replacement (by weight) so that if the chosen
    one cannot produce a move in the current solution, the next one is tried
    instead of wasting the whole iteration. Returns None only if no operator
    yields a move.
    """
    if move_weights is None:
        move_weights = MOVE_WEIGHTS

    pool = [(name, w) for name, w in move_weights.items()
            if w > 0 and name in _SAMPLERS]
    while pool:
        total = sum(w for _, w in pool)
        r = random.uniform(0.0, total)
        upto = 0.0
        pick = len(pool) - 1
        for idx, (_, w) in enumerate(pool):
            upto += w
            if r <= upto:
                pick = idx
                break
        name, _ = pool.pop(pick)
        move = _SAMPLERS[name](sol, inst)
        if move is not None:
            return move
    return None


# -------------------------
# Apply a move to a solution (in-place)
# -------------------------
def apply_move(sol, move, inst):
    mtype = move["type"]

    if mtype in ("relocate", "promote", "demote"):
        order = sol["order"]
        j = order.pop(move["src"])
        order.insert(move["dst"], j)

    elif mtype == "swap":
        order = sol["order"]
        a, b = move["a"], move["b"]
        order[a], order[b] = order[b], order[a]

    elif mtype == "2opt":
        order = sol["order"]
        i, j = move["i"], move["j"]
        order[i:j + 1] = order[i:j + 1][::-1]

    elif mtype == "orient":
        sol["orient"][move["box"]] = move["pref"]

    elif mtype == "container":
        co = sol["cont_order"]
        a, b = move["a"], move["b"]
        co[a], co[b] = co[b], co[a]


# -------------------------
# Diversification (analogue of the shake in 3dp-cptp-SA.py)
# -------------------------
def _diversify(sol, inst, ep_limit=EP_LIMIT):
    """Shake the current solution: promote rejected cargo, scramble a block.

    The routing version un-served a third of the served customers and greedily
    re-inserted them. The equivalent here is to give a third of the currently
    rejected boxes a random high priority, scramble one contiguous block of the
    order, and reshuffle the container sequence.
    """
    order = sol["order"]
    n = len(order)

    unpacked = [j for j in order if j not in sol["packed"]]
    if unpacked:
        n_promote = max(1, len(unpacked) // 3)
        chosen = set(random.sample(unpacked, min(n_promote, len(unpacked))))
        rest = [j for j in order if j not in chosen]
        head = max(1, n // 3)
        for j in chosen:
            rest.insert(random.randrange(head + 1), j)
        order = rest
        sol["order"] = order

    if n >= 4:
        seg = max(2, n // 5)
        i = random.randrange(n - seg + 1)
        block = order[i:i + seg]
        random.shuffle(block)
        order[i:i + seg] = block

    random.shuffle(sol["cont_order"])
    evaluate(sol, inst, ep_limit=ep_limit)


# =========================================================================
# Simulated Annealing main loop
# =========================================================================
def simulated_annealing(inst, max_iterations=MAX_ITERATIONS, time_limit=TIME_LIMIT,
                        t_init=T_INIT, alpha=ALPHA, t_min=T_MIN,
                        reheat_threshold=None, reheat_fraction=REHEAT_FRACTION,
                        t_init_fraction=T_INIT_FRACTION, ep_limit=EP_LIMIT,
                        schedule=SCHEDULE, calibrate=CALIBRATE_T_INIT,
                        reheat_decay=REHEAT_DECAY, verbose=True):
    """Simulated Annealing for the 3DMHKP over decoder inputs.

    Per iteration exactly ONE random neighbour is drawn (random operator +
    random move), decoded into a feasible packing, and then accepted or
    rejected. No neighbourhood is enumerated.

    Acceptance criterion (the objective, stowed value, is MAXIMIZED):
      - Improving moves (delta > 0): always accepted
      - Worsening moves: accepted with probability exp(delta / T)
    Temperature schedule:
      - T0 calibrated from sampled worsening deltas (calibrate=True), else
        t_init_fraction * f(S_0); an explicit t_init overrides both
      - schedule="time": T = T0 * (T_end/T0) ** (elapsed / time_limit), so the
        annealing curve is independent of decode throughput
        schedule="iter": legacy geometric T *= alpha per iteration
      - Reheating after prolonged stagnation, to a ceiling that decays by
        reheat_decay each time so the run still converges, combined with a
        diversification shake
    """
    t_start = time.time()
    deadline = t_start + time_limit
    if reheat_threshold is None:
        reheat_threshold = default_reheat_threshold(alpha)

    # ---- Initial solution ----
    current = build_initial_solution(inst, ep_limit=ep_limit)
    initial_value = current["value"]
    if verbose:
        print(f"  initial (value-density) decode: value {initial_value:.1f} "
              f"({len(current['packed'])}/{len(inst['boxes'])} boxes)")

    best = copy_solution(current)

    # ---- Temperature ----
    t_init_auto = t_init is None
    t_init_source = "fixed, --t-init"
    t_end = t_min
    calib_samples = 0
    if t_init is None:
        if calibrate:
            t_cal, t_floor, calib_samples = calibrate_t_init(
                current, inst, ep_limit=ep_limit)
            if t_cal is not None:
                t_init = t_cal
                # Floor the schedule at the calibrated low-acceptance point
                # rather than at the arbitrary T_MIN, but never above it.
                t_end = max(t_min, min(t_floor, t_init))
                t_init_source = (f"calibrated, {calib_samples} worsening draws, "
                                 f"target accept {TARGET_ACCEPT_INITIAL:.0%}")
        if t_init is None:
            t_init = max(1e-6, t_init_fraction * current["value"])
            t_init_source = f"{t_init_fraction:.0%} of f(S_0)"
        if verbose:
            print(f"  T0 = {t_init:.1f} ({t_init_source})")
            if schedule == "time":
                print(f"  schedule: time-based, T0 -> {t_end:.2f} "
                      f"over {time_limit:.0f}s")
    t_reheat = reheat_fraction * t_init

    T = t_init
    no_improve = 0
    accepted = rejected = no_move = reheats = neutral = 0
    iteration = 0
    # Under the time schedule a reheat cannot just assign T: the next cooling
    # step recomputes T from the clock and would erase it. Instead a reheat
    # raises this floor, which decays per reheat and is released as the
    # schedule's own curve falls back below it.
    reheat_floor = 0.0
    log_span = math.log(t_end / t_init) if t_init > 0 and t_end > 0 else 0.0

    def schedule_temperature():
        """Temperature from the configured schedule, before the reheat floor."""
        if schedule == "time":
            p = min(1.0, (time.time() - t_start) / time_limit)
            return max(t_end, t_init * math.exp(log_span * p))
        return None  # iteration schedule cools multiplicatively in place
    # Recorded in the report so a run stays interpretable after the fact: why
    # the loop ended, and how late in the budget the incumbent was still
    # improving (a best_time near the limit means more time would still pay).
    stopped_on = "iteration cap"
    best_iteration, best_time = 0, 0.0

    for iteration in range(1, max_iterations + 1):
        if time.time() >= deadline:
            stopped_on = "time limit"
            if verbose:
                print(f"  time limit reached at iteration {iteration}")
            break

        # ---- Draw ONE random neighbour ----
        move = sample_random_move(current, inst)
        if move is None:
            # Nothing can change this solution at all — shake it and retry.
            no_move += 1
            no_improve += 1
            if no_improve >= reheat_threshold:
                reheat_floor = t_reheat * (reheat_decay ** reheats)
                _diversify(current, inst, ep_limit=ep_limit)
                no_improve = 0
                reheats += 1
            sched_t = schedule_temperature()
            T = max(sched_t if sched_t is not None else T * alpha,
                    reheat_floor, t_min)
            continue

        # ---- Decode it: every neighbour is feasible by construction ----
        trial = copy_solution(current)
        apply_move(trial, move, inst)
        delta = evaluate(trial, inst, ep_limit=ep_limit) - current["value"]

        # ---- Metropolis acceptance ----
        # Neutral draws are counted apart from the Metropolis decision: on these
        # instances ~73% of sampled moves decode to the same value (the decoder
        # cannot tell many reorderings apart), and exp(0/T)=1 admits every one of
        # them at any temperature. Folding them into "accepted" made the overall
        # rate read 60-77% and hid what the schedule was really doing.
        if delta == 0:
            neutral += 1
        if delta > 0:
            accept = True                       # improving move — always accept
        elif T > 1e-12:
            accept = random.random() < math.exp(delta / T)
        else:
            accept = False

        if accept:
            current = trial
            accepted += 1
            if current["value"] > best["value"] + 1e-6:
                best = copy_solution(current)
                no_improve = 0
                best_iteration, best_time = iteration, time.time() - t_start
                if verbose:
                    print(f"    [iter {iteration}] *** NEW BEST: value "
                          f"{best['value']:.1f}, boxes {len(best['packed'])}/"
                          f"{len(inst['boxes'])}, T={T:.1f}, "
                          f"elapsed={time.time() - t_start:.1f}s")
            else:
                no_improve += 1
        else:
            rejected += 1
            no_improve += 1

        # ---- Reheat + diversify on prolonged stagnation ----
        # Raise the floor BEFORE cooling so the new ceiling takes effect this
        # iteration rather than one iteration late.
        if no_improve >= reheat_threshold:
            reheat_floor = t_reheat * (reheat_decay ** reheats)
            _diversify(current, inst, ep_limit=ep_limit)
            no_improve = 0
            reheats += 1
            if verbose:
                print(f"    [iter {iteration}] reheat + diversify: "
                      f"ceiling={reheat_floor:.2f}, "
                      f"value={current['value']:.1f}")

        # ---- Cool down once per iteration, whatever happened ----
        sched_t = schedule_temperature()
        T = max(sched_t if sched_t is not None else T * alpha,
                reheat_floor, t_min)

    elapsed = time.time() - t_start
    stats = {
        "iterations": iteration,
        "accepted": accepted,
        "rejected": rejected,
        "neutral": neutral,
        "no_move": no_move,
        "reheats": reheats,
        "t_init": t_init,
        "alpha": alpha,
        "reheat_threshold": reheat_threshold,
        "initial_value": initial_value,
        "elapsed": elapsed,
        "iters_per_sec": iteration / elapsed if elapsed > 0 else 0.0,
        # The configuration too, so the report is a complete record of the run
        # and not just of its outcome.
        "time_limit": time_limit,
        "max_iterations": max_iterations,
        "t_init_auto": t_init_auto,
        "t_init_fraction": t_init_fraction,
        "t_init_source": t_init_source,
        "t_min": t_min,
        "t_end": t_end,
        "schedule": schedule,
        "calibrate": calibrate,
        "calib_samples": calib_samples,
        "reheat_decay": reheat_decay,
        "t_reheat": t_reheat,
        "reheat_fraction": reheat_fraction,
        "ep_limit": ep_limit,
        "move_weights": dict(MOVE_WEIGHTS),
        "t_final": T,
        "stopped_on": stopped_on,
        "best_iteration": best_iteration,
        "best_time": best_time,
    }
    if verbose:
        acc_rate = accepted / max(1, accepted + rejected)
        # The rate that actually reflects the schedule: worsening moves admitted
        # out of the worsening moves drawn, with neutral draws excluded.
        worsening = accepted - neutral + rejected
        worse_rate = (accepted - neutral) / max(1, worsening)
        print(f"  SA finished: {iteration} iterations in {elapsed:.1f}s "
              f"({stats['iters_per_sec']:.0f} it/s), accepted {accepted} "
              f"({acc_rate:.1%}), rejected {rejected}, neutral {neutral} "
              f"({neutral / max(1, iteration):.1%}), "
              f"worsening accepted {worse_rate:.1%}, "
              f"no-move draws {no_move}, reheats {reheats}")
    return best, stats


# =========================================================================
# Reporting  (same report layout as 3DMHKP.py, plus the SA block)
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


def report_name(inst_name, solver, args):
    """Report filename, carrying the parameters that define the run.

    A set of 16 reports is only interpretable next to the settings that
    produced it, and two runs at different settings otherwise overwrite each
    other silently - which is how a directory named "Sa900" came to hold
    1800 s results. The name carries the three settings that are varied in
    practice, cooling rate, reheat factor and time limit:

        instance01-3DMHKP-SA-a0.999-r0.25-t1800s.txt

    --tag still appends a free-text suffix on top, for runs that differ in
    something the name does not cover (a seed, a move-weight set).
    """
    if args.greedy_only:
        params = "-greedy"          # no cooling schedule to name
    else:
        params = (f"-a{args.alpha:g}-r{args.reheat_fraction:g}"
                  f"-t{args.time_limit:g}s")
    tag = f"-{args.tag}" if args.tag else ""
    return f"{inst_name}-{solver}{params}{tag}.txt"


def write_report(path, inst, placement, result):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    stats = summarize(inst, placement)
    total_container_vol = sum(c["vol"] for c in inst["containers"])
    packed_vol = sum(inst["boxes"][j]["vol"] for j in placement)
    sa = result["sa"]

    with open(path, "w") as f:
        f.write(f"Instance: {inst['name']}\n")
        f.write("Solver: simulated annealing (sequence decoder)\n")
        formula = ("v_i = c_i" if inst["value_mode"] == "flat"
                   else "v_i = c_i * volume_i")
        f.write(f"Value model: {formula} ({inst['value_mode']})\n")
        f.write(f"Boxes: {len(inst['boxes'])} ({len(inst['box_types'])} types)\n")
        f.write(f"Containers: {len(inst['containers'])} "
                f"({len(inst['container_types'])} types)\n\n")

        tag = f"  (tag: {result['tag']})" if result.get("tag") else ""
        on = [k for k, v in sa["move_weights"].items() if v > 0]
        off = [k for k, v in sa["move_weights"].items() if v <= 0]

        f.write(f"Run started: {result['started']:%Y-%m-%d %H:%M:%S}{tag}\n")
        f.write(f"Command: {result['command']}\n")
        f.write(f"Time limit: {sa['time_limit']:.1f} s per instance\n")
        f.write(f"Runtime: {result['runtime']:.2f} s (SA loop {sa['elapsed']:.2f} s, "
                f"stopped on {sa['stopped_on']})\n")
        f.write(f"Packed value (SA incumbent / lower bound): {result['value']:.1f}\n")
        f.write(f"0-1 knapsack bound (1D, valid under rotation): "
                f"{result['kp_bound']:.1f}\n")
        f.write(f"Continuous-knapsack relaxation: {result['ck_bound']:.1f}\n")
        f.write(f"Gap to the 0-1 bound: {result['gap']:.2%}\n")
        f.write(f"Greedy (initial decode) value: {sa['initial_value']:.1f}\n")
        f.write(f"SA improvement over greedy: "
                f"{result['value'] - sa['initial_value']:+.1f}\n\n")

        f.write("SA parameters:\n")
        f.write(f"  time limit: {sa['time_limit']:.1f} s, "
                f"max iterations: {sa['max_iterations']}\n")
        # The cooling schedule only exists if the SA loop actually ran; under
        # --greedy-only there is no T0 to report a sweep length against.
        if sa["iterations"]:
            f.write(f"  T0: {sa['t_init']:.2f} "
                    f"({sa.get('t_init_source', 'unknown')})\n")
            if sa.get("schedule") == "time":
                f.write(f"  schedule: time-based, T0 -> {sa['t_end']:.2f} "
                        f"over {sa['time_limit']:.0f}s "
                        f"(alpha unused)\n")
            else:
                sweep = cooling_sweep_iters(sa["alpha"], sa["t_min"],
                                            max(sa["t_init"], 1e-6))
                f.write(f"  schedule: iteration-based, alpha: {sa['alpha']}, "
                        f"T_min: {sa['t_min']:g}, "
                        f"one cooling sweep: {sweep:.0f} it\n")
            f.write(f"  reheat threshold: {sa['reheat_threshold']} it, "
                    f"ceiling {sa['t_reheat']:.2f} "
                    f"({sa['reheat_fraction']:.0%} of T0) "
                    f"decaying x{sa.get('reheat_decay', 1.0):g} per reheat\n")
        else:
            f.write("  cooling schedule: not applicable, the SA loop did not "
                    "run\n")
        f.write(f"  ep limit: {sa['ep_limit']}"
                f"{' (unlimited)' if not sa['ep_limit'] else ''}, "
                f"value mode: {inst['value_mode']}\n")
        f.write(f"  move weights: "
                + ", ".join(f"{k}={sa['move_weights'][k]:g}" for k in on)
                + (f" (disabled: {', '.join(off)})" if off else "") + "\n")
        f.write(f"  seed: {result['seed']}\n\n")

        f.write("SA statistics:\n")
        f.write(f"  iterations: {sa['iterations']} in {sa['elapsed']:.2f} s "
                f"({sa['iters_per_sec']:.0f} it/s)\n")
        f.write(f"  stopped on: {sa['stopped_on']}\n")
        f.write(f"  accepted: {sa['accepted']}, rejected: {sa['rejected']}, "
                f"no-move draws: {sa['no_move']}, reheats: {sa['reheats']}\n")
        f.write(f"  acceptance rate: "
                f"{sa['accepted'] / max(1, sa['accepted'] + sa['rejected']):.1%}\n")
        if sa["iterations"]:
            f.write(f"  T at exit: {sa['t_final']:g}\n")
        if sa["best_iteration"]:
            share = (f", {sa['best_time'] / sa['elapsed']:.0%} into the run"
                     if sa["elapsed"] > 0 else "")
            f.write(f"  last improvement: iteration {sa['best_iteration']} at "
                    f"{sa['best_time']:.1f} s{share}\n")
        else:
            f.write("  last improvement: none - the incumbent is the greedy "
                    "decode\n")
        f.write(f"  verification: {result['verification']}\n\n")

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
    _warn_on_value_mode(path, args.value_mode)

    n_boxes = len(inst["boxes"])
    n_containers = len(inst["containers"])
    unfittable = sum(1 for c in inst["compat"] if not c)
    # The integral 0-1 bound is always at least as tight as the continuous one
    # and is the bound that stays valid under rotation, so it is the one gaps
    # are reported against. The continuous bound is kept for continuity with the
    # earlier Bortfeldt-referenced runs.
    ck_bound = continuous_knapsack_bound(inst)
    kp_bound = knapsack01_bound(inst)
    bound = kp_bound if kp_bound > 0 else ck_bound

    print(f"\n{'=' * 70}")
    print(f"{inst['name']}: {n_boxes} boxes ({len(inst['box_types'])} types), "
          f"{n_containers} containers ({len(inst['container_types'])} types)")
    print(f"  0-1 knapsack bound: {kp_bound:.1f} "
          f"(continuous relaxation: {ck_bound:.1f})")
    if unfittable:
        print(f"  note: {unfittable} box(es) fit no container - never packable")

    t0 = time.time()
    started = datetime.now()

    if args.greedy_only:
        best = build_initial_solution(inst, ep_limit=args.ep_limit)
        sa_stats = {"iterations": 0, "accepted": 0, "rejected": 0, "no_move": 0,
                    "reheats": 0, "t_init": 0.0, "alpha": args.alpha,
                    "reheat_threshold": 0, "initial_value": best["value"],
                    "elapsed": time.time() - t0, "iters_per_sec": 0.0,
                    "time_limit": args.time_limit,
                    "max_iterations": args.max_iterations,
                    "t_init_auto": args.t_init is None,
                    "t_init_fraction": args.t_init_fraction,
                    "t_init_source": "n/a", "t_end": args.t_min,
                    "schedule": args.schedule, "calibrate": False,
                    "calib_samples": 0, "reheat_decay": args.reheat_decay,
                    "t_min": args.t_min, "t_reheat": 0.0,
                    "reheat_fraction": args.reheat_fraction,
                    "ep_limit": args.ep_limit,
                    "move_weights": dict(MOVE_WEIGHTS), "t_final": 0.0,
                    "stopped_on": "greedy only, SA not run",
                    "best_iteration": 0, "best_time": 0.0}
        print(f"  greedy-only decode: value {best['value']:.1f} "
              f"({len(best['packed'])}/{n_boxes} boxes)")
    else:
        best, sa_stats = simulated_annealing(
            inst,
            max_iterations=args.max_iterations,
            time_limit=args.time_limit,
            t_init=args.t_init,
            alpha=args.alpha,
            t_min=args.t_min,
            reheat_threshold=args.reheat_threshold,
            reheat_fraction=args.reheat_fraction,
            t_init_fraction=args.t_init_fraction,
            schedule=args.schedule,
            calibrate=not args.no_calibrate,
            reheat_decay=args.reheat_decay,
            ep_limit=args.ep_limit,
            verbose=not args.quiet,
        )

    placement = best["placement"]
    errs = verify_placement(inst, placement)
    if errs:
        print(f"  WARNING: SA packing failed verification: {errs[0]}")

    packed_vol = sum(inst["boxes"][j]["vol"] for j in placement)
    total_vol = sum(c["vol"] for c in inst["containers"])

    result = {
        "instance": inst["name"],
        "n_boxes": n_boxes,
        "n_containers": n_containers,
        "ck_bound": ck_bound,
        "kp_bound": kp_bound,
        "bound": bound,
        "value": best["value"],
        # The bound is an upper bound, so a decode that reaches it is optimal;
        # clamp away the float noise instead of printing a negative gap.
        "gap": max(0.0, (bound - best["value"]) / bound) if bound > 0 else 0.0,
        "n_packed": len(placement),
        "utilization": packed_vol / total_vol if total_vol else 0.0,
        "runtime": time.time() - t0,
        "sa": sa_stats,
        "seed": args.seed,
        "verification": "ok" if not errs else f"{len(errs)} violation(s): {errs[0]}",
        "started": started,
        "command": args.command,
        "tag": args.tag,
    }

    result["optimal"] = result["gap"] <= 1e-9
    print(f"  SA: value {result['value']:.1f} "
          f"(greedy {sa_stats['initial_value']:.1f}, "
          f"{result['value'] - sa_stats['initial_value']:+.1f}), "
          f"bound {bound:.1f}, gap {result['gap']:.2%}"
          f"{' - matches the bound, so PROVEN OPTIMAL' if result['optimal'] else ''}, "
          f"util {result['utilization']:.1%}")

    if not args.no_report:
        results_dir = getattr(args, "results_dir", None) or RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / report_name(inst["name"], "3DMHKP-SA", args)
        write_report(out, inst, placement, result)
        print(f"  report -> {out}")

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulated annealing for the 3D multiple heterogeneous "
                    "knapsack problem of Mohanty, Mathur & Ivancic (1994).")
    parser.add_argument("--instances", nargs="*", type=int, default=None,
                        metavar="N",
                        help="instance numbers to solve (default: all 16)")
    parser.add_argument("--instance-dir", type=Path, default=INSTANCE_DIR,
                        help=f"instance directory (default: {INSTANCE_DIR})")
    parser.add_argument("--instance-files", nargs="*", type=Path, default=None,
                        metavar="PATH",
                        help="explicit instance files to solve, instead of the "
                             "instanceNN.txt numbering of --instances. Accepts "
                             "any file in the BOXES/CONTAINERS format, e.g. the "
                             "converted Egeblad set: --instance-files "
                             "Egeblad/*.txt --value-mode flat")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                        help=f"SA seconds per instance (default: {TIME_LIMIT})")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                        help="maximum SA iterations per instance")
    parser.add_argument("--t-init", type=float, default=T_INIT,
                        help="initial temperature (default: a fraction of the "
                             "initial objective, see --t-init-fraction)")
    parser.add_argument("--t-init-fraction", type=float, default=T_INIT_FRACTION,
                        help=f"T0 as a fraction of the initial objective value "
                             f"(default: {T_INIT_FRACTION})")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"geometric cooling factor (default: {ALPHA})")
    parser.add_argument("--t-min", type=float, default=T_MIN,
                        help="minimum temperature")
    parser.add_argument("--reheat-threshold", type=int, default=REHEAT_THRESHOLD,
                        help="iterations without a new best before reheating "
                             "(default: half a cooling sweep at the given alpha)")
    parser.add_argument("--reheat-fraction", type=float, default=REHEAT_FRACTION,
                        help="T is reset to this fraction of T0 on a reheat")
    parser.add_argument("--schedule", choices=("time", "iter"), default=SCHEDULE,
                        help="cooling schedule: 'time' anneals on wall-clock "
                             "progress (default), 'iter' is the legacy "
                             "geometric T *= alpha per iteration")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="skip T0 calibration from sampled move deltas and "
                             "use --t-init-fraction * f(S_0) instead (legacy)")
    parser.add_argument("--reheat-decay", type=float, default=REHEAT_DECAY,
                        help="reheat ceiling is multiplied by this per reheat, "
                             "so repeated reheats still converge (1.0 = legacy "
                             "constant ceiling)")
    parser.add_argument("--ep-limit", type=int, default=EP_LIMIT,
                        help="max extreme points scanned per box (0 = unlimited)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default: random)")
    parser.add_argument("--value-mode", choices=("volume", "flat"), default="volume",
                        help="volume: v_i = c_i * vol_i (default, matches the "
                             "literature); flat: v_i = c_i")
    parser.add_argument("--greedy-only", action="store_true",
                        help="only decode the initial value-density order, no SA")
    parser.add_argument("--tag", default="",
                        help="extra suffix for the report filenames. The names "
                             "already carry the cooling rate, reheat factor and "
                             "time limit (instanceNN-3DMHKP-SA-a0.999-r0.25-"
                             "t1800s.txt), so a tag is only needed to separate "
                             "runs that differ in something else, such as the "
                             "seed or the move weights")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"directory for the per-instance reports "
                             f"(default: {RESULTS_DIR.name}). Give a separate "
                             f"directory to a run over a different instance "
                             f"set, so its reports stay together.")
    parser.add_argument("--no-report", action="store_true",
                        help="do not write per-instance report files")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the per-improvement SA log")
    args = parser.parse_args(argv)

    if args.seed is None:
        args.seed = random.randrange(2 ** 31)
    random.seed(args.seed)

    # The reports record how this solver was invoked. main_slurm.py calls
    # main(argv) in-process, so sys.argv is the dispatcher's line, not ours -
    # record the arguments we actually parsed, and the launching command after
    # them when the two differ.
    args.command = shlex.join([Path(__file__).name]
                              + (list(argv) if argv is not None else sys.argv[1:]))
    if argv is not None:
        args.command += f"   [launched by: {shlex.join(sys.argv)}]"

    if args.instance_files:
        paths = []
        for p in args.instance_files:
            # A shell that cannot expand the glob (or a quoted pattern) hands
            # us the pattern itself; expand it here so both forms work.
            matches = sorted(Path().glob(str(p))) if any(
                ch in str(p) for ch in "*?[") else [p]
            if not matches:
                print(f"no instance file matches: {p}", file=sys.stderr)
                return 1
            for m in matches:
                if not m.exists():
                    print(f"missing instance file: {m}", file=sys.stderr)
                    return 1
                paths.append(m)
    else:
        numbers = args.instances if args.instances else list(range(1, 17))
        paths = []
        for n in numbers:
            p = args.instance_dir / f"instance{n:02d}.txt"
            if not p.exists():
                print(f"missing instance file: {p}", file=sys.stderr)
                return 1
            paths.append(p)

    reheat = (args.reheat_threshold if args.reheat_threshold is not None
              else default_reheat_threshold(args.alpha))
    print("=" * 70)
    print("3DMHKP Simulated Annealing (random single-neighbour, sequence decoder)")
    t0_desc = (f"{args.t_init_fraction:.0%} of f(S_0)"
               if args.t_init is None else f"{args.t_init}")
    print(f"SA params: T0={t0_desc}, "
          f"alpha={args.alpha}, T_min={args.t_min}, "
          f"reheat_threshold={reheat}, reheat_fraction={args.reheat_fraction}")
    print(f"Move weights: {MOVE_WEIGHTS}")
    print(f"Time limit: {args.time_limit}s per instance, seed {args.seed}")
    print("=" * 70)

    results = []
    for p in paths:
        try:
            results.append(solve_instance(p, args))
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            break

    if not results:
        return 1

    print(f"\n{'=' * 96}")
    print("SUMMARY")
    print(f"{'=' * 96}")
    print(f"{'instance':<12}{'boxes':>7}{'packed':>8}{'greedy':>13}{'SA value':>13}"
          f"{'bound':>13}{'gap':>9}{'util':>8}{'iters':>9}{'time':>8}")
    print("-" * 96)
    for r in results:
        print(f"{r['instance']:<12}{r['n_boxes']:>7}{r['n_packed']:>8}"
              f"{r['sa']['initial_value']:>13.1f}{r['value']:>13.1f}"
              f"{r['bound']:>13.1f}{r['gap']:>8.1%}{r['utilization']:>8.1%}"
              f"{r['sa']['iterations']:>9}{r['runtime']:>7.1f}s")
    print("-" * 96)

    proven = sum(1 for r in results if r.get("optimal"))
    print(f"proven optimal (value reaches the upper bound): {proven}/{len(results)}")
    improved = sum(1 for r in results if r["value"] > r["sa"]["initial_value"] + 1e-6)
    total_greedy = sum(r["sa"]["initial_value"] for r in results)
    total_sa = sum(r["value"] for r in results)
    print(f"improved over greedy: {improved}/{len(results)} "
          f"(total value {total_greedy:.1f} -> {total_sa:.1f}, "
          f"{(total_sa / total_greedy - 1) if total_greedy else 0:+.2%})")
    print(f"mean gap to 0-1 knapsack bound: "
          f"{sum(r['gap'] for r in results) / len(results):.1%}")
    print(f"mean volume utilization: "
          f"{sum(r['utilization'] for r in results) / len(results):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
