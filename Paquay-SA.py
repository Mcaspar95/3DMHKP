#!/usr/bin/env python3
"""Simulated annealing for the Paquay, Limbourg & Schyns (2018) 3D MBSBPP.

    C. Paquay, S. Limbourg, M. Schyns, "A tailored two-phase constructive
    heuristic for the three-dimensional Multiple Bin Size Bin Packing Problem
    with transportation constraints", EJOR 267 (2018) 52-64.

Reads the authors' own instances from Paquay/ (see Paquay/README.md for their
provenance and file formats; load_paquay.py is the parser).

WHAT PROBLEM THIS SOLVES
========================
The paper's problem, not a relaxation of it. Every box must be packed, each
into exactly one ULD, minimising the volume of the ULDs selected. ULD types are
available in unlimited quantity. All seven of the paper's constraints
(Section 2) are enforced here:

  1. every box assigned to exactly one used ULD,
  2. containment, INCLUDING the bevelled ULD shapes (the z = ax + b cuts),
  3. no overlap,
  4. orthogonal rotation, restricted per box by the l+/w+/h+ flags,
  5. fragile boxes support nothing,
  6. per-ULD weight capacity,
  7. stability (support fraction beneath each box).

With --cg, the paper's two centre-of-gravity constraints are enforced as well:
the CG of each ULD must stay below its maximum height and inside the allowable
area around the ULD's geometric centre (Sec. 2). Without --cg they are only
measured and reported. Use --cg for a like-for-like comparison with the paper.

METHOD
======
SA over a box permutation plus a per-box orientation choice, decoded by an
extreme-point placement routine. This mirrors the project's other solvers
(sequence-decoded SA) rather than the paper's deterministic two-phase
constructive heuristic, which is the point of the comparison: the paper's
method is a fast one-shot construction, ours trades time for quality.

Decoder: boxes in sequence order, each placed at the feasible extreme point
minimising Paquay's merit function MF1 = (RSx-l)+(RSy-w)+(RSz-h), opening a new
ULD when none fits. ULD type for a new bin is chosen by trying each type and
keeping the one that ends up cheapest per unit volume packed. A final repack
pass retries each used ULD in every smaller type, which is the paper's phase 2.

Moves: swap two boxes, shift a box, reverse a segment, change one orientation.
Geometric cooling. Objective = total volume of used ULDs (minimise); ties are
broken on packed-volume concentration so the search can see progress.

LIMITATIONS -- read before quoting any number against the paper
===============================================================
  * WITHOUT --cg, the CG constraints are measured but not enforced, so results
    are obtained under a weaker constraint set than the paper's and are to that
    extent optimistic. `--report-cg` sizes the gap. WITH --cg both are hard:
    the height limit filters extreme points during placement, and a bin whose
    lateral CG cannot be repaired makes the decode infeasible.
  * The CG tolerances are read from columns 6-8 of uld.csv (aL, aW, aH). The
    paper states the constraint but defers its numeric definition to Paquay
    et al. (2016) and a Boeing loading manual, neither of which is to hand, so
    this reading is INFERRED from the column values: aL/L and aW/W are
    0.05-0.10 (lateral tolerances) while aH/H is 0.41-0.53 (a mid-height
    ceiling). It is consistent with the paper's prose and with the four
    directional deviations (Left/Right/Front/Rear) of its Fig. 13, but it is
    not quoted from the paper and could differ from what the authors used.
  * Stability is a support-area fraction test (default 100%, i.e. full support
    on the box footprint, configurable via --support). The paper cites the
    coordinate-based conditions of Paquay et al. (2016), which are not
    identical.
  * The paper reports filling rates as BOXPLOTS (Fig. 11), not as a table of
    means, so there is no published per-size mean to difference against. The
    two hard numbers it does give, and which this script reproduces, are
    1010 ULDs used over the 300 final instances, and a maximum runtime of
    12 s per instance. Compare on those, and on the Fig. 11 quantiles.

USAGE
    python Paquay-SA.py --n 10 --time 5            # one size, 5 s per instance
    python Paquay-SA.py --time 10 --out results_Paquay_SA
    python Paquay-SA.py --n 20 --sample 0 --time 30 --verbose
    python Paquay-SA.py --greedy                   # decoder only, no SA
    python Paquay-SA.py --cg --time 12 --out results_Paquay_SA_cg
"""

import argparse
import json
import math
import os
import random
import sys
import time

from load_paquay import (FINAL_SIZES, TRAINING_SIZES, SAMPLES_PER_SIZE,
                         load_boxes, load_ulds, instance_path, iter_instances)

EPS = 1e-6


# --------------------------------------------------------------------------
# ULD geometry, including the bevelled corners
# --------------------------------------------------------------------------

class Uld:
    """A ULD type. Cuts are half-planes in the XZ projection, z*den + num*x <=> b.

    The sense of each cut (which side is inside) is not stored in uld.csv, so it
    is derived: the half-space containing the ULD centroid is the feasible one.
    Validated against column 21 (exact volume in mm^3) for all six types, worst
    error 0.01% -- see the provenance note in Paquay/README.md.
    """

    def __init__(self, rec):
        self.code = rec.code
        self.L, self.W, self.H = rec.length, rec.width, rec.height
        self.max_weight = rec.max_weight
        self.volume_mm3 = rec.volume_mm3
        self.uid = rec.uid
        # Columns 6-8 of uld.csv. Read as the CG tolerances of paper Sec. 2:
        # the CG must stay within +/- aL of the ULD's geometric centre along x
        # and +/- aW along y ("a determined area around the geometrical centre
        # of the ULD"), and below height aH ("and below a maximum height").
        # Consistent with the values: aL/L and aW/W are ~0.05-0.10 while aH/H is
        # ~0.41-0.53, i.e. two lateral tolerances and one mid-height ceiling.
        self.aL, self.aW, self.aH = rec.aL, rec.aW, rec.aH
        cx, cz = self.L / 2.0, self.H / 2.0
        self.cuts = []
        for den, num, b in rec.cuts:
            sense = 1 if (cz * den + num * cx) <= b else -1
            self.cuts.append((den, num, b, sense))

    def _xz_ok(self, x, z):
        for den, num, b, sense in self.cuts:
            lhs = z * den + num * x
            if sense == 1:
                if lhs > b + EPS:
                    return False
            else:
                if lhs < b - EPS:
                    return False
        return True

    def contains(self, x, y, z, l, w, h):
        """Is the whole box inside the ULD? Cuts are in XZ and independent of y,
        so testing the four XZ corners of the box is exact for a convex region."""
        if x < -EPS or y < -EPS or z < -EPS:
            return False
        if x + l > self.L + EPS or y + w > self.W + EPS or z + h > self.H + EPS:
            return False
        if not self.cuts:
            return True
        for xx in (x, x + l):
            for zz in (z, z + h):
                if not self._xz_ok(xx, zz):
                    return False
        return True


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------

def orientations(box):
    """Permutations of (l, w, h) allowed by the box's l+/w+/h+ flags.

    Flag k+ means axis k may be the VERTICAL one. Paper Sec. 6.1.2: heavy boxes
    (> 50 kg) have l+ = w+ = 0, h+ is 1 for every box.
    """
    L, W, H = box.length, box.width, box.height
    out = []
    if box.h_up:                      # h vertical
        out.append((L, W, H))
        out.append((W, L, H))
    if box.l_up:                      # l vertical
        out.append((H, W, L))
        out.append((W, H, L))
    if box.w_up:                      # w vertical
        out.append((L, H, W))
        out.append((H, L, W))
    seen, uniq = set(), []
    for o in out:
        if o not in seen:
            seen.add(o)
            uniq.append(o)
    return uniq or [(L, W, H)]


class Bin:
    """One open ULD and the boxes placed in it."""

    __slots__ = ("uld", "placed", "weight", "eps", "mx", "my", "mz")

    def __init__(self, uld):
        self.uld = uld
        self.placed = []          # (x, y, z, l, w, h, box_index, fragile, weight)
        self.weight = 0.0
        self.eps = [(0.0, 0.0, 0.0)]
        self.mx = self.my = self.mz = 0.0     # weight moments, for the CG

    def packed_volume(self):
        return sum(p[3] * p[4] * p[5] for p in self.placed)

    def filling_rate(self):
        return 100.0 * self.packed_volume() / self.uld.volume_mm3

    def _overlaps(self, x, y, z, l, w, h):
        for (px, py, pz, pl, pw, ph, _i, _f, _wt) in self.placed:
            if (x + l > px + EPS and px + pl > x + EPS and
                    y + w > py + EPS and py + pw > y + EPS and
                    z + h > pz + EPS and pz + ph > z + EPS):
                return True
        return False

    def _support_ok(self, x, y, z, l, w, h, min_frac):
        """Fraction of the box footprint resting on the floor or on box tops.

        Also enforces constraint 5: nothing may rest on a fragile box.
        """
        if z <= EPS:
            return True
        need = l * w * min_frac
        area = 0.0
        for (px, py, pz, pl, pw, ph, _i, pfrag, _wt) in self.placed:
            if abs(pz + ph - z) > EPS:
                continue
            ox = max(0.0, min(x + l, px + pl) - max(x, px))
            oy = max(0.0, min(y + w, py + pw) - max(y, py))
            if ox > 0 and oy > 0:
                if pfrag:
                    return False          # cannot stack on a fragile box
                area += ox * oy
        return area >= need - EPS

    def residual_space(self, x, y, z):
        """RS of an extreme point: free distance to the next obstacle per axis."""
        rx = self.uld.L - x
        ry = self.uld.W - y
        rz = self.uld.H - z
        # A box blocks an axis if it lies ahead on that axis and its extent
        # straddles the EP on the other two.
        for (px, py, pz, pl, pw, ph, _i, _f, _wt) in self.placed:
            span_y = py - EPS <= y < py + pw - EPS
            span_z = pz - EPS <= z < pz + ph - EPS
            span_x = px - EPS <= x < px + pl - EPS
            if px > x + EPS and span_y and span_z:
                rx = min(rx, px - x)
            if py > y + EPS and span_x and span_z:
                ry = min(ry, py - y)
            if pz > z + EPS and span_x and span_y:
                rz = min(rz, pz - z)
        return rx, ry, rz

    def _cg_after(self, x, y, z, l, w, h, wt):
        """CG of the load if this box were added here."""
        tw = self.weight + wt
        if tw <= 0:
            return None
        return ((self.mx + (x + l / 2.0) * wt) / tw,
                (self.my + (y + w / 2.0) * wt) / tw,
                (self.mz + (z + h / 2.0) * wt) / tw)

    def cg_height_ok(self, x, y, z, l, w, h, wt):
        """Paper constraint 5: the CG stays below the ULD's maximum height."""
        cg = self._cg_after(x, y, z, l, w, h, wt)
        return cg is None or cg[2] <= self.uld.aH + EPS

    def cg_lateral_ok(self, cg=None):
        """CG inside the allowable area around the geometric centre in XY.

        Checked on the FINAL load, not per placement: an intermediate CG may
        legitimately sit outside the region and be brought back by later boxes,
        exactly as the paper's jump & shift repair does after the fact.
        """
        cg = cg or self.centre_of_gravity()
        if cg is None:
            return True
        return (abs(cg[0] - self.uld.L / 2.0) <= self.uld.aL + EPS and
                abs(cg[1] - self.uld.W / 2.0) <= self.uld.aW + EPS)

    def try_place(self, box, idx, dims, min_frac, cg_height=False):
        """Best feasible EP for this box/orientation by MF1, or None."""
        l, w, h = dims
        if self.weight + box.weight > self.uld.max_weight + EPS:
            return None
        best = None
        for (x, y, z) in self.eps:
            if not self.uld.contains(x, y, z, l, w, h):
                continue
            if self._overlaps(x, y, z, l, w, h):
                continue
            if not self._support_ok(x, y, z, l, w, h, min_frac):
                continue
            if cg_height and not self.cg_height_ok(x, y, z, l, w, h, box.weight):
                continue
            rx, ry, rz = self.residual_space(x, y, z)
            mf = (rx - l) + (ry - w) + (rz - h)
            if best is None or mf < best[0]:
                best = (mf, x, y, z)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def commit(self, box, idx, dims, pos):
        l, w, h = dims
        x, y, z = pos
        self.placed.append((x, y, z, l, w, h, idx, box.fragile, box.weight))
        self.weight += box.weight
        self.mx += (x + l / 2.0) * box.weight
        self.my += (y + w / 2.0) * box.weight
        self.mz += (z + h / 2.0) * box.weight
        if (x, y, z) in self.eps:
            self.eps.remove((x, y, z))
        for cand in ((x + l, y, z), (x, y + w, z), (x, y, z + h)):
            if (cand not in self.eps and
                    self.uld.contains(cand[0], cand[1], cand[2], 0.0, 0.0, 0.0)):
                self.eps.append(cand)

    def centre_of_gravity(self):
        if not self.placed or self.weight <= 0:
            return None
        sx = sy = sz = 0.0
        for (x, y, z, l, w, h, _i, _f, wt) in self.placed:
            sx += (x + l / 2.0) * wt
            sy += (y + w / 2.0) * wt
            sz += (z + h / 2.0) * wt
        return sx / self.weight, sy / self.weight, sz / self.weight


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------

def _tail_volumes(boxes, order):
    """Suffix sums of box volume along `order`: tail[k] covers order[k:]."""
    tail = [0.0] * (len(order) + 1)
    for k in range(len(order) - 1, -1, -1):
        b = boxes[order[k]]
        tail[k] = tail[k + 1] + b.length * b.width * b.height
    return tail


def decode(boxes, order, orient, ulds, min_frac, repack=True, cg=False):
    """Place every box; open new ULDs as needed. Returns list of Bin.

    Every box must be packed (paper Sec. 2). If a box fits no open bin, a new
    bin is opened; the type chosen is the smallest one that can hold the box,
    which keeps selected volume low while guaranteeing feasibility.
    """
    bins = []
    by_vol = sorted(ulds, key=lambda u: u.volume_mm3)
    tail = _tail_volumes(boxes, order)
    for seq_k, pos_i in enumerate(order):
        box = boxes[pos_i]
        opts = orientations(box)
        dims = opts[orient[pos_i] % len(opts)]
        done = False
        # try open bins first
        for b in bins:
            spot = b.try_place(box, pos_i, dims, min_frac, cg)
            if spot is not None:
                b.commit(box, pos_i, dims, spot)
                done = True
                break
        if done:
            continue
        # try every allowed orientation in open bins before opening a new one
        for d in opts:
            for b in bins:
                spot = b.try_place(box, pos_i, d, min_frac, cg)
                if spot is not None:
                    b.commit(box, pos_i, d, spot)
                    done = True
                    break
            if done:
                break
        if done:
            continue
        # Open a new ULD. Choosing the smallest type that fits this one box is
        # short-sighted -- it ignores the boxes still queued behind it, which
        # then force further bins. Size the new bin against the whole remaining
        # tail instead, and let the phase-2 repack shrink it if that overshoots.
        remaining = tail[seq_k]
        for u in by_vol:
            if u.volume_mm3 < remaining * 0.95 and u is not by_vol[-1]:
                continue
            nb = Bin(u)
            for d in [dims] + opts:
                spot = nb.try_place(box, pos_i, d, min_frac, cg)
                if spot is not None:
                    nb.commit(box, pos_i, d, spot)
                    bins.append(nb)
                    done = True
                    break
            if done:
                break
        if not done:
            # nothing large enough by the tail rule; fall back to any type
            for u in by_vol:
                nb = Bin(u)
                for d in [dims] + opts:
                    spot = nb.try_place(box, pos_i, d, min_frac, cg)
                    if spot is not None:
                        nb.commit(box, pos_i, d, spot)
                        bins.append(nb)
                        done = True
                        break
                if done:
                    break
        if not done:
            return None          # box fits no ULD at all: infeasible decode
    if repack:
        bins = repack_pass(bins, boxes, ulds, min_frac, cg)
    if cg:
        # Paper's post-processing: repair lateral CG deviations.
        #
        # A bin that will not balance is not necessarily a dead end -- the load
        # is often simply too sparse for the ULD it landed in, and the same
        # boxes balance in a different type. So retry the offending bin in every
        # other ULD type (largest first, since room to shift is what is missing)
        # before giving up. Only if no type balances is the decode rejected,
        # which is what makes the constraint HARD rather than merely reported.
        out = []
        for b in bins:
            if balance_bin(b, boxes, min_frac, cg):
                out.append(b)
                continue
            contents = sorted(b.placed, key=lambda p: -(p[3] * p[4] * p[5]))
            fixed = None
            for u in sorted(ulds, key=lambda u: -u.volume_mm3):
                if u is b.uld:
                    continue
                nb = Bin(u)
                ok = True
                for (_x, _y, _z, l, w, h, idx, _f, _wt) in contents:
                    box = boxes[idx]
                    spot = None
                    for d in [(l, w, h)] + orientations(box):
                        spot = nb.try_place(box, idx, d, min_frac, cg)
                        if spot is not None:
                            nb.commit(box, idx, d, spot)
                            break
                    if spot is None:
                        ok = False
                        break
                if ok and balance_bin(nb, boxes, min_frac, cg):
                    fixed = nb
                    break
            if fixed is None:
                return None
            out.append(fixed)
        bins = out
    return bins


def _carries_load(b, p, min_frac):
    """Would removing p leave some other box unsupported?"""
    x, y, z, l, w, h = p[0], p[1], p[2], p[3], p[4], p[5]
    top = z + h
    rest = None
    for q in b.placed:
        if q is p or q[2] <= EPS or abs(q[2] - top) > EPS:
            continue
        ox = max(0.0, min(x + l, q[0] + q[3]) - max(x, q[0]))
        oy = max(0.0, min(y + w, q[1] + q[4]) - max(y, q[1]))
        if ox <= 0 or oy <= 0:
            continue                      # not resting on p at all
        if rest is None:
            rest = [r for r in b.placed if r is not p]
        saved = b.placed
        b.placed = rest
        ok = b._support_ok(q[0], q[1], q[2], q[3], q[4], q[5], min_frac)
        b.placed = saved
        if not ok:
            return True
    return False


def _translate_to_centre(b):
    """Slide the whole load along x and y to bring the CG toward the centre.

    A rigid translation cannot break overlap, support or stacking, so only
    containment needs rechecking. The offset is the one that would centre the
    CG exactly, clipped to what the ULD walls and the cuts allow.
    """
    if not b.placed or b.weight <= 0:
        return
    cg = b.centre_of_gravity()
    want_dx = b.uld.L / 2.0 - cg[0]
    want_dy = b.uld.W / 2.0 - cg[1]

    for axis, want in ((0, want_dx), (1, want_dy)):
        if abs(want) < EPS:
            continue
        lo = min(p[axis] for p in b.placed)
        hi = max(p[axis] + p[axis + 3] for p in b.placed)
        room = (b.uld.L if axis == 0 else b.uld.W)
        d = max(-lo, min(want, room - hi))      # keep inside the walls
        if abs(d) < EPS:
            continue
        # binary-search the largest feasible fraction of d (cuts may bite)
        best = 0.0
        step = d
        for _ in range(12):
            trial = best + step
            ok = True
            for (x, y, z, l, w, h, _i, _f, _wt) in b.placed:
                nx = x + (trial if axis == 0 else 0.0)
                ny = y + (trial if axis == 1 else 0.0)
                if not b.uld.contains(nx, ny, z, l, w, h):
                    ok = False
                    break
            if ok:
                best = trial
            step /= 2.0
            if abs(step) < 1e-3:
                break
        if abs(best) < EPS:
            continue
        moved = []
        for (x, y, z, l, w, h, i, f, wt) in b.placed:
            nx = x + (best if axis == 0 else 0.0)
            ny = y + (best if axis == 1 else 0.0)
            moved.append((nx, ny, z, l, w, h, i, f, wt))
        b.placed[:] = moved
        if axis == 0:
            b.mx += best * b.weight
            b.eps = [(ex + best, ey, ez) for (ex, ey, ez) in b.eps]
        else:
            b.my += best * b.weight
            b.eps = [(ex, ey + best, ez) for (ex, ey, ez) in b.eps]


def balance_bin(b, boxes, min_frac, cg_height, max_rounds=6):
    """Repair a lateral CG deviation, after the paper's jump & shift operators.

    The paper's jump takes boxes from the overloaded side and repacks them on
    extreme points on the opposite side; shift pushes boxes along an axis. Both
    are implemented here as one move: lift a box and re-place it at the feasible
    position that most reduces the CG offset, keeping the pattern feasible at
    every step. Candidate boxes are considered from the overloaded side inward
    and by decreasing density, as the paper orders them.

    Returns True if the bin ends up balanced.
    """
    # Translating the whole load is the cheapest repair and often the only one
    # available: the packing grows from the front-left corner, so the entire
    # group frequently sits against two walls with free space opposite. Moving
    # every box by the same offset preserves overlap, support and stacking
    # exactly, so only containment has to be rechecked.
    _translate_to_centre(b)

    for _ in range(max_rounds):
        cg = b.centre_of_gravity()
        if cg is None or b.cg_lateral_ok(cg):
            return True
        dx = cg[0] - b.uld.L / 2.0
        dy = cg[1] - b.uld.W / 2.0
        over_x = abs(dx) > b.uld.aL + EPS
        over_y = abs(dy) > b.uld.aW + EPS

        # boxes on the heavy side first, densest first
        def key(p):
            x, y, z, l, w, h, _i, _f, wt = p
            vol = l * w * h
            dens = wt / vol if vol > 0 else 0.0
            reach = 0.0
            if over_x:
                reach += (x + l / 2.0 - b.uld.L / 2.0) * (1 if dx > 0 else -1)
            if over_y:
                reach += (y + w / 2.0 - b.uld.W / 2.0) * (1 if dy > 0 else -1)
            return (-reach, -dens)

        moved = False
        for p in sorted(b.placed, key=key):
            x, y, z, l, w, h, idx, frag, wt = p
            if wt <= 0:
                continue
            # Only boxes carrying nothing may be lifted: the paper's jump
            # requires the pattern to stay feasible with the box removed, and
            # anything resting on this one would lose its support.
            if _carries_load(b, p, min_frac):
                continue
            # lift it
            b.placed.remove(p)
            b.weight -= wt
            b.mx -= (x + l / 2.0) * wt
            b.my -= (y + w / 2.0) * wt
            b.mz -= (z + h / 2.0) * wt
            b.eps.append((x, y, z))

            before = abs(dx) / max(b.uld.aL, EPS) + abs(dy) / max(b.uld.aW, EPS)
            best = None
            box = boxes[idx]
            for d in [(l, w, h)] + orientations(box):
                # The paper's shift operator slides boxes along an axis, so the
                # useful positions are not only the extreme points (which sit at
                # a wall or a neighbour's far face) but also the offsets that
                # centre this box. Without these a heavy box can never be moved
                # off a wall and lateral balance is often unreachable.
                cands = list(b.eps)
                for (ex, ey, ez) in list(b.eps):
                    cands.append((b.uld.L / 2.0 - d[0] / 2.0, ey, ez))
                    cands.append((ex, b.uld.W / 2.0 - d[1] / 2.0, ez))
                    cands.append((b.uld.L / 2.0 - d[0] / 2.0,
                                  b.uld.W / 2.0 - d[1] / 2.0, ez))
                for (ex, ey, ez) in cands:
                    if not b.uld.contains(ex, ey, ez, d[0], d[1], d[2]):
                        continue
                    if b._overlaps(ex, ey, ez, d[0], d[1], d[2]):
                        continue
                    if not b._support_ok(ex, ey, ez, d[0], d[1], d[2], min_frac):
                        continue
                    if cg_height and not b.cg_height_ok(ex, ey, ez, d[0], d[1],
                                                        d[2], wt):
                        continue
                    ncg = b._cg_after(ex, ey, ez, d[0], d[1], d[2], wt)
                    if ncg is None:
                        continue
                    score = (abs(ncg[0] - b.uld.L / 2.0) / max(b.uld.aL, EPS) +
                             abs(ncg[1] - b.uld.W / 2.0) / max(b.uld.aW, EPS))
                    if best is None or score < best[0]:
                        best = (score, ex, ey, ez, d)
            if best is not None and best[0] < before - 1e-9:
                _s, ex, ey, ez, d = best
                b.commit(box, idx, d, (ex, ey, ez))
                moved = True
                break
            # put it back untouched
            b.placed.append(p)
            b.weight += wt
            b.mx += (x + l / 2.0) * wt
            b.my += (y + w / 2.0) * wt
            b.mz += (z + h / 2.0) * wt
            if (x, y, z) in b.eps:
                b.eps.remove((x, y, z))
        if not moved:
            break
    return b.cg_lateral_ok()


def repack_pass(bins, boxes, ulds, min_frac, cg=False):
    """Paper phase 2: retry each used ULD's contents in every smaller type."""
    by_vol = sorted(ulds, key=lambda u: u.volume_mm3)
    out = []
    for b in bins:
        contents = sorted(b.placed, key=lambda p: -(p[3] * p[4] * p[5]))
        best = b
        for u in by_vol:
            if u.volume_mm3 >= best.uld.volume_mm3 - EPS:
                break
            nb = Bin(u)
            ok = True
            for (_x, _y, _z, l, w, h, idx, _f, _wt) in contents:
                box = boxes[idx]
                spot = None
                for d in [(l, w, h)] + orientations(box):
                    spot = nb.try_place(box, idx, d, min_frac, cg)
                    if spot is not None:
                        nb.commit(box, idx, d, spot)
                        break
                if spot is None:
                    ok = False
                    break
            if ok:
                best = nb
                break
        out.append(best)
    return out


def objective(bins):
    """Total volume of selected ULDs [mm^3] -- the paper's minimisation target."""
    return sum(b.uld.volume_mm3 for b in bins)


# --------------------------------------------------------------------------
# Simulated annealing
# --------------------------------------------------------------------------

def solve_instance(boxes, ulds, time_limit, seed=0, min_frac=1.0,
                   t0=None, tend=None, greedy=False, verbose=False, cg=False):
    rng = random.Random(seed)
    n = len(boxes)

    # initial order: decreasing base area (irace's choice, boxSort = 2)
    order = sorted(range(n), key=lambda i: -(boxes[i].length * boxes[i].width))
    orient = [0] * n

    cur = decode(boxes, order, orient, ulds, min_frac, cg=cg)
    if cur is None:
        # With --cg the default order can decode into a pattern that will not
        # balance. That is a property of this one sequence, not of the instance,
        # so try other starts before declaring failure -- otherwise the search
        # never begins and a solvable instance is reported infeasible.
        deadline = time.time() + max(time_limit * 0.25, 2.0)
        alts = [sorted(range(n), key=lambda i: -(boxes[i].length * boxes[i].width
                                                 * boxes[i].height)),
                sorted(range(n), key=lambda i: -boxes[i].weight),
                sorted(range(n), key=lambda i: -max(boxes[i].length,
                                                    boxes[i].width,
                                                    boxes[i].height))]
        for cand_order in alts:
            cur = decode(boxes, cand_order, orient, ulds, min_frac, cg=cg)
            if cur is not None:
                order = cand_order
                break
        while cur is None and time.time() < deadline:
            cand_order = list(range(n))
            rng.shuffle(cand_order)
            cur = decode(boxes, cand_order, orient, ulds, min_frac, cg=cg)
            if cur is not None:
                order = cand_order
        if cur is None:
            return None
    cur_obj = objective(cur)
    best, best_obj = cur, cur_obj
    best_order, best_orient = list(order), list(orient)

    if greedy or time_limit <= 0:
        return {"bins": best, "obj": best_obj, "iters": 0, "seconds": 0.0}

    total_vol = sum(b.length * b.width * b.height for b in boxes)
    t0 = t0 if t0 is not None else max(total_vol * 0.02, 1.0)
    tend = tend if tend is not None else t0 * 1e-3

    start = time.time()
    it = 0
    while True:
        elapsed = time.time() - start
        if elapsed >= time_limit:
            break
        frac = elapsed / time_limit
        temp = t0 * (tend / t0) ** frac

        new_order, new_orient = list(order), list(orient)
        m = rng.random()
        if m < 0.35 and n >= 2:                     # swap
            i, j = rng.randrange(n), rng.randrange(n)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif m < 0.60 and n >= 2:                   # shift
            i = rng.randrange(n)
            v = new_order.pop(i)
            new_order.insert(rng.randrange(n), v)
        elif m < 0.80 and n >= 3:                   # reverse a segment
            i, j = sorted(rng.sample(range(n), 2))
            new_order[i:j + 1] = reversed(new_order[i:j + 1])
        else:                                       # re-orient one box
            i = rng.randrange(n)
            new_orient[i] = rng.randrange(6)

        cand = decode(boxes, new_order, new_orient, ulds, min_frac, cg=cg)
        it += 1
        if cand is None:
            continue
        cand_obj = objective(cand)
        d = cand_obj - cur_obj
        if d <= 0 or rng.random() < math.exp(-d / max(temp, EPS)):
            order, orient, cur, cur_obj = new_order, new_orient, cand, cand_obj
            if cand_obj < best_obj - EPS:
                best, best_obj = cand, cand_obj
                best_order, best_orient = list(new_order), list(new_orient)
                if verbose:
                    print("    %6.1fs  it %6d  obj %.3f m3  ULDs %d"
                          % (time.time() - start, it, best_obj / 1e9, len(best)))

    return {"bins": best, "obj": best_obj, "iters": it,
            "seconds": time.time() - start}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def cg_stats(bins):
    """CG diagnostics. Not enforced -- reported so the gap to the paper's
    constrained results stays visible. Returns (n_uld, n_high, n_offcentre)."""
    high = off = 0
    for b in bins:
        cg = b.centre_of_gravity()
        if cg is None:
            continue
        x, y, z = cg
        if z > b.uld.H / 2.0:
            high += 1
        dx = abs(x - b.uld.L / 2.0) / b.uld.L
        dy = abs(y - b.uld.W / 2.0) / b.uld.W
        if dx > 0.10 or dy > 0.10:
            off += 1
    return len(bins), high, off


def summarise(rows):
    print()
    print("  %4s %6s %8s %10s %9s %9s %8s" %
          ("n", "inst", "ULDs", "vol[m3]", "fill%", "maxfill%", "sec"))
    tot_uld = 0
    tot_time = 0.0
    max_time = 0.0
    all_fills = []
    for n in sorted({r["n"] for r in rows}):
        grp = [r for r in rows if r["n"] == n]
        u = sum(r["ulds"] for r in grp)
        vol = sum(r["obj"] for r in grp) / len(grp) / 1e9
        fills = [f for r in grp for f in r["fills"]]
        mean_fill = sum(fills) / len(fills) if fills else 0.0
        secs = sum(r["seconds"] for r in grp) / len(grp)
        tot_uld += u
        tot_time += sum(r["seconds"] for r in grp)
        max_time = max(max_time, max(r["seconds"] for r in grp))
        all_fills.extend(fills)
        print("  %4d %6d %8d %10.2f %9.2f %9.2f %8.2f"
              % (n, len(grp), u, vol, mean_fill, max(fills) if fills else 0, secs))
    print("  " + "-" * 62)
    mean_all = sum(all_fills) / len(all_fills) if all_fills else 0.0
    print("  total ULDs used: %d   mean filling rate: %.2f%%   max: %.2f%%"
          % (tot_uld, mean_all, max(all_fills) if all_fills else 0))
    print("  runtime: %.1f s total, %.2f s max per instance" % (tot_time, max_time))
    if all_fills:
        s = sorted(all_fills)
        q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
        print("  filling-rate quartiles: Q1 %.2f  median %.2f  Q3 %.2f"
              % (q(0.25), q(0.50), q(0.75)))
    return tot_uld


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, help="only this box count")
    ap.add_argument("--sample", type=int, help="only this sample index")
    ap.add_argument("--training", action="store_true",
                    help="use the training sets instead of the final ones")
    ap.add_argument("--time", type=float, default=5.0,
                    help="SA seconds per instance (default 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--support", type=float, default=1.0,
                    help="required support fraction under a box (default 1.0)")
    ap.add_argument("--greedy", action="store_true",
                    help="decoder only, no SA (the constructive baseline)")
    ap.add_argument("--out", help="directory for per-instance JSON reports")
    ap.add_argument("--cg", action="store_true",
                    help="ENFORCE the CG height limit and lateral balance "
                         "(paper Sec. 2); off by default")
    ap.add_argument("--report-cg", action="store_true",
                    help="also print CG diagnostics (not enforced)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ulds = [Uld(u) for u in load_ulds()]
    sizes = [args.n] if args.n else (TRAINING_SIZES if args.training else FINAL_SIZES)
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    rows = []
    cg_tot = cg_high = cg_off = 0
    for n in sizes:
        samples = [args.sample] if args.sample is not None else range(SAMPLES_PER_SIZE)
        for k in samples:
            path = instance_path(n, k, args.training)
            if not os.path.exists(path):
                continue
            boxes = load_boxes(path)
            res = solve_instance(boxes, ulds, args.time, seed=args.seed + k,
                                 min_frac=args.support, greedy=args.greedy,
                                 verbose=args.verbose, cg=args.cg)
            if res is None:
                print("n%dsample%d: INFEASIBLE decode" % (n, k), file=sys.stderr)
                continue
            bins = res["bins"]
            fills = [b.filling_rate() for b in bins]
            t, h, o = cg_stats(bins)
            cg_tot += t
            cg_high += h
            cg_off += o
            row = {"n": n, "sample": k, "ulds": len(bins), "obj": res["obj"],
                   "fills": fills, "seconds": res["seconds"],
                   "iters": res["iters"],
                   "types": [b.uld.code for b in bins],
                   "cg_enforced": bool(args.cg)}
            rows.append(row)
            if args.verbose or args.sample is not None:
                print("n%-4d sample %-3d ULDs %2d  %-22s vol %7.2f m3  "
                      "mean fill %5.2f%%  %5.1fs (%d it)"
                      % (n, k, len(bins), ",".join(b.uld.code for b in bins),
                         res["obj"] / 1e9, sum(fills) / len(fills),
                         res["seconds"], res["iters"]))
            if args.out:
                with open(os.path.join(args.out, "n%dsample%d.json" % (n, k)), "w") as fh:
                    json.dump(row, fh, indent=1)

    if not rows:
        print("No instances solved.")
        return

    total = summarise(rows)
    if args.report_cg:
        print("\n  CG diagnostics (NOT enforced, see module docstring):")
        print("    ULDs %d | CG above half height: %d (%.1f%%) | "
              "CG >10%% off centre in XY: %d (%.1f%%)"
              % (cg_tot, cg_high, 100.0 * cg_high / max(cg_tot, 1),
                 cg_off, 100.0 * cg_off / max(cg_tot, 1)))

    if not args.training and args.n is None and args.sample is None:
        print("\n  Paper reference point: Paquay et al. (2018) use 1010 ULDs over")
        print("  these same 300 instances, max 12 s per instance (Sec. 6.3.1/6.3.3).")
        print("  Ours: %d ULDs. Note our CG constraints are NOT enforced, so this" % total)
        print("  is not yet a like-for-like comparison -- see module docstring.")


if __name__ == "__main__":
    main()
