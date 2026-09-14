#!/usr/bin/env python3
"""Reader for the Paquay, Limbourg & Schyns (2018) 3D MBSBPP instances.

    C. Paquay, S. Limbourg, M. Schyns, "A tailored two-phase constructive
    heuristic for the three-dimensional Multiple Bin Size Bin Packing Problem
    with transportation constraints", EJOR 267 (2018) 52-64.

The files under Paquay/ are the AUTHORS' OWN instances, downloaded from the
ORBi record cited in the paper (handle 2268/206856). They are not regenerated
here - see Paquay/README.md for the provenance and for the format tables.

This module only PARSES. It deliberately does not convert to the
BOXES/CONTAINERS "Mohanty-format" the four solver scripts read, because that
conversion would be lossy in ways that matter more here than they did for
convert_bortfeldt.py. Paquay et al. minimise unused space subject to a set of
transportation constraints that have no representation in this project's
solvers:

  1. ULD WEIGHT CAPACITY. Every ULD has a max payload (1518-10840 kg). The
     solvers have no weight dimension at all.

  2. PER-BOX ORIENTATION. Columns l+/w+/h+ say which axes may point up. Boxes
     over 50 kg may not be turned (l+ = w+ = 0). The solvers assume free
     orthogonal rotation.

  3. FRAGILITY / STACKING. A fragile box (density < 0.05 kg/dm3) may carry
     nothing on top. Not representable.

  4. NON-CUBOID ULDs. Four of the six ULD types have bevelled corners, given
     as the cut equations in uld.csv. The solvers assume rectangular
     containers, which would OVERSTATE capacity for LD1, LD6, PA and PG.

  5. CENTRE-OF-GRAVITY BALANCE. The CG must fall inside an allowable region;
     the paper's whole shift/jump post-processing exists to enforce it.

Consequently a filling rate produced by this project's solvers on these
instances would NOT be comparable to the rates in Paquay et al. Tables 6-9.
Reporting one as if it were would repeat exactly the error flagged in the
header of convert_bortfeldt.py. Parse first, decide what to enforce, and only
then compare.

Usage:
    python load_paquay.py                    # summary of the whole benchmark
    python load_paquay.py --n 50             # just the n=50 final instances
    python load_paquay.py --training         # the tuning instances instead
"""

import argparse
import csv
import os
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
PAQUAY_DIR = os.path.join(HERE, "Paquay")

FINAL_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TRAINING_SIZES = [10, 15, 20, 25, 30]
SAMPLES_PER_SIZE = 30

# Density below which the paper treats a box as fragile [kg/dm3], Sec. 6.1.2.
FRAGILE_DENSITY = 0.05
# Weight above which the paper forbids turning a box [kg], Sec. 6.1.2.
UNTURNABLE_WEIGHT = 50.0

Box = namedtuple("Box", "length width height weight l_up w_up h_up fragile")
ULD = namedtuple("ULD", "code length width height max_weight volume "
                        "aL aW aH cuts volume_mm3 uid")

# uld.csv carries no IATA codes; identify rows by their dimensions (Table 1).
_ULD_CODES = {
    (2337, 1534, 1626): "LD1",
    (3175, 1534, 1626): "LD11",
    (4064, 1534, 1626): "LD6",
    (2235, 3175, 2997): "PA",
    (2438, 6058, 2438): "PG",
    (2438, 3175, 2997): "PM",
}


def load_boxes(path):
    """Read one instance file into a list of Box."""
    boxes = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh, delimiter=";"):
            if not row or not row[0].strip():
                continue
            boxes.append(Box(
                length=float(row[0]), width=float(row[1]), height=float(row[2]),
                weight=float(row[3]),
                l_up=bool(int(row[4])), w_up=bool(int(row[5])),
                h_up=bool(int(row[6])), fragile=bool(int(row[7])),
            ))
    return boxes


def load_ulds(path=None):
    """Read uld.csv into a list of ULD, in the file's own order."""
    path = path or os.path.join(PAQUAY_DIR, "uld.csv")
    ulds = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh, delimiter=";"):
            if not row or not row[0].strip():
                continue
            L, W, H = int(row[0]), int(row[1]), int(row[2])
            # Columns 9-20 are four (denominator, numerator, constant) triples;
            # an all-zero triple means that cut is absent.
            cuts = []
            for i in range(8, 20, 3):
                triple = tuple(float(x) for x in row[i:i + 3])
                if any(triple):
                    cuts.append(triple)
            ulds.append(ULD(
                code=_ULD_CODES.get((L, W, H), "?"),
                length=L, width=W, height=H,
                max_weight=float(row[3]), volume=float(row[4]),
                aL=float(row[5]), aW=float(row[6]), aH=float(row[7]),
                cuts=cuts, volume_mm3=float(row[20]), uid=int(row[21]),
            ))
    return ulds


def instance_path(n, sample, training=False):
    sub = "training_data_sets" if training else "Final_instances"
    return os.path.join(PAQUAY_DIR, sub, "n%dsample%d.csv" % (n, sample))


def iter_instances(sizes=None, training=False):
    """Yield (n, sample, boxes) over a whole data set."""
    sizes = sizes or (TRAINING_SIZES if training else FINAL_SIZES)
    for n in sizes:
        for k in range(SAMPLES_PER_SIZE):
            path = instance_path(n, k, training)
            if os.path.exists(path):
                yield n, k, load_boxes(path)


def density(box):
    """kg per cubic decimetre."""
    return box.weight / (box.length * box.width * box.height / 1e6)


def check_derived_flags(boxes):
    """Re-verify the paper's two derivation rules. Returns (n_orient, n_frag)
    violations; both should be 0 on the published files."""
    bad_orient = sum(1 for b in boxes
                     if b.weight > UNTURNABLE_WEIGHT and (b.l_up or b.w_up))
    bad_frag = sum(1 for b in boxes
                   if b.fragile != (density(b) < FRAGILE_DENSITY))
    return bad_orient, bad_frag


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, help="only this box count")
    ap.add_argument("--training", action="store_true",
                    help="use the training sets instead of the final ones")
    args = ap.parse_args()

    ulds = load_ulds()
    print("ULD types (%d):" % len(ulds))
    print("  %-5s %-22s %9s %8s %5s" % ("code", "L x W x H [mm]", "max[kg]",
                                        "vol[m3]", "cuts"))
    for u in ulds:
        print("  %-5s %-22s %9.0f %8.1f %5d" % (
            u.code, "%d x %d x %d" % (u.length, u.width, u.height),
            u.max_weight, u.volume, len(u.cuts)))

    sizes = [args.n] if args.n else None
    rows = list(iter_instances(sizes, args.training))
    if not rows:
        print("\nNo instances found - is Paquay/ populated?")
        return

    label = "training" if args.training else "final"
    print("\n%s data set: %d instances" % (label, len(rows)))
    print("  %4s %6s %10s %10s %8s %8s" % ("n", "count", "vol[m3]", "weight[kg]",
                                           "fragile", "fixed"))

    tot_o = tot_f = 0
    for n in sorted({r[0] for r in rows}):
        group = [b for nn, _, bs in rows if nn == n for b in bs]
        k = sum(1 for nn, _, _ in rows if nn == n)
        vol = sum(b.length * b.width * b.height for b in group) / 1e9 / k
        wt = sum(b.weight for b in group) / k
        frag = sum(1 for b in group if b.fragile) / k
        fixed = sum(1 for b in group if not (b.l_up or b.w_up)) / k
        o, f = check_derived_flags(group)
        tot_o += o
        tot_f += f
        print("  %4d %6d %10.2f %10.1f %8.2f %8.2f" % (n, k, vol, wt, frag, fixed))

    print("\nderivation-rule violations: orientation %d, fragility %d "
          "(both should be 0)" % (tot_o, tot_f))
    print("NOTE: these instances carry weight, orientation, fragility, "
          "stability\n      and CG constraints. See the module docstring "
          "before comparing\n      filling rates against Paquay et al.")


if __name__ == "__main__":
    main()
