#!/usr/bin/env python3
"""Convert Egeblad ep3 .3kp instances into the BOXES/CONTAINERS format used
by 3DMHKP.py and 3DMHKP-SA.py.

Source format (Egeblad 2007, kp2d3d-data/readme.txt):
    dim, W, H, D
    box, i, w, h, d, p, c

Target format (parse_instance in 3DMHKP.py / 3DMHKP-SA.py):
    BOXES      rows: type_id count length width height value_coefficient
    CONTAINERS rows: type_id count length width height

The ep3 profits p are given explicitly and are NOT proportional to volume, so
they go in the value_coefficient column verbatim and the instances must be run
with --value-mode flat (value = coef). Writing p/vol into the column to use the
default volume mode would be lossy: the ratios are repeating decimals.

Boxes with identical (w, h, d, p) are grouped into one type row with a count.
"""

import argparse
import re
from collections import Counter
from pathlib import Path


def parse_3kp(path):
    """Return (container_dims, [(dims, profit, count), ...]) from a .3kp file."""
    container = None
    # Counter over (w, h, d, p), preserving first-seen order for stable output.
    boxes = Counter()
    order = []

    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        kind = parts[0].lower()

        if kind == "dim":
            if len(parts) != 4:
                raise ValueError(f"{path}:{lineno}: expected 'dim, W, H, D'\n  {line!r}")
            container = tuple(int(x) for x in parts[1:4])
        elif kind in ("box", "rect"):
            # box, i, w, h, d, p, c  -- index parts[1] is ignored per the readme.
            if len(parts) != 7:
                raise ValueError(f"{path}:{lineno}: expected 7 fields\n  {line!r}")
            w, h, d, p, c = (int(x) for x in parts[2:7])
            key = (w, h, d, p)
            if key not in boxes:
                order.append(key)
            boxes[key] += c
        else:
            raise ValueError(f"{path}:{lineno}: unknown record {kind!r}\n  {line!r}")

    if container is None:
        raise ValueError(f"{path}: no 'dim' line found")
    if not boxes:
        raise ValueError(f"{path}: no box lines found")

    return container, [(k[:3], k[3], boxes[k]) for k in order]


def format_coef(p):
    """Profits are integers; keep them exact and unadorned in the coef column."""
    return str(p)


def convert(src, dst):
    container, types = parse_3kp(src)
    W, H, D = container

    total_boxes = sum(c for _, _, c in types)
    total_profit = sum(p * c for _, p, c in types)
    total_vol = sum(dims[0] * dims[1] * dims[2] * c for dims, _, c in types)
    cap = W * H * D

    lines = [
        f"# Egeblad & Pisinger ep3 instance {src.stem}, converted from {src.name}",
        "# Source: Egeblad, J. & Pisinger, D. (2009), \"Heuristic approaches for",
        "# the two- and three-dimensional knapsack packing problem\",",
        "# Computers & Operations Research 36(4):1026-1049. Data set kp2d3d-data/ep3.",
        "#",
        "# Single-container 3D knapsack: one container, boxes carry explicit profits.",
        "# Format: BOXES rows      = type_id count length width height value_coefficient",
        "#         CONTAINERS rows = type_id count length width height",
        "#",
        "# The value_coefficient column holds the ORIGINAL ep3 profit p of the box,",
        "# not a per-volume coefficient: ep3 profits are not proportional to volume.",
        "# Run these instances with --value-mode flat so that value = coefficient.",
        "# Under the default --value-mode volume the objective would be p * volume,",
        "# which is NOT the ep3 objective. The solvers read the machine-readable",
        "# declaration below and warn if run with the wrong mode.",
        "#! value-mode: flat",
        "#",
        f"# Boxes: {total_boxes} ({len(types)} distinct types)",
        f"# Total profit if all packed: {total_profit}",
        f"# Total box volume: {total_vol}   Container volume: {cap}"
        f"   (ratio {total_vol / cap:.3f})",
        "",
        "BOXES",
    ]

    for tid, (dims, p, count) in enumerate(types, 1):
        l, w, h = dims
        lines.append(f"{tid} {count} {l} {w} {h} {format_coef(p)}")

    lines += ["", "CONTAINERS", f"1 1 {W} {H} {D}", ""]
    dst.write_text("\n".join(lines))
    return total_boxes, len(types), total_profit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path,
                    default=Path("/Users/marvincaspar/Downloads/kp2d3d-data/ep3"))
    ap.add_argument("--dst", type=Path,
                    default=Path("/Users/marvincaspar/3DMHKP/Egeblad"))
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.3kp"))
    if not files:
        raise SystemExit(f"no .3kp files in {args.src}")

    for f in files:
        out = args.dst / (f.stem + ".txt")
        n, t, p = convert(f, out)
        print(f"{f.name:24s} -> {out.name:24s} {n:3d} boxes, {t:3d} types, profit {p}")
    print(f"\n{len(files)} instances written to {args.dst}")


if __name__ == "__main__":
    main()
