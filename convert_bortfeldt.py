#!/usr/bin/env python3
"""Convert the Bortfeldt/problemNNx.txt files (destination-restricted Ivancic
et al. instances, see BischoffRatcliff-style header comments in those files)
into 3DMHKP-SA.py's BOXES/CONTAINERS text format.

Why a conversion is needed at all: 3DMHKP-SA.py (and 3DMHKPup.py,
3DMHKPup-SA.py, 3DMHKPst.py - all four solver scripts share the same parser)
read what their own docstrings call "Mohanty-format":

    BOXES      rows: type_id count length width height value_coefficient
    CONTAINERS rows: type_id count length width height

The Bortfeldt/ files use a different, older format instead:

    BOXES      rows: type_id count length width height destination
    CONTAINERS rows: type_id length width height cost

Two real modelling relaxations follow from bridging that gap, and this is a
DOCUMENTED, NOT free, conversion - exactly the same spirit as convert_br.py's
own CAUTION about dropped orientation constraints:

  1. Destination restriction DROPPED. Bortfeldt's extension (Sec. 6/7)
     requires every container to hold boxes of only ONE box type ("Kisten
     eines Kistentyps") at a time. None of the four solver scripts have any
     notion of a destination/compartment constraint, so this is not enforced;
     the destination column is read and then discarded. Results on these
     converted files are therefore a relaxation of Bortfeldt's actual
     destination-restricted problem, not a reproduction of it.

  2. Container COST dropped in favour of a fixed COUNT of 1 per listed
     container type. Bortfeldt's problem is a MULTIPLE container loading
     problem: minimise total container cost, containers available in
     unlimited count per type, every box must be packed. That objective has
     no equivalent in the solvers here, which all maximise packed value
     subject to a FIXED container supply. Giving each listed container type
     a count of 1 turns every converted instance into a bounded 3D knapsack:
     "pack as much value as fits in exactly these containers" - the same
     framing already used throughout this project for BR0-BR18, the ep3
     instances, and the existing Ivancic/-folder problems. It is a
     structurally different problem from Bortfeldt's cost-minimisation one,
     so packed-volume results from these files are NOT comparable to the
     container-cost or volume-utilisation figures Bortfeldt (2000) Table 1
     reports.

value_coefficient is set to 1 for every box type, and the converted files
carry a "#! value-mode: volume" directive, so v_i = volume_i under
--value-mode volume - the same convention as the BR/ep3 conversions.

Usage:
    python3 convert_bortfeldt.py                  # writes Bortfeldt_converted/
    python3 convert_bortfeldt.py --source-dir X --out-dir Y
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "Bortfeldt"
OUT_DIR = ROOT / "Bortfeldt_converted"

HEADER = """\
# Bortfeldt (2000), OR Spektrum 22:239-261 - destination-restricted Ivancic
# et al. (1989) benchmark {stem}, converted from Bortfeldt/{stem}.txt
# into 3DMHKP-SA.py's BOXES/CONTAINERS format.
#
# TWO RELAXATIONS versus the source file - see convert_bortfeldt.py for the
# full rationale:
#   1. Destination restriction DROPPED (not representable by this solver's
#      instance format; the source's per-box-type destination column is
#      discarded).
#   2. Container COST replaced by a fixed COUNT of 1 per listed container
#      type - this turns Bortfeldt's cost-minimising multiple-container
#      problem (unlimited containers per type, pack every box, minimise
#      cost) into a bounded 3D knapsack (fixed containers, maximise packed
#      value) - the same framing used for BR0-BR18 and ep3 elsewhere in this
#      project. Results here are NOT comparable to Bortfeldt's published
#      cost/volume-utilisation figures.
#
# value_coefficient = 1 for every box type: v_i = volume_i under
# --value-mode volume (the source data's "cost" column, being a per-CONTAINER
# figure, has no bearing on box value).
#! value-mode: volume
#
# Format: BOXES rows      = type_id count length width height value_coefficient
#         CONTAINERS rows = type_id count length width height
"""


def convert_one(path, out_dir):
    boxes = []       # (type, count, L, W, H) - destination column dropped
    containers = []  # (type, L, W, H) - cost column dropped, count fixed at 1
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
            if section == "boxes":
                if len(parts) != 6:
                    raise ValueError(f"{path}:{lineno}: expected 6 BOXES fields, "
                                      f"got {len(parts)}: {line!r}")
                t, count, l, w, h, _dest = parts
                boxes.append((int(t), int(count), int(l), int(w), int(h)))
            elif section == "containers":
                if len(parts) != 5:
                    raise ValueError(f"{path}:{lineno}: expected 5 CONTAINERS "
                                      f"fields, got {len(parts)}: {line!r}")
                t, l, w, h, _cost = parts
                containers.append((int(t), int(l), int(w), int(h)))
            else:
                raise ValueError(f"{path}:{lineno}: data outside BOXES/"
                                  f"CONTAINERS section: {line!r}")

    if not boxes:
        raise ValueError(f"{path}: no box types found")
    if not containers:
        raise ValueError(f"{path}: no container types found")

    lines = [HEADER.format(stem=path.stem), "", "BOXES"]
    for (t, count, l, w, h) in boxes:
        lines.append(f"{t} {count} {l} {w} {h} 1")
    lines.append("")
    lines.append("CONTAINERS")
    for (t, l, w, h) in containers:
        lines.append(f"{t} 1 {l} {w} {h}")

    out_path = out_dir / path.name
    out_path.write_text("\n".join(lines) + "\n")
    return out_path, len(boxes), len(containers)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if not args.source_dir.is_dir():
        parser.error(f"missing source directory: {args.source_dir}")
    args.out_dir.mkdir(exist_ok=True)

    files = sorted(args.source_dir.glob("problem*.txt"),
                    key=lambda p: (int(re.match(r"problem(\d+)", p.stem).group(1)),
                                    p.stem))
    if not files:
        parser.error(f"no problem*.txt instances in {args.source_dir}")

    n_ok = 0
    for path in files:
        try:
            out_path, nb, nc = convert_one(path, args.out_dir)
        except ValueError as exc:
            print(f"SKIP {path.name}: {exc}")
            continue
        print(f"{path.name} -> {out_path.name}  ({nb} box types, "
              f"{nc} container types)")
        n_ok += 1

    print(f"\nconverted {n_ok}/{len(files)} instances into {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
