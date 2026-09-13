#!/usr/bin/env python3
"""Convert the kcliu2/CLP-Datasets BR0-BR18 JSON instances into 3DMHKP's
BOXES/CONTAINERS text format (the same format used for the Egeblad ep2/ep3
instances).

Source data: https://github.com/kcliu2/CLP-Datasets  (BR/BR{k}/{n}.json)
  BR1-BR7:       Bischoff, E.E. & Ratcliff, M.S.W. (1995), "Issues in the
                 Development of Approaches to Container Loading",
                 Omega 23(4):377-390.
  BR0, BR8-BR15: Davies, A.P. & Bischoff, E.E. (1999), "Weight distribution
                 considerations in container loading", EJOR 114(3):509-527.
  BR16-BR18:     Liu, C., Smith-Miles, K., Wauters, T. & Costa, A.M. (2025),
                 "A block-building constraint programming model for the
                 container loading problem", Computers & Operations Research
                 182:107111.

Each source instance has exactly one container ("Objects": 1 entry) and a
list of box types ("Items") whose declared Value already equals the box's
volume (Length*Height*Depth) -- i.e. the CLP objective of maximizing packed
volume is encoded directly as knapsack profit. We carry that straight
through: value_coefficient = 1 for every box type, run under
--value-mode volume.

CAUTION: the source JSON also carries per-axis rotation-permission flags
(C1_Length, C1_Height, C1_Depth) restricting which axis may be vertical.
3DMHKP's BOXES row format has no field for that and the solver always
tries all 6 axis-aligned orientations, so these converted instances are the
FREE-ROTATION relaxation of the true (partially orientation-restricted) BR
benchmark. This is flagged in each output file's header; results on these
files are not directly comparable to published BR results that enforce the
orientation restriction.
"""
import json
import os
import sys
from pathlib import Path

SOURCE_NOTE = {
    "BR0": "Davies, A.P. & Bischoff, E.E. (1999), EJOR 114(3):509-527. (BR0: homogeneous)",
    **{f"BR{k}": "Bischoff, E.E. & Ratcliff, M.S.W. (1995), Omega 23(4):377-390."
       for k in range(1, 8)},
    **{f"BR{k}": "Davies, A.P. & Bischoff, E.E. (1999), EJOR 114(3):509-527."
       for k in range(8, 16)},
    **{f"BR{k}": "Liu, C., Smith-Miles, K., Wauters, T. & Costa, A.M. (2025), "
                 "Computers & Operations Research 182:107111."
       for k in range(16, 19)},
}

HEADER_TEMPLATE = """\
# Bischoff & Ratcliff container-loading instance {cls}-{n}, converted from
# BR/{cls}/{n}.json (kcliu2/CLP-Datasets on GitHub).
# Source: {source}
# Dataset repository: https://github.com/kcliu2/CLP-Datasets
#
# Single-container 3D knapsack: one container, boxes carry explicit profits.
# Format: BOXES rows      = type_id count length width height value_coefficient
#         CONTAINERS rows = type_id count length width height
#
# The value_coefficient column is 1 for every box type: the original CLP
# objective (maximize packed volume) is already encoded as profit = volume
# in the source data (Value == Length*Height*Depth for every box type here).
# Run these instances under the default --value-mode volume.
#! value-mode: volume
#
# CAUTION - orientation constraints dropped in this conversion: the source
# data restricts, per box type, which of its 3 axes may be placed vertical
# (fields C1_Length/C1_Height/C1_Depth in the source JSON -- 1 = that axis
# may be vertical, 0 = may not). This solver's BOXES row format has no field
# to carry that restriction, and the solver always tries all 6 axis-aligned
# orientations for every box. These converted instances therefore solve the
# FREE-ROTATION relaxation of the true BR benchmark (partial orientation
# restriction). Expect equal-or-better packed volume than published results
# that enforce the restriction; the two are not directly comparable.
#
# Boxes: {total_boxes} ({n_types} distinct types)
# Total profit if all packed: {total_profit}
# Total box volume: {total_vol}   Container volume: {cont_vol}   (ratio {ratio:.3f})

BOXES
{box_lines}
CONTAINERS
{container_lines}
"""


def convert_one(src_path: Path, cls: str, n: str, out_path: Path):
    data = json.loads(src_path.read_text())
    assert len(data["Objects"]) == 1, f"{src_path}: expected exactly 1 container"
    obj = data["Objects"][0]
    L, H, D = obj["Length"], obj["Height"], obj["Depth"]
    cont_vol = L * H * D

    box_lines = []
    total_boxes = 0
    total_profit = 0
    total_vol = 0
    for i, it in enumerate(data["Items"], start=1):
        l, h, d = it["Length"], it["Height"], it["Depth"]
        count = it["Demand"]
        assert it["DemandMax"] is None, f"{src_path}: unexpected DemandMax on item {i}"
        vol = l * h * d
        if vol != it["Value"]:
            raise ValueError(f"{src_path}: item {i} Value {it['Value']} != volume {vol}")
        box_lines.append(f"{i} {count} {l} {d} {h} 1")
        total_boxes += count
        total_profit += count * vol
        total_vol += count * vol

    container_lines = f"1 1 {L} {D} {H}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(HEADER_TEMPLATE.format(
        cls=cls, n=n,
        source=SOURCE_NOTE[cls],
        total_boxes=total_boxes,
        n_types=len(data["Items"]),
        total_profit=total_profit,
        total_vol=total_vol,
        cont_vol=cont_vol,
        ratio=total_vol / cont_vol if cont_vol else float("nan"),
        box_lines="\n".join(box_lines),
        container_lines=container_lines,
    ))


def main():
    src_root = Path(sys.argv[1])
    out_root = Path(sys.argv[2])
    classes = [f"BR{k}" for k in range(0, 19)]

    n_written = 0
    for cls in classes:
        src_dir = src_root / cls
        json_files = sorted(src_dir.glob("*.json"), key=lambda p: int(p.stem))
        if not json_files:
            print(f"WARNING: no json files found for {cls} in {src_dir}", file=sys.stderr)
            continue
        for jf in json_files:
            n = jf.stem
            out_path = out_root / cls / f"{cls}-{int(n):03d}.txt"
            convert_one(jf, cls, n, out_path)
            n_written += 1
        print(f"{cls}: {len(json_files)} instances -> {out_root / cls}")

    print(f"Done. {n_written} instance files written to {out_root}")


if __name__ == "__main__":
    main()
