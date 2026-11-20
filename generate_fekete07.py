#!/usr/bin/env python3
"""Generate new OKP-3 instances following Fekete et al. (2007), Section 6.2.

These are new random draws, not reproductions of the authors' benchmark files.
"""

import argparse
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "Fekete07_generated"
SETTINGS = ((20, 1), (30, 1), (40, 1), (20, 3), (20, 4))
WEIGHTS = {"I": (20, 20, 20, 20, 20),
           "II": (15, 15, 15, 15, 40),
           "III": (10, 10, 10, 10, 60)}
# Table 5: bulky in the other two axes, large, or small.
RANGES = (((1, 50), (75, 100), (75, 100)),
          ((75, 100), (1, 50), (75, 100)),
          ((75, 100), (75, 100), (1, 50)),
          ((50, 100), (50, 100), (50, 100)),
          ((1, 50), (1, 50), (1, 50)))


def generate_instance(rng, instance_class, m, multiplicity):
    """Independently draw each type's class, continuous dimensions and profit."""
    rows = []
    for type_id in range(1, m + 1):
        box_class = rng.choices(range(5), weights=WEIGHTS[instance_class], k=1)[0]
        dims = tuple(math.ceil(rng.uniform(lo, hi)) for lo, hi in RANGES[box_class])
        profit = math.prod(dims) * rng.randint(1, 3)
        rows.append((type_id, multiplicity, *dims, profit))
    return rows


def render(rows, instance_class, m, multiplicity, replicate, seed):
    header = f"""# Newly generated instance inspired by Fekete, Schepers & van der Veen (2007),
# Section 6.2 and Table 5; not an original published benchmark instance.
# Class {instance_class}, m={m}, multiplicity={multiplicity}, replicate={replicate:02d}, seed={seed}.
# Container: 100 x 100 x 100. Profit = box volume * random integer in {{1,2,3}}.
# Dimensions sampled uniformly on the paper's intervals then rounded up.
# Original paper assumes fixed box orientation; 3DMHKP-SA.py allows rotations.
# Use --value-mode flat: the last BOXES field contains the full box profit.
#! value-mode: flat

BOXES
"""
    box_lines = "\n".join(" ".join(map(str, row)) for row in rows)
    return header + box_lines + "\n\nCONTAINERS\n1 1 100 100 100\n"


def generate_all(output_dir, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"fekete07-3d-{cls}-m{m}-v{v}-{rep:02d}.txt"
             for cls in WEIGHTS for m, v in SETTINGS for rep in range(1, 11)]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite {existing[0]}; choose an empty output directory")
    rng = random.Random(seed)
    for path in paths:
        _, _, cls, m_part, v_part, rep_part = path.stem.split("-")
        m, v, replicate = int(m_part[1:]), int(v_part[1:]), int(rep_part)
        path.write_text(render(generate_instance(rng, cls, m, v), cls, m, v,
                               replicate, seed), encoding="utf-8")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2007)
    args = parser.parse_args()
    try:
        paths = generate_all(args.output_dir, args.seed)
    except FileExistsError as exc:
        parser.error(str(exc))
    print(f"Generated {len(paths)} new OKP-3 instances in {args.output_dir} (seed {args.seed}).")


if __name__ == "__main__":
    main()