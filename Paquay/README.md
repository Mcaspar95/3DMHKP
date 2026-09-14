# Paquay, Limbourg & Schyns (2018) — 3D Multiple Bin Size Bin Packing Problem

Instances for:

> C. Paquay, S. Limbourg, M. Schyns, "A tailored two-phase constructive heuristic
> for the three-dimensional Multiple Bin Size Bin Packing Problem with
> transportation constraints", *European Journal of Operational Research* 267
> (2018) 52–64.

These are the **authors' original files**, downloaded from the ORBi repository
named in the paper (Section 6.1.2):

    https://orbi.uliege.be/handle/2268/206856   ->  data_sets.zip  (508,711 bytes)

`data_sets_orbi_original.zip` is the untouched archive as downloaded. Nothing here
was regenerated or reconstructed — the published box samples are exactly these.

## Contents

| Path | Files | Description |
|---|---|---|
| `Final_instances/` | 300 | 30 instances each for n = 10, 20, 30, …, 100 boxes (Section 6.3) |
| `training_data_sets/` | 150 | 30 instances each for n = 10, 15, 20, 25, 30 (parameter tuning, Section 6.2) |
| `uld.csv` | 1 | The 6 ULD types (paper Table 1) |

Naming is `n<N>sample<K>.csv`, N = box count, K = 0..29. Every file has exactly N
data rows (verified). Separator is `;`, no header, lengths in millimetres.

## Box file format (9 columns)

    L ; W ; H ; weight ; l+ ; w+ ; h+ ; fragile ; (reserved)

| Col | Meaning |
|---|---|
| 1–3 | Length, width, height [mm] |
| 4 | Weight [kg] |
| 5 | `l+` — may the length axis go vertical (1 = allowed) |
| 6 | `w+` — may the width axis go vertical |
| 7 | `h+` — may the height axis go vertical (always 1) |
| 8 | fragile flag — nothing may be stacked on top |
| 9 | reserved / unused (0 throughout) |

The paper states the orientation and fragility flags were *derived*, because the
real-world source data lacked them (Section 6.1.2). Both rules verify exactly
against these files across all 16,500 box rows:

* **Orientation** — boxes heavier than 50 kg are considered too awkward to turn,
  so `l+ = w+ = 0`. Checked: 0 rows violate this. `h+ = 1` everywhere, since the
  box's initial orientation is by assumption feasible.
* **Fragility** — a box with density < 0.05 kg/dm³ is fragile. Checked: the 257
  rows flagged fragile top out at density 0.0496; the 16,243 unflagged rows start
  at 0.0514. No overlap.

## ULD file format (`uld.csv`, 22 columns)

    L ; W ; H ; maxWeight ; volume ; aL ; aW ; aH ; <4 cut triples> ; volume_mm3 ; id

Columns 1–3 are the bounding box [mm]; 4 is the weight capacity [kg]; 5 the
nominal volume [m³]; 6–8 are the `αL; αW; αH` offsets of paper Table 1. Columns
9–20 hold up to four cut definitions as (denominator, numerator, constant)
triples describing the bevelled corners — a cut is absent when its triple is all
zeros. Column 21 is the exact volume in mm³, column 22 an integer id.

The six rows are, in file order: LD1, LD11, PG, PA, LD6, PM — i.e. **not** the
order printed in Table 1. Match rows by dimensions, not by position.

| IATA | L × W × H [mm] | Max cap. [kg] | Vol [m³] | Deck | Cuts |
|---|---|---|---|---|---|
| LD1 | 2337 × 1534 × 1626 | 1518 | 5.0 | lower | 1 |
| LD11 | 3175 × 1534 × 1626 | 2991 | 7.4 | lower | 0 |
| LD6 | 4064 × 1534 × 1626 | 2945 | 9.1 | lower | 2 |
| PA | 2235 × 3175 × 2997 | 5890 | 20.0 | main | 1 |
| PG | 2438 × 6058 × 2438 | 10840 | 31.1 | main | 2 |
| PM | 2438 × 3175 × 2997 | 6680 | 18.9 | main | 0 |

An unlimited supply of each type is assumed.

## Caveat on comparability

The paper's objective minimises unused space subject to transportation
constraints the classical benchmarks do not carry: ULD weight capacity, per-box
allowed orientations, fragility/stacking, load stability, the bevelled ULD
shapes, and centre-of-gravity balance. Filling rates here are therefore **not**
directly comparable to volume utilisation on the Ivancic or Bischoff–Ratcliff
sets unless those constraints are enforced too.

`load_paquay.py` in the repository root reads these files.
