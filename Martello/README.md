# Martello, Pisinger & Vigo (2000) — 3D Bin Packing Problem instances

The Section 6 benchmark of:

> S. Martello, D. Pisinger, D. Vigo, "The Three-Dimensional Bin Packing Problem",
> *Operations Research* 48(2), Mar–Apr 2000, 256–267. JSTOR 223143.
> (`../Martello00.pdf`)

**1170 instances** = 9 classes × 13 sizes (n = 10, 15, 20, 25, 30, 35, 40, 45, 50,
60, 70, 80, 90) × 10 instances each.

## These instances are GENERATED, not downloaded

Section 6 opens: *"To our knowledge no test instances have been published for the
three-dimensional bin packing problem."* The authors therefore generated them, and
what they published is the **generator**, not an instance archive:

> All test instances are available on website `http://www.diku.dk/~pisinger/codes.html`

That page is still live (now `https://hjemmesider.diku.dk/~pisinger/codes.html`) and
offers `test3dbpp.c` — "A C-code which generates test instances for the 3D BPP as
described in the paper". There is no instance ZIP anywhere; the instances *are* the
generator plus its seeds.

Reproduction is exact, not approximate, because:

- `test3dbpp.c` ships its **own** `srand48x`/`lrand48x` implementations rather than
  calling the platform RNG, so output does not vary by OS, libc or architecture.
- Seeding is `srand(v + n)` for test `v = 1..10` at size `n` — fully determined by
  (class, n, test index).

Verified: regenerating a sample of 90 instances produced byte-identical files.

## Classes (Section 6)

Bin is always cubic, `W = H = D`.

| Class | Bin | Items |
|---|---|---|
| 1–5 | 100 | Martello & Vigo (1998) types; in Class *k* each item is of type *k* with prob. 60%, of each other type with prob. 10% |
| 6 | 10 | w, h, d u.r. in [1, 10] |
| 7 | 40 | w, h, d u.r. in [1, 35] |
| 8 | 100 | w, h, d u.r. in [1, 100] |
| 9 | 100 | "all-fill": items produced by *cutting* 3 bins, so the optimum is known to be exactly 3 bins |

The five item types behind Classes 1–5 (`randomtype` in `src/test3dbpp.c`):

| Type | w | h | d |
|---|---|---|---|
| 1 | u.r. [1, W/2] | u.r. [2H/3, H] | u.r. [2D/3, D] |
| 2 | u.r. [2W/3, W] | u.r. [1, H/2] | u.r. [2D/3, D] |
| 3 | u.r. [2W/3, W] | u.r. [2H/3, H] | u.r. [1, D/2] |
| 4 | u.r. [W/2, W] | u.r. [H/2, H] | u.r. [D/2, D] |
| 5 | u.r. [1, W/2] | u.r. [1, H/2] | u.r. [1, D/2] |

(Table 1 in the PDF is garbled by text extraction — these values are taken from the
authors' source, which is authoritative.)

## File format

`instances/class<CC>_n<NNN>_<VV>.3dbpp`, where `VV` = 1..10 is the test index:

```
n  W  H  D          <- item count, then bin dimensions
1  w1 h1 d1         <- item number, then its dimensions
2  w2 h2 d2
...
```

All values are integers. Items are **individually listed** (no type/count grouping),
and there is exactly one bin type, of unlimited supply.

## Validation performed

All 1170 files were checked: item count matches the header, bin dimensions match the
class spec, every dimension ≥ 1 and ≤ bin size, and Classes 6–8 respect their stated
u.r. ranges. Zero discrepancies.

Class 9 carries a self-check: total item volume must equal exactly 3 bins. All 130
Class 9 instances satisfy this, confirming the `specialbin` cutting recursion works.

## Relationship to this project's solvers

This is **3D-BPP**: minimise the number of identical bins holding all items. It is not
the multi-container / knapsack objective the solvers in this repo optimise, and the
format differs from their Mohanty `BOXES`/`CONTAINERS` layout (see
`3DMHKP-SA.py:parse_instance`). A conversion is mechanically simple — group identical
items into types with counts, emit one container type — but note the objectives differ,
so results are not comparable to the paper's tables without care.

The paper reports exact-algorithm results, not heuristic packing ratios. Table 2 gives
the number of instances (out of 10) solved to proved optimality: the benchmark gets
hard around n ≥ 40, with 740 of 1170 solved within the 1000s limit on a HP9000/C160.
Classes 4 and 6 were easiest (111/130 each), Class 7 hardest (55/130).

## Contents

- `instances/` — the 1170 generated instances.
- `src/test3dbpp.c` — the authors' original generator, unmodified.
- `src/3dbpp.c` — the authors' exact 3D-BPP branch-and-bound solver, unmodified.
- `src/readme.3dbpp` — the authors' compilation instructions.
- `src/gen3dbpp.c` — standalone writer used here. It is `test3dbpp.c` with its item
  generation, RNG and seeding kept **verbatim**; only the `main` differs, writing each
  instance to a file instead of calling `binpack3d` to solve it (and `MAXITEMS` raised
  from 201 to 2001). Rebuild and regenerate with:

  ```sh
  cc -O2 -o src/gen3dbpp src/gen3dbpp.c -lm
  ./src/gen3dbpp <n> <bindim> <class> instances    # writes all 10 tests
  ```

Source: <https://hjemmesider.diku.dk/~pisinger/codes.html> (Pisinger's code page).
The code is free for research and academic use per its header.
