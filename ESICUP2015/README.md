# ESICUP Challenge 2015 — 3D Multi-Container Loading Problem (3D-MCLP)

56 real-world instances supplied by **Renault** for the 2014/2015 ESICUP challenge
(EURO Special Interest Group on Cutting and Packing), organised with Université de
Bordeaux. The objective is to minimise the total volume of shipped containers.

## Contents

| Path | Instances | Items (min–max) | Total items |
|---|---|---|---|
| `InstancesA/` | 20 | 117–2447 | 9 845 |
| `InstancesB/` | 19 | 162–1378 | 10 301 |
| `InstancesX/` | 17 | 159–2597 | 12 756 |
| **Total** | **56** | | **32 902** |

Set A was the initial release (27 May 2014), B the qualification set (1 Dec 2014),
and X the final-round set (20 Feb 2015).

**Note on duplicate names:** 10 instance names occur in two sets each (e.g.
`ALG_RIR_PEX_container` in both B and X). These are *different* instances — same
shipping route sampled at a different date, with different item sets. Always key an
instance by `set + name`, never by name alone.

## Instance format

Each instance is a directory with three files (full spec in `doc/input_output.pdf`):

- `input_bin.csv` — available container/truck types: `bin type;length;width;height;maximal weight`
  (mm and kg), e.g. `CA270;13400;2440;2700;24000`.
- `input_items.csv` — items to load: `item code;length;width;height;weight;Package material;Orientation constrained;Product`.
  `Product` is the Renault part number; a non-empty `Orientation constrained` marks
  the "this side up" restriction.
- `parameters.txt` — 8 instance parameters, including `max_number_of_rows_in_a_layer`,
  `max_number_of_items_in_a_row`, `maximum_density_allowed_kg` and
  `maximum_weigth_above_item_kg` (the typo is in the original data).

Items are grouped into stacks, stacks into rows, rows into layers — the hierarchy the
solution format and the official checker expect.

## Provenance

The original challenge site, `http://challenge-esicup-2015.org/`, is **dead** — the
domain no longer resolves, and the Internet Archive holds the HTML pages but never
crawled the instance ZIPs (all return 404). The instances were recovered from the
CODeS / KU Leuven benchmark mirror, which hosts this exact challenge dataset:

- Instances: <https://benchmark.gent.cs.kuleuven.be/mclp/media/instances/instances.zip>
- Benchmark site: <https://benchmark.gent.cs.kuleuven.be/mclp/>

Verified genuine: instance names match the archived official results tables exactly,
and the data carries real Renault container types (CA270/CA29R) and part numbers.

### `doc/`
- `input_output.pdf` — official instance/solution format specification.
- `results_instances_final.html` — per-instance objective values for all finalist
  teams on set X (recovered from the Internet Archive). Useful as best-known values.
- `rankings_instances_A_B_final.html`, `rankings_instances_X_final.html` — final team
  rankings with affiliations.
- `mclp_website.bib` — citation for the benchmark site.

The original French problem statement (`modele_renault.pdf`, Clautiaux et al. 2014)
could **not** be recovered — it was never archived and both hosting sites link to the
dead domain.

### `tools/`
- `solver.zip` — `esicup.jar`, the CODeS reference solver.
- `gui.zip` — solution visualiser.

The official **solution validator is not included**: the benchmark site advertises
`validator.zip` but the file 404s on their server. Request it from CODeS if you need
to verify solutions.

## Best-known results

Challenge winner was team_17 (Olivier Briant & Denis Naddef, Grenoble-INP), with
16 376 m³ on set B and 18 151 m³ on set X. Per-instance values are in
`doc/results_instances_final.html`.

Key reference: Toffolo, Esprit, Wauters & Vanden Berghe (2017), *A two-dimensional
heuristic decomposition approach to a three-dimensional multiple container loading
problem*, EJOR 257(2), 526–538. <https://doi.org/10.1016/j.ejor.2016.07.033>
