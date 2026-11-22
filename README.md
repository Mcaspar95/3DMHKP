# 3DMHKP
Build upon Ivancic89 and Mohanty94

Since your ep3/Egeblad instances use a profit-maximization objective and BR/Mohanty94 instances use volume-occupancy, keep in mind you'd run 3DMHKP in a volume-max mode (or treat volume as the value coefficient) to make results comparable to the papers using this benchmark.

## BR0–BR15 upper bounds

`bound_br15.py` applies the version 1.0.0 `orthogonal-packing-bounds` source
vendored in this project's `packing_bounds/` directory to the converted
BR0–BR15 instances (default BR15). Copy `packing_bounds/` along with the project
to the cluster; no separate installation of that package is required. Python
3.10 or later, NumPy >=1.24,<3 and SciPy >=1.11,<2 must be available in the
chosen Python environment. The optional `highspy` backend is not required.
For example, `python bound_br15.py --stage baseline --time-limit 5
--wall-time-limit 15 --skip-existing` processes all instances; use
`--instances BR15-001` for one instance. The default stage is `full` and each
stage's wall limit includes model construction. Results are saved individually
in `results_bounds_BR15/`; `--skip-existing` does not recompute existing files,
even if the requested time limits have changed.

Pass `--class BR0` through `--class BR15` to select a class. The runner uses
`BischoffRatcliff/<class>/`, compares any available reports in
`results_3DMHKP_<class>/`, and saves results in `results_bounds_<class>/`.
The best independently verified packing is the lower bound; if no matching
report exists the bound is still computed and the lower bound is null.

For a full-stage BR0–BR15 run on SLURM, submit `sbatch slurm_bounds_br.sbatch`
from the project directory. Install NumPy and SciPy in `/home/mcaspar/3DPCPTP`
first, or override `VENV_DIR` when
submitting. It runs **1,600 array tasks**, mapping task 0 to BR0-001, 99 to
BR0-100, 100 to BR1-001, and 1599 to BR15-100. At most ten run concurrently,
and each requests 8 GB and four hours. The default
limits are 3600 seconds for the solver and 12600 seconds for model construction
and solving combined. Adjust `SOLVER_SECONDS`, `WALL_SECONDS`, and the SLURM
time/memory request together as needed. `INSTANCE_CLASS` is no longer used:
unset it in the submission environment if it was previously set. The array
requires the cluster's `MaxArraySize` to allow index 1599. If it does not,
submit chunks using `--array` and `ARRAY_OFFSET`: for example, if index 799
is allowed, submit `sbatch --array=0-799%10 --export=ALL,ARRAY_OFFSET=0
slurm_bounds_br.sbatch` and `sbatch --array=0-799%10
--export=ALL,ARRAY_OFFSET=800 slurm_bounds_br.sbatch`. Each array has its own
ten-task concurrency limit. Smaller limits require smaller chunks. To run
only BR15, use `sbatch --array=0-99%10 --export=ALL,ARRAY_OFFSET=1500
slurm_bounds_br.sbatch`.

Each submission writes to `results_bounds_<class>_full_<array-job-id>/`
and logs to `packing-bounds-<array-job-id>-<task-id>.out`
and `.err`. Override `OUTPUT_DIR` to use a different shared directory; doing so
may overwrite existing results for the same instances. A wall timeout returns
the volume fallback, not a full-stage result. Numerical inconsistencies cause
the affected task to fail; check logs and SLURM job states rather than treating
missing files as valid bounds. The bound is numerical, not an exact certificate.

The runner independently checks the placements in the existing SA reports and
compares the best verified packed volume against the numerical upper bound.
`relative_gap` is `(upper - lower) / upper`. The relaxation's `model_value`
is **not** a feasible packing. An `Optimal` status refers only to the relaxation;
a wall-time fallback is explicitly labeled and uses the volume bound. These are
floating-point numerical bounds, not exact certificates.

The converted instances omit the original per-type vertical orientation
restrictions, so both the saved SA placements and these bounds concern the
free-rotation relaxation. In particular, those placements are not guaranteed
feasible for the original orientation-restricted benchmark.

For the existing 100 BR15 reports, the baseline IP relaxation completed for
all instances with a 5-second solver limit and 15-second wall limit per
instance. The verified packings average 84.735% container utilization; the
upper bounds average 99.445%, with a mean relative gap of 14.790%. All 100
baseline bounds equal the simple minimum of container volume and total box
volume, so this run proves none of the packings optimal. A full-stage trial
on BR15-001 hit its 15-second wall limit and returned the volume fallback.

For BR0, the same baseline limits give 100 numerical relaxation optima,
averaging 99.617% container utilization, versus 87.096% for the best
verified packing across the 900s and 3600s SA reports; the mean relative gap
is 12.569%. All baseline bounds equal the simple volume bound. Full-stage
trials can tighten it, but BR0-012 exposed an inconsistent numerical primal
and upper bound, which the library rejected. Failed instances are reported,
do not produce a result file, and cause the batch to return a nonzero exit
status. Full-stage `Optimal` results remain floating-point numerical bounds,
not exact certificates, and require particular caution given this failure.

## Exact line-pattern bounds for homogeneous BR0

`bound_br0_patterns.py` computes a separate free-rotation bound for the
converted BR0 instances, which each contain one box type. Run
`python bound_br0_patterns.py` for all instances, or append `--instances
BR0-098` to select an instance. It requires the same environment as
`bound_br15.py` (including SciPy). Results and exact certificates are stored
in `results_bounds_BR0_patterns/`; rerunning overwrites successful results.
Each run independently verifies the best available SA placement. An error is
reported and the batch exits nonzero; an older output file is not deleted if
the corresponding rerun fails.

For SLURM, submit `sbatch slurm_bounds_br0_patterns.sbatch` from the project
directory. This runs all 100 BR0 instances as an array, with at most ten tasks
at once, and independently verifies each resulting certificate. It requests
one CPU, 2 GB and 30 minutes per task. The allocation is a maximum, not a
search budget: increasing it alone does not improve the bound. The default
environment is `/home/mcaspar/3DPCPTP`; override `VENV_DIR` on submission if
needed and install the dependencies described above. Results go to
`results_bounds_BR0_patterns_<array-job-id>/` by default, with separate
`BR0-pattern-<array-job-id>-<task-id>.out` and `.err` logs. Override
`OUTPUT_DIR` only when intentionally sharing a results directory; old files
can remain if a task fails, so check the job status. If SA reports are absent,
the upper bound is still computed but no lower bound is reported. The verifier
checks the certificate, not the source instance or lower-bound placement.
This job does **not** cover heterogeneous BR1–BR18: the current line-pattern
implementation cannot process those classes.

For each coordinate axis, a line intersects a set of boxes whose edge lengths
fit within that container dimension. For a nonnegative vector of integer
weights, enumerate all feasible line patterns to find their maximum weight.
Integrating this line inequality over the perpendicular cross-section yields a
valid inequality `A x <= b` on the counts `x` of box orientations. Floating
point convex hulls and an LP suggest weights and dual multipliers only; each
pattern maximum is recomputed exactly. For nonnegative rational multipliers
`lambda` and `0 <= x[j] <= N`, the expression
`b^T lambda + N * sum(max(0, 1 - (A^T lambda)[j]))` is an upper bound on
the box count, even if the numerical LP dual is not feasible. The expression
is evaluated with exact rational arithmetic before flooring; the final bound
also uses the exact supply and volume limits.

Use `python verify_br0_pattern_certificate.py results_bounds_BR0_patterns/*.json`
to independently check the certificates with a different one-dimensional
knapsack dynamic program and exact arithmetic. The verifier checks the data
in the JSON files, not their correspondence to the source instance or the
saved lower-bound packing.

On all 100 BR0 instances, this bound averaged **96.838%** container
utilization versus **99.617%** for the earlier baseline numerical bounds;
the mean gap to the verified packings was **10.033%**. It improved on the
integer-rounded volume/supply bound in 99 cases. BR0-098 has a verified
packing of 132 boxes (29,243,280 volume), matching the exact upper bound;
this proves optimality for that converted free-rotation instance. BR0-012's
exact upper bound is 29,457,600, despite the separate full-stage numerical
failure. Of the 99 available full-stage numerical results, the new bound is
lower in 8, equal in 17, and higher in 74; two of the eight improvements are
over timeout fallbacks. Those numerical full-stage results are not exact
certificates, so the two approaches should not be treated as equivalent proofs.