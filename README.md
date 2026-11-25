# 3DMHKP

Built upon Ivancic89 and Mohanty94.

When comparing results across benchmarks, ensure the solver's value mode matches
the benchmark objective, such as profit maximization or volume occupancy.

## Egeblad 3D bounds

Submit `slurm_eg3d.sbatch` from the repository root to compute upper bounds
for the 60 Egeblad `ep3` instances, one array task per instance. It uses the
repository's `packing_bounds` full integer relaxation, original flat profits and
free orthogonal rotations. Each task allows 900 seconds of solver time and
1,200 seconds including model construction; results are saved as JSON in
`results_EG3D_bounds/`. A wall-time fallback is explicitly identified in the
result. These are upper bounds, not feasible packings or exact certificates.
The cluster Python environment must contain NumPy and SciPy; `highspy` is
optional. The `packing-bounds-1.0.0` release directory is not needed.

## CS instances

Submit `slurm_cs.sbatch` from the repository root to run one SA task per CS
instance, with 1,000 seconds of search per instance. Reports and placements
are written to `results_CS_SA/`. The batch script uses the same cluster Python
environment as the other SLURM scripts.

`CS-SA.py` enforces rotation and weight limits, load bearing, full support and
multi-drop accessibility. It uses conservative single-box support and
non-overlapping destination bands. It lexicographically minimizes unloaded-box
cost, container cost, stops and occupied length; the original paper's
objective weights are not specified in the data. The reported raw metrics
should therefore not be treated as the paper's weighted table values.