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

For exact packings of the 20 `ep3-20` instances, submit
`sbatch slurm_eg3d_opt.sbatch` from the repository root. Each array task runs
the Gurobi MILP with flat profits and free rotations, using eight CPUs and up
to 23 hours 50 minutes of solver time (24 hours of Slurm wall time). Results,
including verified placements, the dual bound and `proven_optimal`, are written
to `results_EG3D_opt/` (`dual_bound` is null if no finite bound is available).
A task exits unsuccessfully if optimality was not proven; the time limit does
not guarantee a proof. The cluster environment
requires Gurobi and a valid license.

## Egeblad 2D exact optimization

Submit `sbatch slurm_eg2d_opt.sbatch` from the repository root for all 80
`ep2` instances in the paper (20 each with 30, 50, 100 and 200 rectangles).
The original-format inputs are in `Egeblad/ep2/`, mirrored from
[PackingSolver](https://github.com/fontanf/packingsolver/tree/759f8627b5e2652014644e23303a5249cb84ca05/data/rectangle_raw/egeblad2009).
The model maximizes the original item profits in one bin, permits 90-degree
rotations and does not impose guillotine cuts. Each task uses Gurobi, eight
CPUs, 16 GB of memory and a 23-hour-50-minute solver limit within 24 hours of
Slurm time. The cluster environment needs Gurobi and a valid license.

Results and verified placements are saved in `results_EG2D_opt/`. Check
`proven_optimal`: a time limit does not guarantee a proof, and an unproven
task exits unsuccessfully after writing its incumbent and dual bound (null if
unavailable). Larger instances may need substantially more time or memory.

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