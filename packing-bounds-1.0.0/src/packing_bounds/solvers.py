"""HiGHS/SciPy optimization and valid upper-bound handling."""
import math
import time
import numpy as np

def _fractional_volume(C, items):
    """Exact rational fractional-knapsack bound, converted outwards to float."""
    from fractions import Fraction
    remaining = math.prod(C); bound = Fraction(0)
    ranked = sorted(((math.prod(d), Fraction(p)) for d,p in items),
                    key=lambda vp: vp[1]/vp[0], reverse=True)
    for volume, profit in ranked:
        used = min(remaining, volume)
        bound += profit * Fraction(used, volume)
        remaining -= used
        if not remaining: break
    return math.nextafter(float(bound), math.inf)


def _round_upper(upper, items):
    if all(float(profit).is_integer() for _, profit in items):
        return math.floor(upper + max(1e-6, 32 * math.ulp(upper)))
    return upper


def _integer_solve(p, A, limit, prefer_highspy, milp, Bounds, LinearConstraint):
    """Use standalone HiGHS when installed; preserve a SciPy-only installation.

    This function runs in the calling process; it does not spawn children.
    Called on the full model for legacy methods and on restricted row sets
    by the extended solver. The latter validates incumbents against every row.
    """
    from types import SimpleNamespace
    highspy = None
    if prefer_highspy:
        try:
            import highspy
        except ImportError:
            pass
    if highspy is None:
        result = milp(-p, integrality=np.ones(len(p)),
                      bounds=Bounds(np.zeros(len(p)), np.ones(len(p))),
                      constraints=LinearConstraint(A, -np.inf, 1),
                      options={'time_limit': limit, 'mip_rel_gap': 0})
        return result, 'SciPy / HiGHS'
    h = highspy.Highs()
    def checked(status):
        if status == highspy.HighsStatus.kError:
            raise RuntimeError('HiGHS rejected a model or solver option')
    for key, value in [('output_flag', False), ('threads', 1),
                       ('time_limit', float(limit)), ('mip_rel_gap', 0.)]:
        checked(h.setOptionValue(key, value))
    n = len(p)
    checked(h.addCols(n, -p, np.zeros(n), np.ones(n), 0,
                     np.zeros(n+1, dtype=np.int32), np.array([], dtype=np.int32),
                     np.array([], dtype=float)))
    checked(h.addRows(A.shape[0], np.full(A.shape[0], -np.inf), np.ones(A.shape[0]),
                     A.nnz, A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data))
    checked(h.changeColsIntegrality(n, np.arange(n, dtype=np.int32),
                                   np.ones(n, dtype=np.uint8)))
    checked(h.run())
    status = h.getModelStatus()
    code = 0 if status == highspy.HighsModelStatus.kOptimal else (
        1 if status == highspy.HighsModelStatus.kTimeLimit else 4)
    info, sol = h.getInfo(), h.getSolution()
    result = SimpleNamespace(status=code, message=h.modelStatusToString(status),
        mip_dual_bound=info.mip_dual_bound,
        x=np.asarray(sol.col_value) if sol.value_valid else None,
        fun=info.objective_function_value if sol.value_valid else None)
    return result, 'HiGHS ' + h.version() + ' (highspy)'


def _extended_integer_solve(p, A, raw_A, b, limit, fallback,
                            linprog, milp, Bounds, LinearConstraint):
    """Separate violated rows while retaining a global upper bound.

    A restricted optimum is a full-model optimum only if it satisfies every
    row. Restricted incumbents violating omitted rows are never reported as
    full-model values. All solves share one time budget in the GUI worker.
    """
    from types import SimpleNamespace
    started = time.perf_counter()
    upper = fallback
    upper_source = 'fractional volume'
    best_value = 0.
    chosen = {0}
    # Includes shared-orientation groups (at most six orientations in 3D).
    for r in np.flatnonzero(np.diff(A.indptr) <= 6):
        if np.all(A.data[A.indptr[r]:A.indptr[r+1]] == 1):
            chosen.add(int(r))
    lp = linprog(-p, A_ub=A, b_ub=np.ones(A.shape[0]), bounds=(0, 1),
                 method='highs', options={'time_limit': max(.001, min(5., limit*.25))})
    if lp.success:
        lam = np.maximum(0, -lp.ineqlin.marginals)
        slack = np.maximum(0, p-A.T@lam)
        lp_upper = math.nextafter(float(lam.sum()+slack.sum()), math.inf)
        if lp_upper <= upper:
            upper, upper_source = lp_upper, 'LP dual bound' 
        chosen.update(map(int, np.flatnonzero(lam > 1e-9)))
    rounds = 0
    code = 1
    backend = 'SciPy / HiGHS (LP initialization)'
    while time.perf_counter()-started < limit:
        rows = np.array(sorted(chosen))
        remaining = max(.001, limit-(time.perf_counter()-started))
        result, backend = _integer_solve(p, A[rows], remaining, True,
                                         milp, Bounds, LinearConstraint)
        rounds += 1
        if result.status not in (0, 1):
            raise RuntimeError(f'Bound solver failed: {result.message}')
        dual = getattr(result, 'mip_dual_bound', None)
        if dual is not None and math.isfinite(dual):
            if -float(dual) <= upper:
                upper, upper_source = -float(dual), 'MIP global dual bound' 
        if result.x is None:
            break
        x = np.rint(result.x)
        if np.any(np.abs(result.x-x) > 1e-5) or np.any(x<0) or np.any(x>1):
            break
        violation = np.asarray(raw_A@x).ravel()-b
        bad = np.flatnonzero(violation > 1e-6)
        if not len(bad):
            best_value = max(best_value, float(p@x))
            if result.status == 0:
                code = 0
            break
        missing = [int(r) for r in bad if r not in chosen]
        if not missing:
            raise RuntimeError('Integer solver returned a solution violating its active constraints')
        chosen.update(sorted(missing, key=lambda r: violation[r]/b[r], reverse=True)[:64])
    result = SimpleNamespace(status=code,
        message='Optimal full relaxation' if code==0 else 'Shared solver time limit reached',
        mip_dual_bound=-upper, fun=-best_value, bound_source=upper_source,
        row_generation_rounds=rounds, active_rows=len(chosen))
    return result, backend

