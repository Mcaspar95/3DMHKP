"""Public JSON-compatible API for 2D/3D knapsack bounds."""
import math
import time
from numbers import Real
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import diags
from .models import STAGES, construct
from .solvers import _fractional_volume, _round_upper, _integer_solve, _extended_integer_solve


def prepare(instance):
    """Validate JSON input, expand copies, exclude individually unfit items.

    Input: {container: [..], items: [{dimensions: [..], profit: number,
    copies: optional positive integer, id: optional string}],
    allow_rotations: optional boolean, name: optional string}.
    Positive integral lengths and nonnegative finite profits are required.
    """
    if not isinstance(instance,dict):raise ValueError('Instance must be an object')
    def dims(values,dimension=None):
        if not isinstance(values,(list,tuple)) or len(values) not in (2,3):
            raise ValueError('Dimensions must contain two or three positive integers')
        if dimension is not None and len(values)!=dimension:
            raise ValueError('Item and container dimension counts must match')
        if any(isinstance(x,bool) or not isinstance(x,Real) or not math.isfinite(x)
               or x<=0 or not float(x).is_integer() or x>1_000_000 for x in values):
            raise ValueError('Dimensions must be positive integers no larger than 1000000')
        values=tuple(map(int,values))
        if math.prod(values)>2**50:raise ValueError('Volume exceeds supported numeric range (2**50)')
        return values
    C=dims(instance.get('container'))
    rotate=instance.get('allow_rotations',True)
    if not isinstance(rotate,bool):raise ValueError('allow_rotations must be boolean')
    entries=instance.get('items')
    if not isinstance(entries,list):raise ValueError('items must be a list')
    items=[];ids=[];excluded=0
    for index,entry in enumerate(entries):
        if not isinstance(entry,dict):raise ValueError('Each item must be an object')
        a=dims(entry.get('dimensions'),len(C));profit=entry.get('profit')
        if isinstance(profit,bool) or not isinstance(profit,Real) or not math.isfinite(profit) or profit<0:
            raise ValueError('Item profit must be finite and nonnegative')
        copies=entry.get('copies',1)
        if isinstance(copies,bool) or not isinstance(copies,int) or copies<1:
            raise ValueError('copies must be a positive integer')
        fits=all(s<=c for s,c in zip(sorted(a),sorted(C))) if rotate else all(s<=c for s,c in zip(a,C))
        if not fits:excluded+=copies;continue
        if copies*float(profit)>2**50:raise ValueError('Total profit exceeds supported numeric range')
        items.extend([(a,float(profit))]*copies)
        ids.extend([{'input_index':index,'id':str(entry.get('id',index)),'copy':j+1} for j in range(copies)])
    if math.fsum(p for _,p in items)>2**50:raise ValueError('Total profit exceeds supported numeric range')
    return dict(container=C,items=items,item_ids=ids,allow_rotations=rotate,
                excluded=excluded,name=str(instance.get('name','Instance')))


def build_model(instance,stage='full'):
    """Return a Model containing a sparse matrix and original RHS capacities."""
    return construct(prepare(instance),stage)


def solve(instance,stage='full',mode='ip',time_limit=30,progress=None):
    """Compute a profit upper bound. time_limit is solver time, excluding build.

    Native solver time limits are advisory. solve_isolated optionally adds a
    hard total wall-time guard. Results are floating-point numerical bounds,
    not exact rational certificates or feasible geometric packings.
    """
    if stage not in STAGES:raise ValueError(f'Unknown stage: {stage}')
    if mode not in ('ip','lp'):raise ValueError('mode must be ip or lp')
    if not isinstance(time_limit,Real) or not math.isfinite(time_limit) or time_limit<=0:
        raise ValueError('time_limit must be finite and positive')
    started=time.perf_counter();payload=prepare(instance)
    model=construct(payload,stage,progress)
    base=dict(name=payload['name'],dimension=len(payload['container']),stage=stage,mode=mode,
              rotations=payload['allow_rotations'],excluded=payload['excluded'],
              rows=model.A.shape[0],variables=len(model.profits),metadata=model.metadata,
              construction_seconds=time.perf_counter()-started,solver_time_limit=time_limit)
    if not len(model.profits):
        return dict(base,upper_bound=0.,model_value=0.,status='Optimal',seconds=time.perf_counter()-started,
                    solver='none',bound_source='empty instance')
    p,A,b=model.profits,model.A,model.b
    scaled=(diags(1/b)@A).tocsr()
    fallback=_fractional_volume(payload['container'],payload['items']);upper=fallback
    source='fractional volume';backend='SciPy / HiGHS'
    if progress:progress({'phase':'solving','detail':stage})
    if mode=='ip':
        if stage in ('floor_mixed','rounding_mixed','data_mixed','full'):
            r,backend=_extended_integer_solve(p,scaled,A,b,time_limit,fallback,linprog,milp,Bounds,LinearConstraint)
            base.update(row_generation_rounds=r.row_generation_rounds,active_rows=r.active_rows)
        else:
            r,backend=_integer_solve(p,scaled,time_limit,False,milp,Bounds,LinearConstraint)
        dual=getattr(r,'mip_dual_bound',None)
        if dual is not None and math.isfinite(dual) and -float(dual)<=upper:
            upper=-float(dual);source=getattr(r,'bound_source','MIP global dual bound')
        upper=_round_upper(upper,payload['items'])
    else:
        r=linprog(-p,A_ub=scaled,b_ub=np.ones(len(b)),bounds=(0,1),method='highs',
                  options={'time_limit':float(time_limit)})
        if r.success:
            lam=np.maximum(0,-r.ineqlin.marginals);slack=np.maximum(0,p-scaled.T@lam)
            dual=math.nextafter(float(lam.sum()+slack.sum()),math.inf)
            if dual<=upper:upper=dual;source='LP dual bound'
    if r.status not in (0,1):raise RuntimeError(f'Bound solver failed: {r.message}')
    value=None if r.fun is None else -float(r.fun)
    if value is not None and upper<value-max(1e-5,64*math.ulp(upper)):
        raise RuntimeError('Solver returned inconsistent primal and upper bounds')
    return dict(base,upper_bound=float(upper),model_value=value,
                status='Optimal' if r.status==0 else 'Time limit (upper bound)',
                seconds=time.perf_counter()-started,solver=backend,bound_source=source,
                solver_message=str(r.message))


def solve_all(instance,mode='ip',time_limit=30):
    """Solve all eight stages independently, with a time limit per stage."""
    return [solve(instance,stage,mode,time_limit) for stage in STAGES]


def _worker(messages,instance,stage,mode,time_limit):
    try:messages.put(('result',solve(instance,stage,mode,time_limit)))
    except Exception as exc:messages.put(('error',str(exc)))


def solve_isolated(instance,stage='full',mode='ip',time_limit=30,wall_time_limit=60):
    """Run in a spawned process with a total build+solve timeout.

    Use under an if __name__ == '__main__' guard in scripts. A wall timeout
    returns a valid volume fallback, not a partial or unverified incumbent.
    Input validation and process startup precede the timed queue wait.
    """
    import multiprocessing,queue
    if stage not in STAGES or mode not in ('ip','lp'):raise ValueError('Unknown stage or mode')
    if not math.isfinite(time_limit) or time_limit<=0:raise ValueError('Invalid solver time limit')
    if not math.isfinite(wall_time_limit) or wall_time_limit<=0:raise ValueError('Invalid wall-time limit')
    payload=prepare(instance);start=time.perf_counter()
    ctx=multiprocessing.get_context('spawn');messages=ctx.Queue()
    process=ctx.Process(target=_worker,args=(messages,instance,stage,mode,time_limit))
    process.start()
    try:
        deadline=time.monotonic()+wall_time_limit
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            try:
                kind,value=messages.get(timeout=min(.1,remaining))
                if kind=='error':raise RuntimeError(value)
                return value
            except queue.Empty:
                if not process.is_alive():
                    try:kind,value=messages.get(timeout=.1)
                    except queue.Empty:raise RuntimeError(f'Worker exited without a result ({process.exitcode})')
                    if kind=='error':raise RuntimeError(value)
                    return value
        upper=_fractional_volume(payload['container'],payload['items']) if payload['items'] else 0.
        if mode=='ip':upper=_round_upper(upper,payload['items'])
        return dict(name=payload['name'],dimension=len(payload['container']),stage=stage,mode=mode,
                    rotations=payload['allow_rotations'],excluded=payload['excluded'],
                    upper_bound=float(upper),model_value=None,status='Wall time limit (volume fallback)',
                    solver='wall-time guard',bound_source='fractional volume',rows=None,variables=None,
                    seconds=time.perf_counter()-start,solver_time_limit=time_limit,
                    wall_time_limit=wall_time_limit,metadata={})
    finally:
        if process.is_alive():process.terminate()
        process.join(timeout=1)
        if process.is_alive():process.kill();process.join()
        process.close();messages.close()
