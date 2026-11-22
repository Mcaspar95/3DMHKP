"""Original grouped-cube and shape threshold bounds."""
import itertools
import math
from collections import Counter
from functools import lru_cache
import numpy as np

@lru_cache(None)
def cube_partition(dims):
 a,b,s=sorted(dims,reverse=True)
 if s==0:return ()
 if s==1:return ((1,a*b),)
 ka,ra=divmod(a,s);kb,rb=divmod(b,s)
 ans=Counter({s:ka*kb})
 # Large tiled block is (ka*s,kb*s,s); two disjoint residual boxes.
 for r in [(ra,b,s),(ka*s,rb,s)]:
  if min(r)>0:ans.update(dict(cube_partition(tuple(sorted(r)))))
 assert sum(k**3*v for k,v in ans.items())==a*b*s
 return tuple(sorted(ans.items()))

def axis_maps(C,lengths):
 lengths=np.asarray(lengths,dtype=np.int64)
 out=[];seen=set()
 for q2 in sorted({0, C} | {int(t) for s in lengths for t in (2*s+1, 2*(C-s)+1) if 0 <= t <= C}):
  q=q2/2
  v=np.where(2*lengths<q2,0,np.where(2*lengths>2*C-q2,C,lengths));key=tuple(v)
  if key not in seen:out.append((q,v));seen.add(key)
 # Strongest componentwise axis maps suffice for every product inequality.
 keep=[]
 for k,(q,v) in enumerate(out):
  if not any(j!=k and np.all(u>=v) for j,(_,u) in enumerate(out)):keep.append((q,v))
 return keep

def prune(rows,base):
 out=[np.asarray(base,dtype=np.int64)];seen={tuple(base)}
 for v in rows:
  v=np.asarray(v,dtype=np.int64);key=tuple(v)
  if key in seen or np.all(v<=base):continue
  seen.add(key);out.append(v)
 return np.array(out,dtype=np.float64)

def cube_rows(C,items):
 parts=[dict(cube_partition(tuple(sorted(d)))) for d,p in items]
 sizes=sorted({s for Q in parts for s in Q if s>1});hist=np.array([[Q.get(s,0) for Q in parts] for s in sizes],dtype=np.int64).reshape((len(sizes),len(items)))
 maps=[axis_maps(c,sizes) for c in C];raw=[]
 for xyz in itertools.product(*maps):raw.append((xyz[0][1]*xyz[1][1]*xyz[2][1])@hist)
 V=math.prod(C)
 # Projection-length and clique constraints scaled to common volume capacity.
 for axis in range(3):
  others=[k for k in range(3) if k!=axis]
  co=np.array([s*(2*s>C[others[0]])*(2*s>C[others[1]]) for s in sizes],dtype=np.int64)@hist
  raw.append(co*(V//C[axis]))
 raw.append((np.array([2*s>max(C) for s in sizes],dtype=np.int64)@hist)*V)
 base=np.array([math.prod(d) for d,p in items],dtype=np.int64)
 return prune(raw,base),{'axis_maps':[len(x) for x in maps],'cubes_including_units':sum(sum(q.values()) for q in parts),'cubes_retained':sum(sum(v for s,v in q.items() if s>1) for q in parts)}

def box_rows(C,items,rotate):
 dims=[];owners=[]
 for i,(d,p) in enumerate(items):
  os=sorted(set(itertools.permutations(d))) if rotate else [d]
  os=[o for o in os if all(v<=c for v,c in zip(o,C))]
  assert os
  dims.extend(os);owners.extend([i]*len(os))
 dims=np.array(dims,dtype=np.int64);owners=np.array(owners,dtype=int)
 maps=[axis_maps(c,dims[:,k]) for k,c in enumerate(C)]
 raw=[x[1]*y[1]*z[1] for x,y,z in itertools.product(*maps)]
 A=prune(raw,np.prod(dims,axis=1));G=np.zeros((len(items),len(dims)))
 G[owners,np.arange(len(dims))]=1
 return A,G,owners,dims,{'axis_maps':[len(x) for x in maps]}
