"""Dimension-independent cumulative necessary-condition models."""
from dataclasses import dataclass
import itertools
import math
import numpy as np
from scipy.sparse import csr_matrix, vstack, save_npz
from .baselines import cube_rows
from .scales import product_rows, conflicts

STAGES = ('volume', 'fragments', 'shapes', 'baseline', 'floor_mixed',
          'rounding_mixed', 'data_mixed', 'full')

@dataclass
class Model:
    """Maximize profits @ y subject to A @ y <= b, 0 <= y <= 1.

    owners[j] identifies the filtered/expanded item for column j.
    orientations[j] gives its dimensions, or the original dimensions for an
    orientation-independent volume/fragment variable. Integrality is chosen
    at solve time. Feasibility here does not imply geometric packability.
    """
    profits: np.ndarray
    A: csr_matrix
    b: np.ndarray
    owners: np.ndarray
    orientations: np.ndarray
    metadata: dict

    def export(self, directory):
        """Export sparse A.npz, vectors.npz and metadata.json for another solver."""
        from pathlib import Path
        import json
        out=Path(directory);out.mkdir(parents=True,exist_ok=True)
        save_npz(out/'A.npz',self.A)
        np.savez(out/'vectors.npz',profits=self.profits,b=self.b,
                 owners=self.owners,orientations=self.orientations)
        (out/'metadata.json').write_text(json.dumps(self.metadata,indent=2)+'\n')


def square_rows(C, items):
    """Grouped CUTSQ tall-width/count and residual-space inequalities.

    Each displayed inequality has its own geometric proof; these rows do not
    represent every bound from Dell'Amico et al. Unit fragments are omitted.
    """
    W,H=sorted(C,reverse=True)
    parts=[]
    for dims,_ in items:
        w,h=sorted(dims,reverse=True);counts={}
        while h>1:
            k,r=divmod(w,h);counts[h]=counts.get(h,0)+k;w,h=h,r
        parts.append(counts)
    rows=[[math.prod(d) for d,_ in items]];rhs=[W*H]
    rows.append([sum(s*n for s,n in q.items() if 2*s>H) for q in parts]);rhs.append(W)
    rows.append([sum(n for s,n in q.items() if 2*s>H) for q in parts]);rhs.append(W//(H//2+1))
    for threshold in range(H//2+1):
        rows.append([sum(n*s*s for s,n in q.items() if s>=threshold)+
                     sum(n*s*(H-s) for s,n in q.items() if s>H-threshold) for q in parts])
        rhs.append(W*H)
    # Exact duplicate and volume-dominance removal (Python integer products).
    seen=set();keep=[];capacities=[]
    for index,(row,cap) in enumerate(zip(rows,rhs)):
        if index and all(a*(W*H)<=v*cap for a,v in zip(row,rows[0])):
            continue
        g=math.gcd(cap,*row);key=(cap//g,tuple(a//g for a in row))
        if key in seen:continue
        seen.add(key);keep.append(row);capacities.append(cap)
    return csr_matrix(np.array(keep,dtype=float)),np.array(capacities,dtype=float)


def construct(payload, stage='full', progress=None):
    if stage not in STAGES:raise ValueError(f'Unknown stage: {stage}')
    C=payload['container'];items=payload['items'];n=len(items)
    V=math.prod(C);d=len(C)
    if not n:
        return Model(np.zeros(0),csr_matrix((0,0)),np.zeros(0),np.zeros(0,dtype=int),
                     np.empty((0,d),dtype=int),{'stage':stage,'dimension':d,'stages':[]})
    volume=np.array([math.prod(a) for a,_ in items],dtype=float)
    profits=np.array([p for _,p in items],dtype=float)
    meta={'stage':stage,'dimension':d,'stages':[],'item_ids':payload['item_ids'],
          'allow_rotations':payload['allow_rotations']}
    if stage=='volume':
        return Model(profits,csr_matrix(volume[None,:]),np.array([V],dtype=float),
                     np.arange(n),np.array([a for a,_ in items]),meta)
    if stage!='shapes':
        if d==2:
            F,fb=square_rows(C,items)
        else:
            raw,_=cube_rows(C,items);F=csr_matrix(raw);fb=np.full(len(raw),V,dtype=float)
        if stage=='fragments':
            return Model(profits,F,fb,np.arange(n),np.array([a for a,_ in items]),meta)
    dims=[];owners=[]
    for i,(a,_) in enumerate(items):
        choices=sorted(set(itertools.permutations(a))) if payload['allow_rotations'] else [a]
        for o in choices:
            if all(s<=c for s,c in zip(o,C)):dims.append(o);owners.append(i)
    dims=np.array(dims,dtype=np.int64);owners=np.array(owners,dtype=int)
    groups=csr_matrix((np.ones(len(owners)),(owners,np.arange(len(owners)))),shape=(n,len(owners)))
    A,b,stats=product_rows(C,dims,'threshold')
    A=vstack([csr_matrix(volume[owners][None,:]),A,groups],format='csr')
    b=np.r_[V,b,np.ones(n)]
    meta['threshold_products']=stats
    if stage!='shapes':
        A=vstack([A,F[1:,owners]],format='csr');b=np.r_[b,fb[1:]]
    meta['baseline_rows']=A.shape[0]
    if stage in ('baseline','shapes'):
        return Model(profits[owners],A,b,owners,dims,meta)
    for name,family,data in [('floor_mixed','floor',False),
                             ('rounding_mixed','rounding',False),
                             ('data_mixed','rounding',True)]:
        if progress:progress({'phase':'building','detail':name})
        new,nb,stats=product_rows(C,dims,family,data)
        A=vstack([A,new],format='csr');b=np.r_[b,nb]
        meta['stages'].append(dict(stage=name,**stats))
        if stage==name:return Model(profits[owners],A,b,owners,dims,meta)
    new,nb,stats=conflicts(np.asarray(C),dims,owners)
    A=vstack([A,new],format='csr');b=np.r_[b,nb]
    meta['stages'].append(dict(stage='full',**stats))
    return Model(profits[owners],A,b,owners,dims,meta)
