"""Threshold, floor, rounding, domain-tightened scales and conflicts."""
import itertools
import math
import numpy as np
from scipy.sparse import csr_matrix

def rawmaps(C, lengths, family):
    s=np.asarray(lengths,dtype=np.int64)
    for q2 in sorted({0, C} | {int(t) for length in s for t in (2*length+1, 2*(C-length)+1) if 0 <= t <= C}):
        yield np.where(2*s<q2,0,np.where(2*s>2*C-q2,C,s)),C,('threshold',q2)
    if family in ('floor','rounding'):
        for k in range(1,C+1):
            K=C//k
            yield s//k,K,('floor',k)
            if family=='rounding':
                yield np.where(2*s>C,2*(K-(C-s)//k),np.where(2*s==C,K,2*(s//k))),2*K,('rounding',k)

def frontier(raw):
    """Exact rational componentwise comparisons, not floating-point pruning."""
    out=[]
    for v,b,label in raw:
        v=np.asarray(v,dtype=np.int64)
        if not np.any(v):continue
        g=math.gcd(int(b),*map(int,v));v=v//g;b=int(b)//g
        if out:
            nums=np.array([z[0] for z in out]);caps=np.array([z[1] for z in out])
            if np.any(np.all(nums*b>=v*caps[:,None],axis=1)):continue
            keep=~np.all(v*caps[:,None]>=nums*b,axis=1)
            out=[z for z,k in zip(out,keep) if k]
        out.append((v,b,label))
    return out

def tighten(C,lengths,v):
    """Exact unbounded knapsack on the union of possible axis lengths.

    Includes every physically possible 1D pattern, even repeated item lengths.
    Returns the largest total transformed weight of any such pattern.
    """
    s=np.asarray(lengths,dtype=int);v=np.asarray(v,dtype=np.int64)
    dp=np.zeros(C+1,dtype=np.int64)
    for c in range(1,C+1):
        ok=s<=c
        dp[c]=max(dp[c-1],np.max(dp[c-s[ok]]+v[ok],initial=0))
    return int(dp[C])

def axis(C,values,family,data=False):
    lengths,inverse=np.unique(values,return_inverse=True)
    out=frontier(rawmaps(C,lengths,family))
    if data:
        lifted=[]
        for v,b,label in out:
            tight=tighten(C,lengths,v)
            assert 0<tight<=b
            lifted.append((v,tight,('domain_tightened',label,b)))
        out=frontier(lifted)
    return [(v[inverse],b,label) for v,b,label in out]

def product_rows(C,dims,family,data=False):
    maps=[axis(c,dims[:,j],family,data) for j,c in enumerate(C)]
    area=np.prod(dims,axis=1);V=math.prod(C)
    rows=[];rhs=[];seen=set()
    for factors in itertools.product(*maps):
        v=np.prod([z[0] for z in factors],axis=0,dtype=np.int64)
        b=math.prod(z[1] for z in factors)
        if np.all(v.astype(object)*V<=area.astype(object)*b):continue
        g=math.gcd(b,*map(int,v));v=v//g;b=b//g
        key=(b,v.tobytes())
        if key in seen:continue
        seen.add(key);rows.append(v);rhs.append(b)
    A=csr_matrix(np.asarray(rows,dtype=float).reshape((-1,len(dims))))
    return A,np.asarray(rhs,dtype=float),{'axis_maps':[len(z) for z in maps],
         'product_candidates':math.prod(map(len,maps)),'retained_product_rows':len(rows)}

def conflicts(C,dims,owners):
    nvars=len(dims);n=int(max(owners))+1
    edges=[];compatible=np.zeros((n,n),dtype=bool)
    for j in range(nvars):
        for k in range(j+1,nvars):
            if owners[j]==owners[k]:continue
            if np.all(dims[j]+dims[k]>C):edges.append((j,k))
            else:compatible[owners[j],owners[k]]=compatible[owners[k],owners[j]]=True
    # Greedy maximal item cliques: all orientation pairs conflict for each pair.
    adj=~compatible;np.fill_diagonal(adj,False)
    order=sorted(range(n),key=lambda i:-int(adj[i].sum()))
    cliques=set()
    for seed in order:
        clique=[seed]
        for i in order:
            if i!=seed and all(adj[i,j] for j in clique):clique.append(i)
        if len(clique)>2:cliques.add(tuple(sorted(clique)))
    rr=[];cc=[]
    for r,(j,k) in enumerate(edges):rr.extend([r,r]);cc.extend([j,k])
    for r,clique in enumerate(sorted(cliques),start=len(edges)):
        js=np.flatnonzero(np.isin(owners,clique));rr.extend([r]*len(js));cc.extend(js.tolist())
    count=len(edges)+len(cliques)
    return csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(count,nvars)),np.ones(count),{'orientation_pair_cuts':len(edges),'item_clique_cuts':len(cliques)}

