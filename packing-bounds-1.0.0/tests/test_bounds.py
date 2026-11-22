import itertools,json,math,os,subprocess,sys,tempfile,unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from scipy.sparse import csr_matrix,load_npz
from packing_bounds import STAGES,solve,solve_isolated,build_model,prepare
from packing_bounds.scales import rawmaps,frontier,tighten,product_rows
from packing_bounds.solvers import _extended_integer_solve


def sample(C=(10,7),a=(3,3),profit=1,copies=7,rotate=True):
 return dict(container=list(C),allow_rotations=rotate,items=[dict(dimensions=list(a),profit=profit,copies=copies)])

@lru_cache(None)
def patterns(C,minimum=1):
 return ((),)+tuple((s,)+tail for s in range(minimum,C+1) for tail in patterns(C-s,s))

class BoundsTests(unittest.TestCase):
 def test_2d_and_3d_nested_improvement(self):
  for inst in [sample(),sample((10,7,1),(3,3,1))]:
   for mode in ('ip','lp'):
    last=math.inf
    for stage in ('baseline','floor_mixed','rounding_mixed','data_mixed','full'):
     r=solve(inst,stage,mode)
     self.assertLessEqual(r['upper_bound'],last+1e-7)
     self.assertAlmostEqual(r['upper_bound'],7 if stage=='baseline' else 6)
     last=r['upper_bound']
 def test_all_stages_and_incompatible_cubes(self):
  inst=sample((11,11,11),(6,6,6),8,3)
  for stage in STAGES:
   self.assertEqual(solve(inst,stage)['upper_bound'],24 if stage=='volume' else 8)
 def test_rotations_and_exclusions(self):
  for C,a in [((2,5),(5,2)),((2,5,2),(5,2,2))]:
   for rotate in (False,True):
    r=solve(sample(C,a,7.5,1,rotate))
    self.assertAlmostEqual(r['upper_bound'],7.5 if rotate else 0)
    self.assertEqual(r['excluded'],0 if rotate else 1)
 def test_binary_vs_lp_decimal(self):
  inst=sample((10,10),(6,10),7.5,2)
  self.assertAlmostEqual(solve(inst,'volume','ip')['upper_bound'],7.5)
  self.assertAlmostEqual(solve(inst,'volume','lp')['upper_bound'],12.5)
 def test_empty(self):
  for C in ([10,10],[10,10,10]):
   for stage in STAGES:
    self.assertEqual(solve(dict(container=C,items=[]),stage)['upper_bound'],0)
 def test_invalid_input(self):
  for inst in [sample(a=(3.5,3)),sample(profit=-1),sample(copies=0),sample(rotate='false'),
               sample(a=(3,3,1)),sample(C=(0,7)),sample(profit=float('nan'))]:
   with self.assertRaises(ValueError):prepare(inst)
  for limit in (0,-1,float('nan'),float('inf')):
   with self.assertRaises(ValueError):solve(sample(),time_limit=limit)
 def test_feasible_tilings_survive_all_rows(self):
  for C,pieces in [((9,8),[(4,8),(5,3),(5,5)]),((9,8,7),[(4,8,7),(5,3,7),(5,5,2),(5,5,5)])]:
   for rotate in (False,True):
    inst=dict(container=C,allow_rotations=rotate,items=[dict(dimensions=tuple(reversed(d)) if rotate else d,profit=1) for d in pieces])
    for stage in STAGES:
     model=build_model(inst,stage);y=np.zeros(len(model.profits))
     for i,target in enumerate(pieces):
      candidates=[j for j,(owner,o) in enumerate(zip(model.owners,model.orientations))
                  if owner==i and (stage in ('volume','fragments') or tuple(o)==target)]
      y[candidates[0]]=1
     self.assertTrue(np.all(model.A@y<=model.b+1e-7),(C,stage))
 def test_finite_map_validity_and_exact_capacity(self):
  for C in range(2,9):
   for v,b,_ in rawmaps(C,np.arange(1,C+1),'rounding'):
    self.assertLessEqual(max(sum(v[s-1] for s in pat) for pat in patterns(C)),b)
   domain=np.arange(1,C+1,2)
   for v,b,_ in frontier(rawmaps(C,domain,'rounding')):
    values=dict(zip(domain,v))
    exact=max(sum(values[s] for s in pat) for pat in patterns(C) if all(s in values for s in pat))
    self.assertEqual(tighten(C,domain,v),exact)
 def test_large_cross_products(self):
  A,b,_=product_rows((100000,)*3,np.array([[60000]*3]),'threshold')
  self.assertAlmostEqual(float(np.max(A.toarray().ravel()/b)),1.)
 def test_restricted_incumbent_checked(self):
  A=csr_matrix([[.5,.5],[2/3,2/3]])
  results=[(SimpleNamespace(status=0,message='',mip_dual_bound=-2.,x=np.array([1.,1.])),'fake'),
           (SimpleNamespace(status=0,message='',mip_dual_bound=-1.,x=np.array([1.,0.])),'fake')]
  with patch('packing_bounds.solvers._integer_solve',side_effect=results):
   r,_=_extended_integer_solve(np.ones(2),A,A,np.ones(2),10,2.,Mock(return_value=SimpleNamespace(success=False)),None,None,None)
  self.assertEqual(r.fun,-1);self.assertEqual(r.row_generation_rounds,2)
 def test_time_limit_uses_dual_not_incumbent(self):
  fake=SimpleNamespace(status=1,fun=-7.5,mip_dual_bound=-11.,message='time limit')
  with patch('packing_bounds.api._integer_solve',return_value=(fake,'fake')):
   r=solve(sample((10,10),(6,10),7.5,2),'volume')
  self.assertEqual(r['upper_bound'],11.);self.assertEqual(r['model_value'],7.5)
 def test_scipy_fallback(self):
  with patch.dict('sys.modules',{'highspy':None}):r=solve(sample())
  self.assertEqual(r['upper_bound'],6)
 def test_matrix_export(self):
  model=build_model(sample())
  with tempfile.TemporaryDirectory() as d:
   model.export(d)
   self.assertEqual((load_npz(Path(d)/'A.npz')!=model.A).nnz,0)
   np.testing.assert_array_equal(np.load(Path(d)/'vectors.npz')['b'],model.b)
 def test_cli_json_and_errors(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'input.json';path.write_text(json.dumps(sample()))
   p=subprocess.run([sys.executable,'-m','packing_bounds',str(path),'--mode','both'],capture_output=True,text=True)
   self.assertEqual(p.returncode,0,p.stderr)
   for result in json.loads(p.stdout):self.assertAlmostEqual(result['upper_bound'],6.)
   path.write_text('{')
   p=subprocess.run([sys.executable,'-m','packing_bounds',str(path)],capture_output=True,text=True)
   self.assertEqual(p.returncode,2)
 def test_isolated_solver_and_wall_timeout(self):
  self.assertEqual(solve_isolated(sample(),wall_time_limit=10)['upper_bound'],6)
  result=solve_isolated(sample(),wall_time_limit=.00001)
  self.assertIn('Wall time limit',result['status'])
  self.assertGreaterEqual(result['upper_bound'],6)
  self.assertIsNone(result['model_value'])

if __name__=='__main__':unittest.main()
