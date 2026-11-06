"""Small mathematical checks; no training, data downloads or analysis imports."""
import ast
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]

def selected_functions(filename,names):
    tree=ast.parse((ROOT/'scripts'/filename).read_text(encoding='utf-8'))
    body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in body}==set(names)
    module=ast.Module(body=body,type_ignores=[])
    scope={'np':np,'log_ndtr':log_ndtr,'LOG_SQRT_2PI':.5*np.log(2*np.pi)}
    exec(compile(module,filename,'exec'),scope)
    return scope

C=selected_functions('censored_lib.py',['_types','tobit_nll','tobit_grad_hess','interval_cindex'])
E=selected_functions('15_global_epistasis.py',['link'])

class CoreChecks(unittest.TestCase):
    def test_all_scripts_compile(self):
        for path in (ROOT/'scripts').glob('*.py'):
            with self.subTest(path=path.name):compile(path.read_text(encoding='utf-8'),str(path),'exec')

    def test_point_and_tail_likelihood(self):
        mu=np.array([.3,-.2,.8]);lo=np.array([.7,-np.inf,1.]);hi=np.array([.7,-.5,np.inf]);sigma=1.2
        expected=np.mean([-norm.logpdf(.7,loc=.3,scale=sigma),-norm.logcdf(-.5,loc=-.2,scale=sigma),-norm.logsf(1.,loc=.8,scale=sigma)])
        self.assertAlmostEqual(C['tobit_nll'](mu,lo,hi,sigma),expected,places=12)

    def test_analytic_gradient_and_curvature(self):
        lo=np.array([.4,-np.inf,.9]);hi=np.array([.4,-.6,np.inf]);mu=np.array([.1,.3,.2]);sigma=.9
        g,h=C['tobit_grad_hess'](mu,lo,hi,sigma)
        eps=1e-4
        for i in range(3):
            def value(x):return C['tobit_nll'](np.array([x]),lo[i:i+1],hi[i:i+1],sigma)
            grad=(value(mu[i]+eps)-value(mu[i]-eps))/(2*eps)
            curvature=(value(mu[i]+eps)-2*value(mu[i])+value(mu[i]-eps))/eps**2
            self.assertAlmostEqual(float(g[i]),grad,places=6)
            self.assertAlmostEqual(float(h[i]),curvature,places=5)

    def test_extreme_tail_stability(self):
        value=C['tobit_nll'](np.array([40.,-40.]),np.array([-np.inf,0.]),np.array([0.,np.inf]),1.)
        self.assertTrue(math.isfinite(value));self.assertGreater(value,800.)

    def test_interval_ordering_and_ties(self):
        lo=np.array([-np.inf,0.,2.]);hi=np.array([-1.,0.,np.inf])
        for pred,want in [(np.array([-2.,0.,3.]),1.),(np.array([3.,0.,-2.]),0.),(np.zeros(3),.5)]:
            actual,n=C['interval_cindex'](pred,lo,hi,n_pairs=10000,seed=42)
            self.assertGreater(n,0);self.assertEqual(actual,want)

    def test_softplus_limits_and_monotonicity(self):
        th=np.array([-5.,np.log(.8),.2,np.log(.7)])
        vals=E['link'](np.array([-100.,-6.,-4.,100.]),th)
        self.assertTrue(np.all(np.diff(vals)>0));self.assertAlmostEqual(vals[0],-5.,places=12);self.assertAlmostEqual(vals[-1],100.2,places=10)

if __name__=='__main__':unittest.main()
