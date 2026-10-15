"""Analytic-moment and custom-derivative tests, no Apple hardware required."""
import math
import unittest
import torch
from analytic_reference import stable_moments, analytic_scores_backward
from architecture import continuous_attention_weights
from acma_metal import METAL_SOURCE, _scores_backward_kernel


class AnalyticBackwardTests(unittest.TestCase):
    def test_shipping_kernel_not_recompute(self):
        self.assertIn('kernel void acma_scores_backward(', METAL_SOURCE)
        import inspect
        import acma_metal
        fn = inspect.getsource(acma_metal._MetalWeights.backward)
        self.assertIn('_scores_backward_kernel(', fn)
        self.assertNotIn('autograd.grad(', fn)
        self.assertNotIn('continuous_attention_weights(', fn)

    def test_cpu_guard(self):
        s = torch.randn(1,1,2,2)
        with self.assertRaisesRegex(RuntimeError,'MPS'):
            _scores_backward_kernel(s,s,s)

    def test_moment_limits_symmetry_and_positivity(self):
        x=torch.tensor([-100.,-10.,-1.,-.1,0.,.1,1.,10.,100.],dtype=torch.float64)
        z0,z1,z2=stable_moments(x)
        self.assertTrue(torch.isfinite(z0).all())
        self.assertTrue(((z0>=z1)&(z1>=z2)&(z2>=0)).all())
        torch.testing.assert_close(z0[4],torch.tensor(1.,dtype=x.dtype))
        torch.testing.assert_close(z1[4],torch.tensor(.5,dtype=x.dtype))
        torch.testing.assert_close(z2[4],torch.tensor(1/3,dtype=x.dtype))
        for delta in (-12.,-1.01,-.8,-.2,0.,.2,.8,1.01,12.):
            t=torch.linspace(0,1,20001,dtype=torch.float64)
            weight=(t*delta - max(delta,0.)).exp()
            moments=stable_moments(torch.tensor(delta,dtype=torch.float64))
            for n,m in enumerate(moments):
                numeric=torch.trapezoid(weight*t.pow(n),t)
                torch.testing.assert_close(m,numeric,rtol=2e-6,atol=2e-7)

    def test_vjp_matches_autograd(self):
        torch.manual_seed(731)
        for size in (1,2,3,7,32,64):
            for amp in (0.0,0.01,1.0,4.0,20.0):
                s=(torch.randn(2,2,size,size,dtype=torch.float64)*amp).requires_grad_(True)
                weights=continuous_attention_weights(s)
                g=torch.randn_like(weights)
                reference=torch.autograd.grad(weights,s,g)[0]
                actual=analytic_scores_backward(s.detach(),g,weights.detach())
                torch.testing.assert_close(actual,reference,atol=1e-9,rtol=5e-8)
                self.assertTrue(torch.isfinite(actual).all())

    def test_large_length_and_causality(self):
        torch.manual_seed(23)
        s=torch.randn(1,1,192,192,dtype=torch.float64,requires_grad=True)
        w=continuous_attention_weights(s)
        g=torch.randn_like(w)
        expect=torch.autograd.grad(w,s,g)[0]
        actual=analytic_scores_backward(s.detach(),g,w.detach())
        torch.testing.assert_close(actual,expect,atol=1e-9,rtol=1e-8)
        self.assertEqual(actual[...,0,1:].abs().max().item(),0)


if __name__=='__main__':
    unittest.main()
