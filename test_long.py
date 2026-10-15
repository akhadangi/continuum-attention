"""CPU-only long-operator algebraic and backward regression tests.

MPS shader numerical correctness must additionally be verified on the Mac.
"""
import unittest
import torch
from architecture import continuous_attention_weights, segment_integral_coefficients, TinyLM
from analytic_reference import analytic_scores_backward


def endpoint_parallel_oracle(scores):
    """Equivalent scalar endpoint contributions to acma_long_raw_weights.

    Shapes [B,H,T,T]. Test implementation, not used in production.
    """
    T=scores.shape[-1]
    q=torch.arange(T,device=scores.device)[:,None]
    j=torch.arange(T,device=scores.device)[None,:]
    valid=j<=q
    m=scores.masked_fill(~valid,-torch.inf).amax(-1,keepdim=True)
    s=scores
    w=torch.zeros_like(s)
    cap=0.5*torch.exp(s-m)
    w=w+torch.where(j==0,cap,0.0)
    w=w+torch.where(j==q,cap,0.0)
    delta=s[...,1:]-s[...,:-1]
    L,mu=segment_integral_coefficients(delta)
    r=torch.exp(torch.maximum(s[...,1:],s[...,:-1])-m)*L
    # These are all parallel slices, never a token scan.
    w[...,:-1] += torch.where(j[...,:-1]<q,r*(1-mu),0.0)
    w[...,1:] += torch.where(j[...,1:]<=q,r*mu,0.0)
    w=w.masked_fill(~valid,0)
    return w/w.sum(-1,keepdim=True)


class LongOperatorTests(unittest.TestCase):
    def test_exactness_beyond_256(self):
        for T in (257,512):
            torch.manual_seed(T)
            s=torch.randn(1,2,T,T,dtype=torch.float64)*1.4
            expected=continuous_attention_weights(s)
            got=endpoint_parallel_oracle(s)
            torch.testing.assert_close(got,expected,atol=2e-10,rtol=2e-10)

    def test_vjp_long(self):
        T=257
        s=torch.randn(1,1,T,T,dtype=torch.float64,requires_grad=True)
        g=torch.randn_like(s)
        w=continuous_attention_weights(s)
        actual=analytic_scores_backward(s.detach(),g.detach(),w.detach())
        expected=torch.autograd.grad(w,s,g)[0]
        torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-8)

    def test_model_512_cpu(self):
        torch.manual_seed(1729)
        m=TinyLM(variant='S',backend='reference',seq_len=512,width=32,hidden=64,heads=4)
        x=torch.randint(0,256,(1,512))
        with torch.no_grad():
            out=m(x)
        self.assertEqual(tuple(out.shape),(1,512,256))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_long_shader_and_dispatch_defined(self):
        from acma_metal import METAL_SOURCE
        self.assertIn('acma_long_raw_weights',METAL_SOURCE)
        from pathlib import Path
        self.assertIn('_MetalLongWeights',Path('acma_metal.py').read_text())


if __name__=='__main__': unittest.main()
