"""Analytic correctness tests for exact continuous-memory attention."""
import ast
import inspect
import textwrap
import unittest
import torch
from architecture import (TinyLM, CausalAttention, parameters,
                          segment_integral_coefficients,
                          continuous_attention_weights)


class ACMATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)

    def tiny_model(self, variant='P', backend='reference'):
        torch.manual_seed(97)
        return TinyLM(variant=variant, backend=backend, seq_len=16,
                      width=32, hidden=64, heads=4, fixed_tau=.30)

    def test_parameter_budget_exact(self):
        self.assertEqual(parameters(TinyLM(variant='B')), 5_008_640)
        self.assertEqual(parameters(TinyLM(variant='P')), 5_008_640)
        self.assertEqual(parameters(TinyLM(variant='R')), 5_008_640)
        self.assertEqual(parameters(TinyLM(variant='F')), 5_008_664)
        self.assertEqual(parameters(TinyLM(variant='S')), 5_008_664)

    def test_shared_initial_weights_equal(self):
        states={v:self.tiny_model(v).state_dict() for v in ('A','B','F','P','R','S')}
        common=set.intersection(*(set(v) for v in states.values()))
        self.assertGreaterEqual(len(common),40)
        for key in common:
            for variant in ('A','B','F','P','R','S'):
                self.assertTrue(torch.equal(states['B'][key],states[variant][key]),key)

    def test_segments_against_direct_numeric_integration(self):
        # Quadrature is solely an offline test of the *closed-form* model.
        pairs = [(-4., 1.5), (2.3,-1.7), (-.02,.01), (3.,3.), (-100.,100.),(100.,-100.)]
        for a,b in pairs:
            delta=torch.tensor([b-a],dtype=torch.float64)
            L,mu=segment_integral_coefficients(delta)
            u=torch.linspace(0,1,40001,dtype=torch.float64)
            exp_stable=torch.exp(torch.tensor(a,dtype=torch.float64)+u*delta[0]-max(a,b))
            mass=torch.trapezoid(exp_stable,u)
            moment=torch.trapezoid(exp_stable*u,u)
            self.assertAlmostEqual((L*torch.exp(torch.tensor(max(a,b),dtype=torch.float64)-max(a,b))).item(),mass.item(),delta=3e-8)
            self.assertAlmostEqual(mu.item(),(moment/mass).item(),delta=3e-8)

    def test_exact_reference_integral_multiple_queries(self):
        # Brute-force high-resolution quadrature per segment and query.
        scores=torch.tensor([.35,-1.8,1.0,2.5],dtype=torch.float64)
        values=torch.tensor([-.7,1.3,5.,-2.],dtype=torch.float64)
        T=scores.numel()
        logits=scores[None,None,None,:].expand(1,1,T,T).clone()
        weights=continuous_attention_weights(logits)[0,0]
        for i in range(T):
            numerator = .5 * torch.exp(scores[0]-scores[:i+1].max())*values[0]
            numerator += .5 * torch.exp(scores[i]-scores[:i+1].max())*values[i]
            denominator = .5*torch.exp(scores[0]-scores[:i+1].max())
            denominator += .5*torch.exp(scores[i]-scores[:i+1].max())
            for j in range(1,i+1):
                u=torch.linspace(0,1,20001,dtype=torch.float64)
                f=torch.exp(scores[j-1]*(1-u)+scores[j]*u-scores[:i+1].max())
                v=values[j-1]*(1-u)+values[j]*u
                numerator += torch.trapezoid(f*v,u)
                denominator += torch.trapezoid(f,u)
            expected=numerator/denominator
            actual=(weights[i]*values).sum()
            self.assertAlmostEqual(actual.item(),expected.item(),delta=5e-8)

    def test_constant_score_limit_equals_uniform_standard_softmax(self):
        z=torch.ones(2,3,14,14,dtype=torch.float64)*4.3
        w=continuous_attention_weights(z)
        causal=torch.ones(14,14,dtype=torch.bool).tril()
        expected=torch.softmax(z.masked_fill(~causal,-torch.inf),dim=-1)
        torch.testing.assert_close(w,expected,atol=1e-14,rtol=1e-14)

    def test_one_token_exact(self):
        w=continuous_attention_weights(torch.tensor([[[[123.]]]]))
        self.assertEqual(w.item(),1.)

    def test_weights_nonnegative_causal_and_sum_to_one(self):
        logits=torch.randn(2,4,13,13)*8
        w=continuous_attention_weights(logits)
        self.assertTrue(torch.isfinite(w).all().item())
        self.assertGreaterEqual(w.min().item(),0)
        torch.testing.assert_close(w.sum(-1),torch.ones_like(w.sum(-1)),atol=1e-6,rtol=1e-6)
        causal=torch.ones(13,13,dtype=torch.bool).tril()
        self.assertEqual(w.masked_select(~causal).abs().sum().item(),0.)

    def test_extreme_logits_finite_and_offset_invariance(self):
        logits=torch.randn(1,3,9,9)*150
        w=continuous_attention_weights(logits)
        self.assertTrue(torch.isfinite(w).all().item())
        torch.testing.assert_close(w,continuous_attention_weights(logits+1000),atol=1e-4,rtol=1e-4)

    def test_analytical_coefficients_at_zero_and_gradients(self):
        delta=torch.zeros(3,dtype=torch.float64,requires_grad=True)
        mass,mu=segment_integral_coefficients(delta)
        torch.testing.assert_close(mass,torch.ones_like(mass))
        torch.testing.assert_close(mu,torch.ones_like(mu)*.5)
        grad=torch.autograd.grad(mu.sum(),delta)[0]
        torch.testing.assert_close(grad,torch.full_like(grad,1/12),atol=1e-12,rtol=1e-12)

    def test_scores_receive_gradients(self):
        logits=torch.randn(2,2,7,7,requires_grad=True)
        weights=continuous_attention_weights(logits)
        loss=(weights*torch.randn_like(weights)).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(logits.grad.abs().sum().item(),0)

    def test_qkv_gradient_and_learned_scale_gradient(self):
        attn=CausalAttention(32,4,16,variant='S')
        out=attn(torch.randn(2,11,32)).square().mean()
        out.backward()
        self.assertIsNotNone(attn.log_qk_gain.grad)
        self.assertTrue(torch.isfinite(attn.log_qk_gain.grad).all().item())
        self.assertGreater(attn.log_qk_gain.grad.abs().sum().item(),0)
        self.assertGreater(attn.qkv.weight.grad.abs().sum().item(),0)

    def test_no_future_information_leak(self):
        for variant in ('P','R','S'):
            model=self.tiny_model(variant).eval()
            ids=torch.randint(0,256,(2,13))
            modified=ids.clone()
            modified[:,7:]=torch.randint(0,256,(2,6))
            with torch.no_grad():
                before=model(ids)[:,:7]
                after=model(modified)[:,:7]
            torch.testing.assert_close(before,after,atol=2e-5,rtol=2e-5)

    def test_segment_evaluates_continuum_not_simple_softmax(self):
        logits=torch.tensor([[[[0.,0.,0.],[0.,2.,0.],[1.,-1.,3.]]]])
        w=continuous_attention_weights(logits)
        causal=torch.ones(3,3,dtype=torch.bool).tril()
        standard=torch.softmax(logits.masked_fill(~causal,-torch.inf),-1)
        self.assertGreater((w-standard).abs().sum().item(),.01)

    def test_unscaled_control_matches_learned_initially(self):
        ids=torch.randint(0,256,(1,13))
        with torch.no_grad():
            a=self.tiny_model('A'); f=self.tiny_model('F')
            r=self.tiny_model('R'); s=self.tiny_model('S')
            torch.testing.assert_close(a(ids),f(ids),atol=0,rtol=0)
            torch.testing.assert_close(r(ids),s(ids),atol=0,rtol=0)

    def test_baseline_reference_and_optimized_match(self):
        ids=torch.randint(0,256,(1,11))
        for variant in ('B','F'):
            ref=self.tiny_model(variant,'reference')
            fast=self.tiny_model(variant,'mixed')
            with torch.no_grad():
                torch.testing.assert_close(ref(ids),fast(ids),atol=3e-5,rtol=3e-5)

    def test_forward_contains_no_loops_scans(self):
        funcs=(segment_integral_coefficients,continuous_attention_weights,
               CausalAttention._qkv,CausalAttention._weights,CausalAttention.forward,
               TinyLM.forward)
        for fn in funcs:
            code=textwrap.dedent(inspect.getsource(fn))
            tree=ast.parse(code)
            banned=(ast.For,ast.AsyncFor,ast.While,ast.ListComp,
                    ast.DictComp,ast.SetComp,ast.GeneratorExp)
            self.assertFalse(any(isinstance(n,banned) for n in ast.walk(tree)),fn.__name__)


if __name__=='__main__':
    unittest.main()
