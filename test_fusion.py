"""CPU guards for shipped Metal formulas; GPU correctness in verify_mps.py."""
import unittest
import torch
from torch.nn import functional as F
from architecture import TinyLM, parameters, continuous_attention_weights
from acma_metal import METAL_SOURCE


class FusedACMATests(unittest.TestCase):
    def test_shipped_metal_contains_true_direct_kernel(self):
        self.assertIn('kernel void acma_direct(',METAL_SOURCE)
        self.assertIn('kernel void acma_weights(',METAL_SOURCE)
        self.assertIn('segment_mass(',METAL_SOURCE)
        self.assertIn('cap0',METAL_SOURCE)
        self.assertIn('capi',METAL_SOURCE)

    def test_baseline_seed_and_parameter_budget_intact(self):
        for name, count in [('B',5008640),('P',5008640),('S',5008664)]:
            self.assertEqual(parameters(TinyLM(variant=name)),count)
        ids=torch.randint(0,256,(1,13))
        torch.manual_seed(123)
        old=TinyLM(variant='B',seq_len=16,width=32,hidden=64,heads=4,backend='reference')
        torch.manual_seed(123)
        new=TinyLM(variant='B',seq_len=16,width=32,hidden=64,heads=4,backend='mixed')
        torch.testing.assert_close(old(ids),new(ids),atol=2e-5,rtol=2e-5)

    def test_cap_integral_and_stochastic_weights(self):
        torch.manual_seed(22)
        scores=torch.randn(2,3,15,15)
        weights=continuous_attention_weights(scores)
        self.assertTrue((weights>=0).all().item())
        torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights.sum(-1)),atol=1e-6,rtol=1e-6)
        v=torch.randn(2,3,15,16)
        self.assertEqual((weights@v).shape,(2,3,15,16))
        torch.testing.assert_close(weights[...,0,0],torch.ones_like(weights[...,0,0]))

    def test_mps_execution_guard_on_cpu(self):
        from acma_metal import acma_metal_direct,acma_metal_weights
        s=torch.randn(1,1,3,3)
        v=torch.randn(1,1,3,4)
        with self.assertRaisesRegex(RuntimeError,'requires an MPS'):
            acma_metal_weights(s)
        with self.assertRaisesRegex(RuntimeError,'requires an MPS'):
            acma_metal_direct(s,v)


if __name__=='__main__':
    unittest.main()
