import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from architecture import TinyLM, parameters
from paper_experiment import arms_for
from summarize_paper import paired_stats, summarize


class PaperProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_parameter_budget_and_new_gain_controls(self):
        models={s:TinyLM(variant=s,backend='reference',seq_len=192)
                for s in ('B','F','G','S','H')}
        self.assertEqual(parameters(models['B']),5008640)
        for s in ('F','G','S','H'):
            self.assertEqual(parameters(models[s]),5008664)
        for s in ('F','S'):
            self.assertTrue(torch.allclose(models[s].b0.attn.log_qk_gain, torch.zeros(4)))
        for s in ('G','H'):
            gain=models[s].b0.attn.log_qk_gain.exp()
            self.assertTrue(torch.allclose(gain, torch.full_like(gain,1/.3),atol=1e-6))
        self.assertFalse(models['G'].b0.attn.continuous)
        self.assertTrue(models['H'].b0.attn.continuous)

    def test_four_arms_exact_parameter_budget_at_512(self):
        for s in ('F','G','S','H'):
            self.assertEqual(parameters(TinyLM(variant=s,backend='reference',seq_len=512)),5090584)

    def test_matched_initial_shared_parameters(self):
        torch.manual_seed(123)
        f=TinyLM(variant='F',backend='reference',seq_len=8,width=32,hidden=64,heads=4)
        torch.manual_seed(123)
        g=TinyLM(variant='G',backend='reference',seq_len=8,width=32,hidden=64,heads=4)
        torch.manual_seed(123)
        s=TinyLM(variant='S',backend='reference',seq_len=8,width=32,hidden=64,heads=4)
        for key, v in f.state_dict().items():
            if not key.endswith('log_qk_gain'):
                self.assertTrue(torch.equal(v,g.state_dict()[key]),key)
                self.assertTrue(torch.equal(v,s.state_dict()[key]),key)

    def test_trainable_warmstart_backward(self):
        model=TinyLM(variant='G',backend='reference',seq_len=8,width=32,hidden=64,heads=4)
        x=torch.randint(0,256,(1,8))
        model(x).square().mean().backward()
        self.assertIsNotNone(model.b0.attn.log_qk_gain.grad)
        self.assertTrue(torch.isfinite(model.b0.attn.log_qk_gain.grad).all())

    def test_paired_ci_single_seed_is_not_a_ci(self):
        self.assertIsNone(paired_stats([.04])['student_t_95pct_CI'])
        stat=paired_stats([.048011,.043859,.034957])
        self.assertAlmostEqual(stat['mean_delta_nats'],.0422756666,places=6)
        low,high=stat['student_t_95pct_CI']
        self.assertAlmostEqual(low,.02570776,places=5)
        self.assertAlmostEqual(high,.05884357,places=5)

    def test_arm_grid_and_ids(self):
        arms=arms_for([.2,.3,.5],include_h=True,include_p=True)
        self.assertEqual(len(arms),8)
        self.assertEqual(len({x['id'] for x in arms}),len(arms))
        self.assertEqual([a['variant'] for a in arms[:3]],['B','B','B'])

    def test_summary_selects_best_development_control(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            arms=arms_for([.2,.3])
            sha={'train':'abc','validation':'def'}
            p={'contexts':[192],'seeds':[1,2,3],'arms':arms,'eval_seed':9090,'data_sha256':sha,'paper_valid_data':True}
            (root/'protocol.json').write_text(json.dumps(p))
            vals={'B_tau0p20':1.19,'B_tau0p30':1.10,'F_tauinit1':1.18,
                  'G_tauinit0p30':1.09,'S_tauinit1':1.05}
            for seed in p['seeds']:
                folder=root/'T192'/f'seed{seed}'
                folder.mkdir(parents=True)
                for arm in arms:
                    id=arm['id']
                    obj={'settings':{'seed':seed,'seq_len':192,'eval_seed':9090},
                         'dataset_sha256':{'train_sha256':'abc','validation_sha256':'def'},
                         'results':[{'variant':arm['variant'],'validation_nll_nats':vals[id]+seed*.001,
                                     'parameters':5008664,'training_tokens_per_second':39000.,
                                     'speed':{'forward_tokens_per_second':120000.}}]}
                    (folder/(id+'.json')).write_text(json.dumps(obj))
            report=summarize(root)
            self.assertEqual(report['contexts']['192']['dev_selected_best_softmax'],'G_tauinit0p30')
            self.assertAlmostEqual(report['contexts']['192']['paired_control_minus_S']['G_tauinit0p30']['mean_delta_nats'],.04)


if __name__=='__main__':unittest.main()
