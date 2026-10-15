"""CPU-only regression tests for add-on two-arm experiment protocol and merger."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from paper_experiment import arms_for
from summarize_hp import calculate


class HPAblationTests(unittest.TestCase):
    def test_variants_have_expected_initialization(self):
        import torch
        from architecture import TinyLM, parameters
        for T in (192,512):
            H=TinyLM(variant='H',backend='reference',seq_len=T)
            P=TinyLM(variant='P',backend='reference',seq_len=T)
            self.assertTrue(H.b0.attn.continuous)
            self.assertTrue(P.b0.attn.continuous)
            self.assertAlmostEqual(torch.exp(-H.b0.attn.log_qk_gain).mean().item(),.3,places=6)
            self.assertEqual(parameters(H),5008664 if T==192 else 5090584)
            self.assertEqual(parameters(P),5008640 if T==192 else 5090560)

    def test_merger_uses_old_results_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'original'
            hp=Path(tmp)/'new_hp'
            old.mkdir(); hp.mkdir()
            proto={'schema_version':1,'contexts':[192,512],
                   'seeds':[1729,2718,3141,4567,6174],
                   'data_sha256':{'train':'trainsha','validation':'valsha'},
                   'paper_valid_data':True,'eval_seed':9090,'val_batches':128,
                   'steps_by_context':{'192':2500,'512':3750},
                   'batch_by_context':{'192':8,'512':2},
                   'lr':4e-4,'geometry_lr_mult':8.,'backend':'auto','probe_reps':20}
            old_protocol={**proto,'arms':[
                {'id':'B_tau0p20','variant':'B','fixed_tau':.2},
                {'id':'B_tau0p30','variant':'B','fixed_tau':.3},
                {'id':'F_tauinit1','variant':'F','fixed_tau':.3},
                {'id':'G_tauinit0p30','variant':'G','fixed_tau':.3},
                {'id':'S_tauinit1','variant':'S','fixed_tau':.3}]}
            hp_protocol={**proto,'arms':[
                {'id':'H_tauinit0p30','variant':'H','fixed_tau':.3},
                {'id':'P_tau0p30','variant':'P','fixed_tau':.3}],
                'ablation_only_hp':True}
            (old/'protocol.json').write_text(json.dumps(old_protocol))
            (hp/'protocol.json').write_text(json.dumps(hp_protocol))
            losses={'B_tau0p20':1.09,'B_tau0p30':1.04,'F_tauinit1':1.12,
                    'G_tauinit0p30':1.02,'S_tauinit1':1.055,
                    'H_tauinit0p30':1.005,'P_tau0p30':1.045}
            for root,pp in ((old,old_protocol),(hp,hp_protocol)):
                for T in proto['contexts']:
                    for seed in proto['seeds']:
                        folder=root/f'T{T}'/f'seed{seed}'
                        folder.mkdir(parents=True)
                        for arm in pp['arms']:
                            name=arm['id']; variant=arm['variant']
                            r={'settings':{'seed':seed,'seq_len':T,'steps':proto['steps_by_context'][str(T)],
                                           'batch_size':proto['batch_by_context'][str(T)],'eval_seed':9090,
                                           'val_batches':128},
                               'dataset_sha256':{'train_sha256':'trainsha','validation_sha256':'valsha'},
                               'results':[{'variant':variant,'validation_nll_nats':losses[name]+.000001*seed,
                                           'training_tokens_per_second':39000.,'parameters':5008664,
                                           'speed':{'forward_tokens_per_second':130000.}}]}
                            (folder/(name+'.json')).write_text(json.dumps(r))
                            (folder/(name+'_'+variant+'.pt')).touch()
            report=calculate(old,hp)
            self.assertEqual(len(report['records']),2*5*5)
            self.assertEqual(report['contexts']['192']['best_original_dev_softmax'],'G_tauinit0p30')
            self.assertAlmostEqual(report['contexts']['192']['comparisons']['G_tauinit0p30_minus_H_tauinit0p30']['mean_delta_nats'],.015,places=7)
            self.assertAlmostEqual(report['contexts']['512']['comparisons']['B_tau0p30_minus_P_tau0p30']['mean_delta_nats'],-.005,places=7)
            self.assertEqual(report['contexts']['192']['comparisons']['G_tauinit0p30_minus_H_tauinit0p30']['positive_count'],5)

if __name__=='__main__':
    unittest.main()
