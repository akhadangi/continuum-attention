"""Exploratory follow-up: evaluate ONLY new H and P checkpoints on REUSED holdout.

The old holdout has already been opened. Do not describe these measurements as
an independent sealed confirmation. Reuse the exact evaluation positions,
seed, batches, and dataset digest recorded by the original holdout study.
Never retrain or even re-evaluate B/F/G/S here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmark import build, choose_device, evaluate
from paper_experiment import digest
from summarize_hp import check_protocols
from summarize_paper import paired_stats

EXPECTED_HOLDOUT='d70cb8d77cfdb50736f37bea5e81a4958e6a1f69bba6630c98d16a65a5ebeaec'
NEW_IDS={'H_tauinit0p30':'H','P_tau0p30':'P'}


def main():
    ap=argparse.ArgumentParser(description='Evaluate only H/P on previously opened holdout (EXPLORATORY)')
    ap.add_argument('--baseline-study',default='results/ca_5seed_confirm')
    ap.add_argument('--hp-study',default='results/ca_hp_only')
    ap.add_argument('--holdout-file',default=str(Path.home()/'Downloads/acma_scale_lab/data/validation_heldout_next4m.bin'))
    ap.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    args=ap.parse_args()
    oldroot=Path(args.baseline_study).expanduser().resolve()
    hproot=Path(args.hp_study).expanduser().resolve()
    old=json.loads((oldroot/'protocol.json').read_text())
    hp=json.loads((hproot/'protocol.json').read_text())
    check_protocols(old,hp)
    original_path=oldroot/'FINAL_HOLDOUT_RESULTS.json'
    if not original_path.exists():
        ap.error(f'Original holdout results missing: {original_path}. Cannot ensure identical evaluation.')
    original=json.loads(original_path.read_text())
    select=json.loads((oldroot/'FROZEN_SELECTION.json').read_text())
    if select['protocol']!=old:
        raise ValueError('Original frozen model-selection protocol differs from development protocol')
    holdout=Path(args.holdout_file).expanduser().resolve()
    if not holdout.exists():
        ap.error(f'Holdout file not found: {holdout}')
    file_sha=digest(holdout)
    if file_sha!=EXPECTED_HOLDOUT or file_sha!=original['holdout_sha256']:
        ap.error('Holdout SHA256 mismatch: refusing to mix datasets.')
    seed=int(original['holdout_eval_seed'])
    batches=int(original['holdout_batches'])
    if seed<0 or batches<1:
        ap.error('Original holdout sampling settings invalid')
    target=hproot/'HP_HOLDOUT_EXPLORATORY.json'
    if target.exists():
        ap.error(f'Results already exist; refusing to overwrite: {target}')
    arr=np.memmap(holdout,dtype=np.uint8,mode='r')
    device=choose_device(args.device)
    entries=[]
    print('REUSED HOLDOUT: H and P only; exploratory because this heldout was previously examined.',flush=True)
    print(f'Validation holdout sha256: {file_sha}; seed={seed}, batches={batches}',flush=True)
    for T in hp['contexts']:
        backend=('metal_analytic' if T==192 else 'metal_long') if hp['backend']=='auto' else hp['backend']
        for training_seed in hp['seeds']:
            for arm_name,variant in NEW_IDS.items():
                arm=next(a for a in hp['arms'] if a['id']==arm_name)
                checkpoint=hproot/f'T{T}'/f'seed{training_seed}'/(arm_name+'_'+variant+'.pt')
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if (ckpt['seed']!=training_seed or ckpt['seq_len']!=T or
                    ckpt['variant']!=variant or
                    ckpt['steps']!=hp['steps_by_context'][str(T)] or
                    abs(float(ckpt['fixed_tau'])-float(arm['fixed_tau']))>1e-8):
                    raise ValueError(f'Incompatible checkpoint: {checkpoint}')
                runargs=SimpleNamespace(seed=training_seed,seq_len=T,
                    batch_size=hp['batch_by_context'][str(T)],width=256,hidden=720,heads=4,
                    fixed_tau=arm['fixed_tau'],backend=backend,eval_seed=hp['eval_seed'],
                    holdout_eval_seed=seed,val_batches=hp['val_batches'])
                model=build(runargs,variant,device)
                model.load_state_dict(ckpt['state_dict'],strict=True)
                nll=evaluate(model,arr,runargs,device,seed_offset=2003,batches=batches)
                entries.append({'context':T,'seed':training_seed,'arm':arm_name,
                                'variant':variant,'holdout_nll':float(nll),
                                'holdout_bpb':float(nll/np.log(2)),
                                'checkpoint':str(checkpoint.relative_to(hproot))})
                print(f'T={T} seed={training_seed} {arm_name}: holdout NLL {nll:.6f}',flush=True)
                del model
    original_by_key={(int(row['context']),int(row['seed']),row['arm']):row
                     for row in original['runs']}
    new_by_key={(row['context'],row['seed'],row['arm']):row for row in entries}
    outcomes={}
    for T in hp['contexts']:
        winner=select['selected_controls'][str(T)]
        comparisons={}
        for control in (winner,'S_tauinit1'):
            for ca_arm in NEW_IDS:
                deltas=[]
                for training_seed in hp['seeds']:
                    baseline=original_by_key[(T,training_seed,control)]['holdout_nll']
                    candidate=new_by_key[(T,training_seed,ca_arm)]['holdout_nll']
                    deltas.append(float(baseline)-float(candidate))
                comparisons[f'{control}_minus_{ca_arm}']=paired_stats(deltas)
        outcomes[str(T)]={'original_selected_softmax':winner,'comparisons':comparisons}
    output={'status':'EXPLORATORY - reused opened holdout, NOT independent confirmation',
            'original_holdout_sha256':file_sha,'original_holdout_seed':seed,
            'original_holdout_batches':batches,
            'original_study':str(oldroot),'hp_study':str(hproot),
            'new_hp_runs':entries,'contexts':outcomes}
    target.write_text(json.dumps(output,indent=2)+'\n')
    print('\nEXPLORATORY HOLDOUT COMPARISONS',json.dumps(outcomes,indent=2))
    print('\nSaved',target)

if __name__=='__main__':
    main()
