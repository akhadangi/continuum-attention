"""One-time heldout confirmation of development-selected standard attention vs CA.

Neither model nor hyperparameters may be changed here. The frozen selection is
read from disk; all seeds and contexts use exactly the same holdout starts.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from benchmark import build, choose_device, evaluate, learned_tau
from paper_experiment import digest
from summarize_paper import paired_stats

EXPECTED_HOLDOUT='d70cb8d77cfdb50736f37bea5e81a4958e6a1f69bba6630c98d16a65a5ebeaec'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--study',default='results/ca_paper_study')
    ap.add_argument('--holdout-file',default=str(Path.home()/'Downloads/acma_scale_lab/data/validation_heldout_next4m.bin'))
    ap.add_argument('--holdout-batches',type=int,default=256)
    ap.add_argument('--holdout-seed',type=int,default=121212)
    ap.add_argument('--device',default='auto')
    ap.add_argument('--allow-custom-holdout',action='store_true')
    args=ap.parse_args()
    root=Path(args.study).expanduser().resolve()
    selection=json.loads((root/'FROZEN_SELECTION.json').read_text())
    proto=json.loads((root/'protocol.json').read_text())
    if selection['protocol']!=proto:
        raise RuntimeError('Protocol differs from frozen development selection')
    if not proto['paper_valid_data']:
        raise RuntimeError('Nonstandard training corpus; cannot produce paper confirmation under this protocol')
    target=root/'FINAL_HOLDOUT_RESULTS.json'
    if target.exists():
        raise RuntimeError(f'Heldout results already exist: {target}. Do not repeatedly inspect the sealed test.')
    holdout_path=Path(args.holdout_file).expanduser().resolve()
    if not holdout_path.is_file():
        raise FileNotFoundError(holdout_path)
    h_sha=digest(holdout_path)
    if not args.allow_custom_holdout and h_sha!=EXPECTED_HOLDOUT:
        raise RuntimeError('Holdout SHA mismatch! Expected previously-created disjoint 4 MB validation tail; use --allow-custom-holdout only for an intentional different corpus.')
    if args.holdout_batches < 1:
        ap.error('--holdout-batches must be >= 1')
    data=np.memmap(holdout_path,mode='r',dtype=np.uint8)
    device=choose_device(args.device)
    entries=[]
    print('SEALED HOLDOUT: comparing FROZEN development-selected softmax with S.',flush=True)
    print(f'Holdout SHA256={h_sha}, device={device}',flush=True)
    for T in proto['contexts']:
        backend=('metal_analytic' if T==192 else 'metal_long') if proto['backend']=='auto' else proto['backend']
        winner=selection['selected_controls'][str(T)]
        arm_lookup={a['id']:a for a in proto['arms']}
        for seed in proto['seeds']:
            for name in (winner,selection['CA_arm']):
                arm=arm_lookup[name]
                checkpoint=root/f'T{T}'/f'seed{seed}'/(name+'_'+arm['variant']+'.pt')
                if not checkpoint.exists():
                    raise RuntimeError('Missing selected-model checkpoint: '+str(checkpoint))
                state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if any([state['variant']!=arm['variant'],state['seq_len']!=T,
                        state['seed']!=seed,state['steps']!=proto['steps_by_context'][str(T)],
                        abs(state['fixed_tau']-arm['fixed_tau'])>1e-8]):
                    raise RuntimeError('Checkpoint metadata does not match frozen protocol: '+str(checkpoint))
                runargs=SimpleNamespace(seed=seed,seq_len=T,
                                        batch_size=proto['batch_by_context'][str(T)],
                                        width=256,hidden=720,heads=4,
                                        fixed_tau=arm['fixed_tau'],backend=backend,
                                        eval_seed=proto['eval_seed'],
                                        holdout_eval_seed=args.holdout_seed,
                                        val_batches=proto['val_batches'])
                model=build(runargs,arm['variant'],device)
                model.load_state_dict(state['state_dict'],strict=True)
                nll=evaluate(model,data,runargs,device,seed_offset=2003,batches=args.holdout_batches)
                entry={'context':T,'seed':seed,'arm':name,'variant':arm['variant'],
                       'holdout_nll':float(nll),'holdout_bpb':float(nll/np.log(2)),
                       'checkpoint':str(checkpoint.relative_to(root))}
                entries.append(entry)
                print(f'T={T} seed={seed} {name} heldout NLL={nll:.6f}',flush=True)
                del model
    outcome={}
    for T in proto['contexts']:
        winner=selection['selected_controls'][str(T)]
        diffs=[]
        for seed in proto['seeds']:
            rows={e['arm']:e for e in entries if e['context']==T and e['seed']==seed}
            diffs.append(rows[winner]['holdout_nll']-rows[selection['CA_arm']]['holdout_nll'])
        outcome[str(T)]={'best_development_softmax':winner,
                         'paired_softmax_minus_CA_holdout':paired_stats(diffs)}
    result={'holdout_sha256':h_sha,'holdout_path':str(holdout_path),
            'holdout_eval_seed':args.holdout_seed,'holdout_batches':args.holdout_batches,
            'frozen_selection':selection, 'runs':entries,'results':outcome,
            'statistical_caveat':'Student-t CI is across training seeds with a shared sampled heldout slice, not a corpus-generalization CI.'}
    target.write_text(json.dumps(result,indent=2)+'\n')
    print('\nFINAL HOLDOUT RESULT:',json.dumps(outcome,indent=2),flush=True)
    print('Recorded:',target)

if __name__=='__main__':
    main()
