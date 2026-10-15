"""Paired seed-level Student-t intervals; freeze dev-selected strongest control.

This is an exploratory small-n interval, not a population guarantee or a
corpus-independent confidence interval. Holdout is not accessed here.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from scipy.stats import t as student_t


def paired_stats(x):
    vals=list(map(float,x))
    n=len(vals)
    mu=statistics.mean(vals)
    sd=statistics.stdev(vals) if n>1 else None
    interval=student_t.ppf(.975,n-1)*sd/math.sqrt(n) if n>1 else None
    return {'n':n,'mean_delta_nats':mu,'sample_sd':sd,
            'student_t_95pct_CI':([mu-interval,mu+interval] if interval is not None else None),
            'positive_count':sum(v>0 for v in vals),
            'method':'paired seed-level Student-t; small-sample and seed-only uncertainty'}


def summarize(root,require_complete=True):
    root=Path(root)
    protocol=json.loads((root/'protocol.json').read_text())
    records=[]
    for T in protocol['contexts']:
        for seed in protocol['seeds']:
            for arm in protocol['arms']:
                path=root/f'T{T}'/f'seed{seed}'/(arm['id']+'.json')
                if not path.exists():
                    if require_complete:
                        raise RuntimeError(f'Missing planned result {path}; cannot freeze baseline.')
                    continue
                obj=json.loads(path.read_text())
                assert obj['settings']['seq_len']==T and obj['settings']['seed']==seed
                assert obj['settings']['eval_seed']==protocol['eval_seed']
                assert obj['dataset_sha256']['train_sha256']==protocol['data_sha256']['train']
                assert obj['dataset_sha256']['validation_sha256']==protocol['data_sha256']['validation']
                result=obj['results'][0]
                assert result['variant']==arm['variant']
                records.append({'context':T,'seed':seed,'arm':arm['id'],
                                'variant':arm['variant'],
                                'nll':result['validation_nll_nats'],
                                'train_tps':result['training_tokens_per_second'],
                                'forward_tps':result['speed']['forward_tokens_per_second'],
                                'params':result['parameters'],
                                'file':str(path.relative_to(root))})
    output={'protocol':protocol,'records':records,'contexts':{}}
    for T in protocol['contexts']:
        subset=[r for r in records if r['context']==T]
        by_arm={arm['id']:{r['seed']:r for r in subset if r['arm']==arm['id']} for arm in protocol['arms']}
        complete=[id for id,seed_map in by_arm.items() if len(seed_map)==len(protocol['seeds'])]
        controls=[arm['id'] for arm in protocol['arms'] if arm['variant'] in ('B','F','G') and arm['id'] in complete]
        if not controls or 'S_tauinit1' not in complete:
            if require_complete:
                raise RuntimeError(f'Incomplete baseline or S for T={T}')
            continue
        means={key:statistics.mean(r['nll'] for r in by_arm[key].values()) for key in complete}
        best_control=min(controls,key=lambda key:means[key])
        intervals={}
        for control in controls:
            deltas=[by_arm[control][seed]['nll']-by_arm['S_tauinit1'][seed]['nll']
                    for seed in protocol['seeds']]
            intervals[control]=paired_stats(deltas)
        output['contexts'][str(T)]={
            'arm_mean_dev_nll':means,
            'dev_selected_best_softmax':best_control,
            'paired_control_minus_S':intervals,
            'S_training_tps_mean':statistics.mean(r['train_tps'] for r in by_arm['S_tauinit1'].values()),
            'best_control_training_tps_mean':statistics.mean(r['train_tps'] for r in by_arm[best_control].values()),
            'S_params':next(iter(by_arm['S_tauinit1'].values()))['params'],
            'control_params':next(iter(by_arm[best_control].values()))['params'],
            'note':'Selected from development validation only. Final heldout remains sealed.'}
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--study',default='results/ca_paper_study')
    ap.add_argument('--allow-incomplete',action='store_true')
    ap.add_argument('--freeze',action='store_true',help='Save immutable control selections for heldout once all runs complete')
    args=ap.parse_args()
    study=Path(args.study).expanduser().resolve()
    result=summarize(study,not args.allow_incomplete)
    (study/'development_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with open(study/'development_runs.csv','w',newline='') as f:
        cols=['context','seed','arm','variant','nll','train_tps','forward_tps','params','file']
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader()
        writer.writerows(result['records'])
    print('\nDEV STUDY: comparison of CA-S versus all softmax controls')
    for T,group in result['contexts'].items():
        winner=group['dev_selected_best_softmax']
        print(f'\nT={T} best development softmax: {winner}')
        for arm,stat in group['paired_control_minus_S'].items():
            ci=stat['student_t_95pct_CI']
            print(f" {arm:16s} mean(F-S)={stat['mean_delta_nats']:+.6f} "
                  f"95% paired t-CI={ci} wins={stat['positive_count']}/{stat['n']}")
    if args.freeze:
        if args.allow_incomplete or len(result['contexts'])!=len(result['protocol']['contexts']):
            ap.error('Cannot freeze incomplete protocol')
        freeze_path=study/'FROZEN_SELECTION.json'
        selected={'schema_version':1,'protocol':result['protocol'],
                  'selected_controls':{T:group['dev_selected_best_softmax'] for T,group in result['contexts'].items()},
                  'CA_arm':'S_tauinit1','selection_basis':'Mean NLL on development validation; NO heldout examined'}
        if freeze_path.exists() and json.loads(freeze_path.read_text())!=selected:
            raise RuntimeError('FROZEN_SELECTION.json differs: not allowed to change after freezing')
        freeze_path.write_text(json.dumps(selected,indent=2)+'\n')
        print('\nDEVELOPMENT SELECTION FROZEN:',freeze_path)
    print('\nSaved:',study/'development_summary.json')

if __name__=='__main__':
    main()
