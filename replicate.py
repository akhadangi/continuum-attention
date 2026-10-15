"""Paired multi-seed TinyStories F/S benchmark orchestrator.

All iterations are OUTSIDE the model; no recurrent operator is introduced.
"""
import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def report(paths, output=None):
    entries=[]
    for path in paths:
        obj=json.loads(Path(path).read_text())
        rs={x['variant']:x for x in obj['results']}
        if not {'F','S'} <= rs.keys():
            raise ValueError(f'Missing F or S in {path}')
        f,s=rs['F'],rs['S']
        hashes=obj.get('dataset_sha256') or {}
        entries.append({
            'seed':obj['settings']['seed'],
            'steps':obj['settings']['steps'],
            'seq_len':obj['settings']['seq_len'],
            'dataset_sha256':hashes,
            'F_nll':f['validation_nll_nats'],
            'S_nll':s['validation_nll_nats'],
            'nll_F_minus_S':round(f['validation_nll_nats']-s['validation_nll_nats'],8),
            'F_train_tps':f['training_tokens_per_second'],
            'S_train_tps':s['training_tokens_per_second'],
            'F_holdout_nll':f.get('holdout_nll_nats'),
            'S_holdout_nll':s.get('holdout_nll_nats'),
        })
    fps={(e['dataset_sha256'].get('train_sha256'), e['dataset_sha256'].get('validation_sha256')) for e in entries}
    if len(fps)>1:
        raise ValueError('Dataset checksums differ between seeds; cannot compare')
    pairs=[e['nll_F_minus_S'] for e in entries]
    n=len(pairs)
    mean=statistics.mean(pairs)
    sd=statistics.stdev(pairs) if n>1 else None
    # Exploratory interval only; not a claim of normality at n=3.
    zbound=(1.96*sd/math.sqrt(n)) if sd is not None else None
    holdout_pairs=[e['F_holdout_nll']-e['S_holdout_nll'] for e in entries
                   if e['F_holdout_nll'] is not None and e['S_holdout_nll'] is not None]
    result={'runs':entries,'paired_nll_improvement_F_minus_S_mean':mean,
            'paired_nll_improvement_std':sd,
            'exploratory_normal_95pct_halfwidth':zbound,
            'S_wins':sum(d>0 for d in pairs),'total_seeds':n,
            'holdout_paired_F_minus_S_mean':statistics.mean(holdout_pairs) if holdout_pairs else None,
            'holdout_S_wins':sum(d>0 for d in holdout_pairs) if holdout_pairs else None,
            'note':'Seed-level uncertainty only; with few seeds, the normal-approximation interval is not a rigorous confidence interval.'}
    print('\nSEED      F_NLL       S_NLL       F-S          F t/s      S t/s')
    for e in entries:
        print(f"{e['seed']:6d}  {e['F_nll']:.6f}    {e['S_nll']:.6f}    {e['nll_F_minus_S']:+.6f}    {e['F_train_tps']:.0f}       {e['S_train_tps']:.0f}")
    print(f'Paired mean F-S NLL: {mean:+.6f} | S wins {result["S_wins"]}/{n}')
    if holdout_pairs:
        print(f'Holdout paired F-S NLL: {statistics.mean(holdout_pairs):+.6f} | S wins {sum(d>0 for d in holdout_pairs)}/{len(holdout_pairs)}')
    if output:
        Path(output).write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',default='1729,2718,3141')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--seq-len',type=int,default=192)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--backend',default='metal_analytic')
    p.add_argument('--cache-dir',default='../focal_attention_lab/data')
    p.add_argument('--val-batches',type=int,default=128)
    p.add_argument('--eval-seed',type=int,default=9090)
    p.add_argument('--holdout-file',default=None)
    p.add_argument('--holdout-batches',type=int,default=128)
    p.add_argument('--out-dir',default='replications')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    out=Path(args.out_dir)
    out.mkdir(exist_ok=True,parents=True)
    seeds=[int(x) for x in args.seeds.split(',')]
    files=[out/f'fs_seed_{seed}_T{args.seq_len}.json' for seed in seeds]
    if not args.summary_only:
        for seed,path in zip(seeds,files):
            cmd=[sys.executable,'benchmark.py','--models','F,S','--backend',args.backend,
                 '--steps',str(args.steps),'--batch-size',str(args.batch_size),
                 '--seq-len',str(args.seq_len),'--val-batches',str(args.val_batches),
                 '--eval-seed',str(args.eval_seed),'--seed',str(seed),
                 '--cache-dir',args.cache_dir,'--out',str(path)]
            if args.holdout_file:
                cmd.extend(['--holdout-file',args.holdout_file,'--holdout-batches',str(args.holdout_batches)])
            print('\nRUNNING:', ' '.join(cmd),flush=True)
            subprocess.run(cmd,check=True)
    report(files,out/f'paired_summary_T{args.seq_len}.json')


if __name__=='__main__':
    main()
