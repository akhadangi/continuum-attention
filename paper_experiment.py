"""Reproducible matched ~5M byte-level Transformer study; only orchestration loops.

All variants use the proven attention/model/optimizer modules. No holdout is
read or used by this file. Trained weights are checkpointed for later blind test.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_TRAIN='7708581afbaf4754daafb34839ced934da37eb3a58ce0daf37e84297bd0e7110'
EXPECTED_VAL='2367cba93ff5aed65961689187c708e59a890f0803249914b8fc0ef6515d9972'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def parse_csv(text,cast):
    out=[cast(t.strip()) for t in text.split(',') if t.strip()]
    if not out or len(out)!=len(set(out)):
        raise ValueError('Expected a nonempty comma-separated list with unique entries')
    return out


def arms_for(taus, include_h=False, include_p=False):
    arms=[{'id':'B_tau'+f'{tau:.2f}'.replace('.','p'), 'variant':'B','fixed_tau':tau} for tau in taus]
    arms += [{'id':'F_tauinit1', 'variant':'F', 'fixed_tau':.30},
             {'id':'G_tauinit0p30','variant':'G','fixed_tau':.30},
             {'id':'S_tauinit1','variant':'S','fixed_tau':.30}]
    if include_h:
        arms.append({'id':'H_tauinit0p30','variant':'H','fixed_tau':.30})
    if include_p:
        arms.append({'id':'P_tau0p30','variant':'P','fixed_tau':.30})
    return arms


def main():
    ap=argparse.ArgumentParser(description='Paired Mac CA vs properly tuned softmax study')
    ap.add_argument('--contexts',default='192,512')
    ap.add_argument('--seeds',default='1729,2718,3141,4567,6174')
    ap.add_argument('--temperature-grid',default='0.20,0.30,0.50')
    ap.add_argument('--only-hp',action='store_true',help='Train ONLY H (CA learned tau init 0.30) and P (CA fixed tau 0.30); never retrain B/F/G/S')
    ap.add_argument('--include-ca-init03',action='store_true',help='Add optional CA learned-scale warm-start H')
    ap.add_argument('--include-fixed-ca',action='store_true',help='Add optional CA with fixed tau .30 (P)')
    ap.add_argument('--cache-dir',default=str(Path.home()/'Downloads/focal_attention_lab/data'))
    ap.add_argument('--output',default='results/ca_paper_study')
    ap.add_argument('--backend',choices=['auto','reference','metal_analytic','metal_long'],default='auto')
    ap.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    ap.add_argument('--eval-seed',type=int,default=9090)
    ap.add_argument('--val-batches',type=int,default=128)
    ap.add_argument('--probe-reps',type=int,default=20)
    ap.add_argument('--log-every',type=int,default=250)
    ap.add_argument('--steps-192',type=int,default=2500)
    ap.add_argument('--steps-512',type=int,default=3750)
    ap.add_argument('--batch-192',type=int,default=8)
    ap.add_argument('--batch-512',type=int,default=2)
    ap.add_argument('--lr',type=float,default=4e-4)
    ap.add_argument('--geometry-lr-mult',type=float,default=8.)
    ap.add_argument('--allow-nonstandard-data',action='store_true',help='ONLY for local smoke testing; prevents paper-valid label')
    ap.add_argument('--train-file',default=None)
    ap.add_argument('--val-file',default=None)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    contexts=parse_csv(args.contexts,int)
    if any(T not in (192,512) for T in contexts):
        ap.error('Paper protocol contexts must be 192 and/or 512')
    seeds=parse_csv(args.seeds,int)
    taus=parse_csv(args.temperature_grid,float)
    if any(t<=0 or t>2 for t in taus):
        ap.error('Fixed temperature must be in (0,2]')
    if args.eval_seed is None or args.val_batches<1:
        ap.error('Evaluation seed and val-batches must be fixed')
    train=Path(args.train_file).expanduser() if args.train_file else Path(args.cache_dir).expanduser()/ 'tinystories_train_16000000.bin'
    val=Path(args.val_file).expanduser() if args.val_file else Path(args.cache_dir).expanduser()/ 'tinystories_validation_2000000.bin'
    if not train.is_file() or not val.is_file():
        ap.error(f'Missing dataset cache: {train} or {val}. Run benchmark.py --prepare-only first.')
    tr_sha,vl_sha=digest(train),digest(val)
    recognized=(tr_sha==EXPECTED_TRAIN and vl_sha==EXPECTED_VAL)
    if not recognized and not args.allow_nonstandard_data:
        ap.error('Dataset SHA256 does not match the historical TinyStories cache! Use --allow-nonstandard-data only for tests.')
    if args.only_hp and (args.include_ca_init03 or args.include_fixed_ca):
        ap.error('With --only-hp, omit --include-ca-init03 and --include-fixed-ca: H and P are already selected')
    arms=([{'id':'H_tauinit0p30','variant':'H','fixed_tau':.30},
           {'id':'P_tau0p30','variant':'P','fixed_tau':.30}] if args.only_hp else
          arms_for(taus,args.include_ca_init03,args.include_fixed_ca))
    out=Path(args.output).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    cfg={'schema_version':1,'contexts':contexts,'seeds':seeds,'arms':arms,
         'data_sha256':{'train':tr_sha,'validation':vl_sha},
         'paper_valid_data':recognized,
         'eval_seed':args.eval_seed,'val_batches':args.val_batches,
         'steps_by_context':{'192':args.steps_192,'512':args.steps_512},
         'batch_by_context':{'192':args.batch_192,'512':args.batch_512},
         'lr':args.lr,'geometry_lr_mult':args.geometry_lr_mult,
         'backend':args.backend,'probe_reps':args.probe_reps}
    if args.only_hp:
        cfg['ablation_only_hp']=True
    config_path=out/'protocol.json'
    if config_path.exists():
        prev=json.loads(config_path.read_text())
        if prev!=cfg:
            raise RuntimeError(f'Protocol changed in existing output directory {out}. Choose a NEW --output folder; do not mix runs.')
    else:
        config_path.write_text(json.dumps(cfg,indent=2)+'\n')
    script=Path(__file__).resolve().with_name('benchmark.py')
    for T in contexts:
        backend=('metal_analytic' if T==192 else 'metal_long') if args.backend=='auto' else args.backend
        steps=getattr(args,'steps_'+str(T))
        batch=getattr(args,'batch_'+str(T))
        if steps<=0 or batch<=0:
            ap.error('Steps and batch must be positive')
        for seed in seeds:
            folder=out/f'T{T}'/f'seed{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            for arm in arms:
                name=arm['id']
                filepath=folder/(name+'.json')
                ckpt=folder/(name+'_'+arm['variant']+'.pt')
                if filepath.exists() and ckpt.exists():
                    previous=json.loads(filepath.read_text())
                    if (previous.get('settings',{}).get('seed')!=seed or
                            previous.get('settings',{}).get('seq_len')!=T or
                            previous.get('settings',{}).get('steps')!=steps or
                            previous.get('results',[{}])[0].get('variant')!=arm['variant'] or
                            abs(previous.get('settings',{}).get('fixed_tau',0)-arm['fixed_tau'])>1e-8):
                        raise RuntimeError(f'Refusing to reuse mismatched run: {filepath}')
                    print(f'SKIP verified existing: {filepath}',flush=True)
                    continue
                if filepath.exists() or ckpt.exists():
                    raise RuntimeError(f'Incomplete run found ({filepath}, {ckpt}); inspect/rename before re-running')
                cmd=[sys.executable,str(script),'--models',arm['variant'],
                     '--backend',backend,'--device',args.device,'--fixed-tau',str(arm['fixed_tau']),
                     '--steps',str(steps),'--batch-size',str(batch),'--seq-len',str(T),
                     '--val-batches',str(args.val_batches),'--eval-seed',str(args.eval_seed),
                     '--seed',str(seed),'--lr',str(args.lr),
                     '--geometry-lr-mult',str(args.geometry_lr_mult),
                     '--probe-reps',str(args.probe_reps),'--log-every',str(args.log_every),
                     '--save-checkpoints','--out',str(filepath)]
                # Keep the historical TinyStories dataset label when using the
                # validated standard cache. Custom synthetic text keeps its
                # explicit local_text label, never masquerading as TinyStories.
                if args.train_file or args.val_file:
                    cmd.extend(['--train-file',str(train),'--val-file',str(val)])
                else:
                    cmd.extend(['--cache-dir',str(Path(args.cache_dir).expanduser())])
                print('\nRUN:',T,seed,name,'tokens=',steps*batch*T,flush=True)
                if args.dry_run:
                    print(' '.join(cmd),flush=True)
                    continue
                subprocess.run(cmd,cwd=str(script.parent),check=True)
                if not (filepath.is_file() and ckpt.is_file()):
                    raise RuntimeError(f'Output checkpoint missing: {filepath} / {ckpt}')
    if args.only_hp:
        print('H/P TRAINING COMPLETE. Compare against existing controls using summarize_hp.py.')
    else:
        print('TRAINING PROTOCOL COMPLETE. Analyze with summarize_paper.py (no holdout yet).')

if __name__=='__main__':
    main()
