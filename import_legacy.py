"""Import a directory/list of older bench JSONs without inventing missing seeds.

Supports prior F/S 512 quality outputs and earlier 192 runs. Paired intervals
only appear when TWO or more independent training seeds are actually available.
"""
import argparse
import json
from pathlib import Path
from summarize_paper import paired_stats


def import_records(paths):
    collected=[]
    for path in paths:
        obj=json.loads(Path(path).read_text())
        results={r['variant']:r for r in obj.get('results',[])}
        for name,r in results.items():
            if 'validation_nll_nats' not in r:
                continue
            collected.append({'path':str(path),'context':obj['settings']['seq_len'],
                              'seed':obj['settings']['seed'],'variant':name,
                              'validation_nll':r['validation_nll_nats'],
                              'train_tokens':r['train_tokens'],
                              'train_tps':r['training_tokens_per_second'],
                              'data_sha256':obj.get('dataset_sha256'),
                              'eval_seed':obj['settings'].get('eval_seed'),
                              'holdout_nll':r.get('holdout_nll_nats')})
    pairs={}
    for T in sorted({r['context'] for r in collected}):
        group=[r for r in collected if r['context']==T]
        seeds=sorted(set(r['seed'] for r in group))
        diffs=[]
        for seed in seeds:
            rows={r['variant']:r for r in group if r['seed']==seed}
            if 'F' in rows and 'S' in rows:
                if rows['F']['train_tokens']!=rows['S']['train_tokens'] or rows['F']['eval_seed']!=rows['S']['eval_seed']:
                    raise ValueError('Incompatible token budget or evaluation sampling for F/S')
                diffs.append((seed,rows['F']['validation_nll']-rows['S']['validation_nll']))
        pairs[str(T)]={'per_seed_deltas':diffs,'seed_level_interval':paired_stats([d for s,d in diffs]) if diffs else None}
    return {'imported_records':collected,'paired_by_context':pairs,
            'note':'Legacy results are not automatically commensurate with the NEW protocol; reports stay separate.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('paths',nargs='+')
    ap.add_argument('--out',default='legacy_import.json')
    args=ap.parse_args()
    output=import_records(args.paths)
    Path(args.out).write_text(json.dumps(output,indent=2)+'\n')
    for T,x in output['paired_by_context'].items():
        print('Context',T,'paired seeds:',x['per_seed_deltas'], 'interval:',x['seed_level_interval'])
    print('Saved',args.out)

if __name__=='__main__':main()
