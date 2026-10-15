"""Compare newly trained H and P against ORIGINAL, already trained B/G/S.

NO model training, NO holdout access, NO baseline re-selection using holdout.
All matched seeds, data hashes, training settings and evaluation seeds are checked.
Paired seed-level Student-t CIs use the same method as summarize_paper.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from summarize_paper import paired_stats, summarize

NEW_IDS = {'H_tauinit0p30': 'H', 'P_tau0p30': 'P'}
OLD_IDS = {'B_tau0p30': 'B', 'G_tauinit0p30': 'G', 'S_tauinit1': 'S'}


def check_protocols(old, hp):
    if not hp.get('ablation_only_hp'):
        raise ValueError('New study must be created with --only-hp')
    if {a['id']: a['variant'] for a in hp['arms']} != NEW_IDS:
        raise ValueError('New study must contain exactly H and P')
    old_ids = {a['id']: a['variant'] for a in old['arms']}
    if any(old_ids.get(name) != variant for name, variant in OLD_IDS.items()):
        raise ValueError('The original study does not contain matching B@0.30, G@init0.30, and S')
    for field in ('contexts','seeds','data_sha256','paper_valid_data','eval_seed',
                  'val_batches','steps_by_context','batch_by_context','lr',
                  'geometry_lr_mult'):
        if old[field] != hp[field]:
            raise ValueError(f'Protocols differ at {field}: original {old[field]} vs H/P {hp[field]}')
    if not old['paper_valid_data']:
        raise ValueError('Original dataset hashes were not official benchmark fingerprints')
    old_arms={a['id']:a for a in old['arms']}
    hp_arms={a['id']:a for a in hp['arms']}
    for name in ('B_tau0p30', 'G_tauinit0p30'):
        if abs(old_arms[name]['fixed_tau']-.3)>1e-8:
            raise ValueError(f'{name} not at tau .30')
    for name in NEW_IDS:
        if abs(hp_arms[name]['fixed_tau']-.3)>1e-8:
            raise ValueError(f'{name} not at tau .30')


def get_record(root, protocol, context, seed, name, variant):
    root=Path(root)
    json_path=root/f'T{context}'/f'seed{seed}'/(name+'.json')
    pt_path=root/f'T{context}'/f'seed{seed}'/(name+'_'+variant+'.pt')
    if not json_path.is_file() or not pt_path.is_file():
        raise FileNotFoundError(f'Incomplete training run: {json_path} or {pt_path}')
    d=json.loads(json_path.read_text())
    st=d['settings']; rr=d['results']
    if len(rr)!=1 or rr[0]['variant']!=variant:
        raise ValueError(f'Wrong variant recorded: {json_path}')
    expected={'seed':seed,'seq_len':context,'steps':protocol['steps_by_context'][str(context)],
              'batch_size':protocol['batch_by_context'][str(context)],
              'eval_seed':protocol['eval_seed'],'val_batches':protocol['val_batches']}
    for key, v in expected.items():
        if st.get(key)!=v:
            raise ValueError(f'{json_path}: {key}={st.get(key)} != {v}')
    hashes=d['dataset_sha256']
    if hashes['train_sha256']!=protocol['data_sha256']['train'] or hashes['validation_sha256']!=protocol['data_sha256']['validation']:
        raise ValueError(f'Dataset mismatch: {json_path}')
    return {'nll':float(rr[0]['validation_nll_nats']),
            'train_tps':float(rr[0]['training_tokens_per_second']),
            'forward_tps':float(rr[0]['speed']['forward_tokens_per_second']),
            'params':int(rr[0]['parameters']),
            'checkpoint':str(pt_path)}


def calculate(original_root, hp_root):
    original_root=Path(original_root).resolve()
    hp_root=Path(hp_root).resolve()
    old=json.loads((original_root/'protocol.json').read_text())
    hp=json.loads((hp_root/'protocol.json').read_text())
    check_protocols(old,hp)
    old_summary=summarize(original_root,require_complete=True)
    output={'note':'Post hoc exploratory CA scale-initialization ablation. No new sealed test.',
            'original_study':str(original_root),'hp_study':str(hp_root),
            'contexts':{},'records':[]}
    for T in hp['contexts']:
        by_id={name:{} for name in (*OLD_IDS,*NEW_IDS)}
        for seed in hp['seeds']:
            for name,variant in {**OLD_IDS,**NEW_IDS}.items():
                src=hp_root if name in NEW_IDS else original_root
                row=get_record(src, hp if name in NEW_IDS else old,T,seed,name,variant)
                by_id[name][seed]=row
                output['records'].append({'context':T,'seed':seed,'arm':name,'variant':variant,
                                           'source':'H/P new' if name in NEW_IDS else 'original',**row})
        means={name:statistics.mean(by_id[name][s]['nll'] for s in hp['seeds']) for name in by_id}
        selected=old_summary['contexts'][str(T)]['dev_selected_best_softmax']
        comparisons={}
        pairs=[('G_tauinit0p30','H_tauinit0p30'),
               ('B_tau0p30','P_tau0p30'),
               ('S_tauinit1','H_tauinit0p30'),
               ('S_tauinit1','P_tau0p30'),
               ('H_tauinit0p30','P_tau0p30'),
               (selected,'H_tauinit0p30'),
               (selected,'P_tau0p30')]
        for baseline,candidate in pairs:
            key=f'{baseline}_minus_{candidate}'
            if key not in comparisons:
                comparisons[key]=paired_stats([by_id[baseline][s]['nll']-by_id[candidate][s]['nll']
                                              for s in hp['seeds']])
        output['contexts'][str(T)]={'means_nll':means,
                                    'best_original_dev_softmax':selected,
                                    'best_hp_on_dev':min(NEW_IDS,key=means.get),
                                    'comparisons':comparisons,
                                    'mean_train_tps':{name:statistics.mean(by_id[name][s]['train_tps'] for s in hp['seeds']) for name in by_id},
                                    'mean_forward_tps':{name:statistics.mean(by_id[name][s]['forward_tps'] for s in hp['seeds']) for name in by_id}}
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline-study',default='results/ca_5seed_confirm')
    p.add_argument('--hp-study',default='results/ca_hp_only')
    args=p.parse_args()
    result=calculate(args.baseline_study,args.hp_study)
    out=Path(args.hp_study).resolve()
    (out/'HP_COMPARISON_DEVELOPMENT.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'HP_COMPARISON_DEVELOPMENT.csv').open('w',newline='') as f:
        cols=['context','seed','arm','variant','source','nll','train_tps','forward_tps','params','checkpoint']
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader();writer.writerows(result['records'])
    for T,group in result['contexts'].items():
        print(f'\nT={T}, best original softmax={group["best_original_dev_softmax"]}')
        for name,val in group['means_nll'].items():
            print(f'  {name:17s} mean dev NLL = {val:.6f}')
        for name,stats in group['comparisons'].items():
            print(f'  {name}: mean baseline - candidate = {stats["mean_delta_nats"]:+.6f}, '
                  f'95% CI={stats["student_t_95pct_CI"]}, '
                  f'baseline worse on {stats["positive_count"]}/{stats["n"]} seeds')
    print(f'\nSaved {out/"HP_COMPARISON_DEVELOPMENT.json"} and CSV.')

if __name__=='__main__':
    main()
