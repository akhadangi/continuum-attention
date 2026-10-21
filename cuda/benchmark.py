#!/usr/bin/env python3
"""CUDA v6 analytical correctness, causality and performance tests."""
import argparse
import json
import math
import os
import statistics
import time
import torch

from continuum_attention import (continuum_attention,
    ContinuumDecoderConfig, ContinuumDecoderLM)


def run_operator(mode,q,k,v,rho,dy,tile):
    os.environ['CA_CUDA_MODE']=mode
    y=continuum_attention(q,k,v,rho,tile)
    grads=torch.autograd.grad((y.float()*dy.float()).sum(),(q,k,v,rho))
    return y,*grads


def cmp(a,b,name,t=1e-3):
    a,b=a.detach().float(),b.detach().float()
    if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
        raise AssertionError(f'{name}: NaN or Inf')
    d=(a-b).abs()
    maximum=d.max().item()
    rms=d.norm().item()/max(a.norm().item(),1e-9)
    print('COMPARE',json.dumps({'name':name,'max_abs':maximum,'relative_l2':rms}),flush=True)
    if not (rms<=t):
        raise AssertionError(f'{name} mismatch (relative_l2={rms:.6g} > {t})')


def tests(args):
    print('GPU',torch.cuda.get_device_name(0),'Torch',torch.__version__,'CUDA',torch.version.cuda,flush=True)
    for T,magnitude in ((9,1.),(32,2.2),(129,1.),(192,2.5),(512,1.),(2048,1.)):
        torch.manual_seed(1633+T)
        q=torch.randn(1,2,T,64,device='cuda',dtype=torch.float16)*magnitude
        k=torch.randn_like(q)*magnitude
        v=torch.randn_like(q)
        rho=torch.tensor([0.0,math.log(1/.3)],device='cuda')
        q.requires_grad_(True);k.requires_grad_(True);v.requires_grad_(True);rho.requires_grad_(True)
        dy=torch.randn_like(q)
        ref=run_operator('prefix',q,k,v,rho,dy,128)
        v5=run_operator('prefix_v5',q,k,v,rho,dy,128)
        cand=run_operator('prefix_v6',q,k,v,rho,dy,128)
        for name,a,old,new in zip(('Y','dQ','dK','dV','dRho'),ref,v5,cand):
            cmp(a,new,f'T={T} magnitude={magnitude} {name} v6-v2',t=args.rel_tol)
            cmp(old,new,f'T={T} magnitude={magnitude} {name} v6-v5',t=args.rel_tol)
        print('NUMERICAL_CASE_PASSED',T,flush=True)
    # Only the operator can depend on earlier positions; future input variations
    # must not change its preceding outputs, and prefix loss must not backprop
    # to any future Q/K/V. These tests target actual T=2048 in native CUDA.
    T=2048
    q=torch.randn(1,2,T,64,device='cuda',dtype=torch.float16)
    k=torch.randn_like(q);v=torch.randn_like(q)
    rho=torch.tensor([0.0,0.2],device='cuda',requires_grad=True)
    os.environ['CA_CUDA_MODE']='prefix_v6'
    with torch.no_grad():
        y0=continuum_attention(q,k,v,rho,128)
        for cut in (0,127,128,1023,1024,2046):
            qp=q.clone();kp=k.clone();vp=v.clone()
            # large future changes, including values and keys
            qp[:,:,cut+1:,:]=torch.randn_like(qp[:,:,cut+1:,:])*2.5
            kp[:,:,cut+1:,:]=torch.randn_like(kp[:,:,cut+1:,:])*2.5
            vp[:,:,cut+1:,:]=torch.randn_like(vp[:,:,cut+1:,:])*2.5
            y1=continuum_attention(qp,kp,vp,rho,128)
            err=(y0[:,:,:cut+1,:]-y1[:,:,:cut+1,:]).abs().max().item()
            print('CAUSAL_OUTPUT',json.dumps({'T':T,'cut':cut,'max_abs':err}),flush=True)
            assert err==0.,'Future data changed preceding attention outputs'
    for cut in (0,127,128,1023,1024,2046):
        qt=q.clone().detach().requires_grad_(True)
        kt=k.clone().detach().requires_grad_(True)
        vt=v.clone().detach().requires_grad_(True)
        y=continuum_attention(qt,kt,vt,rho,128)
        gq,gk,gv=torch.autograd.grad(y[:,:,:cut+1,:].float().sum(),(qt,kt,vt))
        for name,g in (('Q',gq),('K',gk),('V',gv)):
            future=g[:,:,cut+1:,:]
            err=future.abs().max().item() if future.numel() else 0.
            print('CAUSAL_GRAD',json.dumps({'cut':cut,'variable':name,'future_max':err}),flush=True)
            assert err==0.,'Future token receives gradient from earlier outputs'
    print('V6_OPERATOR_NUMERICAL_CAUSALITY_PASSED',flush=True)


def model_causality(args):
    # Exact training architecture. Keep in FP32 storage and use FP16 autocast,
    # matching training. No model source/architecture changes.
    torch.manual_seed(4242)
    config=ContinuumDecoderConfig(block_size=2048,d_model=1024,n_heads=16,
        n_layers=8,mlp_ratio=4.,dropout=0.,ca_query_tile=128)
    model=ContinuumDecoderLM(config).cuda().eval()
    total=sum(p.numel() for p in model.parameters())
    assert total==101205120,total
    ids=torch.randint(0,256,(1,2048),device='cuda')
    os.environ['CA_CUDA_MODE']='prefix_v6'
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        baseline=model(ids)['logits']
        for cut in (127,1023,1024,2046):
            edited=ids.clone()
            edited[:,cut+1:] = torch.randint_like(edited[:,cut+1:],0,256)
            changed=model(edited)['logits']
            err=(baseline[:,:cut+1]-changed[:,:cut+1]).abs().max().item()
            print('DECODER_CAUSAL',json.dumps({'cut':cut,'max_abs':err}),flush=True)
            assert err==0.,'Full decoder violates causal future independence'
    print('V6_FULL_DECODER_CAUSALITY_PASSED',flush=True)


def timed(mode,q,k,v,rho,dy,tile,warmup,repeats):
    for _ in range(warmup):
        run_operator(mode,q,k,v,rho,dy,tile)
    torch.cuda.synchronize()
    fwd,bwd,total=[],[],[]
    for j in range(repeats):
        os.environ['CA_CUDA_MODE']=mode
        ev=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        ev[0].record()
        y=continuum_attention(q,k,v,rho,tile)
        ev[1].record()
        torch.autograd.grad((y.float()*dy.float()).sum(),(q,k,v,rho))
        ev[2].record();ev[2].synchronize()
        fwd.append(ev[0].elapsed_time(ev[1]))
        bwd.append(ev[1].elapsed_time(ev[2]))
        total.append(ev[0].elapsed_time(ev[2]))
    return {'forward_ms':statistics.median(fwd),'backward_ms':statistics.median(bwd),
            'total_ms':statistics.median(total),'samples_total_ms':total}


def speed(args):
    torch.manual_seed(1649)
    shape=(2,16,2048,64)
    q=torch.randn(shape,device='cuda',dtype=torch.float16,requires_grad=True)
    k=torch.randn(shape,device='cuda',dtype=torch.float16,requires_grad=True)
    v=torch.randn(shape,device='cuda',dtype=torch.float16,requires_grad=True)
    rho=torch.zeros(16,device='cuda',dtype=torch.float32,requires_grad=True)
    dy=torch.randn(shape,device='cuda',dtype=torch.float16)
    results={}
    for mode in ('prefix_v5','prefix_v6'):
        results[mode]=timed(mode,q,k,v,rho,dy,128,args.warmup,args.repeats)
        print('SPEED_PHASE',json.dumps({'mode':mode,**results[mode]}),flush=True)
    ratio=results['prefix_v5']['total_ms']/results['prefix_v6']['total_ms']
    report={'baseline_ms':results['prefix_v5']['total_ms'],
            'candidate_ms':results['prefix_v6']['total_ms'],
            'attention_speedup':ratio,'minimum_required':args.min_speedup, 'reference':'v5_at_same_tile'}
    print('V6_SPEED',json.dumps(report),flush=True)
    if ratio < args.min_speedup:
        print('V6_NOT_FAST_ENOUGH -- DO NOT ENABLE IN TRAINING',flush=True)
        if args.require_fast: raise RuntimeError('v6 failed v5-relative minimum speed gate')
    else:
        print('V6_APPROVED_FOR_FURTHER_FULL_TRAINING_EVALUATION',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--rel-tol',type=float,default=1e-3)
    p.add_argument('--min-speedup',type=float,default=1.03)
    p.add_argument('--require-fast',action='store_true')
    p.add_argument('--skip-full-decoder',action='store_true')
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
    capability=torch.cuda.get_device_capability(0)
    if capability != (7, 0):
        print('NOTICE: CUDA v6 speed and full-context numerics were validated on V100 sm_70; '+
              f'this device is sm_{capability[0]}{capability[1]}, please review all checks', flush=True)
    tests(args)
    if not args.skip_full_decoder:model_causality(args)
    speed(args)
    print('V6_BENCH_DONE',flush=True)

if __name__=='__main__':main()
