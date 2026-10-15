"""TinyStories 5M ACMA benchmark, preserving prior data/sampling/evaluation exactly.

Loops exist only in training/evaluation/data loading, never the neural forward.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from architecture import TinyLM, parameters


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def choose_device(choice):
    if choice != "auto":
        result = torch.device(choice)
        if result.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return result
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def stream_tinystories(destination, split, max_bytes):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets: python -m pip install datasets") from exc
    print(f"Streaming TinyStories {split} (up to {max_bytes:,} UTF-8 bytes)...", flush=True)
    # Streaming avoids downloading the multi-GB entire training corpus.
    stream = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    tmp = destination.with_suffix(".partial")
    total = 0
    with open(tmp, "wb") as handle:
        for item in stream:
            raw = (item["text"] + "\n\n").encode("utf-8")
            count = min(len(raw), max_bytes - total)
            if count > 0:
                handle.write(raw[:count])
                total += count
            if total >= max_bytes:
                break
    if total < 2048:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"TinyStories split {split} returned too little text")
    tmp.replace(destination)
    print(f"Cached {total:,} bytes to {destination}", flush=True)


def prepare_data(args):
    if args.train_file is not None:
        if args.val_file is None:
            raise ValueError("Provide --val-file together with --train-file")
        train = Path(args.train_file).expanduser().resolve()
        val = Path(args.val_file).expanduser().resolve()
        if not train.is_file() or not val.is_file():
            raise FileNotFoundError("--train-file and --val-file must exist")
        print("Using supplied UTF-8 files; treating them as raw bytes.")
    else:
        root = Path(args.cache_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        train = root / f"tinystories_train_{args.train_bytes}.bin"
        val = root / f"tinystories_validation_{args.val_bytes}.bin"
        if not train.exists():
            stream_tinystories(train, "train", args.train_bytes)
        if not val.exists():
            stream_tinystories(val, "validation", args.val_bytes)
    train_bytes = np.memmap(train, dtype=np.uint8, mode="r")
    val_bytes = np.memmap(val, dtype=np.uint8, mode="r")
    if min(len(train_bytes), len(val_bytes)) <= args.seq_len + 1:
        raise ValueError("Dataset too small for the selected --seq-len")
    print(f"Dataset bytes: train={len(train_bytes):,}, validation={len(val_bytes):,}")
    return train_bytes, val_bytes


def get_batch(raw, seq_len, batch_size, device, generator):
    starts = torch.randint(0, len(raw) - seq_len - 1, (batch_size,), generator=generator).numpy()
    offsets = np.arange(seq_len + 1, dtype=np.int64)
    ids = np.asarray(raw[starts[:, None] + offsets[None, :]], dtype=np.int64).copy()
    tokens = torch.from_numpy(ids).to(device, non_blocking=False)
    return tokens[:, :-1], tokens[:, 1:]


@torch.no_grad()
def evaluate(model, validation, args, device, seed_offset=877, batches=None):
    model.eval()
    generator_seed = (args.eval_seed if seed_offset == 877 and args.eval_seed is not None
                      else args.holdout_eval_seed if seed_offset == 2003 and args.holdout_eval_seed is not None
                      else args.seed + seed_offset)
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    total = 0.0
    for _ in range(batches if batches is not None else args.val_batches):
        x, target = get_batch(validation, args.seq_len, args.batch_size, device, generator)
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1)).item()
    return total / (batches if batches is not None else args.val_batches)


@torch.no_grad()
def probe(model, args, device):
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 11)
    ids = torch.randint(0, 256, (args.batch_size, args.seq_len), generator=generator).to(device)
    for _ in range(args.probe_warmup):
        model(ids)
    synchronize(device)
    tic = time.perf_counter()
    for _ in range(args.probe_reps):
        model(ids)
    synchronize(device)
    duration = time.perf_counter() - tic
    return {
        "forward_tokens_per_second": round(args.probe_reps * ids.numel() / duration, 1),
        "forward_ms_per_call": round(1000.0 * duration / args.probe_reps, 4),
    }


@torch.no_grad()
def attention_stats(model, validation, args, device):
    """Same first-layer entropy/top1 diagnostic as earlier labs."""
    model.eval()
    generator = torch.Generator(device='cpu').manual_seed(args.seed + 132)
    x, _ = get_batch(validation, args.seq_len, min(4, args.batch_size), device, generator)
    T = x.size(1)
    residual = model.tok_embed(x) + model.pos_embed(torch.arange(T, device=device))[None,:,:]
    attn = model.b0.attn
    normed = model.b0.norm_attn(residual)
    weights = attn.attention_weights(normed)
    selected = weights[:, :, min(16,T-1):, :]
    entropy = -(selected * selected.clamp_min(1e-12).log()).sum(-1).mean().item()
    peak = selected.amax(-1).mean().item()
    q,k,_ = attn._qkv(normed)
    raw = (q @ k.transpose(-2,-1)) * attn.head_dim ** -0.5
    causal = attn.causal_mask[:T,:T]
    gap = raw.masked_fill(~causal, -1e4).amax(-1,keepdim=True) - raw.masked_fill(~causal,-1e4)
    fraction = (((gap > .3) & causal).float().sum() /
                (causal.float().sum() * raw.shape[0]*raw.shape[1])).item()
    result = {'first_layer_attention_entropy_nats': round(entropy,5),
              'first_layer_top1_weight_mean': round(peak,5),
              'fraction_scores_beyond_margin': round(fraction,5)}
    if model.variant in ('P','R','S','H'):
        # W is exactly the integral's discrete equivalent; report how much
        # it differs from regular point-softmax using the SAME Q and K.
        reference = F.softmax(raw.masked_fill(~causal,-torch.inf), dim=-1)
        result['first_layer_mean_abs_mass_difference_vs_softmax'] = round(
            (weights-reference).abs().sum(-1).mean().item() * .5, 6)
    return result


def learned_tau(model):
    if model.variant not in ('F','G','S','H'):
        return None
    logs = torch.cat([block.attn.log_qk_gain.detach().cpu().flatten()
                      for block in (model.b0,model.b1,model.b2,
                                    model.b3,model.b4,model.b5)])
    temps = (-logs).exp()
    return {'mean':round(temps.mean().item(),5),
            'min':round(temps.min().item(),5),
            'max':round(temps.max().item(),5),
            'initial_tau':0.30 if model.variant in ('G','H') else 1.0, 'learned_from_initial_tau_one':model.variant in ('F','S')}


def build(args, variant, device):
    # The exact seed reset and module initialization order from the prior lab.
    torch.manual_seed(args.seed)
    return TinyLM(variant=variant, backend=args.backend,
                  seq_len=args.seq_len, width=args.width,
                  hidden=args.hidden, heads=args.heads,
                  fixed_tau=args.fixed_tau).to(device)


def train_variant(args, variant, train, val, device, holdout=None):
    model=build(args,variant,device)
    count=parameters(model)
    if variant in ('F','G','S','H'):
        small=[p for n,p in model.named_parameters() if n.endswith('log_qk_gain')]
        dense=[p for n,p in model.named_parameters() if not n.endswith('log_qk_gain')]
        optimizer=torch.optim.AdamW([
            {'params':dense, 'lr':args.lr,'weight_decay':args.weight_decay},
            {'params':small, 'lr':args.lr * args.geometry_lr_mult, 'weight_decay':0.0},
        ], betas=(0.9,0.95))
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,
                                    weight_decay=args.weight_decay,betas=(0.9,0.95))
    rng=torch.Generator(device='cpu').manual_seed(args.seed+42)
    print(f'\n{variant}: parameters={count:,}, backend={args.backend}, device={device}', flush=True)
    synchronize(device)
    start=time.perf_counter()
    model.train()
    for step in range(1,args.steps+1):
        x,target=get_batch(train,args.seq_len,args.batch_size,device,rng)
        logits=model(x)
        loss=F.cross_entropy(logits.reshape(-1,256),target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step==1 or step % args.log_every==0 or step==args.steps:
            print(f'{variant} step={step} train_nll={loss.item():.4f}', flush=True)
    synchronize(device)
    elapsed=time.perf_counter()-start
    nll=evaluate(model,val,args,device)
    test_nll=(evaluate(model,holdout,args,device,seed_offset=2003,batches=args.holdout_batches)
              if holdout is not None else None)
    result={
        'variant':variant,'parameters':count,'train_steps':args.steps,
        'train_tokens':args.steps*args.batch_size*args.seq_len,
        'validation_nll_nats':round(nll,6),
        'validation_bits_per_byte':round(nll/math.log(2.0),6),
        'holdout_nll_nats':round(test_nll,6) if test_nll is not None else None,
        'holdout_bits_per_byte':round(test_nll/math.log(2.0),6) if test_nll is not None else None,
        'training_tokens_per_second':round(args.steps*args.batch_size*args.seq_len/elapsed,2),
        'training_seconds':round(elapsed,3),
        'learned_tau':learned_tau(model),
        'attention':attention_stats(model,val,args,device),
        'continuous_integral_exact':model.b0.attn.continuous,
        'fused_sdpa_used':args.backend in ('mixed','metal_weights','metal_analytic','metal_direct','metal_long') and not model.b0.attn.continuous,
        'acma_kernel':args.backend if model.b0.attn.continuous else 'standard',
        'speed':probe(model,args,device),
    }
    print(json.dumps(result,indent=2),flush=True)
    if args.save_checkpoints:
        dest=Path(args.out).with_name(f'{Path(args.out).stem}_{variant}.pt')
        torch.save({'state_dict':model.state_dict(),'variant':variant,
                    'seq_len':args.seq_len,'width':args.width,
                    'hidden':args.hidden,'heads':args.heads,
                    'fixed_tau':args.fixed_tau, 'seed':args.seed,
                    'steps':args.steps, 'parameters':count},dest)
        print(f'Checkpoint: {dest}')
    return result


def main():
    parser=argparse.ArgumentParser(description='Exact ACMA continuum TinyStories 5M benchmark')
    parser.add_argument('--models',default='B,P',help='Comma-separated A,B,F,G,P,R,S,H')
    parser.add_argument('--backend',choices=('reference','mixed','metal_weights','metal_analytic','metal_direct','metal_long'),default='reference')
    parser.add_argument('--device',default='auto',choices=('auto','cpu','mps','cuda'))
    parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--seq-len',type=int,default=192)
    parser.add_argument('--width',type=int,default=256)
    parser.add_argument('--hidden',type=int,default=720)
    parser.add_argument('--heads',type=int,default=4)
    parser.add_argument('--lr',type=float,default=4e-4)
    parser.add_argument('--geometry-lr-mult',type=float,default=8.0)
    parser.add_argument('--weight-decay',type=float,default=.1)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--fixed-tau',type=float,default=.30,
                        help='Applies only to B and P; R/S have no fixed temperature')
    parser.add_argument('--val-batches',type=int,default=128)
    parser.add_argument('--eval-seed',type=int,default=None,help='Optional fixed validation sample RNG seed shared across training seeds')
    parser.add_argument('--holdout-file',default=None,help='An untouched disjoint heldout validation byte stream')
    parser.add_argument('--holdout-batches',type=int,default=128)
    parser.add_argument('--holdout-eval-seed',type=int,default=121212,help='Common heldout evaluation sample positions across training seeds')
    parser.add_argument('--log-every',type=int,default=100)
    parser.add_argument('--probe-reps',type=int,default=30)
    parser.add_argument('--probe-warmup',type=int,default=5)
    parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--save-checkpoints',action='store_true')
    parser.add_argument('--cache-dir',default='data')
    parser.add_argument('--train-bytes',type=int,default=16000000)
    parser.add_argument('--val-bytes',type=int,default=2000000)
    parser.add_argument('--train-file',default=None)
    parser.add_argument('--val-file',default=None)
    parser.add_argument('--out',default='acma_results.json')
    args=parser.parse_args()
    if min(args.steps,args.batch_size,args.seq_len,args.val_batches,args.probe_reps) <= 0:
        parser.error('Steps, batch size, sequence length, validation/probe batches must be positive')
    if args.fixed_tau<=0:
        parser.error('--fixed-tau must be positive')
    if args.backend == 'metal_long' and args.seq_len > 1024:
        parser.error('metal_long v1 is verified only up to T=1024. Use 512/1024 first.')
    if args.backend in ('metal_weights','metal_analytic','metal_direct') and args.seq_len > 256:
        parser.error('Short Metal backends support seq_len <= 256; use metal_long')
    names=[s.strip().upper() for s in args.models.split(',')]
    if not names or any(name not in ('A','B','F','G','P','R','S','H') for name in names):
        parser.error('--models must contain A,B,F,G,P,R,S,H')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device=choose_device(args.device)
    if args.backend.startswith('metal_') and device.type != 'mps':
        parser.error('Metal backends require --device mps on Apple Silicon')
    print(f'Torch {torch.__version__}; device {device}; backend {args.backend}',flush=True)
    # Fail closed: no costly TinyStories run may use a custom analytic backward
    # until its GPU derivative agrees with an independent CPU-f64 oracle at
    # the *actual training sequence length*. Does not alter model seed/data.
    if (args.backend in ('metal_weights', 'metal_analytic', 'metal_long') and not args.probe_only and
            not args.prepare_only and any(v in ('P','R','S','H') for v in names)):
        if args.backend != 'metal_long' and args.seq_len > 256:
            parser.error('Use --backend metal_long for seq_len > 256')
        from verify_backward_mps import compare_vjp
        print('PRE-FLIGHT: validating Metal analytic score VJP against CPU float64 oracle...',flush=True)
        compare_vjp(args.seq_len, backend=('metal_long' if args.backend=='metal_long' else 'metal_analytic'))
        torch.manual_seed(args.seed)
        print('PRE-FLIGHT PASSED. Continuing with original training regime.',flush=True)
    hashes=None
    if args.probe_only:
        results=[]
        for variant in names:
            model=build(args,variant,device)
            results.append({'variant':variant,'parameters':parameters(model),
                            'continuous_integral_exact':model.b0.attn.continuous,
                            'fused_sdpa_used':args.backend in ('mixed','metal_weights','metal_analytic','metal_direct','metal_long') and not model.b0.attn.continuous,
                            'acma_kernel':args.backend if model.b0.attn.continuous else 'standard',
                            'speed':probe(model,args,device)})
    else:
        train,val=prepare_data(args)
        if args.prepare_only:
            return
        holdout=None
        if args.holdout_file is not None:
            fp=Path(args.holdout_file).expanduser().resolve()
            if not fp.is_file():
                raise FileNotFoundError(f'No holdout file: {fp}')
            holdout=np.memmap(fp,dtype=np.uint8,mode='r')
            if len(holdout) <= args.seq_len+1:
                raise ValueError('Holdout too small')
        hashes={'train_sha256':hashlib.sha256(train).hexdigest(),
                'validation_sha256':hashlib.sha256(val).hexdigest(),
                'holdout_sha256':hashlib.sha256(holdout).hexdigest() if holdout is not None else None}
        print(f'Data SHA256 train={hashes["train_sha256"][:16]} '
              f'validation={hashes["validation_sha256"][:16]}',flush=True)
        results=[train_variant(args,variant,train,val,device,holdout) for variant in names]
    report={'dataset':'roneneldan/TinyStories' if args.train_file is None else 'local_text',
            'device':str(device),'torch':torch.__version__,'backend':args.backend,
            'dataset_sha256':hashes,
            'settings':{'steps':args.steps,'batch_size':args.batch_size,
                        'seq_len':args.seq_len,'seed':args.seed,
                        'fixed_tau':args.fixed_tau,
                        'geometry_lr_mult':args.geometry_lr_mult,
                        'train_bytes':args.train_bytes,'val_bytes':args.val_bytes,
                        'val_batches':args.val_batches,'eval_seed':args.eval_seed,
                        'holdout_batches':args.holdout_batches,'holdout_eval_seed':args.holdout_eval_seed,
                        'lr':args.lr,'weight_decay':args.weight_decay,'width':args.width,
                        'hidden':args.hidden,'heads':args.heads},
            'results':results}
    out=Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(f'Saved: {out}')

if __name__=='__main__':
    main()
