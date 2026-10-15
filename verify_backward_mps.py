"""MPS GATE: verify exact ACMA analytic Metal backward before running training.

Run this on Apple Silicon with PyTorch 2.8+ and torch.mps.compile_shader.
Does NOT silently replace the Metal kernel with CPU/PyTorch autograd.
"""
import argparse
import time
import torch
from architecture import TinyLM, continuous_attention_weights
from acma_metal import acma_metal_weights, acma_metal_long
from analytic_reference import analytic_scores_backward


def check(label, got, expected, atol=1.5e-3, rtol=1.5e-3):
    if not torch.isfinite(got).all().item():
        raise AssertionError(label+': nonfinite results')
    maximum=(got-expected).abs().max().item()
    print(f'{label:<53} max_abs={maximum:.6g}',flush=True)
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)


def error_report(label, got, expected, atol=1.5e-3, rtol=1.5e-3):
    # Report the absolute values as well as their difference: a large
    # difference does not indicate which implementation is incorrect.
    got = got.detach().cpu()
    expected = expected.detach().cpu()
    difference = (got - expected).abs()
    tolerance = atol + rtol * expected.abs()
    bad = difference > tolerance
    maximum = difference.max().item()
    index = torch.unravel_index(difference.argmax(),difference.shape)
    idx = tuple(int(v) for v in index)
    print(f'{label:<49} max_abs={maximum:.6g} bad={bad.sum().item()}/{bad.numel()} '
          f'max_actual={got.abs().max().item():.6g} '
          f'max_expected={expected.abs().max().item():.6g}',flush=True)
    if bad.any().item():
        print(f'  worst {idx}: actual={got[idx].item():.9g} '
              f'expected={expected[idx].item():.9g}',flush=True)
    return not bool(bad.any().item())


def compare_vjp(T, magnitude=1., repeat=False, backend="metal_analytic"):
    if backend not in ("metal_analytic", "metal_long"):
        raise ValueError("backend must be metal_analytic or metal_long")
    # Large CPU-f64 oracle can be memory intensive; one head is sufficient
    # to test all causal boundary cases beyond 256.
    B,H=(1,1) if T>=512 else (2,2)
    s=torch.randn(B,H,T,T,device='mps',dtype=torch.float32)*magnitude
    if repeat and T>1:
        s=s.clone()
        s[...,1::2]=s[...,::2][...,:s[...,1::2].shape[-1]]
    s=s.requires_grad_()
    g=torch.randn_like(s)
    wr=continuous_attention_weights(s)
    ref_mps=torch.autograd.grad(wr,s,g)[0]
    wa=(acma_metal_long(s) if backend=="metal_long" else acma_metal_weights(s))
    actual=torch.autograd.grad(wa,s,g)[0]
    check(f'T={T} magnitude={magnitude} repeat={repeat}: forward',wa,wr,
          atol=5e-4,rtol=5e-4)

    # The CPU double oracle computes the algebraic VJP directly (no MPS
    # reference graph). It is independent of BOTH GPU implementations.
    # MPS cannot represent float64: transfer first, then promote on CPU.
    cpu_scores=s.detach().cpu().to(dtype=torch.float64)
    cpu_grad=g.detach().cpu().to(dtype=torch.float64)
    oracle=analytic_scores_backward(cpu_scores,cpu_grad).to(dtype=torch.float32)
    prefix=f'T={T} magnitude={magnitude} repeat={repeat}'
    metal_ok=error_report(prefix+': Metal vs CPU-f64 oracle',actual,oracle)
    mps_ref_ok=error_report(prefix+': MPS autograd vs CPU-f64 oracle',ref_mps,oracle)
    if not metal_ok or not mps_ref_ok:
        sum_residual=actual.detach().sum(-1).abs().max().item()
        print(f'  Metal row-score-gradient shift-invariance residual: '
              f'{sum_residual:.6g}',flush=True)
        if not metal_ok:
            raise AssertionError('Metal analytic backward disagrees with the independent '
                                 'CPU-f64 oracle; training prohibited')
        if not mps_ref_ok:
            print('  WARNING: PyTorch MPS reference autograd differs from the '
                  'independent CPU-f64 oracle. The Metal gradient passed; '
                  'the MPS reference is not a reliable ground truth here.',flush=True)
    causal=torch.ones(T,T,device='mps',dtype=torch.bool).tril()
    if actual.masked_select(~causal).abs().sum().item()!=0.0:
        raise AssertionError('Non-causal score gradients were nonzero')
    if T==1 and actual.abs().max().item()>1e-6:
        raise AssertionError('T=1 should have exactly zero score gradient')


def full_model_test():
    torch.manual_seed(1729)
    reference=TinyLM('S',backend='reference',seq_len=32,
                     width=32,hidden=64,heads=4).to('mps')
    torch.manual_seed(1729)
    accelerated=TinyLM('S',backend='metal_analytic',seq_len=32,
                       width=32,hidden=64,heads=4).to('mps')
    x=torch.randint(0,256,(2,32),device='mps')
    target=torch.randint(0,256,(2,32),device='mps')
    y0=reference(x)
    y1=accelerated(x)
    check('S full-model logits',y1,y0,atol=8e-4,rtol=8e-4)
    loss0=torch.nn.functional.cross_entropy(y0.reshape(-1,256),target.reshape(-1))
    loss1=torch.nn.functional.cross_entropy(y1.reshape(-1,256),target.reshape(-1))
    loss0.backward()
    loss1.backward()
    check('S full-model NLL',loss1,loss0,atol=2e-4,rtol=2e-4)
    reference_p=dict(reference.named_parameters())
    accelerated_p=dict(accelerated.named_parameters())
    for key in ('b0.attn.log_qk_gain','b0.attn.qkv.weight',
                'b3.attn.qkv.weight','b5.attn.out.weight',
                'b0.norm_attn.weight','tok_embed.weight'):
        check('S parameter gradient '+key,accelerated_p[key].grad,
              reference_p[key].grad,atol=2e-3,rtol=2e-3)
    print('FULL MODEL F/G PARAMETER GRADIENTS PASSED.',flush=True)


def time_call(fn, reps=15):
    # GPU GPU workloads are measured with full synchronization, not enqueue time.
    for _ in range(4):fn()
    torch.mps.synchronize()
    start=time.perf_counter()
    for _ in range(reps):fn()
    torch.mps.synchronize()
    return 1000*(time.perf_counter()-start)/reps


def benchmark(reps):
    print('\nIsolated full attention core (B=8,H=4,T=192,D=64):')
    s=torch.randn(8,4,192,192,device='mps',requires_grad=True)
    v=torch.randn(8,4,192,64,device='mps',requires_grad=True)
    for name,weightfn in (('reference',continuous_attention_weights),
                          ('analytic Metal',acma_metal_weights)):
        def forward():
            with torch.no_grad():
                return weightfn(s)@v
        def backward():
            w=weightfn(s)
            y=w@v
            return torch.autograd.grad(y.sum(),(s,v),retain_graph=False)
        print(f'  {name:17s} forward ms={time_call(forward,reps):.3f} '
              f'forward+backward ms={time_call(backward,reps):.3f}',flush=True)
    print('Note: only the reference and analytic Metal kernels are benchmarked.',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark',action='store_true')
    ap.add_argument('--reps',type=int,default=12)
    ap.add_argument('--backend',choices=['metal_analytic','metal_long'],default='metal_analytic')
    ap.add_argument('--lengths',default=None,help='Comma-separated lengths; e.g. 257,512,1024')
    args=ap.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('Apple MPS required; no non-MPS fallback')
    print('Torch',torch.__version__,'MPS',torch.backends.mps.is_available())
    torch.manual_seed(1729)
    lengths=(tuple(int(t) for t in args.lengths.split(','))
             if args.lengths is not None else ((257,512) if args.backend=='metal_long' else (1,2,7,32,192,256)))
    for T in lengths:
        compare_vjp(T,backend=args.backend)
    if args.backend=='metal_analytic':
        for T in (7,32,192):
            compare_vjp(T,magnitude=7.0)
            compare_vjp(T,magnitude=1.0,repeat=True)
        full_model_test()
    print('\nALL METAL ANALYTIC BACKWARD CHECKS PASSED.\n')
    if args.benchmark:
        if args.backend != "metal_analytic":
            print("Long-kernel perf measured via benchmark.py --probe-only")
        else:
            benchmark(args.reps)

if __name__=='__main__': main()
