"""MANDATORY correctness gate before Metal training or performance claims."""
import argparse
import time
import torch
from architecture import continuous_attention_weights, TinyLM
from acma_metal import acma_metal_weights, acma_metal_direct


def assert_match(name, actual, expected, atol=5e-4, rtol=5e-4):
    maximum = (actual-expected).abs().max().item()
    print(f'{name:40s} max_abs={maximum:.7g}', flush=True)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def bench(fn, n=35):
    with torch.no_grad():
        for _ in range(5): fn()
        torch.mps.synchronize()
        start=time.perf_counter()
        for _ in range(n): fn()
        torch.mps.synchronize()
    return 1000*(time.perf_counter()-start)/n


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',action='store_true',help='Time isolated operator paths')
    args=parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('This verifier requires an Apple GPU with MPS enabled.')
    torch.manual_seed(1729)
    print('PyTorch:',torch.__version__,'MPS:',torch.backends.mps.is_available())
    for T in (1,2,7,32,192,256):
        s=(torch.randn(2,3,T,T,device='mps')*3).float()
        v=torch.randn(2,3,T,16,device='mps')
        if T >= 2:
            s[:, :, :, 1:3] = s[:, :, :, :1] # equal-score segments
        ref_w=continuous_attention_weights(s)
        ref_y=ref_w @ v
        w=acma_metal_weights(s)
        y=acma_metal_direct(s,v)
        assert_match(f'T={T}: Metal weights',w,ref_w)
        assert_match(f'T={T}: Metal direct output',y,ref_y)
        assert_match(f'T={T}: Metal weights AV',w@v,ref_y)
        assert_match(f'T={T}: mass per row',w.sum(-1),torch.ones_like(w.sum(-1)))
        causal=torch.ones(T,T,device='mps',dtype=torch.bool).tril()
        if (w.masked_select(~causal).abs().max().item() if (~causal).any().item() else 0) > 1e-7:
            raise AssertionError('Causal mask leaked')
    print('Forward numerical checks PASSED.',flush=True)

    T=13
    s=torch.randn(1,2,T,T,device='mps',requires_grad=True)
    v=torch.randn(1,2,T,8,device='mps',requires_grad=True)
    target=torch.randn(1,2,T,8,device='mps')
    yr=continuous_attention_weights(s)@v
    gsr,gvr=torch.autograd.grad((yr*target).sum(),(s,v))
    yw=acma_metal_weights(s)@v
    gsw,gvw=torch.autograd.grad((yw*target).sum(),(s,v))
    yd=acma_metal_direct(s,v)
    gsd,gvd=torch.autograd.grad((yd*target).sum(),(s,v))
    assert_match('weights grad logits',gsw,gsr,atol=2e-3,rtol=2e-3)
    assert_match('weights grad values',gvw,gvr,atol=2e-3,rtol=2e-3)
    assert_match('direct grad logits',gsd,gsr,atol=2e-3,rtol=2e-3)
    assert_match('direct grad values',gvd,gvr,atol=2e-3,rtol=2e-3)
    print('Gradient numerical checks PASSED.',flush=True)

    ids=torch.randint(0,256,(2,32),device='mps')
    for mode in ('metal_weights','metal_direct'):
        torch.manual_seed(13)
        ref=TinyLM('P',backend='reference',seq_len=32,width=32,hidden=64,heads=4).to('mps')
        torch.manual_seed(13)
        fused=TinyLM('P',backend=mode,seq_len=32,width=32,hidden=64,heads=4).to('mps')
        assert_match(f'full model {mode}',fused(ids),ref(ids),atol=7e-4,rtol=7e-4)
    print('Full model equivalence PASSED.',flush=True)

    if args.benchmark:
        s=torch.randn(8,4,192,192,device='mps')
        v=torch.randn(8,4,192,64,device='mps')
        print('\nIsolated attention core forward timings (ms):')
        print('reference',round(bench(lambda: continuous_attention_weights(s)@v),3))
        print('metal_weights',round(bench(lambda: acma_metal_weights(s)@v),3))
        print('metal_direct',round(bench(lambda: acma_metal_direct(s,v)),3))
    print('\nALL MPS VERIFICATION PASSED.')


if __name__=='__main__':
    run()
