"""Apple Silicon-only long-context ACMA-S shader verifier; never falls back silently."""
import argparse
import torch
from architecture import TinyLM
from verify_backward_mps import compare_vjp


def full_model(T):
    torch.manual_seed(501)
    a=TinyLM(variant='S',backend='reference',seq_len=T,width=32,hidden=64,heads=4).to('mps')
    torch.manual_seed(501)
    b=TinyLM(variant='S',backend='metal_long',seq_len=T,width=32,hidden=64,heads=4).to('mps')
    x=torch.randint(0,256,(1,T),device='mps')
    y=torch.randint(0,256,(1,T),device='mps')
    za=a(x)
    zb=b(x)
    torch.testing.assert_close(zb,za,atol=1e-3,rtol=1e-3)
    loss_a=torch.nn.functional.cross_entropy(za.reshape(-1,256),y.reshape(-1))
    loss_b=torch.nn.functional.cross_entropy(zb.reshape(-1,256),y.reshape(-1))
    loss_a.backward()
    loss_b.backward()
    for key in ('b0.attn.log_qk_gain','b2.attn.qkv.weight',
                'b5.attn.out.weight','tok_embed.weight'):
        ga=dict(a.named_parameters())[key].grad
        gb=dict(b.named_parameters())[key].grad
        torch.testing.assert_close(gb,ga,atol=2e-3,rtol=2e-3)
    print(f'T={T}: full-model logits, NLL and selected gradients PASSED',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lengths',default='257,512,1024')
    ap.add_argument('--skip-full-model',action='store_true')
    args=ap.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('Apple MPS required')
    for T in (int(t) for t in args.lengths.split(',')):
        if T<=0 or T>1024:
            raise ValueError('This verifier supports T=1..1024')
        compare_vjp(T,backend='metal_long')
        if not args.skip_full_model and T<=512:
            full_model(T)
    print('LONG-ACMA MPS VERIFICATION PASSED',flush=True)


if __name__=='__main__': main()
