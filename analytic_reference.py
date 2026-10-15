"""Independent vectorized mathematical oracle for the ACMA first derivative.

This CPU-friendly implementation is exclusively for correctness testing of the
Metal shader. The shipped MPS attention forward/backward does not invoke it.
"""
import math
import torch
import torch.nn.functional as F
from architecture import continuous_attention_weights


def stable_moments(delta):
    """Return exp(-max(delta, 0)) * integral u**n exp(delta*u) du, n=0..2.

    Exact integral identities with a Taylor expansion for |delta|<1.
    Python loops here are ONLY in the numerical-verification oracle, not model.
    """
    t = delta.abs()
    d = t.clamp_min(1.0)
    e = torch.exp(-d)
    m0_large = -torch.expm1(-d) / d
    m1_pos = (d - 1.0 + e) / d.square()
    m2_pos = (d.square() - 2.0*d + 2.0 - 2.0*e) / d.pow(3)
    m1_neg = (1.0 - (d+1.0)*e) / d.square()
    m2_neg = (2.0 - (d.square()+2.0*d+2.0)*e) / d.pow(3)
    m1_large = torch.where(delta >= 0, m1_pos, m1_neg)
    m2_large = torch.where(delta >= 0, m2_pos, m2_neg)

    # positive/negative signs have different *shifted* moments
    def short_moment(n, pos):
        if n == 0:
            cs = [(-1)**k/math.factorial(k+1) for k in range(17)]
        elif n == 1 and pos:
            cs = [(-1)**k/math.factorial(k+2) for k in range(17)]
        elif n == 2 and pos:
            cs = [(-1)**k*2/math.factorial(k+3) for k in range(17)]
        else:
            cs = [(-1)**k/(math.factorial(k)*(k+n+1)) for k in range(17)]
        out = torch.full_like(t, cs[-1])
        for c in cs[-2::-1]:
            out = out*t + c
        return out
    p0 = short_moment(0,True)
    p1 = torch.where(delta >= 0, short_moment(1,True),short_moment(1,False))
    p2 = torch.where(delta >= 0, short_moment(2,True),short_moment(2,False))
    mask = t < 1.0
    return (torch.where(mask,p0,m0_large),
            torch.where(mask,p1,m1_large),
            torch.where(mask,p2,m2_large))


def analytic_scores_backward(logits, grad_weights, weights=None):
    """Analytic VJP dL/d(logits) from arbitrary dL/d(weights), no autograd.

    Matches the exact piecewise-linear ACMA integrals with first/last caps.
    """
    if weights is None:
        weights = continuous_attention_weights(logits)
    T = logits.shape[-1]
    mask = torch.ones(T,T,device=logits.device,dtype=torch.bool).tril()
    m = logits.masked_fill(~mask,-torch.inf).amax(dim=-1,keepdim=True)
    valid = mask[:,1:]
    a = torch.where(valid,logits[...,:-1],m)
    b = torch.where(valid,logits[...,1:],m)
    d = b-a
    z0,z1,z2 = stable_moments(d)
    e = torch.exp(torch.maximum(a,b)-m)*valid
    # As in the forward, unnormalized segment total mass has the endpoint caps.
    capfirst = F.pad(0.5*torch.exp(logits[...,:1]-m),(0,T-1))
    caplast = torch.diag_embed(0.5*torch.exp(torch.diagonal(logits,dim1=-2,dim2=-1)-m.squeeze(-1)))
    Z = (e*z0).sum(dim=-1,keepdim=True) + capfirst.sum(dim=-1,keepdim=True)+caplast.sum(dim=-1,keepdim=True)
    g = grad_weights - (grad_weights*weights).sum(dim=-1,keepdim=True)
    l = e * (g[...,:-1]*(z0-2*z1+z2) + g[...,1:]*(z1-z2))
    r = e * (g[...,:-1]*(z1-z2) + g[...,1:]*z2)
    return ((F.pad(l,(0,1))+F.pad(r,(1,0)) + (capfirst+caplast)*g)/Z).masked_fill(~mask,0)
