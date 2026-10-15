"""Exact algebraic continuous-memory attention, TinyStories 5M backbone.

Variants:
 A standard causal softmax, tau=1
 B standard causal softmax, fixed tau (comparison reference)
 F ordinary softmax, end-to-end learned per-head positive QK gain (tau init 1)
 G ordinary softmax, learned per-head gain initialized at tau 0.30
 P exact continuous memory, fixed tau (matched to B)
 R exact continuous memory, no additional gain (matched to A)
 S exact continuous memory, end-to-end learned per-head QK gain (matched to F)
 H exact continuous memory, learned per-head gain initialized at tau 0.30 (matched to G)

No Python iteration, recurrent scan, numerical quadrature, or iterative solver
in any neural-model forward path. The integral on each key-value segment is exact.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def segment_integral_coefficients(delta):
    """For a segment with endpoint logits a,b, delta=b-a, return (L, mu).

    exp(max(a,b))*L is exact segment mass, where
      L = (1-exp(-abs(delta)))/abs(delta), L(0)=1.
    mu = integral_0^1 u exp(delta*u) du / integral_0^1 exp(delta*u) du,
    mu(0)=1/2. Segment contributes mass*((1-mu)*v_left+mu*v_right).

    Stable at zero and very large |delta|. The Taylor branches are analytic
    approximations at machine precision for float32 and common float64 use.
    No variable-length numerical integration and no token-axis loops.
    """
    t = delta.abs()
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    t6 = t3 * t3
    # L = 1 - t/2 + t^2/6 - t^3/24 + t^4/120 - t^5/720 + t^6/5040 +...
    l_small = (1.0 - 0.5*t + t2/6.0 - t3/24.0
               + t4/120.0 - t5/720.0 + t6/5040.0)
    safe_t = t.clamp_min(0.1)
    l_general = -torch.expm1(-safe_t) / safe_t
    L = torch.where(t < 0.1, l_small, l_general)

    # mu = 1/2 + delta/12 - delta^3/720 + delta^5/30240 + O(delta^7).
    mu_small = 0.5 + delta / 12.0 - delta.pow(3) / 720.0 + delta.pow(5) / 30240.0
    mu_positive = 1.0 / (-torch.expm1(-safe_t)) - 1.0 / safe_t
    mu_general = torch.where(delta >= 0, mu_positive, 1.0 - mu_positive)
    mu = torch.where(t < 0.1, mu_small, mu_general)
    return L, mu


def continuous_attention_weights(logits):
    """Exact continuous causal piecewise-linear K/V memory weights.

    logits: [batch, heads, queries=T, keys=T], *unmasked* scaled QK logits.
    For query i integrate from t=-1/2 to t=i+1/2, where K,V are linearly
    interpolated between adjacent integer-token positions and extended
    constantly over first/last half-token caps of the query's visible prefix.

    A single query-dependent stochastic weight matrix W represents the exact
    integral, and the output is W @ V. The two half-caps are essential: when
    all QK scores are equal, W is exactly ordinary uniform causal attention.
    """
    T = logits.size(-1)
    if logits.size(-2) != T:
        raise ValueError('This benchmark requires square full-sequence causal attention')
    causal = torch.ones(T, T, dtype=torch.bool, device=logits.device).tril()
    row_max = logits.masked_fill(~causal, -torch.inf).amax(dim=-1, keepdim=True)

    # Segment j-1 -> j is visible only if j <= query index i.
    valid_segments = causal[:, 1:]
    # Mask invalid endpoint scores *before* taking exponentials or differences.
    # Otherwise a future score can overflow even when later multiplied by zero.
    a = torch.where(valid_segments, logits[..., :-1], row_max)
    b = torch.where(valid_segments, logits[..., 1:], row_max)
    delta = b - a
    L, mu = segment_integral_coefficients(delta)
    segment_mass = torch.exp(torch.maximum(a, b) - row_max) * L
    segment_mass = segment_mass * valid_segments.to(segment_mass.dtype)

    # Algebraically collect the contributions at endpoint keys without a scan.
    left_contrib = F.pad(segment_mass * (1.0 - mu), (0, 1))
    right_contrib = F.pad(segment_mass * mu, (1, 0))

    # Half-token cap at t in [-1/2,0] (always visible).
    initial_cap = F.pad(0.5 * torch.exp(logits[..., :1] - row_max), (0, T-1))
    # Half-token cap at t in [i,i+1/2] for each query i.
    diagonal_logits = torch.diagonal(logits, dim1=-2, dim2=-1)
    ending_cap = torch.diag_embed(0.5 * torch.exp(diagonal_logits - row_max.squeeze(-1)))

    w = left_contrib + right_contrib + initial_cap + ending_cap
    return w / w.sum(dim=-1, keepdim=True)


class CausalAttention(nn.Module):
    def __init__(self, width, heads, seq_len, variant='B', backend='reference', fixed_tau=0.30):
        super().__init__()
        if variant not in ('A', 'B', 'F', 'G', 'P', 'R', 'S', 'H'):
            raise ValueError('variant must be A, B, F, G, P, R, S or H')
        if backend not in ('reference', 'mixed', 'metal_weights', 'metal_analytic', 'metal_direct', 'metal_long'):
            raise ValueError('backend must be reference, mixed, metal_weights, metal_analytic, metal_direct or metal_long')
        if width % heads:
            raise ValueError('width must be divisible by heads')
        if fixed_tau <= 0:
            raise ValueError('fixed_tau must be > 0')
        self.variant = variant
        self.backend = backend
        self.heads = heads
        self.width = width
        self.head_dim = width // heads
        self.fixed_tau = float(fixed_tau)
        self.continuous = variant in ('P', 'R', 'S', 'H')
        # Learned scale is identical to the F control from earlier experiments.
        if variant in ('F', 'G', 'S', 'H'):
            init_log_gain = -math.log(0.30) if variant in ('G', 'H') else 0.0
            self.log_qk_gain = nn.Parameter(torch.full((heads,), init_log_gain))
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.register_buffer('causal_mask', torch.ones(seq_len, seq_len, dtype=torch.bool).tril(), persistent=False)

    def _qkv(self, x):
        batch, tokens, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        if self.variant in ('B', 'P'):
            q = q * (1.0 / self.fixed_tau)
        elif self.variant in ('F', 'G', 'S', 'H'):
            q = q * self.log_qk_gain.exp().view(1, -1, 1, 1)
        return q, k, v

    def _weights(self, q, k, v=None):
        logits = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if self.continuous:
            if self.backend in ('metal_weights', 'metal_analytic'):
                from acma_metal import acma_metal_weights
                return acma_metal_weights(logits)
            if self.backend == 'metal_long':
                from acma_metal import acma_metal_long
                return acma_metal_long(logits)
            return continuous_attention_weights(logits)
        T = q.size(-2)
        return F.softmax(logits.masked_fill(~self.causal_mask[:T, :T], -torch.inf), dim=-1)

    def attention_weights(self, x):
        q, k, v = self._qkv(x)
        return self._weights(q, k, v)

    def forward(self, x):
        batch, tokens, _ = x.shape
        q, k, v = self._qkv(x)
        if self.backend in ('mixed', 'metal_weights', 'metal_analytic', 'metal_direct', 'metal_long') and not self.continuous:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        elif self.backend == 'metal_direct' and self.continuous:
            from acma_metal import acma_metal_direct
            logits = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            y = acma_metal_direct(logits, v)
        else:
            # Metal-weights fuses exact score-to-probabilities arithmetic;
            # reference uses the original vectorized integral.
            y = self._weights(q, k, v) @ v
        y = y.transpose(1, 2).contiguous().reshape(batch, tokens, self.width)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(self, width, hidden, heads, seq_len, variant, backend, fixed_tau):
        super().__init__()
        self.norm_attn = RMSNorm(width)
        self.attn = CausalAttention(width, heads, seq_len, variant, backend, fixed_tau)
        self.norm_mlp = RMSNorm(width)
        self.mlp = SwiGLU(width, hidden)

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        return x + self.mlp(self.norm_mlp(x))


class TinyLM(nn.Module):
    """Identical six-layer 5,008,640-parameter pre-norm baseline architecture."""
    def __init__(self, variant='B', backend='reference', seq_len=192,
                 width=256, hidden=720, heads=4, fixed_tau=0.30):
        super().__init__()
        self.seq_len = seq_len
        self.variant = variant
        self.tok_embed = nn.Embedding(256, width)
        self.pos_embed = nn.Embedding(seq_len, width)
        self.b0 = Block(width, hidden, heads, seq_len, variant, backend, fixed_tau)
        self.b1 = Block(width, hidden, heads, seq_len, variant, backend, fixed_tau)
        self.b2 = Block(width, hidden, heads, seq_len, variant, backend, fixed_tau)
        self.b3 = Block(width, hidden, heads, seq_len, variant, backend, fixed_tau)
        self.b4 = Block(width, hidden, heads, seq_len, variant, backend, fixed_tau)
        self.b5 = Block(width, hidden, heads, seq_len, variant, backend, fixed_tau)
        self.final_norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, 256, bias=False)
        self.lm_head.weight = self.tok_embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        _, length = tokens.shape
        if length > self.seq_len:
            raise ValueError('Sequence exceeds model seq_len')
        pos = torch.arange(length, device=tokens.device)
        x = self.tok_embed(tokens) + self.pos_embed(pos)[None, :, :]
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.lm_head(self.final_norm(x))


def parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
