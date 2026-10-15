"""Minimal forward and backward demonstration; no dataset download."""
import argparse
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architecture import TinyLM

p = argparse.ArgumentParser()
p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
p.add_argument('--seq-len', choices=[192, 512], default=192, type=int)
a = p.parse_args()
if a.device == 'mps' and not torch.backends.mps.is_available():
    p.error('Apple Metal Performance Shaders is not available')
backend = 'reference' if a.device == 'cpu' else ('metal_analytic' if a.seq_len == 192 else 'metal_long')
torch.manual_seed(1729)
model = TinyLM(variant='P', backend=backend, seq_len=a.seq_len, fixed_tau=0.30).to(a.device)
# A short sanity check of both forward and gradient paths.
x = torch.randint(0, 256, (1, a.seq_len), device=a.device)
out = model(x)
loss = out.float().square().mean()
loss.backward()
assert torch.isfinite(loss).item()
assert all(torch.isfinite(param.grad).all().item() for param in model.parameters() if param.grad is not None)
print(f'OK: device={a.device} backend={backend} shape={tuple(out.shape)} loss={loss.item():.6f}')
