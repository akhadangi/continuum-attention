"""Distributed training, metrics and optimizer utilities for Continuum Attention."""
from __future__ import annotations


import os
import random
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    is_distributed: bool
    device: torch.device


def init_distributed(seed: int = 1337) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed and not torch.cuda.is_available():
        raise RuntimeError("The supplied multi-GPU launcher requires CUDA")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    return DistributedContext(rank, local_rank, world_size, rank == 0, distributed, device)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


import json
import math
from pathlib import Path


def count_parameters(model):
    params = list(model.parameters())
    return {"total": sum(p.numel() for p in params),
            "trainable": sum(p.numel() for p in params if p.requires_grad)}


def nats_to_bpb(nats: float) -> float:
    return float(nats) / math.log(2)


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        with self.path.open("a", encoding="utf8") as f:
            f.write(json.dumps(record) + "\n")


import math
import torch


def build_adamw(model, lr, weight_decay, betas):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


def cosine_lr(step, max_steps, warmup_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    p = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * p))


def set_optimizer_lr(optimizer, lr_value):
    for group in optimizer.param_groups:
        group["lr"] = lr_value

