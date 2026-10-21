"""Hugging Face datasets streamed into contiguous UTF-8 byte batches."""
from __future__ import annotations

from itertools import islice
from typing import Iterator

import torch


def _documents(*, dataset_name, dataset_config, split, text_column,
               data_files, streaming, shuffle_buffer, seed,
               rank, world_size) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets: pip install datasets") from exc

    kwargs = {"path": dataset_name, "split": split, "streaming": streaming}
    if dataset_config:
        kwargs["name"] = dataset_config
    if data_files:
        kwargs["data_files"] = data_files
    dataset = load_dataset(**kwargs)
    if shuffle_buffer > 0:
        if streaming:
            dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        else:
            dataset = dataset.shuffle(seed=seed)

    # Prefer HF's shard-aware multi-worker splitter for data sources with
    # native shards; fall back to deterministic document-index partitioning.
    if world_size > 1:
        try:
            from datasets.distributed import split_dataset_by_node
            dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
            rows = iter(dataset)
        except (ImportError, ValueError, TypeError, AttributeError):
            rows = islice(iter(dataset), rank, None, world_size)
    else:
        rows = iter(dataset)
    for row in rows:
        value = row.get(text_column)
        if value is not None:
            text = value if isinstance(value, str) else str(value)
            if text:
                yield text


def build_byte_batch_iterator(*, dataset_name, dataset_config=None, split="train",
                              text_column="text", data_files=None, streaming=True,
                              shuffle_buffer=20000, seed=1337, block_size=2048,
                              batch_size=1, rank=0, world_size=1,
                              add_document_separator=True, infinite=True,
                              device="cpu"):
    """Yield contiguous byte blocks as [batch_size,block_size] long tensors.

    For portability, documents are UTF-8 encoded and optionally separated by a
    newline byte. In multi-GPU mode, documents are partitioned between ranks.
    At exhaustion, the dataset restarts with a new shuffle seed when infinite.
    """
    if block_size < 2 or batch_size < 1 or world_size < 1 or not (0 <= rank < world_size):
        raise ValueError("Invalid block/batch/distributed dimensions")
    chunk_size = block_size * batch_size
    target_device = torch.device(device)
    epoch = 0
    queue = bytearray()
    consumed = 0
    while True:
        produced = False
        for text in _documents(
            dataset_name=dataset_name, dataset_config=dataset_config,
            split=split, text_column=text_column, data_files=data_files,
            streaming=streaming, shuffle_buffer=shuffle_buffer, seed=seed + epoch,
            rank=rank, world_size=world_size,
        ):
            produced = True
            queue.extend(text.encode("utf-8"))
            if add_document_separator:
                queue.extend(b"\n")
            while len(queue) - consumed >= chunk_size:
                chunk = bytes(memoryview(queue)[consumed:consumed + chunk_size])
                consumed += chunk_size
                # Do not resize a bytearray while a memoryview is alive.
                batch = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
                batch = batch.to(dtype=torch.long).reshape(batch_size, block_size)
                yield batch.to(target_device, non_blocking=False)
            if consumed and (consumed > 4 * chunk_size or consumed == len(queue)):
                del queue[:consumed]
                consumed = 0
        if not infinite:
            return
        if not produced:
            raise RuntimeError("Dataset yielded no nonempty text for this distributed rank")
        epoch += 1
