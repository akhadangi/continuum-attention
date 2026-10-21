# Continuum Attention — NVIDIA CUDA v6

Native CUDA implementation of **Continuum Attention (CA)**, with closed-form integration over causal piecewise-linear key/value segments and analytical first-order gradients. This directory is self-contained: the model, CUDA kernels, data loader, benchmark and launchers live together.

The provided decoder is **CA-S**: 8 layers, width 1,024, 16 heads, 2,048-token context, byte vocabulary 256, and **101,205,120 parameters**. Each head learns a score gain initialized at 1 (effective temperature \(\tau=1\)). This is a separate configuration from the paper's 5M-parameter CA-P TinyStories study.

## Files

```text
cuda/
├── continuum_attention.py   # Model, analytical CA kernels, training and evaluation
├── data.py                  # Hugging Face / local text to UTF-8 byte batches
├── runtime.py               # Single-node DDP, AdamW, metrics and scheduling
├── benchmark.py             # CUDA numerical, gradient, causality and speed tests
├── test_cpu.py              # CPU model and byte-data regression tests
├── run.sh                   # Single GPU or single-node multi-GPU launcher
├── train.sbatch             # Portable Slurm training template
├── benchmark.sbatch         # Portable Slurm verification template
├── requirements.txt
├── example.txt
└── README.md
```

## Requirements

Python 3.10+, NVIDIA CUDA-enabled PyTorch, a **compatible CUDA toolkit with `nvcc`**, a C++ compiler, and the Python packages in `cuda/requirements.txt`. The CUDA extension compiles on first use and caches its build through PyTorch. No Triton installation is required.

Install a suitable PyTorch build following [PyTorch installation instructions](https://pytorch.org/get-started/locally/), then from the **repository root**:

```bash
python -m pip install -r cuda/requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
nvcc --version
```

**Hardware tested:** Tesla V100 (compute capability 7.0), PyTorch 2.6.0+cu118. Different NVIDIA GPUs and software versions should undergo their own benchmark and correctness tests. The accelerated path uses FP16 Q/K/V and is validated up to 2,048 tokens; the moment computations use FP32. "Exact" denotes analytical evaluation of the piecewise-linear integral, not bitwise accuracy across floating-point formats.

## Verify

```bash
# No GPU required
python cuda/test_cpu.py
python cuda/continuum_attention.py --mode ca_check
python cuda/continuum_attention.py --mode param_count

# GPU / CUDA compilation and analytical-gradient checks
bash cuda/run.sh check

# Full 2,048-token verification and attention forward/backward timing
bash cuda/run.sh benchmark --warmup 2 --repeats 7
```

The benchmark compares v6 with the dense causal-prefix reference and v5, checks outputs and gradients for Q/K/V/learned gains, perturbs future positions, and checks a full 100M decoder at context 2,048. Successful audits print `V6_OPERATOR_NUMERICAL_CAUSALITY_PASSED`, `V6_FULL_DECODER_CAUSALITY_PASSED`, and `V6_BENCH_DONE`. On other GPUs, the performance ratio may differ; treat the speed gate as a benchmark comparison, not as a mathematical requirement.

## Train on one GPU (no Slurm)

```bash
DATASET_NAME=HuggingFaceFW/fineweb-edu \
DATASET_CONFIG=sample-10BT \
OUTPUT_DIR=runs/ca_cuda \
BATCH_SIZE=1 GRAD_ACCUM=8 MAX_STEPS=1000 \
bash cuda/run.sh train
```

Or perform a short offline data-pipeline smoke test with the supplied text:

```bash
DATASET_NAME=text DATA_FILES=cuda/example.txt \
OUTPUT_DIR=runs/ca_smoke MAX_STEPS=2 \
BATCH_SIZE=1 GRAD_ACCUM=1 LOG_EVERY=1 SAVE_EVERY=2 \
bash cuda/run.sh train
```

For the 100M model, a 2,048-token context can still require substantial GPU memory. Reduce `BATCH_SIZE` first; changing the context width, layers or head count creates a different model. The dataset stream requires `datasets` and may require network access to download FineWeb-Edu.

### Resume

```bash
RESUME_FROM=runs/ca_cuda/latest OUTPUT_DIR=runs/ca_cuda \
bash cuda/run.sh train
```

Weights, optimizer state, scaler, step and token count are restored. The streaming data iterator is reconstructed and **does not guarantee bitwise-identical document ordering after resume**.

## Slurm: one or more GPUs on one node

Submit from the repository root. Specify your own partition and other site-dependent Slurm settings on the command line:

```bash
# One GPU
sbatch --partition=YOUR_GPU_PARTITION --gres=gpu:1 cuda/train.sbatch

# Four GPUs on one node
BATCH_SIZE=8 GRAD_ACCUM=8 OUTPUT_DIR=runs/ca_cuda_4gpu \
sbatch --export=ALL --partition=YOUR_GPU_PARTITION \
  --gres=gpu:4 --cpus-per-task=16 cuda/train.sbatch

# Correctness and performance benchmark on one GPU
sbatch --partition=YOUR_GPU_PARTITION --gres=gpu:1 cuda/benchmark.sbatch
```

Some Slurm installations use `--gpus=4` instead of `--gres=gpu:4`. Use whichever your site supports. The scripts intentionally have no fixed partition, account, QoS, GPU model constraint or virtual-environment path. If software modules must be loaded **inside the allocation**, set `MODULE_SETUP` to your site's environment initialization commands, for example:

```bash
export MODULE_SETUP='module load cuda; source "$HOME/myenv/bin/activate"'
sbatch --partition=YOUR_GPU_PARTITION cuda/benchmark.sbatch
```

`torchrun` automatically launches one worker per visible GPU. These templates support **one node only**; they do not implement multi-node rendezvous.

### Training settings

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `DATASET_NAME`, `DATASET_CONFIG` | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | Dataset name and subset |
| `DATA_FILES` | unset | Local file when `DATASET_NAME=text` |
| `OUTPUT_DIR` | `runs/continuum_ca_cuda_v6` | Checkpoints and JSONL metrics |
| `RESUME_FROM` | unset | Saved checkpoint directory |
| `NPROC_PER_NODE` | all visible GPUs | One node worker count |
| `BATCH_SIZE`, `GRAD_ACCUM` | `1`, `8` | Microbatch per GPU, accumulation steps |
| `BLOCK_SIZE`, `CA_QUERY_TILE` | `2048`, `128` | Token context, attention query tile |
| `D_MODEL`, `N_HEADS`, `N_LAYERS` | `1024`, `16`, `8` | Model configuration |
| `MAX_STEPS`, `SAVE_EVERY` | `50000`, `1000` | Training length, checkpoints |
| `LR`, `MIN_LR`, `WARMUP_STEPS` | `3e-4`, `3e-5`, `2000` | Learning-rate schedule |
| `CA_RUN_GPU_CHECK` | `1` | Check native CUDA before training |
| `CA_RUN_FULL_BENCH` | `0` | Also perform full-context benchmark before training |
| `MODULE_SETUP` | unset | Slurm environment initialization |

Run `python cuda/continuum_attention.py --help` for the complete set of flags. Extra CLI flags after `bash cuda/run.sh train` are forwarded to the training program.

## Hardware performance result

At `[batch, heads, tokens, head_dim] = [2,16,2048,64]`, query tile 128, on one Tesla V100:

| Implementation | Forward | Backward | Total |
| --- | ---: | ---: | ---: |
| v5 | 5.974 ms | 13.543 ms | 19.519 ms |
| **v6** | **5.012 ms** | **11.587 ms** | **16.603 ms** |

That is **1.176× faster isolated attention forward/backward** relative to v5. 

## Scope

First-order gradients; byte-level autoregressive training; single-node DDP. No KV cache, incremental decoding, arbitrary masking or pretrained checkpoints are provided. The byte-stream loader uses UTF-8 text packing and dataset-shard splitting when available. For scientifically controlled comparisons, record dataset revision, sampling protocol, training configuration and code version.

Paper: [Continuum Attention (arXiv:2610.20586)](https://arxiv.org/abs/2610.20586).
