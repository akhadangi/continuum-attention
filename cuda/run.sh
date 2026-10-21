#!/usr/bin/env bash
# Portable single-node NVIDIA CA v6 entry point. No scheduler required.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CA_CUDA_MODE=prefix_v6
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

usage() {
  echo "Usage: bash cuda/run.sh {check|benchmark|train} [extra_args]" >&2
  echo "Configuration: DATASET_NAME, DATASET_CONFIG, DATA_FILES, OUTPUT_DIR," >&2
  echo "  BATCH_SIZE, GRAD_ACCUM, MAX_STEPS, NPROC_PER_NODE, RESUME_FROM," >&2
  echo "  WANDB_PROJECT, CA_RUN_GPU_CHECK and model hyperparameter env vars." >&2
  exit 2
}
[[ $# -gt 0 ]] || usage
MODE=$1; shift

MODEL_ARGS=(
  --block_size "${BLOCK_SIZE:-2048}"
  --d_model "${D_MODEL:-1024}"
  --n_heads "${N_HEADS:-16}"
  --n_layers "${N_LAYERS:-8}"
  --mlp_ratio "${MLP_RATIO:-4.0}"
  --rope_theta "${ROPE_THETA:-10000}"
  --dropout 0.0
  --ca_query_tile "${CA_QUERY_TILE:-128}"
)
MODEL="$ROOT/cuda/continuum_attention.py"

cuda_environment() {
  command -v nvcc >/dev/null 2>&1 || {
    echo 'CUDA toolkit nvcc is required. Load your site CUDA module or install a CUDA toolkit.' >&2
    exit 1
  }
  python - <<'PY'
import torch
assert torch.cuda.is_available(), 'No NVIDIA CUDA device visible'
print('NVIDIA device:', torch.cuda.get_device_name(0), 'count:', torch.cuda.device_count())
print('Torch:',torch.__version__,'Torch CUDA:',torch.version.cuda)
PY
}

case "$MODE" in
  check)
    python "$MODEL" --mode param_count "${MODEL_ARGS[@]}"
    CA_CUDA_MODE=prefix python "$MODEL" --mode ca_check "${MODEL_ARGS[@]}"
    cuda_environment
    python "$MODEL" --mode ca_cuda_check "${MODEL_ARGS[@]}"
    python "$MODEL" --mode causality_test "${MODEL_ARGS[@]}"
    ;;
  benchmark)
    cuda_environment
    python "$ROOT/cuda/benchmark.py" "$@"
    ;;
  train)
    cuda_environment
    [[ -z "${DATASET_NAME:-}" ]] && DATASET_NAME=HuggingFaceFW/fineweb-edu
    if [[ -z "${DATASET_CONFIG+x}" ]]; then
      DATASET_CONFIG=""
      [[ "$DATASET_NAME" == HuggingFaceFW/fineweb-edu ]] && DATASET_CONFIG=sample-10BT
    fi
    OUTPUT_DIR="${OUTPUT_DIR:-runs/continuum_ca_cuda_v6}"
    if [[ "${DATASET_NAME}" == text && -z "${DATA_FILES:-}" ]]; then
      echo 'Set DATA_FILES when DATASET_NAME=text (e.g., cuda/example.txt)' >&2; exit 1
    fi
      # Microbatch defaults to one to reduce GPU memory requirements.
    python "$MODEL" --mode param_count "${MODEL_ARGS[@]}"
    CA_CUDA_MODE=prefix python "$MODEL" --mode ca_check "${MODEL_ARGS[@]}"
    if [[ "${CA_RUN_GPU_CHECK:-1}" == 1 ]]; then
      python "$MODEL" --mode ca_cuda_check "${MODEL_ARGS[@]}"
      python "$MODEL" --mode causality_test "${MODEL_ARGS[@]}"
    fi
    if [[ "${CA_RUN_FULL_BENCH:-0}" == 1 ]]; then
      python "$ROOT/cuda/benchmark.py" --warmup 2 --repeats 5
    fi
    NGPU="${NPROC_PER_NODE:-auto}"
    if [[ "$NGPU" == auto ]]; then
      NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
    fi
    [[ "$NGPU" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid NPROC_PER_NODE' >&2; exit 2; }
    AVAILABLE="$(python -c 'import torch; print(torch.cuda.device_count())')"
    (( NGPU <= AVAILABLE )) || { echo "Requested $NGPU GPUs but only $AVAILABLE visible" >&2; exit 1; }
    TRAIN_ARGS=(
      --mode train
      --output_dir "$OUTPUT_DIR"
      --dataset_name "$DATASET_NAME"
      --split "${DATASET_SPLIT:-train}"
      --text_column "${TEXT_COLUMN:-text}"
      --batch_size "${BATCH_SIZE:-1}"
      --grad_accum "${GRAD_ACCUM:-8}"
      --max_steps "${MAX_STEPS:-50000}"
      "${MODEL_ARGS[@]}"
      --lr "${LR:-3e-4}"
      --min_lr "${MIN_LR:-3e-5}"
      --warmup_steps "${WARMUP_STEPS:-2000}"
      --weight_decay "${WEIGHT_DECAY:-0.1}"
      --beta1 "${BETA1:-0.9}"
      --beta2 "${BETA2:-0.95}"
      --grad_clip "${GRAD_CLIP:-1.0}"
      --amp_dtype fp16
      --prefetch_batches "${PREFETCH_BATCHES:-2}"
      --shuffle_buffer "${SHUFFLE_BUFFER:-20000}"
      --log_every "${LOG_EVERY:-10}"
      --save_every "${SAVE_EVERY:-1000}"
      --keep_last "${KEEP_LAST:-3}"
    )
    [[ -n "$DATASET_CONFIG" ]] && TRAIN_ARGS+=(--dataset_config "$DATASET_CONFIG")
    [[ -n "${DATA_FILES:-}" ]] && TRAIN_ARGS+=(--data_files "$DATA_FILES")
    [[ "${NO_STREAMING:-0}" == 1 ]] && TRAIN_ARGS+=(--no_streaming)
    [[ -n "${RESUME_FROM:-}" ]] && TRAIN_ARGS+=(--resume "$RESUME_FROM")
    [[ -n "${WANDB_PROJECT:-}" ]] && TRAIN_ARGS+=(--wandb_project "$WANDB_PROJECT")
    [[ -n "${WANDB_RUN_NAME:-}" ]] && TRAIN_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
    exec python -m torch.distributed.run --standalone --nnodes=1 \
      --nproc_per_node="$NGPU" "$MODEL" "${TRAIN_ARGS[@]}" "$@"
    ;;
  *) usage ;;
esac
