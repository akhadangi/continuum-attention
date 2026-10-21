#!/usr/bin/env bash
#SBATCH --job-name=CA-CUDA-v6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Set site-specific Slurm options using sbatch command-line flags.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
# Optional MODULE_SETUP can load dependencies before running.
if [[ -n "${MODULE_SETUP:-}" ]]; then
  set +u
  eval "$MODULE_SETUP"
fi
# srun launches one torchrun parent, which launches one child per visible GPU.
srun --ntasks=1 bash cuda/run.sh train "$@"
