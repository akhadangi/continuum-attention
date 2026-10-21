#!/usr/bin/env bash
#SBATCH --job-name=CA-CUDA-Bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ -n "${MODULE_SETUP:-}" ]]; then
  set +u
  eval "$MODULE_SETUP"
fi
srun --ntasks=1 bash cuda/run.sh benchmark "$@"
