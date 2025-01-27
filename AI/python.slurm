#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=80:00:00
#SBATCH --output=processing_%j.log

# Автоматическое определение путей
INPUT_DIR="/home/ipnthsapst/max_exa/source_files/scripts/python"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/processing_results_${SLURM_JOB_ID}"

module load python/3.11

# Создание выходной директории
mkdir -p "${OUTPUT_DIR}"

srun python plot_spectrum_AI.py \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -j $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))