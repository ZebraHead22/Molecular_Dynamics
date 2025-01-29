#!/bin/bash
#SBATCH --job-name=prc_spectre
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=30:00:00

# Загрузка необходимых модулей
module load python/3.11

# Путь к вашему скрипту
PYTHON_SCRIPT="/home/ipnthsapst/max_exa/source_files/scripts/python/plot_spectrum_AI_simple.py"

# Запуск скрипта
python3 $PYTHON_SCRIPT