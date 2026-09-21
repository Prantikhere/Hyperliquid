#!/bin/bash
cd /home/prantik/Downloads/Personal/TradingBingx
export PYTHONPATH=/home/prantik/Downloads/Personal/TradingBingx
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1
export PYTHONFAULTHANDLER=1
./venv/bin/python3 -u src/execution/hl_executor.py >> logs/polyarb.log 2>&1
