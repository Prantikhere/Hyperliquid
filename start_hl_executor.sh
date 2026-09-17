#!/bin/bash
cd /home/prantik/Downloads/Personal/TradingBingx
export PYTHONPATH=/home/prantik/Downloads/Personal/TradingBingx
export PYTHONDONTWRITEBYTECODE=1
./venv/bin/python3 -u src/execution/hl_executor.py >> logs/polyarb.log 2>&1
