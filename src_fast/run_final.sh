#!/bin/bash
set -e

# 合并lora
python merge_lora.py
python run_gptq.py