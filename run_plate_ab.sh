#!/bin/bash
# A/B test plate_vote (no DB writes), same CUDA env as run_kfz.sh
SP=/home/gh/python/venv_py311/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cufft/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cublas/lib:$SP/torch/lib:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=0
exec /home/gh/python/venv_py311/bin/python3 /home/gh/python/reolink_AI/plate_vote_ab.py "$@"
