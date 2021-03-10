#!/bin/bash
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/chair.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/drums.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/ficus.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/lego.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/mic.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/ship.txt &

## Test
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/lego.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/mic.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/ship.txt &
wait;
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/chair.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/drums.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/ficus.txt &
