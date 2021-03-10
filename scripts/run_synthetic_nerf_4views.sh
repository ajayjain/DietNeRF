#!/bin/bash
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/220_blender_chair_4views.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/221_blender_drums_4views.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=7 python run_nerf.py --config configs/222_blender_ficus_4views.txt &

# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/223_blender_lego_4views.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/224_blender_mic_4views.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/225_blender_ship_4views.txt &


## Test
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/223_blender_lego_4views.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/224_blender_mic_4views.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/225_blender_ship_4views.txt &
wait;
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/220_blender_chair_4views.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/221_blender_drums_4views.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/222_blender_ficus_4views.txt &
