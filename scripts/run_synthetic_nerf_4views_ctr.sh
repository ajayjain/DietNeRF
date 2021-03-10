#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/240_blender_chair_4views_ctr.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/241_blender_drums_4views_ctr.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/242_blender_ficus_4views_ctr.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/243_blender_lego_4views_ctr.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/244_blender_mic_4views_ctr.txt &
# CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/245_blender_ship_4views_ctr.txt &

## Test with 4 views
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/243_blender_lego_4views_ctr.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/244_blender_mic_4views_ctr.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/245_blender_ship_4views_ctr.txt &
wait;
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/240_blender_chair_4views_ctr.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/241_blender_drums_4views_ctr.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/242_blender_ficus_4views_ctr.txt &
