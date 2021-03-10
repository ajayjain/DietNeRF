#!/bin/bash

## Train
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/230_blender_chair_8views.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/231_blender_drums_8views.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/232_blender_ficus_8views.txt &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/233_blender_lego_8views.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/234_blender_mic_8views.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/235_blender_ship_8views.txt &

CUDA_VISIBLE_DEVICES=2,3 python run_nerf.py --config configs/236_blender_hotdog_8views.txt &
CUDA_VISIBLE_DEVICES=5,6,7 python run_nerf.py --config configs/237_blender_materials_8views.txt &

## Test
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/230_blender_chair_8views.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/231_blender_drums_8views.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/232_blender_ficus_8views.txt &
# wait;
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/233_blender_lego_8views.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/234_blender_mic_8views.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/235_blender_ship_8views.txt &