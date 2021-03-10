#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/250_blender_chair_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/251_blender_drums_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/252_blender_ficus_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/253_blender_lego_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/254_blender_mic_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/255_blender_ship_8views_ctr.txt &

## Test with 8 views
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/250_blender_chair_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/251_blender_drums_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/042_blender_paper_lego.txt &