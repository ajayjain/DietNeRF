#!/bin/bash
## Train
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/283_blender_chair_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/284_blender_drums_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/285_blender_ficus_8views_ftctr.txt &

CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/286_blender_lego_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/288_blender_hotdog_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/289_blender_ship_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --config configs/287_blender_materials_8views_ftctr.txt &

# Skip (using run 282_blender_mic_8views_ftctr254)
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/282_blender_mic_8views_ftctr.txt &

## Test with 8 views
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/250_blender_chair_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/251_blender_drums_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/252_blender_ficus_8views_ctr.txt &
# wait;
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/042_blender_paper_lego.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/254_blender_mic_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/255_blender_ship_8views_ctr.txt &
# wait;
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/256_blender_hotdog_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/257_blender_materials_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/258_blender_materials_8views_ctr.txt &
