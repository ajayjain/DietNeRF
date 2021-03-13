#!/bin/bash
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/220_blender_chair_4views.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/221_blender_drums_4views.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=7 python run_nerf.py --config configs/222_blender_ficus_4views.txt &

# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/223_blender_lego_4views.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/224_blender_mic_4views.txt &
# PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/225_blender_ship_4views.txt &

# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/226_blender_hotdog_4views.txt &
# Need to run below
# CUDA_VISIBLE_DEVICES=0,1 python run_nerf.py --config configs/227_blender_materials_4views.txt &


## Test
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/223_blender_lego_4views.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/224_blender_mic_4views.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/225_blender_ship_4views.txt &
# wait;
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/220_blender_chair_4views.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/221_blender_drums_4views.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/222_blender_ficus_4views.txt &
# wait;
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/226_blender_hotdog_4views.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/227_blender_materials_4views.txt &

## Test 4 and 8 view materials
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/236_blender_hotdog_8views.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/237_blender_materials_8views.txt &