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

### FID KID
cp -r logs/226_blender_materials_4views/testset_200000 logs/nerf_images_4/materials_testset_200000
cp -r logs/226_blender_hotdog_4views/testset_200000 logs/nerf_images_4/hotdog_testset_200000
cp -r logs/222_blender_ficus_4views/testset_200000 logs/nerf_images_4/ficus_testset_200000
cp -r logs/221_blender_drums_4views/testset_200000 logs/nerf_images_4/drums_testset_200000
cp -r logs/220_blender_chair_4views/testset_200000 logs/nerf_images_4/chair_testset_200000
cp -r logs/225_blender_ship_4views/testset_200000 logs/nerf_images_4/ship_testset_200000
cp -r logs/224_blender_mic_4views/testset_200000 logs/nerf_images_4/mic_testset_200000
cp -r logs/223_blender_lego_4views/testset_200000 logs/nerf_images_4/lego_testset_200000

220_blender_chair_4views
221_blender_drums_4views
222_blender_ficus_4views
223_blender_lego_4views
224_blender_mic_4views
225_blender_hotdog_4views
225_blender_ship_4views
226_blender_hotdog_4views
226_blender_materials_4views
240_blender_chair_4views_ctr