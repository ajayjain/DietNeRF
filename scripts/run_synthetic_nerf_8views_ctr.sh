#!/bin/bash
## Train
# Running on b7_1
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/250_blender_chair_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/251_blender_drums_8views_ctr.txt &

# Running on pabdgx-1
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/252_blender_ficus_8views_ctr.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/254_blender_mic_8views_ctr.txt &

# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/255_blender_ship_8views_ctr.txt &

# Skip (using earlier run 042 for lego)
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/253_blender_lego_8views_ctr.txt &

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
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/256_blender_hotdog_8views_ctr_resume.txt &


## FID and KID
cp -r logs/256_blender_hotdog_8views_ctr/testset_200000 logs/scarf_images_8/hotdog_testset_200000
cp -r logs/254_blender_mic_8views_ctr/testset_200000 logs/scarf_images_8/mic_testset_200000
cp -r logs/255_blender_ship_8views_ctr/testset_200000 logs/scarf_images_8/ship_testset_200000
cp -r logs/042_blender_paper_lego_ctr_coarseandfine_clip_vit_reuseemb_uniformpose/testset_200000 logs/scarf_images_8/lego_testset_200000
cp -r logs/252_blender_ficus_8views_ctr/testset_200000 logs/scarf_images_8/ficus_testset_200000
cp -r logs/251_blender_drums_8views_ctr/testset_200000 logs/scarf_images_8/drums_testset_200000
cp -r logs/250_blender_chair_8views_ctr/testset_200000 logs/scarf_images_8/chair_testset_200000
cp -r logs/258_blender_materials_8views_ctr/testset_200000 logs/scarf_images_8/materials_testset_200000

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarf_images_8/ logs/true_test_images
Creating feature extractor "inception-v3-compat" with features ['2048', 'logits_unbiased']
Extracting features from input_1
Looking for samples recursively in "logs/scarf_images_8/" with extensions png,jpg,jpeg
Found 200 samples
Processing samples
Extracting features from input_2
Looking for samples recursively in "logs/true_test_images" with extensions png,jpg,jpeg
Found 200 samples
Processing samples
Computing Inception Score
Computing Frechet Inception Distance
Computing Kernel Inception Distance
inception_score_mean: 5.73159
inception_score_std: 0.3925282
frechet_inception_distance: 72.47652
kernel_inception_distance_mean: 0.002727995
kernel_inception_distance_std: 7.708944e-08