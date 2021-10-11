#!/bin/bash

## Train
# Running on pabdgx-1 [3/15/21 6:49 pm]
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/330_blender_chair_8views_simple.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/331_blender_drums_8views_simple.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/332_blender_ficus_8views_simple.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/334_blender_mic_8views_simple.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/335_blender_ship_8views_simple.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/336_blender_hotdog_8views_simple.txt &  ## KILLED. degenerate

# Running on pabrtxs3
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/337_blender_materials_8views_simple.txt &

# Skip. instead use /shared/ajay/clip/nerf/nerf-pytorch/configs/313_blender_lego_8views_tune.txt  ## WHICH WAS DEGENERATE
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/333_blender_lego_8views_simple.txt &

## Simpler experiments for lego, hotdog, materials which were degenerate early
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/343_blender_lego_8views_simpler.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/346_blender_hotdog_8views_simpler.txt &  ## Running pabdgx1 7:08 pm
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/347_blender_materials_8views_simpler.txt &

CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/345_blender_hotdog_8views_simplest.txt &

## Working runs
312_blender_lego_8views_tune
330_blender_chair_8views_simple
331_blender_drums_8views_simple
332_blender_ficus_8views_simple
334_blender_mic_8views_simple
335_blender_ship_8views_simple
## still waiting to tell
346_blender_hotdog_8views_simpler
347_blender_materials_8views_simpler


## Test
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/345_blender_hotdog_8views_simplest.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/347_blender_materials_8views_simpler.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/335_blender_ship_8views_simple.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/331_blender_drums_8views_simple.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/330_blender_chair_8views_simple.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/334_blender_mic_8views_simple.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --render_only --render_test --config configs/332_blender_ficus_8views_simple.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --render_only --render_test --config configs/312_blender_lego_8views_tune.txt &

## FID and KID
cp -r logs/330_blender_chair_8views_simple/testset_200000 logs/nerf_simple_images_8/chair_testset_200000
cp -r logs/334_blender_mic_8views_simple/testset_200000 logs/nerf_simple_images_8/mic_testset_200000
cp -r logs/332_blender_ficus_8views_simple/testset_200000 logs/nerf_simple_images_8/ficus_testset_200000
cp -r logs/312_blender_lego_8views_tune/testset_200000 logs/nerf_simple_images_8/lego_testset_200000

cp -r logs/330_blender_chair_8views_simple/testset logs/true_test_images/chair_testset
cp -r logs/334_blender_mic_8views_simple/testset logs/true_test_images/mic_testset
cp -r logs/332_blender_ficus_8views_simple/testset logs/true_test_images/ficus_testset
cp -r logs/312_blender_lego_8views_tune/testset logs/true_test_images/lego_testset

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_simple_images_8/ logs/true_test_images
Creating feature extractor "inception-v3-compat" with features ['2048', 'logits_unbiased']
Extracting features from input_1
Looking for samples recursively in "logs/nerf_simple_images_8/" with extensions png,jpg,jpeg
Found 200 samples
Processing samples
Extracting features from input_2
Looking for samples recursively in "logs/true_test_images" with extensions png,jpg,jpeg
Found 200 samples
Processing samples
Computing Inception Score
Computing Frechet Inception Distance
Computing Kernel Inception Distance
inception_score_mean: 4.560575
inception_score_std: 0.7868282
frechet_inception_distance: 201.0543
kernel_inception_distance_mean: 0.04715515
kernel_inception_distance_std: 1.051532e-07

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_simple_images_8/ data/nerf_synthetic
Creating feature extractor "inception-v3-compat" with features ['2048', 'logits_unbiased']
Extracting features from input_1
Looking for samples recursively in "logs/nerf_simple_images_8/" with extensions png,jpg,jpeg
Found 200 samples
Processing samples
Extracting features from input_2
Looking for samples recursively in "data/nerf_synthetic" with extensions png,jpg,jpeg
Found 6400 samples
Processing samples
Computing Inception Score
Computing Frechet Inception Distance
Computing Kernel Inception Distance
inception_score_mean: 4.560575
inception_score_std: 0.7868282
frechet_inception_distance: 220.0247
kernel_inception_distance_mean: 0.06602992
kernel_inception_distance_std: 0.001822898