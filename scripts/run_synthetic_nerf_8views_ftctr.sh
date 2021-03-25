#!/bin/bash
## Train
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/283_blender_chair_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/284_blender_drums_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/285_blender_ficus_8views_ftctr.txt &

# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/286_blender_lego_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/288_blender_hotdog_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/289_blender_ship_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=7 python run_nerf.py --config configs/287_blender_materials_8views_ftctr.txt &

# Skip (using run 282_blender_mic_8views_ftctr254)
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/282_blender_mic_8views_ftctr.txt &

## Test with 8 views
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/283_blender_chair_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/284_blender_drums_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/285_blender_ficus_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/286_blender_lego_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/282_blender_mic_8views_ftctr.txt &
# wait;
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/289_blender_ship_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/288_blender_hotdog_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/287_blender_materials_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/288_blender_hotdog_8views_ftctr_200k.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/287_blender_materials_8views_ftctr258_200k.txt &

## Test with 8 views at 210k iterations total
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/283_blender_chair_8views_ftctr.txt --reload_iter 210000 &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/284_blender_drums_8views_ftctr.txt --reload_iter 210000 &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/285_blender_ficus_8views_ftctr.txt --reload_iter 210000 &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/286_blender_lego_8views_ftctr.txt  --reload_iter 210000 &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/282_blender_mic_8views_ftctr.txt   --reload_iter 210000 &
# wait;
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --reload_iter 210000 --config configs/289_blender_ship_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --reload_iter 210000 --config configs/288_blender_hotdog_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --reload_iter 210000 --config configs/287_blender_materials_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --reload_iter 210000 --config configs/288_blender_hotdog_8views_ftctr_200k.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --reload_iter 210000 --config configs/287_blender_materials_8views_ftctr_200k.txt &

## Re-test 288, 287 at 250k iterations total
CUDA_VISIBLE_DEVICES=0,1 python run_nerf.py --render_only --render_test --reload_iter 250000 --config configs/288_blender_hotdog_8views_ftctr_200k.txt &
CUDA_VISIBLE_DEVICES=2,3 python run_nerf.py --render_only --render_test --reload_iter 250000 --config configs/287_blender_materials_8views_ftctr_200k.txt &


cp -r logs/282_blender_mic_8views_ftctr254/testset_250000/ logs/scarfft_images_8/mic_testset_250000
cp -r logs/283_blender_chair_8views_ftctr250/testset_250000/ logs/scarfft_images_8/chair_testset_250000
cp -r logs/284_blender_drums_8views_ftctr251/testset_250000/ logs/scarfft_images_8/drums_testset_250000
cp -r logs/285_blender_ficus_8views_ftctr252/testset_250000/ logs/scarfft_images_8/ficus_testset_250000
cp -r logs/286_blender_lego_8views_ftctr042/testset_250000/ logs/scarfft_images_8/lego_testset_250000
cp -r logs/287_blender_materials_8views_ftctr258_200k/testset_250000/ logs/scarfft_images_8/materials_testset_250000
cp -r logs/288_blender_hotdog_8views_ftctr256_200k/testset_250000/ logs/scarfft_images_8/hotdog_testset_250000
cp -r logs/289_blender_ship_8views_ftctr255/testset_250000/ logs/scarfft_images_8/ship_testset_250000
find logs/scarfft_images_8 -name *.png | wc -l
fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarfft_images_8/ logs/true_test_images
# Creating feature extractor "inception-v3-compat" with features ['2048', 'logits_unbiased']
# Extracting features from input_1
# Looking for samples recursively in "logs/scarfft_images_8/" with extensions png,jpg,jpeg
# Found 200 samples
# Processing samples
# Extracting features from input_2
# Looking for samples recursively in "logs/true_test_images" with extensions png,jpg,jpeg
# Found 200 samples
# Processing samples
# Computing Inception Score
# Computing Frechet Inception Distance
# Computing Kernel Inception Distance
# inception_score_mean: 5.524686
# inception_score_std: 0.4460831
# frechet_inception_distance: 68.10616
# kernel_inception_distance_mean: 0.002675536
# kernel_inception_distance_std: 1.170947e-07
fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarfft_images_8/ data/nerf_synthetic



## For NeRF with 100 views
find logs/nerf_images_100 -name *.png | wc -l
fidelity --gpu 1 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_100/ logs/true_test_images
# Creating feature extractor "inception-v3-compat" with features ['logits_unbiased', '2048']
# Extracting features from input_1
# Looking for samples recursively in "logs/nerf_images_100/" with extensions png,jpg,jpeg
# Found 200 samples
# Processing samples
# Extracting features from input_2
# Looking for samples recursively in "logs/true_test_images" with extensions png,jpg,jpeg
# Found 200 samples
# Processing samples
# Computing Inception Score
# Computing Frechet Inception Distance
# Computing Kernel Inception Distance
# inception_score_mean: 6.067405
# inception_score_std: 0.6212252
# frechet_inception_distance: 34.52845
# kernel_inception_distance_mean: -0.0009028905
# kernel_inception_distance_std: 1.281454e-07
fidelity --gpu 1 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_100/ data/...
200 vs 6400 samples
inception_score_mean: 6.067405
inception_score_std: 0.6212252
frechet_inception_distance: 153.3308
kernel_inception_distance_mean: 0.04256576
kernel_inception_distance_std: 0.002897964

#####

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_100/ data/nerf_synthetic_400
inception_score_mean: 6.067405
inception_score_std: 0.6212252
frechet_inception_distance: 150.9984
kernel_inception_distance_mean: 0.04325217
kernel_inception_distance_std: 0.002748292

fidelity --gpu 3 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_simple_images_8/ data/nerf_synthetic_400
inception_score_mean: 4.560575
inception_score_std: 0.7868282
frechet_inception_distance: 222.1301
kernel_inception_distance_mean: 0.06973465
kernel_inception_distance_std: 0.001912131

fidelity --gpu 1 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarf_images_8/ data/nerf_synthetic_400
inception_score_mean: 5.73159
inception_score_std: 0.3925282
frechet_inception_distance: 163.1665
kernel_inception_distance_mean: 0.04811746
kernel_inception_distance_std: 0.002719963

fidelity --gpu 2 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarfft_images_8/ data/nerf_synthetic_400
inception_score_mean: 5.524686
inception_score_std: 0.4460831
frechet_inception_distance: 161.8603
kernel_inception_distance_mean: 0.04814288
kernel_inception_distance_std: 0.002810767

fidelity --gpu 2 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_8/ data/nerf_synthetic_400
inception_score_mean: 4.347547
inception_score_std: 0.9040484
frechet_inception_distance: 250.6511
kernel_inception_distance_mean: 0.09696839
kernel_inception_distance_std: 0.002371341

fidelity --gpu 2 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_4/ data/nerf_synthetic_400
inception_score_mean: 3.10864
inception_score_std: 0.4524474
frechet_inception_distance: 320.0606
kernel_inception_distance_mean: 0.176157
kernel_inception_distance_std: 0.003247311

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarf_images_4/ data/nerf_synthetic_400
inception_score_mean: 5.995791
inception_score_std: 0.3400172
frechet_inception_distance: 200.609
kernel_inception_distance_mean: 0.05604133
kernel_inception_distance_std: 0.002076527