#!/bin/bash

## Test
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --full_res --num_render_poses=80 --render_only --config configs/345_blender_hotdog_8views_simplest.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --full_res --num_render_poses=80 --render_only --config configs/347_blender_materials_8views_simpler.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --full_res --num_render_poses=80 --render_only --config configs/335_blender_ship_8views_simple.txt &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --full_res --num_render_poses=80 --render_only --config configs/331_blender_drums_8views_simple.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --full_res --num_render_poses=80 --render_only --config configs/330_blender_chair_8views_simple.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --full_res --num_render_poses=80 --render_only --config configs/334_blender_mic_8views_simple.txt &
# CUDA_VISIBLE_DEVICES=6 python run_nerf.py --full_res --num_render_poses=80 --render_only --config configs/332_blender_ficus_8views_simple.txt &
# CUDA_VISIBLE_DEVICES=7 python run_nerf.py --full_res --num_render_poses=80 --render_only --config configs/312_blender_lego_8views_tune.txt &

find logs/345_blender_hotdog_8views_simplest -wholename "logs/345_blender_hotdog_8views_simplest/renderonly_path*/*.mp4"
find logs/347_blender_materials_8views_simpler -wholename "logs/347_blender_materials_8views_simpler/renderonly_path*/*.mp4"
find logs/335_blender_ship_8views_simple -wholename "logs/335_blender_ship_8views_simple/renderonly_path*/*.mp4"
find logs/331_blender_drums_8views_simple -wholename "logs/331_blender_drums_8views_simple/renderonly_path*/*.mp4"
find logs/330_blender_chair_8views_simple -wholename "logs/330_blender_chair_8views_simple/renderonly_path*/*.mp4"
find logs/334_blender_mic_8views_simple -wholename "logs/334_blender_mic_8views_simple/renderonly_path*/*.mp4"
find logs/332_blender_ficus_8views_simple -wholename "logs/332_blender_ficus_8views_simple/renderonly_path*/*.mp4"
find logs/312_blender_lego_8views_tune -wholename "logs/312_blender_lego_8views_tune/renderonly_path*/*.mp4"


mkdir logs/smoothvideos
cp logs/345_blender_hotdog_8views_simplest/renderonly_path_199999/video.mp4       logs/smoothvideos/345_blender_hotdog_8views_simplest-renderonly_path_199999-video.mp4
cp logs/347_blender_materials_8views_simpler/renderonly_path_199999/video.mp4         logs/smoothvideos/347_blender_materials_8views_simpler-renderonly_path_199999-video.mp4
cp logs/335_blender_ship_8views_simple/renderonly_path_199999/video.mp4       logs/smoothvideos/335_blender_ship_8views_simple-renderonly_path_199999-video.mp4
cp logs/331_blender_drums_8views_simple/renderonly_path_199999/video.mp4          logs/smoothvideos/331_blender_drums_8views_simple-renderonly_path_199999-video.mp4
cp logs/330_blender_chair_8views_simple/renderonly_path_199999/video.mp4          logs/smoothvideos/330_blender_chair_8views_simple-renderonly_path_199999-video.mp4
cp logs/334_blender_mic_8views_simple/renderonly_path_199999/video.mp4        logs/smoothvideos/334_blender_mic_8views_simple-renderonly_path_199999-video.mp4
cp logs/332_blender_ficus_8views_simple/renderonly_path_199999/video.mp4          logs/smoothvideos/332_blender_ficus_8views_simple-renderonly_path_199999-video.mp4
cp logs/312_blender_lego_8views_tune/renderonly_path_199999/video.mp4         logs/smoothvideos/312_blender_lego_8views_tune-renderonly_path_199999-video.mp4

## PixelNeRF
python eval/gen_video.py -n pn051_dtu_ftall1v_lr1e-5_scan21 --gpu_id="0 1 2 3" --split val -P '25' -D data/rs_dtu_4 -S 1 --dataset_format dvr_dtu
python eval/gen_video.py -n pn031_dtu_ftall1v_lr1e-5_ctr_scale.4_scan21 --gpu_id="4 5 6" --split val -P '25' -D data/rs_dtu_4 -S 1 --dataset_format dvr_dtu

python eval/gen_video.py -n pn051_dtu_ftall1v_lr1e-5_scan21 --gpu_id="0 1 2 3" --split val -P '25' -D data/rs_dtu_4 -S 1 --dataset_format dvr_dtu --scale 2.0
python eval/gen_video.py -n pn031_dtu_ftall1v_lr1e-5_ctr_scale.4_scan21 --gpu_id="4 5 6" --split val -P '25' -D data/rs_dtu_4 -S 1 --dataset_format dvr_dtu --scale 2.0