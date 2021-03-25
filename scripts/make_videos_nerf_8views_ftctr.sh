#!/bin/bash

## Render with 8 views at 250k iterations total at full 800x800 resolution, more poses for a smoother video
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --full_res --num_render_poses=80 --checkpoint_rendering --render_only --reload_iter 250000 --config configs/283_blender_chair_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --full_res --num_render_poses=80 --checkpoint_rendering --render_only --reload_iter 250000 --config configs/284_blender_drums_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --full_res --num_render_poses=80 --checkpoint_rendering --render_only --reload_iter 250000 --config configs/285_blender_ficus_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --full_res --num_render_poses=80 --checkpoint_rendering --render_only --reload_iter 250000 --config configs/286_blender_lego_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --full_res --num_render_poses=80 --checkpoint_rendering --render_only --reload_iter 250000 --config configs/282_blender_mic_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --full_res --num_render_poses=80 --checkpoint_rendering --render_only --reload_iter 250000 --config configs/289_blender_ship_8views_ftctr.txt &
# CUDA_VISIBLE_DEVICES=6 python run_nerf.py --full_res --num_render_poses=80 --checkpoint_rendering --render_only --reload_iter 250000 --config configs/288_blender_hotdog_8views_ftctr_200k.txt &
# CUDA_VISIBLE_DEVICES=7 python run_nerf.py --full_res --num_render_poses=80 --checkpoint_rendering --render_only --reload_iter 250000 --config configs/287_blender_materials_8views_ftctr_200k.txt &

find logs/283_blender_chair_8views_ftctr250 -wholename "logs/283_blender_chair_8views_ftctr250/renderonly_path*/*.mp4"
find logs/284_blender_drums_8views_ftctr251 -wholename "logs/284_blender_drums_8views_ftctr251/renderonly_path*/*.mp4"
find logs/285_blender_ficus_8views_ftctr252 -wholename "logs/285_blender_ficus_8views_ftctr252/renderonly_path*/*.mp4"
find logs/286_blender_lego_8views_ftctr042 -wholename "logs/286_blender_lego_8views_ftctr042/renderonly_path*/*.mp4"
find logs/282_blender_mic_8views_ftctr254 -wholename "logs/282_blender_mic_8views_ftctr254/renderonly_path*/*.mp4"
find logs/289_blender_ship_8views_ftctr255 -wholename "logs/289_blender_ship_8views_ftctr255/renderonly_path*/*.mp4"
find logs/288_blender_hotdog_8views_ftctr256_200k -wholename "logs/288_blender_hotdog_8views_ftctr256_200k/renderonly_path*/*.mp4"
find logs/287_blender_materials_8views_ftctr258_200k -wholename "logs/287_blender_materials_8views_ftctr258_200k/renderonly_path*/*.mp4"

# 281_blender_mic_8views_ftctr254_fullres
# 282_blender_mic_8views_ftctr254
# 283_blender_chair_8views_ftctr250
# 284_blender_drums_8views_ftctr251
# 285_blender_ficus_8views_ftctr252
# 286_blender_lego_8views_ftctr042
# 287_blender_materials_8views_ftctr258_150k
# 287_blender_materials_8views_ftctr258_200k
# 288_blender_hotdog_8views_ftctr256
# 288_blender_hotdog_8views_ftctr256_150k
# 288_blender_hotdog_8views_ftctr256_200k
# 289_blender_ship_8views_ftctr255
# 321_blender_lego_8views_right_ftctr314_130k


cp logs/283_blender_chair_8views_ftctr250/renderonly_path_249999/video.mp4      logs/smoothvideo_ftctr/logs-283_blender_chair_8views_ftctr250-renderonly_path_249999-video.mp4
cp logs/284_blender_drums_8views_ftctr251/renderonly_path_249999/video.mp4      logs/smoothvideo_ftctr/logs-284_blender_drums_8views_ftctr251-renderonly_path_249999-video.mp4
cp logs/285_blender_ficus_8views_ftctr252/renderonly_path_249999/video.mp4      logs/smoothvideo_ftctr/logs-285_blender_ficus_8views_ftctr252-renderonly_path_249999-video.mp4
cp logs/286_blender_lego_8views_ftctr042/renderonly_path_249999/video.mp4       logs/smoothvideo_ftctr/logs-286_blender_lego_8views_ftctr042-renderonly_path_249999-video.mp4
cp logs/282_blender_mic_8views_ftctr254/renderonly_path_249999/video.mp4        logs/smoothvideo_ftctr/logs-282_blender_mic_8views_ftctr254-renderonly_path_249999-video.mp4
cp logs/289_blender_ship_8views_ftctr255/renderonly_path_249999/video.mp4       logs/smoothvideo_ftctr/logs-289_blender_ship_8views_ftctr255-renderonly_path_249999-video.mp4
cp logs/288_blender_hotdog_8views_ftctr256_200k/renderonly_path_249999/video.mp4        logs/smoothvideo_ftctr/logs-288_blender_hotdog_8views_ftctr256_200k-renderonly_path_249999-video.mp4
cp logs/287_blender_materials_8views_ftctr258_200k/renderonly_path_249999/video.mp4     logs/smoothvideo_ftctr/logs-287_blender_materials_8views_ftctr258_200k-renderonly_path_249999-video.mp4