#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/210_blender_chair_2views.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/211_blender_drums_2views.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/212_blender_ficus_2views.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/213_blender_lego_2views.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/214_blender_mic_2views.txt &
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/215_blender_ship_2views.txt &