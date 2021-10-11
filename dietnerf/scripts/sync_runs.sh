#!/bin/bash

PABDGX_SOURCE=ajay@pabdgx-1.ist.berkeley.edu:/data/ajay/clip/nerf/nerf-pytorch/logs
B7_1_SOURCE=ajay@b7_1:/home/ajay/clip/nerf/nerf-pytorch/logs
DEST=/shared/ajay/clip/nerf/nerf-pytorch/logs/

rsync -avhtr ${PABDGX_SOURCE}/2*_blender_*views_ctr $DEST
rsync -avhtr ${B7_1_SOURCE}/2*_blender_*views_ctr $DEST
rsync -avhtr ${PABDGX_SOURCE}/042_blender_paper_lego_ctr_coarseandfine_clip_vit_reuseemb_uniformpose $DEST

rsync -avhtr ${PABDGX_SOURCE}/2*_blender_*_*views $DEST
rsync -avhtr ${B7_1_SOURCE}/2*_blender_*_*views $DEST
