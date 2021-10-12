# DietNeRF: Semantically consistent radiance fields for few shot view synthesis

## Setup

We will use the following folder structure:
```
dietnerf/
  logs/ (images, videos, checkpoints)
  data/
    nerf_synthetic/
    nerf_llff_data/
  configs/ (run configuration files)
CLIP/ (Fork of OpenAI's clip repository with a wrapper)
```

Create conda environment and login to wandb:
```
conda create -n dietnerf python=3.9
conda activate dietnerf
wandb login
```

Set up requirements and our fork of CLIP:
```
pip install -r requirements.txt
cd CLIP
pip install -e .
```

## Experiments on the Realistic Synthetic dataset
Realistic Synthetic experiments are implemented in the `./dietnerf` subdirectory.

You need to download datasets
from [NeRF's Google Drive folder](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
The dataset was used in the original NeRF paper by Mildenhall et al. For example,
```
mkdir dietnerf/logs/ dietnerf/data/
cd dietnerf/data
pip install gdown
gdown --id 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG -O nerf_synthetic.zip
unzip nerf_synthetic.zip
rm -r __MACOSX
```

Then, shrink images to 400x400:
```
python dietnerf/scripts/bulk_shrink_images.py "dietnerf/data/nerf_synthetic/*/*/*.png" dietnerf/data/nerf_synthetic_400_rgb/ True
```
These images are used for FID/KID computation. The `dietnerf/run_nerf.py` training and evaluation code automatically shrinks images with the `--half_res` argument.

Each experiment has a config file stored in `dietnerf/configs/`. Scripts in `dietnerf/scripts/` can be run to train and evaluate models.
Run these scripts from `./dietnerf`.
The scripts assume you are running one script at a time on a server with 8 NVIDIA GPUs.
```
cd dietnerf
export WANDB_ENTITY=<your wandb username>

# NeRF baselines
sh scripts/run_synthetic_nerf_100v.sh
sh scripts/run_synthetic_nerf_8v.sh
sh scripts/run_synthetic_simplified_nerf_8v.sh

# DietNeRF with 8 observed views
sh scripts/run_synthetic_dietnerf_8v.sh
sh scripts/run_synthetic_dietnerf_ft_8v.sh

# NeRF and DietNeRF with partial observability
sh scripts/run_synthetic_unseen_side_14v.sh
```

## Experiments on the DTU dataset
Coming soon

## Citation and acknowledgements
If DietNeRF is relevant to your project, please cite our associated paper:
```
@article{jain2021dietnerf,
      title={Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis},
      author={Ajay Jain and Matthew Tancik and Pieter Abbeel},
      year={2021},
      journal={arXiv},
}
```
This code is based on Yen-Chen Lin's [PyTorch implementation of NeRF](https://github.com/yenchenlin/nerf-pytorch) and the [official pixelNeRF code](https://github.com/sxyu/pixel-nerf).