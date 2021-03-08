from collections import namedtuple

import fire

Scene = namedtuple('Scene', 'name datadir')

SYNTHETIC_SCENES = [
    Scene('chair', './data/nerf_synthetic/chair'),
    Scene('drums', './data/nerf_synthetic/drums'),
    Scene('ficus', './data/nerf_synthetic/ficus'),
    Scene('lego', './data/nerf_synthetic/lego'),
    Scene('mic', './data/nerf_synthetic/mic'),
    Scene('ship', './data/nerf_synthetic/ship'),
]

base_synthetic_config = \
"""basedir = ./logs
dataset_type = blender
no_batching = True
use_viewdirs = True
white_bkgd = True
lrate_decay = 500
N_samples = 64
N_importance = 128
N_rand = 1024
precrop_iters = 500
precrop_frac = 0.5
half_res = True
"""

def make_synthetic_scenes(start_id, max_train_views=-1):
    commands = []
    for i, scene in enumerate(SYNTHETIC_SCENES):
        expname = f"{start_id + i}_blender_{scene.name}_{max_train_views}views"
        config = \
f"""expname = {expname}
datadir = {scene.datadir}
{base_synthetic_config}
## Additional arguments
max_train_views = {max_train_views}
i_log_raw_hist = 50
i_video = 6250
save_splits = True"""
        out_path = f'configs/{expname}.txt'
        print("==== WRITING TO", out_path)
        print(config)
        with open(out_path, 'w') as f:
            f.write(config)
        print("=============================")

        command = f"PYTHONPATH=/data/ajay/clip/CLIP:$PYTHONPATH CUDA_VISIBLE_DEVICES={i} python run_nerf.py --config {out_path} &"
        commands.append(command)

    print("=========== COMMANDS")
    commands = '#!/bin/bash\n' + '\n'.join(commands)
    print(commands)
    # with open(f'scripts/{start_id}_run_synthetic_{max_train_views}views.sh')



if __name__=='__main__':
    fire.Fire()
