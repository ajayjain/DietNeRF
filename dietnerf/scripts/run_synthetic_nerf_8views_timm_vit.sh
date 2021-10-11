049_blender_lego_ctr_timm_vit-base-patch32-224_coarseandfine_reuseemb_uniformpose
052_blender_lego_ctr_timm_vit-large-patch16-384_coarseandfine_reuseemb_uniformpose
053_blender_lego_ctr_timm_vit-large-patch32-384_coarseandfine_reuseemb_uniformpose
042_blender_paper_lego_ctr_coarseandfine_clip_vit_reuseemb_uniformpose
/shared/ajay/clip/nerf/nerf-pytorch/configs/042_blender_paper_lego.txt

049_blender_lego.txt


cp -r /data/ajay/clip/nerf/nerf-pytorch/logs/049_blender_lego_ctr_timm_vit-base-patch32-224_coarseandfine_reuseemb_uniformpose/* 049_blender_lego_ctr_timm_vit-base-patch32-224_coarseandfine_reuseemb_uniformpose/
cp -r /data/ajay/clip/nerf/nerf-pytorch/logs/052_blender_lego_ctr_timm_vit-large-patch16-384_coarseandfine_reuseemb_uniformpose/* 052_blender_lego_ctr_timm_vit-large-patch16-384_coarseandfine_reuseemb_uniformpose/
cp -r /data/ajay/clip/nerf/nerf-pytorch/logs/053_blender_lego_ctr_timm_vit-large-patch32-384_coarseandfine_reuseemb_uniformpose/* 053_blender_lego_ctr_timm_vit-large-patch32-384_coarseandfine_reuseemb_uniformpose/


## Test at 200k iterations total
random nerfs -- didn\'t load weights
  049_blender_lego_ctr_timm_vit-base-patch32-224_coarseandfine_reuseemb_uniformpose test metrics (./logs/049_blender_lego_ctr_timm_vit-base-patch32-224_coarseandfine_reuseemb_uniformpose/renderonly_test_000000/test_metrics.json): {'mse': 0.15310210997874885, 'psnr': 8.248913593108481, 'ssim': 0.08392188421906084, 'lpips': 0.7751548886299133}
  052_blender_lego_ctr_timm_vit-large-patch16-384_coarseandfine_reuseemb_uniformpose test metrics (./logs/052_blender_lego_ctr_timm_vit-large-patch16-384_coarseandfine_reuseemb_uniformpose/renderonly_test_000000/test_metrics.json): {'mse': 0.15310210997874885, 'psnr': 8.248913593108481, 'ssim': 0.08392188421906084, 'lpips': 0.7751548886299133}
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/049_blender_lego.txt &
  049_blender_lego_ctr_timm_vit-base-patch32-224_coarseandfine_reuseemb_uniformpose test metrics (./logs/049_blender_lego_ctr_timm_vit-base-patch32-224_coarseandfine_reuseemb_uniformpose/renderonly_test_199999/test_metrics.json): {'mse': 0.007752414546123216, 'psnr': 22.05851890801957, 'ssim': 0.8355417396820325, 'lpips': 0.13147972524166107}
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/052_blender_lego.txt &
  052_blender_lego_ctr_timm_vit-large-patch16-384_coarseandfine_reuseemb_uniformpose test metrics (./logs/052_blender_lego_ctr_timm_vit-large-patch16-384_coarseandfine_reuseemb_uniformpose/renderonly_test_199999/test_metrics.json): {'mse': 0.00830004751509752, 'psnr': 21.501397039861093, 'ssim': 0.8088294601423479, 'lpips': 0.16682079434394836}
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/053_blender_lego.txt &
  053_blender_lego_ctr_timm_vit-large-patch32-384_coarseandfine_reuseemb_uniformpose test metrics (./logs/053_blender_lego_ctr_timm_vit-large-patch32-384_coarseandfine_reuseemb_uniformpose/renderonly_test_199999/test_metrics.json): {'mse': 0.010742842160200423, 'psnr': 20.49800814780044, 'ssim': 0.8008540460308295, 'lpips': 0.1741504669189453}
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/042_blender_paper_lego.txt &
  {"mse": 0.005002116529646187, "psnr": 23.89649989400083, "ssim": 0.8626880405360791, "lpips": 0.11029504239559174, "args": {"config": "configs/042_blender_paper_lego.txt", "expname": "042_blender_paper_lego_ctr_coarseandfine_clip_vit_reuseemb_uniformpose", "basedir": "./logs", "datadir": "./data/nerf_synthetic/lego", "netdepth": 8, "netwidth": 256, "netdepth_fine": 8, "netwidth_fine": 256, "N_rand": 1024, "lrate": 0.0005, "lrate_decay": 500, "chunk": 32768, "netchunk_per_gpu": 262144, "no_batching": true, "no_reload": false, "ft_path": null, "seed": 0, "use_softplus_alpha": false, "N_samples": 64, "N_importance": 128, "perturb": 1.0, "use_viewdirs": true, "i_embed": 0, "multires": 10, "multires_views": 4, "raw_noise_std": 0.0, "render_only": true, "render_test": true, "render_factor": 0, "precrop_iters": 500, "precrop_frac": 0.5, "N_iters": 200000, "dataset_type": "blender", "testskip": 8, "shape": "greek", "white_bkgd": true, "half_res": true, "factor": 8, "no_ndc": false, "lindisp": false, "spherify": false, "llffhold": 8, "wandb_entity": "ajayj", "i_log": 1, "i_log_raw_hist": 2, "i_print": 100, "i_img": 500, "i_weights": 10000, "i_testset": 50000, "i_video": 6250, "save_splits": false, "i_log_ctr_img": 10, "max_train_views": 8, "render_loss_interval": 10.0, "render_increase_interval_every": 0, "render_increase_interval_delta": 0, "render_autocast": false, "render_poses": "uniform", "render_poses_translation_jitter_sigma": 0.0, "render_poses_interpolate_range": [0.0, 1.0], "render_theta_range": [-180.0, 180.0], "render_phi_range": [-90.0, 0.0], "render_radius_range": [3.5, 4.5], "render_nH": 168, "render_nW": 168, "render_jitter_rays": true, "checkpoint_rendering": true, "checkpoint_embedding": false, "no_mse": false, "pixel_interp_mode": "bicubic", "feature_interp_mode": "bilinear", "consistency_loss": "consistent_with_target_rep", "consistency_loss_comparison": ["cosine_sim"], "consistency_loss_sampling": "single_random", "consistency_loss_sampling_temp": 1.0, "consistency_loss_lam": 0.1, "consistency_loss_lam0": 0.1, "consistency_target_augmentation": "none", "consistency_size": 224, "consistency_model_type": "clip_vit", "consistency_model_num_layers": -1, "aligned_loss": false, "mask_aligned_loss": false, "spatial_model_type": "clip_rn50", "spatial_model_num_layers": -1, "aligned_loss_comparison": "cosine_sim", "aligned_loss_lam": 0.1, "aligned_loss_lam0": 0.1, "patch_gan_loss": false, "patch_gan_D_inner_steps": 1, "patch_gan_D_grad_clip": -1, "patch_gan_D_nH": 168, "patch_gan_D_nW": 168, "patch_wgan_lambda_gp": 0.1, "patch_gan_D_activation": "none", "patch_gan_mode": "lsgan", "patch_gan_G_lam": 1.0, "patch_gan_lr": 0.0002, "patch_gan_beta1": 0.5, "patch_gan_num_Ds": 1, "patch_gan_netD": "basic_256_multi", "patch_gan_ndf": 64, "patch_gan_norm": "instance", "patch_gan_init_type": "xavier", "patch_gan_init_gain": 0.02, "patch_gan_nl": "relu", "n_gpus": 1}}


049 ImageNet ViT B/32, 224 & 22.05851890801957 & 0.8355417396820325 & 0.13147972524166107
052 ImageNet ViT L/16, 384 & 21.501397039861093 & 0.8088294601423479 & 0.16682079434394836
053 ImageNet ViT L/32, 384 & 20.49800814780044 & 0.8008540460308295 & 0.1741504669189453
042 CLIP ViT B/32, 224     & 23.89649989400083 & 0.8626880405360791 & 0.11029504239559174

ImageNet ViT B/32, 224 & 22.059 & 0.836 & 0.131
ImageNet ViT L/16, 384 & 21.501 & 0.809 & 0.167
ImageNet ViT L/32, 384 & 20.498 & 0.801 & 0.174
CLIP ViT B/32, 224     & 23.896 & 0.863 & 0.110


## FT ablation
\ours{}	                           & 23.147	& 0.866	& 0.109
\ours{}, $\Lmse$ ft for 10k	iters  & 23.524	& 0.872	& 0.101
\ours{}, $\Lmse$ ft for 50k iters  & 23.591	& 0.874	& 0.097
\ours{}, $\Lmse$ ft for 100k iters & 23.521	& 0.874	& 0.097
\ours{}, $\Lmse$ ft for 200k iters & 23.443	& 0.872	& 0.098