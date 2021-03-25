CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/317_blender_lego_8views_right_tune.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/315_blender_lego_8views_right.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/314_blender_lego_8views_ctr_right.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/321_blender_lego_8views_right_ftctr314_130k.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/319_blender_lego_ctr_right_tune.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/configs/lego.txt &

317_blender_lego_8views_right_tune test metrics (./logs/317_blender_lego_8views_right_tune/renderonly_test_199999/test_metrics.json): {'mse': 0.010389125433889257, 'psnr': 21.552794770971776, 'ssim': 0.818427028978732, 'lpips': 0.16045555472373962}
315_blender_lego_8views_right test metrics (./logs/315_blender_lego_8views_right/renderonly_test_199999/test_metrics.json): {'mse': 0.047060517815402186, 'psnr': 19.662171553535135, 'ssim': 0.7989012138491661, 'lpips': 0.20198333263397217}
319_blender_lego_ctr_right_tune test metrics (./logs/319_blender_lego_ctr_right_tune/renderonly_test_199999/test_metrics.json): {'mse': 0.011339320410207528, 'psnr': 20.763049350742996, 'ssim': 0.8010880198380628, 'lpips': 0.16844302415847778}
314_blender_lego_8views_ctr_right test metrics (./logs/314_blender_lego_8views_ctr_right/renderonly_test_199999/test_metrics.json): {'mse': 0.015154157314942004, 'psnr': 20.752635124067698, 'ssim': 0.8099180000596051, 'lpips': 0.15682531893253326}
321_blender_lego_8views_right_ftctr314_130k test metrics (./logs/321_blender_lego_8views_right_ftctr314_130k/renderonly_test_199999/test_metrics.json): {'mse': 0.011828136482482248, 'psnr': 22.211157369285544, 'ssim': 0.8235280440142161, 'lpips': 0.14257486164569855}
blender_paper_lego test metrics (./logs/blender_paper_lego/renderonly_test_199999/test_metrics.json): {'mse': 0.0007208819449603167, 'psnr': 31.617750419505718, 'ssim': 0.965375278364357, 'lpips': 0.032723985612392426}

NeRF             & 19.662171553535135 & 0.7989012138491661 & 0.20198333263397217
Simplified NeRF  & 21.552794770971776 & 0.818427028978732  & 0.16045555472373962
Simplified SCaRF & 20.763049350742996 & 0.8010880198380628 & 0.16844302415847778
SCaRF (ours)     & 20.752635124067698 & 0.8099180000596051 & 0.15682531893253326
SCaRF, ft (ours) & 22.211157369285544 & 0.8235280440142161 & 0.14257486164569855
NeRF, 100 views  & 31.617750419505718 & 0.965375278364357  & 0.032723985612392426


NeRF            & 19.662 & 0.799 & 0.202
Simplified NeRF & 21.553 & 0.818 & 0.160
SCaRF (ours)     & 20.753 & 0.810 & 0.157
SCaRF, ft (ours) & 22.211 & 0.824 & 0.143
NeRF, 100 views  & 31.618 & 0.965  & 0.033