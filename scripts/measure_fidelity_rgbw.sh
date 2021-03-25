fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_100/ data/nerf_synthetic_400_rgbw > logs/nerf_images_100.txt &
fidelity --gpu 1 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_simple_images_8/ data/nerf_synthetic_400_rgbw > logs/nerf_simple_images_8.txt &
fidelity --gpu 2 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarf_images_8/ data/nerf_synthetic_400_rgbw > logs/scarf_images_8.txt &
fidelity --gpu 3 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarfft_images_8/ data/nerf_synthetic_400_rgbw > logs/scarfft_images_8.txt &
fidelity --gpu 4 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_8/ data/nerf_synthetic_400_rgbw > logs/nerf_images_8.txt &
fidelity --gpu 5 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_4/ data/nerf_synthetic_400_rgbw > logs/nerf_images_4.txt &
fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarf_images_4/ data/nerf_synthetic_400_rgbw > logs/scarf_images_4.txt &

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarf_images_4/ data/nerf_synthetic_400_rgbw > logs/scarf_images_4.txt &

# With 1618 samples
fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nv data/nerf_synthetic_400_rgb

# With 1600 samples, 400x400 resolution
fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nv_400_rgbw/ data/nerf_synthetic_400_rgbw > logs/nv_8_400.txt


==> logs/nerf_images_100.txt <==
inception_score_mean: 6.067405
inception_score_std: 0.6212252
frechet_inception_distance: 50.48378
kernel_inception_distance_mean: 0.001273644
kernel_inception_distance_std: 0.0005610814

# Full precision
nerf 4          & 306.1708 & 0.157574
scarf 4         & 159.3824 & 0.02898115
nerf 8          & 228.1117 & 0.07600711
nerf simple 8   & 189.2016 & 0.04666717
nv 8            & 239.4725 & 0.1165526
scarf 8         & 74.87528 & 0.004647122
scarfft 8       & 72.0289  & 0.004457215
nerf 100        & 50.48378 & 0.001273644

# Full precision
nerf 4          & 306.2 & 0.158
scarf 4         & 159.4 & 0.029
nerf 8          & 228.1 & 0.076
nerf simple 8   & 189.2 & 0.047
nv 8            & 239.5 & 0.117
scarf 8         &  74.9 & 0.005
scarfft 8       &  72.0 & 0.004
nerf 100        &  50.5 & 0.001

==> logs/nerf_images_4.txt <==
inception_score_mean: 3.10864
inception_score_std: 0.4524474
frechet_inception_distance: 306.1708
kernel_inception_distance_mean: 0.157574
kernel_inception_distance_std: 0.001958814

==> logs/nerf_images_8.txt <==
inception_score_mean: 4.347547
inception_score_std: 0.9040484
frechet_inception_distance: 228.1117
kernel_inception_distance_mean: 0.07600711
kernel_inception_distance_std: 0.001269367

==> logs/nerf_simple_images_8.txt <==
inception_score_mean: 4.560575
inception_score_std: 0.7868282
frechet_inception_distance: 189.2016
kernel_inception_distance_mean: 0.04666717
kernel_inception_distance_std: 0.002327858

==> logs/scarfft_images_8.txt <==
inception_score_mean: 5.524686
inception_score_std: 0.4460831
frechet_inception_distance: 72.0289
kernel_inception_distance_mean: 0.004457215
kernel_inception_distance_std: 0.0006336758

==> logs/scarf_images_4.txt <==
inception_score_mean: 5.995791
inception_score_std: 0.3400172
frechet_inception_distance: 159.3824
kernel_inception_distance_mean: 0.02898115
kernel_inception_distance_std: 0.0008947263

==> logs/scarf_images_8.txt <==
inception_score_mean: 5.73159
inception_score_std: 0.3925282
frechet_inception_distance: 74.87528
kernel_inception_distance_mean: 0.004647122
kernel_inception_distance_std: 0.0005858962

(clip) ajay@pabrtxs3:/shared/ajay/clip/nerf/nerf-pytorch$ cat logs/nv_8_400.txt
inception_score_mean: 5.125942
inception_score_std: 0.18581
frechet_inception_distance: 239.4725
kernel_inception_distance_mean: 0.1165526
kernel_inception_distance_std: 0.006308419