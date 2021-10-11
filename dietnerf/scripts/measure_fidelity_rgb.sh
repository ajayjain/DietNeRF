fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_100/ data/nerf_synthetic_400_rgb
inception_score_mean: 6.067405
inception_score_std: 0.6212252
frechet_inception_distance: 110.4658
kernel_inception_distance_mean: 0.02722284
kernel_inception_distance_std: 0.001525454

fidelity --gpu 3 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_simple_images_8/ data/nerf_synthetic_400_rgb
inception_score_mean: 4.560575
inception_score_std: 0.7868282
frechet_inception_distance: 213.7481
kernel_inception_distance_mean: 0.06113499
kernel_inception_distance_std: 0.001804599

fidelity --gpu 1 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarf_images_8/ data/nerf_synthetic_400_rgb
inception_score_mean: 5.73159
inception_score_std: 0.3925282
frechet_inception_distance: 124.5834
kernel_inception_distance_mean: 0.03087021
kernel_inception_distance_std: 0.001448755

fidelity --gpu 2 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarfft_images_8/ data/nerf_synthetic_400_rgb
inception_score_mean: 5.524686
inception_score_std: 0.4460831
frechet_inception_distance: 122.4893
kernel_inception_distance_mean: 0.03054625
kernel_inception_distance_std: 0.001519936

fidelity --gpu 2 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_8/ data/nerf_synthetic_400_rgb
inception_score_mean: 4.347547
inception_score_std: 0.9040484
frechet_inception_distance: 246.9516
kernel_inception_distance_mean: 0.09010862
kernel_inception_distance_std: 0.001587372

fidelity --gpu 5 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_4/ data/nerf_synthetic_400_rgb
inception_score_mean: 3.10864
inception_score_std: 0.4524474
frechet_inception_distance: 321.8411
kernel_inception_distance_mean: 0.1712483
kernel_inception_distance_std: 0.002500453

fidelity --gpu 6 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/scarf_images_4/ data/nerf_synthetic_400_rgb
inception_score_mean: 5.995791
inception_score_std: 0.3400172
frechet_inception_distance: 180.7898
kernel_inception_distance_mean: 0.0429789
kernel_inception_distance_std: 0.001000432

# With 1618 samples
fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nv data/nerf_synthetic_400_rgb

# With test images
fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nv data/nerf_synthetic_400_rgb