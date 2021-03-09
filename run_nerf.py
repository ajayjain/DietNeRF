import functools
import json
import os
import time

from numpy.lib.arraysetops import isin

import clip_utils
import configargparse
import imageio
import numpy as np
import PIL
from scipy.spatial.transform import Rotation
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as run_checkpoint
import torchvision
from tqdm import tqdm, trange
import wandb

import geometry
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, pose_spherical_uniform
from run_nerf_helpers import *
import gan_networks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, keep_keys=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    # ret_rgb_only = keep_keys and len(keep_keys) == 1 and keep_keys[0] == 'rgb_map'
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if keep_keys and k not in keep_keys:
                # Don't save this returned value to save memory
                continue
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  keep_keys=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, keep_keys=keep_keys, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    if keep_keys:
        k_extract = [k for k in k_extract if k in keep_keys]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    model = nn.DataParallel(model).to(device)
    wandb.watch(model)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        model_fine = nn.DataParallel(model_fine).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################


    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                verbose=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:
        pts0, raw0, rgb_map_0, disp_map_0, acc_map_0, weights0 = pts, raw, rgb_map, disp_map, acc_map, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map}
    ret['disp_map'] = disp_map
    ret['acc_map'] = acc_map
    ret['weights'] = weights
    ret['pts'] = pts
    if retraw:
        ret['raw'] = raw
        ret['raw0'] = raw0
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['pts0'] = pts0
        ret['weights0'] = weights0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def sample_rays(H, W, rays_o, rays_d, N_rand, i, start, precrop_iters, precrop_frac):
    if i < precrop_iters:
        dH = int(H//2 * precrop_frac)
        dW = int(W//2 * precrop_frac)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            ), -1)
        if i == start:
            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {precrop_iters}")                
    else:
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
    return batch_rays, select_coords


def get_embed_fn(model_type, num_layers=-1, spatial=False, checkpoint=False):
    if model_type.startswith('clip_'):
        if model_type == 'clip_rn50':
            clip_utils.load_rn(jit=False)
            if spatial:
                _clip_dtype = clip_utils.clip_model_rn.clip_model.dtype
                assert num_layers == -1
                def embed(ims):
                    ims = clip_utils.CLIP_NORMALIZE(ims).type(_clip_dtype)
                    return clip_utils.clip_model_rn.clip_model.visual.featurize(ims)  # [N,C,56,56]
            else:
                embed = lambda ims: clip_utils.clip_model_rn(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers).unsqueeze(1)
            assert not clip_utils.clip_model_rn.training
        elif model_type == 'clip_vit':
            clip_utils.load_vit()
            if spatial:
                def embed(ims):
                    emb = clip_utils.clip_model_vit(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers)  # [N,L=50,D]
                    return emb[:, 1:].view(emb.shape[0], 7, 7, emb.shape[2]).permute(0, 3, 1, 2)  # [N,D,7,7]
            else:
                embed = lambda ims: clip_utils.clip_model_vit(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers)  # [N,L=50,D]
            assert not clip_utils.clip_model_vit.training
    elif model_type.startswith('timm_'):
        assert num_layers == -1
        assert not spatial

        model_type = model_type[len('timm_'):]
        encoder = timm.create_model(model_type, pretrained=True, num_classes=0)
        encoder.eval()
        normalize = torchvision.transforms.Normalize(
            encoder.default_cfg['mean'], encoder.default_cfg['std'])  # normalize an image that is already scaled to [0, 1]
        encoder = nn.DataParallel(encoder).to(device)
        embed = lambda ims: encoder(normalize(ims)).unsqueeze(1)
    elif model_type.startswith('torch_'):
        assert num_layers == -1
        assert not spatial

        model_type = model_type[len('torch_'):]
        encoder = torch.hub.load('pytorch/vision:v0.6.0', model_type, pretrained=True)
        encoder.eval()
        encoder = nn.DataParallel(encoder).to(device)
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize an image that is already scaled to [0, 1]
        embed = lambda ims: encoder(normalize(ims)).unsqueeze(1)
    else:
        raise ValueError

    if checkpoint:
        return lambda x: run_checkpoint(embed, x)

    return embed


def compute_aligned_loss(args, features1, features2, acc_map=None):
    assert features1.ndim == 4
    assert features1.shape == features2.shape

    if args.aligned_loss_comparison == 'mse':
        mse = F.mse_loss(features1, features2, reduction='none')
        if args.mask_aligned_loss:
            assert acc_map.shape == features1.shape[2:]  # [H,W]
            mse = mse * acc_map.unsqueeze(0).unsqueeze(1)
        return mse.mean()

    if args.aligned_loss_comparison == 'cosine_sim':
        sim = F.cosine_similarity(features1, features2, dim=1)
        if args.mask_aligned_loss:
            assert acc_map.shape == features1.shape[2:]  # [H,W]
            sim = sim * acc_map.unsqueeze(0)
        return -sim.mean()


def c2w_to_w2c(c2w):
    assert c2w.shape == (3, 4)
    c2w_rotation = c2w[:, :3]
    camera_origin = c2w[:, 3].unsqueeze(1)
    return torch.cat([c2w_rotation.T, -c2w_rotation.T.mm(camera_origin)], dim=1)


def world_to_image(rendered_pts, w2c, H, W, focal, principal_point):
    """Convert from ray pts in world coordinates to image coordinates

    Args:
        rendered_pts (tensor): [cnH,cnW,sampled_pts,3] xyz world coordinates along rays
        w2c (tensor): [3,4] transformation matrix to camera pose
    Returns:
        uv (tensor): [cnH,cnW,sampled_pts,2] uv pixel coordinates between 0 and H or W
        norm_uv (tensor): [cnH,cnW,sampled_pts,2] normalized uv pixel coordinates between -1 and 1
    """
    # Transform rendered rays
    rendered_pts_extended = torch.cat([rendered_pts, torch.ones_like(rendered_pts[..., 0:1])], dim=-1)
    rendered_pts_camera = torch.einsum('cw,absw->absc', w2c, rendered_pts_extended)

    # Convert to image coordinates
    uv = -rendered_pts_camera[..., :2] / rendered_pts_camera[..., 2:]  # [H,W,B,2]
    uv = uv * focal + principal_point
    uv[..., 1] = H - uv[..., 1]
    norm_uv = uv / torch.tensor([H,W], device=uv.device) * 2 - 1  # between [-1, 1] for F.grid_sample

    return uv, norm_uv


def resample_features(features, norm_uv, weights, feature_interp_mode='bilinear'):
    """
    Args:
        features (tensor): [D, H, W] or [1, D, H, W]
        norm_uv (tensor): [cnH, cnW, samples_per_ray, 2]
        weights (tensor): [cnH, cnW, samples_per_ray]
        feature_interp_mode (str): bilinear or nearest
    Returns:
        features (tensor): [1, D, cnH, cnW]
    """
    # Resample features along rays
    features = features.expand(norm_uv.shape[2], -1, -1, -1)  # [samples_per_ray, D, H, W]
    features = F.grid_sample(
        features,
        norm_uv.permute(2, 0, 1, 3).type(features.dtype),  # [samples_per_ray, consistency_nH, consistency_nW, 2]
        align_corners=True,
        padding_mode='border',
        mode=feature_interp_mode,
    )  # [samples_per_ray, D, consistency_nH, consistency_nW], D is 256 for CLIP RN50 featurize

    # Weighted average of target features along each ray
    weights = weights.permute(2, 0, 1).unsqueeze(1)
    features = weights * features
    features = features.sum(dim=0, keepdim=True)  # [1, D, cnH, cnW]

    return features


def get_patch_similarity_matrix(embs1, embs2):
    assert embs1.ndim == 2  # [L1,D]
    assert embs2.ndim == 3  # [L2,D]
    embs1 = F.normalize(embs1, p=2, dim=-1)  # [L1,D]
    embs2 = F.normalize(embs2, p=2, dim=-1)  # [B,L2,D]
    sim = torch.matmul(embs2, embs1.transpose(0,1))  # [B,L2,L1]
    return sim.view(-1, embs1.shape[0]).transpose(0,1)  # [L1,B*L2]


@torch.no_grad()
def make_wandb_image(tensor, preprocess='scale'):
    tensor = tensor.detach()
    tensor = tensor.float()
    if preprocess == 'scale':
        mi = tensor.min()
        tensor = ((tensor - mi) / (tensor.max() - mi))
    elif preprocess == 'clip':
        tensor = tensor.clip(0, 1)
    return wandb.Image(tensor.cpu().numpy())


@torch.no_grad()
def make_wandb_histogram(tensor):
    return wandb.Histogram(tensor.detach().flatten().cpu().numpy())


@torch.no_grad()
def log_discriminator_histograms(metrics, key, preds):
    if isinstance(preds, list):
        for disc_id, pred in enumerate(preds):
            metrics[f'{key}_D{disc_id}'] = make_wandb_histogram(pred)
    else:
        metrics[f'{key}_D0'] = make_wandb_histogram(preds)


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=1024*64*4, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--seed", type=int, default=0)

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--wandb_entity",   type=str, default='ajayj')
    parser.add_argument("--i_log",   type=int, default=1, 
                        help='frequency of metric logging')
    parser.add_argument("--i_log_raw_hist",   type=int, default=2, 
                        help='frequency of logging histogram of raw network outputs')
    parser.add_argument("--i_log_rendered_img", "--i_log_ctr_img",
                        type=int, default=100, 
                        help='frequency of train rendering logging')
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--save_splits", action="store_true",
                        help='save ground truth images and poses in each split')

    ### options for learning with few views
    parser.add_argument("--max_train_views", type=int, default=-1,
                        help='limit number of training views for the mse loss')
    # Options for rendering shared between different losses
    parser.add_argument("--render_loss_interval", "--consistency_loss_interval",
        type=float, default=1)
    parser.add_argument("--render_autocast", action='store_true')
    parser.add_argument("--render_poses", "--consistency_poses",
        type=str, choices=['loaded', 'interpolate_train_all', 'uniform'], default='loaded')
    parser.add_argument("--render_poses_translation_jitter_sigma", "--consistency_poses_translation_jitter_sigma",
        type=float, default=0.)
    parser.add_argument("--render_poses_interpolate_range", "--consistency_poses_interpolate_range",
        type=float, nargs=2, default=[0., 1.])
    # Options for --render_poses=uniform
    parser.add_argument("--render_theta_range", "--consistency_theta_range", type=float, nargs=2)
    parser.add_argument("--render_phi_range", "--consistency_phi_range", type=float, nargs=2)
    parser.add_argument("--render_radius_range", "--consistency_radius_range", type=float, nargs=2)
    parser.add_argument("--render_nH", "--consistency_nH", type=int, default=32, 
                        help='number of rows to render for consistency loss. smaller values use less memory')
    parser.add_argument("--render_nW", "--consistency_nW", type=int, default=32, 
                        help='number of columns to render for consistency loss')
    parser.add_argument("--render_jitter_rays", "--consistency_jitter_rays", action='store_true')
    # Computational options shared between rendering losses
    parser.add_argument("--checkpoint_rendering", action='store_true')
    parser.add_argument("--checkpoint_embedding", action='store_true')
    parser.add_argument("--no_mse", action='store_true')
    parser.add_argument("--pixel_interp_mode", type=str, default='bicubic')
    parser.add_argument("--feature_interp_mode", type=str, default='bilinear')

    # Global semantic consistency loss
    parser.add_argument("--consistency_loss", type=str, default='none', choices=['none', 'consistent_with_target_rep'])
    parser.add_argument("--consistency_loss_comparison", type=str, default=['cosine_sim'], nargs='+', choices=[
        'cosine_sim', 'mse', 'max_patch_match_sametarget', 'max_patch_match_alltargets'])
    parser.add_argument("--consistency_loss_lam", type=float, default=0.2,
                        help="weight for the fine network's semantic consistency loss")
    parser.add_argument("--consistency_loss_lam0", type=float, default=0.2,
                        help="weight for the coarse network's semantic consistency loss")
    parser.add_argument("--consistency_target_augmentation", type=str, default='none', choices=[
        'none', 'flips', 'tencrop'])
    parser.add_argument("--consistency_size", type=int, default=224)
    # Consistency model arguments
    parser.add_argument("--consistency_model_type", type=str, default='clip_vit') # choices=['clip_vit', 'clip_rn50']
    parser.add_argument("--consistency_model_num_layers", type=int, default=-1)

    # Aligned feature consistency arguments
    parser.add_argument("--aligned_loss", action='store_true')
    parser.add_argument("--mask_aligned_loss", action='store_true')
    parser.add_argument("--spatial_model_type", type=str, default='clip_rn50', choices=['clip_vit', 'clip_rn50'])
    parser.add_argument("--spatial_model_num_layers", type=int, default=-1)
    parser.add_argument("--aligned_loss_comparison", type=str, default='cosine_sim', choices=['cosine_sim', 'mse'])
    parser.add_argument("--aligned_loss_lam", type=float, default=0.1,
                        help="weight for the fine network's aligned semantic consistency loss")
    parser.add_argument("--aligned_loss_lam0", type=float, default=0.1,
                        help="weight for the coarse network's aligned semantic consistency loss")

    # Discriminator losses
    parser.add_argument("--patch_gan_loss", action='store_true')
    parser.add_argument("--patch_gan_D_inner_steps", type=int, default=1)
    parser.add_argument("--patch_gan_D_grad_clip", type=float, default=-1)
    parser.add_argument("--patch_gan_D_nH", type=int, default=168)
    parser.add_argument("--patch_gan_D_nW", type=int, default=168)
    parser.add_argument("--patch_wgan_lambda_gp", type=float, default=0.1)  # SinGAN uses 0.1, but the WGAN-GP paper uses 10
    parser.add_argument("--patch_gan_D_activation", type=str, default='none', choices=['none', 'sigmoid'])
    # Train options from BiCycleGAN
    parser.add_argument('--patch_gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla | lsgan ｜ wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--patch_gan_G_lam', type=float, default=1.0, help='weight on D loss backpropped to NeRF. D(G(random_pose))')
    parser.add_argument('--patch_gan_lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--patch_gan_beta1', type=float, default=0.5, help='momentum term of adam')
    # parser.add_argument('--patch_gan_lr_policy', type=str, default='linear', help='learning rate policy: linear | step | plateau | cosine')
    # parser.add_argument('--patch_gan_lr_decay_iters', type=int, default=100, help='multiply by a gamma every lr_decay_iters iterations')
    # Base options from BiCycleGAN
    parser.add_argument('--patch_gan_num_Ds', type=int, default=1, help='number of Discrminators')
    parser.add_argument('--patch_gan_netD', type=str, default='basic_256_multi', help='selects model to use for netD')
    parser.add_argument('--patch_gan_ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--patch_gan_norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--patch_gan_init_type', type=str, default='xavier', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--patch_gan_init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--patch_gan_nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    wandb.init(project="nerf-nl", entity=args.wandb_entity)
    wandb.run.name = args.expname
    wandb.run.save()
    wandb.config.update(args)

    # Re-seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    print('dataset_type:', args.dataset_type)
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Subsample training indices to simulate having fewer training views
    i_train_poses = i_train  # Use all training poses for auxiliary representation consistency loss.
                             # TODO: Could also use any continuous set of poses including the
                             # val and test poses, since we don't use the images for aux loss.
    if args.max_train_views > 0:
        print('Original training views:', i_train)
        i_train = np.random.choice(i_train, size=args.max_train_views, replace=False)
        print('Subsampled train views:', i_train)

    # Load embedding network for rendering losses
    if args.consistency_loss != 'none':
        print(f'Using auxilliary consistency loss [{args.consistency_loss}], fine weight [{args.consistency_loss_lam}], coarse weight [{args.consistency_loss_lam0}]')
        embed = get_embed_fn(args.consistency_model_type, args.consistency_model_num_layers, checkpoint=args.checkpoint_embedding)
    if args.aligned_loss:
        embed_spatial = get_embed_fn(args.spatial_model_type, args.spatial_model_num_layers, spatial=True, checkpoint=args.checkpoint_embedding)

    # Cast intrinsics to right types
    H, W, focal = hwf
    print('hwf', hwf)
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    network_fn = render_kwargs_train['network_fn']
    network_fine = render_kwargs_train['network_fine']
    if args.checkpoint_rendering:
        # Pass a dummy input tensor that requires grad so checkpointing does something
        # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/10
        dummy = torch.ones(1, dtype=torch.float32, requires_grad=True, device=device)
        network_fn_wrapper = lambda x, y: network_fn(x)
        network_fine_wrapper = lambda x, y: network_fine(x)
        render_kwargs_train['network_fn'] = lambda x: run_checkpoint(network_fn_wrapper, x, dummy)
        render_kwargs_train['network_fine'] = lambda x: run_checkpoint(network_fine_wrapper, x, dummy)


    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            if args.render_test:
                # Compute metrics
                mse, psnr, ssim, lpips = get_perceptual_metrics(rgbs, images, device=device)

                metricspath = os.path.join(testsavedir, 'test_metrics.json')
                with open(metricspath, 'w') as test_metrics_f:
                    test_metrics = {
                        'mse': mse,
                        'psnr': psnr,
                        'ssim': ssim,
                        'lpips': lpips,
                    }
                    print(args.expname, f'test metrics ({metricspath}):', test_metrics)
                    test_metrics['args'] = vars(args)
                    json.dump(test_metrics, test_metrics_f)
                    wandb.save(metricspath)

            return

    # Save ground truth splits for visualization
    if args.save_splits:
        for idx, name in [(i_train, 'train'), (i_val, 'val'), (i_test, 'test')]:
            savedir = os.path.join(basedir, expname, '{}set'.format(name))
            os.makedirs(savedir, exist_ok=True)

            torch.save(poses[idx], os.path.join(savedir, 'poses.pth'))
            torch.save(idx, os.path.join(savedir, 'indices.pth'))
            for i in idx:
                rgb8 = to8b(images[i])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

            print(name, 'poses shape', poses[idx].shape, 'images shape', images[idx].shape)
            print(f'Saved ground truth {name} set')

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    calc_ctr_loss = args.consistency_loss.startswith('consistent_with_target_rep')
    any_rendered_loss = (calc_ctr_loss or args.aligned_loss or args.patch_gan_loss)
    if any_rendered_loss:
        with torch.no_grad():
            targets = images[i_train].permute(0, 3, 1, 2)

    # Embed training images for consistency loss
    if calc_ctr_loss:
        with torch.no_grad():
            i_train_map = {it: i for i, it in enumerate(i_train)}  # map from indices into all images to indices in selected training images
            if args.consistency_target_augmentation != 'none':
                raise NotImplementedError
            targets_resize_model = F.interpolate(targets, (args.consistency_size, args.consistency_size), mode=args.pixel_interp_mode)
            target_embeddings = embed(targets_resize_model)  # [N,L,D]

    # Embed training images for aligned consistency loss
    consistency_keep_keys = ['rgb_map', 'rgb0']
    if args.aligned_loss:
        assert calc_ctr_loss
        assert args.dataset_type != 'llff' or args.no_ndc  # Haven't written working transform code for NDC
        assert not args.render_jitter_rays  # Translating the rays leads to a mismatched alignment

        with torch.no_grad():
            # Embed targets
            target_features = embed_spatial(targets_resize_model)  # [N,C,fH,fW]

            # Resize target features at the full resolution (H,W)
            targets_features_resize_full = F.interpolate(target_features, (H, W), mode=args.feature_interp_mode)

            principal_point = torch.tensor([[H/2., W/2.]], device=device)

        consistency_keep_keys = ['rgb_map', 'rgb0', 'pts', 'pts0', 'weights', 'weights0', 'acc_map', 'acc0']
 
    if args.patch_gan_loss:
        netD = gan_networks.define_D(3,
            args.patch_gan_ndf,
            netD=args.patch_gan_netD,
            norm=args.patch_gan_norm,
            nl=args.patch_gan_nl,
            init_type=args.patch_gan_init_type,
            init_gain=args.patch_gan_init_gain,
            num_Ds=args.patch_gan_num_Ds,
            activation=args.patch_gan_D_activation).to(device)
        print('Discriminator:', netD)
        criterionGAN = gan_networks.GANLoss(gan_mode=args.patch_gan_mode).to(device)
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.patch_gan_lr, betas=(args.patch_gan_beta1, 0.999))

    start = start + 1
    for i in trange(start, N_iters):
        metrics = {}

        # Sample random ray batch
        if use_batching:
            assert args.consistency_loss == 'none'

            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            assert N_rand is not None

            # Random rays from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3,:4]

            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            batch_rays, select_coords = sample_rays(H, W, rays_o, rays_d, N_rand=N_rand,
                i=i, start=start, precrop_iters=args.precrop_iters, precrop_frac=args.precrop_frac)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            # Representational consistency loss with rendered image
            render_loss_iter = i % args.render_loss_interval == 0
            if (calc_ctr_loss or args.aligned_loss or args.patch_gan_loss) and render_loss_iter:
                with torch.no_grad():
                    # Render from a random viewpoint
                    if args.render_poses == 'loaded':
                        poses_i = np.random.choice(i_train_poses)
                        pose = poses[poses_i, :3, :4]
                    elif args.render_poses == 'interpolate_train_all':
                        assert len(i_train_poses) >= 3
                        poses_i = np.random.choice(i_train_poses, size=3, replace=False)
                        pose1, pose2, pose3 = poses[poses_i, :3, :4].cpu()
                        s12, s3 = np.random.uniform(*args.render_poses_interpolate_range, size=2)
                        pose = geometry.interp3(pose1, pose2, pose3, s12, s3)
                    elif args.render_poses == 'uniform':
                        assert args.dataset_type == 'blender'
                        pose = pose_spherical_uniform(args.render_theta_range, args.render_phi_range, args.render_radius_range)
                        pose = pose[:3, :4]

                    print('Sampled pose:', Rotation.from_matrix(pose[:, :3].cpu()).as_rotvec(), 'origin:', pose[:, 3])

                    if args.render_poses_translation_jitter_sigma > 0:
                        pose[:, -1] = pose[:, -1] + torch.randn(3, device=pose.device) * args.render_poses_translation_jitter_sigma

                    # TODO: something strange with pts_W in get_rays when 224 nH
                    rays = get_rays(H, W, focal, c2w=pose, nH=args.render_nH, nW=args.render_nW,
                                    jitter=args.render_jitter_rays)

                with torch.cuda.amp.autocast(enabled=args.render_autocast):
                    extras = render(H, W, focal, chunk=args.chunk, rays=rays,
                                    keep_keys=consistency_keep_keys,
                                    **render_kwargs_train)[-1]
                    # rgb0 is the rendering from the coarse network, while rgb_map uses the fine network
                    rgbs = torch.stack([extras['rgb_map'], extras['rgb0']], dim=0)
                    rgbs = rgbs.permute(0, 3, 1, 2).clamp(0, 1)

                if i == 0:
                    print('rendering losses rendered rgb image shape:', rgbs.shape)

                if args.aligned_loss:
                    # TODO: support sampling nearby targets
                    align_target_i = img_i
                    align_target_pose = poses[align_target_i, :3,:4]

                    # Create world to target camera trasformation
                    w2c = c2w_to_w2c(align_target_pose)

                    # Transform rendered rays
                    _, norm_uv = run_checkpoint(world_to_image, extras['pts'], w2c, H, W, focal, principal_point)
                    _, norm_uv0 = run_checkpoint(world_to_image, extras['pts0'], w2c, H, W, focal, principal_point)

                    # Embed rendered images into spatial features
                    rgbs_resize_c = F.interpolate(rgbs, size=(args.consistency_size, args.consistency_size), mode=args.pixel_interp_mode)
                    rendered_features = embed_spatial(rgbs_resize_c)  # [2, D, fH, fW] = [2, 256, 56, 56] for CLIP RN50
                    
                    _resample = functools.partial(resample_features, feature_interp_mode=args.feature_interp_mode)
                    target_feature_samples = run_checkpoint(_resample,
                        targets_features_resize_full[i_train_map[align_target_i]],
                        norm_uv,
                        extras['weights'],
                    )

                    target_feature_samples0 = run_checkpoint(_resample,
                        targets_features_resize_full[i_train_map[align_target_i]],
                        norm_uv0,
                        extras['weights0'],
                    )

                    # Save acc map for loss calculation
                    rendered_acc_map = extras['acc_map']
                    rendered_acc0 = extras['acc0']

                if i%args.i_log_rendered_img==0:
                    metrics['train_ctr/target'] = make_wandb_image(target, 'clip')
                    metrics['train_ctr/rgb'] = make_wandb_image(extras['rgb_map'], 'clip')
                    metrics['train_ctr/rgb0'] = make_wandb_image(extras['rgb0'], 'clip')

                    if args.aligned_loss:
                        metrics['train_aligned/target_feature'] = make_wandb_image(targets_features_resize_full[i_train_map[align_target_i]][...,None].mean(0))
                        metrics['train_aligned/target_feature_samples'] = make_wandb_image(target_feature_samples[0,...,None].mean(0))
                        metrics['train_aligned/rendered_features'] = make_wandb_image(rendered_features[0,...,None].mean(0))
                        metrics['train_aligned/rendered_features0'] = make_wandb_image(rendered_features[1,...,None].mean(0))
                        metrics['train_aligned/acc_map'] = make_wandb_image(rendered_acc_map.unsqueeze(-1))
                        metrics['train_aligned/acc0'] = make_wandb_image(rendered_acc0.unsqueeze(-1))

        #####  Core optimization loop  #####

        optimizer.zero_grad()
        loss = 0

        if calc_ctr_loss and render_loss_iter:
            assert args.consistency_loss == 'consistent_with_target_rep'

            # Resize and embed rendered images
            rgbs_resize_c = F.interpolate(rgbs, size=(args.consistency_size, args.consistency_size), mode=args.pixel_interp_mode)
            rendered_embedding, rendered_embedding0 = embed(rgbs_resize_c)
            assert rendered_embedding.ndim == 2  # [L,D]
            assert target_embeddings.ndim == 3  # [N,L,D]

            # NOTE: Randomly sampling a target (run 031) is worse than sampling the same target as MSE (run 032)
            target_i = i_train_map[img_i]
            target_emb = target_embeddings[target_i, 0]
            rendered_emb = rendered_embedding[0]
            rendered_emb0 = rendered_embedding0[0]

            consistency_loss, consistency_loss0 = [], []
            if 'cosine_sim' in args.consistency_loss_comparison:
                consistency_loss.append(-torch.cosine_similarity(target_emb, rendered_emb, dim=-1))
                consistency_loss0.append(-torch.cosine_similarity(target_emb, rendered_emb0, dim=-1))

            if 'mse' in args.consistency_loss_comparison:
                consistency_loss.append(F.mse_loss(rendered_emb, target_emb))
                consistency_loss0.append(F.mse_loss(rendered_emb0, target_emb))

            if 'max_patch_match_alltargets' in args.consistency_loss_comparison:
                patch_sim = get_patch_similarity_matrix(rendered_embedding[1:], target_embeddings[:, 1:])  # [L-1,num_targ*(L-1)]
                max_patch_sim, _ = torch.max(patch_sim, dim=1)  # max across target patches
                consistency_loss.append(-torch.mean(max_patch_sim))  # mean across rendered patches

                patch_sim0 = get_patch_similarity_matrix(rendered_embedding0[1:], target_embeddings[:, 1:])
                max_patch_sim0, _ = torch.max(patch_sim0, dim=1)
                consistency_loss0.append(-torch.mean(max_patch_sim0))

            if 'max_patch_match_sametarget' in args.consistency_loss_comparison:
                patch_sim = get_patch_similarity_matrix(
                    rendered_embedding[1:], target_embeddings[target_i, 1:].unsqueeze(0))  # [L-1,L-1]
                max_patch_sim, _ = torch.max(patch_sim, dim=1)  # max across target patches
                consistency_loss.append(-torch.mean(max_patch_sim))  # mean across rendered patches

                patch_sim0 = get_patch_similarity_matrix(
                    rendered_embedding0[1:], target_embeddings[target_i, 1:].unsqueeze(0))  # [L-1,L-1]
                max_patch_sim0, _ = torch.max(patch_sim0, dim=1)
                consistency_loss0.append(-torch.mean(max_patch_sim0))

            consistency_loss = torch.mean(torch.stack(consistency_loss))
            consistency_loss0 = torch.mean(torch.stack(consistency_loss0))

            loss = (loss + consistency_loss * args.consistency_loss_lam +
                    consistency_loss0 * args.consistency_loss_lam0)

        if args.aligned_loss and render_loss_iter:
            rendered_features_resize_c = F.interpolate(rendered_features, (args.render_nH, args.render_nW), mode=args.feature_interp_mode)
            rendered_features_resize_c = rendered_features_resize_c.float()

            aligned_loss = compute_aligned_loss(args,
                rendered_features_resize_c[0:1],
                target_feature_samples,  # [1, D, cnH, cnW]
                rendered_acc_map)

            aligned_loss0 = compute_aligned_loss(args,
                rendered_features_resize_c[1:2],
                target_feature_samples0,
                rendered_acc0)

            loss = (loss + aligned_loss * args.aligned_loss_lam +
                    aligned_loss0 * args.aligned_loss_lam0)

        if args.patch_gan_loss and render_loss_iter:
            # Compute realism loss for NeRF training
            gan_networks.set_requires_grad([netD], False)

            # Flip fine network's rendering
            rgbs_G = torch.cat([rgbs[:1], rgbs[:1].flip(3)], dim=0)
            if rgbs_G.shape[2:] != (args.patch_gan_D_nH, args.patch_gan_D_nW):
                rgbs_G = F.interpolate(rgbs_G, (args.patch_gan_D_nH, args.patch_gan_D_nW), mode=args.pixel_interp_mode)
            pred_fake = netD(rgbs_G * 2 - 1)
            if args.patch_gan_num_Ds > 1:
                print('patch gan gen loss pred shapes', 'rgbs_G', rgbs_G.shape, 'pred_fake', len(pred_fake), list(map(lambda x: x.shape, pred_fake)))

            loss_G, _ = criterionGAN(pred_fake, True)
            loss = loss + loss_G * args.patch_gan_G_lam

            metrics['train_patch_gan/G/rgbs_G'] = make_wandb_image(rgbs_G[0].permute(1,2,0), 'clip')  # fine network
            metrics['train_patch_gan/G/loss_G'] = loss_G.item()
            log_discriminator_histograms(metrics, 'train_patch_gan/G/pred_fake', pred_fake)

        if not args.no_mse:
            # Standard NeRF MSE loss with subsampled rays
            rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True,
                                            **render_kwargs_train)
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = loss + img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                if i == start:
                    print('Using auxilliary rgb0 mse loss')
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        if args.patch_gan_loss and render_loss_iter:
            # Compute loss for discriminator training
            gan_networks.set_requires_grad([netD], True)
            for _ in range(args.patch_gan_D_inner_steps):
                optimizer_D.zero_grad()
                ## Flip fine net rendering and targets
                with torch.no_grad():
                    rgbs_D = rgbs.detach()[:1]
                    rgbs_D = torch.cat([rgbs_D, rgbs_D.flip(3)], dim=0)
                    if rgbs_D.shape[2:] != (args.patch_gan_D_nH, args.patch_gan_D_nW):
                        rgbs_D = F.interpolate(rgbs_D, (args.patch_gan_D_nH, args.patch_gan_D_nW), mode=args.pixel_interp_mode)

                    targets_D = targets[np.random.randint(len(targets))].unsqueeze(0)
                    targets_D = torch.cat([targets_D, targets_D.flip(3)], dim=0)
                    if targets_D.shape[2:] != (args.patch_gan_D_nH, args.patch_gan_D_nW):
                        targets_D = F.interpolate(targets.detach(), (args.patch_gan_D_nH, args.patch_gan_D_nW), mode=args.pixel_interp_mode)

                pred_fake = netD(rgbs_D.detach())
                pred_real = netD(targets_D.detach())

                loss_D_fake, _ = criterionGAN(pred_fake, False)
                loss_D_real, _ = criterionGAN(pred_real, True)
                loss_D = loss_D_fake + loss_D_real

                if args.patch_gan_mode == 'wgangp':
                    optimizer_D.zero_grad()
                    penalty_target_idx = np.random.choice(targets_D.shape[0], size=rgbs_D.shape[0], replace=False)
                    gradient_penalty = gan_networks.cal_gradient_penalty(netD,
                        fake_data=rgbs_D.detach(), real_data=targets_D[penalty_target_idx].detach(),
                        lambda_gp=args.patch_wgan_lambda_gp, device=device)
                    loss_D = loss_D + gradient_penalty
                    metrics['train_patch_gan/D/gradient_penality'] = gradient_penalty.item()

                if args.patch_gan_D_grad_clip > 0:
                    nn.utils.clip_grad.clip_grad_norm_(netD.parameters(), max_norm=args.patch_gan_D_grad_clip)
                loss_D.backward()
                optimizer_D.step()

            metrics['train_patch_gan/D/rgbs_D'] = make_wandb_image(rgbs_D[0].permute(1,2,0), 'clip')  # fine network
            metrics['train_patch_gan/D/targets_D'] = make_wandb_image(targets_D[np.random.randint(len(targets_D))].permute(1,2,0), 'clip')
            metrics['train_patch_gan/D/loss_D'] = loss_D
            metrics['train_patch_gan/D/loss_D_fake'] = loss_D_fake
            metrics['train_patch_gan/D/loss_D_real'] = loss_D_real
            log_discriminator_histograms(metrics, 'train_patch_gan/D/pred_fake', pred_fake)
            log_discriminator_histograms(metrics, 'train_patch_gan/D/pred_real', pred_real)

        # # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        if args.patch_gan_loss:
            new_lrate_D = args.patch_gan_lr * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = new_lrate_D
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': network_fn.state_dict(),
                'network_fine_state_dict': network_fine.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
            wandb.save(path)
            print('Uploading checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            metrics["render_path/rgb_video"] = wandb.Video(moviebase + 'rgb.mp4')
            metrics["render_path/disp_video"] = wandb.Video(moviebase + 'disp.mp4')

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        # Log scalars, images and histograms to wandb
        if i%args.i_log==0:
            metrics.update({
                "train/loss": loss.item(),
                "train/psnr": psnr.item(),
                "train/mse": img_loss.item(),
                "train/mse0": img_loss0.item(),
                "gradients/norm": gradient_norm(network_fine.parameters()),
                "gradients/norm0": gradient_norm(network_fn.parameters()),
                "train/lrate": new_lrate,
            })
            if args.N_importance > 0:
                metrics["train/psnr0"] = psnr0.item()
            if render_loss_iter:
                if calc_ctr_loss:
                    metrics["train_ctr/consistency_loss"] = consistency_loss.item()
                    metrics["train_ctr/consistency_loss0"] = consistency_loss0.item()
                if args.aligned_loss:
                    metrics["train_aligned/aligned_loss"] = aligned_loss.item()
                    metrics["train_aligned/aligned_loss0"] = aligned_loss0.item()
                if args.patch_gan_loss:
                    metrics["gradients/normD"] = gradient_norm(netD.parameters())

        if i%args.i_log_raw_hist==0:
            metrics["train/tran"] = wandb.Histogram(trans.detach().cpu().numpy())

        if i%args.i_img==0:
            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            with torch.no_grad():
                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                    **render_kwargs_test)

            psnr = mse2psnr(img2mse(rgb, target))

            metrics = {
                'val/rgb': wandb.Image(to8b(rgb.cpu().numpy())[np.newaxis]),
                'val/disp': wandb.Image(disp.cpu().numpy()[np.newaxis,...,np.newaxis]),
                'val/disp_scaled': make_wandb_image(disp[np.newaxis,...,np.newaxis]),
                'val/acc': wandb.Image(acc.cpu().numpy()[np.newaxis,...,np.newaxis]),
                'val/acc_scaled': make_wandb_image(acc[np.newaxis,...,np.newaxis]),
                'val/psnr_holdout': psnr.item(),
                'val/rgb_holdout': wandb.Image(target.cpu().numpy()[np.newaxis])
            }
            if args.N_importance > 0:
                metrics['rgb0'] = wandb.Image(to8b(extras['rgb0'].cpu().numpy())[np.newaxis])
                metrics['disp0'] = wandb.Image(extras['disp0'].cpu().numpy()[np.newaxis,...,np.newaxis])
                metrics['z_std'] = wandb.Image(extras['z_std'].cpu().numpy()[np.newaxis,...,np.newaxis])

        if metrics:
            wandb.log(metrics, step=i)

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
