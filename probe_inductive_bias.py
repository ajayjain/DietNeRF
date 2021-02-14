import fire
import os

import numpy as np
import torch
import torchvision
import tqdm

import clip_utils
import crw_utils

from load_blender import load_blender_data


def probe(
    dataset_type,
    datadir,
    half_res,
    output_path,
    testskip=8,
    device='cuda',
    batch_size=16,
    model_type='clip_rn50',
):
    if dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(datadir, half_res, testskip)
        print('Loaded blender', images.shape, poses.shape, render_poses.shape, hwf, datadir)
        print('poses[0]', poses[0])
        print('render_poses[0]', render_poses[0])
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        raise NotImplementedError

    # Load embedding model
    if model_type == 'clip_rn50':
        clip_utils.load_rn()
        embed = lambda ims: clip_utils.clip_model_rn(images_or_text=ims)
        assert not clip_utils.clip_model_rn.training
    elif model_type == 'clip_vit':
        clip_utils.load_vit()
        embed = lambda ims: clip_utils.clip_model_vit(images_or_text=ims)
        assert not clip_utils.clip_model_vit.training
    elif model_type == 'crw_rn18':
        crw_utils.load_rn18()
        embed = lambda ims: crw_utils.embed_image(ims, spatial_reduction='mean')
        assert not crw_utils.crw_rn18_model.training

    # Prepare images
    images = torch.from_numpy(images).permute(0, 3, 1, 2)
    print('Loaded images:', images.shape, images.min(), images.max())

    # Embed images
    with torch.no_grad():
        # DEBUG: set some images to junk
        # images[-10:].uniform_()

        embedding = []
        for i in tqdm.trange(0, len(images), batch_size, desc='Embedding images'):
            images_batch = images[i:i+batch_size].to(device)
            images_batch = torch.nn.functional.interpolate(images_batch, size=(224, 224), mode='bicubic')
            images_batch = clip_utils.CLIP_NORMALIZE(images_batch)
            print('images_batch', images_batch.shape)
            embedding_batch = embed(images_batch)
            embedding.append(embedding_batch)
        embedding = torch.cat(embedding, dim=0)
        print('Embedding:', embedding.shape)
        assert embedding.shape[0] == len(images)

        # Write results
        print('Saving embeddings to', output_path)
        torch.save({
            'images': images,
            'poses': poses,
            'render_poses': render_poses,
            'hwf': hwf,
            'i_split': i_split,
            'embedding': embedding.cpu().numpy(),
        }, output_path)


if __name__=='__main__':
    fire.Fire({
        'probe': probe
    })
