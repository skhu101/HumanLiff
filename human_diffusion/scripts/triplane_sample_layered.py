"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import sys, os
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current + "/../")

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import imageio
import torch.nn.functional as F
import trimesh

from improved_diffusion import dist_util2, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from SynBodyView_datasets import SynBodyViewDataset
from TightCapView_datasets import TightCapViewDataset

from NeRF.renderer import Renderer
from NeRF.fields import to_cuda

from torch.utils.data import DataLoader

def main():
    args = create_argparser().parse_args()

    # dist_util2.setup_dist()
    rank, world_size, gpu = dist_util2.setup_dist()
    args.rank, args.world_size, args.gpu = rank, world_size, gpu

    th.cuda.set_device(args.gpu)
    th.backends.cudnn.benchmark = True
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)

    logger.configure(dir=args.log_dir)

    if "ema" in args.model_path:
        suffix = f"ckpt_{args.model_path.split('/')[-1].split('_')[2].split('.')[0]}_ema"
    else:
        suffix = f"ckpt_{args.model_path.split('/')[-1].split('.')[0].split('model')[1]}"

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util2.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util2.dev())
    model.eval()

    logger.log("creating humannerf model...")
    # load human nerf
    if args.data_name == "synbody":
        if args.use_3d_aware:
            human_nerf = Renderer(use_canonical_space=False, num_instances=1, triplane_dim=256, triplane_ch=args.out_channels*3, smpl_type='smplx', test=True)
        else:
            human_nerf = Renderer(use_canonical_space=False, num_instances=1, triplane_dim=256, triplane_ch=args.out_channels, smpl_type='smplx', test=True)
    elif args.data_name == "tightcap":
        if args.use_3d_aware:
            human_nerf = Renderer(use_canonical_space=True, num_instances=1, triplane_dim=256, triplane_ch=args.out_channels*3, test=True)
        else:
            human_nerf = Renderer(use_canonical_space=True, num_instances=1, triplane_dim=256, triplane_ch=args.out_channels, test=True)
    
    ckpt = th.load(args.data_dir, map_location='cpu')
    del ckpt['network_fn_state_dict']['tri_planes']
    human_nerf.load_state_dict(ckpt['network_fn_state_dict'])
    human_nerf.to(dist_util2.dev())

    if args.data_name == "synbody":
        dataset = SynBodyViewDataset(
            triplane_root=args.data_dir, 
            img_root='data/SynBody/20230423_layered/seq_000000', 
            num_instance=1, 
            layer_idx=args.layer_index,
            start_motion_pose_id=args.start_motion_pose_id,
            )
    elif args.data_name == "tightcap":
        dataset = TightCapViewDataset(
            triplane_root=args.data_dir, 
            img_root='data/TightCap/f_c_10412256613', 
            num_instance=1, 
            layer_idx=args.layer_index,
            start_motion_pose_id=args.start_motion_pose_id,
            )


    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, persistent_workers=False
    )

    logger.log("sampling...")
    all_images = []
    all_labels = []

    
    count = 0
    iteration = args.start_id + count
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util2.dev()
            )
            classes = args.layer_index * th.ones_like(classes)
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        if args.use_cond:
            if args.layer_index == 0:
                if args.use_3d_aware:
                    x_cond = th.zeros(args.batch_size, args.out_channels*3, args.image_size, args.image_size).to(dist_util2.dev())
                else:
                    x_cond = th.zeros(args.batch_size, args.out_channels, args.image_size, args.image_size).to(dist_util2.dev())                    
            else:
                sample_arr = np.load(args.sample_npz)
                x_cond = th.from_numpy(sample_arr.f.arr_0.astype(np.float32))[count:count+args.batch_size].to(dist_util2.dev())
        else:
            x_cond = None

        if args.use_3d_aware:
            sample = sample_fn(
                model,
                (args.batch_size, args.out_channels*3, args.image_size, args.image_size),
                x_cond = x_cond,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        else:
            sample = sample_fn(
                model,
                (args.batch_size, args.out_channels, args.image_size, args.image_size),
                x_cond = x_cond,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )        

        count += args.batch_size  

        for id in range(args.batch_size):
            frames = []
            for b, batch in enumerate(loader):
                tri_planes = sample[id:id+1].reshape(1, 3, -1, *sample.shape[-2:])
                tp_input = to_cuda(dist_util2.dev(), batch)
                rays_o = batch['ray_o'].to(dist_util2.dev())
                rays_d = batch['ray_d'].to(dist_util2.dev())
                near = batch['near'].to(dist_util2.dev())
                far = batch['far'].to(dist_util2.dev())
                cloth_layer_index = batch['cloth_layer_index'].item()
                view_index = batch['view_index'].item()

                # # save smpl kpts
                # joints_uv = batch['joints_uv']
                # if th.distributed.get_rank() == 0:
                #     path = os.path.join(logger.get_dir(), f'cloth_layer_{cloth_layer_index}_smpl_kpts')
                #     os.makedirs(path, exist_ok=True) 
                #     for i in range(joints_uv.shape[0]):
                #         np.save(f'{path}/img_{suffix}_cloth_layer_{cloth_layer_index}_human_{iteration}_view_index_{view_index}', joints_uv[i].cpu().numpy())

                with th.no_grad():
                    rgb, acc, normal_map, depth_map = render(chunk=512*512//16, rays_o=rays_o, rays_d=rays_d, tp_input=tp_input, near=near, far=far, tri_planes=tri_planes, renderer=human_nerf, n_samples=128, perturb=0., n_importance=128)

                H, W = int((rgb.shape[1])**0.5), int((rgb.shape[1])**0.5)

                if th.distributed.get_rank() == 0:
                    path = os.path.join(logger.get_dir(), f'cloth_layer_{cloth_layer_index}_image')
                    os.makedirs(path, exist_ok=True) 
                    for i in range(rgb.shape[0]):
                        imageio.imwrite(f'{path}/img_{suffix}_cloth_layer_{cloth_layer_index}_human_{iteration}_view_index_{view_index}.png', rgb[i].detach().reshape(H, W, 3).cpu().numpy())

                # stack for videos
                if th.distributed.get_rank() == 0:
                    for i in range(rgb.shape[0]):
                        frames.append(rgb[0].reshape(H, W, 3).cpu().numpy())

            if th.distributed.get_rank() == 0:
                frames = np.stack(frames)
                # save video 
                vid_path = os.path.join(logger.get_dir(), f'cloth_layer_{cloth_layer_index}_video')
                os.makedirs(vid_path, exist_ok=True) 
                imageio.mimwrite(
                    f'{vid_path}/ckpt_{suffix}_human_{iteration}_cloth_layer_{cloth_layer_index}.mp4', (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8),
                    fps=20, quality=8
                )

            # extract geometry
            vertices, triangles =\
                human_nerf.extract_geometry(tp_input, tri_planes=tri_planes, resolution=512, threshold=0)
            os.makedirs(os.path.join(logger.get_dir(), f'cloth_layer_{cloth_layer_index}_meshes'), exist_ok=True)
            mesh_path = os.path.join(logger.get_dir(), f'cloth_layer_{cloth_layer_index}_meshes', f'mesh_{suffix}_cloth_layer_{cloth_layer_index}_human_{iteration}.ply')
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(mesh_path)

        iteration += 1

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        if args.layer_index == 0:
            out_path = os.path.join(logger.get_dir(), f"samples_person_{shape_str}_{suffix}_start_id_{args.start_id}.npz")
        elif args.layer_index == 1:
            out_path = os.path.join(logger.get_dir(), f"samples_person_pant_{shape_str}_{suffix}_start_id_{args.start_id}.npz")
        elif args.layer_index == 2:
            out_path = os.path.join(logger.get_dir(), f"samples_person_pant_shirt_{shape_str}_{suffix}_start_id_{args.start_id}.npz")
        elif args.layer_index == 3:
            out_path = os.path.join(logger.get_dir(), f"samples_person_pant_shirt_shoes_{shape_str}_{suffix}_start_id_{args.start_id}.npz")

        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def render(chunk=1024*32, rays_o=None, rays_d=None, near=0., far=1., tri_planes=None, tp_input=None, renderer=None, n_samples=128, perturb=0., n_importance=0, white_bkgd=False):
    """Render rays"""

    batch_size, n_rays, _ = rays_d.shape # (bs, N_rays, 3)

    # Create ray batch
    rays_o = rays_o.reshape(batch_size, -1, 3)
    rays_d = rays_d.reshape(batch_size, -1, 3)
    near = near.reshape(batch_size, -1, 1)
    far = far.reshape(batch_size, -1, 1)

    all_ret = {}
    for i in range(0, rays_o.shape[1], chunk):
        rays_o_micro = rays_o[:, i:i+chunk]
        rays_d_micro = rays_d[:, i:i+chunk]
        near_micro = near[:, i:i+chunk]
        far_micro = far[:, i:i+chunk]
        t_vals = th.linspace(0., 1., steps=n_samples, device='cuda')
        z_vals_micro = near_micro * (1.-t_vals) + far_micro * (t_vals) # [bs, N_rays, N_samples]
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals_micro[...,1:] + z_vals_micro[...,:-1])
            upper = th.cat([mids, z_vals_micro[...,-1:]], -1)
            lower = th.cat([z_vals_micro[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = th.rand(z_vals_micro.shape, device='cuda') 
            z_vals_micro = lower + (upper - lower) * t_rand # [bs, N_rays, N_samples]
        pts_micro = rays_o_micro[...,None,:] + rays_d_micro[...,None,:] * z_vals_micro[...,:,None] # [bs, N_rays, N_samples, 3]
        pts_micro = pts_micro.reshape(pts_micro.shape[0], -1, 3)
        ret = renderer.render(tp_input, pts_micro, z_vals_micro, rays_o_micro, rays_d_micro, near_micro, far_micro, tri_planes, n_importance, white_bkgd)

        for k in ret:
            if k not in all_ret: all_ret[k] = [] 
            all_ret[k].append(ret[k])
        th.cuda.empty_cache()

    all_ret = {k : th.cat(all_ret[k], 1) for k in all_ret}
    ret_list = [all_ret[k] for k in all_ret.keys()]
    return ret_list

def create_argparser():
    defaults = dict(
        data_name="synbody",
        log_dir="",
        data_dir="",
        start_id=0,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        layer_index=0,
        sample_npz="",
        use_cond=False,
        local_rank=0,
        cond_person_id=-1,
        cond_pant_id=-1,
        cond_shirt_id=-1,
        start_motion_pose_id=6000,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()




