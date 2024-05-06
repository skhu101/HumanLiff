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

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
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
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            classes = args.layer_index * th.ones_like(classes)
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.layer_index == 0:
            # smplx_tri_plane = th.load(os.path.join('/mnt/lustre/chenzhaoxi1.vendor/human_diff-main_layered/triplane_gen/logs/Synbody_185_view_100_subject_triplane_256x256x18_mlp_2_layer_N_coarse_samples_128_fine_sample_128_no_human_sample_lr_5e-3_tri_plane_lr_1e-1_ft', 'seq_smplx_010000.tar'), map_location='cpu')['network_fn_state_dict']['tri_planes'][0, 0].reshape(args.in_channels//2, args.image_size, args.image_size).unsqueeze(0).repeat(args.batch_size, 1, 1, 1).to(dist_util.dev())
            # x_cond = smplx_tri_plane 
            x_cond = th.zeros(args.batch_size, args.in_channels//2, args.image_size, args.image_size).to(dist_util.dev())
        else:
            sample_arr = np.load(args.sample_npz)
            x_cond = th.from_numpy(sample_arr.f.arr_0.astype(np.float32)).to(dist_util.dev())
            # tri_plane = []
            # triplane_ft_dir = '/mnt/lustre/chenzhaoxi1.vendor/human_diff-main_layered/triplane_gen/logs/Synbody_185_view_100_subject_triplane_256x256x18_mlp_2_layer_N_coarse_samples_128_fine_sample_128_no_human_sample_lr_5e-3_tri_plane_lr_1e-1_ft'
            # with open(os.path.join(triplane_ft_dir, 'seq_triplane.txt')) as f:
            #     for line in f.readlines()[:args.batch_size]:
            #         line = line.strip()
            #         tri_plane.append(th.load(os.path.join(triplane_ft_dir, line), map_location='cpu')['network_fn_state_dict']['tri_planes'])  
            # x_cond = th.stack(tri_plane, 0).squeeze(1)[:, args.layer_index].reshape(args.batch_size, args.in_channels//2, args.image_size, args.image_size).to(dist_util.dev())
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels//2, args.image_size, args.image_size),
            x_cond = x_cond,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample#.permute(0, 2, 3, 1)
        # sample = sample.contiguous()

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

    # reverse triplane
    # x_new = arr.reshape(-1)
    # x_new_min, x_new_max = -0.6611096, 0.6611096 # -0.6611096 0.6611096
    # x_reverse = (x_new + 1) / 2 * (x_new_max - x_new_min) + x_new_min
    # x_reverse_new = np.zeros_like(x_new)
    # for i in range(x_reverse.shape[0]):
    #     if x_reverse[i] > 0:
    #         x_reverse_new[i] = 100 ** (x_reverse[i]) - 1  #math.log(x[i]+1, 100)
    #     elif x_reverse[i] < 0:
    #         x_reverse_new[i] = -100 ** (-x_reverse[i]) + 1 #-math.log(-(x[i]-1), 100)
    # tri_plane_new_reverse = x_reverse_new.reshape(*arr.shape) / 20
    # import pdb; pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.hist(tri_plane_new_reverse.reshape(-1), bins=100)
    # plt.savefig(f'rednderpeople_seq_000000_triplane_256x256x96_norm_diff_steps_1000_hist_0.png')
    # plt.close()
    # plt.hist(x_new, bins=100)
    # plt.savefig(f'rednderpeople_seq_000000_triplane_256x256x96_norm_diff_steps_1000_hist_scale_log_0.png')

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        if args.layer_index == 0:
            out_path = os.path.join(logger.get_dir(), f"samples_person_{shape_str}_{suffix}.npz")
        elif args.layer_index == 1:
            out_path = os.path.join(logger.get_dir(), f"samples_person_pant_{shape_str}_{suffix}.npz")
        elif args.layer_index == 2:
            out_path = os.path.join(logger.get_dir(), f"samples_person_pant_shirt_{shape_str}_{suffix}.npz")
        elif args.layer_index == 3:
            out_path = os.path.join(logger.get_dir(), f"samples_person_pant_shirt_shoes_{shape_str}_{suffix}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)





    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        log_dir="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        layer_index=0,
        sample_npz="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
