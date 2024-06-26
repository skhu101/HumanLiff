"""
Train a diffusion model on images.
"""

import sys, os
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current + "/../")

import argparse
import datetime

from improved_diffusion import dist_util, logger
from improved_diffusion.triplane_datasets import load_triplane_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
import torch as th
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    rank, world_size, gpu = dist_util.setup_dist()
    args.rank, args.world_size, args.gpu = rank, world_size, gpu
    th.cuda.set_device(args.gpu)
    th.backends.cudnn.benchmark = True
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=9000))
    print(args.rank, args.world_size, args.gpu)

    logger.configure(dir=args.log_dir)

    if dist.get_rank() == 0:
        writer = SummaryWriter(os.path.join(args.log_dir, "runs"))
    else:
        writer = None

    if dist.get_rank() == 0:
        logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if dist.get_rank() == 0:
        logger.log("creating data loader...")
    if args.data_name == "imagenet":
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )
    else:
        data = load_triplane_data(
            data_name=args.data_name,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            layer_idx=args.layer_idx,
            num_subjects=args.num_subjects,
            world_size=args.world_size,
            rank=args.rank,
        )

    if dist.get_rank() == 0:
        logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        use_amp=args.use_amp,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_cond=args.use_cond,
        writer=writer,
    ).run_loop()
    if dist.get_rank() == 0:
        writer.close()


def create_argparser():
    defaults = dict(
        data_name="SynBody",
        data_dir="",
        log_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_amp=False,
        num_subjects=1000,
        layer_idx=None,
        use_cond=False,
        local_rank=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
