# -*- coding: utf-8 -*
import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler 
import torch.distributed as dist
import datetime

from parser_config import *
from lib.all_test import test_SynBody, test_TightCap

from lib.SynBody_dataset import SynBodyDatasetBatch
from lib.TightCap_dataset import TightCapDatasetBatch
from lib.renderer import Renderer
from lib.fields import *

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg')

parser = config_parser()
global_args = parser.parse_args()


def render(chunk=1024*32, rays_o=None, rays_d=None, near=0., far=1., tp_input=None, renderer=None, n_samples=128, perturb=0., n_importance=0, white_bkgd=False):
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
        t_vals = torch.linspace(0., 1., steps=n_samples, device='cuda')
        z_vals_micro = near_micro * (1.-t_vals) + far_micro * (t_vals) # [bs, N_rays, N_samples]
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals_micro[...,1:] + z_vals_micro[...,:-1])
            upper = torch.cat([mids, z_vals_micro[...,-1:]], -1)
            lower = torch.cat([z_vals_micro[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals_micro.shape, device='cuda') 
            z_vals_micro = lower + (upper - lower) * t_rand # [bs, N_rays, N_samples]
        pts_micro = rays_o_micro[...,None,:] + rays_d_micro[...,None,:] * z_vals_micro[...,:,None] # [bs, N_rays, N_samples, 3]
        pts_micro = pts_micro.reshape(pts_micro.shape[0], -1, 3)
        ret = renderer.module.render(tp_input, pts_micro, z_vals_micro, rays_o_micro, rays_d_micro, near_micro, far_micro, n_importance, white_bkgd)

        for k in ret:
            if k not in all_ret: all_ret[k] = [] 
            all_ret[k].append(ret[k])
        torch.cuda.empty_cache()

    all_ret = {k : torch.cat(all_ret[k], 1) for k in all_ret}
    ret_list = [all_ret[k] for k in all_ret.keys()]
    return ret_list


def create_nerf(args, device=None):
    """Instantiate renderer.
    """
    model = Renderer(
                        use_canonical_space=args.use_canonical_space,
                        num_instances=args.num_instance,
                        triplane_dim=args.triplane_dim,
                        triplane_ch=args.triplane_ch,
                        test=args.test,
                        )
    grad_vars = []
    tri_plane_vars = []
    for name, params in model.named_parameters():
        if name != 'tri_planes':
            grad_vars.append(params)
        else:
            tri_plane_vars.append(params)
    
    # Create optimizer
    optimizer = torch.optim.Adam([{'params': grad_vars, 'lr': args.lrate}, {'params': tri_plane_vars, 'lr': args.tri_plane_lrate}], betas=(0.9, 0.999))
    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [os.path.join(basedir, expname, args.ft_path)]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if '.tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')

        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()
        model.load_state_dict(ckpt['network_fn_state_dict'], strict=True)

    if global_args.ddp:
        model = model.to(global_args.gpu)
        model = DDP(model, device_ids=[global_args.rank], find_unused_parameters=True).to(device)
    else:
        model = nn.DataParallel(model).to(device)
    
    render_kwargs_train = {
        'perturb' : args.perturb,
        'n_samples' : args.n_samples,
        'renderer' : model,
        'n_importance' : args.n_importance,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def train(nprocs, global_args):
    args = global_args

    if global_args.data_set_type == 'SynBody':
        training_set = SynBodyDatasetBatch(
                data_root=global_args.data_root, 
                split=global_args.train_split, 
                multi_person=global_args.multi_person,
                num_instance=global_args.num_instance,
                pose_start=global_args.start, 
                pose_interval=global_args.interval, 
                poses_num=global_args.poses_num,
                views_num=global_args.views_num,
                N_rand=global_args.n_rand,
                image_scaling=args.image_scaling,
            )
    elif global_args.data_set_type == 'TightCap':
        training_set = TightCapDatasetBatch(
                data_root=global_args.data_root, 
                split=global_args.train_split, 
                multi_person=global_args.multi_person,
                num_instance=global_args.num_instance,
                pose_start=global_args.start, 
                pose_interval=global_args.interval, 
                poses_num=global_args.poses_num,
                views_num=global_args.views_num,
                N_rand=global_args.n_rand,
                image_scaling=args.image_scaling,
            )  

    if args.ddp:
        global_args.rank = int(os.environ["RANK"])
        global_args.world_size = int(os.environ["WORLD_SIZE"])
        global_args.gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(global_args.gpu)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend='nccl', world_size=global_args.world_size, rank=global_args.rank, timeout=datetime.timedelta(seconds=10800))
        print(global_args.rank, global_args.world_size, global_args.gpu)
        device = torch.device('cuda', torch.cuda.current_device())
        train_sampler = torch.utils.data.distributed.DistributedSampler(training_set, shuffle=True)
        training_loader = DataLoader(training_set, batch_size=global_args.batch_size, num_workers=global_args.num_worker, sampler=train_sampler, persistent_workers=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_loader = DataLoader(training_set, batch_size=global_args.batch_size, shuffle=True, num_workers=global_args.num_worker, pin_memory=False, persistent_workers=True)
    
    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print("Using {} GPU(s).".format(args.n_gpus))

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
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, device=device)
    global_step = start
    n_iters = global_args.n_iteration + 1

    # Summary writers
    if not global_args.test:
        writer = SummaryWriter(os.path.join(basedir, expname, "runs"))
    render_kwargs_train['renderer'].train()
    iter_per_epoch = len(training_loader) #skip_step * len(training_loader)
    scaler = GradScaler(enabled=False)
    
    running_loss = 0.0
    running_img_loss = 0.0
    running_acc_loss = 0.0
    running_tv_loss = 0.0
    running_l1_loss = 0.0

    if global_args.test:
        testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(global_step))
        render_kwargs_test['renderer'].eval()
        testsavedir = testsavedir
        if global_args.data_set_type == 'SynBody':
            test_SynBody(args.chunk, render_kwargs_test, savedir=testsavedir, global_args=global_args, device=device, render=render, loss_fn_vgg=loss_fn_vgg)
        elif global_args.data_set_type == 'TightCap':
            test_TightCap(args.chunk, render_kwargs_test, savedir=testsavedir, global_args=global_args, device=device, render=render, loss_fn_vgg=loss_fn_vgg)            
        exit()

    for param_group in optimizer.param_groups:
        new_lrate = param_group['lr']

    while global_step < n_iters:
        epoch = global_step // iter_per_epoch
        if args.ddp:
            training_loader.sampler.set_epoch(epoch)

        for i, data in enumerate(training_loader):
            tp_input = to_cuda(device, data)
            time0 = time.time()

            with autocast(enabled=False):
                k = 0
                rays_o=tp_input['ray_o_all'][:,k]
                rays_d=tp_input['ray_d_all'][:,k]
                near=tp_input['near_all'][:,k]
                far=tp_input['far_all'][:,k]
                target_s = tp_input['rgb_all'][:, k]
                bkgd_msk = tp_input['bkgd_msk_all'][:, k]

                instance_id = tp_input['instance_idx']
                cloth_layer_index = tp_input['cloth_layer_index']

                ###  Core optimization loop  ###
                rgb, acc, _, _ = render(chunk=args.chunk, rays_o=rays_o, rays_d=rays_d, tp_input=tp_input,
                                                near=near, far=far, **render_kwargs_train)

                # ### calc loss ###
                img_loss = img2mse(rgb, target_s) 
                acc_loss = img2mse(bkgd_msk.squeeze(2), acc)
                if global_args.tv_loss:
                    tv_loss_x = F.l1_loss(render_kwargs_train['renderer'].module.tri_planes[instance_id,cloth_layer_index,:,:,0:-1,:],  render_kwargs_train['renderer'].module.tri_planes[instance_id,cloth_layer_index,:,:,1:,:])
                    tv_loss_y = F.l1_loss(render_kwargs_train['renderer'].module.tri_planes[instance_id,cloth_layer_index,:,:,:,0:-1], render_kwargs_train['renderer'].module.tri_planes[instance_id,cloth_layer_index,:,:,:,1:])
                    tv_loss = tv_loss_x + tv_loss_y
                    l1_loss = F.l1_loss(render_kwargs_train['renderer'].module.tri_planes[instance_id,cloth_layer_index], torch.zeros_like(render_kwargs_train['renderer'].module.tri_planes[instance_id,cloth_layer_index]))
                else:
                    tv_loss = torch.Tensor([0]).to(device)
                loss = img_loss + 0.1 * acc_loss + global_args.tv_loss_coef * tv_loss + global_args.l1_loss_coef * l1_loss

            ### update loss ###
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            ### clamp triplane range to [-1, 1]
            if args.use_clamp:
                render_kwargs_train['renderer'].module.tri_planes.data.clamp_(-1.0, 1.0)
            
            ### calculate loss stat ###
            running_loss += loss.item()
            running_img_loss += img_loss.item()
            running_acc_loss += acc_loss.item()
            running_tv_loss += tv_loss.item()
            running_l1_loss += l1_loss.item()

            if global_step <= 300000:
                decay_rate = 0.1
                decay_steps = args.lrate_decay * 600 #args.lrate_decay * 1000
                new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))

                decay_rate = 0.5 #0.1
                decay_steps = args.lrate_decay * 60 #args.lrate_decay * 1000
                new_tri_plane_lrate = args.tri_plane_lrate * (decay_rate ** (global_step / decay_steps))

                count = 0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
                    if count == 0:
                        param_group['lr'] = new_lrate
                    else:
                        param_group['lr'] = new_tri_plane_lrate
                    count += 1

            ###     Logger    ###
            dt = (time.time()-time0)
            global_step += 1

            if (args.ddp and dist.get_rank() == 0) or not args.ddp:
                writer.add_scalar("All training loss", loss.item(), global_step)
                writer.add_scalar("image loss", img_loss.item(), global_step)
                writer.add_scalar("acc weights loss", acc_loss.item(), global_step)
                writer.add_scalar("tv loss", tv_loss.item(), global_step)
                writer.add_scalar("l1 loss", l1_loss.item(), global_step)
                writer.add_scalar("psnr", round(mse2psnr(torch.tensor(img_loss.item())).item(), 3), global_step)

            if (global_step) % args.i_print == 0 and global_step > 1:
                psnr = round(mse2psnr(torch.tensor(running_img_loss / args.i_print)).item(), 3)
                if (args.ddp and dist.get_rank() == 0) or not args.ddp:
                    print("[TRAIN] Epoch:{}  Iter: {} Lr: {} Loss: {} Img Loss: {} Acc Loss: {} tv Loss: {} L1 Loss: {}  PSNR: {}  Time: {} s/iter".format(epoch, global_step, round(new_lrate, 6), round(running_loss / args.i_print, 5), round(running_img_loss / args.i_print, 5), round(running_acc_loss / args.i_print, 5), round(running_tv_loss / args.i_print, 5), round(running_l1_loss / args.i_print, 5), psnr, round(dt, 3)))
                running_loss = 0.0
                running_img_loss = 0.0
                running_acc_loss = 0.0
                running_tv_loss = 0.0
                running_l1_loss = 0.0
            
            if ((global_step)%args.i_weights == 0 and global_step > 1) or global_step == 5000:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(global_step))
                
                if (args.ddp and dist.get_rank() == 0) or not args.ddp:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['renderer'].module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                    print('Saved checkpoints at', path)

if __name__=='__main__':
    print_args(global_args)
    if not global_args.ddp:
        torch.multiprocessing.set_start_method('spawn', force=True)
        global_args.rank = 0
    train(torch.cuda.device_count(), global_args)
