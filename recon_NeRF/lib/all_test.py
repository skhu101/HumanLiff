# -*- coding: utf-8 -*
import os
import numpy as np
import imageio
import time
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from lib.fields import *
from parser_config import *
from lib.SynBody_dataset import SynBodyDatasetBatch
from lib.TightCap_dataset import TightCapDatasetBatch
from skimage.metrics import structural_similarity as compare_ssim
import json 
import trimesh


def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def ssim_metric(rgb_pred, rgb_gt, mask_at_box, H, W, loss_fn_vgg):
    # convert the pixels into an image
    img_pred = np.zeros((H, W, 3))
    img_pred[mask_at_box] = rgb_pred
    img_gt = np.zeros((H, W, 3))
    img_gt[mask_at_box] = rgb_gt

    # crop the object region
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]
    
    # compute the ssim
    ssim = compare_ssim(img_pred, img_gt, multichannel=True)
    
    # compute LPIPS 
    lpips = loss_fn_vgg(torch.from_numpy(img_pred).permute(2, 0, 1).to(torch.float32), torch.from_numpy(img_gt).permute(2, 0, 1).to(torch.float32)).reshape(-1).item()

    return ssim, lpips


def test_SynBody(chunk, render_kwargs, savedir=None, global_args=None, device=None, render=None, loss_fn_vgg=None):

    # novel view synthesis
    batch_size = 1
    pose_start = 0
    pose_interval = 1
    pose_num = 1
    H, W = int(1024*global_args.image_scaling), int(1024*global_args.image_scaling)
    
    data_root_dir = os.path.dirname(global_args.data_root)
    human_list = os.path.join(data_root_dir, 'human_list.txt')
    with open(human_list) as f:
        human_names = f.readlines()
    test_SynBody_list_all = [os.path.join(data_root_dir, human_name.strip()) for human_name in human_names]
    if global_args.num_instance <= 100:
        test_SynBody_list = test_SynBody_list_all[0:global_args.num_instance]
    else:
        test_SynBody_list = test_SynBody_list_all[0:global_args.num_instance]

    # extract depth
    test_SynBody_list = test_SynBody_list_all

    metric = {
            "novel_view_mean_human":[], "novel_view_all_human":[], "novel_view_mse":[], "novel_view_psnr":[], "novel_view_ssim":[], "novel_view_lipis":[], 
            "novel_pose_mean_human":[], "novel_pose_all_human":[], "novel_pose_mse":[], "novel_pose_psnr":[], "novel_pose_ssim":[], "novel_pose_lpips":[],
            "all_human_names":[]
            }

    all_human_psnr = []
    all_human_mse = []
    all_human_ssim = []
    all_human_lpips = []
    for human_data_path in test_SynBody_list:
        data_root = human_data_path
        human_id = test_SynBody_list_all.index(human_data_path)
        human_name = human_names[human_id].strip()

        # novel pose novel view test
        test_set = SynBodyDatasetBatch(
                data_root=data_root, 
                split=global_args.test_split, 
                multi_person=0,
                num_instance=1,
                pose_start=pose_start, 
                pose_interval=pose_interval, 
                poses_num=pose_num,
                views_num=global_args.views_num,
                image_scaling=global_args.image_scaling,
            )
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=False)

        # load triplane
        tri_plane_ckpt = torch.load(f'{global_args.basedir}/{global_args.expname}/{human_name}_002000.tar', map_location='cpu')['network_fn_state_dict']['tri_planes'].repeat(global_args.num_instance, 1, 1, 1, 1, 1).cuda()
        render_kwargs['renderer'].module.tri_planes.data = tri_plane_ckpt

        if global_args.views_num == 185:
            view_id_lst = [145, 165] + [145+global_args.views_num, 165+global_args.views_num] + [145+global_args.views_num*2, 165+global_args.views_num*2] + [145+global_args.views_num*3, 165+global_args.views_num*3]
            if global_args.test_layer_id == 0:
                view_id_lst = [i for i in range(145, 186)] 
            elif global_args.test_layer_id == 1:
                view_id_lst = [i+185 for i in range(145, 186)] 
            elif global_args.test_layer_id == 2:    
                view_id_lst = [i+2*185 for i in range(145, 186)] 
            elif global_args.test_layer_id == 3:      
                view_id_lst = [i+3*185 for i in range(145, 186)]
        human_save_path = os.path.join(savedir, "novel_view", human_names[human_id].strip())
        os.makedirs(human_save_path, exist_ok=True)
        all_pose_psnr = []
        all_pose_mse = []
        all_pose_ssim = []
        all_pose_lpips = []
        for view_id, data in enumerate(test_loader):

            # ### extract geometry ###
            # if view_id in [145, 145+global_args.views_num, 145+global_args.views_num*2, 145+global_args.views_num*3]:
            #     data = to_cuda(device, data)
            #     cloth_layer_index = view_id//global_args.views_num
            #     vertices, triangles =\
            #         render_kwargs['renderer'].module.extract_geometry(data, resolution=512, threshold=0.0)
            #     os.makedirs(os.path.join(human_save_path, 'meshes'), exist_ok=True)

            #     mesh_path = os.path.join(human_save_path, 'meshes', 'mesh_layer{:04d}.ply'.format(cloth_layer_index))
            #     mesh = trimesh.Trimesh(vertices, triangles)
            #     mesh.export(mesh_path)

            if view_id not in view_id_lst:
                continue

            data = to_cuda(device, data)
            tp_input = data
            all_view_psnr = []
            all_view_mse = []
            all_view_ssim = []
            all_view_lpips = []

            with autocast(enabled=False):
                k = 0
                rays_o=tp_input['ray_o_all'][:,k]
                rays_d=tp_input['ray_d_all'][:,k]
                near=tp_input['near_all'][:,k]
                far=tp_input['far_all'][:,k]
                target_s = tp_input['rgb_all'][:, k]
                mask_at_box = tp_input['mask_at_box_all'][:,k]
                tp_input['instance_idx'] = torch.tensor([human_id]).expand(batch_size)
                tp_input['cloth_layer_index'] = torch.tensor([view_id//global_args.views_num]).expand(batch_size)

                # use mask_at_box to discard unimportant pixel
                time_0 = time.time()
                rgb, acc, normal_map, depth_map = render(chunk=H*W//16, rays_o=rays_o, rays_d=rays_d, tp_input=tp_input,
                                                near=near, far=far, **render_kwargs)
                time_1 = time.time()
                print("Time per image: ", time_1 - time_0)

            rgb = rgb.reshape(batch_size, H, W, 3).detach()
            target_s = target_s.reshape(batch_size, H, W, 3)
            mask_at_box = mask_at_box.reshape(batch_size, H, W)
            depth_map = depth_map.reshape(batch_size, H, W).detach()
            depth_map = depth_map[..., None].repeat(1, 1, 1, 3)
            
            # for j in range(batch_size):
            #     depth_map = depth_map[j]
            #     depth_map[~mask_at_box[j]] = 0
            #     depth_rgb8 = to8b(depth_map.cpu().numpy())
                
            #     os.makedirs(os.path.join(savedir, 'depth', 'cloth_layer{:04d}'.format(int(tp_input['cloth_layer_index'][j]))), exist_ok=True)

            #     depth_filename = os.path.join(savedir, 'depth', 'cloth_layer{:04d}'.format(int(tp_input['cloth_layer_index'][j])), '{}_frame{:04d}_view{:04d}.png'.format(human_names[human_id].strip(), int(tp_input['pose_index'][j]), view_id))
            #     imageio.imwrite(depth_filename, depth_rgb8)
            # continue

            for j in range(batch_size):
                img_pred = rgb[j]
                gt_img = target_s[j]
                img_pred[~mask_at_box[j]] = 0
                pred_rgb8 = to8b(img_pred.cpu().numpy())
                gt_rgb8 = to8b(gt_img.cpu().numpy())
                gt_filename = os.path.join(human_save_path, 'cloth_layer{:04d}_frame{:04d}_view{:04d}_gt.png'.format(int(tp_input['cloth_layer_index'][j]), int(tp_input['pose_index'][j]), view_id))
                pred_filename = os.path.join(human_save_path, 'cloth_layer{:04d}_frame{:04d}_view{:04d}.png'.format(int(tp_input['cloth_layer_index'][j]), int(tp_input['pose_index'][j]), view_id))
                imageio.imwrite(gt_filename, gt_rgb8)
                imageio.imwrite(pred_filename, pred_rgb8)
            
                mse = img2mse(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                psnr = psnr_metric(img_pred[mask_at_box[j]].cpu().numpy(), gt_img[mask_at_box[j]].cpu().numpy())
                ssim, lpips = ssim_metric(img_pred[mask_at_box[j]].cpu().numpy(), gt_img[mask_at_box[j]].cpu().numpy(), mask_at_box[j].cpu().numpy(), H, W, loss_fn_vgg)

                print("[Test] ", "human: ", human_names[human_id].strip(), " cloth_layer:", int(tp_input['cloth_layer_index'][j]), " pose:", int(tp_input['pose_index'][j]), " view:", view_id, \
                    " mse:", round(mse.item(), 5), " psnr:", {psnr}, " ssim:", {ssim}, " lpips:", {lpips})
                all_view_mse.append(mse.item())
                all_view_psnr.append(psnr)
                all_view_ssim.append(ssim)
                all_view_lpips.append(lpips)

            all_pose_mse.append(all_view_mse)
            all_pose_psnr.append(all_view_psnr)
            all_pose_ssim.append(all_view_ssim)
            all_pose_lpips.append(all_view_lpips)

        all_human_psnr.append(all_pose_psnr)
        all_human_mse.append(all_pose_mse)
        all_human_ssim.append(all_pose_ssim)
        all_human_lpips.append(all_pose_lpips)

    human_num = len(all_human_psnr)
    metric["novel_view_mse"] = np.array(all_human_mse)
    metric["novel_view_psnr"] = np.array(all_human_psnr)
    metric["novel_view_ssim"] = np.array(all_human_ssim)
    metric["novel_view_lpips"] = np.array(all_human_lpips)
    metric["novel_view_mean_human"] = np.array([np.mean(metric["novel_view_mse"][:, :, :]), np.mean(metric["novel_view_psnr"][:, :, :]), np.mean(metric["novel_view_ssim"][:, :, :])])
    metric["novel_view_all_human"] = np.array([
        np.mean(metric["novel_view_mse"][:, :, :].reshape(human_num, -1), axis=-1), 
        np.mean(metric["novel_view_psnr"][:, :, :].reshape(human_num, -1), axis=-1), 
        np.mean(metric["novel_view_ssim"][:, :, :].reshape(human_num, -1), axis=-1),
        np.mean(metric["novel_view_lpips"][:, :, :].reshape(human_num, -1), axis=-1),
        ])

    metric_json = {}
    with open(savedir+"/metrics.json", 'w') as f:
        metric_json["novel_view_mean_human"] = metric["novel_view_mean_human"].tolist()
        metric_json["novel_view_all_human"] = metric["novel_view_all_human"].tolist()
        
        json.dump(metric_json, f)

    np.save(savedir+"/metrics.npy", metric)

    return

def test_TightCap(chunk, render_kwargs, savedir=None, global_args=None, device=None, render=None, loss_fn_vgg=None):

    # novel view synthesis
    batch_size = 1
    pose_start = 0
    pose_interval = 1
    pose_num = 1
    H, W = int(512*global_args.image_scaling), int(512*global_args.image_scaling)
    
    data_root_dir = os.path.dirname(global_args.data_root)
    human_list = os.path.join(data_root_dir, 'TightCap_human_list.txt')
    with open(human_list) as f:
        human_names = f.readlines()
    test_SynBody_list_all = [os.path.join(data_root_dir, human_name.strip()) for human_name in human_names]
    test_SynBody_list = test_SynBody_list_all[0:global_args.num_instance]

    # extract depth
    test_SynBody_list = test_SynBody_list_all

    metric = {
            "novel_view_mean_human":[], "novel_view_all_human":[], "novel_view_mse":[], "novel_view_psnr":[], "novel_view_ssim":[], "novel_view_lipis":[], 
            "novel_pose_mean_human":[], "novel_pose_all_human":[], "novel_pose_mse":[], "novel_pose_psnr":[], "novel_pose_ssim":[], "novel_pose_lpips":[],
            "all_human_names":[]
            }

    all_human_psnr = []
    all_human_mse = []
    all_human_ssim = []
    all_human_lpips = []
    for human_data_path in test_SynBody_list:
        data_root = human_data_path
        human_id = test_SynBody_list_all.index(human_data_path)
        human_name = human_names[human_id].strip()

        # novel pose novel view test
        test_set = TightCapDatasetBatch(
                data_root=data_root, 
                split=global_args.test_split, 
                multi_person=0,
                num_instance=1,
                pose_start=pose_start, 
                pose_interval=pose_interval, 
                poses_num=pose_num,
                views_num=global_args.views_num,
                image_scaling=global_args.image_scaling,
            )
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=False)
        
        # load triplane
        tri_plane_ckpt = torch.load(f'{global_args.basedir}/{global_args.expname}/{human_name}_002000.tar', map_location='cpu')['network_fn_state_dict']['tri_planes'].repeat(global_args.num_instance, 1, 1, 1, 1, 1).cuda()
        render_kwargs['renderer'].module.tri_planes.data = tri_plane_ckpt

        if global_args.views_num == 185:
            view_id_lst = [53, 146] + [53+global_args.views_num, 146+global_args.views_num] + [53+global_args.views_num*2, 146+global_args.views_num*2] + [53+global_args.views_num*3, 146+global_args.views_num*3]
            if global_args.test_layer_id == 0:
                view_id_lst = [i for i in range(145, 186)] 
            elif global_args.test_layer_id == 1:
                view_id_lst = [i+185 for i in range(145, 186)] 
            elif global_args.test_layer_id == 2:    
                view_id_lst = [i+2*185 for i in range(145, 186)] 
            elif global_args.test_layer_id == 3:      
                view_id_lst = [i+3*185 for i in range(145, 186)]

        human_save_path = os.path.join(savedir, "novel_view", human_names[human_id].strip())
        os.makedirs(human_save_path, exist_ok=True)
        all_pose_psnr = []
        all_pose_mse = []
        all_pose_ssim = []
        all_pose_lpips = []
        for view_id, data in enumerate(test_loader):
            if view_id not in view_id_lst:
                continue
            data = to_cuda(device, data)
            tp_input = data
            all_view_psnr = []
            all_view_mse = []
            all_view_ssim = []
            all_view_lpips = []

            with autocast(enabled=True):
                k = 0
                rays_o=tp_input['ray_o_all'][:,k]
                rays_d=tp_input['ray_d_all'][:,k]
                near=tp_input['near_all'][:,k]
                far=tp_input['far_all'][:,k]
                target_s = tp_input['rgb_all'][:, k]
                msk = tp_input['msk_all'][:,k]
                mask_at_box = tp_input['mask_at_box_all'][:,k]
                tp_input['instance_idx'] = 0 #torch.tensor([human_id]).expand(batch_size)
                tp_input['cloth_layer_index'] = torch.tensor([view_id//global_args.views_num]).expand(batch_size)

                # use mask_at_box to discard unimportant pixel
                time_0 = time.time()
                rgb, acc, normal_map, depth_map = render(chunk=H*W//16, rays_o=rays_o, rays_d=rays_d, tp_input=tp_input,
                                                near=near, far=far, **render_kwargs)
                time_1 = time.time()
                print("Time per image: ", time_1 - time_0)

            rgb = rgb.reshape(batch_size, H, W, 3)
            target_s = target_s.reshape(batch_size, H, W, 3)
            msk = msk.reshape(batch_size, H, W)
            mask_at_box = mask_at_box.reshape(batch_size, H, W)
            depth_map = depth_map.reshape(batch_size, H, W).detach()
            depth_map = depth_map[..., None].repeat(1, 1, 1, 3)
            
            # for j in range(batch_size):
            #     depth_map = depth_map[j]
            #     depth_map[~mask_at_box[j]] = 0
            #     depth_rgb8 = to8b(depth_map.cpu().numpy())
                
            #     os.makedirs(os.path.join(savedir, 'depth', 'cloth_layer{:04d}'.format(int(tp_input['cloth_layer_index'][j]))), exist_ok=True)

            #     depth_filename = os.path.join(savedir, 'depth', 'cloth_layer{:04d}'.format(int(tp_input['cloth_layer_index'][j])), '{}_frame{:04d}_view{:04d}.png'.format(human_names[human_id].strip(), int(tp_input['pose_index'][j]), view_id))
            #     imageio.imwrite(depth_filename, depth_rgb8)
            # continue

            for j in range(batch_size):
                img_pred = rgb[j]
                gt_img = target_s[j]
                img_pred[~mask_at_box[j]] = 0
                pred_rgb8 = to8b(img_pred.cpu().numpy())
                gt_rgb8 = to8b(gt_img.cpu().numpy())
                gt_filename = os.path.join(human_save_path, 'cloth_layer{:04d}_frame{:04d}_view{:04d}_gt.png'.format(int(tp_input['cloth_layer_index'][j]), int(tp_input['pose_index'][j]), view_id))
                pred_filename = os.path.join(human_save_path, 'cloth_layer{:04d}_frame{:04d}_view{:04d}.png'.format(int(tp_input['cloth_layer_index'][j]), int(tp_input['pose_index'][j]), view_id))
                imageio.imwrite(gt_filename, gt_rgb8)
                imageio.imwrite(pred_filename, pred_rgb8)
            
                mse = img2mse(img_pred[mask_at_box[j]], gt_img[mask_at_box[j]])
                psnr = psnr_metric(img_pred[mask_at_box[j]].cpu().numpy(), gt_img[mask_at_box[j]].cpu().numpy())
                ssim, lpips = ssim_metric(img_pred[mask_at_box[j]].cpu().numpy(), gt_img[mask_at_box[j]].cpu().numpy(), mask_at_box[j].cpu().numpy(), H, W, loss_fn_vgg)

                print("[Test] ", "human: ", human_names[human_id].strip(), " cloth_layer:", int(tp_input['cloth_layer_index'][j]), " pose:", int(tp_input['pose_index'][j]), " view:", view_id, \
                    " mse:", round(mse.item(), 5), " psnr:", {psnr}, " ssim:", {ssim}, " lpips:", {lpips})
                all_view_mse.append(mse.item())
                all_view_psnr.append(psnr)
                all_view_ssim.append(ssim)
                all_view_lpips.append(lpips)

            all_pose_mse.append(all_view_mse)
            all_pose_psnr.append(all_view_psnr)
            all_pose_ssim.append(all_view_ssim)
            all_pose_lpips.append(all_view_lpips)

        all_human_psnr.append(all_pose_psnr)
        all_human_mse.append(all_pose_mse)
        all_human_ssim.append(all_pose_ssim) # human * pose * novel_view (5,5,8)
        all_human_lpips.append(all_pose_lpips)

    human_num = len(all_human_psnr)
    metric["novel_view_mse"] = np.array(all_human_mse)
    metric["novel_view_psnr"] = np.array(all_human_psnr)
    metric["novel_view_ssim"] = np.array(all_human_ssim)
    metric["novel_view_lpips"] = np.array(all_human_lpips)
    metric["novel_view_mean_human"] = np.array([np.mean(metric["novel_view_mse"][:, :, :]), np.mean(metric["novel_view_psnr"][:, :, :]), np.mean(metric["novel_view_ssim"][:, :, :])])
    metric["novel_view_all_human"] = np.array([
        np.mean(metric["novel_view_mse"][:, :, :].reshape(human_num, -1), axis=-1), 
        np.mean(metric["novel_view_psnr"][:, :, :].reshape(human_num, -1), axis=-1), 
        np.mean(metric["novel_view_ssim"][:, :, :].reshape(human_num, -1), axis=-1),
        np.mean(metric["novel_view_lpips"][:, :, :].reshape(human_num, -1), axis=-1),
        ])

    metric_json = {}
    with open(savedir+"/metrics.json", 'w') as f:
        metric_json["novel_view_mean_human"] = metric["novel_view_mean_human"].tolist()
        metric_json["novel_view_all_human"] = metric["novel_view_all_human"].tolist()
        
        json.dump(metric_json, f)

    np.save(savedir+"/metrics.npy", metric)

    return


