import torch
from torch.utils.data import DataLoader, dataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import imageio
import cv2
import time
import lib.if_nerf_data_utils as if_nerf_dutils
import copy
from random import sample
from smpl.smpl_numpy import SMPL
from smplx.body_models import SMPLX
from cv2 import Rodrigues as rodrigues

import json

def writeOBJ(file, V, F, Vt=None, Ft=None):    
    if not Vt is None:        
        assert len(F) == len(Ft), 'Inconsistent data, mesh and UV map do not have the same number of faces'            
    with open(file, 'w') as file:        # Vertices        
        for v in V:            
            line = 'v ' + ' '.join([str(_) for _ in v]) + '\n'            
            file.write(line)        # UV verts        
        if not Vt is None:            
            for v in Vt:                
                line = 'vt ' + ' '.join([str(_) for _ in v]) + '\n'                
                file.write(line)        # 3D Faces / UV faces        
                if Ft:            
                    F = [[str(i+1)+'/'+str(j+1) for i,j in zip(f,ft)] for f,ft in zip(F,Ft)]        
                else:            
                    F = [[str(i + 1) for i in f] for f in F]                
        for f in F:            
            line = 'f ' + f'{f[0]} {f[1]} {f[2]}' + '\n'            
            file.write(line)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class SynBodyDatasetBatch(Dataset):
    def __init__(self, data_root=None, split='test', multi_person=True, num_instance=1, pose_start=0, pose_interval=5, poses_num=1, views_num=185, N_rand=4096, image_scaling=0.5, smpl_type='smplx', layer_idx=None):
        super(SynBodyDatasetBatch, self).__init__()
        self.data_root = data_root
        self.split = split
        self.image_scaling = image_scaling

        self.train_view = self.test_view = [x for x in range(views_num)]
        self.output_view = self.train_view if split == 'train' else self.test_view
   
        print("output view: ", self.output_view)

        self.multi_person = multi_person
        self.num_instance = num_instance
        self.cloth_layer_num = 4 if layer_idx is None else 1
        self.i = pose_start # start index 0
        self.i_intv = pose_interval # interval 1
        self.ni = poses_num # number of used poses 30
        self.N_rand = N_rand
        self.smpl_type = smpl_type
        self.layer_idx = layer_idx

        all_human_data_root = os.path.join(os.path.dirname(data_root))
        self.human_list = os.path.join(all_human_data_root, 'human_list.txt')
        with open(self.human_list) as f:
            human_dirs = f.readlines()[0:num_instance]

        self.root_list = [data_root] if not multi_person else [os.path.join(all_human_data_root, x.strip()) for x in human_dirs]
        print('num of subjects: ', len(self.root_list), self.root_list[0])

        self.cams_all = []
        for subject_root in self.root_list:
            camera_file = os.path.join(subject_root, 'cameras.json')
            camera = json.load(open(camera_file))
            self.cams_all.append(camera)

        if self.smpl_type == 'smpl':
            self.smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL_SynBody.pkl')
            self.big_pose_params = self.big_pose_params()
            t_vertices, _ = self.smpl_model(self.big_pose_params['poses'], self.big_pose_params['shapes'].reshape(-1))
        elif self.smpl_type == 'smplx':
            self.smpl_model = {}
            self.smpl_model['male'] = SMPLX('assets/models/smplx/', smpl_type='smplx',
                                        gender='male', use_face_contour=True, flat_hand_mean=True, use_pca=False, 
                                        num_betas=10,
                                        num_expression_coeffs=10,
                                        ext='npz')
            self.smpl_model['female'] = SMPLX('assets/models/smplx/', smpl_type='smplx',
                                        gender='female', use_face_contour=True, flat_hand_mean=True, use_pca=False, 
                                        num_betas=10,
                                        num_expression_coeffs=10,
                                        ext='npz')
            self.smpl_model['neutral'] = SMPLX('assets/models/smplx/', smpl_type='smplx',
                                        gender='neutral', use_face_contour=True, flat_hand_mean=True, use_pca=False, 
                                        num_betas=10,
                                        num_expression_coeffs=10,
                                        ext='npz')
            self.big_pose_params = self.big_pose_params()
            params_tensor = {}
            for key in self.big_pose_params.keys():
                params_tensor[key] = torch.from_numpy(self.big_pose_params[key])

            body_model_output = self.smpl_model['neutral'](
                global_orient=params_tensor['global_orient'],
                betas=params_tensor['betas'],
                body_pose=params_tensor['body_pose'],
                jaw_pose=params_tensor['jaw_pose'],
                left_hand_pose=params_tensor['left_hand_pose'],
                right_hand_pose=params_tensor['right_hand_pose'],
                leye_pose=params_tensor['leye_pose'],
                reye_pose=params_tensor['reye_pose'],
                expression=params_tensor['expression'],
                transl=params_tensor['transl'],
                return_full_pose=True,
            )
            t_vertices = np.array(body_model_output.vertices).reshape(-1,3)

        # prepare t pose and vertex
        self.t_vertices = t_vertices.astype(np.float32)
        min_xyz = np.min(self.t_vertices, axis=0)
        max_xyz = np.max(self.t_vertices, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[1] -= 0.05
        max_xyz[1] += 0.05
        self.t_world_bounds = np.stack([min_xyz, max_xyz], axis=0)

    def get_mask(self, mask_path):
        msk = imageio.imread(mask_path)
        msk[msk!=0]=255
        return msk

    def prepare_smpl_params(self, smpl_path, pose_index):
        if self.smpl_type == 'smpl':
            params_ori = dict(np.load(smpl_path, allow_pickle=True))['smpl'].item()
            params = {}
            params['shapes'] = np.array(params_ori['betas']).astype(np.float32)
            params['poses'] = np.zeros((1,72)).astype(np.float32)
            params['poses'][:, :3] = np.array(params_ori['global_orient'][pose_index]).astype(np.float32)
            params['poses'][:, 3:] = np.array(params_ori['body_pose'][pose_index].reshape(-1)).astype(np.float32)
            params['R'] = np.eye(3).astype(np.float32)
            params['Th'] = np.array(params_ori['transl'][0:1]).astype(np.float32)
        elif self.smpl_type == 'smplx':
            params_ori = dict(np.load(smpl_path, allow_pickle=True))['smplx'].item()
            gender = dict(np.load(smpl_path, allow_pickle=True))['meta'].item()['gender']
            params = {}
            for key in params_ori.keys():
                if key == 'betas':
                    params[key] = torch.from_numpy(params_ori[key])
                else:
                    params[key] = torch.from_numpy(params_ori[key][pose_index:pose_index+1])
            params['R'] = np.eye(3).astype(np.float32)
            params['Th'] = np.zeros((1,3)).astype(np.float32)
        return params, gender

    def prepare_input(self, smpl_path, pose_index):

        params, gender = self.prepare_smpl_params(smpl_path, pose_index)
        if self.smpl_type == 'smpl':
            xyz, joints = self.smpl_model(params['poses'], params['shapes'].reshape(-1))
        elif self.smpl_type == 'smplx':
            body_model_output = self.smpl_model[gender](
                global_orient=params['global_orient'],
                betas=params['betas'],
                body_pose=params['body_pose'],
                jaw_pose=params['jaw_pose'],
                left_hand_pose=params['left_hand_pose'],
                right_hand_pose=params['right_hand_pose'],
                leye_pose=params['leye_pose'],
                reye_pose=params['reye_pose'],
                expression=params['expression'],
                transl=params['transl'],
                return_full_pose=True,
            )
            xyz = np.array(body_model_output.vertices).reshape(-1,3)
            # for code simplicity, follow SMPL to add shapes and poses attributes to params
            params['poses'] = body_model_output.full_pose
            params['shapes'] = torch.cat([params['betas'], params['expression']], dim=-1)
        
        xyz = (np.matmul(xyz, params['R'].transpose()) + params['Th']).astype(np.float32)
        vertices = xyz

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[1] -= 0.05
        max_xyz[1] += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        return world_bounds, vertices, params

    def big_pose_params(self):
        if self.smpl_type == 'smpl':
            big_pose_params = {}
            big_pose_params['R'] = np.eye(3).astype(np.float32)
            big_pose_params['Th'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['shapes'] = np.zeros((1,10)).astype(np.float32)
            big_pose_params['poses'] = np.zeros((1,72)).astype(np.float32)
            big_pose_params['poses'][0, 5] = 45/180*np.array(np.pi)
            big_pose_params['poses'][0, 8] = -45/180*np.array(np.pi)
            big_pose_params['poses'][0, 23] = -30/180*np.array(np.pi)
            big_pose_params['poses'][0, 26] = 30/180*np.array(np.pi)
        elif self.smpl_type == 'smplx':
            big_pose_params = {}
            big_pose_params['R'] = np.eye(3).astype(np.float32)
            big_pose_params['Th'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['global_orient'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['betas'] = np.zeros((1,10)).astype(np.float32)
            big_pose_params['body_pose'] = np.zeros((1,63)).astype(np.float32)
            big_pose_params['jaw_pose'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['left_hand_pose'] = np.zeros((1,45)).astype(np.float32)
            big_pose_params['right_hand_pose'] = np.zeros((1,45)).astype(np.float32)
            big_pose_params['leye_pose'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['reye_pose'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['expression'] = np.zeros((1,10)).astype(np.float32)
            big_pose_params['transl'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['body_pose'][0, 2] = 45/180*np.array(np.pi)
            big_pose_params['body_pose'][0, 5] = -45/180*np.array(np.pi)
            big_pose_params['body_pose'][0, 20] = -30/180*np.array(np.pi)
            big_pose_params['body_pose'][0, 23] = 30/180*np.array(np.pi)
        return big_pose_params

    def __getitem__(self, index):
        """
            pose_index: [0, number of used poses), the pose index of selected poses
            view_index: [0, number of all view)
            mask_at_box_all: mask of 2D bounding box which is the 3D bounding box projection 
                training: all one array, not fixed length, no use for training
                Test: zero and one array, fixed length which can be reshape to (H,W)
            bkgd_msk_all: mask of foreground and background
                trainning: for acc loss
                test: no use
        """    
        instance_idx = index // (self.cloth_layer_num * self.ni * len(self.output_view)) if self.multi_person else 0
        cloth_layer_index = (index - instance_idx * self.cloth_layer_num * self.ni * len(self.output_view)) // (self.ni * len(self.output_view))
        pose_index = (index - instance_idx * self.cloth_layer_num * self.ni * len(self.output_view) - cloth_layer_index * self.ni * len(self.output_view)) // len(self.output_view) * self.i_intv + self.i
        view_index = index % len(self.output_view)
        self.data_root = self.root_list[instance_idx]
        self.cams = self.cams_all[instance_idx]

        if self.layer_idx is not None:
            cloth_layer_index = self.layer_idx

        img_all, rgb_all, ray_o_all, ray_d_all, near_all, far_all, msk_all = [], [], [], [], [], [], []
        mask_at_box_all, bkgd_msk_all = [], []
        tar_K_all, tar_R_all, tar_T_all = [], [], []

        # Load image, mask, K, D, R, T
        if cloth_layer_index == 0:
            img_path = os.path.join(self.data_root, 'person', 'img', f'camera{str(view_index).zfill(4)}', str(pose_index).zfill(4)+'.jpg')            
            mask_path = os.path.join(self.data_root, 'person', 'mask', f'camera{str(view_index).zfill(4)}', str(pose_index).zfill(4)+'.png')
        elif cloth_layer_index == 1:
            img_path = os.path.join(self.data_root, 'person-pants', 'img', f'camera{str(view_index).zfill(4)}', str(pose_index).zfill(4)+'.jpg')            
            mask_path = os.path.join(self.data_root, 'person-pants', 'mask', f'camera{str(view_index).zfill(4)}', str(pose_index).zfill(4)+'.png')            
        elif cloth_layer_index == 2:
            img_path = os.path.join(self.data_root, 'person-pants-shirt', 'img', f'camera{str(view_index).zfill(4)}', str(pose_index).zfill(4)+'.jpg')            
            mask_path = os.path.join(self.data_root, 'person-pants-shirt', 'mask', f'camera{str(view_index).zfill(4)}', str(pose_index).zfill(4)+'.png') 
        elif cloth_layer_index == 3:
            img_path = os.path.join(self.data_root, 'person-pants-shirt-shoes', 'img', f'camera{str(view_index).zfill(4)}', str(pose_index).zfill(4)+'.jpg')            
            mask_path = os.path.join(self.data_root, 'person-pants-shirt-shoes', 'mask', f'camera{str(view_index).zfill(4)}', str(pose_index).zfill(4)+'.png') 

        img = np.array(imageio.imread(img_path).astype(np.float32) / 255.)
        msk = np.array(self.get_mask(mask_path)) / 255.
        img[msk == 0] = 0

        K = np.array(self.cams[f'camera{str(view_index).zfill(4)}']['K'])
        R = np.array(self.cams[f'camera{str(view_index).zfill(4)}']['R'])
        T = np.array(self.cams[f'camera{str(view_index).zfill(4)}']['T']).reshape(-1, 1)

        # rescaling
        H, W = img.shape[:2]
        H, W = int(H * self.image_scaling), int(W * self.image_scaling)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        K[:2] = K[:2]*self.image_scaling

        # Prepare the smpl input, including the current pose and canonical pose
        if self.smpl_type == 'smpl':
            smpl_path = os.path.join(self.data_root, 'person-top-bottom-shoes/outputs_re_fitting/refit_smpl_2nd.npz')
        elif self.smpl_type == 'smplx':  
            smpl_path = os.path.join(self.data_root, 'smplx.npz')
        world_bounds, vertices, params = self.prepare_input(smpl_path, pose_index)

        # Sample rays in target space world coordinate
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box, bkgd_msk = if_nerf_dutils.sample_ray_batch(
                img, msk, K, R, T, world_bounds, self.N_rand, self.split, ratio=0.8)

        # target view
        # img_all.append(img)
        msk_all.append(msk)
        rgb_all.append(rgb)
        ray_o_all.append(ray_o)
        ray_d_all.append(ray_d)
        near_all.append(near)
        far_all.append(far)
        mask_at_box_all.append(mask_at_box)
        bkgd_msk_all.append(bkgd_msk)

        tar_K_all.append(K)
        tar_R_all.append(R)
        tar_T_all.append(T)

        # target view
        # img_all = np.stack(img_all, axis=0)
        msk_all = np.stack(msk_all, axis=0)
        rgb_all = np.stack(rgb_all, axis=0)
        ray_o_all = np.stack(ray_o_all, axis=0)
        ray_d_all = np.stack(ray_d_all, axis=0)
        near_all = np.stack(near_all, axis=0)[...,None]
        far_all = np.stack(far_all, axis=0)[...,None]
        mask_at_box_all = np.stack(mask_at_box_all, axis=0)
        bkgd_msk_all = np.stack(bkgd_msk_all, axis=0)

        tar_K_all = np.stack(tar_K_all, axis=0)
        tar_R_all = np.stack(tar_R_all, axis=0)
        tar_T_all = np.stack(tar_T_all, axis=0)

        ret = {
            "instance_idx": instance_idx, # person instance idx
            "cloth_layer_index": cloth_layer_index, # person instance idx
            'pose_index': pose_index, # pose_index in selected poses

            # canonical space
            't_params': self.big_pose_params,
            't_vertices': self.t_vertices,
            't_world_bounds': self.t_world_bounds,

            # target view
            "params": params, # smpl params including smpl global R, Th
            'vertices': vertices, # world vertices
            'world_bounds': world_bounds,
            # 'img_all': img_all,
            'msk_all': msk_all,
            'rgb_all': rgb_all,
            'ray_o_all': ray_o_all,
            'ray_d_all': ray_d_all,
            'near_all': near_all,
            'far_all': far_all,
            'mask_at_box_all': mask_at_box_all,
            'bkgd_msk_all': bkgd_msk_all,

            'tar_K_all': tar_K_all,
            'tar_R_all': tar_R_all,
            'tar_T_all': tar_T_all,

        }

        return ret

    def __len__(self):
        return self.num_instance * self.cloth_layer_num * self.ni * len(self.output_view)
