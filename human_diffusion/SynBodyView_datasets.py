from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os, imageio, cv2, time, copy, math, json
from random import sample

from smpl.smpl_numpy import SMPL
from smplx.body_models import SMPLX
import torch

class SynBodyViewDataset(Dataset):
    def __init__(self, triplane_root=None, img_root=None, split='test', num_instance=1, pose_start=0, pose_interval=5, pose_num=1, view_num=185, N_rand=2048, image_scaling=0.5, layer_idx=None, smpl_type='smplx', start_motion_pose_id=6000):
        super(SynBodyViewDataset, self).__init__()
        self.img_root = img_root
        self.split = split
        self.image_scaling = image_scaling
        self.layer_idx = layer_idx
        self.start_motion_pose_id = start_motion_pose_id

        self.output_view = [x for x in range(145, 185)]
        self.num_instance = num_instance
        self.cloth_layer_num = 1
        self.pose_start = pose_start 
        self.pose_interval = pose_interval
        self.pose_num = pose_num 
        self.N_rand = N_rand
        self.smpl_type = smpl_type

        all_human_data_root = os.path.join(os.path.dirname(img_root))
        triplane_ft_dir = os.path.dirname(triplane_root)
        triplane_ft_dir = '/mnt/sfs-common/fzhong/skhu/HumanLayerDiffusion/recon_NeRF/logs/Synbody_185_view_100_subject_triplane_256x256x27_mlp_2_layer_N_coarse_samples_128_fine_sample_128_no_human_sample_lr_5e-3_tri_plane_lr_1e-1_nineplane_scale_tv_loss_1e-2_l1_loss_5e-4'
        self.human_list = os.path.join(triplane_ft_dir, 'human_list.txt')
        human_dirs = []
        with open(self.human_list) as f:
            for line in f.readlines()[0:num_instance]:
                human_name = line.strip().split('_')[1]
                human_dirs.append(f'seq_{human_name}')

        self.root_list = [os.path.join(all_human_data_root, x) for x in human_dirs]
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
                transl=torch.zeros_like(params_tensor['transl']),
                return_full_pose=True,
            )
            t_vertices = np.array(body_model_output.vertices).reshape(-1,3)
            t_vertices = (np.matmul(t_vertices, self.big_pose_params['R'].transpose()) + self.big_pose_params['Th']).astype(np.float32)
            # for code simplicity, follow SMPL to add shapes and poses attributes to params
            self.big_pose_params['poses'] = body_model_output.full_pose
            self.big_pose_params['shapes'] = torch.cat([torch.from_numpy(self.big_pose_params['betas']), torch.from_numpy(self.big_pose_params['expression'])], dim=-1)

        # prepare t pose and vertex
        self.t_vertices = t_vertices.astype(np.float32)
        min_xyz = np.min(self.t_vertices, axis=0)
        max_xyz = np.max(self.t_vertices, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[1] -= 0.05
        max_xyz[1] += 0.05
        self.t_world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        tri_plane = []
        with open(os.path.join(triplane_ft_dir, 'human_list.txt')) as f:
            for line in f.readlines()[:num_instance]:
                line = line.strip()
                tri_plane.append(np.array(torch.load(os.path.join(triplane_ft_dir, line), map_location='cpu')['network_fn_state_dict']['tri_planes'])) 
        tri_plane = np.stack(tri_plane, 0).squeeze(1)
        self.tri_plane = tri_plane.reshape(tri_plane.shape[0], tri_plane.shape[1], -1, *tri_plane.shape[-2:])
        print('triplane range: ', tri_plane.shape, tri_plane.min(), tri_plane.max())

    def get_mask(self, mask_path):
        msk = imageio.imread(mask_path)
        msk[msk!=0]=255
        return msk

    def prepare_smpl_params(self, smpl_path, pose_index):
        if self.smpl_type == 'smpl':
            params_ori = dict(np.load(smpl_path, allow_pickle=True))['smpl'].item()
            gender = 'neutral'
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
            params['Th'] = np.array(params_ori['transl'][0:1]).astype(np.float32)
        return params, gender

    def prepare_input(self, smpl_path, pose_index, params=None, gender=None):

        if params is None and gender is None:
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
                transl=torch.zeros_like(params['transl']),#params['transl'],
                return_full_pose=True,
            )
            xyz = np.array(body_model_output.vertices).reshape(-1,3)
            joints = np.array(body_model_output.joints).reshape(-1,3)

            # for code simplicity, follow SMPL to add shapes and poses attributes to params
            params['poses'] = body_model_output.full_pose
            params['shapes'] = torch.cat([params['betas'], params['expression']], dim=-1)

        xyz = (np.matmul(xyz, params['R'].transpose()) + params['Th']).astype(np.float32)
        vertices = xyz

        joints = (np.matmul(joints, params['R'].transpose()) + params['Th']).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        min_xyz[1] -= 0.05
        max_xyz[1] += 0.05
        world_bounds = np.stack([min_xyz, max_xyz], axis=0)

        return world_bounds, vertices, params, joints

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
            big_pose_params['global_orient'] = np.array([[-3.1416, 0.0000, 0.0000]]).astype(np.float32)
            big_pose_params['betas'] = np.zeros((1,10)).astype(np.float32)
            big_pose_params['body_pose'] = np.zeros((1,63)).astype(np.float32)
            big_pose_params['jaw_pose'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['left_hand_pose'] = np.zeros((1,45)).astype(np.float32)
            big_pose_params['right_hand_pose'] = np.zeros((1,45)).astype(np.float32)
            big_pose_params['leye_pose'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['reye_pose'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['expression'] = np.zeros((1,10)).astype(np.float32)
            big_pose_params['transl'] = np.zeros((1,3)).astype(np.float32)
            big_pose_params['body_pose'][0, 2] = 45/180*np.array(np.pi).astype(np.float32)
            big_pose_params['body_pose'][0, 5] = -45/180*np.array(np.pi).astype(np.float32)
            big_pose_params['body_pose'][0, 20] = -30/180*np.array(np.pi).astype(np.float32)
            big_pose_params['body_pose'][0, 23] = 30/180*np.array(np.pi).astype(np.float32)
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
        instance_idx = index // (self.cloth_layer_num * self.pose_num * len(self.output_view))
        cloth_layer_index = (index - instance_idx * self.cloth_layer_num * self.pose_num * len(self.output_view)) // (self.pose_num * len(self.output_view))
        pose_index = (index - instance_idx * self.cloth_layer_num * self.pose_num * len(self.output_view) - cloth_layer_index * self.pose_num * len(self.output_view)) // len(self.output_view) * self.pose_interval + self.pose_start
        view_index = index % len(self.output_view)

        if self.layer_idx is not None:
            cloth_layer_index = int(self.layer_idx)

        view_index = self.output_view[view_index]

        self.data_root = self.root_list[instance_idx]
        self.cams = self.cams_all[instance_idx]

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
            smpl_path = os.path.join(self.data_root, 'smpl.npz')
        elif self.smpl_type == 'smplx':  
            smpl_path = os.path.join(self.data_root, 'smplx.npz')
        world_bounds, vertices, params, _ = self.prepare_input(smpl_path, pose_index)

        # Sample rays in target space world coordinate
        rgb, ray_o, ray_d, near, far, mask_at_box, bkgd_msk = sample_ray_batch(
                img, msk, K, R, T, world_bounds)
        # img = np.transpose(img, (2,0,1))

        if cloth_layer_index == 0:
            cloth_layer_condition = np.zeros_like(self.tri_plane[0,0])
        else:
            cloth_layer_condition = self.tri_plane[instance_idx, cloth_layer_index-1]
        label_dict = {}
        label_dict["y"] = np.array(cloth_layer_index, dtype=np.int64)

        out_dict = {}
        out_dict['cloth_layer_index'] = cloth_layer_index
        out_dict['view_index'] = view_index

        out_dict['x'] = self.tri_plane[instance_idx, cloth_layer_index]
        out_dict['x_cond'] = cloth_layer_condition
        # out_dict['label'] = label_dict
        out_dict['rgb'] = rgb
        out_dict['ray_o'] = ray_o
        out_dict['ray_d'] = ray_d
        out_dict['near'] = near
        out_dict['far'] = far
        out_dict['bkgd_msk'] = bkgd_msk
        out_dict['mask_at_box'] = mask_at_box
        out_dict['params'] = params
        out_dict['vertices'] = vertices
        # out_dict['joints_uv'] = joints_uv
        out_dict['world_bounds'] = world_bounds
        out_dict['t_params'] = self.big_pose_params
        out_dict['t_world_bounds'] = self.t_world_bounds

        return out_dict

    def __len__(self):
        return self.num_instance * self.cloth_layer_num * self.pose_num * len(self.output_view)


################################################################################

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    ray_d[ray_d==0.0] = 1e-8
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2

    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def sample_ray_batch(img, msk, K, R, T, bounds):

    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)
    img_ray_d = ray_d.copy()
    img_ray_d = img_ray_d / np.linalg.norm(img_ray_d, axis=-1, keepdims=True)
    pose = np.concatenate([R, T], axis=1)
    
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    mask_bkgd = True

    msk = msk * bound_mask
    if mask_bkgd:
        img[bound_mask != 1] = 0

    rgb = img.reshape(-1, 3).astype(np.float32)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)

    near_all = np.zeros_like(ray_o[:,0])
    far_all = np.ones_like(ray_o[:,0])
    near_all[mask_at_box] = near 
    far_all[mask_at_box] = far 
    near = near_all
    far = far_all
    bkgd_msk = msk.reshape(-1)

    return rgb, ray_o, ray_d, near, far, mask_at_box.reshape(-1), bkgd_msk