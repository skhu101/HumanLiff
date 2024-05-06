import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import pickle
import mcubes
from pytorch3d.ops.knn import knn_points

from lib.fields import PositionalEncoding

class Renderer(nn.Module):
    def __init__(self, use_canonical_space=False, num_instances=1, triplane_dim=256, triplane_ch=18, test=False):
        super(Renderer, self).__init__()

        self.use_canonical_space = use_canonical_space
        self.num_instances = num_instances
        self.triplane_dim = triplane_dim
        self.triplane_ch = triplane_ch
        self.test = test

        self.view_enc = PositionalEncoding(num_freqs=4)
        self.plane_axes = generate_planes()

        self.tri_planes = nn.Parameter(torch.Tensor(num_instances, 4, 3, self.triplane_ch//3, self.triplane_dim, self.triplane_dim))
        nn.init.normal_(self.tri_planes, mean=0, std=0.1)

        print(self.tri_planes.shape)
        self.triplane_ch = triplane_ch

        # load network
        d_in = triplane_ch
        n_layers = 2
        d_hidden = 128
        d_feature = d_hidden 
        self.skips = [n_layers/2]
        self.pts_linears = nn.ModuleList(
            [nn.Linear(d_in, d_hidden)] + [nn.Linear(d_hidden, d_hidden) if i not in self.skips else nn.Linear(d_hidden + d_in, d_hidden) for i in range(n_layers)])
        self.feature_linear = nn.Linear(d_hidden, d_hidden)
        self.alpha_linear = nn.Linear(d_hidden, 1)
        self.views_linear = nn.Linear(d_feature+27, d_hidden//2)
        self.rgb_linear = nn.Linear(d_hidden//2, 3)

        # load smpl model
        neutral_smpl_path = os.path.join('assets', 'SMPL_NEUTRAL.pkl')
        self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(neutral_smpl_path), device=torch.device('cuda', torch.cuda.current_device()))
        self.faces = self.SMPL_NEUTRAL['f']

    def big_pose_params(self, params):
        big_pose_params = copy.deepcopy(params)
        big_pose_params['poses'] = torch.zeros((params['poses'].shape[0], 1, 72))
        big_pose_params['poses'][:, 0, 5] = 45/180*torch.tensor(np.pi)
        big_pose_params['poses'][:, 0, 8] = -45/180*torch.tensor(np.pi)
        big_pose_params['poses'][:, 0, 23] = -30/180*torch.tensor(np.pi)
        big_pose_params['poses'][:, 0, 26] = 30/180*torch.tensor(np.pi)

        return big_pose_params

    def deform_target2c_op(self, params, vertices, t_params, query_pts, query_viewdirs=None):
        bs = query_pts.shape[0]
        # joints transformation
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params)

        joints_num = joints.shape[1]
        vertices_num = vertices.shape[1]

        # transform smpl vertices from world space to smpl space
        smpl_pts = torch.matmul((vertices - Th), R)
        _, vert_ids, _ = knn_points(query_pts.float(), smpl_pts.float(), K=1)
        bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], 24)
        # From smpl space target pose to smpl space canonical pose
        A = torch.matmul(bweights, A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        can_pts = torch.matmul(R_inv, can_pts[..., None]).squeeze(-1)

        if query_viewdirs is not None:
            query_viewdirs = torch.matmul(R_inv, query_viewdirs[..., None]).squeeze(-1)

        self.mean_shape = True
        if self.mean_shape:
            posedirs = self.SMPL_NEUTRAL['posedirs']#.cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).to(pose_.device).float()
            batch_size = pose_.shape[0] #1
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(6890*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts - pose_offsets

            # To mean shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs']
            shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(params['shapes'].cuda(), (batch_size, 1, -1, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts - shape_offset

        # From T To Big Pose        
        big_pose_params = t_params

        if self.mean_shape:
            big_pose_params['shapes'] = torch.zeros_like(params['shapes'])
            pose_ = big_pose_params['poses']
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            can_pts = can_pts + pose_offsets

        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.matmul(bweights, A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = torch.matmul(A[..., :3, :3], can_pts[..., None]).squeeze(-1)
        can_pts = can_pts + A[..., :3, 3]

        if query_viewdirs is not None:
            query_viewdirs = torch.matmul(A[..., :3, :3], query_viewdirs[..., None]).squeeze(-1)
            return can_pts, query_viewdirs

        return can_pts, None

    def deform_target2c(self, tp_input, pts, viewdir=None):
        '''deform points from target space to caonical space'''

        if self.use_canonical_space:
            # transfer pts and directions to smpl coordinate
            R = tp_input['params']['R'] # [bs, 3, 3]
            Th = tp_input['params']['Th'] # [bs, 1, 3]
            box_warp = tp_input['t_world_bounds']
            smpl_pts = torch.matmul(pts - Th, R) # [bs, N_rays*N_samples, 3]
            smpl_viewdir = torch.matmul(viewdir - Th, R) if viewdir is not None else None
            canonical_pts, canonical_viewdir = self.deform_target2c_op(tp_input['params'], tp_input['vertices'], tp_input['t_params'], smpl_pts, smpl_viewdir)
        else:
            box_warp = tp_input['world_bounds']
            canonical_pts = pts # [bs, N_rays*N_samples, 3]
            canonical_viewdir = viewdir # [bs, N_rays*N_samples, 3]

        return canonical_pts, canonical_viewdir, box_warp

    def NeRF_network(self, point_feature, box_warp=None, canonical_viewdir=None):

        x = point_feature
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.softplus(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        alpha = self.alpha_linear(h)
        
        if canonical_viewdir is None: return alpha 

        feature = self.feature_linear(h)
        viewdir = self.view_enc(canonical_viewdir.squeeze(0))
        h = torch.cat([feature, viewdir], -1)

        h = self.views_linear(h)
        h = F.softplus(h)
        rgb = self.rgb_linear(h)

        return rgb, alpha

    def up_sample(self, densities, z_vals, rays_d, n_importance):

        batch_size, n_rays, n_samples = z_vals.shape
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).to(dists.device).expand(dists[...,:1].shape)], -1)  # [batch_size, n_rays, n_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
        alpha = 1.-torch.exp(-F.softplus(densities)*dists) # [batch_size, n_rays, n_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1)).to(dists.device), 1.-alpha + 1e-10], -1), -1)[:, :, :-1]

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        new_z_vals = sample_pdf(z_vals_mid.reshape(batch_size*n_rays, -1), weights[..., 1:-1].reshape(batch_size*n_rays, -1), n_importance).reshape(batch_size, n_rays, n_importance)

        return new_z_vals

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    viewdirs, 
                    tri_planes,
                    tp_input,
                    white_bkgd=False):

        batch_size = tri_planes.shape[0]
        n_rays, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)], -1)

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3

        pts = pts.reshape(-1, 3)

        ## deform target to caonical
        canonical_pts, canonical_dirs, box_warp = self.deform_target2c(tp_input, pts.reshape(batch_size, -1, 3), viewdirs.reshape(batch_size, -1, 3))

        if self.test: canonical_pts.requires_grad_(True)

        pt_features = sample_from_planes(self.plane_axes, tri_planes, canonical_pts, padding_mode='zeros', box_warp=box_warp)
        pt_features = pt_features.permute(0,2,1,3).reshape(pts.shape[0], -1)
        canonical_dirs = canonical_dirs.reshape(-1, 3)

        sampled_color, alpha = self.NeRF_network(pt_features, box_warp=box_warp, canonical_viewdir=canonical_dirs)

        # if self.test: 
        #     d_output = torch.ones_like(alpha, requires_grad=False, device=alpha.device)
        #     gradients = torch.autograd.grad(
        #         outputs=alpha,
        #         inputs=canonical_pts,
        #         grad_outputs=d_output,
        #         only_inputs=True)[0]
        #     canonical_pts.requires_grad_(True)

        alpha = (alpha + torch.randn_like(alpha)).reshape(n_rays, n_samples) if not self.test else alpha.reshape(n_rays, n_samples)
        alpha = 1.-torch.exp(-F.softplus(alpha) * dists) # [n_rays, n_samples]

        sampled_color = torch.sigmoid(sampled_color).reshape(n_rays, n_samples, 3)

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).to(dists.device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([n_rays, 1]).to(dists.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        acc_map = weights.sum(dim=-1, keepdim=True)

        rgb_map = (sampled_color * weights[:, :, None]).sum(dim=1)
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[...,None])

        normal_map = rgb_map #(gradients.reshape(n_rays, n_samples, 3) * weights[:, :, None]).sum(dim=1) if self.test else rgb_map

        depth_map = torch.sum(weights * z_vals, -1)

        return rgb_map, acc_map, normal_map, depth_map


    def render(self, tp_input, world_pts, z_vals, rays_o, rays_d, near, far, n_importance=128, white_bkgd=False):

        # load human id and cloth layer id
        human_ids = tp_input['instance_idx']
        cloth_layer_index = tp_input['cloth_layer_index']

        # load triplane features
        tri_planes = self.tri_planes[human_ids, cloth_layer_index]
        batch_size, n_planes, C, H, W = tri_planes.shape

        self.plane_axes = self.plane_axes.to(world_pts.device)

        batch_size, n_rays, n_samples = z_vals.shape

        if n_importance > 0:
            with torch.no_grad():

                canonical_pts, _, box_warp = self.deform_target2c(tp_input, world_pts)
                pt_features = sample_from_planes(self.plane_axes, tri_planes, canonical_pts, padding_mode='zeros', box_warp=box_warp)
                pt_features = pt_features.permute(0,2,1,3).reshape(batch_size, n_rays*n_samples, -1)
                densities = self.NeRF_network(pt_features, box_warp=box_warp)

                new_z_vals = self.up_sample(densities[..., 0].reshape(batch_size, n_rays, n_importance), z_vals, rays_d, n_importance) # [batch_size, n_rays, n_samples]

                z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
                z_vals, _ = torch.sort(z_vals, dim=-1)

            n_samples = z_vals.shape[-1]

        # Render core
        viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdir = viewdir.reshape(batch_size, -1, 3) # (bs, N_rays, 3)
        rgb_map, acc_map, normal_map, depth_map = self.render_core(rays_o.reshape(batch_size*n_rays, 3),
                                    rays_d.reshape(batch_size*n_rays, 3),
                                    z_vals.reshape(batch_size*n_rays, -1),
                                    viewdir[:,:,None].expand(batch_size, n_rays, n_samples, 3),
                                    tri_planes,
                                    tp_input,
                                    white_bkgd)

        rgb_map = rgb_map.reshape(batch_size, n_rays, -1)
        acc_map = acc_map.reshape(batch_size, n_rays)
        normal_map = normal_map.reshape(batch_size, n_rays, -1)
        depth_map = depth_map.reshape(batch_size, n_rays)
        depth_map = (depth_map - near[..., 0]) / (far[..., 0] - near[..., 0] + 1e-5)

        if self.test:
            raw  = {'rgb_map': rgb_map.detach(), 'acc_map': acc_map.detach(), 'normal_map': normal_map.detach(), 'depth_map': depth_map.detach()}
            del rgb_map, acc_map, normal_map, depth_map
            return raw

        return {'rgb_map': rgb_map, 'acc_map': acc_map, 'normal_map': normal_map, 'depth_map': depth_map}

    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    def extract_geometry(self, tp_input, resolution, threshold=0.0):
        # load human id and cloth layer id
        human_ids = tp_input['instance_idx']
        cloth_layer_index = tp_input['cloth_layer_index']

        # load triplane features
        tri_planes = self.tri_planes[human_ids, cloth_layer_index]
        self.plane_axes = self.plane_axes.to(tri_planes.device)

        # extract fields
        N = resolution
        bound_min = tp_input['world_bounds'][0,0]
        bound_max = tp_input['world_bounds'][0,1]
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val_all = []
                        chunk = 2000000
                        for i in range(0, pts.shape[0], chunk):
                            pts_micro = pts[i:i+chunk]
                            ## deform target to caonical
                            canonical_pts, _, box_warp = self.deform_target2c(tp_input, pts_micro.reshape(1, -1, 3))
                            pt_features = sample_from_planes(self.plane_axes, tri_planes, canonical_pts.to(tri_planes.device), padding_mode='zeros', box_warp=box_warp)
                            pt_features = pt_features.permute(0,2,1,3).reshape(pts_micro.shape[0], -1)
                            val = -self.NeRF_network(pt_features, box_warp=box_warp).detach().cpu().numpy()
                            val_all.append(val)
                        val = np.concatenate(val_all, axis=0)
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val.reshape(N, N, N)

        print('threshold: {}'.format(threshold))
        smoothed_u = mcubes.smooth(u)
        vertices, triangles = mcubes.marching_cubes(smoothed_u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

        return vertices, triangles


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def SMPL_to_tensor(params, device):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            if isinstance(params[key1], np.ndarray):
                params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
            else:
                params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.long, device=device)
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.float32, device=device)
    return params

def get_transform_params_torch(smpl, params):
    """ obtain the transformation parameters for linear blend skinning
    """

    v_template = smpl['v_template']

    # add shape blend shapes
    shapedirs = smpl['shapedirs']
    betas = params['shapes']
    v_shaped = v_template[None] + torch.sum(shapedirs[None] * betas[:,None], axis=-1).float()

    # add pose blend shapes
    poses = params['poses'].reshape(-1, 3)
    # bs x 24 x 3 x 3
    rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

    # obtain the joints
    joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]

    A = get_rigid_transformation_torch(rot_mats, joints, parents)

    # apply global transformation
    R = params['R'] 
    Th = params['Th'] 

    return A, R, Th, joints

def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3
    parents: 24
    """
    # obtain the relative joints
    bs, joints_num = joints.shape[0:2]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, joints_num, 1, 4]).cuda()
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

    # rotate each part
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # obtain the rigid transformation
    padding = torch.zeros([bs, joints_num, 1]).cuda()
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms

def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros([batch_size, 1], device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'

    plane_features_x = plane_features[:,:,:plane_features.shape[2]//3]
    plane_features_y = plane_features[:,:,plane_features.shape[2]//3:2*plane_features.shape[2]//3]
    plane_features_z = plane_features[:,:,2*plane_features.shape[2]//3:]

    N, n_planes, C, H, W = plane_features_x.shape
    _, M, _ = coordinates.shape
    plane_features_x = plane_features_x.view(N*n_planes, C, H, W)
    plane_features_y = plane_features_y.view(N*n_planes, C, H, W)
    plane_features_z = plane_features_z.view(N*n_planes, C, H, W)
    if box_warp is not None:
        coordinates = 2 * (coordinates - box_warp[:, :1]) / (box_warp[:, 1:2] - box_warp[:, :1]) - 1 # TODO: add specific box bounds
    # coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates_x = project_onto_planes(plane_axes, coordinates).unsqueeze(1) # [3, 1, 786432, 2]
    output_features_x = F.grid_sample(plane_features_x, projected_coordinates_x, mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)

    projected_coordinates_y = projected_coordinates_x.clone()
    projected_coordinates_y[..., 0] = projected_coordinates_y[..., 0] + 1/H
    output_features_y = F.grid_sample(plane_features_y, projected_coordinates_y, mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)

    projected_coordinates_z = projected_coordinates_x.clone()
    projected_coordinates_z[..., 1] = projected_coordinates_z[..., 1] + 1/H
    output_features_z = F.grid_sample(plane_features_z, projected_coordinates_z, mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)

    output_features = torch.cat([output_features_x, output_features_y, output_features_z], dim=-1)

    return output_features

def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

