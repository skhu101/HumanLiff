from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os, imageio, cv2, time, copy, math, json
from random import sample
from cv2 import Rodrigues as rodrigues

import blobfile as bf

from smpl.smpl_numpy import SMPL
import torch
import torch.nn.functional as F

def load_triplane_data(
    *,
    data_name,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    num_subjects=1000,
    layer_idx=None,
    deterministic=False,
    world_size=1,
    rank=0,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if data_name == 'SynBody':
        dataset = SynBodyDataset(
            image_size,
            data_dir,
            num_subjects,
            layer_idx=layer_idx,
            classes=None,
        )
    elif data_name == 'tightcap':
        dataset = TightCapDataset(
            image_size,
            data_dir,
            num_subjects,
            layer_idx=layer_idx,
            classes=None,
        )        


    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, num_workers=3, drop_last=False, pin_memory=True)
    while True:
        yield from loader

class SynBodyDataset(Dataset):
    def __init__(
        self,
        resolution,
        data_dir,
        num_subjects,
        classes=None,
        layer_idx=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.layer_idx = layer_idx
        ckpt_path = data_dir
        self.num_subjects = num_subjects
        self.layer_num = 4
        print('Reloading from', ckpt_path)

        triplane_ft_dir = os.path.dirname(ckpt_path)
        self.tri_plane_lst = []
        with open(os.path.join(triplane_ft_dir, 'human_list.txt')) as f:
            for line in f.readlines()[0:num_subjects]:
                line = line.strip()
                self.tri_plane_lst.append(os.path.join(triplane_ft_dir, line))

    def __len__(self):
        return self.num_subjects * self.layer_num 

    def __getitem__(self, idx):
        instance_idx = idx // self.layer_num
        layer_idx = idx % self.layer_num 

        if self.layer_idx is not None:
            layer_idx = int(self.layer_idx)

        tri_plane = torch.load(self.tri_plane_lst[instance_idx], map_location='cpu')['network_fn_state_dict']['tri_planes'].squeeze(0)
        tri_plane = tri_plane.reshape(tri_plane.shape[0], -1, *tri_plane.shape[-2:])

        if layer_idx == 0:
            layer_condition = torch.zeros((tri_plane.shape[1], tri_plane.shape[2], tri_plane.shape[3]), dtype=tri_plane.dtype)
        else:
            layer_condition = tri_plane[layer_idx-1]
        out_dict = {}
        out_dict["y"] = np.array(layer_idx, dtype=np.int64)
        return tri_plane[layer_idx], layer_condition, out_dict


class TightCapDataset(Dataset):
    def __init__(
        self,
        resolution,
        data_dir,
        num_subjects,
        classes=None,
        layer_idx=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.layer_idx = layer_idx
        ckpt_path = data_dir
        self.num_subjects = num_subjects
        self.layer_num = 4
        print('Reloading from', ckpt_path)

        triplane_ft_dir = os.path.dirname(ckpt_path)
        self.tri_plane_lst = []
        with open(os.path.join(triplane_ft_dir, 'human_list.txt')) as f:
            for line in f.readlines()[0:num_subjects]:
                line = line.strip()
                self.tri_plane_lst.append(os.path.join(triplane_ft_dir, line))

    def __len__(self):
        return self.num_subjects * self.layer_num

    def __getitem__(self, idx):
        instance_idx = idx // self.layer_num
        layer_idx = idx % self.layer_num

        if self.layer_idx is not None:
            layer_idx = int(self.layer_idx)

        tri_plane = torch.load(self.tri_plane_lst[instance_idx], map_location='cpu')['network_fn_state_dict']['tri_planes'].squeeze(0)
        tri_plane = tri_plane.reshape(tri_plane.shape[0], -1, *tri_plane.shape[-2:])

        if layer_idx == 0:
            layer_condition = torch.zeros((tri_plane.shape[1], tri_plane.shape[2], tri_plane.shape[3]), dtype=tri_plane.dtype)
        else:
            layer_condition = tri_plane[layer_idx-1]
        out_dict = {}
        out_dict["y"] = np.array(layer_idx, dtype=np.int64)
        return tri_plane[layer_idx], layer_condition, out_dict
