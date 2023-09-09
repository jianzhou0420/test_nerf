'''
1. Code architecture of this dataset module is inspired by CO-SLAM Available: https://github.com/HengyiWang/Co-SLAM
'''

import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import cv2
from .utils import get_camera_rays


def get_dataset(config):
    '''
    Get the datasets class from the config file.
    '''
    if config['dataset'] == 'replica':
        dataset = ReplicaDataset

    return dataset(config,
                   config['data']['datadir'])


class BaseDataset(Dataset):
    def __init__(self, cfg):
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.H, self.W = cfg['cam']['H'] // cfg['data']['downsample'], \
                         cfg['cam']['W'] // cfg['data']['downsample']

        self.fx, self.fy = cfg['cam']['fx'] // cfg['data']['downsample'], \
                           cfg['cam']['fy'] // cfg['data']['downsample']
        self.cx, self.cy = cfg['cam']['cx'] // cfg['data']['downsample'], \
                           cfg['cam']['cy'] // cfg['data']['downsample']
        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
        self.ignore_w = cfg['tracking']['ignore_edge_W']
        self.ignore_h = cfg['tracking']['ignore_edge_H']

        self.total_pixels = (self.H - self.crop_size * 2) * (self.W - self.crop_size * 2)
        self.num_rays_to_save = int(self.total_pixels * cfg['mapping']['n_pixels'])

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()


class ReplicaDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1,
                 downsample_factor=1, translation=0.0,
                 sc_factor=1., crop=0):
        super(ReplicaDataset, self).__init__(cfg)

        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(glob.glob(f'{self.basedir}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.basedir}/results/depth*.png'))
        self.load_poses(os.path.join(self.basedir, 'traj.txt'))

        self.rays_d = None
        self.tracking_mask = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d,
        }

        return ret

    def load_poses(self, path):
        self.poses = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for i in range(len(self.img_files)):
                singleline = lines[i]
                c2w = np.array(list(map(float, singleline.split()))).reshape(4, 4)
                c2w[:3, 1] *= -1
                c2w[:3, 2] *= -1
                c2w[:3, 3] *= self.sc_factor
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)
