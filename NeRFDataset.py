import os
from os import path
import json
import numpy as np
import cv2
from PIL import Image
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader

from load_blender import load_blender_data
from load_deepvoxels import load_dv_data
from load_LINEMOD import load_LINEMOD_data
from load_llff import load_llff_data

from ray_utils import Rays

class NeRFDataset(Dataset):
    def __init__(
        self,
        base_dir,
        dataset_type,
        split: str = 'train',
        data_type: str = 'rays',
        # For LLFF data
        factor: int = 8,
        recenter: bool = True,
        bd_factor: float = 0.75,
        spherify: bool = False,
        llffhold: int = 8,
        no_ndc: bool = True,
        # For Blender and LINEMOD data
        half_res: bool = False,
        test_skip: Optional[int] = None,
        white_bkgd: bool = True,
        # For Deepvoxels data
        scene: str = 'cube',
        ) -> None:
        super().__init__()
    
        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.split = split
        self.data_type = data_type

        K = None
        
        # Load llff data
        if self.dataset_type == 'llff':

            self.factor = factor
            self.recenter = recenter
            self.bd_factor = bd_factor
            self.spherify = spherify
            self.llffhold = llffhold
            self.no_ndc = no_ndc
            self.camera_intrinsic_type = 'static'
            self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])

            images, poses, bds, render_poses, i_test = load_llff_data(
                self.base_dir, self.factor, self.recenter, self.bd_factor, self.spherify
            )

            hwf = poses[0,:3,-1]
            poses = poses[:,:3,:4]
            print('Loaded llff', images.shape, render_poses.shape, hwf, self.base_dir)
            if not isinstance(i_test, list):
                i_test = [i_test]

            if self.llffhold > 0:
                print('Auto LLFF holdout,', self.llffhold)
                i_test = np.arange(images.shape[0])[::self.llffhold]

            i_val = i_test
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

            print('DEFINING BOUNDS')
            if self.no_ndc:
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
                
            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)
        
        # Load blender data
        elif self.dataset_type == 'blender':

            self.half_res = half_res
            self.testskip = test_skip if test_skip is not None else 1
            self.white_bkgd = white_bkgd
            self.camera_intrinsic_type = 'static'
            self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])

            images, poses, render_poses, hwf, i_split = load_blender_data(
                self.base_dir, self.half_res, self.testskip)
            print('Loaded blender', images.shape, render_poses.shape, hwf, self.base_dir)
            i_train, i_val, i_test = i_split

            near = 2.
            far = 6.

            if self.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]
        
        else:
            raise ValueError(f"Error: Unknow dataset type {self.dataset_type}")
        
        """ Not implemented yet
        # Load LINEMOD data
        elif self.dataset_type == 'LINEMOD':

            self.half_res = half_res
            self.testskip = test_skip if test_skip is not None else 1
            self.white_bkgd = white_bkgd
            self.camera_intrinsic_type = 'static'

            images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
                self.base_dir, self.half_res, self.testskip)
            print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
            print(f'[CHECK HERE] near: {near}, far: {far}.')
            i_train, i_val, i_test = i_split

            if self.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]

        # Load deepvoxels data
        elif self.dataset_type == 'deepvoxels':

            self.scene = scene
            self.testskip = test_skip if test_skip is not None else 8
            self.camera_intrinsic_type = 'static'

            images, poses, render_poses, hwf, i_split = load_dv_data(
                scene=self.scene, basedir=self.base_dir, testskip=self.testskip)

            print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, self.base_dir)
            i_train, i_val, i_test = i_split

            hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
            near = hemi_R-1.
            far = hemi_R+1.
        """
        
        
        # For future can use different camera captured image to train the model
        if self.camera_intrinsic_type == 'static':
            # Cast intrinsics to right types
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]

            if K is None:
                K = np.array([
                    [focal, 0, 0.5*W],
                    [0, focal, 0.5*H],
                    [0, 0,     1]
                ])
        else:
            pass # Different camera captured image <---- TODO
        
        # Transform data to rays and trained with random selected rays
        if self.data_type == 'rays':
            self.rays_data = self._get_all_rays_rgb(poses, images, near, far, 1, hwf, K, i_train)
        
        # Use random data sequence to train a model <--- TODO
        elif self.data_type == 'images':
            pass

    # Get rays of every pixels in one pose(image)
    def _get_rays_from_pose(self, c2w, hwf, K):
        H, W, _ = hwf
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Normalize ray directions
        rays_d_norm = np.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays_d = rays_d / (rays_d_norm + 1e-10)
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
        # Add radii for Mip-NeRF
        dx = np.sqrt(np.sum((rays_d[:-1, :, :] - rays_d[1:, :, :])**2, axis=-1))
        dy = np.sqrt(np.sum((rays_d[:, :-1, :] - rays_d[:, 1:, :])**2, axis=-1))
        dx = np.pad(dx, ((0, 1), (0, 0)), mode='edge')
        dy = np.pad(dy, ((0, 0), (0, 1)), mode='edge')
        radii = 0.5 * (dx + dy) * (2.0 / np.sqrt(12.0))

        return rays_o, rays_d, radii
    
    # Gather every ray and rgb information
    # rays_ro_rd_rgb: [NumRays, 3 (Ro, Rd, RGB, radii, near, far, lossmulti)]
    # 0-2 Ro: Origin of ray    - x y z position           - real world position (float32)
    # 3-5 Rd: Direction of ray - x y z normalized vector  - directional vector  (float32)
    # 6-8 RGB: Color of pixel  - R G B value              - [0-1] (float32)
    #   9 Radii: cone radii    - radii                    - float32
    #  10 near:
    #  11 far: fixed
    #  12 lossmult: fixed 1 if not given
    def _get_all_rays_rgb(self, poses, images, near, far, lossmulti, hwf, K, i_train):
        if self.camera_intrinsic_type == 'static':
            print('get rays')
            # Collect rays from all poses
            rays_list = []
            for p in poses[:, :3, :4]:
                ray_o, ray_d, radii_val = self._get_rays_from_pose(p, hwf, K)
                rays_list.append((ray_o, ray_d, radii_val))
            
            print('done, concats')
            ray_os = np.stack([r[0] for r in rays_list], axis=0)  # [N, H, W, 3]
            ray_ds = np.stack([r[1] for r in rays_list], axis=0)  # [N, H, W, 3]
            radii = np.stack([r[2] for r in rays_list], axis=0)
            H, W = ray_os.shape[1:3]
            N = poses.shape[0]
            rays_data = np.concatenate([
                ray_os.reshape(N, H, W, 3),          # Ro: [N, H, W, 3]
                ray_ds.reshape(N, H, W, 3),          # Rd: [N, H, W, 3]
                images[:, :, :, None],               # RGB: [N, H, W, 3]
                radii.reshape(N, H, W, 1),           # Radii: [N, H, W, 1]
                np.full((N, H, W, 1), near),         # near: [N, H, W, 1]
                np.full((N, H, W, 1), far),          # far: [N, H, W, 1]
                np.full((N, H, W, 1), lossmulti)     # lossmulti: [N, H, W, 1]
            ], axis=-1)                              # Result: [N, H, W, 13]
            rays_data = rays_data[i_train]
            rays_data = rays_data.reshape(-1, 13)    # [NumRays, 13]
            rays_data = rays_data.astype(np.float32)
            print('shuffle rays')
            np.random.shuffle(rays_data)
        else:
            pass

        return rays_data

    def __len__(self):
        if self.data_type == 'rays':
            return self.rays_data.shape[0]
        elif self.data_type == 'images':
            return 0
    
    def __getitem__(self, index):
        if self.data_type == 'rays':
            return self.rays_data[index]
        elif self.data_type == 'images':
            return 0
