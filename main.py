import os
import time
import numpy as np
import imageio
from PIL import Image
from typing import Callable, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from NeRFModel import *
from NeRFDataset import NeRFDataset, get_rays_from_pose

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
mse2psnr_np = lambda x : -10. * np.log(x) / np.log(10)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class NeRF_Trainer():
    def __init__(
            self,
            output_dir: str,
            model: RayRendering,
            model_name: str = 'NeRF',
            dataset: Optional[NeRFDataset] = None,
            loss_fn: Optional[Callable] = None,
            optimizer: Optional[Optimizer] = None,
            lr_scheduler: Optional[LRScheduler] = None,
            batch_size: int = 1024,
            grad_accu_step: int = 4,
            eval_batch_size: Optional[int] = None,
            max_step: int = 20000,
            eval_step: int = 5000,
            save_step: int = 5000,
            update_step: int = 100,
            device: str = 'cpu',
            ):
        
        self.output_dir = output_dir
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.grad_accu_step = grad_accu_step
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.device = device
        self.max_step = max_step
        self.eval_step = eval_step
        self.save_step = save_step
        self.update_step = update_step
        self.name = model_name

        # Make output folder
        os.makedirs(self.output_dir, exist_ok=True)

        # From dataset create dataloader
        if self.dataset is not None:
            self.dataloader = self.create_dataloader()
            self.data_iter = iter(self.dataloader)

        if self.loss_fn is None:
            self.loss_fn = F.mse_loss

    def create_dataloader(self):
        assert self.dataset is not None, 'Must specify dataset'
        return DataLoader(
                self.dataset,
                self.batch_size,
                shuffle=True,
                drop_last=True,
            )

    def render(self, poses, hwf, K, near, far, step = None, images: Optional[np.ndarray] = None, rand_sel: int = 1):
        # Managing files
        if step is not None:
            self.eval_output_dir = os.path.join(self.output_dir, f'eval_{step}')
            self.eval_log_file = os.path.join(self.eval_output_dir, f'eval_{step}_log.txt')
        else:
            self.eval_output_dir = os.path.join(self.output_dir, f'eval')
            self.eval_log_file = os.path.join(self.eval_output_dir, f'eval_log.txt')
        os.makedirs(self.eval_output_dir, exist_ok=True)
        
        # Whether output per image psnr.
        if images is not None:
            assert poses.shape[0] == images.shape[0], \
            f"Poses and images must be same in evaluation mode, get {poses.shape[0]} and {images.shape[0]}"
            if rand_sel > 0:
                np.random.default_rng(int(time.time() * 1000))
                rand_index = np.random.randint(0, poses.shape[0], rand_sel)
                images = images[rand_index]
                poses = poses[rand_index]

        self.model.eval()
        
        # Collect rays from all poses
        H, W, _ = hwf
        N = poses.shape[0]
        rays_list = []
        for p in poses[:, :3, :4]:
            ray_o, ray_d, radii_val = get_rays_from_pose(p, hwf, K)
            rays_list.append((ray_o, ray_d, radii_val))
        ray_os = np.stack([r[0] for r in rays_list], axis=0)  # [N, H, W, 3]
        ray_ds = np.stack([r[1] for r in rays_list], axis=0)  # [N, H, W, 3]
        radii = np.stack([r[2] for r in rays_list], axis=0)
        H, W = ray_os.shape[1:3]
        N = poses.shape[0]
        rays_data = np.concatenate([
            ray_os.reshape(N, H, W, 3),          # Ro: [N, H, W, 3]
            ray_ds.reshape(N, H, W, 3),          # Rd: [N, H, W, 3]
            np.full((N, H, W, 3), 0),            # RGB: [N, H, W, 3] - Now is 0!!!
            radii.reshape(N, H, W, 1),           # Radii: [N, H, W, 1]
            np.full((N, H, W, 1), near),         # near: [N, H, W, 1]
            np.full((N, H, W, 1), far),          # far: [N, H, W, 1]
            np.full((N, H, W, 1), 1)             # lossmulti: [N, H, W, 1] - Now is 1!!!
        ], axis=-1)                              # Result: [N, H, W, 13]
        rays_data = rays_data.reshape(-1, 13)    # [NumRays, 13]
        rays_data = rays_data.astype(np.float32)
        rays_data = torch.tensor(rays_data).to(self.device)

        # Evaluation and gathering results
        keep_eval = True
        current_idx = 0
        next_idx = 0
        total_idx = rays_data.shape[0]
        result_rgb = torch.zeros(total_idx, 3).to(self.device)
        pbar = tqdm(total=total_idx)
        pbar.set_description_str(f"Eval")

        # Core eval
        eval_start_time = time.time()
        while keep_eval:
            next_idx = current_idx + self.eval_batch_size
            if next_idx >= total_idx:
                next_idx = total_idx
                keep_eval = False
            batch = rays_data[current_idx:next_idx]

            with torch.no_grad():
                comp_rgbs, _, _ = self.model(batch)
            result_rgb[current_idx:next_idx] = comp_rgbs[-1]

            pbar.update(next_idx - current_idx)
            current_idx = next_idx

        eval_time = time.time() - eval_start_time
        with open(self.eval_log_file, '+a') as file:
            file.write(f'Eval time, {eval_time}, \n')
        
        # Split results in to images
        result_rgb = result_rgb.view(N, H, W, 3).cpu().numpy()
        psnr_all = []
        mse_all = []
        imageList = []

        # Save image and calculate image quality
        for idx in range(N):
            image = result_rgb[idx]
            if images is not None:
                target = images[idx]
                mse = float(np.mean(np.square(image - target)))
                mse_all.append(mse)
                psnr = float(mse2psnr_np(mse))
                psnr_all.append(psnr)
                with open(self.eval_log_file, '+a') as file:
                    file.write(f'Frame, {idx}, MSE, {mse}, PSNR, {psnr}, \n')
            image = to8b(image)
            imageList.append(to8b(image))
            freme_name = os.path.join(self.eval_output_dir, f'{idx}.jpg')
            image = Image.fromarray(image).save(freme_name)
        
        # Save video if not test
        if images is None:
            video_name = os.path.join(self.eval_output_dir, f'video.mp4')
            imageio.mimwrite(video_name, imageList, fps=30, quality=8)
        avg_mse = np.mean(mse_all)
        avg_psnr = np.mean(psnr_all)
        with open(self.eval_log_file, '+a') as file:
            file.write(f'Totall, {N}, MSE, {avg_mse}, PSNR, {avg_psnr}, \n')
    
    def train(self):
        
        # Check training component
        if self.dataset is None:
            print('Need dataset.')
            return
        if self.loss_fn is None:
            print('Need loss function.')
            return
        if self.optimizer is None:
            print('Need optimizer.')
            return
        if self.lr_scheduler is None:
            print('Need lr_scheduler.')
            return
        
        # Create dir
        self.train_log_dir = os.path.join(self.output_dir, 'train_log.txt')
        self.ckpt_dir = os.path.join(self.output_dir, 'CKPT')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Add progress bar and start timer
        pbar = tqdm(total=self.max_step)
        pbar.set_description_str(f"Train")
        start_time = time.time()

        # Training iters
        self.optimizer.zero_grad()
        for step in range(self.max_step):

            step_start_time = time.time()
            try:
                batch = next(self.data_iter).to(self.device)
            except:
                self.dataloader = self.create_dataloader()
                self.data_iter = iter(self.dataloader)
                batch = next(self.data_iter).to(self.device)
            
            comp_rgbs, _, _ = self.model(batch)
            loss = self.loss_fn(comp_rgbs, batch[:,6:9].expand(comp_rgbs.shape))
            loss.backward()

            if (step+1) % self.grad_accu_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.lr_scheduler.step()

            psnr = mse2psnr(loss.detach().cpu())
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            train_time = step_end_time - start_time

            # Log training info and update pbar
            if (step+1) % self.update_step == 0:
                with open(self.train_log_dir, '+a') as file:
                    file.write(f"Step, {step}, Loss, {float(loss.detach().cpu())}, PSNR, {float(psnr)}, StepTime, {step_time}, TrainTime, {train_time},\n")
                pbar.set_postfix_str(f'loss: {float(loss.detach().cpu())}')
                pbar.update(self.update_step)

            if (step+1) % self.save_step == 0:
                model_dir = os.path.join(self.ckpt_dir, f'{self.name}-{step}-ckpt.pth')
                torch.save(self.model, model_dir)

            if (step+1) % self.eval_step == 0:
                self.render(
                    self.dataset.test_poses, 
                    self.dataset.hwf, 
                    self.dataset.K, 
                    self.dataset.near, 
                    self.dataset.far, 
                    step, 
                    self.dataset.test_images)
                self.model.train()
        
        self.render(
            np.array(self.dataset.render_poses),
            self.dataset.hwf, 
            self.dataset.K, 
            self.dataset.near, 
            self.dataset.far, 
            step,
            None
        )
        model_dir = os.path.join(self.ckpt_dir, f'{self.name}-Last-ckpt.pth')
        torch.save(self.model, model_dir)
    def load_model(self, ckpt):
        self.model = torch.load(ckpt).to(self.device)

if __name__ == '__main__':

    base_dir = "./data/nerf_synthetic/lego"
    device = 'mps'
    batch_size = 1024
    grad_accu_step = 4
    max_step = 200000 * grad_accu_step
    eval_step = 5000 * grad_accu_step
    update_step = 10
    lr_init = 1e-3
    lr_final = 1e-5
    weight_decay = 1e-5
    output_dir = "./output"

    print('Get dataset')
    dataset = NeRFDataset(
        base_dir=base_dir,
        dataset_type='blender',
        split='train',
        data_type='rays',
        half_res=False,
        white_bkgd=True,
    )

    print('Get model')
    tensorrf_model = TensorRF(
        device=device,
        aabb=dataset.scene_bbox,
        resolution=256,
        dense_ch=8,
        color_ch=8,
        app_ch=27,
        use_ipe_tenso=False,
        use_ipe_mlp=False,
        ipe_tol=3,
        ipe_factor=1,
        position_pe_dim=10,
        viewdir_pe_dim=4,
        mlp_color_feature=128,
    )
    
    print('Get renderer')
    tensorrf_renderer = RayRendering(
        model=tensorrf_model,
        device=device,
        num_samples=[64, 128],
        num_levels=2,
        resample_padding=0.01,
        rgb_padding=0.001,
        white_bkgd=True,
        return_raw=False,
    )

    print('Get optimizer')
    optimizer = torch.optim.AdamW(
        tensorrf_model.parameters(),
        lr=lr_init,
        weight_decay=weight_decay,
    )

    print('Get lr_scheduler')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_step,
    )

    print('Get train code')
    trainer = NeRF_Trainer(
        output_dir=output_dir,
        model=tensorrf_renderer,
        model_name='TensoRF',
        dataset=dataset,
        loss_fn=None,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        batch_size=batch_size,
        grad_accu_step=grad_accu_step,
        eval_batch_size=None,
        max_step=max_step,
        eval_step=eval_step,
        update_step=update_step,
        device=device
    )

    print('Train now')
    trainer.train()
