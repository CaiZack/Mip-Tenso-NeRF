from typing import Optional
import numpy as np
from scipy import integrate
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray_utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map
from pose_utils import to8b

def pdf_2d(x, y):
    return np.exp(-(x**2 + y**2)/2) / (2*np.pi)

def pdf_1d(x):
    return np.exp(-(x**2)/2) / (2*np.pi)

"""
Generate pre calculated gaussian weight for gathering features from plane.

Args:
    tol: tolerate for compute region, n*std_var
    factor: inter-points within one var

Return:
    weight map
    vector relative to the center
"""
def create_standard_2d_gaussian_weight_map(
        tol, factor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    weight = np.zeros([tol*2*factor+1, tol*2*factor+1],  dtype=np.float32)
    x_shift = np.zeros([tol*2*factor+1, tol*2*factor+1],  dtype=np.float32)
    y_shift = np.zeros([tol*2*factor+1, tol*2*factor+1],  dtype=np.float32)

    window_length = 1/factor

    for idx, x in enumerate(np.linspace(-tol, tol, tol*2*factor+1)):
        for idy, y in enumerate(np.linspace(-tol, tol, tol*2*factor+1)):
            x_min, x_max = x - window_length/2, x + window_length/2
            y_min, y_max = y - window_length/2, y + window_length/2
            weight[idx][idy], _ = integrate.dblquad( # type: ignore
                lambda y, x: pdf_2d(x, y),
                x_min, x_max,
                lambda x: y_min, lambda x: y_max
            )
            x_shift[idx][idy] = x
            y_shift[idx][idy] = y
    return weight, x_shift, y_shift

def create_standard_1d_gaussian_weight_map(
        tol, factor) -> tuple[np.ndarray, np.ndarray]:
    weight = np.zeros(tol*2*factor+1, dtype=np.float32)
    x_shift = np.zeros(tol*2*factor+1,  dtype=np.float32)

    window_length = 1/factor

    for idx, x in enumerate(np.linspace(-tol, tol, tol*2*factor+1)):
        x_min, x_max = x - window_length/2, x + window_length/2
        weight[idx], _ = integrate.quad(
                lambda x: pdf_1d(x),
                x_min, x_max
            )
        x_shift[idx] = x

    return weight, x_shift

########## Origin from https://github.com/bebeal/mipnerf-pytorch/blob/main/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None].to(x.device)).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None].to(x.device)**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return x_ret, x_ret

"""
    Multiple-NeRF Model

    Args:
        input_ch: input XYZ channel (fixed 3)

"""
class MultiNeRF(nn.Module):
    def __init__(
            self,
            device: str = 'cpu',
            # Basic NeRF mlp
            input_ch: int = 3,
            input_ch_views: int = 3,
            output_ch: int = 4,
            depth: int = 8,
            layer_ch: int = 256,
            skip: int = 4,
            num_samples: list = [64,128],
            num_levels: int = 2,
            resample_padding: float = 0.01,
            use_viewdir: bool = True,
            use_viewdir_in_first_layer: bool = True,
            white_bkgd: bool = True,
            # Positional Encoder
            use_posenc: bool = True,
            posenc_ch: int = 10,
            use_intergrate_posenc: bool = True,
            use_viewdir_posenc: bool = True,
            viewdir_posenc_ch: int = 4,
            # Tenso Encoder
            use_tenso: bool = True,
            tenso_aabb: torch.Tensor =  torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
            tenso_resolution: int = 256,
            tenso_color_ch: int = 8,
            tenso_app_ch: int = 8,
            tenso_dense_ch: int = 8,
            ipe_tol: int = 3,
            ipe_sampling_factor: int = 2,
            # Forward
            return_raw: bool = False,
        ) -> None:
        super().__init__()

        self.device = device
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.depth = depth
        self.layer_ch = layer_ch
        self.skip = skip
        self.num_samples = num_samples
        self.num_levels = num_levels
        self.resample_padding = resample_padding
        self.use_viewdir = use_viewdir
        self.use_viewdir_in_first_layer = use_viewdir_in_first_layer if use_viewdir else False
        self.use_viewdir_pe = use_viewdir_posenc if use_viewdir else False
        self.viewdir_pe_ch = viewdir_posenc_ch
        self.use_pe = use_posenc
        self.use_ipe = use_intergrate_posenc
        self.pe_ch = posenc_ch
        self.use_tenso = use_tenso
        self.tenso_res = tenso_resolution
        self.tenso_aabb = tenso_aabb.to(self.device)
        self.color_ch = tenso_color_ch
        self.app_ch = tenso_app_ch
        self.dense_ch = tenso_dense_ch
        self.ipe_tol = ipe_tol
        self.ipe_factor = ipe_sampling_factor
        self.white_bkgd = white_bkgd
        self.return_raw = return_raw

        # Check if network need tenso decomposition
        if self.use_tenso:
            # Creating VM decomposition parameters - Vectors
            self.color_vector_x = nn.Parameter(
                torch.randn(1, self.color_ch, self.tenso_res, 1) * 0.1
            ).to(self.device)
            self.color_vector_y = nn.Parameter(
                torch.randn(1, self.color_ch, self.tenso_res, 1) * 0.1
            ).to(self.device)
            self.color_vector_z = nn.Parameter(
                torch.randn(1, self.color_ch, self.tenso_res, 1) * 0.1
            ).to(self.device)
            if not self.dense_ch == 0:
                self.dense_vector_x = nn.Parameter(
                    torch.randn(1, self.dense_ch, self.tenso_res, 1) * 0.1
                ).to(self.device)
                self.dense_vector_y = nn.Parameter(
                    torch.randn(1, self.dense_ch, self.tenso_res, 1) * 0.1
                ).to(self.device)
                self.dense_vector_z = nn.Parameter(
                    torch.randn(1, self.dense_ch, self.tenso_res, 1) * 0.1
                ).to(self.device)
            else:
                self.dense_vector_x = self.color_vector_x
                self.dense_vector_y = self.color_vector_y
                self.dense_vector_z = self.color_vector_z
            # Creating VM decomposition parameters - Matrix
            self.color_plane_yz = nn.Parameter(
                torch.randn(1, self.color_ch, self.tenso_res, self.tenso_res) * 0.1
            ).to(self.device)
            self.color_plane_zx = nn.Parameter(
                torch.randn(1, self.color_ch, self.tenso_res, self.tenso_res) * 0.1
            ).to(self.device)
            self.color_plane_xy = nn.Parameter(
                torch.randn(1, self.color_ch, self.tenso_res, self.tenso_res) * 0.1
            ).to(self.device)
            if not self.dense_ch == 0:
                self.dense_plane_yz = nn.Parameter(
                    torch.randn(1, self.dense_ch, self.tenso_res, self.tenso_res) * 0.1
                ).to(self.device)
                self.dense_plane_zx = nn.Parameter(
                    torch.randn(1, self.dense_ch, self.tenso_res, self.tenso_res) * 0.1
                ).to(self.device)
                self.dense_plane_xy = nn.Parameter(
                    torch.randn(1, self.dense_ch, self.tenso_res, self.tenso_res) * 0.1
                ).to(self.device)
            else:
                self.dense_plane_yz = self.color_plane_yz
                self.dense_plane_zx = self.color_plane_zx
                self.dense_plane_xy = self.color_plane_xy
                self.dense_ch = self.color_ch
            self.color_basis = nn.Linear(3*self.color_ch, self.app_ch, bias=False).to(self.device)
        # Check if network need pe/ipe
        if self.use_pe:
            self.positional_encoding = PositionalEncoding(
                min_deg=0,
                max_deg=self.pe_ch
            ).to(self.device)
            self.pe_ch = self.input_ch * self.pe_ch * 2

            # If using ipe and tenso both, pre compute the gaussian weight to fetching from feature map
            if self.use_ipe and self.use_tenso:
                self.vector_weight, vector_shift = create_standard_1d_gaussian_weight_map(
                    self.ipe_tol, self.ipe_factor
                )
                self.plane_weight, plane_shift_x, plane_shift_y = create_standard_2d_gaussian_weight_map(
                    self.ipe_tol, self.ipe_factor
                )
                self.vector_weight = torch.tensor(self.vector_weight).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(self.device) # [1, 1, N, 1]
                vector_shift = torch.tensor(vector_shift, dtype=torch.float32)
                vector_shift_x, vector_shift_y = torch.meshgrid([vector_shift, torch.tensor(1, dtype=torch.float32)], indexing='ij')
                vector_shift_x = vector_shift_x.unsqueeze(-1)
                vector_shift_y = vector_shift_y.unsqueeze(-1)
                self.vector_shift = torch.cat([vector_shift_x,vector_shift_y], dim=-1).unsqueeze(0).to(self.device) # [1, N, 1, 2]
                self.plane_weight = torch.tensor(self.plane_weight).unsqueeze(0).unsqueeze(0).to(self.device) # [1, 1, Nx, Ny]
                plane_shift_x = torch.tensor(plane_shift_x, dtype=torch.float32)
                plane_shift_y = torch.tensor(plane_shift_y, dtype=torch.float32)
                plane_shift_x = plane_shift_x.unsqueeze(-1)
                plane_shift_y = plane_shift_y.unsqueeze(-1)
                self.plane_shift = torch.cat([plane_shift_x,plane_shift_y], dim=-1).unsqueeze(0).to(self.device) # [1, Nx, Ny, 2]
        # Check if network need viewdir and viewdir_pe
        if self.use_viewdir:
            self.viewdir_ch = 3
            if self.use_viewdir_pe:
                self.viewdir_pe = PositionalEncoding(
                    min_deg=0,
                    max_deg=self.viewdir_pe_ch
                ).to(self.device)
                self.viewdir_ch = self.viewdir_ch * self.viewdir_pe_ch * 2
        # Calculate input channel
        self.mlp_input_ch = 0
        if self.use_tenso:
            self.mlp_input_ch = self.mlp_input_ch + self.app_ch 
        if self.use_pe:
            self.mlp_input_ch = self.mlp_input_ch + self.pe_ch
        if self.use_viewdir_in_first_layer:
            self.mlp_input_ch = self.mlp_input_ch + self.viewdir_ch
        if self.mlp_input_ch == 0:
            self.mlp_input_ch = 3

        # If using regular NeRF, create MLP network
        if not self.use_tenso:
            self.mlp_list = []
            for index in range(self.depth):
                if index == 0:
                    self.mlp_list.append(nn.Sequential(nn.Linear(self.mlp_input_ch, self.layer_ch), nn.ReLU(True)).to(self.device))
                elif index == self.skip:
                    self.mlp_skip_ch = self.layer_ch
                    if self.use_tenso:
                        self.mlp_skip_ch = self.mlp_skip_ch + self.app_ch 
                    if self.use_pe:
                        self.mlp_skip_ch = self.mlp_skip_ch + self.pe_ch
                    if self.use_viewdir_pe:
                        self.mlp_skip_ch = self.mlp_skip_ch + self.viewdir_ch
                    self.mlp_list.append(nn.Sequential(nn.Linear(self.mlp_skip_ch, self.layer_ch), nn.ReLU(True)).to(self.device))
                else:
                    self.mlp_list.append(nn.Sequential(nn.Linear(self.layer_ch, self.layer_ch), nn.ReLU(True)).to(self.device))

        # In tenso, density directly comput by sum
        # In other NeRF, density out by one layer
        if not self.use_tenso:
            if self.use_viewdir:
                self.density_out = nn.Linear(self.layer_ch + self.viewdir_ch, 1).to(self.device)
                self.color_out = nn.Sequential(
                    nn.Linear(self.layer_ch + self.viewdir_ch, self.layer_ch//2),
                    nn.ReLU(),
                    nn.Linear(self.layer_ch // 2, 3),
                ).to(self.device)
            else:
                self.density_out = nn.Linear(self.layer_ch, 1).to(self.device)
                self.color_out = nn.Sequential(
                    nn.Linear(self.layer_ch, self.layer_ch//2),
                    nn.ReLU(),
                    nn.Linear(self.layer_ch // 2, 3),
                ).to(self.device)
        else:
            self.color_input_ch = self.mlp_input_ch
            self.color_out = nn.Sequential(
                nn.Linear(self.color_input_ch, self.layer_ch),
                nn.Linear(self.layer_ch, self.layer_ch),
                nn.Linear(self.layer_ch, 3),
                ).to(self.device)

        self.density_ReLU = nn.ReLU(True)
        self.color_Sigmoid = nn.Sigmoid()

    def _bilinear_grid(self, image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [1, C, H0, W0] 
            grid: [N, H, W, 2(x,y)] 
        
        Returns:
            [N, C, H, W] 插值后的图像
        """
        _, C, H0, W0 = image.shape
        N, H, W, _ = grid.shape

        x = grid[..., 0]
        y = grid[..., 1]
        # Restrict grid position
        x0 = torch.floor(x).long().clamp(0, H0 - 1)
        x1 = (x0 + 1).clamp(0, H0 - 1)
        y0 = torch.floor(y).long().clamp(0, W0 - 1)
        y1 = (y0 + 1).clamp(0, W0 - 1)

        # Weight
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        
        wa = (x1_f - x) * (y1_f - y)
        wb = (x1_f - x) * (y - y0_f)
        wc = (x - x0_f) * (y1_f - y)
        wd = (x - x0_f) * (y - y0_f)

        # Flat image
        image_flat = image.view(C, H0 * W0)  # [C, H0*W0]
        
        # Indexes
        indices_x0y0 = (x0 * W0 + y0).view(N, -1)  # [N, H*W]
        indices_x0y1 = (x0 * W0 + y1).view(N, -1)
        indices_x1y0 = (x1 * W0 + y0).view(N, -1)
        indices_x1y1 = (x1 * W0 + y1).view(N, -1)

        # Allocating result
        result = torch.zeros(N, C, H * W, device=image.device, dtype=image.dtype)

        # Flat weight
        wa = wa.view(N, -1)
        wb = wb.view(N, -1)
        wc = wc.view(N, -1)
        wd = wd.view(N, -1)

        # Calculate with each channel
        for c in range(C):
            channel_data = image_flat[c]  # [H0*W0]
            
            # 直接索引获取值（比gather更快，因为batch size=1）
            v00 = channel_data[indices_x0y0]  # [N, H*W]
            v01 = channel_data[indices_x0y1]
            v10 = channel_data[indices_x1y0]
            v11 = channel_data[indices_x1y1]
            
            # 加权求和
            result[:, c, :] = (v00 * wa + 
                            v01 * wb + 
                            v10 * wc + 
                            v11 * wd)
        
        # 重塑为最终形状 [N, C, H, W]
        return result.view(N, C, H, W)

    def _get_features(self, xyz: torch.Tensor, var: Optional[torch.Tensor] = None, param_type: str = 'density', method: str = 'sum'):
        B = xyz.shape[0]
        # xyz: [B, 3]; var: [B, 3]
        rescale_ratio = (self.tenso_aabb[0,:] - self.tenso_aabb[1,:]) * (self.tenso_res - 1)
        # Remap XYZ and VAR to the Parameter Region
        xyz = (xyz - self.tenso_aabb[0,:]) * rescale_ratio

        coefs = []
        if param_type == 'density':
            vectors = [self.dense_vector_x, self.dense_vector_y, self.dense_vector_z]
            planes = [self.dense_plane_yz, self.dense_plane_zx, self.dense_plane_xy]
        else:
            vectors = [self.color_vector_x, self.color_vector_y, self.color_vector_z]
            planes = [self.color_plane_yz, self.color_plane_zx, self.color_plane_xy]

        for i in range(3):
            mask = torch.ones(3, dtype=torch.bool).to(self.device)
            mask[i] = False
            if var is not None:
                stdvar = torch.sqrt(var) * rescale_ratio
                _, N, Hv, Wv = self.vector_shift.shape
                _, N, Hp, Wp = self.plane_shift.shape


                batch_vector_shift = self.vector_shift.expand(B,N,Hv,Wv)
                batch_vector_weight = self.vector_weight.expand(B,self.dense_ch,N,1) # type: ignore
                batch_plane_shift = self.plane_shift.expand(B,N,Hp,Wp)
                batch_plane_weight = self.plane_weight.expand(B,self.dense_ch,N,N) # type: ignore

                # Vector X and Plane YZ, vector_shift: [1, N, 1, 2]
                query_x = batch_vector_shift * torch.cat([stdvar[:,0].unsqueeze(-1), torch.ones(B,1, device=self.device)], dim=-1).view(B,1,1,2).expand(B,N,1,2) + \
                        torch.cat([xyz[:,0].unsqueeze(-1), torch.ones(B,1, device=self.device)], dim=-1).view(B,1,1,2).expand(B,N,1,2) # shape: [B, N, 1, 2]
                vector_coef = (torch.sum(
                    self._bilinear_grid(
                        vectors[i],
                        query_x
                    ) * batch_vector_weight, 
                    dim=(2,3)
                    ) / N).float() # [B, C]
                query_yz = batch_plane_shift * stdvar[:,mask].view(B,1,1,2).expand(B,N,N,2) + \
                        xyz[:,mask].view(B,1,1,2).expand(B,N,N,2) # shape: [B, Nx, Ny, 2]
                plane_coef = (torch.sum(
                    self._bilinear_grid(
                        planes[i],
                        query_yz,
                        ) * batch_plane_weight,
                    dim=(2,3)
                    ) / (N * N)).float() # [B, C]
            else:
                vector_coef = self._bilinear_grid(
                    vectors[i],
                    torch.cat([xyz[:,i].unsqueeze(-1), torch.ones(B,1, device=self.device)], dim=-1).view(B,1,1,2), # [B, 1, 1, 2]
                ).float().squeeze(-1).squeeze(-1) # [B, C]
                plane_coef = self._bilinear_grid(
                    planes[i],
                    xyz[:,mask].view(B,1,1,2), # [B, 1, 1, 2]
                ).float().squeeze(-1).squeeze(-1) # [B, C]

            coefs.append(vector_coef * plane_coef) # [B, C] [3]
        
        if method == 'sum':
            coef = torch.sum(torch.cat(coefs, dim=-1), dim=1) # [B 3*C] -> [B]
        elif method == 'keep_ch':
            coef = torch.cat(coefs, dim=-1) # [B 3*C]
        else:
            coef = torch.cat(coefs, dim=-1) # [B 3*C]
            coef = self.color_basis(coef) # [B, app_ch]
        return coef.to(self.device)

    def _get_density_features(self, xyz: torch.Tensor, var: Optional[torch.Tensor]):
        return self._get_features(xyz, var, 'density', 'sum')
    
    def _get_color_features(self, xyz: torch.Tensor, var: Optional[torch.Tensor]):
        return self._get_features(xyz, var, 'color', 'multiply')
        
    def forward(self, rays_data: torch.Tensor):

        comp_rgbs = []
        distances = []
        accs = []

        # If batchsize == 1, expand to [B, ro+rd+rgb, 3]
        if len(rays_data) < 2:
            rays_data = rays_data.unsqueeze(0)

        rays_origins =    rays_data[:, 0:3]
        rays_directions = rays_data[:, 3:6]
        rays_radii =      rays_data[:, 9].unsqueeze(-1)
        rays_near =       rays_data[:, 10].unsqueeze(-1)
        rays_far =        rays_data[:, 11].unsqueeze(-1)
        
        B = rays_data.shape[0]
            
        for l in range(self.num_levels):
            # sample
            if l == 0:  # coarse grain sample, mean = xyz var = [var_x, var_y, var_z]
                t_vals, (mean, var) = sample_along_rays(rays_origins, rays_directions, rays_radii, self.num_samples[l],
                                                        rays_near, rays_far, randomized=False, lindisp=False,
                                                        ray_shape='cone')
            else:  # fine grain sample/s
                t_vals, (mean, var) = resample_along_rays(rays_origins, rays_directions, rays_radii, self.num_samples[l],
                                                          t_vals.to(rays_origins.device),
                                                          weights.to(rays_origins.device), randomized=False,
                                                          stop_grad=True, resample_padding=self.resample_padding,
                                                          ray_shape='cone')
            mean = mean.view(B*self.num_samples[l], 3).to(self.device)
            var = var.view(B*self.num_samples[l], 3).to(self.device)

            # If not use IPE, change var to None to avoid it.
            if self.use_pe:
                if not self.use_ipe:
                    var = None
                samples_enc = self.positional_encoding(mean, var)[0] # [B*S, PE_CH]
            else:
                samples_enc = mean # [B*S, 3]

            # Compute view enc if use view_dir
            if self.use_viewdir:
                view_dirs = rays_directions.unsqueeze(1).repeat(1,self.num_samples[l],1).view(B*self.num_samples[l],3)
                if self.use_viewdir_pe:
                    view_enc = self.viewdir_pe(view_dirs, None)[0]
                else:
                    view_enc = view_dirs

            # If use tenso, gather feature using interpolation and weighted intergration
            if self.use_tenso:
                sigma = self._get_density_features(mean, var)
                rgb_feature = self._get_color_features(mean, var)
                samples_enc = torch.cat([samples_enc,rgb_feature], dim=-1) # [B, PE_CH+APP_CH]
                if self.use_viewdir:
                    samples_enc = torch.cat([samples_enc,view_enc], dim=-1) # [B, PE_CH+APP_CH+V_CH]
                rgb = self.color_out(samples_enc)
            # If use regular NeRF
            else:
                if self.use_viewdir_in_first_layer:
                    samples_enc = torch.cat([samples_enc,view_enc], dim=-1) # [B, PE_CH+APP_CH+V_CH]
            
                # Main MLP network
                hidden = self.mlp_list[0](samples_enc)
                for layer in range(1, self.depth):
                    if layer == self.skip:
                        skip_input = torch.cat([hidden, samples_enc], dim=-1)
                        hidden = self.mlp_list[layer](skip_input)
                    else:
                        hidden = self.mlp_list[layer](hidden)
            
                if self.use_viewdir:
                    final_input = torch.cat([hidden, view_enc], dim=-1)
                    sigma = self.density_out(final_input)
                    rgb = self.color_out(final_input)

            sigma = self.density_ReLU(sigma).view(B, self.num_samples[l], 1)
            rgb = self.color_Sigmoid(rgb).view(B, self.num_samples[l], 3)
            comp_rgb, distance, acc, weights, alpha = volumetric_rendering(rgb, sigma, t_vals, rays_directions, self.white_bkgd)
            comp_rgbs.append(comp_rgb)
            distances.append(distance)
            accs.append(acc)

        if self.return_raw:
            raws = torch.cat((torch.clone(rgb).detach(), torch.clone(sigma).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)

if __name__ == "__main__":
    import time
        
    # 测试配置参数
    BATCH_SIZE = 1024  # 批量大小
    NUM_SAMPLES = [64, 128]  # 采样点数
    device = 'mps'
    
    # 创建模型实例
    print("\nCreating MultiNeRF model...")
    model = MultiNeRF(
        device=device,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        depth=4,
        layer_ch=256,
        skip=2,
        num_samples=NUM_SAMPLES,
        num_levels=2,
        resample_padding=0.01,
        use_viewdir=True,
        white_bkgd=True,
        use_posenc=True,
        posenc_ch=10,
        use_intergrate_posenc=True,
        use_viewdir_posenc=True,
        viewdir_posenc_ch=4,
        use_tenso=True,
        tenso_aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        tenso_resolution=128,  # 使用较小分辨率以加快测试
        tenso_color_ch=8,
        tenso_app_ch=8,
        tenso_dense_ch=8,
        ipe_tol=2,  # 使用较小的容忍度以加快测试
        ipe_sampling_factor=2,
        return_raw=True,
    )
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建模拟的rays_data
    # 数据格式: [origins(3), directions(3), viewdirs(3), rgb(3), radius(1), near(1), far(1)]
    print("\nCreating test data...")
    rays_data = torch.randn(BATCH_SIZE, 12).to(device)
    
    # 设置合理的值范围
    rays_data[:, 9] = 0.01  # radius
    rays_data[:, 10] = 0.1  # near
    rays_data[:, 11] = 10.0  # far
    
    # 归一化方向向量
    rays_directions = rays_data[:, 3:6]
    rays_directions = F.normalize(rays_directions, dim=-1)
    rays_data[:, 3:6] = rays_directions
    

    # 测试前向传播
    print("\nTesting forward pass...")
    
    # 计时前向传播
    start_time = time.time()
    
    with torch.no_grad():
        comp_rgbs, distances, accs, raws = model(rays_data)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time

    # 检查输出形状
    print(f"\nForward pass completed in {elapsed_time*1000:.2f} ms")
    print(f"Output shapes:")
    print(f"  comp_rgbs: {comp_rgbs.shape}")  # 应该为 [num_levels, BATCH_SIZE, 3]
    print(f"  distances: {distances.shape}")  # 应该为 [num_levels, BATCH_SIZE]
    print(f"  accs: {accs.shape}")           # 应该为 [num_levels, BATCH_SIZE]
    print(f"  raws: {raws.shape}")           # 应该为 [BATCH_SIZE*num_samples, 4]
    
    # 检查值范围
    print(f"\nValue ranges:")
    print(f"  comp_rgbs: [{comp_rgbs.min():.4f}, {comp_rgbs.max():.4f}] (should be [0, 1] for RGB)")
    print(f"  distances: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"  accs: [{accs.min():.4f}, {accs.max():.4f}] (should be [0, 1])")
    print(f"  raws RGB: [{raws[:, :3].min():.4f}, {raws[:, :3].max():.4f}]")
    print(f"  raws sigma: [{raws[:, 3].min():.4f}, {raws[:, 3].max():.4f}]")


    # 测试梯度计算（训练模式）
    print("\nTesting gradient computation...")
    model.train()
    rays_data.requires_grad_(True)
    
    # 计算损失并反向传播
    comp_rgbs, distances, accs, raws = model(rays_data)
    loss = comp_rgbs.mean()  # 简单的损失函数
    loss.backward()
    
    # 检查梯度
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_norm = param.grad.norm().item()
            if "weight" in name or "bias" in name:
                print(f"  {name}: gradient norm = {grad_norm:.6f}")
            break
    
    if has_gradients:
        print("✓ Gradients computed successfully")
    else:
        print("✗ No gradients computed")
    
    # 测试不同配置
    print("\nTesting different configurations...")
    
    # 测试1: 不使用tenso
    print("\n1. Testing without TensoRF...")
    model = MultiNeRF(
        device=device,
        use_tenso=False,
        use_posenc=True,
        use_intergrate_posenc=False,
        num_samples=[32, 64],  # 使用更少的采样点以加快测试
    )
    
    with torch.no_grad():
        comp_rgbs_nt, distances_nt, accs_nt = model(rays_data)
    print(f"  Output shapes: {comp_rgbs_nt.shape}, {distances_nt.shape}, {accs_nt.shape}")
    
    # 测试2: 不使用位置编码
    print("\n2. Testing without intergrated positional encoding...")
    model = MultiNeRF(
        device=device,
        use_tenso=True,
        use_posenc=True,
        use_intergrate_posenc=False,
        num_samples=[32, 64],
    )
    
    with torch.no_grad():
        comp_rgbs_npe, distances_npe, accs_npe = model(rays_data)
    print(f"  Output shapes: {comp_rgbs_npe.shape}, {distances_npe.shape}, {accs_npe.shape}")
    
    # 测试3: 不使用视角方向
    print("\n3. Testing without view directions...")
    model = MultiNeRF(
        device=device,
        use_viewdir=False,
        use_tenso=True,
        num_samples=[32, 64],
    )
    
    with torch.no_grad():
        comp_rgbs_nv, distances_nv, accs_nv = model(rays_data)
    print(f"  Output shapes: {comp_rgbs_nv.shape}, {distances_nv.shape}, {accs_nv.shape}")
    
    # 内存使用情况（如果可用）
    if torch.cuda.is_available():
        print(f"\nGPU Memory usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    print("\n" + "="*50)
    print("All tests completed successfully! ✓")
    print("="*50)
