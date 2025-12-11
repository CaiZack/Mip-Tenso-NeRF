from typing import Optional
import numpy as np
from scipy import integrate
import torch
import torch.nn as nn
import torch.nn.functional as F

########## Origin from https://github.com/bebeal/mipnerf-pytorch/blob/main/model.py
from ray_utils import sample_along_rays, resample_along_rays, volumetric_rendering

"""
    Two gaussian distribution
"""
def pdf_2d(x, y):
    return np.exp(-(x**2 + y**2)/2) / (2*np.pi)

def pdf_1d(x):
    return np.exp(-(x**2)/2) / np.sqrt(2*np.pi)

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
"""
    Positional Encoding with Intergration Implemented
        Do IPE when variance is provided
"""
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
    TensoVMBase Model
        Create VM decomposition parameters
        Get density and color features
    
    If forward function contain position variance, Custom-IPE will be automatically processed.
"""
class TensorVMBase(nn.Module):
    def __init__(
            self,
            device: str = 'cpu',
            aabb: torch.Tensor =  torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
            resolution: int = 256,
            dense_ch: int = 8,
            color_ch: int = 8,
            app_ch: int = 27,
            use_ipe: bool = True,
            ipe_tol: int = 3,
            ipe_factor: int = 2,
        ) -> None:
        super().__init__()

        self.device = device
        self.aabb = aabb.to(self.device)
        self.resolution = resolution
        self.dense_ch = dense_ch
        self.color_ch = color_ch
        self.app_ch = app_ch
        self.use_ipe = use_ipe
        self.ipe_tol = ipe_tol
        self.ipe_factor = ipe_factor

        # Creating VM decomposition parameters - Vectors [C L]
        self.color_vector_x = nn.Parameter(
            torch.randn(self.color_ch, self.resolution) * 0.5
        ).to(self.device)
        self.color_vector_y = nn.Parameter(
            torch.randn(self.color_ch, self.resolution) * 0.5
        ).to(self.device)
        self.color_vector_z = nn.Parameter(
            torch.randn(self.color_ch, self.resolution) * 0.5
        ).to(self.device)
        if not self.dense_ch == 0:
            self.dense_vector_x = nn.Parameter(
                torch.randn(self.dense_ch, self.resolution) * 0.5
            ).to(self.device)
            self.dense_vector_y = nn.Parameter(
                torch.randn(self.dense_ch, self.resolution) * 0.5
            ).to(self.device)
            self.dense_vector_z = nn.Parameter(
                torch.randn(self.dense_ch, self.resolution) * 0.5
            ).to(self.device)
        else:
            self.dense_vector_x = self.color_vector_x
            self.dense_vector_y = self.color_vector_y
            self.dense_vector_z = self.color_vector_z
        # Creating VM decomposition parameters - Matrix [C H W]
        self.color_plane_yz = nn.Parameter(
            torch.randn(self.color_ch, self.resolution, self.resolution) * 0.5
        ).to(self.device)
        self.color_plane_zx = nn.Parameter(
            torch.randn(self.color_ch, self.resolution, self.resolution) * 0.5
        ).to(self.device)
        self.color_plane_xy = nn.Parameter(
            torch.randn(self.color_ch, self.resolution, self.resolution) * 0.5
        ).to(self.device)
        if not self.dense_ch == 0:
            self.dense_plane_yz = nn.Parameter(
                torch.randn(self.dense_ch, self.resolution, self.resolution) * 0.5
            ).to(self.device)
            self.dense_plane_zx = nn.Parameter(
                torch.randn(self.dense_ch, self.resolution, self.resolution) * 0.5
            ).to(self.device)
            self.dense_plane_xy = nn.Parameter(
                torch.randn(self.dense_ch, self.resolution, self.resolution) * 0.5
            ).to(self.device)
        else:
            self.dense_plane_yz = self.color_plane_yz
            self.dense_plane_zx = self.color_plane_zx
            self.dense_plane_xy = self.color_plane_xy
            self.dense_ch = self.color_ch
        self.color_basis = nn.Linear(3*self.color_ch, self.app_ch, bias=False).to(self.device)

        # If using ipe, pre compute the gaussian weight to fetching from feature map
        if self.use_ipe:
            vector_weight, vector_shift = create_standard_1d_gaussian_weight_map(
                self.ipe_tol, self.ipe_factor
            )
            plane_weight, plane_shift_x, plane_shift_y = create_standard_2d_gaussian_weight_map(
                self.ipe_tol, self.ipe_factor
            )
            self.vector_weight = torch.tensor(vector_weight, dtype=torch.float32).to(self.device) # [N]
            self.vector_shift = torch.tensor(vector_shift, dtype=torch.float32).to(self.device) # [N]
            self.plane_weight = torch.tensor(plane_weight).to(self.device) # [Nx, Ny]
            plane_shift_x = torch.tensor(plane_shift_x, dtype=torch.float32)
            plane_shift_y = torch.tensor(plane_shift_y, dtype=torch.float32)
            self.plane_shift = torch.stack([plane_shift_x, plane_shift_y], dim=-1).to(self.device) # [Nx, Ny, 2]
    
    def _bilinear(self, signal: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: [C, L0]  # L0是原始信号长度
            grid: [N]  # x坐标
        
        Returns:
            [N, C]  # 插值后的信号
        """
        # Check grid input
        assert grid.dim() == 1, f"grid shoule be 1-D tensor=, get {grid.dim()}-D."
        
        C, L0 = signal.shape
        N = grid.shape[0]
        
        x = grid  # [N]
        
        x0 = torch.floor(x).long().clamp(0, L0 - 1)  # [N]
        x1 = (x0 + 1).clamp(0, L0 - 1)  # [N]
        
        x0_f = x0.float()
        x1_f = x1.float()
        wa = x1_f - x  # [N] : weight left
        wb = x - x0_f  # [N] : weight right
        
        v0 = signal[:, x0]  # [C, N]
        v1 = signal[:, x1]  # [C, N]
        
        result = (v0 * wa.unsqueeze(0) + v1 * wb.unsqueeze(0)).permute(1, 0)  # [N, C]
        return result

    def _bilinear_grid(self, image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [C, H0, W0] 
            grid: [N, 2(x,y)] 
        
        Returns:
            [N, C] 插值后的图像
        """
        C, H0, W0 = image.shape
        N, _ = grid.shape

        x = grid[..., 0]  # [N]
        y = grid[..., 1]  # [N]
        
        x0 = torch.floor(x).long().clamp(0, H0 - 1)  # [N]
        x1 = (x0 + 1).clamp(0, H0 - 1)  # [N]
        y0 = torch.floor(y).long().clamp(0, W0 - 1)  # [N]
        y1 = (y0 + 1).clamp(0, W0 - 1)  # [N]

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = (x1_f - x) * (y1_f - y)  # [N]
        wb = (x1_f - x) * (y - y0_f)  # [N]
        wc = (x - x0_f) * (y1_f - y)  # [N]
        wd = (x - x0_f) * (y - y0_f)  # [N]

        v00 = image[:, x0, y0]  # [C, N]
        v01 = image[:, x0, y1]  # [C, N]
        v10 = image[:, x1, y0]  # [C, N]
        v11 = image[:, x1, y1]  # [C, N]

        result = (v00 * wa.unsqueeze(0) + 
                v01 * wb.unsqueeze(0) + 
                v10 * wc.unsqueeze(0) + 
                v11 * wd.unsqueeze(0)).permute(1, 0)  # [N, C]
        
        return result

    def _get_features(self, xyz: torch.Tensor, var: Optional[torch.Tensor] = None, param_type: str = 'density', method: str = 'sum'):
        B = xyz.shape[0]
        # xyz: [B, 3]; var: [B, 3]
        rescale_ratio = 1 / (self.aabb[1,:] - self.aabb[0,:]) * (self.resolution - 1)
        # Remap XYZ and VAR to the Parameter Region
        xyz = (xyz - self.aabb[0,:]) * rescale_ratio

        coefs = []
        if param_type == 'density':
            vectors = [self.dense_vector_x, self.dense_vector_y, self.dense_vector_z]
            planes = [self.dense_plane_yz, self.dense_plane_zx, self.dense_plane_xy]
            feature_ch = self.dense_ch
        else:
            vectors = [self.color_vector_x, self.color_vector_y, self.color_vector_z]
            planes = [self.color_plane_yz, self.color_plane_zx, self.color_plane_xy]
            feature_ch = self.color_ch
        
        if var is not None:
            stdvar = torch.sqrt(var) * rescale_ratio
            N, = self.vector_shift.shape
            Nx, Ny, _ = self.plane_shift.shape

            batch_vector_shift = self.vector_shift.view(1,N).expand(B,N) # [B N]
            batch_vector_weight = self.vector_weight.view(1,N,1).expand(B,N,feature_ch) # [B N C]
            batch_plane_shift = self.plane_shift.view(1,Nx*Ny,2).expand(B,Nx*Ny,2) # [B Nx*Ny 2]
            batch_plane_weight = self.plane_weight.view(1,Nx*Ny,1).expand(B,Nx*Ny,feature_ch) # [B Nx*Ny C]

        for i in range(3):
            mask = torch.ones(3, dtype=torch.bool).to(self.device)
            mask[i] = False
            if var is not None:

                # Vector X and Plane YZ
                query_vec = (batch_vector_shift * stdvar[:,i].unsqueeze(-1).expand(B,N)).view(B*N) + \
                          (xyz[:,0].unsqueeze(-1).view(B,1).expand(B,N)).reshape(B*N) # shape: [B, N]
                vector_coef = (torch.sum(
                    self._bilinear(
                        vectors[i], # [C L]
                        query_vec # [B*N]
                    ).view(B,N,feature_ch) * batch_vector_weight, 
                    dim=1) / N) # [B, C]
                
                query_plane = (batch_plane_shift * stdvar[:,mask].view(B,1,2).expand(B,Nx*Ny,2)).view(B*Nx*Ny,2) + \
                        (xyz[:,mask].view(B,1,2).expand(B,Nx*Ny,2)).reshape(B*Nx*Ny,2) # shape: [B*Nx*Ny, 2]
                plane_coef = (torch.sum(
                    self._bilinear_grid(
                        planes[i],
                        query_plane,
                        ).view(B, Nx*Ny, feature_ch) * batch_plane_weight,
                    dim=1) / (Nx * Ny)) # [B, C]
            else:
                vector_coef = self._bilinear(
                    vectors[i],
                    xyz[:,i] # [B]
                ) # [B, C]
                plane_coef = self._bilinear_grid(
                    planes[i],
                    xyz[:,mask], # [B, 2]
                ) # [B, C]

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
        
    def forward(self, xyz: torch.Tensor, var: torch.Tensor):
        if self.use_ipe:
            sigma = self._get_density_features(xyz.to(self.device), var.to(self.device))
            rgb_feature = self._get_color_features(xyz.to(self.device), var.to(self.device))
        else:
            sigma = self._get_density_features(xyz.to(self.device), None)
            rgb_feature = self._get_color_features(xyz.to(self.device), None)
        return sigma, rgb_feature
                
"""
    MLP network for tensoRF with IPE compatible
    If forward function contain position variance, IPE will be automatically processed.
"""
class TensoMLP_PE(nn.Module):
    def __init__(
            self,
            device: str = 'cpu',
            color_in_ch: int = 27,
            feature_ch: int = 128,
            pos_pe_dim: int = 10,
            view_pe_dim: int = 4,
            ) -> None:
        super().__init__()
        
        self.device = device
        self.color_in_ch = color_in_ch
        self.feature_ch = feature_ch
        self.pos_pe_dim = pos_pe_dim
        self.view_pe_dim = view_pe_dim

        self.mlp_input_ch = self.color_in_ch

        if not self.pos_pe_dim == 0:
            self.pos_pe = PositionalEncoding(
                min_deg=0,
                max_deg=self.pos_pe_dim,
                ).to(self.device)
            self.mlp_input_ch = self.mlp_input_ch + 3*2*self.pos_pe_dim
        else:
            self.pos_pe = nn.Identity().to(self.device)
            self.mlp_input_ch = self.mlp_input_ch + 3

        if not self.view_pe_dim == 0:
            self.view_pe = PositionalEncoding(
                min_deg=0,
                max_deg=self.view_pe_dim,
                ).to(self.device)
            self.mlp_input_ch = self.mlp_input_ch + 3*2*self.view_pe_dim
        else:
            self.view_pe = nn.Identity().to(self.device)
            self.mlp_input_ch = self.mlp_input_ch + 3
        
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_ch, self.feature_ch),
            nn.ReLU(True),
            nn.Linear(self.feature_ch, self.feature_ch),
            nn.ReLU(True),
            nn.Linear(self.feature_ch, 3),
            ).to(self.device)
    
    def forward(
            self, 
            color_feature: torch.Tensor,
            viewdir: torch.Tensor,
            position: torch.Tensor,
            position_var: Optional[torch.Tensor] = None,
            ):
        
        pos_enc = self.pos_pe(position.to(self.device), position_var.to(self.device) if position_var is not None else None)[0]
        view_enc = self.view_pe(viewdir.to(self.device), None)[0]
        input = torch.cat([color_feature.to(self.device), pos_enc, view_enc], dim=-1)
        out = self.mlp(input)

        return out

"""
    TensorRF implementation from oritinal paper using VM decomp
    Input: position, variance
    Output: rgb, sigma
    Compatible for volumetric_rendering
"""
class TensorRF(nn.Module):
    def __init__(
            self,
            device: str = 'cpu',
            aabb: torch.Tensor =  torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
            resolution: int = 128,
            dense_ch: int = 8,
            color_ch: int = 8,
            app_ch: int = 27,
            use_ipe_tenso: bool = False,
            use_ipe_mlp: bool = True,
            ipe_tol: int = 3,
            ipe_factor: int = 2,
            position_pe_dim: int = 10,
            viewdir_pe_dim: int = 4,
            mlp_color_feature: int = 128,
        ) -> None:
        super().__init__()
        
        self.device = device
        self.aabb = aabb.to(self.device)
        self.res = resolution
        self.dense_ch = dense_ch
        self.color_ch = color_ch
        self.app_ch = app_ch
        self.use_ipe_tenso = use_ipe_tenso
        self.use_ipe_mlp = use_ipe_mlp
        self.ipe_tol = ipe_tol
        self.ipe_factor = ipe_factor
        self.position_pe_dim = position_pe_dim
        self.viewdir_pe_dim = viewdir_pe_dim
        self.mlp_color_feature = mlp_color_feature

        self.tensoVMModel = TensorVMBase(
            device=self.device,
            aabb=self.aabb,
            resolution=self.res,
            dense_ch=self.dense_ch,
            color_ch=self.color_ch,
            app_ch=self.app_ch,
            use_ipe=self.use_ipe_tenso,
            ipe_tol=self.ipe_tol,
            ipe_factor=self.ipe_factor,
        ).to(self.device)

        self.tensoColorMlp = TensoMLP_PE(
            device=self.device,
            color_in_ch=self.app_ch,
            feature_ch=self.mlp_color_feature,
            pos_pe_dim=self.position_pe_dim,
            view_pe_dim=self.viewdir_pe_dim,
            ).to(self.device)
        
        self.density_ReLU = nn.ReLU(True).to(self.device)
        self.color_Sigmoid = nn.Sigmoid().to(self.device)
        
    def forward(self, viewdir: torch.Tensor, xyz: torch.Tensor, var: torch.Tensor):
        sigma, rgb_feature = self.tensoVMModel(xyz.to(self.device), var.to(self.device) if self.use_ipe_tenso else None)
        rgb = self.tensoColorMlp(rgb_feature, viewdir.to(self.device), xyz.to(self.device), var.to(self.device) if self.use_ipe_mlp else None)
        sigma = self.density_ReLU(sigma)
        rgb = self.color_Sigmoid(rgb)
        return sigma, rgb

"""
    NeRF_MLP
    For normal MLP network for NeRF and Mip-NeRF
"""
class NeRF_MLP(nn.Module):
    def __init__(
            self,
            device: str = 'cpu',
            input_ch: int = 3,
            depth: int = 8,
            skip: list[int] = [4],
            hidden: int = 256,
            ) -> None:
        super().__init__()

        self.device = device
        self.input_ch = input_ch
        self.depth = depth
        self.skip = skip
        self.hidden = hidden

        self.mlp = []
        self.mlp.append(nn.Sequential(
            nn.Linear(self.input_ch, self.hidden),
            nn.ReLU()).to(self.device))
        for layer in range(1, self.depth):
            if layer in self.skip:
                self.mlp.append(nn.Sequential(
                    nn.Linear(self.hidden + self.input_ch, self.hidden),
                    nn.ReLU()).to(self.device))
            else:
                self.mlp.append(nn.Sequential(
                    nn.Linear(self.hidden, self.hidden),
                    nn.ReLU()).to(self.device))

    def forward(self, input_feature):
        hidden = self.mlp[0](input_feature.to(self.device))
        for layer in range(1, self.depth):
            if layer in self.skip:
                hidden_input = torch.cat([hidden, input_feature], dim=-1)
                hidden = self.mlp[layer](hidden_input)
            else:
                hidden = self.mlp[layer](hidden)
        return hidden

"""
    NeRF Network compatible with Mip-NeRF
    If forward function contain position variance, IPE will be automatically processed.
"""
class NeRF_Mip(nn.Module):
    def __init__(
            self,
            device: str = 'cpu',
            pos_pe_dim: int = 10,
            view_pe_dim: int = 4,
            depth: int = 8,
            skip: list[int] = [4],
            hidden: int = 256,
            use_viewdir: bool = True,
            use_ipe: bool = True,
        ) -> None:
        super().__init__()

        self.device = device
        self.pos_pe_dim = pos_pe_dim
        self.view_pe_dim = view_pe_dim
        self.depth = depth
        self.skip = skip
        self.hidden = hidden
        self.use_viewdir = use_viewdir
        self.use_ipe = use_ipe

        self.pos_pe = PositionalEncoding(
            min_deg=0,
            max_deg=self.pos_pe_dim).to(self.device)
        self.mlp = NeRF_MLP(
            device=self.device,
            input_ch=3*2*self.pos_pe_dim,
            depth=8,
            skip=[4],
            hidden=self.hidden).to(self.device)

        self.dense_out = nn.Linear(self.hidden, 1).to(self.device)
        if self.use_viewdir:
            self.view_pe = PositionalEncoding(
                min_deg=0,
                max_deg=self.view_pe_dim).to(self.device)
            self.rgb_out = nn.Sequential(
                nn.Linear(self.hidden+3*2*self.view_pe_dim, self.hidden//2),
                nn.ReLU(),
                nn.Linear(self.hidden // 2, 3)).to(self.device)
        else:
            self.rgb_out = nn.Linear(self.hidden, 3).to(self.device)
        
        self.density_ReLU = nn.ReLU(True).to(self.device)
        self.color_Sigmoid = nn.Sigmoid().to(self.device)
    
    def forward(self, dir: torch.Tensor, xyz: torch.Tensor, var: torch.Tensor):
        pos_enc = self.pos_pe(xyz.to(self.device), var.to(self.device) if self.use_ipe else None)[0]
        mlp_out = self.mlp(pos_enc)

        sigma = self.dense_out(mlp_out)
        if self.use_viewdir:
            view_enc = self.view_pe(dir.to(self.device), None)[0]
            rgb_input = torch.cat([mlp_out, view_enc], dim=-1)
            rgb = self.rgb_out(rgb_input)
        else:
            rgb = self.rgb_out(mlp_out)
        
        sigma = self.density_ReLU(sigma)
        rgb = self.color_Sigmoid(rgb)

        return sigma, rgb
        
"""
    Main Backbone for ray rendering for NeRF
"""
class RayRendering(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            device: str = 'cpu',
            num_samples: list = [64,128],
            num_levels: int = 2,
            resample_padding: float = 0.01,
            rgb_padding: float = 0.001,
            white_bkgd: bool = True,
            return_raw: bool = False,
        ) -> None:
        super().__init__()

        self.device = device
        self.model = model.to(device)
        self.num_samples = num_samples
        self.num_levels = num_levels
        self.resample_padding = resample_padding
        self.rgb_padding = rgb_padding
        self.white_bkgd = white_bkgd
        self.return_raw = return_raw
        
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
            dirs = rays_directions.unsqueeze(1).expand(B,self.num_samples[l],3).reshape(B*self.num_samples[l],3).to(self.device)
            # Run model
            sigma, rgb = self.model(dirs, mean, var)
            # Rendering
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            sigma = sigma.view(B, self.num_samples[l], 1)
            rgb = rgb.view(B, self.num_samples[l], 3)
            comp_rgb, distance, acc, weights, alpha = volumetric_rendering(rgb, sigma, t_vals.to(self.device), rays_directions.to(self.device), self.white_bkgd)
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
    import torch
    import numpy as np
    
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备设置
    device = 'mps'
    print(f"Using device: {device}")
    
    # 测试配置
    batch_size = 2
    num_rays = 4
    num_samples_coarse = 8
    num_samples_fine = 16
    
    # 创建模拟的光线数据
    def create_mock_rays_data(num_rays, device='cpu'):
        """创建模拟的光线数据"""
        rays_data = torch.zeros(num_rays, 12, device=device)
        
        # 光线起点 (origins)
        rays_data[:, 0:3] = torch.randn(num_rays, 3) * 0.1
        
        # 光线方向 (directions) - 归一化
        dirs = torch.randn(num_rays, 3)
        rays_data[:, 3:6] = dirs / torch.norm(dirs, dim=1, keepdim=True)
        
        # 光线颜色 (rgb) - 可选，在训练中使用
        rays_data[:, 6:9] = torch.rand(num_rays, 3)
        
        # 光线半径 (radii) - MipNeRF中使用
        rays_data[:, 9] = torch.rand(num_rays) * 0.01 + 0.001
        
        # 近平面和远平面
        rays_data[:, 10] = torch.rand(num_rays) * 0.5 + 0.5  # near: 0.5-1.0
        rays_data[:, 11] = torch.rand(num_rays) * 2.0 + 3.0  # far: 3.0-5.0
        
        return rays_data
    
    # 创建测试数据
    test_rays = create_mock_rays_data(num_rays, device)
    print(f"Test rays shape: {test_rays.shape}")
    
    # 测试1: TensorRF模型
    print("\n" + "="*60)
    print("Testing TensorRF Model")
    print("="*60)
    
    tensorrf_model = TensorRF(
        device=device,
        aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]),
        resolution=128,
        dense_ch=8,
        color_ch=8,
        app_ch=27,
        use_ipe_tenso=False,
        use_ipe_mlp=False,
        ipe_tol=3,
        ipe_factor=2,
        position_pe_dim=10,
        viewdir_pe_dim=4,
        mlp_color_feature=128,
    )
    
    # 创建渲染器
    tensorrf_renderer = RayRendering(
        model=tensorrf_model,
        device=device,
        num_samples=[num_samples_coarse, num_samples_fine],
        num_levels=2,
        resample_padding=0.01,
        rgb_padding=0.001,
        white_bkgd=True,
        return_raw=True,
    )
    
    # 测试前向传播
    tensorrf_renderer.eval()
    with torch.no_grad():
        comp_rgbs, distances, accs, raws = tensorrf_renderer(test_rays)
        
    print(f"TensorRF Output shapes:")
    print(f"  Coarse RGB: {comp_rgbs[0].shape}")
    print(f"  Fine RGB: {comp_rgbs[1].shape}")
    print(f"  Distances: {distances.shape}")
    print(f"  Accs: {accs.shape}")
    print(f"  Raw outputs: {raws.shape}")
    print(f"  RGB range: [{comp_rgbs[0].min():.3f}, {comp_rgbs[0].max():.3f}]")
    print(f"  Density range: [{raws[:, -1].min():.3f}, {raws[:, -1].max():.3f}]")
    
    # 测试模型直接调用
    test_xyz = torch.randn(10, 3, device=device) * 0.5
    test_var = torch.rand(10, 3, device=device) * 0.01
    test_viewdir = torch.randn(10, 3, device=device)
    test_viewdir = test_viewdir / torch.norm(test_viewdir, dim=1, keepdim=True)
    
    sigma, rgb = tensorrf_model(test_viewdir, test_xyz, test_var)
    print(f"Direct model call:")
    print(f"  Sigma shape: {sigma.shape}, mean: {sigma.mean().item():.4f}")
    print(f"  RGB shape: {rgb.shape}, mean: {rgb.mean().item():.4f}")
    
    print("✓ TensorRF test passed!")
            
    # 测试2: MipNeRF模型
    print("\n" + "="*60)
    print("Testing MipNeRF Model")
    print("="*60)
    
    mipnerf_model = NeRF_Mip(
        device=device,
        pos_pe_dim=10,
        view_pe_dim=4,
        depth=8,
        skip=[4],
        hidden=256,
        use_viewdir=True,
        use_ipe=True,  # MipNeRF使用IPE
    )
    
    # 创建渲染器
    mipnerf_renderer = RayRendering(
        model=mipnerf_model,
        device=device,
        num_samples=[num_samples_coarse, num_samples_fine],
        num_levels=2,
        resample_padding=0.01,
        rgb_padding=0.001,
        white_bkgd=True,
        return_raw=False,
    )
    
    # 测试前向传播
    mipnerf_renderer.eval()
    with torch.no_grad():
        comp_rgbs, distances, accs = mipnerf_renderer(test_rays)
        
    print(f"MipNeRF Output shapes:")
    print(f"  Coarse RGB: {comp_rgbs[0].shape}")
    print(f"  Fine RGB: {comp_rgbs[1].shape}")
    print(f"  Distances: {distances.shape}")
    print(f"  Accs: {accs.shape}")
    print(f"  RGB range: [{comp_rgbs[0].min():.3f}, {comp_rgbs[0].max():.3f}]")
    
    # 测试模型直接调用
    sigma, rgb = mipnerf_model(test_viewdir, test_xyz, test_var)
    print(f"Direct model call:")
    print(f"  Sigma shape: {sigma.shape}, mean: {sigma.mean().item():.4f}")
    print(f"  RGB shape: {rgb.shape}, mean: {rgb.mean().item():.4f}")
    
    print("✓ MipNeRF test passed!")
    
    # 测试3: 原始NeRF模型
    print("\n" + "="*60)
    print("Testing Original NeRF Model")
    print("="*60)
    
    nerf_model = NeRF_Mip(
        device=device,
        pos_pe_dim=10,
        view_pe_dim=4,
        depth=8,
        skip=[4],
        hidden=256,
        use_viewdir=True,
        use_ipe=False,  # 原始NeRF不使用IPE
    )
    
    # 创建渲染器 - 原始NeRF通常只有一层
    nerf_renderer = RayRendering(
        model=nerf_model,
        device=device,
        num_samples=[num_samples_coarse],  # 只有粗采样
        num_levels=1,  # 只有一层
        resample_padding=0.01,
        rgb_padding=0.001,
        white_bkgd=True,
        return_raw=True,
    )
    
    # 测试前向传播
    nerf_renderer.eval()
    with torch.no_grad():
        comp_rgbs, distances, accs, raws = nerf_renderer(test_rays)
        
    print(f"NeRF Output shapes:")
    print(f"  RGB: {comp_rgbs[0].shape}")
    print(f"  Distances: {distances.shape}")
    print(f"  Accs: {accs.shape}")
    print(f"  Raw outputs: {raws.shape}")
    print(f"  RGB range: [{comp_rgbs[0].min():.3f}, {comp_rgbs[0].max():.3f}]")
    
    # 测试模型直接调用（无方差）
    sigma, rgb = nerf_model(test_viewdir, test_xyz, None)
    print(f"Direct model call (no variance):")
    print(f"  Sigma shape: {sigma.shape}, mean: {sigma.mean().item():.4f}")
    print(f"  RGB shape: {rgb.shape}, mean: {rgb.mean().item():.4f}")
    
    print("✓ NeRF test passed!")
        
    # 测试4: 性能基准测试
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)
    
    if torch.cuda.is_available():
        # 创建更大的测试数据
        large_rays = create_mock_rays_data(1024, device)
        
        models_to_test = [
            ("TensorRF", tensorrf_renderer),
            ("MipNeRF", mipnerf_renderer),
            ("NeRF", nerf_renderer),
        ]
        
        for model_name, renderer in models_to_test:
            renderer.eval()
            
            # 预热
            with torch.no_grad():
                _ = renderer(large_rays[:16])
            
            # 性能测试
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record() # type: ignore
            with torch.no_grad():
                for i in range(0, 1024, 64):
                    batch = large_rays[i:i+64]
                    _ = renderer(batch)
            end_time.record() # type: ignore
            torch.cuda.synchronize()
            
            elapsed_time = start_time.elapsed_time(end_time)
            print(f"{model_name}: {elapsed_time:.1f} ms for 1024 rays")
    
    # 测试5: 梯度检查
    print("\n" + "="*60)
    print("Gradient Check")
    print("="*60)
    
    try:
        # 创建一个小的训练场景
        test_model = TensorRF(
            device=device,
            resolution=64,  # 使用较小的分辨率以加快测试
            dense_ch=4,
            color_ch=4,
            app_ch=12,
        )
        
        test_optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        
        # 前向传播
        test_model.train()
        sigma, rgb = test_model(test_viewdir, test_xyz, test_var)
        
        # 创建目标值
        target_sigma = torch.randn_like(sigma) * 0.1 + 0.5
        target_rgb = torch.rand_like(rgb)
        
        # 计算损失和梯度
        loss = loss_fn(sigma, target_sigma) + loss_fn(rgb, target_rgb)
        loss.backward()
        
        # 检查梯度
        has_gradients = False
        for name, param in test_model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        if has_gradients:
            print("✓ Gradients are flowing correctly")
        else:
            print("⚠ No gradients detected - check model architecture")
            
    except Exception as e:
        print(f"✗ Gradient check failed: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)