import math
import torch
from typing import Dict, List, Tuple

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from s9.base import NonLearnableProcessorBase

class FastSurfaceletTransform3D(NonLearnableProcessorBase):
    """
    3D Fast Discrete Surfacelet Transform
    3차원 구면 좌표계를 활용한 표면/볼륨 에지 최적화 타일링
    """
    @override
    def __init__(self, scales: int = 3, angles_coarsest: int = 8):
        super().__init__()
        self.scales: int = scales
        self.angles_coarsest: int = angles_coarsest
        self.D: int = 3
        self._mask_cache: Dict[Tuple[int, int, int, torch.device], List[torch.Tensor]] = {}

    @override
    def is_valid_input(self, x: torch.Tensor) -> bool:
        return (not x.dtype.is_complex) and (x.ndim == 5)

    def _get_wrapping_masks(self, D_dim: int, H: int, W: int, device: torch.device) -> List[torch.Tensor]:
        cache_key = (D_dim, H, W, device)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        masks: List[torch.Tensor] = []
        cd, cy, cx = D_dim // 2, H // 2, W // 2
        
        Z, Y, X = torch.meshgrid(
            torch.arange(D_dim, device=device) - cd,
            torch.arange(H, device=device) - cy,
            torch.arange(W, device=device) - cx,
            indexing='ij'
        )
        
        # 3차원 구면 좌표 변환
        R = torch.sqrt(Z**2 + Y**2 + X**2)
        Theta = torch.atan2(Y, X) # 방위각 (Azimuth)
        Phi = torch.acos(torch.clamp(Z / (R + 1e-8), -1.0, 1.0)) # 고도각 (Elevation)

        max_R = float(min(cd, cy, cx))
        
        # 1. 최저 주파수 대역 (중앙 구체)
        r_mask = R < (max_R / (2**(self.scales - 1)))
        masks.append(r_mask.to(torch.complex64))

        # 2. 세부 주파수 대역 및 3D 입체 쐐기 분할
        for s in range(1, self.scales):
            r_inner = max_R / (2**(self.scales - s))
            r_outer = max_R / (2**(self.scales - s - 1))
            num_angles = self.angles_coarsest * (2**(s - 1))
            
            theta_step = 2 * math.pi / num_angles
            phi_step = math.pi / (num_angles // 2 + 1)

            for t in range(num_angles):
                for p in range(num_angles // 2):
                    theta_c = -math.pi + t * theta_step
                    phi_c = p * phi_step
                    
                    theta_diff = torch.abs(torch.atan2(
                        torch.sin(Theta - theta_c), 
                        torch.cos(Theta - theta_c)
                    ))
                    phi_diff = torch.abs(Phi - phi_c)

                    w_mask = (R >= r_inner) & (R < r_outer) & \
                             (theta_diff <= theta_step / 2) & (phi_diff <= phi_step / 2)
                             
                    masks.append(w_mask.to(torch.complex64))

        if len(self._mask_cache) > 8:
            self._mask_cache.pop(next(iter(self._mask_cache)))
        self._mask_cache[cache_key] = masks
        
        return masks

    @override
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D_dim, H, W = x.shape
        x_c = x.to(torch.complex64)

        X_f = torch.fft.fftshift(torch.fft.fftn(x_c, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        masks = self._get_wrapping_masks(D_dim, H, W, x.device)
        
        Z_list: List[torch.Tensor] = []

        for mask in masks:
            band_f = X_f * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            band_f_ishift = torch.fft.ifftshift(band_f, dim=(-3, -2, -1))
            band_spatial = torch.fft.ifftn(band_f_ishift, dim=(-3, -2, -1))
            Z_list.append(band_spatial)

        return torch.cat(Z_list, dim=1)


class InverseFastSurfaceletTransform3D(NonLearnableProcessorBase):
    """
    3D Fast Discrete Surfacelet Transform 역변환
    """
    @override
    def __init__(self, forward_transform_instance: FastSurfaceletTransform3D):
        super().__init__()
        self.forward_t = forward_transform_instance

    @override
    def is_valid_input(self, z: torch.Tensor) -> bool:
        return z.dtype.is_complex and (z.ndim == 5)

    @override
    def transform(self, z: torch.Tensor) -> torch.Tensor:
        B, C_total, D_dim, H, W = z.shape
        masks = self.forward_t._get_wrapping_masks(D_dim, H, W, z.device)
        num_masks = len(masks)

        if C_total % num_masks != 0:
            raise ValueError(f"입력 채널({C_total})이 타일 마스크 수({num_masks})와 일치하지 않습니다.")

        C = C_total // num_masks
        z_reshaped = z.view(B, C, num_masks, D_dim, H, W)

        X_f_recon = torch.zeros((B, C, D_dim, H, W), dtype=torch.complex64, device=z.device)

        for i, mask in enumerate(masks):
            band_spatial = z_reshaped[:, :, i, :, :, :]
            band_f_ishift = torch.fft.fftn(band_spatial, dim=(-3, -2, -1))
            band_f = torch.fft.fftshift(band_f_ishift, dim=(-3, -2, -1))
            X_f_recon += band_f * mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        x_recon = torch.fft.ifftn(torch.fft.ifftshift(X_f_recon, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        
        return x_recon.real