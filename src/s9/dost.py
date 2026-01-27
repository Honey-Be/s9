import torch
import torch.nn as nn

from functools import reduce

from s9.base import NonLearnableProcessorBase

from typing import override

def _get_dyadic_partitions(self, N: int) -> list[tuple[int, int]]:
    """
    Generates indices for dyadic partitioning of the frequency spectrum.
    Fundamental to the Discrete Orthogonal Stockwell Transform.
    """
    partitions: list[tuple[int, int]] = []
    N_eff: int
    if N % 2 != 0:
        # Handle odd size by padding or simplifying (simplified here for robustness)
        N_eff = N - 1
    else:
        N_eff = N
        
    # DC component
    partitions.append((0, 1))
    
    # Positive frequencies dyadic split
    k: int = 1
    while k < N_eff // 2:
        end = min(2 * k, N_eff // 2)
        partitions.append((k, end))
        k *= 2
        
    # Negative frequencies (Nyquist and beyond)
    # Simplified: Treat remaining high freqs as one block or mirror structure
    # For M-DOST in deep learning, we usually focus on the positive spectrum 
    # and let the complex weights handle the rest, but to be strict:
    if k < N_eff:
        partitions.append((k, N_eff))

    return partitions

def _get_mask_index(D: int, d: int, start: int, end: int) -> tuple[slice, ...]:
    index: tuple[slice, ...] = tuple(slice(None) for _ in range(D))
    index[d] = slice(start, end)
    return index

class DOST(NonLearnableProcessorBase):
    @override
    def __init__(self, D: int):
        super().__init__()
        self.D: int = D
    
    @override
    def is_valid_input(self, x: torch.Tensor) -> bool:
        return (x.dtype in [torch.float64, torch.float32, torch.float16, torch.int32, torch.int16, torch.int8]) and (len(x.shape) >= self.D + 2)

    def _convert_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float64 or x.dtype == torch.int32:
            return x.to(torch.complex128)
        elif x.dtype == torch.float32 or x.dtype == torch.int16:
            return x.to(torch.complex64)
        elif x.dtype == torch.float16 or x.dtype == torch.int8:
            return x.to(torch.complex32)
        else:
            raise RuntimeError("Cannot convert to complex dtype :(")
    
    @override
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, tuple (H, W, ...) as S) - Real domain
        Returns:
            z: Output tensor of shape (B, C', S) - Complex domain
               Where C' is expanded by the number of frequency bands.
        """
        
        S: tuple[int, ...] = tuple(x.shape[2:(self.D + 2)])

        dims = tuple(range(2, self.D + 2))

        converted = self._convert_to_complex(x)
        f_x = torch.fft.fftn(converted, dim=dims)

        all_partitions: list[list[tuple[int, int]]] = [_get_dyadic_partitions(s) for s in S]

        sub_bands: list[torch.Tensor] = []

        mask: torch.Tensor = torch.zeros(
            size = S, device = x.device, dtype=converted.dtype
        )

        for (d, partitions) in enumerate(all_partitions):
            for (start, end) in partitions:
                mask.fill_(0.0)
                mask_index = _get_mask_index(self.D, d, start, end)
                mask[mask_index] = 1.0

                band_freq = f_x * mask
                band_time = torch.fft.ifftn(band_freq, dim=dims)
                sub_bands.append(band_time)
        
        out: torch.Tensor = torch.cat(sub_bands, dim = 1)
        return out


class IDOST(NonLearnableProcessorBase):
    @override
    def __init__(self, D: int):
        super().__init__()
        self.D: int = D
    
    @override
    def is_valid_input(self, z: torch.Tensor) -> bool:
        return (z.dtype.is_complex) and (len(z.shape) >= self.D + 2)

    @override
    def transform(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C', tuple (H, W, ...) as S) - Complex domain
               Where C' is expanded by the number of frequency bands.
        Returns:
            z: Output tensor of shape (B, C, S) - Real domain
        """
        
        S: tuple[int, ...] = tuple(z.shape[2:(self.D + 2)])

        dims = tuple(range(2, self.D + 2))

        all_partitions: list[list[tuple[int, int]]] = [_get_dyadic_partitions(s) for s in S]

        total_bands = sum(len(partitions) for partitions in all_partitions)

        B, C_expanded = z.shape[0], z.shape[1]

        if C_expanded % total_bands != 0:
            raise RuntimeError(
                f"Input channel count ({C_expanded}) is not divisible by total DOST bands ({total_bands}). "
                "The input might not be a valid DOST output."
            )

        C_original = C_expanded // total_bands

        recon_size: list[int] = [B, C_original] + list(S)

        f_recon = torch.zeros(
            recon_size, 
            device=z.device, 
            dtype=z.dtype
        )

        current_idx = 0

        # 마스크 캐시 변수
        mask: torch.Tensor = torch.zeros(size=S, device=z.device, dtype=z.dtype)
        
        # 2. 각 밴드별로 분해 -> FFT -> 마스킹 -> 합산
        for d, partitions in enumerate(all_partitions):
            for start, end in partitions:
                # 현재 밴드에 해당하는 부분 추출
                # z의 구조: [Batch, (Band1_C1..Cn, Band2_C1..Cn, ...), D1, D2...]
                # 따라서 C_original 개수만큼 씩 잘라내야 함
                
                # 슬라이싱 범위: current_idx ~ current_idx + C_original
                band_time = z[:, current_idx : current_idx + C_original, ...]
                current_idx = current_idx + C_original
                
                # 시간/공간 영역 밴드를 주파수 영역으로 변환
                band_freq = torch.fft.fftn(band_time, dim=dims)
                
                # 해당 밴드의 위치에만 값을 남기고 나머지는 0으로 만듦 (Masking)
                # DOST에서 해당 영역만 살려서 IFFT 했으므로, 
                # 여기서도 해당 영역만 다시 살려야 원래 위치로 돌아감.
                mask.fill_(0.0) # 마스크 캐시 초기화
                mask_idx = self._get_mask_index(d, start, end)
                mask[mask_idx] = 1.0
                
                # 주파수 스펙트럼 누적
                f_recon = f_recon + (band_freq * mask)
        
        # 3. 최종 IFFT로 원본 신호 복원\
        recon = torch.fft.ifftn(f_recon, dim=dims)
        
        return recon.real