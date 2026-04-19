"""FFT-based N-D convolution helpers for complex and real domains."""

import torch


def fftn_convolve_nd(u: torch.Tensor, k_global: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """Full-complex N-D convolution via FFT.

    Args:
        u: Input tensor ``(B, C, D1, D2, ...)``, complex.
        k_global: Global kernel ``(C, D1, D2, ...)``, complex.
        spatial_dims: Number of spatial dimensions.

    Returns:
        Convolved tensor, same shape as *u*.
    """
    spatial_shapes = u.shape[2:]
    padded_shapes = [size * 2 for size in spatial_shapes]
    fft_dims = tuple(range(-spatial_dims, 0))

    u_f = torch.fft.fftn(u, s=padded_shapes, dim=fft_dims)
    k_f = torch.fft.fftn(k_global, s=padded_shapes, dim=fft_dims)
    y_f = u_f * k_f
    y = torch.fft.ifftn(y_f, s=padded_shapes, dim=fft_dims)

    slices = [slice(None)] * 2 + [slice(0, size) for size in spatial_shapes]
    return y[tuple(slices)]


def rfftn_convolve_nd(u: torch.Tensor, k_global: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """Real-valued N-D convolution via real FFT.

    Args:
        u: Input tensor ``(B, C, D1, D2, ...)``, real.
        k_global: Global kernel ``(C, D1, D2, ...)``, real.
        spatial_dims: Number of spatial dimensions.

    Returns:
        Convolved tensor, same shape as *u*.
    """
    spatial_shapes = u.shape[2:]
    padded_shapes = [size * 2 for size in spatial_shapes]
    fft_dims = tuple(range(-spatial_dims, 0))

    u_f = torch.fft.rfftn(u, s=padded_shapes, dim=fft_dims)
    k_f = torch.fft.rfftn(k_global, s=padded_shapes, dim=fft_dims)
    y_f = u_f * k_f
    y = torch.fft.irfftn(y_f, s=padded_shapes, dim=fft_dims)

    slices = [slice(None)] * 2 + [slice(0, size) for size in spatial_shapes]
    return y[tuple(slices)]
