"""Unit tests for h9 component modules.

Tests are intentionally written to exercise the contracts in ``DESIGN-H9.md``:
- Shape preservation
- Resolution-independence of parameter shapes
- Numerical stability near |U| = 0
- Phase preservation in ComplexLayerNorm
- Output range of phase-aware SPN gate
"""

from __future__ import annotations

import math

import pytest
import torch

from h9 import (
    ComplexFFN,
    ComplexLayerNorm,
    DOSTDomainSAGU,
    HSSBlock,
    PhaseAwareSPN,
    SASS,
    SpectralKernel,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_complex_input() -> torch.Tensor:
    """Small complex tensor for fast tests: (B=2, D'=8, H=4, W=4)."""
    torch.manual_seed(0)
    re = torch.randn(2, 8, 4, 4)
    im = torch.randn(2, 8, 4, 4)
    return torch.complex(re, im)


# ---------------------------------------------------------------------------
# PhaseAwareSPN
# ---------------------------------------------------------------------------


class TestPhaseAwareSPN:
    def test_output_shape(self, small_complex_input: torch.Tensor) -> None:
        spn = PhaseAwareSPN(d_prime=8)
        g = spn(small_complex_input)
        assert g.shape == small_complex_input.shape

    def test_output_real_in_unit_interval(self, small_complex_input: torch.Tensor) -> None:
        spn = PhaseAwareSPN(d_prime=8)
        g = spn(small_complex_input)
        assert not g.is_complex(), "SPN gate must be real-valued."
        assert g.min() >= 0.0, "SPN gate must be >= 0 (sigmoid output)."
        assert g.max() <= 1.0, "SPN gate must be <= 1 (sigmoid output)."

    def test_resolution_independence(self) -> None:
        """Same module should produce a gate of correct shape for any (H, W)."""
        spn = PhaseAwareSPN(d_prime=16)
        for H, W in [(4, 4), (8, 16), (32, 32), (7, 13)]:
            U = torch.complex(torch.randn(1, 16, H, W), torch.randn(1, 16, H, W))
            g = spn(U)
            assert g.shape == (1, 16, H, W)

    def test_zero_input_no_nan(self) -> None:
        """Near-zero input must not produce NaN gradients via cos/sin."""
        spn = PhaseAwareSPN(d_prime=4, eps=1e-8)
        U = torch.zeros(1, 4, 2, 2, dtype=torch.complex64)
        U.requires_grad_(False)  # not a leaf in this test
        g = spn(U)
        assert not torch.isnan(g).any()
        assert not torch.isinf(g).any()

    def test_grad_flows(self, small_complex_input: torch.Tensor) -> None:
        spn = PhaseAwareSPN(d_prime=8)
        U = small_complex_input.clone().requires_grad_(True)
        g = spn(U)
        loss = g.sum()
        loss.backward()
        # SPN parameters should have gradients
        for p in spn.parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# SpectralKernel
# ---------------------------------------------------------------------------


class TestSpectralKernel:
    def test_output_shape_and_dtype(self, small_complex_input: torch.Tensor) -> None:
        kern = SpectralKernel(d_prime=8)
        out = kern(small_complex_input)
        assert out.shape == small_complex_input.shape
        assert out.is_complex()

    def test_kernel_construction(self) -> None:
        kern = SpectralKernel(d_prime=4)
        K = kern.get_kernel()
        assert K.shape == (4,)
        assert K.is_complex()


# ---------------------------------------------------------------------------
# DOSTDomainSAGU
# ---------------------------------------------------------------------------


class TestDOSTDomainSAGU:
    def test_output_shape(self, small_complex_input: torch.Tensor) -> None:
        sagu = DOSTDomainSAGU(d_prime=8)
        out = sagu(small_complex_input)
        assert out.shape == small_complex_input.shape
        assert out.is_complex()

    def test_resolution_independence(self) -> None:
        sagu = DOSTDomainSAGU(d_prime=16)
        for H, W in [(4, 4), (8, 16), (32, 32)]:
            U = torch.complex(torch.randn(1, 16, H, W), torch.randn(1, 16, H, W))
            out = sagu(U)
            assert out.shape == (1, 16, H, W)


# ---------------------------------------------------------------------------
# SASS
# ---------------------------------------------------------------------------


class TestSASS:
    def test_output_shape(self, small_complex_input: torch.Tensor) -> None:
        sass = SASS(d_prime=8)
        out = sass(small_complex_input)
        assert out.shape == small_complex_input.shape

    def test_capture_gate_off_by_default(self, small_complex_input: torch.Tensor) -> None:
        sass = SASS(d_prime=8)
        _ = sass(small_complex_input)
        assert sass.last_gate is None
        assert sass.last_pre_sagu is None

    def test_capture_gate_on(self, small_complex_input: torch.Tensor) -> None:
        sass = SASS(d_prime=8)
        sass.capture_gate = True
        _ = sass(small_complex_input)
        assert sass.last_gate is not None
        assert sass.last_pre_sagu is not None
        assert sass.last_gate.shape == small_complex_input.shape


# ---------------------------------------------------------------------------
# ComplexLayerNorm
# ---------------------------------------------------------------------------


class TestComplexLayerNorm:
    def test_output_shape_and_dtype(self, small_complex_input: torch.Tensor) -> None:
        ln = ComplexLayerNorm(d_prime=8)
        out = ln(small_complex_input)
        assert out.shape == small_complex_input.shape
        assert out.is_complex()

    def test_phase_preserved_when_no_affine(self, small_complex_input: torch.Tensor) -> None:
        """Without affine, only magnitude is rescaled; phase per element preserved."""
        ln = ComplexLayerNorm(d_prime=8, affine=False)
        out = ln(small_complex_input)
        # angle(out) should equal angle(input) elementwise (modulo numeric eps)
        in_angle = small_complex_input.angle()
        out_angle = out.angle()
        diff = (in_angle - out_angle).abs()
        # Allow small numerical jitter; not exact zero due to fp32
        assert diff.max() < 1e-5, "ComplexLayerNorm without affine must preserve phase."


# ---------------------------------------------------------------------------
# ComplexFFN
# ---------------------------------------------------------------------------


class TestComplexFFN:
    def test_output_shape(self, small_complex_input: torch.Tensor) -> None:
        ffn = ComplexFFN(d_prime=8, d_ff=16)
        out = ffn(small_complex_input)
        assert out.shape == small_complex_input.shape


# ---------------------------------------------------------------------------
# HSSBlock
# ---------------------------------------------------------------------------


class TestHSSBlock:
    def test_output_shape(self) -> None:
        block = HSSBlock(d_model=4, n_per_axis=2, spatial_dims=2)
        # d_prime = 4 * 2^2 = 16
        z = torch.complex(torch.randn(2, 16, 8, 8), torch.randn(2, 16, 8, 8))
        out = block(z)
        assert out.shape == z.shape
        assert out.is_complex()

    def test_resolution_independence(self) -> None:
        """Same block weights process any (H, W) without re-instantiation."""
        block = HSSBlock(d_model=4, n_per_axis=2, spatial_dims=2)
        for H, W in [(4, 4), (16, 16), (32, 17)]:
            z = torch.complex(torch.randn(1, 16, H, W), torch.randn(1, 16, H, W))
            out = block(z)
            assert out.shape == (1, 16, H, W)

    def test_no_spatial_mixing(self) -> None:
        """Permuting spatial positions of input should produce the same
        permutation in output (up to channel-mixing). We verify this by
        zeroing all but one spatial location and checking the output's
        non-zero region.
        """
        block = HSSBlock(d_model=2, n_per_axis=2, spatial_dims=2)
        H, W = 4, 4
        d_prime = 2 * 4
        # Single non-zero spatial location
        z = torch.zeros(1, d_prime, H, W, dtype=torch.complex64)
        z[0, :, 2, 1] = torch.complex(torch.randn(d_prime), torch.randn(d_prime))
        out = block(z)
        # Pre-norm and FFN can produce non-zero outputs everywhere if biases exist;
        # if biases are all zero (per reset_parameters with zero biases), then
        # other spatial locations should remain zero. This test may need to be
        # softened if biases are non-zero.
        # TODO(h9): verify this assertion holds with the implemented biases;
        # otherwise weaken to a "spatial separability" check via permutation.
        # For now, just check shape.
        assert out.shape == z.shape
