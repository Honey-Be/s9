"""Tests for the dual attribution mechanism.

Verifies:
1. Attribution capture is opt-in and produces tensors of correct shape.
2. Without attribution enabled, training-time forward has no overhead
   (no captured tensors, no extra memory).
3. The three views (spatial, spectral, joint) produce correct shapes.

See ``DESIGN-H9.md`` §8.
"""

from __future__ import annotations

import pytest
import torch

from h9.examples import H9ClassifierModelExample
from h9.attribution import (
    DualAttribution,
    joint_view,
    spatial_view,
    spectral_view,
)
from h9.sass import SASS


@pytest.fixture
def calibrated_small_model() -> H9ClassifierModelExample:
    torch.manual_seed(7)
    model = H9ClassifierModelExample(
        in_channels=3,
        d_model=8,
        n_layers=3,
        num_classes=5,
        n_per_axis=3,
    )
    model.calibrate(torch.randn(4, 3, 32, 32))
    return model


class TestAttributionIntrusion:
    def test_default_no_capture(
        self, calibrated_small_model: H9ClassifierModelExample
    ) -> None:
        """Without calling enable(), no SASS module should capture."""
        x = torch.randn(2, 3, 32, 32)
        _ = calibrated_small_model(x)
        for mod in calibrated_small_model.modules():
            if isinstance(mod, SASS):
                assert not mod.capture_gate
                assert mod.last_gate is None
                assert mod.last_pre_sagu is None

    def test_enable_then_disable_clears(
        self, calibrated_small_model: H9ClassifierModelExample
    ) -> None:
        attr = DualAttribution(calibrated_small_model)
        attr.enable()
        x = torch.randn(2, 3, 32, 32)
        _ = calibrated_small_model(x)
        attr.disable()
        for mod in calibrated_small_model.modules():
            if isinstance(mod, SASS):
                assert not mod.capture_gate
                assert mod.last_gate is None
                assert mod.last_pre_sagu is None


class TestAttributionCapture:
    def test_collect_returns_tensors_per_block(
        self, calibrated_small_model: H9ClassifierModelExample
    ) -> None:
        attr = DualAttribution(calibrated_small_model)
        attr.enable()
        x = torch.randn(2, 3, 32, 32)
        _ = calibrated_small_model(x)
        captured = attr.collect()
        # Should have one entry per HSS block (n_layers=3)
        assert len(captured) == 3
        for name, tensor in captured.items():
            assert "sass" in name.lower()
            # d_prime = 8 * 3^2 = 72
            assert tensor.shape == (2, 72, 32, 32)
            assert not tensor.is_complex(), "Attribution should be |.|^2, real."
            assert (tensor >= 0).all()


class TestAttributionViews:
    def test_spatial_view(self) -> None:
        A = torch.rand(2, 32, 8, 8)
        out = spatial_view(A)
        assert out.shape == (2, 8, 8)
        # Sum check
        torch.testing.assert_close(out, A.sum(dim=1))

    def test_spectral_view(self) -> None:
        A = torch.rand(2, 32, 8, 8)
        out = spectral_view(A)
        assert out.shape == (2, 32)
        torch.testing.assert_close(out, A.sum(dim=(-2, -1)))

    def test_joint_view_shape(self) -> None:
        # d_prime = d * n^2 = 8 * 4 = 32
        A = torch.rand(2, 32, 8, 8)
        out = joint_view(A, n=2, d=8, spatial_dims=2)
        assert out.shape == (2, 2, 2, 8, 8)

    def test_joint_view_dim_mismatch_raises(self) -> None:
        A = torch.rand(2, 31, 8, 8)
        with pytest.raises(ValueError):
            _ = joint_view(A, n=2, d=8, spatial_dims=2)

    def test_joint_view_consistent_with_spectral(self) -> None:
        """Sum of joint over (n, n, H, W) equals total of spectral_view sum."""
        A = torch.rand(2, 32, 8, 8)
        joint = joint_view(A, n=2, d=8, spatial_dims=2)
        # joint sums over original d channels, so total energy equals A's total
        # only if we also sum what is left. Specifically:
        #   joint.sum() == A.sum()   (since reshape + sum-over-d covers all of d_prime)
        torch.testing.assert_close(joint.sum(), A.sum())
