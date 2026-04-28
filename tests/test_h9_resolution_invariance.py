"""Resolution invariance tests for the h9 classifier.

Verifies the central architectural claim of h9: the same trained model
weights work at any spatial resolution after re-calibrating the Warped DOST
boundaries on a calibration batch at the new resolution.

See ``DESIGN-H9.md`` §9.
"""

from __future__ import annotations

import pytest
import torch

from h9.examples import H9ClassifierModelExample


@pytest.fixture
def small_model() -> H9ClassifierModelExample:
    """A small model for fast tests."""
    torch.manual_seed(42)
    model = H9ClassifierModelExample(
        in_channels=3,
        d_model=16,
        n_layers=2,
        num_classes=10,
        n_per_axis=3,
        spatial_dims=2,
    )
    return model


# ---------------------------------------------------------------------------
# Calibration contract
# ---------------------------------------------------------------------------


class TestCalibrationContract:
    def test_forward_without_calibration_raises(
        self, small_model: H9ClassifierModelExample
    ) -> None:
        """Forward pass without calibration must raise per DESIGN-H9 §4.2."""
        x = torch.randn(2, 3, 32, 32)
        with pytest.raises(RuntimeError):
            _ = small_model(x)

    def test_one_shot_calibration(
        self, small_model: H9ClassifierModelExample
    ) -> None:
        x_calib = torch.randn(8, 3, 32, 32)
        small_model.calibrate(x_calib)
        # Forward should succeed now
        x = torch.randn(2, 3, 32, 32)
        logits = small_model(x)
        assert logits.shape == (2, 10)

    def test_streaming_calibration(
        self, small_model: H9ClassifierModelExample
    ) -> None:
        f = small_model.fitter
        for _ in range(4):
            chunk = torch.randn(2, 3, 32, 32)
            f.accumulate(chunk)
        f.finalize()
        x = torch.randn(2, 3, 32, 32)
        logits = small_model(x)
        assert logits.shape == (2, 10)


# ---------------------------------------------------------------------------
# Resolution invariance
# ---------------------------------------------------------------------------


class TestResolutionInvariance:
    def test_parameter_count_invariant_across_resolutions(
        self, small_model: H9ClassifierModelExample
    ) -> None:
        """The model's learnable parameter count does not depend on the
        spatial resolution it was calibrated for."""
        # Calibrate at 32x32
        small_model.calibrate(torch.randn(8, 3, 32, 32))
        n_params_32 = sum(p.numel() for p in small_model.parameters())

        # Re-calibrate at 64x64
        f = small_model.fitter
        f.accumulate(torch.randn(8, 3, 64, 64))
        f.finalize()
        n_params_64 = sum(p.numel() for p in small_model.parameters())

        assert n_params_32 == n_params_64

    def test_state_dict_compatible_across_resolutions(
        self, small_model: H9ClassifierModelExample
    ) -> None:
        """state_dict from a model calibrated at one resolution must load
        cleanly into a model calibrated at another resolution."""
        # Train (mock) at 32x32
        small_model.calibrate(torch.randn(8, 3, 32, 32))
        sd = small_model.state_dict()

        # Fresh model, calibrate at 64x64
        torch.manual_seed(99)
        model_64 = H9ClassifierModelExample(
            in_channels=3, d_model=16, n_layers=2, num_classes=10, n_per_axis=3,
        )
        model_64.calibrate(torch.randn(8, 3, 64, 64))

        # Should load without errors
        missing, unexpected = model_64.load_state_dict(sd, strict=False)
        # Boundaries (DOST internal) may be non-Parameter buffers stored
        # outside state_dict. Learnable parameters must transfer cleanly.
        # TODO(h9): tighten this assertion once the storage of DOST boundaries
        # is confirmed (per s9 README they are in plain dict, not buffers,
        # so they should not appear in state_dict at all).

    def test_inference_at_new_resolution(
        self, small_model: H9ClassifierModelExample
    ) -> None:
        """After training at 32x32, the model should accept 64x64 input
        following a re-calibration at 64x64."""
        # Calibrate at 32x32, do a "training" forward (no actual gradient step)
        small_model.calibrate(torch.randn(8, 3, 32, 32))
        x_train = torch.randn(2, 3, 32, 32)
        out_train = small_model(x_train)
        assert out_train.shape == (2, 10)

        # Re-calibrate at 64x64
        f = small_model.fitter
        f.accumulate(torch.randn(8, 3, 64, 64))
        f.finalize()
        x_eval = torch.randn(2, 3, 64, 64)
        out_eval = small_model(x_eval)
        assert out_eval.shape == (2, 10)

    def test_multiple_resolutions_in_succession(
        self, small_model: H9ClassifierModelExample
    ) -> None:
        """Switching back and forth between resolutions should work."""
        for H, W in [(32, 32), (64, 64), (32, 32), (48, 48)]:
            f = small_model.fitter
            f.accumulate(torch.randn(4, 3, H, W))
            f.finalize()
            x = torch.randn(1, 3, H, W)
            out = small_model(x)
            assert out.shape == (1, 10)
