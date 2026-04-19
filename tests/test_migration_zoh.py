"""Tests for checkpoint migration between approx and zoh discretizations."""

import torch
import pytest

from s9.modules import S9SSMKernel, S9Layer
from s9.rs9_modules import RS9SSMKernel, RS9Layer
from s9.ars9_modules import ARS9SSMKernel, ARS9Layer
from s9.activations.real.thash import ThASh
from s9.activations.complex.stable_modrelu import StableModReLU
from s9.migration.zoh import migrate_state_dict_zoh, migrate_state_dict_from_zoh


def _make_real_act(d, eps, idx):
    return ThASh()


@pytest.mark.parametrize("kernel_cls,layer_cls,gen_act,is_complex", [
    (S9SSMKernel, S9Layer, StableModReLU, True),
    (RS9SSMKernel, RS9Layer, _make_real_act, False),
    (ARS9SSMKernel, ARS9Layer, _make_real_act, False),
])
def test_migration_forward_equivalence(kernel_cls, layer_cls, gen_act, is_complex):
    """approx model forward == migrated-to-zoh model forward."""
    torch.manual_seed(42)
    d_model = 16
    spatial_dims = 2

    # Build an "approx" layer
    layer_approx = layer_cls(
        d_model=d_model, spatial_dims=spatial_dims,
        gen_activation=gen_act, dtype_idx=64, discretization="approx",
    )
    layer_approx.eval()

    # Create test input
    if is_complex:
        x = torch.randn(2, d_model, 8, 8, dtype=torch.complex64)
    else:
        x = torch.randn(2, d_model, 8, 8)

    with torch.no_grad():
        y_approx = layer_approx(x)

    # Migrate state dict: approx → zoh
    sd_approx = layer_approx.state_dict()
    sd_zoh = migrate_state_dict_zoh(sd_approx)

    # Load into a "zoh" layer (default discretization)
    layer_zoh = layer_cls(
        d_model=d_model, spatial_dims=spatial_dims,
        gen_activation=gen_act, dtype_idx=64, discretization="zoh",
    )
    layer_zoh.load_state_dict(sd_zoh)
    layer_zoh.eval()

    with torch.no_grad():
        y_zoh = layer_zoh(x)

    torch.testing.assert_close(y_approx, y_zoh, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kernel_cls,layer_cls,gen_act,is_complex", [
    (S9SSMKernel, S9Layer, StableModReLU, True),
    (RS9SSMKernel, RS9Layer, _make_real_act, False),
    (ARS9SSMKernel, ARS9Layer, _make_real_act, False),
])
def test_migration_roundtrip(kernel_cls, layer_cls, gen_act, is_complex):
    """approx → zoh → approx roundtrip is identity."""
    torch.manual_seed(123)
    d_model = 8

    layer = layer_cls(
        d_model=d_model, spatial_dims=1,
        gen_activation=gen_act, dtype_idx=64, discretization="approx",
    )

    sd_orig = layer.state_dict()
    sd_zoh = migrate_state_dict_zoh(sd_orig)
    sd_back = migrate_state_dict_from_zoh(sd_zoh)

    for key in sd_orig:
        torch.testing.assert_close(sd_orig[key], sd_back[key], atol=1e-6, rtol=1e-6,
                                   msg=f"Roundtrip mismatch on {key}")


def test_taylor_fallback_stability():
    """Very small |A*dt| triggers Taylor fallback without NaN/Inf."""
    k = RS9SSMKernel(4, N=4, discretization="approx")
    with torch.no_grad():
        k.log_dt.fill_(-30.0)  # dt ≈ 1e-13

    sd = k.state_dict()
    sd_migrated = migrate_state_dict_zoh({"kernels.0." + k_: v for k_, v in sd.items()})

    for key, val in sd_migrated.items():
        assert not torch.isnan(val).any(), f"NaN in {key}"
        assert not torch.isinf(val).any(), f"Inf in {key}"
