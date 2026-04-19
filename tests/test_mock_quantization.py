"""Tests for quantized SSM layers and quantization utilities."""

import torch
import pytest

from s9.base import FLOAT_DTYPES_DICT, FPDTypeIdx
from s9.quantization import QuantConfig, fake_quant, symmetric_per_tensor_quantize, assert_discrete_stability
from s9.quantization.kernel_cache import QuantizedKernelCache
from s9.qs9_modules import QS9Layer
from s9.qrs9_modules import QRS9Layer
from s9.qars9_modules import QARS9Layer
from s9.activations.complex.stable_modrelu import StableModReLU
from s9.activations.real.thash import ThASh

from . import SPATIAL_SHAPES


def _make_real_act(d, eps, idx):
    return ThASh()


# ---- Quantizer unit tests ---------------------------------------------------


def test_symmetric_quantize_real():
    x = torch.randn(16, 32)
    x_q, scale = symmetric_per_tensor_quantize(x, bits=8)
    assert x_q.shape == x.shape
    assert scale.item() > 0


def test_symmetric_quantize_complex():
    x = torch.randn(16, 32, dtype=torch.complex64)
    x_q, scale = symmetric_per_tensor_quantize(x, bits=8, is_complex=True)
    assert x_q.shape == x.shape
    assert x_q.is_complex()


def test_fake_quant_none_is_noop():
    x = torch.randn(4, 8)
    assert torch.equal(fake_quant(x, None), x)


def test_ste_gradient():
    x = torch.randn(8, requires_grad=True)
    y = fake_quant(x, 8)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---- Stability check tests --------------------------------------------------


def test_stability_passes():
    A_bar = 0.9 * torch.ones(4, 8)
    assert_discrete_stability(A_bar, epsilon=0.05, enforce=True)


def test_stability_fails():
    A_bar = 0.999 * torch.ones(4, 8)
    with pytest.raises(RuntimeError, match="Discrete stability violation"):
        assert_discrete_stability(A_bar, epsilon=0.01, enforce=True)


def test_stability_warns():
    import warnings
    A_bar = 0.999 * torch.ones(4, 8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_discrete_stability(A_bar, epsilon=0.01, enforce=False)
        assert len(w) == 1
        assert "stability violation" in str(w[0].message).lower()


# ---- Kernel cache tests -----------------------------------------------------


def test_kernel_cache_hit():
    cache = QuantizedKernelCache(bits=8)
    k = torch.randn(16, 8, 8)
    q1, s1 = cache.get_or_compute(k, (8, 8))
    q2, s2 = cache.get_or_compute(k, (8, 8))
    assert torch.equal(q1, q2)


def test_kernel_cache_invalidate():
    cache = QuantizedKernelCache(bits=8)
    k = torch.randn(16, 8, 8)
    cache.get_or_compute(k, (8, 8))
    cache.invalidate()
    assert cache._k_quantized is None


# ---- Q Layer smoke tests ----------------------------------------------------


@pytest.mark.parametrize('D', [1, 2])
@pytest.mark.parametrize('dtype_idx', [64])
def test_qs9_layer(D: int, dtype_idx: FPDTypeIdx):
    cfg = QuantConfig(enforce_stability=False)
    spatial_shape = SPATIAL_SHAPES[D - 1]
    d_model = 16
    layer = QS9Layer(d_model, D, StableModReLU, dtype_idx=dtype_idx, quant_config=cfg)
    x = torch.randn([2, d_model] + spatial_shape, dtype=torch.complex64)
    y = layer(x)
    assert y.shape == x.shape


@pytest.mark.parametrize('D', [1, 2])
@pytest.mark.parametrize('dtype_idx', [64])
def test_qrs9_layer(D: int, dtype_idx: FPDTypeIdx):
    cfg = QuantConfig(enforce_stability=False)
    spatial_shape = SPATIAL_SHAPES[D - 1]
    d_model = 16
    layer = QRS9Layer(d_model, D, _make_real_act, dtype_idx=dtype_idx, quant_config=cfg)
    x = torch.randn([2, d_model] + spatial_shape, dtype=FLOAT_DTYPES_DICT[dtype_idx])
    y = layer(x)
    assert y.shape == x.shape
    assert not y.is_complex()


@pytest.mark.parametrize('D', [1, 2])
@pytest.mark.parametrize('dtype_idx', [64])
def test_qars9_layer(D: int, dtype_idx: FPDTypeIdx):
    cfg = QuantConfig(enforce_stability=False)
    spatial_shape = SPATIAL_SHAPES[D - 1]
    d_model = 16
    layer = QARS9Layer(d_model, D, _make_real_act, dtype_idx=dtype_idx, quant_config=cfg)
    x = torch.randn([2, d_model] + spatial_shape, dtype=FLOAT_DTYPES_DICT[dtype_idx])
    y = layer(x)
    assert y.shape == x.shape
    assert not y.is_complex()
