# SPDX-License-Identifier: Apache-2.0
"""MUSA-specific vLLM detection for the DSV4.1 584B geometry."""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
from lmcache.v1.gpu_connector.kv_format.detectors import vllm as vllm_detector
import lmcache.lmcache_native as lmcache_native

pytestmark = pytest.mark.musa


def test_musa_vllm_detects_dsv41_blocked_scale_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUSA routes the validated 584B row to blocked-scale transfer paths."""
    monkeypatch.setattr(vllm_detector, "torch_device_type", "musa")
    kv = [torch.zeros(7, 3, 584, dtype=torch.uint8) for _ in range(5)]

    fmt, out = detect_format(kv, EngineType.VLLM, {"kv_layout": "NHD"})

    assert fmt == lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS
    assert isinstance(out, list)
    assert out[0].shape == (7, 3, 584)
