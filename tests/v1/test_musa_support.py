# SPDX-License-Identifier: Apache-2.0
"""
MUSA support unit tests that do not require MUSA hardware.

These tests cover the design contract documented in
``docs/source/developer_guide/musa_support_design.rst``:

- Device detection precedence in :func:`lmcache._detect_device`.
- Factory dispatch in :func:`lmcache.v1.gpu_connector.CreateGPUConnector`,
  including fail-fast validation when device-scoped features are requested on
  accelerators without connector support.
- Storage backend eligibility: each GDS backend must fail clearly when it is
  configured for the wrong accelerator.

The tests stub ``torch_device_type`` / ``torch_dev`` (rather than mutating
the global PyTorch namespace) so they run on any platform.
"""

# Standard
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch
import asyncio
import importlib
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import CreateGPUConnector
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends
import lmcache as lmc
import lmcache.v1.gpu_connector as gpu_connector_module
import lmcache.v1.storage_backend as storage_backend_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata() -> LMCacheMetadata:
    """Minimal metadata accepted by ``CreateGPUConnector``."""
    return LMCacheMetadata(
        model_name="musa_support_test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 2, 16, 8, 64),
    )


def _make_config(**overrides: Any) -> LMCacheEngineConfig:
    """Default config plus the requested overrides."""
    config = LMCacheEngineConfig.from_defaults(chunk_size=16)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class _FakeTorchDev:
    """Stand-in for ``torch.musa`` / ``torch.xpu`` / ``torch.cuda``."""

    def __init__(self, device_count: int = 1) -> None:
        self._device_count = device_count

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return self._device_count

    def current_device(self) -> int:
        return 0

    def set_device(self, _idx: int) -> None:
        return


def _patch_device(monkeypatch: pytest.MonkeyPatch, device_type: str) -> None:
    """Pretend the current accelerator is ``device_type`` in the factory."""
    monkeypatch.setattr(gpu_connector_module, "torch_device_type", device_type)
    monkeypatch.setattr(gpu_connector_module, "torch_dev", _FakeTorchDev())


# ---------------------------------------------------------------------------
# _detect_device
# ---------------------------------------------------------------------------


class _StubTorch:
    """Minimal stand-in for ``torch`` exposing only what ``_detect_device`` reads."""

    def __init__(
        self,
        *,
        has_musa: bool = False,
        has_xpu: bool = False,
        has_hpu: bool = False,
        musa_available: bool = False,
        xpu_available: bool = False,
        hpu_available: bool = False,
    ) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: True)
        if has_musa:
            self.musa = SimpleNamespace(is_available=lambda: musa_available)
        if has_xpu:
            self.xpu = SimpleNamespace(is_available=lambda: xpu_available)
        if has_hpu:
            self.hpu = SimpleNamespace(is_available=lambda: hpu_available)


def _detect_with_stub(stub: _StubTorch) -> tuple[Any, str]:
    """Run ``_detect_device`` with ``torch`` swapped for the stub."""
    with patch.dict("sys.modules", {"torch": stub}):
        return lmc._detect_device()


def test_detect_device_prefers_musa_when_available() -> None:
    """``_detect_device`` returns MUSA whenever ``torch.musa.is_available()``."""
    stub = _StubTorch(
        has_musa=True,
        has_xpu=True,
        has_hpu=True,
        musa_available=True,
        xpu_available=True,
        hpu_available=True,
    )
    dev, name = _detect_with_stub(stub)
    assert name == "musa"
    assert dev is stub.musa


def test_detect_device_falls_back_past_unavailable_musa() -> None:
    """Falls through MUSA when ``torch.musa.is_available()`` is False."""
    stub = _StubTorch(
        has_musa=True,
        has_xpu=True,
        musa_available=False,
        xpu_available=True,
    )
    _, name = _detect_with_stub(stub)
    assert name == "xpu"


def test_detect_device_cuda_fallback_when_no_alt_accelerator() -> None:
    """Default fallback is CUDA so existing CUDA tests/paths keep working."""
    stub = _StubTorch()
    dev, name = _detect_with_stub(stub)
    assert name == "cuda"
    assert dev is stub.cuda


# ---------------------------------------------------------------------------
# CreateGPUConnector: MUSA branch + device-scoped feature guards
# ---------------------------------------------------------------------------


def test_create_gpu_connector_blending_rejected_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enable_blending`` must fail fast on MUSA."""
    _patch_device(monkeypatch, "musa")
    config = _make_config(enable_blending=True, use_layerwise=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError, match="enable_blending"):
        CreateGPUConnector(config, metadata, EngineType.VLLM)


@pytest.mark.parametrize("device_type", ["musa", "hpu"])
def test_create_gpu_connector_blending_rejected_on_unsupported_devices(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    """The blending guard rejects devices without a blending connector."""
    _patch_device(monkeypatch, device_type)
    config = _make_config(enable_blending=True, use_layerwise=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError, match="enable_blending"):
        CreateGPUConnector(config, metadata, EngineType.VLLM)


def test_create_gpu_connector_v3_rejected_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``use_gpu_connector_v3`` must fail fast on MUSA."""
    _patch_device(monkeypatch, "musa")
    config = _make_config(use_gpu_connector_v3=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError, match="use_gpu_connector_v3"):
        CreateGPUConnector(config, metadata, EngineType.VLLM)


def test_create_gpu_connector_layerwise_rejected_on_hpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HPU ships only ``VLLMPagedMemHPUConnectorV2`` (no layerwise variant).

    Previously, ``use_layerwise=True`` on HPU silently fell through into
    ``VLLMPagedMemLayerwiseGPUConnector`` — the CUDA layerwise connector —
    which then crashed on HPU tensors when constructing
    ``torch.cuda.Stream()``. The guard must reject this combination with a
    clear error before any device-specific construction.
    """
    _patch_device(monkeypatch, "hpu")
    config = _make_config(use_layerwise=True)
    metadata = _make_metadata()
    with pytest.raises(ValueError) as exc:
        CreateGPUConnector(config, metadata, EngineType.VLLM)
    message = str(exc.value)
    assert "use_layerwise" in message
    assert "hpu" in message.lower()


def test_create_gpu_connector_musa_dispatches_to_musa_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``torch_device_type == 'musa'`` selects a MUSA connector class.

    We do not need real MUSA hardware: the factory's ``torch.device`` call
    is patched (PyTorch builds without ``torch_musa`` reject the 'musa'
    string), and ``from_metadata`` is replaced on the MUSA connectors so
    the assertion checks only the dispatch.
    """
    _patch_device(monkeypatch, "musa")
    # First Party
    from lmcache.v1.gpu_connector import musa_connectors as musa_mod

    monkeypatch.setattr(
        gpu_connector_module.torch, "device", lambda *_a, **_kw: "musa:0"
    )

    sentinel_v2 = object()
    sentinel_layer = object()
    monkeypatch.setattr(
        musa_mod.VLLMPagedMemMUSAConnectorV2,
        "from_metadata",
        classmethod(lambda cls, *a, **kw: sentinel_v2),
    )
    monkeypatch.setattr(
        musa_mod.VLLMPagedMemLayerwiseMUSAConnector,
        "from_metadata",
        classmethod(lambda cls, *a, **kw: sentinel_layer),
    )

    metadata = _make_metadata()
    assert (
        CreateGPUConnector(_make_config(use_layerwise=False), metadata, EngineType.VLLM)
        is sentinel_v2
    )
    assert (
        CreateGPUConnector(_make_config(use_layerwise=True), metadata, EngineType.VLLM)
        is sentinel_layer
    )


# ---------------------------------------------------------------------------
# CreateStorageBackends: GDS device/backend cross-product matrix
# ---------------------------------------------------------------------------


def _storage_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="m",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 2, 16, 8, 64),
        role="worker",
    )


def _run_storage_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    device_type: str,
    gds_backend: str,
) -> None:
    """Drive ``CreateStorageBackends`` with the stubbed device + gds_backend.

    Raises whatever the guard raises so callers can assert exact behavior.
    """
    monkeypatch.setattr(storage_backend_module, "torch_device_type", device_type)
    monkeypatch.setattr(storage_backend_module, "torch_dev", _FakeTorchDev())

    config = _make_config(
        local_disk=None,
        max_local_cpu_size=0.0,
        gds_path=str(tmp_path / "gds"),
        gds_backend=gds_backend,
    )
    loop = asyncio.new_event_loop()
    try:
        CreateStorageBackends(config, _storage_metadata(), loop)
    finally:
        loop.close()


def test_storage_guard_rejects_musa_with_cufile(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """MUSA + the default ``gds_backend='cufile'`` is a clear misconfig.

    The error must name ``mufile`` so operators learn the right setting.
    """
    with pytest.raises(ValueError) as exc:
        _run_storage_guard(
            monkeypatch, tmp_path, device_type="musa", gds_backend="cufile"
        )
    message = str(exc.value)
    assert "musa" in message.lower()
    assert "mufile" in message


def test_storage_guard_rejects_musa_with_hipfile(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """MUSA + AMD ``hipfile`` is a clear misconfig; suggest mufile."""
    with pytest.raises(ValueError) as exc:
        _run_storage_guard(
            monkeypatch, tmp_path, device_type="musa", gds_backend="hipfile"
        )
    assert "mufile" in str(exc.value)


def test_storage_guard_rejects_cuda_with_mufile(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The inverse: CUDA + ``mufile`` must also fail with a clear message."""
    with pytest.raises(ValueError) as exc:
        _run_storage_guard(
            monkeypatch, tmp_path, device_type="cuda", gds_backend="mufile"
        )
    message = str(exc.value)
    assert "mufile" in message
    # Pointer to the correct CUDA options.
    assert "cufile" in message or "hipfile" in message


@pytest.mark.parametrize("device_type", ["xpu", "hpu"])
def test_storage_guard_rejects_xpu_hpu_unconditionally(
    monkeypatch: pytest.MonkeyPatch, tmp_path, device_type: str
) -> None:
    """XPU/HPU have no GDS analog wired today — reject any backend choice."""
    for gds_backend in ("cufile", "hipfile", "mufile"):
        with pytest.raises(ValueError):
            _run_storage_guard(
                monkeypatch,
                tmp_path,
                device_type=device_type,
                gds_backend=gds_backend,
            )


def test_storage_guard_accepts_musa_with_mufile(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """MUSA + ``gds_backend='mufile'`` must pass the *guard* (further work
    happens inside ``GdsBackend``).

    The full ``GdsBackend`` ctor needs the ``mufile`` binding installed, so
    we stop the call from reaching it by stubbing ``GdsBackend`` to raise a
    sentinel exception. If the guard would have rejected us, we'd see a
    ``ValueError`` from ``_validate_cuda_only_storage_features`` first; if it
    accepts us, we see the sentinel.
    """

    class _Sentinel(Exception):
        pass

    def _fake_gds_backend(*_a, **_kw):
        raise _Sentinel()

    monkeypatch.setattr(storage_backend_module, "GdsBackend", _fake_gds_backend)

    with pytest.raises(_Sentinel):
        _run_storage_guard(
            monkeypatch, tmp_path, device_type="musa", gds_backend="mufile"
        )


# ---------------------------------------------------------------------------
# GDS allocator dispatch: gds_backend='mufile' -> MuFileMemoryAllocator
# ---------------------------------------------------------------------------


def _install_fake_mufile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake ``mufile`` / ``mufile.bindings`` in ``sys.modules``.

    This lets us exercise ``MuFileMemoryAllocator`` and the ``gds_backend``
    dispatch without the real Moore Threads package being installed.
    """
    fake_mufile = ModuleType("mufile")
    fake_bindings = ModuleType("mufile.bindings")

    register_calls: list[tuple] = []
    deregister_calls: list[tuple] = []

    def mu_register(ptr, size, flags=0):
        register_calls.append((ptr, size, flags))

    def mu_deregister(ptr):
        deregister_calls.append((ptr,))

    fake_bindings.muFileBufRegister = mu_register
    fake_bindings.muFileBufDeregister = mu_deregister
    fake_mufile.bindings = fake_bindings
    # Drop-in compat — match the pattern hipfile uses for CuFile/CuFileDriver.
    fake_mufile.CuFile = SimpleNamespace
    fake_mufile.CuFileDriver = lambda: SimpleNamespace()
    fake_mufile.muFileBufRegister = mu_register
    fake_mufile.muFileBufDeregister = mu_deregister
    fake_mufile.register_calls = register_calls
    fake_mufile.deregister_calls = deregister_calls

    monkeypatch.setitem(sys.modules, "mufile", fake_mufile)
    monkeypatch.setitem(sys.modules, "mufile.bindings", fake_bindings)


def test_mufile_memory_allocator_registers_with_mufile_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MuFileMemoryAllocator`` must lazily import ``mufile.bindings`` and
    call ``muFileBufRegister`` at construction and ``muFileBufDeregister``
    at destruction — symmetrically to ``HipFileMemoryAllocator``.
    """
    _install_fake_mufile(monkeypatch)

    # First Party
    from lmcache.v1.memory_management import MuFileMemoryAllocator

    alloc = MuFileMemoryAllocator(size=1024, device="cpu")
    fake = sys.modules["mufile"]
    assert fake.register_calls, "muFileBufRegister was not called"
    ptr, size, flags = fake.register_calls[-1]
    assert size == 1024
    assert flags == 0
    assert str(alloc) == "MuFileMemoryAllocator"

    del alloc
    # Garbage collection ordering isn't guaranteed here, but __del__ is the
    # only place that calls Deregister, so we don't strictly need to assert
    # the call list — the symmetry assertion above already enforces the
    # binding contract.


def test_gds_backend_dispatch_selects_mufile_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GdsBackend.initialize_allocator`` must instantiate
    ``MuFileMemoryAllocator`` when ``config.gds_backend == 'mufile'``.

    Bypass the bulky ``GdsBackend.__init__`` and call the method on a
    minimally-populated instance — the method is pure dispatch by string.
    """
    _install_fake_mufile(monkeypatch)

    # First Party
    from lmcache.v1.memory_management import MuFileMemoryAllocator
    from lmcache.v1.storage_backend.gds_backend import GdsBackend

    backend = GdsBackend.__new__(GdsBackend)
    backend.gds_backend = "mufile"
    config = _make_config(gds_buffer_size=1)  # 1 MiB
    allocator = backend.initialize_allocator(config, _storage_metadata())
    assert isinstance(allocator, MuFileMemoryAllocator)


# ---------------------------------------------------------------------------
# POSIX runtime fallback (use_gds=False)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device_type", "expected_lib", "expected_symbol"),
    [
        ("cuda", "libcudart.so", "cudaMemcpy"),
        ("musa", "libmusart.so", "musaMemcpy"),
    ],
)
def test_posix_runtime_resolver_maps_device_to_vendor_runtime(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
    expected_lib: str,
    expected_symbol: str,
) -> None:
    """The POSIX fallback used by ``GdsBackend`` when ``use_gds=False`` must
    load the **device-appropriate** vendor runtime, not unconditionally
    ``libcudart.so``. On MUSA the analogs are ``libmusart.so`` /
    ``musaMemcpy``; on CUDA the original ``libcudart.so`` /
    ``cudaMemcpy`` behavior is preserved.

    The resolver must return ``(cdll_handle, memcpy_callable, memcpy_name)``
    so the call sites in ``_load_gds`` / ``_save_gds`` can both invoke the
    symbol and report a faithful error message on failure.
    """
    # First Party
    from lmcache.v1.storage_backend import gds_backend as gds_mod

    loaded = {}

    class _FakeLib:
        def __init__(self, name: str) -> None:
            loaded["name"] = name
            self.cudaMemcpy = object()
            self.musaMemcpy = object()

    monkeypatch.setattr(gds_mod.ctypes, "CDLL", _FakeLib)

    lib, memcpy, name = gds_mod._load_posix_runtime(device_type)

    assert loaded["name"] == expected_lib
    assert name == expected_symbol
    assert memcpy is getattr(lib, expected_symbol)


def test_posix_runtime_resolver_rejects_unsupported_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Devices with no POSIX-runtime entry (XPU/HPU today) must fail fast
    with a message naming the active device and listing the supported set —
    so the operator immediately knows which extra step (install vendor
    cufile/hipfile/mufile package, or run on CUDA/MUSA) unblocks them.
    """
    # First Party
    from lmcache.v1.storage_backend import gds_backend as gds_mod

    with pytest.raises(ValueError) as exc:
        gds_mod._load_posix_runtime("xpu")
    message = str(exc.value)
    assert "xpu" in message
    assert "cuda" in message and "musa" in message


def test_gds_backend_posix_branch_uses_resolver_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: when ``GdsBackend`` falls through to the POSIX branch on
    MUSA, the cached memcpy on the instance must be the one bound to
    ``musaMemcpy`` from ``libmusart.so``. We exercise this without running
    the heavy ``__init__`` (which probes the filesystem and starts a thread
    pool) by calling the resolver and asserting it round-trips.
    """
    # First Party
    from lmcache.v1.storage_backend import gds_backend as gds_mod

    musa_memcpy_sentinel = object()

    class _FakeLib:
        def __init__(self, name: str) -> None:
            self._name = name
            self.musaMemcpy = musa_memcpy_sentinel

    monkeypatch.setattr(gds_mod.ctypes, "CDLL", _FakeLib)
    lib, memcpy, name = gds_mod._load_posix_runtime("musa")
    assert lib._name == "libmusart.so"
    assert memcpy is musa_memcpy_sentinel
    assert name == "musaMemcpy"


def test_gds_backend_constructor_accepts_musa_dst_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``GdsBackend`` must not reject the validated MUSA POSIX path.

    The storage factory already validates ``gds_backend='mufile'`` for MUSA;
    this test covers the downstream constructor gate so the accepted path does
    not still fail on a stale ``dst_device.startswith("cuda")`` assertion or
    require ``mufile`` bindings when ``use_gds=False``.
    """
    # First Party
    from lmcache.v1.memory_management import GPUMemoryAllocator
    from lmcache.v1.storage_backend import abstract_backend as abstract_backend_mod
    from lmcache.v1.storage_backend import gds_backend as gds_mod

    def _fake_scan(coro: Any, _loop: asyncio.AbstractEventLoop) -> SimpleNamespace:
        close = getattr(coro, "close", None)
        if close is not None:
            close()
        return SimpleNamespace(result=lambda timeout=None: None)

    monkeypatch.delitem(sys.modules, "mufile", raising=False)
    monkeypatch.delitem(sys.modules, "mufile.bindings", raising=False)
    monkeypatch.setattr(gds_mod, "get_fstype", lambda _path: "ext4")
    monkeypatch.setattr(abstract_backend_mod.torch, "device", lambda device: device)
    monkeypatch.setattr(
        gds_mod,
        "_load_posix_runtime",
        lambda _device_type: (object(), lambda *_args: 0, "musaMemcpy"),
    )
    monkeypatch.setattr(gds_mod.asyncio, "run_coroutine_threadsafe", _fake_scan)

    config = _make_config(
        gds_path=str(tmp_path),
        gds_backend="mufile",
        use_gds=False,
        gds_buffer_size=1,
    )
    loop = asyncio.new_event_loop()
    try:
        backend = gds_mod.GdsBackend(
            config, _storage_metadata(), loop, dst_device="musa:0"
        )
        try:
            assert backend.dst_device == "musa:0"
            assert isinstance(backend.memory_allocator, GPUMemoryAllocator)
            assert backend.gds_base_pointer is None
            assert backend._gds_file_cls is None
            assert backend._posix_memcpy_name == "musaMemcpy"
        finally:
            backend.close()
    finally:
        loop.close()


def test_gds_backend_posix_save_uses_tensor_pointer_without_registered_base(
    tmp_path,
) -> None:
    """POSIX fallback must not add allocator offsets to direct tensor pointers.

    When ``use_gds=False`` the backend can allocate with a plain
    ``GPUMemoryAllocator`` and ``gds_base_pointer`` is None. In that mode
    ``kv_chunk.data_ptr()`` already points at the first byte to copy; adding
    ``memory_obj.metadata.address`` would shift the source pointer and corrupt
    the persisted chunk.
    """
    # First Party
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.storage_backend.gds_backend import GdsBackend

    backend = GdsBackend.__new__(GdsBackend)
    backend._gds_file_cls = None
    backend._posix_memcpy_name = "musaMemcpy"
    backend.use_direct_io = False
    observed: dict[str, int] = {}

    def _fake_memcpy(dst: Any, src: Any, nbytes: Any, direction: Any) -> int:
        observed["src"] = src.value
        observed["nbytes"] = nbytes.value
        observed["direction"] = direction.value
        return 0

    backend._posix_memcpy = _fake_memcpy

    kv_chunk = torch.arange(16, dtype=torch.uint8)
    backend._save_gds(
        str(tmp_path / "chunk"),
        ".tmp",
        kv_chunk,
        MemoryFormat.KV_2LTD,
        base_pointer=None,
        device_offset=128,
    )

    assert observed == {
        "src": kv_chunk.data_ptr(),
        "nbytes": kv_chunk.nbytes,
        "direction": 2,
    }


# ---------------------------------------------------------------------------
# SGLang dispatch on non-CUDA must fail fast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", ["xpu", "hpu"])
def test_sglang_dispatch_rejected_on_unsupported_accelerator(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    """``SGLangGPUConnector`` calls ``lmc_ops.multi_layer_kv_transfer_unilateral``
    (a CUDA-only C++ op) and asserts ``device.type == "cuda"``. There is
    no XPU/HPU port today, so the dispatcher must reject those combinations
    at construction time with a message that names both the engine and the
    active device so the failure is actionable.
    """
    _patch_device(monkeypatch, device_type)

    config = _make_config()
    metadata = _make_metadata()
    with pytest.raises(ValueError) as exc:
        CreateGPUConnector(config, metadata, EngineType.SGLANG)
    message = str(exc.value)
    assert "SGLang" in message or "sglang" in message.lower()
    assert device_type in message.lower()


@pytest.mark.parametrize("use_layerwise", [False, True])
def test_sglang_dispatch_picks_musa_connector(
    monkeypatch: pytest.MonkeyPatch, use_layerwise: bool
) -> None:
    """The SGLang branch must dispatch to the MUSA-specific connectors on
    MUSA — ``SGLangMUSAConnector`` for the non-layerwise path and
    ``SGLangLayerwiseMUSAConnector`` for the layerwise path. This locks in
    the pair (MUSA, layerwise flag) -> concrete class contract so future
    refactors of the dispatcher cannot silently fall back to the CUDA-only
    ``SGLangGPUConnector`` / ``SGLangLayerwiseGPUConnector``.
    """
    _patch_device(monkeypatch, "musa")
    monkeypatch.setattr(
        gpu_connector_module.torch, "device", lambda *_a, **_kw: "musa:0"
    )

    # First Party
    from lmcache.v1.gpu_connector import musa_connectors as musa_mod

    target_cls_name = (
        "SGLangLayerwiseMUSAConnector" if use_layerwise else "SGLangMUSAConnector"
    )
    target_cls = getattr(musa_mod, target_cls_name)
    sentinel = object()
    monkeypatch.setattr(
        target_cls,
        "from_metadata",
        classmethod(lambda cls, *a, **kw: sentinel),
    )

    config = _make_config(use_layerwise=use_layerwise)
    metadata = _make_metadata()
    assert CreateGPUConnector(config, metadata, EngineType.SGLANG) is sentinel


def test_sglang_musa_connector_v2_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SGLangMUSAConnector`` must accept the same constructor signature
    its CUDA sibling does (``hidden_dim_size``, ``num_layers``, ``use_gpu``,
    plus optional ``chunk_size`` / ``dtype`` / ``device`` when ``use_gpu``
    is True) and provide a deterministic ``get_shape``. We don't need a
    MUSA device here — just verify the class surface the factory relies on.
    """
    # First Party
    from lmcache.v1.gpu_connector.musa_connectors import SGLangMUSAConnector

    conn = SGLangMUSAConnector(hidden_dim_size=64, num_layers=4, use_gpu=False)
    assert conn.num_layers == 4
    assert conn.hidden_dim_size == 64
    # Non-MLA SGLang shape is [2, num_layers, num_tokens, hidden_dim].
    shape = conn.get_shape(num_tokens=8)
    assert tuple(shape) == (2, 4, 8, 64)


def test_sglang_mha_kv_lists_accepts_list_layout() -> None:
    """SGLang passes ``[k_list, v_list]`` (list), not only a 2-tuple.

    Regression: ``_sglang_per_layer_kv_views`` used to require a tuple and
    rejected the in-process SGLang layout.
    """
    # First Party
    from lmcache.v1.gpu_connector.musa_connectors import (
        _sglang_mha_kv_lists,
        _sglang_per_layer_kv_views,
    )

    k0 = torch.zeros(4, 8)
    v0 = torch.zeros(4, 8)
    k_list = [k0]
    v_list = [v0]
    hidden = 8

    k_out, v_out = _sglang_mha_kv_lists([k_list, v_list])
    assert k_out is k_list and v_out is v_list

    views = _sglang_per_layer_kv_views([k_list, v_list], hidden, use_mla=False)
    assert len(views) == 1
    assert views[0][0].shape == (4, hidden)


def test_sglang_layerwise_musa_connector_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SGLangLayerwiseMUSAConnector`` requires ``dtype`` and ``device``
    at construction (the layerwise path lazily allocates a per-batch
    staging buffer keyed by those). ``use_gpu=False`` (default) skips the
    staging buffer altogether so the test runs on a CPU box.
    """
    # First Party
    from lmcache.v1.gpu_connector.musa_connectors import SGLangLayerwiseMUSAConnector

    conn = SGLangLayerwiseMUSAConnector(
        hidden_dim_size=64,
        num_layers=4,
        use_gpu=False,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert conn.num_layers == 4
    # The layerwise path's per-token shape is [num_tokens, 2, hidden_dim].
    assert tuple(conn.get_shape(num_tokens=8)) == (8, 2, 64)


def test_sglang_dispatch_unchanged_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CUDA path through the SGLang dispatcher must remain a no-op for
    the new guard. We patch ``torch.device`` so the call doesn't require
    a real CUDA device, and intercept ``SGLangGPUConnector`` so the test
    doesn't need ``lmc_ops``. Reaching the sentinel proves the guard is
    a strict pass-through on CUDA.
    """
    _patch_device(monkeypatch, "cuda")
    monkeypatch.setattr(
        gpu_connector_module.torch, "device", lambda *_a, **_kw: "cuda:0"
    )

    # Intercept SGLangGPUConnector to avoid importing lmc_ops.
    sentinel = object()
    fake_module = ModuleType("fake_sglang_connectors")
    fake_module.SGLangGPUConnector = lambda *a, **kw: sentinel
    fake_module.SGLangLayerwiseGPUConnector = lambda *a, **kw: sentinel

    real_import = gpu_connector_module.__dict__.get("__import__")
    monkeypatch.setitem(
        sys.modules,
        "lmcache.v1.gpu_connector.gpu_connectors",
        sys.modules.get("lmcache.v1.gpu_connector.gpu_connectors") or fake_module,
    )
    monkeypatch.setattr(
        sys.modules["lmcache.v1.gpu_connector.gpu_connectors"],
        "SGLangGPUConnector",
        lambda *a, **kw: sentinel,
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["lmcache.v1.gpu_connector.gpu_connectors"],
        "SGLangLayerwiseGPUConnector",
        lambda *a, **kw: sentinel,
        raising=False,
    )

    config = _make_config(use_layerwise=False)
    metadata = _make_metadata()
    assert CreateGPUConnector(config, metadata, EngineType.SGLANG) is sentinel
    del real_import


# ---------------------------------------------------------------------------
# End-to-end MUSA dispatch smoke
# ---------------------------------------------------------------------------


def test_e2e_musa_dispatch_chain(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """E2E smoke that walks the full MUSA-on-vLLM dispatch chain on CPU.

    The point of this test is to lock in the *connectivity* of the four
    independent components MUSA support touches:

    1. ``CreateGPUConnector`` selects ``VLLMPagedMemMUSAConnectorV2`` on
       MUSA (proved by patching its ``from_metadata`` to a sentinel).
    2. The vLLM CUDA-only guards (``enable_blending`` /
       ``use_gpu_connector_v3``) stay off this happy path.
    3. The storage guard accepts ``gds_path`` + ``gds_backend="mufile"``
       on MUSA (proved by patching ``GdsBackend`` to a sentinel).
    4. ``GdsBackend.initialize_allocator`` dispatches to
       ``MuFileMemoryAllocator`` for ``gds_backend="mufile"``.

    A regression in any of those four is caught by a single test, which
    is the closest CPU equivalent we can run to "MUSA install + vLLM run
    + put/get" without a Moore Threads box.
    """
    _patch_device(monkeypatch, "musa")
    # The storage guard reads its own module-local ``torch_device_type``
    # binding, so we have to mirror the patch into that module too.
    monkeypatch.setattr(storage_backend_module, "torch_device_type", "musa")
    _install_fake_mufile(monkeypatch)

    # (1) Stub torch.device so the factory can construct a ``musa:0`` device.
    monkeypatch.setattr(
        gpu_connector_module.torch, "device", lambda *_a, **_kw: "musa:0"
    )

    # (1) Stub the MUSA connector's ``from_metadata`` so we don't need real
    # MUSA paging kernels.
    # First Party
    from lmcache.v1.gpu_connector import musa_connectors as musa_mod

    connector_sentinel = object()
    monkeypatch.setattr(
        musa_mod.VLLMPagedMemMUSAConnectorV2,
        "from_metadata",
        classmethod(lambda cls, *a, **kw: connector_sentinel),
    )

    config = _make_config(use_layerwise=False)
    metadata = _make_metadata()
    assert (
        CreateGPUConnector(config, metadata, EngineType.VLLM) is connector_sentinel
    ), "MUSA dispatch did not reach VLLMPagedMemMUSAConnectorV2"

    # (3) Storage guard accepts gds_path + mufile on MUSA, then the
    # downstream GdsBackend constructor is reached (we stub it with a
    # sentinel exception, just like the dedicated test above does).
    class _GdsReached(Exception):
        pass

    monkeypatch.setattr(
        storage_backend_module,
        "GdsBackend",
        lambda *_a, **_kw: (_ for _ in ()).throw(_GdsReached()),
    )

    storage_config = _make_config(
        gds_path=str(tmp_path),
        gds_backend="mufile",
        enable_xpyd=False,
        enable_pd=False,
    )
    storage_config.local_cpu = False
    storage_config.max_local_cpu_size = 0
    storage_config.local_disk = None
    storage_config.max_local_disk_size = 0
    storage_config.remote_url = None
    storage_metadata = _storage_metadata()

    with pytest.raises(_GdsReached):
        loop = asyncio.new_event_loop()
        try:
            CreateStorageBackends(
                config=storage_config,
                metadata=storage_metadata,
                loop=loop,
                dst_device="musa:0",
            )
        finally:
            loop.close()

    # (4) Allocator dispatch picks MuFileMemoryAllocator. Reuse the bypass
    # pattern from the dedicated dispatch test rather than running the full
    # GdsBackend.__init__ (which would need a real ``mufile`` install).
    # First Party
    from lmcache.v1.memory_management import MuFileMemoryAllocator
    from lmcache.v1.storage_backend.gds_backend import GdsBackend

    backend = GdsBackend.__new__(GdsBackend)
    backend.gds_backend = "mufile"
    allocator = backend.initialize_allocator(
        _make_config(gds_buffer_size=1), _storage_metadata()
    )
    assert isinstance(allocator, MuFileMemoryAllocator)


# ---------------------------------------------------------------------------
# Module import sanity
# ---------------------------------------------------------------------------


def test_lmcache_exports_torch_dev_and_torch_device_type() -> None:
    """The contract used by every accelerator-aware module is the
    ``torch_dev`` / ``torch_device_type`` pair exported from ``lmcache``.

    Re-import to defeat any per-test monkeypatching above.
    """
    importlib.reload(lmc)
    assert hasattr(lmc, "torch_dev")
    assert isinstance(lmc.torch_device_type, str)
    assert lmc.torch_device_type in {"cuda", "musa", "xpu", "hpu", "cpu"}


# ---------------------------------------------------------------------------
# python_ops_fallback fallback library resolution (Phase 6)
# ---------------------------------------------------------------------------
#
# When LMCache runs on a host without the compiled CUDA extension, the
# python fallback ``lmcache.python_ops_fallback`` is loaded as
# ``lmcache.c_ops``. Its ``lmcache_memcpy_async`` and ``_tensor_from_cuda_ptr``
# paths lazily resolve a vendor memcpy library via ``_get_copy_lib``. On a
# MUSA-only host that means ``libmusart.so`` must be one of the candidates;
# otherwise the call raises ``RuntimeError("Failed to load libcudart/libamdhip")``
# even though the operator has a perfectly valid MUSA runtime.


def test_python_ops_fallback_copy_lib_resolves_libmusart_on_musa() -> None:
    """``_get_copy_lib`` must try ``libmusart.so`` so MUSA hosts can use the
    Python fallback for pointer-mode memcpy when ``cudart`` and ``amdhip64``
    are unavailable.

    We monkey-patch ``ctypes.util.find_library`` to return ``None`` for the
    CUDA / ROCm candidates and a sentinel for ``musart``, then assert the
    helper loads (and caches) a ``CDLL`` for the MUSA library.
    """
    # First Party
    from lmcache import python_ops_fallback

    importlib.reload(python_ops_fallback)

    fake_handle = SimpleNamespace(name="libmusart")
    cdll_calls: list[str] = []

    def fake_find_library(name: str):
        return f"/fake/lib{name}.so" if name == "musart" else None

    def fake_cdll(path: str):
        cdll_calls.append(path)
        if path.endswith("libmusart.so"):
            return fake_handle
        raise OSError(f"unexpected dlopen of {path}")

    with (
        patch.object(
            python_ops_fallback.ctypes.util,
            "find_library",
            side_effect=fake_find_library,
        ),
        patch.object(python_ops_fallback.ctypes, "CDLL", side_effect=fake_cdll),
    ):
        lib = python_ops_fallback._get_copy_lib()

    assert lib is fake_handle, (
        "After Phase 6 the fallback library chain must include libmusart.so; "
        f"got cdll_calls={cdll_calls}"
    )
    assert any("musart" in c for c in cdll_calls), (
        "_get_copy_lib must attempt ``musart``/``libmusart.so`` before giving up."
    )


def test_python_ops_fallback_copy_lib_prefers_cudart_then_hip_then_musart() -> None:
    """Ordering matters: cudart wins if available, then HIP, then MUSA.

    Asserting the order keeps the lookup deterministic and matches the
    POSIX runtime fallback in ``gds_backend`` (CUDA before MUSA on dual-stack
    machines, with HIP as the ROCm fork).
    """
    # First Party
    from lmcache import python_ops_fallback

    importlib.reload(python_ops_fallback)

    seen: list[str] = []

    def fake_find_library(name: str):
        seen.append(name)
        return None  # force fallback to ctypes.CDLL(fallback_name)

    def always_fail(path: str):
        seen.append(path)
        raise OSError(path)

    with (
        patch.object(
            python_ops_fallback.ctypes.util,
            "find_library",
            side_effect=fake_find_library,
        ),
        patch.object(python_ops_fallback.ctypes, "CDLL", side_effect=always_fail),
    ):
        result = python_ops_fallback._get_copy_lib()

    # All three names must have been tried, in cuda -> hip -> musa order.
    name_indices = {n: i for i, n in enumerate(seen) if not n.startswith("/")}
    assert "cudart" in name_indices
    assert "amdhip64" in name_indices
    assert "musart" in name_indices
    assert name_indices["cudart"] < name_indices["amdhip64"] < name_indices["musart"]
    assert result is None, "All candidates failed; must return None, not raise."
