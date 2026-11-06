# SPDX-License-Identifier: Apache-2.0
"""GLM layout contracts, not a claim of captured MUSA model geometry."""

# Standard
from dataclasses import dataclass
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.types import (
    DiscoverableKVCache,
    KVLayoutName,
)
from lmcache.v1.gpu_connector.utils import (
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.platform.devices.musa.device_ops import MusaDeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc

F = EngineKVFormat
_NB, _NL = 6, 2
_CANARY, _OFFSET = 239, 16


@dataclass(frozen=True)
class PageContract:
    """Small physical fixture independent of vLLM/model imports."""

    fmt: EngineKVFormat
    layout: KVLayoutName
    bs: int
    nh: int
    width: int
    dtype: torch.dtype


_CONTRACTS = [
    PageContract(F.NL_X_NB_BS_HS, "NHD", 8, 1, 16, torch.bfloat16),
    PageContract(F.NL_X_NB_BSV_BSS, "NHD", 4, 1, 132, torch.uint8),
    PageContract(F.NL_X_NB_BS_NH_CS, "NHD", 4, 2, 16, torch.bfloat16),
    PageContract(F.NL_X_NB_NH_BS_CS, "HND", 4, 2, 16, torch.uint8),
]


def _page(
    contract: PageContract, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate a page view with nonzero offset and untouched gap bytes.

    Args:
        contract: Shape, dtype and layout of the physical payload.
        device: Device hosting the allocation.

    Returns:
        The flat owning storage and its strided per-layer page view.
    """
    bs, nh, width = contract.bs, contract.nh, contract.width
    step = bs * nh * width + 32
    storage = torch.full(
        (_OFFSET + _NB * step + 16,), _CANARY, dtype=contract.dtype, device=device
    )
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    if contract.fmt in (F.NL_X_NB_BS_HS, F.NL_X_NB_BSV_BSS):
        shape, strides = (_NB, bs, width), (step, width, 1)
    elif contract.layout == "HND":
        shape, strides = (_NB, nh, bs, width), (step, bs * width, width, 1)
    else:
        shape, strides = (_NB, bs, nh, width), (step, nh * width, width, 1)
    return storage, storage.as_strided(shape, strides, storage_offset=_OFFSET)


def _fill_source(
    contract: PageContract, page: torch.Tensor, rows: torch.Tensor
) -> None:
    """Encode canonical rows independently of the transfer implementation.

    Args:
        contract: Physical layout contract.
        page: Destination engine page.
        rows: Canonical ``[NB, BS, NH * width]`` reference.

    Returns:
        None.
    """
    if contract.fmt == F.NL_X_NB_BSV_BSS:
        for block in range(_NB):
            flat = page[block].view(-1)
            split = contract.bs * 128
            flat[:split].copy_(rows[block, :, :128].reshape(-1))
            flat[split:].copy_(rows[block, :, 128:].reshape(-1))
    elif contract.fmt == F.NL_X_NB_BS_HS:
        page.copy_(rows)
    else:
        shaped = rows.view(_NB, contract.bs, contract.nh, contract.width)
        page.copy_(shaped.permute(0, 2, 1, 3) if contract.layout == "HND" else shaped)


@pytest.fixture(params=["cpu", "musa"])
def transfer_device(request: pytest.FixtureRequest) -> torch.device:
    """Run CPU contracts always and MUSA pointer roundtrips only on hardware.

    Args:
        request: Device parameter supplied by pytest.

    Returns:
        The available transfer device.
    """
    if request.param == "musa":
        if torch_device_type != "musa" or not torch_dev.is_available():
            pytest.skip("MUSA hardware is required for pointer roundtrip")
        return torch.device("musa:0")
    return torch.device("cpu")


@pytest.mark.parametrize("contract", _CONTRACTS, ids=["mla", "indexer", "nhd", "hnd"])
@pytest.mark.parametrize("skip", [0, 3])
@pytest.mark.parametrize("native_enabled", ["0", "1"])
def test_glm_pages_store_and_restore_without_touching_padding(
    contract: PageContract,
    skip: int,
    transfer_device: torch.device,
    native_enabled: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical rows, partial objects and page canaries survive both directions."""
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", native_enabled)
    device = transfer_device
    source = [_page(contract, device) for _ in range(_NL)]
    target = [_page(contract, device) for _ in range(_NL)]
    hidden = contract.nh * contract.width
    rows = (
        torch.arange(_NL * _NB * contract.bs * hidden, device=device)
        .remainder(197)
        .add(1)
        .to(contract.dtype)
        .view(_NL, _NB, contract.bs, hidden)
    )
    for layer, (_, page) in enumerate(source):
        _fill_source(contract, page, rows[layer])
    source_before = [storage.clone() for storage, _ in source]
    chunk_slots = 2 * contract.bs
    objects = [
        torch.full(
            (_NL, chunk_slots, hidden), _CANARY, device=device, dtype=contract.dtype
        )
        for _ in range(3)
    ]
    ids = [4, 1, 5, 0, 3]
    block_ids = torch.tensor(ids, dtype=torch.int64, device=device)
    desc = PageBufferShapeDesc()
    desc.nl, desc.nb, desc.bs = _NL, _NB, contract.bs
    desc.nh, desc.hs, desc.kv_size = contract.nh, contract.width, 1
    desc.element_size, desc.dtype = rows.element_size(), contract.dtype
    desc.block_stride_elems = source[0][1].stride(0)

    ops = MusaDeviceOps()
    source_pages = [page for _, page in source]
    target_pages = [page for _, page in target]
    # Hardware exercises the MP wire operands; CPU tests use actual tensor views.
    source_operand = (
        torch.tensor([p.data_ptr() for p in source_pages], device=device)
        if device.type == "musa"
        else source_pages
    )
    target_operand = (
        torch.tensor([p.data_ptr() for p in target_pages], device=device)
        if device.type == "musa"
        else target_pages
    )
    object_operand = (
        [obj.data_ptr() for obj in objects] if device.type == "musa" else objects
    )
    ops.multi_layer_block_kv_transfer(
        source_operand,
        object_operand,
        block_ids,
        device,
        TransferDirection.D2H,
        desc,
        chunk_slots,
        contract.fmt,
        0,
    )
    expected_rows = rows[:, ids].reshape(_NL, -1, hidden)
    actual_rows = torch.cat(objects, dim=1)
    assert torch.equal(actual_rows[:, : len(ids) * contract.bs], expected_rows)
    assert torch.all(actual_rows[:, len(ids) * contract.bs :] == _CANARY)

    ops.multi_layer_block_kv_transfer(
        target_operand,
        object_operand,
        block_ids,
        device,
        TransferDirection.H2D,
        desc,
        chunk_slots,
        contract.fmt,
        skip,
    )
    payload = contract.bs * hidden
    step = desc.block_stride_elems
    for (src_storage, _), (dst_storage, _), before in zip(
        source, target, source_before, strict=True
    ):
        assert torch.equal(src_storage, before)
        expected = torch.full_like(dst_storage, _CANARY)
        for block in ids[skip:]:
            start = _OFFSET + block * step
            expected[start : start + payload] = src_storage[start : start + payload]
        assert torch.equal(dst_storage, expected)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_unified_glm_planes_and_indexer_keep_distinct_formats(
    layout: KVLayoutName,
) -> None:
    """Rank-4 unified pages and uint8 indexer normalize without losing stride."""
    inner = (4, 1) if layout == "NHD" else (1, 4)
    packed = torch.empty(4096, dtype=torch.bfloat16).as_strided(
        (6, *inner, 16),
        (128, 16, 16, 1) if layout == "NHD" else (128, 64, 16, 1),
        storage_offset=16,
    )
    indexer = torch.empty(8192, dtype=torch.uint8).as_strided(
        (6, *inner, 132),
        (640, 132, 132, 1) if layout == "NHD" else (640, 528, 132, 1),
        storage_offset=16,
    )
    normalized, formats = normalize_and_discover_per_layer_formats(
        [packed, indexer], [[0, 1]], EngineType.VLLM, {"kv_layout": layout}
    )
    expected_packed = F.NL_X_NB_BS_NH_CS if layout == "NHD" else F.NL_X_NB_NH_BS_CS
    assert formats == [expected_packed, F.NL_X_NB_BSV_BSS]
    assert isinstance(normalized, list)
    assert isinstance(normalized[0], torch.Tensor)
    assert isinstance(normalized[1], torch.Tensor)
    assert normalized[0].data_ptr() == packed.data_ptr()
    assert normalized[1].data_ptr() == indexer.data_ptr()
    assert normalized[0].stride(0) == 128
    assert normalized[1].stride(0) == 640


def test_flash_group_geometry_drives_each_transfer_independently() -> None:
    """State, sparse latent and compressed indexer keep independent slot counts."""
    contracts = [_CONTRACTS[2], _CONTRACTS[0], _CONTRACTS[1]]
    device = torch.device("cpu")
    source = [_page(c, device)[1] for c in contracts]
    for layer, page in enumerate(source):
        page.fill_(layer + 1)
    normalized, formats = normalize_and_discover_per_layer_formats(
        cast(DiscoverableKVCache, source),
        [[0], [1, 2]],
        EngineType.VLLM,
        {"kv_layout": "NHD"},
    )
    manager = KVLayerGroupsManager(
        normalized,
        engine_kv_formats=formats,
        engine_group_infos=[
            EngineGroupInfo(
                0, (0,), tokens_per_block=4, sw_size_tokens=4, recurrent_state=True
            ),
            EngineGroupInfo(1, (1,), tokens_per_block=16),
            EngineGroupInfo(1, (2,), tokens_per_block=16),
        ],
        lmcache_tokens_per_chunk=32,
        separate_object_groups=True,
    )
    assert [g.slots_per_block for g in manager.kernel_groups] == [4, 8, 4]
    assert [manager.get_slots_per_chunk_in_sw(i) for i in range(3)] == [4, 16, 8]
    assert [g.engine_group_idx for g in manager.kernel_groups] == [0, 1, 1]
    ids_by_engine_group = [[5], [4, 1]]
    for index, group in enumerate(manager.kernel_groups):
        layer = group.layer_indices[0]
        page = source[layer]
        desc = manager.get_shape_desc(index)
        assert group.engine_kv_format is not None
        assert desc.block_stride_elems == page.stride(0)
        assert group.dtype == page.dtype
        slots = manager.get_slots_per_chunk_in_sw(index)
        obj = torch.empty(1, slots, group.hidden_dim_size, dtype=group.dtype)
        ids = ids_by_engine_group[group.engine_group_idx]
        target = _page(contracts[layer], device)[1]
        ops = MusaDeviceOps()
        ops.multi_layer_block_kv_transfer(
            [page],
            [obj],
            ids,
            device,
            TransferDirection.D2H,
            desc,
            slots,
            group.engine_kv_format,
            0,
        )
        assert torch.all(obj == layer + 1)
        ops.multi_layer_block_kv_transfer(
            [target],
            [obj],
            ids,
            device,
            TransferDirection.H2D,
            desc,
            slots,
            group.engine_kv_format,
            0,
        )
        for block in range(_NB):
            expected = layer + 1 if block in ids else _CANARY
            assert torch.all(target[block] == expected)
