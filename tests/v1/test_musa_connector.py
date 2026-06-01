# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.musa_connectors import (
    SGLangLayerwiseMUSAConnector,
    SGLangMUSAConnector,
    VLLMPagedMemLayerwiseMUSAConnector,
    VLLMPagedMemMUSAConnectorV2,
)
from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    MemoryFormat,
    PinMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from tests.v1.utils import (
    check_paged_kv_cache_equal,
    check_sglang_paged_kv_cache_equal,
    generate_kv_cache_paged_list_tensors,
    generate_sglang_kv_cache_paged_list_tensors,
)


def _skip_if_no_musa():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("torch.musa is not available")


def _make_unique_slot_mapping(
    *, total_slots: int, num_tokens: int, device: torch.device
) -> torch.Tensor:
    return torch.randperm(total_slots, device=device, dtype=torch.int64)[
        :num_tokens
    ]


def _pack_slot_mapping(
    slot_mapping: torch.Tensor, starts: list[int], ends: list[int]
) -> torch.Tensor:
    return torch.cat(
        [slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)],
        dim=0,
    )


@pytest.mark.parametrize("use_gpu", [False, True])
def test_musa_connector_roundtrip_non_layerwise(use_gpu: bool):
    """Round-trip from_gpu -> to_gpu on the non-layerwise MUSA connector."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 2
    num_blocks = 4
    block_size = 16
    head_size = 64
    num_tokens = 32

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )

    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    total_slots = num_blocks * block_size
    slot_mapping = _make_unique_slot_mapping(
        total_slots=total_slots, num_tokens=num_tokens, device=device
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobj = pin_alloc.allocate(
        torch.Size([2, num_layers, num_tokens, hidden_dim_actual]),
        torch.bfloat16,
        MemoryFormat.KV_2LTD,
    )

    meta = LMCacheMetadata(
        model_name="musa_test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            num_layers,
            2,
            num_tokens,
            num_heads_actual,
            head_size_actual,
        ),
    )
    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        meta,
        use_gpu=use_gpu,
        device=device,
    )

    try:
        conn.from_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches,
        )

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for t in kvcaches_dst:
            t.zero_()

        conn.to_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches_dst,
        )

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        memobj.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_gpu", [False, True])
def test_musa_connector_roundtrip_layerwise(use_gpu: bool):
    """Round-trip batched_from_gpu -> batched_to_gpu on layerwise MUSA connector."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 4
    num_blocks = 8
    block_size = 16
    head_size = 64
    num_tokens = 64

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )

    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    total_slots = num_blocks * block_size
    slot_mapping = _make_unique_slot_mapping(
        total_slots=total_slots, num_tokens=num_tokens, device=device
    )

    meta = LMCacheMetadata(
        model_name="musa_test_layerwise",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            num_layers,
            2,
            num_tokens,
            num_heads_actual,
            head_size_actual,
        ),
    )

    conn = VLLMPagedMemLayerwiseMUSAConnector.from_metadata(
        meta,
        use_musa=use_gpu,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 256)

    memobjs_by_layer = [
        [
            pin_alloc.allocate(
                torch.Size([num_tokens, 2, hidden_dim_actual]),
                torch.bfloat16,
                MemoryFormat.KV_T2D,
            )
        ]
        for _ in range(num_layers)
    ]

    try:
        gen = conn.batched_from_gpu(
            memobjs_by_layer,
            starts=[0],
            ends=[num_tokens],
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches,
        )

        for _ in range(num_layers + 1):
            next(gen)

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for t in kvcaches_dst:
            t.zero_()

        gen2 = conn.batched_to_gpu(
            starts=[0],
            ends=[num_tokens],
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches_dst,
        )

        next(gen2)
        for layer_id in range(num_layers):
            gen2.send(memobjs_by_layer[layer_id])

        next(gen2)

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        for layer in memobjs_by_layer:
            for m in layer:
                m.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_gpu", [False, True])
def test_musa_connector_roundtrip_non_layerwise_multi_chunk(
    use_gpu: bool,
) -> None:
    """Non-layerwise multi-chunk round-trip on MUSA connector."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 2
    num_blocks = 6
    block_size = 8
    head_size = 64
    total_tokens = 32

    starts = [0, 7, 19]
    ends = [4, 13, 25]

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )
    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=total_tokens,
        device=device,
    )
    packed_slot_mapping = _pack_slot_mapping(slot_mapping, starts, ends)

    meta = LMCacheMetadata(
        model_name="musa_test_non_layerwise_multi_chunk",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            num_layers,
            2,
            total_tokens,
            num_heads_actual,
            head_size_actual,
        ),
    )
    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        meta,
        use_gpu=use_gpu,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobjs = []
    try:
        for s, e in zip(starts, ends, strict=False):
            n = e - s
            memobj = pin_alloc.allocate(
                torch.Size([2, num_layers, n, hidden_dim_actual]),
                torch.bfloat16,
                MemoryFormat.KV_2LTD,
            )
            conn.from_gpu(
                memobj,
                start=s,
                end=e,
                slot_mapping=slot_mapping,
                kvcaches=kvcaches,
            )
            memobjs.append((s, e, memobj))

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for layer in kvcaches_dst:
            layer.zero_()

        for s, e, memobj in memobjs:
            conn.to_gpu(
                memobj,
                start=s,
                end=e,
                slot_mapping=slot_mapping,
                kvcaches=kvcaches_dst,
            )

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            packed_slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        for _, _, memobj in memobjs:
            memobj.ref_count_down()
        pin_alloc.close()


def test_sglang_musa_connector_roundtrip_non_layerwise_list_cache() -> None:
    """Round-trip on SGLang's native ``[k_list, v_list]`` layout (not a tuple)."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 2
    num_blocks = 4
    block_size = 16
    num_heads = 4
    head_size = 16
    num_tokens = 32
    hidden_dim = num_heads * head_size

    gpu_kv_src = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=False,
        device=device,
        dtype=torch.bfloat16,
    )
    gpu_kv_dst = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=False,
        device=device,
        dtype=torch.bfloat16,
    )

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=num_tokens,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobj = pin_alloc.allocate(
        torch.Size([2, num_layers, num_tokens, hidden_dim]),
        torch.bfloat16,
        MemoryFormat.KV_2LTD,
    )

    conn = SGLangMUSAConnector(
        hidden_dim_size=hidden_dim,
        num_layers=num_layers,
        use_gpu=False,
    )

    try:
        conn.from_gpu(
            memobj,
            0,
            num_tokens,
            kvcaches=gpu_kv_src,
            slot_mapping=slot_mapping,
        )

        conn.to_gpu(
            memobj,
            0,
            num_tokens,
            kvcaches=gpu_kv_dst,
            slot_mapping=slot_mapping,
            offset=0,
        )

        check_sglang_paged_kv_cache_equal(
            gpu_kv_src,
            gpu_kv_dst,
            slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
        )
    finally:
        memobj.ref_count_down()
        pin_alloc.close()


def test_sglang_musa_connector_roundtrip_layerwise_list_cache() -> None:
    """Layerwise SGLang MUSA connector with ``[k_list, v_list]`` kvcaches."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 4
    num_blocks = 8
    block_size = 16
    num_heads = 4
    head_size = 64
    num_tokens = 64
    hidden_dim = num_heads * head_size

    kvcaches = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=False,
        device=device,
        dtype=torch.bfloat16,
    )

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=num_tokens,
        device=device,
    )

    meta = LMCacheMetadata(
        model_name="musa_sglang_layerwise",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(num_layers, 2, num_tokens, num_heads, head_size),
    )
    conn = SGLangLayerwiseMUSAConnector.from_metadata(
        meta,
        use_gpu=False,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 256)
    memobjs_by_layer = [
        [
            pin_alloc.allocate(
                torch.Size([num_tokens, 2, hidden_dim]),
                torch.bfloat16,
                MemoryFormat.KV_T2D,
            )
        ]
        for _ in range(num_layers)
    ]

    try:
        producer = conn.batched_from_gpu(
            memobjs_by_layer,
            starts=[0],
            ends=[num_tokens],
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches,
        )
        for _ in range(num_layers + 1):
            next(producer)

        kvcaches_dst = generate_sglang_kv_cache_paged_list_tensors(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=False,
            device=device,
            dtype=torch.bfloat16,
        )

        consumer = conn.batched_to_gpu(
            starts=[0],
            ends=[num_tokens],
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches_dst,
        )
        next(consumer)
        for layer_id in range(num_layers):
            consumer.send(memobjs_by_layer[layer_id])
        next(consumer)

        check_sglang_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
        )
    finally:
        for layer in memobjs_by_layer:
            for m in layer:
                m.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_musa", [False, True])
def test_musa_connector_roundtrip_layerwise_multi_chunk(
    use_musa: bool,
) -> None:
    """Layerwise multi-chunk round-trip on MUSA connector."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 4
    num_blocks = 8
    block_size = 8
    head_size = 64
    total_tokens = 40

    starts = [0, 9, 21]
    ends = [5, 15, 30]

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )

    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=total_tokens,
        device=device,
    )
    packed_slot_mapping = _pack_slot_mapping(slot_mapping, starts, ends)

    meta = LMCacheMetadata(
        model_name="musa_test_layerwise_multi_chunk",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            num_layers,
            2,
            total_tokens,
            num_heads_actual,
            head_size_actual,
        ),
    )
    conn = VLLMPagedMemLayerwiseMUSAConnector.from_metadata(
        meta,
        use_musa=use_musa,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 128)
    memobjs_by_layer = []
    for _ in range(num_layers):
        per_layer = []
        for s, e in zip(starts, ends, strict=False):
            n = e - s
            per_layer.append(
                pin_alloc.allocate(
                    torch.Size([n, 2, hidden_dim_actual]),
                    torch.bfloat16,
                    MemoryFormat.KV_T2D,
                )
            )
        memobjs_by_layer.append(per_layer)

    try:
        producer = conn.batched_from_gpu(
            memobjs_by_layer,
            starts=starts,
            ends=ends,
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches,
        )
        for _ in range(num_layers + 1):
            next(producer)

        if use_musa:
            assert conn.gpu_buffer_allocator is not None
        else:
            assert conn.gpu_buffer_allocator is None

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for layer in kvcaches_dst:
            layer.zero_()

        consumer = conn.batched_to_gpu(
            starts=starts,
            ends=ends,
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches_dst,
        )
        next(consumer)
        for layer_id in range(num_layers):
            consumer.send(memobjs_by_layer[layer_id])
        next(consumer)

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            packed_slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        for layer in memobjs_by_layer:
            for memobj in layer:
                memobj.ref_count_down()
        pin_alloc.close()
