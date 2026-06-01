# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Union, cast
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector.gpu_connectors import (
    GPUConnectorInterface,
    VLLMPagedMemGPUConnectorV2,
)
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    _get_head_size_view,
    _split_token2d_kv,
    get_block_size,
    get_dtype,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    get_page_buffer_size,
    is_mla,
    normalize_kv_and_discover_format,
)
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

ALLOWED_FORMAT_TRANSITIONS = {
    (None, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_MLA_FMT, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_T2D, MemoryFormat.KV_MLA_FMT),
}


class VLLMPagedMemMUSAConnectorV2(VLLMPagedMemGPUConnectorV2):
    """Non-layerwise paged KV connector for MUSA devices.

    Follows the same contract as VLLMPagedMemXPUConnectorV2: pure torch ops
    (index_copy_ / index_select) with ``torch.musa`` stream and sync APIs.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        **kwargs,
    ):
        self._attributes_initialized = False
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.use_gpu = use_gpu

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemMUSAConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemMUSAConnectorV2.
        """
        return cls(use_gpu=use_gpu)

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Store KV data from a memory object into MUSA paged KV caches.

        Args:
            memory_obj: The memory object containing KV data.
            start: Starting index in the token sequence.
            end: Ending index in the token sequence.

        Keyword Args:
            kvcaches: Nested tuple of K/V tensors for the whole sequence.
            slot_mapping: Full slot mapping tensor.

        Raises:
            ValueError: If slot_mapping is missing from kwargs.
            AssertionError: If memory_obj has no tensor.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)

        if self.use_mla:
            tmp = memory_obj.tensor[0].to(slot_mapping.device)
            total_blocks = self.num_blocks * self.block_size
            for i, kvcache in enumerate(self.kvcaches):
                kvcache.view(total_blocks, self.head_size).index_copy_(
                    0, slices, tmp[i]
                )
        else:
            tmp_k = memory_obj.tensor[0].to(slot_mapping.device)
            tmp_v = memory_obj.tensor[1].to(slot_mapping.device)
            total_blocks = self.num_blocks * self.block_size
            d = self.num_heads * self.head_size
            for i, (kcache, vcache) in enumerate(self.kvcaches):
                kcache.view(total_blocks, d).index_copy_(0, slices, tmp_k[i])
                vcache.view(total_blocks, d).index_copy_(0, slices, tmp_v[i])

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Load KV data from MUSA paged KV caches into a memory object.

        Args:
            memory_obj: The memory object to populate.
            start: Starting index in the token sequence.
            end: Ending index in the token sequence.

        Keyword Args:
            kvcaches: Nested tuple of K/V tensors for the whole sequence.
            slot_mapping: Full slot mapping tensor.

        Raises:
            ValueError: If slot_mapping is missing from kwargs.
            AssertionError: If memory_obj has no tensor.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)

        if self.use_mla:
            total_blocks = self.num_blocks * self.block_size
            tmp = torch.stack(
                [
                    kvcache.view(total_blocks, self.head_size).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
        else:
            total_blocks = self.num_blocks * self.block_size
            d = self.num_heads * self.head_size
            tmp_k = torch.stack(
                [
                    kvcache[0].view(total_blocks, d).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
            tmp_v = torch.stack(
                [
                    kvcache[1].view(total_blocks, d).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
            tmp = torch.stack([tmp_k, tmp_v])
        memory_obj.tensor.copy_(tmp, non_blocking=True)

        if memory_obj.tensor.device.type != "musa":
            torch.musa.synchronize()  # type: ignore[attr-defined]

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(
        self,
        memory_objs: List[MemoryObj],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens.

        Args:
            num_tokens: The number of tokens in the data.

        Returns:
            The shape of the KV cache data.

        Raises:
            RuntimeError: If attributes have not been initialized yet.
        """
        if not self._attributes_initialized:
            raise RuntimeError(
                "Cannot determine shape before attributes are initialized. "
                "Call to_gpu or from_gpu first so that _initialize_attributes "
                "can discover the KV cache layout."
            )
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])

    def _validate_memory_format(self, memory_obj: MemoryObj) -> None:
        """Validate that the memory object has the expected format.

        Args:
            memory_obj: The memory object to validate.

        Raises:
            ValueError: If the memory format does not match.
        """
        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemMUSAConnectorV2"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemMUSAConnectorV2"
                )

    def _initialize_attributes(self, kv_caches: List[torch.Tensor]):
        """Initialize attributes from KV caches using utils functions.

        Args:
            kv_caches: The KV cache tensors from which to discover layout.
        """
        if self._attributes_initialized:
            return

        self.device = kv_caches[0].device
        assert self.device.type == "musa", "The device should be MUSA."

        self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
            kv_caches, EngineType.VLLM
        )
        self.num_layers = get_num_layers(kv_caches, self.gpu_kv_format)
        self.num_blocks = get_num_blocks(kv_caches, self.gpu_kv_format)
        self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
        self.page_buffer_size = get_page_buffer_size(kv_caches, self.gpu_kv_format)
        self.hidden_dim_size = get_hidden_dim_size(kv_caches, self.gpu_kv_format)
        self.head_size = get_head_size(kv_caches, self.gpu_kv_format)
        self.use_mla = is_mla(self.gpu_kv_format)
        self.dtype = get_dtype(kv_caches, self.gpu_kv_format)
        self.num_heads = (
            1 if self.use_mla else get_num_heads(kv_caches, self.gpu_kv_format)
        )

        self._attributes_initialized = True
        logger.info(
            "MUSA: attributes initialized - format: %s, "
            "num_layers: %d, num_blocks: %d, block_size: %d, "
            "page_buffer_size: %d, hidden_dim_size: %d, head_size: %d, "
            "use_mla: %s, dtype: %s, num_heads: %d",
            self.gpu_kv_format,
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.page_buffer_size,
            self.hidden_dim_size,
            self.head_size,
            self.use_mla,
            self.dtype,
            self.num_heads,
        )


class VLLMPagedMemLayerwiseMUSAConnector(GPUConnectorInterface):
    """Layerwise paged KV connector for MUSA devices.

    Implements the same generator contract as VLLMPagedMemLayerwiseXPUConnector:
      - batched_to_gpu(...) yields num_layers + 2 times
      - batched_from_gpu(...) yields num_layers + 1 times

    Transfer is implemented with pure torch ops (index_copy_ / index_select).
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_musa: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_musa = use_musa

        assert "chunk_size" in kwargs, "chunk_size should be provided."
        assert "dtype" in kwargs, "dtype should be provided."
        assert "device" in kwargs, "device should be provided."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.kvcaches: Optional[List[torch.Tensor]] = None

        self.load_stream = torch.musa.Stream()  # type: ignore[attr-defined]
        self.store_stream = torch.musa.Stream()  # type: ignore[attr-defined]

        self.gpu_buffer_allocator: Optional[MemoryAllocatorInterface] = None

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_musa: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemLayerwiseMUSAConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model
                configuration.
            use_musa: Whether to use MUSA intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemLayerwiseMUSAConnector.
        """
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_musa=use_musa,
            chunk_size=metadata.kv_shape[2],
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _validate_format_transition(
        self, mem: MemoryObj, target_fmt: MemoryFormat
    ) -> None:
        current_fmt = mem.metadata.fmt
        if (current_fmt, target_fmt) not in ALLOWED_FORMAT_TRANSITIONS:
            raise ValueError(
                f"Invalid KV format transition: {current_fmt} -> {target_fmt}"
            )

    def _lazy_initialize_buffer(self, kv_caches: List[torch.Tensor]) -> None:
        if self.use_musa and self.gpu_buffer_allocator is None:
            # First Party
            from lmcache.v1.memory_management import MUSAMemoryAllocator

            layer0 = kv_caches[0]
            derived_bytes = layer0.numel() * layer0.element_size()
            staging_bytes = int(
                os.getenv("LMCACHE_GPU_STAGING_BUFFER_BYTES", derived_bytes)
            )
            logger.info(
                "Initializing MUSA staging buffer (derived=%d bytes, final=%d bytes)",
                derived_bytes,
                staging_bytes,
            )
            self.gpu_buffer_allocator = MUSAMemoryAllocator(
                size=staging_bytes, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError("Layerwise uses batched_to_gpu(generator).")

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError("Layerwise uses batched_from_gpu(generator).")

    def _batched_to_gpu_gen(self, starts: List[int], ends: List[int], **kwargs):
        """Generator: CPU token2d -> (optional staging) -> MUSA paged KV."""
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        def _ensure_musa(t: torch.Tensor) -> torch.Tensor:
            if t.device != self.device:
                return t.to(self.device, non_blocking=True)
            return t

        slot_mapping_chunks = [
            slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)
        ]
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        slot_mapping_full = _ensure_musa(slot_mapping_full)

        num_tokens = int(slot_mapping_full.numel())
        if num_tokens <= 0:
            for _ in range(self.num_layers):
                _ = yield
            yield
            if sync:
                torch.musa.current_stream().wait_stream(self.load_stream)  # type: ignore[attr-defined]
            yield
            return

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_musa:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            if tmp_gpu_buffer_obj is None or tmp_gpu_buffer_obj.tensor is None:
                raise RuntimeError(
                    "Failed to allocate MUSA staging buffer for batched_to_gpu."
                )

        current_stream = torch.musa.current_stream()  # type: ignore[attr-defined]

        try:
            for layer_id in range(self.num_layers):
                memory_objs_layer = yield

                if sync:
                    current_stream.wait_stream(self.load_stream)

                with torch.musa.stream(self.load_stream):  # type: ignore[attr-defined]
                    dst_layer = self.kvcaches[layer_id]
                    if self.use_mla:
                        dst_flat = cast(
                            torch.Tensor,
                            _get_head_size_view(dst_layer, use_mla=True),
                        )
                    else:
                        dst_k_flat, dst_v_flat = _get_head_size_view(  # type: ignore[misc]
                            dst_layer, use_mla=False
                        )

                    cursor = 0

                    if self.use_musa:
                        assert tmp_gpu_buffer_obj is not None
                        staged = tmp_gpu_buffer_obj.tensor
                        assert staged is not None

                        for s, e, mem in zip(
                            starts, ends, memory_objs_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue
                            src = _ensure_musa(mem.tensor)
                            staged[cursor : cursor + n].copy_(src, non_blocking=True)
                            cursor += n

                        sl = _ensure_musa(slot_mapping_full)

                        if self.use_mla:
                            staged_dev = _ensure_musa(staged)
                            if staged_dev.dim() == 2:
                                dst_flat.index_copy_(0, sl, staged_dev)
                            elif staged_dev.dim() == 3 and staged_dev.shape[0] == 1:
                                dst_flat.index_copy_(0, sl, staged_dev[0])
                            else:
                                raise ValueError(
                                    f"Unexpected MLA staged tensor: {staged_dev.shape}"
                                )
                        else:
                            k_tok, v_tok = _split_token2d_kv(staged)
                            k_tok = _ensure_musa(k_tok)
                            v_tok = _ensure_musa(v_tok)

                            if (
                                k_tok.dim() == 2
                                and dst_k_flat.dim() == 3
                                and k_tok.shape[1]
                                == dst_k_flat.shape[1] * dst_k_flat.shape[2]
                            ):
                                k_tok = k_tok.reshape(
                                    k_tok.shape[0],
                                    dst_k_flat.shape[1],
                                    dst_k_flat.shape[2],
                                )
                            if (
                                v_tok.dim() == 2
                                and dst_v_flat.dim() == 3
                                and v_tok.shape[1]
                                == dst_v_flat.shape[1] * dst_v_flat.shape[2]
                            ):
                                v_tok = v_tok.reshape(
                                    v_tok.shape[0],
                                    dst_v_flat.shape[1],
                                    dst_v_flat.shape[2],
                                )

                            dst_k_flat.index_copy_(0, sl, k_tok)
                            dst_v_flat.index_copy_(0, sl, v_tok)

                    else:
                        for s, e, mem in zip(
                            starts, ends, memory_objs_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue
                            src = _ensure_musa(mem.tensor)
                            sl = slot_mapping_full[cursor : cursor + n]
                            sl = _ensure_musa(sl)
                            cursor += n

                            if self.use_mla:
                                if src.dim() == 2:
                                    dst_flat.index_copy_(0, sl, src)
                                elif src.dim() == 3 and src.shape[0] == 1:
                                    dst_flat.index_copy_(0, sl, src[0])
                                else:
                                    raise ValueError(
                                        f"Unexpected MLA token tensor: {src.shape}"
                                    )
                            else:
                                k_tok, v_tok = _split_token2d_kv(src)
                                k_tok = _ensure_musa(k_tok)
                                v_tok = _ensure_musa(v_tok)

                                if (
                                    k_tok.dim() == 2
                                    and dst_k_flat.dim() == 3
                                    and k_tok.shape[1]
                                    == dst_k_flat.shape[1] * dst_k_flat.shape[2]
                                ):
                                    k_tok = k_tok.reshape(
                                        k_tok.shape[0],
                                        dst_k_flat.shape[1],
                                        dst_k_flat.shape[2],
                                    )
                                if (
                                    v_tok.dim() == 2
                                    and dst_v_flat.dim() == 3
                                    and v_tok.shape[1]
                                    == dst_v_flat.shape[1] * dst_v_flat.shape[2]
                                ):
                                    v_tok = v_tok.reshape(
                                        v_tok.shape[0],
                                        dst_v_flat.shape[1],
                                        dst_v_flat.shape[2],
                                    )

                                dst_k_flat.index_copy_(0, sl, k_tok)
                                dst_v_flat.index_copy_(0, sl, v_tok)

            yield

            if sync:
                current_stream.wait_stream(self.load_stream)
        finally:
            if tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_from_gpu(
        self,
        memory_objs: List[List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """Generator: MUSA paged KV -> CPU token2d (per layer)."""
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        current_stream = torch.musa.current_stream()  # type: ignore[attr-defined]

        slot_mapping_on_device = slot_mapping.to(self.device)

        for layer_id in range(self.num_layers):
            mem_layer = memory_objs[layer_id]

            with torch.musa.stream(self.store_stream):  # type: ignore[attr-defined]
                self.store_stream.wait_stream(current_stream)

                src_layer = self.kvcaches[layer_id]

                if self.use_mla:
                    src_flat = cast(
                        torch.Tensor,
                        _get_head_size_view(src_layer, use_mla=True),
                    )
                    for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                        assert mem.tensor is not None
                        sl = slot_mapping_on_device[s:e]
                        gathered = src_flat.index_select(0, sl)
                        mem.tensor.copy_(
                            gathered.to(mem.tensor.device),
                            non_blocking=True,
                        )

                    target_fmt = MemoryFormat.KV_MLA_FMT
                    for mem in mem_layer:
                        self._validate_format_transition(mem, target_fmt)
                        mem.metadata.fmt = target_fmt
                else:
                    src_k_flat, src_v_flat = _get_head_size_view(
                        src_layer, use_mla=False
                    )
                    for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                        assert mem.tensor is not None
                        sl = slot_mapping_on_device[s:e]
                        k = src_k_flat.index_select(0, sl)
                        v = src_v_flat.index_select(0, sl)

                        if mem.tensor.shape[0] == 2:
                            mem.tensor[0].copy_(
                                k.to(mem.tensor.device), non_blocking=True
                            )
                            mem.tensor[1].copy_(
                                v.to(mem.tensor.device), non_blocking=True
                            )
                        elif mem.tensor.dim() >= 2 and mem.tensor.shape[1] == 2:
                            mem.tensor[:, 0].copy_(
                                k.to(mem.tensor.device), non_blocking=True
                            )
                            mem.tensor[:, 1].copy_(
                                v.to(mem.tensor.device), non_blocking=True
                            )
                        else:
                            raise ValueError(
                                f"Unrecognized KV tensor layout: {mem.tensor.shape}"
                            )

            if sync:
                self.store_stream.synchronize()
            yield

        yield

    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        return self._batched_to_gpu_gen(starts=starts or [], ends=ends or [], **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data for a single layer.

        Args:
            num_tokens: The number of tokens in the data.

        Returns:
            The shape of the KV cache data for one layer.
        """
        if self.use_mla:
            return torch.Size([num_tokens, self.hidden_dim_size])
        return torch.Size([num_tokens, 2, self.hidden_dim_size])


# ---------------------------------------------------------------------------
# SGLang on MUSA
# ---------------------------------------------------------------------------
#
# SGLang's CUDA connectors call two ``lmc_ops`` kernels:
#
#   - ``lmc_ops.multi_layer_kv_transfer_unilateral`` (non-layerwise)
#   - ``lmc_ops.single_layer_kv_transfer_sgl``        (layerwise)
#
# Both are CUDA-only C++ ops and both assert ``device.type == "cuda"`` on
# their inputs. The MUSA mirrors below replace those calls with the same
# pure-torch ``index_copy_`` / ``index_select`` pattern the MUSA vLLM
# connectors already use, plus ``torch.musa`` streams. The KV cache layout
# (one ``[page_buffer_size, head_num, head_size]`` tensor per layer for K
# and another for V; flat list for MLA) is unchanged so SGLang's
# ``kvcaches`` argument flows through without re-shaping.


def _first_tensor_from_discoverable(kvcaches: object) -> torch.Tensor:
    """Return the first leaf tensor in a nested list/tuple KV cache structure.

    Args:
        kvcaches: SGLang or normalized KV cache (nested lists or tuples of
            tensors).

    Returns:
        The first underlying ``torch.Tensor``.

    Raises:
        ValueError: If the structure is empty.
        TypeError: If no tensor is found at a leaf.
    """
    probe: object = kvcaches
    while isinstance(probe, (list, tuple)):
        if not probe:
            raise ValueError("kvcaches must not be empty.")
        probe = probe[0]
    if not isinstance(probe, torch.Tensor):
        raise TypeError(
            f"Expected a tensor at the leaf of kvcaches, got {type(probe).__name__}."
        )
    return probe


def _sglang_mha_kv_lists(kvcaches: object) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split non-MLA SGLang MHA caches into K and V layer lists.

    SGLang passes ``[k_list, v_list]`` (a length-2 list). Some call sites
    use a 2-tuple; both layouts are accepted.

    Args:
        kvcaches: ``List[KTensor]`` per layer plus ``List[VTensor]`` per layer.

    Returns:
        ``(k_list, v_list)`` with one tensor per layer.

    Raises:
        TypeError: If ``kvcaches`` is not a 2-element list or tuple of lists.
    """
    if isinstance(kvcaches, tuple) and len(kvcaches) == 2:
        k_list, v_list = kvcaches
    elif isinstance(kvcaches, list) and len(kvcaches) == 2:
        k_list, v_list = kvcaches[0], kvcaches[1]
    else:
        raise TypeError(
            "Non-MLA SGLang kvcaches must be a 2-element list or tuple "
            f"(K_list, V_list); got {type(kvcaches).__name__}"
        )
    if not isinstance(k_list, list) or not isinstance(v_list, list):
        raise TypeError(
            "Non-MLA SGLang kvcaches entries must be lists of per-layer tensors."
        )
    return k_list, v_list


def _sglang_per_layer_kv_views(
    kvcaches: object, hidden_dim_size: int, use_mla: bool
) -> "list[tuple[torch.Tensor, Optional[torch.Tensor]]]":
    """Yield ``[page_buf, hidden_dim]``-flat (K, V) views per layer.

    Returns a list rather than a generator so ``len()`` is available for
    consistency checks; the per-layer views are *aliases* of the underlying
    K/V tensors (``Tensor.view``), so callers can ``index_copy_`` them
    in-place to write back into the SGLang paged caches.

    Args:
        kvcaches: SGLang's KV cache structure. For non-MLA it is
            ``(List[KTensor], List[VTensor])``; each tensor has shape
            ``[page_buffer_size, head_num, head_size]``. For MLA it is a
            flat ``List[KTensor]`` (no V), same per-tensor shape.
        hidden_dim_size: ``head_num * head_size`` (or just ``head_size``
            on MLA, depending on how upstream packs the KV).
        use_mla: Whether MLA mode is active. Drives the layout assumption.

    Returns:
        A list of ``(k_view, v_view_or_None)`` pairs, one per layer.

    Raises:
        TypeError: If ``kvcaches`` has an unexpected shape for the mode.
    """
    if use_mla:
        # MLA: kvcaches is a flat list (one tensor per layer, K only).
        if not hasattr(kvcaches, "__iter__"):
            raise TypeError(
                f"MLA SGLang kvcaches must be iterable, got {type(kvcaches)}"
            )
        return [(k.view(-1, hidden_dim_size), None) for k in kvcaches]

    # Non-MLA: kvcaches is [K_list, V_list] (list) or (K_list, V_list) (tuple).
    k_list, v_list = _sglang_mha_kv_lists(kvcaches)
    views: list[tuple[torch.Tensor, Optional[torch.Tensor]]] = []
    for k, v in zip(k_list, v_list, strict=False):
        views.append(
            (
                k.view(-1, hidden_dim_size),
                v.view(-1, hidden_dim_size),
            )
        )
    return views


def _to_musa(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move ``t`` to ``device`` non-blocking if it is not already there."""
    if t.device != device:
        return t.to(device, non_blocking=True)
    return t


class SGLangMUSAConnector(GPUConnectorInterface):
    """Non-layerwise paged KV connector for SGLang on MUSA.

    Pure-torch mirror of :class:`SGLangGPUConnector`. Replaces
    ``lmc_ops.multi_layer_kv_transfer_unilateral`` with a per-layer
    ``index_copy_`` (H2D) or ``index_select`` (D2H) using the same slot
    mapping. The KV cache shape contract is preserved: each layer's K and
    V are ``[page_buffer_size, head_num, head_size]``; the memory object
    is ``KV_2LTD`` (``[2, num_layers, num_tokens, hidden_dim]``) for
    non-MLA or ``KV_MLA_FMT`` (``[num_layers, num_tokens, hidden_dim]``)
    for MLA.

    Performance note: the per-layer torch path is ~functionally identical
    to the CUDA C++ op but launches more kernels (one per layer). For
    production SLAs on MUSA, the same speed-up path used for vLLM-MUSA
    applies: build a MUSA version of ``lmc_ops`` or use a fused kernel
    via ``torch.musa.compile``. Tracked in
    ``docs/source/developer_guide/musa_sglang_integration_debug.rst``.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ) -> None:
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_mla = bool(kwargs.get("use_mla", False))
        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2

        self.use_gpu = use_gpu
        self.gpu_buffer: Optional[torch.Tensor] = None
        self.gpu_kv_format = None
        self.page_buffer_size = 0
        self._metadata_initialized = False
        self._sglang_kvcaches: object = None

        if use_gpu:
            for required in ("chunk_size", "dtype", "device"):
                if required not in kwargs:
                    raise ValueError(
                        f"'{required}' must be provided when use_gpu=True; "
                        f"got kwargs keys={list(kwargs)}."
                    )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        chunk_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "SGLangMUSAConnector":
        """Construct from :class:`LMCacheMetadata`.

        ``CreateGPUConnector`` calls this with ``device`` plus the
        downstream connector's positional args; we accept ``**kwargs`` so
        future fields don't require a signature bump.
        """
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size if chunk_size is not None else metadata.kv_shape[2],
            dtype=dtype if dtype is not None else metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _initialize_metadata(self, kvcaches) -> torch.device:
        """Discover the SGLang KV layout and cache derived attributes.

        Args:
            kvcaches: SGLang's KV cache structure (see
                :func:`_sglang_per_layer_kv_views`).

        Returns:
            The torch device on which the KV caches live.

        Raises:
            AssertionError: If the underlying device is not MUSA.
        """
        if self._metadata_initialized:
            return self._kv_device

        self.gpu_kv_format, normalized = normalize_kv_and_discover_format(
            kvcaches, EngineType.SGLANG
        )
        self.page_buffer_size = get_page_buffer_size(normalized, self.gpu_kv_format)
        self._sglang_kvcaches = normalized

        device = _first_tensor_from_discoverable(normalized).device
        assert device.type == "musa", (
            f"SGLangMUSAConnector requires MUSA tensors; got '{device.type}'."
        )
        self._kv_device = device
        self._metadata_initialized = True
        return device

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Store memory_obj's KV slab into the SGLang paged caches.

        Mirrors :meth:`SGLangGPUConnector.to_gpu`: ``slot_mapping`` is the
        partial slot mapping (length == uncached tokens), and the slice
        ``[start - offset : end - offset]`` is used per call.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        self._validate_memory_format(memory_obj)

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        offset = kwargs.get("offset", 0)

        device = self._initialize_metadata(kwargs["kvcaches"])
        assert self._sglang_kvcaches is not None
        slot_slice = _to_musa(slot_mapping[start - offset : end - offset], device)

        per_layer = _sglang_per_layer_kv_views(
            self._sglang_kvcaches, self.hidden_dim_size, self.use_mla
        )

        if self.use_mla:
            for layer_id, (k_view, _) in enumerate(per_layer):
                src = _to_musa(memory_obj.tensor[layer_id], device)
                k_view.index_copy_(0, slot_slice, src)
        else:
            for layer_id, (k_view, v_view) in enumerate(per_layer):
                src_k = _to_musa(memory_obj.tensor[0, layer_id], device)
                src_v = _to_musa(memory_obj.tensor[1, layer_id], device)
                k_view.index_copy_(0, slot_slice, src_k)
                v_view.index_copy_(0, slot_slice, src_v)  # type: ignore[union-attr]

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Read a slab of KV from the SGLang paged caches into memory_obj.

        Mirrors :meth:`SGLangGPUConnector.from_gpu`. Slot mapping here is
        the *full* slot mapping (no ``offset`` adjustment).
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        device = self._initialize_metadata(kwargs["kvcaches"])
        assert self._sglang_kvcaches is not None
        slot_slice = _to_musa(slot_mapping[start:end], device)

        per_layer = _sglang_per_layer_kv_views(
            self._sglang_kvcaches, self.hidden_dim_size, self.use_mla
        )

        if self.use_mla:
            stacked = torch.stack(
                [k_view.index_select(0, slot_slice) for k_view, _ in per_layer]
            )
            memory_obj.tensor.copy_(stacked, non_blocking=True)
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
        else:
            k_stacked = torch.stack(
                [k_view.index_select(0, slot_slice) for k_view, _ in per_layer]
            )
            v_stacked = torch.stack(
                [
                    v_view.index_select(0, slot_slice)  # type: ignore[union-attr]
                    for _, v_view in per_layer
                ]
            )
            memory_obj.tensor.copy_(
                torch.stack([k_stacked, v_stacked]), non_blocking=True
            )

        if memory_obj.tensor.device.type != "musa":
            # Buffer is on a non-MUSA device (e.g. pinned host); make sure
            # the async stack/copy lands before we return.
            torch.musa.synchronize()  # type: ignore[attr-defined]

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Memory-object layout produced/consumed by this connector.

        Non-MLA: ``[2, num_layers, num_tokens, hidden_dim_size]``
        (``MemoryFormat.KV_2LTD``).

        MLA: ``[num_layers, num_tokens, hidden_dim_size]``
        (``MemoryFormat.KV_MLA_FMT``).
        """
        if self.use_mla:
            return torch.Size([self.num_layers, num_tokens, self.hidden_dim_size])
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    def _validate_memory_format(self, memory_obj: MemoryObj) -> None:
        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "Memory object must be KV_MLA_FMT for SGLangMUSAConnector "
                    f"(MLA mode); got {memory_obj.metadata.fmt}."
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "Memory object must be KV_2LTD for SGLangMUSAConnector "
                    f"(non-MLA mode); got {memory_obj.metadata.fmt}."
                )


class SGLangLayerwiseMUSAConnector(GPUConnectorInterface):
    """Layerwise paged KV connector for SGLang on MUSA.

    Pure-torch mirror of :class:`SGLangLayerwiseGPUConnector`. Drives the
    same generator contract (``batched_to_gpu`` yields ``num_layers + 2``
    times; ``batched_from_gpu`` yields ``num_layers + 1`` times) but with
    per-layer ``index_copy_`` / ``index_select`` instead of
    ``lmc_ops.single_layer_kv_transfer_sgl``.

    The memory object layout is **per-layer ``KV_T2D``**:
    ``[num_tokens, 2, hidden_dim]``. Each yield receives a list of
    memory objects (one per chunk in the batched call) for the current
    layer.

    MLA is not implemented yet — matching the upstream
    ``SGLangLayerwiseGPUConnector`` which carries a ``# TODO: support MLA``
    marker. Setting ``use_mla=True`` raises ``NotImplementedError`` at
    construction so the gap is loud.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ) -> None:
        if kwargs.get("use_mla", False):
            raise NotImplementedError(
                "SGLangLayerwiseMUSAConnector does not yet support MLA. "
                "Use SGLangMUSAConnector (non-layerwise) for MLA on MUSA."
            )

        for required in ("dtype", "device"):
            if required not in kwargs:
                raise ValueError(
                    f"'{required}' must be provided to SGLangLayerwiseMUSAConnector."
                )

        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]
        self.use_mla = False
        self.num_kv_cache = num_layers * 2

        # ``torch.musa.Stream`` only exists on actual MUSA-built torch.
        # During construction on a CPU box (tests, dispatch checks) the
        # attribute is unavailable; we lazily create the streams the first
        # time we actually need them.
        self._load_stream: Optional[object] = None
        self._store_stream: Optional[object] = None

        self.kvcaches = None
        self.page_buffer_size = 0
        self._metadata_initialized = False
        self._sglang_kvcaches: object = None

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        chunk_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "SGLangLayerwiseMUSAConnector":
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            dtype=dtype if dtype is not None else metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _ensure_streams(self) -> None:
        """Lazily create MUSA streams on first transfer (not at __init__)."""
        if self._load_stream is None:
            self._load_stream = torch.musa.Stream()  # type: ignore[attr-defined]
            self._store_stream = torch.musa.Stream()  # type: ignore[attr-defined]

    def _initialize_metadata(self, kvcaches) -> torch.device:
        if self._metadata_initialized:
            return self._kv_device

        self.gpu_kv_format, normalized = normalize_kv_and_discover_format(
            kvcaches, EngineType.SGLANG
        )
        self.page_buffer_size = get_page_buffer_size(normalized, self.gpu_kv_format)
        self._sglang_kvcaches = normalized

        device = _first_tensor_from_discoverable(normalized).device
        assert device.type == "musa", (
            f"SGLangLayerwiseMUSAConnector requires MUSA tensors; got '{device.type}'."
        )
        self._kv_device = device
        self._metadata_initialized = True
        return device

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError(
            "SGLangLayerwiseMUSAConnector uses batched_to_gpu(generator)."
        )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError(
            "SGLangLayerwiseMUSAConnector uses batched_from_gpu(generator)."
        )

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """Generator: CPU/host memory_objs (KV_T2D) -> MUSA paged caches.

        Yields ``num_layers + 2`` times (matching upstream): one yield
        before any layer for setup, ``num_layers`` yields receiving the
        per-layer chunk list via ``.send()``, and one final yield for sync.
        """
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        kvcaches = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        device = self._initialize_metadata(kvcaches)
        self._ensure_streams()
        assert self._load_stream is not None
        assert self._sglang_kvcaches is not None

        per_layer = _sglang_per_layer_kv_views(
            self._sglang_kvcaches, self.hidden_dim_size, use_mla=False
        )

        current_stream = torch.musa.current_stream()  # type: ignore[attr-defined]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield

            if sync:
                current_stream.wait_stream(self._load_stream)

            k_view, v_view = per_layer[layer_id]
            assert v_view is not None
            with torch.musa.stream(self._load_stream):  # type: ignore[attr-defined]
                for s, e, mem in zip(starts, ends, memory_objs_layer, strict=False):
                    if mem.tensor is None or (e - s) <= 0:
                        continue
                    # KV_T2D: [num_tokens, 2, hidden_dim].
                    src = _to_musa(mem.tensor, device)
                    sl = _to_musa(slot_mapping[s:e], device)
                    k_view.index_copy_(0, sl, src[:, 0, :])
                    v_view.index_copy_(0, sl, src[:, 1, :])

        if sync:
            current_stream.wait_stream(self._load_stream)
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: List[List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """Generator: MUSA paged caches -> CPU/host memory_objs (KV_T2D).

        Yields ``num_layers + 1`` times (matching upstream).
        """
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        kvcaches = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        device = self._initialize_metadata(kvcaches)
        self._ensure_streams()
        assert self._store_stream is not None
        assert self._sglang_kvcaches is not None

        per_layer = _sglang_per_layer_kv_views(
            self._sglang_kvcaches, self.hidden_dim_size, use_mla=False
        )

        current_stream = torch.musa.current_stream()  # type: ignore[attr-defined]

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]

            k_view, v_view = per_layer[layer_id]
            assert v_view is not None
            with torch.musa.stream(self._store_stream):  # type: ignore[attr-defined]
                for s, e, mem in zip(starts, ends, memory_objs_layer, strict=False):
                    if mem.tensor is None or (e - s) <= 0:
                        continue
                    sl = _to_musa(slot_mapping[s:e], device)
                    k_chunk = k_view.index_select(0, sl)
                    v_chunk = v_view.index_select(0, sl)
                    # KV_T2D destination: [num_tokens, 2, hidden_dim].
                    mem.tensor[:, 0, :].copy_(k_chunk, non_blocking=True)
                    mem.tensor[:, 1, :].copy_(v_chunk, non_blocking=True)

            yield

        if sync:
            current_stream.wait_stream(self._store_stream)
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Per-layer memory-object layout: ``[num_tokens, 2, hidden_dim_size]``."""
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
