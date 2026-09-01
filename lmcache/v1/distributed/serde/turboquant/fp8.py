# SPDX-License-Identifier: Apache-2.0
"""Portable FP8 E4M3B15 conversion used by TurboQuant.

The CUDA Triton backend supplies an inline-assembly implementation for
``float8e4b15``.  Some accelerator Triton backends (including the MUSA
backend used by LMCache CI) expose the type but do not provide the custom
conversion hook.  This module keeps the serialized representation identical
by implementing the conversion with ordinary PyTorch arithmetic.
"""

# Standard

# Third Party
import torch


def _to_fp16_rtz(values: torch.Tensor) -> torch.Tensor:
    """Convert floating-point values to FP16 with round-toward-zero semantics."""
    rounded = values.to(dtype=torch.float16)
    if values.dtype != torch.float32:
        return rounded

    rounded_fp32 = rounded.to(dtype=torch.float32)
    finite_overshoot = torch.isfinite(values) & (rounded_fp32.abs() > values.abs())
    toward_zero = torch.nextafter(rounded, torch.zeros_like(rounded))
    return torch.where(finite_overshoot, toward_zero, rounded)


def _is_missing_accelerator_op(exc: RuntimeError) -> bool:
    """Return whether an error reports an unavailable accelerator operator."""
    message = str(exc).lower()
    return (
        ("could not run" in message and "privateuse1" in message)
        or "not implemented for" in message
        or "not implemented on" in message
        or "does not have an implementation" in message
    )


def _encode_fp8_e4b15_impl(values: torch.Tensor) -> torch.Tensor:
    """Encode values as the E4M3 format with exponent bias 15.

    Args:
        values: Floating-point tensor on CPU or an accelerator device.

    Returns:
        A contiguous ``torch.uint8`` tensor with the same shape and device.

    Notes:
        E4M3B15 is bit-compatible with E4M3FN after scaling by ``2**8``.
        The arithmetic implementation below avoids relying on an accelerator
        having a native float8 dtype conversion.
    """
    # Triton's E4M3B15 path downcasts FP32 through FP16 with RTZ semantics.
    # Preserve that intermediate rounding so serialized bytes remain compatible
    # across CUDA and MUSA around FP8 bucket boundaries.
    x = _to_fp16_rtz(values).to(dtype=torch.float32)
    nan_mask = torch.isnan(x)
    sign = torch.signbit(x)
    magnitude = torch.where(nan_mask, torch.zeros_like(x), x.abs())
    magnitude = magnitude.clamp(max=1.75)
    scaled = magnitude * 256.0

    # E4M3 normal numbers have an exponent range of -6..7 after scaling.
    fraction, exponent = torch.frexp(scaled)
    exponent = exponent.to(dtype=torch.int32)
    normal_fraction = (fraction * 2.0 - 1.0) * 8.0
    rounded_fraction = torch.round(normal_fraction).to(dtype=torch.int32)
    carry = rounded_fraction >= 8
    rounded_fraction = torch.where(
        carry, torch.zeros_like(rounded_fraction), rounded_fraction
    )
    normal_exponent = exponent + 6 + carry.to(dtype=torch.int32)
    normal = (scaled > 0) & (normal_exponent >= 1)

    # Subnormal unit for E4M3B15 is 2**-17.  Values rounding to eight units
    # become the smallest normal value (exponent=1, mantissa=0).
    subnormal = torch.round(scaled * 512.0).to(dtype=torch.int32)
    subnormal_carry = subnormal >= 8
    subnormal = subnormal.clamp(min=0, max=7)

    encoded_exponent = torch.where(
        normal,
        normal_exponent,
        torch.where(
            subnormal_carry,
            torch.ones_like(normal_exponent),
            torch.zeros_like(normal_exponent),
        ),
    )
    encoded_mantissa = torch.where(
        normal,
        rounded_fraction,
        torch.where(subnormal_carry, torch.zeros_like(subnormal), subnormal),
    )

    # Clamping also handles the E=15 boundary.  E4M3B15 reserves no NaN
    # value for finite inputs; 0x7f is retained for NaN to match hardware
    # conversion behavior while finite values top out at 0x7e (1.75).
    encoded_exponent = encoded_exponent.clamp(min=0, max=15)
    encoded_mantissa = encoded_mantissa.clamp(min=0, max=7)
    result = (
        (sign.to(dtype=torch.uint8) << 7)
        | (encoded_exponent.to(dtype=torch.uint8) << 3)
        | encoded_mantissa.to(dtype=torch.uint8)
    )
    result = torch.where(
        nan_mask,
        torch.full_like(result, 0x7F),
        result,
    )
    return result.contiguous()


def encode_fp8_e4b15(values: torch.Tensor) -> torch.Tensor:
    """Encode values as E4M3B15, falling back to host arithmetic if needed.

    Args:
        values: Floating-point tensor on CPU or an accelerator device.

    Returns:
        A contiguous ``torch.uint8`` tensor with the same shape and device.

    Raises:
        RuntimeError: If neither device nor CPU arithmetic can process the
            input tensor.
    """
    try:
        return _encode_fp8_e4b15_impl(values)
    except NotImplementedError:
        if values.device.type == "cpu":
            raise
        # A few accelerator builds do not expose ``frexp``.  The host path is
        # a correctness fallback; it is selected only for those builds and
        # still returns bytes on the original device.
        return _encode_fp8_e4b15_impl(values.cpu()).to(device=values.device)
    except RuntimeError as exc:
        if values.device.type == "cpu" or not _is_missing_accelerator_op(exc):
            raise
        return _encode_fp8_e4b15_impl(values.cpu()).to(device=values.device)


def decode_fp8_e4b15(values: torch.Tensor) -> torch.Tensor:
    """Decode E4M3B15 bytes to float32 using portable tensor operations.

    Args:
        values: ``torch.uint8`` tensor containing E4M3B15 bytes.

    Returns:
        Float32 tensor on the same device as ``values``.
    """
    raw = values.to(dtype=torch.int32)
    sign = (raw >> 7) & 1
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    exponent_f = exponent.to(dtype=torch.float32)
    mantissa_f = mantissa.to(dtype=torch.float32)
    normal = (1.0 + mantissa_f / 8.0) * torch.pow(
        torch.full_like(exponent_f, 2.0), exponent_f - 15.0
    )
    subnormal = mantissa_f * (2.0**-17)
    decoded = torch.where(exponent == 0, subnormal, normal)
    # E=15,m=7 is the software representation of the finite endpoint.
    decoded = torch.where(
        exponent == 15,
        decoded.clamp(max=1.75),
        decoded,
    )
    return torch.where(sign != 0, -decoded, decoded)
