"""Python replicas of the device's vector functions (benzene/dsp/vector_math/functions.h), operation by
operation in fp32 (every torch op on fp32 tensors rounds on its own, as the device's HVX IEEE ops), for the models'
device models."""

from __future__ import annotations

import math

import numpy as np
import torch

# vector_math::exp2's Taylor coefficients 1 / k! as the C++ float divisions (1.0f / 6, ...)
EXP2_TAYLOR = [np.float32(1.0), np.float32(1.0), np.float32(0.5)] + [np.float32(1.0) / np.float32(math.factorial(k))
                                                                     for k in range(3, 8)]


def f32(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def exp2_f32(y: torch.Tensor) -> torch.Tensor:
    """vector_math::exp2: |y| <= 100, n = trunc(y), 2^(y - n) by the degree-7 Taylor polynomial in (y - n) ln 2,
    the exponent add."""
    y = y.clamp(-100.0, 100.0)
    n = torch.trunc(y)
    fraction = (y - n) * f32(0.69314718055994531)
    power = torch.full_like(y, float(EXP2_TAYLOR[7]))
    for c in EXP2_TAYLOR[6::-1]:
        power = power * fraction
        power = power + f32(float(c))
    return (power.view(torch.int32) + (n.to(torch.int32) << 23)).view(torch.float32)


def reciprocal_f32(x: torch.Tensor) -> torch.Tensor:
    """vector_math::reciprocal: bit-trick start, 3 Newton steps."""
    r = (torch.tensor(0x7EF311C3, dtype=torch.int32) - x.view(torch.int32)).view(torch.float32)
    for _ in range(3):
        correction = f32(2.0) - x * r
        r = r * correction
    return r
