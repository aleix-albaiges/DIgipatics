from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


MAGNIFICATION_TO_MPP = 10.0
DEFAULT_FALLBACK_SOURCE_MAGNIFICATION = 20.0


@dataclass(frozen=True)
class ResolutionPlan:
    level_index: int
    level_downsample: float
    output_downsample: float
    read_scale: float
    target_mpp: Optional[float]
    base_mpp: Optional[float]
    source_magnification: Optional[float]
    read_magnification: Optional[float]
    output_magnification: Optional[float]
    source: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "level_index": int(self.level_index),
            "level_downsample": float(self.level_downsample),
            "output_downsample": float(self.output_downsample),
            "read_scale": float(self.read_scale),
            "target_mpp": None if self.target_mpp is None else float(self.target_mpp),
            "base_mpp": None if self.base_mpp is None else float(self.base_mpp),
            "source_magnification": (
                None if self.source_magnification is None else float(self.source_magnification)
            ),
            "read_magnification": (
                None if self.read_magnification is None else float(self.read_magnification)
            ),
            "output_magnification": (
                None if self.output_magnification is None else float(self.output_magnification)
            ),
            "source": self.source,
        }


def level_downsamples(structure: Dict[str, object]) -> List[float]:
    level_shapes = structure["level_shapes"]
    base_h, base_w = level_shapes[0][0], level_shapes[0][1]
    factors: List[float] = []
    for shape in level_shapes:
        h, w = shape[0], shape[1]
        factors.append(float((base_h / h + base_w / w) / 2.0))
    return factors


def magnification_to_mpp(magnification: float) -> float:
    if magnification <= 0:
        raise ValueError("magnification must be positive")
    return MAGNIFICATION_TO_MPP / float(magnification)


def mpp_to_magnification(mpp: float) -> float:
    if mpp <= 0:
        raise ValueError("mpp must be positive")
    return MAGNIFICATION_TO_MPP / float(mpp)


def structure_mpp(structure: Dict[str, object]) -> Optional[float]:
    value = structure.get("mpp")
    if value is None:
        xs = structure.get("mpp_x")
        ys = structure.get("mpp_y")
        values = [float(v) for v in (xs, ys) if v is not None and float(v) > 0]
        if not values:
            return None
        return float(sum(values) / len(values))
    value_f = float(value)
    return value_f if value_f > 0 else None


def _choose_finer_or_equal_level(factors: List[float], desired_downsample: float) -> int:
    if desired_downsample <= 0:
        raise ValueError("desired_downsample must be positive")
    candidates = [
        (idx, factor)
        for idx, factor in enumerate(factors)
        if factor <= desired_downsample * (1.0 + 1e-6)
    ]
    if not candidates:
        return min(range(len(factors)), key=lambda idx: factors[idx])
    return max(candidates, key=lambda item: item[1])[0]


def choose_resolution_plan(
    structure: Dict[str, object],
    target_magnification: float,
    source_magnification: Optional[float] = None,
    target_mpp: Optional[float] = None,
) -> ResolutionPlan:
    factors = level_downsamples(structure)
    base_mpp = structure_mpp(structure)

    if target_mpp is not None and target_mpp <= 0:
        raise ValueError("target_mpp must be positive")
    if target_magnification <= 0 and target_mpp is None:
        raise ValueError("target_magnification must be positive when target_mpp is not set")

    if base_mpp is not None:
        resolved_target_mpp = (
            float(target_mpp) if target_mpp is not None else magnification_to_mpp(target_magnification)
        )
        desired_downsample = resolved_target_mpp / base_mpp
        level_index = _choose_finer_or_equal_level(factors, desired_downsample)
        level_downsample = factors[level_index]
        read_scale = level_downsample / desired_downsample
        source_mag = mpp_to_magnification(base_mpp)
        return ResolutionPlan(
            level_index=level_index,
            level_downsample=level_downsample,
            output_downsample=desired_downsample,
            read_scale=read_scale,
            target_mpp=resolved_target_mpp,
            base_mpp=base_mpp,
            source_magnification=source_mag,
            read_magnification=source_mag / level_downsample,
            output_magnification=mpp_to_magnification(resolved_target_mpp),
            source="tiff_mpp",
        )

    fallback_source_mag = float(source_magnification or DEFAULT_FALLBACK_SOURCE_MAGNIFICATION)
    desired_downsample = fallback_source_mag / float(target_magnification)
    level_index = _choose_finer_or_equal_level(factors, desired_downsample)
    level_downsample = factors[level_index]
    read_scale = level_downsample / desired_downsample
    resolved_target_mpp = (
        float(target_mpp) if target_mpp is not None else magnification_to_mpp(target_magnification)
    )
    return ResolutionPlan(
        level_index=level_index,
        level_downsample=level_downsample,
        output_downsample=desired_downsample,
        read_scale=read_scale,
        target_mpp=resolved_target_mpp,
        base_mpp=None,
        source_magnification=fallback_source_mag,
        read_magnification=fallback_source_mag / level_downsample,
        output_magnification=float(target_magnification),
        source="fallback_magnification",
    )


def resize_to_scale(image: np.ndarray, scale: float, is_mask: bool = False) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * float(scale))))
    new_h = max(1, int(round(h * float(scale))))
    if (new_h, new_w) == (h, w):
        return image

    import cv2

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    if not is_mask and scale > 1.0:
        interpolation = cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
