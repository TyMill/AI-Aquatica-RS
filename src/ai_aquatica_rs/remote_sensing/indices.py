"""Spectral indices for aquatic remote-sensing workflows.

The module provides normalized-difference indices frequently used for
vegetation vigor, surface water delineation, and turbidity proxies.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

ArrayLike = np.ndarray


def _as_float_array(name: str, values: ArrayLike) -> np.ndarray:
    """Convert an input band into a floating-point ``numpy`` array.

    Args:
        name: Logical name used in error messages.
        values: Band values to convert.

    Returns:
        A floating-point ``numpy`` array.

    Raises:
        ValueError: If ``values`` cannot be represented as numeric floats.
    """
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Band '{name}' must contain numeric values.") from exc


def _require_same_shape(
    left: np.ndarray, right: np.ndarray, *, left_name: str, right_name: str
) -> None:
    """Validate two arrays have identical shape.

    Raises:
        ValueError: If shapes differ.
    """
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch for '{left_name}' and '{right_name}': {left.shape} != {right.shape}."
        )


def _safe_normalized_difference(
    first: ArrayLike,
    second: ArrayLike,
    *,
    first_name: str,
    second_name: str,
) -> np.ndarray:
    """Compute ``(first - second) / (first + second)`` safely.

    Division-by-zero locations return ``0.0`` to keep outputs finite and
    numerically stable for downstream feature engineering.
    """
    first_arr = _as_float_array(first_name, first)
    second_arr = _as_float_array(second_name, second)
    _require_same_shape(first_arr, second_arr, left_name=first_name, right_name=second_name)

    denominator = first_arr + second_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            first_arr - second_arr,
            denominator,
            out=np.zeros_like(first_arr, dtype=float),
            where=denominator != 0,
        )


def ndvi(nir: ArrayLike, red: ArrayLike) -> np.ndarray:
    """Compute NDVI: ``(NIR - Red) / (NIR + Red)``.

    NDVI is a vegetation vigor proxy and is often included as a covariate in
    aquatic ecosystem modeling.
    """
    return _safe_normalized_difference(nir, red, first_name="nir", second_name="red")


def ndwi(green: ArrayLike, nir: ArrayLike) -> np.ndarray:
    """Compute NDWI (McFeeters): ``(Green - NIR) / (Green + NIR)``.

    NDWI emphasizes open-water signal relative to vegetation and soil.
    """
    return _safe_normalized_difference(green, nir, first_name="green", second_name="nir")


def mndwi(green: ArrayLike, swir: ArrayLike) -> np.ndarray:
    """Compute MNDWI: ``(Green - SWIR) / (Green + SWIR)``.

    MNDWI can better suppress built-up land and highlights water bodies in
    mixed landscapes.
    """
    return _safe_normalized_difference(green, swir, first_name="green", second_name="swir")


def ndti(red: ArrayLike, green: ArrayLike) -> np.ndarray:
    """Compute NDTI: ``(Red - Green) / (Red + Green)``.

    NDTI is used as a relative suspended sediment/turbidity indicator.
    """
    return _safe_normalized_difference(red, green, first_name="red", second_name="green")


def compute_indices(bands: Mapping[str, ArrayLike]) -> dict[str, np.ndarray]:
    """Compute all supported indices from a dictionary of spectral bands.

    Args:
        bands: Mapping containing ``nir``, ``red``, ``green``, and ``swir``
            arrays with matching shapes.

    Returns:
        Dictionary with keys ``ndvi``, ``ndwi``, ``mndwi``, and ``ndti``.

    Raises:
        ValueError: If any required band is missing or if inputs are invalid.
    """
    required = ("nir", "red", "green", "swir")
    missing = [name for name in required if name not in bands]
    if missing:
        raise ValueError(f"Missing required band(s): {missing}.")

    return {
        "ndvi": ndvi(bands["nir"], bands["red"]),
        "ndwi": ndwi(bands["green"], bands["nir"]),
        "mndwi": mndwi(bands["green"], bands["swir"]),
        "ndti": ndti(bands["red"], bands["green"]),
    }
