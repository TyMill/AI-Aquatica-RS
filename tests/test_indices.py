import numpy as np
import pytest

from ai_aquatica_rs.remote_sensing.indices import compute_indices, mndwi, ndti, ndvi, ndwi


def test_index_functions_return_expected_values() -> None:
    nir = np.array([0.8, 0.6], dtype=float)
    red = np.array([0.2, 0.3], dtype=float)
    green = np.array([0.3, 0.2], dtype=float)
    swir = np.array([0.1, 0.25], dtype=float)

    assert np.allclose(ndvi(nir, red), np.array([0.6, 1.0 / 3.0]))
    assert np.allclose(ndwi(green, nir), np.array([-0.45454545, -0.5]))
    assert np.allclose(mndwi(green, swir), np.array([0.5, -0.11111111]))
    assert np.allclose(ndti(red, green), np.array([-0.2, 0.2]))


def test_safe_division_returns_zero_for_zero_denominator() -> None:
    result = ndvi(np.array([0.0, 1.0]), np.array([0.0, -1.0]))
    assert np.allclose(result, np.array([0.0, 0.0]))
    assert np.isfinite(result).all()


def test_shape_mismatch_raises_explicit_error() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        ndwi(np.array([0.2, 0.3]), np.array([0.2]))


def test_compute_indices_batch_helper_is_complete_and_numeric() -> None:
    bands = {
        "nir": np.array([0.8, 0.5], dtype=np.float32),
        "red": np.array([0.2, 0.1], dtype=np.float32),
        "green": np.array([0.3, 0.2], dtype=np.float32),
        "swir": np.array([0.1, 0.05], dtype=np.float32),
    }

    results = compute_indices(bands)

    assert set(results) == {"ndvi", "ndwi", "mndwi", "ndti"}
    for values in results.values():
        assert np.issubdtype(values.dtype, np.floating)
        assert np.isfinite(values).all()


def test_compute_indices_requires_all_expected_bands() -> None:
    with pytest.raises(ValueError, match="Missing required band"):
        compute_indices({"nir": np.array([0.1]), "red": np.array([0.1]), "green": np.array([0.1])})
