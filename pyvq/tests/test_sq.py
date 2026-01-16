import numpy as np
import pytest
import pyvq


@pytest.fixture
def scalar_quantizer():
    """
    Fixture to create a ScalarQuantizer instance for testing.

    For this test, we use:
      - min = -1.0
      - max = 1.0
      - levels = 5

    This defines quantization levels as:
      0 -> -1.0
      1 -> -0.5
      2 ->  0.0
      3 ->  0.5
      4 ->  1.0
    """
    return pyvq.ScalarQuantizer(-1.0, 1.0, 5)


def test_quantize_single_value(scalar_quantizer):
    """Test quantization of a single value."""
    # For x = -0.8:
    #   (x - min)/step = (-0.8 - (-1.0)) / 0.5 = 0.2/0.5 = 0.4, which rounds to 0.
    data = np.array([-0.8], dtype=np.float32)
    result = scalar_quantizer.quantize(data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.array([0], dtype=np.uint8))


def test_quantize_multiple_values(scalar_quantizer):
    """Test quantization of multiple values."""
    # Test input: [-1.2, -1.0, -0.8, -0.3, 0.0, 0.3, 0.6, 1.0, 1.2]
    # Expected behavior:
    #  - -1.2 clamps to -1.0 -> index 0.
    #  - -1.0 -> index 0.
    #  - -0.8 -> index 0.
    #  - -0.3 -> ((-0.3 - (-1.0))=0.7/0.5=1.4 rounds to 1).
    #  -  0.0 -> ((0.0 - (-1.0))=1.0/0.5=2.0 -> index 2).
    #  -  0.3 -> ((0.3 - (-1.0))=1.3/0.5=2.6 rounds to 3).
    #  -  0.6 -> ((0.6 - (-1.0))=1.6/0.5=3.2 rounds to 3).
    #  -  1.0 -> index 4.
    #  -  1.2 clamps to 1.0 -> index 4.
    data = np.array([-1.2, -1.0, -0.8, -0.3, 0.0, 0.3, 0.6, 1.0, 1.2], dtype=np.float32)
    result = scalar_quantizer.quantize(data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.array([0, 0, 0, 1, 2, 3, 3, 4, 4], dtype=np.uint8))


def test_quantize_empty_array(scalar_quantizer):
    """Test quantization of an empty array."""
    data = np.array([], dtype=np.float32)
    result = scalar_quantizer.quantize(data)
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_quantize_values_outside_range(scalar_quantizer):
    """Test quantization of values far outside the range."""
    data = np.array([-100.0, 100.0], dtype=np.float32)
    result = scalar_quantizer.quantize(data)
    np.testing.assert_array_equal(result, np.array([0, 4], dtype=np.uint8))


def test_dequantize(scalar_quantizer):
    """Test dequantization."""
    codes = np.array([0, 2, 4], dtype=np.uint8)
    result = scalar_quantizer.dequantize(codes)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_almost_equal(result, np.array([-1.0, 0.0, 1.0], dtype=np.float32))


def test_properties():
    """Test ScalarQuantizer properties."""
    sq = pyvq.ScalarQuantizer(-1.0, 1.0, 5)
    assert sq.min == -1.0
    assert sq.max == 1.0
    assert sq.levels == 5
    assert sq.step == 0.5


def test_repr():
    """Test __repr__."""
    sq = pyvq.ScalarQuantizer(-1.0, 1.0, 256)
    assert "ScalarQuantizer" in repr(sq)


def test_too_many_levels_rejected():
    """Test that levels > 256 raises ValueError."""
    with pytest.raises(ValueError):
        pyvq.ScalarQuantizer(-1.0, 1.0, 257)


def test_nan_min_max_rejected():
    """Test that NaN min/max raises ValueError."""
    with pytest.raises(ValueError):
        pyvq.ScalarQuantizer(float('nan'), 1.0, 256)
    with pytest.raises(ValueError):
        pyvq.ScalarQuantizer(-1.0, float('nan'), 256)


def test_infinity_rejected():
    """Test that Infinity min/max raises ValueError."""
    with pytest.raises(ValueError):
        pyvq.ScalarQuantizer(float('-inf'), 1.0, 256)
    with pytest.raises(ValueError):
        pyvq.ScalarQuantizer(-1.0, float('inf'), 256)


if __name__ == "__main__":
    pytest.main()
