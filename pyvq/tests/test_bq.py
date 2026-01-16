import numpy as np
import pytest
import pyvq


@pytest.fixture
def binary_quantizer():
    """Fixture to create a BinaryQuantizer instance for testing."""
    return pyvq.BinaryQuantizer(0.5, 0, 1)


def test_quantize_single_value(binary_quantizer):
    """Test quantization of a single value."""
    data = np.array([0.2], dtype=np.float32)
    result = binary_quantizer.quantize(data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.array([0], dtype=np.uint8))


def test_quantize_multiple_values(binary_quantizer):
    """Test quantization of multiple values."""
    data = np.array([0.2, 1.0, 0.5], dtype=np.float32)
    result = binary_quantizer.quantize(data)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.array([0, 1, 1], dtype=np.uint8))


def test_quantize_empty_array(binary_quantizer):
    """Test quantization of an empty array."""
    data = np.array([], dtype=np.float32)
    result = binary_quantizer.quantize(data)
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_quantize_large_values(binary_quantizer):
    """Test quantization of large values."""
    data = np.array([100.0, 200.0], dtype=np.float32)
    result = binary_quantizer.quantize(data)
    np.testing.assert_array_equal(result, np.array([1, 1], dtype=np.uint8))


def test_dequantize(binary_quantizer):
    """Test dequantization."""
    codes = np.array([0, 1, 1], dtype=np.uint8)
    result = binary_quantizer.dequantize(codes)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array([0.0, 1.0, 1.0], dtype=np.float32))


def test_properties():
    """Test BinaryQuantizer properties."""
    bq = pyvq.BinaryQuantizer(0.5, 10, 20)
    assert bq.threshold == 0.5
    assert bq.low == 10
    assert bq.high == 20


def test_repr():
    """Test __repr__."""
    bq = pyvq.BinaryQuantizer(0.5, 0, 1)
    assert "BinaryQuantizer" in repr(bq)
    assert "0.5" in repr(bq)


def test_invalid_parameters():
    """Test that invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        pyvq.BinaryQuantizer(0.5, 5, 5)  # low == high
    with pytest.raises(ValueError):
        pyvq.BinaryQuantizer(0.5, 6, 5)  # low > high


def test_nan_threshold_rejected():
    """Test that NaN threshold raises ValueError."""
    import math
    with pytest.raises(ValueError):
        pyvq.BinaryQuantizer(float('nan'), 0, 1)


if __name__ == "__main__":
    pytest.main()
