import numpy as np
import pytest
import pyvq


@pytest.fixture
def binary_quantizer():
    """Fixture to create a BinaryQuantizer instance for testing."""
    bq = pyvq.BinaryQuantizer(0.5, 0, 1)
    return bq


def test_quantize_single_value(binary_quantizer):
    """Test quantization of a single value."""
    result_bytes = binary_quantizer.quantize([0.2])
    result_array = np.frombuffer(result_bytes, dtype=np.uint8)
    assert np.array_equal(result_array, np.array([0], dtype=np.uint8))


def test_quantize_multiple_values(binary_quantizer):
    """Test quantization of multiple values."""
    result_bytes = binary_quantizer.quantize([0.2, 1.0, 0.5])
    result_array = np.frombuffer(result_bytes, dtype=np.uint8)
    assert np.array_equal(result_array, np.array([0, 1, 1], dtype=np.uint8))


def test_quantize_empty_list(binary_quantizer):
    """Test quantization of an empty list."""
    result_bytes = binary_quantizer.quantize([])
    assert result_bytes == b''


def test_quantize_large_values(binary_quantizer):
    """Test quantization of large values."""
    result_bytes = binary_quantizer.quantize([100.0, 200.0])
    result_array = np.frombuffer(result_bytes, dtype=np.uint8)
    assert np.array_equal(result_array, np.array([1, 1], dtype=np.uint8))


if __name__ == "__main__":
    pytest.main()
