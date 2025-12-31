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
    result_bytes = scalar_quantizer.quantize([-0.8])
    result_array = np.frombuffer(result_bytes, dtype=np.uint8)
    expected = np.array([0], dtype=np.uint8)
    np.testing.assert_array_equal(result_array, expected)


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
    result_bytes = scalar_quantizer.quantize([-1.2, -1.0, -0.8, -0.3, 0.0, 0.3, 0.6, 1.0, 1.2])
    result_array = np.frombuffer(result_bytes, dtype=np.uint8)
    expected = np.array([0, 0, 0, 1, 2, 3, 3, 4, 4], dtype=np.uint8)
    np.testing.assert_array_equal(result_array, expected)


def test_quantize_empty_list(scalar_quantizer):
    """Test quantization of an empty list."""
    result_bytes = scalar_quantizer.quantize([])
    result_array = np.frombuffer(result_bytes, dtype=np.uint8)
    expected = np.array([], dtype=np.uint8)
    np.testing.assert_array_equal(result_array, expected)


def test_quantize_values_outside_range(scalar_quantizer):
    """Test quantization of values far outside the range."""
    # For input [-100.0, 100.0]:
    #  - -100.0 clamps to -1.0 -> index 0.
    #  - 100.0 clamps to 1.0 -> index 4.
    result_bytes = scalar_quantizer.quantize([-100.0, 100.0])
    result_array = np.frombuffer(result_bytes, dtype=np.uint8)
    expected = np.array([0, 4], dtype=np.uint8)
    np.testing.assert_array_equal(result_array, expected)


if __name__ == "__main__":
    pytest.main()
