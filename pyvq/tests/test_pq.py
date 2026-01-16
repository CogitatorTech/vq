import numpy as np
import pytest
import pyvq


def test_product_quantizer_creation():
    """Test ProductQuantizer creation."""
    training = np.random.rand(100, 16).astype(np.float32)
    pq = pyvq.ProductQuantizer(
        training_data=training,
        num_subspaces=4,
        num_centroids=8,
        max_iters=10,
        seed=42
    )
    assert pq.dim == 16
    assert pq.num_subspaces == 4
    assert pq.sub_dim == 4


def test_product_quantizer_with_distance():
    """Test ProductQuantizer with explicit distance metric."""
    training = np.random.rand(50, 8).astype(np.float32)
    pq = pyvq.ProductQuantizer(
        training_data=training,
        num_subspaces=2,
        num_centroids=4,
        distance=pyvq.Distance.euclidean()
    )
    assert pq.dim == 8
    assert pq.num_subspaces == 2


def test_product_quantizer_quantize():
    """Test ProductQuantizer quantize method."""
    training = np.random.rand(100, 12).astype(np.float32)
    pq = pyvq.ProductQuantizer(
        training_data=training,
        num_subspaces=3,
        num_centroids=8,
        seed=42
    )
    
    vector = training[0].copy()
    codes = pq.quantize(vector)
    assert isinstance(codes, np.ndarray)
    assert codes.dtype == np.float32
    assert len(codes) == 12


def test_product_quantizer_dequantize():
    """Test ProductQuantizer dequantize method."""
    training = np.random.rand(100, 8).astype(np.float32)
    pq = pyvq.ProductQuantizer(
        training_data=training,
        num_subspaces=2,
        num_centroids=4,
        seed=42
    )
    
    vector = training[0].copy()
    codes = pq.quantize(vector)
    reconstructed = pq.dequantize(codes)
    
    assert isinstance(reconstructed, np.ndarray)
    assert reconstructed.dtype == np.float32
    assert len(reconstructed) == 8


def test_product_quantizer_repr():
    """Test __repr__."""
    training = np.random.rand(50, 8).astype(np.float32)
    pq = pyvq.ProductQuantizer(training, 2, 4)
    assert "ProductQuantizer" in repr(pq)
    assert "dim=8" in repr(pq)


def test_product_quantizer_empty_training():
    """Test that empty training data raises ValueError."""
    training = np.array([]).reshape(0, 8).astype(np.float32)
    with pytest.raises(ValueError, match="empty"):
        pyvq.ProductQuantizer(training, 2, 4)


def test_product_quantizer_invalid_subspaces():
    """Test that invalid num_subspaces raises ValueError."""
    training = np.random.rand(50, 7).astype(np.float32)
    # 7 is not divisible by 2
    with pytest.raises(ValueError):
        pyvq.ProductQuantizer(training, 2, 4)


def test_dimension_mismatch():
    """Test that quantizing wrong dimension vector raises ValueError."""
    training = np.random.rand(50, 8).astype(np.float32)
    pq = pyvq.ProductQuantizer(training, 2, 4)
    
    wrong_dim_vector = np.random.rand(10).astype(np.float32)  # dim 10 != 8
    with pytest.raises(ValueError, match="Dimension mismatch"):
        pq.quantize(wrong_dim_vector)


if __name__ == "__main__":
    pytest.main()
