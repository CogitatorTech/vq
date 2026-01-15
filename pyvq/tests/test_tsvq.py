import numpy as np
import pytest
import pyvq


def test_tsvq_creation():
    """Test TSVQ creation."""
    training = np.random.rand(100, 8).astype(np.float32)
    tsvq = pyvq.TSVQ(
        training_data=training,
        max_depth=5
    )
    assert tsvq.dim == 8


def test_tsvq_with_distance():
    """Test TSVQ with explicit distance metric."""
    training = np.random.rand(50, 6).astype(np.float32)
    tsvq = pyvq.TSVQ(
        training_data=training,
        max_depth=3,
        distance=pyvq.Distance.squared_euclidean()
    )
    assert tsvq.dim == 6


def test_tsvq_quantize():
    """Test TSVQ quantize method."""
    training = np.random.rand(100, 10).astype(np.float32)
    tsvq = pyvq.TSVQ(training, max_depth=4)
    
    vector = training[0].copy()
    codes = tsvq.quantize(vector)
    assert isinstance(codes, np.ndarray)
    assert codes.dtype == np.float32
    assert len(codes) == 10


def test_tsvq_dequantize():
    """Test TSVQ dequantize method."""
    training = np.random.rand(100, 6).astype(np.float32)
    tsvq = pyvq.TSVQ(training, max_depth=3)
    
    vector = training[0].copy()
    codes = tsvq.quantize(vector)
    reconstructed = tsvq.dequantize(codes)
    
    assert isinstance(reconstructed, np.ndarray)
    assert reconstructed.dtype == np.float32
    assert len(reconstructed) == 6


def test_tsvq_repr():
    """Test __repr__."""
    training = np.random.rand(50, 8).astype(np.float32)
    tsvq = pyvq.TSVQ(training, 3)
    assert "TSVQ" in repr(tsvq)
    assert "dim=8" in repr(tsvq)


def test_tsvq_empty_training():
    """Test that empty training data raises ValueError."""
    training = np.array([]).reshape(0, 8).astype(np.float32)
    with pytest.raises(ValueError, match="empty"):
        pyvq.TSVQ(training, 3)


def test_tsvq_reconstruction_quality():
    """Test that TSVQ reconstruction is reasonable."""
    # Create clustered data
    training = np.vstack([
        np.random.randn(50, 4) + np.array([0, 0, 0, 0]),
        np.random.randn(50, 4) + np.array([10, 10, 10, 10]),
    ]).astype(np.float32)
    training = np.ascontiguousarray(training)
    tsvq = pyvq.TSVQ(training, max_depth=3)
    
    # Quantize a sample from the first cluster
    sample = training[0]
    codes = tsvq.quantize(sample)
    recon = tsvq.dequantize(codes)
    
    # The reconstruction should be closer to the cluster center (0,0,0,0) than to (10,10,10,10)
    dist_to_origin = np.linalg.norm(recon)
    dist_to_ten = np.linalg.norm(recon - 10)
    assert dist_to_origin < dist_to_ten


if __name__ == "__main__":
    pytest.main()
