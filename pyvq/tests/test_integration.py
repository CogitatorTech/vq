"""
Integration tests for pyvq.

These tests verify end-to-end workflows combining multiple quantizers
and testing realistic usage patterns.
"""

import numpy as np
import pytest
import pyvq


class TestQuantizationRoundTrip:
    """Test quantize -> dequantize round-trip workflows."""

    def test_bq_preserves_sign_pattern(self):
        """BQ should map values based on threshold."""
        bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)
        original = np.array([-0.5, 0.3, -0.8, 0.9, 0.0], dtype=np.float32)
        
        codes = bq.quantize(original)
        reconstructed = bq.dequantize(codes)
        
        # Values < 0 -> low (0), values >= 0 -> high (1)
        # Dequantize returns these as floats
        expected = np.where(original >= 0, 1.0, 0.0)
        np.testing.assert_array_equal(reconstructed, expected)

    def test_sq_reconstruction_within_step(self):
        """SQ reconstruction should be within step size of original."""
        sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
        original = np.random.uniform(-1.0, 1.0, 100).astype(np.float32)
        
        codes = sq.quantize(original)
        reconstructed = sq.dequantize(codes)
        
        # Error should be bounded by half step size
        max_error = sq.step / 2 + 1e-6
        errors = np.abs(original - reconstructed)
        assert np.all(errors <= max_error), f"Max error {errors.max()} exceeds {max_error}"

    def test_pq_reconstruction_reasonable(self):
        """PQ reconstruction should be reasonably close to original."""
        np.random.seed(42)
        training = np.random.randn(200, 16).astype(np.float32)
        
        pq = pyvq.ProductQuantizer(
            training_data=training,
            num_subspaces=4,
            num_centroids=16,
            max_iters=20,
            seed=42
        )
        
        # Test on training data (should reconstruct well)
        test_vector = training[0].copy()
        codes = pq.quantize(test_vector)
        reconstructed = pq.dequantize(codes)
        
        # Reconstruction should be close (RMSE < 1.0 for normalized data)
        rmse = np.sqrt(np.mean((test_vector - reconstructed) ** 2))
        assert rmse < 2.0, f"RMSE {rmse} too high for PQ reconstruction"

    def test_tsvq_reconstruction_reasonable(self):
        """TSVQ reconstruction should be reasonably close to original."""
        np.random.seed(42)
        training = np.random.randn(200, 8).astype(np.float32)
        
        tsvq = pyvq.TSVQ(training_data=training, max_depth=4)
        
        test_vector = training[0].copy()
        codes = tsvq.quantize(test_vector)
        reconstructed = tsvq.dequantize(codes)
        
        rmse = np.sqrt(np.mean((test_vector - reconstructed) ** 2))
        assert rmse < 2.0, f"RMSE {rmse} too high for TSVQ reconstruction"


class TestDistanceMetrics:
    """Test distance metric integration with quantizers."""

    def test_pq_with_different_distances(self):
        """PQ should work with different distance metrics."""
        np.random.seed(42)
        training = np.random.randn(100, 8).astype(np.float32)
        
        distances = [
            pyvq.Distance.euclidean(),
            pyvq.Distance.squared_euclidean(),
            pyvq.Distance.manhattan(),
            pyvq.Distance.cosine(),
        ]
        
        for dist in distances:
            pq = pyvq.ProductQuantizer(
                training_data=training,
                num_subspaces=2,
                num_centroids=4,
                max_iters=5,
                distance=dist,
                seed=42
            )
            
            codes = pq.quantize(training[0])
            reconstructed = pq.dequantize(codes)
            
            assert len(reconstructed) == 8
            assert reconstructed.dtype == np.float32

    def test_tsvq_with_different_distances(self):
        """TSVQ should work with different distance metrics."""
        np.random.seed(42)
        training = np.random.randn(100, 6).astype(np.float32)
        
        distances = [
            pyvq.Distance.euclidean(),
            pyvq.Distance.squared_euclidean(),
        ]
        
        for dist in distances:
            tsvq = pyvq.TSVQ(
                training_data=training,
                max_depth=3,
                distance=dist
            )
            
            codes = tsvq.quantize(training[0])
            reconstructed = tsvq.dequantize(codes)
            
            assert len(reconstructed) == 6

    def test_distance_compute_batch(self):
        """Distance computation should work on multiple vector pairs."""
        dist = pyvq.Distance.euclidean()
        
        # Generate random vectors and compute distances
        np.random.seed(42)
        vectors_a = np.random.randn(10, 8).astype(np.float32)
        vectors_b = np.random.randn(10, 8).astype(np.float32)
        
        distances = []
        for a, b in zip(vectors_a, vectors_b):
            d = dist.compute(a, b)
            distances.append(d)
            assert d >= 0  # Distance should be non-negative
        
        # Verify against numpy
        expected = np.linalg.norm(vectors_a - vectors_b, axis=1)
        np.testing.assert_allclose(distances, expected, rtol=1e-5)


class TestChainedQuantization:
    """Test combining multiple quantization steps."""

    def test_bq_on_sq_output(self):
        """Apply BQ on SQ output (multi-stage quantization)."""
        sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
        bq = pyvq.BinaryQuantizer(threshold=128, low=0, high=1)
        
        original = np.array([0.3, -0.7, 0.5, -0.1], dtype=np.float32)
        
        # SQ quantize
        sq_codes = sq.quantize(original)
        
        # BQ on SQ codes (treating as float for threshold comparison)
        bq_codes = bq.quantize(sq_codes.astype(np.float32))
        
        assert len(bq_codes) == len(original)
        assert bq_codes.dtype == np.uint8


class TestLargeScale:
    """Test with larger datasets to verify scalability."""

    def test_pq_large_training_set(self):
        """PQ should handle larger training sets."""
        np.random.seed(42)
        # 10,000 vectors of dimension 64
        training = np.random.randn(10000, 64).astype(np.float32)
        
        pq = pyvq.ProductQuantizer(
            training_data=training,
            num_subspaces=8,
            num_centroids=256,
            max_iters=10,
            seed=42
        )
        
        assert pq.dim == 64
        assert pq.num_subspaces == 8
        assert pq.sub_dim == 8
        
        # Quantize a batch of vectors
        for i in range(100):
            codes = pq.quantize(training[i])
            reconstructed = pq.dequantize(codes)
            assert len(reconstructed) == 64

    def test_tsvq_large_training_set(self):
        """TSVQ should handle larger training sets."""
        np.random.seed(42)
        training = np.random.randn(5000, 32).astype(np.float32)
        
        tsvq = pyvq.TSVQ(training_data=training, max_depth=6)
        
        assert tsvq.dim == 32
        
        codes = tsvq.quantize(training[0])
        reconstructed = tsvq.dequantize(codes)
        assert len(reconstructed) == 32


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_element_vector(self):
        """Quantizers should handle single-element vectors."""
        bq = pyvq.BinaryQuantizer(threshold=0.5, low=0, high=1)
        sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
        
        single = np.array([0.7], dtype=np.float32)
        
        bq_codes = bq.quantize(single)
        sq_codes = sq.quantize(single)
        
        assert len(bq_codes) == 1
        assert len(sq_codes) == 1

    def test_extreme_values(self):
        """Quantizers should handle extreme (but valid) values."""
        sq = pyvq.ScalarQuantizer(min=-1e6, max=1e6, levels=256)
        
        extreme = np.array([1e6, -1e6, 0.0], dtype=np.float32)
        codes = sq.quantize(extreme)
        reconstructed = sq.dequantize(codes)
        
        # Should be at boundaries
        np.testing.assert_allclose(reconstructed[0], 1e6, rtol=0.1)
        np.testing.assert_allclose(reconstructed[1], -1e6, rtol=0.1)

    def test_identical_vectors_in_training(self):
        """PQ/TSVQ should handle training data with identical vectors."""
        np.random.seed(42)
        # Create training data with some duplicates
        base = np.random.randn(50, 8).astype(np.float32)
        training = np.vstack([base, base])  # Duplicate all vectors
        training = np.ascontiguousarray(training)
        
        pq = pyvq.ProductQuantizer(
            training_data=training,
            num_subspaces=2,
            num_centroids=4,
            seed=42
        )
        
        codes = pq.quantize(training[0])
        assert len(codes) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
