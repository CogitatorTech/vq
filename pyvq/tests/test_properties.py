"""
Property-based tests for pyvq using Hypothesis.

These tests verify invariants and properties that should hold
for all valid inputs, not just specific examples.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import pyvq


# =============================================================================
# Custom Strategies
# =============================================================================

def float32_arrays(min_size=1, max_size=100, min_value=-1e6, max_value=1e6):
    """Strategy for generating float32 numpy arrays."""
    return arrays(
        dtype=np.float32,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def valid_training_data(dim_range=(4, 32), n_samples_range=(50, 200)):
    """Strategy for generating valid 2D training data."""
    return st.builds(
        lambda dim, n_samples: np.random.randn(n_samples, dim).astype(np.float32),
        dim=st.integers(min_value=dim_range[0], max_value=dim_range[1]),
        n_samples=st.integers(min_value=n_samples_range[0], max_value=n_samples_range[1]),
    )


# =============================================================================
# BinaryQuantizer Properties
# =============================================================================

class TestBinaryQuantizerProperties:
    """Property-based tests for BinaryQuantizer."""

    @given(
        threshold=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        low=st.integers(min_value=0, max_value=127),
        high=st.integers(min_value=128, max_value=255),
        values=float32_arrays(min_size=1, max_size=50, min_value=-100, max_value=100),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_bq_output_is_binary(self, threshold, low, high, values):
        """BQ output should only contain low or high values."""
        assume(low < high)

        bq = pyvq.BinaryQuantizer(threshold=threshold, low=low, high=high)
        codes = bq.quantize(values)

        assert set(codes).issubset({low, high}), f"Codes contain unexpected values: {set(codes)}"

    @given(
        threshold=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        values=float32_arrays(min_size=1, max_size=50, min_value=-100, max_value=100),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_bq_dequantize_only_returns_low_or_high(self, threshold, values):
        """Dequantized values should only be low or high (as floats)."""
        low, high = 0, 1  # integers for BinaryQuantizer
        bq = pyvq.BinaryQuantizer(threshold=threshold, low=low, high=high)

        codes = bq.quantize(values)
        reconstructed = bq.dequantize(codes)

        # Dequantize returns floats
        assert set(reconstructed).issubset({float(low), float(high)})

    @given(values=float32_arrays(min_size=1, max_size=100, min_value=-10, max_value=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_bq_preserves_length(self, values):
        """Quantize/dequantize should preserve array length."""
        bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)  # integers

        codes = bq.quantize(values)
        reconstructed = bq.dequantize(codes)

        assert len(codes) == len(values)
        assert len(reconstructed) == len(values)


# =============================================================================
# ScalarQuantizer Properties
# =============================================================================

class TestScalarQuantizerProperties:
    """Property-based tests for ScalarQuantizer."""

    @given(
        min_val=st.floats(min_value=-1000, max_value=0, allow_nan=False, allow_infinity=False),
        max_val=st.floats(min_value=0.01, max_value=1000, allow_nan=False, allow_infinity=False),
        levels=st.integers(min_value=2, max_value=256),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_sq_codes_in_valid_range(self, min_val, max_val, levels):
        """SQ codes should be in range [0, levels-1]."""
        assume(min_val < max_val)

        sq = pyvq.ScalarQuantizer(min=min_val, max=max_val, levels=levels)
        values = np.random.uniform(min_val, max_val, 50).astype(np.float32)

        codes = sq.quantize(values)

        assert np.all(codes >= 0), "Codes contain negative values"
        assert np.all(codes < levels), f"Codes exceed max level {levels-1}"

    @given(
        min_val=st.floats(min_value=-100, max_value=-0.01, allow_nan=False, allow_infinity=False),
        max_val=st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
        levels=st.integers(min_value=2, max_value=256),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_sq_reconstruction_in_range(self, min_val, max_val, levels):
        """SQ reconstructed values should be within [min, max]."""
        assume(min_val < max_val)

        sq = pyvq.ScalarQuantizer(min=min_val, max=max_val, levels=levels)
        values = np.random.uniform(min_val, max_val, 50).astype(np.float32)

        codes = sq.quantize(values)
        reconstructed = sq.dequantize(codes)

        assert np.all(reconstructed >= min_val - 1e-5)
        assert np.all(reconstructed <= max_val + 1e-5)

    @given(levels=st.integers(min_value=2, max_value=256))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_sq_step_size_correct(self, levels):
        """SQ step size should equal (max - min) / (levels - 1)."""
        min_val, max_val = -1.0, 1.0
        sq = pyvq.ScalarQuantizer(min=min_val, max=max_val, levels=levels)

        expected_step = (max_val - min_val) / (levels - 1)
        assert abs(sq.step - expected_step) < 1e-6


# =============================================================================
# Distance Properties
# =============================================================================

@st.composite
def matched_float32_arrays(draw, min_size=2, max_size=50, min_value=-10, max_value=10):
    """Strategy for generating two float32 arrays with the same length."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    elements = st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False)
    a = draw(arrays(dtype=np.float32, shape=size, elements=elements))
    b = draw(arrays(dtype=np.float32, shape=size, elements=elements))
    return a, b


class TestDistanceProperties:
    """Property-based tests for Distance metrics."""

    @given(
        a=float32_arrays(min_size=2, max_size=50, min_value=-10, max_value=10),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_euclidean_self_distance_is_zero(self, a):
        """Euclidean distance of a vector to itself should be 0."""
        dist = pyvq.Distance.euclidean()
        result = dist.compute(a, a.copy())
        assert abs(result) < 1e-5, f"Self-distance is {result}, expected ~0"

    @given(pair=matched_float32_arrays())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_euclidean_is_symmetric(self, pair):
        """Euclidean distance should be symmetric: d(a,b) = d(b,a)."""
        a, b = pair

        dist = pyvq.Distance.euclidean()
        d_ab = dist.compute(a, b)
        d_ba = dist.compute(b, a)

        assert abs(d_ab - d_ba) < 1e-5, f"Asymmetry: d(a,b)={d_ab}, d(b,a)={d_ba}"

    @given(pair=matched_float32_arrays())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_euclidean_is_non_negative(self, pair):
        """Euclidean distance should be non-negative."""
        a, b = pair

        dist = pyvq.Distance.euclidean()
        result = dist.compute(a, b)

        assert result >= 0, f"Negative distance: {result}"

    @given(pair=matched_float32_arrays())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_manhattan_is_symmetric(self, pair):
        """Manhattan distance should be symmetric."""
        a, b = pair

        dist = pyvq.Distance.manhattan()
        d_ab = dist.compute(a, b)
        d_ba = dist.compute(b, a)

        assert abs(d_ab - d_ba) < 1e-5

    @given(
        a=float32_arrays(min_size=2, max_size=50, min_value=0.1, max_value=10),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_cosine_self_distance_is_zero(self, a):
        """Cosine distance of a vector to itself should be 0."""
        # Guarantee vectors are non-zero
        assume(np.linalg.norm(a) > 1e-3)

        dist = pyvq.Distance.cosine()
        result = dist.compute(a, a.copy())

        assert abs(result) < 1e-4, f"Cosine self-distance is {result}, expected ~0"


# =============================================================================
# ProductQuantizer Properties
# =============================================================================

class TestProductQuantizerProperties:
    """Property-based tests for ProductQuantizer."""

    @given(
        dim=st.integers(min_value=4, max_value=32).filter(lambda x: x % 2 == 0),
        n_samples=st.integers(min_value=50, max_value=150),
    )
    @settings(max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow])
    def test_pq_preserves_dimension(self, dim, n_samples):
        """PQ quantize/dequantize should preserve vector dimension."""
        np.random.seed(42)
        training = np.random.randn(n_samples, dim).astype(np.float32)

        pq = pyvq.ProductQuantizer(
            training_data=training,
            num_subspaces=2,
            num_centroids=4,
            max_iters=5,
            seed=42
        )

        test_vec = training[0].copy()
        codes = pq.quantize(test_vec)
        reconstructed = pq.dequantize(codes)

        assert len(reconstructed) == dim
        assert pq.dim == dim

    @given(
        num_subspaces=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=5, deadline=30000, suppress_health_check=[HealthCheck.too_slow])
    def test_pq_sub_dim_correct(self, num_subspaces):
        """PQ sub_dim should equal dim / num_subspaces."""
        dim = 16  # Fixed dim divisible by common subspace counts
        assume(dim % num_subspaces == 0)

        np.random.seed(42)
        training = np.random.randn(100, dim).astype(np.float32)

        pq = pyvq.ProductQuantizer(
            training_data=training,
            num_subspaces=num_subspaces,
            num_centroids=4,
            seed=42
        )

        expected_sub_dim = dim // num_subspaces
        assert pq.sub_dim == expected_sub_dim


# =============================================================================
# TSVQ Properties
# =============================================================================

class TestTSVQProperties:
    """Property-based tests for TSVQ."""

    @given(
        dim=st.integers(min_value=4, max_value=16),
        n_samples=st.integers(min_value=50, max_value=150),
    )
    @settings(max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow])
    def test_tsvq_preserves_dimension(self, dim, n_samples):
        """TSVQ quantize/dequantize should preserve vector dimension."""
        np.random.seed(42)
        training = np.random.randn(n_samples, dim).astype(np.float32)

        tsvq = pyvq.TSVQ(training_data=training, max_depth=3)

        test_vec = training[0].copy()
        codes = tsvq.quantize(test_vec)
        reconstructed = tsvq.dequantize(codes)

        assert len(reconstructed) == dim
        assert tsvq.dim == dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
