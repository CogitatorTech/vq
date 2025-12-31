import numpy as np
import pytest
import pyvq


@pytest.fixture
def distance_euclidean():
    """Fixture to create a Distance instance for testing Euclidean distance."""
    return pyvq.Distance("euclidean")


@pytest.fixture
def distance_squared_euclidean():
    """Fixture to create a Distance instance for testing Squared Euclidean distance."""
    return pyvq.Distance("squared_euclidean")


@pytest.fixture
def distance_cosine():
    """Fixture to create a Distance instance for testing Cosine distance."""
    return pyvq.Distance("cosine")


@pytest.fixture
def distance_manhattan():
    """Fixture to create a Distance instance for testing Manhattan distance."""
    return pyvq.Distance("manhattan")


def test_distance_compute_euclidean(distance_euclidean):
    """Test computing Euclidean distance."""
    result = distance_euclidean.compute([1.0, 2.0], [3.0, 4.0])
    assert np.isclose(result, 2.8284, rtol=1e-4)


def test_distance_compute_squared_euclidean(distance_squared_euclidean):
    """Test computing Squared Euclidean distance."""
    result = distance_squared_euclidean.compute([1.0, 2.0], [3.0, 4.0])
    assert np.isclose(result, 8.0, rtol=1e-4)


def test_distance_compute_cosine(distance_cosine):
    """Test computing Cosine distance."""
    result = distance_cosine.compute([1.0, 2.0], [3.0, 4.0])
    assert np.isclose(result, 0.01613, rtol=1e-3)


def test_distance_compute_manhattan(distance_manhattan):
    """Test computing Manhattan distance."""
    result = distance_manhattan.compute([1.0, 2.0], [3.0, 4.0])
    assert np.isclose(result, 4.0, rtol=1e-4)


def test_distance_compute_different_lengths(distance_euclidean):
    """Test computing distance with vectors of different lengths."""
    with pytest.raises(ValueError, match="Vectors must have the same length"):
        distance_euclidean.compute([1.0, 2.0], [3.0, 4.0, 5.0])


def test_distance_invalid_metric():
    """Test initializing Distance with an invalid metric."""
    with pytest.raises(ValueError, match="Invalid distance metric"):
        pyvq.Distance("invalid_metric")


if __name__ == "__main__":
    pytest.main()
