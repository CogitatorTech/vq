import numpy as np
import pytest
import pyvq


@pytest.fixture
def distance_euclidean():
    """Fixture to create a Distance instance for Euclidean distance."""
    return pyvq.Distance.euclidean()


@pytest.fixture
def distance_squared_euclidean():
    """Fixture to create a Distance instance for Squared Euclidean distance."""
    return pyvq.Distance.squared_euclidean()


@pytest.fixture
def distance_cosine():
    """Fixture to create a Distance instance for Cosine distance."""
    return pyvq.Distance.cosine()


@pytest.fixture
def distance_manhattan():
    """Fixture to create a Distance instance for Manhattan distance."""
    return pyvq.Distance.manhattan()


def test_distance_compute_euclidean(distance_euclidean):
    """Test computing Euclidean distance."""
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    result = distance_euclidean.compute(a, b)
    assert np.isclose(result, 2.8284, rtol=1e-4)


def test_distance_compute_squared_euclidean(distance_squared_euclidean):
    """Test computing Squared Euclidean distance."""
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    result = distance_squared_euclidean.compute(a, b)
    assert np.isclose(result, 8.0, rtol=1e-4)


def test_distance_compute_cosine(distance_cosine):
    """Test computing Cosine distance."""
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    result = distance_cosine.compute(a, b)
    assert np.isclose(result, 0.01613, rtol=1e-3)


def test_distance_compute_manhattan(distance_manhattan):
    """Test computing Manhattan distance."""
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    result = distance_manhattan.compute(a, b)
    assert np.isclose(result, 4.0, rtol=1e-4)


def test_distance_compute_different_lengths(distance_euclidean):
    """Test computing distance with vectors of different lengths."""
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        distance_euclidean.compute(a, b)


def test_distance_invalid_metric():
    """Test initializing Distance with an invalid metric."""
    with pytest.raises(ValueError, match="Invalid distance metric"):
        pyvq.Distance("invalid_metric")


def test_distance_static_constructors():
    """Test static constructor methods."""
    eucl = pyvq.Distance.euclidean()
    sqeucl = pyvq.Distance.squared_euclidean()
    manh = pyvq.Distance.manhattan()
    cos = pyvq.Distance.cosine()

    assert "euclidean" in repr(eucl)
    assert "squared_euclidean" in repr(sqeucl)
    assert "manhattan" in repr(manh)
    assert "cosine" in repr(cos)


def test_distance_string_constructor():
    """Test string-based constructor."""
    eucl = pyvq.Distance("euclidean")
    sqeucl = pyvq.Distance("squared_euclidean")
    manh = pyvq.Distance("manhattan")
    cos = pyvq.Distance("cosine")

    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)

    assert np.isclose(eucl.compute(a, b), 2.8284, rtol=1e-4)
    assert np.isclose(sqeucl.compute(a, b), 8.0, rtol=1e-4)
    assert np.isclose(manh.compute(a, b), 4.0, rtol=1e-4)


if __name__ == "__main__":
    pytest.main()
