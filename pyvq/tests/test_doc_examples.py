#!/usr/bin/env python3
"""Test all code examples from pyvq documentation."""
import numpy as np
import pyvq

def test_index_examples():
    """Examples from index.md"""
    print("Testing index.md examples...")
    
    # Binary Quantization
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)
    vector = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    codes = bq.quantize(vector)
    print(f"  Quantized: {codes}")  # [0, 1, 1, 1]
    assert list(codes) == [0, 1, 1, 1], f"Expected [0, 1, 1, 1], got {list(codes)}"

    # Scalar Quantization
    sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
    quantized = sq.quantize(vector)
    reconstructed = sq.dequantize(quantized)
    print(f"  Reconstructed: {reconstructed}")

    # Distance Computation
    dist = pyvq.Distance.euclidean()
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = dist.compute(a, b)
    print(f"  Euclidean distance: {result}")
    
    print("  ✓ index.md examples passed")

def test_getting_started_examples():
    """Examples from getting-started.md"""
    print("Testing getting-started.md examples...")
    
    # Binary Quantization
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)
    vector = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    codes = bq.quantize(vector)
    print(f"  Input:  {vector}")
    print(f"  Output: {codes}")
    assert list(codes) == [0, 0, 1, 1, 1]
    
    # Scalar Quantization
    sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
    vector = np.array([0.1, -0.3, 0.7, -0.9], dtype=np.float32)
    quantized = sq.quantize(vector)
    reconstructed = sq.dequantize(quantized)
    print(f"  Original:      {vector}")
    print(f"  Reconstructed: {reconstructed}")
    
    # Product Quantization
    training = np.random.randn(100, 16).astype(np.float32)
    pq = pyvq.ProductQuantizer(
        training_data=training,
        num_subspaces=4,
        num_centroids=8,
        max_iters=10,
        distance=pyvq.Distance.euclidean(),
        seed=42
    )
    vector = training[0]
    quantized = pq.quantize(vector)
    reconstructed = pq.dequantize(quantized)
    print(f"  PQ Original dimension: {len(vector)}")
    print(f"  PQ Quantized dimension: {len(quantized)}")
    
    # TSVQ
    training = np.random.randn(100, 32).astype(np.float32)
    tsvq = pyvq.TSVQ(
        training_data=training,
        max_depth=5,
        distance=pyvq.Distance.squared_euclidean()
    )
    vector = training[0]
    quantized = tsvq.quantize(vector)
    reconstructed = tsvq.dequantize(quantized)
    print(f"  TSVQ dimension: {len(quantized)}")
    
    # Distance Computation
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    euclidean = pyvq.Distance.euclidean()
    manhattan = pyvq.Distance.manhattan()
    cosine = pyvq.Distance.cosine()
    sq_euclidean = pyvq.Distance.squared_euclidean()
    print(f"  Euclidean: {euclidean.compute(a, b)}")
    print(f"  Manhattan: {manhattan.compute(a, b)}")
    print(f"  Cosine: {cosine.compute(a, b)}")
    print(f"  Squared Euclidean: {sq_euclidean.compute(a, b)}")
    
    print("  ✓ getting-started.md examples passed")

def test_examples_page():
    """Examples from examples.md"""
    print("Testing examples.md examples...")
    
    # Embedding Compression with Scalar Quantization
    embeddings = np.random.randn(100, 64).astype(np.float32)  # Smaller for speed
    embeddings = embeddings / np.abs(embeddings).max()
    sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
    compressed = [sq.quantize(e) for e in embeddings]
    original_bytes = embeddings.nbytes
    compressed_bytes = sum(c.nbytes for c in compressed)
    print(f"  SQ Compression: {original_bytes} -> {compressed_bytes} bytes")
    
    # Binary Hashing
    def hamming_distance(a, b):
        return np.sum(a != b)
    
    bq = pyvq.BinaryQuantizer(threshold=0.0, low=0, high=1)
    vectors = [
        np.array([0.5, -0.3, 0.1, -0.8, 0.2], dtype=np.float32),
        np.array([0.4, -0.2, 0.0, -0.7, 0.3], dtype=np.float32),
        np.array([-0.6, 0.4, -0.2, 0.9, -0.1], dtype=np.float32),
    ]
    hashes = [bq.quantize(v) for v in vectors]
    h01 = hamming_distance(hashes[0], hashes[1])
    h02 = hamming_distance(hashes[0], hashes[2])
    print(f"  Hash 0 vs 1: {h01}")
    print(f"  Hash 0 vs 2: {h02}")
    assert h01 < h02, "Similar vectors should have lower Hamming distance"
    
    # Distance Metrics
    np.random.seed(42)
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)
    metrics = [
        ("Euclidean", pyvq.Distance.euclidean()),
        ("Squared Euclidean", pyvq.Distance.squared_euclidean()),
        ("Manhattan", pyvq.Distance.manhattan()),
        ("Cosine Distance", pyvq.Distance.cosine()),
    ]
    print("  Distance between random 100-d vectors:")
    for name, dist in metrics:
        result = dist.compute(a, b)
        print(f"    {name:20s}: {result:.4f}")
    
    # Error Analysis
    vector = np.random.randn(64).astype(np.float32)
    bq = pyvq.BinaryQuantizer(0.0, 0, 1)
    bq_q = bq.quantize(vector)
    bq_r = bq.dequantize(bq_q)
    bq_mse = np.mean((vector - bq_r) ** 2)
    
    sq = pyvq.ScalarQuantizer(min=-3.0, max=3.0, levels=256)
    sq_q = sq.quantize(vector)
    sq_r = sq.dequantize(sq_q)
    sq_mse = np.mean((vector - sq_r) ** 2)
    
    print(f"  Binary Quantizer MSE: {bq_mse:.4f}")
    print(f"  Scalar Quantizer MSE: {sq_mse:.6f}")
    
    print("  ✓ examples.md examples passed")

def test_api_reference():
    """Examples from api-reference.md"""
    print("Testing api-reference.md examples...")
    
    # Distance
    dist = pyvq.Distance.euclidean()
    result = dist.compute(np.array([1.0, 2.0], dtype=np.float32),
                          np.array([3.0, 4.0], dtype=np.float32))
    print(f"  Distance: {result}")
    
    # BinaryQuantizer
    bq = pyvq.BinaryQuantizer(threshold=0.0)
    codes = bq.quantize(np.array([-0.5, 0.5], dtype=np.float32))
    assert list(codes) == [0, 1]
    print(f"  BinaryQuantizer: {codes}")
    
    # ScalarQuantizer
    sq = pyvq.ScalarQuantizer(min=-1.0, max=1.0, levels=256)
    codes = sq.quantize(np.array([0.0, 0.5], dtype=np.float32))
    reconstructed = sq.dequantize(codes)
    print(f"  ScalarQuantizer: {codes} -> {reconstructed}")
    
    # ProductQuantizer
    training = np.random.randn(100, 16).astype(np.float32)
    pq = pyvq.ProductQuantizer(
        training_data=training,
        num_subspaces=4,
        num_centroids=8,
        distance=pyvq.Distance.euclidean()
    )
    codes = pq.quantize(training[0])
    print(f"  ProductQuantizer: dim={pq.dim}, subspaces={pq.num_subspaces}")
    
    # TSVQ
    training = np.random.randn(100, 32).astype(np.float32)
    tsvq = pyvq.TSVQ(
        training_data=training,
        max_depth=5,
        distance=pyvq.Distance.squared_euclidean()
    )
    codes = tsvq.quantize(training[0])
    print(f"  TSVQ: dim={tsvq.dim}")
    
    # get_simd_backend
    backend = pyvq.get_simd_backend()
    print(f"  SIMD Backend: {backend}")
    
    print("  ✓ api-reference.md examples passed")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing all PyVq documentation examples")
    print("=" * 60)
    
    test_index_examples()
    test_getting_started_examples()
    test_examples_page()
    test_api_reference()
    
    print("=" * 60)
    print("ALL DOCUMENTATION EXAMPLES PASSED ✓")
    print("=" * 60)
