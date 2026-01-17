#!/usr/bin/env python3
"""
Image Quantization Demo

Demonstrates the four quantization algorithms (Binary, Scalar, Product, TSVQ)
on an image, showing the visual effects and file size reduction.
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import pyvq

# Configuration
# Images are stored in main docs folder
IMAGE_PATH = Path(__file__).parent.parent.parent / "docs" / "assets" / "images" / "nixon_visions_base_1024.png"
# Output to same directory as original image
OUTPUT_DIR = IMAGE_PATH.parent


def load_image(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image as float32 numpy array."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    return arr, img.size


def save_image(arr: np.ndarray, path: Path, original_size: tuple[int, int]) -> int:
    """Save numpy array as image and return file size in bytes."""
    # Clip values to valid range and convert to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(path, format="PNG", optimize=True)
    return path.stat().st_size


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"


def binary_quantization(image: np.ndarray) -> np.ndarray:
    """Apply binary quantization to image with fixed threshold."""
    # Fixed threshold at midpoint
    bq = pyvq.BinaryQuantizer(threshold=127.5, low=0, high=1)
    
    result = np.zeros_like(image)
    for c in range(3):
        channel = image[:, :, c].flatten().astype(np.float32)
        # Quantize to 0/1 binary values
        quantized = bq.quantize(channel)
        # Scale 0/1 to 0/255 for image display
        result[:, :, c] = (quantized.astype(np.float32) * 255.0).reshape(image.shape[:2])
    
    return result


def scalar_quantization(image: np.ndarray, levels: int = 16) -> np.ndarray:
    """Apply scalar quantization to image with specified number of levels."""
    sq = pyvq.ScalarQuantizer(min=0.0, max=255.0, levels=levels)
    
    # Quantize each channel separately
    result = np.zeros_like(image)
    for c in range(3):
        channel = image[:, :, c].flatten().astype(np.float32)
        quantized = sq.quantize(channel)
        dequantized = sq.dequantize(quantized)
        result[:, :, c] = dequantized.reshape(image.shape[:2])
    
    return result


def product_quantization(image: np.ndarray, num_subspaces: int = 8, num_centroids: int = 16) -> np.ndarray:
    """Apply product quantization to image rows."""
    height, width, channels = image.shape
    
    # Reshape to treat each row as a vector (flatten width * channels)
    vectors = image.reshape(height, width * channels).astype(np.float32)
    
    # Use all rows for training
    training_data = vectors
    
    # Create product quantizer
    pq = pyvq.ProductQuantizer(
        training_data=training_data,
        num_subspaces=num_subspaces,
        num_centroids=num_centroids,
        max_iters=5,
        seed=42
    )
    
    # Quantize and dequantize each row
    result = np.zeros_like(vectors)
    for i in range(height):
        codes = pq.quantize(vectors[i])
        result[i] = pq.dequantize(codes)
    
    return result.reshape(height, width, channels)


def tsvq_quantization(image: np.ndarray, max_depth: int = 6) -> np.ndarray:
    """Apply tree-structured vector quantization to image."""
    height, width, channels = image.shape
    
    # Reshape to treat each row as a vector
    vectors = image.reshape(height, width * channels).astype(np.float32)
    
    # Use all rows for training
    training_data = vectors
    
    # Create TSVQ
    tsvq = pyvq.TSVQ(
        training_data=training_data,
        max_depth=max_depth
    )
    
    # Quantize and dequantize each row
    result = np.zeros_like(vectors)
    for i in range(height):
        codes = tsvq.quantize(vectors[i])
        result[i] = tsvq.dequantize(codes)
    
    return result.reshape(height, width, channels)


def main():
    print("=" * 70)
    print("Image Quantization Demo - PyVQ")
    print("=" * 70)
    print(f"\nSIMD Backend: {pyvq.get_simd_backend()}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load original image
    print(f"\nLoading image: {IMAGE_PATH}")
    image, size = load_image(IMAGE_PATH)
    original_size = IMAGE_PATH.stat().st_size
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} (WxH)")
    print(f"Original file size: {format_size(original_size)}")
    
    results = []
    
    # 1. Binary Quantization (2 levels per channel = 8 colors)
    print("\n" + "-" * 50)
    print("1. Binary Quantization (2 levels)")
    print("-" * 50)
    print("   Reducing each color channel to 2 values (0 or 255)")
    binary_result = binary_quantization(image)
    binary_path = OUTPUT_DIR / "nixon_visions_binary_1024.png"
    binary_size = save_image(binary_result, binary_path, size)
    reduction = (1 - binary_size / original_size) * 100
    print(f"   Output: {binary_path}")
    print(f"   File size: {format_size(binary_size)} ({reduction:+.1f}% reduction)")
    results.append(("Binary (2 levels)", binary_size, reduction))
    
    # 2. Scalar Quantization (16 levels = 4 bits per channel)
    print("\n" + "-" * 50)
    print("2. Scalar Quantization (16 levels)")
    print("-" * 50)
    print("   Reducing each color channel to 16 discrete values")
    scalar16_result = scalar_quantization(image, levels=16)
    scalar16_path = OUTPUT_DIR / "nixon_visions_scalar16_1024.png"
    scalar16_size = save_image(scalar16_result, scalar16_path, size)
    reduction = (1 - scalar16_size / original_size) * 100
    print(f"   Output: {scalar16_path}")
    print(f"   File size: {format_size(scalar16_size)} ({reduction:+.1f}% reduction)")
    results.append(("Scalar (16 levels)", scalar16_size, reduction))
    
    # 3. Scalar Quantization (8 levels = 3 bits per channel)
    print("\n" + "-" * 50)
    print("3. Scalar Quantization (8 levels)")
    print("-" * 50)
    print("   More aggressive reduction to 8 discrete values per channel")
    scalar8_result = scalar_quantization(image, levels=8)
    scalar8_path = OUTPUT_DIR / "nixon_visions_scalar8_1024.png"
    scalar8_size = save_image(scalar8_result, scalar8_path, size)
    reduction = (1 - scalar8_size / original_size) * 100
    print(f"   Output: {scalar8_path}")
    print(f"   File size: {format_size(scalar8_size)} ({reduction:+.1f}% reduction)")
    results.append(("Scalar (8 levels)", scalar8_size, reduction))
    
    # 4. Product Quantization
    print("\n" + "-" * 50)
    print("4. Product Quantization (8 subspaces, 16 centroids)")
    print("-" * 50)
    print("   Dividing image rows into subspaces and clustering")
    pq_result = product_quantization(image, num_subspaces=8, num_centroids=16)
    pq_path = OUTPUT_DIR / "nixon_visions_pq8x16_1024.png"
    pq_size = save_image(pq_result, pq_path, size)
    reduction = (1 - pq_size / original_size) * 100
    print(f"   Output: {pq_path}")
    print(f"   File size: {format_size(pq_size)} ({reduction:+.1f}% reduction)")
    results.append(("Product (8x16)", pq_size, reduction))
    
    # 5. TSVQ
    print("\n" + "-" * 50)
    print("5. Tree-Structured Vector Quantization (depth=6)")
    print("-" * 50)
    print("   Hierarchical clustering using binary tree")
    tsvq_result = tsvq_quantization(image, max_depth=6)
    tsvq_path = OUTPUT_DIR / "nixon_visions_tsvq6_1024.png"
    tsvq_size = save_image(tsvq_result, tsvq_path, size)
    reduction = (1 - tsvq_size / original_size) * 100
    print(f"   Output: {tsvq_path}")
    print(f"   File size: {format_size(tsvq_size)} ({reduction:+.1f}% reduction)")
    results.append(("TSVQ (depth=6)", tsvq_size, reduction))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Algorithm':<25} {'Size':>12} {'Reduction':>12}")
    print("-" * 50)
    print(f"{'Original':<25} {format_size(original_size):>12} {'---':>12}")
    for name, size, reduction in results:
        print(f"{name:<25} {format_size(size):>12} {reduction:>+11.1f}%")
    
    print(f"\nOutput files saved to: {OUTPUT_DIR.absolute()}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
