use crate::exceptions::VqError;
use crate::vector::{Vector, PARALLEL_THRESHOLD};
use rayon::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// A simple binary quantizer that maps floating-point values to one of two discrete values (levels).
pub struct BinaryQuantizer {
    /// The threshold value used to determine whether an element is quantized to `high` or `low`.
    pub threshold: f32,
    /// The quantized value assigned to inputs that are below the threshold.
    pub low: u8,
    /// The quantized value assigned to inputs that are at or above the threshold.
    pub high: u8,
}

impl BinaryQuantizer {
    /// Creates a new `BinaryQuantizer` with the specified threshold and quantization levels.
    ///
    /// # Parameters
    /// - `threshold`: The threshold value used for quantization.
    /// - `low`: The quantized value to assign for input values below the threshold.
    /// - `high`: The quantized value to assign for input values at or above the threshold.
    ///
    /// # Panics
    /// Panics with a custom error if `low` is not less than `high`.
    pub fn new(threshold: f32, low: u8, high: u8) -> Self {
        if low >= high {
            panic!(
                "{}",
                VqError::InvalidParameter(
                    "Low quantization level must be less than high quantization level".to_string()
                )
            );
        }
        Self {
            threshold,
            low,
            high,
        }
    }

    /// Quantizes an input slice by mapping each element to either the low or high value based on the threshold.
    ///
    /// For each element in the input slice:
    /// - If the value is greater than or equal to `self.threshold`, it is mapped to `self.high`.
    /// - Otherwise, it is mapped to `self.low`.
    ///
    /// For large input slices (length > `PARALLEL_THRESHOLD`), parallel processing is used.
    /// Internally the slice is processed without copying and SIMD (AVX2) is used when available.
    ///
    /// # Parameters
    /// - `slice`: A reference to the input slice (`&[f32]`) to be quantized.
    ///
    /// # Returns
    /// A new vector (`Vector<u8>`) containing the quantized values.
    pub fn quantize(&self, slice: &[f32]) -> Vector<u8> {
        let n = slice.len();

        let quantized: Vec<u8> = if n > PARALLEL_THRESHOLD {
            // Process in parallel in chunks.
            let mut result = vec![0u8; n];
            result
                .par_chunks_mut(8)
                .enumerate()
                .for_each(|(chunk_index, out_chunk)| {
                    let start = chunk_index * 8;
                    let end = usize::min(start + 8, n);
                    let input_chunk = &slice[start..end];

                    // Use AVX2 if available and if we have a full 8-element chunk.
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if is_x86_feature_detected!("avx2") && input_chunk.len() == 8 {
                            unsafe {
                                let threshold_vec = _mm256_set1_ps(self.threshold);
                                let v = _mm256_loadu_ps(input_chunk.as_ptr());
                                // Using 14 (_CMP_GE_OS) as the compare predicate.
                                let cmp = _mm256_cmp_ps::<14>(v, threshold_vec);
                                let mask = _mm256_movemask_ps(cmp);
                                for j in 0..8 {
                                    let bit = (mask >> j) & 1;
                                    out_chunk[j] = if bit == 1 { self.high } else { self.low };
                                }
                            }
                            return;
                        }
                    }
                    // Fallback: process the chunk sequentially.
                    for (j, &x) in input_chunk.iter().enumerate() {
                        out_chunk[j] = if x >= self.threshold {
                            self.high
                        } else {
                            self.low
                        };
                    }
                });
            result
        } else {
            // For small inputs, process sequentially with SIMD fallback.
            quantize_simd(slice, self.threshold, self.low, self.high)
        };

        Vector::new(quantized)
    }
}

/// Sequential quantization using SIMD (AVX2) if available, with fallback to elementwise processing.
fn quantize_simd(slice: &[f32], threshold: f32, low: u8, high: u8) -> Vec<u8> {
    let n = slice.len();
    let mut result = vec![0u8; n];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            let threshold_vec = _mm256_set1_ps(threshold);
            let mut i = 0;
            while i + 8 <= n {
                let v = _mm256_loadu_ps(slice.as_ptr().add(i));
                let cmp = _mm256_cmp_ps::<14>(v, threshold_vec);
                let mask = _mm256_movemask_ps(cmp);
                for j in 0..8 {
                    let bit = (mask >> j) & 1;
                    result[i + j] = if bit == 1 { high } else { low };
                }
                i += 8;
            }
            // Process any remaining elements.
            while i < n {
                let x = *slice.get_unchecked(i);
                result[i] = if x >= threshold { high } else { low };
                i += 1;
            }
        }
        return result;
    }

    // Fallback sequential processing without SIMD.
    for (i, &x) in slice.iter().enumerate() {
        result[i] = if x >= threshold { high } else { low };
    }
    result
}
