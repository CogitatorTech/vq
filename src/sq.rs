use crate::exceptions::VqError;
use crate::vector::{Vector, PARALLEL_THRESHOLD};
use rayon::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// A scalar quantizer that maps floating-point values to a set of discrete levels.
pub struct ScalarQuantizer {
    /// The minimum value in the quantizer range.
    pub min: f32,
    /// The maximum value in the quantizer range.
    pub max: f32,
    /// The number of quantization levels (must be at least 2 and no more than 256).
    pub levels: usize,
    /// The step size computed as `(max - min) / (levels - 1)`.
    pub step: f32,
}

impl ScalarQuantizer {
    /// Creates a new `ScalarQuantizer`.
    ///
    /// # Parameters
    /// - `min`: The minimum value in the quantizer's range.
    /// - `max`: The maximum value in the quantizer's range. Must be greater than `min`.
    /// - `levels`: The number of quantization levels. Must be between 2 and 256.
    ///
    /// # Panics
    /// Panics with a custom error if `max` is not greater than `min`, or if `levels` is not within the valid range.
    pub fn new(min: f32, max: f32, levels: usize) -> Self {
        if max <= min {
            panic!(
                "{}",
                VqError::InvalidParameter("max must be greater than min".to_string())
            );
        }
        if levels < 2 {
            panic!(
                "{}",
                VqError::InvalidParameter("levels must be at least 2".to_string())
            );
        }
        if levels > 256 {
            panic!(
                "{}",
                VqError::InvalidParameter("levels must be no more than 256".to_string())
            );
        }
        let step = (max - min) / (levels - 1) as f32;
        Self {
            min,
            max,
            levels,
            step,
        }
    }

    /// Quantizes an input slice by mapping each element to one of the discrete levels.
    ///
    /// The input slice is not modified. Instead, a new `Vector<u8>` is allocated and filled
    /// with the quantized values. SIMD (AVX2) is used if available and the slice is large enough.
    ///
    /// # Parameters
    /// - `slice`: An immutable slice (`&[f32]`) to quantize.
    ///
    /// # Returns
    /// A new vector (`Vector<u8>`) containing the quantized values.
    pub fn quantize(&self, slice: &[f32]) -> Vector<u8> {
        let n = slice.len();
        let mut output = Vec::<u8>::with_capacity(n);
        // Pre-allocate the full output.
        output.resize(n, 0u8);

        if n > PARALLEL_THRESHOLD {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        const CHUNK: usize = 8;
                        let num_chunks = n / CHUNK;
                        let min_vec = _mm256_set1_ps(self.min);
                        let max_vec = _mm256_set1_ps(self.max);
                        let step_vec = _mm256_set1_ps(self.step);
                        let levels_minus_one = self.levels - 1;
                        let levels_minus_one_vec = _mm256_set1_ps(levels_minus_one as f32);

                        for chunk_index in 0..num_chunks {
                            let start = chunk_index * CHUNK;
                            let input_ptr = slice.as_ptr().add(start);
                            let out_ptr = output.as_mut_ptr().add(start) as *mut __m128i;

                            let input_chunk = _mm256_loadu_ps(input_ptr);
                            // Clamp to [min, max]
                            let clamped =
                                _mm256_min_ps(_mm256_max_ps(input_chunk, min_vec), max_vec);
                            // Compute the quantization index
                            let diff = _mm256_sub_ps(clamped, min_vec);
                            let quant = _mm256_div_ps(diff, step_vec);
                            let quant = _mm256_round_ps(quant, _MM_FROUND_TO_NEAREST_INT);
                            let quant = _mm256_min_ps(quant, levels_minus_one_vec);
                            let quant = _mm256_max_ps(quant, _mm256_setzero_ps());
                            // Convert to 32-bit integers.
                            let quant_int = _mm256_cvttps_epi32(quant);
                            // Pack 8 i32 values into 8 u8 values:
                            // First, pack into 16-bit integers.
                            let packed16 = _mm256_packs_epi32(quant_int, _mm256_setzero_si256());
                            // Extract the lower 128 bits.
                            let packed16_low = _mm256_castsi256_si128(packed16);
                            // Then pack into 8-bit integers.
                            let packed8 = _mm_packus_epi16(packed16_low, _mm_setzero_si128());
                            // Store the 8 packed u8 values.
                            _mm_storel_epi64(out_ptr, packed8);
                        }
                        // Process the remaining elements sequentially.
                        let start = num_chunks * CHUNK;
                        for i in start..n {
                            output[i] = self.quantize_scalar(slice[i]) as u8;
                        }
                    }
                    return Vector::new(output);
                }
            }
            // Fallback to parallel sequential processing if SIMD is unavailable.
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out)| *out = self.quantize_scalar(slice[i]) as u8);
            return Vector::new(output);
        }

        // For small slices, process sequentially.
        for (i, &x) in slice.iter().enumerate() {
            output[i] = self.quantize_scalar(x) as u8;
        }
        Vector::new(output)
    }

    /// Quantizes a single scalar value.
    ///
    /// Clamps the value to the `[min, max]` range and maps it uniformly to one of the levels.
    ///
    /// # Parameters
    /// - `x`: The scalar value to quantize.
    ///
    /// # Returns
    /// The quantized value as a `usize`.
    fn quantize_scalar(&self, x: f32) -> usize {
        let clamped = x.max(self.min).min(self.max);
        let index = ((clamped - self.min) / self.step).round() as usize;
        index.min(self.levels - 1)
    }
}
