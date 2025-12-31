#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Enum listing the available distance metrics.
pub enum Distance {
    SquaredEuclidean,
    Euclidean,
    Manhattan,
    CosineDistance,
}

impl Distance {
    /// Compute the distance between two slices `a` and `b` using the selected metric.
    ///
    /// Panics with a custom error if the lengths of `a` and `b` differ or if a metric-specific
    /// parameter is invalid.
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            panic!(
                "Dimension mismatch: Expected length {}, found length {}",
                a.len(),
                b.len()
            );
        }

        match self {
            Distance::SquaredEuclidean => compute_squared_euclidean(a, b),
            Distance::Euclidean => compute_euclidean(a, b),
            Distance::CosineDistance => compute_cosine(a, b),
            Distance::Manhattan => compute_manhattan(a, b),
        }
    }
}

/// Compute squared Euclidean distance using SIMD if available, otherwise use scalar.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn compute_squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx") {
        unsafe { compute_squared_euclidean_avx(a, b) }
    } else {
        compute_squared_euclidean_scalar(a, b)
    }
}

/// Scalar implementation of squared Euclidean distance.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn compute_squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    compute_squared_euclidean_scalar(a, b)
}

/// Compute Euclidean distance using SIMD if available, otherwise use scalar.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn compute_euclidean(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx") {
        unsafe { compute_euclidean_avx(a, b) }
    } else {
        compute_euclidean_scalar(a, b)
    }
}

/// Scalar implementation of Euclidean distance.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn compute_euclidean(a: &[f32], b: &[f32]) -> f32 {
    compute_euclidean_scalar(a, b)
}

/// Compute Manhattan distance using SIMD if available, otherwise use scalar.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn compute_manhattan(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx") {
        unsafe { compute_manhattan_avx(a, b) }
    } else {
        compute_manhattan_scalar(a, b)
    }
}

/// Scalar implementation of Manhattan distance.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn compute_manhattan(a: &[f32], b: &[f32]) -> f32 {
    compute_manhattan_scalar(a, b)
}

/// Compute cosine distance using SIMD if available, otherwise use scalar.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn compute_cosine(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx") {
        unsafe { compute_cosine_avx(a, b) }
    } else {
        compute_cosine_scalar(a, b)
    }
}

/// Scalar implementation of cosine distance.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn compute_cosine(a: &[f32], b: &[f32]) -> f32 {
    compute_cosine_scalar(a, b)
}

/// SIMD implementation of squared Euclidean distance for x86/x86_64 using AVX.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn compute_squared_euclidean_avx(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut sum_vec = _mm256_setzero_ps();
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);
        sum_vec = _mm256_add_ps(sum_vec, sq);
        i += 8;
    }
    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum_vec);
    let mut sum = temp.iter().sum::<f32>();
    for j in i..len {
        let d = a[j] - b[j];
        sum += d * d;
    }
    sum
}

/// Scalar implementation of squared Euclidean distance.
fn compute_squared_euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// SIMD implementation of Euclidean distance for x86/x86_64 using AVX.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn compute_euclidean_avx(a: &[f32], b: &[f32]) -> f32 {
    compute_squared_euclidean_avx(a, b).sqrt()
}

/// Scalar implementation of Euclidean distance.
fn compute_euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
    compute_squared_euclidean_scalar(a, b).sqrt()
}

/// SIMD implementation of Manhattan distance for x86/x86_64 using AVX.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn compute_manhattan_avx(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut sum_vec = _mm256_setzero_ps();
    let abs_mask = _mm256_set1_ps(-0.0);
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        let abs_diff = _mm256_andnot_ps(abs_mask, diff);
        sum_vec = _mm256_add_ps(sum_vec, abs_diff);
        i += 8;
    }
    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum_vec);
    let mut sum = temp.iter().sum::<f32>();
    for j in i..len {
        sum += (a[j] - b[j]).abs();
    }
    sum
}

/// Scalar implementation of Manhattan distance.
fn compute_manhattan_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
}

/// SIMD implementation of cosine distance for x86/x86_64 using AVX.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn compute_cosine_avx(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>();

    let mut norm_a_sq = 0.0;
    let mut norm_b_sq = 0.0;

    let len = a.len();
    let mut i = 0;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let va_sq = _mm256_mul_ps(va, va);
        let vb_sq = _mm256_mul_ps(vb, vb);
        let va_sq_arr: [f32; 8] = std::mem::transmute_copy(&va_sq);
        let vb_sq_arr: [f32; 8] = std::mem::transmute_copy(&vb_sq);
        norm_a_sq += va_sq_arr.iter().sum::<f32>();
        norm_b_sq += vb_sq_arr.iter().sum::<f32>();
        i += 8;
    }
    for j in i..len {
        norm_a_sq += a[j] * a[j];
        norm_b_sq += b[j] * b[j];
    }

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

/// Scalar implementation of cosine distance.
fn compute_cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>();

    let norm_a_sq = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b_sq = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a_sq * norm_b_sq))
    }
}
