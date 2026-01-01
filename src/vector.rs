//! # Vector Representation and Operations
//!
//! This module defines a `Vector` type and operations for real numbers. It includes basic
//! arithmetic (addition, subtraction, scalar multiplication), dot product, norm, and a function
//! to compute the mean vector from a slice of vectors. When the input size exceeds a threshold,
//! Rayon is used to perform operations in parallel for better performance.

use half::f16;
use rayon::prelude::*;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

use crate::exceptions::VqError;

use rand::prelude::{IndexedRandom, SeedableRng, StdRng};

/// Size threshold for enabling parallel computation (via multi-threading).
pub const PARALLEL_THRESHOLD: usize = 1024;

/// Trait for basic operations on real numbers.
pub trait Real:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn from_f64(x: f64) -> Self;
}

impl Real for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn powf(self, n: Self) -> Self {
        f32::powf(self, n)
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

impl Real for f16 {
    fn zero() -> Self {
        f16::from_f32(0.0)
    }
    fn one() -> Self {
        f16::from_f32(1.0)
    }
    fn sqrt(self) -> Self {
        f16::from_f32(f32::from(self).sqrt())
    }
    fn abs(self) -> Self {
        if self < f16::from_f32(0.0) {
            -self
        } else {
            self
        }
    }
    fn powf(self, n: Self) -> Self {
        f16::from_f32(f32::from(self).powf(f32::from(n)))
    }
    fn from_f64(x: f64) -> Self {
        f16::from_f32(x as f32)
    }
}

impl Real for u8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn sqrt(self) -> Self {
        (self as f32).sqrt() as u8
    }
    fn abs(self) -> Self {
        self
    }
    fn powf(self, n: Self) -> Self {
        f32::from(self).powf(f32::from(n)) as u8
    }
    fn from_f64(x: f64) -> Self {
        x as u8
    }
}

/// A vector of real numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T: Real> {
    pub data: Vec<T>,
}

impl<T: Real> Vector<T> {
    /// Create a new vector from a `Vec<T>`.
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a slice of the data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Compute the dot product with another vector.
    ///
    /// If the vector length exceeds `PARALLEL_THRESHOLD`, this is computed in parallel.
    pub fn dot(&self, other: &Vector<T>) -> T
    where
        T: Send + Sync,
    {
        self.assert_same_dim(other);
        if self.len() > PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a * b)
                .reduce(|| T::zero(), |x, y| x + y)
        } else {
            self.data
                .iter()
                .zip(other.data.iter())
                .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
        }
    }

    /// Compute the Euclidean norm.
    pub fn norm(&self) -> T
    where
        T: Send + Sync,
    {
        self.dot(self).sqrt()
    }

    /// Compute the squared distance between two vectors.
    pub fn distance2(&self, other: &Vector<T>) -> T
    where
        T: Send + Sync,
    {
        let diff = self - other;
        diff.dot(&diff)
    }

    /// Private helper to verify that two vectors have the same dimension.
    fn assert_same_dim(&self, other: &Self) {
        if self.len() != other.len() {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: self.len(),
                    found: other.len()
                }
            );
        }
    }
}

/// Vector addition.
impl<'b, T: Real> Add<&'b Vector<T>> for &Vector<T> {
    type Output = Vector<T>;
    fn add(self, rhs: &'b Vector<T>) -> Vector<T> {
        self.assert_same_dim(rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Vector::new(data)
    }
}

/// Vector subtraction.
impl<'b, T: Real> Sub<&'b Vector<T>> for &Vector<T> {
    type Output = Vector<T>;
    fn sub(self, rhs: &'b Vector<T>) -> Vector<T> {
        self.assert_same_dim(rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Vector::new(data)
    }
}

/// Scalar multiplication.
impl<T: Real> Mul<T> for &Vector<T> {
    type Output = Vector<T>;
    fn mul(self, scalar: T) -> Vector<T> {
        let data = self.data.iter().map(|&a| a * scalar).collect();
        Vector::new(data)
    }
}

/// Computes the mean vector from a slice of vectors.
///
/// All vectors must have the same dimension. For many vectors (more than `PARALLEL_THRESHOLD`),
/// the summation is done in parallel.
pub fn mean_vector<T: Real + Send + Sync>(vectors: &[Vector<T>]) -> Vector<T> {
    if vectors.is_empty() {
        panic!("{}", VqError::EmptyInput);
    }
    let dim = vectors[0].len();
    for v in vectors {
        if v.len() != dim {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: dim,
                    found: v.len()
                }
            );
        }
    }
    let sum: Vec<T> = if vectors.len() > PARALLEL_THRESHOLD {
        // Parallel reduction: sum all vectors into one.
        let summed = vectors
            .par_iter()
            .cloned()
            .reduce(|| Vector::new(vec![T::zero(); dim]), |a, b| &a + &b);
        summed.data
    } else {
        let mut sum = vec![T::zero(); dim];
        for v in vectors {
            for (s, &value) in sum.iter_mut().zip(v.data.iter()) {
                *s = *s + value;
            }
        }
        sum
    };
    let n = T::from_f64(vectors.len() as f64);
    let mean_data = sum.into_iter().map(|s| s / n).collect();
    Vector::new(mean_data)
}

/// Custom display for vectors.
impl<T: Real + fmt::Display> fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector [")?;
        let mut first = true;
        for elem in &self.data {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{}", elem)?;
        }
        write!(f, "]")
    }
}

//
// --- SIMD Implementations for f32 ---
//

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

impl Vector<f32> {
    /// Compute the dot product using SIMD instructions (AVX2) if available,
    /// falling back to a scalar loop without copying the input.
    pub fn simd_dot(&self, other: &Self) -> f32 {
        self.assert_same_dim(other);
        let n = self.len();
        let a = &self.data;
        let b = &other.data;
        let mut sum = 0.0;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let mut acc = _mm256_setzero_ps();
                    let mut i = 0;
                    while i + 8 <= n {
                        let va = _mm256_loadu_ps(a.as_ptr().add(i));
                        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                        let prod = _mm256_mul_ps(va, vb);
                        acc = _mm256_add_ps(acc, prod);
                        i += 8;
                    }
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
                    for &x in &tmp {
                        sum += x;
                    }
                    for i in i..n {
                        sum += a[i] * b[i];
                    }
                    return sum;
                }
            }
        }
        // Fallback to scalar computation.
        for i in 0..n {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Compute the Euclidean norm using the SIMD dot product.
    pub fn simd_norm(&self) -> f32 {
        self.simd_dot(self).sqrt()
    }

    /// Compute the squared Euclidean distance using the SIMD dot product.
    pub fn simd_distance2(&self, other: &Self) -> f32 {
        let diff = self - other;
        diff.simd_dot(&diff)
    }
}

/// Compute the squared Euclidean distance between two slices using SIMD if available.
#[inline]
fn simd_distance2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slices must have the same length");
    let n = a.len();
    let mut sum = 0.0;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut acc = _mm256_setzero_ps();
                let mut i = 0;
                while i + 8 <= n {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    let diff = _mm256_sub_ps(va, vb);
                    let diff2 = _mm256_mul_ps(diff, diff);
                    acc = _mm256_add_ps(acc, diff2);
                    i += 8;
                }
                let mut tmp = [0.0f32; 8];
                _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
                for &x in &tmp {
                    sum += x;
                }
                for i in i..n {
                    let d = a[i] - b[i];
                    sum += d * d;
                }
                return sum;
            }
        }
    }
    for i in 0..n {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Helper to find the index of the nearest centroid to a given vector.
fn find_nearest(v: &Vector<f32>, centroids: &[Vector<f32>]) -> usize {
    let (best_idx, _) = centroids.iter().enumerate().fold(
        (0, simd_distance2(&v.data, &centroids[0].data)),
        |(best_idx, best_dist), (i, centroid)| {
            let dist = simd_distance2(&v.data, &centroid.data);
            if dist < best_dist {
                (i, dist)
            } else {
                (best_idx, best_dist)
            }
        },
    );
    best_idx
}

/// Quantizes the input data into `k` clusters using the LBG algorithm.
///
/// The function randomly selects `k` initial centroids and iteratively refines them by
/// assigning each data point to the nearest centroid and then recomputing the centroids.
/// Parallel iteration is used for assignments and cluster grouping when possible.
///
/// # Parameters
/// - `data`: A slice of vectors to quantize.
/// - `k`: The number of clusters (must be > 0 and ≤ number of data points).
/// - `max_iters`: Maximum iterations for the refinement process.
/// - `seed`: A seed for random number generation to ensure reproducibility.
///
/// # Returns
/// A vector of centroids (quantized vectors).
///
/// # Panics
/// - If `k` is 0.
/// - If there are fewer data points than clusters.
pub fn lbg_quantize(
    data: &[Vector<f32>],
    k: usize,
    max_iters: usize,
    seed: u64,
) -> Vec<Vector<f32>> {
    let n = data.len();
    if k == 0 {
        panic!(
            "{}",
            VqError::InvalidParameter("k must be greater than 0".to_string())
        );
    }
    if n < k {
        panic!(
            "{}",
            VqError::InvalidParameter("Not enough data points for k clusters".to_string())
        );
    }

    let mut rng = StdRng::seed_from_u64(seed);
    // Randomly select k initial centroids.
    let mut centroids: Vec<Vector<f32>> = data.choose_multiple(&mut rng, k).cloned().collect();
    let mut assignments = vec![0; n];

    for _ in 0..max_iters {
        // Assignment step: assign each vector to the nearest centroid.
        let new_assignments: Vec<usize> = data
            .par_iter()
            .map(|v| find_nearest(v, &centroids))
            .collect();

        // Check if any assignment changed.
        let changed = new_assignments
            .iter()
            .zip(assignments.iter())
            .any(|(new, old)| new != old);
        assignments = new_assignments;

        // Update step: group data points into clusters.
        let clusters: Vec<Vec<Vector<f32>>> = (0..k)
            .into_par_iter()
            .map(|cluster_idx| {
                data.iter()
                    .zip(assignments.iter())
                    .filter(|(_, &assign)| assign == cluster_idx)
                    .map(|(v, _)| v.clone())
                    .collect::<Vec<_>>()
            })
            .collect();

        // Recompute centroids for each cluster.
        for j in 0..k {
            if !clusters[j].is_empty() {
                centroids[j] = mean_vector(&clusters[j]);
            } else {
                // Reinitialize an empty cluster with a random data point.
                // Safety: data.len() >= k is validated at function start.
                #[allow(clippy::expect_used)]
                let random_point = data.choose(&mut rng).expect("data should not be empty");
                centroids[j] = random_point.clone();
            }
        }

        if !changed {
            break;
        }
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::panic;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn get_data() -> Vec<Vector<f32>> {
        vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![2.0, 3.0]),
            Vector::new(vec![3.0, 4.0]),
            Vector::new(vec![4.0, 5.0]),
        ]
    }

    // --- Vector operations tests ---

    #[test]
    fn test_addition() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let b = Vector::new(vec![4.0f32, 5.0, 6.0]);
        let result = &a + &b;
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_subtraction() {
        let a = Vector::new(vec![4.0f32, 5.0, 6.0]);
        let b = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let result = &a - &b;
        assert_eq!(result.data, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_scalar_multiplication() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let result = &a * 2.0f32;
        assert_eq!(result.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_dot_product_sequential() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let b = Vector::new(vec![4.0f32, 5.0, 6.0]);
        let dot = a.dot(&b);
        assert!(approx_eq(dot, 32.0, 1e-6));
    }

    #[test]
    fn test_dot_product_parallel() {
        let len = PARALLEL_THRESHOLD + 1;
        let a = Vector::new((0..len).map(|i| i as f32).collect());
        let b = Vector::new((0..len).map(|i| (i as f32) * 2.0).collect());
        let expected: f32 = 2.0 * (0..len).map(|i| (i as f32).powi(2)).sum::<f32>();
        let dot = a.dot(&b);
        assert!(approx_eq(dot, expected, 1e3));
    }

    #[test]
    fn test_norm() {
        let a = Vector::new(vec![3.0f32, 4.0]);
        let norm = a.norm();
        assert!(approx_eq(norm, 5.0, 1e-6));
    }

    #[test]
    fn test_distance2() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let b = Vector::new(vec![4.0f32, 5.0, 6.0]);
        let dist2 = a.distance2(&b);
        assert!(approx_eq(dist2, 27.0, 1e-6));
    }

    #[test]
    fn test_mean_vector_sequential() {
        let vectors = vec![
            Vector::new(vec![1.0f32, 2.0, 3.0]),
            Vector::new(vec![4.0f32, 5.0, 6.0]),
            Vector::new(vec![7.0f32, 8.0, 9.0]),
        ];
        let mean = mean_vector(&vectors);
        assert!(approx_eq(mean.data[0], 4.0, 1e-6));
        assert!(approx_eq(mean.data[1], 5.0, 1e-6));
        assert!(approx_eq(mean.data[2], 6.0, 1e-6));
    }

    #[test]
    fn test_addition_mismatched_dimensions() {
        let a = Vector::new(vec![1.0f32, 2.0]);
        let b = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let result = panic::catch_unwind(|| {
            let _ = &a + &b;
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_mean_vector_empty() {
        let vectors: Vec<Vector<f32>> = vec![];
        let result = panic::catch_unwind(|| {
            let _ = mean_vector(&vectors);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_display() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let s = format!("{}", a);
        assert!(s.starts_with("Vector ["));
        assert!(s.ends_with("]"));
    }

    #[test]
    fn test_f16_operations() {
        let a = Vector::new(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
        ]);
        let b = Vector::new(vec![
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ]);
        let dot = a.dot(&b);
        let dot_f32 = f32::from(dot);
        assert!((dot_f32 - 32.0).abs() < 1e-1);
    }

    // --- LBG quantizer tests ---

    #[test]
    fn lbg_quantize_basic_functionality() {
        let data = get_data();
        let centroids = lbg_quantize(&data, 2, 10, 42);
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    #[should_panic(expected = "k must be greater than 0")]
    fn lbg_quantize_k_zero() {
        let data = vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![2.0, 3.0])];
        lbg_quantize(&data, 0, 10, 42);
    }

    #[test]
    #[should_panic(expected = "Not enough data points for k clusters")]
    fn lbg_quantize_not_enough_data_points() {
        let data = vec![Vector::new(vec![1.0, 2.0])];
        lbg_quantize(&data, 2, 10, 42);
    }

    #[test]
    fn lbg_quantize_single_data_point() {
        let data = vec![Vector::new(vec![1.0, 2.0])];
        let centroids = lbg_quantize(&data, 1, 10, 42);
        assert_eq!(centroids.len(), 1);
        assert_eq!(centroids[0], Vector::new(vec![1.0, 2.0]));
    }
}
