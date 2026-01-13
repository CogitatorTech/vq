use half::f16;
use rand::prelude::{IndexedRandom, SeedableRng, StdRng};
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

use crate::core::error::{VqError, VqResult};

pub trait Real:
    Copy
    + Clone
    + Default
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + fmt::Display
    + Send
    + Sync
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_usize(n: usize) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
}

impl Real for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_usize(n: usize) -> Self {
        n as f32
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn abs(self) -> Self {
        self.abs()
    }
}

impl Real for f16 {
    fn zero() -> Self {
        f16::from_f32(0.0)
    }
    fn one() -> Self {
        f16::from_f32(1.0)
    }
    fn from_usize(n: usize) -> Self {
        f16::from_f32(n as f32)
    }
    fn sqrt(self) -> Self {
        f16::from_f32(f16::to_f32(self).sqrt())
    }
    fn abs(self) -> Self {
        f16::from_f32(f16::to_f32(self).abs())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vector<T: Real> {
    pub data: Vec<T>,
}

impl<T: Real> Vector<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn dot(&self, other: &Self) -> T {
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }

    #[inline]
    pub fn norm(&self) -> T {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn distance2(&self, other: &Self) -> T {
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(T::zero(), |acc, (&a, &b)| {
                let diff = a - b;
                acc + diff * diff
            })
    }
}

impl<T: Real> Add for &Vector<T> {
    type Output = Vector<T>;

    fn add(self, other: Self) -> Vector<T> {
        if self.len() != other.len() {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: self.len(),
                    found: other.len()
                }
            );
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Vector::new(data)
    }
}

impl<T: Real> Sub for &Vector<T> {
    type Output = Vector<T>;

    fn sub(self, other: Self) -> Vector<T> {
        if self.len() != other.len() {
            panic!(
                "{}",
                VqError::DimensionMismatch {
                    expected: self.len(),
                    found: other.len()
                }
            );
        }
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Vector::new(data)
    }
}

impl<T: Real> Mul<T> for &Vector<T> {
    type Output = Vector<T>;

    fn mul(self, scalar: T) -> Vector<T> {
        let data = self.data.iter().map(|&a| a * scalar).collect();
        Vector::new(data)
    }
}

impl<T: Real> Div<T> for &Vector<T> {
    type Output = Vector<T>;

    fn div(self, scalar: T) -> Vector<T> {
        let data = self.data.iter().map(|&a| a / scalar).collect();
        Vector::new(data)
    }
}

impl<T: Real> fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elements: Vec<String> = self.data.iter().map(|x| format!("{}", x)).collect();
        write!(f, "Vector [{}]", elements.join(", "))
    }
}

pub fn mean_vector<T: Real>(vectors: &[Vector<T>]) -> VqResult<Vector<T>> {
    if vectors.is_empty() {
        return Err(VqError::EmptyInput);
    }
    let dim = vectors[0].len();
    let n = T::from_usize(vectors.len());
    let mut sum = vec![T::zero(); dim];

    for v in vectors {
        for (i, &val) in v.data.iter().enumerate() {
            sum[i] = sum[i] + val;
        }
    }

    let data = sum.into_iter().map(|s| s / n).collect();
    Ok(Vector::new(data))
}

/// Finds the index of the nearest centroid to a vector.
#[inline]
fn find_nearest_centroid(v: &Vector<f32>, centroids: &[Vector<f32>]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = v.distance2(&centroids[0]);
    for (j, c) in centroids.iter().enumerate().skip(1) {
        let dist = v.distance2(c);
        if dist < best_dist {
            best_dist = dist;
            best_idx = j;
        }
    }
    best_idx
}

/// Computes the mean vector from a slice of data using only the specified indices.
/// This avoids cloning vectors into temporary storage.
#[inline]
fn mean_vector_by_indices(data: &[Vector<f32>], indices: &[usize]) -> VqResult<Vector<f32>> {
    if indices.is_empty() {
        return Err(VqError::EmptyInput);
    }
    let dim = data[indices[0]].len();
    let n = indices.len() as f32;
    let mut sum = vec![0.0f32; dim];

    for &idx in indices {
        for (i, &val) in data[idx].data.iter().enumerate() {
            sum[i] += val;
        }
    }

    let result = sum.into_iter().map(|s| s / n).collect();
    Ok(Vector::new(result))
}

/// LBG/k-means quantization algorithm.
///
/// When compiled with the `parallel` feature, the assignment step is parallelized
/// using Rayon for improved performance on large datasets.
pub fn lbg_quantize(
    data: &[Vector<f32>],
    k: usize,
    max_iters: usize,
    seed: u64,
) -> VqResult<Vec<Vector<f32>>> {
    if data.is_empty() {
        return Err(VqError::EmptyInput);
    }
    if k == 0 {
        return Err(VqError::InvalidParameter(
            "k must be greater than 0".to_string(),
        ));
    }
    if data.len() < k {
        return Err(VqError::InvalidParameter(
            "Not enough data points for k clusters".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids: Vec<Vector<f32>> = data.choose_multiple(&mut rng, k).cloned().collect();

    for _ in 0..max_iters {
        // Compute assignments (parallel when feature enabled)
        #[cfg(feature = "parallel")]
        let assignments: Vec<usize> = {
            use rayon::prelude::*;
            data.par_iter()
                .map(|v| find_nearest_centroid(v, &centroids))
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let assignments: Vec<usize> = data
            .iter()
            .map(|v| find_nearest_centroid(v, &centroids))
            .collect();

        // Build cluster indices (no cloning - just track which data points belong to each cluster)
        let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &cluster_idx) in assignments.iter().enumerate() {
            cluster_indices[cluster_idx].push(i);
        }

        // Update centroids
        let mut changed = false;
        for j in 0..k {
            if !cluster_indices[j].is_empty() {
                let new_centroid = mean_vector_by_indices(data, &cluster_indices[j])?;
                if new_centroid != centroids[j] {
                    changed = true;
                }
                centroids[j] = new_centroid;
            } else {
                #[allow(clippy::expect_used)]
                let random_point = data.choose(&mut rng).expect("data should not be empty");
                centroids[j] = random_point.clone();
            }
        }

        if !changed {
            break;
        }
    }

    Ok(centroids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

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
    fn test_dot_product() {
        let a = Vector::new(vec![1.0f32, 2.0, 3.0]);
        let b = Vector::new(vec![4.0f32, 5.0, 6.0]);
        let dot = a.dot(&b);
        assert!(approx_eq(dot, 32.0, 1e-6));
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
    fn test_mean_vector() {
        let vectors = vec![
            Vector::new(vec![1.0f32, 2.0, 3.0]),
            Vector::new(vec![4.0f32, 5.0, 6.0]),
            Vector::new(vec![7.0f32, 8.0, 9.0]),
        ];
        let mean = mean_vector(&vectors).unwrap();
        assert!(approx_eq(mean.data[0], 4.0, 1e-6));
        assert!(approx_eq(mean.data[1], 5.0, 1e-6));
        assert!(approx_eq(mean.data[2], 6.0, 1e-6));
    }

    #[test]
    fn test_mean_vector_empty() {
        let vectors: Vec<Vector<f32>> = vec![];
        let result = mean_vector(&vectors);
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

    #[test]
    fn lbg_quantize_basic() {
        let data = get_data();
        let centroids = lbg_quantize(&data, 2, 10, 42).unwrap();
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn lbg_quantize_k_zero() {
        let data = vec![Vector::new(vec![1.0, 2.0])];
        let result = lbg_quantize(&data, 0, 10, 42);
        assert!(result.is_err());
    }

    #[test]
    fn lbg_quantize_not_enough_data() {
        let data = vec![Vector::new(vec![1.0, 2.0])];
        let result = lbg_quantize(&data, 2, 10, 42);
        assert!(result.is_err());
    }
}
