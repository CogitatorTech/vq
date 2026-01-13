use crate::core::error::{VqError, VqResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    SquaredEuclidean,
    Euclidean,
    Manhattan,
    CosineDistance,
}

impl Distance {
    #[inline]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> VqResult<f32> {
        if a.len() != b.len() {
            return Err(VqError::DimensionMismatch {
                expected: a.len(),
                found: b.len(),
            });
        }

        let result = match self {
            Distance::SquaredEuclidean => compute_squared_euclidean(a, b),
            Distance::Euclidean => compute_squared_euclidean(a, b).sqrt(),
            Distance::Manhattan => compute_manhattan(a, b),
            Distance::CosineDistance => compute_cosine(a, b),
        };

        Ok(result)
    }
}

#[inline]
fn compute_squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

#[inline]
fn compute_manhattan(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
}

#[inline]
fn compute_cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_squared_euclidean() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 6.0, 8.0];
        let result = Distance::SquaredEuclidean.compute(&a, &b).unwrap();
        assert!(approx_eq(result, 50.0, 1e-6));
    }

    #[test]
    fn test_euclidean() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 6.0, 8.0];
        let expected = 50.0f32.sqrt();
        let result = Distance::Euclidean.compute(&a, &b).unwrap();
        assert!(approx_eq(result, expected, 1e-6));
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let result = Distance::CosineDistance.compute(&a, &b).unwrap();
        assert!(approx_eq(result, 1.0, 1e-6));

        let a = vec![1.0f32, 1.0];
        let b = vec![1.0f32, 1.0];
        let result = Distance::CosineDistance.compute(&a, &b).unwrap();
        assert!(approx_eq(result, 0.0, 1e-6));
    }

    #[test]
    fn test_manhattan() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 6.0, 8.0];
        let result = Distance::Manhattan.compute(&a, &b).unwrap();
        assert!(approx_eq(result, 12.0, 1e-6));
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        let result = Distance::Euclidean.compute(&a, &b);
        assert!(result.is_err());
    }
}
