use crate::distance::Distance;
use crate::exceptions::{VqError, VqResult};
use crate::vector::{lbg_quantize, Vector};
use half::f16;

pub struct ProductQuantizer {
    codebooks: Vec<Vec<Vector<f32>>>,
    sub_dim: usize,
    m: usize,
    distance: Distance,
}

impl ProductQuantizer {
    pub fn new(
        training_data: &[&[f32]],
        m: usize,
        k: usize,
        max_iters: usize,
        distance: Distance,
        seed: u64,
    ) -> VqResult<Self> {
        if training_data.is_empty() {
            return Err(VqError::EmptyInput);
        }
        let n = training_data[0].len();
        if n < m {
            return Err(VqError::InvalidParameter(
                "Data dimension must be at least m".to_string(),
            ));
        }
        if n % m != 0 {
            return Err(VqError::InvalidParameter(
                "Data dimension must be divisible by m".to_string(),
            ));
        }
        let sub_dim = n / m;

        let mut codebooks = Vec::with_capacity(m);
        for i in 0..m {
            let sub_training: Vec<Vector<f32>> = training_data
                .iter()
                .map(|v| {
                    let start = i * sub_dim;
                    let end = start + sub_dim;
                    Vector::new(v[start..end].to_vec())
                })
                .collect();
            let codebook = lbg_quantize(&sub_training, k, max_iters, seed + i as u64)?;
            codebooks.push(codebook);
        }

        Ok(Self {
            codebooks,
            sub_dim,
            m,
            distance,
        })
    }

    pub fn quantize(&self, vector: &[f32]) -> VqResult<Vec<f16>> {
        let n = vector.len();
        if n != self.sub_dim * self.m {
            return Err(VqError::DimensionMismatch {
                expected: self.sub_dim * self.m,
                found: n,
            });
        }

        let mut result = Vec::with_capacity(n);
        for i in 0..self.m {
            let start = i * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_vector = &vector[start..end];
            let codebook = &self.codebooks[i];

            let mut best_idx = 0;
            let mut best_dist = self.distance.compute(sub_vector, &codebook[0].data)?;
            for (j, centroid) in codebook.iter().enumerate().skip(1) {
                let dist = self.distance.compute(sub_vector, &centroid.data)?;
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }

            for &val in &codebook[best_idx].data {
                result.push(f16::from_f32(val));
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|j| ((i + j) % 100) as f32).collect())
            .collect()
    }

    #[test]
    fn test_basic() {
        let data: Vec<Vec<f32>> = generate_test_data(100, 10);
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::new(&data_refs, 2, 4, 10, Distance::Euclidean, 42).unwrap();

        let quantized = pq.quantize(&data[0]).unwrap();
        assert_eq!(quantized.len(), 10);
    }

    #[test]
    fn test_empty_training() {
        let data: Vec<&[f32]> = vec![];
        let result = ProductQuantizer::new(&data, 2, 2, 10, Distance::Euclidean, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_not_divisible() {
        let data = vec![vec![1.0, 2.0, 3.0]];
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let result = ProductQuantizer::new(&data_refs, 2, 2, 10, Distance::Euclidean, 42);
        assert!(result.is_err());
    }
}
