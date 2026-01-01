use crate::distance::Distance;
use crate::exceptions::{VqError, VqResult};
use crate::vector::{mean_vector, Vector};
use half::f16;

struct TSVQNode {
    centroid: Vector<f32>,
    left: Option<Box<TSVQNode>>,
    right: Option<Box<TSVQNode>>,
}

impl TSVQNode {
    fn build(training_data: &[Vector<f32>], max_depth: usize) -> VqResult<Self> {
        if training_data.is_empty() {
            return Err(VqError::EmptyInput);
        }

        let centroid = mean_vector(training_data)?;

        if max_depth == 0 || training_data.len() <= 1 {
            return Ok(TSVQNode {
                centroid,
                left: None,
                right: None,
            });
        }

        let dim = centroid.len();
        let variances: Vec<f32> = (0..dim)
            .map(|i| {
                training_data
                    .iter()
                    .map(|v| {
                        let diff = v.data[i] - centroid.data[i];
                        diff * diff
                    })
                    .sum()
            })
            .collect();

        let split_dim = variances
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut values: Vec<f32> = training_data.iter().map(|v| v.data[split_dim]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if values.len() % 2 == 0 {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
        } else {
            values[values.len() / 2]
        };

        let (left_data, right_data): (Vec<_>, Vec<_>) = training_data
            .iter()
            .cloned()
            .partition(|v| v.data[split_dim] <= median);

        let left = if !left_data.is_empty() && left_data.len() < training_data.len() {
            Some(Box::new(TSVQNode::build(&left_data, max_depth - 1)?))
        } else {
            None
        };

        let right = if !right_data.is_empty() && right_data.len() < training_data.len() {
            Some(Box::new(TSVQNode::build(&right_data, max_depth - 1)?))
        } else {
            None
        };

        Ok(TSVQNode {
            centroid,
            left,
            right,
        })
    }

    fn find_leaf<'a>(&'a self, vector: &[f32], distance: &Distance) -> VqResult<&'a TSVQNode> {
        match (&self.left, &self.right) {
            (Some(left), Some(right)) => {
                let dist_left = distance.compute(vector, &left.centroid.data)?;
                let dist_right = distance.compute(vector, &right.centroid.data)?;
                if dist_left <= dist_right {
                    left.find_leaf(vector, distance)
                } else {
                    right.find_leaf(vector, distance)
                }
            }
            (Some(left), None) => left.find_leaf(vector, distance),
            (None, Some(right)) => right.find_leaf(vector, distance),
            (None, None) => Ok(self),
        }
    }
}

pub struct TSVQ {
    root: TSVQNode,
    distance: Distance,
}

impl TSVQ {
    pub fn new(training_data: &[&[f32]], max_depth: usize, distance: Distance) -> VqResult<Self> {
        if training_data.is_empty() {
            return Err(VqError::EmptyInput);
        }

        let vectors: Vec<Vector<f32>> = training_data
            .iter()
            .map(|&slice| Vector::new(slice.to_vec()))
            .collect();

        let root = TSVQNode::build(&vectors, max_depth)?;
        Ok(TSVQ { root, distance })
    }

    pub fn quantize(&self, vector: &[f32]) -> VqResult<Vec<f16>> {
        if vector.len() != self.root.centroid.len() {
            return Err(VqError::DimensionMismatch {
                expected: self.root.centroid.len(),
                found: vector.len(),
            });
        }

        let leaf = self.root.find_leaf(vector, &self.distance)?;
        let result = leaf
            .centroid
            .data
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data: Vec<&[f32]> = (0..10).map(|_| vec.as_slice()).collect();

        let tsvq = TSVQ::new(&data, 3, Distance::SquaredEuclidean).unwrap();
        let quantized = tsvq.quantize(&vec).unwrap();

        assert_eq!(quantized.len(), vec.len());
        for (orig, q) in vec.iter().zip(quantized.iter()) {
            assert!((orig - f16::to_f32(*q)).abs() < 1e-2);
        }
    }

    #[test]
    fn test_random_vectors() {
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..10).map(|j| ((i + j) % 50) as f32).collect())
            .collect();
        let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

        let tsvq = TSVQ::new(&data_refs, 3, Distance::SquaredEuclidean).unwrap();

        for vec in &data {
            let quantized = tsvq.quantize(vec).unwrap();
            assert_eq!(quantized.len(), vec.len());
        }
    }

    #[test]
    fn test_empty_training() {
        let data: Vec<&[f32]> = vec![];
        let result = TSVQ::new(&data, 3, Distance::Euclidean);
        assert!(result.is_err());
    }
}
