use crate::exceptions::{VqError, VqResult};

pub struct ScalarQuantizer {
    pub min: f32,
    pub max: f32,
    pub levels: usize,
    pub step: f32,
}

impl ScalarQuantizer {
    pub fn new(min: f32, max: f32, levels: usize) -> VqResult<Self> {
        if max <= min {
            return Err(VqError::InvalidParameter(
                "max must be greater than min".to_string(),
            ));
        }
        if levels < 2 {
            return Err(VqError::InvalidParameter(
                "levels must be at least 2".to_string(),
            ));
        }
        if levels > 256 {
            return Err(VqError::InvalidParameter(
                "levels must be no more than 256".to_string(),
            ));
        }
        let step = (max - min) / (levels - 1) as f32;
        Ok(Self {
            min,
            max,
            levels,
            step,
        })
    }

    pub fn quantize(&self, slice: &[f32]) -> Vec<u8> {
        slice
            .iter()
            .map(|&x| self.quantize_scalar(x) as u8)
            .collect()
    }

    fn quantize_scalar(&self, x: f32) -> usize {
        let clamped = x.clamp(self.min, self.max);
        let index = ((clamped - self.min) / self.step).round() as usize;
        index.min(self.levels - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_on_scalars() {
        let sq = ScalarQuantizer::new(-1.0, 1.0, 5).unwrap();
        let test_values = vec![-1.2, -1.0, -0.8, -0.3, 0.0, 0.3, 0.6, 1.0, 1.2];
        for x in test_values {
            let indices = sq.quantize(&[x]);
            assert_eq!(indices.len(), 1);
            let reconstructed = sq.min + indices[0] as f32 * sq.step;
            let clamped = x.clamp(sq.min, sq.max);
            let error = (reconstructed - clamped).abs();
            assert!(error <= sq.step / 2.0 + 1e-6);
        }
    }

    #[test]
    fn test_large_vectors() {
        let sq = ScalarQuantizer::new(-1000.0, 1000.0, 256).unwrap();
        let input: Vec<f32> = (0..1024).map(|i| (i as f32) - 512.0).collect();
        let result = sq.quantize(&input);
        assert_eq!(result.len(), 1024);
    }

    #[test]
    fn test_invalid_range() {
        let result = ScalarQuantizer::new(1.0, -1.0, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_too_few_levels() {
        let result = ScalarQuantizer::new(-1.0, 1.0, 1);
        assert!(result.is_err());
    }
}
