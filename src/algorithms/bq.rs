use crate::exceptions::{VqError, VqResult};

pub struct BinaryQuantizer {
    pub threshold: f32,
    pub low: u8,
    pub high: u8,
}

impl BinaryQuantizer {
    pub fn new(threshold: f32, low: u8, high: u8) -> VqResult<Self> {
        if low >= high {
            return Err(VqError::InvalidParameter(
                "Low quantization level must be less than high quantization level".to_string(),
            ));
        }
        Ok(Self {
            threshold,
            low,
            high,
        })
    }

    pub fn quantize(&self, slice: &[f32]) -> Vec<u8> {
        slice
            .iter()
            .map(|&x| {
                if x >= self.threshold {
                    self.high
                } else {
                    self.low
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
        let input = vec![-1.0, 0.0, 1.0, -0.5, 0.5];
        let result = bq.quantize(&input);
        assert_eq!(result, vec![0, 1, 1, 0, 1]);
    }

    #[test]
    fn test_large_vector() {
        let bq = BinaryQuantizer::new(0.0, 0, 1).unwrap();
        let input: Vec<f32> = (0..1024).map(|i| (i as f32) - 512.0).collect();
        let result = bq.quantize(&input);
        assert_eq!(result.len(), 1024);

        for (i, &val) in result.iter().enumerate() {
            let expected = if input[i] >= 0.0 { 1 } else { 0 };
            assert_eq!(val, expected);
        }
    }

    #[test]
    fn test_invalid_levels() {
        let result = BinaryQuantizer::new(0.0, 1, 0);
        assert!(result.is_err());
    }
}
