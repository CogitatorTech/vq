#![allow(dead_code)]

use anyhow::Result;
use rand::prelude::*;
use rand_distr::{Distribution, Uniform};
use std::collections::HashSet;
use vq::Vector;

pub const SEED: u64 = 66;
pub const NUM_SAMPLES: [usize; 6] = [1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000];
pub const DIM: usize = 128;
pub const M: usize = 16;
pub const K: usize = 256;
pub const MAX_ITERS: usize = 10;

/// Results from a benchmark run.
#[derive(serde::Serialize)]
pub struct BenchmarkResult {
    /// Number of samples used.
    pub n_samples: usize,
    /// Dimension of the vectors.
    pub n_dims: usize,
    /// Time taken for training in milliseconds.
    pub training_time_ms: f64,
    /// Time taken for quantization in milliseconds.
    pub quantization_time_ms: f64,
    /// Mean squared reconstruction error.
    pub reconstruction_error: f32,
    /// Recall at k.
    pub recall: f32,
    /// Ratio of original size to quantized size.
    pub memory_reduction_ratio: f32,
}

/// Generates synthetic random vector data.
///
/// # Arguments
///
/// * `n_samples` - Number of vectors to generate
/// * `n_dims` - Dimension of each vector
/// * `seed` - Random seed
pub fn generate_synthetic_data(n_samples: usize, n_dims: usize, seed: u64) -> Vec<Vector<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    #[allow(clippy::unwrap_used)]
    let uniform = Uniform::new(0.0, 1.0).unwrap();
    (0..n_samples)
        .map(|_| {
            let data: Vec<f32> = (0..n_dims).map(|_| uniform.sample(&mut rng)).collect();
            Vector::new(data)
        })
        .collect()
}

/// Computes the Euclidean distance between two vectors.
pub fn euclidean_distance(a: &Vector<f32>, b: &Vector<f32>) -> f32 {
    a.distance2(b).sqrt()
}

/// Calculates the mean squared reconstruction error between original and reconstructed vectors.
pub fn calculate_reconstruction_error(
    original: &[Vector<f32>],
    reconstructed: &[Vector<f32>],
) -> f32 {
    let total_elements = (original.len() * original[0].len()) as f32;
    let sum_error: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| {
            o.data
                .iter()
                .zip(r.data.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
        })
        .sum();
    sum_error / total_elements
}

/// Calculates the recall@k for approximate nearest neighbor search.
///
/// Estimates recall by sampling a subset of queries.
///
/// # Arguments
///
/// * `original` - Original dataset vectors
/// * `approx` - Reconstructed/Approximate vectors
/// * `k` - Number of neighbors to check
pub fn calculate_recall(original: &[Vector<f32>], approx: &[Vector<f32>], k: usize) -> Result<f32> {
    let n_samples = original.len();
    let max_eval_samples = 1000;
    let eval_samples = n_samples.min(max_eval_samples);
    let step = (n_samples / eval_samples).max(1);
    let mut total_recall = 0.0;

    for i in (0..n_samples).step_by(step) {
        let query = &original[i];
        let search_window = if n_samples > 10_000 { 5000 } else { n_samples };

        let start_idx = i.saturating_sub(search_window / 2);
        let end_idx = (i + search_window / 2).min(n_samples);

        let mut true_neighbors: Vec<(usize, f32)> = (start_idx..end_idx)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(query, &original[j])))
            .collect();
        true_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let true_neighbors: Vec<usize> =
            true_neighbors.iter().take(k).map(|&(idx, _)| idx).collect();

        let mut approx_neighbors: Vec<(usize, f32)> = (start_idx..end_idx)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(&approx[i], &approx[j])))
            .collect();
        approx_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let approx_neighbors: Vec<usize> = approx_neighbors
            .iter()
            .take(k)
            .map(|&(idx, _)| idx)
            .collect();

        let approx_set: HashSet<_> = approx_neighbors.into_iter().collect();
        let intersection = true_neighbors
            .iter()
            .filter(|&&idx| approx_set.contains(&idx))
            .count() as f32;
        total_recall += intersection / k as f32;
    }
    Ok(total_recall / (n_samples / step) as f32)
}

fn main() {}
