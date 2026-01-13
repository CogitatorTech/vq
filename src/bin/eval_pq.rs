#![allow(dead_code)]
#[path = "utils.rs"]
mod utils;

use anyhow::Result;
use clap::Parser;
use half::f16;
use std::time::Instant;
use vq::distance::Distance;
use vq::pq::ProductQuantizer;
use vq::vector::Vector;
use vq::Quantizer;

#[derive(Parser)]
#[command(name = "eval_pq")]
#[command(about = "Evaluate Product Quantizer performance")]
struct Args {
    #[arg(long, default_value_t = utils::SEED)]
    seed: u64,

    #[arg(long, default_value_t = utils::DIM)]
    dim: usize,

    #[arg(long, default_value_t = utils::M)]
    m: usize,

    #[arg(long, default_value_t = utils::K)]
    k: usize,

    #[arg(long, default_value_t = utils::MAX_ITERS)]
    max_iters: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Product Quantizer Evaluation");
    println!("============================");

    for &n_samples in &utils::NUM_SAMPLES {
        println!("\nSamples: {}", n_samples);

        let original_data = utils::generate_synthetic_data(n_samples, args.dim, args.seed);
        let training_refs: Vec<&[f32]> = original_data.iter().map(|v| v.data.as_slice()).collect();

        let start = Instant::now();
        let pq = ProductQuantizer::new(
            &training_refs,
            args.m,
            args.k,
            args.max_iters,
            Distance::Euclidean,
            args.seed,
        )?;
        let training_time = start.elapsed().as_millis() as f64;

        let start = Instant::now();
        let quantized: Vec<Vec<f16>> = original_data
            .iter()
            .filter_map(|vec| pq.quantize(&vec.data).ok())
            .collect();
        let quantization_time = start.elapsed().as_millis() as f64;

        let reconstructed: Vec<Vector<f32>> = quantized
            .iter()
            .map(|q| Vector::new(q.iter().map(|&x| f16::to_f32(x)).collect()))
            .collect();

        let error = utils::calculate_reconstruction_error(&original_data, &reconstructed);

        println!("  Training time: {} ms", training_time);
        println!("  Quantization time: {} ms", quantization_time);
        println!("  Reconstruction error: {:.6}", error);
    }

    Ok(())
}
