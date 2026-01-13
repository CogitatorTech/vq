#![allow(dead_code)]
#[path = "common.rs"]
mod common;

use anyhow::Result;
use clap::Parser;
use std::time::Instant;
use vq::{Quantizer, ScalarQuantizer};

#[derive(Parser)]
#[command(name = "eval_sq")]
#[command(about = "Evaluate Scalar Quantizer performance")]
struct Args {
    #[arg(long, default_value_t = common::SEED)]
    seed: u64,

    #[arg(long, default_value_t = common::DIM)]
    dim: usize,

    #[arg(long, default_value_t = 256)]
    levels: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Scalar Quantizer Evaluation");
    println!("===========================");

    for &n_samples in &common::NUM_SAMPLES {
        println!("\nSamples: {}", n_samples);

        let original_data = common::generate_synthetic_data(n_samples, args.dim, args.seed);

        let start = Instant::now();
        let sq = ScalarQuantizer::new(0.0, 1.0, args.levels)?;
        let training_time = start.elapsed().as_millis() as f64;

        let start = Instant::now();
        let _quantized: Vec<Vec<u8>> = original_data
            .iter()
            .filter_map(|vec| sq.quantize(&vec.data).ok())
            .collect();
        let quantization_time = start.elapsed().as_millis() as f64;

        println!("  Training time: {} ms", training_time);
        println!("  Quantization time: {} ms", quantization_time);
    }

    Ok(())
}
