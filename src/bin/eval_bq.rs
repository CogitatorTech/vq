#![allow(dead_code)]
#[path = "common.rs"]
mod common;

use anyhow::Result;
use clap::Parser;
use std::time::Instant;
use vq::{BinaryQuantizer, Quantizer};

#[derive(Parser)]
#[command(name = "eval_bq")]
#[command(about = "Evaluate Binary Quantizer performance")]
struct Args {
    #[arg(long, default_value_t = common::SEED)]
    seed: u64,

    #[arg(long, default_value_t = common::DIM)]
    dim: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Binary Quantizer Evaluation");
    println!("===========================");

    for &n_samples in &common::NUM_SAMPLES {
        println!("\nSamples: {}", n_samples);

        let original_data = common::generate_synthetic_data(n_samples, args.dim, args.seed);

        let start = Instant::now();
        let bq = BinaryQuantizer::new(0.5, 0, 1)?;
        let training_time = start.elapsed().as_millis() as f64;

        let start = Instant::now();
        let _quantized: Vec<Vec<u8>> = original_data
            .iter()
            .filter_map(|vec| bq.quantize(&vec.data).ok())
            .collect();
        let quantization_time = start.elapsed().as_millis() as f64;

        println!("  Training time: {} ms", training_time);
        println!("  Quantization time: {} ms", quantization_time);
    }

    Ok(())
}
