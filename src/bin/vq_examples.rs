use vq::{BinaryQuantizer, Distance, ProductQuantizer, Quantizer, ScalarQuantizer, VqResult, TSVQ};

fn main() -> VqResult<()> {
    println!("=== Vq Examples ===\n");

    example_bq()?;
    example_sq()?;
    example_pq()?;
    example_tsvq()?;

    println!("All examples completed successfully!");
    Ok(())
}

fn example_bq() -> VqResult<()> {
    println!("Binary Quantizer:");
    let bq = BinaryQuantizer::new(0.0, 0, 1)?;
    let input = vec![-0.5, 0.0, 0.5, 1.0, -1.0];
    let quantized = bq.quantize(&input)?;
    println!("  Input: {:?}", input);
    println!("  Output: {:?}\n", quantized);
    Ok(())
}

fn example_sq() -> VqResult<()> {
    println!("Scalar Quantizer:");
    let sq = ScalarQuantizer::new(-1.0, 1.0, 256)?;
    let input = vec![-0.5, 0.0, 0.5, 1.0, -1.0];
    let quantized = sq.quantize(&input)?;
    println!("  Input: {:?}", input);
    println!("  Output: {:?}\n", quantized);
    Ok(())
}

fn example_pq() -> VqResult<()> {
    println!("Product Quantizer:");
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..10).map(|j| ((i + j) % 50) as f32 / 50.0).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let pq = ProductQuantizer::new(&training_refs, 2, 4, 10, Distance::Euclidean, 42)?;
    let quantized = pq.quantize(&training[0])?;
    println!("  Training vectors: {}", training.len());
    println!("  Input dim: {}", training[0].len());
    println!("  Output len: {}\n", quantized.len());
    Ok(())
}

fn example_tsvq() -> VqResult<()> {
    println!("Tree-Structured VQ:");
    let training: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..10).map(|j| ((i + j) % 50) as f32 / 50.0).collect())
        .collect();
    let training_refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

    let tsvq = TSVQ::new(&training_refs, 3, Distance::Euclidean)?;
    let quantized = tsvq.quantize(&training[0])?;
    println!("  Training vectors: {}", training.len());
    println!("  Input dim: {}", training[0].len());
    println!("  Output len: {}\n", quantized.len());
    Ok(())
}
