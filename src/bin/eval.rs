use anyhow::Result;
use clap::Parser;

mod eval_bq;
mod eval_pq;
mod eval_sq;
mod eval_tsvq;

/// Simple CLI to run different evaluations.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Algorithm to evaluate (bq, sq, pq, or tsvq)
    #[arg(short, long)]
    eval: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.eval.as_str() {
        "pq" => eval_pq::main()?,
        "bq" => eval_bq::main()?,
        "sq" => eval_sq::main()?,
        "tsvq" => eval_tsvq::main()?,
        other => {
            eprintln!("Unknown evaluation: {}", other);
            std::process::exit(1);
        }
    }
    Ok(())
}
