use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    eval: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Run individual eval binaries: eval_bq, eval_sq, eval_pq, eval_tsvq");
    println!("Requested: {}", args.eval);
    Ok(())
}
