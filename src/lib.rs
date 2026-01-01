pub mod algorithms;
pub mod distance;
pub mod exceptions;
mod logging;
pub mod vector;

// Re-export algorithms at crate root for backward compatibility
pub use algorithms::bq;
pub use algorithms::pq;
pub use algorithms::sq;
pub use algorithms::tsvq;
