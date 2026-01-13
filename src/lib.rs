pub mod algorithms;
pub mod distance;
pub mod exceptions;
pub mod traits;
pub mod vector;

pub use algorithms::bq;
pub use algorithms::pq;
pub use algorithms::sq;
pub use algorithms::tsvq;

pub use exceptions::{VqError, VqResult};
pub use traits::Quantizer;
