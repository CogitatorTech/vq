//! FFI bindings for hsdlib SIMD distance functions.
//!
//! This module is only compiled when the `simd` feature is enabled.

use std::os::raw::c_int;

/// Status codes returned by hsdlib functions.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HsdStatus {
    Success = 0,
    ErrNullPtr = -1,
    ErrInvalidInput = -3,
    ErrCpuNotSupported = -4,
    Failure = -99,
}

impl HsdStatus {
    #[inline]
    pub fn is_success(self) -> bool {
        self == HsdStatus::Success
    }
}

impl From<c_int> for HsdStatus {
    fn from(value: c_int) -> Self {
        match value {
            0 => HsdStatus::Success,
            -1 => HsdStatus::ErrNullPtr,
            -3 => HsdStatus::ErrInvalidInput,
            -4 => HsdStatus::ErrCpuNotSupported,
            _ => HsdStatus::Failure,
        }
    }
}

extern "C" {
    /// Compute squared Euclidean distance between two float vectors.
    pub fn hsd_dist_sqeuclidean_f32(
        a: *const f32,
        b: *const f32,
        n: usize,
        result: *mut f32,
    ) -> c_int;

    /// Compute Manhattan distance (L1) between two float vectors.
    pub fn hsd_dist_manhattan_f32(
        a: *const f32,
        b: *const f32,
        n: usize,
        result: *mut f32,
    ) -> c_int;

    /// Compute cosine similarity for float vectors.
    pub fn hsd_sim_cosine_f32(a: *const f32, b: *const f32, n: usize, result: *mut f32) -> c_int;

    /// Compute dot product for float vectors.
    pub fn hsd_sim_dot_f32(a: *const f32, b: *const f32, n: usize, result: *mut f32) -> c_int;

    /// Get a human-readable description of the selected backend.
    pub fn hsd_get_backend() -> *const std::os::raw::c_char;
}

/// Compute squared Euclidean distance using SIMD acceleration.
///
/// Returns `None` if the C function fails.
#[inline]
pub fn sqeuclidean_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut result: f32 = 0.0;
    let status: HsdStatus =
        unsafe { hsd_dist_sqeuclidean_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) }.into();
    if status.is_success() {
        Some(result)
    } else {
        None
    }
}

/// Compute Manhattan distance using SIMD acceleration.
///
/// Returns `None` if the C function fails.
#[inline]
pub fn manhattan_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut result: f32 = 0.0;
    let status: HsdStatus =
        unsafe { hsd_dist_manhattan_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) }.into();
    if status.is_success() {
        Some(result)
    } else {
        None
    }
}

/// Compute cosine similarity using SIMD acceleration.
///
/// Returns `None` if the C function fails.
#[inline]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut result: f32 = 0.0;
    let status: HsdStatus =
        unsafe { hsd_sim_cosine_f32(a.as_ptr(), b.as_ptr(), a.len(), &mut result) }.into();
    if status.is_success() {
        Some(result)
    } else {
        None
    }
}

/// Returns the name of the currently active SIMD backend.
///
/// This function queries hsdlib to determine which SIMD implementation is being used.
/// Possible return values include:
/// - `"Scalar (Auto)"` - Fallback scalar implementation
/// - `"AVX (Auto)"` / `"AVX2 (Auto)"` / `"AVX512F (Auto)"` - x86 SIMD
/// - `"NEON (Auto)"` / `"SVE (Auto)"` - ARM SIMD
///
/// # Example
///
/// ```ignore
/// use vq::get_simd_backend;
///
/// if let Some(backend) = get_simd_backend() {
///     println!("Using SIMD backend: {}", backend);
/// }
/// ```
pub fn get_simd_backend() -> String {
    unsafe {
        let ptr = hsd_get_backend();
        if ptr.is_null() {
            return "Unknown".to_string();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
