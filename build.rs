fn main() {
    #[cfg(feature = "simd")]
    {
        let mut build = cc::Build::new();

        // Set include path
        build.include("external/hsdlib/include");

        // Add C source files
        build.file("external/hsdlib/src/utils.c");
        build.file("external/hsdlib/src/distance/euclidean.c");
        build.file("external/hsdlib/src/distance/manhattan.c");
        build.file("external/hsdlib/src/similarity/cosine.c");
        build.file("external/hsdlib/src/similarity/dot.c");

        // Enable optimizations
        build.opt_level(3);

        // Architecture-specific SIMD flags
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

        match target_arch.as_str() {
            "x86_64" | "x86" => {
                // Enable AVX/AVX2/FMA for Intel/AMD
                build.flag_if_supported("-mavx");
                build.flag_if_supported("-mavx2");
                build.flag_if_supported("-mfma");
            }
            "aarch64" => {
                // ARM64 has NEON by default, but enable it explicitly
                build.flag_if_supported("-march=armv8-a");
            }
            "arm" => {
                // ARM32 NEON
                build.flag_if_supported("-mfpu=neon");
            }
            _ => {}
        }

        // Compile as static library
        build.compile("hsd");

        println!("cargo:rerun-if-changed=external/hsdlib/include/hsdlib.h");
        println!("cargo:rerun-if-changed=external/hsdlib/src/");
    }
}
