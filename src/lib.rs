//! # bitnet-rs — 1-bit LLM Inference Engine
//!
//! Pre-alpha (v0.2.1-dev) Rust inference engine for `BitNet` 1-bit large language models.
//!
//! ## Status
//!
//! This is **pre-alpha software**. Correctness, performance, and validation work is ongoing.
//! CPU inference with SIMD optimization works; GPU backends are scaffolded but not validated.
//! Do not use in production.
//!
//! ## Feature Flags
//!
//! Default features are **empty** — always specify features explicitly:
//!
//! - `cpu`: SIMD-optimised CPU inference (AVX2 / AVX-512 / NEON)
//! - `gpu`: GPU acceleration (CUDA umbrella; requires CUDA 12.x)
//! - `full-cli`: Enable all CLI subcommands
//! - `ffi`: C++ FFI bridge for cross-validation
//! - `fixtures`: GGUF fixture-based integration tests (test-only)
//!
//! ## Architecture
//!
//! The workspace contains ~200 crates. Key crates:
//!
//! - [`bitnet_common`]: Shared types, config, error types
//! - [`bitnet_models`]: GGUF / `SafeTensors` model loading
//! - [`bitnet_quantization`]: `I2_S`, `TL1`, `TL2` quantization
//! - `bitnet_kernels`: AVX2 / AVX-512 / NEON / CUDA compute kernels
//! - `bitnet_inference`: Autoregressive generation engine
//! - `bitnet_tokenizers`: Universal tokenizer with auto-discovery
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use bitnet::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a BitNet model config (load tensors from GGUF in practice)
//! let device = Device::Cpu;
//! let config = BitNetConfig::default();
//! let _model = BitNetModel::new(config, device);
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

/// Build metadata captured at compile time.
pub mod build_info;
mod constants;
/// Convenient imports for common `BitNet` types and traits.
pub mod prelude;

pub use build_info::*;
pub use constants::{MSRV, VERSION};

// Re-export core functionality
pub use bitnet_common as common;
pub use bitnet_models as models;
pub use bitnet_quantization as quantization;

#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use bitnet_inference as inference;

#[cfg(feature = "tokenizers")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizers")))]
pub use bitnet_tokenizers as tokenizers;

#[cfg(feature = "kernels")]
#[cfg_attr(docsrs, doc(cfg(feature = "kernels")))]
pub use bitnet_kernels as kernels;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        #[allow(clippy::const_is_empty)]
        {
            assert!(!VERSION.is_empty());
        }
        // Version is read from Cargo.toml via env!(); don't hardcode it here
        assert!(VERSION.starts_with("0."), "expected 0.x version, got {VERSION}");
    }

    #[test]
    fn test_msrv() -> Result<(), Box<dyn std::error::Error>> {
        // The MSRV constant is the public, programmatic declaration of the
        // minimum supported Rust version. It MUST agree with the two other
        // repo sources of truth: `rust-toolchain.toml` (the toolchain CI
        // actually pins and uses) and `Cargo.toml`'s `rust-version` field
        // (what cargo enforces). This test catches drift between them.
        //
        // Returns Result and uses `?`/`Err` rather than `unwrap`/`panic!` so
        // the no-panic no-new-debt gate stays clean (see policy/no-panic-*).
        //
        // If this test fails, one of the three has drifted; reconcile them
        // and update the hardcoded literal below.
        const EXPECTED_MSRV: &str = "1.95.0";

        // 1. The constant itself must match the expected MSRV.
        assert_eq!(MSRV, EXPECTED_MSRV, "src/constants.rs MSRV drifted");

        // 2. rust-toolchain.toml must pin the same channel.
        //    Walk up from CARGO_MANIFEST_DIR (the crate root ".") to find it.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let toolchain_path = std::path::Path::new(manifest_dir).join("rust-toolchain.toml");
        let toolchain_content = std::fs::read_to_string(&toolchain_path)?;
        let toolchain_channel = toolchain_content
            .lines()
            .find_map(|line| {
                let trimmed = line.trim();
                trimmed.strip_prefix("channel = ")?.trim_matches('"').into()
            })
            .ok_or_else(|| format!("no `channel = ` line in {}", toolchain_path.display()))?;
        assert_eq!(
            toolchain_channel, EXPECTED_MSRV,
            "rust-toolchain.toml channel drifted from MSRV constant"
        );

        // 3. Cargo.toml's rust-version must match.
        let cargo_path = std::path::Path::new(manifest_dir).join("Cargo.toml");
        let cargo_content = std::fs::read_to_string(&cargo_path)?;
        let cargo_rust_version = cargo_content
            .lines()
            .find_map(|line| {
                let trimmed = line.trim();
                trimmed.strip_prefix("rust-version = ")?.trim_matches('"').into()
            })
            .ok_or_else(|| format!("no `rust-version = ` line in {}", cargo_path.display()))?;
        assert_eq!(
            cargo_rust_version, EXPECTED_MSRV,
            "Cargo.toml rust-version drifted from MSRV constant"
        );
        Ok(())
    }

    #[test]
    fn test_build_info() {
        // These should not panic
        let _ = build_info::GIT_HASH;
        let _ = build_info::BUILD_TIMESTAMP;
        let _ = build_info::TARGET;
        let _ = build_info::RUSTC_VERSION;
        let _ = GIT_HASH;
        let _ = BUILD_TIMESTAMP;
        let _ = TARGET;
        let _ = RUSTC_VERSION;
    }

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;
        // Test that prelude imports work
        let _config = BitNetConfig::default();
    }
}
