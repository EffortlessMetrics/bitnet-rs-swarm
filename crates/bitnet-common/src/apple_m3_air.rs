//! Canonical Apple M3 MacBook Air device identity.
//!
//! This module owns the exact M3 Air backend labels and profile facts used by
//! CLI config, device config, probes, backend selection, and Apple receipt code.
//! Historical shorthand labels such as `apple-m3-air`, `m3-air-*`, or generic
//! `metal`/`mpsgraph` are intentionally not aliases for this proof lane.

/// Canonical M3 Air backend labels in operator-facing help order.
#[macro_export]
macro_rules! apple_m3_air_backend_labels_csv {
    () => {
        "apple-m3-air-metal, apple-m3-air-mpsgraph, apple-m3-air-cpu-neon"
    };
}

pub const PROFILE_ID: &str = "apple-m3-macbook-air";
pub const MACHINE_ID: &str = PROFILE_ID;
pub const SOC_FAMILY: &str = "Apple M3";
pub const THERMAL_POLICY: &str = "fanless_mobile";
pub const MODEL_NAME: &str = "MacBook Air";
pub const MODEL_IDENTIFIERS: &[&str] = &["Mac15,12", "Mac15,13"];

pub const METAL_BACKEND: &str = "apple-m3-air-metal";
pub const MPSGRAPH_BACKEND: &str = "apple-m3-air-mpsgraph";
pub const CPU_NEON_BACKEND: &str = "apple-m3-air-cpu-neon";

pub const METAL_RUNTIME_API: &str = "metal";
pub const MPSGRAPH_RUNTIME_API: &str = "mpsgraph";
pub const CPU_NEON_RUNTIME_API: &str = "cpu-neon";

pub const CPU_NEON_CLAIM_SCOPE: &str =
    "M3 Air Apple CPU/NEON dense SLM and receipt-checked host evidence only";
pub const METAL_CLAIM_SCOPE: &str =
    "M3 Air Metal visibility/request identity only until receipt-backed runtime work lands";
pub const MPSGRAPH_CLAIM_SCOPE: &str =
    "M3 Air MPSGraph visibility/request identity only until receipt-backed runtime work lands";

pub const BACKEND_LABELS_CSV: &str = crate::apple_m3_air_backend_labels_csv!();
pub const DEVICE_LABELS_TEXT: &str = "apple-m3-air-metal = strict request identity for future M3 MacBook Air Metal receipts, apple-m3-air-mpsgraph = strict request identity for future M3 MacBook Air MPSGraph/reference receipts, apple-m3-air-cpu-neon = M3 MacBook Air Apple CPU/NEON lane";
pub const BACKEND_LABELS: [&str; 3] = [METAL_BACKEND, MPSGRAPH_BACKEND, CPU_NEON_BACKEND];
pub const VISIBILITY_PREFLIGHT_BACKENDS: [&str; 2] = [METAL_BACKEND, MPSGRAPH_BACKEND];

pub const REJECTED_BACKEND_ALIASES: &[&str] = &[
    "apple-m3-air",
    "m3-air-metal",
    "m3-air-mpsgraph",
    "m3-air-cpu-neon",
    "apple-m3-metal",
    "apple-m3-mpsgraph",
    "apple-m3-cpu-neon",
    "apple-m3-macbook-air-metal",
    "apple-m3-macbook-air-mpsgraph",
    "apple-m3-macbook-air-cpu-neon",
    "mac15_13_m3_air_local",
];

/// Stable proof-lane label carried by the M3 Air device profile contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofLabel {
    pub backend_label: &'static str,
    pub runtime_api: &'static str,
    pub execution_available: bool,
    pub claim_scope: &'static str,
}

/// Storage and artifact-retention policy for the M3 Air profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StoragePolicy {
    pub cache_root_required: bool,
    pub large_artifact_sweep_allowed: bool,
    pub model_binaries_committed: bool,
}

/// Claims explicitly rejected by the M3 Air profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnsupportedClaim {
    MetalModelInference,
    MpsGraphModelInference,
    NeuralEngineExecution,
    Qk256AppleSilicon,
    M4MacMiniPerformance,
    BroadAppleSiliconPerformance,
    BitNetLocalAnswerQualityFromDenseSlm,
}

/// Shared host/profile contract for the M3 Air lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostProfileContract {
    pub profile_id: &'static str,
    pub machine_id: &'static str,
    pub soc_family: &'static str,
    pub thermal_policy: &'static str,
    pub core_split_required: bool,
    pub memory_tier_required: bool,
    pub storage: StoragePolicy,
    pub labels: &'static [ProofLabel],
    pub unsupported_claims: &'static [UnsupportedClaim],
}

pub const STORAGE_POLICY: StoragePolicy = StoragePolicy {
    cache_root_required: true,
    large_artifact_sweep_allowed: true,
    model_binaries_committed: false,
};

pub const LABELS: [ProofLabel; 3] = [
    ProofLabel {
        backend_label: CPU_NEON_BACKEND,
        runtime_api: CPU_NEON_RUNTIME_API,
        execution_available: true,
        claim_scope: CPU_NEON_CLAIM_SCOPE,
    },
    ProofLabel {
        backend_label: METAL_BACKEND,
        runtime_api: METAL_RUNTIME_API,
        execution_available: false,
        claim_scope: METAL_CLAIM_SCOPE,
    },
    ProofLabel {
        backend_label: MPSGRAPH_BACKEND,
        runtime_api: MPSGRAPH_RUNTIME_API,
        execution_available: false,
        claim_scope: MPSGRAPH_CLAIM_SCOPE,
    },
];

pub const UNSUPPORTED_CLAIMS: [UnsupportedClaim; 7] = [
    UnsupportedClaim::MetalModelInference,
    UnsupportedClaim::MpsGraphModelInference,
    UnsupportedClaim::NeuralEngineExecution,
    UnsupportedClaim::Qk256AppleSilicon,
    UnsupportedClaim::M4MacMiniPerformance,
    UnsupportedClaim::BroadAppleSiliconPerformance,
    UnsupportedClaim::BitNetLocalAnswerQualityFromDenseSlm,
];

pub const CONTRACT: HostProfileContract = HostProfileContract {
    profile_id: PROFILE_ID,
    machine_id: MACHINE_ID,
    soc_family: SOC_FAMILY,
    thermal_policy: THERMAL_POLICY,
    core_split_required: true,
    memory_tier_required: true,
    storage: STORAGE_POLICY,
    labels: &LABELS,
    unsupported_claims: &UNSUPPORTED_CLAIMS,
};

#[must_use]
pub fn label(backend_label: &str) -> Option<ProofLabel> {
    LABELS.iter().copied().find(|label| label.backend_label == backend_label)
}

#[must_use]
pub fn is_backend_label(label: &str) -> bool {
    BACKEND_LABELS.contains(&label)
}

#[must_use]
pub fn is_visibility_preflight_backend(label: &str) -> bool {
    VISIBILITY_PREFLIGHT_BACKENDS.contains(&label)
}

#[must_use]
pub fn is_model_identifier(identifier: &str) -> bool {
    MODEL_IDENTIFIERS.contains(&identifier)
}

#[must_use]
pub fn rejects(claim: UnsupportedClaim) -> bool {
    UNSUPPORTED_CLAIMS.contains(&claim)
}
