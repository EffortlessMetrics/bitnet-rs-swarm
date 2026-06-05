use crate::DeviceConfig;
use bitnet_common::apple_m3_air;

/// Thermal policy class that affects how performance evidence must be scoped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalPolicy {
    /// Mobile Apple Silicon host with no active fan.
    FanlessMobile,
}

pub use bitnet_common::apple_m3_air::{
    ProofLabel as DeviceProfileLabel, StoragePolicy as DeviceProfileStoragePolicy,
    UnsupportedClaim as DeviceProfileUnsupportedClaim,
};

/// Structured host/profile contract for proof-lane routing and receipt wording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProfileContract {
    pub profile_id: &'static str,
    pub machine_id: &'static str,
    pub soc_family: &'static str,
    pub thermal_policy: ThermalPolicy,
    pub core_split_required: bool,
    pub memory_tier_required: bool,
    pub storage: DeviceProfileStoragePolicy,
    pub labels: &'static [DeviceProfileLabel],
    pub unsupported_claims: &'static [DeviceProfileUnsupportedClaim],
}

impl DeviceProfileContract {
    #[must_use]
    pub fn apple_m3_air() -> Self {
        Self {
            profile_id: apple_m3_air::CONTRACT.profile_id,
            machine_id: apple_m3_air::CONTRACT.machine_id,
            soc_family: apple_m3_air::CONTRACT.soc_family,
            thermal_policy: ThermalPolicy::FanlessMobile,
            core_split_required: apple_m3_air::CONTRACT.core_split_required,
            memory_tier_required: apple_m3_air::CONTRACT.memory_tier_required,
            storage: apple_m3_air::CONTRACT.storage,
            labels: apple_m3_air::CONTRACT.labels,
            unsupported_claims: apple_m3_air::CONTRACT.unsupported_claims,
        }
    }

    #[must_use]
    pub fn label(&self, backend_label: &str) -> Option<DeviceProfileLabel> {
        self.labels.iter().copied().find(|label| label.backend_label == backend_label)
    }

    #[must_use]
    pub fn rejects(&self, claim: DeviceProfileUnsupportedClaim) -> bool {
        self.unsupported_claims.contains(&claim)
    }
}

impl DeviceConfig {
    /// Return the structured host/profile contract for proof-lane-specific labels.
    #[must_use]
    pub fn device_profile_contract(&self) -> Option<DeviceProfileContract> {
        match self {
            DeviceConfig::AppleM3AirMetal
            | DeviceConfig::AppleM3AirMpsGraph
            | DeviceConfig::AppleM3AirCpuNeon => Some(DeviceProfileContract::apple_m3_air()),
            _ => None,
        }
    }
}
