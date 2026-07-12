//! Typed operator profiles for the dense SLM warm-session command.

#[cfg(feature = "full-cli")]
pub mod kaby;
#[cfg(feature = "full-cli")]
pub mod receipt;
#[cfg(feature = "full-cli")]
pub mod resolve;
#[cfg(feature = "full-cli")]
pub mod self_test;

#[cfg(feature = "full-cli")]
pub use receipt::profile_receipt;
#[cfg(feature = "full-cli")]
pub use resolve::{
    CliOverrides, LoadedModelMetadata, inspect_model_metadata, resolve_profile,
    validate_profile_request,
};
#[cfg(feature = "full-cli")]
pub use self_test::{ProfileGate, ProfilePromptInput, profile_prompt_inputs};

#[cfg(feature = "full-cli")]
use anyhow::{Result, bail};
#[cfg(feature = "full-cli")]
use clap::{Args, Subcommand};
#[cfg(feature = "full-cli")]
use serde_json::json;
#[cfg(feature = "full-cli")]
use std::path::{Path, PathBuf};

#[cfg(feature = "full-cli")]
#[derive(Debug, Args)]
pub struct ProfileCommand {
    #[command(subcommand)]
    pub action: ProfileAction,
}

#[cfg(feature = "full-cli")]
#[derive(Debug, Subcommand)]
pub enum ProfileAction {
    /// Show the supported Kaby profile contract without loading a model.
    Show {
        /// Profile identifier.
        id: String,
        /// Emit the contract as JSON.
        #[arg(long)]
        json: bool,
    },
}

#[cfg(feature = "full-cli")]
#[derive(Debug, Args)]
pub struct DoctorCommand {
    /// Profile identifier to diagnose.
    #[arg(long)]
    pub profile: String,
    /// Verified GGUF artifact to inspect.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Optional explicit tokenizer authority.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Emit JSON instead of the concise operator summary.
    #[arg(long)]
    pub json: bool,
    /// Write the doctor report to a JSON file.
    #[arg(long)]
    pub json_out: Option<PathBuf>,
}

#[cfg(feature = "full-cli")]
pub async fn execute_profile_command(command: ProfileCommand) -> Result<()> {
    match command.action {
        ProfileAction::Show { id, json: as_json } => {
            let report = profile_contract(&id)?;
            if as_json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("profile: {}", report["profile_id"]);
                println!("primary model: {}", report["models"][0]["file"]);
                println!("second model: {}", report["models"][1]["file"]);
                println!(
                    "backend: {} (fallback={})",
                    report["selected_backend"], report["fallback"]
                );
                println!("threads: {}", report["defaults"]["threads"]);
                println!(
                    "normal mode: prompts required; use --self-test for bounded certification"
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "full-cli")]
pub async fn execute_doctor_command(command: DoctorCommand, requested_backend: &str) -> Result<()> {
    validate_profile_request(Some(command.profile.as_str()), requested_backend)?;
    let mut blockers = Vec::new();
    let metadata = match command.model.as_deref() {
        Some(path) => match inspect_model_metadata(path, command.tokenizer.as_deref()) {
            Ok(metadata) => Some(metadata),
            Err(error) => {
                blockers.push(error.to_string());
                None
            }
        },
        None => {
            blockers.push(
                "model artifact is missing; pass --model <verified.gguf> after `bitnet model verify`"
                    .to_string(),
            );
            None
        }
    };
    if !matches!(requested_backend, "cpu" | "auto") {
        blockers.push(format!(
            "profile requires --device cpu; requested backend is {requested_backend}"
        ));
    }
    let profile_result = metadata.as_ref().map(|metadata| {
        kaby::classify_model(
            &metadata.architecture,
            &metadata.quant_format,
            &metadata.model_sha256,
            &metadata.tokenizer_authority,
            metadata.chat_template.as_deref(),
            metadata.context_limit,
        )
    });
    if let Some(Err(error)) = profile_result.as_ref() {
        blockers.push(error.to_string());
    }
    let resources = resource_report(command.model.as_deref());
    if resources["memory"]["available_bytes"].as_u64().is_some_and(|bytes| bytes < MIN_MEMORY_BYTES)
    {
        blockers.push(format!(
            "available RAM is below the profile minimum of {} bytes",
            MIN_MEMORY_BYTES
        ));
    }
    if resources["storage"]["available_bytes"]
        .as_u64()
        .is_some_and(|bytes| bytes < MIN_STORAGE_HEADROOM_BYTES)
    {
        blockers.push(format!(
            "available storage is below the profile headroom of {} bytes",
            MIN_STORAGE_HEADROOM_BYTES
        ));
    }
    let ready = blockers.is_empty();
    let report = json!({
        "schema_version": "1.0.0",
        "artifact_kind": "slm_cpu_profile_doctor",
        "profile_id": command.profile,
        "result": if ready { "pass" } else { "fail" },
        "ready": ready,
        "requested_backend": requested_backend,
        "model_path": command.model.as_ref().map(|path| path.display().to_string()),
        "metadata": metadata.clone(),
        "resources": resources,
        "thread_recommendation": {
            "recommended": kaby::RECOMMENDED_THREADS,
            "available_parallelism": std::thread::available_parallelism().map(|n| n.get()).ok(),
        },
        "tokenizer_authority": metadata.as_ref().map(|m| json!({
            "source": m.tokenizer_source,
            "authority": m.tokenizer_authority,
            "strict": m.tokenizer_strict,
        })),
        "supported_optimizations": {
            "default_path": "eager_f32_candle",
            "proven_no_bias_role": "feed_forward.down_proj",
            "packed_q8_runtime": false,
            "candidate_execution": false,
        },
        "thermal_power": {
            "available": false,
            "reason": "platform telemetry is not exposed by the portable doctor path",
        },
        "blockers": blockers.clone(),
        "next_steps": [
            "verify the exact GGUF artifact and tokenizer before retrying",
            "use --self-test for bounded quality and determinism certification",
            "use --allocation-audit only when collecting allocation evidence",
        ],
    });
    if let Some(path) = command.json_out.as_deref() {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    if command.json || command.json_out.is_some() {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("profile doctor: {}", report["result"]);
        println!("model: {}", report["model_path"].as_str().unwrap_or("missing"));
        println!("threads: {} recommended", kaby::RECOMMENDED_THREADS);
        if !blockers.is_empty() {
            println!("blockers:");
            for blocker in &blockers {
                println!("- {blocker}");
            }
        }
    }
    if ready { Ok(()) } else { bail!("profile doctor found {} blocker(s)", blockers.len()) }
}

#[cfg(feature = "full-cli")]
const MIN_MEMORY_BYTES: u64 = 4 * 1024 * 1024 * 1024;
#[cfg(feature = "full-cli")]
const MIN_STORAGE_HEADROOM_BYTES: u64 = 1024 * 1024 * 1024;

#[cfg(feature = "full-cli")]
fn profile_contract(id: &str) -> Result<serde_json::Value> {
    if id != kaby::PROFILE_ID {
        bail!("unsupported profile {id}; supported profiles: {}", kaby::PROFILE_ID);
    }
    Ok(json!({
        "profile_id": kaby::PROFILE_ID,
        "models": [
            {
                "file": "Qwen3-0.6B-Q8_0.gguf",
                "architecture": "qwen3",
                "quantization": "Q8_0",
                "sha256": kaby::QWEN3_SHA256,
                "role": "primary",
                "prompt_template": kaby::ModelRole::Qwen3Primary.prompt_template(),
                "no_think": kaby::ModelRole::Qwen3Primary.no_think(),
                "max_new_tokens": kaby::ModelRole::Qwen3Primary.max_new_tokens(),
                "context_limit": kaby::ModelRole::Qwen3Primary.context_limit(),
                "self_test_corpus": kaby::ModelRole::Qwen3Primary.self_test_corpus(),
                "stop_policy": kaby::ModelRole::Qwen3Primary.stop_policy()
            },
            {
                "file": "qwen2.5-0.5b-instruct-q8_0.gguf",
                "architecture": "qwen2",
                "quantization": "Q8_0",
                "sha256": kaby::QWEN25_SHA256,
                "role": "second_model_proof",
                "prompt_template": kaby::ModelRole::Qwen25SecondModel.prompt_template(),
                "no_think": kaby::ModelRole::Qwen25SecondModel.no_think(),
                "max_new_tokens": kaby::ModelRole::Qwen25SecondModel.max_new_tokens(),
                "context_limit": kaby::ModelRole::Qwen25SecondModel.context_limit(),
                "self_test_corpus": kaby::ModelRole::Qwen25SecondModel.self_test_corpus(),
                "stop_policy": kaby::ModelRole::Qwen25SecondModel.stop_policy()
            }
        ],
        "runtime_api": "cpu",
        "selected_backend": "cpu-rust",
        "fallback": false,
        "defaults": {
            "threads": kaby::RECOMMENDED_THREADS,
            "strict_loader": true,
            "strict_tokenizer": true,
            "deterministic": true,
            "warm_session": true,
            "allocation_audit": false,
            "self_test": false,
        },
        "optimizations": {
            "default_path": "eager_f32_candle",
            "proven_no_bias_role": "feed_forward.down_proj",
            "packed_q8_runtime": false,
        },
        "unsupported": ["Q4/Q5 runtime", "server", "GPU/NPU/OpenVINO/UHD 620", "Qwen3.5", "sustained throughput"]
    }))
}

#[cfg(feature = "full-cli")]
fn resource_report(model_path: Option<&Path>) -> serde_json::Value {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let storage_path = model_path.unwrap_or_else(|| Path::new("."));
    let absolute_path = storage_path.canonicalize().unwrap_or_else(|_| storage_path.to_path_buf());
    let disk_lookup_path = absolute_path
        .to_str()
        .and_then(|path| path.strip_prefix("\\\\?\\"))
        .map(PathBuf::from)
        .unwrap_or_else(|| absolute_path.clone());
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let disk = disks
        .list()
        .iter()
        .filter(|disk| disk_lookup_path.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().components().count());
    let storage = disk.map_or_else(
        || json!({ "path": absolute_path.display().to_string(), "available_bytes": null, "total_bytes": null }),
        |disk| json!({ "path": absolute_path.display().to_string(), "mount_point": disk.mount_point().display().to_string(), "available_bytes": disk.available_space(), "total_bytes": disk.total_space() }),
    );
    json!({
        "cpu": {
            "arch": std::env::consts::ARCH,
            "avx2": cpu_avx2_available(),
            "fma": cpu_fma_available(),
        },
        "memory": {
            "available_bytes": system.available_memory(),
            "total_bytes": system.total_memory(),
        },
        "storage": storage,
    })
}

#[cfg(feature = "full-cli")]
fn cpu_avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(feature = "full-cli")]
fn cpu_fma_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(all(test, feature = "full-cli"))]
mod tests {
    use super::*;

    #[test]
    fn profile_contract_names_both_supported_models() -> anyhow::Result<()> {
        let report = match profile_contract(kaby::PROFILE_ID) {
            Ok(report) => report,
            Err(error) => return Err(error),
        };
        assert_eq!(report["models"].as_array().map(Vec::len), Some(2));
        assert_eq!(report["defaults"]["threads"], kaby::RECOMMENDED_THREADS);
        assert_eq!(report["defaults"]["allocation_audit"], false);
        assert_eq!(report["models"][0]["prompt_template"], "qwen");
        assert_eq!(report["models"][0]["max_new_tokens"], 4);
        assert_eq!(report["models"][0]["context_limit"], 40_960);
        assert_eq!(report["models"][1]["prompt_template"], "qwen2.5");
        assert_eq!(report["models"][1]["max_new_tokens"], 8);
        assert_eq!(report["models"][1]["context_limit"], 32_768);
        assert_ne!(
            report["models"][0]["self_test_corpus"],
            report["models"][1]["self_test_corpus"]
        );
        Ok(())
    }

    #[test]
    fn auto_backend_is_accepted_as_cpu_profile_default() {
        assert!(validate_profile_request(Some(kaby::PROFILE_ID), "auto").is_ok());
    }
}
