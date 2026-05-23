#![recursion_limit = "256"]

use serde_json::{Value, json};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_RECEIPT_OUT: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/strict-bitnet-repeated-profiles.json";
const CAMPAIGN_ITEM: &str = "CUDA-BITNET-PERF-005";

const DEFAULT_ONE_TOKEN_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-01/official-bitnet-one-token.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-02/official-bitnet-one-token.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-03/official-bitnet-one-token.json",
];
const DEFAULT_SHORT_DECODE_8_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-01/official-bitnet-short-decode-8.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-02/official-bitnet-short-decode-8.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-03/official-bitnet-short-decode-8.json",
];
const DEFAULT_SHORT_DECODE_32_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-01/official-bitnet-short-decode-32.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-02/official-bitnet-short-decode-32.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-03/official-bitnet-short-decode-32.json",
];
const DEFAULT_PREFILL_128_DECODE_16_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-01/official-bitnet-prefill-128-decode-16.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-02/official-bitnet-prefill-128-decode-16.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-03/official-bitnet-prefill-128-decode-16.json",
];
const DEFAULT_PREFILL_512_DECODE_32_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-01/official-bitnet-prefill-512-decode-32.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-02/official-bitnet-prefill-512-decode-32.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-03/official-bitnet-prefill-512-decode-32.json",
];
const DEFAULT_WARM_SESSION_3_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-01/official-bitnet-warm-session-3.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-02/official-bitnet-warm-session-3.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-03/official-bitnet-warm-session-3.json",
];
const DEFAULT_WARM_SESSION_10_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-01/official-bitnet-warm-session-10.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-02/official-bitnet-warm-session-10.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-03/official-bitnet-warm-session-10.json",
];
const DEFAULT_DECODE_128_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-01/official-bitnet-decode-128-from-warm-context.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-02/official-bitnet-decode-128-from-warm-context.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-03/official-bitnet-decode-128-from-warm-context.json",
];

#[derive(Debug)]
struct Args {
    one_token_runs: Vec<PathBuf>,
    short_decode_8_runs: Vec<PathBuf>,
    short_decode_32_runs: Vec<PathBuf>,
    prefill_128_decode_16_runs: Vec<PathBuf>,
    prefill_512_decode_32_runs: Vec<PathBuf>,
    warm_session_3_runs: Vec<PathBuf>,
    warm_session_10_runs: Vec<PathBuf>,
    decode_128_runs: Vec<PathBuf>,
    receipt_out: PathBuf,
    manifest_out: Option<PathBuf>,
    print_manifest: bool,
}

struct ProfileInputGroup<'a> {
    profile: &'static str,
    flag: &'static str,
    expected_input_tokens: Option<u64>,
    expected_generated_tokens: u64,
    paths: &'a [PathBuf],
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.print_manifest || args.manifest_out.is_some() {
        let manifest_json = serde_json::to_string_pretty(&source_manifest(&args))?;
        if args.print_manifest {
            println!("{manifest_json}");
        }
        if let Some(path) = &args.manifest_out {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, manifest_json)?;
        }
        return Ok(());
    }

    assert_input_paths_exist(&args)?;
    Err("aggregate generation for CUDA-BITNET-PERF-005 is intentionally gated until current-source profile receipts are committed; use --print-manifest or --manifest-out PATH for the governed capture contract".into())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut one_token_runs = default_paths(DEFAULT_ONE_TOKEN_RUNS);
    let mut short_decode_8_runs = default_paths(DEFAULT_SHORT_DECODE_8_RUNS);
    let mut short_decode_32_runs = default_paths(DEFAULT_SHORT_DECODE_32_RUNS);
    let mut prefill_128_decode_16_runs = default_paths(DEFAULT_PREFILL_128_DECODE_16_RUNS);
    let mut prefill_512_decode_32_runs = default_paths(DEFAULT_PREFILL_512_DECODE_32_RUNS);
    let mut warm_session_3_runs = default_paths(DEFAULT_WARM_SESSION_3_RUNS);
    let mut warm_session_10_runs = default_paths(DEFAULT_WARM_SESSION_10_RUNS);
    let mut decode_128_runs = default_paths(DEFAULT_DECODE_128_RUNS);
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut manifest_out = None;
    let mut print_manifest = false;
    let mut iter = env::args().skip(1);

    let mut one_token_overridden = false;
    let mut short_decode_8_overridden = false;
    let mut short_decode_32_overridden = false;
    let mut prefill_128_decode_16_overridden = false;
    let mut prefill_512_decode_32_overridden = false;
    let mut warm_session_3_overridden = false;
    let mut warm_session_10_overridden = false;
    let mut decode_128_overridden = false;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--one-token-run" => push_override(
                &mut one_token_runs,
                &mut one_token_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--short-decode-8-run" => push_override(
                &mut short_decode_8_runs,
                &mut short_decode_8_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--short-decode-32-run" => push_override(
                &mut short_decode_32_runs,
                &mut short_decode_32_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--prefill-128-decode-16-run" => push_override(
                &mut prefill_128_decode_16_runs,
                &mut prefill_128_decode_16_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--prefill-512-decode-32-run" => push_override(
                &mut prefill_512_decode_32_runs,
                &mut prefill_512_decode_32_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--warm-session-3-run" => push_override(
                &mut warm_session_3_runs,
                &mut warm_session_3_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--warm-session-10-run" => push_override(
                &mut warm_session_10_runs,
                &mut warm_session_10_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--decode-128-from-warm-context-run" => push_override(
                &mut decode_128_runs,
                &mut decode_128_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--receipt-out" => receipt_out = PathBuf::from(next_value(&mut iter, &arg)?),
            "--manifest-out" => manifest_out = Some(PathBuf::from(next_value(&mut iter, &arg)?)),
            "--print-manifest" => print_manifest = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args {
        one_token_runs,
        short_decode_8_runs,
        short_decode_32_runs,
        prefill_128_decode_16_runs,
        prefill_512_decode_32_runs,
        warm_session_3_runs,
        warm_session_10_runs,
        decode_128_runs,
        receipt_out,
        manifest_out,
        print_manifest,
    })
}

fn default_paths<const N: usize>(paths: [&str; N]) -> Vec<PathBuf> {
    paths.into_iter().map(PathBuf::from).collect()
}

fn push_override(paths: &mut Vec<PathBuf>, overridden: &mut bool, value: String) {
    if !*overridden {
        paths.clear();
        *overridden = true;
    }
    paths.push(PathBuf::from(value));
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: strict_bitnet_cuda_repeated_profiles_receipt [--one-token-run PATH ...] [--short-decode-8-run PATH ...] [--short-decode-32-run PATH ...] [--prefill-128-decode-16-run PATH ...] [--prefill-512-decode-32-run PATH ...] [--warm-session-3-run PATH ...] [--warm-session-10-run PATH ...] [--decode-128-from-warm-context-run PATH ...] [--receipt-out PATH] [--manifest-out PATH] [--print-manifest]"
    );
}

fn input_groups(args: &Args) -> Vec<ProfileInputGroup<'_>> {
    vec![
        ProfileInputGroup {
            profile: "one_token",
            flag: "--one-token-run",
            expected_input_tokens: None,
            expected_generated_tokens: 1,
            paths: &args.one_token_runs,
        },
        ProfileInputGroup {
            profile: "short_decode_8",
            flag: "--short-decode-8-run",
            expected_input_tokens: None,
            expected_generated_tokens: 8,
            paths: &args.short_decode_8_runs,
        },
        ProfileInputGroup {
            profile: "short_decode_32",
            flag: "--short-decode-32-run",
            expected_input_tokens: None,
            expected_generated_tokens: 32,
            paths: &args.short_decode_32_runs,
        },
        ProfileInputGroup {
            profile: "prefill_128_decode_16",
            flag: "--prefill-128-decode-16-run",
            expected_input_tokens: Some(128),
            expected_generated_tokens: 16,
            paths: &args.prefill_128_decode_16_runs,
        },
        ProfileInputGroup {
            profile: "prefill_512_decode_32",
            flag: "--prefill-512-decode-32-run",
            expected_input_tokens: Some(512),
            expected_generated_tokens: 32,
            paths: &args.prefill_512_decode_32_runs,
        },
        ProfileInputGroup {
            profile: "warm_session_3_turns",
            flag: "--warm-session-3-run",
            expected_input_tokens: None,
            expected_generated_tokens: 24,
            paths: &args.warm_session_3_runs,
        },
        ProfileInputGroup {
            profile: "warm_session_10_turns",
            flag: "--warm-session-10-run",
            expected_input_tokens: None,
            expected_generated_tokens: 80,
            paths: &args.warm_session_10_runs,
        },
        ProfileInputGroup {
            profile: "decode_128_from_warm_context",
            flag: "--decode-128-from-warm-context-run",
            expected_input_tokens: None,
            expected_generated_tokens: 128,
            paths: &args.decode_128_runs,
        },
    ]
}

fn source_manifest(args: &Args) -> Value {
    let profiles = input_groups(args)
        .into_iter()
        .map(|group| {
            json!({
                "profile": group.profile,
                "run_flag": group.flag,
                "min_runs": 3,
                "expected_input_tokens": group.expected_input_tokens,
                "expected_generated_tokens": group.expected_generated_tokens,
                "source_paths": group.paths.iter().map(|path| path_label(path)).collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();

    json!({
        "schema": 1,
        "artifact_kind": "strict_bitnet_cuda_repeated_profiles_source_manifest",
        "campaign_item": CAMPAIGN_ITEM,
        "aggregate_artifact_kind": "strict_bitnet_cuda_repeated_profiles",
        "aggregate_receipt_out": path_label(&args.receipt_out),
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "selected_route": "bitnet_qk256_cuda",
        "kernel_id": "qk256_gemv_cuda",
        "min_runs_per_profile": 3,
        "profiles": profiles,
        "model": {
            "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
            "file": "ggml-model-i2_s.gguf",
            "sha256": "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162",
            "format": "gguf",
            "architecture": "bitnet_b1_58",
            "quantization_layout": "I2_S/QK256"
        },
        "tokenizer_prompt_authority": {
            "tokenizer_authority": "external_tokenizer",
            "prompt_authority": "bitnetcpp-answer",
            "prompt_template": "bitnetcpp-answer"
        },
        "required_source_fields": [
            "/artifact_kind",
            "/model/repo",
            "/model/file",
            "/model/sha256",
            "/tokenizer/source",
            "/tokenizer/pretokenizer_authority",
            "/execution_plan/selected_route",
            "/selected_backend",
            "/runtime_api",
            "/fallback_used",
            "/quality_summary/passed or /quality/garbage_filter_passed",
            "/timing/model_load_ms",
            "/timing/tokenizer_load_ms",
            "/timing/prompt_render_ms",
            "/timing/tokenize_ms",
            "/timing/cuda_context_init_ms",
            "/timing/weight_upload_ms",
            "/timing/prefill_ms",
            "/timing/first_token_ms",
            "/timing/decode_total_ms",
            "/timing/steady_tok_per_s",
            "/kernel_stats/0/kernel_id",
            "/kernel_stats/0/kernel_time_ms",
            "/kernel_stats/0/invocations",
            "/kernel_stats/0/fallback_invocations",
            "/timing/host_to_device_bytes or /kernel_stats/0/host_to_device_bytes",
            "/timing/host_to_device_ms or explicit source blocker",
            "/timing/device_to_host_bytes or /kernel_stats/0/device_to_host_bytes",
            "/timing/device_to_host_ms or explicit source blocker",
            "/cuda/memory_hwm_bytes or /cuda/vram_bytes",
            "/cuda/power_draw_watts",
            "/cuda/temperature_c"
        ],
        "strict_rejection_rules": [
            "selected_backend must be nvidia-rtx-5070-ti-cuda",
            "runtime_api must be cuda for accelerator receipts",
            "selected_route must be bitnet_qk256_cuda",
            "kernel_id must be qk256_gemv_cuda",
            "fallback_used must be false",
            "kernel fallback invocations must be zero",
            "dense_regular_llm_cuda receipts are rejected for this proof family",
            "generic cuda backend labels are rejected for strict RTX 5070 Ti claims"
        ],
        "claim_boundaries": [
            "source manifest is not a receipt and does not prove hardware execution",
            "speedup_claim=false until a separate exact-profile review accepts a profile",
            "benchmark_qualified_speedup=false until CUDA-BITNET-PERF-006",
            "full_cuda_residency_claimed=false until every required phase proves residency",
            "broad_server_readiness=false; existing server evidence remains smoke or exact-profile only",
            "BitNet packed I2_S/QK256 proof cannot be satisfied by dense_regular_llm_cuda evidence",
            "dense regular-LLM CUDA proof cannot be satisfied by BitNet QK256 evidence"
        ]
    })
}

fn assert_input_paths_exist(args: &Args) -> Result<(), Box<dyn Error>> {
    let missing = input_groups(args)
        .into_iter()
        .flat_map(|group| {
            group.paths.iter().filter_map(move |path| {
                if path.is_file() {
                    None
                } else {
                    Some(format!("{}: {}", group.profile, path_label(path)))
                }
            })
        })
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    Err(format!(
        "missing CUDA-BITNET-PERF-005 source receipts:\n  - {}\nrun with --print-manifest or --manifest-out PATH to inspect the expected capture set",
        missing.join("\n  - ")
    )
    .into())
}

fn path_label(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_with_prefix(prefix: &str) -> Args {
        Args {
            one_token_runs: vec![PathBuf::from(format!("{prefix}/one-token.json"))],
            short_decode_8_runs: vec![PathBuf::from(format!("{prefix}/short-decode-8.json"))],
            short_decode_32_runs: vec![PathBuf::from(format!("{prefix}/short-decode-32.json"))],
            prefill_128_decode_16_runs: vec![PathBuf::from(format!(
                "{prefix}/prefill-128-decode-16.json"
            ))],
            prefill_512_decode_32_runs: vec![PathBuf::from(format!(
                "{prefix}/prefill-512-decode-32.json"
            ))],
            warm_session_3_runs: vec![PathBuf::from(format!("{prefix}/warm-session-3.json"))],
            warm_session_10_runs: vec![PathBuf::from(format!("{prefix}/warm-session-10.json"))],
            decode_128_runs: vec![PathBuf::from(format!(
                "{prefix}/decode-128-from-warm-context.json"
            ))],
            receipt_out: PathBuf::from(format!("{prefix}/aggregate.json")),
            manifest_out: None,
            print_manifest: false,
        }
    }

    #[test]
    fn bitnet_perf_005_source_manifest_names_profiles_and_boundaries() {
        let args = args_with_prefix("target/bitnet-perf-005");
        let manifest = source_manifest(&args);

        assert_eq!(
            manifest["artifact_kind"],
            "strict_bitnet_cuda_repeated_profiles_source_manifest"
        );
        assert_eq!(manifest["campaign_item"], CAMPAIGN_ITEM);
        assert_eq!(manifest["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(manifest["selected_route"], "bitnet_qk256_cuda");
        assert_eq!(manifest["kernel_id"], "qk256_gemv_cuda");
        assert_eq!(
            manifest["model"]["sha256"],
            "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
        );

        let profiles = manifest["profiles"].as_array();
        assert_eq!(profiles.map(Vec::len), Some(8));
        assert!(profiles.is_some_and(|profiles| {
            profiles.iter().any(|profile| {
                profile["profile"] == "prefill_512_decode_32"
                    && profile["expected_input_tokens"] == 512
                    && profile["expected_generated_tokens"] == 32
            })
        }));
        assert!(profiles.is_some_and(|profiles| {
            profiles.iter().any(|profile| {
                profile["profile"] == "decode_128_from_warm_context"
                    && profile["run_flag"] == "--decode-128-from-warm-context-run"
            })
        }));
        assert!(manifest["strict_rejection_rules"].as_array().is_some_and(|rules| {
            rules.iter().any(|rule| {
                rule.as_str().is_some_and(|text| {
                    text.contains("dense_regular_llm_cuda receipts are rejected")
                })
            })
        }));
        assert!(manifest["claim_boundaries"].as_array().is_some_and(|boundaries| {
            boundaries.iter().any(|boundary| {
                boundary
                    .as_str()
                    .is_some_and(|text| text.contains("does not prove hardware execution"))
            })
        }));
    }

    #[test]
    fn bitnet_perf_005_preflight_reports_all_missing_profile_inputs() {
        let args = args_with_prefix("target/bitnet-perf-005/missing");
        let message = match assert_input_paths_exist(&args) {
            Ok(()) => String::new(),
            Err(err) => err.to_string(),
        };
        assert!(!message.is_empty(), "missing inputs should fail");

        for profile in [
            "one_token",
            "short_decode_8",
            "short_decode_32",
            "prefill_128_decode_16",
            "prefill_512_decode_32",
            "warm_session_3_turns",
            "warm_session_10_turns",
            "decode_128_from_warm_context",
        ] {
            assert!(message.contains(profile), "missing {profile} in {message}");
        }
        assert!(message.contains("--print-manifest"), "missing manifest hint in {message}");
    }
}
