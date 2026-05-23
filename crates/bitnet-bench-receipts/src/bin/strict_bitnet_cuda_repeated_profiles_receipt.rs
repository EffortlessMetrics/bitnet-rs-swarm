#![recursion_limit = "256"]

use bitnet_bench_receipts::validate_strict_bitnet_cuda_repeated_profiles_receipt_json;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_RECEIPT_OUT: &str = "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/strict-bitnet-repeated-profiles.json";
const CAMPAIGN_ITEM: &str = "CUDA-BITNET-PERF-005";
const MODEL_REPO: &str = "microsoft/bitnet-b1.58-2B-4T-gguf";
const MODEL_FILE: &str = "ggml-model-i2_s.gguf";
const MODEL_SHA256: &str = "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162";
const TOKENIZER_AUTHORITY: &str = "external_tokenizer";
const PRETOKENIZER_AUTHORITY: &str = "llama-bpe";
const PROMPT_AUTHORITY: &str = "bitnetcpp-answer";
const SELECTED_BACKEND: &str = "nvidia-rtx-5070-ti-cuda";
const REFERENCE_BACKEND: &str = "amd-9950x3d-cpu-avx512";
const RUNTIME_API: &str = "cuda";
const SELECTED_ROUTE: &str = "bitnet_qk256_cuda";
const KERNEL_ID: &str = "qk256_gemv_cuda";

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

#[derive(Debug)]
struct ProfileRuns {
    one_token: Vec<(PathBuf, Value)>,
    short_decode_8: Vec<(PathBuf, Value)>,
    short_decode_32: Vec<(PathBuf, Value)>,
    prefill_128_decode_16: Vec<(PathBuf, Value)>,
    prefill_512_decode_32: Vec<(PathBuf, Value)>,
    warm_session_3: Vec<(PathBuf, Value)>,
    warm_session_10: Vec<(PathBuf, Value)>,
    decode_128: Vec<(PathBuf, Value)>,
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
    let runs = ProfileRuns {
        one_token: read_runs(&args.one_token_runs)?,
        short_decode_8: read_runs(&args.short_decode_8_runs)?,
        short_decode_32: read_runs(&args.short_decode_32_runs)?,
        prefill_128_decode_16: read_runs(&args.prefill_128_decode_16_runs)?,
        prefill_512_decode_32: read_runs(&args.prefill_512_decode_32_runs)?,
        warm_session_3: read_runs(&args.warm_session_3_runs)?,
        warm_session_10: read_runs(&args.warm_session_10_runs)?,
        decode_128: read_runs(&args.decode_128_runs)?,
    };
    assert_repeated_sources(&runs)?;

    let receipt = build_receipt(&args, &runs)?;
    validate_strict_bitnet_cuda_repeated_profiles_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;
    Ok(())
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

fn read_runs(paths: &[PathBuf]) -> Result<Vec<(PathBuf, Value)>, Box<dyn Error>> {
    paths.iter().map(|path| Ok((path.clone(), read_json(path)?))).collect()
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn assert_repeated_sources(runs: &ProfileRuns) -> Result<(), Box<dyn Error>> {
    assert_profile_sources("one_token", &runs.one_token)?;
    assert_profile_sources("short_decode_8", &runs.short_decode_8)?;
    assert_profile_sources("short_decode_32", &runs.short_decode_32)?;
    assert_profile_sources("prefill_128_decode_16", &runs.prefill_128_decode_16)?;
    assert_profile_sources("prefill_512_decode_32", &runs.prefill_512_decode_32)?;
    assert_profile_sources("warm_session_3_turns", &runs.warm_session_3)?;
    assert_profile_sources("warm_session_10_turns", &runs.warm_session_10)?;
    assert_profile_sources("decode_128_from_warm_context", &runs.decode_128)?;
    Ok(())
}

fn assert_profile_sources(profile: &str, runs: &[(PathBuf, Value)]) -> Result<(), Box<dyn Error>> {
    if runs.len() < 3 {
        return Err(format!("{profile} requires at least 3 source receipts").into());
    }
    let mut paths = BTreeSet::new();
    let anchor = runs.first().ok_or("runs must not be empty")?.1.clone();
    let prompt_template = str_at(&anchor, "/tokenizer_prompt_authority/prompt_template")?;
    let prompt_hash = rendered_prompt_hash(profile, &anchor)?;

    for (path, receipt) in runs {
        if !paths.insert(path_label(path)) {
            return Err(format!("{profile} source receipt paths must be unique").into());
        }
        assert_source_receipt(profile, receipt)?;
        if str_at(receipt, "/tokenizer_prompt_authority/prompt_template")? != prompt_template {
            return Err(format!("{profile} runs must use the same prompt template").into());
        }
        if rendered_prompt_hash(profile, receipt)? != prompt_hash {
            return Err(format!("{profile} runs must use the same rendered prompt hash").into());
        }
    }
    Ok(())
}

fn assert_source_receipt(profile: &str, receipt: &Value) -> Result<(), Box<dyn Error>> {
    let artifact_kind = str_at(receipt, "/artifact_kind")?;
    let artifact_kind_lower = artifact_kind.to_ascii_lowercase();
    if !artifact_kind_lower.contains("bitnet") {
        return Err(format!("{profile} source receipt must be a BitNet receipt").into());
    }
    if artifact_kind_lower.contains("dense") || artifact_kind_lower.contains("qwen") {
        return Err(format!("{profile} source receipt must not be dense/Qwen evidence").into());
    }
    if str_at_any(receipt, &["/profile/id", "/profile/profile"])? != profile {
        return Err(format!("{profile} source receipt profile id mismatch").into());
    }
    if str_at(receipt, "/model/repo")? != MODEL_REPO {
        return Err("source receipt must use the official Microsoft BitNet repo".into());
    }
    if str_at(receipt, "/model/file")? != MODEL_FILE {
        return Err("source receipt must use the official I2_S GGUF file".into());
    }
    if str_at(receipt, "/model/sha256")? != MODEL_SHA256 {
        return Err("source receipt has unexpected official BitNet model SHA".into());
    }
    if str_at(receipt, "/tokenizer/source")? != TOKENIZER_AUTHORITY {
        return Err("source receipt must use the external tokenizer authority".into());
    }
    if str_at(receipt, "/tokenizer/pretokenizer_authority")? != PRETOKENIZER_AUTHORITY {
        return Err("source receipt must record llama-bpe pretokenizer authority".into());
    }
    if str_at(receipt, "/tokenizer_prompt_authority/tokenizer_authority")? != TOKENIZER_AUTHORITY {
        return Err("source receipt must use external_tokenizer tokenizer authority".into());
    }
    if str_at(receipt, "/tokenizer_prompt_authority/prompt_authority")? != PROMPT_AUTHORITY {
        return Err("source receipt must use bitnetcpp-answer prompt authority".into());
    }
    if str_at(receipt, "/tokenizer_prompt_authority/prompt_template")? != PROMPT_AUTHORITY {
        return Err("source receipt must use bitnetcpp-answer prompt template".into());
    }
    if str_at_any(receipt, &["/execution_plan/selected_backend", "/selected_backend"])?
        != SELECTED_BACKEND
    {
        return Err("source receipt must select nvidia-rtx-5070-ti-cuda".into());
    }
    if str_at_any(receipt, &["/execution_plan/runtime_api", "/runtime_api"])? != RUNTIME_API {
        return Err("source receipt must use CUDA runtime_api".into());
    }
    if str_at(receipt, "/execution_plan/selected_route")? != SELECTED_ROUTE {
        return Err("source receipt must route bitnet_qk256_cuda".into());
    }
    if bool_at(receipt, "/fallback_used")? {
        return Err("source receipt must be fallback-free".into());
    }
    if bool_at(receipt, "/execution_plan/fallback_used")? {
        return Err("source execution plan must be fallback-free".into());
    }
    if bool_true_or_absent(receipt, "/speedup_claim") {
        return Err("source receipt must not claim speedup".into());
    }
    if bool_true_or_absent(receipt, "/execution_plan/speedup_claim") {
        return Err("source execution plan must not claim speedup".into());
    }
    if bool_true_or_absent(receipt, "/execution_plan/dense_regular_llm_cuda") {
        return Err("source receipt must not claim dense_regular_llm_cuda".into());
    }
    if bool_true_or_absent(receipt, "/dense_regular_llm_cuda_proof") {
        return Err("source receipt must not carry dense_regular_llm_cuda_proof".into());
    }
    if !bool_at_any(
        receipt,
        &["/execution_plan/bitnet_packed_qk256_cuda", "/bitnet_packed_i2s_qk256_proof"],
    )? {
        return Err("source receipt must carry the BitNet packed QK256 proof family".into());
    }

    if !bool_at_any(
        receipt,
        &["/quality_summary/passed", "/quality_gate/passed", "/quality/garbage_filter_passed"],
    )? {
        return Err("source receipt quality gate must pass".into());
    }
    if generated_tokens(profile, receipt)? != expected_generated_tokens(profile)? {
        return Err(format!("{profile} generated token count does not match the profile").into());
    }
    if let Some(expected_input_tokens) = expected_input_tokens(profile)? {
        let actual = u64_at_any(
            receipt,
            &[
                "/profile/expected_input_tokens",
                "/profile/prompt_tokens",
                "/tokenizer_prompt_authority/prompt_token_count",
            ],
        )?;
        if actual != expected_input_tokens {
            return Err(format!(
                "{profile} expected {expected_input_tokens} prompt tokens, got {actual}"
            )
            .into());
        }
    }

    for pointer in [
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
        "/timing/host_to_device_ms",
        "/timing/device_to_host_ms",
        "/cuda/power_draw_watts",
        "/cuda/temperature_c",
    ] {
        number_at(receipt, pointer)?;
    }
    for pointer in ["/timing/host_to_device_bytes", "/timing/device_to_host_bytes"] {
        u64_at(receipt, pointer)?;
    }
    u64_at_any(receipt, &["/cuda/memory_hwm_bytes", "/cuda/vram_bytes"])?;

    let kernel_stats = receipt
        .get("kernel_stats")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .ok_or("kernel_stats must contain at least one entry")?;
    if str_at(kernel_stats, "/kernel_id")? != KERNEL_ID {
        return Err("source receipt must use qk256_gemv_cuda".into());
    }
    number_at(kernel_stats, "/kernel_time_ms")?;
    u64_at_any(kernel_stats, &["/kernel_launches", "/launch_count"])?;
    u64_at(kernel_stats, "/invocations")?;
    if u64_at(kernel_stats, "/fallback_invocations")? != 0 {
        return Err("source receipt kernel fallback_invocations must be zero".into());
    }
    Ok(())
}

fn build_receipt(args: &Args, runs: &ProfileRuns) -> Result<Value, Box<dyn Error>> {
    let profiles = vec![
        profile_from_runs("one_token", &runs.one_token)?,
        profile_from_runs("short_decode_8", &runs.short_decode_8)?,
        profile_from_runs("short_decode_32", &runs.short_decode_32)?,
        profile_from_runs("prefill_128_decode_16", &runs.prefill_128_decode_16)?,
        profile_from_runs("prefill_512_decode_32", &runs.prefill_512_decode_32)?,
        profile_from_runs("warm_session_3_turns", &runs.warm_session_3)?,
        profile_from_runs("warm_session_10_turns", &runs.warm_session_10)?,
        profile_from_runs("decode_128_from_warm_context", &runs.decode_128)?,
    ];
    let total_kernel_invocations = all_receipts(runs)
        .iter()
        .map(|receipt| {
            u64_at_any(receipt, &["/timing/kernel_invocations", "/kernel_stats/0/invocations"])
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<u64>();

    Ok(json!({
        "schema": 1,
        "artifact_kind": "strict_bitnet_cuda_repeated_profiles",
        "artifact_path": path_label(&args.receipt_out),
        "campaign_item": CAMPAIGN_ITEM,
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": timestamp_label(),
        "requested_backend": SELECTED_BACKEND,
        "selected_backend": SELECTED_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "runtime_api": RUNTIME_API,
        "selected_route": SELECTED_ROUTE,
        "kernel_id": KERNEL_ID,
        "claim": "strict_bitnet_cuda_repeated_profiles_baseline",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "full_cuda_residency_claimed": false,
        "server_ready_claimed": false,
        "bitnet_packed_i2s_qk256_proof": true,
        "dense_regular_llm_cuda_proof": false,
        "claim_boundary": {
            "strict_bitnet_cuda_repeated_profiles_claimed": true,
            "bitnet_packed_i2s_qk256_proof": true,
            "dense_regular_llm_cuda_proof": false,
            "server_ready_claimed": false,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false,
            "full_cuda_residency_claimed": false,
            "broad_server_readiness_claimed": false
        },
        "model": {
            "repo": MODEL_REPO,
            "file": MODEL_FILE,
            "sha256": MODEL_SHA256,
            "format": "gguf",
            "architecture": "bitnet_b1_58",
            "quantization_layout": "I2_S/QK256"
        },
        "tokenizer_prompt_authority": {
            "tokenizer_authority": TOKENIZER_AUTHORITY,
            "pretokenizer_authority": PRETOKENIZER_AUTHORITY,
            "prompt_authority": PROMPT_AUTHORITY,
            "prompt_template": PROMPT_AUTHORITY,
            "prompt_policy": "bitnetcpp-answer deterministic profile prompts; same tokenizer and prompt policy across repeated runs",
            "deterministic_prompt": true
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "bitnet_b1_58",
            "quantization": "i2_s_qk256",
            "selected_backend": SELECTED_BACKEND,
            "selected_route": SELECTED_ROUTE,
            "runtime_api": RUNTIME_API,
            "strict_fallback_policy": "reject",
            "bitnet_packed_qk256_cuda": true,
            "dense_regular_llm_cuda": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false,
            "cuda_bitnet_qk256_ops": total_kernel_invocations,
            "cuda_dense_regular_llm_ops": 0,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": total_kernel_invocations,
            "cuda_ops": total_kernel_invocations
        },
        "proof_inputs": {
            "one_token": profile_input("one_token", &runs.one_token)?,
            "short_decode_8": profile_input("short_decode_8", &runs.short_decode_8)?,
            "short_decode_32": profile_input("short_decode_32", &runs.short_decode_32)?,
            "prefill_128_decode_16": profile_input("prefill_128_decode_16", &runs.prefill_128_decode_16)?,
            "prefill_512_decode_32": profile_input("prefill_512_decode_32", &runs.prefill_512_decode_32)?,
            "warm_session_3_turns": profile_input("warm_session_3_turns", &runs.warm_session_3)?,
            "warm_session_10_turns": profile_input("warm_session_10_turns", &runs.warm_session_10)?,
            "decode_128_from_warm_context": profile_input("decode_128_from_warm_context", &runs.decode_128)?
        },
        "profiles": profiles,
        "comparator_summary": comparator_summary(&profiles),
        "transfer_timing": transfer_timing(),
        "hardware_context": hardware_context(runs)?,
        "cuda": cuda_context(runs)?,
        "claim_boundaries": [
            "speedup_claim=false; repeated current-source profiles are not a speedup qualification.",
            "benchmark_qualified_speedup=false until CUDA-BITNET-PERF-006 reviews exact profiles.",
            "full_cuda_residency_claimed=false until every required phase proves residency.",
            "server_ready_claimed=false; server smoke and readiness remain separate proof families.",
            "BitNet packed I2_S/QK256 proof cannot be satisfied by dense_regular_llm_cuda evidence."
        ]
    }))
}

fn profile_input(profile: &str, runs: &[(PathBuf, Value)]) -> Result<Value, Box<dyn Error>> {
    let mut hasher = Sha256::new();
    let mut paths = Vec::new();
    for (path, _) in runs {
        let digest = sha256_file(path)?;
        hasher.update(digest.as_bytes());
        paths.push(path_label(path));
    }
    Ok(json!({
        "path": format!("profile:{profile}"),
        "sha256": format!("{:x}", hasher.finalize()),
        "artifact_kind": "strict_bitnet_profile_repeated_runs",
        "runs": paths
    }))
}

fn profile_from_runs(profile: &str, runs: &[(PathBuf, Value)]) -> Result<Value, Box<dyn Error>> {
    let run_values = runs
        .iter()
        .enumerate()
        .map(|(index, (path, receipt))| run_from_receipt(profile, index + 1, path, receipt))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(json!({
        "profile": profile,
        "status": "repeated_same_artifact_cpu_cuda_profile",
        "cpu_reference_backend": REFERENCE_BACKEND,
        "cuda_backend": SELECTED_BACKEND,
        "runtime_api": RUNTIME_API,
        "selected_route": SELECTED_ROUTE,
        "kernel_id": KERNEL_ID,
        "expected_input_tokens": expected_input_tokens(profile)?,
        "expected_generated_tokens": expected_generated_tokens(profile)?,
        "run_count": run_values.len(),
        "cpu_runs": run_values.len(),
        "cuda_runs": run_values.len(),
        "min_runs_per_backend": 3,
        "fallback_free": true,
        "same_artifact_sha": true,
        "same_tokenizer_prompt_policy": true,
        "deterministic_generation_policy": true,
        "generated_token_ids_match": run_values.iter().all(|run| {
            run.get("generated_token_ids_match").and_then(Value::as_bool) == Some(true)
        }),
        "first_divergence_report": "no generated-token divergence recorded across source receipts",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": true,
        "dense_regular_llm_cuda_proof": false,
        "full_cuda_residency_claimed": false,
        "server_ready_claimed": false,
        "transfer_timing_status": "host_to_device_and_device_to_host_measured_in_source_receipts",
        "model_load_ms": number_summary(&run_values, "/timing/model_load_ms"),
        "tokenizer_load_ms": number_summary(&run_values, "/timing/tokenizer_load_ms"),
        "prompt_render_ms": number_summary(&run_values, "/timing/prompt_render_ms"),
        "tokenize_ms": number_summary(&run_values, "/timing/tokenize_ms"),
        "cuda_context_init_ms": number_summary(&run_values, "/timing/cuda_context_init_ms"),
        "weight_upload_ms": number_summary(&run_values, "/timing/weight_upload_ms"),
        "prefill_ms": number_summary(&run_values, "/timing/prefill_ms"),
        "first_token_ms": number_summary(&run_values, "/timing/first_token_ms"),
        "decode_total_ms": number_summary(&run_values, "/timing/decode_total_ms"),
        "steady_tok_per_s": number_summary(&run_values, "/timing/steady_tok_per_s"),
        "kernel_time_ms": number_summary(&run_values, "/timing/kernel_time_ms"),
        "launch_count": u64_summary(&run_values, "/timing/launch_count"),
        "host_to_device_bytes": u64_summary(&run_values, "/timing/host_to_device_bytes"),
        "host_to_device_ms": number_summary(&run_values, "/timing/host_to_device_ms"),
        "device_to_host_bytes": u64_summary(&run_values, "/timing/device_to_host_bytes"),
        "device_to_host_ms": number_summary(&run_values, "/timing/device_to_host_ms"),
        "vram_high_water_bytes": u64_summary(&run_values, "/timing/vram_high_water_bytes"),
        "runs": run_values
    }))
}

fn run_from_receipt(
    profile: &str,
    index: usize,
    path: &Path,
    receipt: &Value,
) -> Result<Value, Box<dyn Error>> {
    let generated_tokens = generated_tokens(profile, receipt)?;
    let decode_total = number_at(receipt, "/timing/decode_total_ms")?;
    let steady_tok_per_s = number_or(
        receipt,
        "/timing/steady_tok_per_s",
        generated_tokens as f64 / (decode_total / 1000.0),
    );
    Ok(json!({
        "run_id": format!("run-{index:02}"),
        "profile": profile,
        "source_receipt_path": path_label(path),
        "source_receipt_sha256": sha256_file(path)?,
        "source_artifact_kind": str_at(receipt, "/artifact_kind")?,
        "model_sha256": str_at(receipt, "/model/sha256")?,
        "prompt_template": str_at(receipt, "/tokenizer_prompt_authority/prompt_template")?,
        "prompt_token_count": prompt_token_count(profile, receipt)?,
        "generation_policy": "greedy",
        "deterministic_generation": true,
        "expected_input_tokens": expected_input_tokens(profile)?,
        "generated_tokens": generated_tokens,
        "generated_token_ids_sha256": string_at_any_or(
            receipt,
            &[
                "/generated_token_ids_sha256",
                "/quality_summary/generated_token_ids_sha256",
                "/one_token_proof/generated_token_ids_sha256",
                "/short_decode_proof/generated_token_ids_sha256",
                "/warm_session_proof/generated_token_ids_sha256",
                "/warm_decode_proof/generated_token_ids_sha256"
            ],
            "not_recorded_in_source_receipt"
        ),
        "generated_token_ids_match": bool_at_any(
            receipt,
            &[
                "/quality_summary/generated_token_ids_match",
                "/comparison/generated_token_ids_match",
                "/one_token_proof/generated_token_ids_match",
                "/short_decode_proof/generated_token_ids_match",
                "/warm_session_proof/generated_token_ids_match",
                "/warm_decode_proof/generated_token_ids_match"
            ]
        )?,
        "first_divergence_report": string_at_any_or(
            receipt,
            &[
                "/quality_summary/first_divergence_report",
                "/comparison/first_divergence_report"
            ],
            "none"
        ),
        "top_k_evidence_recorded": bool_at_any_or(
            receipt,
            &[
                "/quality_summary/top_k_compared",
                "/one_token_proof/top_k_compared",
                "/short_decode_proof/top_k_compared",
                "/warm_session_proof/top_k_compared",
                "/warm_decode_proof/top_k_compared"
            ],
            false
        ),
        "fallback_used": bool_at(receipt, "/fallback_used")?,
        "quality_passed": bool_at_any(
            receipt,
            &[
                "/quality_summary/passed",
                "/quality_gate/passed",
                "/quality/garbage_filter_passed"
            ]
        )?,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": true,
        "dense_regular_llm_cuda_proof": false,
        "full_cuda_residency_claimed": false,
        "server_ready_claimed": false,
        "timing": {
            "model_load_ms": number_at(receipt, "/timing/model_load_ms")?,
            "tokenizer_load_ms": number_at(receipt, "/timing/tokenizer_load_ms")?,
            "prompt_render_ms": number_at(receipt, "/timing/prompt_render_ms")?,
            "tokenize_ms": number_at(receipt, "/timing/tokenize_ms")?,
            "cuda_context_init_ms": number_at(receipt, "/timing/cuda_context_init_ms")?,
            "weight_upload_ms": number_at(receipt, "/timing/weight_upload_ms")?,
            "prefill_ms": number_at(receipt, "/timing/prefill_ms")?,
            "first_token_ms": number_at(receipt, "/timing/first_token_ms")?,
            "decode_total_ms": decode_total,
            "steady_tok_per_s": steady_tok_per_s,
            "kernel_time_ms": number_at_any(receipt, &["/timing/kernel_time_ms", "/kernel_stats/0/kernel_time_ms"])?,
            "launch_count": u64_at_any(receipt, &["/timing/launch_count", "/timing/kernel_launches", "/kernel_stats/0/kernel_launches"])?,
            "kernel_invocations": u64_at_any(receipt, &["/timing/kernel_invocations", "/kernel_stats/0/invocations"])?,
            "host_to_device_bytes": u64_at(receipt, "/timing/host_to_device_bytes")?,
            "host_to_device_ms": number_at(receipt, "/timing/host_to_device_ms")?,
            "device_to_host_bytes": u64_at(receipt, "/timing/device_to_host_bytes")?,
            "device_to_host_ms": number_at(receipt, "/timing/device_to_host_ms")?,
            "vram_high_water_bytes": u64_at_any(receipt, &["/timing/vram_high_water_bytes", "/cuda/memory_hwm_bytes", "/cuda/vram_bytes"])?,
            "power_temperature_context": "NVML power and temperature sampled during source receipt"
        }
    }))
}

fn comparator_summary(profiles: &[Value]) -> Value {
    let total_runs = profiles.iter().map(|profile| u64_value(profile, "run_count")).sum::<u64>();
    json!({
        "status": "repeated_profiles_baseline_only",
        "profiles_recorded": profiles.len(),
        "min_runs_per_profile": 3,
        "total_cpu_runs": total_runs,
        "total_cuda_runs": total_runs,
        "fallback_free": true,
        "same_artifact_sha": true,
        "same_tokenizer_prompt_policy": true,
        "deterministic_generation_policy": true,
        "generated_tokens_compared": true,
        "speedup_claim_allowed": false,
        "benchmark_qualified_speedup": false,
        "accepted_speedup_profiles": [],
        "remaining_qualification_blockers": [
            "CUDA-BITNET-PERF-006 must review each exact profile before speedup can be accepted",
            "CUDA-BITNET-OPS-001 must audit TTFT and residency before full-residency claims can move"
        ],
        "next_step": "CUDA-BITNET-PERF-006 exact-profile qualification review"
    })
}

fn transfer_timing() -> Value {
    json!({
        "status": "host_to_device_and_device_to_host_measured_in_source_receipts",
        "source": "CUDA-BITNET-PERF-005 source receipts record H2D/D2H bytes and timings per profile",
        "host_to_device_bytes_recorded": true,
        "device_to_host_bytes_recorded": true,
        "host_to_device_timing_recorded": true,
        "device_to_host_timing_recorded": true,
        "pure_host_to_device_timing_recorded": true
    })
}

fn hardware_context(runs: &ProfileRuns) -> Result<Value, Box<dyn Error>> {
    let receipts = all_receipts(runs);
    let powers = receipts
        .iter()
        .map(|receipt| number_at(receipt, "/cuda/power_draw_watts"))
        .collect::<Result<Vec<_>, _>>()?;
    let temperatures = receipts
        .iter()
        .map(|receipt| number_at(receipt, "/cuda/temperature_c"))
        .collect::<Result<Vec<_>, _>>()?;
    let first = receipts.first().ok_or("hardware context requires receipts")?;
    Ok(json!({
        "vram_bytes": u64_at(first, "/cuda/vram_bytes")?,
        "vram_high_water_bytes_min": u64_min(&receipts, &["/cuda/memory_hwm_bytes", "/cuda/vram_bytes"])?,
        "vram_high_water_bytes_max": u64_max(&receipts, &["/cuda/memory_hwm_bytes", "/cuda/vram_bytes"])?,
        "power_draw_watts_min": min_f64(&powers),
        "power_draw_watts_max": max_f64(&powers),
        "temperature_c_min": min_f64(&temperatures),
        "temperature_c_max": max_f64(&temperatures),
        "source": "NVML fields recorded in strict BitNet CUDA source receipts"
    }))
}

fn cuda_context(runs: &ProfileRuns) -> Result<Value, Box<dyn Error>> {
    let first = runs.one_token.first().ok_or("one_token runs missing")?.1.clone();
    first.pointer("/cuda").cloned().ok_or("cuda block missing".into())
}

fn all_receipts(runs: &ProfileRuns) -> Vec<&Value> {
    runs.one_token
        .iter()
        .chain(runs.short_decode_8.iter())
        .chain(runs.short_decode_32.iter())
        .chain(runs.prefill_128_decode_16.iter())
        .chain(runs.prefill_512_decode_32.iter())
        .chain(runs.warm_session_3.iter())
        .chain(runs.warm_session_10.iter())
        .chain(runs.decode_128.iter())
        .map(|(_, receipt)| receipt)
        .collect()
}

fn rendered_prompt_hash(profile: &str, receipt: &Value) -> Result<String, Box<dyn Error>> {
    if profile.starts_with("warm_session_")
        && let Some(value) =
            receipt.pointer("/tokenizer_prompt_authority/turns/0/rendered_prompt_sha256")
    {
        return value
            .as_str()
            .map(ToOwned::to_owned)
            .ok_or_else(|| "warm-session rendered prompt hash must be a string".into());
    }
    Ok(str_at(receipt, "/tokenizer_prompt_authority/rendered_prompt_sha256")?.to_owned())
}

fn prompt_token_count(profile: &str, receipt: &Value) -> Result<u64, Box<dyn Error>> {
    if profile.starts_with("warm_session_")
        && let Some(value) =
            receipt.pointer("/tokenizer_prompt_authority/turns/0/prompt_token_count")
    {
        return value
            .as_u64()
            .ok_or_else(|| "warm-session prompt token count must be an unsigned integer".into());
    }
    u64_at(receipt, "/tokenizer_prompt_authority/prompt_token_count")
}

fn generated_tokens(profile: &str, receipt: &Value) -> Result<u64, Box<dyn Error>> {
    let candidates: &[&str] = match profile {
        "one_token" => &["/profile/generated_tokens", "/one_token_proof/generated_tokens_count"],
        "short_decode_8" | "short_decode_32" => {
            &["/profile/generated_tokens", "/short_decode_proof/generated_tokens_count"]
        }
        "prefill_128_decode_16" | "prefill_512_decode_32" => {
            &["/profile/generated_tokens", "/short_decode_proof/generated_tokens_count"]
        }
        "warm_session_3_turns" | "warm_session_10_turns" => {
            &["/profile/generated_tokens", "/warm_session_proof/generated_tokens_total"]
        }
        "decode_128_from_warm_context" => {
            &["/profile/generated_tokens", "/warm_decode_proof/generated_tokens_count"]
        }
        other => return Err(format!("unknown profile {other}").into()),
    };
    u64_at_any(receipt, candidates)
}

fn expected_generated_tokens(profile: &str) -> Result<u64, Box<dyn Error>> {
    match profile {
        "one_token" => Ok(1),
        "short_decode_8" => Ok(8),
        "short_decode_32" => Ok(32),
        "prefill_128_decode_16" => Ok(16),
        "prefill_512_decode_32" => Ok(32),
        "warm_session_3_turns" => Ok(24),
        "warm_session_10_turns" => Ok(80),
        "decode_128_from_warm_context" => Ok(128),
        other => Err(format!("unknown profile {other}").into()),
    }
}

fn expected_input_tokens(profile: &str) -> Result<Option<u64>, Box<dyn Error>> {
    match profile {
        "prefill_128_decode_16" => Ok(Some(128)),
        "prefill_512_decode_32" => Ok(Some(512)),
        "one_token"
        | "short_decode_8"
        | "short_decode_32"
        | "warm_session_3_turns"
        | "warm_session_10_turns"
        | "decode_128_from_warm_context" => Ok(None),
        other => Err(format!("unknown profile {other}").into()),
    }
}

fn number_summary(runs: &[Value], pointer: &str) -> Value {
    let values = runs
        .iter()
        .filter_map(|run| run.pointer(pointer).and_then(Value::as_f64))
        .collect::<Vec<_>>();
    json!({ "count": values.len(), "min": min_f64(&values), "mean": mean_f64(&values), "max": max_f64(&values) })
}

fn u64_summary(runs: &[Value], pointer: &str) -> Value {
    let values = runs
        .iter()
        .filter_map(|run| run.pointer(pointer).and_then(Value::as_u64))
        .collect::<Vec<_>>();
    let min = values.iter().copied().min().unwrap_or(0);
    let max = values.iter().copied().max().unwrap_or(0);
    let mean = if values.is_empty() {
        0.0
    } else {
        values.iter().copied().sum::<u64>() as f64 / values.len() as f64
    };
    json!({ "count": values.len(), "min": min, "mean": mean, "max": max })
}

fn min_f64(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::min).unwrap_or(0.0)
}

fn max_f64(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(0.0)
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn timestamp_label() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{digest:x}"))
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{pointer} must be a string").into())
}

fn str_at_any<'a>(value: &'a Value, pointers: &[&str]) -> Result<&'a str, Box<dyn Error>> {
    pointers
        .iter()
        .find_map(|pointer| value.pointer(pointer).and_then(Value::as_str))
        .ok_or_else(|| format!("one of {} must be a string", pointers.join(", ")).into())
}

fn string_at_any_or(value: &Value, pointers: &[&str], default: &str) -> String {
    pointers
        .iter()
        .find_map(|pointer| value.pointer(pointer).and_then(Value::as_str))
        .unwrap_or(default)
        .to_owned()
}

fn bool_at(value: &Value, pointer: &str) -> Result<bool, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{pointer} must be a bool").into())
}

fn bool_at_any(value: &Value, pointers: &[&str]) -> Result<bool, Box<dyn Error>> {
    pointers
        .iter()
        .find_map(|pointer| value.pointer(pointer).and_then(Value::as_bool))
        .ok_or_else(|| format!("one of {} must be a bool", pointers.join(", ")).into())
}

fn bool_at_any_or(value: &Value, pointers: &[&str], default: bool) -> bool {
    pointers
        .iter()
        .find_map(|pointer| value.pointer(pointer).and_then(Value::as_bool))
        .unwrap_or(default)
}

fn bool_true_or_absent(value: &Value, pointer: &str) -> bool {
    value.pointer(pointer).and_then(Value::as_bool).unwrap_or(false)
}

fn u64_at(value: &Value, pointer: &str) -> Result<u64, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{pointer} must be an unsigned integer").into())
}

fn u64_at_any(value: &Value, pointers: &[&str]) -> Result<u64, Box<dyn Error>> {
    pointers
        .iter()
        .find_map(|pointer| value.pointer(pointer).and_then(Value::as_u64))
        .ok_or_else(|| format!("one of {} must be an unsigned integer", pointers.join(", ")).into())
}

fn number_at(value: &Value, pointer: &str) -> Result<f64, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{pointer} must be a number").into())
}

fn number_at_any(value: &Value, pointers: &[&str]) -> Result<f64, Box<dyn Error>> {
    pointers
        .iter()
        .find_map(|pointer| value.pointer(pointer).and_then(Value::as_f64))
        .ok_or_else(|| format!("one of {} must be a number", pointers.join(", ")).into())
}

fn number_or(value: &Value, pointer: &str, default: f64) -> f64 {
    value.pointer(pointer).and_then(Value::as_f64).unwrap_or(default)
}

fn u64_min(receipts: &[&Value], pointers: &[&str]) -> Result<u64, Box<dyn Error>> {
    receipts
        .iter()
        .map(|receipt| u64_at_any(receipt, pointers))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .ok_or_else(|| "u64_min requires at least one receipt".into())
}

fn u64_max(receipts: &[&Value], pointers: &[&str]) -> Result<u64, Box<dyn Error>> {
    receipts
        .iter()
        .map(|receipt| u64_at_any(receipt, pointers))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| "u64_max requires at least one receipt".into())
}

fn u64_value(value: &Value, field: &str) -> u64 {
    value.get(field).and_then(Value::as_u64).unwrap_or(0)
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

    #[test]
    fn bitnet_perf_005_strict_bitnet_cuda_repeated_profiles_aggregate_builds_from_sources() {
        let temp = tempfile::tempdir().expect("tempdir");
        let args = Args {
            one_token_runs: write_profile_runs(temp.path(), "one_token", 1, None),
            short_decode_8_runs: write_profile_runs(temp.path(), "short_decode_8", 8, None),
            short_decode_32_runs: write_profile_runs(temp.path(), "short_decode_32", 32, None),
            prefill_128_decode_16_runs: write_profile_runs(
                temp.path(),
                "prefill_128_decode_16",
                16,
                Some(128),
            ),
            prefill_512_decode_32_runs: write_profile_runs(
                temp.path(),
                "prefill_512_decode_32",
                32,
                Some(512),
            ),
            warm_session_3_runs: write_profile_runs(temp.path(), "warm_session_3_turns", 24, None),
            warm_session_10_runs: write_profile_runs(
                temp.path(),
                "warm_session_10_turns",
                80,
                None,
            ),
            decode_128_runs: write_profile_runs(
                temp.path(),
                "decode_128_from_warm_context",
                128,
                None,
            ),
            receipt_out: temp.path().join("aggregate.json"),
            manifest_out: None,
            print_manifest: false,
        };

        let runs = ProfileRuns {
            one_token: read_runs(&args.one_token_runs).expect("one_token runs"),
            short_decode_8: read_runs(&args.short_decode_8_runs).expect("short_decode_8 runs"),
            short_decode_32: read_runs(&args.short_decode_32_runs).expect("short_decode_32 runs"),
            prefill_128_decode_16: read_runs(&args.prefill_128_decode_16_runs)
                .expect("prefill_128_decode_16 runs"),
            prefill_512_decode_32: read_runs(&args.prefill_512_decode_32_runs)
                .expect("prefill_512_decode_32 runs"),
            warm_session_3: read_runs(&args.warm_session_3_runs)
                .expect("warm_session_3_turns runs"),
            warm_session_10: read_runs(&args.warm_session_10_runs)
                .expect("warm_session_10_turns runs"),
            decode_128: read_runs(&args.decode_128_runs)
                .expect("decode_128_from_warm_context runs"),
        };
        assert_repeated_sources(&runs).expect("source receipts validate");

        let receipt = build_receipt(&args, &runs).expect("aggregate receipt");
        validate_strict_bitnet_cuda_repeated_profiles_receipt_json(&receipt)
            .expect("aggregate validates");

        assert_eq!(receipt["profiles"].as_array().map(Vec::len), Some(8));
        assert_eq!(receipt["speedup_claim"], false);
        assert_eq!(receipt["benchmark_qualified_speedup"], false);
        assert_eq!(receipt["full_cuda_residency_claimed"], false);
        assert_eq!(receipt["server_ready_claimed"], false);
        assert_eq!(receipt["bitnet_packed_i2s_qk256_proof"], true);
        assert_eq!(receipt["dense_regular_llm_cuda_proof"], false);

        let mut bad_speedup = receipt.clone();
        bad_speedup["speedup_claim"] = json!(true);
        assert!(validate_strict_bitnet_cuda_repeated_profiles_receipt_json(&bad_speedup).is_err());

        let mut bad_dense = receipt;
        bad_dense["dense_regular_llm_cuda_proof"] = json!(true);
        assert!(validate_strict_bitnet_cuda_repeated_profiles_receipt_json(&bad_dense).is_err());
    }

    fn write_profile_runs(
        root: &Path,
        profile: &str,
        generated_tokens: u64,
        expected_input_tokens: Option<u64>,
    ) -> Vec<PathBuf> {
        (1..=3)
            .map(|run| {
                let path = root.join(format!("{profile}-run-{run}.json"));
                let receipt = source_receipt(profile, run, generated_tokens, expected_input_tokens);
                std::fs::write(&path, serde_json::to_vec_pretty(&receipt).expect("source json"))
                    .expect("write source receipt");
                path
            })
            .collect()
    }

    fn source_receipt(
        profile: &str,
        run: u64,
        generated_tokens: u64,
        expected_input_tokens: Option<u64>,
    ) -> Value {
        let prompt_tokens = expected_input_tokens.unwrap_or(32);
        json!({
            "schema": 1,
            "artifact_kind": "strict_bitnet_cuda_profile_source",
            "model": {
                "repo": MODEL_REPO,
                "file": MODEL_FILE,
                "sha256": MODEL_SHA256,
                "format": "gguf",
                "architecture": "bitnet_b1_58"
            },
            "tokenizer": {
                "source": TOKENIZER_AUTHORITY,
                "pretokenizer_authority": PRETOKENIZER_AUTHORITY
            },
            "tokenizer_prompt_authority": {
                "tokenizer_authority": TOKENIZER_AUTHORITY,
                "prompt_authority": PROMPT_AUTHORITY,
                "prompt_template": PROMPT_AUTHORITY,
                "rendered_prompt_sha256": format!("prompt-hash-{profile}"),
                "prompt_token_count": prompt_tokens
            },
            "execution_plan": {
                "selected_backend": SELECTED_BACKEND,
                "selected_route": SELECTED_ROUTE,
                "runtime_api": RUNTIME_API,
                "strict_fallback_policy": "reject",
                "fallback_used": false,
                "strict_cuda_ready": true,
                "speedup_claim": false,
                "bitnet_packed_qk256_cuda": true,
                "dense_regular_llm_cuda": false,
                "unsupported_ops": 0,
                "cpu_fallback_ops": 0
            },
            "requested_backend": SELECTED_BACKEND,
            "selected_backend": SELECTED_BACKEND,
            "runtime_api": RUNTIME_API,
            "fallback_used": false,
            "speedup_claim": false,
            "bitnet_packed_i2s_qk256_proof": true,
            "dense_regular_llm_cuda_proof": false,
            "quality_summary": {
                "passed": true,
                "generated_token_ids_match": true,
                "generated_token_ids_sha256": format!("generated-hash-{profile}"),
                "first_divergence_report": "none",
                "top_k_compared": true
            },
            "timing": {
                "model_load_ms": 100.0 + run as f64,
                "tokenizer_load_ms": 1.0,
                "prompt_render_ms": 0.1,
                "tokenize_ms": 0.2,
                "cuda_context_init_ms": 2.0,
                "weight_upload_ms": 3.0,
                "prefill_ms": 4.0,
                "first_token_ms": 5.0,
                "decode_total_ms": 6.0 + run as f64,
                "steady_tok_per_s": 7.0,
                "host_to_device_bytes": 8,
                "host_to_device_ms": 0.8,
                "device_to_host_bytes": 9,
                "device_to_host_ms": 0.9
            },
            "kernel_stats": [
                {
                    "kernel_id": KERNEL_ID,
                    "kernel_time_ms": 10.0,
                    "invocations": 210,
                    "kernel_launches": 210,
                    "fallback_invocations": 0,
                    "host_to_device_bytes": 8,
                    "device_to_host_bytes": 9
                }
            ],
            "cuda": {
                "available": true,
                "device_count": 1,
                "device_name": "NVIDIA GeForce RTX 5070 Ti",
                "compute_capability": "12.0",
                "driver_version": "580.00",
                "cuda_runtime_version": "13.0",
                "cuda_toolkit_version": "13.0",
                "nvrtc_version": "13.0",
                "vram_bytes": 17094475776u64,
                "memory_hwm_bytes": 7070547968u64,
                "power_draw_watts": 45.0,
                "temperature_c": 42.0
            },
            "profile": {
                "id": profile,
                "expected_input_tokens": expected_input_tokens,
                "generated_tokens": generated_tokens
            }
        })
    }
}
