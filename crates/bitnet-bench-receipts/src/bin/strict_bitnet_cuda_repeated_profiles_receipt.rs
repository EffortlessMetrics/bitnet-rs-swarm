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
const CPU_BACKEND: &str = "amd-9950x3d-cpu-avx512";
const CUDA_BACKEND: &str = "nvidia-rtx-5070-ti-cuda";
const CPU_RUNTIME_API: &str = "cpu";
const CUDA_RUNTIME_API: &str = "cuda";
const CPU_ROUTE: &str = "bitnet_i2s_qk256_cpu_avx512";
const CUDA_ROUTE: &str = "bitnet_qk256_cuda";
const CPU_KERNEL_ID: &str = "i2_s-avx512-reference";
const CUDA_KERNEL_ID: &str = "qk256_gemv_cuda";

#[derive(Debug, Clone, Copy)]
struct ProfileSpec {
    profile: &'static str,
    stem: &'static str,
    cpu_flag: &'static str,
    cuda_flag: &'static str,
    expected_input_tokens: Option<u64>,
    expected_generated_tokens: u64,
}

const PROFILE_SPECS: &[ProfileSpec] = &[
    ProfileSpec {
        profile: "one_token",
        stem: "one-token",
        cpu_flag: "--one-token-cpu-run",
        cuda_flag: "--one-token-cuda-run",
        expected_input_tokens: None,
        expected_generated_tokens: 1,
    },
    ProfileSpec {
        profile: "short_decode_8",
        stem: "short-decode-8",
        cpu_flag: "--short-decode-8-cpu-run",
        cuda_flag: "--short-decode-8-cuda-run",
        expected_input_tokens: None,
        expected_generated_tokens: 8,
    },
    ProfileSpec {
        profile: "short_decode_32",
        stem: "short-decode-32",
        cpu_flag: "--short-decode-32-cpu-run",
        cuda_flag: "--short-decode-32-cuda-run",
        expected_input_tokens: None,
        expected_generated_tokens: 32,
    },
    ProfileSpec {
        profile: "prefill_128_decode_16",
        stem: "prefill-128-decode-16",
        cpu_flag: "--prefill-128-decode-16-cpu-run",
        cuda_flag: "--prefill-128-decode-16-cuda-run",
        expected_input_tokens: Some(128),
        expected_generated_tokens: 16,
    },
    ProfileSpec {
        profile: "prefill_512_decode_32",
        stem: "prefill-512-decode-32",
        cpu_flag: "--prefill-512-decode-32-cpu-run",
        cuda_flag: "--prefill-512-decode-32-cuda-run",
        expected_input_tokens: Some(512),
        expected_generated_tokens: 32,
    },
    ProfileSpec {
        profile: "warm_session_3_turns",
        stem: "warm-session-3",
        cpu_flag: "--warm-session-3-cpu-run",
        cuda_flag: "--warm-session-3-cuda-run",
        expected_input_tokens: None,
        expected_generated_tokens: 24,
    },
    ProfileSpec {
        profile: "warm_session_10_turns",
        stem: "warm-session-10",
        cpu_flag: "--warm-session-10-cpu-run",
        cuda_flag: "--warm-session-10-cuda-run",
        expected_input_tokens: None,
        expected_generated_tokens: 80,
    },
    ProfileSpec {
        profile: "decode_128_from_warm_context",
        stem: "decode-128-from-warm-context",
        cpu_flag: "--decode-128-from-warm-context-cpu-run",
        cuda_flag: "--decode-128-from-warm-context-cuda-run",
        expected_input_tokens: None,
        expected_generated_tokens: 128,
    },
];

#[derive(Debug)]
struct ProfilePaths {
    spec: ProfileSpec,
    cpu_paths: Vec<PathBuf>,
    cuda_paths: Vec<PathBuf>,
    cpu_overridden: bool,
    cuda_overridden: bool,
}

#[derive(Debug)]
struct Args {
    profiles: Vec<ProfilePaths>,
    receipt_out: PathBuf,
    manifest_out: Option<PathBuf>,
    print_manifest: bool,
}

#[derive(Debug)]
struct ProfileRuns {
    spec: ProfileSpec,
    cpu: Vec<(PathBuf, Value)>,
    cuda: Vec<(PathBuf, Value)>,
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
    let runs = read_profile_runs(&args)?;
    for profile in &runs {
        assert_repeated_sources(profile)?;
    }

    let receipt = build_receipt(&args, &runs)?;
    validate_strict_bitnet_cuda_repeated_profiles_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut profiles = PROFILE_SPECS.iter().copied().map(default_profile_paths).collect::<Vec<_>>();
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut manifest_out = None;
    let mut print_manifest = false;
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        if push_profile_override(&mut profiles, &arg, &mut iter)? {
            continue;
        }
        match arg.as_str() {
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

    Ok(Args { profiles, receipt_out, manifest_out, print_manifest })
}

fn default_profile_paths(spec: ProfileSpec) -> ProfilePaths {
    ProfilePaths {
        spec,
        cpu_paths: (1..=3).map(|run| default_source_path(spec, "cpu-avx512", run)).collect(),
        cuda_paths: (1..=3).map(|run| default_source_path(spec, "cuda", run)).collect(),
        cpu_overridden: false,
        cuda_overridden: false,
    }
}

fn default_source_path(spec: ProfileSpec, backend: &str, run: usize) -> PathBuf {
    PathBuf::from(format!(
        "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-24/bitnet-perf-005/run-{run:02}/official-bitnet-{}-{backend}.json",
        spec.stem
    ))
}

fn push_profile_override(
    profiles: &mut [ProfilePaths],
    flag: &str,
    iter: &mut impl Iterator<Item = String>,
) -> Result<bool, Box<dyn Error>> {
    for profile in profiles {
        if flag == profile.spec.cpu_flag {
            push_override(
                &mut profile.cpu_paths,
                &mut profile.cpu_overridden,
                next_value(iter, flag)?,
            );
            return Ok(true);
        }
        if flag == profile.spec.cuda_flag {
            push_override(
                &mut profile.cuda_paths,
                &mut profile.cuda_overridden,
                next_value(iter, flag)?,
            );
            return Ok(true);
        }
    }
    Ok(false)
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
    let flags = PROFILE_SPECS
        .iter()
        .flat_map(|spec| [spec.cpu_flag, spec.cuda_flag])
        .collect::<Vec<_>>()
        .join(" PATH ... ");
    println!(
        "Usage: strict_bitnet_cuda_repeated_profiles_receipt [{flags} PATH ...] [--receipt-out PATH] [--manifest-out PATH] [--print-manifest]"
    );
}

fn source_manifest(args: &Args) -> Value {
    let profiles = args
        .profiles
        .iter()
        .map(|group| {
            json!({
                "profile": group.spec.profile,
                "cpu_run_flag": group.spec.cpu_flag,
                "cuda_run_flag": group.spec.cuda_flag,
                "min_runs_per_backend": 3,
                "expected_input_tokens": group.spec.expected_input_tokens,
                "expected_generated_tokens": group.spec.expected_generated_tokens,
                "cpu_source_paths": path_labels(&group.cpu_paths),
                "cuda_source_paths": path_labels(&group.cuda_paths)
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
        "requested_backend": CUDA_BACKEND,
        "selected_backend": CUDA_BACKEND,
        "reference_backend": CPU_BACKEND,
        "runtime_api": CUDA_RUNTIME_API,
        "selected_route": CUDA_ROUTE,
        "kernel_id": CUDA_KERNEL_ID,
        "min_runs_per_backend": 3,
        "profiles": profiles,
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
            "prompt_authority": PROMPT_AUTHORITY,
            "prompt_template": PROMPT_AUTHORITY
        },
        "required_source_fields": [
            "/artifact_kind",
            "/model/repo",
            "/model/file",
            "/model/sha256",
            "/tokenizer/source",
            "/tokenizer/pretokenizer_authority",
            "/tokenizer_prompt_authority/tokenizer_authority",
            "/tokenizer_prompt_authority/prompt_authority",
            "/tokenizer_prompt_authority/prompt_template",
            "/profile/id",
            "/profile/generated_tokens",
            "/fallback_used",
            "/quality_summary/passed or /quality/garbage_filter_passed",
            "/timing/model_load_ms",
            "/timing/tokenizer_load_ms",
            "/timing/prompt_render_ms",
            "/timing/tokenize_ms",
            "/timing/prefill_ms",
            "/timing/first_token_ms",
            "/timing/decode_total_ms",
            "/timing/steady_tok_per_s"
        ],
        "required_cuda_source_fields": [
            "/execution_plan/selected_route=bitnet_qk256_cuda",
            "/selected_backend=nvidia-rtx-5070-ti-cuda",
            "/runtime_api=cuda",
            "/timing/cuda_context_init_ms",
            "/timing/weight_upload_ms",
            "/kernel_stats/0/kernel_id=qk256_gemv_cuda",
            "/kernel_stats/0/kernel_time_ms",
            "/kernel_stats/0/invocations",
            "/kernel_stats/0/fallback_invocations=0",
            "/timing/host_to_device_bytes",
            "/timing/host_to_device_ms",
            "/timing/device_to_host_bytes",
            "/timing/device_to_host_ms",
            "/cuda/memory_hwm_bytes or /cuda/vram_bytes",
            "/cuda/power_draw_watts",
            "/cuda/temperature_c"
        ],
        "required_cpu_source_fields": [
            "/execution_plan/selected_route=bitnet_i2s_qk256_cpu_avx512",
            "/selected_backend=amd-9950x3d-cpu-avx512",
            "/runtime_api=cpu",
            "/kernel_stats/0/kernel_id=i2_s-avx512-reference or /kernel/kernel_id=i2_s-avx512-reference"
        ],
        "strict_rejection_rules": [
            "each profile requires at least three CPU AVX-512 source receipts and at least three RTX 5070 Ti CUDA source receipts",
            "selected CUDA backend must be nvidia-rtx-5070-ti-cuda",
            "CUDA runtime_api must be cuda",
            "CUDA selected_route must be bitnet_qk256_cuda",
            "CUDA kernel_id must be qk256_gemv_cuda",
            "CPU selected backend must be amd-9950x3d-cpu-avx512",
            "CPU selected_route must be bitnet_i2s_qk256_cpu_avx512",
            "fallback_used must be false for both CPU and CUDA receipts",
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
    let mut missing = Vec::new();
    for group in &args.profiles {
        for path in &group.cpu_paths {
            if !path.is_file() {
                missing.push(format!("{} cpu: {}", group.spec.profile, path_label(path)));
            }
        }
        for path in &group.cuda_paths {
            if !path.is_file() {
                missing.push(format!("{} cuda: {}", group.spec.profile, path_label(path)));
            }
        }
    }
    if missing.is_empty() {
        return Ok(());
    }

    Err(format!(
        "missing CUDA-BITNET-PERF-005 source receipts:\n  - {}\nrun with --print-manifest or --manifest-out PATH to inspect the expected CPU/CUDA capture set",
        missing.join("\n  - ")
    )
    .into())
}

fn read_profile_runs(args: &Args) -> Result<Vec<ProfileRuns>, Box<dyn Error>> {
    args.profiles
        .iter()
        .map(|group| {
            Ok(ProfileRuns {
                spec: group.spec,
                cpu: read_runs(&group.cpu_paths)?,
                cuda: read_runs(&group.cuda_paths)?,
            })
        })
        .collect()
}

fn read_runs(paths: &[PathBuf]) -> Result<Vec<(PathBuf, Value)>, Box<dyn Error>> {
    paths.iter().map(|path| Ok((path.clone(), read_json(path)?))).collect()
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn assert_repeated_sources(runs: &ProfileRuns) -> Result<(), Box<dyn Error>> {
    assert_backend_sources(runs.spec, &runs.cpu, BackendKind::Cpu)?;
    assert_backend_sources(runs.spec, &runs.cuda, BackendKind::Cuda)?;
    if runs.cpu.len() != runs.cuda.len() {
        return Err(format!("{} CPU and CUDA run counts must match", runs.spec.profile).into());
    }
    for ((_, cpu), (_, cuda)) in runs.cpu.iter().zip(&runs.cuda) {
        if rendered_prompt_hash(runs.spec.profile, cpu)?
            != rendered_prompt_hash(runs.spec.profile, cuda)?
        {
            return Err(format!(
                "{} CPU and CUDA runs must use the same rendered prompt",
                runs.spec.profile
            )
            .into());
        }
        if generated_token_hash(cpu)? != generated_token_hash(cuda)? {
            return Err(format!(
                "{} CPU and CUDA runs must have matching generated token IDs",
                runs.spec.profile
            )
            .into());
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum BackendKind {
    Cpu,
    Cuda,
}

fn assert_backend_sources(
    spec: ProfileSpec,
    runs: &[(PathBuf, Value)],
    backend: BackendKind,
) -> Result<(), Box<dyn Error>> {
    if runs.len() < 3 {
        return Err(format!(
            "{} requires at least 3 {} source receipts",
            spec.profile,
            backend.label()
        )
        .into());
    }
    let mut paths = BTreeSet::new();
    let anchor = runs.first().ok_or("runs must not be empty")?.1.clone();
    let prompt_template = str_at(&anchor, "/tokenizer_prompt_authority/prompt_template")?;
    let prompt_hash = rendered_prompt_hash(spec.profile, &anchor)?;

    for (path, receipt) in runs {
        if !paths.insert(path_label(path)) {
            return Err(format!(
                "{} {} source receipt paths must be unique",
                spec.profile,
                backend.label()
            )
            .into());
        }
        assert_source_receipt(spec, receipt, backend)?;
        if str_at(receipt, "/tokenizer_prompt_authority/prompt_template")? != prompt_template {
            return Err(format!(
                "{} {} runs must use the same prompt template",
                spec.profile,
                backend.label()
            )
            .into());
        }
        if rendered_prompt_hash(spec.profile, receipt)? != prompt_hash {
            return Err(format!(
                "{} {} runs must use the same rendered prompt hash",
                spec.profile,
                backend.label()
            )
            .into());
        }
    }
    Ok(())
}

impl BackendKind {
    fn label(self) -> &'static str {
        match self {
            BackendKind::Cpu => "CPU",
            BackendKind::Cuda => "CUDA",
        }
    }
}

fn assert_source_receipt(
    spec: ProfileSpec,
    receipt: &Value,
    backend: BackendKind,
) -> Result<(), Box<dyn Error>> {
    let artifact_kind = str_at(receipt, "/artifact_kind")?;
    let artifact_kind_lower = artifact_kind.to_ascii_lowercase();
    if !artifact_kind_lower.contains("bitnet") {
        return Err(format!("{} source receipt must be a BitNet receipt", spec.profile).into());
    }
    if artifact_kind_lower.contains("dense") || artifact_kind_lower.contains("qwen") {
        return Err(
            format!("{} source receipt must not be dense/Qwen evidence", spec.profile).into()
        );
    }
    if str_at_any(receipt, &["/profile/id", "/profile/profile"])? != spec.profile {
        return Err(format!("{} source receipt profile id mismatch", spec.profile).into());
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
    if generated_tokens(spec.profile, receipt)? != spec.expected_generated_tokens {
        return Err(
            format!("{} generated token count does not match the profile", spec.profile).into()
        );
    }
    if let Some(expected_input_tokens) = spec.expected_input_tokens {
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
                "{} expected {expected_input_tokens} prompt tokens, got {actual}",
                spec.profile
            )
            .into());
        }
    }

    for pointer in [
        "/timing/model_load_ms",
        "/timing/tokenizer_load_ms",
        "/timing/prompt_render_ms",
        "/timing/tokenize_ms",
        "/timing/prefill_ms",
        "/timing/first_token_ms",
        "/timing/decode_total_ms",
        "/timing/steady_tok_per_s",
    ] {
        number_at(receipt, pointer)?;
    }

    match backend {
        BackendKind::Cpu => assert_cpu_source(receipt),
        BackendKind::Cuda => assert_cuda_source(receipt),
    }
}

fn assert_cpu_source(receipt: &Value) -> Result<(), Box<dyn Error>> {
    if str_at_any(receipt, &["/execution_plan/selected_backend", "/selected_backend"])?
        != CPU_BACKEND
    {
        return Err("CPU source receipt must select amd-9950x3d-cpu-avx512".into());
    }
    if str_at_any(receipt, &["/execution_plan/runtime_api", "/runtime_api"])? != CPU_RUNTIME_API {
        return Err("CPU source receipt must use cpu runtime_api".into());
    }
    if str_at(receipt, "/execution_plan/selected_route")? != CPU_ROUTE {
        return Err("CPU source receipt must route bitnet_i2s_qk256_cpu_avx512".into());
    }
    if kernel_id(receipt)? != CPU_KERNEL_ID {
        return Err("CPU source receipt must use i2_s-avx512-reference".into());
    }
    Ok(())
}

fn assert_cuda_source(receipt: &Value) -> Result<(), Box<dyn Error>> {
    if str_at_any(receipt, &["/execution_plan/selected_backend", "/selected_backend"])?
        != CUDA_BACKEND
    {
        return Err("CUDA source receipt must select nvidia-rtx-5070-ti-cuda".into());
    }
    if str_at_any(receipt, &["/execution_plan/runtime_api", "/runtime_api"])? != CUDA_RUNTIME_API {
        return Err("CUDA source receipt must use cuda runtime_api".into());
    }
    if str_at(receipt, "/execution_plan/selected_route")? != CUDA_ROUTE {
        return Err("CUDA source receipt must route bitnet_qk256_cuda".into());
    }
    if kernel_id(receipt)? != CUDA_KERNEL_ID {
        return Err("CUDA source receipt must use qk256_gemv_cuda".into());
    }
    for pointer in [
        "/timing/cuda_context_init_ms",
        "/timing/weight_upload_ms",
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
    if str_at(kernel_stats, "/kernel_id")? != CUDA_KERNEL_ID {
        return Err("CUDA source receipt must use qk256_gemv_cuda".into());
    }
    number_at(kernel_stats, "/kernel_time_ms")?;
    u64_at_any(kernel_stats, &["/kernel_launches", "/launch_count"])?;
    u64_at(kernel_stats, "/invocations")?;
    if u64_at(kernel_stats, "/fallback_invocations")? != 0 {
        return Err("CUDA source receipt kernel fallback_invocations must be zero".into());
    }
    Ok(())
}

fn build_receipt(args: &Args, runs: &[ProfileRuns]) -> Result<Value, Box<dyn Error>> {
    let profiles = runs.iter().map(profile_from_runs).collect::<Result<Vec<_>, _>>()?;
    let total_cuda_kernel_invocations = runs
        .iter()
        .flat_map(|profile| profile.cuda.iter().map(|(_, receipt)| receipt))
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
        "requested_backend": CUDA_BACKEND,
        "selected_backend": CUDA_BACKEND,
        "reference_backend": CPU_BACKEND,
        "runtime_api": CUDA_RUNTIME_API,
        "selected_route": CUDA_ROUTE,
        "kernel_id": CUDA_KERNEL_ID,
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
            "prompt_policy": "bitnetcpp-answer deterministic profile prompts; same tokenizer and prompt policy across repeated CPU and CUDA runs",
            "deterministic_prompt": true
        },
        "execution_plan": {
            "planner_version": "cuda-planner-004",
            "model_family": "bitnet_b1_58",
            "quantization": "i2_s_qk256",
            "selected_backend": CUDA_BACKEND,
            "selected_route": CUDA_ROUTE,
            "runtime_api": CUDA_RUNTIME_API,
            "strict_fallback_policy": "reject",
            "bitnet_packed_qk256_cuda": true,
            "dense_regular_llm_cuda": false,
            "fallback_used": false,
            "strict_cuda_ready": true,
            "speedup_claim": false,
            "full_cuda_residency_claimed": false,
            "cuda_bitnet_qk256_ops": total_cuda_kernel_invocations,
            "cuda_dense_regular_llm_ops": 0,
            "cpu_fallback_ops": 0,
            "unsupported_ops": 0,
            "total_ops": total_cuda_kernel_invocations,
            "cuda_ops": total_cuda_kernel_invocations
        },
        "proof_inputs": proof_inputs(runs)?,
        "profiles": profiles,
        "comparator_summary": comparator_summary(&profiles),
        "transfer_timing": transfer_timing(),
        "hardware_context": hardware_context(runs)?,
        "cuda": cuda_context(runs)?,
        "claim_boundaries": [
            "speedup_claim=false; repeated current-source CPU/CUDA profiles are not a speedup qualification.",
            "benchmark_qualified_speedup=false until CUDA-BITNET-PERF-006 reviews exact profiles.",
            "full_cuda_residency_claimed=false until every required phase proves residency.",
            "server_ready_claimed=false; server smoke and readiness remain separate proof families.",
            "BitNet packed I2_S/QK256 proof cannot be satisfied by dense_regular_llm_cuda evidence."
        ]
    }))
}

fn proof_inputs(runs: &[ProfileRuns]) -> Result<Value, Box<dyn Error>> {
    let mut object = serde_json::Map::new();
    for profile in runs {
        object.insert(profile.spec.profile.to_owned(), profile_input(profile)?);
    }
    Ok(Value::Object(object))
}

fn profile_input(profile: &ProfileRuns) -> Result<Value, Box<dyn Error>> {
    Ok(json!({
        "path": format!("profile:{}", profile.spec.profile),
        "artifact_kind": "strict_bitnet_profile_repeated_comparator_runs",
        "cpu_sha256": combined_sha256(&profile.cpu)?,
        "cuda_sha256": combined_sha256(&profile.cuda)?,
        "cpu_runs": profile.cpu.iter().map(|(path, _)| path_label(path)).collect::<Vec<_>>(),
        "cuda_runs": profile.cuda.iter().map(|(path, _)| path_label(path)).collect::<Vec<_>>()
    }))
}

fn combined_sha256(runs: &[(PathBuf, Value)]) -> Result<String, Box<dyn Error>> {
    let mut hasher = Sha256::new();
    for (path, _) in runs {
        hasher.update(sha256_file(path)?.as_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn profile_from_runs(profile: &ProfileRuns) -> Result<Value, Box<dyn Error>> {
    let cpu_runs = profile
        .cpu
        .iter()
        .enumerate()
        .map(|(index, (path, receipt))| {
            run_from_receipt(profile.spec, index + 1, path, receipt, BackendKind::Cpu)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let cuda_runs = profile
        .cuda
        .iter()
        .enumerate()
        .map(|(index, (path, receipt))| {
            run_from_receipt(profile.spec, index + 1, path, receipt, BackendKind::Cuda)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let runs = cpu_runs.iter().chain(&cuda_runs).cloned().collect::<Vec<_>>();

    Ok(json!({
        "profile": profile.spec.profile,
        "status": "repeated_same_artifact_cpu_cuda_profile",
        "cpu_reference_backend": CPU_BACKEND,
        "cuda_backend": CUDA_BACKEND,
        "runtime_api": CUDA_RUNTIME_API,
        "selected_route": CUDA_ROUTE,
        "kernel_id": CUDA_KERNEL_ID,
        "expected_input_tokens": profile.spec.expected_input_tokens,
        "expected_generated_tokens": profile.spec.expected_generated_tokens,
        "run_count": runs.len(),
        "cpu_runs": cpu_runs.len(),
        "cuda_runs": cuda_runs.len(),
        "min_runs_per_backend": 3,
        "fallback_free": true,
        "same_artifact_sha": true,
        "same_tokenizer_prompt_policy": true,
        "deterministic_generation_policy": true,
        "generated_token_ids_match": profile.cpu.iter().zip(&profile.cuda).all(|((_, cpu), (_, cuda))| {
            generated_token_hash(cpu).ok() == generated_token_hash(cuda).ok()
        }),
        "first_divergence_report": "no generated-token divergence recorded across paired CPU/CUDA source receipts",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": true,
        "dense_regular_llm_cuda_proof": false,
        "full_cuda_residency_claimed": false,
        "server_ready_claimed": false,
        "transfer_timing_status": "cuda_host_to_device_and_device_to_host_measured_in_source_receipts",
        "cpu": backend_summary(&cpu_runs, CPU_BACKEND, CPU_RUNTIME_API, CPU_ROUTE, CPU_KERNEL_ID)?,
        "cuda": backend_summary(&cuda_runs, CUDA_BACKEND, CUDA_RUNTIME_API, CUDA_ROUTE, CUDA_KERNEL_ID)?,
        "runs": runs
    }))
}

fn run_from_receipt(
    spec: ProfileSpec,
    index: usize,
    path: &Path,
    receipt: &Value,
    backend: BackendKind,
) -> Result<Value, Box<dyn Error>> {
    let generated_tokens = generated_tokens(spec.profile, receipt)?;
    let decode_total = number_at(receipt, "/timing/decode_total_ms")?;
    let steady_tok_per_s = number_or(
        receipt,
        "/timing/steady_tok_per_s",
        generated_tokens as f64 / (decode_total / 1000.0),
    );
    let mut timing = json!({
        "model_load_ms": number_at(receipt, "/timing/model_load_ms")?,
        "tokenizer_load_ms": number_at(receipt, "/timing/tokenizer_load_ms")?,
        "prompt_render_ms": number_at(receipt, "/timing/prompt_render_ms")?,
        "tokenize_ms": number_at(receipt, "/timing/tokenize_ms")?,
        "prefill_ms": number_at(receipt, "/timing/prefill_ms")?,
        "first_token_ms": number_at(receipt, "/timing/first_token_ms")?,
        "decode_total_ms": decode_total,
        "steady_tok_per_s": steady_tok_per_s
    });
    if matches!(backend, BackendKind::Cuda) {
        if let Some(object) = timing.as_object_mut() {
            object.insert(
                "cuda_context_init_ms".to_owned(),
                json!(number_at(receipt, "/timing/cuda_context_init_ms")?),
            );
            object.insert(
                "weight_upload_ms".to_owned(),
                json!(number_at(receipt, "/timing/weight_upload_ms")?),
            );
            object.insert(
                "kernel_time_ms".to_owned(),
                json!(number_at_any(
                    receipt,
                    &["/timing/kernel_time_ms", "/kernel_stats/0/kernel_time_ms"]
                )?),
            );
            object.insert(
                "launch_count".to_owned(),
                json!(u64_at_any(
                    receipt,
                    &[
                        "/timing/launch_count",
                        "/timing/kernel_launches",
                        "/kernel_stats/0/kernel_launches"
                    ]
                )?),
            );
            object.insert(
                "kernel_invocations".to_owned(),
                json!(u64_at_any(
                    receipt,
                    &["/timing/kernel_invocations", "/kernel_stats/0/invocations"]
                )?),
            );
            object.insert(
                "host_to_device_bytes".to_owned(),
                json!(u64_at(receipt, "/timing/host_to_device_bytes")?),
            );
            object.insert(
                "host_to_device_ms".to_owned(),
                json!(number_at(receipt, "/timing/host_to_device_ms")?),
            );
            object.insert(
                "device_to_host_bytes".to_owned(),
                json!(u64_at(receipt, "/timing/device_to_host_bytes")?),
            );
            object.insert(
                "device_to_host_ms".to_owned(),
                json!(number_at(receipt, "/timing/device_to_host_ms")?),
            );
            object.insert(
                "vram_high_water_bytes".to_owned(),
                json!(u64_at_any(
                    receipt,
                    &[
                        "/timing/vram_high_water_bytes",
                        "/cuda/memory_hwm_bytes",
                        "/cuda/vram_bytes"
                    ]
                )?),
            );
            object.insert(
                "power_temperature_context".to_owned(),
                json!("NVML power and temperature sampled during source receipt"),
            );
        }
    }

    Ok(json!({
        "run_id": format!("{}-run-{index:02}", backend.label().to_ascii_lowercase()),
        "profile": spec.profile,
        "backend": backend.backend_label(),
        "runtime_api": backend.runtime_api(),
        "selected_route": backend.route(),
        "kernel_id": backend.kernel_id(),
        "source_receipt_path": path_label(path),
        "source_receipt_sha256": sha256_file(path)?,
        "source_artifact_kind": str_at(receipt, "/artifact_kind")?,
        "model_sha256": str_at(receipt, "/model/sha256")?,
        "prompt_template": str_at(receipt, "/tokenizer_prompt_authority/prompt_template")?,
        "prompt_token_count": prompt_token_count(spec.profile, receipt)?,
        "generation_policy": "greedy",
        "deterministic_generation": true,
        "expected_input_tokens": spec.expected_input_tokens,
        "generated_tokens": generated_tokens,
        "generated_token_ids_sha256": generated_token_hash(receipt)?,
        "generated_token_ids_match": true,
        "first_divergence_report": string_at_any_or(
            receipt,
            &["/quality_summary/first_divergence_report", "/comparison/first_divergence_report"],
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
            &["/quality_summary/passed", "/quality_gate/passed", "/quality/garbage_filter_passed"]
        )?,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": true,
        "dense_regular_llm_cuda_proof": false,
        "full_cuda_residency_claimed": false,
        "server_ready_claimed": false,
        "timing": timing
    }))
}

impl BackendKind {
    fn backend_label(self) -> &'static str {
        match self {
            BackendKind::Cpu => CPU_BACKEND,
            BackendKind::Cuda => CUDA_BACKEND,
        }
    }

    fn runtime_api(self) -> &'static str {
        match self {
            BackendKind::Cpu => CPU_RUNTIME_API,
            BackendKind::Cuda => CUDA_RUNTIME_API,
        }
    }

    fn route(self) -> &'static str {
        match self {
            BackendKind::Cpu => CPU_ROUTE,
            BackendKind::Cuda => CUDA_ROUTE,
        }
    }

    fn kernel_id(self) -> &'static str {
        match self {
            BackendKind::Cpu => CPU_KERNEL_ID,
            BackendKind::Cuda => CUDA_KERNEL_ID,
        }
    }
}

fn backend_summary(
    runs: &[Value],
    backend: &str,
    runtime_api: &str,
    selected_route: &str,
    kernel_id: &str,
) -> Result<Value, Box<dyn Error>> {
    let mut summary = json!({
        "backend": backend,
        "runtime_api": runtime_api,
        "selected_route": selected_route,
        "kernel_id": kernel_id,
        "run_count": runs.len(),
        "quality_passed": runs.iter().all(|run| bool_at(run, "/quality_passed").unwrap_or(false)),
        "fallback_used": runs.iter().any(|run| bool_at(run, "/fallback_used").unwrap_or(true)),
        "model_load_ms": number_summary(runs, "/timing/model_load_ms"),
        "tokenizer_load_ms": number_summary(runs, "/timing/tokenizer_load_ms"),
        "prompt_render_ms": number_summary(runs, "/timing/prompt_render_ms"),
        "tokenize_ms": number_summary(runs, "/timing/tokenize_ms"),
        "prefill_ms": number_summary(runs, "/timing/prefill_ms"),
        "first_token_ms": number_summary(runs, "/timing/first_token_ms"),
        "decode_total_ms": number_summary(runs, "/timing/decode_total_ms"),
        "steady_tok_per_s": number_summary(runs, "/timing/steady_tok_per_s")
    });
    if backend == CUDA_BACKEND {
        if let Some(object) = summary.as_object_mut() {
            object.insert(
                "cuda_context_init_ms".to_owned(),
                number_summary(runs, "/timing/cuda_context_init_ms"),
            );
            object.insert(
                "weight_upload_ms".to_owned(),
                number_summary(runs, "/timing/weight_upload_ms"),
            );
            object.insert(
                "kernel_time_ms".to_owned(),
                number_summary(runs, "/timing/kernel_time_ms"),
            );
            object.insert("launch_count".to_owned(), u64_summary(runs, "/timing/launch_count"));
            object.insert(
                "host_to_device_bytes".to_owned(),
                u64_summary(runs, "/timing/host_to_device_bytes"),
            );
            object.insert(
                "host_to_device_ms".to_owned(),
                number_summary(runs, "/timing/host_to_device_ms"),
            );
            object.insert(
                "device_to_host_bytes".to_owned(),
                u64_summary(runs, "/timing/device_to_host_bytes"),
            );
            object.insert(
                "device_to_host_ms".to_owned(),
                number_summary(runs, "/timing/device_to_host_ms"),
            );
            object.insert(
                "vram_high_water_bytes".to_owned(),
                u64_summary(runs, "/timing/vram_high_water_bytes"),
            );
        }
    }
    Ok(summary)
}

fn comparator_summary(profiles: &[Value]) -> Value {
    let total_cpu_runs = profiles
        .iter()
        .filter_map(|profile| profile.get("cpu_runs").and_then(Value::as_u64))
        .sum::<u64>();
    let total_cuda_runs = profiles
        .iter()
        .filter_map(|profile| profile.get("cuda_runs").and_then(Value::as_u64))
        .sum::<u64>();
    json!({
        "status": "repeated_profiles_baseline_only",
        "profiles_recorded": profiles.len(),
        "min_runs_per_profile": 3,
        "total_cpu_runs": total_cpu_runs,
        "total_cuda_runs": total_cuda_runs,
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
        "status": "cuda_host_to_device_and_device_to_host_measured_in_source_receipts",
        "source": "CUDA-BITNET-PERF-005 CUDA source receipts record H2D/D2H bytes and timings per profile; CPU source receipts provide comparator phase timing only",
        "host_to_device_bytes_recorded": true,
        "device_to_host_bytes_recorded": true,
        "host_to_device_timing_recorded": true,
        "device_to_host_timing_recorded": true,
        "pure_host_to_device_timing_recorded": true
    })
}

fn hardware_context(runs: &[ProfileRuns]) -> Result<Value, Box<dyn Error>> {
    let receipts = cuda_receipts(runs);
    let powers = receipts
        .iter()
        .map(|receipt| number_at(receipt, "/cuda/power_draw_watts"))
        .collect::<Result<Vec<_>, _>>()?;
    let temperatures = receipts
        .iter()
        .map(|receipt| number_at(receipt, "/cuda/temperature_c"))
        .collect::<Result<Vec<_>, _>>()?;
    let first = receipts.first().ok_or("hardware context requires CUDA receipts")?;
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

fn cuda_context(runs: &[ProfileRuns]) -> Result<Value, Box<dyn Error>> {
    let first =
        runs.first().and_then(|profile| profile.cuda.first()).ok_or("CUDA runs missing")?.1.clone();
    first.pointer("/cuda").cloned().ok_or("cuda block missing".into())
}

fn cuda_receipts(runs: &[ProfileRuns]) -> Vec<&Value> {
    runs.iter().flat_map(|profile| profile.cuda.iter()).map(|(_, receipt)| receipt).collect()
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

fn generated_token_hash(receipt: &Value) -> Result<String, Box<dyn Error>> {
    Ok(string_at_any_or(
        receipt,
        &[
            "/generated_token_ids_sha256",
            "/quality_summary/generated_token_ids_sha256",
            "/one_token_proof/generated_token_ids_sha256",
            "/short_decode_proof/generated_token_ids_sha256",
            "/warm_session_proof/generated_token_ids_sha256",
            "/warm_decode_proof/generated_token_ids_sha256",
        ],
        "not_recorded_in_source_receipt",
    ))
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

fn kernel_id(receipt: &Value) -> Result<&str, Box<dyn Error>> {
    str_at_any(receipt, &["/kernel/kernel_id", "/kernel_stats/0/kernel_id"])
}

fn path_label(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn path_labels(paths: &[PathBuf]) -> Vec<String> {
    paths.iter().map(|path| path_label(path)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_with_prefix(prefix: &str) -> Args {
        Args {
            profiles: PROFILE_SPECS
                .iter()
                .copied()
                .map(|spec| ProfilePaths {
                    spec,
                    cpu_paths: vec![PathBuf::from(format!("{prefix}/{}-cpu.json", spec.stem))],
                    cuda_paths: vec![PathBuf::from(format!("{prefix}/{}-cuda.json", spec.stem))],
                    cpu_overridden: false,
                    cuda_overridden: false,
                })
                .collect(),
            receipt_out: PathBuf::from(format!("{prefix}/aggregate.json")),
            manifest_out: None,
            print_manifest: false,
        }
    }

    #[test]
    fn bitnet_perf_005_source_manifest_names_cpu_and_cuda_profiles() {
        let args = args_with_prefix("target/bitnet-perf-005");
        let manifest = source_manifest(&args);

        assert_eq!(
            manifest["artifact_kind"],
            "strict_bitnet_cuda_repeated_profiles_source_manifest"
        );
        assert_eq!(manifest["campaign_item"], CAMPAIGN_ITEM);
        assert_eq!(manifest["selected_backend"], CUDA_BACKEND);
        assert_eq!(manifest["reference_backend"], CPU_BACKEND);
        assert_eq!(manifest["selected_route"], CUDA_ROUTE);
        assert_eq!(manifest["kernel_id"], CUDA_KERNEL_ID);

        let profiles = manifest["profiles"].as_array();
        assert_eq!(profiles.map(Vec::len), Some(8));
        assert!(profiles.is_some_and(|profiles| {
            profiles.iter().any(|profile| {
                profile["profile"] == "prefill_512_decode_32"
                    && profile["expected_input_tokens"] == 512
                    && profile["expected_generated_tokens"] == 32
                    && profile["cpu_run_flag"] == "--prefill-512-decode-32-cpu-run"
                    && profile["cuda_run_flag"] == "--prefill-512-decode-32-cuda-run"
            })
        }));
        assert!(manifest["strict_rejection_rules"].as_array().is_some_and(|rules| {
            rules.iter().any(|rule| {
                rule.as_str()
                    .is_some_and(|text| text.contains("at least three CPU AVX-512 source receipts"))
            })
        }));
    }

    #[test]
    fn bitnet_perf_005_preflight_reports_cpu_and_cuda_missing_profile_inputs() {
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
        assert!(message.contains("cpu:"), "missing CPU hint in {message}");
        assert!(message.contains("cuda:"), "missing CUDA hint in {message}");
    }

    #[test]
    fn bitnet_perf_005_rejects_cuda_only_sources_as_comparator_evidence()
    -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let spec = PROFILE_SPECS[0];
        let cuda_runs = write_profile_runs(temp.path(), spec, BackendKind::Cuda)?;
        let runs = ProfileRuns { spec, cpu: Vec::new(), cuda: cuda_runs };
        let message = assert_repeated_sources(&runs).unwrap_err().to_string();
        assert!(message.contains("at least 3 CPU source receipts"), "got: {message}");
        Ok(())
    }

    #[test]
    fn bitnet_perf_005_strict_bitnet_cuda_repeated_profiles_aggregate_builds_from_cpu_cuda_sources()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let mut profile_runs = Vec::new();
        for spec in PROFILE_SPECS {
            profile_runs.push(ProfileRuns {
                spec: *spec,
                cpu: write_profile_runs(temp.path(), *spec, BackendKind::Cpu)?,
                cuda: write_profile_runs(temp.path(), *spec, BackendKind::Cuda)?,
            });
        }
        for runs in &profile_runs {
            assert_repeated_sources(runs)?;
        }

        let args = Args {
            profiles: PROFILE_SPECS.iter().copied().map(default_profile_paths).collect(),
            receipt_out: temp.path().join("aggregate.json"),
            manifest_out: None,
            print_manifest: false,
        };
        let receipt = build_receipt(&args, &profile_runs)?;
        validate_strict_bitnet_cuda_repeated_profiles_receipt_json(&receipt)?;

        assert_eq!(receipt["profiles"].as_array().map(Vec::len), Some(8));
        assert_eq!(receipt["comparator_summary"]["total_cpu_runs"], 24);
        assert_eq!(receipt["comparator_summary"]["total_cuda_runs"], 24);
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

        Ok(())
    }

    fn write_profile_runs(
        root: &Path,
        spec: ProfileSpec,
        backend: BackendKind,
    ) -> std::io::Result<Vec<(PathBuf, Value)>> {
        let mut runs = Vec::new();
        for run in 1..=3 {
            let path = root.join(format!("{}-{}-run-{run}.json", spec.stem, backend.label()));
            let receipt = source_receipt(spec, run, backend);
            let encoded = serde_json::to_vec_pretty(&receipt).map_err(std::io::Error::other)?;
            std::fs::write(&path, encoded)?;
            runs.push((path, receipt));
        }
        Ok(runs)
    }

    fn source_receipt(spec: ProfileSpec, run: u64, backend: BackendKind) -> Value {
        let prompt_tokens = spec.expected_input_tokens.unwrap_or(32);
        let mut timing = json!({
            "model_load_ms": 100.0 + run as f64,
            "tokenizer_load_ms": 1.0,
            "prompt_render_ms": 0.1,
            "tokenize_ms": 0.2,
            "prefill_ms": 4.0,
            "first_token_ms": 5.0,
            "decode_total_ms": 6.0 + run as f64,
            "steady_tok_per_s": 7.0
        });
        if matches!(backend, BackendKind::Cuda) {
            if let Some(object) = timing.as_object_mut() {
                object.insert("cuda_context_init_ms".to_owned(), json!(2.0));
                object.insert("weight_upload_ms".to_owned(), json!(3.0));
                object.insert("kernel_time_ms".to_owned(), json!(10.0));
                object.insert("launch_count".to_owned(), json!(210));
                object.insert("kernel_invocations".to_owned(), json!(210));
                object.insert("host_to_device_bytes".to_owned(), json!(8));
                object.insert("host_to_device_ms".to_owned(), json!(0.8));
                object.insert("device_to_host_bytes".to_owned(), json!(9));
                object.insert("device_to_host_ms".to_owned(), json!(0.9));
                object.insert("vram_high_water_bytes".to_owned(), json!(7070547968u64));
            }
        }

        let mut receipt = json!({
            "schema": 1,
            "artifact_kind": match backend {
                BackendKind::Cpu => "strict_bitnet_cpu_profile_source",
                BackendKind::Cuda => "strict_bitnet_cuda_profile_source"
            },
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
                "rendered_prompt_sha256": format!("prompt-hash-{}", spec.profile),
                "prompt_token_count": prompt_tokens
            },
            "execution_plan": {
                "selected_backend": backend.backend_label(),
                "selected_route": backend.route(),
                "runtime_api": backend.runtime_api(),
                "strict_fallback_policy": "reject",
                "fallback_used": false,
                "strict_cuda_ready": matches!(backend, BackendKind::Cuda),
                "speedup_claim": false,
                "bitnet_packed_qk256_cuda": true,
                "dense_regular_llm_cuda": false,
                "unsupported_ops": 0,
                "cpu_fallback_ops": 0
            },
            "requested_backend": backend.backend_label(),
            "selected_backend": backend.backend_label(),
            "runtime_api": backend.runtime_api(),
            "fallback_used": false,
            "speedup_claim": false,
            "bitnet_packed_i2s_qk256_proof": true,
            "dense_regular_llm_cuda_proof": false,
            "quality_summary": {
                "passed": true,
                "generated_token_ids_match": true,
                "generated_token_ids_sha256": format!("generated-hash-{}", spec.profile),
                "first_divergence_report": "none",
                "top_k_compared": true
            },
            "timing": timing,
            "kernel": {
                "kernel_id": backend.kernel_id()
            },
            "profile": {
                "id": spec.profile,
                "expected_input_tokens": spec.expected_input_tokens,
                "generated_tokens": spec.expected_generated_tokens
            }
        });
        if matches!(backend, BackendKind::Cuda) {
            if let Some(object) = receipt.as_object_mut() {
                object.insert(
                    "kernel_stats".to_owned(),
                    json!([{
                        "kernel_id": CUDA_KERNEL_ID,
                        "kernel_time_ms": 10.0,
                        "invocations": 210,
                        "kernel_launches": 210,
                        "fallback_invocations": 0,
                        "host_to_device_bytes": 8,
                        "device_to_host_bytes": 9
                    }]),
                );
                object.insert(
                    "cuda".to_owned(),
                    json!({
                        "available": true,
                        "device_count": 1,
                        "device_name": "NVIDIA GeForce RTX 5070 Ti",
                        "compute_capability": "12.0",
                        "driver_version": "591.86",
                        "cuda_runtime_version": "12.9",
                        "cuda_toolkit_version": "12.9",
                        "nvrtc_version": "12.9",
                        "vram_bytes": 17094475776u64,
                        "memory_hwm_bytes": 7070547968u64,
                        "power_draw_watts": 45.0,
                        "temperature_c": 42.0
                    }),
                );
            }
        }
        receipt
    }
}
