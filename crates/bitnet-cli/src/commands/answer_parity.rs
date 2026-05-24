//! Answer-corpus parity comparator.

use anyhow::{Context, Result};
use clap::Args;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

/// Compare answer-corpus receipts.
#[derive(Args, Debug)]
pub struct AnswerParityCommand {
    /// Scalar strict CPU answer-corpus receipt.
    #[arg(long, value_name = "PATH")]
    pub scalar: Option<PathBuf>,

    /// AVX2 strict CPU answer-corpus receipt.
    #[arg(long, value_name = "PATH")]
    pub avx2: Option<PathBuf>,

    /// Left/baseline answer-corpus receipt for generic backend comparison.
    #[arg(long, value_name = "PATH")]
    pub left: Option<PathBuf>,

    /// Right/variant answer-corpus receipt for generic backend comparison.
    #[arg(long, value_name = "PATH")]
    pub right: Option<PathBuf>,

    /// Label to use for the left/baseline receipt in generic comparison output.
    #[arg(long, value_name = "LABEL")]
    pub left_label: Option<String>,

    /// Label to use for the right/variant receipt in generic comparison output.
    #[arg(long, value_name = "LABEL")]
    pub right_label: Option<String>,

    /// Output parity receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "target/bitnet/receipts/cpu-answer-parity.json"
    )]
    pub json_out: PathBuf,

    /// Machine identifier for hardware-scoped parity artifacts.
    #[arg(long, value_name = "ID")]
    pub machine: Option<String>,

    /// Same-machine platform probe artifact to cross-link for CPU topology and power context.
    #[arg(long, value_name = "PATH")]
    pub platform_artifact: Option<PathBuf>,
}

impl AnswerParityCommand {
    /// Execute the offline answer parity comparison.
    pub async fn execute(&self) -> Result<()> {
        let inputs = self.resolve_inputs()?;
        let left = read_json(inputs.left_path)?;
        let right = read_json(inputs.right_path)?;
        let left_label = inputs
            .left_label
            .map(str::to_string)
            .unwrap_or_else(|| infer_lane_label(&left, inputs.left_path, "left"));
        let right_label = inputs
            .right_label
            .map(str::to_string)
            .unwrap_or_else(|| infer_lane_label(&right, inputs.right_path, "right"));
        let platform = match &self.platform_artifact {
            Some(path) => Some(read_json(path)?),
            None => None,
        };
        let receipt = build_answer_parity_receipt(
            inputs.left_path,
            &left,
            inputs.right_path,
            &right,
            &left_label,
            &right_label,
            inputs.legacy_scalar_avx2,
            self.machine.as_deref(),
            self.platform_artifact.as_deref(),
            platform.as_ref(),
        );
        let failed = receipt["summary"]["failed"].as_u64().unwrap_or(1);

        if let Some(parent) = self.json_out.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.json_out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("answer parity receipt written to {}", self.json_out.display());

        if failed > 0 {
            anyhow::bail!(
                "{}-vs-{} answer parity failed for {failed} case(s); receipt written to {}",
                left_label,
                right_label,
                self.json_out.display()
            );
        }
        Ok(())
    }

    fn resolve_inputs(&self) -> Result<ParityInputs<'_>> {
        let generic_requested = self.left.is_some()
            || self.right.is_some()
            || self.left_label.is_some()
            || self.right_label.is_some();
        if generic_requested {
            if self.scalar.is_some() || self.avx2.is_some() {
                anyhow::bail!(
                    "use either --left/--right for generic parity or --scalar/--avx2 for legacy CPU parity"
                );
            }
            let left_path =
                self.left.as_deref().context("--left is required for generic answer parity")?;
            let right_path =
                self.right.as_deref().context("--right is required for generic answer parity")?;
            return Ok(ParityInputs {
                left_path,
                right_path,
                left_label: self.left_label.as_deref(),
                right_label: self.right_label.as_deref(),
                legacy_scalar_avx2: false,
            });
        }

        let left_path = self
            .scalar
            .as_deref()
            .context("--scalar and --avx2 are required for legacy CPU answer parity")?;
        let right_path = self
            .avx2
            .as_deref()
            .context("--scalar and --avx2 are required for legacy CPU answer parity")?;
        Ok(ParityInputs {
            left_path,
            right_path,
            left_label: Some("scalar"),
            right_label: Some("avx2"),
            legacy_scalar_avx2: true,
        })
    }
}

struct ParityInputs<'a> {
    left_path: &'a Path,
    right_path: &'a Path,
    left_label: Option<&'a str>,
    right_label: Option<&'a str>,
    legacy_scalar_avx2: bool,
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

#[allow(clippy::too_many_arguments)]
fn build_answer_parity_receipt(
    left_path: &Path,
    left: &Value,
    right_path: &Path,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    machine: Option<&str>,
    platform_artifact: Option<&Path>,
    platform: Option<&Value>,
) -> Value {
    let mut shared_failures = Vec::new();
    compare_top_level_contract(left, right, legacy_scalar_avx2, &mut shared_failures);

    let left_cases = cases_by_id(left);
    let right_cases = cases_by_id(right);
    let case_ids = left_cases.keys().chain(right_cases.keys()).cloned().collect::<BTreeSet<_>>();

    let mut first_divergence = None;
    let cases = case_ids
        .iter()
        .map(|id| {
            compare_case(
                id,
                left_cases.get(id).copied(),
                right_cases.get(id).copied(),
                left_label,
                right_label,
                legacy_scalar_avx2,
                &mut first_divergence,
            )
        })
        .collect::<Vec<_>>();
    let logits_topk_frontier =
        build_logits_topk_frontier(&case_ids, &left_cases, &right_cases, left_label, right_label);
    let generated_output_frontier = build_generated_output_frontier(
        &case_ids,
        &left_cases,
        &right_cases,
        left_label,
        right_label,
    );
    let generated_output_logit_margin_frontier = build_generated_output_logit_margin_frontier(
        &case_ids,
        &left_cases,
        &right_cases,
        left_label,
        right_label,
    );
    let generated_output_argmax_source_frontier = build_generated_output_argmax_source_frontier(
        &case_ids,
        &left_cases,
        &right_cases,
        left_label,
        right_label,
    );
    let generated_output_internal_logit_source_frontier =
        build_generated_output_internal_logit_source_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_hidden_state_source_frontier =
        build_generated_output_hidden_state_source_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_model_forward_source_frontier =
        build_generated_output_model_forward_source_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_final_block_source_frontier =
        build_generated_output_final_block_source_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_penultimate_block_source_frontier =
        build_generated_output_penultimate_block_source_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_antepenultimate_block_source_frontier =
        build_generated_output_antepenultimate_block_source_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_pre_antepenultimate_block_source_frontier =
        build_generated_output_pre_antepenultimate_block_source_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_earlier_block_source_frontier =
        build_generated_output_earlier_block_source_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_transformer_block_source_stack_frontier =
        build_generated_output_transformer_block_source_stack_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_attention_output_source_frontier =
        build_generated_output_attention_output_source_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_qkv_projection_source_frontier =
        build_generated_output_qkv_projection_source_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_qkv_projection_dispatch_replay_frontier =
        build_generated_output_qkv_projection_dispatch_replay_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );
    let generated_output_qk256_numeric_policy_frontier =
        build_generated_output_qk256_numeric_policy_frontier(
            &case_ids,
            &left_cases,
            &right_cases,
            left_label,
            right_label,
        );

    let passed = cases.iter().filter(|case| case["passed"] == true).count();
    let failed = cases.len().saturating_sub(passed) + usize::from(!shared_failures.is_empty());
    let backend = parity_backend_summary(left, right, left_label, right_label, legacy_scalar_avx2);
    let kernel = parity_kernel_summary(left, right, left_label, right_label, legacy_scalar_avx2);

    let inputs = if legacy_scalar_avx2 {
        json!({
            "scalar_receipt_path": left_path.display().to_string(),
            "avx2_receipt_path": right_path.display().to_string(),
        })
    } else {
        json!({
            "left_receipt_path": left_path.display().to_string(),
            "right_receipt_path": right_path.display().to_string(),
            "left_label": left_label,
            "right_label": right_label,
        })
    };
    let may_claim = if legacy_scalar_avx2 {
        json!([
            "Scalar versus AVX2 full-decode answer parity can be audited for the compared receipts.",
            "First divergence evidence can separate AVX2 kernel issues from shared prompt, tokenizer, logits, sampler, or text decoding issues."
        ])
    } else {
        json!([
            "Full-decode answer parity can be audited for the compared receipts.",
            "First divergence evidence can separate backend issues from shared prompt, tokenizer, logits, sampler, or text decoding issues."
        ])
    };
    let must_not_claim = if legacy_scalar_avx2 {
        json!([
            "General chat quality is proven.",
            "Sustained CPU throughput is proven.",
            "Server inference is complete.",
            "GPU or NPU execution is involved."
        ])
    } else {
        json!([
            "General chat quality is proven.",
            "Sustained throughput is proven.",
            "Server inference is complete.",
            "A backend is answer-ready when the compared model artifact is not answer-ready."
        ])
    };

    json!({
        "schema_version": "1.0.0",
        "artifact_kind": if legacy_scalar_avx2 {
            "bitnet_cpu_answer_parity"
        } else {
            "bitnet_answer_corpus_parity"
        },
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "proof_stage": "full_decode_parity_compared",
        "requested_backend": backend["requested_backend"],
        "selected_backend": backend["selected_backend"],
        "runtime_api": backend["runtime_api"],
        "fallback_used": backend["fallback_used"],
        "backend": backend,
        "kernel": kernel,
        "claim": if legacy_scalar_avx2 {
            "scalar_avx2_full_decode_answer_parity"
        } else {
            "full_decode_answer_parity_diagnostic"
        },
        "speedup_claim": false,
        "inputs": inputs,
        "machine": {
            "machine_id": machine,
            "platform_artifact": platform_artifact.map(|path| path.display().to_string()),
            "cpu": platform.and_then(|value| value.get("cpu")).cloned().unwrap_or(Value::Null),
            "memory": platform.and_then(|value| value.get("memory")).cloned().unwrap_or(Value::Null),
            "power": platform.and_then(|value| value.get("power")).cloned().unwrap_or(Value::Null),
        },
        "shared_contract": {
            "same_real_gguf": shared_failures.iter().all(|failure| *failure != "model_contract"),
            "same_tokenizer": shared_failures.iter().all(|failure| *failure != "tokenizer_contract"),
            "same_prompt_template": shared_failures.iter().all(|failure| *failure != "prompt_template"),
            "same_greedy_settings": shared_failures.iter().all(|failure| *failure != "generation_contract"),
            "failed_rules": shared_failures,
        },
        "summary": {
            "total": cases.len(),
            "passed": passed,
            "failed": failed,
            "first_divergence": first_divergence,
        },
        "logits_topk_frontier": logits_topk_frontier,
        "generated_output_frontier": generated_output_frontier,
        "generated_output_logit_margin_frontier": generated_output_logit_margin_frontier,
        "generated_output_argmax_source_frontier": generated_output_argmax_source_frontier,
        "generated_output_internal_logit_source_frontier": generated_output_internal_logit_source_frontier,
        "generated_output_hidden_state_source_frontier": generated_output_hidden_state_source_frontier,
        "generated_output_model_forward_source_frontier": generated_output_model_forward_source_frontier,
        "generated_output_final_block_source_frontier": generated_output_final_block_source_frontier,
        "generated_output_penultimate_block_source_frontier": generated_output_penultimate_block_source_frontier,
        "generated_output_antepenultimate_block_source_frontier": generated_output_antepenultimate_block_source_frontier,
        "generated_output_pre_antepenultimate_block_source_frontier": generated_output_pre_antepenultimate_block_source_frontier,
        "generated_output_earlier_block_source_frontier": generated_output_earlier_block_source_frontier,
        "generated_output_transformer_block_source_stack_frontier": generated_output_transformer_block_source_stack_frontier,
        "generated_output_attention_output_source_frontier": generated_output_attention_output_source_frontier,
        "generated_output_qkv_projection_source_frontier": generated_output_qkv_projection_source_frontier,
        "generated_output_qkv_projection_dispatch_replay_frontier": generated_output_qkv_projection_dispatch_replay_frontier,
        "generated_output_qk256_numeric_policy_frontier": generated_output_qk256_numeric_policy_frontier,
        "cases": cases,
        "may_claim": may_claim,
        "must_not_claim": must_not_claim,
    })
}

fn parity_backend_summary(
    left: &Value,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
) -> Value {
    let left_backend = receipt_backend_identity(left);
    let right_backend = receipt_backend_identity(right);
    let fallback_used = left_backend["fallback_used"].as_bool().unwrap_or(true)
        || right_backend["fallback_used"].as_bool().unwrap_or(true);
    let lane_labels = if legacy_scalar_avx2 {
        json!({
            "scalar": left_backend,
            "avx2": right_backend,
        })
    } else {
        json!({
            "left": {
                "label": left_label,
                "backend": left_backend,
            },
            "right": {
                "label": right_label,
                "backend": right_backend,
            },
        })
    };

    json!({
        "requested_backend": common_or_mixed(
            left_backend["requested_backend"].as_str(),
            right_backend["requested_backend"].as_str(),
        ),
        "selected_backend": common_or_mixed(
            left_backend["selected_backend"].as_str(),
            right_backend["selected_backend"].as_str(),
        ),
        "runtime_api": common_or_mixed(
            left_backend["runtime_api"].as_str(),
            right_backend["runtime_api"].as_str(),
        ),
        "fallback_used": fallback_used,
        "lanes": lane_labels,
    })
}

fn receipt_backend_identity(receipt: &Value) -> Value {
    json!({
        "requested_backend": backend_str(receipt, "requested_backend").unwrap_or("unknown"),
        "selected_backend": backend_str(receipt, "selected_backend").unwrap_or("unknown"),
        "runtime_api": backend_str(receipt, "runtime_api").unwrap_or("unknown"),
        "fallback_used": receipt_fallback_used(receipt).unwrap_or(true),
    })
}

fn backend_str<'a>(receipt: &'a Value, field: &str) -> Option<&'a str> {
    first_case_backend_field(receipt, field)
        .and_then(Value::as_str)
        .or_else(|| receipt["backend"][field].as_str())
        .or_else(|| receipt[field].as_str())
}

fn first_case_backend_field<'a>(receipt: &'a Value, field: &str) -> Option<&'a Value> {
    receipt["cases"].as_array()?.iter().find_map(|case| case["backend"].get(field))
}

fn receipt_fallback_used(receipt: &Value) -> Option<bool> {
    let mut saw_evidence = false;
    let mut fallback_used = false;
    for candidate in [
        receipt.get("fallback_used"),
        receipt.get("backend").and_then(|backend| backend.get("fallback_used")),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(value) = candidate.as_bool() {
            saw_evidence = true;
            fallback_used |= value;
        }
    }
    for case in receipt["cases"].as_array().into_iter().flatten() {
        if let Some(value) = case["backend"]["fallback_used"].as_bool() {
            saw_evidence = true;
            fallback_used |= value;
        }
    }
    saw_evidence.then_some(fallback_used)
}

fn common_or_mixed(left: Option<&str>, right: Option<&str>) -> Value {
    match (left, right) {
        (Some(left), Some(right)) if left == right && !left.is_empty() => json!(left),
        (None, None) => json!("unknown"),
        _ => json!("mixed"),
    }
}

fn parity_kernel_summary(
    left: &Value,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
) -> Value {
    let left_kernels = selected_kernels(left);
    let right_kernels = selected_kernels(right);
    if legacy_scalar_avx2 {
        json!({
            "scalar_selected_kernels": left_kernels,
            "avx2_selected_kernels": right_kernels,
        })
    } else {
        json!({
            "left_label": left_label,
            "right_label": right_label,
            "left_selected_kernels": left_kernels,
            "right_selected_kernels": right_kernels,
        })
    }
}

fn selected_kernels(receipt: &Value) -> Vec<String> {
    receipt["cases"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|case| case["kernel"]["selected_kernel"].as_str())
        .filter(|kernel| !kernel.is_empty())
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn compare_top_level_contract(
    left: &Value,
    right: &Value,
    legacy_scalar_avx2: bool,
    failures: &mut Vec<&'static str>,
) {
    if !answer_corpus_artifact_kind_allowed(left["artifact_kind"].as_str())
        || !answer_corpus_artifact_kind_allowed(right["artifact_kind"].as_str())
        || (legacy_scalar_avx2
            && (left["artifact_kind"] != "bitnet_cpu_answer_corpus"
                || right["artifact_kind"] != "bitnet_cpu_answer_corpus"))
    {
        failures.push("artifact_kind_contract");
    }

    let path_mismatch = match (left["model"]["path"].as_str(), right["model"]["path"].as_str()) {
        (Some(left_path), Some(right_path)) if !left_path.is_empty() && !right_path.is_empty() => {
            left_path != right_path
        }
        _ => false,
    };
    if left["model"]["loader_mode"] != "real_gguf"
        || right["model"]["loader_mode"] != "real_gguf"
        || left["model"]["repo"] != right["model"]["repo"]
        || left["model"]["file"] != right["model"]["file"]
        || path_mismatch
    {
        failures.push("model_contract");
    }

    if left["model"]["tokenizer_path"] != right["model"]["tokenizer_path"] {
        failures.push("tokenizer_contract");
    }

    if left["prompt_template"] != right["prompt_template"] {
        failures.push("prompt_template");
    }

    for field in [
        "/generation/mode",
        "/generation/temperature",
        "/generation/deterministic",
        "/generation/strict_loader",
        "/generation/default_max_new_tokens",
        "/generation/logits_dump_steps",
        "/generation/logits_topk",
    ] {
        if left.pointer(field) != right.pointer(field) {
            failures.push("generation_contract");
            break;
        }
    }

    if legacy_scalar_avx2 && (!strict_cpu_backend(left) || !strict_cpu_backend(right)) {
        failures.push("strict_cpu_backend");
    }

    if !legacy_scalar_avx2
        && (!generic_backend_contract(left["backend"].as_object())
            || !generic_backend_contract(right["backend"].as_object()))
    {
        failures.push("backend_contract");
    }
}

fn answer_corpus_artifact_kind_allowed(kind: Option<&str>) -> bool {
    matches!(
        kind,
        Some(
            "bitnet_cpu_answer_corpus"
                | "bitnet_a770_opencl_answer_diagnostic_corpus"
                | "bitnet_cuda_answer_corpus"
                | "bitnet_cuda_answer_diagnostic_corpus"
                | "bitnet_apple_m4_local_answer_corpus"
        )
    )
}

fn strict_cpu_backend(receipt: &Value) -> bool {
    receipt["backend"]["requested_backend"] == "cpu"
        && receipt["backend"]["selected_backend"] == "cpu"
        && receipt["backend"]["runtime_api"] == "cpu"
        && receipt["backend"]["fallback_used"] == false
}

fn generic_backend_contract(backend: Option<&serde_json::Map<String, Value>>) -> bool {
    let Some(backend) = backend else {
        return false;
    };
    !backend.get("requested_backend").and_then(Value::as_str).unwrap_or_default().is_empty()
        && !backend.get("selected_backend").and_then(Value::as_str).unwrap_or_default().is_empty()
        && !backend.get("runtime_api").and_then(Value::as_str).unwrap_or_default().is_empty()
        && backend.get("fallback_used").and_then(Value::as_bool) == Some(false)
}

fn infer_lane_label(receipt: &Value, path: &Path, fallback: &str) -> String {
    receipt["generation"]["requested_cpu_kernel"]
        .as_str()
        .or_else(|| receipt["backend"]["requested_backend"].as_str())
        .or_else(|| receipt["backend"]["selected_backend"].as_str())
        .map(str::to_string)
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .filter(|stem| !stem.is_empty())
                .unwrap_or(fallback)
                .to_string()
        })
}

fn cases_by_id(receipt: &Value) -> BTreeMap<String, &Value> {
    receipt["cases"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|case| case["id"].as_str().map(|id| (id.to_string(), case)))
        .collect()
}

fn compare_case(
    id: &str,
    left: Option<&Value>,
    right: Option<&Value>,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    first_divergence: &mut Option<Value>,
) -> Value {
    let missing_left_rule =
        if legacy_scalar_avx2 { "case_missing_in_scalar" } else { "case_missing_in_left" };
    let missing_right_rule =
        if legacy_scalar_avx2 { "case_missing_in_avx2" } else { "case_missing_in_right" };
    let Some(left) = left else {
        set_first(
            first_divergence,
            id,
            missing_left_rule,
            None,
            Value::Null,
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return failed_case(id, &[missing_left_rule]);
    };
    let Some(right) = right else {
        set_first(
            first_divergence,
            id,
            missing_right_rule,
            None,
            Value::Null,
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return failed_case(id, &[missing_right_rule]);
    };

    let mut failures = Vec::new();
    if !case_has_execution_evidence(left) || !case_has_execution_evidence(right) {
        set_first(
            first_divergence,
            id,
            "execution_evidence_recorded",
            None,
            case_status_summary(left),
            case_status_summary(right),
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        check_equal(
            id,
            "question",
            None,
            &left["question"],
            &right["question"],
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
        check_equal(
            id,
            "status",
            None,
            &left["status"],
            &right["status"],
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
        check_equal(
            id,
            "quality_failed_rules",
            None,
            &left["quality"]["failed_rules"],
            &right["quality"]["failed_rules"],
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
        failures.push("execution_evidence_recorded");
        return case_comparison_row(
            id,
            failures,
            left,
            right,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
    }

    check_case_contract(
        id,
        left_label,
        left,
        legacy_scalar_avx2,
        true,
        left_label,
        right_label,
        &mut failures,
        first_divergence,
    );
    check_case_contract(
        id,
        right_label,
        right,
        legacy_scalar_avx2,
        false,
        left_label,
        right_label,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "question",
        None,
        &left["question"],
        &right["question"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "prompt_template",
        None,
        &left["prompt_template"],
        &right["prompt_template"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "tokenizer_source",
        None,
        &left["tokenizer"]["source"],
        &right["tokenizer"]["source"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "tokenizer_strict",
        None,
        &left["tokenizer"]["strict"],
        &right["tokenizer"]["strict"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "prompt_token_ids",
        None,
        &left["token_ids"]["prompt"],
        &right["token_ids"]["prompt"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "generated_token_ids",
        None,
        &left["token_ids"]["generated"],
        &right["token_ids"]["generated"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    check_equal(
        id,
        "decoded_text",
        None,
        &left["answer"],
        &right["answer"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );
    if legacy_scalar_avx2 {
        check_kernel_lane(
            id,
            "scalar",
            left,
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
        check_kernel_lane(
            id,
            "avx2",
            right,
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
    } else {
        check_generic_kernel_recorded(
            id,
            "left",
            left,
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
        check_generic_kernel_recorded(
            id,
            "right",
            right,
            left_label,
            right_label,
            legacy_scalar_avx2,
            &mut failures,
            first_divergence,
        );
    }
    compare_logits_dump(
        id,
        &left["logits_dump"],
        &right["logits_dump"],
        left_label,
        right_label,
        legacy_scalar_avx2,
        &mut failures,
        first_divergence,
    );

    case_comparison_row(id, failures, left, right, left_label, right_label, legacy_scalar_avx2)
}

fn failed_case(id: &str, failures: &[&str]) -> Value {
    json!({
        "id": id,
        "passed": false,
        "failed_rules": failures,
    })
}

fn case_has_execution_evidence(case: &Value) -> bool {
    case["backend"].is_object()
        && case["kernel"]["selected_kernel"].as_str().is_some_and(|kernel| !kernel.is_empty())
        && case["token_ids"]["prompt"].is_array()
        && case["token_ids"]["generated"].is_array()
}

fn case_comparison_row(
    id: &str,
    failures: Vec<&'static str>,
    left: &Value,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
) -> Value {
    if legacy_scalar_avx2 {
        json!({
            "id": id,
            "passed": failures.is_empty(),
            "failed_rules": failures,
            "scalar": case_summary(left),
            "avx2": case_summary(right),
        })
    } else {
        json!({
            "id": id,
            "passed": failures.is_empty(),
            "failed_rules": failures,
            "left": labeled_case_summary(left, left_label),
            "right": labeled_case_summary(right, right_label),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn check_case_contract(
    id: &str,
    lane: &str,
    case: &Value,
    legacy_scalar_avx2: bool,
    is_left: bool,
    left_label: &str,
    right_label: &str,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let backend_rule = match (legacy_scalar_avx2, is_left) {
        (true, true) => "scalar_strict_cpu_backend",
        (true, false) => "avx2_strict_cpu_backend",
        (false, true) => "left_backend_contract",
        (false, false) => "right_backend_contract",
    };
    let backend_ok = if legacy_scalar_avx2 {
        case["backend"]["requested_backend"] == "cpu"
            && matches!(case["backend"]["selected_backend"].as_str(), Some("cpu" | "cpu-rust"))
            && case["backend"]["runtime_api"] == "cpu"
            && case["backend"]["fallback_used"] == false
    } else {
        generic_backend_contract(case["backend"].as_object())
    };
    if !backend_ok {
        failures.push(backend_rule);
        set_first(
            first_divergence,
            id,
            backend_rule,
            None,
            case["backend"].clone(),
            json!(lane),
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn check_equal(
    id: &str,
    rule: &'static str,
    step: Option<usize>,
    left: &Value,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    if left != right {
        failures.push(rule);
        set_first(
            first_divergence,
            id,
            rule,
            step,
            left.clone(),
            right.clone(),
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
    }
}

fn check_kernel_lane(
    id: &str,
    lane: &'static str,
    case: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let selected = case["kernel"]["selected_kernel"].as_str().unwrap_or_default();
    if !selected.to_ascii_lowercase().contains(lane) {
        let rule = if lane == "scalar" { "scalar_kernel_identity" } else { "avx2_kernel_identity" };
        failures.push(rule);
        set_first(
            first_divergence,
            id,
            rule,
            None,
            json!(selected),
            json!(lane),
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
    }
}

fn check_generic_kernel_recorded(
    id: &str,
    lane: &'static str,
    case: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let selected = case["kernel"]["selected_kernel"].as_str().unwrap_or_default();
    if selected.is_empty() || selected.contains("mock") || selected.contains("diagnostic") {
        let rule = if lane == "left" { "left_kernel_recorded" } else { "right_kernel_recorded" };
        failures.push(rule);
        set_first(
            first_divergence,
            id,
            rule,
            None,
            json!(selected),
            json!(lane),
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
    }
}

fn compare_logits_dump(
    id: &str,
    left: &Value,
    right: &Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
    failures: &mut Vec<&'static str>,
    first_divergence: &mut Option<Value>,
) {
    let left_logits_rule = if legacy_scalar_avx2 {
        "scalar_logits_dump_recorded"
    } else {
        "left_logits_dump_recorded"
    };
    let right_logits_rule =
        if legacy_scalar_avx2 { "avx2_logits_dump_recorded" } else { "right_logits_dump_recorded" };
    let Some(left_steps) = left.as_array() else {
        failures.push(left_logits_rule);
        set_first(
            first_divergence,
            id,
            left_logits_rule,
            None,
            left.clone(),
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return;
    };
    let Some(right_steps) = right.as_array() else {
        failures.push(right_logits_rule);
        set_first(
            first_divergence,
            id,
            right_logits_rule,
            None,
            right.clone(),
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return;
    };
    if left_steps.is_empty() {
        failures.push(left_logits_rule);
        set_first(
            first_divergence,
            id,
            left_logits_rule,
            None,
            left.clone(),
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return;
    }
    if right_steps.is_empty() {
        failures.push(right_logits_rule);
        set_first(
            first_divergence,
            id,
            right_logits_rule,
            None,
            right.clone(),
            Value::Null,
            left_label,
            right_label,
            legacy_scalar_avx2,
        );
        return;
    }
    check_equal(
        id,
        "logits_step_count",
        None,
        &json!(left_steps.len()),
        &json!(right_steps.len()),
        left_label,
        right_label,
        legacy_scalar_avx2,
        failures,
        first_divergence,
    );
    for (step, (left_step, right_step)) in left_steps.iter().zip(right_steps).enumerate() {
        check_equal(
            id,
            "logits_topk",
            Some(step),
            left_step,
            right_step,
            left_label,
            right_label,
            legacy_scalar_avx2,
            failures,
            first_divergence,
        );
    }
}

fn build_logits_topk_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut compared_step_count = 0usize;
    let mut mismatch_count = 0usize;
    let mut same_generated_output_count = 0usize;
    let mut generated_output_divergence_count = 0usize;
    let mut same_chosen_token_count = 0usize;
    let mut different_chosen_token_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut max_common_token_abs_delta = 0.0f64;

    for id in case_ids {
        let Some(left_case) = left_cases.get(id).copied() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                json!({
                    "case_id": id,
                    "classification": "logits_topk_missing_context",
                    "reason": "left_case_missing",
                }),
            );
            continue;
        };
        let Some(right_case) = right_cases.get(id).copied() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                json!({
                    "case_id": id,
                    "classification": "logits_topk_missing_context",
                    "reason": "right_case_missing",
                }),
            );
            continue;
        };
        let Some(left_steps) = left_case["logits_dump"].as_array() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                logits_missing_context_row(id, None, "left_logits_dump_missing"),
            );
            continue;
        };
        let Some(right_steps) = right_case["logits_dump"].as_array() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                logits_missing_context_row(id, None, "right_logits_dump_missing"),
            );
            continue;
        };
        if left_steps.is_empty() || right_steps.is_empty() {
            missing_context_count += 1;
            let reason = if left_steps.is_empty() {
                "left_logits_dump_empty"
            } else {
                "right_logits_dump_empty"
            };
            push_limited_row(&mut rows, ROW_LIMIT, logits_missing_context_row(id, None, reason));
            continue;
        }

        for (step, (left_step, right_step)) in left_steps.iter().zip(right_steps).enumerate() {
            compared_step_count += 1;
            if left_step == right_step {
                continue;
            }
            mismatch_count += 1;
            let row =
                logits_topk_frontier_row(id, step, left_case, right_case, left_step, right_step);
            if row["generated_token_ids_match"].as_bool().unwrap_or(false)
                && row["decoded_text_match"].as_bool().unwrap_or(false)
            {
                same_generated_output_count += 1;
            } else {
                generated_output_divergence_count += 1;
            }
            if row["same_chosen_id"].as_bool().unwrap_or(false) {
                same_chosen_token_count += 1;
            } else {
                different_chosen_token_count += 1;
            }
            max_common_token_abs_delta = max_common_token_abs_delta
                .max(row["max_common_token_abs_delta"].as_f64().unwrap_or(0.0));
            push_limited_row(&mut rows, ROW_LIMIT, row);
        }

        if left_steps.len() != right_steps.len() {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                json!({
                    "case_id": id,
                    "classification": "logits_topk_missing_context",
                    "reason": "logits_step_count_mismatch",
                    "left_step_count": left_steps.len(),
                    "right_step_count": right_steps.len(),
                }),
            );
        }
    }

    let classification = if missing_context_count > 0 {
        "logits_topk_frontier_missing_context"
    } else if generated_output_divergence_count > 0 {
        "logits_topk_frontier_generated_output_divergence"
    } else if different_chosen_token_count > 0 {
        "logits_topk_frontier_different_chosen_same_output"
    } else if mismatch_count > 0 {
        "logits_topk_frontier_same_chosen_same_output"
    } else {
        "logits_topk_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "compared_step_count": compared_step_count,
        "logits_topk_mismatch_count": mismatch_count,
        "same_generated_output_count": same_generated_output_count,
        "generated_output_divergence_count": generated_output_divergence_count,
        "same_chosen_token_count": same_chosen_token_count,
        "different_chosen_token_count": different_chosen_token_count,
        "missing_context_count": missing_context_count,
        "max_common_token_abs_delta": max_common_token_abs_delta,
        "rows_truncated": mismatch_count.saturating_add(missing_context_count) > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn push_limited_row(rows: &mut Vec<Value>, row_limit: usize, row: Value) {
    if rows.len() < row_limit {
        rows.push(row);
    }
}

fn logits_missing_context_row(id: &str, step: Option<usize>, reason: &str) -> Value {
    json!({
        "case_id": id,
        "step": step,
        "classification": "logits_topk_missing_context",
        "reason": reason,
    })
}

fn logits_topk_frontier_row(
    id: &str,
    step: usize,
    left_case: &Value,
    right_case: &Value,
    left_step: &Value,
    right_step: &Value,
) -> Value {
    let left_chosen_id = left_step["chosen_id"].as_u64();
    let right_chosen_id = right_step["chosen_id"].as_u64();
    let generated_token_ids_match =
        left_case["token_ids"]["generated"] == right_case["token_ids"]["generated"];
    let decoded_text_match = left_case["answer"] == right_case["answer"];
    let same_chosen_id = left_chosen_id == right_chosen_id;
    let (common_top_token_count, max_common_delta, left_only, right_only) =
        top_logits_common_delta(&left_step["top_logits"], &right_step["top_logits"]);

    let classification = if !generated_token_ids_match || !decoded_text_match {
        "logits_topk_generated_output_divergence"
    } else if same_chosen_id {
        "logits_topk_same_chosen_same_output"
    } else {
        "logits_topk_different_chosen_same_output"
    };

    json!({
        "case_id": id,
        "step": step,
        "classification": classification,
        "left_chosen_id": left_chosen_id,
        "right_chosen_id": right_chosen_id,
        "same_chosen_id": same_chosen_id,
        "generated_token_ids_match": generated_token_ids_match,
        "decoded_text_match": decoded_text_match,
        "first_different_rank": first_different_topk_rank(&left_step["top_logits"], &right_step["top_logits"]),
        "left_topk_count": left_step["top_logits"].as_array().map(Vec::len),
        "right_topk_count": right_step["top_logits"].as_array().map(Vec::len),
        "common_top_token_count": common_top_token_count,
        "max_common_token_abs_delta": max_common_delta,
        "left_only_top_token_ids": left_only,
        "right_only_top_token_ids": right_only,
    })
}

fn top_logits_common_delta(left: &Value, right: &Value) -> (usize, f64, Vec<u64>, Vec<u64>) {
    let left_logits = top_logits_by_token(left);
    let right_logits = top_logits_by_token(right);
    let mut common_count = 0usize;
    let mut max_delta = 0.0f64;
    for (token, left_logit) in &left_logits {
        if let Some(right_logit) = right_logits.get(token) {
            common_count += 1;
            max_delta = max_delta.max((left_logit - right_logit).abs());
        }
    }
    let left_only = left_logits
        .keys()
        .filter(|token| !right_logits.contains_key(token))
        .take(3)
        .copied()
        .collect();
    let right_only = right_logits
        .keys()
        .filter(|token| !left_logits.contains_key(token))
        .take(3)
        .copied()
        .collect();
    (common_count, max_delta, left_only, right_only)
}

fn top_logits_by_token(top_logits: &Value) -> BTreeMap<u64, f64> {
    top_logits
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| Some((entry["token_id"].as_u64()?, entry["logit"].as_f64()?)))
        .collect()
}

fn first_different_topk_rank(left: &Value, right: &Value) -> Value {
    let left = left.as_array().map(Vec::as_slice).unwrap_or(&[]);
    let right = right.as_array().map(Vec::as_slice).unwrap_or(&[]);
    for index in 0..left.len().max(right.len()) {
        let left_token = left.get(index).and_then(|entry| entry["token_id"].as_u64());
        let right_token = right.get(index).and_then(|entry| entry["token_id"].as_u64());
        if left_token != right_token {
            return json!(index);
        }
    }
    Value::Null
}

fn build_generated_output_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut generated_output_mismatch_count = 0usize;
    let mut decoded_text_mismatch_count = 0usize;
    let mut mismatch_with_logit_context_count = 0usize;
    let mut missing_logit_context_count = 0usize;
    let mut missing_context_count = 0usize;

    for id in case_ids {
        let Some(left_case) = left_cases.get(id).copied() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_missing_context_row(id, "left_case_missing"),
            );
            continue;
        };
        let Some(right_case) = right_cases.get(id).copied() else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_missing_context_row(id, "right_case_missing"),
            );
            continue;
        };

        let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_missing_context_row(id, "left_generated_token_ids_missing"),
            );
            continue;
        };
        let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_missing_context_row(id, "right_generated_token_ids_missing"),
            );
            continue;
        };

        let decoded_text_match = left_case["answer"] == right_case["answer"];
        if !decoded_text_match {
            decoded_text_mismatch_count += 1;
        }

        let Some(first_mismatch_index) =
            first_different_token_index(&left_generated, &right_generated)
        else {
            continue;
        };

        generated_output_mismatch_count += 1;
        let row = generated_output_frontier_row(id, left_case, right_case, first_mismatch_index);
        if row["has_logit_context_at_first_mismatch"].as_bool().unwrap_or(false) {
            mismatch_with_logit_context_count += 1;
        } else {
            missing_logit_context_count += 1;
        }
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_frontier_missing_context"
    } else if missing_logit_context_count > 0 {
        "generated_output_frontier_first_mismatch_missing_logit_context"
    } else if generated_output_mismatch_count > 0 {
        "generated_output_frontier_first_mismatch_has_logit_context"
    } else {
        "generated_output_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "generated_output_mismatch_count": generated_output_mismatch_count,
        "decoded_text_mismatch_count": decoded_text_mismatch_count,
        "mismatch_with_logit_context_count": mismatch_with_logit_context_count,
        "missing_logit_context_count": missing_logit_context_count,
        "missing_context_count": missing_context_count,
        "rows_truncated": generated_output_mismatch_count.saturating_add(missing_context_count) > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_missing_context_row(id: &str, reason: &str) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_missing_context",
        "reason": reason,
    })
}

fn generated_output_frontier_row(
    id: &str,
    left_case: &Value,
    right_case: &Value,
    first_mismatch_index: usize,
) -> Value {
    let left_generated = token_id_vec(&left_case["token_ids"]["generated"]).unwrap_or_default();
    let right_generated = token_id_vec(&right_case["token_ids"]["generated"]).unwrap_or_default();
    let left_steps = left_case["logits_dump"].as_array();
    let right_steps = right_case["logits_dump"].as_array();
    let left_step = left_steps.and_then(|steps| steps.get(first_mismatch_index));
    let right_step = right_steps.and_then(|steps| steps.get(first_mismatch_index));
    let has_logit_context_at_first_mismatch = left_step.is_some() && right_step.is_some();
    let left_chosen_id = left_step.and_then(|step| step["chosen_id"].as_u64());
    let right_chosen_id = right_step.and_then(|step| step["chosen_id"].as_u64());
    let same_chosen_id_at_first_mismatch = match (left_chosen_id, right_chosen_id) {
        (Some(left), Some(right)) => json!(left == right),
        _ => Value::Null,
    };
    let classification = if has_logit_context_at_first_mismatch {
        "generated_output_first_mismatch_has_logit_context"
    } else {
        "generated_output_first_mismatch_missing_logit_context"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_generated_len": left_generated.len(),
        "right_generated_len": right_generated.len(),
        "decoded_text_match": left_case["answer"] == right_case["answer"],
        "left_logits_step_count": left_steps.map(Vec::len),
        "right_logits_step_count": right_steps.map(Vec::len),
        "has_logit_context_at_first_mismatch": has_logit_context_at_first_mismatch,
        "left_chosen_id_at_first_mismatch": left_chosen_id,
        "right_chosen_id_at_first_mismatch": right_chosen_id,
        "same_chosen_id_at_first_mismatch": same_chosen_id_at_first_mismatch,
        "first_different_rank_at_first_mismatch": match (left_step, right_step) {
            (Some(left), Some(right)) => {
                first_different_topk_rank(&left["top_logits"], &right["top_logits"])
            }
            _ => Value::Null,
        },
    })
}

fn build_generated_output_logit_margin_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;
    const NEAR_TIE_THRESHOLD: f64 = 0.01;

    let mut rows = Vec::new();
    let mut generated_output_mismatch_count = 0usize;
    let mut margin_available_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut missing_cross_chosen_logit_count = 0usize;
    let mut opposite_argmax_count = 0usize;
    let mut right_near_tie_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let Some(left_case) = left_cases.get(id).copied() else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_logit_margin_missing_context_row(id, "left_case_missing"),
            );
            continue;
        };
        let Some(right_case) = right_cases.get(id).copied() else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_logit_margin_missing_context_row(id, "right_case_missing"),
            );
            continue;
        };

        let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_logit_margin_missing_context_row(
                    id,
                    "left_generated_token_ids_missing",
                ),
            );
            continue;
        };
        let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_logit_margin_missing_context_row(
                    id,
                    "right_generated_token_ids_missing",
                ),
            );
            continue;
        };
        let Some(first_mismatch_index) =
            first_different_token_index(&left_generated, &right_generated)
        else {
            continue;
        };

        generated_output_mismatch_count += 1;
        let row = generated_output_logit_margin_row(
            id,
            left_case,
            right_case,
            first_mismatch_index,
            NEAR_TIE_THRESHOLD,
        );
        match row["classification"].as_str() {
            Some("generated_output_logit_margin_first_mismatch_margin_available")
            | Some("generated_output_logit_margin_first_mismatch_opposite_argmax")
            | Some("generated_output_logit_margin_first_mismatch_opposite_argmax_right_near_tie") =>
            {
                margin_available_count += 1;
            }
            Some("generated_output_logit_margin_missing_cross_chosen_logit") => {
                missing_cross_chosen_logit_count += 1;
            }
            _ => {
                missing_context_count += 1;
            }
        }
        if row["opposite_argmax"].as_bool().unwrap_or(false) {
            opposite_argmax_count += 1;
        }
        if row["right_margin_near_tie"].as_bool().unwrap_or(false) {
            right_near_tie_count += 1;
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_logit_margin_frontier_missing_context"
    } else if missing_cross_chosen_logit_count > 0 {
        "generated_output_logit_margin_frontier_missing_cross_chosen_logit"
    } else if right_near_tie_count > 0 {
        "generated_output_logit_margin_frontier_opposite_argmax_right_near_tie"
    } else if opposite_argmax_count > 0 {
        "generated_output_logit_margin_frontier_opposite_argmax"
    } else if generated_output_mismatch_count > 0 {
        "generated_output_logit_margin_frontier_margin_available"
    } else {
        "generated_output_logit_margin_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "generated_output_mismatch_count": generated_output_mismatch_count,
        "margin_available_count": margin_available_count,
        "missing_cross_chosen_logit_count": missing_cross_chosen_logit_count,
        "missing_context_count": missing_context_count,
        "opposite_argmax_count": opposite_argmax_count,
        "right_near_tie_count": right_near_tie_count,
        "near_tie_abs_logit_delta_threshold": NEAR_TIE_THRESHOLD,
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_logit_margin_missing_context_row(id: &str, reason: &str) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_logit_margin_missing_context",
        "reason": reason,
    })
}

fn generated_output_logit_margin_row(
    id: &str,
    left_case: &Value,
    right_case: &Value,
    first_mismatch_index: usize,
    near_tie_threshold: f64,
) -> Value {
    let left_steps = left_case["logits_dump"].as_array();
    let right_steps = right_case["logits_dump"].as_array();
    let Some(left_step) = left_steps.and_then(|steps| steps.get(first_mismatch_index)) else {
        return json!({
            "case_id": id,
            "classification": "generated_output_logit_margin_missing_context",
            "reason": "left_logits_step_missing",
            "first_mismatch_index": first_mismatch_index,
            "left_logits_step_count": left_steps.map(Vec::len),
            "right_logits_step_count": right_steps.map(Vec::len),
        });
    };
    let Some(right_step) = right_steps.and_then(|steps| steps.get(first_mismatch_index)) else {
        return json!({
            "case_id": id,
            "classification": "generated_output_logit_margin_missing_context",
            "reason": "right_logits_step_missing",
            "first_mismatch_index": first_mismatch_index,
            "left_logits_step_count": left_steps.map(Vec::len),
            "right_logits_step_count": right_steps.map(Vec::len),
        });
    };
    let Some(left_chosen_id) = left_step["chosen_id"].as_u64() else {
        return json!({
            "case_id": id,
            "classification": "generated_output_logit_margin_missing_context",
            "reason": "left_chosen_id_missing",
            "first_mismatch_index": first_mismatch_index,
        });
    };
    let Some(right_chosen_id) = right_step["chosen_id"].as_u64() else {
        return json!({
            "case_id": id,
            "classification": "generated_output_logit_margin_missing_context",
            "reason": "right_chosen_id_missing",
            "first_mismatch_index": first_mismatch_index,
        });
    };

    let left_top_logits = top_logits_by_token(&left_step["top_logits"]);
    let right_top_logits = top_logits_by_token(&right_step["top_logits"]);
    let left_chosen_logit_on_left = left_top_logits.get(&left_chosen_id).copied();
    let right_chosen_logit_on_left = left_top_logits.get(&right_chosen_id).copied();
    let left_chosen_logit_on_right = right_top_logits.get(&left_chosen_id).copied();
    let right_chosen_logit_on_right = right_top_logits.get(&right_chosen_id).copied();

    let left_margin = match (left_chosen_logit_on_left, right_chosen_logit_on_left) {
        (Some(left_chosen), Some(right_chosen)) => Some(left_chosen - right_chosen),
        _ => None,
    };
    let right_margin = match (right_chosen_logit_on_right, left_chosen_logit_on_right) {
        (Some(right_chosen), Some(left_chosen)) => Some(right_chosen - left_chosen),
        _ => None,
    };
    let left_chosen_delta_across_lanes =
        match (left_chosen_logit_on_right, left_chosen_logit_on_left) {
            (Some(right_lane), Some(left_lane)) => Some(right_lane - left_lane),
            _ => None,
        };
    let right_chosen_delta_across_lanes =
        match (right_chosen_logit_on_right, right_chosen_logit_on_left) {
            (Some(right_lane), Some(left_lane)) => Some(right_lane - left_lane),
            _ => None,
        };

    let has_cross_chosen_logits = left_margin.is_some() && right_margin.is_some();
    let opposite_argmax = left_chosen_id != right_chosen_id
        && left_margin.is_some_and(|margin| margin > 0.0)
        && right_margin.is_some_and(|margin| margin > 0.0);
    let right_margin_near_tie =
        right_margin.is_some_and(|margin| margin.abs() <= near_tie_threshold);
    let classification = if !has_cross_chosen_logits {
        "generated_output_logit_margin_missing_cross_chosen_logit"
    } else if opposite_argmax && right_margin_near_tie {
        "generated_output_logit_margin_first_mismatch_opposite_argmax_right_near_tie"
    } else if opposite_argmax {
        "generated_output_logit_margin_first_mismatch_opposite_argmax"
    } else {
        "generated_output_logit_margin_first_mismatch_margin_available"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_chosen_id": left_chosen_id,
        "right_chosen_id": right_chosen_id,
        "same_chosen_id": left_chosen_id == right_chosen_id,
        "has_cross_chosen_logits": has_cross_chosen_logits,
        "opposite_argmax": opposite_argmax,
        "right_margin_near_tie": right_margin_near_tie,
        "near_tie_abs_logit_delta_threshold": near_tie_threshold,
        "left_chosen_logit_on_left": left_chosen_logit_on_left,
        "right_chosen_logit_on_left": right_chosen_logit_on_left,
        "left_margin_over_right_chosen_on_left": left_margin,
        "right_chosen_logit_on_right": right_chosen_logit_on_right,
        "left_chosen_logit_on_right": left_chosen_logit_on_right,
        "right_margin_over_left_chosen_on_right": right_margin,
        "left_chosen_delta_across_lanes": left_chosen_delta_across_lanes,
        "right_chosen_delta_across_lanes": right_chosen_delta_across_lanes,
        "left_topk_count": left_step["top_logits"].as_array().map(Vec::len),
        "right_topk_count": right_step["top_logits"].as_array().map(Vec::len),
    })
}

fn build_generated_output_argmax_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut clean_count = 0usize;
    let mut prompt_history_serialization_count = 0usize;
    let mut sampler_logit_extraction_count = 0usize;
    let mut trace_capture_context_loss_count = 0usize;
    let mut internal_logit_source_missing_context_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let Some(left_case) = left_cases.get(id).copied() else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_argmax_source_missing_context_row(id, "left_case_missing"),
            );
            continue;
        };
        let Some(right_case) = right_cases.get(id).copied() else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_argmax_source_missing_context_row(id, "right_case_missing"),
            );
            continue;
        };

        let Some(left_prompt) = token_id_vec(&left_case["token_ids"]["prompt"]) else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_argmax_source_missing_context_row(
                    id,
                    "left_prompt_token_ids_missing",
                ),
            );
            continue;
        };
        let Some(right_prompt) = token_id_vec(&right_case["token_ids"]["prompt"]) else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_argmax_source_missing_context_row(
                    id,
                    "right_prompt_token_ids_missing",
                ),
            );
            continue;
        };

        if left_prompt != right_prompt {
            prompt_history_serialization_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_argmax_source_prompt_history_row(id, left_case, right_case),
            );
            continue;
        }

        let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_argmax_source_missing_context_row(
                    id,
                    "left_generated_token_ids_missing",
                ),
            );
            continue;
        };
        let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_argmax_source_missing_context_row(
                    id,
                    "right_generated_token_ids_missing",
                ),
            );
            continue;
        };
        let Some(first_mismatch_index) =
            first_different_token_index(&left_generated, &right_generated)
        else {
            clean_count += 1;
            continue;
        };

        let row = generated_output_argmax_source_row(
            id,
            left_case,
            right_case,
            &left_generated,
            &right_generated,
            first_mismatch_index,
        );
        match row["classification"].as_str() {
            Some("generated_output_argmax_source_sampler_logit_extraction_policy") => {
                sampler_logit_extraction_count += 1;
            }
            Some("generated_output_argmax_source_trace_capture_context_loss") => {
                trace_capture_context_loss_count += 1;
            }
            Some("generated_output_argmax_source_internal_logit_source_missing_context") => {
                internal_logit_source_missing_context_count += 1;
            }
            _ => missing_context_count += 1,
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_argmax_source_frontier_missing_context"
    } else if prompt_history_serialization_count > 0 {
        "generated_output_argmax_source_frontier_prompt_history_serialization"
    } else if trace_capture_context_loss_count > 0 {
        "generated_output_argmax_source_frontier_trace_capture_context_loss"
    } else if sampler_logit_extraction_count > 0 {
        "generated_output_argmax_source_frontier_sampler_logit_extraction_policy"
    } else if internal_logit_source_missing_context_count > 0 {
        "generated_output_argmax_source_frontier_internal_logit_source_missing_context"
    } else {
        "generated_output_argmax_source_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "clean_count": clean_count,
        "prompt_history_serialization_count": prompt_history_serialization_count,
        "sampler_logit_extraction_count": sampler_logit_extraction_count,
        "trace_capture_context_loss_count": trace_capture_context_loss_count,
        "internal_logit_source_missing_context_count": internal_logit_source_missing_context_count,
        "missing_context_count": missing_context_count,
        "qk256_operand_context_available": false,
        "output_head_logit_accumulation_context_available": false,
        "next_diagnostic": if internal_logit_source_missing_context_count > 0 {
            "capture first-mismatch QK256 operand and output-head logit accumulation context"
        } else {
            "none"
        },
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn build_generated_output_internal_logit_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut clean_count = 0usize;
    let mut hidden_operand_drift_count = 0usize;
    let mut output_head_logit_accumulation_count = 0usize;
    let mut route_context_only_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let Some(left_case) = left_cases.get(id).copied() else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_internal_logit_source_missing_context_row(id, "left_case_missing"),
            );
            continue;
        };
        let Some(right_case) = right_cases.get(id).copied() else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_internal_logit_source_missing_context_row(
                    id,
                    "right_case_missing",
                ),
            );
            continue;
        };
        let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_internal_logit_source_missing_context_row(
                    id,
                    "left_generated_token_ids_missing",
                ),
            );
            continue;
        };
        let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_internal_logit_source_missing_context_row(
                    id,
                    "right_generated_token_ids_missing",
                ),
            );
            continue;
        };
        let Some(first_mismatch_index) =
            first_different_token_index(&left_generated, &right_generated)
        else {
            clean_count += 1;
            continue;
        };

        let row = generated_output_internal_logit_source_row(
            id,
            left_case,
            right_case,
            &left_generated,
            &right_generated,
            first_mismatch_index,
        );
        match row["classification"].as_str() {
            Some("generated_output_internal_logit_source_hidden_operand_drift") => {
                hidden_operand_drift_count += 1;
            }
            Some("generated_output_internal_logit_source_output_head_logit_accumulation") => {
                output_head_logit_accumulation_count += 1;
            }
            Some("generated_output_internal_logit_source_route_context_only") => {
                route_context_only_count += 1;
            }
            _ => missing_context_count += 1,
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_internal_logit_source_frontier_missing_context"
    } else if hidden_operand_drift_count > 0 {
        "generated_output_internal_logit_source_frontier_hidden_operand_drift"
    } else if output_head_logit_accumulation_count > 0 {
        "generated_output_internal_logit_source_frontier_output_head_logit_accumulation"
    } else if route_context_only_count > 0 {
        "generated_output_internal_logit_source_frontier_route_context_only"
    } else {
        "generated_output_internal_logit_source_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "clean_count": clean_count,
        "hidden_operand_drift_count": hidden_operand_drift_count,
        "output_head_logit_accumulation_count": output_head_logit_accumulation_count,
        "route_context_only_count": route_context_only_count,
        "missing_context_count": missing_context_count,
        "hidden_operand_context_available": rows.iter().any(|row| {
            row["hidden_operand_context_available"].as_bool().unwrap_or(false)
        }),
        "qk256_operand_context_available": rows.iter().any(|row| {
            row["qk256_operand_context_available"].as_bool().unwrap_or(false)
        }),
        "output_head_logit_accumulation_context_available": rows.iter().any(|row| {
            row["output_head_logit_accumulation_context_available"]
                .as_bool()
                .unwrap_or(false)
        }),
        "next_diagnostic": match classification {
            "generated_output_internal_logit_source_frontier_hidden_operand_drift" => {
                "localize hidden-state operand drift before output-head QK256"
            }
            "generated_output_internal_logit_source_frontier_output_head_logit_accumulation" => {
                "replay output-head QK256 accumulation for the selected mismatch tokens"
            }
            "generated_output_internal_logit_source_frontier_route_context_only" => {
                "capture hidden operand fingerprints and selected-token output-head accumulation"
            }
            "generated_output_internal_logit_source_frontier_missing_context" => {
                "rerun focused receipts with logit_source_context enabled"
            }
            _ => "none",
        },
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_internal_logit_source_missing_context_row(id: &str, reason: &str) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_internal_logit_source_missing_context",
        "reason": reason,
        "qk256_operand_context_available": false,
        "output_head_logit_accumulation_context_available": false,
    })
}

fn generated_output_internal_logit_source_row(
    id: &str,
    left_case: &Value,
    right_case: &Value,
    left_generated: &[u64],
    right_generated: &[u64],
    first_mismatch_index: usize,
) -> Value {
    let Some(left_step) =
        left_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_internal_logit_source_missing_context_row(
            id,
            "left_logits_step_missing",
        );
    };
    let Some(right_step) =
        right_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_internal_logit_source_missing_context_row(
            id,
            "right_logits_step_missing",
        );
    };
    let left_context = &left_step["logit_source_context"];
    let right_context = &right_step["logit_source_context"];
    if !left_context.is_object() {
        return generated_output_internal_logit_source_missing_context_row(
            id,
            "left_logit_source_context_missing",
        );
    }
    if !right_context.is_object() {
        return generated_output_internal_logit_source_missing_context_row(
            id,
            "right_logit_source_context_missing",
        );
    }

    let left_hidden = &left_context["hidden_operand"];
    let right_hidden = &right_context["hidden_operand"];
    let left_hidden_available = left_hidden["available"].as_bool().unwrap_or(false);
    let right_hidden_available = right_hidden["available"].as_bool().unwrap_or(false);
    let left_output_head_context =
        left_context["output_head_logit_accumulation_context_available"].as_bool().unwrap_or(false);
    let right_output_head_context =
        right_context["output_head_logit_accumulation_context_available"]
            .as_bool()
            .unwrap_or(false);
    let has_route_context = left_output_head_context || right_output_head_context;
    let left_hidden_sha = left_hidden["sha256_f32_le"].as_str();
    let right_hidden_sha = right_hidden["sha256_f32_le"].as_str();
    let hidden_sha_match = match (left_hidden_sha, right_hidden_sha) {
        (Some(left), Some(right)) => Some(left == right),
        _ => None,
    };
    let qk256_operand_context_available =
        left_hidden_available && right_hidden_available && hidden_sha_match.is_some();
    let hidden_operand_context_available = qk256_operand_context_available;
    let output_head_logit_accumulation_context_available =
        left_output_head_context && right_output_head_context;
    let margin_row =
        generated_output_logit_margin_row(id, left_case, right_case, first_mismatch_index, 0.01);

    let classification = if hidden_operand_context_available && hidden_sha_match == Some(false) {
        "generated_output_internal_logit_source_hidden_operand_drift"
    } else if qk256_operand_context_available
        && hidden_sha_match == Some(true)
        && output_head_logit_accumulation_context_available
    {
        "generated_output_internal_logit_source_output_head_logit_accumulation"
    } else if has_route_context {
        "generated_output_internal_logit_source_route_context_only"
    } else {
        "generated_output_internal_logit_source_missing_context"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_chosen_id": left_step["chosen_id"],
        "right_chosen_id": right_step["chosen_id"],
        "opposite_argmax": margin_row["opposite_argmax"],
        "left_margin_over_right_chosen_on_left": margin_row["left_margin_over_right_chosen_on_left"],
        "right_margin_over_left_chosen_on_right": margin_row["right_margin_over_left_chosen_on_right"],
        "hidden_operand_context_available": hidden_operand_context_available,
        "qk256_operand_context_available": qk256_operand_context_available,
        "output_head_logit_accumulation_context_available": output_head_logit_accumulation_context_available,
        "left_hidden_operand_available": left_hidden_available,
        "right_hidden_operand_available": right_hidden_available,
        "hidden_operand_sha256_match": hidden_sha_match,
        "left_hidden_operand_sha256_f32_le": left_hidden_sha,
        "right_hidden_operand_sha256_f32_le": right_hidden_sha,
        "left_hidden_operand_shape": left_hidden["shape"],
        "right_hidden_operand_shape": right_hidden["shape"],
        "left_hidden_operand_rms": left_hidden["rms"],
        "right_hidden_operand_rms": right_hidden["rms"],
        "hidden_operand_rms_abs_delta": number_abs_delta(&left_hidden["rms"], &right_hidden["rms"]),
        "left_output_head_qk256_dispatch_delta": left_context["output_head_qk256_dispatch_delta"],
        "right_output_head_qk256_dispatch_delta": right_context["output_head_qk256_dispatch_delta"],
        "left_output_head_a770_opencl_runtime_delta": left_context["output_head_a770_opencl_runtime_delta"],
        "right_output_head_a770_opencl_runtime_delta": right_context["output_head_a770_opencl_runtime_delta"],
        "next_diagnostic": match classification {
            "generated_output_internal_logit_source_hidden_operand_drift" => {
                "localize hidden-state operand drift before output-head QK256"
            }
            "generated_output_internal_logit_source_output_head_logit_accumulation" => {
                "replay output-head QK256 accumulation for the selected mismatch tokens"
            }
            "generated_output_internal_logit_source_route_context_only" => {
                "capture hidden operand fingerprints and selected-token output-head accumulation"
            }
            _ => "rerun focused receipts with logit_source_context enabled",
        },
    })
}

fn build_generated_output_hidden_state_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut clean_count = 0usize;
    let mut forward_output_drift_count = 0usize;
    let mut last_hidden_extraction_drift_count = 0usize;
    let mut qk256_residual_context_missing_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let Some(left_case) = left_cases.get(id).copied() else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_hidden_state_source_missing_context_row(id, "left_case_missing"),
            );
            continue;
        };
        let Some(right_case) = right_cases.get(id).copied() else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_hidden_state_source_missing_context_row(id, "right_case_missing"),
            );
            continue;
        };
        let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_hidden_state_source_missing_context_row(
                    id,
                    "left_generated_token_ids_missing",
                ),
            );
            continue;
        };
        let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
            missing_context_count += 1;
            row_candidate_count += 1;
            push_limited_row(
                &mut rows,
                ROW_LIMIT,
                generated_output_hidden_state_source_missing_context_row(
                    id,
                    "right_generated_token_ids_missing",
                ),
            );
            continue;
        };
        let Some(first_mismatch_index) =
            first_different_token_index(&left_generated, &right_generated)
        else {
            clean_count += 1;
            continue;
        };

        let row = generated_output_hidden_state_source_row(
            id,
            left_case,
            right_case,
            &left_generated,
            &right_generated,
            first_mismatch_index,
        );
        match row["classification"].as_str() {
            Some("generated_output_hidden_state_source_forward_output_drift") => {
                forward_output_drift_count += 1;
            }
            Some("generated_output_hidden_state_source_last_hidden_extraction_drift") => {
                last_hidden_extraction_drift_count += 1;
            }
            Some("generated_output_hidden_state_source_qk256_residual_context_missing") => {
                qk256_residual_context_missing_count += 1;
            }
            Some("generated_output_hidden_state_source_clean") => clean_count += 1,
            _ => missing_context_count += 1,
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_hidden_state_source_frontier_missing_context"
    } else if forward_output_drift_count > 0 {
        "generated_output_hidden_state_source_frontier_forward_output_drift"
    } else if last_hidden_extraction_drift_count > 0 {
        "generated_output_hidden_state_source_frontier_last_hidden_extraction_drift"
    } else if qk256_residual_context_missing_count > 0 {
        "generated_output_hidden_state_source_frontier_qk256_residual_context_missing"
    } else {
        "generated_output_hidden_state_source_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "clean_count": clean_count,
        "forward_output_drift_count": forward_output_drift_count,
        "last_hidden_extraction_drift_count": last_hidden_extraction_drift_count,
        "qk256_residual_context_missing_count": qk256_residual_context_missing_count,
        "missing_context_count": missing_context_count,
        "hidden_state_source_context_available": rows.iter().any(|row| {
            row["hidden_state_source_context_available"].as_bool().unwrap_or(false)
        }),
        "forward_output_context_available": rows.iter().any(|row| {
            row["forward_output_context_available"].as_bool().unwrap_or(false)
        }),
        "last_hidden_context_available": rows.iter().any(|row| {
            row["last_hidden_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": match classification {
            "generated_output_hidden_state_source_frontier_forward_output_drift" => {
                "capture final norm and prior layer output fingerprints before model.forward output"
            }
            "generated_output_hidden_state_source_frontier_last_hidden_extraction_drift" => {
                "inspect last-hidden extraction and tensor serialization boundary"
            }
            "generated_output_hidden_state_source_frontier_qk256_residual_context_missing" => {
                "capture final norm, prior layer output, and residual contribution context"
            }
            "generated_output_hidden_state_source_frontier_missing_context" => {
                "rerun focused receipts with hidden_state_source context enabled"
            }
            _ => "none",
        },
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_hidden_state_source_missing_context_row(id: &str, reason: &str) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_hidden_state_source_missing_context",
        "reason": reason,
        "hidden_state_source_context_available": false,
        "forward_output_context_available": false,
        "last_hidden_context_available": false,
    })
}

fn generated_output_hidden_state_source_row(
    id: &str,
    left_case: &Value,
    right_case: &Value,
    left_generated: &[u64],
    right_generated: &[u64],
    first_mismatch_index: usize,
) -> Value {
    let Some(left_step) =
        left_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_hidden_state_source_missing_context_row(
            id,
            "left_logits_step_missing",
        );
    };
    let Some(right_step) =
        right_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_hidden_state_source_missing_context_row(
            id,
            "right_logits_step_missing",
        );
    };

    let left_context = &left_step["logit_source_context"];
    let right_context = &right_step["logit_source_context"];
    if !left_context.is_object() {
        return generated_output_hidden_state_source_missing_context_row(
            id,
            "left_logit_source_context_missing",
        );
    }
    if !right_context.is_object() {
        return generated_output_hidden_state_source_missing_context_row(
            id,
            "right_logit_source_context_missing",
        );
    }

    let left_source = &left_context["hidden_state_source"];
    let right_source = &right_context["hidden_state_source"];
    if !left_source.is_object() {
        return generated_output_hidden_state_source_row_without_source_context(
            id,
            left_context,
            right_context,
            left_step,
            right_step,
            left_generated,
            right_generated,
            first_mismatch_index,
            "left_hidden_state_source_missing",
        );
    }
    if !right_source.is_object() {
        return generated_output_hidden_state_source_row_without_source_context(
            id,
            left_context,
            right_context,
            left_step,
            right_step,
            left_generated,
            right_generated,
            first_mismatch_index,
            "right_hidden_state_source_missing",
        );
    }

    let left_forward = &left_source["forward_output"];
    let right_forward = &right_source["forward_output"];
    let left_last_hidden = &left_source["last_hidden"];
    let right_last_hidden = &right_source["last_hidden"];
    let left_forward_available = left_forward["available"].as_bool().unwrap_or(false);
    let right_forward_available = right_forward["available"].as_bool().unwrap_or(false);
    let left_last_hidden_available = left_last_hidden["available"].as_bool().unwrap_or(false);
    let right_last_hidden_available = right_last_hidden["available"].as_bool().unwrap_or(false);
    let forward_sha_match = optional_str_eq(
        left_forward["sha256_f32_le"].as_str(),
        right_forward["sha256_f32_le"].as_str(),
    );
    let last_hidden_sha_match = optional_str_eq(
        left_last_hidden["sha256_f32_le"].as_str(),
        right_last_hidden["sha256_f32_le"].as_str(),
    );
    let hidden_state_source_context_available = left_forward_available
        && right_forward_available
        && left_last_hidden_available
        && right_last_hidden_available
        && forward_sha_match.is_some()
        && last_hidden_sha_match.is_some();
    let classification =
        if hidden_state_source_context_available && forward_sha_match == Some(false) {
            "generated_output_hidden_state_source_forward_output_drift"
        } else if hidden_state_source_context_available
            && forward_sha_match == Some(true)
            && last_hidden_sha_match == Some(false)
        {
            "generated_output_hidden_state_source_last_hidden_extraction_drift"
        } else if hidden_state_source_context_available {
            "generated_output_hidden_state_source_clean"
        } else {
            "generated_output_hidden_state_source_missing_context"
        };

    generated_output_hidden_state_source_common_row(
        id,
        classification,
        left_step,
        right_step,
        left_generated,
        right_generated,
        first_mismatch_index,
        json!({
            "hidden_state_source_context_available": hidden_state_source_context_available,
            "forward_output_context_available": left_forward_available && right_forward_available && forward_sha_match.is_some(),
            "last_hidden_context_available": left_last_hidden_available && right_last_hidden_available && last_hidden_sha_match.is_some(),
            "forward_output_sha256_match": forward_sha_match,
            "last_hidden_sha256_match": last_hidden_sha_match,
            "left_forward_output_sha256_f32_le": left_forward["sha256_f32_le"],
            "right_forward_output_sha256_f32_le": right_forward["sha256_f32_le"],
            "left_forward_output_shape": left_forward["shape"],
            "right_forward_output_shape": right_forward["shape"],
            "left_forward_output_rms": left_forward["rms"],
            "right_forward_output_rms": right_forward["rms"],
            "forward_output_rms_abs_delta": number_abs_delta(&left_forward["rms"], &right_forward["rms"]),
            "left_last_hidden_sha256_f32_le": left_last_hidden["sha256_f32_le"],
            "right_last_hidden_sha256_f32_le": right_last_hidden["sha256_f32_le"],
            "left_last_hidden_shape": left_last_hidden["shape"],
            "right_last_hidden_shape": right_last_hidden["shape"],
            "left_last_hidden_rms": left_last_hidden["rms"],
            "right_last_hidden_rms": right_last_hidden["rms"],
            "last_hidden_rms_abs_delta": number_abs_delta(&left_last_hidden["rms"], &right_last_hidden["rms"]),
            "reason": Value::Null,
        }),
    )
}

#[allow(clippy::too_many_arguments)]
fn generated_output_hidden_state_source_row_without_source_context(
    id: &str,
    left_context: &Value,
    right_context: &Value,
    left_step: &Value,
    right_step: &Value,
    left_generated: &[u64],
    right_generated: &[u64],
    first_mismatch_index: usize,
    reason: &str,
) -> Value {
    let left_hidden = &left_context["hidden_operand"];
    let right_hidden = &right_context["hidden_operand"];
    let hidden_sha_match = optional_str_eq(
        left_hidden["sha256_f32_le"].as_str(),
        right_hidden["sha256_f32_le"].as_str(),
    );
    let classification = if hidden_sha_match == Some(false) {
        "generated_output_hidden_state_source_qk256_residual_context_missing"
    } else {
        "generated_output_hidden_state_source_missing_context"
    };

    generated_output_hidden_state_source_common_row(
        id,
        classification,
        left_step,
        right_step,
        left_generated,
        right_generated,
        first_mismatch_index,
        json!({
            "hidden_state_source_context_available": false,
            "forward_output_context_available": false,
            "last_hidden_context_available": false,
            "hidden_operand_sha256_match": hidden_sha_match,
            "left_hidden_operand_sha256_f32_le": left_hidden["sha256_f32_le"],
            "right_hidden_operand_sha256_f32_le": right_hidden["sha256_f32_le"],
            "reason": reason,
        }),
    )
}

fn generated_output_hidden_state_source_common_row(
    id: &str,
    classification: &str,
    left_step: &Value,
    right_step: &Value,
    left_generated: &[u64],
    right_generated: &[u64],
    first_mismatch_index: usize,
    mut fields: Value,
) -> Value {
    let mut row = json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_chosen_id": left_step["chosen_id"],
        "right_chosen_id": right_step["chosen_id"],
        "next_diagnostic": match classification {
            "generated_output_hidden_state_source_forward_output_drift" => {
                "capture final norm and prior layer output fingerprints before model.forward output"
            }
            "generated_output_hidden_state_source_last_hidden_extraction_drift" => {
                "inspect last-hidden extraction and tensor serialization boundary"
            }
            "generated_output_hidden_state_source_qk256_residual_context_missing" => {
                "capture final norm, prior layer output, and residual contribution context"
            }
            "generated_output_hidden_state_source_missing_context" => {
                "rerun focused receipts with hidden_state_source context enabled"
            }
            _ => "none",
        },
    });
    if let (Some(row_object), Some(fields_object)) = (row.as_object_mut(), fields.as_object_mut()) {
        row_object.append(fields_object);
    }
    row
}

fn build_generated_output_model_forward_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut clean_count = 0usize;
    let mut prior_layer_output_drift_count = 0usize;
    let mut final_norm_output_drift_count = 0usize;
    let mut forward_output_serialization_drift_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_model_forward_source_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        match row["classification"].as_str() {
            Some("generated_output_model_forward_source_prior_layer_output_drift") => {
                prior_layer_output_drift_count += 1;
            }
            Some("generated_output_model_forward_source_final_norm_output_drift") => {
                final_norm_output_drift_count += 1;
            }
            Some("generated_output_model_forward_source_forward_output_serialization_drift") => {
                forward_output_serialization_drift_count += 1;
            }
            Some("generated_output_model_forward_source_clean") => clean_count += 1,
            Some("generated_output_model_forward_source_not_applicable") => {
                continue;
            }
            _ => missing_context_count += 1,
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_model_forward_source_frontier_missing_context"
    } else if prior_layer_output_drift_count > 0 {
        "generated_output_model_forward_source_frontier_prior_layer_output_drift"
    } else if final_norm_output_drift_count > 0 {
        "generated_output_model_forward_source_frontier_final_norm_output_drift"
    } else if forward_output_serialization_drift_count > 0 {
        "generated_output_model_forward_source_frontier_forward_output_serialization_drift"
    } else {
        "generated_output_model_forward_source_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "clean_count": clean_count,
        "prior_layer_output_drift_count": prior_layer_output_drift_count,
        "final_norm_output_drift_count": final_norm_output_drift_count,
        "forward_output_serialization_drift_count": forward_output_serialization_drift_count,
        "missing_context_count": missing_context_count,
        "model_forward_source_context_available": rows.iter().any(|row| {
            row["model_forward_source_context_available"].as_bool().unwrap_or(false)
        }),
        "prior_layer_output_context_available": rows.iter().any(|row| {
            row["prior_layer_output_context_available"].as_bool().unwrap_or(false)
        }),
        "final_norm_output_context_available": rows.iter().any(|row| {
            row["final_norm_output_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": match classification {
            "generated_output_model_forward_source_frontier_prior_layer_output_drift" => {
                "capture final transformer block residual, attention output, and FFN output fingerprints"
            }
            "generated_output_model_forward_source_frontier_final_norm_output_drift" => {
                "replay final norm input/output numeric policy for selected generated step"
            }
            "generated_output_model_forward_source_frontier_forward_output_serialization_drift" => {
                "inspect model.forward output serialization after final norm"
            }
            "generated_output_model_forward_source_frontier_missing_context" => {
                "rerun focused receipts with model_forward_source context enabled"
            }
            _ => "none",
        },
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_model_forward_source_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let Some(left_case) = left_case else {
        return generated_output_model_forward_source_missing_context_row(id, "left_case_missing");
    };
    let Some(right_case) = right_case else {
        return generated_output_model_forward_source_missing_context_row(id, "right_case_missing");
    };
    let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
        return generated_output_model_forward_source_missing_context_row(
            id,
            "left_generated_token_ids_missing",
        );
    };
    let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
        return generated_output_model_forward_source_missing_context_row(
            id,
            "right_generated_token_ids_missing",
        );
    };
    let Some(first_mismatch_index) = first_different_token_index(&left_generated, &right_generated)
    else {
        return json!({
            "case_id": id,
            "classification": "generated_output_model_forward_source_not_applicable",
            "reason": "generated_token_ids_match",
        });
    };

    let Some(left_step) =
        left_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_model_forward_source_missing_context_row(
            id,
            "left_logits_step_missing",
        );
    };
    let Some(right_step) =
        right_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_model_forward_source_missing_context_row(
            id,
            "right_logits_step_missing",
        );
    };

    let left_source = &left_step["logit_source_context"]["hidden_state_source"];
    let right_source = &right_step["logit_source_context"]["hidden_state_source"];
    if !left_source.is_object() {
        return generated_output_model_forward_source_missing_context_row(
            id,
            "left_hidden_state_source_missing",
        );
    }
    if !right_source.is_object() {
        return generated_output_model_forward_source_missing_context_row(
            id,
            "right_hidden_state_source_missing",
        );
    }

    let left_model_source = &left_source["model_forward_source"];
    let right_model_source = &right_source["model_forward_source"];
    if !left_model_source.is_object() {
        return generated_output_model_forward_source_missing_context_row(
            id,
            "left_model_forward_source_missing",
        );
    }
    if !right_model_source.is_object() {
        return generated_output_model_forward_source_missing_context_row(
            id,
            "right_model_forward_source_missing",
        );
    }

    let left_prior = &left_model_source["prior_layer_output"];
    let right_prior = &right_model_source["prior_layer_output"];
    let left_final_norm = &left_model_source["final_norm_output"];
    let right_final_norm = &right_model_source["final_norm_output"];
    let left_forward = &left_source["forward_output"];
    let right_forward = &right_source["forward_output"];

    let left_prior_available = left_prior["available"].as_bool().unwrap_or(false);
    let right_prior_available = right_prior["available"].as_bool().unwrap_or(false);
    let left_final_norm_available = left_final_norm["available"].as_bool().unwrap_or(false);
    let right_final_norm_available = right_final_norm["available"].as_bool().unwrap_or(false);
    let prior_sha_match = optional_str_eq(
        left_prior["sha256_f32_le"].as_str(),
        right_prior["sha256_f32_le"].as_str(),
    );
    let final_norm_sha_match = optional_str_eq(
        left_final_norm["sha256_f32_le"].as_str(),
        right_final_norm["sha256_f32_le"].as_str(),
    );
    let forward_sha_match = optional_str_eq(
        left_forward["sha256_f32_le"].as_str(),
        right_forward["sha256_f32_le"].as_str(),
    );
    let model_forward_source_context_available = left_prior_available
        && right_prior_available
        && left_final_norm_available
        && right_final_norm_available
        && prior_sha_match.is_some()
        && final_norm_sha_match.is_some();
    let classification = if model_forward_source_context_available && prior_sha_match == Some(false)
    {
        "generated_output_model_forward_source_prior_layer_output_drift"
    } else if model_forward_source_context_available
        && prior_sha_match == Some(true)
        && final_norm_sha_match == Some(false)
    {
        "generated_output_model_forward_source_final_norm_output_drift"
    } else if model_forward_source_context_available
        && final_norm_sha_match == Some(true)
        && forward_sha_match == Some(false)
    {
        "generated_output_model_forward_source_forward_output_serialization_drift"
    } else if model_forward_source_context_available {
        "generated_output_model_forward_source_clean"
    } else {
        "generated_output_model_forward_source_missing_context"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_chosen_id": left_step["chosen_id"],
        "right_chosen_id": right_step["chosen_id"],
        "model_forward_source_context_available": model_forward_source_context_available,
        "prior_layer_output_context_available": left_prior_available && right_prior_available && prior_sha_match.is_some(),
        "final_norm_output_context_available": left_final_norm_available && right_final_norm_available && final_norm_sha_match.is_some(),
        "prior_layer_output_sha256_match": prior_sha_match,
        "final_norm_output_sha256_match": final_norm_sha_match,
        "forward_output_sha256_match": forward_sha_match,
        "left_prior_layer_output_sha256_f32_le": left_prior["sha256_f32_le"],
        "right_prior_layer_output_sha256_f32_le": right_prior["sha256_f32_le"],
        "left_prior_layer_output_shape": left_prior["shape"],
        "right_prior_layer_output_shape": right_prior["shape"],
        "left_prior_layer_output_rms": left_prior["rms"],
        "right_prior_layer_output_rms": right_prior["rms"],
        "prior_layer_output_rms_abs_delta": number_abs_delta(&left_prior["rms"], &right_prior["rms"]),
        "left_final_norm_output_sha256_f32_le": left_final_norm["sha256_f32_le"],
        "right_final_norm_output_sha256_f32_le": right_final_norm["sha256_f32_le"],
        "left_final_norm_output_shape": left_final_norm["shape"],
        "right_final_norm_output_shape": right_final_norm["shape"],
        "left_final_norm_output_rms": left_final_norm["rms"],
        "right_final_norm_output_rms": right_final_norm["rms"],
        "final_norm_output_rms_abs_delta": number_abs_delta(&left_final_norm["rms"], &right_final_norm["rms"]),
        "left_final_norm_matches_forward_output": left_model_source["final_norm_matches_forward_output"],
        "right_final_norm_matches_forward_output": right_model_source["final_norm_matches_forward_output"],
        "next_diagnostic": match classification {
            "generated_output_model_forward_source_prior_layer_output_drift" => {
                "capture final transformer block residual, attention output, and FFN output fingerprints"
            }
            "generated_output_model_forward_source_final_norm_output_drift" => {
                "replay final norm input/output numeric policy for selected generated step"
            }
            "generated_output_model_forward_source_forward_output_serialization_drift" => {
                "inspect model.forward output serialization after final norm"
            }
            "generated_output_model_forward_source_missing_context" => {
                "rerun focused receipts with model_forward_source context enabled"
            }
            _ => "none",
        },
    })
}

fn generated_output_model_forward_source_missing_context_row(id: &str, reason: &str) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_model_forward_source_missing_context",
        "reason": reason,
        "model_forward_source_context_available": false,
        "prior_layer_output_context_available": false,
        "final_norm_output_context_available": false,
    })
}

fn build_generated_output_final_block_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut clean_count = 0usize;
    let mut block_input_drift_count = 0usize;
    let mut attention_output_drift_count = 0usize;
    let mut attention_residual_drift_count = 0usize;
    let mut ffn_output_drift_count = 0usize;
    let mut block_output_drift_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_final_block_source_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        match row["classification"].as_str() {
            Some("generated_output_final_block_source_block_input_drift") => {
                block_input_drift_count += 1;
            }
            Some("generated_output_final_block_source_attention_output_drift") => {
                attention_output_drift_count += 1;
            }
            Some("generated_output_final_block_source_attention_residual_drift") => {
                attention_residual_drift_count += 1;
            }
            Some("generated_output_final_block_source_ffn_output_drift") => {
                ffn_output_drift_count += 1;
            }
            Some("generated_output_final_block_source_block_output_drift") => {
                block_output_drift_count += 1;
            }
            Some("generated_output_final_block_source_clean") => clean_count += 1,
            Some("generated_output_final_block_source_not_applicable") => continue,
            _ => missing_context_count += 1,
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_final_block_source_frontier_missing_context"
    } else if block_input_drift_count > 0 {
        "generated_output_final_block_source_frontier_block_input_drift"
    } else if attention_output_drift_count > 0 {
        "generated_output_final_block_source_frontier_attention_output_drift"
    } else if attention_residual_drift_count > 0 {
        "generated_output_final_block_source_frontier_attention_residual_drift"
    } else if ffn_output_drift_count > 0 {
        "generated_output_final_block_source_frontier_ffn_output_drift"
    } else if block_output_drift_count > 0 {
        "generated_output_final_block_source_frontier_block_output_drift"
    } else {
        "generated_output_final_block_source_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "clean_count": clean_count,
        "block_input_drift_count": block_input_drift_count,
        "attention_output_drift_count": attention_output_drift_count,
        "attention_residual_drift_count": attention_residual_drift_count,
        "ffn_output_drift_count": ffn_output_drift_count,
        "block_output_drift_count": block_output_drift_count,
        "missing_context_count": missing_context_count,
        "final_block_source_context_available": rows.iter().any(|row| {
            row["final_block_source_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": final_block_source_next_diagnostic(classification),
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_final_block_source_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let Some(left_case) = left_case else {
        return generated_output_final_block_source_missing_context_row(id, "left_case_missing");
    };
    let Some(right_case) = right_case else {
        return generated_output_final_block_source_missing_context_row(id, "right_case_missing");
    };
    let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
        return generated_output_final_block_source_missing_context_row(
            id,
            "left_generated_token_ids_missing",
        );
    };
    let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
        return generated_output_final_block_source_missing_context_row(
            id,
            "right_generated_token_ids_missing",
        );
    };
    let Some(first_mismatch_index) = first_different_token_index(&left_generated, &right_generated)
    else {
        return json!({
            "case_id": id,
            "classification": "generated_output_final_block_source_not_applicable",
            "reason": "generated_token_ids_match",
        });
    };

    let Some(left_step) =
        left_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_final_block_source_missing_context_row(
            id,
            "left_logits_step_missing",
        );
    };
    let Some(right_step) =
        right_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_final_block_source_missing_context_row(
            id,
            "right_logits_step_missing",
        );
    };

    let left_final_block = &left_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["final_block_source"];
    let right_final_block = &right_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["final_block_source"];
    if !left_final_block.is_object() {
        return generated_output_final_block_source_missing_context_row(
            id,
            "left_final_block_source_missing",
        );
    }
    if !right_final_block.is_object() {
        return generated_output_final_block_source_missing_context_row(
            id,
            "right_final_block_source_missing",
        );
    }

    let (block_input_available, block_input_sha_match) =
        final_block_tensor_pair_status(left_final_block, right_final_block, "block_input");
    let (attention_output_available, attention_output_sha_match) =
        final_block_tensor_pair_status(left_final_block, right_final_block, "attention_output");
    let (post_attention_residual_available, post_attention_residual_sha_match) =
        final_block_tensor_pair_status(
            left_final_block,
            right_final_block,
            "post_attention_residual",
        );
    let (feed_forward_output_available, feed_forward_output_sha_match) =
        final_block_tensor_pair_status(left_final_block, right_final_block, "feed_forward_output");
    let (block_output_available, block_output_sha_match) =
        final_block_tensor_pair_status(left_final_block, right_final_block, "block_output");
    let final_block_source_context_available = block_input_available
        && attention_output_available
        && post_attention_residual_available
        && feed_forward_output_available
        && block_output_available;

    let classification = if !final_block_source_context_available {
        "generated_output_final_block_source_missing_context"
    } else if block_input_sha_match == Some(false) {
        "generated_output_final_block_source_block_input_drift"
    } else if attention_output_sha_match == Some(false) {
        "generated_output_final_block_source_attention_output_drift"
    } else if post_attention_residual_sha_match == Some(false) {
        "generated_output_final_block_source_attention_residual_drift"
    } else if feed_forward_output_sha_match == Some(false) {
        "generated_output_final_block_source_ffn_output_drift"
    } else if block_output_sha_match == Some(false) {
        "generated_output_final_block_source_block_output_drift"
    } else {
        "generated_output_final_block_source_clean"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_chosen_id": left_step["chosen_id"],
        "right_chosen_id": right_step["chosen_id"],
        "final_block_source_context_available": final_block_source_context_available,
        "block_input_context_available": block_input_available,
        "attention_output_context_available": attention_output_available,
        "post_attention_residual_context_available": post_attention_residual_available,
        "feed_forward_output_context_available": feed_forward_output_available,
        "block_output_context_available": block_output_available,
        "block_input_sha256_match": block_input_sha_match,
        "attention_output_sha256_match": attention_output_sha_match,
        "post_attention_residual_sha256_match": post_attention_residual_sha_match,
        "feed_forward_output_sha256_match": feed_forward_output_sha_match,
        "block_output_sha256_match": block_output_sha_match,
        "left_block_input_sha256_f32_le": left_final_block["block_input"]["sha256_f32_le"],
        "right_block_input_sha256_f32_le": right_final_block["block_input"]["sha256_f32_le"],
        "left_block_input_rms": left_final_block["block_input"]["rms"],
        "right_block_input_rms": right_final_block["block_input"]["rms"],
        "block_input_rms_abs_delta": number_abs_delta(&left_final_block["block_input"]["rms"], &right_final_block["block_input"]["rms"]),
        "left_attention_output_sha256_f32_le": left_final_block["attention_output"]["sha256_f32_le"],
        "right_attention_output_sha256_f32_le": right_final_block["attention_output"]["sha256_f32_le"],
        "left_attention_output_rms": left_final_block["attention_output"]["rms"],
        "right_attention_output_rms": right_final_block["attention_output"]["rms"],
        "attention_output_rms_abs_delta": number_abs_delta(&left_final_block["attention_output"]["rms"], &right_final_block["attention_output"]["rms"]),
        "left_post_attention_residual_sha256_f32_le": left_final_block["post_attention_residual"]["sha256_f32_le"],
        "right_post_attention_residual_sha256_f32_le": right_final_block["post_attention_residual"]["sha256_f32_le"],
        "left_post_attention_residual_rms": left_final_block["post_attention_residual"]["rms"],
        "right_post_attention_residual_rms": right_final_block["post_attention_residual"]["rms"],
        "post_attention_residual_rms_abs_delta": number_abs_delta(&left_final_block["post_attention_residual"]["rms"], &right_final_block["post_attention_residual"]["rms"]),
        "left_feed_forward_output_sha256_f32_le": left_final_block["feed_forward_output"]["sha256_f32_le"],
        "right_feed_forward_output_sha256_f32_le": right_final_block["feed_forward_output"]["sha256_f32_le"],
        "left_feed_forward_output_rms": left_final_block["feed_forward_output"]["rms"],
        "right_feed_forward_output_rms": right_final_block["feed_forward_output"]["rms"],
        "feed_forward_output_rms_abs_delta": number_abs_delta(&left_final_block["feed_forward_output"]["rms"], &right_final_block["feed_forward_output"]["rms"]),
        "left_block_output_sha256_f32_le": left_final_block["block_output"]["sha256_f32_le"],
        "right_block_output_sha256_f32_le": right_final_block["block_output"]["sha256_f32_le"],
        "left_block_output_rms": left_final_block["block_output"]["rms"],
        "right_block_output_rms": right_final_block["block_output"]["rms"],
        "block_output_rms_abs_delta": number_abs_delta(&left_final_block["block_output"]["rms"], &right_final_block["block_output"]["rms"]),
        "next_diagnostic": final_block_source_next_diagnostic(classification),
    })
}

fn final_block_tensor_pair_status(
    left_final_block: &Value,
    right_final_block: &Value,
    field: &str,
) -> (bool, Option<bool>) {
    let left = &left_final_block[field];
    let right = &right_final_block[field];
    let sha_match =
        optional_str_eq(left["sha256_f32_le"].as_str(), right["sha256_f32_le"].as_str());
    (
        left["available"].as_bool().unwrap_or(false)
            && right["available"].as_bool().unwrap_or(false)
            && sha_match.is_some(),
        sha_match,
    )
}

fn final_block_source_next_diagnostic(classification: &str) -> &'static str {
    match classification {
        "generated_output_final_block_source_frontier_block_input_drift"
        | "generated_output_final_block_source_block_input_drift" => {
            "capture penultimate transformer block source frontier"
        }
        "generated_output_final_block_source_frontier_attention_output_drift"
        | "generated_output_final_block_source_attention_output_drift" => {
            "replay final transformer block attention output source"
        }
        "generated_output_final_block_source_frontier_attention_residual_drift"
        | "generated_output_final_block_source_attention_residual_drift" => {
            "inspect final transformer block attention residual add serialization"
        }
        "generated_output_final_block_source_frontier_ffn_output_drift"
        | "generated_output_final_block_source_ffn_output_drift" => {
            "replay final transformer block FFN output source"
        }
        "generated_output_final_block_source_frontier_block_output_drift"
        | "generated_output_final_block_source_block_output_drift" => {
            "inspect final transformer block FFN residual add serialization"
        }
        "generated_output_final_block_source_frontier_missing_context"
        | "generated_output_final_block_source_missing_context" => {
            "rerun focused receipts with final_block_source context enabled"
        }
        _ => "none",
    }
}

fn generated_output_final_block_source_missing_context_row(id: &str, reason: &str) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_final_block_source_missing_context",
        "reason": reason,
        "final_block_source_context_available": false,
    })
}

fn build_generated_output_penultimate_block_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut clean_count = 0usize;
    let mut block_input_drift_count = 0usize;
    let mut attention_output_drift_count = 0usize;
    let mut attention_residual_drift_count = 0usize;
    let mut ffn_output_drift_count = 0usize;
    let mut block_output_drift_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_penultimate_block_source_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        match row["classification"].as_str() {
            Some("generated_output_penultimate_block_source_block_input_drift") => {
                block_input_drift_count += 1;
            }
            Some("generated_output_penultimate_block_source_attention_output_drift") => {
                attention_output_drift_count += 1;
            }
            Some("generated_output_penultimate_block_source_attention_residual_drift") => {
                attention_residual_drift_count += 1;
            }
            Some("generated_output_penultimate_block_source_ffn_output_drift") => {
                ffn_output_drift_count += 1;
            }
            Some("generated_output_penultimate_block_source_block_output_drift") => {
                block_output_drift_count += 1;
            }
            Some("generated_output_penultimate_block_source_clean") => clean_count += 1,
            Some("generated_output_penultimate_block_source_not_applicable") => continue,
            _ => missing_context_count += 1,
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_penultimate_block_source_frontier_missing_context"
    } else if block_input_drift_count > 0 {
        "generated_output_penultimate_block_source_frontier_block_input_drift"
    } else if attention_output_drift_count > 0 {
        "generated_output_penultimate_block_source_frontier_attention_output_drift"
    } else if attention_residual_drift_count > 0 {
        "generated_output_penultimate_block_source_frontier_attention_residual_drift"
    } else if ffn_output_drift_count > 0 {
        "generated_output_penultimate_block_source_frontier_ffn_output_drift"
    } else if block_output_drift_count > 0 {
        "generated_output_penultimate_block_source_frontier_block_output_drift"
    } else {
        "generated_output_penultimate_block_source_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "clean_count": clean_count,
        "block_input_drift_count": block_input_drift_count,
        "attention_output_drift_count": attention_output_drift_count,
        "attention_residual_drift_count": attention_residual_drift_count,
        "ffn_output_drift_count": ffn_output_drift_count,
        "block_output_drift_count": block_output_drift_count,
        "missing_context_count": missing_context_count,
        "penultimate_block_source_context_available": rows.iter().any(|row| {
            row["penultimate_block_source_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": penultimate_block_source_next_diagnostic(classification),
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_penultimate_block_source_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let Some(left_case) = left_case else {
        return generated_output_penultimate_block_source_missing_context_row(
            id,
            "left_case_missing",
        );
    };
    let Some(right_case) = right_case else {
        return generated_output_penultimate_block_source_missing_context_row(
            id,
            "right_case_missing",
        );
    };
    let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
        return generated_output_penultimate_block_source_missing_context_row(
            id,
            "left_generated_token_ids_missing",
        );
    };
    let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
        return generated_output_penultimate_block_source_missing_context_row(
            id,
            "right_generated_token_ids_missing",
        );
    };
    let Some(first_mismatch_index) = first_different_token_index(&left_generated, &right_generated)
    else {
        return json!({
            "case_id": id,
            "classification": "generated_output_penultimate_block_source_not_applicable",
            "reason": "generated_token_ids_match",
        });
    };

    let Some(left_step) =
        left_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_penultimate_block_source_missing_context_row(
            id,
            "left_logits_step_missing",
        );
    };
    let Some(right_step) =
        right_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_penultimate_block_source_missing_context_row(
            id,
            "right_logits_step_missing",
        );
    };

    let left_block = &left_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["penultimate_block_source"];
    let right_block = &right_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["penultimate_block_source"];
    if !left_block.is_object() {
        return generated_output_penultimate_block_source_missing_context_row(
            id,
            "left_penultimate_block_source_missing",
        );
    }
    if !right_block.is_object() {
        return generated_output_penultimate_block_source_missing_context_row(
            id,
            "right_penultimate_block_source_missing",
        );
    }

    let (block_input_available, block_input_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "block_input");
    let (attention_output_available, attention_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "attention_output");
    let (post_attention_residual_available, post_attention_residual_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "post_attention_residual");
    let (feed_forward_output_available, feed_forward_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "feed_forward_output");
    let (block_output_available, block_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "block_output");
    let penultimate_block_source_context_available = block_input_available
        && attention_output_available
        && post_attention_residual_available
        && feed_forward_output_available
        && block_output_available;

    let classification = if !penultimate_block_source_context_available {
        "generated_output_penultimate_block_source_missing_context"
    } else if block_input_sha_match == Some(false) {
        "generated_output_penultimate_block_source_block_input_drift"
    } else if attention_output_sha_match == Some(false) {
        "generated_output_penultimate_block_source_attention_output_drift"
    } else if post_attention_residual_sha_match == Some(false) {
        "generated_output_penultimate_block_source_attention_residual_drift"
    } else if feed_forward_output_sha_match == Some(false) {
        "generated_output_penultimate_block_source_ffn_output_drift"
    } else if block_output_sha_match == Some(false) {
        "generated_output_penultimate_block_source_block_output_drift"
    } else {
        "generated_output_penultimate_block_source_clean"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_chosen_id": left_step["chosen_id"],
        "right_chosen_id": right_step["chosen_id"],
        "penultimate_block_source_context_available": penultimate_block_source_context_available,
        "block_input_context_available": block_input_available,
        "attention_output_context_available": attention_output_available,
        "post_attention_residual_context_available": post_attention_residual_available,
        "feed_forward_output_context_available": feed_forward_output_available,
        "block_output_context_available": block_output_available,
        "block_input_sha256_match": block_input_sha_match,
        "attention_output_sha256_match": attention_output_sha_match,
        "post_attention_residual_sha256_match": post_attention_residual_sha_match,
        "feed_forward_output_sha256_match": feed_forward_output_sha_match,
        "block_output_sha256_match": block_output_sha_match,
        "left_block_input_sha256_f32_le": left_block["block_input"]["sha256_f32_le"],
        "right_block_input_sha256_f32_le": right_block["block_input"]["sha256_f32_le"],
        "left_block_input_rms": left_block["block_input"]["rms"],
        "right_block_input_rms": right_block["block_input"]["rms"],
        "block_input_rms_abs_delta": number_abs_delta(&left_block["block_input"]["rms"], &right_block["block_input"]["rms"]),
        "left_attention_output_sha256_f32_le": left_block["attention_output"]["sha256_f32_le"],
        "right_attention_output_sha256_f32_le": right_block["attention_output"]["sha256_f32_le"],
        "left_attention_output_rms": left_block["attention_output"]["rms"],
        "right_attention_output_rms": right_block["attention_output"]["rms"],
        "attention_output_rms_abs_delta": number_abs_delta(&left_block["attention_output"]["rms"], &right_block["attention_output"]["rms"]),
        "left_post_attention_residual_sha256_f32_le": left_block["post_attention_residual"]["sha256_f32_le"],
        "right_post_attention_residual_sha256_f32_le": right_block["post_attention_residual"]["sha256_f32_le"],
        "left_post_attention_residual_rms": left_block["post_attention_residual"]["rms"],
        "right_post_attention_residual_rms": right_block["post_attention_residual"]["rms"],
        "post_attention_residual_rms_abs_delta": number_abs_delta(&left_block["post_attention_residual"]["rms"], &right_block["post_attention_residual"]["rms"]),
        "left_feed_forward_output_sha256_f32_le": left_block["feed_forward_output"]["sha256_f32_le"],
        "right_feed_forward_output_sha256_f32_le": right_block["feed_forward_output"]["sha256_f32_le"],
        "left_feed_forward_output_rms": left_block["feed_forward_output"]["rms"],
        "right_feed_forward_output_rms": right_block["feed_forward_output"]["rms"],
        "feed_forward_output_rms_abs_delta": number_abs_delta(&left_block["feed_forward_output"]["rms"], &right_block["feed_forward_output"]["rms"]),
        "left_block_output_sha256_f32_le": left_block["block_output"]["sha256_f32_le"],
        "right_block_output_sha256_f32_le": right_block["block_output"]["sha256_f32_le"],
        "left_block_output_rms": left_block["block_output"]["rms"],
        "right_block_output_rms": right_block["block_output"]["rms"],
        "block_output_rms_abs_delta": number_abs_delta(&left_block["block_output"]["rms"], &right_block["block_output"]["rms"]),
        "next_diagnostic": penultimate_block_source_next_diagnostic(classification),
    })
}

fn penultimate_block_source_next_diagnostic(classification: &str) -> &'static str {
    match classification {
        "generated_output_penultimate_block_source_frontier_block_input_drift"
        | "generated_output_penultimate_block_source_block_input_drift" => {
            "capture antepenultimate transformer block source frontier"
        }
        "generated_output_penultimate_block_source_frontier_attention_output_drift"
        | "generated_output_penultimate_block_source_attention_output_drift" => {
            "replay penultimate transformer block attention output source"
        }
        "generated_output_penultimate_block_source_frontier_attention_residual_drift"
        | "generated_output_penultimate_block_source_attention_residual_drift" => {
            "inspect penultimate transformer block attention residual add serialization"
        }
        "generated_output_penultimate_block_source_frontier_ffn_output_drift"
        | "generated_output_penultimate_block_source_ffn_output_drift" => {
            "replay penultimate transformer block FFN output source"
        }
        "generated_output_penultimate_block_source_frontier_block_output_drift"
        | "generated_output_penultimate_block_source_block_output_drift" => {
            "inspect penultimate transformer block FFN residual add serialization"
        }
        "generated_output_penultimate_block_source_frontier_missing_context"
        | "generated_output_penultimate_block_source_missing_context" => {
            "rerun focused receipts with penultimate_block_source context enabled"
        }
        _ => "none",
    }
}

fn generated_output_penultimate_block_source_missing_context_row(id: &str, reason: &str) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_penultimate_block_source_missing_context",
        "reason": reason,
        "penultimate_block_source_context_available": false,
    })
}

fn build_generated_output_antepenultimate_block_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut clean_count = 0usize;
    let mut block_input_drift_count = 0usize;
    let mut attention_output_drift_count = 0usize;
    let mut attention_residual_drift_count = 0usize;
    let mut ffn_output_drift_count = 0usize;
    let mut block_output_drift_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_antepenultimate_block_source_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        match row["classification"].as_str() {
            Some("generated_output_antepenultimate_block_source_block_input_drift") => {
                block_input_drift_count += 1;
            }
            Some("generated_output_antepenultimate_block_source_attention_output_drift") => {
                attention_output_drift_count += 1;
            }
            Some("generated_output_antepenultimate_block_source_attention_residual_drift") => {
                attention_residual_drift_count += 1;
            }
            Some("generated_output_antepenultimate_block_source_ffn_output_drift") => {
                ffn_output_drift_count += 1;
            }
            Some("generated_output_antepenultimate_block_source_block_output_drift") => {
                block_output_drift_count += 1;
            }
            Some("generated_output_antepenultimate_block_source_clean") => clean_count += 1,
            Some("generated_output_antepenultimate_block_source_not_applicable") => continue,
            _ => missing_context_count += 1,
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_antepenultimate_block_source_frontier_missing_context"
    } else if block_input_drift_count > 0 {
        "generated_output_antepenultimate_block_source_frontier_block_input_drift"
    } else if attention_output_drift_count > 0 {
        "generated_output_antepenultimate_block_source_frontier_attention_output_drift"
    } else if attention_residual_drift_count > 0 {
        "generated_output_antepenultimate_block_source_frontier_attention_residual_drift"
    } else if ffn_output_drift_count > 0 {
        "generated_output_antepenultimate_block_source_frontier_ffn_output_drift"
    } else if block_output_drift_count > 0 {
        "generated_output_antepenultimate_block_source_frontier_block_output_drift"
    } else {
        "generated_output_antepenultimate_block_source_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "clean_count": clean_count,
        "block_input_drift_count": block_input_drift_count,
        "attention_output_drift_count": attention_output_drift_count,
        "attention_residual_drift_count": attention_residual_drift_count,
        "ffn_output_drift_count": ffn_output_drift_count,
        "block_output_drift_count": block_output_drift_count,
        "missing_context_count": missing_context_count,
        "antepenultimate_block_source_context_available": rows.iter().any(|row| {
            row["antepenultimate_block_source_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": antepenultimate_block_source_next_diagnostic(classification),
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_antepenultimate_block_source_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let Some(left_case) = left_case else {
        return generated_output_antepenultimate_block_source_missing_context_row(
            id,
            "left_case_missing",
        );
    };
    let Some(right_case) = right_case else {
        return generated_output_antepenultimate_block_source_missing_context_row(
            id,
            "right_case_missing",
        );
    };
    let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
        return generated_output_antepenultimate_block_source_missing_context_row(
            id,
            "left_generated_token_ids_missing",
        );
    };
    let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
        return generated_output_antepenultimate_block_source_missing_context_row(
            id,
            "right_generated_token_ids_missing",
        );
    };
    let Some(first_mismatch_index) = first_different_token_index(&left_generated, &right_generated)
    else {
        return json!({
            "case_id": id,
            "classification": "generated_output_antepenultimate_block_source_not_applicable",
            "reason": "generated_token_ids_match",
        });
    };

    let Some(left_step) =
        left_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_antepenultimate_block_source_missing_context_row(
            id,
            "left_logits_step_missing",
        );
    };
    let Some(right_step) =
        right_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_antepenultimate_block_source_missing_context_row(
            id,
            "right_logits_step_missing",
        );
    };

    let left_block = &left_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["antepenultimate_block_source"];
    let right_block = &right_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["antepenultimate_block_source"];
    if !left_block.is_object() {
        return generated_output_antepenultimate_block_source_missing_context_row(
            id,
            "left_antepenultimate_block_source_missing",
        );
    }
    if !right_block.is_object() {
        return generated_output_antepenultimate_block_source_missing_context_row(
            id,
            "right_antepenultimate_block_source_missing",
        );
    }

    let (block_input_available, block_input_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "block_input");
    let (attention_output_available, attention_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "attention_output");
    let (post_attention_residual_available, post_attention_residual_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "post_attention_residual");
    let (feed_forward_output_available, feed_forward_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "feed_forward_output");
    let (block_output_available, block_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "block_output");
    let antepenultimate_block_source_context_available = block_input_available
        && attention_output_available
        && post_attention_residual_available
        && feed_forward_output_available
        && block_output_available;

    let classification = if !antepenultimate_block_source_context_available {
        "generated_output_antepenultimate_block_source_missing_context"
    } else if block_input_sha_match == Some(false) {
        "generated_output_antepenultimate_block_source_block_input_drift"
    } else if attention_output_sha_match == Some(false) {
        "generated_output_antepenultimate_block_source_attention_output_drift"
    } else if post_attention_residual_sha_match == Some(false) {
        "generated_output_antepenultimate_block_source_attention_residual_drift"
    } else if feed_forward_output_sha_match == Some(false) {
        "generated_output_antepenultimate_block_source_ffn_output_drift"
    } else if block_output_sha_match == Some(false) {
        "generated_output_antepenultimate_block_source_block_output_drift"
    } else {
        "generated_output_antepenultimate_block_source_clean"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_chosen_id": left_step["chosen_id"],
        "right_chosen_id": right_step["chosen_id"],
        "antepenultimate_block_source_context_available": antepenultimate_block_source_context_available,
        "block_input_context_available": block_input_available,
        "attention_output_context_available": attention_output_available,
        "post_attention_residual_context_available": post_attention_residual_available,
        "feed_forward_output_context_available": feed_forward_output_available,
        "block_output_context_available": block_output_available,
        "block_input_sha256_match": block_input_sha_match,
        "attention_output_sha256_match": attention_output_sha_match,
        "post_attention_residual_sha256_match": post_attention_residual_sha_match,
        "feed_forward_output_sha256_match": feed_forward_output_sha_match,
        "block_output_sha256_match": block_output_sha_match,
        "left_block_input_sha256_f32_le": left_block["block_input"]["sha256_f32_le"],
        "right_block_input_sha256_f32_le": right_block["block_input"]["sha256_f32_le"],
        "left_block_input_rms": left_block["block_input"]["rms"],
        "right_block_input_rms": right_block["block_input"]["rms"],
        "block_input_rms_abs_delta": number_abs_delta(&left_block["block_input"]["rms"], &right_block["block_input"]["rms"]),
        "left_attention_output_sha256_f32_le": left_block["attention_output"]["sha256_f32_le"],
        "right_attention_output_sha256_f32_le": right_block["attention_output"]["sha256_f32_le"],
        "left_attention_output_rms": left_block["attention_output"]["rms"],
        "right_attention_output_rms": right_block["attention_output"]["rms"],
        "attention_output_rms_abs_delta": number_abs_delta(&left_block["attention_output"]["rms"], &right_block["attention_output"]["rms"]),
        "left_post_attention_residual_sha256_f32_le": left_block["post_attention_residual"]["sha256_f32_le"],
        "right_post_attention_residual_sha256_f32_le": right_block["post_attention_residual"]["sha256_f32_le"],
        "left_post_attention_residual_rms": left_block["post_attention_residual"]["rms"],
        "right_post_attention_residual_rms": right_block["post_attention_residual"]["rms"],
        "post_attention_residual_rms_abs_delta": number_abs_delta(&left_block["post_attention_residual"]["rms"], &right_block["post_attention_residual"]["rms"]),
        "left_feed_forward_output_sha256_f32_le": left_block["feed_forward_output"]["sha256_f32_le"],
        "right_feed_forward_output_sha256_f32_le": right_block["feed_forward_output"]["sha256_f32_le"],
        "left_feed_forward_output_rms": left_block["feed_forward_output"]["rms"],
        "right_feed_forward_output_rms": right_block["feed_forward_output"]["rms"],
        "feed_forward_output_rms_abs_delta": number_abs_delta(&left_block["feed_forward_output"]["rms"], &right_block["feed_forward_output"]["rms"]),
        "left_block_output_sha256_f32_le": left_block["block_output"]["sha256_f32_le"],
        "right_block_output_sha256_f32_le": right_block["block_output"]["sha256_f32_le"],
        "left_block_output_rms": left_block["block_output"]["rms"],
        "right_block_output_rms": right_block["block_output"]["rms"],
        "block_output_rms_abs_delta": number_abs_delta(&left_block["block_output"]["rms"], &right_block["block_output"]["rms"]),
        "next_diagnostic": antepenultimate_block_source_next_diagnostic(classification),
    })
}

fn antepenultimate_block_source_next_diagnostic(classification: &str) -> &'static str {
    match classification {
        "generated_output_antepenultimate_block_source_frontier_block_input_drift"
        | "generated_output_antepenultimate_block_source_block_input_drift" => {
            "capture pre-antepenultimate transformer block source frontier"
        }
        "generated_output_antepenultimate_block_source_frontier_attention_output_drift"
        | "generated_output_antepenultimate_block_source_attention_output_drift" => {
            "replay antepenultimate transformer block attention output source"
        }
        "generated_output_antepenultimate_block_source_frontier_attention_residual_drift"
        | "generated_output_antepenultimate_block_source_attention_residual_drift" => {
            "inspect antepenultimate transformer block attention residual add serialization"
        }
        "generated_output_antepenultimate_block_source_frontier_ffn_output_drift"
        | "generated_output_antepenultimate_block_source_ffn_output_drift" => {
            "replay antepenultimate transformer block FFN output source"
        }
        "generated_output_antepenultimate_block_source_frontier_block_output_drift"
        | "generated_output_antepenultimate_block_source_block_output_drift" => {
            "inspect antepenultimate transformer block FFN residual add serialization"
        }
        "generated_output_antepenultimate_block_source_frontier_missing_context"
        | "generated_output_antepenultimate_block_source_missing_context" => {
            "rerun focused receipts with antepenultimate_block_source context enabled"
        }
        _ => "none",
    }
}

fn generated_output_antepenultimate_block_source_missing_context_row(
    id: &str,
    reason: &str,
) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_antepenultimate_block_source_missing_context",
        "reason": reason,
        "antepenultimate_block_source_context_available": false,
    })
}

fn build_generated_output_pre_antepenultimate_block_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut clean_count = 0usize;
    let mut block_input_drift_count = 0usize;
    let mut attention_output_drift_count = 0usize;
    let mut attention_residual_drift_count = 0usize;
    let mut ffn_output_drift_count = 0usize;
    let mut block_output_drift_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_pre_antepenultimate_block_source_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        match row["classification"].as_str() {
            Some("generated_output_pre_antepenultimate_block_source_block_input_drift") => {
                block_input_drift_count += 1;
            }
            Some("generated_output_pre_antepenultimate_block_source_attention_output_drift") => {
                attention_output_drift_count += 1;
            }
            Some("generated_output_pre_antepenultimate_block_source_attention_residual_drift") => {
                attention_residual_drift_count += 1;
            }
            Some("generated_output_pre_antepenultimate_block_source_ffn_output_drift") => {
                ffn_output_drift_count += 1;
            }
            Some("generated_output_pre_antepenultimate_block_source_block_output_drift") => {
                block_output_drift_count += 1;
            }
            Some("generated_output_pre_antepenultimate_block_source_clean") => clean_count += 1,
            Some("generated_output_pre_antepenultimate_block_source_not_applicable") => continue,
            _ => missing_context_count += 1,
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_pre_antepenultimate_block_source_frontier_missing_context"
    } else if block_input_drift_count > 0 {
        "generated_output_pre_antepenultimate_block_source_frontier_block_input_drift"
    } else if attention_output_drift_count > 0 {
        "generated_output_pre_antepenultimate_block_source_frontier_attention_output_drift"
    } else if attention_residual_drift_count > 0 {
        "generated_output_pre_antepenultimate_block_source_frontier_attention_residual_drift"
    } else if ffn_output_drift_count > 0 {
        "generated_output_pre_antepenultimate_block_source_frontier_ffn_output_drift"
    } else if block_output_drift_count > 0 {
        "generated_output_pre_antepenultimate_block_source_frontier_block_output_drift"
    } else {
        "generated_output_pre_antepenultimate_block_source_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "clean_count": clean_count,
        "block_input_drift_count": block_input_drift_count,
        "attention_output_drift_count": attention_output_drift_count,
        "attention_residual_drift_count": attention_residual_drift_count,
        "ffn_output_drift_count": ffn_output_drift_count,
        "block_output_drift_count": block_output_drift_count,
        "missing_context_count": missing_context_count,
        "pre_antepenultimate_block_source_context_available": rows.iter().any(|row| {
            row["pre_antepenultimate_block_source_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": pre_antepenultimate_block_source_next_diagnostic(classification),
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_pre_antepenultimate_block_source_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let Some(left_case) = left_case else {
        return generated_output_pre_antepenultimate_block_source_missing_context_row(
            id,
            "left_case_missing",
        );
    };
    let Some(right_case) = right_case else {
        return generated_output_pre_antepenultimate_block_source_missing_context_row(
            id,
            "right_case_missing",
        );
    };
    let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
        return generated_output_pre_antepenultimate_block_source_missing_context_row(
            id,
            "left_generated_token_ids_missing",
        );
    };
    let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
        return generated_output_pre_antepenultimate_block_source_missing_context_row(
            id,
            "right_generated_token_ids_missing",
        );
    };
    let Some(first_mismatch_index) = first_different_token_index(&left_generated, &right_generated)
    else {
        return json!({
            "case_id": id,
            "classification": "generated_output_pre_antepenultimate_block_source_not_applicable",
            "reason": "generated_token_ids_match",
        });
    };

    let Some(left_step) =
        left_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_pre_antepenultimate_block_source_missing_context_row(
            id,
            "left_logits_step_missing",
        );
    };
    let Some(right_step) =
        right_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_pre_antepenultimate_block_source_missing_context_row(
            id,
            "right_logits_step_missing",
        );
    };

    let left_block = &left_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["pre_antepenultimate_block_source"];
    let right_block = &right_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["pre_antepenultimate_block_source"];
    if !left_block.is_object() {
        return generated_output_pre_antepenultimate_block_source_missing_context_row(
            id,
            "left_pre_antepenultimate_block_source_missing",
        );
    }
    if !right_block.is_object() {
        return generated_output_pre_antepenultimate_block_source_missing_context_row(
            id,
            "right_pre_antepenultimate_block_source_missing",
        );
    }

    let (block_input_available, block_input_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "block_input");
    let (attention_output_available, attention_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "attention_output");
    let (post_attention_residual_available, post_attention_residual_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "post_attention_residual");
    let (feed_forward_output_available, feed_forward_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "feed_forward_output");
    let (block_output_available, block_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "block_output");
    let pre_antepenultimate_block_source_context_available = block_input_available
        && attention_output_available
        && post_attention_residual_available
        && feed_forward_output_available
        && block_output_available;

    let classification = if !pre_antepenultimate_block_source_context_available {
        "generated_output_pre_antepenultimate_block_source_missing_context"
    } else if block_input_sha_match == Some(false) {
        "generated_output_pre_antepenultimate_block_source_block_input_drift"
    } else if attention_output_sha_match == Some(false) {
        "generated_output_pre_antepenultimate_block_source_attention_output_drift"
    } else if post_attention_residual_sha_match == Some(false) {
        "generated_output_pre_antepenultimate_block_source_attention_residual_drift"
    } else if feed_forward_output_sha_match == Some(false) {
        "generated_output_pre_antepenultimate_block_source_ffn_output_drift"
    } else if block_output_sha_match == Some(false) {
        "generated_output_pre_antepenultimate_block_source_block_output_drift"
    } else {
        "generated_output_pre_antepenultimate_block_source_clean"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_chosen_id": left_step["chosen_id"],
        "right_chosen_id": right_step["chosen_id"],
        "pre_antepenultimate_block_source_context_available": pre_antepenultimate_block_source_context_available,
        "block_input_context_available": block_input_available,
        "attention_output_context_available": attention_output_available,
        "post_attention_residual_context_available": post_attention_residual_available,
        "feed_forward_output_context_available": feed_forward_output_available,
        "block_output_context_available": block_output_available,
        "block_input_sha256_match": block_input_sha_match,
        "attention_output_sha256_match": attention_output_sha_match,
        "post_attention_residual_sha256_match": post_attention_residual_sha_match,
        "feed_forward_output_sha256_match": feed_forward_output_sha_match,
        "block_output_sha256_match": block_output_sha_match,
        "left_block_input_sha256_f32_le": left_block["block_input"]["sha256_f32_le"],
        "right_block_input_sha256_f32_le": right_block["block_input"]["sha256_f32_le"],
        "left_block_input_rms": left_block["block_input"]["rms"],
        "right_block_input_rms": right_block["block_input"]["rms"],
        "block_input_rms_abs_delta": number_abs_delta(&left_block["block_input"]["rms"], &right_block["block_input"]["rms"]),
        "left_attention_output_sha256_f32_le": left_block["attention_output"]["sha256_f32_le"],
        "right_attention_output_sha256_f32_le": right_block["attention_output"]["sha256_f32_le"],
        "left_attention_output_rms": left_block["attention_output"]["rms"],
        "right_attention_output_rms": right_block["attention_output"]["rms"],
        "attention_output_rms_abs_delta": number_abs_delta(&left_block["attention_output"]["rms"], &right_block["attention_output"]["rms"]),
        "left_post_attention_residual_sha256_f32_le": left_block["post_attention_residual"]["sha256_f32_le"],
        "right_post_attention_residual_sha256_f32_le": right_block["post_attention_residual"]["sha256_f32_le"],
        "left_post_attention_residual_rms": left_block["post_attention_residual"]["rms"],
        "right_post_attention_residual_rms": right_block["post_attention_residual"]["rms"],
        "post_attention_residual_rms_abs_delta": number_abs_delta(&left_block["post_attention_residual"]["rms"], &right_block["post_attention_residual"]["rms"]),
        "left_feed_forward_output_sha256_f32_le": left_block["feed_forward_output"]["sha256_f32_le"],
        "right_feed_forward_output_sha256_f32_le": right_block["feed_forward_output"]["sha256_f32_le"],
        "left_feed_forward_output_rms": left_block["feed_forward_output"]["rms"],
        "right_feed_forward_output_rms": right_block["feed_forward_output"]["rms"],
        "feed_forward_output_rms_abs_delta": number_abs_delta(&left_block["feed_forward_output"]["rms"], &right_block["feed_forward_output"]["rms"]),
        "left_block_output_sha256_f32_le": left_block["block_output"]["sha256_f32_le"],
        "right_block_output_sha256_f32_le": right_block["block_output"]["sha256_f32_le"],
        "left_block_output_rms": left_block["block_output"]["rms"],
        "right_block_output_rms": right_block["block_output"]["rms"],
        "block_output_rms_abs_delta": number_abs_delta(&left_block["block_output"]["rms"], &right_block["block_output"]["rms"]),
        "next_diagnostic": pre_antepenultimate_block_source_next_diagnostic(classification),
    })
}

fn pre_antepenultimate_block_source_next_diagnostic(classification: &str) -> &'static str {
    match classification {
        "generated_output_pre_antepenultimate_block_source_frontier_block_input_drift"
        | "generated_output_pre_antepenultimate_block_source_block_input_drift" => {
            "capture earlier transformer block source frontier"
        }
        "generated_output_pre_antepenultimate_block_source_frontier_attention_output_drift"
        | "generated_output_pre_antepenultimate_block_source_attention_output_drift" => {
            "replay pre-antepenultimate transformer block attention output source"
        }
        "generated_output_pre_antepenultimate_block_source_frontier_attention_residual_drift"
        | "generated_output_pre_antepenultimate_block_source_attention_residual_drift" => {
            "inspect pre-antepenultimate transformer block attention residual add serialization"
        }
        "generated_output_pre_antepenultimate_block_source_frontier_ffn_output_drift"
        | "generated_output_pre_antepenultimate_block_source_ffn_output_drift" => {
            "replay pre-antepenultimate transformer block FFN output source"
        }
        "generated_output_pre_antepenultimate_block_source_frontier_block_output_drift"
        | "generated_output_pre_antepenultimate_block_source_block_output_drift" => {
            "inspect pre-antepenultimate transformer block FFN residual add serialization"
        }
        "generated_output_pre_antepenultimate_block_source_frontier_missing_context"
        | "generated_output_pre_antepenultimate_block_source_missing_context" => {
            "rerun focused receipts with pre_antepenultimate_block_source context enabled"
        }
        _ => "none",
    }
}

fn generated_output_pre_antepenultimate_block_source_missing_context_row(
    id: &str,
    reason: &str,
) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_pre_antepenultimate_block_source_missing_context",
        "reason": reason,
        "pre_antepenultimate_block_source_context_available": false,
    })
}

fn build_generated_output_earlier_block_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut clean_count = 0usize;
    let mut block_input_drift_count = 0usize;
    let mut attention_output_drift_count = 0usize;
    let mut attention_residual_drift_count = 0usize;
    let mut ffn_output_drift_count = 0usize;
    let mut block_output_drift_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_earlier_block_source_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        match row["classification"].as_str() {
            Some("generated_output_earlier_block_source_block_input_drift") => {
                block_input_drift_count += 1;
            }
            Some("generated_output_earlier_block_source_attention_output_drift") => {
                attention_output_drift_count += 1;
            }
            Some("generated_output_earlier_block_source_attention_residual_drift") => {
                attention_residual_drift_count += 1;
            }
            Some("generated_output_earlier_block_source_ffn_output_drift") => {
                ffn_output_drift_count += 1;
            }
            Some("generated_output_earlier_block_source_block_output_drift") => {
                block_output_drift_count += 1;
            }
            Some("generated_output_earlier_block_source_clean") => clean_count += 1,
            Some("generated_output_earlier_block_source_not_applicable") => continue,
            _ => missing_context_count += 1,
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_earlier_block_source_frontier_missing_context"
    } else if block_input_drift_count > 0 {
        "generated_output_earlier_block_source_frontier_block_input_drift"
    } else if attention_output_drift_count > 0 {
        "generated_output_earlier_block_source_frontier_attention_output_drift"
    } else if attention_residual_drift_count > 0 {
        "generated_output_earlier_block_source_frontier_attention_residual_drift"
    } else if ffn_output_drift_count > 0 {
        "generated_output_earlier_block_source_frontier_ffn_output_drift"
    } else if block_output_drift_count > 0 {
        "generated_output_earlier_block_source_frontier_block_output_drift"
    } else {
        "generated_output_earlier_block_source_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "clean_count": clean_count,
        "block_input_drift_count": block_input_drift_count,
        "attention_output_drift_count": attention_output_drift_count,
        "attention_residual_drift_count": attention_residual_drift_count,
        "ffn_output_drift_count": ffn_output_drift_count,
        "block_output_drift_count": block_output_drift_count,
        "missing_context_count": missing_context_count,
        "earlier_block_source_context_available": rows.iter().any(|row| {
            row["earlier_block_source_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": earlier_block_source_next_diagnostic(classification),
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_earlier_block_source_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let Some(left_case) = left_case else {
        return generated_output_earlier_block_source_missing_context_row(id, "left_case_missing");
    };
    let Some(right_case) = right_case else {
        return generated_output_earlier_block_source_missing_context_row(id, "right_case_missing");
    };
    let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
        return generated_output_earlier_block_source_missing_context_row(
            id,
            "left_generated_token_ids_missing",
        );
    };
    let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
        return generated_output_earlier_block_source_missing_context_row(
            id,
            "right_generated_token_ids_missing",
        );
    };
    let Some(first_mismatch_index) = first_different_token_index(&left_generated, &right_generated)
    else {
        return json!({
            "case_id": id,
            "classification": "generated_output_earlier_block_source_not_applicable",
            "reason": "generated_token_ids_match",
        });
    };

    let Some(left_step) =
        left_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_earlier_block_source_missing_context_row(
            id,
            "left_logits_step_missing",
        );
    };
    let Some(right_step) =
        right_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_earlier_block_source_missing_context_row(
            id,
            "right_logits_step_missing",
        );
    };

    let left_block = &left_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["earlier_block_source"];
    let right_block = &right_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["earlier_block_source"];
    if !left_block.is_object() {
        return generated_output_earlier_block_source_missing_context_row(
            id,
            "left_earlier_block_source_missing",
        );
    }
    if !right_block.is_object() {
        return generated_output_earlier_block_source_missing_context_row(
            id,
            "right_earlier_block_source_missing",
        );
    }

    let (block_input_available, block_input_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "block_input");
    let (attention_output_available, attention_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "attention_output");
    let (post_attention_residual_available, post_attention_residual_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "post_attention_residual");
    let (feed_forward_output_available, feed_forward_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "feed_forward_output");
    let (block_output_available, block_output_sha_match) =
        final_block_tensor_pair_status(left_block, right_block, "block_output");
    let earlier_block_source_context_available = block_input_available
        && attention_output_available
        && post_attention_residual_available
        && feed_forward_output_available
        && block_output_available;

    let classification = if !earlier_block_source_context_available {
        "generated_output_earlier_block_source_missing_context"
    } else if block_input_sha_match == Some(false) {
        "generated_output_earlier_block_source_block_input_drift"
    } else if attention_output_sha_match == Some(false) {
        "generated_output_earlier_block_source_attention_output_drift"
    } else if post_attention_residual_sha_match == Some(false) {
        "generated_output_earlier_block_source_attention_residual_drift"
    } else if feed_forward_output_sha_match == Some(false) {
        "generated_output_earlier_block_source_ffn_output_drift"
    } else if block_output_sha_match == Some(false) {
        "generated_output_earlier_block_source_block_output_drift"
    } else {
        "generated_output_earlier_block_source_clean"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_chosen_id": left_step["chosen_id"],
        "right_chosen_id": right_step["chosen_id"],
        "earlier_block_source_context_available": earlier_block_source_context_available,
        "block_input_context_available": block_input_available,
        "attention_output_context_available": attention_output_available,
        "post_attention_residual_context_available": post_attention_residual_available,
        "feed_forward_output_context_available": feed_forward_output_available,
        "block_output_context_available": block_output_available,
        "block_input_sha256_match": block_input_sha_match,
        "attention_output_sha256_match": attention_output_sha_match,
        "post_attention_residual_sha256_match": post_attention_residual_sha_match,
        "feed_forward_output_sha256_match": feed_forward_output_sha_match,
        "block_output_sha256_match": block_output_sha_match,
        "left_block_input_sha256_f32_le": left_block["block_input"]["sha256_f32_le"],
        "right_block_input_sha256_f32_le": right_block["block_input"]["sha256_f32_le"],
        "left_block_input_rms": left_block["block_input"]["rms"],
        "right_block_input_rms": right_block["block_input"]["rms"],
        "block_input_rms_abs_delta": number_abs_delta(&left_block["block_input"]["rms"], &right_block["block_input"]["rms"]),
        "left_attention_output_sha256_f32_le": left_block["attention_output"]["sha256_f32_le"],
        "right_attention_output_sha256_f32_le": right_block["attention_output"]["sha256_f32_le"],
        "left_attention_output_rms": left_block["attention_output"]["rms"],
        "right_attention_output_rms": right_block["attention_output"]["rms"],
        "attention_output_rms_abs_delta": number_abs_delta(&left_block["attention_output"]["rms"], &right_block["attention_output"]["rms"]),
        "left_post_attention_residual_sha256_f32_le": left_block["post_attention_residual"]["sha256_f32_le"],
        "right_post_attention_residual_sha256_f32_le": right_block["post_attention_residual"]["sha256_f32_le"],
        "left_post_attention_residual_rms": left_block["post_attention_residual"]["rms"],
        "right_post_attention_residual_rms": right_block["post_attention_residual"]["rms"],
        "post_attention_residual_rms_abs_delta": number_abs_delta(&left_block["post_attention_residual"]["rms"], &right_block["post_attention_residual"]["rms"]),
        "left_feed_forward_output_sha256_f32_le": left_block["feed_forward_output"]["sha256_f32_le"],
        "right_feed_forward_output_sha256_f32_le": right_block["feed_forward_output"]["sha256_f32_le"],
        "left_feed_forward_output_rms": left_block["feed_forward_output"]["rms"],
        "right_feed_forward_output_rms": right_block["feed_forward_output"]["rms"],
        "feed_forward_output_rms_abs_delta": number_abs_delta(&left_block["feed_forward_output"]["rms"], &right_block["feed_forward_output"]["rms"]),
        "left_block_output_sha256_f32_le": left_block["block_output"]["sha256_f32_le"],
        "right_block_output_sha256_f32_le": right_block["block_output"]["sha256_f32_le"],
        "left_block_output_rms": left_block["block_output"]["rms"],
        "right_block_output_rms": right_block["block_output"]["rms"],
        "block_output_rms_abs_delta": number_abs_delta(&left_block["block_output"]["rms"], &right_block["block_output"]["rms"]),
        "next_diagnostic": earlier_block_source_next_diagnostic(classification),
    })
}

fn earlier_block_source_next_diagnostic(classification: &str) -> &'static str {
    match classification {
        "generated_output_earlier_block_source_frontier_block_input_drift"
        | "generated_output_earlier_block_source_block_input_drift" => {
            "capture preceding transformer block source frontier"
        }
        "generated_output_earlier_block_source_frontier_attention_output_drift"
        | "generated_output_earlier_block_source_attention_output_drift" => {
            "replay earlier transformer block attention output source"
        }
        "generated_output_earlier_block_source_frontier_attention_residual_drift"
        | "generated_output_earlier_block_source_attention_residual_drift" => {
            "inspect earlier transformer block attention residual add serialization"
        }
        "generated_output_earlier_block_source_frontier_ffn_output_drift"
        | "generated_output_earlier_block_source_ffn_output_drift" => {
            "replay earlier transformer block FFN output source"
        }
        "generated_output_earlier_block_source_frontier_block_output_drift"
        | "generated_output_earlier_block_source_block_output_drift" => {
            "inspect earlier transformer block FFN residual add serialization"
        }
        "generated_output_earlier_block_source_frontier_missing_context"
        | "generated_output_earlier_block_source_missing_context" => {
            "rerun focused receipts with earlier_block_source context enabled"
        }
        _ => "none",
    }
}

fn generated_output_earlier_block_source_missing_context_row(id: &str, reason: &str) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_earlier_block_source_missing_context",
        "reason": reason,
        "earlier_block_source_context_available": false,
    })
}

fn build_generated_output_transformer_block_source_stack_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut clean_count = 0usize;
    let mut block_input_drift_count = 0usize;
    let mut attention_output_drift_count = 0usize;
    let mut attention_residual_drift_count = 0usize;
    let mut ffn_output_drift_count = 0usize;
    let mut block_output_drift_count = 0usize;
    let mut missing_context_count = 0usize;
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_transformer_block_source_stack_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        match row["classification"].as_str() {
            Some("generated_output_transformer_block_source_stack_block_input_drift") => {
                block_input_drift_count += 1;
            }
            Some("generated_output_transformer_block_source_stack_attention_output_drift") => {
                attention_output_drift_count += 1;
            }
            Some("generated_output_transformer_block_source_stack_attention_residual_drift") => {
                attention_residual_drift_count += 1;
            }
            Some("generated_output_transformer_block_source_stack_ffn_output_drift") => {
                ffn_output_drift_count += 1;
            }
            Some("generated_output_transformer_block_source_stack_block_output_drift") => {
                block_output_drift_count += 1;
            }
            Some("generated_output_transformer_block_source_stack_clean") => clean_count += 1,
            Some("generated_output_transformer_block_source_stack_not_applicable") => continue,
            _ => missing_context_count += 1,
        }
        row_candidate_count += 1;
        push_limited_row(&mut rows, ROW_LIMIT, row);
    }

    let classification = if missing_context_count > 0 {
        "generated_output_transformer_block_source_stack_frontier_missing_context"
    } else if block_input_drift_count > 0 {
        "generated_output_transformer_block_source_stack_frontier_block_input_drift"
    } else if attention_output_drift_count > 0 {
        "generated_output_transformer_block_source_stack_frontier_attention_output_drift"
    } else if attention_residual_drift_count > 0 {
        "generated_output_transformer_block_source_stack_frontier_attention_residual_drift"
    } else if ffn_output_drift_count > 0 {
        "generated_output_transformer_block_source_stack_frontier_ffn_output_drift"
    } else if block_output_drift_count > 0 {
        "generated_output_transformer_block_source_stack_frontier_block_output_drift"
    } else {
        "generated_output_transformer_block_source_stack_frontier_clean"
    };

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "clean_count": clean_count,
        "block_input_drift_count": block_input_drift_count,
        "attention_output_drift_count": attention_output_drift_count,
        "attention_residual_drift_count": attention_residual_drift_count,
        "ffn_output_drift_count": ffn_output_drift_count,
        "block_output_drift_count": block_output_drift_count,
        "missing_context_count": missing_context_count,
        "transformer_block_source_stack_context_available": rows.iter().any(|row| {
            row["transformer_block_source_stack_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": transformer_block_source_stack_next_diagnostic(classification),
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_transformer_block_source_stack_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let Some(left_case) = left_case else {
        return generated_output_transformer_block_source_stack_missing_context_row(
            id,
            "left_case_missing",
        );
    };
    let Some(right_case) = right_case else {
        return generated_output_transformer_block_source_stack_missing_context_row(
            id,
            "right_case_missing",
        );
    };
    let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
        return generated_output_transformer_block_source_stack_missing_context_row(
            id,
            "left_generated_token_ids_missing",
        );
    };
    let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
        return generated_output_transformer_block_source_stack_missing_context_row(
            id,
            "right_generated_token_ids_missing",
        );
    };
    let Some(first_mismatch_index) = first_different_token_index(&left_generated, &right_generated)
    else {
        return json!({
            "case_id": id,
            "classification": "generated_output_transformer_block_source_stack_not_applicable",
            "reason": "generated_token_ids_match",
        });
    };

    let Some(left_step) =
        left_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_transformer_block_source_stack_missing_context_row(
            id,
            "left_logits_step_missing",
        );
    };
    let Some(right_step) =
        right_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_transformer_block_source_stack_missing_context_row(
            id,
            "right_logits_step_missing",
        );
    };

    let left_stack = &left_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["block_sources"];
    let right_stack = &right_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["block_sources"];
    let Some(left_blocks) = left_stack["blocks"].as_array() else {
        return generated_output_transformer_block_source_stack_missing_context_row(
            id,
            "left_transformer_block_source_stack_missing",
        );
    };
    let Some(right_blocks) = right_stack["blocks"].as_array() else {
        return generated_output_transformer_block_source_stack_missing_context_row(
            id,
            "right_transformer_block_source_stack_missing",
        );
    };
    if left_blocks.is_empty() || right_blocks.is_empty() {
        return generated_output_transformer_block_source_stack_missing_context_row(
            id,
            "transformer_block_source_stack_empty",
        );
    }
    if left_blocks.len() != right_blocks.len() {
        return generated_output_transformer_block_source_stack_missing_context_row(
            id,
            "transformer_block_source_stack_length_mismatch",
        );
    }

    for (block_index, (left_block, right_block)) in
        left_blocks.iter().zip(right_blocks.iter()).enumerate()
    {
        let left_layer_idx = left_block["layer_idx"].as_u64();
        let right_layer_idx = right_block["layer_idx"].as_u64();
        if left_layer_idx != right_layer_idx {
            return generated_output_transformer_block_source_stack_missing_context_row(
                id,
                "transformer_block_source_stack_layer_mismatch",
            );
        }
        let (block_input_available, block_input_sha_match) =
            final_block_tensor_pair_status(left_block, right_block, "block_input");
        let (attention_output_available, attention_output_sha_match) =
            final_block_tensor_pair_status(left_block, right_block, "attention_output");
        let (post_attention_residual_available, post_attention_residual_sha_match) =
            final_block_tensor_pair_status(left_block, right_block, "post_attention_residual");
        let (feed_forward_output_available, feed_forward_output_sha_match) =
            final_block_tensor_pair_status(left_block, right_block, "feed_forward_output");
        let (block_output_available, block_output_sha_match) =
            final_block_tensor_pair_status(left_block, right_block, "block_output");
        let block_context_available = block_input_available
            && attention_output_available
            && post_attention_residual_available
            && feed_forward_output_available
            && block_output_available;

        let classification = if !block_context_available {
            "generated_output_transformer_block_source_stack_missing_context"
        } else if block_input_sha_match == Some(false) {
            "generated_output_transformer_block_source_stack_block_input_drift"
        } else if attention_output_sha_match == Some(false) {
            "generated_output_transformer_block_source_stack_attention_output_drift"
        } else if post_attention_residual_sha_match == Some(false) {
            "generated_output_transformer_block_source_stack_attention_residual_drift"
        } else if feed_forward_output_sha_match == Some(false) {
            "generated_output_transformer_block_source_stack_ffn_output_drift"
        } else if block_output_sha_match == Some(false) {
            "generated_output_transformer_block_source_stack_block_output_drift"
        } else {
            continue;
        };

        return json!({
            "case_id": id,
            "classification": classification,
            "first_mismatch_index": first_mismatch_index,
            "left_token_id": left_generated.get(first_mismatch_index).copied(),
            "right_token_id": right_generated.get(first_mismatch_index).copied(),
            "left_chosen_id": left_step["chosen_id"],
            "right_chosen_id": right_step["chosen_id"],
            "transformer_block_source_stack_context_available": true,
            "compared_block_count": left_blocks.len(),
            "earliest_divergent_block_index": block_index,
            "earliest_divergent_layer_idx": left_layer_idx,
            "block_context_available": block_context_available,
            "block_input_context_available": block_input_available,
            "attention_output_context_available": attention_output_available,
            "post_attention_residual_context_available": post_attention_residual_available,
            "feed_forward_output_context_available": feed_forward_output_available,
            "block_output_context_available": block_output_available,
            "block_input_sha256_match": block_input_sha_match,
            "attention_output_sha256_match": attention_output_sha_match,
            "post_attention_residual_sha256_match": post_attention_residual_sha_match,
            "feed_forward_output_sha256_match": feed_forward_output_sha_match,
            "block_output_sha256_match": block_output_sha_match,
            "left_block_input_sha256_f32_le": left_block["block_input"]["sha256_f32_le"],
            "right_block_input_sha256_f32_le": right_block["block_input"]["sha256_f32_le"],
            "left_block_input_rms": left_block["block_input"]["rms"],
            "right_block_input_rms": right_block["block_input"]["rms"],
            "block_input_rms_abs_delta": number_abs_delta(&left_block["block_input"]["rms"], &right_block["block_input"]["rms"]),
            "left_attention_output_sha256_f32_le": left_block["attention_output"]["sha256_f32_le"],
            "right_attention_output_sha256_f32_le": right_block["attention_output"]["sha256_f32_le"],
            "left_attention_output_rms": left_block["attention_output"]["rms"],
            "right_attention_output_rms": right_block["attention_output"]["rms"],
            "attention_output_rms_abs_delta": number_abs_delta(&left_block["attention_output"]["rms"], &right_block["attention_output"]["rms"]),
            "left_post_attention_residual_sha256_f32_le": left_block["post_attention_residual"]["sha256_f32_le"],
            "right_post_attention_residual_sha256_f32_le": right_block["post_attention_residual"]["sha256_f32_le"],
            "left_post_attention_residual_rms": left_block["post_attention_residual"]["rms"],
            "right_post_attention_residual_rms": right_block["post_attention_residual"]["rms"],
            "post_attention_residual_rms_abs_delta": number_abs_delta(&left_block["post_attention_residual"]["rms"], &right_block["post_attention_residual"]["rms"]),
            "left_feed_forward_output_sha256_f32_le": left_block["feed_forward_output"]["sha256_f32_le"],
            "right_feed_forward_output_sha256_f32_le": right_block["feed_forward_output"]["sha256_f32_le"],
            "left_feed_forward_output_rms": left_block["feed_forward_output"]["rms"],
            "right_feed_forward_output_rms": right_block["feed_forward_output"]["rms"],
            "feed_forward_output_rms_abs_delta": number_abs_delta(&left_block["feed_forward_output"]["rms"], &right_block["feed_forward_output"]["rms"]),
            "left_block_output_sha256_f32_le": left_block["block_output"]["sha256_f32_le"],
            "right_block_output_sha256_f32_le": right_block["block_output"]["sha256_f32_le"],
            "left_block_output_rms": left_block["block_output"]["rms"],
            "right_block_output_rms": right_block["block_output"]["rms"],
            "block_output_rms_abs_delta": number_abs_delta(&left_block["block_output"]["rms"], &right_block["block_output"]["rms"]),
            "next_diagnostic": transformer_block_source_stack_next_diagnostic(classification),
        });
    }

    json!({
        "case_id": id,
        "classification": "generated_output_transformer_block_source_stack_clean",
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_chosen_id": left_step["chosen_id"],
        "right_chosen_id": right_step["chosen_id"],
        "transformer_block_source_stack_context_available": true,
        "compared_block_count": left_blocks.len(),
        "next_diagnostic": transformer_block_source_stack_next_diagnostic(
            "generated_output_transformer_block_source_stack_clean",
        ),
    })
}

fn transformer_block_source_stack_next_diagnostic(classification: &str) -> &'static str {
    match classification {
        "generated_output_transformer_block_source_stack_frontier_block_input_drift"
        | "generated_output_transformer_block_source_stack_block_input_drift" => {
            "inspect transformer stack input source before earliest divergent block"
        }
        "generated_output_transformer_block_source_stack_frontier_attention_output_drift"
        | "generated_output_transformer_block_source_stack_attention_output_drift" => {
            "replay earliest divergent transformer block attention output source"
        }
        "generated_output_transformer_block_source_stack_frontier_attention_residual_drift"
        | "generated_output_transformer_block_source_stack_attention_residual_drift" => {
            "inspect earliest divergent transformer block attention residual add serialization"
        }
        "generated_output_transformer_block_source_stack_frontier_ffn_output_drift"
        | "generated_output_transformer_block_source_stack_ffn_output_drift" => {
            "replay earliest divergent transformer block FFN output source"
        }
        "generated_output_transformer_block_source_stack_frontier_block_output_drift"
        | "generated_output_transformer_block_source_stack_block_output_drift" => {
            "inspect earliest divergent transformer block FFN residual add serialization"
        }
        "generated_output_transformer_block_source_stack_frontier_missing_context"
        | "generated_output_transformer_block_source_stack_missing_context" => {
            "rerun focused receipts with transformer block source stack context enabled"
        }
        _ => "none",
    }
}

fn generated_output_transformer_block_source_stack_missing_context_row(
    id: &str,
    reason: &str,
) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_transformer_block_source_stack_missing_context",
        "reason": reason,
        "transformer_block_source_stack_context_available": false,
    })
}

fn build_generated_output_attention_output_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut classification_counts = BTreeMap::<String, usize>::new();
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_attention_output_source_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        let classification = row["classification"]
            .as_str()
            .unwrap_or("generated_output_attention_output_source_missing_context");
        if classification != "generated_output_attention_output_source_not_applicable" {
            row_candidate_count += 1;
            *classification_counts.entry(classification.to_string()).or_default() += 1;
            push_limited_row(&mut rows, ROW_LIMIT, row);
        }
    }

    let priority = [
        "generated_output_attention_output_source_missing_context",
        "generated_output_attention_output_source_attention_input_drift",
        "generated_output_attention_output_source_q_projection_drift",
        "generated_output_attention_output_source_k_projection_drift",
        "generated_output_attention_output_source_v_projection_drift",
        "generated_output_attention_output_source_q_heads_drift",
        "generated_output_attention_output_source_k_heads_drift",
        "generated_output_attention_output_source_v_heads_drift",
        "generated_output_attention_output_source_q_norm_drift",
        "generated_output_attention_output_source_k_norm_drift",
        "generated_output_attention_output_source_q_rope_drift",
        "generated_output_attention_output_source_k_rope_drift",
        "generated_output_attention_output_source_k_context_drift",
        "generated_output_attention_output_source_v_context_drift",
        "generated_output_attention_output_source_expanded_k_drift",
        "generated_output_attention_output_source_expanded_v_drift",
        "generated_output_attention_output_source_scores_drift",
        "generated_output_attention_output_source_probabilities_drift",
        "generated_output_attention_output_source_value_mix_drift",
        "generated_output_attention_output_source_output_projection_input_drift",
        "generated_output_attention_output_source_sub_layernorm_drift",
        "generated_output_attention_output_source_output_projection_drift",
        "generated_output_attention_output_source_clean",
    ];
    let row_classification = priority
        .iter()
        .find(|classification| classification_counts.contains_key::<str>(*classification))
        .copied()
        .unwrap_or("generated_output_attention_output_source_clean");
    let classification = row_classification.replace(
        "generated_output_attention_output_source_",
        "generated_output_attention_output_source_frontier_",
    );

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "classification_counts": classification_counts,
        "attention_output_source_context_available": rows.iter().any(|row| {
            row["attention_output_source_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": attention_output_source_next_diagnostic(&classification),
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_attention_output_source_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let block_row = generated_output_transformer_block_source_stack_row(id, left_case, right_case);
    let block_classification = block_row["classification"].as_str();
    if block_classification
        != Some("generated_output_transformer_block_source_stack_attention_output_drift")
    {
        return json!({
            "case_id": id,
            "classification": "generated_output_attention_output_source_not_applicable",
            "reason": "transformer_block_source_stack_not_attention_output_drift",
            "transformer_block_source_stack_classification": block_row["classification"],
        });
    }

    let Some(left_case) = left_case else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "left_case_missing",
            block_row,
        );
    };
    let Some(right_case) = right_case else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "right_case_missing",
            block_row,
        );
    };
    let Some(left_generated) = token_id_vec(&left_case["token_ids"]["generated"]) else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "left_generated_token_ids_missing",
            block_row,
        );
    };
    let Some(right_generated) = token_id_vec(&right_case["token_ids"]["generated"]) else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "right_generated_token_ids_missing",
            block_row,
        );
    };
    let Some(first_mismatch_index) = first_different_token_index(&left_generated, &right_generated)
    else {
        return json!({
            "case_id": id,
            "classification": "generated_output_attention_output_source_not_applicable",
            "reason": "generated_token_ids_match",
            "transformer_block_source_stack_classification": block_row["classification"],
        });
    };
    let Some(target_layer_idx) = block_row["earliest_divergent_layer_idx"].as_u64() else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "earliest_divergent_layer_idx_missing",
            block_row,
        );
    };

    let Some(left_step) =
        left_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "left_logits_step_missing",
            block_row,
        );
    };
    let Some(right_step) =
        right_case["logits_dump"].as_array().and_then(|steps| steps.get(first_mismatch_index))
    else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "right_logits_step_missing",
            block_row,
        );
    };

    let left_stack = &left_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["attention_output_sources"];
    let right_stack = &right_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["attention_output_sources"];
    let Some(left_sources) = left_stack["sources"].as_array() else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "left_attention_output_sources_missing",
            block_row,
        );
    };
    let Some(right_sources) = right_stack["sources"].as_array() else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "right_attention_output_sources_missing",
            block_row,
        );
    };
    let Some(left_source) = attention_output_source_by_layer(left_sources, target_layer_idx) else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "left_target_layer_attention_output_source_missing",
            block_row,
        );
    };
    let Some(right_source) = attention_output_source_by_layer(right_sources, target_layer_idx)
    else {
        return generated_output_attention_output_source_missing_context_row(
            id,
            "right_target_layer_attention_output_source_missing",
            block_row,
        );
    };

    let fields = [
        ("attention_input", "attention_input_drift"),
        ("q_projection", "q_projection_drift"),
        ("k_projection", "k_projection_drift"),
        ("v_projection", "v_projection_drift"),
        ("q_heads", "q_heads_drift"),
        ("k_heads", "k_heads_drift"),
        ("v_heads", "v_heads_drift"),
        ("q_norm", "q_norm_drift"),
        ("k_norm", "k_norm_drift"),
        ("q_rope", "q_rope_drift"),
        ("k_rope", "k_rope_drift"),
        ("k_context", "k_context_drift"),
        ("v_context", "v_context_drift"),
        ("expanded_k", "expanded_k_drift"),
        ("expanded_v", "expanded_v_drift"),
        ("scores", "scores_drift"),
        ("probabilities", "probabilities_drift"),
        ("value_mix_output_heads", "value_mix_drift"),
        ("output_projection_input", "output_projection_input_drift"),
        ("sub_layernorm_output", "sub_layernorm_drift"),
        ("attention_output", "output_projection_drift"),
    ];

    let mut field_rows = Vec::new();
    let mut first_drift = None;
    let mut missing_field_context = None;
    for (field, drift_label) in fields {
        let (available, sha_match) =
            final_block_tensor_pair_status(left_source, right_source, field);
        field_rows.push(json!({
            "field": field,
            "available": available,
            "sha256_match": sha_match,
            "left_sha256_f32_le": left_source[field]["sha256_f32_le"],
            "right_sha256_f32_le": right_source[field]["sha256_f32_le"],
            "left_rms": left_source[field]["rms"],
            "right_rms": right_source[field]["rms"],
            "rms_abs_delta": number_abs_delta(&left_source[field]["rms"], &right_source[field]["rms"]),
        }));
        if !available && field != "sub_layernorm_output" {
            missing_field_context = Some(field);
            break;
        }
        if sha_match == Some(false) {
            first_drift = Some(drift_label);
            break;
        }
    }

    let classification = if missing_field_context.is_some() {
        "generated_output_attention_output_source_missing_context".to_string()
    } else if let Some(first_drift) = first_drift {
        format!("generated_output_attention_output_source_{first_drift}")
    } else {
        "generated_output_attention_output_source_clean".to_string()
    };

    json!({
        "case_id": id,
        "classification": classification,
        "reason": missing_field_context,
        "first_mismatch_index": first_mismatch_index,
        "left_token_id": left_generated.get(first_mismatch_index).copied(),
        "right_token_id": right_generated.get(first_mismatch_index).copied(),
        "left_chosen_id": left_step["chosen_id"],
        "right_chosen_id": right_step["chosen_id"],
        "target_layer_idx": target_layer_idx,
        "transformer_block_source_stack_classification": block_row["classification"],
        "attention_output_source_context_available": missing_field_context.is_none(),
        "left_attention_output_source_count": left_sources.len(),
        "right_attention_output_source_count": right_sources.len(),
        "fields": field_rows,
        "next_diagnostic": attention_output_source_next_diagnostic(&classification),
    })
}

fn attention_output_source_by_layer(sources: &[Value], layer_idx: u64) -> Option<&Value> {
    sources.iter().find(|source| source["layer_idx"].as_u64() == Some(layer_idx))
}

fn generated_output_attention_output_source_missing_context_row(
    id: &str,
    reason: &str,
    block_row: Value,
) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_attention_output_source_missing_context",
        "reason": reason,
        "transformer_block_source_stack_classification": block_row["classification"],
        "attention_output_source_context_available": false,
        "next_diagnostic": attention_output_source_next_diagnostic(
            "generated_output_attention_output_source_missing_context",
        ),
    })
}

fn attention_output_source_next_diagnostic(classification: &str) -> &'static str {
    match classification {
        "generated_output_attention_output_source_frontier_attention_input_drift"
        | "generated_output_attention_output_source_attention_input_drift" => {
            "replay earliest divergent block attention RMSNorm output"
        }
        "generated_output_attention_output_source_frontier_q_projection_drift"
        | "generated_output_attention_output_source_frontier_k_projection_drift"
        | "generated_output_attention_output_source_frontier_v_projection_drift"
        | "generated_output_attention_output_source_q_projection_drift"
        | "generated_output_attention_output_source_k_projection_drift"
        | "generated_output_attention_output_source_v_projection_drift" => {
            "replay earliest divergent block QKV projection source"
        }
        "generated_output_attention_output_source_frontier_q_norm_drift"
        | "generated_output_attention_output_source_frontier_k_norm_drift"
        | "generated_output_attention_output_source_q_norm_drift"
        | "generated_output_attention_output_source_k_norm_drift" => {
            "replay earliest divergent block QK norm source"
        }
        "generated_output_attention_output_source_frontier_q_rope_drift"
        | "generated_output_attention_output_source_frontier_k_rope_drift"
        | "generated_output_attention_output_source_q_rope_drift"
        | "generated_output_attention_output_source_k_rope_drift" => {
            "replay earliest divergent block RoPE source"
        }
        "generated_output_attention_output_source_frontier_scores_drift"
        | "generated_output_attention_output_source_scores_drift" => {
            "replay earliest divergent block raw attention score source"
        }
        "generated_output_attention_output_source_frontier_probabilities_drift"
        | "generated_output_attention_output_source_probabilities_drift" => {
            "replay earliest divergent block softmax probability source"
        }
        "generated_output_attention_output_source_frontier_value_mix_drift"
        | "generated_output_attention_output_source_value_mix_drift" => {
            "replay earliest divergent block value-mix source"
        }
        "generated_output_attention_output_source_frontier_output_projection_input_drift"
        | "generated_output_attention_output_source_frontier_sub_layernorm_drift"
        | "generated_output_attention_output_source_frontier_output_projection_drift"
        | "generated_output_attention_output_source_output_projection_input_drift"
        | "generated_output_attention_output_source_sub_layernorm_drift"
        | "generated_output_attention_output_source_output_projection_drift" => {
            "replay earliest divergent block attention output projection source"
        }
        "generated_output_attention_output_source_frontier_missing_context"
        | "generated_output_attention_output_source_missing_context" => {
            "rerun focused receipts with attention output source context enabled"
        }
        _ => "none",
    }
}

fn build_generated_output_qkv_projection_source_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut classification_counts = BTreeMap::<String, usize>::new();
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_qkv_projection_source_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        let classification = row["classification"]
            .as_str()
            .unwrap_or("generated_output_qkv_projection_source_missing_context");
        if classification != "generated_output_qkv_projection_source_not_applicable" {
            row_candidate_count += 1;
            *classification_counts.entry(classification.to_string()).or_default() += 1;
            push_limited_row(&mut rows, ROW_LIMIT, row);
        }
    }

    let priority = [
        "generated_output_qkv_projection_source_missing_context",
        "generated_output_qkv_projection_source_projection_input_drift",
        "generated_output_qkv_projection_source_projection_metadata_drift",
        "generated_output_qkv_projection_source_dispatch_path_drift",
        "generated_output_qkv_projection_source_projection_output_drift",
        "generated_output_qkv_projection_source_clean",
    ];
    let row_classification = priority
        .iter()
        .find(|classification| classification_counts.contains_key::<str>(*classification))
        .copied()
        .unwrap_or("generated_output_qkv_projection_source_clean");
    let classification = row_classification.replace(
        "generated_output_qkv_projection_source_",
        "generated_output_qkv_projection_source_frontier_",
    );

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "classification_counts": classification_counts,
        "qkv_projection_source_context_available": rows.iter().any(|row| {
            row["qkv_projection_source_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": qkv_projection_source_next_diagnostic(&classification),
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn build_generated_output_qkv_projection_dispatch_replay_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut classification_counts = BTreeMap::<String, usize>::new();
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_qkv_projection_dispatch_replay_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        let classification = row["classification"]
            .as_str()
            .unwrap_or("generated_output_qkv_projection_dispatch_replay_missing_context");
        if classification != "generated_output_qkv_projection_dispatch_replay_not_applicable" {
            row_candidate_count += 1;
            *classification_counts.entry(classification.to_string()).or_default() += 1;
            push_limited_row(&mut rows, ROW_LIMIT, row);
        }
    }

    let priority = [
        "generated_output_qkv_projection_dispatch_replay_missing_context",
        "generated_output_qkv_projection_dispatch_replay_runtime_replay_mismatch",
        "generated_output_qkv_projection_dispatch_replay_cpu_a770_output_drift",
        "generated_output_qkv_projection_dispatch_replay_clean",
    ];
    let row_classification = priority
        .iter()
        .find(|classification| classification_counts.contains_key::<str>(*classification))
        .copied()
        .unwrap_or("generated_output_qkv_projection_dispatch_replay_clean");
    let classification = row_classification.replace(
        "generated_output_qkv_projection_dispatch_replay_",
        "generated_output_qkv_projection_dispatch_replay_frontier_",
    );

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "classification_counts": classification_counts,
        "qkv_projection_dispatch_replay_context_available": rows.iter().any(|row| {
            row["qkv_projection_dispatch_replay_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": qkv_projection_dispatch_replay_next_diagnostic(&classification),
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn build_generated_output_qk256_numeric_policy_frontier(
    case_ids: &BTreeSet<String>,
    left_cases: &BTreeMap<String, &Value>,
    right_cases: &BTreeMap<String, &Value>,
    left_label: &str,
    right_label: &str,
) -> Value {
    const ROW_LIMIT: usize = 16;

    let mut rows = Vec::new();
    let mut classification_counts = BTreeMap::<String, usize>::new();
    let mut row_candidate_count = 0usize;

    for id in case_ids {
        let row = generated_output_qk256_numeric_policy_row(
            id,
            left_cases.get(id).copied(),
            right_cases.get(id).copied(),
        );
        let classification = row["classification"]
            .as_str()
            .unwrap_or("generated_output_qk256_numeric_policy_missing_context");
        if classification != "generated_output_qk256_numeric_policy_not_applicable" {
            row_candidate_count += 1;
            *classification_counts.entry(classification.to_string()).or_default() += 1;
            push_limited_row(&mut rows, ROW_LIMIT, row);
        }
    }

    let priority = [
        "generated_output_qk256_numeric_policy_missing_context",
        "generated_output_qk256_numeric_policy_raw_input_materialization",
        "generated_output_qk256_numeric_policy_packed_weight_decode",
        "generated_output_qk256_numeric_policy_scale_application",
        "generated_output_qk256_numeric_policy_accumulation_order",
        "generated_output_qk256_numeric_policy_output_casting_serialization",
        "generated_output_qk256_numeric_policy_clean",
    ];
    let row_classification = priority
        .iter()
        .find(|classification| classification_counts.contains_key::<str>(*classification))
        .copied()
        .unwrap_or("generated_output_qk256_numeric_policy_clean");
    let classification = row_classification.replace(
        "generated_output_qk256_numeric_policy_",
        "generated_output_qk256_numeric_policy_frontier_",
    );

    json!({
        "classification": classification,
        "left_label": left_label,
        "right_label": right_label,
        "case_count": case_ids.len(),
        "classification_counts": classification_counts,
        "qk256_numeric_policy_context_available": rows.iter().any(|row| {
            row["qk256_numeric_policy_context_available"].as_bool().unwrap_or(false)
        }),
        "next_diagnostic": qk256_numeric_policy_next_diagnostic(&classification),
        "rows_truncated": row_candidate_count > rows.len(),
        "row_limit": ROW_LIMIT,
        "rows": rows,
    })
}

fn generated_output_qkv_projection_dispatch_replay_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let source_row = generated_output_qkv_projection_source_row(id, left_case, right_case);
    let source_classification = source_row["classification"]
        .as_str()
        .unwrap_or("generated_output_qkv_projection_source_missing_context");
    if source_classification == "generated_output_qkv_projection_source_not_applicable" {
        return json!({
            "case_id": id,
            "classification": "generated_output_qkv_projection_dispatch_replay_not_applicable",
            "reason": "qkv_projection_source_not_applicable",
            "qkv_projection_source_classification": source_row["classification"],
        });
    }
    if source_classification == "generated_output_qkv_projection_source_missing_context" {
        let reason = source_row["reason"]
            .as_str()
            .unwrap_or("qkv_projection_source_missing_context")
            .to_string();
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id, &reason, source_row,
        );
    }

    let Some(first_mismatch_index) = source_row["first_mismatch_index"].as_u64() else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "first_mismatch_index_missing",
            source_row,
        );
    };
    let Some(target_layer_idx) = source_row["target_layer_idx"].as_u64() else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "target_layer_idx_missing",
            source_row,
        );
    };
    let Some(projection) = source_row["projection"].as_str() else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "projection_missing",
            source_row,
        );
    };
    let Some(left_case) = left_case else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "left_case_missing",
            source_row,
        );
    };
    let Some(right_case) = right_case else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "right_case_missing",
            source_row,
        );
    };
    let Some(left_step) = left_case["logits_dump"]
        .as_array()
        .and_then(|steps| steps.get(first_mismatch_index as usize))
    else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "left_logits_step_missing",
            source_row,
        );
    };
    let Some(right_step) = right_case["logits_dump"]
        .as_array()
        .and_then(|steps| steps.get(first_mismatch_index as usize))
    else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "right_logits_step_missing",
            source_row,
        );
    };

    let Some(left_sources) = left_step["logit_source_context"]["hidden_state_source"]
        ["model_forward_source"]["qkv_projection_sources"]["sources"]
        .as_array()
    else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "left_qkv_projection_sources_missing",
            source_row,
        );
    };
    let Some(right_sources) = right_step["logit_source_context"]["hidden_state_source"]
        ["model_forward_source"]["qkv_projection_sources"]["sources"]
        .as_array()
    else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "right_qkv_projection_sources_missing",
            source_row,
        );
    };
    let Some(left_source) =
        qkv_projection_source_by_layer_projection(left_sources, target_layer_idx, projection)
    else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "left_target_qkv_projection_source_missing",
            source_row,
        );
    };
    let Some(right_source) =
        qkv_projection_source_by_layer_projection(right_sources, target_layer_idx, projection)
    else {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            "right_target_qkv_projection_source_missing",
            source_row,
        );
    };
    let left_replay = &left_source["dispatch_replay"];
    let right_replay = &right_source["dispatch_replay"];
    let left_replay_available = qkv_projection_dispatch_replay_available(left_replay);
    let right_replay_available = qkv_projection_dispatch_replay_available(right_replay);
    if !left_replay_available || !right_replay_available {
        return generated_output_qkv_projection_dispatch_replay_missing_context_row(
            id,
            if !left_replay_available {
                "left_dispatch_replay_missing"
            } else {
                "right_dispatch_replay_missing"
            },
            source_row,
        );
    }

    let left_runtime_cpu_match =
        optional_json_sha_eq(&left_source["output"], &left_replay["cpu_output"]);
    let right_runtime_a770_match =
        optional_json_sha_eq(&right_source["output"], &right_replay["a770_output"]);
    let left_cpu_a770_match =
        optional_json_sha_eq(&left_replay["cpu_output"], &left_replay["a770_output"]);
    let right_cpu_a770_match =
        optional_json_sha_eq(&right_replay["cpu_output"], &right_replay["a770_output"]);
    let cpu_replay_match =
        optional_json_sha_eq(&left_replay["cpu_output"], &right_replay["cpu_output"]);
    let a770_replay_match =
        optional_json_sha_eq(&left_replay["a770_output"], &right_replay["a770_output"]);

    let classification =
        if left_runtime_cpu_match == Some(false) || right_runtime_a770_match == Some(false) {
            "generated_output_qkv_projection_dispatch_replay_runtime_replay_mismatch"
        } else if left_cpu_a770_match == Some(false)
            || right_cpu_a770_match == Some(false)
            || cpu_replay_match == Some(false)
            || a770_replay_match == Some(false)
        {
            "generated_output_qkv_projection_dispatch_replay_cpu_a770_output_drift"
        } else {
            "generated_output_qkv_projection_dispatch_replay_clean"
        };

    json!({
        "case_id": id,
        "classification": classification,
        "reason": Value::Null,
        "first_mismatch_index": first_mismatch_index,
        "target_layer_idx": target_layer_idx,
        "projection": projection,
        "qkv_projection_source_classification": source_row["classification"],
        "qkv_projection_dispatch_replay_context_available": true,
        "left_runtime_output_matches_cpu_replay": left_runtime_cpu_match,
        "right_runtime_output_matches_a770_replay": right_runtime_a770_match,
        "left_cpu_a770_replay_output_match": left_cpu_a770_match,
        "right_cpu_a770_replay_output_match": right_cpu_a770_match,
        "cpu_replay_output_match_across_receipts": cpu_replay_match,
        "a770_replay_output_match_across_receipts": a770_replay_match,
        "left_replay": qkv_projection_dispatch_replay_summary(left_replay),
        "right_replay": qkv_projection_dispatch_replay_summary(right_replay),
        "next_diagnostic": qkv_projection_dispatch_replay_next_diagnostic(classification),
    })
}

fn generated_output_qk256_numeric_policy_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let replay_row = generated_output_qkv_projection_dispatch_replay_row(id, left_case, right_case);
    let replay_classification = replay_row["classification"]
        .as_str()
        .unwrap_or("generated_output_qkv_projection_dispatch_replay_missing_context");
    if replay_classification == "generated_output_qkv_projection_dispatch_replay_not_applicable" {
        return json!({
            "case_id": id,
            "classification": "generated_output_qk256_numeric_policy_not_applicable",
            "reason": "qkv_projection_dispatch_replay_not_applicable",
            "qkv_projection_dispatch_replay_classification": replay_row["classification"],
        });
    }

    let source_classification =
        replay_row["qkv_projection_source_classification"].as_str().unwrap_or_default();
    if source_classification == "generated_output_qkv_projection_source_projection_input_drift" {
        return generated_output_qk256_numeric_policy_context_row(
            id,
            "generated_output_qk256_numeric_policy_raw_input_materialization",
            "qkv_projection_input_sha_drift",
            replay_row,
        );
    }
    if source_classification == "generated_output_qkv_projection_source_projection_metadata_drift" {
        return generated_output_qk256_numeric_policy_context_row(
            id,
            "generated_output_qk256_numeric_policy_packed_weight_decode",
            "qk256_projection_metadata_drift",
            replay_row,
        );
    }

    if replay_classification == "generated_output_qkv_projection_dispatch_replay_missing_context" {
        let reason =
            replay_row["reason"].as_str().unwrap_or("dispatch_replay_missing_context").to_string();
        return generated_output_qk256_numeric_policy_context_row(
            id,
            "generated_output_qk256_numeric_policy_missing_context",
            &reason,
            replay_row,
        );
    }
    if replay_classification
        == "generated_output_qkv_projection_dispatch_replay_runtime_replay_mismatch"
    {
        return generated_output_qk256_numeric_policy_context_row(
            id,
            "generated_output_qk256_numeric_policy_missing_context",
            "runtime_replay_mismatch_blocks_numeric_policy_classification",
            replay_row,
        );
    }

    let Some(left_replay) = replay_row.get("left_replay") else {
        return generated_output_qk256_numeric_policy_context_row(
            id,
            "generated_output_qk256_numeric_policy_missing_context",
            "left_replay_summary_missing",
            replay_row,
        );
    };
    let Some(right_replay) = replay_row.get("right_replay") else {
        return generated_output_qk256_numeric_policy_context_row(
            id,
            "generated_output_qk256_numeric_policy_missing_context",
            "right_replay_summary_missing",
            replay_row,
        );
    };

    if left_replay["inline_scale"].is_null() || right_replay["inline_scale"].is_null() {
        return generated_output_qk256_numeric_policy_context_row(
            id,
            "generated_output_qk256_numeric_policy_scale_application",
            "inline_scale_missing",
            replay_row,
        );
    }
    if left_replay["inline_scale"] != right_replay["inline_scale"] {
        return generated_output_qk256_numeric_policy_context_row(
            id,
            "generated_output_qk256_numeric_policy_scale_application",
            "inline_scale_mismatch_across_receipts",
            replay_row,
        );
    }

    let left_cpu_opencl_policy_match =
        optional_json_sha_eq(&left_replay["cpu_output"], &left_replay["opencl_policy_output"]);
    let right_cpu_opencl_policy_match =
        optional_json_sha_eq(&right_replay["cpu_output"], &right_replay["opencl_policy_output"]);
    let left_opencl_policy_a770_match =
        optional_json_sha_eq(&left_replay["opencl_policy_output"], &left_replay["a770_output"]);
    let right_opencl_policy_a770_match =
        optional_json_sha_eq(&right_replay["opencl_policy_output"], &right_replay["a770_output"]);
    let opencl_policy_match_across_receipts = optional_json_sha_eq(
        &left_replay["opencl_policy_output"],
        &right_replay["opencl_policy_output"],
    );

    if [
        left_cpu_opencl_policy_match,
        right_cpu_opencl_policy_match,
        left_opencl_policy_a770_match,
        right_opencl_policy_a770_match,
        opencl_policy_match_across_receipts,
    ]
    .iter()
    .any(Option::is_none)
    {
        return generated_output_qk256_numeric_policy_context_row(
            id,
            "generated_output_qk256_numeric_policy_missing_context",
            "numeric_policy_output_fingerprint_missing",
            replay_row,
        );
    }

    let cpu_differs_from_opencl_policy =
        left_cpu_opencl_policy_match == Some(false) || right_cpu_opencl_policy_match == Some(false);
    let a770_matches_opencl_policy = left_opencl_policy_a770_match == Some(true)
        && right_opencl_policy_a770_match == Some(true)
        && opencl_policy_match_across_receipts == Some(true);
    let all_clean = left_cpu_opencl_policy_match == Some(true)
        && right_cpu_opencl_policy_match == Some(true)
        && a770_matches_opencl_policy;

    let classification = if all_clean {
        "generated_output_qk256_numeric_policy_clean"
    } else if cpu_differs_from_opencl_policy && a770_matches_opencl_policy {
        "generated_output_qk256_numeric_policy_accumulation_order"
    } else if !cpu_differs_from_opencl_policy && !a770_matches_opencl_policy {
        "generated_output_qk256_numeric_policy_output_casting_serialization"
    } else {
        "generated_output_qk256_numeric_policy_packed_weight_decode"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "reason": Value::Null,
        "first_mismatch_index": replay_row["first_mismatch_index"],
        "target_layer_idx": replay_row["target_layer_idx"],
        "projection": replay_row["projection"],
        "qkv_projection_source_classification": replay_row["qkv_projection_source_classification"],
        "qkv_projection_dispatch_replay_classification": replay_row["classification"],
        "qk256_numeric_policy_context_available": true,
        "left_cpu_opencl_policy_output_match": left_cpu_opencl_policy_match,
        "right_cpu_opencl_policy_output_match": right_cpu_opencl_policy_match,
        "left_opencl_policy_a770_output_match": left_opencl_policy_a770_match,
        "right_opencl_policy_a770_output_match": right_opencl_policy_a770_match,
        "opencl_policy_output_match_across_receipts": opencl_policy_match_across_receipts,
        "left_replay": left_replay,
        "right_replay": right_replay,
        "next_diagnostic": qk256_numeric_policy_next_diagnostic(classification),
    })
}

fn generated_output_qk256_numeric_policy_context_row(
    id: &str,
    classification: &str,
    reason: &str,
    replay_row: Value,
) -> Value {
    json!({
        "case_id": id,
        "classification": classification,
        "reason": reason,
        "qkv_projection_source_classification": replay_row["qkv_projection_source_classification"],
        "qkv_projection_dispatch_replay_classification": replay_row["classification"],
        "qk256_numeric_policy_context_available":
            classification != "generated_output_qk256_numeric_policy_missing_context",
        "left_replay": replay_row["left_replay"],
        "right_replay": replay_row["right_replay"],
        "next_diagnostic": qk256_numeric_policy_next_diagnostic(classification),
    })
}

fn qkv_projection_dispatch_replay_available(replay: &Value) -> bool {
    replay["source_context_available"].as_bool().unwrap_or(false)
        && replay["cpu_output"]["sha256_f32_le"].as_str().is_some()
        && replay["opencl_policy_output"]["sha256_f32_le"].as_str().is_some()
        && replay["a770_output"]["sha256_f32_le"].as_str().is_some()
}

fn qkv_projection_dispatch_replay_summary(replay: &Value) -> Value {
    json!({
        "input_rows": replay["input_rows"],
        "output_rows": replay["output_rows"],
        "cols": replay["cols"],
        "row_stride_bytes": replay["row_stride_bytes"],
        "inline_scale": replay["inline_scale"],
        "cpu_output": {
            "sha256_f32_le": replay["cpu_output"]["sha256_f32_le"],
            "rms": replay["cpu_output"]["rms"],
        },
        "opencl_policy_output": {
            "sha256_f32_le": replay["opencl_policy_output"]["sha256_f32_le"],
            "rms": replay["opencl_policy_output"]["rms"],
        },
        "a770_output": {
            "sha256_f32_le": replay["a770_output"]["sha256_f32_le"],
            "rms": replay["a770_output"]["rms"],
        },
        "cpu_a770_output_sha256_match": replay["cpu_a770_output_sha256_match"],
        "cpu_opencl_policy_output_sha256_match": replay["cpu_opencl_policy_output_sha256_match"],
        "opencl_policy_a770_output_sha256_match": replay["opencl_policy_a770_output_sha256_match"],
        "cpu_a770_output_rms_abs_delta": replay["cpu_a770_output_rms_abs_delta"],
        "cpu_opencl_policy_output_rms_abs_delta": replay["cpu_opencl_policy_output_rms_abs_delta"],
        "opencl_policy_a770_output_rms_abs_delta": replay["opencl_policy_a770_output_rms_abs_delta"],
        "numeric_policy": replay["numeric_policy"],
        "cpu": replay["cpu"],
        "a770": replay["a770"],
    })
}

fn generated_output_qkv_projection_dispatch_replay_missing_context_row(
    id: &str,
    reason: &str,
    source_row: Value,
) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_qkv_projection_dispatch_replay_missing_context",
        "reason": reason,
        "qkv_projection_source_classification": source_row["classification"],
        "qkv_projection_dispatch_replay_context_available": false,
        "next_diagnostic": qkv_projection_dispatch_replay_next_diagnostic(
            "generated_output_qkv_projection_dispatch_replay_missing_context",
        ),
    })
}

fn generated_output_qkv_projection_source_row(
    id: &str,
    left_case: Option<&Value>,
    right_case: Option<&Value>,
) -> Value {
    let attention_row = generated_output_attention_output_source_row(id, left_case, right_case);
    let Some((attention_field, projection)) =
        qkv_projection_from_attention_classification(&attention_row["classification"])
    else {
        return json!({
            "case_id": id,
            "classification": "generated_output_qkv_projection_source_not_applicable",
            "reason": "attention_output_source_not_qkv_projection_drift",
            "attention_output_source_classification": attention_row["classification"],
        });
    };

    let Some(left_case) = left_case else {
        return generated_output_qkv_projection_source_missing_context_row(
            id,
            "left_case_missing",
            attention_row,
        );
    };
    let Some(right_case) = right_case else {
        return generated_output_qkv_projection_source_missing_context_row(
            id,
            "right_case_missing",
            attention_row,
        );
    };
    let Some(first_mismatch_index) = attention_row["first_mismatch_index"].as_u64() else {
        return generated_output_qkv_projection_source_missing_context_row(
            id,
            "first_mismatch_index_missing",
            attention_row,
        );
    };
    let Some(target_layer_idx) = attention_row["target_layer_idx"].as_u64() else {
        return generated_output_qkv_projection_source_missing_context_row(
            id,
            "target_layer_idx_missing",
            attention_row,
        );
    };
    let Some(left_step) = left_case["logits_dump"]
        .as_array()
        .and_then(|steps| steps.get(first_mismatch_index as usize))
    else {
        return generated_output_qkv_projection_source_missing_context_row(
            id,
            "left_logits_step_missing",
            attention_row,
        );
    };
    let Some(right_step) = right_case["logits_dump"]
        .as_array()
        .and_then(|steps| steps.get(first_mismatch_index as usize))
    else {
        return generated_output_qkv_projection_source_missing_context_row(
            id,
            "right_logits_step_missing",
            attention_row,
        );
    };

    let left_stack = &left_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["qkv_projection_sources"];
    let right_stack = &right_step["logit_source_context"]["hidden_state_source"]["model_forward_source"]
        ["qkv_projection_sources"];
    let Some(left_sources) = left_stack["sources"].as_array() else {
        return generated_output_qkv_projection_source_missing_context_row(
            id,
            "left_qkv_projection_sources_missing",
            attention_row,
        );
    };
    let Some(right_sources) = right_stack["sources"].as_array() else {
        return generated_output_qkv_projection_source_missing_context_row(
            id,
            "right_qkv_projection_sources_missing",
            attention_row,
        );
    };
    let Some(left_source) =
        qkv_projection_source_by_layer_projection(left_sources, target_layer_idx, projection)
    else {
        return generated_output_qkv_projection_source_missing_context_row(
            id,
            "left_target_qkv_projection_source_missing",
            attention_row,
        );
    };
    let Some(right_source) =
        qkv_projection_source_by_layer_projection(right_sources, target_layer_idx, projection)
    else {
        return generated_output_qkv_projection_source_missing_context_row(
            id,
            "right_target_qkv_projection_source_missing",
            attention_row,
        );
    };

    let (input_available, input_sha_match) =
        final_block_tensor_pair_status(left_source, right_source, "input");
    let (output_available, output_sha_match) =
        final_block_tensor_pair_status(left_source, right_source, "output");
    let metadata_match = qkv_projection_source_metadata_signature(left_source)
        == qkv_projection_source_metadata_signature(right_source);
    let dispatch_match = qkv_projection_source_dispatch_signature(left_source)
        == qkv_projection_source_dispatch_signature(right_source);

    let classification = if !input_available || !output_available {
        "generated_output_qkv_projection_source_missing_context"
    } else if input_sha_match == Some(false) {
        "generated_output_qkv_projection_source_projection_input_drift"
    } else if !metadata_match {
        "generated_output_qkv_projection_source_projection_metadata_drift"
    } else if !dispatch_match {
        "generated_output_qkv_projection_source_dispatch_path_drift"
    } else if output_sha_match == Some(false) {
        "generated_output_qkv_projection_source_projection_output_drift"
    } else {
        "generated_output_qkv_projection_source_clean"
    };

    json!({
        "case_id": id,
        "classification": classification,
        "reason": if input_available && output_available { Value::Null } else { json!("projection_input_or_output_missing") },
        "first_mismatch_index": first_mismatch_index,
        "target_layer_idx": target_layer_idx,
        "attention_output_source_classification": attention_row["classification"],
        "attention_output_source_field": attention_field,
        "projection": projection,
        "qkv_projection_source_context_available": input_available && output_available,
        "input": {
            "available": input_available,
            "sha256_match": input_sha_match,
            "left_sha256_f32_le": left_source["input"]["sha256_f32_le"],
            "right_sha256_f32_le": right_source["input"]["sha256_f32_le"],
            "left_rms": left_source["input"]["rms"],
            "right_rms": right_source["input"]["rms"],
            "rms_abs_delta": number_abs_delta(&left_source["input"]["rms"], &right_source["input"]["rms"]),
        },
        "output": {
            "available": output_available,
            "sha256_match": output_sha_match,
            "left_sha256_f32_le": left_source["output"]["sha256_f32_le"],
            "right_sha256_f32_le": right_source["output"]["sha256_f32_le"],
            "left_rms": left_source["output"]["rms"],
            "right_rms": right_source["output"]["rms"],
            "rms_abs_delta": number_abs_delta(&left_source["output"]["rms"], &right_source["output"]["rms"]),
        },
        "metadata_match": metadata_match,
        "dispatch_match": dispatch_match,
        "left_metadata": qkv_projection_source_metadata_summary(left_source),
        "right_metadata": qkv_projection_source_metadata_summary(right_source),
        "left_dispatch": qkv_projection_source_dispatch_summary(left_source),
        "right_dispatch": qkv_projection_source_dispatch_summary(right_source),
        "next_diagnostic": qkv_projection_source_next_diagnostic(classification),
    })
}

fn qkv_projection_from_attention_classification(
    classification: &Value,
) -> Option<(&'static str, &'static str)> {
    match classification.as_str()? {
        "generated_output_attention_output_source_frontier_q_projection_drift"
        | "generated_output_attention_output_source_q_projection_drift" => {
            Some(("q_projection", "q_proj"))
        }
        "generated_output_attention_output_source_frontier_k_projection_drift"
        | "generated_output_attention_output_source_k_projection_drift" => {
            Some(("k_projection", "k_proj"))
        }
        "generated_output_attention_output_source_frontier_v_projection_drift"
        | "generated_output_attention_output_source_v_projection_drift" => {
            Some(("v_projection", "v_proj"))
        }
        _ => None,
    }
}

fn qkv_projection_source_by_layer_projection<'a>(
    sources: &'a [Value],
    layer_idx: u64,
    projection: &str,
) -> Option<&'a Value> {
    sources.iter().find(|source| {
        source["layer_idx"].as_u64() == Some(layer_idx)
            && source["projection"].as_str() == Some(projection)
    })
}

fn qkv_projection_source_metadata_signature(source: &Value) -> Vec<String> {
    vec![
        source["tensor_name"].as_str().unwrap_or_default().to_string(),
        source["qk256_key"].as_str().unwrap_or_default().to_string(),
        source["qk256_raw_tensor_present"].as_bool().unwrap_or(false).to_string(),
    ]
}

fn qkv_projection_source_dispatch_signature(source: &Value) -> Vec<String> {
    let dispatch = &source["dispatch_delta"];
    let cpu = &source["cpu_hot_path_delta"];
    let a770 = &source["a770_opencl_runtime_delta"];
    vec![
        dispatch["execution_claim"].as_str().unwrap_or_default().to_string(),
        dispatch["bitnet_linear_layers_total"].as_u64().unwrap_or(0).to_string(),
        dispatch["bitnet_linear_layers_on_cuda"].as_u64().unwrap_or(0).to_string(),
        dispatch["bitnet_linear_layers_on_a770_opencl"].as_u64().unwrap_or(0).to_string(),
        dispatch["bitnet_linear_layers_cpu_fallback"].as_u64().unwrap_or(0).to_string(),
        cpu["qk256_f32_scalar_gemv_invocations"].as_u64().unwrap_or(0).to_string(),
        cpu["qk256_f32_avx2_gemv_invocations"].as_u64().unwrap_or(0).to_string(),
        cpu["qk256_i8s_scaled_scalar_invocations"].as_u64().unwrap_or(0).to_string(),
        cpu["qk256_i8s_scaled_avx2_gemv_invocations"].as_u64().unwrap_or(0).to_string(),
        a770["kernel_invocations"].as_u64().unwrap_or(0).to_string(),
    ]
}

fn qkv_projection_source_metadata_summary(source: &Value) -> Value {
    json!({
        "tensor_name": source["tensor_name"],
        "qk256_key": source["qk256_key"],
        "qk256_raw_tensor_present": source["qk256_raw_tensor_present"],
    })
}

fn qkv_projection_source_dispatch_summary(source: &Value) -> Value {
    json!({
        "dispatch_delta": source["dispatch_delta"],
        "cpu_hot_path_delta": source["cpu_hot_path_delta"],
        "a770_opencl_runtime_delta": source["a770_opencl_runtime_delta"],
    })
}

fn generated_output_qkv_projection_source_missing_context_row(
    id: &str,
    reason: &str,
    attention_row: Value,
) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_qkv_projection_source_missing_context",
        "reason": reason,
        "attention_output_source_classification": attention_row["classification"],
        "qkv_projection_source_context_available": false,
        "next_diagnostic": qkv_projection_source_next_diagnostic(
            "generated_output_qkv_projection_source_missing_context",
        ),
    })
}

fn qkv_projection_source_next_diagnostic(classification: &str) -> &'static str {
    match classification {
        "generated_output_qkv_projection_source_frontier_projection_input_drift"
        | "generated_output_qkv_projection_source_projection_input_drift" => {
            "replay earliest divergent block attention input source"
        }
        "generated_output_qkv_projection_source_frontier_projection_metadata_drift"
        | "generated_output_qkv_projection_source_projection_metadata_drift" => {
            "inspect selected QKV projection tensor naming and raw QK256 source metadata"
        }
        "generated_output_qkv_projection_source_frontier_dispatch_path_drift"
        | "generated_output_qkv_projection_source_dispatch_path_drift" => {
            "replay selected QKV projection CPU versus A770 dispatch policy"
        }
        "generated_output_qkv_projection_source_frontier_projection_output_drift"
        | "generated_output_qkv_projection_source_projection_output_drift" => {
            "replay selected QKV projection numeric output against shared dispatch metadata"
        }
        "generated_output_qkv_projection_source_frontier_missing_context"
        | "generated_output_qkv_projection_source_missing_context" => {
            "rerun focused receipts with QKV projection source context enabled"
        }
        _ => "none",
    }
}

fn qkv_projection_dispatch_replay_next_diagnostic(classification: &str) -> &'static str {
    match classification {
        "generated_output_qkv_projection_dispatch_replay_frontier_cpu_a770_output_drift"
        | "generated_output_qkv_projection_dispatch_replay_cpu_a770_output_drift" => {
            "inspect selected QK256 CPU scalar versus A770 OpenCL GEMV numeric policy"
        }
        "generated_output_qkv_projection_dispatch_replay_frontier_runtime_replay_mismatch"
        | "generated_output_qkv_projection_dispatch_replay_runtime_replay_mismatch" => {
            "inspect selected QKV projection dispatch replay capture scope"
        }
        "generated_output_qkv_projection_dispatch_replay_frontier_missing_context"
        | "generated_output_qkv_projection_dispatch_replay_missing_context" => {
            "rerun focused receipts with QKV projection dispatch replay enabled"
        }
        _ => "none",
    }
}

fn qk256_numeric_policy_next_diagnostic(classification: &str) -> &'static str {
    match classification {
        "generated_output_qk256_numeric_policy_frontier_raw_input_materialization"
        | "generated_output_qk256_numeric_policy_raw_input_materialization" => {
            "inspect selected QK256 replay input materialization before numeric policy"
        }
        "generated_output_qk256_numeric_policy_frontier_packed_weight_decode"
        | "generated_output_qk256_numeric_policy_packed_weight_decode" => {
            "inspect selected QK256 packed weight decode and byte-order policy"
        }
        "generated_output_qk256_numeric_policy_frontier_scale_application"
        | "generated_output_qk256_numeric_policy_scale_application" => {
            "inspect selected QK256 inline scale and activation scale application"
        }
        "generated_output_qk256_numeric_policy_frontier_accumulation_order"
        | "generated_output_qk256_numeric_policy_accumulation_order" => {
            "align or gate selected QK256 OpenCL accumulation policy after before/after receipts"
        }
        "generated_output_qk256_numeric_policy_frontier_output_casting_serialization"
        | "generated_output_qk256_numeric_policy_output_casting_serialization" => {
            "inspect selected QK256 OpenCL output casting and receipt serialization"
        }
        "generated_output_qk256_numeric_policy_frontier_missing_context"
        | "generated_output_qk256_numeric_policy_missing_context" => {
            "rerun focused receipts with QK256 numeric policy replay context enabled"
        }
        _ => "none",
    }
}

fn optional_json_sha_eq(left: &Value, right: &Value) -> Option<bool> {
    Some(left["sha256_f32_le"].as_str()? == right["sha256_f32_le"].as_str()?)
}

fn optional_str_eq(left: Option<&str>, right: Option<&str>) -> Option<bool> {
    Some(left? == right?)
}

fn number_abs_delta(left: &Value, right: &Value) -> Value {
    match (left.as_f64(), right.as_f64()) {
        (Some(left), Some(right)) => json!((left - right).abs()),
        _ => Value::Null,
    }
}

fn generated_output_argmax_source_missing_context_row(id: &str, reason: &str) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_argmax_source_missing_context",
        "reason": reason,
    })
}

fn generated_output_argmax_source_prompt_history_row(
    id: &str,
    left_case: &Value,
    right_case: &Value,
) -> Value {
    let left_prompt = token_id_vec(&left_case["token_ids"]["prompt"]).unwrap_or_default();
    let right_prompt = token_id_vec(&right_case["token_ids"]["prompt"]).unwrap_or_default();
    json!({
        "case_id": id,
        "classification": "generated_output_argmax_source_prompt_history_serialization",
        "prompt_token_ids_match": false,
        "first_prompt_mismatch_index": first_different_token_index(&left_prompt, &right_prompt),
        "left_prompt_len": left_prompt.len(),
        "right_prompt_len": right_prompt.len(),
        "left_generated_len": token_id_vec(&left_case["token_ids"]["generated"]).map(|tokens| tokens.len()),
        "right_generated_len": token_id_vec(&right_case["token_ids"]["generated"]).map(|tokens| tokens.len()),
    })
}

fn generated_output_argmax_source_row(
    id: &str,
    left_case: &Value,
    right_case: &Value,
    left_generated: &[u64],
    right_generated: &[u64],
    first_mismatch_index: usize,
) -> Value {
    let left_steps = left_case["logits_dump"].as_array();
    let right_steps = right_case["logits_dump"].as_array();
    let Some(left_step) = left_steps.and_then(|steps| steps.get(first_mismatch_index)) else {
        return generated_output_argmax_source_trace_row(
            id,
            first_mismatch_index,
            "left_logits_step_missing",
            left_steps.map(Vec::len),
            right_steps.map(Vec::len),
        );
    };
    let Some(right_step) = right_steps.and_then(|steps| steps.get(first_mismatch_index)) else {
        return generated_output_argmax_source_trace_row(
            id,
            first_mismatch_index,
            "right_logits_step_missing",
            left_steps.map(Vec::len),
            right_steps.map(Vec::len),
        );
    };
    let Some(left_chosen_id) = left_step["chosen_id"].as_u64() else {
        return generated_output_argmax_source_trace_row(
            id,
            first_mismatch_index,
            "left_chosen_id_missing",
            left_steps.map(Vec::len),
            right_steps.map(Vec::len),
        );
    };
    let Some(right_chosen_id) = right_step["chosen_id"].as_u64() else {
        return generated_output_argmax_source_trace_row(
            id,
            first_mismatch_index,
            "right_chosen_id_missing",
            left_steps.map(Vec::len),
            right_steps.map(Vec::len),
        );
    };
    let Some(left_top1_id) = top_logit_token_id(&left_step["top_logits"]) else {
        return generated_output_argmax_source_trace_row(
            id,
            first_mismatch_index,
            "left_top_logits_missing",
            left_steps.map(Vec::len),
            right_steps.map(Vec::len),
        );
    };
    let Some(right_top1_id) = top_logit_token_id(&right_step["top_logits"]) else {
        return generated_output_argmax_source_trace_row(
            id,
            first_mismatch_index,
            "right_top_logits_missing",
            left_steps.map(Vec::len),
            right_steps.map(Vec::len),
        );
    };

    let left_token_id = left_generated.get(first_mismatch_index).copied();
    let right_token_id = right_generated.get(first_mismatch_index).copied();
    let left_generated_matches_chosen = Some(left_chosen_id) == left_token_id;
    let right_generated_matches_chosen = Some(right_chosen_id) == right_token_id;
    let left_chosen_is_top1 = left_chosen_id == left_top1_id;
    let right_chosen_is_top1 = right_chosen_id == right_top1_id;

    let classification = if !left_generated_matches_chosen
        || !right_generated_matches_chosen
        || !left_chosen_is_top1
        || !right_chosen_is_top1
    {
        "generated_output_argmax_source_sampler_logit_extraction_policy"
    } else {
        "generated_output_argmax_source_internal_logit_source_missing_context"
    };

    let margin_row =
        generated_output_logit_margin_row(id, left_case, right_case, first_mismatch_index, 0.01);
    json!({
        "case_id": id,
        "classification": classification,
        "first_mismatch_index": first_mismatch_index,
        "prompt_token_ids_match": true,
        "left_token_id": left_token_id,
        "right_token_id": right_token_id,
        "left_generated_len": left_generated.len(),
        "right_generated_len": right_generated.len(),
        "left_logits_step_count": left_steps.map(Vec::len),
        "right_logits_step_count": right_steps.map(Vec::len),
        "left_chosen_id": left_chosen_id,
        "right_chosen_id": right_chosen_id,
        "same_chosen_id": left_chosen_id == right_chosen_id,
        "left_generated_matches_chosen": left_generated_matches_chosen,
        "right_generated_matches_chosen": right_generated_matches_chosen,
        "left_top1_id": left_top1_id,
        "right_top1_id": right_top1_id,
        "left_chosen_is_top1": left_chosen_is_top1,
        "right_chosen_is_top1": right_chosen_is_top1,
        "first_different_rank_at_first_mismatch": first_different_topk_rank(&left_step["top_logits"], &right_step["top_logits"]),
        "left_topk_count": left_step["top_logits"].as_array().map(Vec::len),
        "right_topk_count": right_step["top_logits"].as_array().map(Vec::len),
        "common_top_token_count": top_logits_common_delta(&left_step["top_logits"], &right_step["top_logits"]).0,
        "max_common_token_abs_delta": top_logits_common_delta(&left_step["top_logits"], &right_step["top_logits"]).1,
        "logit_margin_classification": margin_row["classification"],
        "has_cross_chosen_logits": margin_row["has_cross_chosen_logits"],
        "opposite_argmax": margin_row["opposite_argmax"],
        "left_margin_over_right_chosen_on_left": margin_row["left_margin_over_right_chosen_on_left"],
        "right_margin_over_left_chosen_on_right": margin_row["right_margin_over_left_chosen_on_right"],
        "left_chosen_delta_across_lanes": margin_row["left_chosen_delta_across_lanes"],
        "right_chosen_delta_across_lanes": margin_row["right_chosen_delta_across_lanes"],
        "left_selected_kernel": left_case["kernel"]["selected_kernel"],
        "right_selected_kernel": right_case["kernel"]["selected_kernel"],
        "left_execution_claim": left_case["execution_coverage"]["execution_claim"],
        "right_execution_claim": right_case["execution_coverage"]["execution_claim"],
        "left_qk256_execution_path": left_case["execution_coverage"]["qk256_execution_path"],
        "right_qk256_execution_path": right_case["execution_coverage"]["qk256_execution_path"],
        "qk256_operand_context_available": false,
        "output_head_logit_accumulation_context_available": false,
        "next_diagnostic": if classification == "generated_output_argmax_source_internal_logit_source_missing_context" {
            "capture first-mismatch QK256 operand and output-head logit accumulation context"
        } else {
            "none"
        },
    })
}

fn generated_output_argmax_source_trace_row(
    id: &str,
    first_mismatch_index: usize,
    reason: &str,
    left_logits_step_count: Option<usize>,
    right_logits_step_count: Option<usize>,
) -> Value {
    json!({
        "case_id": id,
        "classification": "generated_output_argmax_source_trace_capture_context_loss",
        "reason": reason,
        "first_mismatch_index": first_mismatch_index,
        "left_logits_step_count": left_logits_step_count,
        "right_logits_step_count": right_logits_step_count,
    })
}

fn top_logit_token_id(top_logits: &Value) -> Option<u64> {
    top_logits.as_array()?.first()?.get("token_id")?.as_u64()
}

fn token_id_vec(value: &Value) -> Option<Vec<u64>> {
    value.as_array()?.iter().map(Value::as_u64).collect()
}

fn first_different_token_index(left: &[u64], right: &[u64]) -> Option<usize> {
    (0..left.len().max(right.len())).find(|&index| left.get(index) != right.get(index))
}

fn case_summary(case: &Value) -> Value {
    json!({
        "status": case["status"],
        "quality_passed": case["quality"]["passed"],
        "quality_failed_rules": case["quality"]["failed_rules"],
        "selected_kernel": case["kernel"]["selected_kernel"],
        "prompt_token_ids": case["token_ids"]["prompt"],
        "generated_token_ids": case["token_ids"]["generated"],
        "answer": case["answer"],
        "logits_steps": case["logits_dump"].as_array().map(Vec::len),
    })
}

fn labeled_case_summary(case: &Value, label: &str) -> Value {
    let mut summary = case_summary(case);
    summary["label"] = json!(label);
    summary["backend"] = case["backend"].clone();
    summary
}

fn case_status_summary(case: &Value) -> Value {
    json!({
        "status": case["status"],
        "quality_failed_rules": case["quality"]["failed_rules"],
        "run_receipt_path": case["run_receipt_path"],
        "exit_code": case["exit_code"],
        "child_process": case["child_process"],
        "child_invocation": {
            "expected_receipt_path": case["child_invocation"]["expected_receipt_path"],
            "timeout_seconds": case["child_invocation"]["timeout_seconds"],
        },
        "reason": case["reason"],
    })
}

#[allow(clippy::too_many_arguments)]
fn set_first(
    first_divergence: &mut Option<Value>,
    id: &str,
    kind: &'static str,
    step: Option<usize>,
    left: Value,
    right: Value,
    left_label: &str,
    right_label: &str,
    legacy_scalar_avx2: bool,
) {
    if first_divergence.is_none() {
        let divergence = if legacy_scalar_avx2 {
            json!({
                "case_id": id,
                "kind": kind,
                "step": step,
                "scalar": left,
                "avx2": right,
                "scope": divergence_scope(kind),
            })
        } else {
            json!({
                "case_id": id,
                "kind": kind,
                "step": step,
                "left_label": left_label,
                "right_label": right_label,
                "left": left,
                "right": right,
                "scope": divergence_scope(kind),
            })
        };
        *first_divergence = Some(divergence);
    }
}

fn divergence_scope(kind: &str) -> &'static str {
    match kind {
        "prompt_token_ids" | "question" | "prompt_template" => "prompt_or_tokenizer",
        "generated_token_ids" => "decode_or_sampler",
        "decoded_text" => "text_decode",
        kind if kind.contains("logits") => "logits_or_kernel",
        kind if kind.contains("kernel_identity") => "kernel_selection",
        _ => "receipt_contract",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(kernel: &str, generated: &[u64], answer: &str, logits: Value) -> Value {
        json!({
            "artifact_kind": "bitnet_cpu_answer_corpus",
            "model": {
                "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
                "file": "ggml-model-i2_s.gguf",
                "path": "models/ggml-model-i2_s.gguf",
                "loader_mode": "real_gguf",
                "tokenizer_path": "models/tokenizer.json"
            },
            "backend": {
                "requested_backend": "cpu",
                "selected_backend": "cpu",
                "runtime_api": "cpu",
                "fallback_used": false
            },
            "prompt_template": { "family": "llama3-chat" },
            "generation": {
                "mode": "greedy",
                "temperature": 0.0,
                "deterministic": true,
                "strict_loader": true,
                "default_max_new_tokens": 1
            },
            "cases": [{
                "id": "math",
                "question": "Answer with a single digit: 2+2=",
                "status": "passed",
                "answer": answer,
                "quality": { "passed": true },
                "backend": {
                    "requested_backend": "cpu",
                    "selected_backend": "cpu-rust",
                    "runtime_api": "cpu",
                    "fallback_used": false
                },
                "tokenizer": {
                    "source": "explicit",
                    "strict": true
                },
                "token_ids": {
                    "prompt": [1, 2, 3],
                    "generated": generated
                },
                "logits_dump": logits,
                "prompt_template": "llama3-chat",
                "kernel": {
                    "selected_kernel": kernel,
                    "family": "i2_s"
                }
            }]
        })
    }

    fn logits() -> Value {
        json!([{
            "step": 0,
            "chosen_id": 4,
            "top_logits": [
                { "token_id": 4, "logit": 10.0 },
                { "token_id": 5, "logit": 1.0 }
            ]
        }])
    }

    fn logits_same_chosen_drift() -> Value {
        json!([{
            "step": 0,
            "chosen_id": 4,
            "top_logits": [
                { "token_id": 4, "logit": 9.75 },
                { "token_id": 5, "logit": 1.5 },
                { "token_id": 6, "logit": 0.25 }
            ]
        }])
    }

    fn logits_different_chosen() -> Value {
        json!([{
            "step": 0,
            "chosen_id": 5,
            "top_logits": [
                { "token_id": 5, "logit": 10.0 },
                { "token_id": 4, "logit": 9.0 }
            ]
        }])
    }

    fn logits_for_chosen(chosen: &[u64]) -> Value {
        Value::Array(
            chosen
                .iter()
                .enumerate()
                .map(|(step, token)| {
                    json!({
                        "step": step,
                        "chosen_id": token,
                        "top_logits": [
                            { "token_id": token, "logit": 10.0 },
                            { "token_id": 999, "logit": 1.0 }
                        ]
                    })
                })
                .collect(),
        )
    }

    fn logits_first_mismatch_margin_left() -> Value {
        json!([
            {
                "step": 0,
                "chosen_id": 4,
                "top_logits": [
                    { "token_id": 4, "logit": 10.0 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            },
            {
                "step": 1,
                "chosen_id": 5,
                "top_logits": [
                    { "token_id": 5, "logit": 10.0 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            },
            {
                "step": 2,
                "chosen_id": 6,
                "top_logits": [
                    { "token_id": 6, "logit": 10.0 },
                    { "token_id": 7, "logit": 9.5 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            }
        ])
    }

    fn logits_first_mismatch_margin_right_near_tie() -> Value {
        json!([
            {
                "step": 0,
                "chosen_id": 4,
                "top_logits": [
                    { "token_id": 4, "logit": 10.0 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            },
            {
                "step": 1,
                "chosen_id": 5,
                "top_logits": [
                    { "token_id": 5, "logit": 10.0 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            },
            {
                "step": 2,
                "chosen_id": 7,
                "top_logits": [
                    { "token_id": 7, "logit": 9.25 },
                    { "token_id": 6, "logit": 9.245 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            }
        ])
    }

    fn logit_source_context(hidden_sha: &str, hidden_rms: f64, on_a770: u64) -> Value {
        json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_output_head_logit_source",
            "diagnostic_only": true,
            "claim_allowed": false,
            "hidden_operand": {
                "available": true,
                "shape": [1, 4],
                "value_count": 4,
                "finite_count": 4,
                "nan_count": 0,
                "infinite_count": 0,
                "sha256_f32_le": hidden_sha,
                "rms": hidden_rms,
                "first_values": [0.1, 0.2, 0.3, 0.4]
            },
            "output_head_qk256_dispatch_delta": {
                "bitnet_linear_layers_total": 1,
                "bitnet_linear_layers_on_cuda": 0,
                "bitnet_linear_layers_on_a770_opencl": on_a770,
                "bitnet_linear_layers_cpu_fallback": 0,
                "unsupported_ops": [],
                "execution_claim": if on_a770 > 0 {
                    "a770_opencl_qk256_contribution"
                } else {
                    "cpu_reference"
                }
            },
            "output_head_a770_opencl_runtime_delta": {
                "host_to_device_bytes": if on_a770 > 0 { 16 } else { 0 },
                "device_to_host_bytes": if on_a770 > 0 { 8 } else { 0 },
                "kernel_invocations": on_a770
            },
            "qk256_operand_context_available": true,
            "output_head_logit_accumulation_context_available": true
        })
    }

    fn logit_source_context_with_hidden_state_source(
        hidden_sha: &str,
        hidden_rms: f64,
        on_a770: u64,
        forward_sha: &str,
        forward_rms: f64,
        last_hidden_sha: &str,
        last_hidden_rms: f64,
    ) -> Value {
        let mut context = logit_source_context(hidden_sha, hidden_rms, on_a770);
        context["hidden_state_source"] = json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_hidden_state_source",
            "diagnostic_only": true,
            "claim_allowed": false,
            "forward_output": {
                "available": true,
                "shape": [1, 1, 4],
                "value_count": 4,
                "finite_count": 4,
                "nan_count": 0,
                "infinite_count": 0,
                "sha256_f32_le": forward_sha,
                "rms": forward_rms,
            },
            "last_hidden": {
                "available": true,
                "shape": [1, 4],
                "value_count": 4,
                "finite_count": 4,
                "nan_count": 0,
                "infinite_count": 0,
                "sha256_f32_le": last_hidden_sha,
                "rms": last_hidden_rms,
            },
            "extraction_context_available": true,
        });
        context["hidden_state_source_context_available"] = json!(true);
        context
    }

    fn logits_first_mismatch_margin_left_with_context(hidden_sha: &str) -> Value {
        let mut logits = logits_first_mismatch_margin_left();
        logits[2]["logit_source_context"] = logit_source_context(hidden_sha, 1.0, 0);
        logits
    }

    fn logits_first_mismatch_margin_right_with_context(hidden_sha: &str) -> Value {
        let mut logits = logits_first_mismatch_margin_right_near_tie();
        logits[2]["logit_source_context"] = logit_source_context(hidden_sha, 1.5, 1);
        logits
    }

    fn logits_first_mismatch_margin_left_with_hidden_state_source(
        hidden_sha: &str,
        forward_sha: &str,
        last_hidden_sha: &str,
    ) -> Value {
        let mut logits = logits_first_mismatch_margin_left();
        logits[2]["logit_source_context"] = logit_source_context_with_hidden_state_source(
            hidden_sha,
            1.0,
            0,
            forward_sha,
            1.0,
            last_hidden_sha,
            1.0,
        );
        logits
    }

    fn logits_first_mismatch_margin_right_with_hidden_state_source(
        hidden_sha: &str,
        forward_sha: &str,
        last_hidden_sha: &str,
    ) -> Value {
        let mut logits = logits_first_mismatch_margin_right_near_tie();
        logits[2]["logit_source_context"] = logit_source_context_with_hidden_state_source(
            hidden_sha,
            1.5,
            1,
            forward_sha,
            1.5,
            last_hidden_sha,
            1.5,
        );
        logits
    }

    fn final_block_tensor_fixture(sha: &str, rms: f64) -> Value {
        json!({
            "available": true,
            "shape": [1, 1, 4],
            "value_count": 4,
            "finite_count": 4,
            "nan_count": 0,
            "infinite_count": 0,
            "sha256_f32_le": sha,
            "rms": rms,
        })
    }

    fn final_block_source_fixture(
        block_input_sha: &str,
        attention_output_sha: &str,
        post_attention_residual_sha: &str,
        feed_forward_output_sha: &str,
        block_output_sha: &str,
        rms: f64,
    ) -> Value {
        json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_final_transformer_block_source",
            "diagnostic_only": true,
            "claim_allowed": false,
            "layer_idx": 0,
            "block_input": final_block_tensor_fixture(block_input_sha, rms),
            "attention_output": final_block_tensor_fixture(attention_output_sha, rms),
            "post_attention_residual": final_block_tensor_fixture(post_attention_residual_sha, rms),
            "feed_forward_output": final_block_tensor_fixture(feed_forward_output_sha, rms),
            "block_output": final_block_tensor_fixture(block_output_sha, rms),
            "source_context_available": true,
        })
    }

    fn logits_first_mismatch_margin_left_with_model_forward_source(
        hidden_sha: &str,
        forward_sha: &str,
        last_hidden_sha: &str,
        prior_sha: &str,
        final_norm_sha: &str,
    ) -> Value {
        let mut logits = logits_first_mismatch_margin_left_with_hidden_state_source(
            hidden_sha,
            forward_sha,
            last_hidden_sha,
        );
        logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"] = json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_model_forward_source",
            "diagnostic_only": true,
            "claim_allowed": false,
            "prior_layer_output": {
                "available": true,
                "shape": [1, 1, 4],
                "value_count": 4,
                "finite_count": 4,
                "nan_count": 0,
                "infinite_count": 0,
                "sha256_f32_le": prior_sha,
                "rms": 1.0,
            },
            "final_norm_output": {
                "available": true,
                "shape": [1, 1, 4],
                "value_count": 4,
                "finite_count": 4,
                "nan_count": 0,
                "infinite_count": 0,
                "sha256_f32_le": final_norm_sha,
                "rms": 1.0,
            },
            "final_block_source": final_block_source_fixture(
                prior_sha,
                prior_sha,
                prior_sha,
                prior_sha,
                prior_sha,
                1.0,
            ),
            "source_context_available": true,
            "final_norm_matches_forward_output": final_norm_sha == forward_sha,
        });
        logits
    }

    fn logits_first_mismatch_margin_right_with_model_forward_source(
        hidden_sha: &str,
        forward_sha: &str,
        last_hidden_sha: &str,
        prior_sha: &str,
        final_norm_sha: &str,
    ) -> Value {
        let mut logits = logits_first_mismatch_margin_right_with_hidden_state_source(
            hidden_sha,
            forward_sha,
            last_hidden_sha,
        );
        logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"] = json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_model_forward_source",
            "diagnostic_only": true,
            "claim_allowed": false,
            "prior_layer_output": {
                "available": true,
                "shape": [1, 1, 4],
                "value_count": 4,
                "finite_count": 4,
                "nan_count": 0,
                "infinite_count": 0,
                "sha256_f32_le": prior_sha,
                "rms": 1.5,
            },
            "final_norm_output": {
                "available": true,
                "shape": [1, 1, 4],
                "value_count": 4,
                "finite_count": 4,
                "nan_count": 0,
                "infinite_count": 0,
                "sha256_f32_le": final_norm_sha,
                "rms": 1.5,
            },
            "final_block_source": final_block_source_fixture(
                prior_sha,
                prior_sha,
                prior_sha,
                prior_sha,
                prior_sha,
                1.5,
            ),
            "source_context_available": true,
            "final_norm_matches_forward_output": final_norm_sha == forward_sha,
        });
        logits
    }

    fn with_final_block_source(
        mut logits: Value,
        block_input_sha: &str,
        attention_output_sha: &str,
        post_attention_residual_sha: &str,
        feed_forward_output_sha: &str,
        block_output_sha: &str,
        rms: f64,
    ) -> Value {
        logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["final_block_source"] =
            final_block_source_fixture(
                block_input_sha,
                attention_output_sha,
                post_attention_residual_sha,
                feed_forward_output_sha,
                block_output_sha,
                rms,
            );
        logits
    }

    fn with_penultimate_block_source(
        mut logits: Value,
        block_input_sha: &str,
        attention_output_sha: &str,
        post_attention_residual_sha: &str,
        feed_forward_output_sha: &str,
        block_output_sha: &str,
        rms: f64,
    ) -> Value {
        logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["penultimate_block_source"] =
            final_block_source_fixture(
                block_input_sha,
                attention_output_sha,
                post_attention_residual_sha,
                feed_forward_output_sha,
                block_output_sha,
                rms,
            );
        logits
    }

    fn with_antepenultimate_block_source(
        mut logits: Value,
        block_input_sha: &str,
        attention_output_sha: &str,
        post_attention_residual_sha: &str,
        feed_forward_output_sha: &str,
        block_output_sha: &str,
        rms: f64,
    ) -> Value {
        logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["antepenultimate_block_source"] =
            final_block_source_fixture(
                block_input_sha,
                attention_output_sha,
                post_attention_residual_sha,
                feed_forward_output_sha,
                block_output_sha,
                rms,
            );
        logits
    }

    fn with_pre_antepenultimate_block_source(
        mut logits: Value,
        block_input_sha: &str,
        attention_output_sha: &str,
        post_attention_residual_sha: &str,
        feed_forward_output_sha: &str,
        block_output_sha: &str,
        rms: f64,
    ) -> Value {
        logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["pre_antepenultimate_block_source"] =
            final_block_source_fixture(
                block_input_sha,
                attention_output_sha,
                post_attention_residual_sha,
                feed_forward_output_sha,
                block_output_sha,
                rms,
            );
        logits
    }

    fn with_earlier_block_source(
        mut logits: Value,
        block_input_sha: &str,
        attention_output_sha: &str,
        post_attention_residual_sha: &str,
        feed_forward_output_sha: &str,
        block_output_sha: &str,
        rms: f64,
    ) -> Value {
        logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["earlier_block_source"] =
            final_block_source_fixture(
                block_input_sha,
                attention_output_sha,
                post_attention_residual_sha,
                feed_forward_output_sha,
                block_output_sha,
                rms,
            );
        logits
    }

    fn transformer_block_source_fixture(
        layer_idx: u64,
        block_input_sha: &str,
        attention_output_sha: &str,
        post_attention_residual_sha: &str,
        feed_forward_output_sha: &str,
        block_output_sha: &str,
        rms: f64,
    ) -> Value {
        let mut block = final_block_source_fixture(
            block_input_sha,
            attention_output_sha,
            post_attention_residual_sha,
            feed_forward_output_sha,
            block_output_sha,
            rms,
        );
        block["context_kind"] = json!("decode_step_transformer_block_source");
        block["layer_idx"] = json!(layer_idx);
        block
    }

    fn with_transformer_block_source_stack(mut logits: Value, blocks: Vec<Value>) -> Value {
        logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["block_sources"] = json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_transformer_block_source_stack",
            "diagnostic_only": true,
            "claim_allowed": false,
            "block_count": blocks.len(),
            "source_context_available": !blocks.is_empty(),
            "blocks": blocks,
        });
        logits
    }

    fn attention_output_source_fixture(layer_idx: u64, rms: f64) -> Value {
        json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_attention_output_source",
            "diagnostic_only": true,
            "claim_allowed": false,
            "layer_idx": layer_idx,
            "attention_input": final_block_tensor_fixture("same-attention-input", rms),
            "q_projection": final_block_tensor_fixture("same-q-projection", rms),
            "k_projection": final_block_tensor_fixture("same-k-projection", rms),
            "v_projection": final_block_tensor_fixture("same-v-projection", rms),
            "q_heads": final_block_tensor_fixture("same-q-heads", rms),
            "k_heads": final_block_tensor_fixture("same-k-heads", rms),
            "v_heads": final_block_tensor_fixture("same-v-heads", rms),
            "q_norm": final_block_tensor_fixture("same-q-norm", rms),
            "k_norm": final_block_tensor_fixture("same-k-norm", rms),
            "q_rope": final_block_tensor_fixture("same-q-rope", rms),
            "k_rope": final_block_tensor_fixture("same-k-rope", rms),
            "k_context": final_block_tensor_fixture("same-k-context", rms),
            "v_context": final_block_tensor_fixture("same-v-context", rms),
            "expanded_k": final_block_tensor_fixture("same-expanded-k", rms),
            "expanded_v": final_block_tensor_fixture("same-expanded-v", rms),
            "scores": final_block_tensor_fixture("same-scores", rms),
            "probabilities": final_block_tensor_fixture("same-probabilities", rms),
            "value_mix_output_heads": final_block_tensor_fixture("same-value-mix-output-heads", rms),
            "output_projection_input": final_block_tensor_fixture("same-output-projection-input", rms),
            "sub_layernorm_output": {
                "available": false,
                "reason": "sub_layernorm_not_present"
            },
            "attention_output": final_block_tensor_fixture("same-attention-output", rms),
            "required_context_available": true,
            "source_context_available": true,
        })
    }

    fn with_attention_output_sources(mut logits: Value, sources: Vec<Value>) -> Value {
        logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["attention_output_sources"] = json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_attention_output_source_stack",
            "diagnostic_only": true,
            "claim_allowed": false,
            "source_count": sources.len(),
            "source_context_available": !sources.is_empty(),
            "sources": sources,
        });
        logits
    }

    fn qkv_projection_source_fixture(
        layer_idx: u64,
        projection: &str,
        input_sha: &str,
        output_sha: &str,
        execution_claim: &str,
        on_a770_opencl: u64,
        scaled_scalar: u64,
        a770_kernel_invocations: u64,
        rms: f64,
    ) -> Value {
        json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_qkv_projection_source",
            "diagnostic_only": true,
            "claim_allowed": false,
            "layer_idx": layer_idx,
            "projection": projection,
            "tensor_name": format!("layers.{layer_idx}.attention.{projection}.weight"),
            "qk256_key": format!("layers.{layer_idx}.attention.{projection}.weight.qk256_qs"),
            "qk256_raw_tensor_present": true,
            "input": final_block_tensor_fixture(input_sha, rms),
            "output": final_block_tensor_fixture(output_sha, rms),
            "dispatch_delta": {
                "bitnet_linear_layers_total": 1,
                "bitnet_linear_layers_on_cuda": 0,
                "bitnet_linear_layers_on_a770_opencl": on_a770_opencl,
                "bitnet_linear_layers_cpu_fallback": 0,
                "unsupported_ops": [],
                "execution_claim": execution_claim,
            },
            "cpu_hot_path_delta": {
                "qk256_f32_scalar_gemv_invocations": 0,
                "qk256_f32_avx2_gemv_invocations": 0,
                "qk256_i8s_scaled_scalar_invocations": scaled_scalar,
                "qk256_i8s_scaled_avx2_gemv_invocations": 0,
                "qk256_flat_bytes_extracted_count": 1,
                "input_rows_materialized_count": 1,
                "output_rows_allocated_count": 1,
                "requested_kernel": null,
                "selected_kernel": if scaled_scalar > 0 { "qk256-i8s-scaled-scalar" } else { "a770-opencl-qk256-i8s-scaled" },
                "qk256_execution_path": if scaled_scalar > 0 { "scaled_i2s_i8s" } else { "a770_opencl_scaled_i2s_i8s" },
            },
            "a770_opencl_runtime_delta": {
                "host_to_device_bytes": if a770_kernel_invocations > 0 { 1024 } else { 0 },
                "device_to_host_bytes": if a770_kernel_invocations > 0 { 128 } else { 0 },
                "kernel_invocations": a770_kernel_invocations,
            },
            "source_context_available": true,
        })
    }

    fn qkv_projection_dispatch_replay_fixture(
        cpu_sha: &str,
        a770_sha: &str,
        cpu_rms: f64,
        a770_rms: f64,
    ) -> Value {
        qkv_projection_dispatch_replay_fixture_with_opencl_policy(
            cpu_sha, a770_sha, a770_sha, cpu_rms, a770_rms, a770_rms,
        )
    }

    fn qkv_projection_dispatch_replay_fixture_with_opencl_policy(
        cpu_sha: &str,
        a770_sha: &str,
        opencl_policy_sha: &str,
        cpu_rms: f64,
        a770_rms: f64,
        opencl_policy_rms: f64,
    ) -> Value {
        json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_qkv_projection_dispatch_replay",
            "diagnostic_only": true,
            "claim_allowed": false,
            "input_rows": 1,
            "output_rows": 2560,
            "cols": 2560,
            "row_stride_bytes": 640,
            "inline_scale": 0.25,
            "cpu_output": final_block_tensor_fixture(cpu_sha, cpu_rms),
            "opencl_policy_output": final_block_tensor_fixture(opencl_policy_sha, opencl_policy_rms),
            "a770_output": final_block_tensor_fixture(a770_sha, a770_rms),
            "cpu_a770_output_sha256_match": cpu_sha == a770_sha,
            "cpu_opencl_policy_output_sha256_match": cpu_sha == opencl_policy_sha,
            "opencl_policy_a770_output_sha256_match": opencl_policy_sha == a770_sha,
            "cpu_a770_output_rms_abs_delta": (cpu_rms - a770_rms).abs(),
            "cpu_opencl_policy_output_rms_abs_delta": (cpu_rms - opencl_policy_rms).abs(),
            "opencl_policy_a770_output_rms_abs_delta": (opencl_policy_rms - a770_rms).abs(),
            "numeric_policy": {
                "cpu_replay": "bitnet_i8s_scaled_wrapping_accumulation",
                "host_opencl_policy_replay": "opencl_linear_i32_accumulation",
            },
            "cpu": {
                "scalar_invocations": 1,
                "execution_path": "cpu_qk256_i2s_i8s_scaled_scalar_replay",
            },
            "a770": {
                "compiled_opencl": true,
                "attempted": true,
                "success": true,
                "host_to_device_bytes": 1640960,
                "device_to_host_bytes": 10240,
                "kernel_invocations": 1,
                "last_device": {
                    "platform_index": 0,
                    "device_index": 0,
                    "platform_name": "Intel OpenCL",
                    "runtime_device": "Intel(R) Arc(TM) A770 Graphics",
                    "vendor": "Intel",
                    "driver_version": "test",
                },
                "error": null,
                "execution_path": "a770_opencl_qk256_i2s_i8s_scaled_replay",
            },
            "source_context_available": true,
        })
    }

    fn with_qkv_projection_dispatch_replay(
        mut source: Value,
        cpu_sha: &str,
        a770_sha: &str,
        cpu_rms: f64,
        a770_rms: f64,
    ) -> Value {
        source["dispatch_replay"] =
            qkv_projection_dispatch_replay_fixture(cpu_sha, a770_sha, cpu_rms, a770_rms);
        source["dispatch_replay_error"] = Value::Null;
        source
    }

    fn with_qkv_projection_dispatch_replay_policy(
        mut source: Value,
        cpu_sha: &str,
        a770_sha: &str,
        opencl_policy_sha: &str,
        cpu_rms: f64,
        a770_rms: f64,
        opencl_policy_rms: f64,
    ) -> Value {
        source["dispatch_replay"] = qkv_projection_dispatch_replay_fixture_with_opencl_policy(
            cpu_sha,
            a770_sha,
            opencl_policy_sha,
            cpu_rms,
            a770_rms,
            opencl_policy_rms,
        );
        source["dispatch_replay_error"] = Value::Null;
        source
    }

    fn with_qkv_projection_sources(mut logits: Value, sources: Vec<Value>) -> Value {
        logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["qkv_projection_sources"] = json!({
            "schema_version": "1.0.0",
            "context_kind": "decode_step_qkv_projection_source_stack",
            "diagnostic_only": true,
            "claim_allowed": false,
            "source_count": sources.len(),
            "source_context_available": !sources.is_empty(),
            "sources": sources,
        });
        logits
    }

    fn logits_first_mismatch_missing_cross_chosen() -> Value {
        json!([
            {
                "step": 0,
                "chosen_id": 4,
                "top_logits": [
                    { "token_id": 4, "logit": 10.0 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            },
            {
                "step": 1,
                "chosen_id": 5,
                "top_logits": [
                    { "token_id": 5, "logit": 10.0 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            },
            {
                "step": 2,
                "chosen_id": 7,
                "top_logits": [
                    { "token_id": 7, "logit": 9.25 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            }
        ])
    }

    fn logits_first_mismatch_sampler_policy_right() -> Value {
        json!([
            {
                "step": 0,
                "chosen_id": 4,
                "top_logits": [
                    { "token_id": 4, "logit": 10.0 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            },
            {
                "step": 1,
                "chosen_id": 5,
                "top_logits": [
                    { "token_id": 5, "logit": 10.0 },
                    { "token_id": 999, "logit": 1.0 }
                ]
            },
            {
                "step": 2,
                "chosen_id": 7,
                "top_logits": [
                    { "token_id": 6, "logit": 10.0 },
                    { "token_id": 7, "logit": 9.0 }
                ]
            }
        ])
    }

    fn cuda_receipt(generated: &[u64], answer: &str, logits: Value) -> Value {
        let mut receipt = receipt("qk256_gemv_cuda", generated, answer, logits);
        receipt["artifact_kind"] = json!("bitnet_cuda_answer_diagnostic_corpus");
        receipt["backend"] = json!({
            "requested_backend": "nvidia-rtx-5070-ti-cuda",
            "selected_backend": "nvidia-rtx-5070-ti-cuda",
            "runtime_api": "cuda",
            "fallback_used": false
        });
        receipt["cases"][0]["backend"] = receipt["backend"].clone();
        receipt
    }

    fn a770_receipt(generated: &[u64], answer: &str, logits: Value) -> Value {
        let mut receipt = receipt(
            "a770_opencl_qk256_i2s_i8s_scaled_dispatch_candidate",
            generated,
            answer,
            logits,
        );
        receipt["artifact_kind"] = json!("bitnet_a770_opencl_answer_diagnostic_corpus");
        receipt["backend"] = json!({
            "requested_backend": "intel-a770-opencl",
            "selected_backend": "intel-a770-opencl",
            "runtime_api": "opencl",
            "fallback_used": false
        });
        receipt["cases"][0]["backend"] = receipt["backend"].clone();
        receipt
    }

    fn build_legacy_report(scalar: &Value, avx2: &Value) -> Value {
        build_answer_parity_receipt(
            Path::new("scalar.json"),
            scalar,
            Path::new("avx2.json"),
            avx2,
            "scalar",
            "avx2",
            true,
            None,
            None,
            None,
        )
    }

    fn build_generic_report(left: &Value, right: &Value) -> Value {
        build_answer_parity_receipt(
            Path::new("left.json"),
            left,
            Path::new("right.json"),
            right,
            "scalar",
            "cuda",
            false,
            None,
            None,
            None,
        )
    }

    #[test]
    fn parity_receipt_passes_matching_scalar_and_avx2_runs() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let avx2 = receipt("i2_s-avx2-reference", &[4], "4", logits());

        let report = build_legacy_report(&scalar, &avx2);

        assert_eq!(report["artifact_kind"], "bitnet_cpu_answer_parity");
        assert_eq!(report["claim"], "scalar_avx2_full_decode_answer_parity");
        assert_eq!(report["inputs"]["scalar_receipt_path"], "scalar.json");
        assert!(report["inputs"]["left_receipt_path"].is_null());
        assert_eq!(report["requested_backend"], "cpu");
        assert_eq!(report["selected_backend"], "cpu-rust");
        assert_eq!(report["runtime_api"], "cpu");
        assert_eq!(report["fallback_used"], false);
        assert_eq!(report["backend"]["fallback_used"], false);
        assert_eq!(report["backend"]["lanes"]["scalar"]["selected_backend"], "cpu-rust");
        assert_eq!(report["backend"]["lanes"]["avx2"]["selected_backend"], "cpu-rust");
        assert_eq!(report["kernel"]["scalar_selected_kernels"], json!(["i2_s-scalar-reference"]));
        assert_eq!(report["kernel"]["avx2_selected_kernels"], json!(["i2_s-avx2-reference"]));
        assert_eq!(report["summary"]["failed"], 0);
        assert_eq!(report["cases"][0]["passed"], true);
        assert!(report["cases"][0]["left"].is_null());
        assert!(report["summary"]["first_divergence"].is_null());
    }

    #[test]
    fn parity_receipt_passes_matching_shared_quality_failures() {
        let mut scalar = receipt("i2_s-scalar-reference", &[999], "bad", logits());
        let mut avx2 = receipt("i2_s-avx2-reference", &[999], "bad", logits());
        scalar["cases"][0]["status"] = json!("quality_failed");
        scalar["cases"][0]["quality"] = json!({
            "passed": false,
            "failed_rules": ["exact_trimmed"]
        });
        avx2["cases"][0]["status"] = json!("quality_failed");
        avx2["cases"][0]["quality"] = json!({
            "passed": false,
            "failed_rules": ["exact_trimmed"]
        });

        let report = build_answer_parity_receipt(
            Path::new("scalar.json"),
            &scalar,
            Path::new("avx2.json"),
            &avx2,
            "scalar",
            "avx2",
            true,
            Some("intel-258v"),
            Some(Path::new("platform-probe.json")),
            None,
        );

        assert_eq!(report["summary"]["failed"], 0);
        assert_eq!(report["machine"]["machine_id"], "intel-258v");
        assert_eq!(report["cases"][0]["passed"], true);
        assert_eq!(report["cases"][0]["scalar"]["quality_passed"], false);
        assert!(report["summary"]["first_divergence"].is_null());
    }

    #[test]
    fn parity_receipt_records_first_generated_token_divergence() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let avx2 = receipt("i2_s-avx2-reference", &[5], "5", logits());

        let report = build_legacy_report(&scalar, &avx2);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["cases"][0]["passed"], false);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "generated_token_ids");
        assert_eq!(report["summary"]["first_divergence"]["scope"], "decode_or_sampler");
    }

    #[test]
    fn parity_receipt_requires_logit_evidence() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", Value::Null);
        let avx2 = receipt("i2_s-avx2-reference", &[4], "4", logits());

        let report = build_legacy_report(&scalar, &avx2);

        let failed = report["cases"][0]["failed_rules"].as_array().unwrap();
        assert!(failed.iter().any(|rule| rule == "scalar_logits_dump_recorded"));
    }

    #[test]
    fn generic_parity_accepts_matching_cpu_and_cuda_answer_corpus_receipts() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let cuda = cuda_receipt(&[4], "4", logits());

        let report = build_generic_report(&scalar, &cuda);

        assert_eq!(report["artifact_kind"], "bitnet_answer_corpus_parity");
        assert_eq!(report["claim"], "full_decode_answer_parity_diagnostic");
        assert_eq!(report["summary"]["failed"], 0);
        assert_eq!(report["inputs"]["left_label"], "scalar");
        assert_eq!(report["inputs"]["right_label"], "cuda");
        assert_eq!(report["requested_backend"], "mixed");
        assert_eq!(report["selected_backend"], "mixed");
        assert_eq!(report["runtime_api"], "mixed");
        assert_eq!(report["fallback_used"], false);
        assert_eq!(report["backend"]["lanes"]["left"]["label"], "scalar");
        assert_eq!(report["backend"]["lanes"]["right"]["label"], "cuda");
        assert_eq!(report["kernel"]["left_selected_kernels"], json!(["i2_s-scalar-reference"]));
        assert_eq!(report["kernel"]["right_selected_kernels"], json!(["qk256_gemv_cuda"]));
        assert_eq!(report["cases"][0]["passed"], true);
        assert_eq!(report["cases"][0]["right"]["selected_kernel"], "qk256_gemv_cuda");
        assert!(report["cases"][0]["scalar"].is_null());
    }

    #[test]
    fn generic_parity_accepts_matching_cpu_and_a770_answer_corpus_receipts() {
        let scalar = receipt("i2_s-avx2-reference", &[4], "4", logits());
        let a770 = a770_receipt(&[4], "4", logits());

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(report["artifact_kind"], "bitnet_answer_corpus_parity");
        assert_eq!(report["summary"]["failed"], 0);
        assert!(
            report["shared_contract"]["failed_rules"]
                .as_array()
                .is_some_and(|shared| !shared.iter().any(|rule| rule == "artifact_kind_contract"))
        );
        assert_eq!(
            report["backend"]["lanes"]["right"]["backend"]["selected_backend"],
            "intel-a770-opencl"
        );
        assert_eq!(
            report["kernel"]["right_selected_kernels"],
            json!(["a770_opencl_qk256_i2s_i8s_scaled_dispatch_candidate"])
        );
    }

    #[test]
    fn generic_parity_summarizes_same_output_logits_topk_frontier() {
        let scalar = receipt("i2_s-avx2-reference", &[4], "4", logits());
        let a770 = a770_receipt(&[4], "4", logits_same_chosen_drift());

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "logits_topk");
        assert_eq!(
            report["logits_topk_frontier"]["classification"],
            "logits_topk_frontier_same_chosen_same_output"
        );
        assert_eq!(report["logits_topk_frontier"]["logits_topk_mismatch_count"], 1);
        assert_eq!(report["logits_topk_frontier"]["same_generated_output_count"], 1);
        assert_eq!(report["logits_topk_frontier"]["generated_output_divergence_count"], 0);
        assert_eq!(report["logits_topk_frontier"]["same_chosen_token_count"], 1);
        assert_eq!(
            report["logits_topk_frontier"]["rows"][0]["classification"],
            "logits_topk_same_chosen_same_output"
        );
        assert_eq!(report["logits_topk_frontier"]["rows"][0]["same_chosen_id"], true);
        assert_eq!(report["logits_topk_frontier"]["rows"][0]["generated_token_ids_match"], true);
    }

    #[test]
    fn generic_parity_summarizes_argmax_source_internal_context_gap() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left(),
        );
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits_first_mismatch_margin_right_near_tie());

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_argmax_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_argmax_source_frontier_internal_logit_source_missing_context"
        );
        assert_eq!(frontier["internal_logit_source_missing_context_count"], 1);
        assert_eq!(frontier["sampler_logit_extraction_count"], 0);
        assert_eq!(frontier["qk256_operand_context_available"], false);
        assert_eq!(frontier["output_head_logit_accumulation_context_available"], false);
        assert_eq!(
            report["generated_output_internal_logit_source_frontier"]["classification"],
            "generated_output_internal_logit_source_frontier_missing_context"
        );
        assert_eq!(
            report["generated_output_hidden_state_source_frontier"]["classification"],
            "generated_output_hidden_state_source_frontier_missing_context"
        );
        let row = &frontier["rows"][0];
        assert_eq!(
            row["classification"],
            "generated_output_argmax_source_internal_logit_source_missing_context"
        );
        assert_eq!(row["left_chosen_is_top1"], true);
        assert_eq!(row["right_chosen_is_top1"], true);
        assert_eq!(row["opposite_argmax"], true);
        assert_eq!(
            row["next_diagnostic"],
            "capture first-mismatch QK256 operand and output-head logit accumulation context"
        );
    }

    #[test]
    fn generic_parity_summarizes_internal_logit_source_hidden_operand_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left_with_context("left-hidden"),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            logits_first_mismatch_margin_right_with_context("right-hidden"),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_internal_logit_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_internal_logit_source_frontier_hidden_operand_drift"
        );
        assert_eq!(frontier["hidden_operand_drift_count"], 1);
        assert_eq!(frontier["hidden_operand_context_available"], true);
        assert_eq!(frontier["qk256_operand_context_available"], true);
        assert_eq!(frontier["output_head_logit_accumulation_context_available"], true);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_internal_logit_source_hidden_operand_drift"
        );
        assert_eq!(frontier["rows"][0]["hidden_operand_context_available"], true);
        assert_eq!(frontier["rows"][0]["hidden_operand_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "localize hidden-state operand drift before output-head QK256"
        );
        assert_eq!(
            report["generated_output_hidden_state_source_frontier"]["classification"],
            "generated_output_hidden_state_source_frontier_qk256_residual_context_missing"
        );
    }

    #[test]
    fn generic_parity_summarizes_hidden_state_source_forward_output_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left_with_hidden_state_source(
                "left-hidden",
                "left-forward",
                "left-hidden",
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            logits_first_mismatch_margin_right_with_hidden_state_source(
                "right-hidden",
                "right-forward",
                "right-hidden",
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_hidden_state_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_hidden_state_source_frontier_forward_output_drift"
        );
        assert_eq!(frontier["forward_output_drift_count"], 1);
        assert_eq!(frontier["hidden_state_source_context_available"], true);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_hidden_state_source_forward_output_drift"
        );
        assert_eq!(frontier["rows"][0]["forward_output_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "capture final norm and prior layer output fingerprints before model.forward output"
        );
    }

    #[test]
    fn generic_parity_summarizes_model_forward_source_prior_layer_output_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left_with_model_forward_source(
                "left-hidden",
                "left-forward",
                "left-hidden",
                "left-prior-layer",
                "left-forward",
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            logits_first_mismatch_margin_right_with_model_forward_source(
                "right-hidden",
                "right-forward",
                "right-hidden",
                "right-prior-layer",
                "right-forward",
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_model_forward_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_model_forward_source_frontier_prior_layer_output_drift"
        );
        assert_eq!(frontier["prior_layer_output_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_model_forward_source_prior_layer_output_drift"
        );
        assert_eq!(frontier["rows"][0]["prior_layer_output_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "capture final transformer block residual, attention output, and FFN output fingerprints"
        );
    }

    #[test]
    fn generic_parity_summarizes_model_forward_source_final_norm_output_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left_with_model_forward_source(
                "left-hidden",
                "left-forward",
                "left-hidden",
                "same-prior-layer",
                "left-forward",
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            logits_first_mismatch_margin_right_with_model_forward_source(
                "right-hidden",
                "right-forward",
                "right-hidden",
                "same-prior-layer",
                "right-forward",
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_model_forward_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_model_forward_source_frontier_final_norm_output_drift"
        );
        assert_eq!(frontier["final_norm_output_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_model_forward_source_final_norm_output_drift"
        );
        assert_eq!(frontier["rows"][0]["prior_layer_output_sha256_match"], true);
        assert_eq!(frontier["rows"][0]["final_norm_output_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "replay final norm input/output numeric policy for selected generated step"
        );
    }

    #[test]
    fn generic_parity_summarizes_final_block_source_block_input_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_final_block_source(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "left-block-input",
                "left-attention-output",
                "left-post-attention-residual",
                "left-ffn-output",
                "left-block-output",
                1.0,
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_final_block_source(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "right-block-input",
                "right-attention-output",
                "right-post-attention-residual",
                "right-ffn-output",
                "right-block-output",
                1.5,
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_final_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_final_block_source_frontier_block_input_drift"
        );
        assert_eq!(frontier["block_input_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_final_block_source_block_input_drift"
        );
        assert_eq!(frontier["rows"][0]["block_input_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "capture penultimate transformer block source frontier"
        );
    }

    #[test]
    fn generic_parity_summarizes_final_block_source_ffn_output_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_final_block_source(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "same-block-input",
                "same-attention-output",
                "same-post-attention-residual",
                "left-ffn-output",
                "left-block-output",
                1.0,
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_final_block_source(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "same-block-input",
                "same-attention-output",
                "same-post-attention-residual",
                "right-ffn-output",
                "right-block-output",
                1.5,
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_final_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_final_block_source_frontier_ffn_output_drift"
        );
        assert_eq!(frontier["ffn_output_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_final_block_source_ffn_output_drift"
        );
        assert_eq!(frontier["rows"][0]["feed_forward_output_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "replay final transformer block FFN output source"
        );
    }

    #[test]
    fn generic_parity_summarizes_final_block_source_missing_context() {
        let mut left_logits = logits_first_mismatch_margin_left_with_model_forward_source(
            "left-hidden",
            "same-forward",
            "left-hidden",
            "same-prior-layer",
            "same-forward",
        );
        let right_logits = logits_first_mismatch_margin_right_with_model_forward_source(
            "right-hidden",
            "same-forward",
            "right-hidden",
            "same-prior-layer",
            "same-forward",
        );
        left_logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["final_block_source"] =
            Value::Null;
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", left_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", right_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_final_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_final_block_source_frontier_missing_context"
        );
        assert_eq!(frontier["missing_context_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_final_block_source_missing_context"
        );
    }

    #[test]
    fn generic_parity_summarizes_penultimate_block_source_block_input_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_penultimate_block_source(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "left-block-input",
                "left-attention-output",
                "left-post-attention-residual",
                "left-ffn-output",
                "left-block-output",
                1.0,
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_penultimate_block_source(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "right-block-input",
                "right-attention-output",
                "right-post-attention-residual",
                "right-ffn-output",
                "right-block-output",
                1.5,
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_penultimate_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_penultimate_block_source_frontier_block_input_drift"
        );
        assert_eq!(frontier["block_input_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_penultimate_block_source_block_input_drift"
        );
        assert_eq!(frontier["rows"][0]["block_input_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "capture antepenultimate transformer block source frontier"
        );
    }

    #[test]
    fn generic_parity_summarizes_penultimate_block_source_ffn_output_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_penultimate_block_source(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "same-block-input",
                "same-attention-output",
                "same-post-attention-residual",
                "left-ffn-output",
                "left-block-output",
                1.0,
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_penultimate_block_source(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "same-block-input",
                "same-attention-output",
                "same-post-attention-residual",
                "right-ffn-output",
                "right-block-output",
                1.5,
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_penultimate_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_penultimate_block_source_frontier_ffn_output_drift"
        );
        assert_eq!(frontier["ffn_output_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_penultimate_block_source_ffn_output_drift"
        );
        assert_eq!(frontier["rows"][0]["feed_forward_output_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "replay penultimate transformer block FFN output source"
        );
    }

    #[test]
    fn generic_parity_summarizes_penultimate_block_source_missing_context() {
        let mut left_logits = logits_first_mismatch_margin_left_with_model_forward_source(
            "left-hidden",
            "same-forward",
            "left-hidden",
            "same-prior-layer",
            "same-forward",
        );
        let right_logits = logits_first_mismatch_margin_right_with_model_forward_source(
            "right-hidden",
            "same-forward",
            "right-hidden",
            "same-prior-layer",
            "same-forward",
        );
        left_logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["penultimate_block_source"] =
            Value::Null;
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", left_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", right_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_penultimate_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_penultimate_block_source_frontier_missing_context"
        );
        assert_eq!(frontier["missing_context_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_penultimate_block_source_missing_context"
        );
    }

    #[test]
    fn generic_parity_summarizes_antepenultimate_block_source_block_input_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_antepenultimate_block_source(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "left-block-input",
                "left-attention-output",
                "left-post-attention-residual",
                "left-ffn-output",
                "left-block-output",
                1.0,
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_antepenultimate_block_source(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "right-block-input",
                "right-attention-output",
                "right-post-attention-residual",
                "right-ffn-output",
                "right-block-output",
                1.5,
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_antepenultimate_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_antepenultimate_block_source_frontier_block_input_drift"
        );
        assert_eq!(frontier["block_input_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_antepenultimate_block_source_block_input_drift"
        );
        assert_eq!(frontier["rows"][0]["block_input_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "capture pre-antepenultimate transformer block source frontier"
        );
    }

    #[test]
    fn generic_parity_summarizes_antepenultimate_block_source_ffn_output_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_antepenultimate_block_source(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "same-block-input",
                "same-attention-output",
                "same-post-attention-residual",
                "left-ffn-output",
                "left-block-output",
                1.0,
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_antepenultimate_block_source(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "same-block-input",
                "same-attention-output",
                "same-post-attention-residual",
                "right-ffn-output",
                "right-block-output",
                1.5,
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_antepenultimate_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_antepenultimate_block_source_frontier_ffn_output_drift"
        );
        assert_eq!(frontier["ffn_output_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_antepenultimate_block_source_ffn_output_drift"
        );
        assert_eq!(frontier["rows"][0]["feed_forward_output_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "replay antepenultimate transformer block FFN output source"
        );
    }

    #[test]
    fn generic_parity_summarizes_antepenultimate_block_source_missing_context() {
        let mut left_logits = logits_first_mismatch_margin_left_with_model_forward_source(
            "left-hidden",
            "same-forward",
            "left-hidden",
            "same-prior-layer",
            "same-forward",
        );
        let right_logits = logits_first_mismatch_margin_right_with_model_forward_source(
            "right-hidden",
            "same-forward",
            "right-hidden",
            "same-prior-layer",
            "same-forward",
        );
        left_logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["antepenultimate_block_source"] =
            Value::Null;
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", left_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", right_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_antepenultimate_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_antepenultimate_block_source_frontier_missing_context"
        );
        assert_eq!(frontier["missing_context_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_antepenultimate_block_source_missing_context"
        );
    }

    #[test]
    fn generic_parity_summarizes_pre_antepenultimate_block_source_block_input_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_pre_antepenultimate_block_source(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "left-block-input",
                "left-attention-output",
                "left-post-attention-residual",
                "left-ffn-output",
                "left-block-output",
                1.0,
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_pre_antepenultimate_block_source(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "right-block-input",
                "right-attention-output",
                "right-post-attention-residual",
                "right-ffn-output",
                "right-block-output",
                1.5,
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_pre_antepenultimate_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_pre_antepenultimate_block_source_frontier_block_input_drift"
        );
        assert_eq!(frontier["block_input_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_pre_antepenultimate_block_source_block_input_drift"
        );
        assert_eq!(frontier["rows"][0]["block_input_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "capture earlier transformer block source frontier"
        );
    }

    #[test]
    fn generic_parity_summarizes_pre_antepenultimate_block_source_ffn_output_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_pre_antepenultimate_block_source(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "same-block-input",
                "same-attention-output",
                "same-post-attention-residual",
                "left-ffn-output",
                "left-block-output",
                1.0,
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_pre_antepenultimate_block_source(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "same-block-input",
                "same-attention-output",
                "same-post-attention-residual",
                "right-ffn-output",
                "right-block-output",
                1.5,
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_pre_antepenultimate_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_pre_antepenultimate_block_source_frontier_ffn_output_drift"
        );
        assert_eq!(frontier["ffn_output_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_pre_antepenultimate_block_source_ffn_output_drift"
        );
        assert_eq!(frontier["rows"][0]["feed_forward_output_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "replay pre-antepenultimate transformer block FFN output source"
        );
    }

    #[test]
    fn generic_parity_summarizes_pre_antepenultimate_block_source_missing_context() {
        let mut left_logits = logits_first_mismatch_margin_left_with_model_forward_source(
            "left-hidden",
            "same-forward",
            "left-hidden",
            "same-prior-layer",
            "same-forward",
        );
        let right_logits = logits_first_mismatch_margin_right_with_model_forward_source(
            "right-hidden",
            "same-forward",
            "right-hidden",
            "same-prior-layer",
            "same-forward",
        );
        left_logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["pre_antepenultimate_block_source"] =
            Value::Null;
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", left_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", right_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_pre_antepenultimate_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_pre_antepenultimate_block_source_frontier_missing_context"
        );
        assert_eq!(frontier["missing_context_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_pre_antepenultimate_block_source_missing_context"
        );
    }

    #[test]
    fn generic_parity_summarizes_earlier_block_source_block_input_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_earlier_block_source(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "left-block-input",
                "left-attention-output",
                "left-post-attention-residual",
                "left-ffn-output",
                "left-block-output",
                1.0,
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_earlier_block_source(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "right-block-input",
                "right-attention-output",
                "right-post-attention-residual",
                "right-ffn-output",
                "right-block-output",
                1.5,
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_earlier_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_earlier_block_source_frontier_block_input_drift"
        );
        assert_eq!(frontier["block_input_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_earlier_block_source_block_input_drift"
        );
        assert_eq!(frontier["rows"][0]["block_input_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "capture preceding transformer block source frontier"
        );
    }

    #[test]
    fn generic_parity_summarizes_earlier_block_source_ffn_output_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_earlier_block_source(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "same-block-input",
                "same-attention-output",
                "same-post-attention-residual",
                "left-ffn-output",
                "left-block-output",
                1.0,
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_earlier_block_source(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                "same-block-input",
                "same-attention-output",
                "same-post-attention-residual",
                "right-ffn-output",
                "right-block-output",
                1.5,
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_earlier_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_earlier_block_source_frontier_ffn_output_drift"
        );
        assert_eq!(frontier["ffn_output_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_earlier_block_source_ffn_output_drift"
        );
        assert_eq!(frontier["rows"][0]["feed_forward_output_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "replay earlier transformer block FFN output source"
        );
    }

    #[test]
    fn generic_parity_summarizes_earlier_block_source_missing_context() {
        let mut left_logits = logits_first_mismatch_margin_left_with_model_forward_source(
            "left-hidden",
            "same-forward",
            "left-hidden",
            "same-prior-layer",
            "same-forward",
        );
        let right_logits = logits_first_mismatch_margin_right_with_model_forward_source(
            "right-hidden",
            "same-forward",
            "right-hidden",
            "same-prior-layer",
            "same-forward",
        );
        left_logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["earlier_block_source"] =
            Value::Null;
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", left_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", right_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_earlier_block_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_earlier_block_source_frontier_missing_context"
        );
        assert_eq!(frontier["missing_context_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_earlier_block_source_missing_context"
        );
    }

    #[test]
    fn generic_parity_summarizes_transformer_block_source_stack_block_input_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_transformer_block_source_stack(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                vec![
                    transformer_block_source_fixture(
                        0,
                        "left-block-input",
                        "left-attention-output",
                        "left-post-attention-residual",
                        "left-ffn-output",
                        "left-block-output",
                        1.0,
                    ),
                    transformer_block_source_fixture(
                        1,
                        "left-layer1-input",
                        "left-layer1-attention",
                        "left-layer1-residual",
                        "left-layer1-ffn",
                        "left-layer1-output",
                        1.0,
                    ),
                ],
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_transformer_block_source_stack(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                vec![
                    transformer_block_source_fixture(
                        0,
                        "right-block-input",
                        "right-attention-output",
                        "right-post-attention-residual",
                        "right-ffn-output",
                        "right-block-output",
                        1.5,
                    ),
                    transformer_block_source_fixture(
                        1,
                        "right-layer1-input",
                        "right-layer1-attention",
                        "right-layer1-residual",
                        "right-layer1-ffn",
                        "right-layer1-output",
                        1.5,
                    ),
                ],
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_transformer_block_source_stack_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_transformer_block_source_stack_frontier_block_input_drift"
        );
        assert_eq!(frontier["block_input_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_transformer_block_source_stack_block_input_drift"
        );
        assert_eq!(frontier["rows"][0]["earliest_divergent_layer_idx"], 0);
        assert_eq!(frontier["rows"][0]["block_input_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "inspect transformer stack input source before earliest divergent block"
        );
    }

    #[test]
    fn generic_parity_summarizes_transformer_block_source_stack_ffn_output_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_transformer_block_source_stack(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                vec![
                    transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "same-layer0-attention",
                        "same-layer0-residual",
                        "same-layer0-ffn",
                        "same-layer0-output",
                        1.0,
                    ),
                    transformer_block_source_fixture(
                        1,
                        "same-layer1-input",
                        "same-layer1-attention",
                        "same-layer1-residual",
                        "left-layer1-ffn",
                        "left-layer1-output",
                        1.0,
                    ),
                ],
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_transformer_block_source_stack(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                vec![
                    transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "same-layer0-attention",
                        "same-layer0-residual",
                        "same-layer0-ffn",
                        "same-layer0-output",
                        1.5,
                    ),
                    transformer_block_source_fixture(
                        1,
                        "same-layer1-input",
                        "same-layer1-attention",
                        "same-layer1-residual",
                        "right-layer1-ffn",
                        "right-layer1-output",
                        1.5,
                    ),
                ],
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_transformer_block_source_stack_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_transformer_block_source_stack_frontier_ffn_output_drift"
        );
        assert_eq!(frontier["ffn_output_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_transformer_block_source_stack_ffn_output_drift"
        );
        assert_eq!(frontier["rows"][0]["earliest_divergent_layer_idx"], 1);
        assert_eq!(frontier["rows"][0]["feed_forward_output_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "replay earliest divergent transformer block FFN output source"
        );
    }

    #[test]
    fn generic_parity_summarizes_transformer_block_source_stack_missing_context() {
        let mut left_logits = logits_first_mismatch_margin_left_with_model_forward_source(
            "left-hidden",
            "same-forward",
            "left-hidden",
            "same-prior-layer",
            "same-forward",
        );
        let right_logits = logits_first_mismatch_margin_right_with_model_forward_source(
            "right-hidden",
            "same-forward",
            "right-hidden",
            "same-prior-layer",
            "same-forward",
        );
        left_logits[2]["logit_source_context"]["hidden_state_source"]["model_forward_source"]["block_sources"] =
            Value::Null;
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", left_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", right_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_transformer_block_source_stack_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_transformer_block_source_stack_frontier_missing_context"
        );
        assert_eq!(frontier["missing_context_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_transformer_block_source_stack_missing_context"
        );
    }

    #[test]
    fn generic_parity_summarizes_attention_output_source_q_projection_drift() {
        let left_source = attention_output_source_fixture(0, 1.0);
        let mut right_source = attention_output_source_fixture(0, 1.5);
        right_source["q_projection"] = final_block_tensor_fixture("right-q-projection", 1.5);

        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_left_with_model_forward_source(
                        "left-hidden",
                        "same-forward",
                        "left-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "left-layer0-attention",
                        "left-layer0-residual",
                        "left-layer0-ffn",
                        "left-layer0-output",
                        1.0,
                    )],
                ),
                vec![left_source],
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_right_with_model_forward_source(
                        "right-hidden",
                        "same-forward",
                        "right-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "right-layer0-attention",
                        "right-layer0-residual",
                        "right-layer0-ffn",
                        "right-layer0-output",
                        1.5,
                    )],
                ),
                vec![right_source],
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_attention_output_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_attention_output_source_frontier_q_projection_drift"
        );
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_attention_output_source_q_projection_drift"
        );
        assert_eq!(frontier["rows"][0]["target_layer_idx"], 0);
        assert_eq!(frontier["rows"][0]["fields"][1]["field"], "q_projection");
        assert_eq!(frontier["rows"][0]["fields"][1]["sha256_match"], false);
        assert_eq!(
            frontier["next_diagnostic"],
            "replay earliest divergent block QKV projection source"
        );
    }

    #[test]
    fn generic_parity_summarizes_attention_output_source_value_mix_drift() {
        let left_source = attention_output_source_fixture(0, 1.0);
        let mut right_source = attention_output_source_fixture(0, 1.5);
        right_source["value_mix_output_heads"] =
            final_block_tensor_fixture("right-value-mix-output-heads", 1.5);

        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_left_with_model_forward_source(
                        "left-hidden",
                        "same-forward",
                        "left-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "left-layer0-attention",
                        "left-layer0-residual",
                        "left-layer0-ffn",
                        "left-layer0-output",
                        1.0,
                    )],
                ),
                vec![left_source],
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_right_with_model_forward_source(
                        "right-hidden",
                        "same-forward",
                        "right-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "right-layer0-attention",
                        "right-layer0-residual",
                        "right-layer0-ffn",
                        "right-layer0-output",
                        1.5,
                    )],
                ),
                vec![right_source],
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_attention_output_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_attention_output_source_frontier_value_mix_drift"
        );
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_attention_output_source_value_mix_drift"
        );
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "replay earliest divergent block value-mix source"
        );
    }

    #[test]
    fn generic_parity_summarizes_attention_output_source_output_projection_drift() {
        let left_source = attention_output_source_fixture(0, 1.0);
        let mut right_source = attention_output_source_fixture(0, 1.5);
        right_source["attention_output"] =
            final_block_tensor_fixture("right-attention-output-source", 1.5);

        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_left_with_model_forward_source(
                        "left-hidden",
                        "same-forward",
                        "left-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "left-layer0-attention",
                        "left-layer0-residual",
                        "left-layer0-ffn",
                        "left-layer0-output",
                        1.0,
                    )],
                ),
                vec![left_source],
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_right_with_model_forward_source(
                        "right-hidden",
                        "same-forward",
                        "right-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "right-layer0-attention",
                        "right-layer0-residual",
                        "right-layer0-ffn",
                        "right-layer0-output",
                        1.5,
                    )],
                ),
                vec![right_source],
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_attention_output_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_attention_output_source_frontier_output_projection_drift"
        );
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_attention_output_source_output_projection_drift"
        );
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "replay earliest divergent block attention output projection source"
        );
    }

    #[test]
    fn generic_parity_summarizes_attention_output_source_missing_context() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_transformer_block_source_stack(
                logits_first_mismatch_margin_left_with_model_forward_source(
                    "left-hidden",
                    "same-forward",
                    "left-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                vec![transformer_block_source_fixture(
                    0,
                    "same-layer0-input",
                    "left-layer0-attention",
                    "left-layer0-residual",
                    "left-layer0-ffn",
                    "left-layer0-output",
                    1.0,
                )],
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_transformer_block_source_stack(
                logits_first_mismatch_margin_right_with_model_forward_source(
                    "right-hidden",
                    "same-forward",
                    "right-hidden",
                    "same-prior-layer",
                    "same-forward",
                ),
                vec![transformer_block_source_fixture(
                    0,
                    "same-layer0-input",
                    "right-layer0-attention",
                    "right-layer0-residual",
                    "right-layer0-ffn",
                    "right-layer0-output",
                    1.5,
                )],
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_attention_output_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_attention_output_source_frontier_missing_context"
        );
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_attention_output_source_missing_context"
        );
        assert_eq!(frontier["rows"][0]["reason"], "left_attention_output_sources_missing");
    }

    #[test]
    fn generic_parity_summarizes_qkv_projection_source_dispatch_path_drift() {
        let left_source = attention_output_source_fixture(0, 1.0);
        let mut right_source = attention_output_source_fixture(0, 1.5);
        right_source["q_projection"] = final_block_tensor_fixture("right-q-projection", 1.5);

        let scalar_logits = with_qkv_projection_sources(
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_left_with_model_forward_source(
                        "left-hidden",
                        "same-forward",
                        "left-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "left-layer0-attention",
                        "left-layer0-residual",
                        "left-layer0-ffn",
                        "left-layer0-output",
                        1.0,
                    )],
                ),
                vec![left_source],
            ),
            vec![qkv_projection_source_fixture(
                0,
                "q_proj",
                "same-projection-input",
                "left-q-projection-output",
                "cpu_qk256_reference",
                0,
                1,
                0,
                1.0,
            )],
        );
        let a770_logits = with_qkv_projection_sources(
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_right_with_model_forward_source(
                        "right-hidden",
                        "same-forward",
                        "right-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "right-layer0-attention",
                        "right-layer0-residual",
                        "right-layer0-ffn",
                        "right-layer0-output",
                        1.5,
                    )],
                ),
                vec![right_source],
            ),
            vec![qkv_projection_source_fixture(
                0,
                "q_proj",
                "same-projection-input",
                "right-q-projection-output",
                "a770_opencl_qk256_contribution",
                1,
                0,
                1,
                1.5,
            )],
        );
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", scalar_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", a770_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_qkv_projection_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_qkv_projection_source_frontier_dispatch_path_drift"
        );
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_qkv_projection_source_dispatch_path_drift"
        );
        assert_eq!(frontier["rows"][0]["projection"], "q_proj");
        assert_eq!(frontier["rows"][0]["input"]["sha256_match"], true);
        assert_eq!(frontier["rows"][0]["dispatch_match"], false);
        assert_eq!(
            frontier["next_diagnostic"],
            "replay selected QKV projection CPU versus A770 dispatch policy"
        );
    }

    #[test]
    fn generic_parity_summarizes_qkv_projection_dispatch_replay_cpu_a770_output_drift() {
        let left_source = attention_output_source_fixture(0, 1.0);
        let mut right_source = attention_output_source_fixture(0, 1.5);
        right_source["q_projection"] = final_block_tensor_fixture("a770-replay-output", 1.5);

        let scalar_logits = with_qkv_projection_sources(
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_left_with_model_forward_source(
                        "left-hidden",
                        "same-forward",
                        "left-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "left-layer0-attention",
                        "left-layer0-residual",
                        "left-layer0-ffn",
                        "left-layer0-output",
                        1.0,
                    )],
                ),
                vec![left_source],
            ),
            vec![with_qkv_projection_dispatch_replay(
                qkv_projection_source_fixture(
                    0,
                    "q_proj",
                    "same-projection-input",
                    "cpu-replay-output",
                    "cpu_qk256_reference",
                    0,
                    1,
                    0,
                    1.0,
                ),
                "cpu-replay-output",
                "a770-replay-output",
                1.0,
                1.5,
            )],
        );
        let a770_logits = with_qkv_projection_sources(
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_right_with_model_forward_source(
                        "right-hidden",
                        "same-forward",
                        "right-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "right-layer0-attention",
                        "right-layer0-residual",
                        "right-layer0-ffn",
                        "right-layer0-output",
                        1.5,
                    )],
                ),
                vec![right_source],
            ),
            vec![with_qkv_projection_dispatch_replay(
                qkv_projection_source_fixture(
                    0,
                    "q_proj",
                    "same-projection-input",
                    "a770-replay-output",
                    "a770_opencl_qk256_contribution",
                    1,
                    0,
                    1,
                    1.5,
                ),
                "cpu-replay-output",
                "a770-replay-output",
                1.0,
                1.5,
            )],
        );
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", scalar_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", a770_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_qkv_projection_dispatch_replay_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_qkv_projection_dispatch_replay_frontier_cpu_a770_output_drift"
        );
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_qkv_projection_dispatch_replay_cpu_a770_output_drift"
        );
        assert_eq!(frontier["rows"][0]["left_runtime_output_matches_cpu_replay"], true);
        assert_eq!(frontier["rows"][0]["right_runtime_output_matches_a770_replay"], true);
        assert_eq!(frontier["rows"][0]["left_cpu_a770_replay_output_match"], false);
        assert_eq!(
            frontier["next_diagnostic"],
            "inspect selected QK256 CPU scalar versus A770 OpenCL GEMV numeric policy"
        );

        let numeric_frontier = &report["generated_output_qk256_numeric_policy_frontier"];
        assert_eq!(
            numeric_frontier["classification"],
            "generated_output_qk256_numeric_policy_frontier_accumulation_order"
        );
        assert_eq!(
            numeric_frontier["rows"][0]["classification"],
            "generated_output_qk256_numeric_policy_accumulation_order"
        );
        assert_eq!(numeric_frontier["rows"][0]["left_cpu_opencl_policy_output_match"], false);
        assert_eq!(numeric_frontier["rows"][0]["right_opencl_policy_a770_output_match"], true);
    }

    #[test]
    fn generic_parity_summarizes_qkv_projection_dispatch_replay_numeric_policy_output_casting() {
        let left_source = attention_output_source_fixture(0, 1.0);
        let mut right_source = attention_output_source_fixture(0, 1.5);
        right_source["q_projection"] = final_block_tensor_fixture("a770-replay-output", 1.5);

        let scalar_logits = with_qkv_projection_sources(
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_left_with_model_forward_source(
                        "left-hidden",
                        "same-forward",
                        "left-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "left-layer0-attention",
                        "left-layer0-residual",
                        "left-layer0-ffn",
                        "left-layer0-output",
                        1.0,
                    )],
                ),
                vec![left_source],
            ),
            vec![with_qkv_projection_dispatch_replay_policy(
                qkv_projection_source_fixture(
                    0,
                    "q_proj",
                    "same-projection-input",
                    "cpu-replay-output",
                    "cpu_qk256_reference",
                    0,
                    1,
                    0,
                    1.0,
                ),
                "cpu-replay-output",
                "a770-replay-output",
                "cpu-replay-output",
                1.0,
                1.5,
                1.0,
            )],
        );
        let a770_logits = with_qkv_projection_sources(
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_right_with_model_forward_source(
                        "right-hidden",
                        "same-forward",
                        "right-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "right-layer0-attention",
                        "right-layer0-residual",
                        "right-layer0-ffn",
                        "right-layer0-output",
                        1.5,
                    )],
                ),
                vec![right_source],
            ),
            vec![with_qkv_projection_dispatch_replay_policy(
                qkv_projection_source_fixture(
                    0,
                    "q_proj",
                    "same-projection-input",
                    "a770-replay-output",
                    "a770_opencl_qk256_contribution",
                    1,
                    0,
                    1,
                    1.5,
                ),
                "cpu-replay-output",
                "a770-replay-output",
                "cpu-replay-output",
                1.0,
                1.5,
                1.0,
            )],
        );
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", scalar_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", a770_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_qk256_numeric_policy_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_qk256_numeric_policy_frontier_output_casting_serialization"
        );
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_qk256_numeric_policy_output_casting_serialization"
        );
        assert_eq!(frontier["rows"][0]["left_cpu_opencl_policy_output_match"], true);
        assert_eq!(frontier["rows"][0]["right_opencl_policy_a770_output_match"], false);
    }

    #[test]
    fn generic_parity_summarizes_qkv_projection_dispatch_replay_runtime_mismatch() {
        let left_source = attention_output_source_fixture(0, 1.0);
        let mut right_source = attention_output_source_fixture(0, 1.5);
        right_source["q_projection"] = final_block_tensor_fixture("right-runtime-output", 1.5);

        let scalar_logits = with_qkv_projection_sources(
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_left_with_model_forward_source(
                        "left-hidden",
                        "same-forward",
                        "left-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "left-layer0-attention",
                        "left-layer0-residual",
                        "left-layer0-ffn",
                        "left-layer0-output",
                        1.0,
                    )],
                ),
                vec![left_source],
            ),
            vec![with_qkv_projection_dispatch_replay(
                qkv_projection_source_fixture(
                    0,
                    "q_proj",
                    "same-projection-input",
                    "cpu-replay-output",
                    "cpu_qk256_reference",
                    0,
                    1,
                    0,
                    1.0,
                ),
                "cpu-replay-output",
                "a770-replay-output",
                1.0,
                1.5,
            )],
        );
        let a770_logits = with_qkv_projection_sources(
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_right_with_model_forward_source(
                        "right-hidden",
                        "same-forward",
                        "right-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "right-layer0-attention",
                        "right-layer0-residual",
                        "right-layer0-ffn",
                        "right-layer0-output",
                        1.5,
                    )],
                ),
                vec![right_source],
            ),
            vec![with_qkv_projection_dispatch_replay(
                qkv_projection_source_fixture(
                    0,
                    "q_proj",
                    "same-projection-input",
                    "right-runtime-output",
                    "a770_opencl_qk256_contribution",
                    1,
                    0,
                    1,
                    1.5,
                ),
                "cpu-replay-output",
                "a770-replay-output",
                1.0,
                1.5,
            )],
        );
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", scalar_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", a770_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_qkv_projection_dispatch_replay_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_qkv_projection_dispatch_replay_frontier_runtime_replay_mismatch"
        );
        assert_eq!(frontier["rows"][0]["right_runtime_output_matches_a770_replay"], false);
        assert_eq!(
            frontier["next_diagnostic"],
            "inspect selected QKV projection dispatch replay capture scope"
        );
    }

    #[test]
    fn generic_parity_summarizes_qkv_projection_dispatch_replay_missing_context() {
        let left_source = attention_output_source_fixture(0, 1.0);
        let mut right_source = attention_output_source_fixture(0, 1.5);
        right_source["q_projection"] = final_block_tensor_fixture("right-q-projection", 1.5);

        let scalar_logits = with_qkv_projection_sources(
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_left_with_model_forward_source(
                        "left-hidden",
                        "same-forward",
                        "left-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "left-layer0-attention",
                        "left-layer0-residual",
                        "left-layer0-ffn",
                        "left-layer0-output",
                        1.0,
                    )],
                ),
                vec![left_source],
            ),
            vec![qkv_projection_source_fixture(
                0,
                "q_proj",
                "same-projection-input",
                "left-q-projection-output",
                "cpu_qk256_reference",
                0,
                1,
                0,
                1.0,
            )],
        );
        let a770_logits = with_qkv_projection_sources(
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_right_with_model_forward_source(
                        "right-hidden",
                        "same-forward",
                        "right-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "right-layer0-attention",
                        "right-layer0-residual",
                        "right-layer0-ffn",
                        "right-layer0-output",
                        1.5,
                    )],
                ),
                vec![right_source],
            ),
            vec![qkv_projection_source_fixture(
                0,
                "q_proj",
                "same-projection-input",
                "right-q-projection-output",
                "a770_opencl_qk256_contribution",
                1,
                0,
                1,
                1.5,
            )],
        );
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", scalar_logits);
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", a770_logits);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_qkv_projection_dispatch_replay_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_qkv_projection_dispatch_replay_frontier_missing_context"
        );
        assert_eq!(frontier["rows"][0]["reason"], "left_dispatch_replay_missing");

        let numeric_frontier = &report["generated_output_qk256_numeric_policy_frontier"];
        assert_eq!(
            numeric_frontier["classification"],
            "generated_output_qk256_numeric_policy_frontier_missing_context"
        );
        assert_eq!(numeric_frontier["rows"][0]["reason"], "left_dispatch_replay_missing");
    }

    #[test]
    fn generic_parity_summarizes_qkv_projection_source_output_drift() {
        let left_source = attention_output_source_fixture(0, 1.0);
        let mut right_source = attention_output_source_fixture(0, 1.5);
        right_source["q_projection"] = final_block_tensor_fixture("right-q-projection", 1.5);

        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_qkv_projection_sources(
                with_attention_output_sources(
                    with_transformer_block_source_stack(
                        logits_first_mismatch_margin_left_with_model_forward_source(
                            "left-hidden",
                            "same-forward",
                            "left-hidden",
                            "same-prior-layer",
                            "same-forward",
                        ),
                        vec![transformer_block_source_fixture(
                            0,
                            "same-layer0-input",
                            "left-layer0-attention",
                            "left-layer0-residual",
                            "left-layer0-ffn",
                            "left-layer0-output",
                            1.0,
                        )],
                    ),
                    vec![left_source],
                ),
                vec![qkv_projection_source_fixture(
                    0,
                    "q_proj",
                    "same-projection-input",
                    "left-q-projection-output",
                    "cpu_qk256_reference",
                    0,
                    1,
                    0,
                    1.0,
                )],
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_qkv_projection_sources(
                with_attention_output_sources(
                    with_transformer_block_source_stack(
                        logits_first_mismatch_margin_right_with_model_forward_source(
                            "right-hidden",
                            "same-forward",
                            "right-hidden",
                            "same-prior-layer",
                            "same-forward",
                        ),
                        vec![transformer_block_source_fixture(
                            0,
                            "same-layer0-input",
                            "right-layer0-attention",
                            "right-layer0-residual",
                            "right-layer0-ffn",
                            "right-layer0-output",
                            1.5,
                        )],
                    ),
                    vec![right_source],
                ),
                vec![qkv_projection_source_fixture(
                    0,
                    "q_proj",
                    "same-projection-input",
                    "right-q-projection-output",
                    "cpu_qk256_reference",
                    0,
                    1,
                    0,
                    1.5,
                )],
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_qkv_projection_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_qkv_projection_source_frontier_projection_output_drift"
        );
        assert_eq!(frontier["rows"][0]["dispatch_match"], true);
        assert_eq!(frontier["rows"][0]["output"]["sha256_match"], false);
    }

    #[test]
    fn generic_parity_summarizes_qkv_projection_source_missing_context() {
        let left_source = attention_output_source_fixture(0, 1.0);
        let mut right_source = attention_output_source_fixture(0, 1.5);
        right_source["q_projection"] = final_block_tensor_fixture("right-q-projection", 1.5);

        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_left_with_model_forward_source(
                        "left-hidden",
                        "same-forward",
                        "left-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "left-layer0-attention",
                        "left-layer0-residual",
                        "left-layer0-ffn",
                        "left-layer0-output",
                        1.0,
                    )],
                ),
                vec![left_source],
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            with_attention_output_sources(
                with_transformer_block_source_stack(
                    logits_first_mismatch_margin_right_with_model_forward_source(
                        "right-hidden",
                        "same-forward",
                        "right-hidden",
                        "same-prior-layer",
                        "same-forward",
                    ),
                    vec![transformer_block_source_fixture(
                        0,
                        "same-layer0-input",
                        "right-layer0-attention",
                        "right-layer0-residual",
                        "right-layer0-ffn",
                        "right-layer0-output",
                        1.5,
                    )],
                ),
                vec![right_source],
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_qkv_projection_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_qkv_projection_source_frontier_missing_context"
        );
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_qkv_projection_source_missing_context"
        );
        assert_eq!(frontier["rows"][0]["reason"], "left_qkv_projection_sources_missing");
    }

    #[test]
    fn generic_parity_summarizes_hidden_state_source_last_hidden_extraction_drift() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left_with_hidden_state_source(
                "left-hidden",
                "same-forward",
                "left-hidden",
            ),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            logits_first_mismatch_margin_right_with_hidden_state_source(
                "right-hidden",
                "same-forward",
                "right-hidden",
            ),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_hidden_state_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_hidden_state_source_frontier_last_hidden_extraction_drift"
        );
        assert_eq!(frontier["last_hidden_extraction_drift_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_hidden_state_source_last_hidden_extraction_drift"
        );
        assert_eq!(frontier["rows"][0]["forward_output_sha256_match"], true);
        assert_eq!(frontier["rows"][0]["last_hidden_sha256_match"], false);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "inspect last-hidden extraction and tensor serialization boundary"
        );
    }

    #[test]
    fn generic_parity_summarizes_internal_logit_source_output_head_accumulation() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left_with_context("same-hidden"),
        );
        let a770 = a770_receipt(
            &[4, 5, 7],
            "4 5 7",
            logits_first_mismatch_margin_right_with_context("same-hidden"),
        );

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_internal_logit_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_internal_logit_source_frontier_output_head_logit_accumulation"
        );
        assert_eq!(frontier["output_head_logit_accumulation_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_internal_logit_source_output_head_logit_accumulation"
        );
        assert_eq!(frontier["rows"][0]["hidden_operand_sha256_match"], true);
        assert_eq!(
            frontier["rows"][0]["next_diagnostic"],
            "replay output-head QK256 accumulation for the selected mismatch tokens"
        );
    }

    #[test]
    fn generic_parity_summarizes_argmax_source_sampler_policy() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left(),
        );
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits_first_mismatch_sampler_policy_right());

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_argmax_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_argmax_source_frontier_sampler_logit_extraction_policy"
        );
        assert_eq!(frontier["sampler_logit_extraction_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_argmax_source_sampler_logit_extraction_policy"
        );
        assert_eq!(frontier["rows"][0]["right_chosen_is_top1"], false);
    }

    #[test]
    fn generic_parity_summarizes_argmax_source_prompt_history() {
        let scalar =
            receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", logits_for_chosen(&[4, 5, 6]));
        let mut a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits_for_chosen(&[4, 5, 7]));
        a770["cases"][0]["token_ids"]["prompt"] = json!([1, 2, 999]);

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_argmax_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_argmax_source_frontier_prompt_history_serialization"
        );
        assert_eq!(frontier["prompt_history_serialization_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_argmax_source_prompt_history_serialization"
        );
        assert_eq!(frontier["rows"][0]["first_prompt_mismatch_index"], 2);
    }

    #[test]
    fn generic_parity_summarizes_argmax_source_trace_capture_loss() {
        let scalar =
            receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", logits_for_chosen(&[4, 5]));
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits_for_chosen(&[4, 5, 7]));

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_argmax_source_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_argmax_source_frontier_trace_capture_context_loss"
        );
        assert_eq!(frontier["trace_capture_context_loss_count"], 1);
        assert_eq!(
            frontier["rows"][0]["classification"],
            "generated_output_argmax_source_trace_capture_context_loss"
        );
        assert_eq!(frontier["rows"][0]["reason"], "left_logits_step_missing");
    }

    #[test]
    fn generic_parity_summarizes_generated_output_logits_topk_frontier() {
        let scalar = receipt("i2_s-avx2-reference", &[4], "4", logits());
        let a770 = a770_receipt(&[5], "5", logits_different_chosen());

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "generated_token_ids");
        assert_eq!(
            report["logits_topk_frontier"]["classification"],
            "logits_topk_frontier_generated_output_divergence"
        );
        assert_eq!(report["logits_topk_frontier"]["logits_topk_mismatch_count"], 1);
        assert_eq!(report["logits_topk_frontier"]["generated_output_divergence_count"], 1);
        assert_eq!(report["logits_topk_frontier"]["different_chosen_token_count"], 1);
        assert_eq!(
            report["logits_topk_frontier"]["rows"][0]["classification"],
            "logits_topk_generated_output_divergence"
        );
        assert_eq!(report["logits_topk_frontier"]["rows"][0]["same_chosen_id"], false);
        assert_eq!(report["logits_topk_frontier"]["rows"][0]["generated_token_ids_match"], false);
    }

    #[test]
    fn generic_parity_summarizes_generated_output_frontier_with_logit_context() {
        let scalar =
            receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", logits_for_chosen(&[4, 5, 6]));
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits_for_chosen(&[4, 5, 7]));

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(
            report["generated_output_frontier"]["classification"],
            "generated_output_frontier_first_mismatch_has_logit_context"
        );
        assert_eq!(report["generated_output_frontier"]["generated_output_mismatch_count"], 1);
        assert_eq!(report["generated_output_frontier"]["mismatch_with_logit_context_count"], 1);
        assert_eq!(report["generated_output_frontier"]["missing_logit_context_count"], 0);
        assert_eq!(
            report["generated_output_frontier"]["rows"][0]["classification"],
            "generated_output_first_mismatch_has_logit_context"
        );
        assert_eq!(report["generated_output_frontier"]["rows"][0]["first_mismatch_index"], 2);
        assert_eq!(report["generated_output_frontier"]["rows"][0]["left_token_id"], 6);
        assert_eq!(report["generated_output_frontier"]["rows"][0]["right_token_id"], 7);
        assert_eq!(
            report["generated_output_frontier"]["rows"][0]["has_logit_context_at_first_mismatch"],
            true
        );
        assert_eq!(
            report["generated_output_frontier"]["rows"][0]["same_chosen_id_at_first_mismatch"],
            false
        );
    }

    #[test]
    fn generic_parity_summarizes_first_mismatch_logit_margin_frontier() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left(),
        );
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits_first_mismatch_margin_right_near_tie());

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(
            report["generated_output_logit_margin_frontier"]["classification"],
            "generated_output_logit_margin_frontier_opposite_argmax_right_near_tie"
        );
        assert_eq!(
            report["generated_output_logit_margin_frontier"]["generated_output_mismatch_count"],
            1
        );
        assert_eq!(report["generated_output_logit_margin_frontier"]["margin_available_count"], 1);
        assert_eq!(report["generated_output_logit_margin_frontier"]["right_near_tie_count"], 1);
        let row = &report["generated_output_logit_margin_frontier"]["rows"][0];
        assert_eq!(
            row["classification"],
            "generated_output_logit_margin_first_mismatch_opposite_argmax_right_near_tie"
        );
        assert_eq!(row["first_mismatch_index"], 2);
        assert_eq!(row["left_chosen_id"], 6);
        assert_eq!(row["right_chosen_id"], 7);
        assert_eq!(row["opposite_argmax"], true);
        assert_eq!(row["right_margin_near_tie"], true);
        assert_eq!(row["left_margin_over_right_chosen_on_left"], json!(0.5));
        assert!(matches!(
            row["right_margin_over_left_chosen_on_right"].as_f64(),
            Some(value) if (value - 0.005).abs() < 1e-12
        ));
    }

    #[test]
    fn generic_parity_summarizes_first_mismatch_margin_missing_cross_chosen_logit() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left(),
        );
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits_first_mismatch_missing_cross_chosen());

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(
            report["generated_output_logit_margin_frontier"]["classification"],
            "generated_output_logit_margin_frontier_missing_cross_chosen_logit"
        );
        assert_eq!(report["generated_output_logit_margin_frontier"]["margin_available_count"], 0);
        assert_eq!(
            report["generated_output_logit_margin_frontier"]["missing_cross_chosen_logit_count"],
            1
        );
        assert_eq!(
            report["generated_output_logit_margin_frontier"]["rows"][0]["classification"],
            "generated_output_logit_margin_missing_cross_chosen_logit"
        );
    }

    #[test]
    fn generic_parity_does_not_truncate_single_missing_context_logit_margin_row() {
        let scalar = receipt(
            "i2_s-avx2-reference",
            &[4, 5, 6],
            "4 5 6",
            logits_first_mismatch_margin_left(),
        );
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits_for_chosen(&[4, 5]));

        let report = build_generic_report(&scalar, &a770);
        let frontier = &report["generated_output_logit_margin_frontier"];

        assert_eq!(
            frontier["classification"],
            "generated_output_logit_margin_frontier_missing_context"
        );
        assert_eq!(frontier["generated_output_mismatch_count"], 1);
        assert_eq!(frontier["missing_context_count"], 1);
        assert_eq!(frontier["rows_truncated"], false);
        assert_eq!(
            frontier["rows"],
            json!([
                {
                    "case_id": "math",
                    "classification": "generated_output_logit_margin_missing_context",
                    "reason": "right_logits_step_missing",
                    "first_mismatch_index": 2,
                    "left_logits_step_count": 3,
                    "right_logits_step_count": 2
                }
            ])
        );
    }

    #[test]
    fn generic_parity_summarizes_generated_output_frontier_missing_logit_context() {
        let scalar = receipt("i2_s-avx2-reference", &[4, 5, 6], "4 5 6", logits());
        let a770 = a770_receipt(&[4, 5, 7], "4 5 7", logits());

        let report = build_generic_report(&scalar, &a770);

        assert_eq!(
            report["generated_output_frontier"]["classification"],
            "generated_output_frontier_first_mismatch_missing_logit_context"
        );
        assert_eq!(report["generated_output_frontier"]["generated_output_mismatch_count"], 1);
        assert_eq!(report["generated_output_frontier"]["mismatch_with_logit_context_count"], 0);
        assert_eq!(report["generated_output_frontier"]["missing_logit_context_count"], 1);
        assert_eq!(
            report["generated_output_frontier"]["rows"][0]["classification"],
            "generated_output_first_mismatch_missing_logit_context"
        );
        assert_eq!(report["generated_output_frontier"]["rows"][0]["first_mismatch_index"], 2);
        assert_eq!(
            report["generated_output_frontier"]["rows"][0]["has_logit_context_at_first_mismatch"],
            false
        );
        assert!(
            report["generated_output_frontier"]["rows"][0]["same_chosen_id_at_first_mismatch"]
                .is_null()
        );
    }

    #[test]
    fn generic_parity_records_left_right_divergence() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let cuda = cuda_receipt(&[5], "5", logits());

        let report = build_generic_report(&scalar, &cuda);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "generated_token_ids");
        assert_eq!(report["summary"]["first_divergence"]["left_label"], "scalar");
        assert_eq!(report["summary"]["first_divergence"]["right_label"], "cuda");
        assert!(report["summary"]["first_divergence"]["left"].is_array());
        assert!(report["summary"]["first_divergence"]["scalar"].is_null());
    }

    #[test]
    fn legacy_parity_still_rejects_cuda_receipts() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let cuda = cuda_receipt(&[4], "4", logits());

        let report = build_legacy_report(&scalar, &cuda);

        assert_ne!(report["summary"]["failed"], 0);
        let shared = report["shared_contract"]["failed_rules"].as_array().unwrap();
        assert!(shared.iter().any(|rule| rule == "artifact_kind_contract"));
        assert!(shared.iter().any(|rule| rule == "strict_cpu_backend"));
    }

    #[test]
    fn generic_parity_classifies_matching_command_failures_as_missing_execution_evidence() {
        let mut scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let mut cuda = cuda_receipt(&[4], "4", logits());
        let failed_case = json!({
            "id": "math",
            "question": "Answer with a single digit: 2+2=",
            "status": "command_failed",
            "exit_code": 7,
            "run_receipt_path": "target/bitnet/receipts/math.json",
            "quality": {
                "passed": false,
                "failed_rules": ["command_failed"]
            }
        });
        scalar["cases"][0] = failed_case.clone();
        cuda["cases"][0] = failed_case;

        let report = build_generic_report(&scalar, &cuda);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "execution_evidence_recorded");
        let failed = report["cases"][0]["failed_rules"].as_array().unwrap();
        assert!(failed.iter().any(|rule| rule == "execution_evidence_recorded"));
        assert!(!failed.iter().any(|rule| rule == "left_backend_contract"));
        assert!(!failed.iter().any(|rule| rule == "right_backend_contract"));
    }

    #[test]
    fn generic_parity_prioritizes_missing_child_receipt_over_status_difference() {
        let scalar = receipt("i2_s-scalar-reference", &[4], "4", logits());
        let mut cuda = cuda_receipt(&[4], "4", logits());
        cuda["cases"][0] = json!({
            "id": "math",
            "question": "Answer with a single digit: 2+2=",
            "status": "command_failed",
            "exit_code": -1_073_740_791,
            "run_receipt_path": "target/bitnet/receipts/math.json",
            "quality": {
                "passed": false,
                "failed_rules": ["command_failed"]
            },
            "child_process": {
                "success": false,
                "timed_out": false,
                "exit_code": -1_073_740_791,
                "exit_code_hex": "0xC0000409",
                "crash_class": "windows_stack_buffer_overrun_or_fast_fail",
                "receipt_observed": false
            },
            "child_invocation": {
                "expected_receipt_path": "target/bitnet/receipts/math.json",
                "timeout_seconds": 120
            }
        });

        let report = build_generic_report(&scalar, &cuda);

        assert_eq!(report["summary"]["failed"], 1);
        assert_eq!(report["summary"]["first_divergence"]["kind"], "execution_evidence_recorded");
        assert_eq!(
            report["summary"]["first_divergence"]["right"]["child_process"]["crash_class"],
            "windows_stack_buffer_overrun_or_fast_fail"
        );
        let failed = report["cases"][0]["failed_rules"].as_array().unwrap();
        assert!(failed.iter().any(|rule| rule == "status"));
        assert!(failed.iter().any(|rule| rule == "execution_evidence_recorded"));
    }
}
