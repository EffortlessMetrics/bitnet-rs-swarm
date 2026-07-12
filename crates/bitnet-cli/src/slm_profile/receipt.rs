//! Receipt emission for resolved operator profiles.

use super::kaby::ModelRole;
use super::resolve::ResolvedProfile;
use serde_json::json;

pub fn profile_receipt(
    resolved: &ResolvedProfile,
    metadata: Option<&super::resolve::LoadedModelMetadata>,
    profile_supplied_prompts: bool,
    prompt_count: usize,
    threads: usize,
) -> serde_json::Value {
    let Some(profile_id) = resolved.profile_id else {
        return serde_json::Value::Null;
    };
    let (
        role,
        architecture,
        quant_format,
        model_sha256,
        tokenizer_source,
        tokenizer_authority,
        tokenizer_strict,
        chat_template,
        context_limit,
    ) = metadata.map_or_else(
        || {
            (
                None,
                "unknown".to_string(),
                "unknown".to_string(),
                "unknown".to_string(),
                "unknown".to_string(),
                "unknown".to_string(),
                false,
                None,
                0,
            )
        },
        |metadata| {
            (
                resolved.model_role,
                metadata.architecture.clone(),
                metadata.quant_format.clone(),
                metadata.model_sha256.clone(),
                metadata.tokenizer_source.clone(),
                metadata.tokenizer_authority.clone(),
                metadata.tokenizer_strict,
                metadata.chat_template.clone(),
                metadata.context_limit,
            )
        },
    );
    let role = role.or(resolved.model_role);
    let mode = profile_mode(resolved);
    json!({
        "schema_version": "1.1.0",
        "artifact_kind": "slm_cpu_kaby_opt_in_profile",
        "tracking_item": "SLM-CPU-247",
        "profile_id": profile_id,
        "enabled": true,
        "mode": mode,
        "profile_supplied_prompts": profile_supplied_prompts,
        "prompt_source": if profile_supplied_prompts { "profile_builtin_bounded_prompts" } else { "user_prompt_or_corpus" },
        "prompt_count": prompt_count,
        "model": {
            "role": role.map(ModelRole::id),
            "architecture": architecture,
            "quant_format": quant_format,
            "sha256": model_sha256,
            "tokenizer_source": tokenizer_source,
            "tokenizer_authority": tokenizer_authority,
            "tokenizer_strict": tokenizer_strict,
            "chat_template": chat_template,
            "context_limit": context_limit,
            "behavior_contract": role.map(|role| json!({
                "prompt_template": role.prompt_template(),
                "thinking_policy": if role.no_think() { "no_think" } else { "model_default" },
                "stop_policy": role.stop_policy(),
                "bounded_self_test_corpus": role.self_test_corpus(),
                "artifact_sha256": role.artifact_sha256(),
            })),
            "primary_model": { "file": "Qwen3-0.6B-Q8_0.gguf", "sha256": super::kaby::QWEN3_SHA256 },
            "second_model_proof": { "file": "qwen2.5-0.5b-instruct-q8_0.gguf", "sha256": super::kaby::QWEN25_SHA256 },
        },
        "applied_contract": {
            "runtime_api": "cpu",
            "selected_backend": "cpu-rust",
            "fallback_required": false,
            "strict_loader": resolved.strict_loader,
            "strict_tokenizer": resolved.strict_tokenizer,
            "prompt_template": resolved.prompt_template,
            "qwen_no_think": resolved.no_think,
            "greedy": resolved.greedy,
            "deterministic": resolved.deterministic,
            "max_new_tokens": resolved.max_new_tokens,
            "threads": threads,
            "recommended_threads": super::kaby::RECOMMENDED_THREADS,
            "fail_on_quality": resolved.fail_on_quality,
            "require_determinism": resolved.require_determinism,
            "allocation_audit": resolved.allocation_audit,
            "self_test": resolved.self_test,
            "warm_session_first": true,
            "receipt_output_required": true,
        },
        "no_bias_policy": {
            "only_proven_executable_role": "feed_forward.down_proj",
            "next_receipt_target": "feed_forward.up_proj",
            "candidate_execution_opt_in_only": true,
            "candidate_execution_enabled_by_profile": false,
            "default_path_when_gate_absent": "eager_f32_candle",
        },
        "unsupported_until_receipts": [
            "Q4/Q5 runtime", "server", "GPU/NPU/OpenVINO/UHD 620", "Qwen3.5",
            "BitNet QK256 changes", "broad chat quality", "sustained throughput", "default runtime promotion"
        ],
        "evidence": {
            "thread_recommendation_source": "SLM-CPU-205 dashboard envelope and SLM-CPU-245 timing/allocation receipt pair",
            "role_policy_source": "ci/slm-cpu/intel-i5-8250u/2026-06-05/qwen3-qwen25-slm-cpu-246-no-bias-role-expansion-policy.json",
        },
        "claim_boundary": {
            "default_runtime_changed": false,
            "candidate_execution_enabled": false,
            "speedup_claim": false,
            "allocation_reduction_claim": false,
            "broad_performance_claim": false,
            "broad_quality_claim": false,
            "q4_q5_runtime_support": false,
            "server_or_accelerator_claim": false,
            "qwen35_claim": false,
            "bitnet_qk256_claim": false,
        },
    })
}

fn profile_mode(resolved: &ResolvedProfile) -> &'static str {
    match (resolved.self_test, resolved.allocation_audit) {
        (false, false) => "normal",
        (true, false) => "self_test",
        (false, true) => "normal_allocation_audit",
        (true, true) => "self_test_allocation_audit",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(self_test: bool, allocation_audit: bool) -> ResolvedProfile {
        ResolvedProfile {
            profile_id: Some(super::super::kaby::PROFILE_ID),
            model_role: None,
            max_new_tokens: 4,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.1,
            strict_tokenizer: true,
            strict_loader: true,
            greedy: true,
            deterministic: true,
            threads: 4,
            prompt_template: "qwen".to_string(),
            no_think: true,
            fail_on_quality: self_test,
            require_determinism: self_test,
            allocation_audit,
            profile_supplied_prompts: self_test,
            self_test,
        }
    }

    #[test]
    fn receipt_mode_distinguishes_proof_and_audit_paths() {
        for (self_test, allocation_audit, expected) in [
            (false, false, "normal"),
            (true, false, "self_test"),
            (false, true, "normal_allocation_audit"),
            (true, true, "self_test_allocation_audit"),
        ] {
            let receipt =
                profile_receipt(&resolved(self_test, allocation_audit), None, false, 0, 4);
            assert_eq!(receipt["mode"], expected);
        }
    }
}
