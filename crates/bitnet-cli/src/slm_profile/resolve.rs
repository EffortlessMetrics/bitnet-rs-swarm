//! Profile request and post-load resolution.

use anyhow::{Result, bail};
use std::collections::BTreeSet;
use std::path::Path;

use super::kaby::{self, ModelRole};

#[derive(Clone, Debug, Default)]
pub struct CliOverrides {
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub strict_tokenizer: Option<bool>,
    pub strict_loader: Option<bool>,
    pub greedy: Option<bool>,
    pub deterministic: Option<bool>,
    pub threads: Option<usize>,
    pub prompt_template: Option<String>,
    pub no_think: Option<bool>,
    pub fail_on_quality: Option<bool>,
    pub require_determinism: Option<bool>,
    pub allocation_audit: Option<bool>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LoadedModelMetadata {
    pub architecture: String,
    pub quant_format: String,
    pub model_sha256: String,
    pub tokenizer_source: String,
    pub tokenizer_authority: String,
    pub tokenizer_strict: bool,
    pub chat_template: Option<String>,
    pub context_limit: usize,
}

/// Inspect a GGUF and its strict tokenizer without loading model tensors.
///
/// This is the shared metadata boundary for both warm-session resolution and
/// the operator doctor. Profile authorization must consume this result rather
/// than a path or filename heuristic.
pub fn inspect_model_metadata(
    model_path: &Path,
    tokenizer_path: Option<&Path>,
) -> Result<LoadedModelMetadata> {
    if !model_path.is_file() {
        bail!("model artifact does not exist or is not a file: {}", model_path.display());
    }
    let bytes = std::fs::read(model_path)
        .map_err(|error| anyhow::anyhow!("failed to read {}: {error}", model_path.display()))?;
    let reader = bitnet_models::GgufReader::new(&bytes)
        .map_err(|error| anyhow::anyhow!("failed to parse GGUF metadata: {error}"))?;
    let architecture =
        reader.get_string_metadata("general.architecture").unwrap_or_else(|| "unknown".to_string());
    let quant_format = inspect_quant_format(&reader);
    let context_limit =
        reader.get_u32_metadata(&format!("{architecture}.context_length")).unwrap_or(0) as usize;
    let chat_template = reader.get_string_metadata("tokenizer.chat_template");
    let tokenizer_resolution =
        bitnet_tokenizers::auto::resolve_tokenizer(model_path, tokenizer_path, true)
            .map_err(|error| anyhow::anyhow!("failed to resolve strict tokenizer: {error}"))?;
    let tokenizer_source = tokenizer_resolution.source.as_str().to_string();
    let tokenizer_authority = match tokenizer_resolution.source {
        bitnet_tokenizers::auto::TokenizerSource::GgufMetadata => "present",
        bitnet_tokenizers::auto::TokenizerSource::Explicit
        | bitnet_tokenizers::auto::TokenizerSource::Sibling => "externally_supplied",
        bitnet_tokenizers::auto::TokenizerSource::CompatibilityFallback => "defaulted",
    }
    .to_string();
    let model_sha256 = sha256_bytes(&bytes);
    Ok(LoadedModelMetadata {
        architecture,
        quant_format,
        model_sha256,
        tokenizer_source,
        tokenizer_authority,
        tokenizer_strict: tokenizer_resolution.strict,
        chat_template,
        context_limit,
    })
}

fn inspect_quant_format(reader: &bitnet_models::GgufReader<'_>) -> String {
    let mut families = BTreeSet::new();
    for index in 0..reader.tensor_count() as usize {
        let Ok(info) = reader.get_tensor_info(index) else { continue };
        if info.tensor_type.is_quantized() {
            families.insert(format!("{:?}", info.tensor_type).to_ascii_uppercase());
        }
    }
    if families.len() == 1 {
        families.into_iter().next().unwrap_or_else(|| "unknown".to_string())
    } else if families.is_empty() {
        "unquantized".to_string()
    } else {
        families.into_iter().collect::<Vec<_>>().join(",")
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[derive(Clone, Debug)]
pub struct ResolvedProfile {
    pub profile_id: Option<&'static str>,
    pub model_role: Option<ModelRole>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub strict_tokenizer: bool,
    pub strict_loader: bool,
    pub greedy: bool,
    pub deterministic: bool,
    pub threads: usize,
    pub prompt_template: String,
    pub no_think: bool,
    pub fail_on_quality: bool,
    pub require_determinism: bool,
    pub allocation_audit: bool,
    pub profile_supplied_prompts: bool,
    pub self_test: bool,
}

pub fn validate_profile_request(
    profile: Option<&str>,
    requested_backend: &str,
) -> Result<Option<&'static str>> {
    let Some(profile) = profile.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    if profile != kaby::PROFILE_ID {
        bail!(
            "unsupported slm-warm-session --profile {profile}; supported profiles: {}",
            kaby::PROFILE_ID
        );
    }
    if !matches!(requested_backend, "cpu" | "auto") {
        bail!(
            "slm-warm-session --profile {profile} requires CPU routing (use --device cpu or leave device auto); got {requested_backend}"
        );
    }
    Ok(Some(kaby::PROFILE_ID))
}

pub fn resolve_profile(
    profile: Option<&str>,
    requested_backend: &str,
    overrides: CliOverrides,
    metadata: Option<&LoadedModelMetadata>,
    self_test: bool,
    has_external_prompts: bool,
    has_corpus: bool,
) -> Result<ResolvedProfile> {
    let profile_id = validate_profile_request(profile, requested_backend)?;
    let role = match (profile_id, metadata) {
        (Some(_profile_id), Some(metadata)) => Some(kaby::classify_model(
            &metadata.architecture,
            &metadata.quant_format,
            &metadata.model_sha256,
            &metadata.tokenizer_authority,
            metadata.chat_template.as_deref(),
            metadata.context_limit,
        )?),
        (Some(_), None) => None,
        (None, _) => None,
    };

    let profile_active = profile_id.is_some();
    let profile_self_test = profile_active && self_test;
    let prompt_template = overrides
        .prompt_template
        .clone()
        .or_else(|| role.map(ModelRole::prompt_template).map(str::to_string))
        .unwrap_or_else(|| "qwen2.5".to_string());
    let profile_role = role;
    let default_threads = profile_role.map(|_| kaby::RECOMMENDED_THREADS).unwrap_or(0);
    let default_max_new_tokens = profile_role.map(ModelRole::max_new_tokens).unwrap_or(32);
    let default_no_think = profile_role.map(ModelRole::no_think).unwrap_or(false);
    Ok(ResolvedProfile {
        profile_id,
        model_role: profile_role,
        max_new_tokens: overrides.max_new_tokens.unwrap_or(default_max_new_tokens),
        temperature: overrides.temperature.unwrap_or(if profile_active { 0.0 } else { 0.0 }),
        top_k: overrides.top_k.unwrap_or(0),
        top_p: overrides.top_p.unwrap_or(1.0),
        repetition_penalty: overrides.repetition_penalty.unwrap_or(1.1),
        strict_tokenizer: overrides.strict_tokenizer.unwrap_or(profile_active),
        strict_loader: overrides.strict_loader.unwrap_or(profile_active),
        greedy: overrides.greedy.unwrap_or(profile_active),
        deterministic: overrides.deterministic.unwrap_or(profile_active),
        threads: overrides.threads.unwrap_or(default_threads),
        prompt_template,
        no_think: overrides.no_think.unwrap_or(default_no_think),
        fail_on_quality: overrides.fail_on_quality.unwrap_or(profile_self_test),
        require_determinism: overrides.require_determinism.unwrap_or(profile_self_test),
        allocation_audit: overrides.allocation_audit.unwrap_or(false),
        profile_supplied_prompts: profile_self_test && !has_external_prompts && !has_corpus,
        self_test,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_metadata() -> LoadedModelMetadata {
        LoadedModelMetadata {
            architecture: "qwen3".to_string(),
            quant_format: "Q8_0".to_string(),
            model_sha256: kaby::QWEN3_SHA256.to_string(),
            tokenizer_source: "gguf".to_string(),
            tokenizer_authority: "gguf_tokenizer".to_string(),
            tokenizer_strict: true,
            chat_template: Some("{{ messages }}".to_string()),
            context_limit: 40_960,
        }
    }

    #[test]
    fn explicit_values_override_profile_values() -> Result<()> {
        let resolved = resolve_profile(
            Some(kaby::PROFILE_ID),
            "cpu",
            CliOverrides {
                max_new_tokens: Some(32),
                threads: Some(8),
                prompt_template: Some("raw".to_string()),
                no_think: Some(false),
                greedy: Some(false),
                deterministic: Some(false),
                allocation_audit: Some(true),
                ..CliOverrides::default()
            },
            Some(&qwen3_metadata()),
            false,
            true,
            false,
        )?;

        assert_eq!(resolved.max_new_tokens, 32);
        assert_eq!(resolved.threads, 8);
        assert_eq!(resolved.prompt_template, "raw");
        assert!(!resolved.no_think);
        assert!(!resolved.greedy);
        assert!(!resolved.deterministic);
        assert!(resolved.allocation_audit);
        assert!(!resolved.profile_supplied_prompts);
        Ok(())
    }

    #[test]
    fn normal_profile_does_not_enable_proof_instrumentation() -> Result<()> {
        let resolved = resolve_profile(
            Some(kaby::PROFILE_ID),
            "cpu",
            CliOverrides::default(),
            Some(&qwen3_metadata()),
            false,
            true,
            false,
        )?;

        assert_eq!(resolved.threads, kaby::RECOMMENDED_THREADS);
        assert_eq!(resolved.max_new_tokens, kaby::PROFILE_MAX_NEW_TOKENS);
        assert!(!resolved.fail_on_quality);
        assert!(!resolved.require_determinism);
        assert!(!resolved.allocation_audit);
        assert!(!resolved.profile_supplied_prompts);
        Ok(())
    }

    #[test]
    fn qwen25_profile_default_leaves_room_for_its_answer_corpus() -> Result<()> {
        let mut metadata = qwen3_metadata();
        metadata.architecture = "qwen2".to_string();
        metadata.model_sha256 = kaby::QWEN25_SHA256.to_string();
        metadata.context_limit = 32768;

        let resolved = resolve_profile(
            Some(kaby::PROFILE_ID),
            "cpu",
            CliOverrides::default(),
            Some(&metadata),
            true,
            false,
            false,
        )?;

        assert_eq!(resolved.max_new_tokens, 8);
        assert_eq!(resolved.model_role, Some(ModelRole::Qwen25SecondModel));
        Ok(())
    }

    #[test]
    fn loaded_context_limit_is_part_of_profile_authority() -> Result<()> {
        let mut metadata = qwen3_metadata();
        metadata.context_limit = 4096;

        let error = match resolve_profile(
            Some(kaby::PROFILE_ID),
            "cpu",
            CliOverrides::default(),
            Some(&metadata),
            false,
            true,
            false,
        ) {
            Ok(_) => return Err(anyhow::anyhow!("profile accepted an unexpected context limit")),
            Err(error) => error,
        };

        assert!(error.to_string().contains("context limit mismatch"));
        Ok(())
    }

    #[test]
    fn self_test_is_the_only_source_of_builtin_profile_prompts() -> Result<()> {
        let resolved = resolve_profile(
            Some(kaby::PROFILE_ID),
            "cpu",
            CliOverrides::default(),
            Some(&qwen3_metadata()),
            true,
            false,
            false,
        )?;

        assert!(resolved.self_test);
        assert!(resolved.profile_supplied_prompts);
        assert!(resolved.fail_on_quality);
        assert!(resolved.require_determinism);
        assert!(!resolved.allocation_audit);
        Ok(())
    }
}
