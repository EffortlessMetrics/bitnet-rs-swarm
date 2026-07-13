//! Kaby Lake dense-Qwen profile contract.

use anyhow::{bail, Result};

pub const PROFILE_ID: &str = "kaby-qwen-q8";
pub const RECOMMENDED_THREADS: usize = 4;
pub const PROFILE_MAX_NEW_TOKENS: usize = 4;
pub const QWEN3_SHA256: &str = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031";
pub const QWEN25_SHA256: &str = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
pub const TOKENIZER_SOURCE: &str = "gguf_metadata";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelRole {
    Qwen3Primary,
    Qwen25SecondModel,
}

impl ModelRole {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Qwen3Primary => "primary_qwen3_q8_0",
            Self::Qwen25SecondModel => "second_model_qwen25_q8_0",
        }
    }

    pub const fn architecture(self) -> &'static str {
        match self {
            Self::Qwen3Primary => "qwen3",
            Self::Qwen25SecondModel => "qwen2",
        }
    }

    pub const fn prompt_template(self) -> &'static str {
        match self {
            Self::Qwen3Primary => "qwen",
            Self::Qwen25SecondModel => "qwen2.5",
        }
    }

    pub const fn no_think(self) -> bool {
        matches!(self, Self::Qwen3Primary)
    }

    pub const fn max_new_tokens(self) -> usize {
        match self {
            Self::Qwen3Primary => PROFILE_MAX_NEW_TOKENS,
            // The Qwen2.5 sanity corpus needs enough room to finish
            // "2+2 equals 4." under its model-specific chat behavior.
            Self::Qwen25SecondModel => 8,
        }
    }

    pub const fn context_limit(self) -> usize {
        match self {
            Self::Qwen3Primary => 40_960,
            Self::Qwen25SecondModel => 32_768,
        }
    }

    pub const fn artifact_sha256(self) -> &'static str {
        match self {
            Self::Qwen3Primary => QWEN3_SHA256,
            Self::Qwen25SecondModel => QWEN25_SHA256,
        }
    }

    pub const fn stop_policy(self) -> &'static str {
        match self {
            Self::Qwen3Primary => "qwen_template_defaults_plus_manual_stops",
            Self::Qwen25SecondModel => "qwen2.5_template_defaults_plus_manual_stops",
        }
    }

    pub const fn self_test_corpus(self) -> &'static str {
        match self {
            Self::Qwen3Primary => "slm-cpu-247-kaby-qwen3-self-test-v1",
            Self::Qwen25SecondModel => "slm-cpu-247-kaby-qwen25-self-test-v1",
        }
    }
}

pub fn classify_model(
    architecture: &str,
    quant_format: &str,
    model_sha256: &str,
    tokenizer_source: &str,
    tokenizer_authority: &str,
    tokenizer_strict: bool,
    chat_template: Option<&str>,
    context_limit: usize,
) -> Result<ModelRole> {
    let architecture = architecture.trim().to_ascii_lowercase().replace(['-', '.', ' '], "_");
    let role = match architecture.as_str() {
        "qwen3" | "qwen_3" => ModelRole::Qwen3Primary,
        "qwen2" | "qwen2_5" | "qwen_2" | "qwen_2_5" => ModelRole::Qwen25SecondModel,
        value if value.starts_with("qwen3_5") || value.starts_with("qwen35") => {
            bail!("profile {PROFILE_ID} rejects Qwen3.5 hybrid architecture metadata")
        }
        other => bail!(
            "profile {PROFILE_ID} requires Qwen2/Qwen2.5 or Qwen3 architecture metadata; got {other}"
        ),
    };
    if !quant_format.eq_ignore_ascii_case("Q8_0") {
        bail!("profile {PROFILE_ID} requires Q8_0 quantization metadata; got {quant_format}");
    }
    if model_sha256 != role.artifact_sha256() {
        bail!(
            "profile {PROFILE_ID} model SHA mismatch for {}: expected {}, got {}",
            role.architecture(),
            role.artifact_sha256(),
            model_sha256
        );
    }
    if !tokenizer_source.eq_ignore_ascii_case(TOKENIZER_SOURCE)
        || !tokenizer_authority.eq_ignore_ascii_case(TOKENIZER_SOURCE)
    {
        bail!(
            "profile {PROFILE_ID} requires tokenizer authority {TOKENIZER_SOURCE}; got source={tokenizer_source}, authority={tokenizer_authority}"
        );
    }
    if !tokenizer_strict {
        bail!("profile {PROFILE_ID} requires strict tokenizer resolution (tokenizer_strict=true)");
    }
    if chat_template.map(str::trim).filter(|value| !value.is_empty()).is_none() {
        bail!("profile {PROFILE_ID} requires GGUF tokenizer.chat_template metadata");
    }
    if context_limit != role.context_limit() {
        bail!(
            "profile {PROFILE_ID} context limit mismatch for {}: expected {}, got {}",
            role.architecture(),
            role.context_limit(),
            context_limit
        );
    }
    Ok(role)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHAT_TEMPLATE: &str = "{{ messages }}";

    fn classify(
        tokenizer_source: &str,
        tokenizer_authority: &str,
        tokenizer_strict: bool,
    ) -> Result<ModelRole> {
        classify_model(
            "qwen3",
            "Q8_0",
            QWEN3_SHA256,
            tokenizer_source,
            tokenizer_authority,
            tokenizer_strict,
            Some(CHAT_TEMPLATE),
            40_960,
        )
    }

    #[test]
    fn accepts_only_strict_gguf_tokenizer_authority() -> Result<()> {
        assert_eq!(classify(TOKENIZER_SOURCE, TOKENIZER_SOURCE, true)?, ModelRole::Qwen3Primary);
        Ok(())
    }

    #[test]
    fn rejects_external_tokenizer_authority() {
        let error =
            classify("explicit", "explicit", true).expect_err("external tokenizer accepted");
        assert!(error.to_string().contains("gguf_metadata"));
    }

    #[test]
    fn rejects_non_strict_tokenizer_resolution() {
        let error = classify(TOKENIZER_SOURCE, TOKENIZER_SOURCE, false)
            .expect_err("non-strict tokenizer accepted");
        assert!(error.to_string().contains("tokenizer_strict=true"));
    }
}
