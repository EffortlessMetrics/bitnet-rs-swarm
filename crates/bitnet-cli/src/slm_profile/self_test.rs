//! Bounded certification prompts, kept separate from normal profile use.

use super::kaby::ModelRole;

#[derive(Clone, Debug)]
pub struct ProfilePromptInput {
    pub case_id: String,
    pub prompt: String,
    pub repeat_index: usize,
    pub gate: Option<ProfileGate>,
    pub min_generated_tokens: usize,
    pub min_distinct_generated_tokens: usize,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ProfileGate {
    pub kind: String,
    #[serde(default)]
    pub expected: Option<String>,
    #[serde(default)]
    pub contains_any: Option<Vec<String>>,
    #[serde(default)]
    pub starts_with_any: Option<Vec<String>>,
    #[serde(default)]
    pub min_words: Option<usize>,
}

pub fn profile_prompt_inputs(profile_id: &str, role: ModelRole) -> Vec<ProfilePromptInput> {
    if profile_id != super::kaby::PROFILE_ID {
        return Vec::new();
    }
    match role {
        ModelRole::Qwen3Primary => {
            let water_gate = ProfileGate {
                kind: "starts_with_any".to_string(),
                expected: None,
                contains_any: None,
                starts_with_any: Some(vec!["Yes".to_string()]),
                min_words: None,
            };
            let capital_gate = ProfileGate {
                kind: "starts_with_any".to_string(),
                expected: None,
                contains_any: None,
                starts_with_any: Some(vec!["Paris".to_string()]),
                min_words: None,
            };
            vec![
                repeated_prompt(
                    "kaby_qwen3_water",
                    "Answer yes or no: is water wet?",
                    water_gate.clone(),
                    0,
                ),
                repeated_prompt(
                    "kaby_qwen3_water",
                    "Answer yes or no: is water wet?",
                    water_gate,
                    1,
                ),
                repeated_prompt(
                    "kaby_qwen3_capital_france",
                    "What is the capital of France? Answer with one word.",
                    capital_gate.clone(),
                    0,
                ),
                repeated_prompt(
                    "kaby_qwen3_capital_france",
                    "What is the capital of France? Answer with one word.",
                    capital_gate,
                    1,
                ),
            ]
        }
        ModelRole::Qwen25SecondModel => {
            let math_gate = ProfileGate {
                kind: "contains_any".to_string(),
                expected: None,
                contains_any: Some(vec!["4".to_string()]),
                starts_with_any: None,
                min_words: None,
            };
            let capital_gate = ProfileGate {
                kind: "contains_any".to_string(),
                expected: None,
                contains_any: Some(vec!["Paris".to_string()]),
                starts_with_any: None,
                min_words: None,
            };
            vec![
                repeated_prompt(
                    "kaby_qwen25_math_2_plus_2",
                    "What is 2+2? Answer briefly.",
                    math_gate.clone(),
                    0,
                ),
                repeated_prompt(
                    "kaby_qwen25_math_2_plus_2",
                    "What is 2+2? Answer briefly.",
                    math_gate,
                    1,
                ),
                repeated_prompt(
                    "kaby_qwen25_capital_france",
                    "What is the capital of France? Answer briefly.",
                    capital_gate.clone(),
                    0,
                ),
                repeated_prompt(
                    "kaby_qwen25_capital_france",
                    "What is the capital of France? Answer briefly.",
                    capital_gate,
                    1,
                ),
            ]
        }
    }
}

fn repeated_prompt(
    case_id: &str,
    prompt: &str,
    gate: ProfileGate,
    repeat_index: usize,
) -> ProfilePromptInput {
    ProfilePromptInput {
        case_id: case_id.to_string(),
        prompt: prompt.to_string(),
        repeat_index,
        gate: Some(gate),
        min_generated_tokens: 1,
        min_distinct_generated_tokens: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_uses_established_answer_corpus() {
        let prompts =
            profile_prompt_inputs(super::super::kaby::PROFILE_ID, ModelRole::Qwen3Primary);

        assert_eq!(prompts.len(), 4);
        assert_eq!(prompts[0].case_id, "kaby_qwen3_water");
        assert_eq!(prompts[1].case_id, "kaby_qwen3_water");
        assert_eq!(prompts[2].case_id, "kaby_qwen3_capital_france");
        assert_eq!(prompts[3].case_id, "kaby_qwen3_capital_france");
        assert_eq!(
            prompts[0].gate.as_ref().and_then(|gate| gate.starts_with_any.clone()),
            Some(vec!["Yes".to_string()])
        );
        assert_eq!(
            prompts[2].gate.as_ref().and_then(|gate| gate.starts_with_any.clone()),
            Some(vec!["Paris".to_string()])
        );
    }

    #[test]
    fn qwen25_uses_separate_completed_answer_corpus() {
        let prompts =
            profile_prompt_inputs(super::super::kaby::PROFILE_ID, ModelRole::Qwen25SecondModel);

        assert_eq!(prompts.len(), 4);
        assert_eq!(prompts[0].case_id, "kaby_qwen25_math_2_plus_2");
        assert_eq!(prompts[2].case_id, "kaby_qwen25_capital_france");
        assert_eq!(
            prompts[0].gate.as_ref().and_then(|gate| gate.contains_any.clone()),
            Some(vec!["4".to_string()])
        );
        assert_eq!(
            prompts[2].gate.as_ref().and_then(|gate| gate.contains_any.clone()),
            Some(vec!["Paris".to_string()])
        );
    }
}
