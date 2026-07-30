//! BitNet model implementation

use crate::dense_gguf_q8_dispatch::{DenseQ8DispatchSelection, select_dense_q8_runtime};
use crate::dense_gguf_q8_sidecar::{
    DenseGgufQ8SidecarDescriptor, DenseGgufQ8SidecarRegistry,
    DenseQ8SourceOrderKernelContractStatus, dense_q8_runtime_compute_tensor_from_env,
};
use crate::transformer::{KVCache, TransformerModel};
use bitnet_common::{
    BitNetConfig, BitNetError, BitNetTensor, ConcreteTensor, Device, Result, Tensor,
};
use bitnet_transformer::{
    DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary, DenseLinearPackedQ8Payload,
    DenseLinearRuntimeHookDescriptor, DenseLinearRuntimeHookRegistry,
};
use candle_core::{DType, Tensor as CandleTensor};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ModelForwardSourceContext {
    pub prior_layer_output: ConcreteTensor,
    pub final_norm_output: ConcreteTensor,
    pub final_block_source: Option<ModelFinalBlockSourceContext>,
    pub penultimate_block_source: Option<ModelFinalBlockSourceContext>,
    pub antepenultimate_block_source: Option<ModelFinalBlockSourceContext>,
    pub pre_antepenultimate_block_source: Option<ModelFinalBlockSourceContext>,
    pub earlier_block_source: Option<ModelFinalBlockSourceContext>,
    pub block_sources: Vec<ModelFinalBlockSourceContext>,
    pub attention_output_sources: Vec<ModelAttentionOutputSourceContext>,
    pub qkv_projection_sources: Vec<ModelQkvProjectionSourceContext>,
}

#[derive(Debug, Clone)]
pub struct ModelFinalBlockSourceContext {
    pub layer_idx: usize,
    pub block_input: ConcreteTensor,
    pub attention_output: ConcreteTensor,
    pub post_attention_residual: ConcreteTensor,
    pub feed_forward_output: ConcreteTensor,
    pub block_output: ConcreteTensor,
}

#[derive(Debug, Clone)]
pub struct ModelAttentionOutputSourceContext {
    pub layer_idx: usize,
    pub attention_input: ConcreteTensor,
    pub q_projection: ConcreteTensor,
    pub k_projection: ConcreteTensor,
    pub v_projection: ConcreteTensor,
    pub q_heads: ConcreteTensor,
    pub k_heads: ConcreteTensor,
    pub v_heads: ConcreteTensor,
    pub q_norm: ConcreteTensor,
    pub k_norm: ConcreteTensor,
    pub q_rope: ConcreteTensor,
    pub k_rope: ConcreteTensor,
    pub k_context: ConcreteTensor,
    pub v_context: ConcreteTensor,
    pub expanded_k: ConcreteTensor,
    pub expanded_v: ConcreteTensor,
    pub scores: ConcreteTensor,
    pub probabilities: ConcreteTensor,
    pub value_mix_output_heads: ConcreteTensor,
    pub output_projection_input: ConcreteTensor,
    pub sub_layernorm_output: Option<ConcreteTensor>,
    pub attention_output: ConcreteTensor,
}

#[derive(Debug, Clone)]
pub struct ModelQk256DispatchDeltaContext {
    pub bitnet_linear_layers_total: u64,
    pub bitnet_linear_layers_on_cuda: u64,
    pub bitnet_linear_layers_on_a770_opencl: u64,
    pub bitnet_linear_layers_cpu_fallback: u64,
    pub unsupported_ops: Vec<String>,
    pub execution_claim: String,
}

#[derive(Debug, Clone)]
pub struct ModelQk256CpuHotPathDeltaContext {
    pub qk256_f32_scalar_gemv_invocations: u64,
    pub qk256_f32_avx2_gemv_invocations: u64,
    pub qk256_i8s_scaled_scalar_invocations: u64,
    pub qk256_i8s_scaled_avx2_invocations: u64,
    pub qk256_flat_bytes_extracted_count: u64,
    pub input_rows_materialized_count: u64,
    pub output_rows_allocated_count: u64,
    pub requested_kernel: Option<String>,
    pub selected_kernel: Option<String>,
    pub qk256_execution_path: String,
}

#[derive(Debug, Clone)]
pub struct ModelA770OpenClRuntimeDeltaContext {
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub kernel_invocations: u64,
}

#[derive(Debug, Clone)]
pub struct ModelQkvProjectionDispatchReplayContext {
    pub input_rows: usize,
    pub output_rows: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,
    pub inline_scale: Option<f32>,
    pub cpu_output: ConcreteTensor,
    pub opencl_policy_output: ConcreteTensor,
    pub a770_output: Option<ConcreteTensor>,
    pub device_expression_trace: Option<ModelQk256DeviceExpressionTraceContext>,
    pub device_intermediate_trace: Option<ModelQk256DeviceIntermediateTraceContext>,
    pub focused_operands: Option<ModelQk256FocusedRawOperandsContext>,
    pub full_projection_operands: Option<ModelQk256FullProjectionRawOperandsContext>,
    pub cpu: ModelQkvProjectionDispatchReplayCpuContext,
    pub a770: ModelQkvProjectionDispatchReplayA770Context,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelQk256FocusedRawOperandsContext {
    pub input_row_index: usize,
    pub output_index: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,
    pub packed_qk256_scope: String,
    pub activation_sum: i32,
    pub activation_scale_bits: u32,
    pub weight_scale_bits: u32,
    pub activations_i8: Vec<i8>,
    pub packed_qk256: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelQk256FullProjectionRawOperandsContext {
    pub input_row_index: usize,
    pub rows: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,
    pub packed_qk256_scope: String,
    pub activation_sum: i32,
    pub activation_scale_bits: u32,
    pub weight_scale_bits: u32,
    pub activations_i8: Vec<i8>,
    pub packed_qk256: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ModelQk256DeviceExpressionTraceContext {
    pub input_row_index: usize,
    pub sample_limit: usize,
    pub sample_count: usize,
    pub samples: Vec<ModelQk256DeviceExpressionSampleContext>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelQk256DeviceExpressionSampleContext {
    pub output_index: usize,
    pub int_dot: i32,
    pub activation_sum: i32,
    pub adjusted_dot: i32,
    pub activation_scale: f32,
    pub activation_scale_bits: u32,
    pub weight_scale: f32,
    pub weight_scale_bits: u32,
    pub div_then_mul: f32,
    pub mul_then_div: f32,
    pub reciprocal_then_mul: f32,
    pub f64_div_then_mul_cast: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelQk256DeviceIntermediateTraceContext {
    pub compiled_opencl: bool,
    pub attempted: bool,
    pub success: bool,
    pub error: Option<String>,
    pub input_row_index: usize,
    pub sample_limit: usize,
    pub sample_count: usize,
    pub platform_index: Option<usize>,
    pub device_index: Option<usize>,
    pub platform_name: Option<String>,
    pub runtime_device: Option<String>,
    pub vendor: Option<String>,
    pub driver_version: Option<String>,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub kernel_invocations: usize,
    pub samples: Vec<ModelQk256DeviceIntermediateSampleContext>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelQk256DeviceIntermediateSampleContext {
    pub output_index: usize,
    pub int_dot: i32,
    pub activation_sum: i32,
    pub adjusted_dot: i32,
    pub activation_scale_bits: u32,
    pub weight_scale_bits: u32,
    pub adjusted_f32_bits: u32,
    pub output_bits: u32,
    pub output: f32,
    pub div_then_mul_bits: u32,
    pub div_then_mul: f32,
    pub mul_then_div_bits: u32,
    pub mul_then_div: f32,
    pub reciprocal_then_mul_bits: u32,
    pub reciprocal_then_mul: f32,
    pub volatile_div_then_mul_bits: u32,
    pub volatile_div_then_mul: f32,
}

#[derive(Debug, Clone)]
pub struct ModelQkvProjectionDispatchReplayCpuContext {
    pub scalar_invocations: u64,
    pub execution_path: String,
}

#[derive(Debug, Clone)]
pub struct ModelQkvProjectionDispatchReplayA770Context {
    pub compiled_opencl: bool,
    pub attempted: bool,
    pub success: bool,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub kernel_invocations: u64,
    pub last_device: Option<ModelA770OpenClRuntimeDeviceContext>,
    pub error: Option<String>,
    pub execution_path: String,
}

#[derive(Debug, Clone)]
pub struct ModelA770OpenClRuntimeDeviceContext {
    pub platform_index: usize,
    pub device_index: usize,
    pub platform_name: String,
    pub runtime_device: String,
    pub vendor: String,
    pub driver_version: String,
}

#[derive(Debug, Clone)]
pub struct ModelQkvProjectionSourceContext {
    pub layer_idx: usize,
    pub projection: String,
    pub tensor_name: String,
    pub qk256_key: String,
    pub qk256_raw_tensor_present: bool,
    pub input: ConcreteTensor,
    pub output: ConcreteTensor,
    pub dispatch_delta: ModelQk256DispatchDeltaContext,
    pub cpu_hot_path_delta: ModelQk256CpuHotPathDeltaContext,
    pub a770_opencl_runtime_delta: ModelA770OpenClRuntimeDeltaContext,
    pub dispatch_replay: Option<ModelQkvProjectionDispatchReplayContext>,
    pub dispatch_replay_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelForwardDiagnosticOutput {
    pub output: ConcreteTensor,
    pub source_context: Option<ModelForwardSourceContext>,
}

/// Trait for BitNet models
pub trait Model: Send + Sync {
    fn config(&self) -> &BitNetConfig;
    fn forward(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor>;
    fn forward_with_no_bias_callsite_descriptor(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
        descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    ) -> Result<ConcreteTensor> {
        let _ = (input, cache, descriptor);
        Err(BitNetError::Validation(
            "prompt-bound no-bias apply-linear descriptor forwarding is only implemented for transformer-backed GGUF models".to_string(),
        ))
    }
    fn forward_with_source_context(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
    ) -> Result<ModelForwardDiagnosticOutput> {
        Ok(ModelForwardDiagnosticOutput {
            output: self.forward(input, cache)?,
            source_context: None,
        })
    }
    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor>;
    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor>;
    fn dense_q8_hook_selection_receipt(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": 1,
            "artifact_kind": "dense_gguf_q8_hook_selection_receipt_gate",
            "selected_path": "eager_f32_candle",
            "selected_kernel": "dense-f32-candle-linear",
            "sidecar_descriptor_count": 0,
            "hook_boundary_count": 0,
            "sidecar_descriptors_reached_transformer": false,
            "example_boundary": serde_json::Value::Null,
            "all_boundaries_preserve_eager_f32": true,
            "runtime_compute_enabled": false,
            "dense_runtime_replaced": false,
            "speedup_claim": false,
            "fallback_used": false,
            "remaining_blockers": [
                "dense_q8_sidecar_registry_missing_or_not_exposed_for_model_trait_object"
            ],
            "next_safe_step": "use a strict dense GGUF model with carried Q8_0 sidecar descriptors before evaluating packed sidecar compute"
        })
    }
}

/// BitNet model implementation
pub struct BitNetModel {
    config: BitNetConfig,
    device: Device,
    tensors: HashMap<String, CandleTensor>,
    transformer: Option<Arc<TransformerModel>>,
    dense_q8_sidecars: DenseGgufQ8SidecarRegistry,
}

impl BitNetModel {
    pub fn new(config: BitNetConfig, device: Device) -> Self {
        Self {
            config,
            device,
            tensors: HashMap::new(),
            transformer: None,
            dense_q8_sidecars: DenseGgufQ8SidecarRegistry::default(),
        }
    }

    /// Create a BitNet model from GGUF tensors
    pub fn from_gguf(
        config: BitNetConfig,
        tensors: HashMap<String, CandleTensor>,
        raw_tensors: HashMap<String, CandleTensor>,
        device: Device,
    ) -> Result<Self> {
        Self::from_gguf_with_dense_q8_sidecars(
            config,
            tensors,
            raw_tensors,
            DenseGgufQ8SidecarRegistry::default(),
            device,
        )
    }

    /// Create a BitNet model from GGUF tensors and inert dense Q8 sidecar metadata.
    pub fn from_gguf_with_dense_q8_sidecars(
        config: BitNetConfig,
        tensors: HashMap<String, CandleTensor>,
        raw_tensors: HashMap<String, CandleTensor>,
        dense_q8_sidecars: DenseGgufQ8SidecarRegistry,
        device: Device,
    ) -> Result<Self> {
        tracing::debug!(
            "from_gguf: received config: hidden={}, n_heads={}, n_kv_heads={}",
            config.model.hidden_size,
            config.model.num_heads,
            config.model.num_key_value_heads
        );
        tracing::debug!(
            "from_gguf: received {} tensors, {} raw QK256 tensors",
            tensors.len(),
            raw_tensors.len()
        );
        if !dense_q8_sidecars.is_empty() {
            tracing::debug!(
                "from_gguf: carrying {} inert dense Q8_0 sidecar descriptors; eager F32 runtime remains selected",
                dense_q8_sidecars.descriptor_count()
            );
        }

        // Validate that required tensors are present
        // LM head can be tied to embeddings, so check for either output.weight or embeddings
        let has_output = tensors.contains_key("output.weight")
            || tensors.contains_key("lm_head.weight")
            || tensors.contains_key("head.weight");

        let has_embeddings = tensors.contains_key("token_embd.weight")
            || tensors.contains_key("tok_embeddings.weight")
            || tensors.contains_key("model.embed_tokens.weight");

        if !has_embeddings {
            return Err(BitNetError::Validation(
                "Missing required tensor: token embeddings (token_embd.weight or equivalent)"
                    .to_string(),
            ));
        }

        if !has_output && !has_embeddings {
            return Err(BitNetError::Validation(
                "Missing both output.weight and token_embd.weight - cannot compute logits"
                    .to_string(),
            ));
        }

        // Try to build transformer model; propagate errors so missing weights fail fast
        let transformer =
            Self::build_transformer(&config, &tensors, &raw_tensors, &dense_q8_sidecars, &device)?;

        Ok(Self { config, device, tensors, transformer: Some(transformer), dense_q8_sidecars })
    }

    /// Build transformer model from loaded tensors
    fn build_transformer(
        config: &BitNetConfig,
        tensors: &HashMap<String, CandleTensor>,
        raw_tensors: &HashMap<String, CandleTensor>,
        dense_q8_sidecars: &DenseGgufQ8SidecarRegistry,
        device: &Device,
    ) -> Result<Arc<TransformerModel>> {
        use crate::weight_mapper::{
            create_var_builder, normalize_model_tensors, remap_gguf_weights,
        };

        // Create a VarBuilder that uses our loaded tensors
        let device = device.to_candle().map_err(|e| BitNetError::Validation(e.to_string()))?;

        if tensors.is_empty() {
            return Err(BitNetError::Validation("No model tensors provided".to_string()));
        }

        // Remap tensor names to match our transformer module structure
        let mut mapped = remap_gguf_weights(tensors)?;

        // Normalize embeddings, lm_head, and all layer tensors, detect vocab size and hidden size
        let (detected_vocab, detected_hidden) = normalize_model_tensors(&mut mapped, config)?;

        // Update config with detected values
        let mut updated_config = config.clone();
        if updated_config.model.vocab_size != detected_vocab {
            tracing::info!(
                "Updating vocab_size from {} to {} based on tensor shapes",
                updated_config.model.vocab_size,
                detected_vocab
            );
            updated_config.model.vocab_size = detected_vocab;
        }
        if updated_config.model.hidden_size != detected_hidden {
            tracing::info!(
                "Updating hidden_size from {} to {} based on tensor shapes",
                updated_config.model.hidden_size,
                detected_hidden
            );
            updated_config.model.hidden_size = detected_hidden;
        }

        // Remap raw_tensors keys (QK256 tensors) to match transformer structure
        // Keys like "blk.0.attn_q.weight.qk256_qs" -> "layers.0.attention.q_proj.weight.qk256_qs"
        // The remapper now handles .qk256_qs suffix (strips, remaps, re-appends)
        let raw_mapped = remap_gguf_weights(raw_tensors)?;
        let dense_linear_hooks = dense_q8_runtime_hooks_from_sidecars(dense_q8_sidecars);

        let vb = create_var_builder(mapped.clone(), DType::F32, &device)?;
        let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
            updated_config,
            vb,
            raw_mapped,
            dense_linear_hooks,
        )?;
        Ok(Arc::new(model))
    }

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&CandleTensor> {
        self.tensors.get(name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Inert dense Q8_0 sidecar descriptors carried from GGUF loading.
    pub fn dense_q8_sidecars(&self) -> &DenseGgufQ8SidecarRegistry {
        &self.dense_q8_sidecars
    }

    /// Behavior-preserving dense Q8_0 dispatch selector.
    ///
    /// This currently always selects the eager F32 Candle path. Packed Q8_0
    /// sidecar descriptors are exposed only as unavailable candidates until a
    /// later slice proves generated-ID/text and strict-receipt equivalence.
    pub fn dense_q8_dispatch_selection(&self, tensor_name: &str) -> DenseQ8DispatchSelection {
        select_dense_q8_runtime(tensor_name, &self.dense_q8_sidecars)
    }

    /// Receipt-oriented dense-linear hook selection identity.
    ///
    /// This is intentionally observational. It records which Q8_0 sidecar
    /// descriptors reached the transformer hook boundary while preserving the
    /// eager F32 Candle runtime path until behavior-preserving after receipts
    /// and a packed compute kernel exist.
    pub fn dense_q8_hook_selection_receipt(&self) -> serde_json::Value {
        let hook_boundaries = self
            .transformer
            .as_ref()
            .map(|transformer| transformer.dense_linear_runtime_hook_boundaries())
            .unwrap_or_default();
        let example = hook_boundaries.first();
        let runtime_boundary = hook_boundaries.iter().find(|boundary| {
            boundary.runtime_compute_enabled || boundary.source_order_candidate_runtime_enabled
        });
        let payload_boundary =
            hook_boundaries.iter().find(|boundary| boundary.sidecar_payload_bytes_available);
        let selected_boundary = runtime_boundary.or(payload_boundary).or(example);
        let payload_bearing_boundary_count = hook_boundaries
            .iter()
            .filter(|boundary| boundary.sidecar_payload_bytes_available)
            .count();
        let runtime_compute_enabled = hook_boundaries.iter().any(|boundary| {
            boundary.runtime_compute_enabled || boundary.source_order_candidate_runtime_enabled
        });
        let dense_runtime_replaced =
            hook_boundaries.iter().any(|boundary| boundary.dense_runtime_replaced);
        let has_payload_order_mismatch = hook_boundaries.iter().any(|boundary| {
            boundary.sidecar_payload_bytes_available
                && boundary.sidecar_payload_contract_valid
                && !boundary.sidecar_payload_order_matches_runtime_shape
        });
        let sidecar_descriptor_count = self.dense_q8_sidecars.descriptor_count();
        let hook_boundary_count = hook_boundaries.len();
        let sidecar_descriptors_reached_transformer = hook_boundary_count > 0
            && hook_boundaries.iter().any(|boundary| boundary.sidecar_descriptor_present);
        let boundary_json = |boundary: &bitnet_transformer::DenseLinearRuntimeHookBoundary| {
            serde_json::json!({
                "tensor_name": boundary.tensor_name,
                "selected_path": boundary.selected_path,
                "selected_kernel": boundary.selected_kernel,
                "sidecar_descriptor_present": boundary.sidecar_descriptor_present,
                "sidecar_role": boundary.sidecar_role,
                "sidecar_payload_sha256": boundary.sidecar_payload_sha256,
                "sidecar_payload_bytes_available": boundary.sidecar_payload_bytes_available,
                "sidecar_payload_bytes": boundary.sidecar_payload_bytes,
                "sidecar_q8_block_count": boundary.sidecar_q8_block_count,
                "sidecar_matrix_rows": boundary.sidecar_matrix_rows,
                "sidecar_matrix_cols": boundary.sidecar_matrix_cols,
                "sidecar_payload_contract_valid": boundary.sidecar_payload_contract_valid,
                "sidecar_payload_order_matches_runtime_shape": boundary.sidecar_payload_order_matches_runtime_shape,
                "source_order_q8_matvec_candidate": boundary.source_order_q8_matvec_candidate,
                "source_order_selected_path": boundary.source_order_selected_path,
                "source_order_selected_kernel": boundary.source_order_selected_kernel,
                "source_order_input_dim": boundary.source_order_input_dim,
                "source_order_output_dim": boundary.source_order_output_dim,
                "source_order_candidate_receipt_identity": boundary.source_order_candidate_receipt_identity,
                "source_order_candidate_runtime_enabled": boundary.source_order_candidate_runtime_enabled,
                "runtime_compute_enabled": boundary.runtime_compute_enabled,
                "eager_f32_runtime_preserved": boundary.eager_f32_runtime_preserved,
                "dense_runtime_replaced": boundary.dense_runtime_replaced,
                "speedup_claim": boundary.speedup_claim,
                "generated_id_preservation_required_before_runtime_use": boundary.generated_id_preservation_required_before_runtime_use,
                "next_receipt_gate": boundary.next_receipt_gate,
            })
        };

        serde_json::json!({
            "schema": 1,
            "artifact_kind": "dense_gguf_q8_hook_selection_receipt_gate",
            "selected_path": selected_boundary
                .map(|boundary| boundary.selected_path)
                .unwrap_or("eager_f32_candle"),
            "selected_kernel": selected_boundary
                .map(|boundary| boundary.selected_kernel)
                .unwrap_or("dense-f32-candle-linear"),
            "sidecar_descriptor_count": sidecar_descriptor_count,
            "hook_boundary_count": hook_boundary_count,
            "sidecar_descriptors_reached_transformer": sidecar_descriptors_reached_transformer,
            "example_boundary": example.map(boundary_json),
            "payload_bearing_boundary_count": payload_bearing_boundary_count,
            "payload_bearing_boundary": payload_boundary.map(boundary_json),
            "runtime_boundary": runtime_boundary.map(boundary_json),
            "all_boundaries_preserve_eager_f32": hook_boundaries.iter().all(|boundary| boundary.preserves_eager_f32()),
            "runtime_compute_enabled": runtime_compute_enabled,
            "dense_runtime_replaced": dense_runtime_replaced,
            "speedup_claim": false,
            "fallback_used": false,
            "remaining_blockers": if runtime_compute_enabled {
                serde_json::json!([
                    "before_after_qwen3_q8_warm_session_receipts_with_hook_selection_identity_missing"
                ])
            } else if has_payload_order_mismatch {
                serde_json::json!([
                    "packed_q8_sidecar_payload_order_does_not_match_runtime_matrix_shape",
                    "before_after_qwen3_q8_warm_session_receipts_with_hook_selection_identity_missing"
                ])
            } else {
                serde_json::json!([
                    "packed_q8_sidecar_compute_kernel_not_enabled",
                    "before_after_qwen3_q8_warm_session_receipts_with_hook_selection_identity_missing"
                ])
            },
            "next_safe_step": "run before/after Qwen3 Q8_0 warm-session receipts that record identical model SHA, tokenizer source/strictness, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook selection identity, and fallback_used=false before enabling packed sidecar compute"
        })
    }

    /// Convert ConcreteTensor to Candle tensor
    fn to_candle_tensor(&self, tensor: &ConcreteTensor) -> Result<CandleTensor> {
        match tensor {
            ConcreteTensor::BitNet(t) => t.to_candle(),
            ConcreteTensor::Mock(mock) => {
                // Create a dummy tensor for mock
                let shape = mock.shape();
                let device =
                    self.device.to_candle().map_err(|e| BitNetError::Validation(e.to_string()))?;
                Ok(CandleTensor::zeros(shape, DType::F32, &device)?)
            }
        }
    }

    /// Convert Candle tensor to ConcreteTensor
    fn candle_to_concrete(&self, tensor: CandleTensor) -> ConcreteTensor {
        ConcreteTensor::BitNet(BitNetTensor::new(tensor))
    }
}

fn dense_q8_runtime_hooks_from_sidecars(
    registry: &DenseGgufQ8SidecarRegistry,
) -> DenseLinearRuntimeHookRegistry {
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    let runtime_compute_tensor = dense_q8_runtime_compute_tensor_from_env();
    for descriptor in &registry.descriptors {
        let Some(canonical_name) = dense_q8_transformer_hook_name(descriptor) else {
            continue;
        };
        let payload_order_matches_runtime_shape = descriptor.payload_order_matches_runtime_shape();
        let source_order_contract = descriptor.source_order_kernel_contract();
        let source_order_q8_matvec_candidate = matches!(
            source_order_contract.contract_status,
            DenseQ8SourceOrderKernelContractStatus::RuntimeDisabledSourceOrderMatvecCandidate
        ) && descriptor.packed_q8_bytes.is_some();
        let runtime_tensor_matches = runtime_compute_tensor
            .as_deref()
            .is_some_and(|tensor| tensor == descriptor.tensor_name);
        let runtime_compute_enabled = runtime_tensor_matches
            && descriptor.packed_q8_bytes.is_some()
            && (payload_order_matches_runtime_shape || source_order_q8_matvec_candidate);
        hooks.insert(
            canonical_name,
            DenseLinearRuntimeHookDescriptor {
                tensor_name: descriptor.tensor_name.clone(),
                role: format!("{:?}", descriptor.role),
                sidecar_payload_sha256: Some(descriptor.packed_q8_bytes_sha256.clone()),
                packed_q8_payload: dense_q8_packed_payload_from_sidecar(descriptor),
                payload_order_matches_runtime_shape,
                source_order_q8_matvec_candidate,
                source_order_input_dim: source_order_contract.source_input_dim,
                source_order_output_dim: source_order_contract.source_output_dim,
                runtime_compute_enabled,
                receipt_bound_no_bias_selector: None,
            },
        );
    }
    hooks
}

fn dense_q8_packed_payload_from_sidecar(
    descriptor: &DenseGgufQ8SidecarDescriptor,
) -> Option<DenseLinearPackedQ8Payload> {
    let [matrix_rows, matrix_cols]: [usize; 2] =
        descriptor.runtime_candle_shape.as_slice().try_into().ok()?;
    Some(DenseLinearPackedQ8Payload {
        tensor_name: descriptor.tensor_name.clone(),
        packed_q8_bytes: descriptor.packed_q8_bytes.clone()?,
        q8_block_size: descriptor.q8_block_size,
        q8_block_count: descriptor.q8_block_count,
        matrix_rows,
        matrix_cols,
    })
}

fn dense_q8_transformer_hook_name(descriptor: &DenseGgufQ8SidecarDescriptor) -> Option<String> {
    bitnet_weight_name_core::normalize_vendor_key(&descriptor.tensor_name).or_else(|| {
        match descriptor.tensor_name.as_str() {
            "output.weight" | "lm_head.weight" => Some("lm_head.weight".to_string()),
            "token_embd.weight" | "model.embed_tokens.weight" => {
                Some("embed_tokens.weight".to_string())
            }
            _ => None,
        }
    })
}

#[cfg(test)]
mod dense_q8_runtime_hook_tests {
    use super::*;
    use crate::dense_gguf_q8_sidecar::{DENSE_Q8_RUNTIME_ENABLE_ENV, DENSE_Q8_RUNTIME_TENSOR_ENV};
    use crate::formats::gguf::{GgufTensorType, TensorInfo};
    use serial_test::serial;

    fn q8_tensor_info(name: &str, shape: Vec<usize>) -> TensorInfo {
        let value_count = shape.iter().copied().product::<usize>();
        let block_count = value_count.div_ceil(GgufTensorType::Q8_0.block_size());
        let size = block_count * GgufTensorType::Q8_0.element_size();
        TensorInfo {
            name: name.to_string(),
            shape,
            tensor_type: GgufTensorType::Q8_0,
            offset: 0,
            size: size as u64,
        }
    }

    #[test]
    #[serial]
    fn dense_gguf_q8_sidecar_hooks_map_vendor_tensor_names() -> Result<()> {
        unsafe {
            std::env::remove_var(DENSE_Q8_RUNTIME_ENABLE_ENV);
            std::env::remove_var(DENSE_Q8_RUNTIME_TENSOR_ENV);
        }
        let mut registry = DenseGgufQ8SidecarRegistry::default();
        let tensor_info = q8_tensor_info("blk.0.attn_q.weight", vec![64, 32]);
        let data = vec![0_u8; tensor_info.size as usize];
        registry.try_push_tensor(&tensor_info, &data)?;

        let hooks = dense_q8_runtime_hooks_from_sidecars(&registry);
        let Some(hook) = hooks.get("layers.0.attention.q_proj.weight") else {
            return Err(BitNetError::Validation(
                "expected canonical attention q_proj hook".to_string(),
            ));
        };

        assert_eq!(hook.tensor_name, "blk.0.attn_q.weight");
        assert_eq!(hook.role, "AttentionQ");
        assert_eq!(hook.runtime_compute_enabled, false);
        assert!(hook.sidecar_payload_sha256.is_some());
        Ok(())
    }

    #[test]
    #[serial]
    fn dense_gguf_q8_sidecar_hook_can_carry_one_real_payload_candidate() -> Result<()> {
        unsafe {
            std::env::remove_var(DENSE_Q8_RUNTIME_ENABLE_ENV);
            std::env::remove_var(DENSE_Q8_RUNTIME_TENSOR_ENV);
        }
        let mut registry = DenseGgufQ8SidecarRegistry::default();
        let tensor_info = q8_tensor_info("blk.0.attn_q.weight", vec![64, 32]);
        let data = vec![7_u8; tensor_info.size as usize];
        registry.try_push_tensor_with_payload_candidate(
            &tensor_info,
            &data,
            Some("blk.0.attn_q.weight"),
        )?;

        let hooks = dense_q8_runtime_hooks_from_sidecars(&registry);
        let Some(hook) = hooks.get("layers.0.attention.q_proj.weight") else {
            return Err(BitNetError::Validation(
                "expected canonical attention q_proj hook".to_string(),
            ));
        };
        let Some(payload) = hook.packed_q8_payload.as_ref() else {
            return Err(BitNetError::Validation("expected packed payload".to_string()));
        };

        assert_eq!(payload.tensor_name, "blk.0.attn_q.weight");
        assert_eq!(payload.payload_len(), tensor_info.size as usize);
        assert_eq!(payload.q8_block_size, 32);
        assert_eq!(payload.q8_block_count, 64);
        assert_eq!(payload.matrix_rows, 32);
        assert_eq!(payload.matrix_cols, 64);
        assert!(payload.shape_matches_matvec_contract());
        assert!(payload.payload_len_matches_contract());
        assert!(!hook.runtime_compute_enabled);
        Ok(())
    }

    #[test]
    #[serial]
    fn dense_gguf_q8_sidecar_runtime_compute_allows_source_order_qproj_candidate() -> Result<()> {
        unsafe {
            std::env::set_var(DENSE_Q8_RUNTIME_ENABLE_ENV, "1");
            std::env::set_var(DENSE_Q8_RUNTIME_TENSOR_ENV, "blk.0.attn_q.weight");
        }
        let result = (|| -> Result<()> {
            let mut registry = DenseGgufQ8SidecarRegistry::default();
            let tensor_info = q8_tensor_info("blk.0.attn_q.weight", vec![64, 32]);
            let data = vec![7_u8; tensor_info.size as usize];
            registry.try_push_tensor_with_payload_candidate(
                &tensor_info,
                &data,
                Some("blk.0.attn_q.weight"),
            )?;

            let hooks = dense_q8_runtime_hooks_from_sidecars(&registry);
            let Some(hook) = hooks.get("layers.0.attention.q_proj.weight") else {
                return Err(BitNetError::Validation(
                    "expected canonical attention q_proj hook".to_string(),
                ));
            };

            assert!(hook.runtime_compute_enabled);
            assert!(hook.packed_q8_payload.is_some());
            assert!(!hook.payload_order_matches_runtime_shape);
            assert!(hook.source_order_q8_matvec_candidate);
            assert_eq!(hook.source_order_input_dim, Some(64));
            assert_eq!(hook.source_order_output_dim, Some(32));
            Ok(())
        })();
        unsafe {
            std::env::remove_var(DENSE_Q8_RUNTIME_ENABLE_ENV);
            std::env::remove_var(DENSE_Q8_RUNTIME_TENSOR_ENV);
        }
        result
    }
}

impl Model for BitNetModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor> {
        // Fail fast if transformer not initialized - prevents silent zero-logit failures
        let transformer = self.transformer.as_ref().ok_or_else(|| {
            BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: "BitNetModel::transformer not initialized (GGUF load failed or build_transformer returned error)".to_string()
            })
        })?;

        // Get or create KV cache
        let kv_cache = cache.downcast_mut::<KVCache>();

        // Convert input to Candle tensor
        let input_tensor = self.to_candle_tensor(input)?;

        // Run transformer forward pass (passes ownership to avoid clone on hot path)
        let output = transformer.forward(input_tensor, kv_cache)?;

        // Convert back to ConcreteTensor
        Ok(self.candle_to_concrete(output))
    }

    fn forward_with_no_bias_callsite_descriptor(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
        descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    ) -> Result<ConcreteTensor> {
        let transformer = self.transformer.as_ref().ok_or_else(|| {
            BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: "BitNetModel::transformer not initialized (GGUF load failed or build_transformer returned error)".to_string()
            })
        })?;

        let kv_cache = cache.downcast_mut::<KVCache>();
        let input_tensor = self.to_candle_tensor(input)?;
        let output = transformer.forward_with_no_bias_callsite_descriptor(
            input_tensor,
            kv_cache,
            descriptor,
        )?;
        Ok(self.candle_to_concrete(output))
    }

    fn forward_with_source_context(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
    ) -> Result<ModelForwardDiagnosticOutput> {
        let transformer = self.transformer.as_ref().ok_or_else(|| {
            BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: "BitNetModel::transformer not initialized (GGUF load failed or build_transformer returned error)".to_string()
            })
        })?;

        let kv_cache = cache.downcast_mut::<KVCache>();
        let input_tensor = self.to_candle_tensor(input)?;
        let mut workspace = bitnet_transformer::TransformerForwardWorkspace::new();
        let output = transformer.forward_with_workspace(input_tensor, kv_cache, &mut workspace)?;
        let final_block_source =
            workspace.final_block_source_tensors().map(|source| ModelFinalBlockSourceContext {
                layer_idx: source.layer_idx,
                block_input: self.candle_to_concrete(source.block_input.clone()),
                attention_output: self.candle_to_concrete(source.attention_output.clone()),
                post_attention_residual: self
                    .candle_to_concrete(source.post_attention_residual.clone()),
                feed_forward_output: self.candle_to_concrete(source.feed_forward_output.clone()),
                block_output: self.candle_to_concrete(source.block_output.clone()),
            });
        let penultimate_block_source = workspace.penultimate_block_source_tensors().map(|source| {
            ModelFinalBlockSourceContext {
                layer_idx: source.layer_idx,
                block_input: self.candle_to_concrete(source.block_input.clone()),
                attention_output: self.candle_to_concrete(source.attention_output.clone()),
                post_attention_residual: self
                    .candle_to_concrete(source.post_attention_residual.clone()),
                feed_forward_output: self.candle_to_concrete(source.feed_forward_output.clone()),
                block_output: self.candle_to_concrete(source.block_output.clone()),
            }
        });
        let antepenultimate_block_source =
            workspace.antepenultimate_block_source_tensors().map(|source| {
                ModelFinalBlockSourceContext {
                    layer_idx: source.layer_idx,
                    block_input: self.candle_to_concrete(source.block_input.clone()),
                    attention_output: self.candle_to_concrete(source.attention_output.clone()),
                    post_attention_residual: self
                        .candle_to_concrete(source.post_attention_residual.clone()),
                    feed_forward_output: self
                        .candle_to_concrete(source.feed_forward_output.clone()),
                    block_output: self.candle_to_concrete(source.block_output.clone()),
                }
            });
        let pre_antepenultimate_block_source = workspace
            .pre_antepenultimate_block_source_tensors()
            .map(|source| ModelFinalBlockSourceContext {
                layer_idx: source.layer_idx,
                block_input: self.candle_to_concrete(source.block_input.clone()),
                attention_output: self.candle_to_concrete(source.attention_output.clone()),
                post_attention_residual: self
                    .candle_to_concrete(source.post_attention_residual.clone()),
                feed_forward_output: self.candle_to_concrete(source.feed_forward_output.clone()),
                block_output: self.candle_to_concrete(source.block_output.clone()),
            });
        let earlier_block_source =
            workspace.earlier_block_source_tensors().map(|source| ModelFinalBlockSourceContext {
                layer_idx: source.layer_idx,
                block_input: self.candle_to_concrete(source.block_input.clone()),
                attention_output: self.candle_to_concrete(source.attention_output.clone()),
                post_attention_residual: self
                    .candle_to_concrete(source.post_attention_residual.clone()),
                feed_forward_output: self.candle_to_concrete(source.feed_forward_output.clone()),
                block_output: self.candle_to_concrete(source.block_output.clone()),
            });
        let block_sources = workspace
            .block_source_tensors()
            .iter()
            .map(|source| ModelFinalBlockSourceContext {
                layer_idx: source.layer_idx,
                block_input: self.candle_to_concrete(source.block_input.clone()),
                attention_output: self.candle_to_concrete(source.attention_output.clone()),
                post_attention_residual: self
                    .candle_to_concrete(source.post_attention_residual.clone()),
                feed_forward_output: self.candle_to_concrete(source.feed_forward_output.clone()),
                block_output: self.candle_to_concrete(source.block_output.clone()),
            })
            .collect();
        let attention_output_sources = workspace
            .attention_output_source_tensors()
            .iter()
            .map(|source| ModelAttentionOutputSourceContext {
                layer_idx: source.layer_idx,
                attention_input: self.candle_to_concrete(source.attention_input.clone()),
                q_projection: self.candle_to_concrete(source.q_projection.clone()),
                k_projection: self.candle_to_concrete(source.k_projection.clone()),
                v_projection: self.candle_to_concrete(source.v_projection.clone()),
                q_heads: self.candle_to_concrete(source.q_heads.clone()),
                k_heads: self.candle_to_concrete(source.k_heads.clone()),
                v_heads: self.candle_to_concrete(source.v_heads.clone()),
                q_norm: self.candle_to_concrete(source.q_norm.clone()),
                k_norm: self.candle_to_concrete(source.k_norm.clone()),
                q_rope: self.candle_to_concrete(source.q_rope.clone()),
                k_rope: self.candle_to_concrete(source.k_rope.clone()),
                k_context: self.candle_to_concrete(source.k_context.clone()),
                v_context: self.candle_to_concrete(source.v_context.clone()),
                expanded_k: self.candle_to_concrete(source.expanded_k.clone()),
                expanded_v: self.candle_to_concrete(source.expanded_v.clone()),
                scores: self.candle_to_concrete(source.scores.clone()),
                probabilities: self.candle_to_concrete(source.probabilities.clone()),
                value_mix_output_heads: self
                    .candle_to_concrete(source.value_mix_output_heads.clone()),
                output_projection_input: self
                    .candle_to_concrete(source.output_projection_input.clone()),
                sub_layernorm_output: source
                    .sub_layernorm_output
                    .clone()
                    .map(|tensor| self.candle_to_concrete(tensor)),
                attention_output: self.candle_to_concrete(source.attention_output.clone()),
            })
            .collect();
        let qkv_projection_sources = workspace
            .qkv_projection_source_tensors()
            .iter()
            .map(|source| ModelQkvProjectionSourceContext {
                layer_idx: source.layer_idx,
                projection: source.projection.clone(),
                tensor_name: source.tensor_name.clone(),
                qk256_key: source.qk256_key.clone(),
                qk256_raw_tensor_present: source.qk256_raw_tensor_present,
                input: self.candle_to_concrete(source.input.clone()),
                output: self.candle_to_concrete(source.output.clone()),
                dispatch_delta: ModelQk256DispatchDeltaContext {
                    bitnet_linear_layers_total: source.dispatch_delta.bitnet_linear_layers_total,
                    bitnet_linear_layers_on_cuda: source
                        .dispatch_delta
                        .bitnet_linear_layers_on_cuda,
                    bitnet_linear_layers_on_a770_opencl: source
                        .dispatch_delta
                        .bitnet_linear_layers_on_a770_opencl,
                    bitnet_linear_layers_cpu_fallback: source
                        .dispatch_delta
                        .bitnet_linear_layers_cpu_fallback,
                    unsupported_ops: source.dispatch_delta.unsupported_ops.clone(),
                    execution_claim: source.dispatch_delta.execution_claim.clone(),
                },
                cpu_hot_path_delta: ModelQk256CpuHotPathDeltaContext {
                    qk256_f32_scalar_gemv_invocations: source
                        .cpu_hot_path_delta
                        .qk256_f32_scalar_gemv_invocations,
                    qk256_f32_avx2_gemv_invocations: source
                        .cpu_hot_path_delta
                        .qk256_f32_avx2_gemv_invocations,
                    qk256_i8s_scaled_scalar_invocations: source
                        .cpu_hot_path_delta
                        .qk256_i8s_scaled_scalar_invocations,
                    qk256_i8s_scaled_avx2_invocations: source
                        .cpu_hot_path_delta
                        .qk256_i8s_scaled_avx2_invocations,
                    qk256_flat_bytes_extracted_count: source
                        .cpu_hot_path_delta
                        .qk256_flat_bytes_extracted_count,
                    input_rows_materialized_count: source
                        .cpu_hot_path_delta
                        .input_rows_materialized_count,
                    output_rows_allocated_count: source
                        .cpu_hot_path_delta
                        .output_rows_allocated_count,
                    requested_kernel: source.cpu_hot_path_delta.requested_kernel.clone(),
                    selected_kernel: source.cpu_hot_path_delta.selected_kernel.clone(),
                    qk256_execution_path: source.cpu_hot_path_delta.qk256_execution_path.clone(),
                },
                a770_opencl_runtime_delta: ModelA770OpenClRuntimeDeltaContext {
                    host_to_device_bytes: source.a770_opencl_runtime_delta.host_to_device_bytes,
                    device_to_host_bytes: source.a770_opencl_runtime_delta.device_to_host_bytes,
                    kernel_invocations: source.a770_opencl_runtime_delta.kernel_invocations,
                },
                dispatch_replay: source.dispatch_replay.as_ref().map(|replay| {
                    ModelQkvProjectionDispatchReplayContext {
                        input_rows: replay.input_rows,
                        output_rows: replay.output_rows,
                        cols: replay.cols,
                        row_stride_bytes: replay.row_stride_bytes,
                        inline_scale: replay.inline_scale,
                        cpu_output: self.candle_to_concrete(replay.cpu_output.clone()),
                        opencl_policy_output: self
                            .candle_to_concrete(replay.opencl_policy_output.clone()),
                        a770_output: replay
                            .a770_output
                            .clone()
                            .map(|tensor| self.candle_to_concrete(tensor)),
                        device_expression_trace: replay.device_expression_trace.as_ref().map(
                            |trace| ModelQk256DeviceExpressionTraceContext {
                                input_row_index: trace.input_row_index,
                                sample_limit: trace.sample_limit,
                                sample_count: trace.sample_count,
                                samples: trace
                                    .samples
                                    .iter()
                                    .map(|sample| ModelQk256DeviceExpressionSampleContext {
                                        output_index: sample.output_index,
                                        int_dot: sample.int_dot,
                                        activation_sum: sample.activation_sum,
                                        adjusted_dot: sample.adjusted_dot,
                                        activation_scale: sample.activation_scale,
                                        activation_scale_bits: sample.activation_scale_bits,
                                        weight_scale: sample.weight_scale,
                                        weight_scale_bits: sample.weight_scale_bits,
                                        div_then_mul: sample.div_then_mul,
                                        mul_then_div: sample.mul_then_div,
                                        reciprocal_then_mul: sample.reciprocal_then_mul,
                                        f64_div_then_mul_cast: sample.f64_div_then_mul_cast,
                                    })
                                    .collect(),
                            },
                        ),
                        device_intermediate_trace: replay.device_intermediate_trace.as_ref().map(
                            |trace| ModelQk256DeviceIntermediateTraceContext {
                                compiled_opencl: trace.compiled_opencl,
                                attempted: trace.attempted,
                                success: trace.success,
                                error: trace.error.clone(),
                                input_row_index: trace.input_row_index,
                                sample_limit: trace.sample_limit,
                                sample_count: trace.sample_count,
                                platform_index: trace.platform_index,
                                device_index: trace.device_index,
                                platform_name: trace.platform_name.clone(),
                                runtime_device: trace.runtime_device.clone(),
                                vendor: trace.vendor.clone(),
                                driver_version: trace.driver_version.clone(),
                                host_to_device_bytes: trace.host_to_device_bytes,
                                device_to_host_bytes: trace.device_to_host_bytes,
                                kernel_invocations: trace.kernel_invocations,
                                samples: trace
                                    .samples
                                    .iter()
                                    .map(|sample| ModelQk256DeviceIntermediateSampleContext {
                                        output_index: sample.output_index,
                                        int_dot: sample.int_dot,
                                        activation_sum: sample.activation_sum,
                                        adjusted_dot: sample.adjusted_dot,
                                        activation_scale_bits: sample.activation_scale_bits,
                                        weight_scale_bits: sample.weight_scale_bits,
                                        adjusted_f32_bits: sample.adjusted_f32_bits,
                                        output_bits: sample.output_bits,
                                        output: sample.output,
                                        div_then_mul_bits: sample.div_then_mul_bits,
                                        div_then_mul: sample.div_then_mul,
                                        mul_then_div_bits: sample.mul_then_div_bits,
                                        mul_then_div: sample.mul_then_div,
                                        reciprocal_then_mul_bits: sample.reciprocal_then_mul_bits,
                                        reciprocal_then_mul: sample.reciprocal_then_mul,
                                        volatile_div_then_mul_bits: sample
                                            .volatile_div_then_mul_bits,
                                        volatile_div_then_mul: sample.volatile_div_then_mul,
                                    })
                                    .collect(),
                            },
                        ),
                        focused_operands: replay.focused_operands.as_ref().map(|operands| {
                            ModelQk256FocusedRawOperandsContext {
                                input_row_index: operands.input_row_index,
                                output_index: operands.output_index,
                                cols: operands.cols,
                                row_stride_bytes: operands.row_stride_bytes,
                                packed_qk256_scope: operands.packed_qk256_scope.clone(),
                                activation_sum: operands.activation_sum,
                                activation_scale_bits: operands.activation_scale_bits,
                                weight_scale_bits: operands.weight_scale_bits,
                                activations_i8: operands.activations_i8.clone(),
                                packed_qk256: operands.packed_qk256.clone(),
                            }
                        }),
                        full_projection_operands: replay.full_projection_operands.as_ref().map(
                            |operands| ModelQk256FullProjectionRawOperandsContext {
                                input_row_index: operands.input_row_index,
                                rows: operands.rows,
                                cols: operands.cols,
                                row_stride_bytes: operands.row_stride_bytes,
                                packed_qk256_scope: operands.packed_qk256_scope.clone(),
                                activation_sum: operands.activation_sum,
                                activation_scale_bits: operands.activation_scale_bits,
                                weight_scale_bits: operands.weight_scale_bits,
                                activations_i8: operands.activations_i8.clone(),
                                packed_qk256: operands.packed_qk256.clone(),
                            },
                        ),
                        cpu: ModelQkvProjectionDispatchReplayCpuContext {
                            scalar_invocations: replay.cpu.scalar_invocations,
                            execution_path: replay.cpu.execution_path.clone(),
                        },
                        a770: ModelQkvProjectionDispatchReplayA770Context {
                            compiled_opencl: replay.a770.compiled_opencl,
                            attempted: replay.a770.attempted,
                            success: replay.a770.success,
                            host_to_device_bytes: replay.a770.host_to_device_bytes,
                            device_to_host_bytes: replay.a770.device_to_host_bytes,
                            kernel_invocations: replay.a770.kernel_invocations,
                            last_device: replay.a770.last_device.as_ref().map(|device| {
                                ModelA770OpenClRuntimeDeviceContext {
                                    platform_index: device.platform_index,
                                    device_index: device.device_index,
                                    platform_name: device.platform_name.clone(),
                                    runtime_device: device.runtime_device.clone(),
                                    vendor: device.vendor.clone(),
                                    driver_version: device.driver_version.clone(),
                                }
                            }),
                            error: replay.a770.error.clone(),
                            execution_path: replay.a770.execution_path.clone(),
                        },
                    }
                }),
                dispatch_replay_error: source.dispatch_replay_error.clone(),
            })
            .collect();
        let source_context =
            workspace.model_forward_source_tensors().map(|source| ModelForwardSourceContext {
                prior_layer_output: self.candle_to_concrete(source.prior_layer_output.clone()),
                final_norm_output: self.candle_to_concrete(source.final_norm_output.clone()),
                final_block_source,
                penultimate_block_source,
                antepenultimate_block_source,
                pre_antepenultimate_block_source,
                earlier_block_source,
                block_sources,
                attention_output_sources,
                qkv_projection_sources,
            });

        Ok(ModelForwardDiagnosticOutput { output: self.candle_to_concrete(output), source_context })
    }

    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor> {
        // Fail fast if transformer not initialized
        let transformer = self.transformer.as_ref().ok_or_else(|| {
            BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: "BitNetModel::transformer not initialized (cannot embed tokens)"
                    .to_string(),
            })
        })?;

        let embedded = transformer.embed(tokens)?;
        Ok(self.candle_to_concrete(embedded))
    }

    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor> {
        // Fail fast if transformer not initialized
        let transformer = self.transformer.as_ref().ok_or_else(|| {
            BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: "BitNetModel::transformer not initialized (cannot compute logits)"
                    .to_string(),
            })
        })?;

        let hidden_tensor = self.to_candle_tensor(hidden)?;
        let logits = transformer.logits(&hidden_tensor)?;
        Ok(self.candle_to_concrete(logits))
    }

    fn dense_q8_hook_selection_receipt(&self) -> serde_json::Value {
        BitNetModel::dense_q8_hook_selection_receipt(self)
    }
}
