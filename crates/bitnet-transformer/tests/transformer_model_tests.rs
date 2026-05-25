//! Integration tests for `TransformerModel` — covers embed, logits, and
//! forward_full on zero-initialized weights (no real GGUF required).
//!
//! Uses `VarBuilder::zeros` which auto-fills any requested tensor key with zeros,
//! eliminating the need to manually enumerate all weight keys.
//!
//! Verifies:
//!   - Shape invariants (embed output, logit output, forward_full output)
//!   - Finite-value guarantees (no NaN / Inf in output)
//!   - Determinism (same input → same output)
//!   - Model construction with different config shapes
//!   - Validation errors for incompatible config values
#![cfg(feature = "cpu")]

use bitnet_common::config::{BitNetConfig, ModelConfig};
use bitnet_transformer::{
    DenseLinearOutputStorageApiBoundary, DenseLinearPackedQ8Payload,
    DenseLinearRuntimeHookBoundary, DenseLinearRuntimeHookDescriptor,
    DenseLinearRuntimeHookRegistry, DenseQ8SidecarQNormInputReceiptIdentity, KVCache,
    LayerOutputStorageApiBoundary, NormOutputStorageApiBoundary, TransformerForwardWorkspace,
    TransformerModel, compare_dense_q8_sidecar_q_norm_input_receipts,
    dense_q8_sidecar_fused_consumer_boundary,
    dense_q8_sidecar_fused_q_projection_consumer_contract,
    dense_q8_sidecar_q_norm_input_proof_gate,
    dense_q8_sidecar_q_norm_input_receipt_comparator_gate,
    dense_q8_sidecar_q_norm_input_runtime_hook_gate,
    dense_q8_sidecar_q_norm_input_tensor_identity_surface,
    dense_q8_sidecar_q_norm_materialization_boundary_gate,
    dense_q8_sidecar_typed_attention_head_consumer_gate,
    dense_q8_sidecar_typed_attention_head_view_gate,
    dense_q8_sidecar_typed_fused_q_projection_implementation_gate,
    dense_q8_sidecar_typed_q_norm_rope_consumer_gate,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Minimal config for a 1-layer, small-vocab model — fast to construct.
fn tiny_config(hidden: usize, vocab: usize, heads: usize) -> BitNetConfig {
    BitNetConfig {
        model: ModelConfig {
            hidden_size: hidden,
            vocab_size: vocab,
            num_heads: heads,
            num_key_value_heads: heads,
            num_layers: 1,
            intermediate_size: hidden * 4,
            max_position_embeddings: 64,
            rms_norm_eps: Some(1e-5),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Build a `TransformerModel` with all-zero weights via `VarBuilder::zeros`.
fn make_model(hidden: usize, vocab: usize, heads: usize) -> anyhow::Result<TransformerModel> {
    let device = Device::Cpu;
    let cfg = tiny_config(hidden, vocab, heads);
    let vb = VarBuilder::zeros(DType::F32, &device);
    Ok(TransformerModel::new(cfg, vb)?)
}

// ── embed tests ───────────────────────────────────────────────────────────────

/// The `embed` method must return shape `[1, seq_len, hidden]`.
#[test]
fn test_embed_shape() -> anyhow::Result<()> {
    let model = make_model(64, 128, 4)?;
    let tokens: &[u32] = &[1, 2, 3, 4, 5];
    let out = model.embed(tokens)?;
    assert_eq!(out.dims(), &[1, 5, 64], "embed shape should be [1, seq, hidden]");
    Ok(())
}

/// Embedding output must be finite.
#[test]
fn test_embed_finite() -> anyhow::Result<()> {
    let model = make_model(64, 128, 4)?;
    let tokens: &[u32] = &[0, 1, 2];
    let out = model.embed(tokens)?;
    let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
    assert!(vals.iter().all(|v| v.is_finite()), "embed output must be finite");
    Ok(())
}

/// `embed` is deterministic — same tokens → same tensor every call.
#[test]
fn test_embed_determinism() -> anyhow::Result<()> {
    let model = make_model(64, 128, 4)?;
    let tokens: &[u32] = &[10, 20, 30];
    let a: Vec<f32> = model.embed(tokens)?.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = model.embed(tokens)?.flatten_all()?.to_vec1()?;
    assert_eq!(a, b, "embed must be deterministic");
    Ok(())
}

// ── logits tests ──────────────────────────────────────────────────────────────

/// `logits` should accept a 3D hidden state and return `[B, seq, vocab]`.
#[test]
fn test_logits_shape_3d() -> anyhow::Result<()> {
    let hidden = 64;
    let vocab = 128;
    let model = make_model(hidden, vocab, 4)?;

    let device = Device::Cpu;
    let hidden_state = Tensor::zeros((1usize, 3usize, hidden), DType::F32, &device)?;
    let out = model.logits(&hidden_state)?;
    assert_eq!(out.dims(), &[1, 3, vocab], "logits shape should be [1, seq, vocab]");
    Ok(())
}

/// `logits` should accept a 2D hidden state (last-token only) and return `[B, vocab]`.
#[test]
fn test_logits_shape_2d() -> anyhow::Result<()> {
    let hidden = 64;
    let vocab = 128;
    let model = make_model(hidden, vocab, 4)?;

    let device = Device::Cpu;
    let hidden_state = Tensor::zeros((1usize, hidden), DType::F32, &device)?;
    let out = model.logits(&hidden_state)?;
    // logits() returns [B, V] for 2D input (incremental decode path)
    assert_eq!(out.dims()[out.dims().len() - 1], vocab, "last dim should be vocab");
    Ok(())
}

/// `logits` output must be finite.
#[test]
fn test_logits_finite() -> anyhow::Result<()> {
    let hidden = 64;
    let vocab = 128;
    let model = make_model(hidden, vocab, 4)?;

    let device = Device::Cpu;
    let hidden_state = Tensor::zeros((1usize, 2usize, hidden), DType::F32, &device)?;
    let out = model.logits(&hidden_state)?;
    let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
    assert!(vals.iter().all(|v| v.is_finite()), "logits must be finite");
    Ok(())
}

// ── forward_full tests ────────────────────────────────────────────────────────

/// `forward_full` must return shape `[1, seq, vocab]` for a 3-token sequence.
#[test]
fn test_forward_full_shape() -> anyhow::Result<()> {
    let hidden = 64;
    let vocab = 128;
    let model = make_model(hidden, vocab, 4)?;

    let device = Device::Cpu;
    let token_ids = Tensor::from_slice(&[1u32, 2, 3], (1usize, 3usize), &device)?;
    let out = model.forward_full(&token_ids)?;
    assert_eq!(out.dims(), &[1, 3, vocab], "forward_full shape should be [1, seq, vocab]");
    Ok(())
}

/// `forward_full` must produce finite values.
#[test]
fn test_forward_full_finite() -> anyhow::Result<()> {
    let model = make_model(64, 128, 4)?;
    let device = Device::Cpu;
    let token_ids = Tensor::from_slice(&[0u32, 1], (1usize, 2usize), &device)?;
    let out = model.forward_full(&token_ids)?;
    let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
    assert!(vals.iter().all(|v| v.is_finite()), "forward_full must not produce NaN/Inf");
    Ok(())
}

/// `forward_full` must be deterministic — same input → same output.
#[test]
fn test_forward_full_determinism() -> anyhow::Result<()> {
    let model = make_model(64, 128, 4)?;
    let device = Device::Cpu;
    let token_ids = Tensor::from_slice(&[5u32, 10, 15], (1usize, 3usize), &device)?;

    let a: Vec<f32> = model.forward_full(&token_ids)?.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = model.forward_full(&token_ids)?.flatten_all()?.to_vec1()?;
    assert_eq!(a, b, "forward_full must be deterministic");
    Ok(())
}

// ── incremental (forward) tests ───────────────────────────────────────────────

/// Incremental `forward` (single token at a time with KV cache) must return
/// rank-2 `[B, H]` per step and produce finite logits.
#[test]
fn test_incremental_forward_shape_and_finite() -> anyhow::Result<()> {
    let hidden = 64;
    let vocab = 128;
    let model = make_model(hidden, vocab, 4)?;
    let device = Device::Cpu;

    let tokens: &[u32] = &[1, 2, 3];
    let mut kv = KVCache::new(&model.config, 1, &device)?;

    for &t in tokens {
        let h = model.embed(std::slice::from_ref(&t))?;
        let out = model.forward(h, Some(&mut kv))?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!(vals.iter().all(|v| v.is_finite()), "incremental forward must be finite");
    }
    Ok(())
}

#[test]
fn test_incremental_forward_workspace_matches_existing_path() -> anyhow::Result<()> {
    let hidden = 64;
    let model = make_model(hidden, 128, 4)?;
    let device = Device::Cpu;
    let token = 7u32;
    let h = model.embed(std::slice::from_ref(&token))?;

    let mut existing_kv = KVCache::new(&model.config, 1, &device)?;
    let mut workspace_kv = KVCache::new(&model.config, 1, &device)?;
    let mut workspace = TransformerForwardWorkspace::new();

    let existing: Vec<f32> =
        model.forward(h.clone(), Some(&mut existing_kv))?.flatten_all()?.to_vec1()?;
    let with_workspace =
        model.forward_with_workspace(h, Some(&mut workspace_kv), &mut workspace)?;
    let with_workspace_values: Vec<f32> = with_workspace.flatten_all()?.to_vec1()?;

    assert_eq!(existing, with_workspace_values, "workspace API must preserve forward output");
    assert_eq!(workspace.model_forward_calls(), 1);
    assert_eq!(workspace.block_forward_calls(), model.config.model.num_layers);
    assert_eq!(workspace.feed_forward_calls(), model.config.model.num_layers);
    assert_eq!(workspace.last_output_shape(), with_workspace.dims());
    assert_eq!(workspace.last_output_shape(), &[1, 1, hidden]);
    assert_eq!(
        workspace.reuse_status(),
        "dense_linear_output_storage_blocked_by_candle_tensor_ops"
    );
    assert_eq!(workspace.workspace_owned_output_count(), model.config.model.num_layers);
    assert_eq!(workspace.model_workspace_owned_output_count(), 1);
    assert_eq!(workspace.model_output_storage_attempts(), 1);
    assert_eq!(workspace.down_proj_output_storage_attempts(), model.config.model.num_layers);
    assert_eq!(workspace.layer_output_storage_attempts(), model.config.model.num_layers);
    assert_eq!(workspace.final_norm_output_storage_attempts(), 1);
    let Some(source_tensors) = workspace.model_forward_source_tensors() else {
        anyhow::bail!("workspace should retain model forward source tensors");
    };
    assert_eq!(source_tensors.prior_layer_output.dims(), &[1, 1, hidden]);
    assert_eq!(source_tensors.final_norm_output.dims(), &[1, 1, hidden]);
    assert_eq!(
        source_tensors.final_norm_output.flatten_all()?.to_vec1::<f32>()?,
        with_workspace_values,
        "final norm output source tensor should match model.forward output"
    );
    let Some(model_surface) = workspace.model_output_surface() else {
        anyhow::bail!("workspace should classify the model forward output surface");
    };
    assert_eq!(model_surface.name, "model.forward.output");
    assert_eq!(model_surface.storage_owner, "TransformerForwardWorkspace");
    assert_eq!(
        model_surface.status,
        "model_forward_output_storage_api_surface_present_reuse_blocked_by_candle_tensor_ops"
    );
    assert_eq!(model_surface.last_shape, vec![1, 1, hidden]);
    assert!(!model_surface.weight_accessible);
    assert!(!model_surface.bias_accessible);
    assert!(!model_surface.can_fill_caller_output_storage);
    let Some(surface) = workspace.first_output_surface() else {
        anyhow::bail!("workspace should classify one output surface");
    };
    assert_eq!(surface.name, "feed_forward.down_proj.output");
    assert_eq!(surface.storage_owner, "TransformerForwardWorkspace");
    assert_eq!(surface.status, "dense_linear_output_storage_blocked_by_candle_tensor_ops");
    assert_eq!(surface.last_shape, vec![1, 1, hidden]);
    assert_eq!(surface.linear_weight_shape, vec![hidden, hidden * 4]);
    assert_eq!(surface.linear_bias_shape, Some(vec![hidden]));
    assert!(surface.weight_accessible);
    assert!(surface.bias_accessible);
    assert!(!surface.can_fill_caller_output_storage);
    let Some(layer_surface) = workspace.layer_output_surface() else {
        anyhow::bail!("workspace should classify the transformer block output surface");
    };
    assert_eq!(layer_surface.name, "transformer.block.output");
    assert_eq!(layer_surface.status, "layer_output_storage_blocked_by_candle_tensor_add_ops");
    assert_eq!(layer_surface.last_shape, vec![1, 1, hidden]);
    assert_eq!(layer_surface.residual_input_shape, Some(vec![1, 1, hidden]));
    assert_eq!(layer_surface.branch_output_shape, Some(vec![1, 1, hidden]));
    assert!(layer_surface.residual_add_involved);
    assert!(!layer_surface.weight_accessible);
    assert!(!layer_surface.can_fill_caller_output_storage);
    let Some(final_norm_surface) = workspace.final_norm_output_surface() else {
        anyhow::bail!("workspace should classify the final norm output surface");
    };
    assert_eq!(final_norm_surface.name, "model.final_norm.output");
    assert_eq!(
        final_norm_surface.status,
        "final_norm_output_storage_blocked_by_candle_layer_norm_ops"
    );
    assert_eq!(final_norm_surface.last_shape, vec![1, 1, hidden]);
    assert_eq!(final_norm_surface.weight_shape, Some(vec![hidden]));
    assert!(final_norm_surface.weight_accessible);
    assert!(!final_norm_surface.can_fill_caller_output_storage);
    assert!(
        !workspace.tensor_reuse_enabled(),
        "SLM-CPU-041 proves the dense linear output hook still lacks reusable Candle storage"
    );
    Ok(())
}

#[test]
fn final_norm_output_storage_boundary_records_candle_layer_norm_blocker() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::ones(4, DType::F32, &device)?;
    let norm = LayerNorm::rms_norm(weight, 1e-5);

    let boundary =
        NormOutputStorageApiBoundary::from_candle_layer_norm("model.final_norm.output", &norm);

    assert_eq!(boundary.role, "model.final_norm.output");
    assert_eq!(boundary.status, "final_norm_output_storage_blocked_by_candle_layer_norm_ops");
    assert_eq!(boundary.weight_shape, vec![4]);
    assert!(boundary.bias_shape.is_none());
    assert!(!boundary.remove_mean);
    assert!(boundary.weight_accessible);
    assert!(!boundary.bias_accessible);
    assert!(!boundary.can_fill_caller_output_storage);
    assert!(
        boundary.reason.contains("LayerNorm::forward")
            && boundary.reason.contains("caller-provided output-storage")
    );
    Ok(())
}

#[test]
fn layer_output_storage_boundary_records_candle_residual_add_blocker() {
    let boundary =
        LayerOutputStorageApiBoundary::from_candle_residual_add("transformer.block.output");

    assert_eq!(boundary.role, "transformer.block.output");
    assert_eq!(boundary.status, "layer_output_storage_blocked_by_candle_tensor_add_ops");
    assert!(boundary.residual_add_involved);
    assert!(!boundary.can_fill_caller_output_storage);
    assert!(
        boundary.reason.contains("residual-add")
            && boundary.reason.contains("caller-provided output-storage")
    );
}

#[test]
fn dense_linear_output_storage_boundary_records_candle_tensor_blocker() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::zeros((3, 2), DType::F32, &device)?;
    let bias = Tensor::zeros(3, DType::F32, &device)?;
    let linear = Linear::new(weight, Some(bias));

    let boundary = DenseLinearOutputStorageApiBoundary::from_candle_linear(
        "feed_forward.down_proj.output",
        &linear,
    );

    assert_eq!(boundary.role, "feed_forward.down_proj.output");
    assert_eq!(boundary.weight_shape, vec![3, 2]);
    assert_eq!(boundary.bias_shape, Some(vec![3]));
    assert!(boundary.weight_accessible);
    assert!(boundary.bias_accessible);
    assert!(!boundary.can_fill_caller_output_storage);
    assert_eq!(boundary.status, "dense_linear_output_storage_blocked_by_candle_tensor_ops");
    assert!(
        boundary.reason.contains("Tensor::matmul")
            && boundary.reason.contains("caller-provided output-storage")
    );
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundary_defaults_to_eager_f32() -> anyhow::Result<()> {
    let model = make_model(8, 16, 2)?;

    let boundary = model.dense_linear_runtime_hook_boundary("layers.0.attention.q_proj.weight");

    assert_eq!(boundary.selected_path, "eager_f32_candle");
    assert_eq!(boundary.selected_kernel, "dense-f32-candle-linear");
    assert!(!boundary.sidecar_descriptor_present);
    assert!(!boundary.runtime_compute_enabled);
    assert!(boundary.preserves_eager_f32());
    assert!(!boundary.speedup_claim);
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundary_accepts_inert_q8_sidecar_descriptor() -> anyhow::Result<()> {
    let config = tiny_config(8, 16, 2);
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    hooks.insert(
        "layers.0.attention.q_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: "AttentionQ".to_string(),
            sidecar_payload_sha256: Some("abc123".to_string()),
            packed_q8_payload: None,
            runtime_compute_enabled: false,
        },
    );
    let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
        config,
        vb,
        Default::default(),
        hooks,
    )?;

    let boundary = model.dense_linear_runtime_hook_boundary("layers.0.attention.q_proj.weight");

    assert_eq!(boundary.selected_path, "eager_f32_candle");
    assert_eq!(boundary.selected_kernel, "dense-f32-candle-linear");
    assert!(boundary.sidecar_descriptor_present);
    assert_eq!(boundary.sidecar_role.as_deref(), Some("AttentionQ"));
    assert_eq!(boundary.sidecar_payload_sha256.as_deref(), Some("abc123"));
    assert!(!boundary.sidecar_payload_bytes_available);
    assert_eq!(boundary.sidecar_payload_bytes, None);
    assert!(!boundary.sidecar_payload_contract_valid);
    assert!(!boundary.runtime_compute_enabled);
    assert!(boundary.preserves_eager_f32());
    assert!(boundary.generated_id_preservation_required_before_runtime_use);
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundary_can_carry_payload_without_enabling_compute()
-> anyhow::Result<()> {
    let config = tiny_config(8, 16, 2);
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let payload_bytes = vec![0_u8; 68];
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    hooks.insert(
        "layers.0.attention.q_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: "AttentionQ".to_string(),
            sidecar_payload_sha256: Some("sha256:payload".to_string()),
            packed_q8_payload: Some(DenseLinearPackedQ8Payload {
                tensor_name: "blk.0.attn_q.weight".to_string(),
                packed_q8_bytes: std::sync::Arc::from(payload_bytes.into_boxed_slice()),
                q8_block_size: 32,
                q8_block_count: 2,
                matrix_rows: 8,
                matrix_cols: 8,
            }),
            runtime_compute_enabled: false,
        },
    );
    let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
        config,
        vb,
        Default::default(),
        hooks,
    )?;

    let boundary = model.dense_linear_runtime_hook_boundary("layers.0.attention.q_proj.weight");

    assert_eq!(boundary.selected_path, "eager_f32_candle");
    assert_eq!(boundary.selected_kernel, "dense-f32-candle-linear");
    assert!(boundary.sidecar_descriptor_present);
    assert!(boundary.sidecar_payload_bytes_available);
    assert_eq!(boundary.sidecar_payload_bytes, Some(68));
    assert_eq!(boundary.sidecar_q8_block_count, Some(2));
    assert_eq!(boundary.sidecar_matrix_rows, Some(8));
    assert_eq!(boundary.sidecar_matrix_cols, Some(8));
    assert!(boundary.sidecar_payload_contract_valid);
    assert!(!boundary.runtime_compute_enabled);
    assert!(boundary.preserves_eager_f32());
    assert!(boundary.generated_id_preservation_required_before_runtime_use);
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundary_rejects_payload_tensor_mismatch() -> anyhow::Result<()> {
    let config = tiny_config(8, 16, 2);
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let payload_bytes = vec![0_u8; 68];
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    hooks.insert(
        "layers.0.attention.q_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: "AttentionQ".to_string(),
            sidecar_payload_sha256: Some("sha256:payload".to_string()),
            packed_q8_payload: Some(DenseLinearPackedQ8Payload {
                tensor_name: "blk.0.attn_k.weight".to_string(),
                packed_q8_bytes: std::sync::Arc::from(payload_bytes.into_boxed_slice()),
                q8_block_size: 32,
                q8_block_count: 2,
                matrix_rows: 8,
                matrix_cols: 8,
            }),
            runtime_compute_enabled: true,
        },
    );
    let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
        config,
        vb,
        Default::default(),
        hooks,
    )?;

    let boundary = model.dense_linear_runtime_hook_boundary("layers.0.attention.q_proj.weight");

    assert!(boundary.sidecar_payload_bytes_available);
    assert_eq!(boundary.sidecar_payload_bytes, Some(68));
    assert!(!boundary.sidecar_payload_contract_valid);
    assert!(!boundary.runtime_compute_enabled);
    assert!(boundary.preserves_eager_f32());
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundaries_report_sorted_receipt_identity() -> anyhow::Result<()> {
    let config = tiny_config(8, 16, 2);
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    hooks.insert(
        "layers.0.feed_forward.down_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.ffn_down.weight".to_string(),
            role: "MlpDown".to_string(),
            sidecar_payload_sha256: Some("sha256:down".to_string()),
            packed_q8_payload: None,
            runtime_compute_enabled: false,
        },
    );
    hooks.insert(
        "layers.0.attention.q_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: "AttentionQ".to_string(),
            sidecar_payload_sha256: Some("sha256:q".to_string()),
            packed_q8_payload: None,
            runtime_compute_enabled: false,
        },
    );
    let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
        config,
        vb,
        Default::default(),
        hooks,
    )?;

    let boundaries = model.dense_linear_runtime_hook_boundaries();

    assert_eq!(boundaries.len(), 2);
    assert_eq!(boundaries[0].tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(boundaries[1].tensor_name, "layers.0.feed_forward.down_proj.weight");
    assert!(boundaries.iter().all(DenseLinearRuntimeHookBoundary::preserves_eager_f32));
    assert!(boundaries.iter().all(|boundary| boundary.sidecar_descriptor_present));
    assert!(boundaries.iter().all(|boundary| !boundary.runtime_compute_enabled));
    Ok(())
}

#[test]
fn dense_linear_runtime_hook_boundary_does_not_enable_packed_compute() -> anyhow::Result<()> {
    let config = tiny_config(8, 16, 2);
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let mut hooks = DenseLinearRuntimeHookRegistry::default();
    hooks.insert(
        "layers.0.attention.q_proj.weight".to_string(),
        DenseLinearRuntimeHookDescriptor {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            role: "AttentionQ".to_string(),
            sidecar_payload_sha256: Some("sha256:future".to_string()),
            packed_q8_payload: None,
            runtime_compute_enabled: true,
        },
    );
    let model = TransformerModel::new_with_tensors_and_dense_linear_hooks(
        config,
        vb,
        Default::default(),
        hooks,
    )?;

    let boundary = model.dense_linear_runtime_hook_boundary("layers.0.attention.q_proj.weight");

    assert_eq!(boundary.selected_path, "eager_f32_candle");
    assert_eq!(boundary.selected_kernel, "dense-f32-candle-linear");
    assert!(boundary.sidecar_descriptor_present);
    assert!(!boundary.runtime_compute_enabled);
    assert!(!boundary.dense_runtime_replaced);
    assert!(boundary.preserves_eager_f32());
    Ok(())
}

#[test]
fn dense_q8_sidecar_fused_consumer_boundary_names_downstream_tensor_blocker() {
    let boundary = dense_q8_sidecar_fused_consumer_boundary();

    assert_eq!(boundary.role, "attention.q_proj.fused_output_consumer");
    assert_eq!(boundary.status, "blocked_by_downstream_candle_tensor_consumers");
    assert_eq!(boundary.exact_tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(boundary.exact_tensor_role, "AttentionQ");
    assert!(boundary.sidecar_inner_matvec_accepts_output_slice);
    assert!(!boundary.can_avoid_returned_candle_tensor_for_current_consumer);
    assert!(boundary.downstream_consumers_require_tensor_semantics);
    assert!(boundary.appliance_oracle_required_before_claim);
    assert!(boundary.exact_blocking_ops.iter().any(|op| op.contains("Tensor::from_vec")));
    assert!(boundary.exact_blocking_ops.iter().any(|op| op.contains("reshape_qkv_heads")));
    assert!(boundary.required_missing_api.contains("typed fused Q projection consumer"));
}

#[test]
fn dense_q8_sidecar_fused_q_projection_contract_is_design_only() {
    let boundary = dense_q8_sidecar_fused_consumer_boundary();
    let contract = dense_q8_sidecar_fused_q_projection_consumer_contract();

    assert_eq!(contract.role, "attention.q_proj.typed_fused_consumer_contract");
    assert_eq!(contract.status, "contract_defined_runtime_disabled");
    assert_eq!(contract.source_boundary_status, boundary.status);
    assert_eq!(contract.exact_tensor_name, boundary.exact_tensor_name);
    assert_eq!(contract.exact_tensor_role, "AttentionQ");
    assert!(contract.owns_packed_q8_matvec_output_slice);
    assert!(!contract.intermediate_returned_candle_tensor_allowed);
    assert!(!contract.runtime_execution_enabled);
    assert!(!contract.default_runtime_changed);
    assert!(!contract.allocation_reduction_claim);
    assert!(!contract.speedup_claim);
}

#[test]
fn dense_q8_sidecar_fused_q_projection_contract_covers_q_downstream_stages() {
    let contract = dense_q8_sidecar_fused_q_projection_consumer_contract();
    let stages: Vec<_> = contract.stages.iter().map(|stage| stage.stage).collect();

    assert_eq!(
        stages,
        vec![
            "packed_q8_matvec_output_slice",
            "q_proj_reshape",
            "q_proj_transpose",
            "optional_q_norm",
            "q_rope",
            "trace_workspace_identity",
            "attention_head_handoff",
        ]
    );
    assert!(contract.stages.iter().all(|stage| stage.fused_consumer_must_own));
    assert!(
        contract
            .stages
            .iter()
            .find(|stage| stage.stage == "optional_q_norm")
            .is_some_and(|stage| stage.optional)
    );
    assert_eq!(contract.shape.input_rank, 3);
    assert_eq!(contract.shape.projected_rank, 3);
    assert_eq!(contract.shape.attention_heads_rank, 4);
    assert_eq!(contract.shape.head_handoff_shape, "AttentionHeads.q");
}

#[test]
fn dense_q8_sidecar_fused_q_projection_contract_requires_behavior_receipts() {
    let contract = dense_q8_sidecar_fused_q_projection_consumer_contract();

    assert!(contract.receipt.required_before_runtime_execution);
    assert!(contract.receipt.required_before_allocation_claim);
    assert!(contract.receipt.required_before_speedup_claim);
    for required in [
        "model.sha256",
        "tokenizer.strict=true",
        "prompt_ids",
        "generated_ids",
        "decoded_text",
        "selected_backend=cpu-rust",
        "dense_hook identity",
        "fallback_used=false",
    ] {
        assert!(
            contract.receipt.required_fields.contains(&required),
            "contract receipt fields should include {required}"
        );
    }
    assert!(contract.receipt.gate.contains("before_after_receipts"));
}

#[test]
fn dense_q8_sidecar_typed_fused_q_projection_gate_blocks_runtime_execution() {
    let gate = dense_q8_sidecar_typed_fused_q_projection_implementation_gate();

    assert_eq!(gate.role, "attention.q_proj.typed_fused_consumer_implementation_gate");
    assert_eq!(gate.status, "blocked_runtime_disabled");
    assert_eq!(gate.source_contract_status, "contract_defined_runtime_disabled");
    assert_eq!(gate.exact_tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(gate.exact_tensor_role, "AttentionQ");
    assert!(!gate.attempted_runtime_implementation);
    assert!(gate.can_own_packed_q8_matvec_output_slice);
    assert!(!gate.can_preserve_downstream_tensor_semantics_without_intermediate_tensor);
    assert!(!gate.runtime_execution_enabled);
    assert!(!gate.default_runtime_changed);
    assert!(!gate.allocation_reduction_claim);
    assert!(!gate.speedup_claim);
}

#[test]
fn dense_q8_sidecar_typed_fused_q_projection_gate_names_exact_blockers() {
    let gate = dense_q8_sidecar_typed_fused_q_projection_implementation_gate();
    let blockers: Vec<_> = gate.blockers.iter().map(|blocker| blocker.blocker).collect();

    assert_eq!(
        blockers,
        vec![
            "q_heads_tensor_semantics",
            "q_norm_tensor_api",
            "rope_tensor_api",
            "trace_workspace_tensor_identity",
            "attention_handoff_tensor_contract",
            "receipt_safety_evidence",
        ]
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("LayerNorm::forward"))
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("RotaryEmbedding::apply"))
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("prepare_attention_scores"))
    );
    assert!(gate.receipt_gate.required_before_runtime_execution);
    assert!(gate.receipt_gate.required_before_allocation_claim);
    assert!(gate.next_required_slice.contains("typed attention-head buffer"));
}

#[test]
fn dense_q8_sidecar_typed_attention_head_view_gate_is_design_only() {
    let source = dense_q8_sidecar_typed_fused_q_projection_implementation_gate();
    let gate = dense_q8_sidecar_typed_attention_head_view_gate();

    assert_eq!(gate.role, "attention.q_proj.typed_attention_head_view_gate");
    assert_eq!(gate.status, "contract_defined_runtime_disabled");
    assert_eq!(gate.source_gate_status, source.status);
    assert_eq!(gate.exact_tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(gate.exact_tensor_role, "AttentionQ");
    assert!(gate.can_represent_q_heads_without_candle_tensor);
    assert!(!gate.can_feed_current_attention_score_api_without_materialization);
    assert_eq!(gate.selected_materialization_point, None);
    assert!(!gate.runtime_execution_enabled);
    assert!(!gate.default_runtime_changed);
    assert!(!gate.allocation_reduction_claim);
    assert!(!gate.speedup_claim);
}

#[test]
fn dense_q8_sidecar_typed_attention_head_view_gate_covers_downstream_stages() {
    let gate = dense_q8_sidecar_typed_attention_head_view_gate();
    let stages: Vec<_> = gate.stages.iter().map(|stage| stage.stage).collect();

    assert_eq!(
        stages,
        vec![
            "q_projection_output_slice",
            "logical_head_view",
            "optional_q_norm_handoff",
            "q_rope_handoff",
            "trace_workspace_identity",
            "attention_score_handoff",
        ]
    );
    assert_eq!(gate.layout.projected_rank, 3);
    assert_eq!(gate.layout.attention_heads_rank, 4);
    assert_eq!(gate.layout.projected_shape, "[batch, seq, n_heads * head_dim]");
    assert_eq!(gate.layout.attention_heads_shape, "[batch, n_heads, seq, head_dim]");
    assert!(
        gate.stages
            .iter()
            .find(|stage| stage.stage == "logical_head_view")
            .is_some_and(|stage| !stage.candle_tensor_semantics_required_today)
    );
    assert!(
        gate.stages
            .iter()
            .find(|stage| stage.stage == "optional_q_norm_handoff")
            .is_some_and(|stage| stage.optional && stage.candle_tensor_semantics_required_today)
    );
}

#[test]
fn dense_q8_sidecar_typed_attention_head_view_gate_names_runtime_blockers() {
    let gate = dense_q8_sidecar_typed_attention_head_view_gate();
    let blockers: Vec<_> = gate.blockers.iter().map(|blocker| blocker.blocker).collect();

    assert_eq!(
        blockers,
        vec![
            "q_norm_requires_tensor_or_typed_norm",
            "rope_requires_tensor_or_typed_rope",
            "trace_source_identity_requires_tensor_mapping",
            "attention_scores_require_tensor_or_typed_score_path",
            "receipt_safety_evidence",
        ]
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("LayerNorm::forward"))
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("RotaryEmbedding::apply"))
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("prepare_attention_scores"))
    );
    assert!(gate.receipt_gate.required_before_runtime_execution);
    assert!(gate.receipt_gate.required_before_allocation_claim);
    assert!(gate.next_required_slice.contains("typed q_norm/RoPE"));
}

#[test]
fn dense_q8_sidecar_typed_attention_head_consumer_gate_blocks_runtime_execution() {
    let source = dense_q8_sidecar_typed_attention_head_view_gate();
    let gate = dense_q8_sidecar_typed_attention_head_consumer_gate();

    assert_eq!(gate.role, "attention.q_proj.typed_attention_head_consumer_gate");
    assert_eq!(gate.status, "blocked_runtime_disabled");
    assert_eq!(gate.source_gate_status, source.status);
    assert_eq!(gate.exact_tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(gate.exact_tensor_role, "AttentionQ");
    assert!(gate.can_consume_projection_output_slice);
    assert!(gate.can_apply_logical_head_view_without_candle_tensor);
    assert!(!gate.can_apply_q_norm_without_candle_tensor);
    assert!(!gate.can_apply_rope_without_candle_tensor);
    assert!(!gate.can_feed_attention_scores_without_candle_tensor);
    assert_eq!(gate.first_blocking_stage, "q_norm_consumer");
    assert_eq!(gate.accepted_single_materialization_point, None);
    assert!(!gate.runtime_execution_enabled);
    assert!(!gate.default_runtime_changed);
    assert!(!gate.allocation_reduction_claim);
    assert!(!gate.speedup_claim);
}

#[test]
fn dense_q8_sidecar_typed_attention_head_consumer_gate_names_handoff_blockers() {
    let gate = dense_q8_sidecar_typed_attention_head_consumer_gate();
    let stages: Vec<_> = gate.stages.iter().map(|stage| stage.stage).collect();
    let blockers: Vec<_> = gate.blockers.iter().map(|blocker| blocker.blocker).collect();

    assert_eq!(
        stages,
        vec![
            "projection_slice_ingress",
            "logical_head_view_ingress",
            "q_norm_consumer",
            "rope_consumer",
            "trace_identity_consumer",
            "attention_score_consumer",
            "receipt_safety_gate",
        ]
    );
    assert_eq!(
        blockers,
        vec![
            "q_norm_typed_consumer_absent",
            "rope_typed_consumer_absent",
            "trace_identity_typed_receipt_gap",
            "attention_score_typed_path_absent",
            "accumulator_order_unproven",
            "receipt_safety_evidence",
        ]
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("LayerNorm::forward"))
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("RotaryEmbedding::apply"))
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("prepare_attention_scores"))
    );
    assert!(gate.blockers.iter().any(|blocker| blocker.category == "accumulator-order"));
}

#[test]
fn dense_q8_sidecar_typed_attention_head_consumer_gate_keeps_receipt_gate_strict() {
    let gate = dense_q8_sidecar_typed_attention_head_consumer_gate();

    assert!(gate.receipt_gate.required_before_runtime_execution);
    assert!(gate.receipt_gate.required_before_allocation_claim);
    assert!(gate.receipt_gate.required_before_speedup_claim);
    assert!(gate.receipt_gate.gate.contains("repeated_qwen3_q8_before_after_receipts"));
    assert!(gate.receipt_gate.required_fields.contains(&"fallback_used=false"));
    assert!(
        gate.candidate_materialization_points
            .contains(&"after_q_rope_before_attention_scores_candle_tensor_boundary")
    );
    assert!(gate.next_required_slice.contains("strict Qwen3/Qwen2.5 CPU receipts"));
}

#[test]
fn dense_q8_sidecar_typed_q_norm_rope_consumer_gate_blocks_runtime_execution() {
    let source = dense_q8_sidecar_typed_attention_head_consumer_gate();
    let gate = dense_q8_sidecar_typed_q_norm_rope_consumer_gate();

    assert_eq!(gate.role, "attention.q_proj.typed_q_norm_rope_consumer_gate");
    assert_eq!(gate.status, "blocked_runtime_disabled");
    assert_eq!(gate.source_gate_status, source.status);
    assert_eq!(gate.exact_tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(gate.exact_tensor_role, "AttentionQ");
    assert!(gate.can_consume_logical_q_head_view);
    assert!(!gate.can_apply_typed_q_norm_without_candle_tensor);
    assert!(!gate.can_apply_typed_rope_without_candle_tensor);
    assert!(!gate.can_preserve_trace_identity_without_tensor_mapping);
    assert!(!gate.can_feed_attention_scores_without_candle_tensor);
    assert_eq!(gate.first_blocking_stage, "typed_q_norm_consumer");
    assert_eq!(gate.accepted_single_materialization_point, None);
    assert!(!gate.runtime_execution_enabled);
    assert!(!gate.default_runtime_changed);
    assert!(!gate.packed_q8_sidecar_default_enabled);
    assert!(!gate.allocation_reduction_claim);
    assert!(!gate.speedup_claim);
}

#[test]
fn dense_q8_sidecar_typed_q_norm_rope_consumer_gate_names_precise_blockers() {
    let gate = dense_q8_sidecar_typed_q_norm_rope_consumer_gate();
    let stages: Vec<_> = gate.stages.iter().map(|stage| stage.stage).collect();
    let blockers: Vec<_> = gate.blockers.iter().map(|blocker| blocker.blocker).collect();

    assert_eq!(
        stages,
        vec![
            "typed_q_head_view_ingress",
            "typed_q_norm_consumer",
            "typed_rope_consumer",
            "trace_workspace_identity_handoff",
            "attention_score_handoff",
            "receipt_safety_gate",
        ]
    );
    assert_eq!(
        blockers,
        vec![
            "typed_q_norm_kernel_absent",
            "typed_rope_kernel_absent",
            "trace_identity_typed_surface_absent",
            "score_handoff_typed_surface_absent",
            "single_materialization_boundary_unproven",
            "accumulator_order_receipt_absent",
            "receipt_safety_evidence",
        ]
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("LayerNorm::forward"))
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("RotaryEmbedding::apply"))
    );
    assert!(
        gate.blockers
            .iter()
            .any(|blocker| blocker.exact_api_or_surface.contains("prepare_attention_scores"))
    );
    assert!(gate.blockers.iter().any(|blocker| blocker.category == "accumulator-order"));
}

#[test]
fn dense_q8_sidecar_typed_q_norm_rope_consumer_gate_keeps_receipt_gate_strict() {
    let gate = dense_q8_sidecar_typed_q_norm_rope_consumer_gate();

    assert!(gate.receipt_gate.required_before_runtime_execution);
    assert!(gate.receipt_gate.required_before_allocation_claim);
    assert!(gate.receipt_gate.required_before_speedup_claim);
    assert!(gate.receipt_gate.required_fields.contains(&"fallback_used=false"));
    assert!(
        gate.candidate_materialization_points
            .contains(&"after_q_rope_before_attention_scores_candle_tensor_boundary")
    );
    assert!(
        gate.next_required_slice
            .contains("one proven materialization boundary before attention scores")
    );
    assert!(
        gate.stages
            .iter()
            .any(|stage| stage.current_status == "blocked_until_behavior_oracles_pass")
    );
}

#[test]
fn dense_q8_sidecar_q_norm_materialization_boundary_selects_one_point() {
    let source = dense_q8_sidecar_typed_q_norm_rope_consumer_gate();
    let gate = dense_q8_sidecar_q_norm_materialization_boundary_gate();

    assert_eq!(gate.role, "attention.q_proj.q_norm_input_materialization_boundary_gate");
    assert_eq!(gate.status, "boundary_selected_runtime_disabled");
    assert_eq!(gate.source_gate_status, source.status);
    assert_eq!(gate.exact_tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(gate.exact_tensor_role, "AttentionQ");
    assert_eq!(gate.accepted_single_materialization_point, "q_norm_input_candle_tensor_boundary");
    assert_eq!(gate.materializes_before_stage, "typed_q_norm_consumer");
    assert_eq!(
        gate.rejected_materialization_points,
        &[
            "after_q_norm_before_rope_candle_tensor_boundary",
            "after_q_rope_before_attention_scores_candle_tensor_boundary",
        ]
    );
}

#[test]
fn dense_q8_sidecar_q_norm_materialization_boundary_preserves_candle_consumers() {
    let gate = dense_q8_sidecar_q_norm_materialization_boundary_gate();

    assert!(
        gate.preserved_candle_consumers
            .iter()
            .any(|consumer| consumer.contains("LayerNorm::forward"))
    );
    assert!(
        gate.preserved_candle_consumers
            .iter()
            .any(|consumer| consumer.contains("RotaryEmbedding::apply"))
    );
    assert!(
        gate.preserved_candle_consumers
            .iter()
            .any(|consumer| consumer.contains("TransformerAttentionOutputSourceTensors"))
    );
    assert!(
        gate.preserved_candle_consumers
            .iter()
            .any(|consumer| consumer.contains("prepare_attention_scores"))
    );
}

#[test]
fn dense_q8_sidecar_q_norm_materialization_boundary_keeps_runtime_disabled() {
    let gate = dense_q8_sidecar_q_norm_materialization_boundary_gate();

    assert!(!gate.runtime_execution_enabled);
    assert!(!gate.default_runtime_changed);
    assert!(!gate.packed_q8_sidecar_default_enabled);
    assert!(!gate.allocation_reduction_claim);
    assert!(!gate.speedup_claim);
    assert!(!gate.sustained_throughput_claim);
    assert!(!gate.q4_q5_runtime_claim);
    assert!(gate.qwen3_q8_before_after_receipts_required);
    assert!(gate.qwen25_q8_before_after_receipts_required);
    assert!(gate.receipt_gate.required_before_runtime_execution);
    assert!(gate.receipt_gate.required_before_allocation_claim);
    assert!(gate.receipt_gate.required_before_speedup_claim);
    assert!(gate.receipt_gate.required_fields.contains(&"fallback_used=false"));
    assert!(gate.next_required_slice.contains("before/after Qwen3 Q8_0 and Qwen2.5 Q8_0 receipts"));
}

#[test]
fn dense_q8_sidecar_q_norm_input_proof_gate_blocks_without_receipts() {
    let source = dense_q8_sidecar_q_norm_materialization_boundary_gate();
    let gate = dense_q8_sidecar_q_norm_input_proof_gate();

    assert_eq!(gate.role, "attention.q_proj.q_norm_input_materialization_proof_gate");
    assert_eq!(
        gate.status,
        "blocked_missing_runtime_hook_and_before_after_receipts_comparator_defined"
    );
    assert_eq!(gate.source_boundary_status, source.status);
    assert_eq!(gate.exact_tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(gate.exact_tensor_role, "AttentionQ");
    assert_eq!(gate.selected_materialization_boundary, "q_norm_input_candle_tensor_boundary");
    assert!(!gate.proof_ready);
    assert!(gate.missing_runtime_hook);
    assert!(gate.missing_receipt_field);
    assert!(!gate.missing_comparator);
    assert!(gate.comparator_contract_defined);
    assert!(gate.tensor_identity_unrecorded);
    assert!(gate.accumulator_order_unproven);
    assert!(gate.artifact_gap);
}

#[test]
fn dense_q8_sidecar_q_norm_input_proof_gate_names_required_receipt_pairs() {
    let gate = dense_q8_sidecar_q_norm_input_proof_gate();

    assert_eq!(gate.required_receipts.len(), 2);
    assert!(gate.required_receipts.iter().any(|receipt| {
        receipt.model_id == "qwen3-0.6b-q8_0"
            && receipt.model_architecture == "qwen3"
            && receipt.required_fields.contains(&"generated_ids")
            && receipt.required_fields.contains(&"fallback_used=false")
    }));
    assert!(gate.required_receipts.iter().any(|receipt| {
        receipt.model_id == "qwen2.5-0.5b-instruct-q8_0"
            && receipt.model_architecture == "qwen2"
            && receipt.required_fields.contains(&"dense_hook identity")
            && receipt.required_fields.contains(&"selected_backend=cpu-rust")
    }));
}

#[test]
fn dense_q8_sidecar_q_norm_input_proof_gate_names_precise_blockers() {
    let gate = dense_q8_sidecar_q_norm_input_proof_gate();

    for blocker in [
        "q_norm_input_runtime_hook_missing",
        "qwen3_q8_before_after_receipts_missing",
        "qwen25_q8_before_after_receipts_missing",
        "q_norm_input_tensor_identity_unrecorded",
        "accumulator_order_unproven",
    ] {
        assert!(
            gate.blockers.iter().any(|candidate| candidate.blocker == blocker),
            "missing blocker {blocker}"
        );
    }
    assert!(
        !gate
            .blockers
            .iter()
            .any(|candidate| candidate.blocker == "q_norm_input_receipt_comparator_missing")
    );
}

#[test]
fn dense_q8_sidecar_q_norm_input_proof_gate_keeps_claim_boundary() {
    let gate = dense_q8_sidecar_q_norm_input_proof_gate();

    assert!(!gate.runtime_execution_enabled);
    assert!(!gate.default_runtime_changed);
    assert!(!gate.packed_q8_sidecar_default_enabled);
    assert!(!gate.allocation_reduction_claim);
    assert!(!gate.speedup_claim);
    assert!(!gate.sustained_throughput_claim);
    assert!(!gate.q4_q5_runtime_claim);
    assert!(!gate.server_or_accelerator_claim);
    assert!(!gate.qwen35_claim);
    assert!(!gate.bitnet_qk256_claim);
    assert!(gate.next_required_slice.contains("fail-closed comparator"));
}

fn sample_q_norm_input_receipt_identity() -> DenseQ8SidecarQNormInputReceiptIdentity {
    DenseQ8SidecarQNormInputReceiptIdentity {
        model_id: "qwen3-0.6b-q8_0",
        model_sha256: "model-sha256",
        tokenizer_source: "gguf_metadata",
        tokenizer_strict: true,
        prompt_ids_digest: "prompt-ids-digest",
        generated_ids_digest: "generated-ids-digest",
        decoded_text_digest: "decoded-text-digest",
        selected_backend: "cpu-rust",
        selected_kernel_identity: "eager_f32_candle",
        dense_hook_identity: "layers.0.attention.q_proj.weight",
        q_norm_input_boundary: "q_norm_input_candle_tensor_boundary",
        q_norm_input_tensor_identity: "shape=[1,1,1024];dtype=f32;source=q_proj",
        fallback_used: false,
    }
}

#[test]
fn dense_q8_sidecar_q_norm_input_receipt_comparator_gate_names_contract() {
    let gate = dense_q8_sidecar_q_norm_input_receipt_comparator_gate();

    assert_eq!(gate.role, "attention.q_proj.q_norm_input_receipt_identity_comparator");
    assert_eq!(gate.selected_materialization_boundary, "q_norm_input_candle_tensor_boundary");
    assert!(gate.fail_closed_on_missing_field);
    assert!(gate.fail_closed_on_mismatch);
    assert!(gate.fail_closed_on_fallback);
    assert!(gate.compares_qwen3_q8);
    assert!(gate.compares_qwen25_q8);
    for required in [
        "model_sha256",
        "tokenizer_source=gguf_metadata",
        "prompt_ids_digest",
        "generated_ids_digest",
        "decoded_text_digest",
        "selected_backend=cpu-rust",
        "q_norm_input_boundary=q_norm_input_candle_tensor_boundary",
        "q_norm_input_tensor_identity",
        "fallback_used=false",
    ] {
        assert!(
            gate.required_identity_fields.contains(&required),
            "missing required field {required}"
        );
    }
    assert!(gate.remaining_blockers.contains(&"q_norm_input_runtime_hook_missing"));
    assert!(gate.remaining_blockers.contains(&"qwen25_q8_before_after_receipts_missing"));
    assert!(!gate.runtime_execution_enabled);
    assert!(!gate.speedup_claim);
    assert!(!gate.server_or_accelerator_claim);
    assert!(!gate.bitnet_qk256_claim);
}

#[test]
fn dense_q8_sidecar_q_norm_input_receipt_comparator_passes_identical_identity() {
    let before = sample_q_norm_input_receipt_identity();
    let after = before.clone();

    let comparison = compare_dense_q8_sidecar_q_norm_input_receipts(&before, &after);

    assert!(comparison.passed);
    assert!(comparison.failed_fields.is_empty());
}

#[test]
fn dense_q8_sidecar_q_norm_input_receipt_comparator_fails_closed_on_gaps() {
    let before = sample_q_norm_input_receipt_identity();
    let after = DenseQ8SidecarQNormInputReceiptIdentity {
        generated_ids_digest: "different-generated-ids",
        q_norm_input_boundary: "wrong_boundary",
        q_norm_input_tensor_identity: "",
        fallback_used: true,
        ..before.clone()
    };

    let comparison = compare_dense_q8_sidecar_q_norm_input_receipts(&before, &after);

    assert!(!comparison.passed);
    for failed in [
        "fallback_used",
        "generated_ids_digest",
        "q_norm_input_boundary",
        "q_norm_input_tensor_identity",
    ] {
        assert!(comparison.failed_fields.contains(&failed), "missing failed field {failed}");
    }
}

#[test]
fn dense_q8_sidecar_q_norm_input_tensor_identity_surface_records_boundary_source_shape_dtype() {
    let tensor =
        Tensor::new(&[1f32, 2., 3., 4.], &Device::Cpu).unwrap().reshape((1, 1, 4)).unwrap();

    let identity = dense_q8_sidecar_q_norm_input_tensor_identity_surface(
        "layers.0.attention.q_proj.weight",
        &tensor,
    );

    assert_eq!(identity.boundary, "q_norm_input_candle_tensor_boundary");
    assert_eq!(identity.source_tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(identity.shape, vec![1, 1, 4]);
    assert_eq!(identity.dtype, "F32");
    assert!(identity.identity.contains("boundary=q_norm_input_candle_tensor_boundary"));
    assert!(identity.identity.contains("source=layers.0.attention.q_proj.weight"));
    assert!(identity.identity.contains("shape=[1, 1, 4]"));
    assert!(identity.identity.contains("dtype=F32"));
}

#[test]
fn dense_q8_sidecar_q_norm_input_runtime_hook_gate_defines_disabled_hook_surface() {
    let comparator = dense_q8_sidecar_q_norm_input_receipt_comparator_gate();
    let gate = dense_q8_sidecar_q_norm_input_runtime_hook_gate();

    assert_eq!(gate.role, "attention.q_proj.q_norm_input_runtime_hook_gate");
    assert_eq!(
        gate.status,
        "runtime_disabled_hook_and_tensor_identity_surface_defined_receipts_still_required"
    );
    assert_eq!(gate.source_comparator_status, comparator.status);
    assert_eq!(gate.exact_tensor_name, "layers.0.attention.q_proj.weight");
    assert_eq!(gate.exact_tensor_role, "AttentionQ");
    assert_eq!(gate.selected_materialization_boundary, "q_norm_input_candle_tensor_boundary");
    assert!(gate.runtime_disabled_hook_surface_defined);
    assert!(gate.receipt_tensor_identity_surface_defined);
    assert!(gate.comparator_contract_defined);
    assert!(!gate.proof_ready);
}

#[test]
fn dense_q8_sidecar_q_norm_input_runtime_hook_gate_names_remaining_receipt_blockers_only() {
    let gate = dense_q8_sidecar_q_norm_input_runtime_hook_gate();

    for blocker in [
        "qwen3_q8_before_after_receipts_missing",
        "qwen25_q8_before_after_receipts_missing",
        "accumulator_order_unproven",
    ] {
        assert!(gate.remaining_blockers.contains(&blocker), "missing remaining blocker {blocker}");
    }
    assert!(!gate.remaining_blockers.contains(&"q_norm_input_runtime_hook_missing"));
    assert!(!gate.remaining_blockers.contains(&"q_norm_input_tensor_identity_unrecorded"));
    assert!(!gate.remaining_blockers.contains(&"q_norm_input_receipt_comparator_missing"));
}

#[test]
fn dense_q8_sidecar_q_norm_input_runtime_hook_gate_keeps_claim_boundary() {
    let gate = dense_q8_sidecar_q_norm_input_runtime_hook_gate();

    assert!(!gate.runtime_execution_enabled);
    assert!(!gate.default_runtime_changed);
    assert!(!gate.packed_q8_sidecar_default_enabled);
    assert!(!gate.allocation_reduction_claim);
    assert!(!gate.speedup_claim);
    assert!(!gate.sustained_throughput_claim);
    assert!(!gate.q4_q5_runtime_claim);
    assert!(!gate.server_or_accelerator_claim);
    assert!(!gate.qwen35_claim);
    assert!(!gate.bitnet_qk256_claim);
}

// ── construction tests ────────────────────────────────────────────────────────

/// Model construction must succeed for different hidden/vocab/head combinations.
#[test]
fn test_construction_variants() -> anyhow::Result<()> {
    let cases = [(32, 64, 2), (64, 128, 4), (128, 256, 8)];
    for (h, v, n) in cases {
        make_model(h, v, n)
            .unwrap_or_else(|e| panic!("construction failed for h={h}, v={v}, n={n}: {e}"));
    }
    Ok(())
}

/// Construction must fail when hidden is not divisible by num_heads.
#[test]
fn test_construction_fails_bad_head_dim() {
    // hidden=60, heads=8: 60 % 8 != 0 → should fail
    let result = make_model(60, 64, 8);
    assert!(result.is_err(), "Should fail: hidden=60 not divisible by heads=8");
}
