//! Single-responsibility stages for [`MultiHeadAttention::forward`].
//!
//! The public forward path is intentionally a thin pipeline: projection,
//! rotary/cache preparation, grouped-query expansion, score preparation,
//! softmax, and output projection are each isolated below.

#[cfg(feature = "trace")]
use super::BitNetError;
use super::{
    DenseLinearRuntimeHookRegistry, LayerKVCache, MultiHeadAttention,
    QWEN_QPROJ_OUTPUT_PRE_OPTIONAL_QNORM_BOUNDARY, QWEN_QPROJ_OUTPUT_PRE_OPTIONAL_QNORM_STAGE,
    QwenTraceDenseHookIdentity, TransformerA770OpenClRuntimeDelta,
    TransformerA770OpenClRuntimeDevice, TransformerForwardWorkspace,
    TransformerQk256CpuHotPathDelta, TransformerQk256DeviceExpressionSample,
    TransformerQk256DeviceExpressionTrace, TransformerQk256DeviceIntermediateSample,
    TransformerQk256DeviceIntermediateTrace, TransformerQk256DispatchDelta,
    TransformerQk256FocusedRawOperands, TransformerQk256FullProjectionRawOperands,
    TransformerQkvProjectionDispatchReplayA770Stats,
    TransformerQkvProjectionDispatchReplayCpuStats, TransformerQkvProjectionDispatchReplayTensors,
    TransformerQkvProjectionSourceTensors, attention_f16_dot_input, attention_score_key_input,
    dbg_finite, dbg_stats, debug_attn_enabled, debug_attn_scale_enabled, debug_gqa_enabled,
    debug_rope_enabled, maybe_trace_dense_q8_source_order_qproj_candidate, qk256_inline_scale,
    qwen_trace_event, qwen_trace_events_enabled, qwen_trace_layer_enabled, qwen_trace_number,
    qwen_trace_tensor, qwen_trace_tensor_fingerprint,
    qwen_trace_tensor_fingerprint_with_dense_hook, trace_rms_enabled,
};
use bitnet_common::Result;
use candle_core::{DType, Module, Tensor};
use std::time::Instant;

struct QkvProjections {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

struct AttentionHeads {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

struct ExpandedKv {
    k: Tensor,
    v: Tensor,
}

struct AttentionOutputProjection {
    projection_input: Tensor,
    sub_layernorm_output: Option<Tensor>,
    output: Tensor,
}

impl MultiHeadAttention {
    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        mut workspace: Option<&mut TransformerForwardWorkspace>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let trace_attention = qwen_trace_events_enabled();
        let attention_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.forward_start", || {
            format!(
                "\"layer\":{},\"batch\":{},\"seq_len\":{},\"n_heads\":{},\"n_kv_heads\":{},\"head_dim\":{}",
                self.layer_idx, batch_size, seq_len, self.n_heads, self.n_kv_heads, self.head_dim
            )
        });

        let projection_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.qkv_projection_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let projections =
            self.project_qkv(x, raw_tensors, dense_linear_hooks, workspace.as_deref_mut())?;
        self.trace_qproj_output_pre_optional_qnorm(&projections.q)?;
        let q_projection_for_source = projections.q.clone();
        let k_projection_for_source = projections.k.clone();
        let v_projection_for_source = projections.v.clone();
        qwen_attention_trace_event(trace_attention, "attention.qkv_projection_finish", || {
            format!(
                "\"layer\":{},\"projection_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(projection_start)
            )
        });
        self.trace_projection_rms_once(&projections)?;
        self.trace_q_projection(&projections.q)?;

        let reshape_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.reshape_heads_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let heads = self.reshape_qkv_heads(projections, batch_size, seq_len)?;
        let q_heads_for_source = heads.q.clone();
        let k_heads_for_source = heads.k.clone();
        let v_heads_for_source = heads.v.clone();
        qwen_attention_trace_event(trace_attention, "attention.reshape_heads_finish", || {
            format!(
                "\"layer\":{},\"reshape_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(reshape_start)
            )
        });
        let qk_norm_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.qk_norm_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let heads = self.apply_qk_norms(heads)?;
        let q_norm_for_source = heads.q.clone();
        let k_norm_for_source = heads.k.clone();
        qwen_attention_trace_event(trace_attention, "attention.qk_norm_finish", || {
            format!(
                "\"layer\":{},\"qk_norm_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(qk_norm_start)
            )
        });
        self.log_gqa_shapes_once(&heads)?;

        let rope_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.rope_start", || {
            format!(
                "\"layer\":{},\"cache_seq_len\":{}",
                self.layer_idx,
                kv_cache.as_ref().map(|cache| cache.seq_len).unwrap_or(0)
            )
        });
        let heads = self.apply_rotary_embeddings(heads, kv_cache.as_ref().map(|c| c.seq_len))?;
        let q_rope_for_source = heads.q.clone();
        let k_rope_for_source = heads.k.clone();
        qwen_attention_trace_event(trace_attention, "attention.rope_finish", || {
            format!(
                "\"layer\":{},\"rope_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(rope_start)
            )
        });
        let kv_cache_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.kv_cache_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let (k_ctx, v_ctx) = Self::kv_context(&heads.k, &heads.v, kv_cache)?;
        let k_context_for_source = k_ctx.clone();
        let v_context_for_source = v_ctx.clone();
        qwen_attention_trace_event(trace_attention, "attention.kv_cache_finish", || {
            format!(
                "\"layer\":{},\"kv_cache_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(kv_cache_start)
            )
        });
        let gqa_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.gqa_expand_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let expanded = self.expand_grouped_query_kv(k_ctx, v_ctx, batch_size)?;
        let expanded_k_for_source = expanded.k.clone();
        let expanded_v_for_source = expanded.v.clone();
        qwen_attention_trace_event(trace_attention, "attention.gqa_expand_finish", || {
            format!(
                "\"layer\":{},\"gqa_expand_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(gqa_start)
            )
        });

        let scores_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.scores_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let scores = self.prepare_attention_scores(&heads.q, &expanded.k, seq_len)?;
        let scores_for_source = scores.clone();
        qwen_attention_trace_event(trace_attention, "attention.scores_finish", || {
            format!(
                "\"layer\":{},\"scores_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(scores_start)
            )
        });
        let softmax_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.softmax_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let attn_weights = self.softmax_attention_scores(&scores)?;
        let probabilities_for_source = attn_weights.clone();
        qwen_attention_trace_event(trace_attention, "attention.softmax_finish", || {
            format!(
                "\"layer\":{},\"softmax_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(softmax_start)
            )
        });
        let value_mix_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.value_mix_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let output_heads = self.apply_attention_weights(&attn_weights, &expanded.v)?;
        let output_heads_for_source = output_heads.clone();
        qwen_attention_trace_event(trace_attention, "attention.value_mix_finish", || {
            format!(
                "\"layer\":{},\"value_mix_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(value_mix_start)
            )
        });

        let output_projection_start = Instant::now();
        qwen_attention_trace_event(trace_attention, "attention.output_projection_start", || {
            format!("\"layer\":{}", self.layer_idx)
        });
        let output_projection = self.project_attention_output(
            output_heads,
            batch_size,
            seq_len,
            raw_tensors,
            dense_linear_hooks,
        )?;
        if let Some(workspace) = workspace.as_mut() {
            workspace.record_attention_output_source_tensors(
                self.layer_idx,
                x,
                &q_projection_for_source,
                &k_projection_for_source,
                &v_projection_for_source,
                &q_heads_for_source,
                &k_heads_for_source,
                &v_heads_for_source,
                &q_norm_for_source,
                &k_norm_for_source,
                &q_rope_for_source,
                &k_rope_for_source,
                &k_context_for_source,
                &v_context_for_source,
                &expanded_k_for_source,
                &expanded_v_for_source,
                &scores_for_source,
                &probabilities_for_source,
                &output_heads_for_source,
                &output_projection.projection_input,
                output_projection.sub_layernorm_output.as_ref(),
                &output_projection.output,
            );
        }
        qwen_attention_trace_event(trace_attention, "attention.output_projection_finish", || {
            format!(
                "\"layer\":{},\"output_projection_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(output_projection_start)
            )
        });
        qwen_attention_trace_event(trace_attention, "attention.forward_finish", || {
            format!(
                "\"layer\":{},\"attention_ms\":{}",
                self.layer_idx,
                qwen_attention_elapsed_ms(attention_start)
            )
        });
        Ok(output_projection.output)
    }

    fn project_qkv(
        &self,
        x: &Tensor,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        mut workspace: Option<&mut TransformerForwardWorkspace>,
    ) -> Result<QkvProjections> {
        // PATCH 3: Project to Q, K, V separately (NOT fused QKV)
        // This is the correct implementation - separate projections ensure proper shape handling
        // Q: [B, T, hidden] -> [B, T, n_heads * head_dim] -> [B, n_heads, T, head_dim]
        // K: [B, T, hidden] -> [B, T, n_kv_heads * head_dim] -> [B, n_kv_heads, T, head_dim]
        // V: [B, T, hidden] -> [B, T, n_kv_heads * head_dim] -> [B, n_kv_heads, T, head_dim]
        let q = self.apply_linear_with_qkv_projection_source(
            x,
            &self.q_proj,
            "q_proj",
            raw_tensors,
            dense_linear_hooks,
            workspace.as_deref_mut(),
        )?;
        let k = self.apply_linear_with_qkv_projection_source(
            x,
            &self.k_proj,
            "k_proj",
            raw_tensors,
            dense_linear_hooks,
            workspace.as_deref_mut(),
        )?;
        let v = self.apply_linear_with_qkv_projection_source(
            x,
            &self.v_proj,
            "v_proj",
            raw_tensors,
            dense_linear_hooks,
            workspace,
        )?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.q_proj", Some(self.layer_idx), &q)?;
            qwen_trace_tensor("attention.k_proj", Some(self.layer_idx), &k)?;
            qwen_trace_tensor("attention.v_proj", Some(self.layer_idx), &v)?;
        }
        Ok(QkvProjections { q, k, v })
    }

    fn apply_linear_with_qkv_projection_source(
        &self,
        input: &Tensor,
        linear: &candle_nn::Linear,
        proj_name: &str,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        workspace: Option<&mut TransformerForwardWorkspace>,
    ) -> Result<Tensor> {
        let qk256_key =
            format!("layers.{}.attention.{}.weight.qk256_qs", self.layer_idx, proj_name);
        let tensor_name = format!("layers.{}.attention.{}.weight", self.layer_idx, proj_name);
        let qk256_raw_tensor = raw_tensors.get(&qk256_key);
        let qk256_raw_tensor_present = qk256_raw_tensor.is_some();
        let source_input = workspace.as_ref().map(|_| input.clone());
        let dispatch_before = bitnet_qk256_dispatch::qk256_dispatch_coverage();
        let cpu_hot_path_before = bitnet_qk256_dispatch::qk256_cpu_hot_path_counters();
        let a770_runtime_before = bitnet_qk256_dispatch::qk256_a770_opencl_runtime_stats();

        let output =
            self.apply_linear(input, linear, proj_name, raw_tensors, dense_linear_hooks)?;
        maybe_trace_dense_q8_source_order_qproj_candidate(
            input,
            &output,
            linear,
            &tensor_name,
            dense_linear_hooks,
            self.layer_idx,
        )?;

        if let (Some(workspace), Some(source_input)) = (workspace, source_input) {
            let dispatch_after = bitnet_qk256_dispatch::qk256_dispatch_coverage();
            let cpu_hot_path_after = bitnet_qk256_dispatch::qk256_cpu_hot_path_counters();
            let a770_runtime_after = bitnet_qk256_dispatch::qk256_a770_opencl_runtime_stats();
            let (dispatch_replay, dispatch_replay_error) =
                if qkv_projection_dispatch_replay_enabled(self.layer_idx, proj_name) {
                    match qk256_raw_tensor {
                        Some(qk256_tensor) => match qk256_inline_scale(raw_tensors, &qk256_key)
                            .and_then(|inline_scale| {
                                bitnet_qk256_dispatch::replay_qk256_cpu_vs_a770_with_scale(
                                    &source_input,
                                    qk256_tensor,
                                    &qk256_key,
                                    inline_scale,
                                )
                            }) {
                            Ok(replay) => (Some(transformer_dispatch_replay_tensors(replay)), None),
                            Err(err) => (None, Some(err.to_string())),
                        },
                        None => (None, Some(format!("qk256 raw tensor {qk256_key} missing"))),
                    }
                } else {
                    (None, None)
                };
            workspace.record_qkv_projection_source_tensors(TransformerQkvProjectionSourceTensors {
                layer_idx: self.layer_idx,
                projection: proj_name.to_string(),
                tensor_name,
                qk256_key,
                qk256_raw_tensor_present,
                input: source_input,
                output: output.clone(),
                dispatch_delta: qk256_dispatch_delta_for_projection(
                    &dispatch_before,
                    &dispatch_after,
                ),
                cpu_hot_path_delta: qk256_cpu_hot_path_delta_for_projection(
                    &cpu_hot_path_before,
                    &cpu_hot_path_after,
                ),
                a770_opencl_runtime_delta: qk256_a770_runtime_delta_for_projection(
                    &a770_runtime_before,
                    &a770_runtime_after,
                ),
                dispatch_replay,
                dispatch_replay_error,
            });
        }

        Ok(output)
    }

    fn trace_projection_rms_once(&self, projections: &QkvProjections) -> Result<()> {
        // Probe A3: Q/K/V projection RMS (layer 0, step 0 only)
        if trace_rms_enabled() && self.layer_idx == 0 {
            static PROJ_LOGGED: std::sync::Once = std::sync::Once::new();
            PROJ_LOGGED.call_once(|| {
                let _ = (|| -> candle_core::Result<()> {
                    let q_vec = projections.q.flatten_all()?.to_vec1::<f32>()?;
                    let q_rms = (q_vec.iter().map(|x| x * x).sum::<f32>()
                        / q_vec.len().max(1) as f32)
                        .sqrt();
                    let k_vec = projections.k.flatten_all()?.to_vec1::<f32>()?;
                    let k_rms = (k_vec.iter().map(|x| x * x).sum::<f32>()
                        / k_vec.len().max(1) as f32)
                        .sqrt();
                    let v_vec = projections.v.flatten_all()?.to_vec1::<f32>()?;
                    let v_rms = (v_vec.iter().map(|x| x * x).sum::<f32>()
                        / v_vec.len().max(1) as f32)
                        .sqrt();
                    eprintln!(
                        "trace: q_proj_rms={:.6} k_proj_rms={:.6} v_proj_rms={:.6}",
                        q_rms, k_rms, v_rms
                    );
                    Ok(())
                })();
            });
        }
        Ok(())
    }

    fn trace_q_projection(&self, _q_proj_out: &Tensor) -> Result<()> {
        // Tracepoint 3: Q projection output (layer-specific)
        #[cfg(feature = "trace")]
        {
            let trace_name = format!("t0/blk{}/q_proj", self.layer_idx);
            bitnet_trace::dump_trace(
                &trace_name,
                _q_proj_out,
                Some(0),
                Some(self.layer_idx as isize),
                Some("q_proj"),
            )
            .map_err(BitNetError::from)?;
        }
        Ok(())
    }

    fn trace_qproj_output_pre_optional_qnorm(&self, q_proj_out: &Tensor) -> Result<()> {
        let source_tensor = format!("layers.{}.attention.q_proj.weight", self.layer_idx);
        let gguf_tensor = format!("blk.{}.attn_q.weight", self.layer_idx);
        let dense_hook_identity = format!(
            "{}:{}:runtime_disabled",
            source_tensor, QWEN_QPROJ_OUTPUT_PRE_OPTIONAL_QNORM_BOUNDARY
        );
        qwen_trace_tensor_fingerprint_with_dense_hook(
            QWEN_QPROJ_OUTPUT_PRE_OPTIONAL_QNORM_STAGE,
            Some(self.layer_idx),
            q_proj_out,
            &source_tensor,
            QWEN_QPROJ_OUTPUT_PRE_OPTIONAL_QNORM_BOUNDARY,
            QwenTraceDenseHookIdentity {
                dense_hook_identity: &dense_hook_identity,
                gguf_tensor: &gguf_tensor,
                runtime_disabled: true,
            },
        )?;
        Ok(())
    }

    fn reshape_qkv_heads(
        &self,
        projections: QkvProjections,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<AttentionHeads> {
        let q = projections
            .q
            .reshape(&[batch_size, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, Hq, T, D]

        let k = projections
            .k
            .reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, HKV, T, D]

        let v = projections
            .v
            .reshape(&[batch_size, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, HKV, T, D]

        // Debug Q, K, V projections
        dbg_stats("Q", &q)?;
        dbg_stats("K", &k)?;
        dbg_stats("V", &v)?;

        Ok(AttentionHeads { q, k, v })
    }

    fn apply_qk_norms(&self, heads: AttentionHeads) -> Result<AttentionHeads> {
        let q = if let Some(norm) = &self.q_norm {
            if self.layer_idx == 0 {
                qwen_trace_tensor_fingerprint(
                    "attention.q_norm_input",
                    Some(self.layer_idx),
                    &heads.q,
                    "layers.0.attention.q_proj.weight",
                    "q_norm_input_candle_tensor_boundary",
                )?;
            }
            let normalized = norm.forward(&heads.q)?;
            if qwen_trace_layer_enabled(self.layer_idx) {
                qwen_trace_tensor("attention.q_norm", Some(self.layer_idx), &normalized)?;
            }
            normalized
        } else {
            heads.q
        };

        let k = if let Some(norm) = &self.k_norm {
            let normalized = norm.forward(&heads.k)?;
            if qwen_trace_layer_enabled(self.layer_idx) {
                qwen_trace_tensor("attention.k_norm", Some(self.layer_idx), &normalized)?;
            }
            normalized
        } else {
            heads.k
        };

        Ok(AttentionHeads { q, k, v: heads.v })
    }

    fn log_gqa_shapes_once(&self, heads: &AttentionHeads) -> Result<()> {
        // GQA diagnostic: log Q/K/V dimensions and norms (once per run)
        if debug_gqa_enabled() {
            static GQA_LOGGED: std::sync::Once = std::sync::Once::new();
            GQA_LOGGED.call_once(|| {
                let q_dims = heads.q.dims();
                let k_dims = heads.k.dims();
                let v_dims = heads.v.dims();
                if let (Ok(q_mean), Ok(k_mean), Ok(v_mean)) = (
                    heads.q.mean_all().and_then(|m| m.to_scalar::<f32>()),
                    heads.k.mean_all().and_then(|m| m.to_scalar::<f32>()),
                    heads.v.mean_all().and_then(|m| m.to_scalar::<f32>()),
                ) {
                    tracing::info!(
                        "GQA shapes - Q: {:?} (mean {:.3}), K: {:?} (mean {:.3}), V: {:?} (mean {:.3})",
                        q_dims, q_mean, k_dims, k_mean, v_dims, v_mean
                    );
                    tracing::info!(
                        "GQA config - n_heads={}, n_kv_heads={}, head_dim={}, group_size={}",
                        self.n_heads, self.n_kv_heads, self.head_dim, self.group_size
                    );
                }
            });
        }
        Ok(())
    }

    fn apply_rotary_embeddings(
        &self,
        heads: AttentionHeads,
        cache_seq_len: Option<usize>,
    ) -> Result<AttentionHeads> {
        // Apply rotary embeddings if available (need to handle different K/V head counts)
        let Some(rope) = &self.rope else {
            return Ok(heads);
        };

        let position = cache_seq_len.unwrap_or(0);

        // Log ROPE application details (once)
        if debug_rope_enabled() {
            static ROPE_LOGGED: std::sync::Once = std::sync::Once::new();
            ROPE_LOGGED.call_once(|| {
                tracing::info!(
                    "ROPE applied: position={}, q_shape={:?}, k_shape={:?}, head_dim={}",
                    position,
                    heads.q.dims(),
                    heads.k.dims(),
                    self.head_dim
                );
            });
        }

        let q = rope.apply(&heads.q, position)?;
        let k = rope.apply(&heads.k, position)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_event(
                "attention.rope_metadata",
                &format!(
                    "\"layer\":{},\"position\":{},\"head_dim\":{},\"n_heads\":{},\"n_kv_heads\":{}",
                    self.layer_idx, position, self.head_dim, self.n_heads, self.n_kv_heads
                ),
            );
            qwen_trace_tensor("attention.q_rope", Some(self.layer_idx), &q)?;
            qwen_trace_tensor("attention.k_rope", Some(self.layer_idx), &k)?;
        }

        Ok(AttentionHeads { q, k, v: heads.v })
    }

    fn kv_context<'a>(
        k: &'a Tensor,
        v: &'a Tensor,
        kv_cache: Option<&'a mut LayerKVCache>,
    ) -> Result<(&'a Tensor, &'a Tensor)> {
        // Update KV cache if provided (store HKV heads, not Hq)
        // **Performance note**: Borrow references instead of cloning after append.
        // Candle operations accept both owned and borrowed tensors.
        if let Some(cache) = kv_cache {
            cache.append(k, v)?;
            // Borrow from cache - avoids cloning full KV history
            Ok((&cache.k, &cache.v))
        } else {
            // No cache: use freshly computed K/V from this step
            Ok((k, v))
        }
    }

    fn expand_grouped_query_kv(
        &self,
        k_ctx: &Tensor,
        v_ctx: &Tensor,
        batch_size: usize,
    ) -> Result<ExpandedKv> {
        // GQA core: expand K/V to Hq heads (repeat along head axis)
        // We want K,V of shape [B,Hq,Tk,D]. Repeat every KV head group_size times.
        let t_k = k_ctx.dims()[2];

        // Expand K: [B, HKV, Tk, D] -> [B, Hq, Tk, D]
        let k = k_ctx
            .unsqueeze(2)? // [B, HKV, 1, Tk, D]
            .repeat(&[1, 1, self.group_size, 1, 1])? // [B, HKV, group, Tk, D]
            .reshape(&[batch_size, self.n_heads, t_k, self.head_dim])?; // [B, Hq, Tk, D]

        // Expand V: [B, HKV, Tk, D] -> [B, Hq, Tk, D]
        let v = v_ctx
            .unsqueeze(2)? // [B, HKV, 1, Tk, D]
            .repeat(&[1, 1, self.group_size, 1, 1])? // [B, HKV, group, Tk, D]
            .reshape(&[batch_size, self.n_heads, t_k, self.head_dim])?; // [B, Hq, Tk, D]

        Ok(ExpandedKv { k, v })
    }

    fn prepare_attention_scores(
        &self,
        q: &Tensor,
        k_expanded: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Scaled dot-product attention with explicit fp32 handling
        // For head_dim=128, scale = 1/sqrt(128) ≈ 0.0883883
        let scale_factor = (self.head_dim as f32).sqrt().recip();

        // Log scale computation once
        if debug_attn_scale_enabled() {
            static SCALE_LOGGED: std::sync::Once = std::sync::Once::new();
            SCALE_LOGGED.call_once(|| {
                tracing::info!(
                    "Attention scale: head_dim={}, scale_factor=1/sqrt({})={:.7}",
                    self.head_dim,
                    self.head_dim,
                    scale_factor
                );
            });
        }

        let q_for_scores = attention_f16_dot_input(q)?;
        let k_for_scores = attention_score_key_input(k_expanded)?;
        let scores = q_for_scores.matmul(&k_for_scores.transpose(2, 3)?)?;

        // Convert to fp32 for numerically stable computation
        let scores_f32 = scores.to_dtype(DType::F32)?;

        // Scale in fp32
        let scores_f32 = scores_f32.affine(scale_factor as f64, 0.0)?;

        // Debug scores before mask
        dbg_stats("scores pre-mask", &scores_f32)?;
        dbg_finite("scores pre-mask", &scores_f32)?;

        // Apply causal mask so queries cannot attend to future positions.
        // When using a KV cache, k includes past tokens, so the mask must
        // account for the total key length. Single-token decode has no future
        // key positions, so the mask is all zeros and can be skipped.
        let total_len = k_expanded.dims()[2];
        let scores_f32 = if seq_len == 1 {
            scores_f32
        } else {
            // PATCH 5: create_causal_mask now returns [1, 1, Tq, Tk] directly - no need for unsqueeze
            let mask = self.create_causal_mask(seq_len, total_len, scores_f32.device())?;
            scores_f32.broadcast_add(&mask)?
        };
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.scores_post_mask", Some(self.layer_idx), &scores_f32)?;
        }

        // Debug scores after mask and before softmax (critical diagnostics)
        dbg_stats("scores post-mask", &scores_f32)?;
        dbg_finite("scores post-mask", &scores_f32)?;

        // Log scores range after mask for layer 0 (user's diagnostic request)
        if debug_attn_scale_enabled() {
            static LAYER_LOGGED: std::sync::Once = std::sync::Once::new();
            LAYER_LOGGED.call_once(|| {
                if let Ok(flat) = scores_f32.flatten_all()
                    && let Ok(vals) = flat.to_vec1::<f32>()
                    && let (Some(&min_val), Some(&max_val)) = (
                        vals.iter().filter(|v| v.is_finite()).min_by(|a, b| a.total_cmp(b)),
                        vals.iter().filter(|v| v.is_finite()).max_by(|a, b| a.total_cmp(b)),
                    )
                {
                    tracing::info!(
                        "Layer 0 scores post-mask range: min={:.6}, max={:.6}",
                        min_val,
                        max_val
                    );
                }
            });
        }

        Ok(scores_f32)
    }

    fn softmax_attention_scores(&self, scores_f32: &Tensor) -> Result<Tensor> {
        // PATCH 4: Softmax path verification
        // Apply max-subtraction for numerical stability before softmax
        // Compute row-wise max and subtract for stability (explicit max-subtraction)
        // VERIFIED: axis=3 is correct for [B, H, Tq, Tk] layout - normalizes across keys (Tk)
        let row_max = scores_f32.max_keepdim(3)?;
        let scores_stabilized = scores_f32.broadcast_sub(&row_max)?;

        // Log that max-subtraction ran (user's diagnostic request)
        if debug_attn_scale_enabled() {
            static MAX_SUB_LOGGED: std::sync::Once = std::sync::Once::new();
            MAX_SUB_LOGGED.call_once(|| {
                tracing::info!("Attention: max-subtraction applied for numerical stability");
            });
        }

        // Apply softmax (exp then normalize)
        // VERIFIED: axis=3 is correct - softmax over keys (Tk dimension) in [B, H, Tq, Tk]
        let attn_weights = candle_nn::ops::softmax(&scores_stabilized, 3)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.weights", Some(self.layer_idx), &attn_weights)?;
        }

        // Tracepoint 4: Attention scores post-softmax (layer-specific)
        #[cfg(feature = "trace")]
        {
            let trace_name = format!("t0/blk{}/attn_scores_softmax", self.layer_idx);
            bitnet_trace::dump_trace(
                &trace_name,
                &attn_weights,
                Some(0),
                Some(self.layer_idx as isize),
                Some("attn_scores_softmax"),
            )
            .map_err(BitNetError::from)?;
        }

        // Debug attention weights and row sums
        dbg_stats("attn softmax", &attn_weights)?;
        if debug_attn_enabled() {
            let sums = attn_weights.sum(3)?;
            let sums_host: Vec<f32> = sums.flatten_all()?.to_vec1()?;
            let take = sums_host.iter().take(4).cloned().collect::<Vec<_>>();
            eprintln!("[dbg] attn row-sums (first 4): {:?}", take);
        }

        Ok(attn_weights)
    }

    fn apply_attention_weights(
        &self,
        attn_weights: &Tensor,
        v_expanded: &Tensor,
    ) -> Result<Tensor> {
        let v_for_value_mix = attention_f16_dot_input(v_expanded)?;
        let attn_output = attn_weights.matmul(&v_for_value_mix)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.output_heads", Some(self.layer_idx), &attn_output)?;
        }
        Ok(attn_output)
    }

    fn project_attention_output(
        &self,
        attn_output: Tensor,
        batch_size: usize,
        seq_len: usize,
        raw_tensors: &std::collections::HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<AttentionOutputProjection> {
        // Reshape and project output
        let projection_input = attn_output.transpose(1, 2)?.reshape(&[
            batch_size,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;
        let sub_layernorm_output = if let Some(sub_layernorm) = &self.sub_layernorm {
            let normalized = sub_layernorm.forward(&projection_input)?;
            if qwen_trace_layer_enabled(self.layer_idx) {
                qwen_trace_tensor("attention.sub_layernorm", Some(self.layer_idx), &normalized)?;
            }
            Some(normalized)
        } else {
            None
        };
        let projected_input = sub_layernorm_output.as_ref().unwrap_or(&projection_input);

        let projected = self.apply_linear(
            projected_input,
            &self.o_proj,
            "o_proj",
            raw_tensors,
            dense_linear_hooks,
        )?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("attention.o_proj", Some(self.layer_idx), &projected)?;
        }
        Ok(AttentionOutputProjection { projection_input, sub_layernorm_output, output: projected })
    }
}

fn qwen_attention_trace_event(enabled: bool, stage: &str, fields_json: impl FnOnce() -> String) {
    if enabled {
        qwen_trace_event(stage, &fields_json());
    }
}

fn qwen_attention_elapsed_ms(start: Instant) -> String {
    qwen_trace_number(start.elapsed().as_secs_f64() * 1000.0)
}

fn qkv_projection_dispatch_replay_enabled(layer_idx: usize, projection: &str) -> bool {
    if !env_truthy("BITNET_QKV_PROJECTION_DISPATCH_REPLAY") {
        return false;
    }
    if let Ok(layer) = std::env::var("BITNET_QKV_PROJECTION_DISPATCH_REPLAY_LAYER")
        && layer.parse::<usize>().ok() != Some(layer_idx)
    {
        return false;
    }
    if let Ok(filter) = std::env::var("BITNET_QKV_PROJECTION_DISPATCH_REPLAY_PROJECTION")
        && filter != projection
    {
        return false;
    }
    true
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn transformer_dispatch_replay_tensors(
    replay: bitnet_qk256_dispatch::Qk256CpuA770DispatchReplay,
) -> TransformerQkvProjectionDispatchReplayTensors {
    TransformerQkvProjectionDispatchReplayTensors {
        input_rows: replay.input_rows,
        output_rows: replay.output_rows,
        cols: replay.cols,
        row_stride_bytes: replay.row_stride_bytes,
        inline_scale: replay.inline_scale,
        cpu_output: replay.cpu_output,
        opencl_policy_output: replay.opencl_policy_output,
        a770_output: replay.a770_output,
        device_expression_trace: replay.device_expression_trace.map(|trace| {
            TransformerQk256DeviceExpressionTrace {
                input_row_index: trace.input_row_index,
                sample_limit: trace.sample_limit,
                sample_count: trace.sample_count,
                samples: trace
                    .samples
                    .into_iter()
                    .map(|sample| TransformerQk256DeviceExpressionSample {
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
            }
        }),
        device_intermediate_trace: replay.device_intermediate_trace.map(|trace| {
            TransformerQk256DeviceIntermediateTrace {
                compiled_opencl: trace.compiled_opencl,
                attempted: trace.attempted,
                success: trace.success,
                error: trace.error,
                input_row_index: trace.input_row_index,
                sample_limit: trace.sample_limit,
                sample_count: trace.sample_count,
                platform_index: trace.platform_index,
                device_index: trace.device_index,
                platform_name: trace.platform_name,
                runtime_device: trace.runtime_device,
                vendor: trace.vendor,
                driver_version: trace.driver_version,
                host_to_device_bytes: trace.host_to_device_bytes,
                device_to_host_bytes: trace.device_to_host_bytes,
                kernel_invocations: trace.kernel_invocations,
                samples: trace
                    .samples
                    .into_iter()
                    .map(|sample| TransformerQk256DeviceIntermediateSample {
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
                        volatile_div_then_mul_bits: sample.volatile_div_then_mul_bits,
                        volatile_div_then_mul: sample.volatile_div_then_mul,
                    })
                    .collect(),
            }
        }),
        focused_operands: replay.focused_operands.map(|operands| {
            TransformerQk256FocusedRawOperands {
                input_row_index: operands.input_row_index,
                output_index: operands.output_index,
                cols: operands.cols,
                row_stride_bytes: operands.row_stride_bytes,
                packed_qk256_scope: operands.packed_qk256_scope.to_string(),
                activation_sum: operands.activation_sum,
                activation_scale_bits: operands.activation_scale_bits,
                weight_scale_bits: operands.weight_scale_bits,
                activations_i8: operands.activations_i8,
                packed_qk256: operands.packed_qk256,
            }
        }),
        full_projection_operands: replay.full_projection_operands.map(|operands| {
            TransformerQk256FullProjectionRawOperands {
                input_row_index: operands.input_row_index,
                rows: operands.rows,
                cols: operands.cols,
                row_stride_bytes: operands.row_stride_bytes,
                packed_qk256_scope: operands.packed_qk256_scope.to_string(),
                activation_sum: operands.activation_sum,
                activation_scale_bits: operands.activation_scale_bits,
                weight_scale_bits: operands.weight_scale_bits,
                activations_i8: operands.activations_i8,
                packed_qk256: operands.packed_qk256,
            }
        }),
        cpu: TransformerQkvProjectionDispatchReplayCpuStats {
            scalar_invocations: replay.cpu.scalar_invocations,
            execution_path: replay.cpu.execution_path.to_string(),
        },
        a770: TransformerQkvProjectionDispatchReplayA770Stats {
            compiled_opencl: replay.a770.compiled_opencl,
            attempted: replay.a770.attempted,
            success: replay.a770.success,
            host_to_device_bytes: replay.a770.host_to_device_bytes,
            device_to_host_bytes: replay.a770.device_to_host_bytes,
            kernel_invocations: replay.a770.kernel_invocations,
            last_device: replay.a770.last_device.map(|device| TransformerA770OpenClRuntimeDevice {
                platform_index: device.platform_index,
                device_index: device.device_index,
                platform_name: device.platform_name,
                runtime_device: device.runtime_device,
                vendor: device.vendor,
                driver_version: device.driver_version,
            }),
            error: replay.a770.error,
            execution_path: replay.a770.execution_path.to_string(),
        },
    }
}

fn qk256_dispatch_delta_for_projection(
    before: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
    after: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
) -> TransformerQk256DispatchDelta {
    let unsupported_before =
        before.unsupported_ops.iter().cloned().collect::<std::collections::BTreeSet<_>>();
    let unsupported_after =
        after.unsupported_ops.iter().cloned().collect::<std::collections::BTreeSet<_>>();
    let bitnet_linear_layers_total =
        after.bitnet_linear_layers_total.saturating_sub(before.bitnet_linear_layers_total);
    let bitnet_linear_layers_on_cuda =
        after.bitnet_linear_layers_on_cuda.saturating_sub(before.bitnet_linear_layers_on_cuda);
    let bitnet_linear_layers_on_a770_opencl = after
        .bitnet_linear_layers_on_a770_opencl
        .saturating_sub(before.bitnet_linear_layers_on_a770_opencl);
    let bitnet_linear_layers_cpu_fallback = after
        .bitnet_linear_layers_cpu_fallback
        .saturating_sub(before.bitnet_linear_layers_cpu_fallback);
    let execution_claim = if bitnet_linear_layers_on_a770_opencl > 0 {
        "a770_opencl_qk256_contribution"
    } else if bitnet_linear_layers_on_cuda > 0 {
        "cuda_qk256_contribution"
    } else if bitnet_linear_layers_cpu_fallback > 0 {
        "cpu_fallback"
    } else if bitnet_linear_layers_total > 0 {
        "cpu_qk256_reference"
    } else {
        "dense_f32_candle_linear"
    };

    TransformerQk256DispatchDelta {
        bitnet_linear_layers_total,
        bitnet_linear_layers_on_cuda,
        bitnet_linear_layers_on_a770_opencl,
        bitnet_linear_layers_cpu_fallback,
        unsupported_ops: unsupported_after.difference(&unsupported_before).cloned().collect(),
        execution_claim: execution_claim.to_string(),
    }
}

fn qk256_cpu_hot_path_delta_for_projection(
    before: &bitnet_qk256_dispatch::Qk256CpuHotPathCounters,
    after: &bitnet_qk256_dispatch::Qk256CpuHotPathCounters,
) -> TransformerQk256CpuHotPathDelta {
    TransformerQk256CpuHotPathDelta {
        qk256_f32_scalar_gemv_invocations: after
            .qk256_f32_scalar_gemv_invocations
            .saturating_sub(before.qk256_f32_scalar_gemv_invocations),
        qk256_f32_avx2_gemv_invocations: after
            .qk256_f32_avx2_gemv_invocations
            .saturating_sub(before.qk256_f32_avx2_gemv_invocations),
        qk256_i8s_scaled_scalar_invocations: after
            .qk256_i8s_scaled_scalar_invocations
            .saturating_sub(before.qk256_i8s_scaled_scalar_invocations),
        qk256_i8s_scaled_avx2_invocations: after
            .qk256_i8s_scaled_avx2_invocations
            .saturating_sub(before.qk256_i8s_scaled_avx2_invocations),
        qk256_flat_bytes_extracted_count: after
            .qk256_flat_bytes_extracted_count
            .saturating_sub(before.qk256_flat_bytes_extracted_count),
        input_rows_materialized_count: after
            .input_rows_materialized_count
            .saturating_sub(before.input_rows_materialized_count),
        output_rows_allocated_count: after
            .output_rows_allocated_count
            .saturating_sub(before.output_rows_allocated_count),
        requested_kernel: after.requested_kernel.clone(),
        selected_kernel: after.selected_kernel.clone(),
        qk256_execution_path: after.qk256_execution_path.to_string(),
    }
}

fn qk256_a770_runtime_delta_for_projection(
    before: &bitnet_qk256_dispatch::Qk256A770OpenClRuntimeStats,
    after: &bitnet_qk256_dispatch::Qk256A770OpenClRuntimeStats,
) -> TransformerA770OpenClRuntimeDelta {
    TransformerA770OpenClRuntimeDelta {
        host_to_device_bytes: after
            .host_to_device_bytes
            .saturating_sub(before.host_to_device_bytes),
        device_to_host_bytes: after
            .device_to_host_bytes
            .saturating_sub(before.device_to_host_bytes),
        kernel_invocations: after.kernel_invocations.saturating_sub(before.kernel_invocations),
    }
}
