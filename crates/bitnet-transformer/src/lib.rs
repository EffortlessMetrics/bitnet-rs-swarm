#[cfg(test)]
use bitnet_common::config::NormType;
#[cfg(test)]
use bitnet_common::dtype_convert::f32_to_fp16;
use bitnet_common::dtype_convert::fp16_to_f32;
use bitnet_common::{BitNetConfig, BitNetError, Result, config::ActivationType};
use bitnet_qk256_dispatch::{
    forward_qk256_with_scale, record_bitnet_linear_cpu_fallback, record_bitnet_linear_unsupported,
    strict_cuda_bitnet_backend_requested,
};
use bitnet_rope::{build_tables as build_rope_tables, resolve_base as resolve_rope_base};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
mod attention_forward;

mod diagnostics;
mod layer_builders;
mod qk256;

use diagnostics::{
    QWEN_QPROJ_OUTPUT_PRE_OPTIONAL_QNORM_BOUNDARY, QWEN_QPROJ_OUTPUT_PRE_OPTIONAL_QNORM_STAGE,
    QWEN_QPROJ_SOURCE_ORDER_Q8_ACCUMULATOR_AUDIT_BOUNDARY,
    QWEN_QPROJ_SOURCE_ORDER_Q8_ACCUMULATOR_AUDIT_STAGE,
    QWEN_QPROJ_SOURCE_ORDER_Q8_CANDIDATE_BOUNDARY, QWEN_QPROJ_SOURCE_ORDER_Q8_CANDIDATE_STAGE,
    QWEN_QPROJ_SOURCE_ORDER_Q8_CANDLE_SLICE_COMPARE_BOUNDARY,
    QWEN_QPROJ_SOURCE_ORDER_Q8_CANDLE_SLICE_COMPARE_STAGE,
    QWEN_QPROJ_SOURCE_ORDER_Q8_ROW_MAPPING_PROOF_BOUNDARY,
    QWEN_QPROJ_SOURCE_ORDER_Q8_ROW_MAPPING_PROOF_STAGE, QwenTraceDenseHookIdentity,
    QwenTraceSourceOrderQ8AccumulatorAudit, QwenTraceSourceOrderQ8AccumulatorAuditEntry,
    QwenTraceSourceOrderQ8Candidate, QwenTraceSourceOrderQ8CandleSliceCompare,
    QwenTraceSourceOrderQ8CandleSliceCompareEntry, QwenTraceSourceOrderQ8RowMappingProof,
    QwenTraceSourceOrderQ8RowMappingProofEntry, dbg_finite, dbg_stats, debug_attn_enabled,
    debug_attn_scale_enabled, debug_gqa_enabled, debug_mlp_enabled, debug_rmsnorm_enabled,
    debug_rope_enabled, qwen_trace_event, qwen_trace_events_enabled, qwen_trace_layer_enabled,
    qwen_trace_number, qwen_trace_source_order_q8_accumulator_audit,
    qwen_trace_source_order_q8_candidate, qwen_trace_source_order_q8_candle_slice_compare,
    qwen_trace_source_order_q8_row_mapping_proof, qwen_trace_tensor, qwen_trace_tensor_fingerprint,
    qwen_trace_tensor_fingerprint_with_dense_hook, trace_rms_enabled,
};
#[cfg(test)]
use layer_builders::layer_norm_with_optional_bias;
#[cfg(test)]
use layer_builders::linear_with_optional_bias;
use layer_builders::{norm_with_optional_bias, optional_layer_norm_with_optional_bias};
use qk256::{TIED_EMBED_QK256_KEY, qk256_inline_scale};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub type DenseLinearRuntimeHookRegistry = HashMap<String, DenseLinearRuntimeHookDescriptor>;
const SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR: &str = "layers.0.attention.q_proj.weight";

fn qwen_trace_device_kind(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}

fn qwen_trace_elapsed_ms(start: Instant) -> String {
    qwen_trace_number(start.elapsed().as_secs_f64() * 1000.0)
}

fn qwen_trace_model_init_event(enabled: bool, stage: &str, fields_json: impl FnOnce() -> String) {
    if enabled {
        qwen_trace_event(stage, &fields_json());
    }
}

fn qwen_trace_runtime_event(enabled: bool, stage: &str, fields_json: impl FnOnce() -> String) {
    if enabled {
        qwen_trace_event(stage, &fields_json());
    }
}

fn qwen_trace_dims_json(dims: &[usize]) -> String {
    dims.iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join(",")
}

fn qwen_trace_rms_norm_fused(
    norm_input: &Tensor,
    output_dtype: DType,
    weight: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let weight = if weight.dtype() == norm_input.dtype() {
        weight.clone()
    } else {
        weight.to_dtype(norm_input.dtype())?
    };
    let output = candle_nn::ops::rms_norm(norm_input, &weight, eps as f32)?;
    Ok(output.to_dtype(output_dtype)?)
}

struct LinearInitTrace<'a> {
    enabled: bool,
    init_start: Instant,
    layer_idx: Option<usize>,
    device: &'a Device,
    scope: &'static str,
    name: &'static str,
}

#[derive(Clone, Copy)]
struct RopeInitTrace<'a> {
    enabled: bool,
    init_start: Instant,
    layer_idx: usize,
    device: &'a Device,
}

fn qwen_trace_linear_init_event(
    trace: &LinearInitTrace<'_>,
    stage: &str,
    fields_json: impl FnOnce() -> String,
) {
    qwen_trace_model_init_event(trace.enabled, stage, || {
        let layer = trace.layer_idx.map_or_else(|| "null".to_string(), |layer| layer.to_string());
        format!(
            "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\",\"scope\":\"{}\",\"linear\":\"{}\",{}",
            qwen_trace_elapsed_ms(trace.init_start),
            layer,
            qwen_trace_device_kind(trace.device),
            trace.scope,
            trace.name,
            fields_json()
        )
    });
}

fn linear_with_optional_bias_traced(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    trace: LinearInitTrace<'_>,
) -> Result<Linear> {
    qwen_trace_linear_init_event(&trace, "model_init.linear_start", || {
        format!("\"in_dim\":{},\"out_dim\":{}", in_dim, out_dim)
    });

    let weight_start = Instant::now();
    qwen_trace_linear_init_event(&trace, "model_init.linear_weight_start", || {
        format!("\"in_dim\":{},\"out_dim\":{}", in_dim, out_dim)
    });
    let weight = match vb.get((out_dim, in_dim), "weight") {
        Ok(weight) => {
            qwen_trace_linear_init_event(&trace, "model_init.linear_weight_finish", || {
                format!(
                    "\"weight_ms\":{},\"dtype\":\"{:?}\",\"dims\":[{}]",
                    qwen_trace_elapsed_ms(weight_start),
                    weight.dtype(),
                    qwen_trace_dims_json(weight.dims())
                )
            });
            weight
        }
        Err(err) => {
            qwen_trace_linear_init_event(&trace, "model_init.linear_weight_error", || {
                format!("\"weight_ms\":{}", qwen_trace_elapsed_ms(weight_start))
            });
            return Err(BitNetError::from(err));
        }
    };

    let bias_start = Instant::now();
    qwen_trace_linear_init_event(&trace, "model_init.linear_bias_start", || {
        format!("\"out_dim\":{}", out_dim)
    });
    let bias = match vb.get(out_dim, "bias") {
        Ok(bias) => {
            qwen_trace_linear_init_event(&trace, "model_init.linear_bias_finish", || {
                format!(
                    "\"bias_ms\":{},\"present\":true,\"dtype\":\"{:?}\",\"dims\":[{}]",
                    qwen_trace_elapsed_ms(bias_start),
                    bias.dtype(),
                    qwen_trace_dims_json(bias.dims())
                )
            });
            Some(bias)
        }
        Err(_) => {
            tracing::debug!("Bias tensor missing for linear layer; using no-bias path [{out_dim}]");
            qwen_trace_linear_init_event(&trace, "model_init.linear_bias_finish", || {
                format!("\"bias_ms\":{},\"present\":false", qwen_trace_elapsed_ms(bias_start))
            });
            None
        }
    };

    qwen_trace_linear_init_event(&trace, "model_init.linear_finish", || {
        format!("\"in_dim\":{},\"out_dim\":{}", in_dim, out_dim)
    });

    Ok(Linear::new(weight, bias))
}

fn qwen_trace_rope_init_event(
    trace: Option<RopeInitTrace<'_>>,
    stage: &str,
    fields_json: impl FnOnce() -> String,
) {
    if let Some(trace) = trace {
        qwen_trace_model_init_event(trace.enabled, stage, || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\",{}",
                qwen_trace_elapsed_ms(trace.init_start),
                trace.layer_idx,
                qwen_trace_device_kind(trace.device),
                fields_json()
            )
        });
    }
}

fn rope_table_device_for_target(device: &Device) -> Device {
    match device {
        Device::Cuda(_) => Device::Cpu,
        _ => device.clone(),
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DenseQ8SidecarInstrumentationSnapshot {
    pub selector_dispatch_calls: u64,
    pub selector_selected_calls: u64,
    pub selector_declined_calls: u64,
    pub selector_error_calls: u64,
    pub selector_dispatch_ns: u64,
    pub input_materialization_calls: u64,
    pub input_materialization_ns: u64,
    pub input_values_materialized: u64,
    pub bias_materialization_calls: u64,
    pub bias_materialization_ns: u64,
    pub bias_values_materialized: u64,
    pub packed_matvec_calls: u64,
    pub packed_matvec_ns: u64,
    pub packed_matvec_input_rows: u64,
    pub packed_matvec_output_values: u64,
    pub output_tensor_construction_calls: u64,
    pub output_tensor_construction_ns: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DenseLinearNoBiasCandidateInstrumentationSnapshot {
    pub selector_dispatch_calls: u64,
    pub selector_selected_calls: u64,
    pub selector_declined_calls: u64,
    pub selector_error_calls: u64,
    pub selector_dispatch_ns: u64,
    pub candidate_forward_calls: u64,
    pub candidate_forward_ns: u64,
}

struct DenseQ8SidecarInstrumentationCounters {
    selector_dispatch_calls: AtomicU64,
    selector_selected_calls: AtomicU64,
    selector_declined_calls: AtomicU64,
    selector_error_calls: AtomicU64,
    selector_dispatch_ns: AtomicU64,
    input_materialization_calls: AtomicU64,
    input_materialization_ns: AtomicU64,
    input_values_materialized: AtomicU64,
    bias_materialization_calls: AtomicU64,
    bias_materialization_ns: AtomicU64,
    bias_values_materialized: AtomicU64,
    packed_matvec_calls: AtomicU64,
    packed_matvec_ns: AtomicU64,
    packed_matvec_input_rows: AtomicU64,
    packed_matvec_output_values: AtomicU64,
    output_tensor_construction_calls: AtomicU64,
    output_tensor_construction_ns: AtomicU64,
}

struct DenseLinearNoBiasCandidateInstrumentationCounters {
    selector_dispatch_calls: AtomicU64,
    selector_selected_calls: AtomicU64,
    selector_declined_calls: AtomicU64,
    selector_error_calls: AtomicU64,
    selector_dispatch_ns: AtomicU64,
    candidate_forward_calls: AtomicU64,
    candidate_forward_ns: AtomicU64,
}

impl DenseQ8SidecarInstrumentationCounters {
    const fn new() -> Self {
        Self {
            selector_dispatch_calls: AtomicU64::new(0),
            selector_selected_calls: AtomicU64::new(0),
            selector_declined_calls: AtomicU64::new(0),
            selector_error_calls: AtomicU64::new(0),
            selector_dispatch_ns: AtomicU64::new(0),
            input_materialization_calls: AtomicU64::new(0),
            input_materialization_ns: AtomicU64::new(0),
            input_values_materialized: AtomicU64::new(0),
            bias_materialization_calls: AtomicU64::new(0),
            bias_materialization_ns: AtomicU64::new(0),
            bias_values_materialized: AtomicU64::new(0),
            packed_matvec_calls: AtomicU64::new(0),
            packed_matvec_ns: AtomicU64::new(0),
            packed_matvec_input_rows: AtomicU64::new(0),
            packed_matvec_output_values: AtomicU64::new(0),
            output_tensor_construction_calls: AtomicU64::new(0),
            output_tensor_construction_ns: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.selector_dispatch_calls.store(0, Ordering::Relaxed);
        self.selector_selected_calls.store(0, Ordering::Relaxed);
        self.selector_declined_calls.store(0, Ordering::Relaxed);
        self.selector_error_calls.store(0, Ordering::Relaxed);
        self.selector_dispatch_ns.store(0, Ordering::Relaxed);
        self.input_materialization_calls.store(0, Ordering::Relaxed);
        self.input_materialization_ns.store(0, Ordering::Relaxed);
        self.input_values_materialized.store(0, Ordering::Relaxed);
        self.bias_materialization_calls.store(0, Ordering::Relaxed);
        self.bias_materialization_ns.store(0, Ordering::Relaxed);
        self.bias_values_materialized.store(0, Ordering::Relaxed);
        self.packed_matvec_calls.store(0, Ordering::Relaxed);
        self.packed_matvec_ns.store(0, Ordering::Relaxed);
        self.packed_matvec_input_rows.store(0, Ordering::Relaxed);
        self.packed_matvec_output_values.store(0, Ordering::Relaxed);
        self.output_tensor_construction_calls.store(0, Ordering::Relaxed);
        self.output_tensor_construction_ns.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> DenseQ8SidecarInstrumentationSnapshot {
        DenseQ8SidecarInstrumentationSnapshot {
            selector_dispatch_calls: self.selector_dispatch_calls.load(Ordering::Relaxed),
            selector_selected_calls: self.selector_selected_calls.load(Ordering::Relaxed),
            selector_declined_calls: self.selector_declined_calls.load(Ordering::Relaxed),
            selector_error_calls: self.selector_error_calls.load(Ordering::Relaxed),
            selector_dispatch_ns: self.selector_dispatch_ns.load(Ordering::Relaxed),
            input_materialization_calls: self.input_materialization_calls.load(Ordering::Relaxed),
            input_materialization_ns: self.input_materialization_ns.load(Ordering::Relaxed),
            input_values_materialized: self.input_values_materialized.load(Ordering::Relaxed),
            bias_materialization_calls: self.bias_materialization_calls.load(Ordering::Relaxed),
            bias_materialization_ns: self.bias_materialization_ns.load(Ordering::Relaxed),
            bias_values_materialized: self.bias_values_materialized.load(Ordering::Relaxed),
            packed_matvec_calls: self.packed_matvec_calls.load(Ordering::Relaxed),
            packed_matvec_ns: self.packed_matvec_ns.load(Ordering::Relaxed),
            packed_matvec_input_rows: self.packed_matvec_input_rows.load(Ordering::Relaxed),
            packed_matvec_output_values: self.packed_matvec_output_values.load(Ordering::Relaxed),
            output_tensor_construction_calls: self
                .output_tensor_construction_calls
                .load(Ordering::Relaxed),
            output_tensor_construction_ns: self
                .output_tensor_construction_ns
                .load(Ordering::Relaxed),
        }
    }
}

impl DenseLinearNoBiasCandidateInstrumentationCounters {
    const fn new() -> Self {
        Self {
            selector_dispatch_calls: AtomicU64::new(0),
            selector_selected_calls: AtomicU64::new(0),
            selector_declined_calls: AtomicU64::new(0),
            selector_error_calls: AtomicU64::new(0),
            selector_dispatch_ns: AtomicU64::new(0),
            candidate_forward_calls: AtomicU64::new(0),
            candidate_forward_ns: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.selector_dispatch_calls.store(0, Ordering::Relaxed);
        self.selector_selected_calls.store(0, Ordering::Relaxed);
        self.selector_declined_calls.store(0, Ordering::Relaxed);
        self.selector_error_calls.store(0, Ordering::Relaxed);
        self.selector_dispatch_ns.store(0, Ordering::Relaxed);
        self.candidate_forward_calls.store(0, Ordering::Relaxed);
        self.candidate_forward_ns.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> DenseLinearNoBiasCandidateInstrumentationSnapshot {
        DenseLinearNoBiasCandidateInstrumentationSnapshot {
            selector_dispatch_calls: self.selector_dispatch_calls.load(Ordering::Relaxed),
            selector_selected_calls: self.selector_selected_calls.load(Ordering::Relaxed),
            selector_declined_calls: self.selector_declined_calls.load(Ordering::Relaxed),
            selector_error_calls: self.selector_error_calls.load(Ordering::Relaxed),
            selector_dispatch_ns: self.selector_dispatch_ns.load(Ordering::Relaxed),
            candidate_forward_calls: self.candidate_forward_calls.load(Ordering::Relaxed),
            candidate_forward_ns: self.candidate_forward_ns.load(Ordering::Relaxed),
        }
    }
}

static DENSE_Q8_SIDECAR_INSTRUMENTATION: DenseQ8SidecarInstrumentationCounters =
    DenseQ8SidecarInstrumentationCounters::new();
static DENSE_LINEAR_NO_BIAS_CANDIDATE_INSTRUMENTATION:
    DenseLinearNoBiasCandidateInstrumentationCounters =
    DenseLinearNoBiasCandidateInstrumentationCounters::new();

fn elapsed_ns_u64(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn add_counter(counter: &AtomicU64, value: u64) {
    counter.fetch_add(value, Ordering::Relaxed);
}

pub fn reset_dense_q8_sidecar_instrumentation() {
    DENSE_Q8_SIDECAR_INSTRUMENTATION.reset();
}

pub fn dense_q8_sidecar_instrumentation_snapshot() -> DenseQ8SidecarInstrumentationSnapshot {
    DENSE_Q8_SIDECAR_INSTRUMENTATION.snapshot()
}

pub fn reset_dense_linear_no_bias_candidate_instrumentation() {
    DENSE_LINEAR_NO_BIAS_CANDIDATE_INSTRUMENTATION.reset();
}

pub fn dense_linear_no_bias_candidate_instrumentation_snapshot()
-> DenseLinearNoBiasCandidateInstrumentationSnapshot {
    DENSE_LINEAR_NO_BIAS_CANDIDATE_INSTRUMENTATION.snapshot()
}

/// Evidence-scoped packed Q8_0 payload for a dense-linear runtime hook.
///
/// This carries bytes only when a caller intentionally wires one tensor path
/// for before/after proof. Runtime compute stays disabled until receipts prove
/// generated-ID/text preservation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearPackedQ8Payload {
    pub tensor_name: String,
    pub packed_q8_bytes: std::sync::Arc<[u8]>,
    pub q8_block_size: usize,
    pub q8_block_count: usize,
    pub matrix_rows: usize,
    pub matrix_cols: usize,
}

impl DenseLinearPackedQ8Payload {
    pub fn payload_len(&self) -> usize {
        self.packed_q8_bytes.len()
    }

    pub fn expected_q8_payload_len(&self) -> Option<usize> {
        self.q8_block_count.checked_mul(2 + self.q8_block_size)
    }

    pub fn shape_matches_matvec_contract(&self) -> bool {
        self.matrix_rows > 0
            && self.matrix_cols > 0
            && self.q8_block_size == 32
            && self
                .matrix_rows
                .checked_mul(self.matrix_cols)
                .is_some_and(|values| self.q8_block_count == values.div_ceil(self.q8_block_size))
    }

    pub fn payload_len_matches_contract(&self) -> bool {
        self.expected_q8_payload_len().is_some_and(|expected| expected == self.payload_len())
    }
}

/// Descriptor passed from model loading into transformer dense-linear calls.
///
/// Production loading may still pass metadata-only descriptors. A later
/// evidence-scoped slice can attach `packed_q8_payload` for exactly one tensor
/// path, but this descriptor still does not enable packed compute by itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearRuntimeHookDescriptor {
    pub tensor_name: String,
    pub role: String,
    pub sidecar_payload_sha256: Option<String>,
    pub packed_q8_payload: Option<DenseLinearPackedQ8Payload>,
    pub payload_order_matches_runtime_shape: bool,
    pub source_order_q8_matvec_candidate: bool,
    pub source_order_input_dim: Option<usize>,
    pub source_order_output_dim: Option<usize>,
    pub runtime_compute_enabled: bool,
    pub receipt_bound_no_bias_selector: Option<DenseLinearNoBiasReceiptBoundSelectorDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearRuntimeHookBoundary {
    pub tensor_name: String,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub sidecar_descriptor_present: bool,
    pub sidecar_role: Option<String>,
    pub sidecar_payload_sha256: Option<String>,
    pub sidecar_payload_bytes_available: bool,
    pub sidecar_payload_bytes: Option<usize>,
    pub sidecar_q8_block_count: Option<usize>,
    pub sidecar_matrix_rows: Option<usize>,
    pub sidecar_matrix_cols: Option<usize>,
    pub sidecar_payload_contract_valid: bool,
    pub sidecar_payload_order_matches_runtime_shape: bool,
    pub source_order_q8_matvec_candidate: bool,
    pub source_order_selected_path: Option<&'static str>,
    pub source_order_selected_kernel: Option<&'static str>,
    pub source_order_input_dim: Option<usize>,
    pub source_order_output_dim: Option<usize>,
    pub source_order_candidate_receipt_identity: Option<String>,
    pub source_order_candidate_runtime_enabled: bool,
    pub runtime_compute_enabled: bool,
    pub eager_f32_runtime_preserved: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub next_receipt_gate: &'static str,
}

/// Audit-only no-bias dense-linear selector decision.
///
/// This is deliberately separate from the runtime hook descriptor. It reports
/// whether a manifest role is a future no-bias candidate, but it never selects
/// compute and always preserves the eager F32 Candle path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasSelectorAudit {
    pub role_id: String,
    pub model_sha256: String,
    pub manifest_sha256: String,
    pub bias_present: Option<bool>,
    pub decision: &'static str,
    pub reason: &'static str,
    pub runtime_gate_name: &'static str,
    pub runtime_gate_default_enabled: bool,
    pub runtime_selection_enabled: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub eager_f32_runtime_preserved: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub fail_closed_conditions: Vec<&'static str>,
}

impl DenseLinearNoBiasSelectorAudit {
    pub const RUNTIME_GATE_NAME: &'static str = "BITNET_DENSE_NO_BIAS_LINEAR_ENABLE";

    pub fn from_role_evidence(
        role_id: impl Into<String>,
        model_sha256: impl Into<String>,
        manifest_sha256: impl Into<String>,
        bias_present: Option<bool>,
    ) -> Self {
        let (decision, reason, fail_closed_conditions) = match bias_present {
            Some(false) => (
                "eligible_no_bias_candidate_runtime_disabled",
                "bias_present_false_and_audit_hook_preserves_eager_f32",
                Vec::new(),
            ),
            Some(true) => (
                "blocked_fail_closed",
                "bias_present_true_blocks_no_bias_selector",
                vec!["bias_present_true"],
            ),
            None => (
                "blocked_fail_closed",
                "unknown_bias_present_blocks_no_bias_selector",
                vec!["unknown_bias_present"],
            ),
        };
        Self {
            role_id: role_id.into(),
            model_sha256: model_sha256.into(),
            manifest_sha256: manifest_sha256.into(),
            bias_present,
            decision,
            reason,
            runtime_gate_name: Self::RUNTIME_GATE_NAME,
            runtime_gate_default_enabled: false,
            runtime_selection_enabled: false,
            selected_path: "eager_f32_candle",
            selected_kernel: "dense-f32-candle-linear",
            eager_f32_runtime_preserved: true,
            dense_runtime_replaced: false,
            speedup_claim: false,
            generated_id_preservation_required_before_runtime_use: true,
            fail_closed_conditions,
        }
    }

    pub fn preserves_eager_f32(&self) -> bool {
        !self.runtime_selection_enabled
            && self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.eager_f32_runtime_preserved
            && !self.dense_runtime_replaced
            && !self.speedup_claim
    }

    pub fn is_eligible_future_candidate(&self) -> bool {
        self.decision == "eligible_no_bias_candidate_runtime_disabled"
            && self.bias_present == Some(false)
            && self.fail_closed_conditions.is_empty()
            && self.preserves_eager_f32()
    }
}

pub const SLM_CPU_195_QWEN3_Q8_MODEL_SHA256: &str =
    "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031";
pub const SLM_CPU_QWEN25_Q8_MODEL_SHA256: &str =
    "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
pub const SLM_CPU_195_QWEN3_DOWN_PROJ_LAYER_COUNT: usize = 28;
pub const SLM_CPU_195_NO_BIAS_CANDIDATE_PATH: &str =
    "qwen3_feed_forward_down_proj_no_bias_candidate";
pub const SLM_CPU_QWEN25_NO_BIAS_CANDIDATE_PATH: &str =
    "qwen25_feed_forward_down_proj_no_bias_candidate";
pub const SLM_CPU_195_NO_BIAS_CANDIDATE_KERNEL: &str = "dense-f32-no-bias-matmul-candidate";
pub const SLM_CPU_APPLY_LINEAR_NO_BIAS_CANDIDATE_KERNEL: &str =
    "dense-f32-candle-linear-no-bias-candidate";

/// Runtime-disabled no-bias dense-linear candidate for the SLM-CPU-195 slice.
///
/// This records that the exact Qwen3 Q8_0 `feed_forward.down_proj` role can be
/// computed without a bias add, but it does not select that candidate for model
/// execution. Runtime still uses the existing eager F32 Candle linear path until
/// a later PR supplies before/after warm-session receipts proving identical
/// generated IDs and decoded text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasFastPathCandidate {
    pub role_id: String,
    pub model_sha256: String,
    pub manifest_sha256: String,
    pub layer_idx: usize,
    pub scope: &'static str,
    pub linear: &'static str,
    pub bias_present: Option<bool>,
    pub decision: &'static str,
    pub reason: &'static str,
    pub runtime_gate_name: &'static str,
    pub runtime_gate_default_enabled: bool,
    pub runtime_selection_enabled: bool,
    pub candidate_compute_implemented: bool,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub eager_f32_runtime_preserved: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub fail_closed_conditions: Vec<&'static str>,
    pub required_receipt_fields_before_runtime_use: Vec<&'static str>,
}

impl DenseLinearNoBiasFastPathCandidate {
    pub fn qwen3_down_proj(
        layer_idx: usize,
        role_id: impl Into<String>,
        model_sha256: impl Into<String>,
        manifest_sha256: impl Into<String>,
        bias_present: Option<bool>,
    ) -> Self {
        let role_id = role_id.into();
        let model_sha256 = model_sha256.into();
        let expected_role_id = format!("layers.{layer_idx}.feed_forward.down_proj");
        let mut fail_closed_conditions = Vec::new();
        if layer_idx >= SLM_CPU_195_QWEN3_DOWN_PROJ_LAYER_COUNT {
            fail_closed_conditions.push("layer_outside_qwen3_0_6b_range");
        }
        if role_id != expected_role_id {
            fail_closed_conditions.push("role_not_qwen3_feed_forward_down_proj");
        }
        if model_sha256 != SLM_CPU_195_QWEN3_Q8_MODEL_SHA256 {
            fail_closed_conditions.push("model_sha_not_qwen3_0_6b_q8_0");
        }
        match bias_present {
            Some(false) => {}
            Some(true) => fail_closed_conditions.push("bias_present_true"),
            None => fail_closed_conditions.push("unknown_bias_present"),
        }

        let eligible = fail_closed_conditions.is_empty();
        Self {
            role_id,
            model_sha256,
            manifest_sha256: manifest_sha256.into(),
            layer_idx,
            scope: "feed_forward",
            linear: "down_proj",
            bias_present,
            decision: if eligible {
                "candidate_compute_available_runtime_disabled"
            } else {
                "blocked_fail_closed"
            },
            reason: if eligible {
                "exact_qwen3_q8_down_proj_bias_false_candidate_preserves_eager_runtime"
            } else {
                "candidate_scope_or_bias_evidence_failed_closed"
            },
            runtime_gate_name: DenseLinearNoBiasSelectorAudit::RUNTIME_GATE_NAME,
            runtime_gate_default_enabled: false,
            runtime_selection_enabled: false,
            candidate_compute_implemented: eligible,
            candidate_path: SLM_CPU_195_NO_BIAS_CANDIDATE_PATH,
            candidate_kernel: SLM_CPU_195_NO_BIAS_CANDIDATE_KERNEL,
            selected_path: "eager_f32_candle",
            selected_kernel: "dense-f32-candle-linear",
            eager_f32_runtime_preserved: true,
            dense_runtime_replaced: false,
            speedup_claim: false,
            generated_id_preservation_required_before_runtime_use: true,
            fail_closed_conditions,
            required_receipt_fields_before_runtime_use: vec![
                "model_sha256",
                "tokenizer.source",
                "tokenizer.strict",
                "selected_backend",
                "runtime_api",
                "fallback_used",
                "prompt_ids_hash",
                "generated_ids",
                "decoded_text",
                "dense_path_identity",
                "manifest_sha256",
                "role_id",
                "bias_present",
            ],
        }
    }

    pub fn is_runtime_disabled_candidate(&self) -> bool {
        self.decision == "candidate_compute_available_runtime_disabled"
            && self.candidate_compute_implemented
            && !self.runtime_selection_enabled
            && self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.eager_f32_runtime_preserved
            && !self.dense_runtime_replaced
            && !self.speedup_claim
            && self.fail_closed_conditions.is_empty()
    }
}

/// Candidate no-bias matmul used by tests and receipt-gated runtime work.
///
/// This is a narrow implementation surface for roles whose prompt/session
/// descriptor and receipts prove `bias_present=false`. It preserves the
/// default eager F32 path unless an explicit receipt gate routes execution here.
pub fn dense_linear_no_bias_candidate_forward(
    input: &Tensor,
    linear: &Linear,
) -> candle_core::Result<Tensor> {
    if linear.bias().is_some() {
        candle_core::bail!("no-bias dense-linear candidate requires bias_present=false");
    }
    let weight_dims = linear.weight().dims();
    if weight_dims.len() != 2 {
        candle_core::bail!(
            "no-bias dense-linear candidate requires 2D weight, got {:?}",
            weight_dims
        );
    }
    let output_dim = weight_dims[0];
    let input_dim = weight_dims[1];
    let input_dims = input.dims();
    if input_dims.last().copied() != Some(input_dim) {
        candle_core::bail!(
            "no-bias dense-linear candidate input last dim {:?} does not match weight input dim {}",
            input_dims.last(),
            input_dim
        );
    }
    if input_dims.len() == 2 {
        return input.matmul(&linear.weight().t()?);
    }

    let row_count = input_dims[..input_dims.len() - 1]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| {
            candle_core::Error::Msg("no-bias dense-linear candidate shape overflow".into())
        })?;
    let projected = input.reshape(&[row_count, input_dim])?.matmul(&linear.weight().t()?)?;
    let mut output_shape = input_dims.to_vec();
    if let Some(last) = output_shape.last_mut() {
        *last = output_dim;
    }
    projected.reshape(output_shape.as_slice())
}

const FEED_FORWARD_APPLY_LINEAR_CALLSITE: &str = "bitnet_transformer::FeedForward::apply_linear";

fn feed_forward_dense_tensor_name(layer_idx: usize, proj_name: &str) -> String {
    format!("layers.{layer_idx}.feed_forward.{proj_name}.weight")
}

fn feed_forward_apply_linear_callsite_identity(tensor_name: &str) -> String {
    format!("{FEED_FORWARD_APPLY_LINEAR_CALLSITE}:{tensor_name}")
}

pub fn dense_linear_no_bias_feed_forward_apply_linear_callsite_identity(
    layer_idx: usize,
    proj_name: &str,
) -> String {
    feed_forward_apply_linear_callsite_identity(&feed_forward_dense_tensor_name(
        layer_idx, proj_name,
    ))
}

fn prompt_bound_no_bias_descriptor_targets_feed_forward_down_proj_layer(
    descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    layer_idx: usize,
) -> bool {
    descriptor.tensor_name == feed_forward_dense_tensor_name(layer_idx, "down_proj")
}

fn feed_forward_no_bias_apply_linear_descriptor_fail_closed_conditions(
    descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    proj_name: &str,
    tensor_name: &str,
) -> Vec<&'static str> {
    let mut fail_closed_conditions = Vec::new();
    let expected_callsite_identity = feed_forward_apply_linear_callsite_identity(tensor_name);

    if proj_name != "down_proj" {
        fail_closed_conditions.push("feed_forward_projection_not_down_proj");
    }
    if descriptor.tensor_name != tensor_name {
        fail_closed_conditions.push("prompt_bound_descriptor_tensor_name_mismatch");
    }
    if descriptor.callsite_identity != expected_callsite_identity {
        fail_closed_conditions.push("prompt_bound_descriptor_callsite_identity_mismatch");
    }
    if !descriptor.per_callsite_receipt_emitter_present {
        fail_closed_conditions.push("per_callsite_receipt_emitter_missing");
    }
    if !descriptor.per_callsite_identity_matches_descriptor {
        fail_closed_conditions.push("per_callsite_identity_does_not_match_descriptor");
    }
    if !descriptor.explicit_runtime_gate_requested {
        fail_closed_conditions.push("explicit_runtime_gate_not_requested");
    }
    if descriptor.runtime_api != "cpu" {
        fail_closed_conditions.push("runtime_api_not_cpu");
    }
    if descriptor.selected_backend != "cpu-rust" {
        fail_closed_conditions.push("selected_backend_not_cpu_rust");
    }
    if descriptor.fallback_used {
        fail_closed_conditions.push("fallback_used");
    }
    if descriptor.selected_path != "eager_f32_candle" {
        fail_closed_conditions.push("selected_path_not_eager_f32_candle");
    }
    if descriptor.selected_kernel != "dense-f32-candle-linear" {
        fail_closed_conditions.push("selected_kernel_not_dense_f32_candle_linear");
    }
    if descriptor.normal_inference_runtime_selection_enabled {
        fail_closed_conditions.push("normal_inference_runtime_selection_enabled");
    }
    if descriptor.candidate_execution_enabled {
        fail_closed_conditions.push("candidate_execution_enabled");
    }
    if !descriptor.preserves_normal_inference() {
        fail_closed_conditions.push("normal_inference_not_preserved");
    }

    fail_closed_conditions.sort_unstable();
    fail_closed_conditions.dedup();
    fail_closed_conditions
}

fn feed_forward_no_bias_candidate_dispatch_fail_closed_conditions(
    descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    linear: &Linear,
    proj_name: &str,
    tensor_name: &str,
) -> Vec<&'static str> {
    let mut fail_closed_conditions =
        feed_forward_no_bias_apply_linear_descriptor_fail_closed_conditions(
            descriptor,
            proj_name,
            tensor_name,
        );
    match descriptor.model_architecture {
        "qwen3" => {
            if descriptor.model_sha256 != SLM_CPU_195_QWEN3_Q8_MODEL_SHA256 {
                fail_closed_conditions.push("model_sha_not_qwen3_0_6b_q8_0");
            }
            if descriptor.candidate_path != SLM_CPU_195_NO_BIAS_CANDIDATE_PATH {
                fail_closed_conditions.push("candidate_path_not_qwen3_down_proj_no_bias");
            }
        }
        "qwen2" => {
            if descriptor.model_sha256 != SLM_CPU_QWEN25_Q8_MODEL_SHA256 {
                fail_closed_conditions.push("model_sha_not_qwen25_0_5b_q8_0");
            }
            if descriptor.candidate_path != SLM_CPU_QWEN25_NO_BIAS_CANDIDATE_PATH {
                fail_closed_conditions.push("candidate_path_not_qwen25_down_proj_no_bias");
            }
        }
        _ => fail_closed_conditions.push("model_architecture_not_qwen2_or_qwen3"),
    }
    if descriptor.quant_format != "Q8_0" {
        fail_closed_conditions.push("quant_format_not_q8_0");
    }
    if descriptor.tokenizer_source != "gguf_metadata" {
        fail_closed_conditions.push("tokenizer_source_not_gguf_metadata");
    }
    if !descriptor.tokenizer_strict {
        fail_closed_conditions.push("tokenizer_not_strict");
    }
    if descriptor.candidate_kernel != SLM_CPU_APPLY_LINEAR_NO_BIAS_CANDIDATE_KERNEL {
        fail_closed_conditions.push("candidate_kernel_not_apply_linear_no_bias");
    }
    if linear.bias().is_some() {
        fail_closed_conditions.push("bias_present_true");
    }

    fail_closed_conditions.sort_unstable();
    fail_closed_conditions.dedup();
    fail_closed_conditions
}

fn maybe_forward_feed_forward_no_bias_candidate_linear(
    input: &Tensor,
    linear: &Linear,
    proj_name: &str,
    tensor_name: &str,
    prompt_bound_no_bias_descriptor: Option<
        &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    >,
) -> Result<Option<Tensor>> {
    let Some(descriptor) = prompt_bound_no_bias_descriptor else {
        return Ok(None);
    };
    let selector_start = Instant::now();
    add_counter(&DENSE_LINEAR_NO_BIAS_CANDIDATE_INSTRUMENTATION.selector_dispatch_calls, 1);
    let fail_closed_conditions = feed_forward_no_bias_candidate_dispatch_fail_closed_conditions(
        descriptor,
        linear,
        proj_name,
        tensor_name,
    );
    if !fail_closed_conditions.is_empty() {
        add_counter(&DENSE_LINEAR_NO_BIAS_CANDIDATE_INSTRUMENTATION.selector_error_calls, 1);
        add_counter(
            &DENSE_LINEAR_NO_BIAS_CANDIDATE_INSTRUMENTATION.selector_dispatch_ns,
            elapsed_ns_u64(selector_start),
        );
        return Err(BitNetError::Validation(format!(
            "prompt-bound no-bias descriptor for {tensor_name} failed closed before candidate dispatch: {}",
            fail_closed_conditions.join(",")
        )));
    }

    add_counter(&DENSE_LINEAR_NO_BIAS_CANDIDATE_INSTRUMENTATION.selector_selected_calls, 1);
    add_counter(
        &DENSE_LINEAR_NO_BIAS_CANDIDATE_INSTRUMENTATION.selector_dispatch_ns,
        elapsed_ns_u64(selector_start),
    );
    tracing::trace!(
        tensor_name = %tensor_name,
        callsite_identity = %descriptor.callsite_identity,
        selected_path = descriptor.selected_path,
        candidate_path = descriptor.candidate_path,
        "prompt-bound no-bias descriptor selected dense_linear_no_bias_candidate_forward"
    );

    let candidate_start = Instant::now();
    let output =
        dense_linear_no_bias_candidate_forward(input, linear).map_err(BitNetError::from)?;
    add_counter(&DENSE_LINEAR_NO_BIAS_CANDIDATE_INSTRUMENTATION.candidate_forward_calls, 1);
    add_counter(
        &DENSE_LINEAR_NO_BIAS_CANDIDATE_INSTRUMENTATION.candidate_forward_ns,
        elapsed_ns_u64(candidate_start),
    );
    Ok(Some(output))
}

/// Disabled-by-default preflight for future no-bias runtime selection.
///
/// This is an audit surface only. It can report that an exact Qwen3 Q8_0
/// down-projection candidate would be selectable in a receipt-gated experiment,
/// but normal inference still selects the eager F32 Candle path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasRuntimeSelectionPreflight {
    pub role_id: String,
    pub model_sha256: String,
    pub manifest_sha256: String,
    pub layer_idx: usize,
    pub runtime_gate_name: &'static str,
    pub runtime_gate_default_enabled: bool,
    pub runtime_gate_requested_enabled: bool,
    pub paired_strict_receipts_present: bool,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub normal_inference_runtime_selection_enabled: bool,
    pub would_select_candidate_in_receipt_gated_experiment: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub eager_f32_runtime_preserved: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub fail_closed_conditions: Vec<&'static str>,
    pub required_receipt_fields_before_runtime_use: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

impl DenseLinearNoBiasRuntimeSelectionPreflight {
    pub fn from_candidate(
        candidate: &DenseLinearNoBiasFastPathCandidate,
        runtime_gate_requested_enabled: bool,
        paired_strict_receipts_present: bool,
    ) -> Self {
        let mut fail_closed_conditions = candidate.fail_closed_conditions.clone();
        if !runtime_gate_requested_enabled {
            fail_closed_conditions.push("runtime_gate_not_requested");
        }
        if !paired_strict_receipts_present {
            fail_closed_conditions.push("paired_strict_receipts_missing");
        }

        let would_select_candidate_in_receipt_gated_experiment = candidate
            .is_runtime_disabled_candidate()
            && runtime_gate_requested_enabled
            && paired_strict_receipts_present;

        let (decision, reason) = if would_select_candidate_in_receipt_gated_experiment {
            (
                "would_select_candidate_in_receipt_gated_experiment",
                "exact_qwen3_q8_down_proj_candidate_gate_requested_and_receipts_present",
            )
        } else if !candidate.is_runtime_disabled_candidate() {
            ("blocked_fail_closed", "candidate_scope_or_bias_evidence_failed_closed")
        } else if !runtime_gate_requested_enabled {
            ("default_disabled_preserves_eager_f32", "runtime_gate_not_requested")
        } else {
            (
                "blocked_before_after_receipts_missing",
                "paired_strict_warm_session_receipts_required_before_runtime_selection",
            )
        };

        Self {
            role_id: candidate.role_id.clone(),
            model_sha256: candidate.model_sha256.clone(),
            manifest_sha256: candidate.manifest_sha256.clone(),
            layer_idx: candidate.layer_idx,
            runtime_gate_name: candidate.runtime_gate_name,
            runtime_gate_default_enabled: false,
            runtime_gate_requested_enabled,
            paired_strict_receipts_present,
            candidate_path: candidate.candidate_path,
            candidate_kernel: candidate.candidate_kernel,
            selected_path: "eager_f32_candle",
            selected_kernel: "dense-f32-candle-linear",
            normal_inference_runtime_selection_enabled: false,
            would_select_candidate_in_receipt_gated_experiment,
            decision,
            reason,
            eager_f32_runtime_preserved: true,
            generated_id_preservation_required_before_runtime_use: true,
            fail_closed_conditions,
            required_receipt_fields_before_runtime_use: candidate
                .required_receipt_fields_before_runtime_use
                .clone(),
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        !self.runtime_gate_default_enabled
            && !self.normal_inference_runtime_selection_enabled
            && self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.eager_f32_runtime_preserved
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

pub fn dense_linear_no_bias_runtime_gate_requested(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Manifest-bound no-bias runtime descriptor contract for SLM-CPU-199.
///
/// This is still a contract/audit surface, not a compute selector. It carries
/// the exact role identity that a future receipt-gated runtime experiment must
/// preserve before `FeedForward::apply_linear` can select the no-bias candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasRuntimeDescriptorContract {
    pub role_id: String,
    pub model_sha256: String,
    pub quant_format: &'static str,
    pub manifest_sha256: String,
    pub layer_idx: usize,
    pub scope: &'static str,
    pub linear: &'static str,
    pub bias_present: Option<bool>,
    pub runtime_gate_name: &'static str,
    pub runtime_gate_default_enabled: bool,
    pub runtime_gate_requested_enabled: bool,
    pub descriptor_fields_present: bool,
    pub receipt_identity_fields_present: bool,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub normal_inference_runtime_selection_enabled: bool,
    pub descriptor_ready_for_future_receipt_gate: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub required_descriptor_fields: Vec<&'static str>,
    pub required_receipt_fields_before_runtime_use: Vec<&'static str>,
    pub fail_closed_conditions: Vec<&'static str>,
    pub fallback_used_required: bool,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

impl DenseLinearNoBiasRuntimeDescriptorContract {
    pub fn from_preflight(
        preflight: &DenseLinearNoBiasRuntimeSelectionPreflight,
        quant_format: &'static str,
        bias_present: Option<bool>,
        descriptor_fields_present: bool,
        receipt_identity_fields_present: bool,
    ) -> Self {
        let mut fail_closed_conditions = Vec::new();
        if quant_format != "Q8_0" {
            fail_closed_conditions.push("quant_format_not_q8_0");
        }
        match bias_present {
            Some(false) => {}
            Some(true) => fail_closed_conditions.push("bias_present_true"),
            None => fail_closed_conditions.push("unknown_bias_present"),
        }
        if !descriptor_fields_present {
            fail_closed_conditions.push("descriptor_fields_missing");
        }
        if !receipt_identity_fields_present {
            fail_closed_conditions.push("receipt_identity_fields_missing");
        }
        if !preflight.would_select_candidate_in_receipt_gated_experiment {
            fail_closed_conditions.push("preflight_not_receipt_gate_selectable");
        }
        if !preflight.preserves_normal_inference() {
            fail_closed_conditions.push("preflight_does_not_preserve_normal_inference");
        }
        for condition in &preflight.fail_closed_conditions {
            if !fail_closed_conditions.contains(condition) {
                fail_closed_conditions.push(condition);
            }
        }

        let descriptor_ready_for_future_receipt_gate = fail_closed_conditions.is_empty();
        let (decision, reason) = if descriptor_ready_for_future_receipt_gate {
            (
                "descriptor_contract_ready_runtime_disabled",
                "manifest_bound_no_bias_descriptor_identity_complete",
            )
        } else {
            ("blocked_fail_closed", "manifest_bound_no_bias_descriptor_identity_incomplete")
        };

        Self {
            role_id: preflight.role_id.clone(),
            model_sha256: preflight.model_sha256.clone(),
            quant_format,
            manifest_sha256: preflight.manifest_sha256.clone(),
            layer_idx: preflight.layer_idx,
            scope: "feed_forward",
            linear: "down_proj",
            bias_present,
            runtime_gate_name: preflight.runtime_gate_name,
            runtime_gate_default_enabled: false,
            runtime_gate_requested_enabled: preflight.runtime_gate_requested_enabled,
            descriptor_fields_present,
            receipt_identity_fields_present,
            candidate_path: preflight.candidate_path,
            candidate_kernel: preflight.candidate_kernel,
            selected_path: preflight.selected_path,
            selected_kernel: preflight.selected_kernel,
            normal_inference_runtime_selection_enabled: false,
            descriptor_ready_for_future_receipt_gate,
            decision,
            reason,
            required_descriptor_fields: vec![
                "model_sha256",
                "quant_format",
                "manifest_sha256",
                "role_id",
                "layer",
                "scope",
                "linear",
                "bias_present",
                "candidate_path",
                "candidate_kernel",
                "selected_path",
                "selected_kernel",
                "runtime_gate_state",
                "fallback_used",
            ],
            required_receipt_fields_before_runtime_use: preflight
                .required_receipt_fields_before_runtime_use
                .clone(),
            fail_closed_conditions,
            fallback_used_required: false,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        !self.runtime_gate_default_enabled
            && !self.normal_inference_runtime_selection_enabled
            && self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && !self.fallback_used_required
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

/// Apply-linear audit boundary for a manifest-bound no-bias descriptor.
///
/// This boundary models the last check before a future `FeedForward` runtime
/// selector. It proves whether the descriptor identity matches the dense tensor
/// callsite, but it still preserves eager F32 Candle execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasApplyLinearAuditBoundary {
    pub tensor_name: String,
    pub role_id: String,
    pub model_sha256: String,
    pub quant_format: &'static str,
    pub manifest_sha256: String,
    pub layer_idx: usize,
    pub scope: &'static str,
    pub linear: &'static str,
    pub bias_present: Option<bool>,
    pub runtime_gate_name: &'static str,
    pub runtime_gate_requested_enabled: bool,
    pub descriptor_decision: &'static str,
    pub descriptor_ready_for_future_receipt_gate: bool,
    pub callsite_descriptor_observed: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasApplyLinearReceiptBoundary {
    pub tensor_name: String,
    pub role_id: String,
    pub model_sha256: String,
    pub quant_format: &'static str,
    pub manifest_sha256: String,
    pub layer_idx: usize,
    pub scope: &'static str,
    pub linear: &'static str,
    pub bias_present: Option<bool>,
    pub runtime_gate_name: &'static str,
    pub runtime_gate_requested_enabled: bool,
    pub descriptor_decision: &'static str,
    pub callsite_descriptor_observed: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub receipt_fields_present: bool,
    pub required_receipt_fields: Vec<&'static str>,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasApplyLinearBeforeAfterReceiptGate {
    pub tensor_name: String,
    pub role_id: String,
    pub model_sha256: String,
    pub quant_format: &'static str,
    pub manifest_sha256: String,
    pub layer_idx: usize,
    pub scope: &'static str,
    pub linear: &'static str,
    pub bias_present: Option<bool>,
    pub runtime_gate_name: &'static str,
    pub runtime_gate_requested_enabled: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub before_after_receipts_present: bool,
    pub descriptor_callsite_identity_preserved: bool,
    pub prompt_ids_digest_preserved: bool,
    pub generated_ids_digest_preserved: bool,
    pub decoded_text_digest_preserved: bool,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasReceiptBoundSelectorDescriptor {
    pub tensor_name: String,
    pub role_id: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub manifest_sha256: String,
    pub layer_idx: usize,
    pub scope: &'static str,
    pub linear: &'static str,
    pub bias_present: Option<bool>,
    pub runtime_gate_name: &'static str,
    pub runtime_gate_requested_enabled: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub before_after_receipts_present: bool,
    pub before_after_receipt_pair_identity: String,
    pub descriptor_callsite_identity_preserved: bool,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub prompt_ids_digest_preserved: bool,
    pub generated_ids_digest_preserved: bool,
    pub decoded_text_digest_preserved: bool,
    pub qwen2_candidate_policy_present: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub descriptor_ready_for_apply_linear_callsite: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasSelectorPropagationBoundary {
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub prompt_digest_lifetime: &'static str,
    pub hook_registry_owner: &'static str,
    pub hook_construction_callsite: &'static str,
    pub apply_linear_callsite: &'static str,
    pub hook_registry_selector_present: bool,
    pub hook_registry_mutation_point_present: bool,
    pub per_callsite_receipt_emitter_present: bool,
    pub descriptor_ready_for_apply_linear_callsite: bool,
    pub can_attach_after_prompt_digests_known: bool,
    pub can_attach_before_same_prompt_candidate_execution: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub descriptor_ready_for_apply_linear_callsite: bool,
    pub per_callsite_receipt_emitter_present: bool,
    pub per_callsite_identity_matches_descriptor: bool,
    pub explicit_runtime_gate_requested: bool,
    pub candidate_off_on_receipts_present: bool,
    pub generated_id_preservation_proven: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasPromptSessionDescriptor {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub prompt_ids: Vec<u32>,
    pub prompt_ids_digest: String,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub bias_present: Option<bool>,
    pub explicit_runtime_gate_requested: bool,
    pub descriptor_ready_for_apply_linear_callsite: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub generated_ids_bound_before_decode: bool,
    pub decoded_text_bound_before_decode: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasPromptSessionDescriptorInput<'a> {
    pub tensor_name: &'a str,
    pub callsite_identity: &'a str,
    pub model_sha256: &'a str,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub prompt_ids: &'a [u32],
    pub prompt_ids_digest: &'a str,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub bias_present: Option<bool>,
    pub explicit_runtime_gate_requested: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasCandidateOffOnReceiptPairGate {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub per_callsite_receipt_emitter_present: bool,
    pub per_callsite_identity_matches_descriptor: bool,
    pub explicit_runtime_gate_requested: bool,
    pub candidate_off_receipt_present: bool,
    pub candidate_on_receipt_present: bool,
    pub prompt_ids_preserved: bool,
    pub generated_ids_preserved: bool,
    pub decoded_text_preserved: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasPerCallsiteDispatchDescriptorBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub candidate_off_on_receipt_pair_gate_ready: bool,
    pub explicit_runtime_gate_requested: bool,
    pub prompt_bound_candidate_descriptor_argument_present: bool,
    pub prompt_bound_session_descriptor_constructed: bool,
    pub descriptor_identity_reaches_apply_linear_callsite: bool,
    pub prompt_digest_available_at_apply_linear: bool,
    pub generated_text_digests_available_at_apply_linear: bool,
    pub feed_forward_apply_linear_no_bias_dispatch_branch_present: bool,
    pub dispatch_calls_no_bias_candidate_forward: bool,
    pub candidate_on_receipt_emitted_at_apply_linear_callsite: bool,
    pub feed_forward_down_proj_scope_preserved: bool,
    pub default_runtime_path_preserved: bool,
    pub candidate_execution_attempt_allowed: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled_by_default: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasCandidateOnBehaviorEvidenceGate {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub candidate_off_on_pair_gate_ready: bool,
    pub candidate_on_behavior_evidence_present: bool,
    pub candidate_on_runtime_attachment_point_present: bool,
    pub candidate_on_receipt_fields_complete: bool,
    pub explicit_runtime_gate_requested: bool,
    pub prompt_ids_preserved: bool,
    pub generated_ids_preserved: bool,
    pub decoded_text_preserved: bool,
    pub default_runtime_path_preserved: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasCandidateRuntimeAttachmentBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub candidate_on_behavior_gate_ready: bool,
    pub explicit_runtime_gate_requested: bool,
    pub apply_linear_candidate_attachment_wired: bool,
    pub candidate_runtime_owner_present: bool,
    pub candidate_receipt_emitter_wired: bool,
    pub candidate_compute_callable: bool,
    pub default_runtime_path_preserved: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasCandidateRuntimeOwnerBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub candidate_runtime_attachment_boundary_defined: bool,
    pub explicit_runtime_gate_requested: bool,
    pub apply_linear_runtime_owner_present: bool,
    pub owner_has_apply_linear_inputs: bool,
    pub owner_has_linear_weight_access: bool,
    pub candidate_compute_callable: bool,
    pub same_callsite_candidate_on_receipt_emitter_wired: bool,
    pub candidate_off_on_strict_receipts_present: bool,
    pub prompt_ids_preserved: bool,
    pub generated_ids_preserved: bool,
    pub decoded_text_preserved: bool,
    pub default_runtime_path_preserved: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasCandidateRuntimeOwnerInputs {
    pub apply_linear_runtime_owner_present: bool,
    pub owner_has_apply_linear_inputs: bool,
    pub owner_has_linear_weight_access: bool,
    pub candidate_compute_callable: bool,
    pub same_callsite_candidate_on_receipt_emitter_wired: bool,
    pub candidate_off_on_strict_receipts_present: bool,
    pub prompt_ids_preserved: bool,
    pub generated_ids_preserved: bool,
    pub decoded_text_preserved: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub runtime_owner_boundary_defined: bool,
    pub apply_linear_runtime_owner_present: bool,
    pub owner_has_apply_linear_inputs: bool,
    pub owner_has_linear_weight_access: bool,
    pub candidate_compute_callable: bool,
    pub same_callsite_candidate_receipt_emitter_present: bool,
    pub candidate_off_strict_receipt_present: bool,
    pub candidate_on_strict_receipt_present: bool,
    pub strict_receipts_bind_owner_identity: bool,
    pub prompt_ids_preserved: bool,
    pub generated_ids_preserved: bool,
    pub decoded_text_preserved: bool,
    pub default_runtime_path_preserved: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
    pub same_callsite_candidate_receipt_emitter_present: bool,
    pub candidate_off_strict_receipt_present: bool,
    pub candidate_on_strict_receipt_present: bool,
    pub strict_receipts_bind_owner_identity: bool,
    pub prompt_ids_preserved: bool,
    pub generated_ids_preserved: bool,
    pub decoded_text_preserved: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub same_callsite_receipt_emitter_ready: bool,
    pub candidate_off_strict_receipt_artifact_present: bool,
    pub candidate_on_strict_receipt_artifact_present: bool,
    pub candidate_off_receipt_binds_owner_identity: bool,
    pub candidate_on_receipt_binds_owner_identity: bool,
    pub candidate_off_on_same_callsite_identity: bool,
    pub prompt_ids_preserved: bool,
    pub generated_ids_preserved: bool,
    pub decoded_text_preserved: bool,
    pub default_runtime_path_preserved: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptInputs {
    pub candidate_off_strict_receipt_artifact_present: bool,
    pub candidate_on_strict_receipt_artifact_present: bool,
    pub candidate_off_receipt_binds_owner_identity: bool,
    pub candidate_on_receipt_binds_owner_identity: bool,
    pub candidate_off_on_same_callsite_identity: bool,
    pub prompt_ids_preserved: bool,
    pub generated_ids_preserved: bool,
    pub decoded_text_preserved: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub off_on_strict_receipt_boundary_ready: bool,
    pub explicit_gate_identity_present: bool,
    pub descriptor_identity_present: bool,
    pub owner_callsite_identity_present: bool,
    pub prompt_generated_text_digests_bound: bool,
    pub default_runtime_path_preserved: bool,
    pub candidate_execution_attempt_allowed: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
    pub explicit_gate_identity_present: bool,
    pub descriptor_identity_present: bool,
    pub owner_callsite_identity_present: bool,
    pub prompt_generated_text_digests_bound: bool,
    pub explicit_candidate_execution_gate_requested: bool,
    pub default_runtime_path_preserved: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasStrictReceiptArtifactPairBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub receipt_gated_candidate_execution_boundary_ready: bool,
    pub candidate_off_strict_receipt_artifact_path: Option<String>,
    pub candidate_on_strict_receipt_artifact_path: Option<String>,
    pub candidate_off_strict_receipt_artifact_present: bool,
    pub candidate_on_strict_receipt_artifact_present: bool,
    pub candidate_off_receipt_binds_gate_identity: bool,
    pub candidate_on_receipt_binds_gate_identity: bool,
    pub candidate_off_receipt_binds_descriptor_identity: bool,
    pub candidate_on_receipt_binds_descriptor_identity: bool,
    pub candidate_off_receipt_binds_owner_callsite_identity: bool,
    pub candidate_on_receipt_binds_owner_callsite_identity: bool,
    pub candidate_off_on_same_callsite_identity: bool,
    pub candidate_off_on_same_prompt_digest: bool,
    pub candidate_off_on_same_generated_digest: bool,
    pub candidate_off_on_same_decoded_text_digest: bool,
    pub candidate_off_on_same_model_backend_identity: bool,
    pub default_runtime_path_preserved: bool,
    pub candidate_execution_attempt_allowed: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasStrictReceiptArtifactPairInputs {
    pub candidate_off_strict_receipt_artifact_path: Option<&'static str>,
    pub candidate_on_strict_receipt_artifact_path: Option<&'static str>,
    pub candidate_off_strict_receipt_artifact_present: bool,
    pub candidate_on_strict_receipt_artifact_present: bool,
    pub candidate_off_receipt_binds_gate_identity: bool,
    pub candidate_on_receipt_binds_gate_identity: bool,
    pub candidate_off_receipt_binds_descriptor_identity: bool,
    pub candidate_on_receipt_binds_descriptor_identity: bool,
    pub candidate_off_receipt_binds_owner_callsite_identity: bool,
    pub candidate_on_receipt_binds_owner_callsite_identity: bool,
    pub candidate_off_on_same_callsite_identity: bool,
    pub candidate_off_on_same_prompt_digest: bool,
    pub candidate_off_on_same_generated_digest: bool,
    pub candidate_off_on_same_decoded_text_digest: bool,
    pub candidate_off_on_same_model_backend_identity: bool,
    pub default_runtime_path_preserved: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasStrictArtifactCaptureBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub candidate_off_strict_receipt_artifact_path: Option<String>,
    pub candidate_on_strict_receipt_artifact_path: Option<String>,
    pub strict_receipt_artifact_pair_boundary_ready: bool,
    pub candidate_off_capture_artifact_validated: bool,
    pub candidate_on_capture_artifact_validated: bool,
    pub candidate_off_capture_command_recorded: bool,
    pub candidate_on_capture_command_recorded: bool,
    pub candidate_off_on_capture_same_callsite_identity: bool,
    pub candidate_off_on_capture_same_prompt_digest: bool,
    pub candidate_off_on_capture_same_generated_digest: bool,
    pub candidate_off_on_capture_same_decoded_text_digest: bool,
    pub candidate_off_on_capture_same_model_backend_identity: bool,
    pub capture_blocker_recorded: bool,
    pub default_runtime_path_preserved: bool,
    pub candidate_execution_prereqs_complete: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasStrictCaptureArtifactPairBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub strict_artifact_capture_boundary_ready: bool,
    pub candidate_off_strict_capture_artifact_path: Option<String>,
    pub candidate_on_strict_capture_artifact_path: Option<String>,
    pub candidate_off_strict_capture_artifact_present: bool,
    pub candidate_on_strict_capture_artifact_present: bool,
    pub candidate_off_capture_command_recorded: bool,
    pub candidate_on_capture_command_recorded: bool,
    pub candidate_off_capture_binds_gate_identity: bool,
    pub candidate_on_capture_binds_gate_identity: bool,
    pub candidate_off_capture_binds_descriptor_identity: bool,
    pub candidate_on_capture_binds_descriptor_identity: bool,
    pub candidate_off_capture_binds_owner_callsite_identity: bool,
    pub candidate_on_capture_binds_owner_callsite_identity: bool,
    pub candidate_off_on_capture_same_callsite_identity: bool,
    pub candidate_off_on_capture_same_prompt_digest: bool,
    pub candidate_off_on_capture_same_generated_digest: bool,
    pub candidate_off_on_capture_same_decoded_text_digest: bool,
    pub candidate_off_on_capture_same_model_backend_identity: bool,
    pub capture_prerequisite_blocker_recorded: bool,
    pub default_runtime_path_preserved: bool,
    pub strict_capture_artifact_pair_validated: bool,
    pub candidate_execution_prereqs_complete: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasRuntimeAttemptBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub strict_capture_artifact_pair_validated: bool,
    pub explicit_candidate_execution_gate_requested: bool,
    pub runtime_hook_registry_attachment_present: bool,
    pub runtime_hook_descriptor_binds_selector_identity: bool,
    pub runtime_hook_descriptor_binds_strict_capture_pair: bool,
    pub apply_linear_dispatch_wired_to_no_bias_candidate: bool,
    pub feed_forward_down_proj_scope_preserved: bool,
    pub default_runtime_path_preserved: bool,
    pub candidate_execution_attempt_allowed: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasRuntimeHookAttachmentBoundary {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub strict_capture_artifact_pair_validated: bool,
    pub explicit_candidate_execution_gate_requested: bool,
    pub runtime_hook_registry_attachment_present: bool,
    pub runtime_hook_descriptor_binds_selector_identity: bool,
    pub runtime_hook_descriptor_binds_strict_capture_pair: bool,
    pub registry_key_matches_tensor_name: bool,
    pub descriptor_ready_for_apply_linear_callsite: bool,
    pub feed_forward_down_proj_scope_preserved: bool,
    pub default_runtime_path_preserved: bool,
    pub candidate_execution_attempt_allowed: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearNoBiasCandidateExecutionReceiptGate {
    pub tensor_name: String,
    pub callsite_identity: String,
    pub model_sha256: String,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub runtime_api: &'static str,
    pub selected_backend: &'static str,
    pub fallback_used: bool,
    pub selected_path: &'static str,
    pub selected_kernel: &'static str,
    pub candidate_path: &'static str,
    pub candidate_kernel: &'static str,
    pub prompt_ids_digest: String,
    pub generated_ids_digest: String,
    pub decoded_text_digest: String,
    pub runtime_hook_attachment_ready: bool,
    pub explicit_candidate_execution_gate_requested: bool,
    pub runtime_hook_registry_attachment_present: bool,
    pub runtime_hook_descriptor_binds_selector_identity: bool,
    pub runtime_hook_descriptor_binds_strict_capture_pair: bool,
    pub registry_key_matches_tensor_name: bool,
    pub descriptor_ready_for_apply_linear_callsite: bool,
    pub candidate_off_execution_receipt_present: bool,
    pub candidate_on_execution_receipt_present: bool,
    pub candidate_off_execution_binds_registry_attachment: bool,
    pub candidate_on_execution_binds_registry_attachment: bool,
    pub candidate_off_on_same_callsite_identity: bool,
    pub candidate_off_on_same_prompt_digest: bool,
    pub candidate_off_on_same_generated_digest: bool,
    pub candidate_off_on_same_decoded_text_digest: bool,
    pub candidate_off_on_same_model_backend_identity: bool,
    pub prompt_ids_preserved: bool,
    pub generated_ids_preserved: bool,
    pub decoded_text_preserved: bool,
    pub default_runtime_path_preserved: bool,
    pub candidate_execution_receipt_pair_ready: bool,
    pub normal_inference_runtime_selection_enabled: bool,
    pub candidate_execution_enabled_by_default: bool,
    pub decision: &'static str,
    pub reason: &'static str,
    pub remaining_runtime_selection_blocker: &'static str,
    pub fail_closed_conditions: Vec<&'static str>,
    pub allocation_reduction_claim: bool,
    pub timing_improvement_claim: bool,
    pub speedup_claim: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasRuntimeAttemptInputs {
    pub explicit_candidate_execution_gate_requested: bool,
    pub runtime_hook_registry_attachment_present: bool,
    pub runtime_hook_descriptor_binds_selector_identity: bool,
    pub runtime_hook_descriptor_binds_strict_capture_pair: bool,
    pub apply_linear_dispatch_wired_to_no_bias_candidate: bool,
    pub feed_forward_down_proj_scope_preserved: bool,
    pub default_runtime_path_preserved: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasCandidateExecutionReceiptInputs {
    pub candidate_off_execution_receipt_present: bool,
    pub candidate_on_execution_receipt_present: bool,
    pub candidate_off_execution_binds_registry_attachment: bool,
    pub candidate_on_execution_binds_registry_attachment: bool,
    pub candidate_off_on_same_callsite_identity: bool,
    pub candidate_off_on_same_prompt_digest: bool,
    pub candidate_off_on_same_generated_digest: bool,
    pub candidate_off_on_same_decoded_text_digest: bool,
    pub candidate_off_on_same_model_backend_identity: bool,
    pub prompt_ids_preserved: bool,
    pub generated_ids_preserved: bool,
    pub decoded_text_preserved: bool,
    pub execution_receipt_blocker_recorded: bool,
    pub default_runtime_path_preserved: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasPerCallsiteDispatchDescriptorInputs {
    pub prompt_bound_candidate_descriptor_argument_present: bool,
    pub prompt_bound_session_descriptor_constructed: bool,
    pub descriptor_identity_reaches_apply_linear_callsite: bool,
    pub prompt_digest_available_at_apply_linear: bool,
    pub generated_text_digests_available_at_apply_linear: bool,
    pub feed_forward_apply_linear_no_bias_dispatch_branch_present: bool,
    pub dispatch_calls_no_bias_candidate_forward: bool,
    pub candidate_on_receipt_emitted_at_apply_linear_callsite: bool,
    pub feed_forward_down_proj_scope_preserved: bool,
    pub default_runtime_path_preserved: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasStrictArtifactCaptureInputs {
    pub candidate_off_capture_artifact_validated: bool,
    pub candidate_on_capture_artifact_validated: bool,
    pub candidate_off_capture_command_recorded: bool,
    pub candidate_on_capture_command_recorded: bool,
    pub candidate_off_on_capture_same_callsite_identity: bool,
    pub candidate_off_on_capture_same_prompt_digest: bool,
    pub candidate_off_on_capture_same_generated_digest: bool,
    pub candidate_off_on_capture_same_decoded_text_digest: bool,
    pub candidate_off_on_capture_same_model_backend_identity: bool,
    pub capture_blocker_recorded: bool,
    pub default_runtime_path_preserved: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseLinearNoBiasStrictCaptureArtifactPairInputs {
    pub candidate_off_strict_capture_artifact_path: Option<&'static str>,
    pub candidate_on_strict_capture_artifact_path: Option<&'static str>,
    pub candidate_off_strict_capture_artifact_present: bool,
    pub candidate_on_strict_capture_artifact_present: bool,
    pub candidate_off_capture_command_recorded: bool,
    pub candidate_on_capture_command_recorded: bool,
    pub candidate_off_capture_binds_gate_identity: bool,
    pub candidate_on_capture_binds_gate_identity: bool,
    pub candidate_off_capture_binds_descriptor_identity: bool,
    pub candidate_on_capture_binds_descriptor_identity: bool,
    pub candidate_off_capture_binds_owner_callsite_identity: bool,
    pub candidate_on_capture_binds_owner_callsite_identity: bool,
    pub candidate_off_on_capture_same_callsite_identity: bool,
    pub candidate_off_on_capture_same_prompt_digest: bool,
    pub candidate_off_on_capture_same_generated_digest: bool,
    pub candidate_off_on_capture_same_decoded_text_digest: bool,
    pub candidate_off_on_capture_same_model_backend_identity: bool,
    pub capture_prerequisite_blocker_recorded: bool,
    pub default_runtime_path_preserved: bool,
}

impl DenseLinearNoBiasApplyLinearAuditBoundary {
    pub fn from_descriptor_contract(
        tensor_name: impl Into<String>,
        contract: &DenseLinearNoBiasRuntimeDescriptorContract,
        runtime_gate_requested_enabled: bool,
    ) -> Self {
        let tensor_name = tensor_name.into();
        let expected_tensor_name =
            format!("layers.{}.{}.{}.weight", contract.layer_idx, contract.scope, contract.linear);
        let mut fail_closed_conditions = contract.fail_closed_conditions.clone();
        if tensor_name != expected_tensor_name {
            fail_closed_conditions.push("tensor_name_not_descriptor_role");
        }
        if contract.role_id
            != format!("layers.{}.{}.{}", contract.layer_idx, contract.scope, contract.linear)
        {
            fail_closed_conditions.push("role_id_not_descriptor_role");
        }
        if !runtime_gate_requested_enabled {
            fail_closed_conditions.push("runtime_gate_not_requested");
        }
        if contract.selected_path != "eager_f32_candle"
            || contract.selected_kernel != "dense-f32-candle-linear"
            || !contract.preserves_normal_inference()
        {
            fail_closed_conditions.push("descriptor_does_not_preserve_eager_f32");
        }

        let callsite_descriptor_observed = fail_closed_conditions.is_empty();
        let (decision, reason, remaining_runtime_selection_blocker) =
            if callsite_descriptor_observed {
                (
                    "descriptor_observed_at_apply_linear_runtime_disabled",
                    "exact_qwen3_down_proj_descriptor_matches_dense_tensor_callsite",
                    "fresh_before_after_strict_warm_session_receipts_and_receipt_emission_wiring",
                )
            } else {
                (
                    "blocked_fail_closed",
                    "descriptor_identity_or_gate_not_valid_for_apply_linear_callsite",
                    "descriptor_callsite_identity_or_runtime_gate",
                )
            };

        Self {
            tensor_name,
            role_id: contract.role_id.clone(),
            model_sha256: contract.model_sha256.clone(),
            quant_format: contract.quant_format,
            manifest_sha256: contract.manifest_sha256.clone(),
            layer_idx: contract.layer_idx,
            scope: contract.scope,
            linear: contract.linear,
            bias_present: contract.bias_present,
            runtime_gate_name: contract.runtime_gate_name,
            runtime_gate_requested_enabled,
            descriptor_decision: contract.decision,
            descriptor_ready_for_future_receipt_gate: contract
                .descriptor_ready_for_future_receipt_gate,
            callsite_descriptor_observed,
            selected_path: "eager_f32_candle",
            selected_kernel: "dense-f32-candle-linear",
            candidate_path: contract.candidate_path,
            candidate_kernel: contract.candidate_kernel,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasApplyLinearReceiptBoundary {
    pub fn from_apply_linear_boundary(
        boundary: &DenseLinearNoBiasApplyLinearAuditBoundary,
        runtime_api: &'static str,
        selected_backend: &'static str,
        fallback_used: bool,
        prompt_ids_digest: impl Into<String>,
        generated_ids_digest: impl Into<String>,
        decoded_text_digest: impl Into<String>,
        receipt_fields_present: bool,
    ) -> Self {
        let prompt_ids_digest = prompt_ids_digest.into();
        let generated_ids_digest = generated_ids_digest.into();
        let decoded_text_digest = decoded_text_digest.into();
        let mut fail_closed_conditions = boundary.fail_closed_conditions.clone();
        if !boundary.callsite_descriptor_observed
            || boundary.decision != "descriptor_observed_at_apply_linear_runtime_disabled"
        {
            fail_closed_conditions.push("apply_linear_descriptor_not_observed");
        }
        if runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if fallback_used {
            fail_closed_conditions.push("fallback_used");
        }
        if prompt_ids_digest.is_empty() {
            fail_closed_conditions.push("prompt_ids_digest_missing");
        }
        if generated_ids_digest.is_empty() {
            fail_closed_conditions.push("generated_ids_digest_missing");
        }
        if decoded_text_digest.is_empty() {
            fail_closed_conditions.push("decoded_text_digest_missing");
        }
        if !receipt_fields_present {
            fail_closed_conditions.push("receipt_fields_missing");
        }
        if !boundary.preserves_normal_inference() {
            fail_closed_conditions.push("apply_linear_boundary_does_not_preserve_eager_f32");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let receipt_identity_ready = fail_closed_conditions.is_empty();
        let (decision, reason, remaining_runtime_selection_blocker) = if receipt_identity_ready {
            (
                "receipt_emission_boundary_ready_runtime_disabled",
                "descriptor_callsite_identity_receipt_visible",
                "fresh_before_after_strict_warm_session_receipts",
            )
        } else {
            (
                "blocked_fail_closed",
                "descriptor_callsite_receipt_identity_incomplete",
                "descriptor_callsite_receipt_identity",
            )
        };

        Self {
            tensor_name: boundary.tensor_name.clone(),
            role_id: boundary.role_id.clone(),
            model_sha256: boundary.model_sha256.clone(),
            quant_format: boundary.quant_format,
            manifest_sha256: boundary.manifest_sha256.clone(),
            layer_idx: boundary.layer_idx,
            scope: boundary.scope,
            linear: boundary.linear,
            bias_present: boundary.bias_present,
            runtime_gate_name: boundary.runtime_gate_name,
            runtime_gate_requested_enabled: boundary.runtime_gate_requested_enabled,
            descriptor_decision: boundary.descriptor_decision,
            callsite_descriptor_observed: boundary.callsite_descriptor_observed,
            selected_path: boundary.selected_path,
            selected_kernel: boundary.selected_kernel,
            candidate_path: boundary.candidate_path,
            candidate_kernel: boundary.candidate_kernel,
            runtime_api,
            selected_backend,
            fallback_used,
            prompt_ids_digest,
            generated_ids_digest,
            decoded_text_digest,
            receipt_fields_present,
            required_receipt_fields: vec![
                "model_sha256",
                "quant_format",
                "manifest_sha256",
                "role_id",
                "layer",
                "scope",
                "linear",
                "bias_present=false",
                "tensor_name",
                "selected_path=eager_f32_candle",
                "selected_kernel=dense-f32-candle-linear",
                "candidate_path",
                "candidate_kernel",
                "runtime_gate_state",
                "runtime_api=cpu",
                "selected_backend=cpu-rust",
                "fallback=false",
                "prompt_ids_digest",
                "generated_ids_digest",
                "decoded_text_digest",
            ],
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasApplyLinearBeforeAfterReceiptGate {
    pub fn from_receipt_boundaries(
        before: &DenseLinearNoBiasApplyLinearReceiptBoundary,
        after: &DenseLinearNoBiasApplyLinearReceiptBoundary,
        before_after_receipts_present: bool,
    ) -> Self {
        let mut fail_closed_conditions = Vec::new();
        if before.decision != "receipt_emission_boundary_ready_runtime_disabled" {
            fail_closed_conditions.push("before_receipt_boundary_not_ready");
        }
        if after.decision != "receipt_emission_boundary_ready_runtime_disabled" {
            fail_closed_conditions.push("after_receipt_boundary_not_ready");
        }
        if !before_after_receipts_present {
            fail_closed_conditions.push("before_after_receipts_missing");
        }
        if before.tensor_name != after.tensor_name {
            fail_closed_conditions.push("tensor_name_changed");
        }
        if before.role_id != after.role_id {
            fail_closed_conditions.push("role_id_changed");
        }
        if before.model_sha256 != after.model_sha256 {
            fail_closed_conditions.push("model_sha256_changed");
        }
        if before.quant_format != after.quant_format {
            fail_closed_conditions.push("quant_format_changed");
        }
        if before.manifest_sha256 != after.manifest_sha256 {
            fail_closed_conditions.push("manifest_sha256_changed");
        }
        if before.layer_idx != after.layer_idx {
            fail_closed_conditions.push("layer_idx_changed");
        }
        if before.scope != after.scope {
            fail_closed_conditions.push("scope_changed");
        }
        if before.linear != after.linear {
            fail_closed_conditions.push("linear_changed");
        }
        if before.bias_present != Some(false) || after.bias_present != Some(false) {
            fail_closed_conditions.push("bias_present_not_false");
        }
        if before.runtime_gate_name != after.runtime_gate_name {
            fail_closed_conditions.push("runtime_gate_name_changed");
        }
        if before.runtime_gate_requested_enabled != after.runtime_gate_requested_enabled {
            fail_closed_conditions.push("runtime_gate_request_changed");
        }
        if before.selected_path != after.selected_path || after.selected_path != "eager_f32_candle"
        {
            fail_closed_conditions.push("selected_path_not_preserved_eager_f32");
        }
        if before.selected_kernel != after.selected_kernel
            || after.selected_kernel != "dense-f32-candle-linear"
        {
            fail_closed_conditions.push("selected_kernel_not_preserved_eager_f32");
        }
        if before.candidate_path != after.candidate_path {
            fail_closed_conditions.push("candidate_path_changed");
        }
        if before.candidate_kernel != after.candidate_kernel {
            fail_closed_conditions.push("candidate_kernel_changed");
        }
        if before.runtime_api != after.runtime_api || after.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_preserved_cpu");
        }
        if before.selected_backend != after.selected_backend || after.selected_backend != "cpu-rust"
        {
            fail_closed_conditions.push("selected_backend_not_preserved_cpu_rust");
        }
        if before.fallback_used || after.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }
        if before.prompt_ids_digest.is_empty()
            || after.prompt_ids_digest.is_empty()
            || before.prompt_ids_digest != after.prompt_ids_digest
        {
            fail_closed_conditions.push("prompt_ids_digest_not_preserved");
        }
        if before.generated_ids_digest.is_empty()
            || after.generated_ids_digest.is_empty()
            || before.generated_ids_digest != after.generated_ids_digest
        {
            fail_closed_conditions.push("generated_ids_digest_not_preserved");
        }
        if before.decoded_text_digest.is_empty()
            || after.decoded_text_digest.is_empty()
            || before.decoded_text_digest != after.decoded_text_digest
        {
            fail_closed_conditions.push("decoded_text_digest_not_preserved");
        }
        if !before.receipt_fields_present || !after.receipt_fields_present {
            fail_closed_conditions.push("receipt_fields_missing");
        }
        if !before.callsite_descriptor_observed || !after.callsite_descriptor_observed {
            fail_closed_conditions.push("descriptor_callsite_not_observed");
        }
        if !before.preserves_normal_inference() || !after.preserves_normal_inference() {
            fail_closed_conditions.push("normal_inference_not_preserved");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let descriptor_callsite_identity_preserved = before.tensor_name == after.tensor_name
            && before.role_id == after.role_id
            && before.model_sha256 == after.model_sha256
            && before.quant_format == after.quant_format
            && before.manifest_sha256 == after.manifest_sha256
            && before.layer_idx == after.layer_idx
            && before.scope == after.scope
            && before.linear == after.linear
            && before.bias_present == Some(false)
            && after.bias_present == Some(false)
            && before.candidate_path == after.candidate_path
            && before.candidate_kernel == after.candidate_kernel
            && before.callsite_descriptor_observed
            && after.callsite_descriptor_observed;
        let prompt_ids_digest_preserved = !before.prompt_ids_digest.is_empty()
            && before.prompt_ids_digest == after.prompt_ids_digest;
        let generated_ids_digest_preserved = !before.generated_ids_digest.is_empty()
            && before.generated_ids_digest == after.generated_ids_digest;
        let decoded_text_digest_preserved = !before.decoded_text_digest.is_empty()
            && before.decoded_text_digest == after.decoded_text_digest;

        let gate_ready = fail_closed_conditions.is_empty();
        let (decision, reason, remaining_runtime_selection_blocker) = if gate_ready {
            (
                "before_after_receipt_gate_ready_runtime_disabled",
                "strict_warm_session_identity_preserved",
                "candidate_execution_still_disabled_until_explicit_runtime_selection_pr",
            )
        } else {
            (
                "blocked_fail_closed",
                "before_after_strict_warm_session_identity_incomplete_or_drifted",
                "before_after_strict_warm_session_identity",
            )
        };

        Self {
            tensor_name: before.tensor_name.clone(),
            role_id: before.role_id.clone(),
            model_sha256: before.model_sha256.clone(),
            quant_format: before.quant_format,
            manifest_sha256: before.manifest_sha256.clone(),
            layer_idx: before.layer_idx,
            scope: before.scope,
            linear: before.linear,
            bias_present: before.bias_present,
            runtime_gate_name: before.runtime_gate_name,
            runtime_gate_requested_enabled: before.runtime_gate_requested_enabled,
            selected_path: before.selected_path,
            selected_kernel: before.selected_kernel,
            candidate_path: before.candidate_path,
            candidate_kernel: before.candidate_kernel,
            runtime_api: before.runtime_api,
            selected_backend: before.selected_backend,
            fallback_used: before.fallback_used || after.fallback_used,
            before_after_receipts_present,
            descriptor_callsite_identity_preserved,
            prompt_ids_digest_preserved,
            generated_ids_digest_preserved,
            decoded_text_digest_preserved,
            prompt_ids_digest: before.prompt_ids_digest.clone(),
            generated_ids_digest: before.generated_ids_digest.clone(),
            decoded_text_digest: before.decoded_text_digest.clone(),
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasReceiptBoundSelectorDescriptor {
    pub fn from_before_after_gate(
        gate: &DenseLinearNoBiasApplyLinearBeforeAfterReceiptGate,
        model_architecture: &'static str,
        tokenizer_source: &'static str,
        tokenizer_strict: bool,
        before_after_receipt_pair_identity: impl Into<String>,
        qwen2_candidate_policy_present: bool,
    ) -> Self {
        let before_after_receipt_pair_identity = before_after_receipt_pair_identity.into();
        let mut fail_closed_conditions = gate.fail_closed_conditions.clone();
        if !matches!(model_architecture, "qwen3" | "qwen2") {
            fail_closed_conditions.push("model_architecture_not_qwen2_or_qwen3");
        }
        if model_architecture == "qwen2" && !qwen2_candidate_policy_present {
            fail_closed_conditions.push("qwen2_candidate_policy_missing");
        }
        if gate.quant_format != "Q8_0" {
            fail_closed_conditions.push("quant_format_not_q8_0");
        }
        if tokenizer_source != "gguf_metadata" {
            fail_closed_conditions.push("tokenizer_source_not_gguf_metadata");
        }
        if !tokenizer_strict {
            fail_closed_conditions.push("tokenizer_not_strict");
        }
        if gate.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if gate.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if gate.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }
        if !gate.before_after_receipts_present {
            fail_closed_conditions.push("before_after_receipts_missing");
        }
        if !gate.descriptor_callsite_identity_preserved {
            fail_closed_conditions.push("descriptor_callsite_identity_not_preserved");
        }
        if !gate.prompt_ids_digest_preserved || gate.prompt_ids_digest.is_empty() {
            fail_closed_conditions.push("prompt_ids_digest_not_preserved");
        }
        if !gate.generated_ids_digest_preserved || gate.generated_ids_digest.is_empty() {
            fail_closed_conditions.push("generated_ids_digest_not_preserved");
        }
        if !gate.decoded_text_digest_preserved || gate.decoded_text_digest.is_empty() {
            fail_closed_conditions.push("decoded_text_digest_not_preserved");
        }
        if before_after_receipt_pair_identity.is_empty() {
            fail_closed_conditions.push("before_after_receipt_pair_identity_missing");
        }
        if gate.selected_path != "eager_f32_candle" {
            fail_closed_conditions.push("selected_path_not_eager_f32_candle");
        }
        if gate.selected_kernel != "dense-f32-candle-linear" {
            fail_closed_conditions.push("selected_kernel_not_dense_f32_candle_linear");
        }
        if gate.normal_inference_runtime_selection_enabled {
            fail_closed_conditions.push("normal_inference_runtime_selection_enabled");
        }
        if gate.candidate_execution_enabled {
            fail_closed_conditions.push("candidate_execution_enabled");
        }
        if !gate.preserves_normal_inference() {
            fail_closed_conditions.push("normal_inference_not_preserved");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let descriptor_ready_for_apply_linear_callsite = fail_closed_conditions.is_empty();
        let (decision, reason, remaining_runtime_selection_blocker) =
            if descriptor_ready_for_apply_linear_callsite {
                (
                    "receipt_bound_selector_descriptor_ready_runtime_disabled",
                    "slm_cpu_209_identity_can_reach_apply_linear_selector",
                    "fresh_candidate_off_on_strict_warm_session_receipts_before_runtime_execution",
                )
            } else {
                (
                    "blocked_fail_closed",
                    "receipt_bound_selector_descriptor_identity_incomplete",
                    "receipt_bound_descriptor_identity_or_candidate_policy",
                )
            };

        Self {
            tensor_name: gate.tensor_name.clone(),
            role_id: gate.role_id.clone(),
            model_sha256: gate.model_sha256.clone(),
            model_architecture,
            quant_format: gate.quant_format,
            tokenizer_source,
            tokenizer_strict,
            manifest_sha256: gate.manifest_sha256.clone(),
            layer_idx: gate.layer_idx,
            scope: gate.scope,
            linear: gate.linear,
            bias_present: gate.bias_present,
            runtime_gate_name: gate.runtime_gate_name,
            runtime_gate_requested_enabled: gate.runtime_gate_requested_enabled,
            selected_path: gate.selected_path,
            selected_kernel: gate.selected_kernel,
            candidate_path: gate.candidate_path,
            candidate_kernel: gate.candidate_kernel,
            runtime_api: gate.runtime_api,
            selected_backend: gate.selected_backend,
            fallback_used: gate.fallback_used,
            before_after_receipts_present: gate.before_after_receipts_present,
            before_after_receipt_pair_identity,
            descriptor_callsite_identity_preserved: gate.descriptor_callsite_identity_preserved,
            prompt_ids_digest: gate.prompt_ids_digest.clone(),
            generated_ids_digest: gate.generated_ids_digest.clone(),
            decoded_text_digest: gate.decoded_text_digest.clone(),
            prompt_ids_digest_preserved: gate.prompt_ids_digest_preserved,
            generated_ids_digest_preserved: gate.generated_ids_digest_preserved,
            decoded_text_digest_preserved: gate.decoded_text_digest_preserved,
            qwen2_candidate_policy_present,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            descriptor_ready_for_apply_linear_callsite,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasSelectorPropagationBoundary {
    pub fn from_receipt_bound_selector_descriptor(
        descriptor: &DenseLinearNoBiasReceiptBoundSelectorDescriptor,
        hook_registry_selector_present: bool,
        hook_registry_mutation_point_present: bool,
        per_callsite_receipt_emitter_present: bool,
    ) -> Self {
        let mut fail_closed_conditions = descriptor.fail_closed_conditions.clone();
        if !descriptor.descriptor_ready_for_apply_linear_callsite {
            fail_closed_conditions.push("receipt_bound_selector_descriptor_not_ready");
        }
        if !hook_registry_selector_present {
            fail_closed_conditions.push("hook_registry_selector_identity_missing");
        }
        if !hook_registry_mutation_point_present && !per_callsite_receipt_emitter_present {
            fail_closed_conditions.push("session_selector_mutation_point_missing");
            fail_closed_conditions.push("per_callsite_candidate_receipt_emitter_missing");
        }
        if descriptor.prompt_ids_digest.is_empty() {
            fail_closed_conditions.push("prompt_ids_digest_missing");
        }
        if descriptor.generated_ids_digest.is_empty() {
            fail_closed_conditions.push("generated_ids_digest_missing");
        }
        if descriptor.decoded_text_digest.is_empty() {
            fail_closed_conditions.push("decoded_text_digest_missing");
        }
        if !descriptor.preserves_normal_inference() {
            fail_closed_conditions.push("descriptor_does_not_preserve_normal_inference");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let can_attach_after_prompt_digests_known = descriptor
            .descriptor_ready_for_apply_linear_callsite
            && (hook_registry_mutation_point_present || per_callsite_receipt_emitter_present);
        let can_attach_before_same_prompt_candidate_execution =
            can_attach_after_prompt_digests_known
                && hook_registry_selector_present
                && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) =
            if can_attach_before_same_prompt_candidate_execution {
                (
                    "selector_propagation_boundary_ready_runtime_disabled",
                    "receipt_bound_selector_identity_has_safe_session_or_callsite_attachment_point",
                    "candidate_off_on_strict_warm_session_receipts_before_runtime_selection",
                )
            } else {
                (
                    "blocked_fail_closed",
                    "receipt_bound_selector_identity_cannot_reach_apply_linear_before_candidate_execution",
                    "session_hook_registry_mutation_point_or_per_callsite_candidate_receipt_emitter",
                )
            };

        Self {
            model_sha256: descriptor.model_sha256.clone(),
            model_architecture: descriptor.model_architecture,
            quant_format: descriptor.quant_format,
            tokenizer_source: descriptor.tokenizer_source,
            tokenizer_strict: descriptor.tokenizer_strict,
            runtime_api: descriptor.runtime_api,
            selected_backend: descriptor.selected_backend,
            fallback_used: descriptor.fallback_used,
            selected_path: descriptor.selected_path,
            selected_kernel: descriptor.selected_kernel,
            candidate_path: descriptor.candidate_path,
            candidate_kernel: descriptor.candidate_kernel,
            prompt_ids_digest: descriptor.prompt_ids_digest.clone(),
            generated_ids_digest: descriptor.generated_ids_digest.clone(),
            decoded_text_digest: descriptor.decoded_text_digest.clone(),
            prompt_digest_lifetime: "available_after_warm_session_prompt_execution",
            hook_registry_owner: "bitnet_models::bitnet::dense_q8_runtime_hooks_from_sidecars",
            hook_construction_callsite: "bitnet_models::bitnet::dense_q8_runtime_hooks_from_sidecars",
            apply_linear_callsite: "bitnet_transformer::FeedForward::apply_linear",
            hook_registry_selector_present,
            hook_registry_mutation_point_present,
            per_callsite_receipt_emitter_present,
            descriptor_ready_for_apply_linear_callsite: descriptor
                .descriptor_ready_for_apply_linear_callsite,
            can_attach_after_prompt_digests_known,
            can_attach_before_same_prompt_candidate_execution,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasPromptSessionDescriptor {
    pub fn from_prompt_session(input: DenseLinearNoBiasPromptSessionDescriptorInput<'_>) -> Self {
        let mut fail_closed_conditions = Vec::new();
        if input.tensor_name.is_empty() {
            fail_closed_conditions.push("tensor_name_missing");
        }
        if input.callsite_identity.is_empty() {
            fail_closed_conditions.push("callsite_identity_missing");
        }
        if input.model_sha256.is_empty() {
            fail_closed_conditions.push("model_sha256_missing");
        }
        if !matches!(input.model_architecture, "qwen2" | "qwen3") {
            fail_closed_conditions.push("model_architecture_not_qwen2_or_qwen3");
        }
        if input.quant_format != "Q8_0" {
            fail_closed_conditions.push("quant_format_not_q8_0");
        }
        if input.tokenizer_source != "gguf_metadata" {
            fail_closed_conditions.push("tokenizer_source_not_gguf_metadata");
        }
        if !input.tokenizer_strict {
            fail_closed_conditions.push("tokenizer_not_strict");
        }
        if input.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if input.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if input.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }
        if input.prompt_ids.is_empty() {
            fail_closed_conditions.push("prompt_ids_missing");
        }
        if input.prompt_ids_digest.is_empty() {
            fail_closed_conditions.push("prompt_ids_digest_missing");
        }
        if input.selected_path != "eager_f32_candle" {
            fail_closed_conditions.push("selected_path_not_eager_f32_candle");
        }
        if input.selected_kernel != "dense-f32-candle-linear" {
            fail_closed_conditions.push("selected_kernel_not_dense_f32_candle_linear");
        }
        if !matches!(input.bias_present, Some(false)) {
            fail_closed_conditions.push("bias_present_not_false");
        }
        if !input.explicit_runtime_gate_requested {
            fail_closed_conditions.push("explicit_runtime_gate_not_requested");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let descriptor_ready_for_apply_linear_callsite = fail_closed_conditions.is_empty();
        let (decision, reason, remaining_runtime_selection_blocker) =
            if descriptor_ready_for_apply_linear_callsite {
                (
                    "prompt_session_descriptor_ready_for_apply_linear_runtime_disabled",
                    "prompt_session_identity_is_bound_before_decode_without_generated_text_fields",
                    "post_decode_candidate_off_on_receipts_and_dispatch_branch",
                )
            } else {
                (
                    "blocked_fail_closed",
                    "prompt_session_descriptor_identity_incomplete_or_not_explicitly_gated",
                    "prompt_session_descriptor_construction_inputs",
                )
            };

        Self {
            tensor_name: input.tensor_name.to_string(),
            callsite_identity: input.callsite_identity.to_string(),
            model_sha256: input.model_sha256.to_string(),
            model_architecture: input.model_architecture,
            quant_format: input.quant_format,
            tokenizer_source: input.tokenizer_source,
            tokenizer_strict: input.tokenizer_strict,
            runtime_api: input.runtime_api,
            selected_backend: input.selected_backend,
            fallback_used: input.fallback_used,
            prompt_ids: input.prompt_ids.to_vec(),
            prompt_ids_digest: input.prompt_ids_digest.to_string(),
            selected_path: input.selected_path,
            selected_kernel: input.selected_kernel,
            candidate_path: input.candidate_path,
            candidate_kernel: input.candidate_kernel,
            bias_present: input.bias_present,
            explicit_runtime_gate_requested: input.explicit_runtime_gate_requested,
            descriptor_ready_for_apply_linear_callsite,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            generated_ids_bound_before_decode: false,
            decoded_text_bound_before_decode: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.generated_ids_bound_before_decode
            && !self.decoded_text_bound_before_decode
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary {
    pub fn from_prompt_session_descriptor(
        descriptor: &DenseLinearNoBiasPromptSessionDescriptor,
    ) -> Self {
        let mut fail_closed_conditions = descriptor.fail_closed_conditions.clone();
        let expected_callsite_identity =
            feed_forward_apply_linear_callsite_identity(&descriptor.tensor_name);
        if !descriptor.descriptor_ready_for_apply_linear_callsite {
            fail_closed_conditions.push("prompt_session_descriptor_not_ready");
        }
        if descriptor.callsite_identity != expected_callsite_identity {
            fail_closed_conditions.push("prompt_session_callsite_identity_mismatch");
        }
        if !descriptor.preserves_normal_inference() {
            fail_closed_conditions
                .push("prompt_session_descriptor_does_not_preserve_normal_inference");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let per_callsite_identity_matches_descriptor = descriptor.callsite_identity
            == expected_callsite_identity
            && descriptor.descriptor_ready_for_apply_linear_callsite;
        let per_callsite_receipt_emitter_present = per_callsite_identity_matches_descriptor;
        let (decision, reason, remaining_runtime_selection_blocker) =
            if per_callsite_identity_matches_descriptor && fail_closed_conditions.is_empty() {
                (
                    "per_callsite_prompt_session_descriptor_ready_runtime_disabled",
                    "prompt_session_descriptor_identity_reaches_apply_linear_without_generated_text_binding",
                    "post_decode_candidate_off_on_receipts_and_dispatch_branch",
                )
            } else {
                (
                    "blocked_fail_closed",
                    "prompt_session_descriptor_identity_does_not_match_apply_linear_callsite",
                    "per_callsite_prompt_session_descriptor_identity",
                )
            };

        Self {
            tensor_name: descriptor.tensor_name.clone(),
            callsite_identity: descriptor.callsite_identity.clone(),
            model_sha256: descriptor.model_sha256.clone(),
            model_architecture: descriptor.model_architecture,
            quant_format: descriptor.quant_format,
            tokenizer_source: descriptor.tokenizer_source,
            tokenizer_strict: descriptor.tokenizer_strict,
            runtime_api: descriptor.runtime_api,
            selected_backend: descriptor.selected_backend,
            fallback_used: descriptor.fallback_used,
            selected_path: descriptor.selected_path,
            selected_kernel: descriptor.selected_kernel,
            candidate_path: descriptor.candidate_path,
            candidate_kernel: descriptor.candidate_kernel,
            prompt_ids_digest: descriptor.prompt_ids_digest.clone(),
            generated_ids_digest: String::new(),
            decoded_text_digest: String::new(),
            descriptor_ready_for_apply_linear_callsite: descriptor
                .descriptor_ready_for_apply_linear_callsite,
            per_callsite_receipt_emitter_present,
            per_callsite_identity_matches_descriptor,
            explicit_runtime_gate_requested: descriptor.explicit_runtime_gate_requested,
            candidate_off_on_receipts_present: false,
            generated_id_preservation_proven: false,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn from_receipt_bound_selector_descriptor(
        descriptor: &DenseLinearNoBiasReceiptBoundSelectorDescriptor,
        tensor_name: impl Into<String>,
        callsite_identity: impl Into<String>,
        explicit_runtime_gate_requested: bool,
        candidate_off_on_receipts_present: bool,
        generated_id_preservation_proven: bool,
    ) -> Self {
        let tensor_name = tensor_name.into();
        let callsite_identity = callsite_identity.into();
        let mut fail_closed_conditions = descriptor.fail_closed_conditions.clone();
        if !descriptor.descriptor_ready_for_apply_linear_callsite {
            fail_closed_conditions.push("receipt_bound_selector_descriptor_not_ready");
        }
        if tensor_name != descriptor.tensor_name {
            fail_closed_conditions.push("callsite_tensor_name_mismatch");
        }
        if callsite_identity.is_empty() {
            fail_closed_conditions.push("callsite_identity_missing");
        }
        if !descriptor.preserves_normal_inference() {
            fail_closed_conditions.push("descriptor_does_not_preserve_normal_inference");
        }
        if descriptor.prompt_ids_digest.is_empty() {
            fail_closed_conditions.push("prompt_ids_digest_missing");
        }
        if descriptor.generated_ids_digest.is_empty() {
            fail_closed_conditions.push("generated_ids_digest_missing");
        }
        if descriptor.decoded_text_digest.is_empty() {
            fail_closed_conditions.push("decoded_text_digest_missing");
        }
        if !explicit_runtime_gate_requested {
            fail_closed_conditions.push("explicit_runtime_gate_not_requested");
        }
        if !candidate_off_on_receipts_present {
            fail_closed_conditions.push("candidate_off_on_receipts_missing");
        }
        if !generated_id_preservation_proven {
            fail_closed_conditions.push("generated_id_preservation_not_proven");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let per_callsite_identity_matches_descriptor = tensor_name == descriptor.tensor_name
            && !callsite_identity.is_empty()
            && descriptor.descriptor_ready_for_apply_linear_callsite;
        let per_callsite_receipt_emitter_present = per_callsite_identity_matches_descriptor;

        let (decision, reason, remaining_runtime_selection_blocker) =
            if per_callsite_identity_matches_descriptor
                && explicit_runtime_gate_requested
                && candidate_off_on_receipts_present
                && generated_id_preservation_proven
                && fail_closed_conditions.is_empty()
            {
                (
                    "per_callsite_candidate_receipt_emitter_ready_runtime_disabled",
                    "descriptor_identity_reaches_apply_linear_callsite_with_receipt_fields",
                    "candidate_execution_enablement_pr_with_strict_off_on_receipts",
                )
            } else if per_callsite_identity_matches_descriptor {
                (
                    "per_callsite_candidate_receipt_emitter_defined_fail_closed",
                    "descriptor_identity_reaches_apply_linear_but_candidate_on_proof_is_incomplete",
                    "explicit_gate_and_candidate_off_on_generated_id_preservation_receipts",
                )
            } else {
                (
                    "blocked_fail_closed",
                    "descriptor_identity_does_not_match_apply_linear_callsite",
                    "per_callsite_descriptor_identity",
                )
            };

        Self {
            tensor_name,
            callsite_identity,
            model_sha256: descriptor.model_sha256.clone(),
            model_architecture: descriptor.model_architecture,
            quant_format: descriptor.quant_format,
            tokenizer_source: descriptor.tokenizer_source,
            tokenizer_strict: descriptor.tokenizer_strict,
            runtime_api: descriptor.runtime_api,
            selected_backend: descriptor.selected_backend,
            fallback_used: descriptor.fallback_used,
            selected_path: descriptor.selected_path,
            selected_kernel: descriptor.selected_kernel,
            candidate_path: descriptor.candidate_path,
            candidate_kernel: descriptor.candidate_kernel,
            prompt_ids_digest: descriptor.prompt_ids_digest.clone(),
            generated_ids_digest: descriptor.generated_ids_digest.clone(),
            decoded_text_digest: descriptor.decoded_text_digest.clone(),
            descriptor_ready_for_apply_linear_callsite: descriptor
                .descriptor_ready_for_apply_linear_callsite,
            per_callsite_receipt_emitter_present,
            per_callsite_identity_matches_descriptor,
            explicit_runtime_gate_requested,
            candidate_off_on_receipts_present,
            generated_id_preservation_proven,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasCandidateOffOnReceiptPairGate {
    pub fn from_per_callsite_emitter(
        emitter: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
        candidate_off_receipt_present: bool,
        candidate_on_receipt_present: bool,
        prompt_ids_preserved: bool,
        generated_ids_preserved: bool,
        decoded_text_preserved: bool,
    ) -> Self {
        let mut fail_closed_conditions = emitter.fail_closed_conditions.clone();
        if !emitter.per_callsite_receipt_emitter_present {
            fail_closed_conditions.push("per_callsite_receipt_emitter_missing");
        }
        if !emitter.per_callsite_identity_matches_descriptor {
            fail_closed_conditions.push("per_callsite_identity_not_preserved");
        }
        if !emitter.explicit_runtime_gate_requested {
            fail_closed_conditions.push("explicit_runtime_gate_not_requested");
        }
        if !candidate_off_receipt_present {
            fail_closed_conditions.push("candidate_off_receipt_missing");
        }
        if !candidate_on_receipt_present {
            fail_closed_conditions.push("candidate_on_receipt_missing");
        }
        if !prompt_ids_preserved {
            fail_closed_conditions.push("prompt_ids_not_preserved");
        }
        if !generated_ids_preserved {
            fail_closed_conditions.push("generated_ids_not_preserved");
        }
        if !decoded_text_preserved {
            fail_closed_conditions.push("decoded_text_not_preserved");
        }
        if emitter.prompt_ids_digest.is_empty() {
            fail_closed_conditions.push("prompt_ids_digest_missing");
        }
        if emitter.generated_ids_digest.is_empty() {
            fail_closed_conditions.push("generated_ids_digest_missing");
        }
        if emitter.decoded_text_digest.is_empty() {
            fail_closed_conditions.push("decoded_text_digest_missing");
        }
        if !emitter.preserves_normal_inference() {
            fail_closed_conditions.push("emitter_does_not_preserve_normal_inference");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let candidate_off_on_receipt_pair_ready = emitter.per_callsite_receipt_emitter_present
            && emitter.per_callsite_identity_matches_descriptor
            && emitter.explicit_runtime_gate_requested
            && candidate_off_receipt_present
            && candidate_on_receipt_present
            && prompt_ids_preserved
            && generated_ids_preserved
            && decoded_text_preserved
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) =
            if candidate_off_on_receipt_pair_ready {
                (
                    "candidate_off_on_receipt_pair_gate_ready_runtime_disabled",
                    "candidate_off_on_receipts_preserve_strict_warm_session_identity",
                    "candidate_execution_enablement_pr_must_consume_receipt_pair_gate",
                )
            } else if emitter.per_callsite_receipt_emitter_present {
                (
                    "candidate_off_on_receipt_pair_gate_defined_fail_closed",
                    "candidate_off_on_receipt_pair_incomplete",
                    "candidate_on_strict_warm_session_receipt_artifact",
                )
            } else {
                (
                    "blocked_fail_closed",
                    "per_callsite_receipt_emitter_identity_incomplete",
                    "per_callsite_candidate_receipt_emitter",
                )
            };

        Self {
            tensor_name: emitter.tensor_name.clone(),
            callsite_identity: emitter.callsite_identity.clone(),
            model_sha256: emitter.model_sha256.clone(),
            model_architecture: emitter.model_architecture,
            quant_format: emitter.quant_format,
            tokenizer_source: emitter.tokenizer_source,
            tokenizer_strict: emitter.tokenizer_strict,
            runtime_api: emitter.runtime_api,
            selected_backend: emitter.selected_backend,
            fallback_used: emitter.fallback_used,
            selected_path: emitter.selected_path,
            selected_kernel: emitter.selected_kernel,
            candidate_path: emitter.candidate_path,
            candidate_kernel: emitter.candidate_kernel,
            prompt_ids_digest: emitter.prompt_ids_digest.clone(),
            generated_ids_digest: emitter.generated_ids_digest.clone(),
            decoded_text_digest: emitter.decoded_text_digest.clone(),
            per_callsite_receipt_emitter_present: emitter.per_callsite_receipt_emitter_present,
            per_callsite_identity_matches_descriptor: emitter
                .per_callsite_identity_matches_descriptor,
            explicit_runtime_gate_requested: emitter.explicit_runtime_gate_requested,
            candidate_off_receipt_present,
            candidate_on_receipt_present,
            prompt_ids_preserved,
            generated_ids_preserved,
            decoded_text_preserved,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasPerCallsiteDispatchDescriptorBoundary {
    pub fn from_candidate_off_on_pair_gate(
        pair_gate: &DenseLinearNoBiasCandidateOffOnReceiptPairGate,
        inputs: DenseLinearNoBiasPerCallsiteDispatchDescriptorInputs,
    ) -> Self {
        let mut fail_closed_conditions = pair_gate.fail_closed_conditions.clone();
        let candidate_off_on_receipt_pair_gate_ready = pair_gate.decision
            == "candidate_off_on_receipt_pair_gate_ready_runtime_disabled"
            && pair_gate.per_callsite_receipt_emitter_present
            && pair_gate.per_callsite_identity_matches_descriptor
            && pair_gate.explicit_runtime_gate_requested
            && pair_gate.candidate_off_receipt_present
            && pair_gate.candidate_on_receipt_present
            && pair_gate.prompt_ids_preserved
            && pair_gate.generated_ids_preserved
            && pair_gate.decoded_text_preserved
            && pair_gate.preserves_normal_inference()
            && pair_gate.fail_closed_conditions.is_empty();
        let default_runtime_path_preserved = inputs.default_runtime_path_preserved
            && pair_gate.selected_path == "eager_f32_candle"
            && pair_gate.selected_kernel == "dense-f32-candle-linear"
            && pair_gate.preserves_normal_inference();

        if !candidate_off_on_receipt_pair_gate_ready {
            fail_closed_conditions.push("candidate_off_on_receipt_pair_gate_not_ready");
        }
        if !inputs.prompt_bound_candidate_descriptor_argument_present {
            fail_closed_conditions.push(
                "feed_forward_apply_linear_prompt_bound_candidate_descriptor_argument_missing",
            );
        }
        if !inputs.prompt_bound_session_descriptor_constructed {
            fail_closed_conditions.push("prompt_bound_session_descriptor_not_constructed");
        }
        if !inputs.descriptor_identity_reaches_apply_linear_callsite {
            fail_closed_conditions
                .push("per_callsite_descriptor_identity_not_available_at_apply_linear");
        }
        if !inputs.prompt_digest_available_at_apply_linear {
            fail_closed_conditions.push("prompt_digest_not_available_at_apply_linear");
        }
        if !inputs.generated_text_digests_available_at_apply_linear {
            fail_closed_conditions
                .push("generated_text_digests_not_available_before_apply_linear_dispatch");
        }
        if !inputs.feed_forward_apply_linear_no_bias_dispatch_branch_present {
            fail_closed_conditions
                .push("feed_forward_apply_linear_no_bias_dispatch_branch_missing");
        }
        if !inputs.dispatch_calls_no_bias_candidate_forward {
            fail_closed_conditions.push("dense_linear_no_bias_candidate_forward_dispatch_missing");
        }
        if !inputs.candidate_on_receipt_emitted_at_apply_linear_callsite {
            fail_closed_conditions
                .push("candidate_on_receipt_not_emitted_at_apply_linear_callsite");
        }
        if !inputs.feed_forward_down_proj_scope_preserved {
            fail_closed_conditions.push("feed_forward_down_proj_scope_not_preserved");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if pair_gate.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if pair_gate.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if pair_gate.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let candidate_execution_attempt_allowed = candidate_off_on_receipt_pair_gate_ready
            && pair_gate.explicit_runtime_gate_requested
            && inputs.prompt_bound_candidate_descriptor_argument_present
            && inputs.prompt_bound_session_descriptor_constructed
            && inputs.descriptor_identity_reaches_apply_linear_callsite
            && inputs.prompt_digest_available_at_apply_linear
            && inputs.generated_text_digests_available_at_apply_linear
            && inputs.feed_forward_apply_linear_no_bias_dispatch_branch_present
            && inputs.dispatch_calls_no_bias_candidate_forward
            && inputs.candidate_on_receipt_emitted_at_apply_linear_callsite
            && inputs.feed_forward_down_proj_scope_preserved
            && default_runtime_path_preserved
            && pair_gate.runtime_api == "cpu"
            && pair_gate.selected_backend == "cpu-rust"
            && !pair_gate.fallback_used
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) =
            if candidate_execution_attempt_allowed {
                (
                    "per_callsite_dispatch_descriptor_ready_runtime_disabled",
                    "prompt_bound_candidate_descriptor_reaches_apply_linear_with_dispatch_branch_but_runtime_enablement_remains_separate",
                    "fresh_candidate_off_on_execution_receipts_from_apply_linear",
                )
            } else if candidate_off_on_receipt_pair_gate_ready {
                let blocker = if !inputs.prompt_bound_candidate_descriptor_argument_present {
                    "feed_forward_apply_linear_prompt_bound_candidate_descriptor_argument"
                } else if !inputs.prompt_bound_session_descriptor_constructed {
                    "prompt_bound_session_descriptor_construction"
                } else if !inputs.descriptor_identity_reaches_apply_linear_callsite {
                    "per_callsite_descriptor_identity_to_apply_linear"
                } else if !inputs.prompt_digest_available_at_apply_linear {
                    "prompt_digest_lifetime_at_apply_linear"
                } else if !inputs.generated_text_digests_available_at_apply_linear {
                    "generated_text_digest_lifetime_before_apply_linear_dispatch"
                } else if !inputs.feed_forward_apply_linear_no_bias_dispatch_branch_present {
                    "feed_forward_apply_linear_no_bias_dispatch_branch"
                } else if !inputs.dispatch_calls_no_bias_candidate_forward {
                    "dense_linear_no_bias_candidate_forward_dispatch_call"
                } else if !inputs.candidate_on_receipt_emitted_at_apply_linear_callsite {
                    "apply_linear_callsite_candidate_on_receipt_emitter"
                } else if !inputs.feed_forward_down_proj_scope_preserved {
                    "feed_forward_down_proj_scope"
                } else {
                    "default_runtime_path_preservation"
                };
                (
                    "per_callsite_dispatch_descriptor_blocked_fail_closed",
                    "candidate_off_on_identity_exists_but_prompt_bound_descriptor_or_apply_linear_dispatch_is_missing",
                    blocker,
                )
            } else {
                (
                    "blocked_fail_closed",
                    "candidate_off_on_receipt_pair_gate_incomplete",
                    "candidate_off_on_receipt_pair_gate",
                )
            };

        Self {
            tensor_name: pair_gate.tensor_name.clone(),
            callsite_identity: pair_gate.callsite_identity.clone(),
            model_sha256: pair_gate.model_sha256.clone(),
            model_architecture: pair_gate.model_architecture,
            quant_format: pair_gate.quant_format,
            tokenizer_source: pair_gate.tokenizer_source,
            tokenizer_strict: pair_gate.tokenizer_strict,
            runtime_api: pair_gate.runtime_api,
            selected_backend: pair_gate.selected_backend,
            fallback_used: pair_gate.fallback_used,
            selected_path: pair_gate.selected_path,
            selected_kernel: pair_gate.selected_kernel,
            candidate_path: pair_gate.candidate_path,
            candidate_kernel: pair_gate.candidate_kernel,
            prompt_ids_digest: pair_gate.prompt_ids_digest.clone(),
            generated_ids_digest: pair_gate.generated_ids_digest.clone(),
            decoded_text_digest: pair_gate.decoded_text_digest.clone(),
            candidate_off_on_receipt_pair_gate_ready,
            explicit_runtime_gate_requested: pair_gate.explicit_runtime_gate_requested,
            prompt_bound_candidate_descriptor_argument_present: inputs
                .prompt_bound_candidate_descriptor_argument_present,
            prompt_bound_session_descriptor_constructed: inputs
                .prompt_bound_session_descriptor_constructed,
            descriptor_identity_reaches_apply_linear_callsite: inputs
                .descriptor_identity_reaches_apply_linear_callsite,
            prompt_digest_available_at_apply_linear: inputs.prompt_digest_available_at_apply_linear,
            generated_text_digests_available_at_apply_linear: inputs
                .generated_text_digests_available_at_apply_linear,
            feed_forward_apply_linear_no_bias_dispatch_branch_present: inputs
                .feed_forward_apply_linear_no_bias_dispatch_branch_present,
            dispatch_calls_no_bias_candidate_forward: inputs
                .dispatch_calls_no_bias_candidate_forward,
            candidate_on_receipt_emitted_at_apply_linear_callsite: inputs
                .candidate_on_receipt_emitted_at_apply_linear_callsite,
            feed_forward_down_proj_scope_preserved: inputs.feed_forward_down_proj_scope_preserved,
            default_runtime_path_preserved,
            candidate_execution_attempt_allowed,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled_by_default: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled_by_default
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasCandidateOnBehaviorEvidenceGate {
    pub fn from_candidate_off_on_pair_gate(
        pair_gate: &DenseLinearNoBiasCandidateOffOnReceiptPairGate,
        candidate_on_behavior_evidence_present: bool,
        candidate_on_runtime_attachment_point_present: bool,
        candidate_on_receipt_fields_complete: bool,
    ) -> Self {
        let mut fail_closed_conditions = pair_gate.fail_closed_conditions.clone();
        let candidate_off_on_pair_gate_ready = pair_gate.decision
            == "candidate_off_on_receipt_pair_gate_ready_runtime_disabled"
            && pair_gate.fail_closed_conditions.is_empty();
        let default_runtime_path_preserved = pair_gate.selected_path == "eager_f32_candle"
            && pair_gate.selected_kernel == "dense-f32-candle-linear"
            && pair_gate.preserves_normal_inference();

        if !candidate_off_on_pair_gate_ready {
            fail_closed_conditions.push("candidate_off_on_pair_gate_not_ready");
        }
        if !candidate_on_behavior_evidence_present {
            fail_closed_conditions.push("candidate_on_behavior_evidence_missing");
        }
        if !candidate_on_runtime_attachment_point_present {
            fail_closed_conditions.push("candidate_on_runtime_attachment_point_missing");
        }
        if !candidate_on_receipt_fields_complete {
            fail_closed_conditions.push("candidate_on_receipt_fields_incomplete");
        }
        if !pair_gate.explicit_runtime_gate_requested {
            fail_closed_conditions.push("explicit_runtime_gate_not_requested");
        }
        if !pair_gate.prompt_ids_preserved {
            fail_closed_conditions.push("prompt_ids_not_preserved");
        }
        if !pair_gate.generated_ids_preserved {
            fail_closed_conditions.push("generated_ids_not_preserved");
        }
        if !pair_gate.decoded_text_preserved {
            fail_closed_conditions.push("decoded_text_not_preserved");
        }
        if pair_gate.prompt_ids_digest.is_empty() {
            fail_closed_conditions.push("prompt_ids_digest_missing");
        }
        if pair_gate.generated_ids_digest.is_empty() {
            fail_closed_conditions.push("generated_ids_digest_missing");
        }
        if pair_gate.decoded_text_digest.is_empty() {
            fail_closed_conditions.push("decoded_text_digest_missing");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if pair_gate.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if pair_gate.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if pair_gate.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let candidate_on_behavior_evidence_ready = candidate_off_on_pair_gate_ready
            && candidate_on_behavior_evidence_present
            && candidate_on_runtime_attachment_point_present
            && candidate_on_receipt_fields_complete
            && pair_gate.explicit_runtime_gate_requested
            && pair_gate.prompt_ids_preserved
            && pair_gate.generated_ids_preserved
            && pair_gate.decoded_text_preserved
            && default_runtime_path_preserved
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) =
            if candidate_on_behavior_evidence_ready {
                (
                    "candidate_on_behavior_evidence_ready_runtime_disabled",
                    "candidate_on_behavior_preserves_strict_warm_session_identity",
                    "candidate_execution_enablement_pr_must_consume_behavior_evidence_gate",
                )
            } else if candidate_off_on_pair_gate_ready {
                let blocker = if !candidate_on_runtime_attachment_point_present {
                    "candidate_on_apply_linear_runtime_attachment_point"
                } else if !candidate_on_receipt_fields_complete {
                    "candidate_on_strict_receipt_fields"
                } else {
                    "candidate_on_strict_warm_session_receipt_artifact"
                };
                (
                    "candidate_on_behavior_evidence_gate_defined_fail_closed",
                    "candidate_on_behavior_evidence_or_runtime_attachment_incomplete",
                    blocker,
                )
            } else {
                (
                    "blocked_fail_closed",
                    "candidate_off_on_pair_gate_incomplete",
                    "candidate_off_on_receipt_pair_gate",
                )
            };

        Self {
            tensor_name: pair_gate.tensor_name.clone(),
            callsite_identity: pair_gate.callsite_identity.clone(),
            model_sha256: pair_gate.model_sha256.clone(),
            model_architecture: pair_gate.model_architecture,
            quant_format: pair_gate.quant_format,
            tokenizer_source: pair_gate.tokenizer_source,
            tokenizer_strict: pair_gate.tokenizer_strict,
            runtime_api: pair_gate.runtime_api,
            selected_backend: pair_gate.selected_backend,
            fallback_used: pair_gate.fallback_used,
            selected_path: pair_gate.selected_path,
            selected_kernel: pair_gate.selected_kernel,
            candidate_path: pair_gate.candidate_path,
            candidate_kernel: pair_gate.candidate_kernel,
            prompt_ids_digest: pair_gate.prompt_ids_digest.clone(),
            generated_ids_digest: pair_gate.generated_ids_digest.clone(),
            decoded_text_digest: pair_gate.decoded_text_digest.clone(),
            candidate_off_on_pair_gate_ready,
            candidate_on_behavior_evidence_present,
            candidate_on_runtime_attachment_point_present,
            candidate_on_receipt_fields_complete,
            explicit_runtime_gate_requested: pair_gate.explicit_runtime_gate_requested,
            prompt_ids_preserved: pair_gate.prompt_ids_preserved,
            generated_ids_preserved: pair_gate.generated_ids_preserved,
            decoded_text_preserved: pair_gate.decoded_text_preserved,
            default_runtime_path_preserved,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasCandidateRuntimeAttachmentBoundary {
    pub fn from_candidate_on_behavior_gate(
        gate: &DenseLinearNoBiasCandidateOnBehaviorEvidenceGate,
        explicit_runtime_gate_requested: bool,
        apply_linear_candidate_attachment_wired: bool,
        candidate_runtime_owner_present: bool,
        candidate_receipt_emitter_wired: bool,
        candidate_compute_callable: bool,
    ) -> Self {
        let mut fail_closed_conditions = gate.fail_closed_conditions.clone();
        let candidate_on_behavior_gate_ready = gate.decision
            == "candidate_on_behavior_evidence_ready_runtime_disabled"
            && gate.fail_closed_conditions.is_empty();
        let default_runtime_path_preserved = gate.preserves_normal_inference()
            && gate.default_runtime_path_preserved
            && gate.selected_path == "eager_f32_candle"
            && gate.selected_kernel == "dense-f32-candle-linear";

        if !candidate_on_behavior_gate_ready {
            fail_closed_conditions.push("candidate_on_behavior_gate_not_ready");
        }
        if !explicit_runtime_gate_requested {
            fail_closed_conditions.push("explicit_runtime_gate_not_requested");
        }
        if !apply_linear_candidate_attachment_wired {
            fail_closed_conditions.push("apply_linear_candidate_attachment_not_wired");
        }
        if !candidate_runtime_owner_present {
            fail_closed_conditions.push("candidate_runtime_owner_missing");
        }
        if !candidate_receipt_emitter_wired {
            fail_closed_conditions.push("candidate_receipt_emitter_not_wired");
        }
        if !candidate_compute_callable {
            fail_closed_conditions.push("candidate_compute_not_callable");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if gate.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if gate.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if gate.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let attachment_ready = candidate_on_behavior_gate_ready
            && explicit_runtime_gate_requested
            && apply_linear_candidate_attachment_wired
            && candidate_runtime_owner_present
            && candidate_receipt_emitter_wired
            && candidate_compute_callable
            && default_runtime_path_preserved
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) = if attachment_ready {
            (
                "candidate_runtime_attachment_ready_runtime_disabled",
                "apply_linear_candidate_runtime_attachment_preserves_default_runtime",
                "candidate_off_on_behavior_receipt_capture_must_consume_attachment_boundary",
            )
        } else if candidate_on_behavior_gate_ready {
            let blocker = if !candidate_runtime_owner_present {
                "candidate_runtime_owner"
            } else if !candidate_receipt_emitter_wired {
                "candidate_on_receipt_emitter"
            } else if !apply_linear_candidate_attachment_wired {
                "apply_linear_candidate_attachment"
            } else {
                "candidate_compute_callable"
            };
            (
                "candidate_runtime_attachment_defined_fail_closed",
                "apply_linear_runtime_ownership_or_receipt_emission_incomplete",
                blocker,
            )
        } else {
            (
                "blocked_fail_closed",
                "candidate_on_behavior_gate_incomplete",
                "candidate_on_behavior_evidence_gate",
            )
        };

        Self {
            tensor_name: gate.tensor_name.clone(),
            callsite_identity: gate.callsite_identity.clone(),
            model_sha256: gate.model_sha256.clone(),
            model_architecture: gate.model_architecture,
            quant_format: gate.quant_format,
            tokenizer_source: gate.tokenizer_source,
            tokenizer_strict: gate.tokenizer_strict,
            runtime_api: gate.runtime_api,
            selected_backend: gate.selected_backend,
            fallback_used: gate.fallback_used,
            selected_path: gate.selected_path,
            selected_kernel: gate.selected_kernel,
            candidate_path: gate.candidate_path,
            candidate_kernel: gate.candidate_kernel,
            prompt_ids_digest: gate.prompt_ids_digest.clone(),
            generated_ids_digest: gate.generated_ids_digest.clone(),
            decoded_text_digest: gate.decoded_text_digest.clone(),
            candidate_on_behavior_gate_ready,
            explicit_runtime_gate_requested,
            apply_linear_candidate_attachment_wired,
            candidate_runtime_owner_present,
            candidate_receipt_emitter_wired,
            candidate_compute_callable,
            default_runtime_path_preserved,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasCandidateRuntimeOwnerBoundary {
    pub fn from_runtime_attachment_boundary(
        attachment: &DenseLinearNoBiasCandidateRuntimeAttachmentBoundary,
        inputs: DenseLinearNoBiasCandidateRuntimeOwnerInputs,
    ) -> Self {
        let mut fail_closed_conditions: Vec<_> = attachment
            .fail_closed_conditions
            .iter()
            .copied()
            .filter(|condition| {
                !matches!(
                    *condition,
                    "candidate_runtime_owner_missing"
                        | "candidate_receipt_emitter_not_wired"
                        | "candidate_compute_not_callable"
                )
            })
            .collect();
        let candidate_runtime_attachment_boundary_defined = matches!(
            attachment.decision,
            "candidate_runtime_attachment_ready_runtime_disabled"
                | "candidate_runtime_attachment_defined_fail_closed"
        ) && attachment
            .candidate_on_behavior_gate_ready
            && attachment.explicit_runtime_gate_requested
            && attachment.apply_linear_candidate_attachment_wired;
        let default_runtime_path_preserved = attachment.preserves_normal_inference()
            && attachment.default_runtime_path_preserved
            && attachment.selected_path == "eager_f32_candle"
            && attachment.selected_kernel == "dense-f32-candle-linear";

        if !candidate_runtime_attachment_boundary_defined {
            fail_closed_conditions.push("candidate_runtime_attachment_boundary_not_defined");
        }
        if !attachment.explicit_runtime_gate_requested {
            fail_closed_conditions.push("explicit_runtime_gate_not_requested");
        }
        if !inputs.apply_linear_runtime_owner_present {
            fail_closed_conditions.push("apply_linear_runtime_owner_missing");
        }
        if !inputs.owner_has_apply_linear_inputs {
            fail_closed_conditions.push("runtime_owner_missing_apply_linear_inputs");
        }
        if !inputs.owner_has_linear_weight_access {
            fail_closed_conditions.push("runtime_owner_missing_linear_weight_access");
        }
        if !inputs.candidate_compute_callable {
            fail_closed_conditions.push("candidate_compute_not_callable");
        }
        if !inputs.same_callsite_candidate_on_receipt_emitter_wired {
            fail_closed_conditions.push("same_callsite_candidate_on_receipt_emitter_missing");
        }
        if !inputs.candidate_off_on_strict_receipts_present {
            fail_closed_conditions.push("candidate_off_on_strict_receipts_missing");
        }
        if !inputs.prompt_ids_preserved {
            fail_closed_conditions.push("prompt_ids_preservation_receipt_missing");
        }
        if !inputs.generated_ids_preserved {
            fail_closed_conditions.push("generated_ids_preservation_receipt_missing");
        }
        if !inputs.decoded_text_preserved {
            fail_closed_conditions.push("decoded_text_preservation_receipt_missing");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if attachment.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if attachment.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if attachment.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let owner_ready = candidate_runtime_attachment_boundary_defined
            && attachment.explicit_runtime_gate_requested
            && inputs.apply_linear_runtime_owner_present
            && inputs.owner_has_apply_linear_inputs
            && inputs.owner_has_linear_weight_access
            && inputs.candidate_compute_callable
            && default_runtime_path_preserved
            && attachment.runtime_api == "cpu"
            && attachment.selected_backend == "cpu-rust"
            && !attachment.fallback_used;
        let receipt_ready = owner_ready
            && inputs.same_callsite_candidate_on_receipt_emitter_wired
            && inputs.candidate_off_on_strict_receipts_present
            && inputs.prompt_ids_preserved
            && inputs.generated_ids_preserved
            && inputs.decoded_text_preserved
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) = if receipt_ready {
            (
                "candidate_runtime_owner_and_receipt_emitter_ready_runtime_disabled",
                "same_callsite_candidate_on_receipt_emission_preserves_default_runtime",
                "candidate_execution_enablement_requires_explicit_receipt_gated_pr",
            )
        } else if owner_ready {
            let blocker = if !inputs.same_callsite_candidate_on_receipt_emitter_wired {
                "same_callsite_candidate_on_receipt_emitter"
            } else if !inputs.candidate_off_on_strict_receipts_present {
                "candidate_off_on_strict_receipts"
            } else {
                "prompt_generated_text_preservation_receipts"
            };
            (
                "candidate_runtime_owner_defined_fail_closed",
                "same_callsite_candidate_on_receipt_emission_incomplete",
                blocker,
            )
        } else if candidate_runtime_attachment_boundary_defined {
            let blocker = if !inputs.apply_linear_runtime_owner_present {
                "apply_linear_runtime_owner"
            } else if !inputs.owner_has_apply_linear_inputs {
                "apply_linear_input_ownership"
            } else if !inputs.owner_has_linear_weight_access {
                "linear_weight_ownership"
            } else {
                "candidate_compute_callable"
            };
            (
                "candidate_runtime_owner_blocked_fail_closed",
                "apply_linear_runtime_owner_incomplete",
                blocker,
            )
        } else {
            (
                "blocked_fail_closed",
                "candidate_runtime_attachment_boundary_incomplete",
                "candidate_runtime_attachment_boundary",
            )
        };

        Self {
            tensor_name: attachment.tensor_name.clone(),
            callsite_identity: attachment.callsite_identity.clone(),
            model_sha256: attachment.model_sha256.clone(),
            model_architecture: attachment.model_architecture,
            quant_format: attachment.quant_format,
            tokenizer_source: attachment.tokenizer_source,
            tokenizer_strict: attachment.tokenizer_strict,
            runtime_api: attachment.runtime_api,
            selected_backend: attachment.selected_backend,
            fallback_used: attachment.fallback_used,
            selected_path: attachment.selected_path,
            selected_kernel: attachment.selected_kernel,
            candidate_path: attachment.candidate_path,
            candidate_kernel: attachment.candidate_kernel,
            prompt_ids_digest: attachment.prompt_ids_digest.clone(),
            generated_ids_digest: attachment.generated_ids_digest.clone(),
            decoded_text_digest: attachment.decoded_text_digest.clone(),
            candidate_runtime_attachment_boundary_defined,
            explicit_runtime_gate_requested: attachment.explicit_runtime_gate_requested,
            apply_linear_runtime_owner_present: inputs.apply_linear_runtime_owner_present,
            owner_has_apply_linear_inputs: inputs.owner_has_apply_linear_inputs,
            owner_has_linear_weight_access: inputs.owner_has_linear_weight_access,
            candidate_compute_callable: inputs.candidate_compute_callable,
            same_callsite_candidate_on_receipt_emitter_wired: inputs
                .same_callsite_candidate_on_receipt_emitter_wired,
            candidate_off_on_strict_receipts_present: inputs
                .candidate_off_on_strict_receipts_present,
            prompt_ids_preserved: inputs.prompt_ids_preserved,
            generated_ids_preserved: inputs.generated_ids_preserved,
            decoded_text_preserved: inputs.decoded_text_preserved,
            default_runtime_path_preserved,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary {
    pub fn from_runtime_owner_boundary(
        owner: &DenseLinearNoBiasCandidateRuntimeOwnerBoundary,
        inputs: DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs,
    ) -> Self {
        let mut fail_closed_conditions: Vec<_> = owner
            .fail_closed_conditions
            .iter()
            .copied()
            .filter(|condition| {
                !matches!(
                    *condition,
                    "same_callsite_candidate_on_receipt_emitter_missing"
                        | "candidate_off_on_strict_receipts_missing"
                        | "prompt_ids_preservation_receipt_missing"
                        | "generated_ids_preservation_receipt_missing"
                        | "decoded_text_preservation_receipt_missing"
                )
            })
            .collect();
        let runtime_owner_boundary_defined = matches!(
            owner.decision,
            "candidate_runtime_owner_defined_fail_closed"
                | "candidate_runtime_owner_and_receipt_emitter_ready_runtime_disabled"
        ) && owner
            .candidate_runtime_attachment_boundary_defined
            && owner.apply_linear_runtime_owner_present
            && owner.owner_has_apply_linear_inputs
            && owner.owner_has_linear_weight_access
            && owner.candidate_compute_callable;
        let default_runtime_path_preserved = owner.preserves_normal_inference()
            && owner.default_runtime_path_preserved
            && owner.selected_path == "eager_f32_candle"
            && owner.selected_kernel == "dense-f32-candle-linear";

        if !runtime_owner_boundary_defined {
            fail_closed_conditions.push("candidate_runtime_owner_boundary_not_defined");
        }
        if !inputs.same_callsite_candidate_receipt_emitter_present {
            fail_closed_conditions.push("same_callsite_candidate_receipt_emitter_missing");
        }
        if !inputs.candidate_off_strict_receipt_present {
            fail_closed_conditions.push("candidate_off_strict_receipt_missing");
        }
        if !inputs.candidate_on_strict_receipt_present {
            fail_closed_conditions.push("candidate_on_strict_receipt_missing");
        }
        if !inputs.strict_receipts_bind_owner_identity {
            fail_closed_conditions.push("strict_receipts_do_not_bind_runtime_owner_identity");
        }
        if !inputs.prompt_ids_preserved {
            fail_closed_conditions.push("prompt_ids_preservation_receipt_missing");
        }
        if !inputs.generated_ids_preserved {
            fail_closed_conditions.push("generated_ids_preservation_receipt_missing");
        }
        if !inputs.decoded_text_preserved {
            fail_closed_conditions.push("decoded_text_preservation_receipt_missing");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if owner.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if owner.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if owner.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let owner_ready = runtime_owner_boundary_defined
            && default_runtime_path_preserved
            && owner.runtime_api == "cpu"
            && owner.selected_backend == "cpu-rust"
            && !owner.fallback_used;
        let receipt_ready = owner_ready
            && inputs.same_callsite_candidate_receipt_emitter_present
            && inputs.candidate_off_strict_receipt_present
            && inputs.candidate_on_strict_receipt_present
            && inputs.strict_receipts_bind_owner_identity
            && inputs.prompt_ids_preserved
            && inputs.generated_ids_preserved
            && inputs.decoded_text_preserved
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) = if receipt_ready {
            (
                "same_callsite_candidate_receipt_emitter_ready_runtime_disabled",
                "same_callsite_candidate_off_on_receipts_preserve_runtime_owner_identity",
                "candidate_execution_enablement_requires_explicit_receipt_gated_pr",
            )
        } else if owner_ready && inputs.same_callsite_candidate_receipt_emitter_present {
            let blocker = if !inputs.candidate_off_strict_receipt_present
                || !inputs.candidate_on_strict_receipt_present
            {
                "fresh_candidate_off_on_strict_receipts"
            } else if !inputs.strict_receipts_bind_owner_identity {
                "strict_receipts_bind_owner_identity"
            } else {
                "prompt_generated_text_preservation_receipts"
            };
            (
                "same_callsite_candidate_receipt_emitter_defined_fail_closed",
                "same_callsite_candidate_off_on_receipts_incomplete",
                blocker,
            )
        } else if owner_ready {
            (
                "same_callsite_candidate_receipt_emitter_blocked_fail_closed",
                "same_callsite_candidate_receipt_emitter_missing",
                "same_callsite_candidate_receipt_emitter",
            )
        } else {
            (
                "blocked_fail_closed",
                "candidate_runtime_owner_boundary_incomplete",
                "candidate_runtime_owner_boundary",
            )
        };

        Self {
            tensor_name: owner.tensor_name.clone(),
            callsite_identity: owner.callsite_identity.clone(),
            model_sha256: owner.model_sha256.clone(),
            model_architecture: owner.model_architecture,
            quant_format: owner.quant_format,
            tokenizer_source: owner.tokenizer_source,
            tokenizer_strict: owner.tokenizer_strict,
            runtime_api: owner.runtime_api,
            selected_backend: owner.selected_backend,
            fallback_used: owner.fallback_used,
            selected_path: owner.selected_path,
            selected_kernel: owner.selected_kernel,
            candidate_path: owner.candidate_path,
            candidate_kernel: owner.candidate_kernel,
            prompt_ids_digest: owner.prompt_ids_digest.clone(),
            generated_ids_digest: owner.generated_ids_digest.clone(),
            decoded_text_digest: owner.decoded_text_digest.clone(),
            runtime_owner_boundary_defined,
            apply_linear_runtime_owner_present: owner.apply_linear_runtime_owner_present,
            owner_has_apply_linear_inputs: owner.owner_has_apply_linear_inputs,
            owner_has_linear_weight_access: owner.owner_has_linear_weight_access,
            candidate_compute_callable: owner.candidate_compute_callable,
            same_callsite_candidate_receipt_emitter_present: inputs
                .same_callsite_candidate_receipt_emitter_present,
            candidate_off_strict_receipt_present: inputs.candidate_off_strict_receipt_present,
            candidate_on_strict_receipt_present: inputs.candidate_on_strict_receipt_present,
            strict_receipts_bind_owner_identity: inputs.strict_receipts_bind_owner_identity,
            prompt_ids_preserved: inputs.prompt_ids_preserved,
            generated_ids_preserved: inputs.generated_ids_preserved,
            decoded_text_preserved: inputs.decoded_text_preserved,
            default_runtime_path_preserved,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary {
    pub fn from_same_callsite_receipt_emitter_boundary(
        emitter: &DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary,
        inputs: DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptInputs,
    ) -> Self {
        let mut fail_closed_conditions: Vec<_> = emitter
            .fail_closed_conditions
            .iter()
            .copied()
            .filter(|condition| {
                !matches!(
                    *condition,
                    "candidate_off_strict_receipt_missing"
                        | "candidate_on_strict_receipt_missing"
                        | "strict_receipts_do_not_bind_runtime_owner_identity"
                        | "prompt_ids_preservation_receipt_missing"
                        | "generated_ids_preservation_receipt_missing"
                        | "decoded_text_preservation_receipt_missing"
                )
            })
            .collect();
        let same_callsite_receipt_emitter_ready = matches!(
            emitter.decision,
            "same_callsite_candidate_receipt_emitter_defined_fail_closed"
                | "same_callsite_candidate_receipt_emitter_ready_runtime_disabled"
        ) && emitter.runtime_owner_boundary_defined
            && emitter.same_callsite_candidate_receipt_emitter_present
            && emitter.apply_linear_runtime_owner_present
            && emitter.owner_has_apply_linear_inputs
            && emitter.owner_has_linear_weight_access
            && emitter.candidate_compute_callable;
        let default_runtime_path_preserved = emitter.preserves_normal_inference()
            && emitter.default_runtime_path_preserved
            && emitter.selected_path == "eager_f32_candle"
            && emitter.selected_kernel == "dense-f32-candle-linear";

        if !same_callsite_receipt_emitter_ready {
            fail_closed_conditions.push("same_callsite_receipt_emitter_boundary_not_ready");
        }
        if !inputs.candidate_off_strict_receipt_artifact_present {
            fail_closed_conditions.push("candidate_off_strict_receipt_artifact_missing");
        }
        if !inputs.candidate_on_strict_receipt_artifact_present {
            fail_closed_conditions.push("candidate_on_strict_receipt_artifact_missing");
        }
        if !inputs.candidate_off_receipt_binds_owner_identity {
            fail_closed_conditions.push("candidate_off_receipt_does_not_bind_owner_identity");
        }
        if !inputs.candidate_on_receipt_binds_owner_identity {
            fail_closed_conditions.push("candidate_on_receipt_does_not_bind_owner_identity");
        }
        if !inputs.candidate_off_on_same_callsite_identity {
            fail_closed_conditions.push("candidate_off_on_callsite_identity_mismatch");
        }
        if !inputs.prompt_ids_preserved {
            fail_closed_conditions.push("prompt_ids_preservation_receipt_missing");
        }
        if !inputs.generated_ids_preserved {
            fail_closed_conditions.push("generated_ids_preservation_receipt_missing");
        }
        if !inputs.decoded_text_preserved {
            fail_closed_conditions.push("decoded_text_preservation_receipt_missing");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if emitter.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if emitter.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if emitter.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let receipt_pair_ready = same_callsite_receipt_emitter_ready
            && inputs.candidate_off_strict_receipt_artifact_present
            && inputs.candidate_on_strict_receipt_artifact_present
            && inputs.candidate_off_receipt_binds_owner_identity
            && inputs.candidate_on_receipt_binds_owner_identity
            && inputs.candidate_off_on_same_callsite_identity
            && inputs.prompt_ids_preserved
            && inputs.generated_ids_preserved
            && inputs.decoded_text_preserved
            && default_runtime_path_preserved
            && emitter.runtime_api == "cpu"
            && emitter.selected_backend == "cpu-rust"
            && !emitter.fallback_used
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) = if receipt_pair_ready {
            (
                "same_callsite_candidate_off_on_strict_receipts_ready_runtime_disabled",
                "same_callsite_candidate_off_on_receipts_bind_owner_identity_and_preserve_outputs",
                "candidate_execution_enablement_requires_explicit_receipt_gated_pr",
            )
        } else if same_callsite_receipt_emitter_ready {
            let blocker = if !inputs.candidate_on_strict_receipt_artifact_present {
                "candidate_on_strict_receipt_artifact"
            } else if !inputs.candidate_off_strict_receipt_artifact_present {
                "candidate_off_strict_receipt_artifact"
            } else if !inputs.candidate_off_receipt_binds_owner_identity
                || !inputs.candidate_on_receipt_binds_owner_identity
            {
                "strict_receipts_bind_owner_identity"
            } else if !inputs.candidate_off_on_same_callsite_identity {
                "candidate_off_on_same_callsite_identity"
            } else {
                "prompt_generated_text_preservation_receipts"
            };
            (
                "same_callsite_candidate_off_on_strict_receipts_blocked_fail_closed",
                "same_callsite_candidate_off_on_strict_receipt_artifacts_incomplete",
                blocker,
            )
        } else {
            (
                "blocked_fail_closed",
                "same_callsite_receipt_emitter_boundary_incomplete",
                "same_callsite_candidate_receipt_emitter_boundary",
            )
        };

        Self {
            tensor_name: emitter.tensor_name.clone(),
            callsite_identity: emitter.callsite_identity.clone(),
            model_sha256: emitter.model_sha256.clone(),
            model_architecture: emitter.model_architecture,
            quant_format: emitter.quant_format,
            tokenizer_source: emitter.tokenizer_source,
            tokenizer_strict: emitter.tokenizer_strict,
            runtime_api: emitter.runtime_api,
            selected_backend: emitter.selected_backend,
            fallback_used: emitter.fallback_used,
            selected_path: emitter.selected_path,
            selected_kernel: emitter.selected_kernel,
            candidate_path: emitter.candidate_path,
            candidate_kernel: emitter.candidate_kernel,
            prompt_ids_digest: emitter.prompt_ids_digest.clone(),
            generated_ids_digest: emitter.generated_ids_digest.clone(),
            decoded_text_digest: emitter.decoded_text_digest.clone(),
            same_callsite_receipt_emitter_ready,
            candidate_off_strict_receipt_artifact_present: inputs
                .candidate_off_strict_receipt_artifact_present,
            candidate_on_strict_receipt_artifact_present: inputs
                .candidate_on_strict_receipt_artifact_present,
            candidate_off_receipt_binds_owner_identity: inputs
                .candidate_off_receipt_binds_owner_identity,
            candidate_on_receipt_binds_owner_identity: inputs
                .candidate_on_receipt_binds_owner_identity,
            candidate_off_on_same_callsite_identity: inputs.candidate_off_on_same_callsite_identity,
            prompt_ids_preserved: inputs.prompt_ids_preserved,
            generated_ids_preserved: inputs.generated_ids_preserved,
            decoded_text_preserved: inputs.decoded_text_preserved,
            default_runtime_path_preserved,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary {
    pub fn from_off_on_strict_receipt_boundary(
        boundary: &DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary,
        inputs: DenseLinearNoBiasReceiptGatedCandidateExecutionInputs,
    ) -> Self {
        let mut fail_closed_conditions = boundary.fail_closed_conditions.clone();
        let off_on_strict_receipt_boundary_ready = boundary.decision
            == "same_callsite_candidate_off_on_strict_receipts_ready_runtime_disabled"
            && boundary.same_callsite_receipt_emitter_ready
            && boundary.candidate_off_strict_receipt_artifact_present
            && boundary.candidate_on_strict_receipt_artifact_present
            && boundary.candidate_off_receipt_binds_owner_identity
            && boundary.candidate_on_receipt_binds_owner_identity
            && boundary.candidate_off_on_same_callsite_identity
            && boundary.prompt_ids_preserved
            && boundary.generated_ids_preserved
            && boundary.decoded_text_preserved
            && boundary.preserves_normal_inference();
        let prompt_generated_text_digests_bound = inputs.prompt_generated_text_digests_bound
            && !boundary.prompt_ids_digest.is_empty()
            && !boundary.generated_ids_digest.is_empty()
            && !boundary.decoded_text_digest.is_empty();
        let default_runtime_path_preserved = inputs.default_runtime_path_preserved
            && boundary.default_runtime_path_preserved
            && boundary.selected_path == "eager_f32_candle"
            && boundary.selected_kernel == "dense-f32-candle-linear"
            && boundary.preserves_normal_inference();

        if !off_on_strict_receipt_boundary_ready {
            fail_closed_conditions.push("off_on_strict_receipt_boundary_not_ready");
        }
        if !inputs.explicit_gate_identity_present {
            fail_closed_conditions.push("explicit_gate_identity_missing");
        }
        if !inputs.descriptor_identity_present {
            fail_closed_conditions.push("descriptor_identity_missing");
        }
        if !inputs.owner_callsite_identity_present {
            fail_closed_conditions.push("owner_callsite_identity_missing");
        }
        if !prompt_generated_text_digests_bound {
            fail_closed_conditions.push("prompt_generated_text_digests_not_bound");
        }
        if !inputs.explicit_candidate_execution_gate_requested {
            fail_closed_conditions.push("explicit_candidate_execution_gate_not_requested");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if boundary.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if boundary.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if boundary.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let candidate_execution_attempt_allowed = off_on_strict_receipt_boundary_ready
            && inputs.explicit_gate_identity_present
            && inputs.descriptor_identity_present
            && inputs.owner_callsite_identity_present
            && prompt_generated_text_digests_bound
            && inputs.explicit_candidate_execution_gate_requested
            && default_runtime_path_preserved
            && boundary.runtime_api == "cpu"
            && boundary.selected_backend == "cpu-rust"
            && !boundary.fallback_used
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) =
            if candidate_execution_attempt_allowed {
                (
                    "receipt_gated_candidate_execution_prereqs_ready_runtime_disabled",
                    "strict_identity_artifacts_allow_candidate_execution_attempt_but_runtime_remains_disabled",
                    "candidate_execution_enablement_requires_separate_receipt_gated_pr",
                )
            } else if off_on_strict_receipt_boundary_ready {
                let blocker = if !inputs.explicit_candidate_execution_gate_requested {
                    "explicit_candidate_execution_gate"
                } else if !inputs.explicit_gate_identity_present {
                    "explicit_gate_identity"
                } else if !inputs.descriptor_identity_present {
                    "descriptor_identity"
                } else if !inputs.owner_callsite_identity_present {
                    "owner_callsite_identity"
                } else if !prompt_generated_text_digests_bound {
                    "prompt_generated_text_digests"
                } else {
                    "default_runtime_path_preservation"
                };
                (
                    "receipt_gated_candidate_execution_blocked_fail_closed",
                    "strict_identity_artifacts_incomplete_for_candidate_execution_attempt",
                    blocker,
                )
            } else {
                (
                    "blocked_fail_closed",
                    "off_on_strict_receipt_boundary_incomplete",
                    "same_callsite_candidate_off_on_strict_receipts",
                )
            };

        Self {
            tensor_name: boundary.tensor_name.clone(),
            callsite_identity: boundary.callsite_identity.clone(),
            model_sha256: boundary.model_sha256.clone(),
            model_architecture: boundary.model_architecture,
            quant_format: boundary.quant_format,
            tokenizer_source: boundary.tokenizer_source,
            tokenizer_strict: boundary.tokenizer_strict,
            runtime_api: boundary.runtime_api,
            selected_backend: boundary.selected_backend,
            fallback_used: boundary.fallback_used,
            selected_path: boundary.selected_path,
            selected_kernel: boundary.selected_kernel,
            candidate_path: boundary.candidate_path,
            candidate_kernel: boundary.candidate_kernel,
            prompt_ids_digest: boundary.prompt_ids_digest.clone(),
            generated_ids_digest: boundary.generated_ids_digest.clone(),
            decoded_text_digest: boundary.decoded_text_digest.clone(),
            off_on_strict_receipt_boundary_ready,
            explicit_gate_identity_present: inputs.explicit_gate_identity_present,
            descriptor_identity_present: inputs.descriptor_identity_present,
            owner_callsite_identity_present: inputs.owner_callsite_identity_present,
            prompt_generated_text_digests_bound,
            default_runtime_path_preserved,
            candidate_execution_attempt_allowed,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasStrictReceiptArtifactPairBoundary {
    pub fn from_receipt_gated_candidate_execution_boundary(
        boundary: &DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary,
        inputs: DenseLinearNoBiasStrictReceiptArtifactPairInputs,
    ) -> Self {
        let mut fail_closed_conditions = boundary.fail_closed_conditions.clone();
        let receipt_gated_candidate_execution_boundary_ready = boundary.decision
            == "receipt_gated_candidate_execution_prereqs_ready_runtime_disabled"
            && boundary.candidate_execution_attempt_allowed
            && boundary.explicit_gate_identity_present
            && boundary.descriptor_identity_present
            && boundary.owner_callsite_identity_present
            && boundary.prompt_generated_text_digests_bound
            && boundary.preserves_normal_inference();

        let candidate_off_strict_receipt_artifact_path = inputs
            .candidate_off_strict_receipt_artifact_path
            .filter(|path| !path.is_empty())
            .map(str::to_owned);
        let candidate_on_strict_receipt_artifact_path = inputs
            .candidate_on_strict_receipt_artifact_path
            .filter(|path| !path.is_empty())
            .map(str::to_owned);
        let candidate_off_strict_receipt_artifact_present = inputs
            .candidate_off_strict_receipt_artifact_present
            && candidate_off_strict_receipt_artifact_path.is_some();
        let candidate_on_strict_receipt_artifact_present = inputs
            .candidate_on_strict_receipt_artifact_present
            && candidate_on_strict_receipt_artifact_path.is_some();
        let default_runtime_path_preserved = inputs.default_runtime_path_preserved
            && boundary.default_runtime_path_preserved
            && boundary.selected_path == "eager_f32_candle"
            && boundary.selected_kernel == "dense-f32-candle-linear"
            && boundary.preserves_normal_inference();

        if !receipt_gated_candidate_execution_boundary_ready {
            fail_closed_conditions.push("receipt_gated_candidate_execution_boundary_not_ready");
        }
        if !candidate_off_strict_receipt_artifact_present {
            fail_closed_conditions.push("candidate_off_strict_receipt_artifact_missing");
        }
        if !candidate_on_strict_receipt_artifact_present {
            fail_closed_conditions.push("candidate_on_strict_receipt_artifact_missing");
        }
        if !inputs.candidate_off_receipt_binds_gate_identity {
            fail_closed_conditions.push("candidate_off_receipt_does_not_bind_gate_identity");
        }
        if !inputs.candidate_on_receipt_binds_gate_identity {
            fail_closed_conditions.push("candidate_on_receipt_does_not_bind_gate_identity");
        }
        if !inputs.candidate_off_receipt_binds_descriptor_identity {
            fail_closed_conditions.push("candidate_off_receipt_does_not_bind_descriptor_identity");
        }
        if !inputs.candidate_on_receipt_binds_descriptor_identity {
            fail_closed_conditions.push("candidate_on_receipt_does_not_bind_descriptor_identity");
        }
        if !inputs.candidate_off_receipt_binds_owner_callsite_identity {
            fail_closed_conditions
                .push("candidate_off_receipt_does_not_bind_owner_callsite_identity");
        }
        if !inputs.candidate_on_receipt_binds_owner_callsite_identity {
            fail_closed_conditions
                .push("candidate_on_receipt_does_not_bind_owner_callsite_identity");
        }
        if !inputs.candidate_off_on_same_callsite_identity {
            fail_closed_conditions.push("candidate_off_on_callsite_identity_mismatch");
        }
        if !inputs.candidate_off_on_same_prompt_digest {
            fail_closed_conditions.push("candidate_off_on_prompt_digest_mismatch");
        }
        if !inputs.candidate_off_on_same_generated_digest {
            fail_closed_conditions.push("candidate_off_on_generated_digest_mismatch");
        }
        if !inputs.candidate_off_on_same_decoded_text_digest {
            fail_closed_conditions.push("candidate_off_on_decoded_text_digest_mismatch");
        }
        if !inputs.candidate_off_on_same_model_backend_identity {
            fail_closed_conditions.push("candidate_off_on_model_backend_identity_mismatch");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if boundary.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if boundary.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if boundary.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let candidate_execution_attempt_allowed = receipt_gated_candidate_execution_boundary_ready
            && candidate_off_strict_receipt_artifact_present
            && candidate_on_strict_receipt_artifact_present
            && inputs.candidate_off_receipt_binds_gate_identity
            && inputs.candidate_on_receipt_binds_gate_identity
            && inputs.candidate_off_receipt_binds_descriptor_identity
            && inputs.candidate_on_receipt_binds_descriptor_identity
            && inputs.candidate_off_receipt_binds_owner_callsite_identity
            && inputs.candidate_on_receipt_binds_owner_callsite_identity
            && inputs.candidate_off_on_same_callsite_identity
            && inputs.candidate_off_on_same_prompt_digest
            && inputs.candidate_off_on_same_generated_digest
            && inputs.candidate_off_on_same_decoded_text_digest
            && inputs.candidate_off_on_same_model_backend_identity
            && default_runtime_path_preserved
            && boundary.runtime_api == "cpu"
            && boundary.selected_backend == "cpu-rust"
            && !boundary.fallback_used
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) =
            if candidate_execution_attempt_allowed {
                (
                    "same_callsite_strict_receipt_artifact_pair_ready_runtime_disabled",
                    "strict_artifact_pair_binds_gate_descriptor_owner_and_preserves_outputs",
                    "candidate_execution_enablement_requires_separate_receipt_gated_pr",
                )
            } else if receipt_gated_candidate_execution_boundary_ready {
                let blocker = if !candidate_on_strict_receipt_artifact_present {
                    "candidate_on_strict_receipt_artifact"
                } else if !candidate_off_strict_receipt_artifact_present {
                    "candidate_off_strict_receipt_artifact"
                } else if !inputs.candidate_off_receipt_binds_gate_identity
                    || !inputs.candidate_on_receipt_binds_gate_identity
                {
                    "strict_receipts_bind_gate_identity"
                } else if !inputs.candidate_off_receipt_binds_descriptor_identity
                    || !inputs.candidate_on_receipt_binds_descriptor_identity
                {
                    "strict_receipts_bind_descriptor_identity"
                } else if !inputs.candidate_off_receipt_binds_owner_callsite_identity
                    || !inputs.candidate_on_receipt_binds_owner_callsite_identity
                    || !inputs.candidate_off_on_same_callsite_identity
                {
                    "strict_receipts_bind_owner_callsite_identity"
                } else if !inputs.candidate_off_on_same_model_backend_identity {
                    "strict_receipts_bind_model_backend_identity"
                } else if !inputs.candidate_off_on_same_prompt_digest
                    || !inputs.candidate_off_on_same_generated_digest
                    || !inputs.candidate_off_on_same_decoded_text_digest
                {
                    "strict_receipts_preserve_prompt_generated_text_digests"
                } else {
                    "default_runtime_path_preservation"
                };
                (
                    "same_callsite_strict_receipt_artifact_pair_blocked_fail_closed",
                    "same_callsite_strict_receipt_artifact_pair_incomplete",
                    blocker,
                )
            } else {
                (
                    "blocked_fail_closed",
                    "receipt_gated_candidate_execution_boundary_incomplete",
                    "receipt_gated_candidate_execution_boundary",
                )
            };

        Self {
            tensor_name: boundary.tensor_name.clone(),
            callsite_identity: boundary.callsite_identity.clone(),
            model_sha256: boundary.model_sha256.clone(),
            model_architecture: boundary.model_architecture,
            quant_format: boundary.quant_format,
            tokenizer_source: boundary.tokenizer_source,
            tokenizer_strict: boundary.tokenizer_strict,
            runtime_api: boundary.runtime_api,
            selected_backend: boundary.selected_backend,
            fallback_used: boundary.fallback_used,
            selected_path: boundary.selected_path,
            selected_kernel: boundary.selected_kernel,
            candidate_path: boundary.candidate_path,
            candidate_kernel: boundary.candidate_kernel,
            prompt_ids_digest: boundary.prompt_ids_digest.clone(),
            generated_ids_digest: boundary.generated_ids_digest.clone(),
            decoded_text_digest: boundary.decoded_text_digest.clone(),
            receipt_gated_candidate_execution_boundary_ready,
            candidate_off_strict_receipt_artifact_path,
            candidate_on_strict_receipt_artifact_path,
            candidate_off_strict_receipt_artifact_present,
            candidate_on_strict_receipt_artifact_present,
            candidate_off_receipt_binds_gate_identity: inputs
                .candidate_off_receipt_binds_gate_identity,
            candidate_on_receipt_binds_gate_identity: inputs
                .candidate_on_receipt_binds_gate_identity,
            candidate_off_receipt_binds_descriptor_identity: inputs
                .candidate_off_receipt_binds_descriptor_identity,
            candidate_on_receipt_binds_descriptor_identity: inputs
                .candidate_on_receipt_binds_descriptor_identity,
            candidate_off_receipt_binds_owner_callsite_identity: inputs
                .candidate_off_receipt_binds_owner_callsite_identity,
            candidate_on_receipt_binds_owner_callsite_identity: inputs
                .candidate_on_receipt_binds_owner_callsite_identity,
            candidate_off_on_same_callsite_identity: inputs.candidate_off_on_same_callsite_identity,
            candidate_off_on_same_prompt_digest: inputs.candidate_off_on_same_prompt_digest,
            candidate_off_on_same_generated_digest: inputs.candidate_off_on_same_generated_digest,
            candidate_off_on_same_decoded_text_digest: inputs
                .candidate_off_on_same_decoded_text_digest,
            candidate_off_on_same_model_backend_identity: inputs
                .candidate_off_on_same_model_backend_identity,
            default_runtime_path_preserved,
            candidate_execution_attempt_allowed,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasStrictArtifactCaptureBoundary {
    pub fn from_strict_receipt_artifact_pair_boundary(
        boundary: &DenseLinearNoBiasStrictReceiptArtifactPairBoundary,
        inputs: DenseLinearNoBiasStrictArtifactCaptureInputs,
    ) -> Self {
        let mut fail_closed_conditions = boundary.fail_closed_conditions.clone();
        let strict_receipt_artifact_pair_boundary_ready = boundary.decision
            == "same_callsite_strict_receipt_artifact_pair_ready_runtime_disabled"
            && boundary.candidate_execution_attempt_allowed
            && boundary.candidate_off_strict_receipt_artifact_present
            && boundary.candidate_on_strict_receipt_artifact_present
            && boundary.candidate_off_receipt_binds_gate_identity
            && boundary.candidate_on_receipt_binds_gate_identity
            && boundary.candidate_off_receipt_binds_descriptor_identity
            && boundary.candidate_on_receipt_binds_descriptor_identity
            && boundary.candidate_off_receipt_binds_owner_callsite_identity
            && boundary.candidate_on_receipt_binds_owner_callsite_identity
            && boundary.candidate_off_on_same_callsite_identity
            && boundary.candidate_off_on_same_prompt_digest
            && boundary.candidate_off_on_same_generated_digest
            && boundary.candidate_off_on_same_decoded_text_digest
            && boundary.candidate_off_on_same_model_backend_identity
            && boundary.preserves_normal_inference();
        let default_runtime_path_preserved = inputs.default_runtime_path_preserved
            && boundary.default_runtime_path_preserved
            && boundary.selected_path == "eager_f32_candle"
            && boundary.selected_kernel == "dense-f32-candle-linear"
            && boundary.preserves_normal_inference();

        if !strict_receipt_artifact_pair_boundary_ready {
            fail_closed_conditions.push("strict_receipt_artifact_pair_boundary_not_ready");
        }
        if !inputs.candidate_off_capture_artifact_validated {
            fail_closed_conditions.push("candidate_off_capture_artifact_not_validated");
        }
        if !inputs.candidate_on_capture_artifact_validated {
            fail_closed_conditions.push("candidate_on_capture_artifact_not_validated");
        }
        if !inputs.candidate_off_capture_command_recorded {
            fail_closed_conditions.push("candidate_off_capture_command_not_recorded");
        }
        if !inputs.candidate_on_capture_command_recorded {
            fail_closed_conditions.push("candidate_on_capture_command_not_recorded");
        }
        if !inputs.candidate_off_on_capture_same_callsite_identity {
            fail_closed_conditions.push("candidate_off_on_capture_callsite_identity_mismatch");
        }
        if !inputs.candidate_off_on_capture_same_prompt_digest {
            fail_closed_conditions.push("candidate_off_on_capture_prompt_digest_mismatch");
        }
        if !inputs.candidate_off_on_capture_same_generated_digest {
            fail_closed_conditions.push("candidate_off_on_capture_generated_digest_mismatch");
        }
        if !inputs.candidate_off_on_capture_same_decoded_text_digest {
            fail_closed_conditions.push("candidate_off_on_capture_decoded_text_digest_mismatch");
        }
        if !inputs.candidate_off_on_capture_same_model_backend_identity {
            fail_closed_conditions.push("candidate_off_on_capture_model_backend_identity_mismatch");
        }
        if !inputs.capture_blocker_recorded
            && (!strict_receipt_artifact_pair_boundary_ready
                || !inputs.candidate_off_capture_artifact_validated
                || !inputs.candidate_on_capture_artifact_validated)
        {
            fail_closed_conditions.push("strict_artifact_capture_blocker_not_recorded");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if boundary.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if boundary.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if boundary.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let candidate_execution_prereqs_complete = strict_receipt_artifact_pair_boundary_ready
            && inputs.candidate_off_capture_artifact_validated
            && inputs.candidate_on_capture_artifact_validated
            && inputs.candidate_off_capture_command_recorded
            && inputs.candidate_on_capture_command_recorded
            && inputs.candidate_off_on_capture_same_callsite_identity
            && inputs.candidate_off_on_capture_same_prompt_digest
            && inputs.candidate_off_on_capture_same_generated_digest
            && inputs.candidate_off_on_capture_same_decoded_text_digest
            && inputs.candidate_off_on_capture_same_model_backend_identity
            && default_runtime_path_preserved
            && boundary.runtime_api == "cpu"
            && boundary.selected_backend == "cpu-rust"
            && !boundary.fallback_used
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) =
            if candidate_execution_prereqs_complete {
                (
                    "same_callsite_strict_artifact_capture_ready_runtime_disabled",
                    "strict_capture_artifact_pair_validated_runtime_disabled",
                    "candidate_execution_enablement_requires_separate_receipt_gated_pr",
                )
            } else if strict_receipt_artifact_pair_boundary_ready {
                let blocker = if !inputs.candidate_on_capture_artifact_validated {
                    "candidate_on_capture_artifact"
                } else if !inputs.candidate_off_capture_artifact_validated {
                    "candidate_off_capture_artifact"
                } else if !inputs.candidate_off_capture_command_recorded
                    || !inputs.candidate_on_capture_command_recorded
                {
                    "candidate_off_on_capture_commands"
                } else if !inputs.candidate_off_on_capture_same_callsite_identity {
                    "candidate_off_on_capture_callsite_identity"
                } else if !inputs.candidate_off_on_capture_same_model_backend_identity {
                    "candidate_off_on_capture_model_backend_identity"
                } else if !inputs.candidate_off_on_capture_same_prompt_digest
                    || !inputs.candidate_off_on_capture_same_generated_digest
                    || !inputs.candidate_off_on_capture_same_decoded_text_digest
                {
                    "candidate_off_on_capture_prompt_generated_text_digests"
                } else if !inputs.capture_blocker_recorded {
                    "strict_artifact_capture_blocker_record"
                } else {
                    "default_runtime_path_preservation"
                };
                (
                    "same_callsite_strict_artifact_capture_blocked_fail_closed",
                    "same_callsite_strict_artifact_capture_incomplete",
                    blocker,
                )
            } else {
                (
                    "blocked_fail_closed",
                    "strict_receipt_artifact_pair_boundary_incomplete",
                    "strict_receipt_artifact_pair_boundary",
                )
            };

        Self {
            tensor_name: boundary.tensor_name.clone(),
            callsite_identity: boundary.callsite_identity.clone(),
            model_sha256: boundary.model_sha256.clone(),
            model_architecture: boundary.model_architecture,
            quant_format: boundary.quant_format,
            tokenizer_source: boundary.tokenizer_source,
            tokenizer_strict: boundary.tokenizer_strict,
            runtime_api: boundary.runtime_api,
            selected_backend: boundary.selected_backend,
            fallback_used: boundary.fallback_used,
            selected_path: boundary.selected_path,
            selected_kernel: boundary.selected_kernel,
            candidate_path: boundary.candidate_path,
            candidate_kernel: boundary.candidate_kernel,
            prompt_ids_digest: boundary.prompt_ids_digest.clone(),
            generated_ids_digest: boundary.generated_ids_digest.clone(),
            decoded_text_digest: boundary.decoded_text_digest.clone(),
            candidate_off_strict_receipt_artifact_path: boundary
                .candidate_off_strict_receipt_artifact_path
                .clone(),
            candidate_on_strict_receipt_artifact_path: boundary
                .candidate_on_strict_receipt_artifact_path
                .clone(),
            strict_receipt_artifact_pair_boundary_ready,
            candidate_off_capture_artifact_validated: inputs
                .candidate_off_capture_artifact_validated,
            candidate_on_capture_artifact_validated: inputs.candidate_on_capture_artifact_validated,
            candidate_off_capture_command_recorded: inputs.candidate_off_capture_command_recorded,
            candidate_on_capture_command_recorded: inputs.candidate_on_capture_command_recorded,
            candidate_off_on_capture_same_callsite_identity: inputs
                .candidate_off_on_capture_same_callsite_identity,
            candidate_off_on_capture_same_prompt_digest: inputs
                .candidate_off_on_capture_same_prompt_digest,
            candidate_off_on_capture_same_generated_digest: inputs
                .candidate_off_on_capture_same_generated_digest,
            candidate_off_on_capture_same_decoded_text_digest: inputs
                .candidate_off_on_capture_same_decoded_text_digest,
            candidate_off_on_capture_same_model_backend_identity: inputs
                .candidate_off_on_capture_same_model_backend_identity,
            capture_blocker_recorded: inputs.capture_blocker_recorded,
            default_runtime_path_preserved,
            candidate_execution_prereqs_complete,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasStrictCaptureArtifactPairBoundary {
    pub fn from_strict_artifact_capture_boundary(
        boundary: &DenseLinearNoBiasStrictArtifactCaptureBoundary,
        inputs: DenseLinearNoBiasStrictCaptureArtifactPairInputs,
    ) -> Self {
        let mut fail_closed_conditions = boundary.fail_closed_conditions.clone();
        let strict_artifact_capture_boundary_ready = boundary.decision
            == "same_callsite_strict_artifact_capture_ready_runtime_disabled"
            && boundary.strict_receipt_artifact_pair_boundary_ready
            && boundary.candidate_off_capture_artifact_validated
            && boundary.candidate_on_capture_artifact_validated
            && boundary.candidate_off_capture_command_recorded
            && boundary.candidate_on_capture_command_recorded
            && boundary.candidate_off_on_capture_same_callsite_identity
            && boundary.candidate_off_on_capture_same_prompt_digest
            && boundary.candidate_off_on_capture_same_generated_digest
            && boundary.candidate_off_on_capture_same_decoded_text_digest
            && boundary.candidate_off_on_capture_same_model_backend_identity
            && boundary.candidate_execution_prereqs_complete
            && boundary.preserves_normal_inference();
        let default_runtime_path_preserved = inputs.default_runtime_path_preserved
            && boundary.default_runtime_path_preserved
            && boundary.selected_path == "eager_f32_candle"
            && boundary.selected_kernel == "dense-f32-candle-linear"
            && boundary.preserves_normal_inference();

        if !strict_artifact_capture_boundary_ready {
            fail_closed_conditions.push("strict_artifact_capture_boundary_not_ready");
        }
        if inputs.candidate_off_strict_capture_artifact_path.is_none() {
            fail_closed_conditions.push("candidate_off_strict_capture_artifact_path_missing");
        }
        if inputs.candidate_on_strict_capture_artifact_path.is_none() {
            fail_closed_conditions.push("candidate_on_strict_capture_artifact_path_missing");
        }
        if !inputs.candidate_off_strict_capture_artifact_present {
            fail_closed_conditions.push("candidate_off_strict_capture_artifact_missing");
        }
        if !inputs.candidate_on_strict_capture_artifact_present {
            fail_closed_conditions.push("candidate_on_strict_capture_artifact_missing");
        }
        if !inputs.candidate_off_capture_command_recorded {
            fail_closed_conditions.push("candidate_off_capture_command_not_recorded");
        }
        if !inputs.candidate_on_capture_command_recorded {
            fail_closed_conditions.push("candidate_on_capture_command_not_recorded");
        }
        if !inputs.candidate_off_capture_binds_gate_identity {
            fail_closed_conditions.push("candidate_off_capture_gate_identity_missing");
        }
        if !inputs.candidate_on_capture_binds_gate_identity {
            fail_closed_conditions.push("candidate_on_capture_gate_identity_missing");
        }
        if !inputs.candidate_off_capture_binds_descriptor_identity {
            fail_closed_conditions.push("candidate_off_capture_descriptor_identity_missing");
        }
        if !inputs.candidate_on_capture_binds_descriptor_identity {
            fail_closed_conditions.push("candidate_on_capture_descriptor_identity_missing");
        }
        if !inputs.candidate_off_capture_binds_owner_callsite_identity {
            fail_closed_conditions.push("candidate_off_capture_owner_callsite_identity_missing");
        }
        if !inputs.candidate_on_capture_binds_owner_callsite_identity {
            fail_closed_conditions.push("candidate_on_capture_owner_callsite_identity_missing");
        }
        if !inputs.candidate_off_on_capture_same_callsite_identity {
            fail_closed_conditions.push("candidate_off_on_capture_callsite_identity_mismatch");
        }
        if !inputs.candidate_off_on_capture_same_prompt_digest {
            fail_closed_conditions.push("candidate_off_on_capture_prompt_digest_mismatch");
        }
        if !inputs.candidate_off_on_capture_same_generated_digest {
            fail_closed_conditions.push("candidate_off_on_capture_generated_digest_mismatch");
        }
        if !inputs.candidate_off_on_capture_same_decoded_text_digest {
            fail_closed_conditions.push("candidate_off_on_capture_decoded_text_digest_mismatch");
        }
        if !inputs.candidate_off_on_capture_same_model_backend_identity {
            fail_closed_conditions.push("candidate_off_on_capture_model_backend_identity_mismatch");
        }
        if !inputs.capture_prerequisite_blocker_recorded
            && (!strict_artifact_capture_boundary_ready
                || !inputs.candidate_off_strict_capture_artifact_present
                || !inputs.candidate_on_strict_capture_artifact_present)
        {
            fail_closed_conditions.push("strict_capture_artifact_pair_blocker_not_recorded");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if boundary.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if boundary.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if boundary.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let strict_capture_artifact_pair_validated = strict_artifact_capture_boundary_ready
            && inputs.candidate_off_strict_capture_artifact_path.is_some()
            && inputs.candidate_on_strict_capture_artifact_path.is_some()
            && inputs.candidate_off_strict_capture_artifact_present
            && inputs.candidate_on_strict_capture_artifact_present
            && inputs.candidate_off_capture_command_recorded
            && inputs.candidate_on_capture_command_recorded
            && inputs.candidate_off_capture_binds_gate_identity
            && inputs.candidate_on_capture_binds_gate_identity
            && inputs.candidate_off_capture_binds_descriptor_identity
            && inputs.candidate_on_capture_binds_descriptor_identity
            && inputs.candidate_off_capture_binds_owner_callsite_identity
            && inputs.candidate_on_capture_binds_owner_callsite_identity
            && inputs.candidate_off_on_capture_same_callsite_identity
            && inputs.candidate_off_on_capture_same_prompt_digest
            && inputs.candidate_off_on_capture_same_generated_digest
            && inputs.candidate_off_on_capture_same_decoded_text_digest
            && inputs.candidate_off_on_capture_same_model_backend_identity
            && default_runtime_path_preserved
            && boundary.runtime_api == "cpu"
            && boundary.selected_backend == "cpu-rust"
            && !boundary.fallback_used
            && fail_closed_conditions.is_empty();

        let candidate_execution_prereqs_complete = strict_capture_artifact_pair_validated;

        let (decision, reason, remaining_runtime_selection_blocker) =
            if strict_capture_artifact_pair_validated {
                (
                    "strict_capture_artifact_pair_validated_runtime_disabled",
                    "candidate_off_on_strict_capture_artifact_pair_validated",
                    "candidate_execution_enablement_requires_separate_receipt_gated_pr",
                )
            } else if strict_artifact_capture_boundary_ready {
                let blocker = if !inputs.candidate_on_strict_capture_artifact_present
                    || inputs.candidate_on_strict_capture_artifact_path.is_none()
                {
                    "candidate_on_strict_capture_artifact"
                } else if !inputs.candidate_off_strict_capture_artifact_present
                    || inputs.candidate_off_strict_capture_artifact_path.is_none()
                {
                    "candidate_off_strict_capture_artifact"
                } else if !inputs.candidate_off_capture_command_recorded
                    || !inputs.candidate_on_capture_command_recorded
                {
                    "candidate_off_on_capture_commands"
                } else if !inputs.candidate_off_capture_binds_gate_identity
                    || !inputs.candidate_on_capture_binds_gate_identity
                {
                    "candidate_off_on_capture_gate_identity"
                } else if !inputs.candidate_off_capture_binds_descriptor_identity
                    || !inputs.candidate_on_capture_binds_descriptor_identity
                {
                    "candidate_off_on_capture_descriptor_identity"
                } else if !inputs.candidate_off_capture_binds_owner_callsite_identity
                    || !inputs.candidate_on_capture_binds_owner_callsite_identity
                    || !inputs.candidate_off_on_capture_same_callsite_identity
                {
                    "candidate_off_on_capture_owner_callsite_identity"
                } else if !inputs.candidate_off_on_capture_same_model_backend_identity {
                    "candidate_off_on_capture_model_backend_identity"
                } else if !inputs.candidate_off_on_capture_same_prompt_digest
                    || !inputs.candidate_off_on_capture_same_generated_digest
                    || !inputs.candidate_off_on_capture_same_decoded_text_digest
                {
                    "candidate_off_on_capture_prompt_generated_text_digests"
                } else if !inputs.capture_prerequisite_blocker_recorded {
                    "strict_capture_artifact_pair_blocker_record"
                } else {
                    "default_runtime_path_preservation"
                };
                (
                    "strict_capture_artifact_pair_blocked_fail_closed",
                    "strict_capture_artifact_pair_incomplete",
                    blocker,
                )
            } else {
                (
                    "blocked_fail_closed",
                    "strict_artifact_capture_boundary_incomplete",
                    "strict_artifact_capture_boundary",
                )
            };

        Self {
            tensor_name: boundary.tensor_name.clone(),
            callsite_identity: boundary.callsite_identity.clone(),
            model_sha256: boundary.model_sha256.clone(),
            model_architecture: boundary.model_architecture,
            quant_format: boundary.quant_format,
            tokenizer_source: boundary.tokenizer_source,
            tokenizer_strict: boundary.tokenizer_strict,
            runtime_api: boundary.runtime_api,
            selected_backend: boundary.selected_backend,
            fallback_used: boundary.fallback_used,
            selected_path: boundary.selected_path,
            selected_kernel: boundary.selected_kernel,
            candidate_path: boundary.candidate_path,
            candidate_kernel: boundary.candidate_kernel,
            prompt_ids_digest: boundary.prompt_ids_digest.clone(),
            generated_ids_digest: boundary.generated_ids_digest.clone(),
            decoded_text_digest: boundary.decoded_text_digest.clone(),
            strict_artifact_capture_boundary_ready,
            candidate_off_strict_capture_artifact_path: inputs
                .candidate_off_strict_capture_artifact_path
                .map(str::to_owned),
            candidate_on_strict_capture_artifact_path: inputs
                .candidate_on_strict_capture_artifact_path
                .map(str::to_owned),
            candidate_off_strict_capture_artifact_present: inputs
                .candidate_off_strict_capture_artifact_present,
            candidate_on_strict_capture_artifact_present: inputs
                .candidate_on_strict_capture_artifact_present,
            candidate_off_capture_command_recorded: inputs.candidate_off_capture_command_recorded,
            candidate_on_capture_command_recorded: inputs.candidate_on_capture_command_recorded,
            candidate_off_capture_binds_gate_identity: inputs
                .candidate_off_capture_binds_gate_identity,
            candidate_on_capture_binds_gate_identity: inputs
                .candidate_on_capture_binds_gate_identity,
            candidate_off_capture_binds_descriptor_identity: inputs
                .candidate_off_capture_binds_descriptor_identity,
            candidate_on_capture_binds_descriptor_identity: inputs
                .candidate_on_capture_binds_descriptor_identity,
            candidate_off_capture_binds_owner_callsite_identity: inputs
                .candidate_off_capture_binds_owner_callsite_identity,
            candidate_on_capture_binds_owner_callsite_identity: inputs
                .candidate_on_capture_binds_owner_callsite_identity,
            candidate_off_on_capture_same_callsite_identity: inputs
                .candidate_off_on_capture_same_callsite_identity,
            candidate_off_on_capture_same_prompt_digest: inputs
                .candidate_off_on_capture_same_prompt_digest,
            candidate_off_on_capture_same_generated_digest: inputs
                .candidate_off_on_capture_same_generated_digest,
            candidate_off_on_capture_same_decoded_text_digest: inputs
                .candidate_off_on_capture_same_decoded_text_digest,
            candidate_off_on_capture_same_model_backend_identity: inputs
                .candidate_off_on_capture_same_model_backend_identity,
            capture_prerequisite_blocker_recorded: inputs.capture_prerequisite_blocker_recorded,
            default_runtime_path_preserved,
            strict_capture_artifact_pair_validated,
            candidate_execution_prereqs_complete,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasRuntimeAttemptBoundary {
    pub fn from_strict_capture_artifact_pair(
        boundary: &DenseLinearNoBiasStrictCaptureArtifactPairBoundary,
        inputs: DenseLinearNoBiasRuntimeAttemptInputs,
    ) -> Self {
        let mut fail_closed_conditions = boundary.fail_closed_conditions.clone();
        let strict_capture_artifact_pair_validated = boundary.decision
            == "strict_capture_artifact_pair_validated_runtime_disabled"
            && boundary.strict_capture_artifact_pair_validated
            && boundary.candidate_execution_prereqs_complete
            && boundary.preserves_normal_inference();
        let default_runtime_path_preserved = inputs.default_runtime_path_preserved
            && boundary.default_runtime_path_preserved
            && boundary.selected_path == "eager_f32_candle"
            && boundary.selected_kernel == "dense-f32-candle-linear"
            && boundary.preserves_normal_inference();

        if !strict_capture_artifact_pair_validated {
            fail_closed_conditions.push("strict_capture_artifact_pair_not_validated");
        }
        if !inputs.explicit_candidate_execution_gate_requested {
            fail_closed_conditions.push("explicit_candidate_execution_gate_not_requested");
        }
        if !inputs.runtime_hook_registry_attachment_present {
            fail_closed_conditions
                .push("receipt_bound_selector_not_attached_to_runtime_hook_registry");
        }
        if !inputs.runtime_hook_descriptor_binds_selector_identity {
            fail_closed_conditions.push("runtime_hook_descriptor_selector_identity_missing");
        }
        if !inputs.runtime_hook_descriptor_binds_strict_capture_pair {
            fail_closed_conditions
                .push("runtime_hook_descriptor_strict_capture_pair_identity_missing");
        }
        if !inputs.apply_linear_dispatch_wired_to_no_bias_candidate {
            fail_closed_conditions.push("apply_linear_no_bias_candidate_dispatch_not_wired");
        }
        if !inputs.feed_forward_down_proj_scope_preserved {
            fail_closed_conditions.push("feed_forward_down_proj_scope_not_preserved");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if boundary.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if boundary.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if boundary.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let candidate_execution_attempt_allowed = strict_capture_artifact_pair_validated
            && inputs.explicit_candidate_execution_gate_requested
            && inputs.runtime_hook_registry_attachment_present
            && inputs.runtime_hook_descriptor_binds_selector_identity
            && inputs.runtime_hook_descriptor_binds_strict_capture_pair
            && inputs.apply_linear_dispatch_wired_to_no_bias_candidate
            && inputs.feed_forward_down_proj_scope_preserved
            && default_runtime_path_preserved
            && boundary.runtime_api == "cpu"
            && boundary.selected_backend == "cpu-rust"
            && !boundary.fallback_used
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) =
            if candidate_execution_attempt_allowed {
                (
                    "candidate_execution_attempt_prereqs_ready_runtime_disabled",
                    "strict_capture_pair_and_runtime_attachment_are_ready_but_runtime_enablement_remains_separate",
                    "fresh_candidate_off_on_execution_receipts",
                )
            } else if strict_capture_artifact_pair_validated {
                let blocker = if !inputs.runtime_hook_registry_attachment_present {
                    "receipt_bound_selector_runtime_hook_registry_attachment"
                } else if !inputs.runtime_hook_descriptor_binds_selector_identity {
                    "runtime_hook_descriptor_selector_identity"
                } else if !inputs.runtime_hook_descriptor_binds_strict_capture_pair {
                    "runtime_hook_descriptor_strict_capture_pair_identity"
                } else if !inputs.apply_linear_dispatch_wired_to_no_bias_candidate {
                    "apply_linear_no_bias_candidate_dispatch"
                } else if !inputs.explicit_candidate_execution_gate_requested {
                    "explicit_candidate_execution_gate"
                } else if !inputs.feed_forward_down_proj_scope_preserved {
                    "feed_forward_down_proj_scope"
                } else {
                    "default_runtime_path_preservation"
                };
                (
                    "candidate_execution_attempt_blocked_fail_closed",
                    "strict_capture_pair_is_validated_but_runtime_attachment_is_incomplete",
                    blocker,
                )
            } else {
                (
                    "blocked_fail_closed",
                    "strict_capture_artifact_pair_incomplete",
                    "strict_capture_artifact_pair",
                )
            };

        Self {
            tensor_name: boundary.tensor_name.clone(),
            callsite_identity: boundary.callsite_identity.clone(),
            model_sha256: boundary.model_sha256.clone(),
            model_architecture: boundary.model_architecture,
            quant_format: boundary.quant_format,
            tokenizer_source: boundary.tokenizer_source,
            tokenizer_strict: boundary.tokenizer_strict,
            runtime_api: boundary.runtime_api,
            selected_backend: boundary.selected_backend,
            fallback_used: boundary.fallback_used,
            selected_path: boundary.selected_path,
            selected_kernel: boundary.selected_kernel,
            candidate_path: boundary.candidate_path,
            candidate_kernel: boundary.candidate_kernel,
            prompt_ids_digest: boundary.prompt_ids_digest.clone(),
            generated_ids_digest: boundary.generated_ids_digest.clone(),
            decoded_text_digest: boundary.decoded_text_digest.clone(),
            strict_capture_artifact_pair_validated,
            explicit_candidate_execution_gate_requested: inputs
                .explicit_candidate_execution_gate_requested,
            runtime_hook_registry_attachment_present: inputs
                .runtime_hook_registry_attachment_present,
            runtime_hook_descriptor_binds_selector_identity: inputs
                .runtime_hook_descriptor_binds_selector_identity,
            runtime_hook_descriptor_binds_strict_capture_pair: inputs
                .runtime_hook_descriptor_binds_strict_capture_pair,
            apply_linear_dispatch_wired_to_no_bias_candidate: inputs
                .apply_linear_dispatch_wired_to_no_bias_candidate,
            feed_forward_down_proj_scope_preserved: inputs.feed_forward_down_proj_scope_preserved,
            default_runtime_path_preserved,
            candidate_execution_attempt_allowed,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasRuntimeHookAttachmentBoundary {
    pub fn from_runtime_attempt_and_registry(
        attempt: &DenseLinearNoBiasRuntimeAttemptBoundary,
        registry: &DenseLinearRuntimeHookRegistry,
    ) -> Self {
        let mut fail_closed_conditions = Vec::new();
        let hook = registry.get(&attempt.tensor_name);
        let selector = hook.and_then(|hook| hook.receipt_bound_no_bias_selector.as_ref());

        let runtime_hook_registry_attachment_present = hook.is_some();
        let registry_key_matches_tensor_name =
            hook.is_some_and(|hook| hook.tensor_name == attempt.tensor_name);
        let descriptor_ready_for_apply_linear_callsite =
            selector.is_some_and(|selector| selector.descriptor_ready_for_apply_linear_callsite);

        let runtime_hook_descriptor_binds_selector_identity =
            hook.zip(selector).is_some_and(|(hook, selector)| {
                hook.tensor_name == attempt.tensor_name
                    && selector.tensor_name == attempt.tensor_name
                    && selector.model_sha256 == attempt.model_sha256
                    && selector.model_architecture == attempt.model_architecture
                    && selector.quant_format == attempt.quant_format
                    && selector.tokenizer_source == attempt.tokenizer_source
                    && selector.tokenizer_strict == attempt.tokenizer_strict
                    && selector.runtime_api == attempt.runtime_api
                    && selector.selected_backend == attempt.selected_backend
                    && selector.fallback_used == attempt.fallback_used
                    && selector.selected_path == attempt.selected_path
                    && selector.selected_kernel == attempt.selected_kernel
                    && selector.candidate_path == attempt.candidate_path
                    && selector.candidate_kernel == attempt.candidate_kernel
                    && selector.bias_present == Some(false)
                    && selector.runtime_gate_requested_enabled
                        == attempt.explicit_candidate_execution_gate_requested
                    && selector.descriptor_ready_for_apply_linear_callsite
            });

        let runtime_hook_descriptor_binds_strict_capture_pair = selector.is_some_and(|selector| {
            selector.before_after_receipts_present
                && !selector.before_after_receipt_pair_identity.is_empty()
                && selector.prompt_ids_digest == attempt.prompt_ids_digest
                && selector.generated_ids_digest == attempt.generated_ids_digest
                && selector.decoded_text_digest == attempt.decoded_text_digest
                && selector.prompt_ids_digest_preserved
                && selector.generated_ids_digest_preserved
                && selector.decoded_text_digest_preserved
        });

        let default_runtime_path_preserved = attempt.default_runtime_path_preserved
            && attempt.preserves_normal_inference()
            && hook.is_none_or(|hook| !hook.runtime_compute_enabled);

        if !runtime_hook_registry_attachment_present {
            fail_closed_conditions
                .push("receipt_bound_selector_not_attached_to_runtime_hook_registry");
        }
        if !registry_key_matches_tensor_name {
            fail_closed_conditions.push("runtime_hook_registry_key_tensor_name_mismatch");
        }
        if !runtime_hook_descriptor_binds_selector_identity {
            fail_closed_conditions.push("runtime_hook_descriptor_selector_identity_missing");
        }
        if !runtime_hook_descriptor_binds_strict_capture_pair {
            fail_closed_conditions
                .push("runtime_hook_descriptor_strict_capture_pair_identity_missing");
        }
        if !descriptor_ready_for_apply_linear_callsite {
            fail_closed_conditions.push("selector_descriptor_not_ready_for_apply_linear_callsite");
        }
        if !attempt.strict_capture_artifact_pair_validated {
            fail_closed_conditions.push("strict_capture_artifact_pair_not_validated");
        }
        if !attempt.explicit_candidate_execution_gate_requested {
            fail_closed_conditions.push("explicit_candidate_execution_gate_not_requested");
        }
        if !attempt.feed_forward_down_proj_scope_preserved {
            fail_closed_conditions.push("feed_forward_down_proj_scope_not_preserved");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if attempt.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if attempt.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if attempt.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let runtime_hook_attachment_ready = attempt.strict_capture_artifact_pair_validated
            && attempt.explicit_candidate_execution_gate_requested
            && runtime_hook_registry_attachment_present
            && runtime_hook_descriptor_binds_selector_identity
            && runtime_hook_descriptor_binds_strict_capture_pair
            && registry_key_matches_tensor_name
            && descriptor_ready_for_apply_linear_callsite
            && attempt.feed_forward_down_proj_scope_preserved
            && default_runtime_path_preserved
            && attempt.runtime_api == "cpu"
            && attempt.selected_backend == "cpu-rust"
            && !attempt.fallback_used
            && fail_closed_conditions.is_empty();
        let candidate_execution_attempt_allowed = false;

        let (decision, reason, remaining_runtime_selection_blocker) =
            if runtime_hook_attachment_ready {
                (
                    "runtime_hook_attachment_ready_runtime_disabled",
                    "receipt_bound_selector_identity_reaches_runtime_hook_registry_but_candidate_execution_remains_separate",
                    "fresh_candidate_off_on_execution_receipts",
                )
            } else if attempt.strict_capture_artifact_pair_validated {
                let blocker = if !runtime_hook_registry_attachment_present {
                    "receipt_bound_selector_runtime_hook_registry_attachment"
                } else if !runtime_hook_descriptor_binds_selector_identity {
                    "runtime_hook_descriptor_selector_identity"
                } else if !runtime_hook_descriptor_binds_strict_capture_pair {
                    "runtime_hook_descriptor_strict_capture_pair_identity"
                } else if !descriptor_ready_for_apply_linear_callsite {
                    "selector_descriptor_apply_linear_readiness"
                } else if !registry_key_matches_tensor_name {
                    "runtime_hook_registry_key_tensor_name"
                } else if !default_runtime_path_preserved {
                    "default_runtime_path_preservation"
                } else {
                    "candidate_off_on_execution_receipts"
                };
                (
                    "runtime_hook_attachment_blocked_fail_closed",
                    "strict_capture_pair_is_validated_but_runtime_hook_attachment_identity_is_incomplete",
                    blocker,
                )
            } else {
                (
                    "blocked_fail_closed",
                    "strict_capture_artifact_pair_incomplete",
                    "strict_capture_artifact_pair",
                )
            };

        Self {
            tensor_name: attempt.tensor_name.clone(),
            callsite_identity: attempt.callsite_identity.clone(),
            model_sha256: attempt.model_sha256.clone(),
            model_architecture: attempt.model_architecture,
            quant_format: attempt.quant_format,
            tokenizer_source: attempt.tokenizer_source,
            tokenizer_strict: attempt.tokenizer_strict,
            runtime_api: attempt.runtime_api,
            selected_backend: attempt.selected_backend,
            fallback_used: attempt.fallback_used,
            selected_path: attempt.selected_path,
            selected_kernel: attempt.selected_kernel,
            candidate_path: attempt.candidate_path,
            candidate_kernel: attempt.candidate_kernel,
            prompt_ids_digest: attempt.prompt_ids_digest.clone(),
            generated_ids_digest: attempt.generated_ids_digest.clone(),
            decoded_text_digest: attempt.decoded_text_digest.clone(),
            strict_capture_artifact_pair_validated: attempt.strict_capture_artifact_pair_validated,
            explicit_candidate_execution_gate_requested: attempt
                .explicit_candidate_execution_gate_requested,
            runtime_hook_registry_attachment_present,
            runtime_hook_descriptor_binds_selector_identity,
            runtime_hook_descriptor_binds_strict_capture_pair,
            registry_key_matches_tensor_name,
            descriptor_ready_for_apply_linear_callsite,
            feed_forward_down_proj_scope_preserved: attempt.feed_forward_down_proj_scope_preserved,
            default_runtime_path_preserved,
            candidate_execution_attempt_allowed,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearNoBiasCandidateExecutionReceiptGate {
    pub fn from_runtime_hook_attachment(
        attachment: &DenseLinearNoBiasRuntimeHookAttachmentBoundary,
        inputs: DenseLinearNoBiasCandidateExecutionReceiptInputs,
    ) -> Self {
        let mut fail_closed_conditions = attachment.fail_closed_conditions.clone();
        let runtime_hook_attachment_ready = attachment.decision
            == "runtime_hook_attachment_ready_runtime_disabled"
            && attachment.strict_capture_artifact_pair_validated
            && attachment.explicit_candidate_execution_gate_requested
            && attachment.runtime_hook_registry_attachment_present
            && attachment.runtime_hook_descriptor_binds_selector_identity
            && attachment.runtime_hook_descriptor_binds_strict_capture_pair
            && attachment.registry_key_matches_tensor_name
            && attachment.descriptor_ready_for_apply_linear_callsite
            && attachment.feed_forward_down_proj_scope_preserved
            && attachment.preserves_normal_inference();
        let default_runtime_path_preserved = inputs.default_runtime_path_preserved
            && attachment.default_runtime_path_preserved
            && attachment.selected_path == "eager_f32_candle"
            && attachment.selected_kernel == "dense-f32-candle-linear"
            && attachment.preserves_normal_inference();

        if !runtime_hook_attachment_ready {
            fail_closed_conditions.push("runtime_hook_attachment_not_ready");
        }
        if !inputs.candidate_off_execution_receipt_present {
            fail_closed_conditions.push("candidate_off_execution_receipt_missing");
        }
        if !inputs.candidate_on_execution_receipt_present {
            fail_closed_conditions.push("candidate_on_execution_receipt_missing");
        }
        if !inputs.candidate_off_execution_binds_registry_attachment {
            fail_closed_conditions
                .push("candidate_off_execution_registry_attachment_identity_missing");
        }
        if !inputs.candidate_on_execution_binds_registry_attachment {
            fail_closed_conditions
                .push("candidate_on_execution_registry_attachment_identity_missing");
        }
        if !inputs.candidate_off_on_same_callsite_identity {
            fail_closed_conditions.push("candidate_off_on_callsite_identity_mismatch");
        }
        if !inputs.candidate_off_on_same_prompt_digest {
            fail_closed_conditions.push("candidate_off_on_prompt_digest_mismatch");
        }
        if !inputs.candidate_off_on_same_generated_digest {
            fail_closed_conditions.push("candidate_off_on_generated_digest_mismatch");
        }
        if !inputs.candidate_off_on_same_decoded_text_digest {
            fail_closed_conditions.push("candidate_off_on_decoded_text_digest_mismatch");
        }
        if !inputs.candidate_off_on_same_model_backend_identity {
            fail_closed_conditions.push("candidate_off_on_model_backend_identity_mismatch");
        }
        if !inputs.prompt_ids_preserved {
            fail_closed_conditions.push("prompt_ids_not_preserved");
        }
        if !inputs.generated_ids_preserved {
            fail_closed_conditions.push("generated_ids_not_preserved");
        }
        if !inputs.decoded_text_preserved {
            fail_closed_conditions.push("decoded_text_not_preserved");
        }
        if attachment.prompt_ids_digest.is_empty() {
            fail_closed_conditions.push("prompt_ids_digest_missing");
        }
        if attachment.generated_ids_digest.is_empty() {
            fail_closed_conditions.push("generated_ids_digest_missing");
        }
        if attachment.decoded_text_digest.is_empty() {
            fail_closed_conditions.push("decoded_text_digest_missing");
        }
        if !inputs.execution_receipt_blocker_recorded
            && (!inputs.candidate_off_execution_receipt_present
                || !inputs.candidate_on_execution_receipt_present)
        {
            fail_closed_conditions.push("candidate_execution_receipt_blocker_not_recorded");
        }
        if !default_runtime_path_preserved {
            fail_closed_conditions.push("default_runtime_path_not_preserved");
        }
        if attachment.runtime_api != "cpu" {
            fail_closed_conditions.push("runtime_api_not_cpu");
        }
        if attachment.selected_backend != "cpu-rust" {
            fail_closed_conditions.push("selected_backend_not_cpu_rust");
        }
        if attachment.fallback_used {
            fail_closed_conditions.push("fallback_used");
        }

        fail_closed_conditions.sort_unstable();
        fail_closed_conditions.dedup();

        let candidate_execution_receipt_pair_ready = runtime_hook_attachment_ready
            && inputs.candidate_off_execution_receipt_present
            && inputs.candidate_on_execution_receipt_present
            && inputs.candidate_off_execution_binds_registry_attachment
            && inputs.candidate_on_execution_binds_registry_attachment
            && inputs.candidate_off_on_same_callsite_identity
            && inputs.candidate_off_on_same_prompt_digest
            && inputs.candidate_off_on_same_generated_digest
            && inputs.candidate_off_on_same_decoded_text_digest
            && inputs.candidate_off_on_same_model_backend_identity
            && inputs.prompt_ids_preserved
            && inputs.generated_ids_preserved
            && inputs.decoded_text_preserved
            && default_runtime_path_preserved
            && attachment.runtime_api == "cpu"
            && attachment.selected_backend == "cpu-rust"
            && !attachment.fallback_used
            && fail_closed_conditions.is_empty();

        let (decision, reason, remaining_runtime_selection_blocker) =
            if candidate_execution_receipt_pair_ready {
                (
                    "candidate_execution_receipt_pair_ready_default_disabled",
                    "candidate_off_on_execution_receipts_preserve_registry_bound_identity",
                    "candidate_timing_allocation_receipts_or_opt_in_profile",
                )
            } else if runtime_hook_attachment_ready {
                let blocker = if !inputs.candidate_on_execution_receipt_present {
                    "candidate_on_execution_receipt"
                } else if !inputs.candidate_off_execution_receipt_present {
                    "candidate_off_execution_receipt"
                } else if !inputs.candidate_on_execution_binds_registry_attachment
                    || !inputs.candidate_off_execution_binds_registry_attachment
                {
                    "candidate_execution_registry_attachment_identity"
                } else if !inputs.generated_ids_preserved || !inputs.decoded_text_preserved {
                    "candidate_execution_generated_id_text_preservation"
                } else if !inputs.prompt_ids_preserved {
                    "candidate_execution_prompt_id_preservation"
                } else if !inputs.candidate_off_on_same_model_backend_identity {
                    "candidate_execution_model_backend_identity"
                } else if !inputs.execution_receipt_blocker_recorded {
                    "candidate_execution_receipt_blocker_record"
                } else {
                    "candidate_off_on_execution_receipts"
                };
                (
                    "candidate_execution_receipt_pair_blocked_fail_closed",
                    "runtime_hook_attachment_ready_but_fresh_execution_receipts_are_missing",
                    blocker,
                )
            } else {
                (
                    "blocked_fail_closed",
                    "runtime_hook_attachment_incomplete",
                    "runtime_hook_attachment_boundary",
                )
            };

        Self {
            tensor_name: attachment.tensor_name.clone(),
            callsite_identity: attachment.callsite_identity.clone(),
            model_sha256: attachment.model_sha256.clone(),
            model_architecture: attachment.model_architecture,
            quant_format: attachment.quant_format,
            tokenizer_source: attachment.tokenizer_source,
            tokenizer_strict: attachment.tokenizer_strict,
            runtime_api: attachment.runtime_api,
            selected_backend: attachment.selected_backend,
            fallback_used: attachment.fallback_used,
            selected_path: attachment.selected_path,
            selected_kernel: attachment.selected_kernel,
            candidate_path: attachment.candidate_path,
            candidate_kernel: attachment.candidate_kernel,
            prompt_ids_digest: attachment.prompt_ids_digest.clone(),
            generated_ids_digest: attachment.generated_ids_digest.clone(),
            decoded_text_digest: attachment.decoded_text_digest.clone(),
            runtime_hook_attachment_ready,
            explicit_candidate_execution_gate_requested: attachment
                .explicit_candidate_execution_gate_requested,
            runtime_hook_registry_attachment_present: attachment
                .runtime_hook_registry_attachment_present,
            runtime_hook_descriptor_binds_selector_identity: attachment
                .runtime_hook_descriptor_binds_selector_identity,
            runtime_hook_descriptor_binds_strict_capture_pair: attachment
                .runtime_hook_descriptor_binds_strict_capture_pair,
            registry_key_matches_tensor_name: attachment.registry_key_matches_tensor_name,
            descriptor_ready_for_apply_linear_callsite: attachment
                .descriptor_ready_for_apply_linear_callsite,
            candidate_off_execution_receipt_present: inputs.candidate_off_execution_receipt_present,
            candidate_on_execution_receipt_present: inputs.candidate_on_execution_receipt_present,
            candidate_off_execution_binds_registry_attachment: inputs
                .candidate_off_execution_binds_registry_attachment,
            candidate_on_execution_binds_registry_attachment: inputs
                .candidate_on_execution_binds_registry_attachment,
            candidate_off_on_same_callsite_identity: inputs.candidate_off_on_same_callsite_identity,
            candidate_off_on_same_prompt_digest: inputs.candidate_off_on_same_prompt_digest,
            candidate_off_on_same_generated_digest: inputs.candidate_off_on_same_generated_digest,
            candidate_off_on_same_decoded_text_digest: inputs
                .candidate_off_on_same_decoded_text_digest,
            candidate_off_on_same_model_backend_identity: inputs
                .candidate_off_on_same_model_backend_identity,
            prompt_ids_preserved: inputs.prompt_ids_preserved,
            generated_ids_preserved: inputs.generated_ids_preserved,
            decoded_text_preserved: inputs.decoded_text_preserved,
            default_runtime_path_preserved,
            candidate_execution_receipt_pair_ready,
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled_by_default: false,
            decision,
            reason,
            remaining_runtime_selection_blocker,
            fail_closed_conditions,
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    pub fn preserves_normal_inference(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && self.runtime_api == "cpu"
            && self.selected_backend == "cpu-rust"
            && !self.fallback_used
            && self.default_runtime_path_preserved
            && !self.normal_inference_runtime_selection_enabled
            && !self.candidate_execution_enabled_by_default
            && !self.allocation_reduction_claim
            && !self.timing_improvement_claim
            && !self.speedup_claim
    }
}

impl DenseLinearRuntimeHookBoundary {
    pub fn eager_f32(tensor_name: impl Into<String>) -> Self {
        Self {
            tensor_name: tensor_name.into(),
            selected_path: "eager_f32_candle",
            selected_kernel: "dense-f32-candle-linear",
            sidecar_descriptor_present: false,
            sidecar_role: None,
            sidecar_payload_sha256: None,
            sidecar_payload_bytes_available: false,
            sidecar_payload_bytes: None,
            sidecar_q8_block_count: None,
            sidecar_matrix_rows: None,
            sidecar_matrix_cols: None,
            sidecar_payload_contract_valid: false,
            sidecar_payload_order_matches_runtime_shape: false,
            source_order_q8_matvec_candidate: false,
            source_order_selected_path: None,
            source_order_selected_kernel: None,
            source_order_input_dim: None,
            source_order_output_dim: None,
            source_order_candidate_receipt_identity: None,
            source_order_candidate_runtime_enabled: false,
            runtime_compute_enabled: false,
            eager_f32_runtime_preserved: true,
            dense_runtime_replaced: false,
            speedup_claim: false,
            generated_id_preservation_required_before_runtime_use: true,
            next_receipt_gate: "before_after_qwen3_q8_generated_id_text_receipts",
        }
    }

    pub fn from_sidecar_descriptor(
        tensor_name: impl Into<String>,
        descriptor: &DenseLinearRuntimeHookDescriptor,
    ) -> Self {
        let tensor_name = tensor_name.into();
        let payload = descriptor.packed_q8_payload.as_ref();
        let payload_contract_valid = payload.is_some_and(|payload| {
            payload.tensor_name == descriptor.tensor_name
                && payload.shape_matches_matvec_contract()
                && payload.payload_len_matches_contract()
        });
        let source_order_q8_matvec_candidate = descriptor.source_order_q8_matvec_candidate
            && tensor_name == SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR
            && payload_contract_valid
            && !descriptor.payload_order_matches_runtime_shape
            && descriptor.source_order_input_dim.is_some()
            && descriptor.source_order_output_dim.is_some();
        let source_order_candidate_runtime_enabled =
            descriptor.runtime_compute_enabled && source_order_q8_matvec_candidate;
        let source_order_candidate_receipt_identity = source_order_q8_matvec_candidate.then(|| {
            let runtime_status = if source_order_candidate_runtime_enabled {
                "runtime_enabled"
            } else {
                "runtime_disabled"
            };
            format!(
                "{}:source_order_q8_0_qproj_matvec:{runtime_status}",
                SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR,
            )
        });
        let runtime_compute_enabled = descriptor.runtime_compute_enabled
            && tensor_name == SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR
            && descriptor.payload_order_matches_runtime_shape
            && payload_contract_valid;
        Self {
            tensor_name,
            selected_path: if source_order_candidate_runtime_enabled {
                "source_order_q8_0_qproj_matvec"
            } else if runtime_compute_enabled {
                "packed_q8_sidecar"
            } else {
                "eager_f32_candle"
            },
            selected_kernel: if source_order_candidate_runtime_enabled {
                "dense-q8-source-order-qproj-matvec"
            } else if runtime_compute_enabled {
                "dense-q8-sidecar-linear"
            } else {
                "dense-f32-candle-linear"
            },
            sidecar_descriptor_present: true,
            sidecar_role: Some(descriptor.role.clone()),
            sidecar_payload_sha256: descriptor.sidecar_payload_sha256.clone(),
            sidecar_payload_bytes_available: payload.is_some(),
            sidecar_payload_bytes: payload.map(DenseLinearPackedQ8Payload::payload_len),
            sidecar_q8_block_count: payload.map(|payload| payload.q8_block_count),
            sidecar_matrix_rows: payload.map(|payload| payload.matrix_rows),
            sidecar_matrix_cols: payload.map(|payload| payload.matrix_cols),
            sidecar_payload_contract_valid: payload_contract_valid,
            sidecar_payload_order_matches_runtime_shape: descriptor
                .payload_order_matches_runtime_shape,
            source_order_q8_matvec_candidate,
            source_order_selected_path: source_order_q8_matvec_candidate
                .then_some("source_order_q8_0_qproj_matvec"),
            source_order_selected_kernel: source_order_q8_matvec_candidate
                .then_some("dense-q8-source-order-qproj-matvec"),
            source_order_input_dim: descriptor.source_order_input_dim,
            source_order_output_dim: descriptor.source_order_output_dim,
            source_order_candidate_receipt_identity,
            source_order_candidate_runtime_enabled,
            runtime_compute_enabled,
            eager_f32_runtime_preserved: !(runtime_compute_enabled
                || source_order_candidate_runtime_enabled),
            dense_runtime_replaced: runtime_compute_enabled
                || source_order_candidate_runtime_enabled,
            speedup_claim: false,
            generated_id_preservation_required_before_runtime_use: true,
            next_receipt_gate: "before_after_qwen3_q8_generated_id_text_receipts",
        }
    }

    pub fn preserves_eager_f32(&self) -> bool {
        self.selected_path == "eager_f32_candle"
            && self.selected_kernel == "dense-f32-candle-linear"
            && !self.runtime_compute_enabled
            && self.eager_f32_runtime_preserved
            && !self.dense_runtime_replaced
            && !self.speedup_claim
    }
}

fn dense_linear_runtime_hook_boundary(
    tensor_name: &str,
    hooks: &DenseLinearRuntimeHookRegistry,
) -> DenseLinearRuntimeHookBoundary {
    hooks
        .get(tensor_name)
        .map(|descriptor| {
            DenseLinearRuntimeHookBoundary::from_sidecar_descriptor(tensor_name, descriptor)
        })
        .unwrap_or_else(|| DenseLinearRuntimeHookBoundary::eager_f32(tensor_name))
}

fn maybe_forward_dense_q8_sidecar_linear(
    input: &Tensor,
    linear: &Linear,
    tensor_name: &str,
    hooks: &DenseLinearRuntimeHookRegistry,
) -> Result<Option<Tensor>> {
    let selector_start = Instant::now();
    let Some(descriptor) = hooks.get(tensor_name) else {
        return Ok(None);
    };
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_dispatch_calls, 1);
    let boundary = DenseLinearRuntimeHookBoundary::from_sidecar_descriptor(tensor_name, descriptor);
    if boundary.source_order_candidate_runtime_enabled {
        let Some(payload) = descriptor.packed_q8_payload.as_ref() else {
            add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_error_calls, 1);
            add_counter(
                &DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_dispatch_ns,
                elapsed_ns_u64(selector_start),
            );
            return Err(BitNetError::Validation(format!(
                "source-order Q8 runtime hook for {tensor_name} was enabled without payload bytes"
            )));
        };
        add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_selected_calls, 1);
        add_counter(
            &DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_dispatch_ns,
            elapsed_ns_u64(selector_start),
        );
        return dense_q8_source_order_qproj_linear_forward(
            input,
            linear.bias(),
            payload,
            boundary.source_order_input_dim.ok_or_else(|| {
                BitNetError::Validation(format!(
                    "source-order Q8 runtime hook for {tensor_name} is missing source input dim"
                ))
            })?,
            boundary.source_order_output_dim.ok_or_else(|| {
                BitNetError::Validation(format!(
                    "source-order Q8 runtime hook for {tensor_name} is missing source output dim"
                ))
            })?,
        )
        .map(Some);
    }
    if !boundary.runtime_compute_enabled {
        add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_declined_calls, 1);
        add_counter(
            &DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_dispatch_ns,
            elapsed_ns_u64(selector_start),
        );
        return Ok(None);
    }
    let Some(payload) = descriptor.packed_q8_payload.as_ref() else {
        add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_error_calls, 1);
        add_counter(
            &DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_dispatch_ns,
            elapsed_ns_u64(selector_start),
        );
        return Err(BitNetError::Validation(format!(
            "packed Q8 runtime hook for {tensor_name} was enabled without payload bytes"
        )));
    };
    if tensor_name != SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR {
        add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_error_calls, 1);
        add_counter(
            &DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_dispatch_ns,
            elapsed_ns_u64(selector_start),
        );
        return Err(BitNetError::Validation(format!(
            "packed Q8 runtime hook is scoped to {SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR}, got {tensor_name}"
        )));
    }
    if !payload.shape_matches_matvec_contract() || !payload.payload_len_matches_contract() {
        add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_error_calls, 1);
        add_counter(
            &DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_dispatch_ns,
            elapsed_ns_u64(selector_start),
        );
        return Err(BitNetError::Validation(format!(
            "packed Q8 runtime hook payload contract is invalid for {tensor_name}"
        )));
    }
    if linear.weight().dims() != [payload.matrix_rows, payload.matrix_cols] {
        add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_error_calls, 1);
        add_counter(
            &DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_dispatch_ns,
            elapsed_ns_u64(selector_start),
        );
        return Err(BitNetError::Validation(format!(
            "packed Q8 runtime hook shape {:?} does not match Candle linear weight {:?} for {tensor_name}",
            [payload.matrix_rows, payload.matrix_cols],
            linear.weight().dims()
        )));
    }
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_selected_calls, 1);
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.selector_dispatch_ns,
        elapsed_ns_u64(selector_start),
    );

    dense_q8_sidecar_linear_forward(input, linear.bias(), payload)
        .map(Some)
        .map_err(BitNetError::from)
}

pub(crate) fn maybe_trace_dense_q8_source_order_qproj_candidate(
    input: &Tensor,
    eager_output: &Tensor,
    linear: &Linear,
    tensor_name: &str,
    hooks: &DenseLinearRuntimeHookRegistry,
    layer_idx: usize,
) -> Result<()> {
    if std::env::var("BITNET_QWEN_TRACE_SOURCE_ORDER_QPROJ_CANDIDATE").as_deref() != Ok("1")
        || !qwen_trace_layer_enabled(layer_idx)
    {
        return Ok(());
    }
    if tensor_name != SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR {
        return Ok(());
    }
    let Some(descriptor) = hooks.get(tensor_name) else {
        return Ok(());
    };
    let boundary = DenseLinearRuntimeHookBoundary::from_sidecar_descriptor(tensor_name, descriptor);
    if !boundary.source_order_q8_matvec_candidate {
        return Ok(());
    }
    let Some(payload) = descriptor.packed_q8_payload.as_ref() else {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 candidate capture for {tensor_name} requires payload bytes"
        )));
    };
    let source_input_dim = boundary.source_order_input_dim.ok_or_else(|| {
        BitNetError::Validation(format!(
            "source-order Q8 candidate capture for {tensor_name} is missing source input dim"
        ))
    })?;
    let source_output_dim = boundary.source_order_output_dim.ok_or_else(|| {
        BitNetError::Validation(format!(
            "source-order Q8 candidate capture for {tensor_name} is missing source output dim"
        ))
    })?;
    let Some((&input_cols, _)) = input.dims().split_last() else {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 candidate capture for {tensor_name} requires input rank >= 1"
        )));
    };
    if input_cols != source_input_dim {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 candidate capture input cols {input_cols} do not match source input dim {source_input_dim} for {tensor_name}"
        )));
    }
    let Some((&output_cols, _)) = eager_output.dims().split_last() else {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 candidate capture for {tensor_name} requires output rank >= 1"
        )));
    };
    if output_cols != source_output_dim {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 candidate capture eager output cols {output_cols} do not match source output dim {source_output_dim} for {tensor_name}"
        )));
    }

    let input_values = input.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if !input_values.len().is_multiple_of(source_input_dim) {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 candidate capture input value count {} is not divisible by source input dim {source_input_dim} for {tensor_name}",
            input_values.len()
        )));
    }
    let eager_values = eager_output.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let input_rows = input_values.len() / source_input_dim;
    let expected_output_len = input_rows.checked_mul(source_output_dim).ok_or_else(|| {
        BitNetError::Validation(format!(
            "source-order Q8 candidate capture output length overflows for {tensor_name}"
        ))
    })?;
    if eager_values.len() != expected_output_len {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 candidate capture eager output len {} does not match expected len {expected_output_len} for {tensor_name}",
            eager_values.len()
        )));
    }

    let bias_values = match linear.bias() {
        Some(bias) => Some(bias.to_dtype(DType::F32)?.to_vec1::<f32>()?),
        None => None,
    };
    if let Some(bias_values) = bias_values.as_ref()
        && bias_values.len() != source_output_dim
    {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 candidate capture bias length {} does not match source output dim {source_output_dim} for {tensor_name}",
            bias_values.len()
        )));
    }

    let mut candidate = vec![0.0f32; expected_output_len];
    dense_q8_source_order_qproj_matvec_into(
        &input_values,
        bias_values.as_deref(),
        payload,
        source_input_dim,
        source_output_dim,
        &mut candidate,
    )?;
    let dense_hook_identity =
        boundary.source_order_candidate_receipt_identity.unwrap_or_else(|| {
            format!("{tensor_name}:source_order_q8_0_qproj_matvec:runtime_disabled")
        });
    qwen_trace_source_order_q8_candidate(QwenTraceSourceOrderQ8Candidate {
        stage: QWEN_QPROJ_SOURCE_ORDER_Q8_CANDIDATE_STAGE,
        layer_idx,
        source_tensor: tensor_name,
        gguf_tensor: "blk.0.attn_q.weight",
        boundary: QWEN_QPROJ_SOURCE_ORDER_Q8_CANDIDATE_BOUNDARY,
        dense_hook_identity: &dense_hook_identity,
        candidate_path: boundary
            .source_order_selected_path
            .unwrap_or("source_order_q8_0_qproj_matvec"),
        candidate_kernel: boundary
            .source_order_selected_kernel
            .unwrap_or("dense-q8-source-order-qproj-matvec"),
        source_input_dim,
        source_output_dim,
        input_rows,
        candidate: &candidate,
        eager: &eager_values,
    });
    maybe_trace_dense_q8_source_order_qproj_accumulator_audit(
        &input_values,
        bias_values.as_deref(),
        payload,
        source_input_dim,
        source_output_dim,
        &candidate,
        &eager_values,
        &dense_hook_identity,
        tensor_name,
        layer_idx,
    )?;
    maybe_trace_dense_q8_source_order_qproj_candle_slice_compare(
        &input_values,
        bias_values.as_deref(),
        payload,
        linear,
        source_input_dim,
        source_output_dim,
        &candidate,
        &eager_values,
        &dense_hook_identity,
        tensor_name,
        layer_idx,
    )?;
    maybe_trace_dense_q8_source_order_qproj_row_mapping_proof(
        &input_values,
        bias_values.as_deref(),
        payload,
        linear,
        source_input_dim,
        source_output_dim,
        &candidate,
        &eager_values,
        &dense_hook_identity,
        tensor_name,
        layer_idx,
    )?;
    Ok(())
}

fn source_order_qproj_accumulator_audit_enabled() -> bool {
    std::env::var("BITNET_QWEN_TRACE_SOURCE_ORDER_QPROJ_ACCUMULATOR_AUDIT").as_deref() == Ok("1")
}

fn source_order_qproj_candle_slice_compare_enabled() -> bool {
    std::env::var("BITNET_QWEN_TRACE_SOURCE_ORDER_QPROJ_CANDLE_SLICE_COMPARE").as_deref() == Ok("1")
}

fn source_order_qproj_row_mapping_proof_enabled() -> bool {
    std::env::var("BITNET_QWEN_TRACE_SOURCE_ORDER_QPROJ_ROW_MAPPING_PROOF").as_deref() == Ok("1")
}

fn source_order_qproj_accumulator_audit_indices(source_output_dim: usize) -> Vec<usize> {
    let mut indices = Vec::new();
    if let Ok(raw) = std::env::var("BITNET_QWEN_TRACE_SOURCE_ORDER_QPROJ_ACCUMULATOR_INDICES") {
        for part in raw.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Ok(index) = part.parse::<usize>()
                && index < source_output_dim
                && !indices.contains(&index)
            {
                indices.push(index);
            }
        }
    }
    if indices.is_empty() {
        for index in [0usize, 1419, 1970] {
            if index < source_output_dim && !indices.contains(&index) {
                indices.push(index);
            }
        }
        if indices.is_empty() && source_output_dim > 0 {
            indices.push(0);
        }
    }
    indices
}

#[allow(clippy::too_many_arguments)]
fn maybe_trace_dense_q8_source_order_qproj_accumulator_audit(
    input_values: &[f32],
    bias_values: Option<&[f32]>,
    payload: &DenseLinearPackedQ8Payload,
    source_input_dim: usize,
    source_output_dim: usize,
    candidate: &[f32],
    eager: &[f32],
    dense_hook_identity: &str,
    tensor_name: &str,
    layer_idx: usize,
) -> Result<()> {
    if !source_order_qproj_accumulator_audit_enabled() || !qwen_trace_layer_enabled(layer_idx) {
        return Ok(());
    }
    if input_values.len() < source_input_dim || candidate.len() < source_output_dim {
        return Ok(());
    }

    let input_row = &input_values[..source_input_dim];
    let indices = source_order_qproj_accumulator_audit_indices(source_output_dim);
    let mut entry_data = Vec::with_capacity(indices.len());
    for output_index in indices {
        let initial_bias = bias_values.map_or(0.0, |bias| bias[output_index]);
        let (candidate_output, terms_json) = dense_q8_source_order_qproj_accumulator_audit_entry(
            input_row,
            initial_bias,
            payload,
            source_input_dim,
            source_output_dim,
            output_index,
        )?;
        let eager_output = eager.get(output_index).copied().unwrap_or(0.0);
        entry_data.push((
            output_index,
            initial_bias,
            candidate_output,
            eager_output,
            (candidate_output - eager_output).abs(),
            terms_json,
        ));
    }
    let entries = entry_data
        .iter()
        .map(
            |(
                output_index,
                initial_bias,
                candidate_output,
                eager_output,
                abs_diff_vs_eager,
                partial_terms_json,
            )| QwenTraceSourceOrderQ8AccumulatorAuditEntry {
                output_index: *output_index,
                initial_bias: *initial_bias,
                candidate_output: *candidate_output,
                eager_output: *eager_output,
                abs_diff_vs_eager: *abs_diff_vs_eager,
                partial_terms_json,
            },
        )
        .collect::<Vec<_>>();
    qwen_trace_source_order_q8_accumulator_audit(QwenTraceSourceOrderQ8AccumulatorAudit {
        stage: QWEN_QPROJ_SOURCE_ORDER_Q8_ACCUMULATOR_AUDIT_STAGE,
        layer_idx,
        source_tensor: tensor_name,
        gguf_tensor: "blk.0.attn_q.weight",
        boundary: QWEN_QPROJ_SOURCE_ORDER_Q8_ACCUMULATOR_AUDIT_BOUNDARY,
        dense_hook_identity,
        source_input_dim,
        source_output_dim,
        input_row: 0,
        q8_block_size: payload.q8_block_size,
        entries: &entries,
    });
    for (output_index, _, candidate_output, _, _, _) in entry_data {
        if let Some(existing) = candidate.get(output_index) {
            let diff = (candidate_output - *existing).abs();
            if diff > 1e-4 {
                return Err(BitNetError::Validation(format!(
                    "source-order Q8 q_proj accumulator audit output {output_index} recomputed {candidate_output} but candidate vector has {existing}"
                )));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maybe_trace_dense_q8_source_order_qproj_candle_slice_compare(
    input_values: &[f32],
    bias_values: Option<&[f32]>,
    payload: &DenseLinearPackedQ8Payload,
    linear: &Linear,
    source_input_dim: usize,
    source_output_dim: usize,
    candidate: &[f32],
    eager: &[f32],
    dense_hook_identity: &str,
    tensor_name: &str,
    layer_idx: usize,
) -> Result<()> {
    if !source_order_qproj_candle_slice_compare_enabled() || !qwen_trace_layer_enabled(layer_idx) {
        return Ok(());
    }
    if input_values.len() < source_input_dim || candidate.len() < source_output_dim {
        return Ok(());
    }
    if linear.weight().dims() != [source_output_dim, source_input_dim] {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 q_proj Candle slice compare expected Candle weight dims [{source_output_dim}, {source_input_dim}] for {tensor_name}, got {:?}",
            linear.weight().dims()
        )));
    }

    let candle_weight = linear.weight().to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let expected_candle_values =
        source_output_dim.checked_mul(source_input_dim).ok_or_else(|| {
            BitNetError::Validation(
                "source-order Q8 q_proj Candle slice compare value count overflow".to_string(),
            )
        })?;
    if candle_weight.len() != expected_candle_values {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 q_proj Candle slice compare materialized {} values, expected {expected_candle_values}",
            candle_weight.len()
        )));
    }

    let input_row = &input_values[..source_input_dim];
    let indices = source_order_qproj_accumulator_audit_indices(source_output_dim);
    struct OwnedEntry {
        output_index: usize,
        initial_bias: f32,
        source_order_output: f32,
        candle_recomputed_output: f32,
        eager_output: f32,
        abs_diff_source_order_vs_candle: f32,
        abs_diff_candle_vs_eager: f32,
        terms_json: String,
    }
    let mut entry_data = Vec::with_capacity(indices.len());
    for output_index in indices {
        let initial_bias = bias_values.map_or(0.0, |bias| bias[output_index]);
        let (candle_recomputed_output, terms_json) =
            dense_q8_source_order_qproj_candle_slice_compare_entry(
                input_row,
                initial_bias,
                payload,
                &candle_weight,
                source_input_dim,
                source_output_dim,
                output_index,
            )?;
        let source_order_output = candidate.get(output_index).copied().unwrap_or(0.0);
        let eager_output = eager.get(output_index).copied().unwrap_or(0.0);
        entry_data.push(OwnedEntry {
            output_index,
            initial_bias,
            source_order_output,
            candle_recomputed_output,
            eager_output,
            abs_diff_source_order_vs_candle: (source_order_output - candle_recomputed_output).abs(),
            abs_diff_candle_vs_eager: (candle_recomputed_output - eager_output).abs(),
            terms_json,
        });
    }
    let entries = entry_data
        .iter()
        .map(|entry| QwenTraceSourceOrderQ8CandleSliceCompareEntry {
            output_index: entry.output_index,
            initial_bias: entry.initial_bias,
            source_order_output: entry.source_order_output,
            candle_recomputed_output: entry.candle_recomputed_output,
            eager_output: entry.eager_output,
            abs_diff_source_order_vs_candle: entry.abs_diff_source_order_vs_candle,
            abs_diff_candle_vs_eager: entry.abs_diff_candle_vs_eager,
            terms_json: &entry.terms_json,
        })
        .collect::<Vec<_>>();
    qwen_trace_source_order_q8_candle_slice_compare(QwenTraceSourceOrderQ8CandleSliceCompare {
        stage: QWEN_QPROJ_SOURCE_ORDER_Q8_CANDLE_SLICE_COMPARE_STAGE,
        layer_idx,
        source_tensor: tensor_name,
        gguf_tensor: "blk.0.attn_q.weight",
        boundary: QWEN_QPROJ_SOURCE_ORDER_Q8_CANDLE_SLICE_COMPARE_BOUNDARY,
        dense_hook_identity,
        source_input_dim,
        source_output_dim,
        input_row: 0,
        q8_block_size: payload.q8_block_size,
        entries: &entries,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maybe_trace_dense_q8_source_order_qproj_row_mapping_proof(
    input_values: &[f32],
    bias_values: Option<&[f32]>,
    payload: &DenseLinearPackedQ8Payload,
    linear: &Linear,
    source_input_dim: usize,
    source_output_dim: usize,
    candidate: &[f32],
    eager: &[f32],
    dense_hook_identity: &str,
    tensor_name: &str,
    layer_idx: usize,
) -> Result<()> {
    if !source_order_qproj_row_mapping_proof_enabled() || !qwen_trace_layer_enabled(layer_idx) {
        return Ok(());
    }
    if input_values.len() < source_input_dim || candidate.len() < source_output_dim {
        return Ok(());
    }
    if linear.weight().dims() != [source_output_dim, source_input_dim] {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 q_proj row mapping proof expected Candle weight dims [{source_output_dim}, {source_input_dim}] for {tensor_name}, got {:?}",
            linear.weight().dims()
        )));
    }

    let candle_weight = linear.weight().to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let expected_candle_values =
        source_output_dim.checked_mul(source_input_dim).ok_or_else(|| {
            BitNetError::Validation(
                "source-order Q8 q_proj row mapping proof value count overflow".to_string(),
            )
        })?;
    if candle_weight.len() != expected_candle_values {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 q_proj row mapping proof materialized {} values, expected {expected_candle_values}",
            candle_weight.len()
        )));
    }

    let input_row = &input_values[..source_input_dim];
    let indices = source_order_qproj_accumulator_audit_indices(source_output_dim);
    struct OwnedEntry {
        output_index: usize,
        initial_bias: f32,
        source_order_output: f32,
        mapped_recomputed_output: f32,
        candle_recomputed_output: f32,
        eager_output: f32,
        abs_diff_mapped_vs_candle: f32,
        abs_diff_mapped_vs_eager: f32,
        terms_json: String,
    }
    let mut entry_data = Vec::with_capacity(indices.len());
    for output_index in indices {
        let initial_bias = bias_values.map_or(0.0, |bias| bias[output_index]);
        let (mapped_recomputed_output, candle_recomputed_output, terms_json) =
            dense_q8_source_order_qproj_row_mapping_proof_entry(
                input_row,
                initial_bias,
                payload,
                &candle_weight,
                source_input_dim,
                source_output_dim,
                output_index,
            )?;
        let source_order_output = candidate.get(output_index).copied().unwrap_or(0.0);
        let eager_output = eager.get(output_index).copied().unwrap_or(0.0);
        entry_data.push(OwnedEntry {
            output_index,
            initial_bias,
            source_order_output,
            mapped_recomputed_output,
            candle_recomputed_output,
            eager_output,
            abs_diff_mapped_vs_candle: (mapped_recomputed_output - candle_recomputed_output).abs(),
            abs_diff_mapped_vs_eager: (mapped_recomputed_output - eager_output).abs(),
            terms_json,
        });
    }
    let entries = entry_data
        .iter()
        .map(|entry| QwenTraceSourceOrderQ8RowMappingProofEntry {
            output_index: entry.output_index,
            initial_bias: entry.initial_bias,
            source_order_output: entry.source_order_output,
            mapped_recomputed_output: entry.mapped_recomputed_output,
            candle_recomputed_output: entry.candle_recomputed_output,
            eager_output: entry.eager_output,
            abs_diff_mapped_vs_candle: entry.abs_diff_mapped_vs_candle,
            abs_diff_mapped_vs_eager: entry.abs_diff_mapped_vs_eager,
            terms_json: &entry.terms_json,
        })
        .collect::<Vec<_>>();
    qwen_trace_source_order_q8_row_mapping_proof(QwenTraceSourceOrderQ8RowMappingProof {
        stage: QWEN_QPROJ_SOURCE_ORDER_Q8_ROW_MAPPING_PROOF_STAGE,
        layer_idx,
        source_tensor: tensor_name,
        gguf_tensor: "blk.0.attn_q.weight",
        boundary: QWEN_QPROJ_SOURCE_ORDER_Q8_ROW_MAPPING_PROOF_BOUNDARY,
        dense_hook_identity,
        source_input_dim,
        source_output_dim,
        input_row: 0,
        q8_block_size: payload.q8_block_size,
        entries: &entries,
    });
    Ok(())
}

struct DenseQ8AccumulatorTerm {
    input_index: usize,
    weight_idx: usize,
    q8_block_index: usize,
    q8_block_value_offset: usize,
    q8_block_scale: f32,
    q: i8,
    input_value: f32,
    contribution: f32,
    partial_sum_after: f32,
    term_kind: &'static str,
}

struct DenseQ8CandleSliceTerm {
    term_kind: &'static str,
    input_index: usize,
    source_weight_idx: usize,
    candle_weight_idx: usize,
    q8_block_index: usize,
    q8_block_value_offset: usize,
    q8_block_scale: f32,
    q: i8,
    source_order_weight_value: f32,
    candle_weight_value: f32,
    input_value: f32,
    source_order_contribution: f32,
    candle_contribution: f32,
    contribution_delta: f32,
    source_order_partial_sum_after: f32,
    candle_partial_sum_after: f32,
}

struct DenseQ8RowMappingTerm {
    term_kind: &'static str,
    input_index: usize,
    source_order_weight_idx: usize,
    runtime_weight_idx: usize,
    candle_weight_idx: usize,
    source_q8_block_index: usize,
    source_q8_block_value_offset: usize,
    runtime_q8_block_index: usize,
    runtime_q8_block_value_offset: usize,
    source_q8_block_scale: f32,
    runtime_q8_block_scale: f32,
    source_q: i8,
    runtime_q: i8,
    source_order_weight_value: f32,
    mapped_weight_value: f32,
    candle_weight_value: f32,
    input_value: f32,
    mapped_contribution: f32,
    candle_contribution: f32,
    contribution_delta_mapped_vs_candle: f32,
    mapped_partial_sum_after: f32,
    candle_partial_sum_after: f32,
}

fn dense_q8_source_order_qproj_accumulator_audit_entry(
    input_row: &[f32],
    initial_bias: f32,
    payload: &DenseLinearPackedQ8Payload,
    source_input_dim: usize,
    source_output_dim: usize,
    output_index: usize,
) -> Result<(f32, String)> {
    if output_index >= source_output_dim {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 q_proj accumulator audit output index {output_index} is out of range for source output dim {source_output_dim}"
        )));
    }
    let block_stride = 2 + payload.q8_block_size;
    let mut sum = initial_bias;
    let mut first_terms = Vec::new();
    let mut max_abs_term: Option<DenseQ8AccumulatorTerm> = None;
    for (in_row, input_value) in input_row.iter().enumerate() {
        let weight_idx = output_index * source_input_dim + in_row;
        let block_idx = weight_idx / payload.q8_block_size;
        let block_value_offset = weight_idx % payload.q8_block_size;
        let block_offset = block_idx * block_stride;
        let scale = dense_q8_sidecar_block_scale(&payload.packed_q8_bytes, block_offset);
        let q_idx = block_offset + 2 + block_value_offset;
        let q = payload.packed_q8_bytes.get(q_idx).copied().ok_or_else(|| {
            BitNetError::Validation(format!(
                "source-order Q8 q_proj accumulator audit q byte index {q_idx} is out of range"
            ))
        })? as i8;
        let contribution = scale * f32::from(q) * *input_value;
        sum += contribution;
        if first_terms.len() < 8 {
            first_terms.push(DenseQ8AccumulatorTerm {
                input_index: in_row,
                weight_idx,
                q8_block_index: block_idx,
                q8_block_value_offset: block_value_offset,
                q8_block_scale: scale,
                q,
                input_value: *input_value,
                contribution,
                partial_sum_after: sum,
                term_kind: "prefix",
            });
        }
        let replace_max = match max_abs_term.as_ref() {
            Some(existing) => contribution.abs() > existing.contribution.abs(),
            None => true,
        };
        if replace_max {
            max_abs_term = Some(DenseQ8AccumulatorTerm {
                input_index: in_row,
                weight_idx,
                q8_block_index: block_idx,
                q8_block_value_offset: block_value_offset,
                q8_block_scale: scale,
                q,
                input_value: *input_value,
                contribution,
                partial_sum_after: sum,
                term_kind: "max_abs_contribution",
            });
        }
    }
    let mut terms_json = first_terms
        .into_iter()
        .map(|term| dense_q8_source_order_qproj_accumulator_term_json(&term))
        .collect::<Vec<_>>();
    if let Some(term) = max_abs_term
        && term.input_index >= 8
    {
        terms_json.push(dense_q8_source_order_qproj_accumulator_term_json(&term));
    }
    Ok((sum, terms_json.join(",")))
}

fn dense_q8_source_order_qproj_accumulator_term_json(term: &DenseQ8AccumulatorTerm) -> String {
    format!(
        "{{\"term_kind\":\"{}\",\"input_index\":{},\"weight_idx\":{},\"q8_block_index\":{},\"q8_block_value_offset\":{},\"q8_block_scale\":{},\"q\":{},\"input_value\":{},\"contribution\":{},\"partial_sum_after\":{}}}",
        term.term_kind,
        term.input_index,
        term.weight_idx,
        term.q8_block_index,
        term.q8_block_value_offset,
        qwen_trace_number(f64::from(term.q8_block_scale)),
        term.q,
        qwen_trace_number(f64::from(term.input_value)),
        qwen_trace_number(f64::from(term.contribution)),
        qwen_trace_number(f64::from(term.partial_sum_after))
    )
}

#[allow(clippy::too_many_arguments)]
fn dense_q8_source_order_qproj_candle_slice_compare_entry(
    input_row: &[f32],
    initial_bias: f32,
    payload: &DenseLinearPackedQ8Payload,
    candle_weight: &[f32],
    source_input_dim: usize,
    source_output_dim: usize,
    output_index: usize,
) -> Result<(f32, String)> {
    let block_stride = 2 + payload.q8_block_size;
    let mut source_sum = initial_bias;
    let mut candle_sum = initial_bias;
    let mut first_terms = Vec::new();
    let mut max_abs_delta_term: Option<DenseQ8CandleSliceTerm> = None;
    for (input_index, input_value) in input_row.iter().enumerate() {
        let source_weight_idx = input_index * source_output_dim + output_index;
        let q8_block_index = source_weight_idx / payload.q8_block_size;
        let q8_block_value_offset = source_weight_idx % payload.q8_block_size;
        let block_offset = q8_block_index * block_stride;
        let q8_block_scale = dense_q8_sidecar_block_scale(&payload.packed_q8_bytes, block_offset);
        let q_idx = block_offset + 2 + q8_block_value_offset;
        let q = payload.packed_q8_bytes.get(q_idx).copied().ok_or_else(|| {
            BitNetError::Validation(format!(
                "source-order Q8 q_proj Candle slice compare q byte index {q_idx} is out of range"
            ))
        })? as i8;
        let source_order_weight_value = q8_block_scale * f32::from(q);
        let candle_weight_idx = output_index
            .checked_mul(source_input_dim)
            .and_then(|base| base.checked_add(input_index))
            .ok_or_else(|| {
                BitNetError::Validation(
                    "source-order Q8 q_proj Candle slice compare weight index overflow".to_string(),
                )
            })?;
        let candle_weight_value = candle_weight.get(candle_weight_idx).copied().ok_or_else(|| {
            BitNetError::Validation(format!(
                "source-order Q8 q_proj Candle slice compare weight index {candle_weight_idx} is out of range"
            ))
        })?;
        let source_order_contribution = source_order_weight_value * *input_value;
        let candle_contribution = candle_weight_value * *input_value;
        source_sum += source_order_contribution;
        candle_sum += candle_contribution;
        let contribution_delta = source_order_contribution - candle_contribution;
        let term_kind = if first_terms.len() < 8 { Some("prefix") } else { None };
        let term = DenseQ8CandleSliceTerm {
            term_kind: term_kind.unwrap_or("max_abs_contribution_delta"),
            input_index,
            source_weight_idx,
            candle_weight_idx,
            q8_block_index,
            q8_block_value_offset,
            q8_block_scale,
            q,
            source_order_weight_value,
            candle_weight_value,
            input_value: *input_value,
            source_order_contribution,
            candle_contribution,
            contribution_delta,
            source_order_partial_sum_after: source_sum,
            candle_partial_sum_after: candle_sum,
        };
        if term_kind.is_some() {
            first_terms.push(term);
        } else {
            let replace_max = match max_abs_delta_term.as_ref() {
                Some(existing) => contribution_delta.abs() > existing.contribution_delta.abs(),
                None => true,
            };
            if replace_max {
                max_abs_delta_term = Some(term);
            }
        }
    }
    let mut terms_json = first_terms
        .into_iter()
        .map(|term| dense_q8_source_order_qproj_candle_slice_term_json(&term))
        .collect::<Vec<_>>();
    if let Some(term) = max_abs_delta_term {
        terms_json.push(dense_q8_source_order_qproj_candle_slice_term_json(&term));
    }
    Ok((candle_sum, terms_json.join(",")))
}

fn dense_q8_source_order_qproj_candle_slice_term_json(term: &DenseQ8CandleSliceTerm) -> String {
    format!(
        "{{\"term_kind\":\"{}\",\"input_index\":{},\"source_weight_idx\":{},\"candle_weight_idx\":{},\"q8_block_index\":{},\"q8_block_value_offset\":{},\"q8_block_scale\":{},\"q\":{},\"source_order_weight_value\":{},\"candle_weight_value\":{},\"input_value\":{},\"source_order_contribution\":{},\"candle_contribution\":{},\"contribution_delta\":{},\"source_order_partial_sum_after\":{},\"candle_partial_sum_after\":{}}}",
        term.term_kind,
        term.input_index,
        term.source_weight_idx,
        term.candle_weight_idx,
        term.q8_block_index,
        term.q8_block_value_offset,
        qwen_trace_number(f64::from(term.q8_block_scale)),
        term.q,
        qwen_trace_number(f64::from(term.source_order_weight_value)),
        qwen_trace_number(f64::from(term.candle_weight_value)),
        qwen_trace_number(f64::from(term.input_value)),
        qwen_trace_number(f64::from(term.source_order_contribution)),
        qwen_trace_number(f64::from(term.candle_contribution)),
        qwen_trace_number(f64::from(term.contribution_delta)),
        qwen_trace_number(f64::from(term.source_order_partial_sum_after)),
        qwen_trace_number(f64::from(term.candle_partial_sum_after))
    )
}

fn dense_q8_payload_value_at(
    payload: &DenseLinearPackedQ8Payload,
    weight_idx: usize,
) -> Result<(usize, usize, f32, i8, f32)> {
    let block_stride = 2 + payload.q8_block_size;
    let block_index = weight_idx / payload.q8_block_size;
    let block_value_offset = weight_idx % payload.q8_block_size;
    let block_offset = block_index * block_stride;
    let q8_block_scale = dense_q8_sidecar_block_scale(&payload.packed_q8_bytes, block_offset);
    let q_idx = block_offset + 2 + block_value_offset;
    let q = payload.packed_q8_bytes.get(q_idx).copied().ok_or_else(|| {
        BitNetError::Validation(format!(
            "source-order Q8 q_proj row mapping proof q byte index {q_idx} is out of range"
        ))
    })? as i8;
    Ok((block_index, block_value_offset, q8_block_scale, q, q8_block_scale * f32::from(q)))
}

fn dense_q8_source_order_qproj_row_mapping_proof_entry(
    input_row: &[f32],
    initial_bias: f32,
    payload: &DenseLinearPackedQ8Payload,
    candle_weight: &[f32],
    source_input_dim: usize,
    source_output_dim: usize,
    output_index: usize,
) -> Result<(f32, f32, String)> {
    let mut mapped_sum = initial_bias;
    let mut candle_sum = initial_bias;
    let mut first_terms = Vec::new();
    let mut max_abs_delta_term: Option<DenseQ8RowMappingTerm> = None;
    for (input_index, input_value) in input_row.iter().enumerate() {
        let source_order_weight_idx = input_index
            .checked_mul(source_output_dim)
            .and_then(|base| base.checked_add(output_index))
            .ok_or_else(|| {
                BitNetError::Validation(
                    "source-order Q8 q_proj row mapping proof source index overflow".to_string(),
                )
            })?;
        let runtime_weight_idx = output_index
            .checked_mul(source_input_dim)
            .and_then(|base| base.checked_add(input_index))
            .ok_or_else(|| {
                BitNetError::Validation(
                    "source-order Q8 q_proj row mapping proof runtime index overflow".to_string(),
                )
            })?;
        let (
            source_q8_block_index,
            source_q8_block_value_offset,
            source_q8_block_scale,
            source_q,
            source_order_weight_value,
        ) = dense_q8_payload_value_at(payload, source_order_weight_idx)?;
        let (
            runtime_q8_block_index,
            runtime_q8_block_value_offset,
            runtime_q8_block_scale,
            runtime_q,
            mapped_weight_value,
        ) = dense_q8_payload_value_at(payload, runtime_weight_idx)?;
        let candle_weight_idx = runtime_weight_idx;
        let candle_weight_value = candle_weight.get(candle_weight_idx).copied().ok_or_else(|| {
            BitNetError::Validation(format!(
                "source-order Q8 q_proj row mapping proof Candle weight index {candle_weight_idx} is out of range"
            ))
        })?;
        let mapped_contribution = mapped_weight_value * *input_value;
        let candle_contribution = candle_weight_value * *input_value;
        mapped_sum += mapped_contribution;
        candle_sum += candle_contribution;
        let contribution_delta_mapped_vs_candle = mapped_contribution - candle_contribution;
        let term_kind = if first_terms.len() < 8 { Some("prefix") } else { None };
        let term = DenseQ8RowMappingTerm {
            term_kind: term_kind.unwrap_or("max_abs_contribution_delta"),
            input_index,
            source_order_weight_idx,
            runtime_weight_idx,
            candle_weight_idx,
            source_q8_block_index,
            source_q8_block_value_offset,
            runtime_q8_block_index,
            runtime_q8_block_value_offset,
            source_q8_block_scale,
            runtime_q8_block_scale,
            source_q,
            runtime_q,
            source_order_weight_value,
            mapped_weight_value,
            candle_weight_value,
            input_value: *input_value,
            mapped_contribution,
            candle_contribution,
            contribution_delta_mapped_vs_candle,
            mapped_partial_sum_after: mapped_sum,
            candle_partial_sum_after: candle_sum,
        };
        if term_kind.is_some() {
            first_terms.push(term);
        } else {
            let replace_max = match max_abs_delta_term.as_ref() {
                Some(existing) => {
                    contribution_delta_mapped_vs_candle.abs()
                        > existing.contribution_delta_mapped_vs_candle.abs()
                }
                None => true,
            };
            if replace_max {
                max_abs_delta_term = Some(term);
            }
        }
    }
    let mut terms_json = first_terms
        .into_iter()
        .map(|term| dense_q8_source_order_qproj_row_mapping_term_json(&term))
        .collect::<Vec<_>>();
    if let Some(term) = max_abs_delta_term {
        terms_json.push(dense_q8_source_order_qproj_row_mapping_term_json(&term));
    }
    Ok((mapped_sum, candle_sum, terms_json.join(",")))
}

fn dense_q8_source_order_qproj_row_mapping_term_json(term: &DenseQ8RowMappingTerm) -> String {
    format!(
        "{{\"term_kind\":\"{}\",\"input_index\":{},\"source_order_weight_idx\":{},\"runtime_weight_idx\":{},\"candle_weight_idx\":{},\"source_q8_block_index\":{},\"source_q8_block_value_offset\":{},\"runtime_q8_block_index\":{},\"runtime_q8_block_value_offset\":{},\"source_q8_block_scale\":{},\"runtime_q8_block_scale\":{},\"source_q\":{},\"runtime_q\":{},\"source_order_weight_value\":{},\"mapped_weight_value\":{},\"candle_weight_value\":{},\"input_value\":{},\"mapped_contribution\":{},\"candle_contribution\":{},\"contribution_delta_mapped_vs_candle\":{},\"mapped_partial_sum_after\":{},\"candle_partial_sum_after\":{}}}",
        term.term_kind,
        term.input_index,
        term.source_order_weight_idx,
        term.runtime_weight_idx,
        term.candle_weight_idx,
        term.source_q8_block_index,
        term.source_q8_block_value_offset,
        term.runtime_q8_block_index,
        term.runtime_q8_block_value_offset,
        qwen_trace_number(f64::from(term.source_q8_block_scale)),
        qwen_trace_number(f64::from(term.runtime_q8_block_scale)),
        term.source_q,
        term.runtime_q,
        qwen_trace_number(f64::from(term.source_order_weight_value)),
        qwen_trace_number(f64::from(term.mapped_weight_value)),
        qwen_trace_number(f64::from(term.candle_weight_value)),
        qwen_trace_number(f64::from(term.input_value)),
        qwen_trace_number(f64::from(term.mapped_contribution)),
        qwen_trace_number(f64::from(term.candle_contribution)),
        qwen_trace_number(f64::from(term.contribution_delta_mapped_vs_candle)),
        qwen_trace_number(f64::from(term.mapped_partial_sum_after)),
        qwen_trace_number(f64::from(term.candle_partial_sum_after))
    )
}

fn dense_q8_source_order_qproj_matvec_into(
    input_values: &[f32],
    bias_values: Option<&[f32]>,
    payload: &DenseLinearPackedQ8Payload,
    source_input_dim: usize,
    source_output_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    if source_input_dim == 0 || source_output_dim == 0 {
        return Err(BitNetError::Validation(
            "source-order Q8 q_proj matvec requires nonzero source dimensions".to_string(),
        ));
    }
    if !input_values.len().is_multiple_of(source_input_dim) {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 q_proj matvec input len {} is not divisible by source input dim {source_input_dim}",
            input_values.len()
        )));
    }
    let input_rows = input_values.len() / source_input_dim;
    let expected_output = input_rows.checked_mul(source_output_dim).ok_or_else(|| {
        BitNetError::Validation("source-order Q8 q_proj matvec output length overflow".to_string())
    })?;
    if output.len() != expected_output {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 q_proj matvec output len {} does not match expected len {expected_output}",
            output.len()
        )));
    }
    let value_count = source_input_dim.checked_mul(source_output_dim).ok_or_else(|| {
        BitNetError::Validation("source-order Q8 q_proj matvec value count overflow".to_string())
    })?;
    let expected_blocks = value_count.div_ceil(payload.q8_block_size);
    if expected_blocks != payload.q8_block_count {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 q_proj matvec expected {expected_blocks} blocks from source dims [{source_input_dim}, {source_output_dim}], payload has {}",
            payload.q8_block_count
        )));
    }
    if payload.expected_q8_payload_len() != Some(payload.payload_len()) {
        return Err(BitNetError::Validation(
            "source-order Q8 q_proj matvec payload length does not match q8 block contract"
                .to_string(),
        ));
    }
    if let Some(bias_values) = bias_values
        && bias_values.len() != source_output_dim
    {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 q_proj matvec bias len {} does not match source output dim {source_output_dim}",
            bias_values.len()
        )));
    }

    let block_stride = 2 + payload.q8_block_size;
    for (input_row_idx, input_row) in input_values.chunks_exact(source_input_dim).enumerate() {
        for out_col in 0..source_output_dim {
            let mut sum = bias_values.map_or(0.0, |bias| bias[out_col]);
            for (in_row, input_value) in input_row.iter().enumerate().take(source_input_dim) {
                let weight_idx = out_col * source_input_dim + in_row;
                let block_idx = weight_idx / payload.q8_block_size;
                let block_value_offset = weight_idx % payload.q8_block_size;
                let block_offset = block_idx * block_stride;
                let scale = dense_q8_sidecar_block_scale(&payload.packed_q8_bytes, block_offset);
                let q_idx = block_offset + 2 + block_value_offset;
                let q = payload.packed_q8_bytes[q_idx] as i8;
                sum += scale * f32::from(q) * *input_value;
            }
            output[input_row_idx * source_output_dim + out_col] = sum;
        }
    }
    Ok(())
}

fn dense_q8_source_order_qproj_linear_forward(
    input: &Tensor,
    bias: Option<&Tensor>,
    payload: &DenseLinearPackedQ8Payload,
    source_input_dim: usize,
    source_output_dim: usize,
) -> Result<Tensor> {
    let dims = input.dims();
    let Some((&input_cols, prefix)) = dims.split_last() else {
        return Err(BitNetError::Validation(
            "source-order Q8 runtime hook requires a tensor with at least one dimension"
                .to_string(),
        ));
    };
    if input_cols != source_input_dim {
        return Err(BitNetError::Validation(format!(
            "source-order Q8 runtime hook input cols {input_cols} do not match source input dim {source_input_dim}"
        )));
    }

    let input_materialization_start = Instant::now();
    let input_values = input.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.input_materialization_calls, 1);
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.input_materialization_ns,
        elapsed_ns_u64(input_materialization_start),
    );
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.input_values_materialized,
        input_values.len() as u64,
    );

    let bias_materialization_start = Instant::now();
    let bias_values = match bias {
        Some(bias) => Some(bias.to_dtype(DType::F32)?.to_vec1::<f32>()?),
        None => None,
    };
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.bias_materialization_calls, 1);
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.bias_materialization_ns,
        elapsed_ns_u64(bias_materialization_start),
    );
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.bias_values_materialized,
        bias_values.as_ref().map_or(0, Vec::len) as u64,
    );

    let input_rows = input_values.len().checked_div(source_input_dim).unwrap_or(0);
    let output_values = input_rows.checked_mul(source_output_dim).ok_or_else(|| {
        BitNetError::Validation("source-order Q8 runtime hook output length overflow".to_string())
    })?;
    let mut output = vec![0.0f32; output_values];
    let matvec_start = Instant::now();
    dense_q8_source_order_qproj_matvec_into(
        &input_values,
        bias_values.as_deref(),
        payload,
        source_input_dim,
        source_output_dim,
        &mut output,
    )?;
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.packed_matvec_calls, 1);
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.packed_matvec_ns, elapsed_ns_u64(matvec_start));
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.packed_matvec_input_rows, input_rows as u64);
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.packed_matvec_output_values,
        output_values as u64,
    );

    let mut output_shape = prefix.to_vec();
    output_shape.push(source_output_dim);
    let output_construction_start = Instant::now();
    let tensor = Tensor::from_vec(output, output_shape, input.device()).map_err(BitNetError::from);
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.output_tensor_construction_calls, 1);
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.output_tensor_construction_ns,
        elapsed_ns_u64(output_construction_start),
    );
    tensor
}

fn dense_q8_sidecar_linear_forward(
    input: &Tensor,
    bias: Option<&Tensor>,
    payload: &DenseLinearPackedQ8Payload,
) -> candle_core::Result<Tensor> {
    let dims = input.dims();
    let Some((&input_cols, prefix)) = dims.split_last() else {
        candle_core::bail!("packed Q8 runtime hook requires a tensor with at least one dimension");
    };
    if input_cols != payload.matrix_cols {
        candle_core::bail!(
            "packed Q8 runtime hook input cols {} do not match payload matrix cols {}",
            input_cols,
            payload.matrix_cols
        );
    }
    let input_materialization_start = Instant::now();
    let input_values = input.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.input_materialization_calls, 1);
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.input_materialization_ns,
        elapsed_ns_u64(input_materialization_start),
    );
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.input_values_materialized,
        input_values.len() as u64,
    );
    if input_values.len() % payload.matrix_cols != 0 {
        candle_core::bail!(
            "packed Q8 runtime hook input value count {} is not divisible by cols {}",
            input_values.len(),
            payload.matrix_cols
        );
    }

    let bias_materialization_start = Instant::now();
    let bias_values = match bias {
        Some(bias) => Some(bias.to_dtype(DType::F32)?.to_vec1::<f32>()?),
        None => None,
    };
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.bias_materialization_calls, 1);
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.bias_materialization_ns,
        elapsed_ns_u64(bias_materialization_start),
    );
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.bias_values_materialized,
        bias_values.as_ref().map_or(0, Vec::len) as u64,
    );
    if let Some(bias_values) = bias_values.as_ref()
        && bias_values.len() != payload.matrix_rows
    {
        candle_core::bail!(
            "packed Q8 runtime hook bias length {} does not match rows {}",
            bias_values.len(),
            payload.matrix_rows
        );
    }

    let input_rows = input_values.len().checked_div(payload.matrix_cols).unwrap_or(0);
    let output_values = input_rows.saturating_mul(payload.matrix_rows);
    let mut output = Vec::with_capacity(output_values);
    let matvec_start = Instant::now();
    if payload.matrix_cols.is_multiple_of(payload.q8_block_size) {
        dense_q8_sidecar_matvec_block_aligned(
            &input_values,
            bias_values.as_deref(),
            payload,
            &mut output,
        );
    } else {
        dense_q8_sidecar_matvec_generic(
            &input_values,
            bias_values.as_deref(),
            payload,
            &mut output,
        );
    }
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.packed_matvec_calls, 1);
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.packed_matvec_ns, elapsed_ns_u64(matvec_start));
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.packed_matvec_input_rows, input_rows as u64);
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.packed_matvec_output_values,
        output_values as u64,
    );

    let mut output_shape = prefix.to_vec();
    output_shape.push(payload.matrix_rows);
    let output_construction_start = Instant::now();
    let tensor = Tensor::from_vec(output, output_shape, input.device());
    add_counter(&DENSE_Q8_SIDECAR_INSTRUMENTATION.output_tensor_construction_calls, 1);
    add_counter(
        &DENSE_Q8_SIDECAR_INSTRUMENTATION.output_tensor_construction_ns,
        elapsed_ns_u64(output_construction_start),
    );
    tensor
}

fn dense_q8_sidecar_matvec_block_aligned(
    input_values: &[f32],
    bias_values: Option<&[f32]>,
    payload: &DenseLinearPackedQ8Payload,
    output: &mut Vec<f32>,
) {
    let input_rows = input_values.len() / payload.matrix_cols;
    let output_start = output.len();
    let output_values = input_rows.saturating_mul(payload.matrix_rows);
    output.resize(output_start + output_values, 0.0);
    dense_q8_sidecar_matvec_block_aligned_into(
        input_values,
        bias_values,
        payload,
        &mut output[output_start..],
    );
}

fn dense_q8_sidecar_matvec_block_aligned_into(
    input_values: &[f32],
    bias_values: Option<&[f32]>,
    payload: &DenseLinearPackedQ8Payload,
    output: &mut [f32],
) {
    let block_stride = 2 + payload.q8_block_size;
    let blocks_per_row = payload.matrix_cols / payload.q8_block_size;
    debug_assert_eq!(output.len(), input_values.len() / payload.matrix_cols * payload.matrix_rows);
    for (input_row_idx, input_row) in input_values.chunks_exact(payload.matrix_cols).enumerate() {
        for row in 0..payload.matrix_rows {
            let mut sum = bias_values.map_or(0.0, |bias| bias[row]);
            let row_block_start = row * blocks_per_row;
            for block_in_row in 0..blocks_per_row {
                let input_start = block_in_row * payload.q8_block_size;
                let block_offset = (row_block_start + block_in_row) * block_stride;
                let scale = dense_q8_sidecar_block_scale(&payload.packed_q8_bytes, block_offset);
                let q_start = block_offset + 2;
                for offset in 0..payload.q8_block_size {
                    let q = payload.packed_q8_bytes[q_start + offset] as i8;
                    sum += scale * f32::from(q) * input_row[input_start + offset];
                }
            }
            output[input_row_idx * payload.matrix_rows + row] = sum;
        }
    }
}

fn dense_q8_sidecar_matvec_generic(
    input_values: &[f32],
    bias_values: Option<&[f32]>,
    payload: &DenseLinearPackedQ8Payload,
    output: &mut Vec<f32>,
) {
    let input_rows = input_values.len() / payload.matrix_cols;
    let output_start = output.len();
    let output_values = input_rows.saturating_mul(payload.matrix_rows);
    output.resize(output_start + output_values, 0.0);
    dense_q8_sidecar_matvec_generic_into(
        input_values,
        bias_values,
        payload,
        &mut output[output_start..],
    );
}

fn dense_q8_sidecar_matvec_generic_into(
    input_values: &[f32],
    bias_values: Option<&[f32]>,
    payload: &DenseLinearPackedQ8Payload,
    output: &mut [f32],
) {
    let block_stride = 2 + payload.q8_block_size;
    debug_assert_eq!(output.len(), input_values.len() / payload.matrix_cols * payload.matrix_rows);
    for (input_row_idx, input_row) in input_values.chunks_exact(payload.matrix_cols).enumerate() {
        for row in 0..payload.matrix_rows {
            let mut sum = bias_values.map_or(0.0, |bias| bias[row]);
            let row_start = row * payload.matrix_cols;
            let mut col = 0usize;
            while col < payload.matrix_cols {
                let weight_idx = row_start + col;
                let block_idx = weight_idx / payload.q8_block_size;
                let block_value_offset = weight_idx % payload.q8_block_size;
                let block_offset = block_idx * block_stride;
                let scale = dense_q8_sidecar_block_scale(&payload.packed_q8_bytes, block_offset);
                let values_in_block = payload.q8_block_size - block_value_offset;
                let values_in_row = payload.matrix_cols - col;
                let values_to_process = values_in_block.min(values_in_row);
                for offset in 0..values_to_process {
                    let q_idx = block_offset + 2 + block_value_offset + offset;
                    let q = payload.packed_q8_bytes[q_idx] as i8;
                    sum += scale * f32::from(q) * input_row[col + offset];
                }
                col += values_to_process;
            }
            output[input_row_idx * payload.matrix_rows + row] = sum;
        }
    }
}

fn dense_q8_sidecar_block_scale(packed_q8_bytes: &[u8], block_offset: usize) -> f32 {
    fp16_to_f32(u16::from_le_bytes([
        packed_q8_bytes[block_offset],
        packed_q8_bytes[block_offset + 1],
    ]))
}

fn attention_f16_dot_input(tensor: &Tensor) -> Result<Tensor> {
    Ok(tensor.to_dtype(DType::F16)?.to_dtype(DType::F32)?)
}

fn attention_score_key_input(tensor: &Tensor) -> Result<Tensor> {
    Ok(tensor.to_dtype(DType::F32)?)
}

/// Rotary Position Embedding
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dim: usize,
        max_seq_len: usize,
        rope_theta: Option<f32>,
        device: &Device,
    ) -> Result<Self> {
        Self::new_impl(dim, max_seq_len, rope_theta, device, None)
    }

    fn new_traced(
        dim: usize,
        max_seq_len: usize,
        rope_theta: Option<f32>,
        device: &Device,
        trace: RopeInitTrace<'_>,
    ) -> Result<Self> {
        Self::new_impl(dim, max_seq_len, rope_theta, device, Some(trace))
    }

    fn new_impl(
        dim: usize,
        max_seq_len: usize,
        rope_theta: Option<f32>,
        device: &Device,
        trace: Option<RopeInitTrace<'_>>,
    ) -> Result<Self> {
        let theta = resolve_rope_base(rope_theta);
        qwen_trace_rope_init_event(trace, "model_init.rope_start", || {
            format!("\"dim\":{},\"max_seq_len\":{},\"theta\":{}", dim, max_seq_len, theta)
        });
        let tables_start = Instant::now();
        qwen_trace_rope_init_event(trace, "model_init.rope_tables_start", || {
            format!("\"dim\":{},\"max_seq_len\":{}", dim, max_seq_len)
        });
        let tables = build_rope_tables(dim, max_seq_len, theta)
            .map_err(|err| BitNetError::Validation(format!("invalid RoPE configuration: {err}")))?;
        let bitnet_rope::RopeTables { half_dim, sin, cos } = tables;
        qwen_trace_rope_init_event(trace, "model_init.rope_tables_finish", || {
            format!(
                "\"tables_ms\":{},\"half_dim\":{},\"sin_len\":{},\"cos_len\":{}",
                qwen_trace_elapsed_ms(tables_start),
                half_dim,
                sin.len(),
                cos.len()
            )
        });
        let table_device = rope_table_device_for_target(device);
        qwen_trace_rope_init_event(trace, "model_init.rope_table_storage", || {
            format!(
                "\"target_device\":\"{}\",\"table_device\":\"{}\",\"reason\":\"{}\"",
                qwen_trace_device_kind(device),
                qwen_trace_device_kind(&table_device),
                if matches!(device, Device::Cuda(_)) {
                    "cpu_staged_to_avoid_constructor_full_table_cuda_upload"
                } else {
                    "target_device_storage"
                }
            )
        });

        let sin_start = Instant::now();
        qwen_trace_rope_init_event(trace, "model_init.rope_sin_tensor_start", || {
            format!(
                "\"rows\":{},\"cols\":{},\"table_device\":\"{}\"",
                max_seq_len,
                half_dim,
                qwen_trace_device_kind(&table_device)
            )
        });
        let sin = Tensor::from_vec(sin, &[max_seq_len, half_dim], &table_device)?;
        qwen_trace_rope_init_event(trace, "model_init.rope_sin_tensor_finish", || {
            format!(
                "\"tensor_ms\":{},\"dtype\":\"{:?}\",\"dims\":[{}],\"table_device\":\"{}\"",
                qwen_trace_elapsed_ms(sin_start),
                sin.dtype(),
                qwen_trace_dims_json(sin.dims()),
                qwen_trace_device_kind(sin.device())
            )
        });
        let cos_start = Instant::now();
        qwen_trace_rope_init_event(trace, "model_init.rope_cos_tensor_start", || {
            format!(
                "\"rows\":{},\"cols\":{},\"table_device\":\"{}\"",
                max_seq_len,
                half_dim,
                qwen_trace_device_kind(&table_device)
            )
        });
        let cos = Tensor::from_vec(cos, &[max_seq_len, half_dim], &table_device)?;
        qwen_trace_rope_init_event(trace, "model_init.rope_cos_tensor_finish", || {
            format!(
                "\"tensor_ms\":{},\"dtype\":\"{:?}\",\"dims\":[{}],\"table_device\":\"{}\"",
                qwen_trace_elapsed_ms(cos_start),
                cos.dtype(),
                qwen_trace_dims_json(cos.dims()),
                qwen_trace_device_kind(cos.device())
            )
        });

        // Log ROPE initialization parameters
        tracing::info!(
            "ROPE initialized: base={}, rope_dims={}, max_seq_len={}",
            theta,
            dim,
            max_seq_len
        );
        qwen_trace_rope_init_event(trace, "model_init.rope_finish", || {
            format!("\"dim\":{},\"max_seq_len\":{},\"half_dim\":{}", dim, max_seq_len, half_dim)
        });

        Ok(Self { sin, cos })
    }

    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        let trace_rope_apply = qwen_trace_events_enabled();
        // x shape: [B, H, T, D] for multi-head attention
        if x.dims().len() == 4 {
            let (batch, n_heads, seq_len, head_dim) = x.dims4()?;
            let half_dim = head_dim / 2;
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_4d_start", || {
                format!(
                    "\"position\":{},\"seq_len\":{},\"head_dim\":{},\"half_dim\":{},\"table_device\":\"{}\",\"target_device\":\"{}\"",
                    position,
                    seq_len,
                    head_dim,
                    half_dim,
                    qwen_trace_device_kind(self.cos.device()),
                    qwen_trace_device_kind(x.device())
                )
            });

            // LLaMA RoPE uses SPLIT layout: [r0,r1,...,r_{d/2-1}, i0,i1,...,i_{d/2-1}]
            // NOT interleaved [r0,i0,r1,i1,...]
            let x0 = x.narrow(3, 0, half_dim)?; // First half (real)
            let x1 = x.narrow(3, half_dim, half_dim)?; // Second half (imaginary)

            // Get cos/sin for the position
            let cos_slice_start = Instant::now();
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_4d_cos_slice_start", || {
                format!(
                    "\"position\":{},\"seq_len\":{},\"table_device\":\"{}\",\"target_device\":\"{}\"",
                    position,
                    seq_len,
                    qwen_trace_device_kind(self.cos.device()),
                    qwen_trace_device_kind(x.device())
                )
            });
            let cos = self.cos.narrow(0, position, seq_len)?.to_device(x.device())?;
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_4d_cos_slice_finish", || {
                format!(
                    "\"slice_ms\":{},\"dims\":[{}],\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(cos_slice_start),
                    qwen_trace_dims_json(cos.dims()),
                    qwen_trace_device_kind(cos.device())
                )
            });
            let cos = cos
                .unsqueeze(0)? // Add batch dim
                .unsqueeze(1)? // Add heads dim
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;

            let sin_slice_start = Instant::now();
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_4d_sin_slice_start", || {
                format!(
                    "\"position\":{},\"seq_len\":{},\"table_device\":\"{}\",\"target_device\":\"{}\"",
                    position,
                    seq_len,
                    qwen_trace_device_kind(self.sin.device()),
                    qwen_trace_device_kind(x.device())
                )
            });
            let sin = self.sin.narrow(0, position, seq_len)?.to_device(x.device())?;
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_4d_sin_slice_finish", || {
                format!(
                    "\"slice_ms\":{},\"dims\":[{}],\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(sin_slice_start),
                    qwen_trace_dims_json(sin.dims()),
                    qwen_trace_device_kind(sin.device())
                )
            });
            let sin = sin
                .unsqueeze(0)?
                .unsqueeze(1)?
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            // Concatenate back in split layout [real, imag]
            let rotated = Tensor::cat(&[x0_rot, x1_rot], 3)?;
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_4d_finish", || {
                format!(
                    "\"dims\":[{}],\"device\":\"{}\"",
                    qwen_trace_dims_json(rotated.dims()),
                    qwen_trace_device_kind(rotated.device())
                )
            });

            Ok(rotated)
        } else {
            // Original 3D implementation for other uses
            let (_batch, _seq, dim) = x.dims3()?;
            let half_dim = dim / 2;
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_3d_start", || {
                format!(
                    "\"position\":{},\"dim\":{},\"half_dim\":{},\"table_device\":\"{}\",\"target_device\":\"{}\"",
                    position,
                    dim,
                    half_dim,
                    qwen_trace_device_kind(self.cos.device()),
                    qwen_trace_device_kind(x.device())
                )
            });

            // LLaMA RoPE uses SPLIT layout: [r0,r1,...,i0,i1,...]
            let x0 = x.narrow(2, 0, half_dim)?; // First half (real)
            let x1 = x.narrow(2, half_dim, half_dim)?; // Second half (imaginary)

            let cos_slice_start = Instant::now();
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_3d_cos_slice_start", || {
                format!(
                    "\"position\":{},\"table_device\":\"{}\",\"target_device\":\"{}\"",
                    position,
                    qwen_trace_device_kind(self.cos.device()),
                    qwen_trace_device_kind(x.device())
                )
            });
            let cos = self.cos.narrow(0, position, 1)?.to_device(x.device())?;
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_3d_cos_slice_finish", || {
                format!(
                    "\"slice_ms\":{},\"dims\":[{}],\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(cos_slice_start),
                    qwen_trace_dims_json(cos.dims()),
                    qwen_trace_device_kind(cos.device())
                )
            });
            let sin_slice_start = Instant::now();
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_3d_sin_slice_start", || {
                format!(
                    "\"position\":{},\"table_device\":\"{}\",\"target_device\":\"{}\"",
                    position,
                    qwen_trace_device_kind(self.sin.device()),
                    qwen_trace_device_kind(x.device())
                )
            });
            let sin = self.sin.narrow(0, position, 1)?.to_device(x.device())?;
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_3d_sin_slice_finish", || {
                format!(
                    "\"slice_ms\":{},\"dims\":[{}],\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(sin_slice_start),
                    qwen_trace_dims_json(sin.dims()),
                    qwen_trace_device_kind(sin.device())
                )
            });

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            // Concatenate back in split layout [real, imag]
            let rotated = Tensor::cat(&[x0_rot, x1_rot], 2)?;
            qwen_trace_runtime_event(trace_rope_apply, "rope.apply_3d_finish", || {
                format!(
                    "\"dims\":[{}],\"device\":\"{}\"",
                    qwen_trace_dims_json(rotated.dims()),
                    qwen_trace_device_kind(rotated.device())
                )
            });

            Ok(rotated)
        }
    }
}

/// Multi-Head Attention Layer
pub struct MultiHeadAttention {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    group_size: usize, // n_heads / n_kv_heads
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
    sub_layernorm: Option<LayerNorm>,
    rope: Option<RotaryEmbedding>,
    layer_idx: usize, // Layer index for QK256 weight name generation
}

impl MultiHeadAttention {
    pub fn new(config: &BitNetConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let init_start = Instant::now();
        let trace_model_init = qwen_trace_events_enabled();
        let device = vb.device().clone();
        let hidden_size = config.model.hidden_size;
        let n_heads = config.model.num_heads;
        let head_dim = config.model.attention_head_dim.unwrap_or_else(|| hidden_size / n_heads);

        if config.model.attention_head_dim.is_none() && !hidden_size.is_multiple_of(n_heads) {
            return Err(BitNetError::Validation(format!(
                "hidden_size {} not divisible by num_heads {}",
                hidden_size, n_heads
            )));
        }

        let n_kv_heads = config.model.num_key_value_heads.max(1).min(n_heads);
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(BitNetError::Validation(format!(
                "num_heads {} must be divisible by num_key_value_heads {}",
                n_heads, n_kv_heads
            )));
        }
        let group_size = n_heads / n_kv_heads;
        let q_out = n_heads * head_dim;
        let kv_out = n_kv_heads * head_dim;

        qwen_trace_model_init_event(trace_model_init, "model_init.attention_start", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\",\"hidden_size\":{},\"n_heads\":{},\"n_kv_heads\":{},\"head_dim\":{},\"q_out\":{},\"kv_out\":{}",
                qwen_trace_elapsed_ms(init_start),
                layer_idx,
                qwen_trace_device_kind(&device),
                hidden_size,
                n_heads,
                n_kv_heads,
                head_dim,
                q_out,
                kv_out
            )
        });

        tracing::info!(
            "layer{}: MultiHeadAttention dims: hidden={}, n_heads={}, n_kv_heads={}, head_dim={}, q_out={}, kv_out={}, group_size={}",
            layer_idx,
            hidden_size,
            n_heads,
            n_kv_heads,
            head_dim,
            q_out,
            kv_out,
            group_size
        );

        tracing::info!(
            "layer{}: About to create linear layers with: q_proj([{}, {}]), k_proj([{}, {}]), v_proj([{}, {}]), o_proj([{}, {}])",
            layer_idx,
            q_out,
            hidden_size,
            kv_out,
            hidden_size,
            kv_out,
            hidden_size,
            hidden_size,
            q_out
        );

        let q_proj = linear_with_optional_bias_traced(
            hidden_size,
            q_out,
            vb.pp("q_proj"),
            LinearInitTrace {
                enabled: trace_model_init,
                init_start,
                layer_idx: Some(layer_idx),
                device: &device,
                scope: "attention",
                name: "q_proj",
            },
        )?;
        let k_proj = linear_with_optional_bias_traced(
            hidden_size,
            kv_out,
            vb.pp("k_proj"),
            LinearInitTrace {
                enabled: trace_model_init,
                init_start,
                layer_idx: Some(layer_idx),
                device: &device,
                scope: "attention",
                name: "k_proj",
            },
        )?;
        let v_proj = linear_with_optional_bias_traced(
            hidden_size,
            kv_out,
            vb.pp("v_proj"),
            LinearInitTrace {
                enabled: trace_model_init,
                init_start,
                layer_idx: Some(layer_idx),
                device: &device,
                scope: "attention",
                name: "v_proj",
            },
        )?;
        let o_proj = linear_with_optional_bias_traced(
            q_out,
            hidden_size,
            vb.pp("o_proj"),
            LinearInitTrace {
                enabled: trace_model_init,
                init_start,
                layer_idx: Some(layer_idx),
                device: &device,
                scope: "attention",
                name: "o_proj",
            },
        )?;
        qwen_trace_model_init_event(
            trace_model_init,
            "model_init.attention_linears_finish",
            || {
                format!(
                    "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(init_start),
                    layer_idx,
                    qwen_trace_device_kind(&device)
                )
            },
        );
        let q_norm = optional_layer_norm_with_optional_bias(
            config.model.norm_type,
            head_dim,
            eps_from_config(config),
            vb.pp("q_norm"),
        )?;
        let k_norm = optional_layer_norm_with_optional_bias(
            config.model.norm_type,
            head_dim,
            eps_from_config(config),
            vb.pp("k_norm"),
        )?;
        let sub_layernorm = optional_layer_norm_with_optional_bias(
            config.model.norm_type,
            q_out,
            eps_from_config(config),
            vb.pp("sub_layernorm"),
        )?;
        qwen_trace_model_init_event(trace_model_init, "model_init.attention_norms_finish", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\",\"q_norm_present\":{},\"k_norm_present\":{},\"sub_layernorm_present\":{}",
                qwen_trace_elapsed_ms(init_start),
                layer_idx,
                qwen_trace_device_kind(&device),
                q_norm.is_some(),
                k_norm.is_some(),
                sub_layernorm.is_some()
            )
        });

        qwen_trace_model_init_event(trace_model_init, "model_init.attention_rope_start", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\",\"head_dim\":{},\"max_seq_len\":{}",
                qwen_trace_elapsed_ms(init_start),
                layer_idx,
                qwen_trace_device_kind(&device),
                head_dim,
                config.model.max_position_embeddings
            )
        });
        let rope = RotaryEmbedding::new_traced(
            head_dim,
            config.model.max_position_embeddings,
            config.model.rope_theta,
            vb.device(),
            RopeInitTrace { enabled: trace_model_init, init_start, layer_idx, device: &device },
        )
        .ok();
        qwen_trace_model_init_event(trace_model_init, "model_init.attention_rope_finish", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\",\"rope_present\":{}",
                qwen_trace_elapsed_ms(init_start),
                layer_idx,
                qwen_trace_device_kind(&device),
                rope.is_some()
            )
        });
        qwen_trace_model_init_event(trace_model_init, "model_init.attention_finish", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\",\"rope_present\":{}",
                qwen_trace_elapsed_ms(init_start),
                layer_idx,
                qwen_trace_device_kind(&device),
                rope.is_some()
            )
        });

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            group_size,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            sub_layernorm,
            rope,
            layer_idx,
        })
    }

    // `forward` lives in the `attention_forward` module so the projection,
    // RoPE/cache, GQA, score, softmax, and output-projection responsibilities
    // stay isolated from QK256 linear dispatch helpers below.

    /// Apply linear transformation with QK256 dispatch
    /// Apply linear transformation with QK256 dispatch
    /// Apply linear transformation with QK256 dispatch
    fn apply_linear(
        &self,
        input: &Tensor,
        linear: &Linear,
        proj_name: &str,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<Tensor> {
        // Generate weight name based on layer index and projection name
        // Format: "layers.{idx}.attention.{proj_name}.weight.qk256_qs"
        let qk256_key =
            format!("layers.{}.attention.{}.weight.qk256_qs", self.layer_idx, proj_name);

        // Check for QK256 data
        if let Some(qk256_tensor) = raw_tensors.get(&qk256_key) {
            tracing::debug!("Using QK256 kernel for {}", qk256_key);
            let inline_scale = qk256_inline_scale(raw_tensors, &qk256_key)?;
            return forward_qk256_with_scale(input, qk256_tensor, &qk256_key, inline_scale);
        }

        if strict_cuda_bitnet_backend_requested() {
            record_bitnet_linear_unsupported();
            return Err(BitNetError::Validation(format!(
                "strict CUDA BitNet linear dispatch requires QK256 raw tensor {}; refusing CPU fallback",
                qk256_key
            )));
        }

        // Probe: Why is QK256 not found? (layer 0 only, once)
        if trace_rms_enabled() && self.layer_idx == 0 {
            static FALLBACK_LOGGED: std::sync::Once = std::sync::Once::new();
            FALLBACK_LOGGED.call_once(|| {
                eprintln!(
                    "trace_fallback: QK256 key '{}' not found in raw_tensors ({}keys total)",
                    qk256_key,
                    raw_tensors.len()
                );
                // Show first few keys for debugging
                let sample_keys: Vec<_> = raw_tensors.keys().take(5).collect();
                eprintln!("trace_fallback: Sample keys: {:?}", sample_keys);
            });
        }

        // Fall back to standard linear
        tracing::trace!(
            "Using standard linear for layers.{}.attention.{}",
            self.layer_idx,
            proj_name
        );
        let dense_tensor_name = format!("layers.{}.attention.{}.weight", self.layer_idx, proj_name);
        let hook_boundary =
            dense_linear_runtime_hook_boundary(&dense_tensor_name, dense_linear_hooks);
        tracing::trace!(
            tensor_name = %hook_boundary.tensor_name,
            selected_path = hook_boundary.selected_path,
            selected_kernel = hook_boundary.selected_kernel,
            sidecar_descriptor_present = hook_boundary.sidecar_descriptor_present,
            runtime_compute_enabled = hook_boundary.runtime_compute_enabled,
            "dense linear production hook boundary"
        );
        if let Some(output) = maybe_forward_dense_q8_sidecar_linear(
            input,
            linear,
            &dense_tensor_name,
            dense_linear_hooks,
        )? {
            return Ok(output);
        }
        record_bitnet_linear_cpu_fallback();
        linear.forward(input).map_err(BitNetError::from)
    }

    /// PATCH 5: Create causal mask with [1, 1, Tq, Tk] shape
    fn create_causal_mask(&self, q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
        // Past tokens are stored in the KV cache and increase k_len.
        // For each query position i, disallow attention to key positions
        // greater than past_len + i.
        let past_len = k_len.saturating_sub(q_len);
        let mut mask_vec = vec![0.0f32; q_len * k_len];
        for i in 0..q_len {
            let start = past_len + i + 1;
            for j in start..k_len {
                mask_vec[i * k_len + j] = f32::NEG_INFINITY;
            }
        }
        // Create [1, 1, q_len, k_len] shape directly for broadcast compatibility
        Tensor::from_vec(mask_vec, &[1, 1, q_len, k_len], device).map_err(BitNetError::from)
    }
}

/// Feed-Forward Network
pub struct FeedForward {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    sub_layernorm: Option<LayerNorm>,
    activation_type: ActivationType,
    layer_idx: usize, // Layer index for QK256 weight name generation
}

impl FeedForward {
    pub fn new(config: &BitNetConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let init_start = Instant::now();
        let trace_model_init = qwen_trace_events_enabled();
        let device = vb.device().clone();
        let hidden_size = config.model.hidden_size;
        let intermediate_size = config.model.intermediate_size;

        Ok(Self {
            gate_proj: linear_with_optional_bias_traced(
                hidden_size,
                intermediate_size,
                vb.pp("gate_proj"),
                LinearInitTrace {
                    enabled: trace_model_init,
                    init_start,
                    layer_idx: Some(layer_idx),
                    device: &device,
                    scope: "feed_forward",
                    name: "gate_proj",
                },
            )?,
            up_proj: linear_with_optional_bias_traced(
                hidden_size,
                intermediate_size,
                vb.pp("up_proj"),
                LinearInitTrace {
                    enabled: trace_model_init,
                    init_start,
                    layer_idx: Some(layer_idx),
                    device: &device,
                    scope: "feed_forward",
                    name: "up_proj",
                },
            )?,
            down_proj: linear_with_optional_bias_traced(
                intermediate_size,
                hidden_size,
                vb.pp("down_proj"),
                LinearInitTrace {
                    enabled: trace_model_init,
                    init_start,
                    layer_idx: Some(layer_idx),
                    device: &device,
                    scope: "feed_forward",
                    name: "down_proj",
                },
            )?,
            sub_layernorm: optional_layer_norm_with_optional_bias(
                config.model.norm_type,
                intermediate_size,
                eps_from_config(config),
                vb.pp("sub_layernorm"),
            )?,
            activation_type: config.model.activation_type,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<Tensor> {
        self.forward_impl(x, raw_tensors, dense_linear_hooks, None, None)
    }

    pub fn forward_with_no_bias_callsite_descriptor(
        &self,
        x: &Tensor,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        prompt_bound_no_bias_descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    ) -> Result<Tensor> {
        self.forward_impl(
            x,
            raw_tensors,
            dense_linear_hooks,
            Some(prompt_bound_no_bias_descriptor),
            None,
        )
    }

    fn forward_impl(
        &self,
        x: &Tensor,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        prompt_bound_no_bias_descriptor: Option<
            &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
        >,
        workspace: Option<&mut TransformerForwardWorkspace>,
    ) -> Result<Tensor> {
        let gate = self.apply_linear(
            x,
            &self.gate_proj,
            "gate_proj",
            raw_tensors,
            dense_linear_hooks,
            None,
        )?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.gate_proj", Some(self.layer_idx), &gate)?;
        }

        // MLP gating diagnostics (point 3 of user's plan)
        if debug_mlp_enabled()
            && let Ok(u_norm) = gate.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||u|| (gate_proj): {:.6e}", u_norm);
        }

        let gate = self.apply_activation(&gate)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.gate_activation", Some(self.layer_idx), &gate)?;
        }

        if debug_mlp_enabled()
            && let Ok(activation_norm) = gate.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||activation(u)||: {:.6e}", activation_norm);
        }

        let up =
            self.apply_linear(x, &self.up_proj, "up_proj", raw_tensors, dense_linear_hooks, None)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.up_proj", Some(self.layer_idx), &up)?;
        }

        if debug_mlp_enabled()
            && let Ok(v_norm) = up.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||v|| (up_proj): {:.6e}", v_norm);
        }

        let hidden = gate.mul(&up)?;
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.gated_product", Some(self.layer_idx), &hidden)?;
        }
        let hidden = if let Some(sub_layernorm) = &self.sub_layernorm {
            let normalized = sub_layernorm.forward(&hidden)?;
            if qwen_trace_layer_enabled(self.layer_idx) {
                qwen_trace_tensor("mlp.sub_layernorm", Some(self.layer_idx), &normalized)?;
            }
            normalized
        } else {
            hidden
        };

        if debug_mlp_enabled()
            && let Ok(prod_norm) = hidden.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||silu(u) * v||: {:.6e}", prod_norm);
        }

        let output = self.apply_linear(
            &hidden,
            &self.down_proj,
            "down_proj",
            raw_tensors,
            dense_linear_hooks,
            prompt_bound_no_bias_descriptor,
        )?;
        if let Some(workspace) = workspace {
            let boundary = self.down_proj_output_storage_boundary(&output);
            workspace.record_down_proj_output_storage_boundary(&output, boundary);
        }
        if qwen_trace_layer_enabled(self.layer_idx) {
            qwen_trace_tensor("mlp.down_proj", Some(self.layer_idx), &output)?;
        }

        if debug_mlp_enabled()
            && let Ok(out_norm) = output.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            tracing::debug!("MLP ||W2 * (...)||: {:.6e}", out_norm);
        }

        Ok(output)
    }

    pub fn forward_with_workspace(
        &self,
        x: &Tensor,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        workspace: &mut TransformerForwardWorkspace,
    ) -> Result<Tensor> {
        workspace.record_feed_forward_input(x);
        let output =
            self.forward_impl(x, raw_tensors, dense_linear_hooks, None, Some(workspace))?;
        workspace.record_feed_forward_output(&output);
        workspace.store_feed_forward_output(output);
        workspace.take_feed_forward_output()
    }

    pub fn forward_with_workspace_and_no_bias_callsite_descriptor(
        &self,
        x: &Tensor,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        workspace: &mut TransformerForwardWorkspace,
        prompt_bound_no_bias_descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    ) -> Result<Tensor> {
        workspace.record_feed_forward_input(x);
        let output = self.forward_impl(
            x,
            raw_tensors,
            dense_linear_hooks,
            Some(prompt_bound_no_bias_descriptor),
            Some(workspace),
        )?;
        workspace.record_feed_forward_output(&output);
        workspace.store_feed_forward_output(output);
        workspace.take_feed_forward_output()
    }

    fn apply_activation(&self, input: &Tensor) -> Result<Tensor> {
        apply_ffn_activation(input, self.activation_type)
    }

    /// Apply linear transformation with QK256 dispatch
    fn apply_linear(
        &self,
        input: &Tensor,
        linear: &Linear,
        proj_name: &str,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        prompt_bound_no_bias_descriptor: Option<
            &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
        >,
    ) -> Result<Tensor> {
        // Generate weight name based on layer index and projection name
        // Format: "layers.{idx}.feed_forward.{proj_name}.weight.qk256_qs"
        let qk256_key =
            format!("layers.{}.feed_forward.{}.weight.qk256_qs", self.layer_idx, proj_name);

        // Check for QK256 data
        if let Some(qk256_tensor) = raw_tensors.get(&qk256_key) {
            tracing::debug!("Using QK256 kernel for {}", qk256_key);
            let inline_scale = qk256_inline_scale(raw_tensors, &qk256_key)?;
            return forward_qk256_with_scale(input, qk256_tensor, &qk256_key, inline_scale);
        }

        if strict_cuda_bitnet_backend_requested() {
            record_bitnet_linear_unsupported();
            return Err(BitNetError::Validation(format!(
                "strict CUDA BitNet linear dispatch requires QK256 raw tensor {}; refusing CPU fallback",
                qk256_key
            )));
        }

        // Fall back to standard linear
        tracing::trace!(
            "Using standard linear for layers.{}.feed_forward.{}",
            self.layer_idx,
            proj_name
        );
        let dense_tensor_name = feed_forward_dense_tensor_name(self.layer_idx, proj_name);
        if let Some(output) = maybe_forward_feed_forward_no_bias_candidate_linear(
            input,
            linear,
            proj_name,
            &dense_tensor_name,
            prompt_bound_no_bias_descriptor,
        )? {
            return Ok(output);
        }
        let hook_boundary =
            dense_linear_runtime_hook_boundary(&dense_tensor_name, dense_linear_hooks);
        tracing::trace!(
            tensor_name = %hook_boundary.tensor_name,
            selected_path = hook_boundary.selected_path,
            selected_kernel = hook_boundary.selected_kernel,
            sidecar_descriptor_present = hook_boundary.sidecar_descriptor_present,
            runtime_compute_enabled = hook_boundary.runtime_compute_enabled,
            "dense linear production hook boundary"
        );
        if let Some(output) = maybe_forward_dense_q8_sidecar_linear(
            input,
            linear,
            &dense_tensor_name,
            dense_linear_hooks,
        )? {
            return Ok(output);
        }
        record_bitnet_linear_cpu_fallback();
        linear.forward(input).map_err(BitNetError::from)
    }

    fn down_proj_output_storage_boundary(
        &self,
        output: &Tensor,
    ) -> TransformerWorkspaceOutputSurface {
        let boundary = DenseLinearOutputStorageApiBoundary::from_candle_linear(
            "feed_forward.down_proj.output",
            &self.down_proj,
        );
        TransformerWorkspaceOutputSurface {
            name: "feed_forward.down_proj.output",
            storage_owner: "TransformerForwardWorkspace",
            status: boundary.status,
            reason: boundary.reason,
            next_api_hook: boundary.next_api_hook,
            last_shape: output.dims().to_vec(),
            linear_weight_shape: boundary.weight_shape,
            linear_bias_shape: boundary.bias_shape,
            weight_accessible: boundary.weight_accessible,
            bias_accessible: boundary.bias_accessible,
            can_fill_caller_output_storage: boundary.can_fill_caller_output_storage,
        }
    }
}

fn apply_ffn_activation(input: &Tensor, activation_type: ActivationType) -> Result<Tensor> {
    match activation_type {
        ActivationType::Silu => input.silu().map_err(BitNetError::from),
        ActivationType::Relu2 => input.relu()?.sqr().map_err(BitNetError::from),
        ActivationType::Gelu => input.gelu_erf().map_err(BitNetError::from),
    }
}

/// Transformer Block
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    attention_norm: LayerNorm,
    ffn_norm: LayerNorm,
}

/// Typed transformer forward workspace boundary.
///
/// This is intentionally conservative: it records the API surface that future
/// slices can use for reusable transformer-owned buffers, but it does not reuse
/// Candle tensor outputs yet. Keeping tensor math delegated to the existing
/// `forward` path preserves the current Qwen3 Q8_0 behavior oracle while making
/// the owned-output boundary explicit in code.
#[derive(Debug, Clone, Default)]
pub struct TransformerForwardWorkspace {
    model_forward_calls: usize,
    block_forward_calls: usize,
    feed_forward_calls: usize,
    last_input_shape: Vec<usize>,
    last_output_shape: Vec<usize>,
    feed_forward_output_slot: Option<Tensor>,
    model_output_slot: Option<Tensor>,
    model_output_surface: Option<TransformerWorkspaceOutputSurface>,
    model_forward_source_tensors: Option<TransformerModelForwardSourceTensors>,
    final_block_source_tensors: Option<TransformerFinalBlockSourceTensors>,
    penultimate_block_source_tensors: Option<TransformerFinalBlockSourceTensors>,
    antepenultimate_block_source_tensors: Option<TransformerFinalBlockSourceTensors>,
    pre_antepenultimate_block_source_tensors: Option<TransformerFinalBlockSourceTensors>,
    earlier_block_source_tensors: Option<TransformerFinalBlockSourceTensors>,
    block_source_tensors: Vec<TransformerFinalBlockSourceTensors>,
    attention_output_source_tensors: Vec<TransformerAttentionOutputSourceTensors>,
    qkv_projection_source_tensors: Vec<TransformerQkvProjectionSourceTensors>,
    feed_forward_output_surface: Option<TransformerWorkspaceOutputSurface>,
    final_norm_output_surface: Option<TransformerWorkspaceOutputStorageBoundary>,
    layer_output_surface: Option<TransformerWorkspaceOutputStorageBoundary>,
    workspace_owned_output_count: usize,
    model_workspace_owned_output_count: usize,
    model_output_storage_attempts: usize,
    down_proj_output_storage_attempts: usize,
    final_norm_output_storage_attempts: usize,
    layer_output_storage_attempts: usize,
    tensor_reuse_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct TransformerModelForwardSourceTensors {
    pub prior_layer_output: Tensor,
    pub final_norm_output: Tensor,
}

#[derive(Debug, Clone)]
pub struct TransformerFinalBlockSourceTensors {
    pub layer_idx: usize,
    pub block_input: Tensor,
    pub attention_output: Tensor,
    pub post_attention_residual: Tensor,
    pub feed_forward_output: Tensor,
    pub block_output: Tensor,
}

#[derive(Debug, Clone)]
pub struct TransformerAttentionOutputSourceTensors {
    pub layer_idx: usize,
    pub attention_input: Tensor,
    pub q_projection: Tensor,
    pub k_projection: Tensor,
    pub v_projection: Tensor,
    pub q_heads: Tensor,
    pub k_heads: Tensor,
    pub v_heads: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub q_rope: Tensor,
    pub k_rope: Tensor,
    pub k_context: Tensor,
    pub v_context: Tensor,
    pub expanded_k: Tensor,
    pub expanded_v: Tensor,
    pub scores: Tensor,
    pub probabilities: Tensor,
    pub value_mix_output_heads: Tensor,
    pub output_projection_input: Tensor,
    pub sub_layernorm_output: Option<Tensor>,
    pub attention_output: Tensor,
}

#[derive(Debug, Clone)]
pub struct TransformerQk256DispatchDelta {
    pub bitnet_linear_layers_total: u64,
    pub bitnet_linear_layers_on_cuda: u64,
    pub bitnet_linear_layers_on_a770_opencl: u64,
    pub bitnet_linear_layers_cpu_fallback: u64,
    pub unsupported_ops: Vec<String>,
    pub execution_claim: String,
}

#[derive(Debug, Clone)]
pub struct TransformerQk256CpuHotPathDelta {
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
pub struct TransformerA770OpenClRuntimeDelta {
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub kernel_invocations: u64,
}

#[derive(Debug, Clone)]
pub struct TransformerQkvProjectionDispatchReplayTensors {
    pub input_rows: usize,
    pub output_rows: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,
    pub inline_scale: Option<f32>,
    pub cpu_output: Tensor,
    pub opencl_policy_output: Tensor,
    pub a770_output: Option<Tensor>,
    pub device_expression_trace: Option<TransformerQk256DeviceExpressionTrace>,
    pub device_intermediate_trace: Option<TransformerQk256DeviceIntermediateTrace>,
    pub focused_operands: Option<TransformerQk256FocusedRawOperands>,
    pub full_projection_operands: Option<TransformerQk256FullProjectionRawOperands>,
    pub cpu: TransformerQkvProjectionDispatchReplayCpuStats,
    pub a770: TransformerQkvProjectionDispatchReplayA770Stats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerQk256FocusedRawOperands {
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
pub struct TransformerQk256FullProjectionRawOperands {
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
pub struct TransformerQk256DeviceExpressionTrace {
    pub input_row_index: usize,
    pub sample_limit: usize,
    pub sample_count: usize,
    pub samples: Vec<TransformerQk256DeviceExpressionSample>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerQk256DeviceExpressionSample {
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
pub struct TransformerQk256DeviceIntermediateTrace {
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
    pub samples: Vec<TransformerQk256DeviceIntermediateSample>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerQk256DeviceIntermediateSample {
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
pub struct TransformerQkvProjectionDispatchReplayCpuStats {
    pub scalar_invocations: u64,
    pub execution_path: String,
}

#[derive(Debug, Clone)]
pub struct TransformerQkvProjectionDispatchReplayA770Stats {
    pub compiled_opencl: bool,
    pub attempted: bool,
    pub success: bool,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub kernel_invocations: u64,
    pub last_device: Option<TransformerA770OpenClRuntimeDevice>,
    pub error: Option<String>,
    pub execution_path: String,
}

#[derive(Debug, Clone)]
pub struct TransformerA770OpenClRuntimeDevice {
    pub platform_index: usize,
    pub device_index: usize,
    pub platform_name: String,
    pub runtime_device: String,
    pub vendor: String,
    pub driver_version: String,
}

#[derive(Debug, Clone)]
pub struct TransformerQkvProjectionSourceTensors {
    pub layer_idx: usize,
    pub projection: String,
    pub tensor_name: String,
    pub qk256_key: String,
    pub qk256_raw_tensor_present: bool,
    pub input: Tensor,
    pub output: Tensor,
    pub dispatch_delta: TransformerQk256DispatchDelta,
    pub cpu_hot_path_delta: TransformerQk256CpuHotPathDelta,
    pub a770_opencl_runtime_delta: TransformerA770OpenClRuntimeDelta,
    pub dispatch_replay: Option<TransformerQkvProjectionDispatchReplayTensors>,
    pub dispatch_replay_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerWorkspaceOutputSurface {
    pub name: &'static str,
    pub storage_owner: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub next_api_hook: &'static str,
    pub last_shape: Vec<usize>,
    pub linear_weight_shape: Vec<usize>,
    pub linear_bias_shape: Option<Vec<usize>>,
    pub weight_accessible: bool,
    pub bias_accessible: bool,
    pub can_fill_caller_output_storage: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerWorkspaceOutputStorageBoundary {
    pub name: &'static str,
    pub storage_owner: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub next_api_hook: &'static str,
    pub last_shape: Vec<usize>,
    pub operation_family: &'static str,
    pub operation_detail: &'static str,
    pub residual_input_shape: Option<Vec<usize>>,
    pub branch_output_shape: Option<Vec<usize>>,
    pub weight_shape: Option<Vec<usize>>,
    pub bias_shape: Option<Vec<usize>>,
    pub epsilon: Option<String>,
    pub input_accessible: bool,
    pub weight_accessible: bool,
    pub bias_accessible: bool,
    pub residual_add_involved: bool,
    pub caller_output_helper_status: &'static str,
    pub can_fill_caller_output_storage: bool,
    pub exact_blocking_ops: Option<&'static [&'static str]>,
    pub public_api_return_type: Option<&'static str>,
    pub required_missing_api: Option<&'static str>,
    pub public_api_accepts_output_storage: Option<bool>,
    pub backend_internal_in_place_api_exposed: Option<bool>,
}

/// Behavior-preserving dense linear output-storage API boundary.
///
/// Candle exposes the read-side pieces (`Linear::weight` and `Linear::bias`),
/// but the compute-side `Tensor::matmul` and optional bias add still allocate
/// and return owned tensors. This boundary records that narrower fact so Kaby
/// SLM allocation work does not mistake read access for a reusable output slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLinearOutputStorageApiBoundary {
    pub role: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub next_api_hook: &'static str,
    pub weight_shape: Vec<usize>,
    pub bias_shape: Option<Vec<usize>>,
    pub weight_accessible: bool,
    pub bias_accessible: bool,
    pub can_fill_caller_output_storage: bool,
}

impl DenseLinearOutputStorageApiBoundary {
    pub fn from_candle_linear(role: &'static str, linear: &Linear) -> Self {
        let weight_shape = linear.weight().dims().to_vec();
        let bias_shape = linear.bias().map(|bias| bias.dims().to_vec());

        Self {
            role,
            status: "dense_linear_output_storage_blocked_by_candle_tensor_ops",
            reason: "candle_nn::Linear exposes weight and optional bias tensors, but its behavior-preserving compute path is Tensor::matmul plus optional broadcast_add, and those operations return owned Tensors without a caller-provided output-storage parameter",
            next_api_hook: "add or adopt a Candle Tensor matmul/bias-add output-storage API before replacing FeedForward::down_proj output construction with reusable workspace-backed storage",
            weight_shape,
            bias_shape,
            weight_accessible: true,
            bias_accessible: true,
            can_fill_caller_output_storage: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormOutputStorageApiBoundary {
    pub role: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub next_api_hook: &'static str,
    pub norm_kind: &'static str,
    pub weight_shape: Vec<usize>,
    pub bias_shape: Option<Vec<usize>>,
    pub epsilon: String,
    pub remove_mean: bool,
    pub input_accessible: bool,
    pub weight_accessible: bool,
    pub bias_accessible: bool,
    pub caller_output_helper_status: &'static str,
    pub can_fill_caller_output_storage: bool,
}

impl NormOutputStorageApiBoundary {
    pub fn from_candle_layer_norm(role: &'static str, norm: &LayerNorm) -> Self {
        let weight_shape = norm.weight().dims().to_vec();
        let bias_shape = norm.bias().map(|bias| bias.dims().to_vec());

        Self {
            role,
            status: "final_norm_output_storage_blocked_by_candle_layer_norm_ops",
            reason: "candle_nn::LayerNorm exposes input, weight, optional bias, epsilon, and remove-mean metadata, but both LayerNorm::forward and the public candle_nn::ops norm helpers return owned Tensors without a caller-provided output-storage parameter",
            next_api_hook: "add or adopt a Candle LayerNorm/RMSNorm output-storage API or apply_op output-storage hook before replacing model.final_norm output construction with reusable workspace-backed storage",
            norm_kind: if norm.remove_mean() { "layer_norm" } else { "rms_norm" },
            weight_shape,
            bias_shape,
            epsilon: format!("{:.8e}", norm.eps()),
            remove_mean: norm.remove_mean(),
            input_accessible: true,
            weight_accessible: true,
            bias_accessible: norm.bias().is_some(),
            caller_output_helper_status: "final_norm_output_storage_helper_blocked_by_owned_candle_norm_output",
            can_fill_caller_output_storage: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerOutputStorageApiBoundary {
    pub role: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub next_api_hook: &'static str,
    pub runtime_slice_status: &'static str,
    pub runtime_slice_blocker: &'static str,
    pub candle_api_evidence: &'static [&'static str],
    pub required_shape_contract: &'static str,
    pub ownership_contract: &'static str,
    pub behavior_preservation_gate: &'static str,
    pub residual_add_involved: bool,
    pub can_fill_caller_output_storage: bool,
    pub exact_blocking_ops: &'static [&'static str],
    pub public_api_return_type: &'static str,
    pub required_missing_api: &'static str,
    pub public_api_accepts_output_storage: bool,
    pub backend_internal_in_place_api_exposed: bool,
}

pub const CANDLE_RESIDUAL_ADD_EXACT_BLOCKING_OPS: &[&str] = &[
    "Tensor::add(&self, &Tensor) -> Result<Tensor>",
    "Tensor::broadcast_add(&self, &Tensor) -> Result<Tensor>",
    "std::ops::Add for Tensor/&Tensor delegates to Tensor::add and returns Result<Tensor>",
];

pub const CANDLE_RESIDUAL_ADD_PUBLIC_API_RETURN_TYPE: &str = "Result<Tensor>";

pub const CANDLE_RESIDUAL_ADD_REQUIRED_MISSING_API: &str = "Tensor residual-add API accepting caller-provided output storage, e.g. add_out/broadcast_add_out(&self, rhs, &mut output)";

pub const CANDLE_RESIDUAL_ADD_RUNTIME_SLICE_STATUS: &str =
    "runtime_slice_blocked_by_missing_caller_output_storage_api";

pub const CANDLE_RESIDUAL_ADD_RUNTIME_SLICE_BLOCKER: &str = "Kaby SLM residual block output allocation behavior cannot safely change while Candle Tensor::add/broadcast_add only return owned Result<Tensor> outputs; implementing storage reuse here requires an output-storage API or a verified backend-local equivalent before paired Qwen3/Qwen2.5 receipts can prove unchanged generated IDs";

pub const CANDLE_RESIDUAL_ADD_API_EVIDENCE: &[&str] = &[
    "Cargo.lock resolves candle-core 0.10.2 for the current workspace",
    "candle-core-0.10.2/src/tensor.rs binary_op!(add, Add) returns Result<Tensor>",
    "candle-core-0.10.2/src/tensor.rs broadcast_binary_op!(broadcast_add, add) delegates to owned Tensor::add output",
    "candle-core-0.10.2/src/tensor.rs notes TODO: make an inplace version or a pre-allocated version",
];

impl LayerOutputStorageApiBoundary {
    pub fn from_candle_residual_add(role: &'static str) -> Self {
        Self {
            role,
            status: "layer_output_storage_blocked_by_candle_tensor_add_ops",
            reason: "TransformerBlock layer output is produced by Candle Tensor::add/broadcast_add residual-add operations whose public API returns owned Result<Tensor> values and exposes no caller-provided output-storage parameter",
            next_api_hook: CANDLE_RESIDUAL_ADD_REQUIRED_MISSING_API,
            runtime_slice_status: CANDLE_RESIDUAL_ADD_RUNTIME_SLICE_STATUS,
            runtime_slice_blocker: CANDLE_RESIDUAL_ADD_RUNTIME_SLICE_BLOCKER,
            candle_api_evidence: CANDLE_RESIDUAL_ADD_API_EVIDENCE,
            required_shape_contract: "caller-provided block output storage must have the same shape, dtype, and device as both residual input and branch output, and the receipt must record all three shapes before any storage reuse is enabled",
            ownership_contract: "TransformerForwardWorkspace may own reusable block output storage only after residual add can write into caller-provided output without aliasing residual input or branch output in a way that changes Candle semantics",
            behavior_preservation_gate: "Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU before/after receipts must preserve prompt IDs, generated IDs, decoded text, selected backend/kernel, tokenizer authority, model SHA, and fallback_used=false",
            residual_add_involved: true,
            can_fill_caller_output_storage: false,
            exact_blocking_ops: CANDLE_RESIDUAL_ADD_EXACT_BLOCKING_OPS,
            public_api_return_type: CANDLE_RESIDUAL_ADD_PUBLIC_API_RETURN_TYPE,
            required_missing_api: CANDLE_RESIDUAL_ADD_REQUIRED_MISSING_API,
            public_api_accepts_output_storage: false,
            backend_internal_in_place_api_exposed: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogitsOutputStorageApiBoundary {
    pub role: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub next_api_hook: &'static str,
    pub exact_blocking_ops: &'static [&'static str],
    pub fused_selection_blocking_ops: &'static [&'static str],
    pub public_api_return_type: &'static str,
    pub required_missing_api: &'static str,
    pub public_api_accepts_output_storage: bool,
    pub backend_internal_in_place_api_exposed: bool,
    pub can_fill_caller_output_storage: bool,
    pub device_argmax_available_after_logits_tensor: bool,
    pub topk_sort_available_after_logits_tensor: bool,
    pub can_fuse_output_head_and_selection: bool,
}

pub const CANDLE_LOGITS_EXACT_BLOCKING_OPS: &[&str] = &[
    "candle_nn::Linear::forward(&self, &Tensor) -> Result<Tensor>",
    "Tensor::matmul(&self, &Tensor) -> Result<Tensor>",
    "Tensor::reshape(&self, shape) -> Result<Tensor>",
    "Tensor::to_vec1::<f32>(&self) -> Result<Vec<f32>> when host logits extraction is requested",
];

pub const CANDLE_LOGITS_PUBLIC_API_RETURN_TYPE: &str = "Result<Tensor>";

pub const CANDLE_LOGITS_REQUIRED_MISSING_API: &str = "logits/output-head API accepting caller-provided output storage or a fused top-k/argmax path that avoids materializing a full owned logits tensor";

pub const CANDLE_LOGITS_FUSED_SELECTION_BLOCKING_OPS: &[&str] = &[
    "candle_nn::Linear::forward(&self, &Tensor) -> Result<Tensor> materializes full logits before selection",
    "Tensor::matmul(&self, &Tensor) -> Result<Tensor> materializes full logits before selection",
    "Tensor::argmax(&self, dim) -> Result<Tensor> selects after full logits Tensor materialization",
    "Tensor::sort_last_dim(&self, asc) -> Result<(Tensor, Tensor)> sorts after full logits Tensor materialization",
    "Tensor::arg_sort_last_dim(&self, asc) -> Result<Tensor> sorts indices after full logits Tensor materialization",
];

impl LogitsOutputStorageApiBoundary {
    pub fn from_candle_logits(role: &'static str) -> Self {
        Self {
            role,
            status: "logits_output_storage_blocked_by_candle_tensor_ops",
            reason: "TransformerModel::logits produces an owned Candle Tensor through lm_head.forward or tied-embedding Tensor::matmul plus reshape; the public APIs expose no caller-provided output-storage parameter, and host logits extraction still allocates when full logits are requested",
            next_api_hook: CANDLE_LOGITS_REQUIRED_MISSING_API,
            exact_blocking_ops: CANDLE_LOGITS_EXACT_BLOCKING_OPS,
            fused_selection_blocking_ops: CANDLE_LOGITS_FUSED_SELECTION_BLOCKING_OPS,
            public_api_return_type: CANDLE_LOGITS_PUBLIC_API_RETURN_TYPE,
            required_missing_api: CANDLE_LOGITS_REQUIRED_MISSING_API,
            public_api_accepts_output_storage: false,
            backend_internal_in_place_api_exposed: false,
            can_fill_caller_output_storage: false,
            device_argmax_available_after_logits_tensor: true,
            topk_sort_available_after_logits_tensor: true,
            can_fuse_output_head_and_selection: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarFusedConsumerBoundary {
    pub role: &'static str,
    pub status: &'static str,
    pub reason: &'static str,
    pub exact_tensor_name: &'static str,
    pub exact_tensor_role: &'static str,
    pub sidecar_inner_matvec_accepts_output_slice: bool,
    pub can_avoid_returned_candle_tensor_for_current_consumer: bool,
    pub downstream_consumers_require_tensor_semantics: bool,
    pub exact_blocking_ops: &'static [&'static str],
    pub required_missing_api: &'static str,
    pub appliance_oracle_required_before_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarFusedQProjectionStageContract {
    pub stage: &'static str,
    pub consumes: &'static str,
    pub produces: &'static str,
    pub fused_consumer_must_own: bool,
    pub candle_tensor_semantics_required_today: bool,
    pub optional: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarFusedQProjectionShapeContract {
    pub input_rank: usize,
    pub projected_rank: usize,
    pub attention_heads_rank: usize,
    pub projected_shape: &'static str,
    pub attention_heads_shape: &'static str,
    pub head_handoff_shape: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarFusedQProjectionReceiptContract {
    pub required_before_runtime_execution: bool,
    pub required_before_allocation_claim: bool,
    pub required_before_speedup_claim: bool,
    pub required_fields: &'static [&'static str],
    pub gate: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarFusedQProjectionConsumerContract {
    pub role: &'static str,
    pub status: &'static str,
    pub source_boundary_status: &'static str,
    pub exact_tensor_name: &'static str,
    pub exact_tensor_role: &'static str,
    pub shape: DenseQ8SidecarFusedQProjectionShapeContract,
    pub stages: &'static [DenseQ8SidecarFusedQProjectionStageContract],
    pub receipt: DenseQ8SidecarFusedQProjectionReceiptContract,
    pub owns_packed_q8_matvec_output_slice: bool,
    pub intermediate_returned_candle_tensor_allowed: bool,
    pub runtime_execution_enabled: bool,
    pub allocation_reduction_claim: bool,
    pub speedup_claim: bool,
    pub default_runtime_changed: bool,
    pub required_missing_implementation: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedFusedQProjectionImplementationBlocker {
    pub blocker: &'static str,
    pub category: &'static str,
    pub exact_api_or_surface: &'static str,
    pub required_before_runtime_execution: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedFusedQProjectionImplementationGate {
    pub role: &'static str,
    pub status: &'static str,
    pub source_contract_status: &'static str,
    pub exact_tensor_name: &'static str,
    pub exact_tensor_role: &'static str,
    pub attempted_runtime_implementation: bool,
    pub can_own_packed_q8_matvec_output_slice: bool,
    pub can_preserve_downstream_tensor_semantics_without_intermediate_tensor: bool,
    pub runtime_execution_enabled: bool,
    pub default_runtime_changed: bool,
    pub allocation_reduction_claim: bool,
    pub speedup_claim: bool,
    pub blockers: &'static [DenseQ8SidecarTypedFusedQProjectionImplementationBlocker],
    pub receipt_gate: DenseQ8SidecarFusedQProjectionReceiptContract,
    pub next_required_slice: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedAttentionHeadViewStageContract {
    pub stage: &'static str,
    pub consumes: &'static str,
    pub produces: &'static str,
    pub storage_or_view_contract: &'static str,
    pub candle_tensor_semantics_required_today: bool,
    pub optional: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedAttentionHeadViewLayoutContract {
    pub projected_rank: usize,
    pub attention_heads_rank: usize,
    pub projected_shape: &'static str,
    pub attention_heads_shape: &'static str,
    pub logical_storage_order: &'static str,
    pub head_stride_contract: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedAttentionHeadViewBlocker {
    pub blocker: &'static str,
    pub category: &'static str,
    pub exact_api_or_surface: &'static str,
    pub required_before_runtime_execution: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedAttentionHeadViewGate {
    pub role: &'static str,
    pub status: &'static str,
    pub source_gate_status: &'static str,
    pub exact_tensor_name: &'static str,
    pub exact_tensor_role: &'static str,
    pub layout: DenseQ8SidecarTypedAttentionHeadViewLayoutContract,
    pub stages: &'static [DenseQ8SidecarTypedAttentionHeadViewStageContract],
    pub blockers: &'static [DenseQ8SidecarTypedAttentionHeadViewBlocker],
    pub receipt_gate: DenseQ8SidecarFusedQProjectionReceiptContract,
    pub can_represent_q_heads_without_candle_tensor: bool,
    pub can_feed_current_attention_score_api_without_materialization: bool,
    pub selected_materialization_point: Option<&'static str>,
    pub runtime_execution_enabled: bool,
    pub default_runtime_changed: bool,
    pub allocation_reduction_claim: bool,
    pub speedup_claim: bool,
    pub next_required_slice: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedAttentionHeadConsumerStageContract {
    pub stage: &'static str,
    pub consumes: &'static str,
    pub required_surface: &'static str,
    pub current_status: &'static str,
    pub candle_tensor_materialization_required_today: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedAttentionHeadConsumerBlocker {
    pub blocker: &'static str,
    pub category: &'static str,
    pub exact_api_or_surface: &'static str,
    pub required_before_runtime_execution: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedAttentionHeadConsumerGate {
    pub role: &'static str,
    pub status: &'static str,
    pub source_gate_status: &'static str,
    pub exact_tensor_name: &'static str,
    pub exact_tensor_role: &'static str,
    pub stages: &'static [DenseQ8SidecarTypedAttentionHeadConsumerStageContract],
    pub blockers: &'static [DenseQ8SidecarTypedAttentionHeadConsumerBlocker],
    pub receipt_gate: DenseQ8SidecarFusedQProjectionReceiptContract,
    pub can_consume_projection_output_slice: bool,
    pub can_apply_logical_head_view_without_candle_tensor: bool,
    pub can_apply_q_norm_without_candle_tensor: bool,
    pub can_apply_rope_without_candle_tensor: bool,
    pub can_feed_attention_scores_without_candle_tensor: bool,
    pub first_blocking_stage: &'static str,
    pub accepted_single_materialization_point: Option<&'static str>,
    pub candidate_materialization_points: &'static [&'static str],
    pub runtime_execution_enabled: bool,
    pub default_runtime_changed: bool,
    pub allocation_reduction_claim: bool,
    pub speedup_claim: bool,
    pub next_required_slice: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedQNormRopeConsumerStageContract {
    pub stage: &'static str,
    pub consumes: &'static str,
    pub required_surface: &'static str,
    pub current_status: &'static str,
    pub selected_materialization_boundary: Option<&'static str>,
    pub candle_tensor_materialization_required_today: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedQNormRopeConsumerBlocker {
    pub blocker: &'static str,
    pub category: &'static str,
    pub exact_api_or_surface: &'static str,
    pub required_before_runtime_execution: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarTypedQNormRopeConsumerGate {
    pub role: &'static str,
    pub status: &'static str,
    pub source_gate_status: &'static str,
    pub exact_tensor_name: &'static str,
    pub exact_tensor_role: &'static str,
    pub stages: &'static [DenseQ8SidecarTypedQNormRopeConsumerStageContract],
    pub blockers: &'static [DenseQ8SidecarTypedQNormRopeConsumerBlocker],
    pub receipt_gate: DenseQ8SidecarFusedQProjectionReceiptContract,
    pub can_consume_logical_q_head_view: bool,
    pub can_apply_typed_q_norm_without_candle_tensor: bool,
    pub can_apply_typed_rope_without_candle_tensor: bool,
    pub can_preserve_trace_identity_without_tensor_mapping: bool,
    pub can_feed_attention_scores_without_candle_tensor: bool,
    pub first_blocking_stage: &'static str,
    pub accepted_single_materialization_point: Option<&'static str>,
    pub candidate_materialization_points: &'static [&'static str],
    pub runtime_execution_enabled: bool,
    pub default_runtime_changed: bool,
    pub packed_q8_sidecar_default_enabled: bool,
    pub allocation_reduction_claim: bool,
    pub speedup_claim: bool,
    pub next_required_slice: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarQNormMaterializationBoundaryGate {
    pub role: &'static str,
    pub status: &'static str,
    pub source_gate_status: &'static str,
    pub exact_tensor_name: &'static str,
    pub exact_tensor_role: &'static str,
    pub accepted_single_materialization_point: &'static str,
    pub rejected_materialization_points: &'static [&'static str],
    pub materializes_before_stage: &'static str,
    pub preserved_candle_consumers: &'static [&'static str],
    pub receipt_gate: DenseQ8SidecarFusedQProjectionReceiptContract,
    pub runtime_execution_enabled: bool,
    pub default_runtime_changed: bool,
    pub packed_q8_sidecar_default_enabled: bool,
    pub allocation_reduction_claim: bool,
    pub speedup_claim: bool,
    pub sustained_throughput_claim: bool,
    pub q4_q5_runtime_claim: bool,
    pub qwen3_q8_before_after_receipts_required: bool,
    pub qwen25_q8_before_after_receipts_required: bool,
    pub next_required_slice: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarQNormInputProofReceiptRequirement {
    pub model_id: &'static str,
    pub model_architecture: &'static str,
    pub quant_format: &'static str,
    pub required_before_receipt: &'static str,
    pub required_after_receipt: &'static str,
    pub required_fields: &'static [&'static str],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarQNormInputProofBlocker {
    pub blocker: &'static str,
    pub category: &'static str,
    pub exact_api_or_surface: &'static str,
    pub required_before_proof: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarQNormInputReceiptComparatorGate {
    pub role: &'static str,
    pub status: &'static str,
    pub selected_materialization_boundary: &'static str,
    pub required_identity_fields: &'static [&'static str],
    pub fail_closed_on_missing_field: bool,
    pub fail_closed_on_mismatch: bool,
    pub fail_closed_on_fallback: bool,
    pub compares_qwen3_q8: bool,
    pub compares_qwen25_q8: bool,
    pub runtime_execution_enabled: bool,
    pub allocation_reduction_claim: bool,
    pub speedup_claim: bool,
    pub sustained_throughput_claim: bool,
    pub q4_q5_runtime_claim: bool,
    pub server_or_accelerator_claim: bool,
    pub qwen35_claim: bool,
    pub bitnet_qk256_claim: bool,
    pub remaining_blockers: &'static [&'static str],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarQNormInputRuntimeHookGate {
    pub role: &'static str,
    pub status: &'static str,
    pub source_comparator_status: &'static str,
    pub exact_tensor_name: &'static str,
    pub exact_tensor_role: &'static str,
    pub selected_materialization_boundary: &'static str,
    pub hook_identity: &'static str,
    pub hook_scope: &'static str,
    pub hook_runtime_enabled: bool,
    pub hook_default_enabled: bool,
    pub tensor_identity_surface_defined: bool,
    pub tensor_identity_fields: &'static [&'static str],
    pub after_receipt_field: &'static str,
    pub preserves_eager_f32_default: bool,
    pub packed_q8_sidecar_default_enabled: bool,
    pub required_receipts: &'static [DenseQ8SidecarQNormInputProofReceiptRequirement],
    pub remaining_blockers: &'static [&'static str],
    pub allocation_reduction_claim: bool,
    pub speedup_claim: bool,
    pub sustained_throughput_claim: bool,
    pub q4_q5_runtime_claim: bool,
    pub server_or_accelerator_claim: bool,
    pub qwen35_claim: bool,
    pub bitnet_qk256_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarQNormInputReceiptIdentity {
    pub model_id: &'static str,
    pub model_sha256: &'static str,
    pub tokenizer_source: &'static str,
    pub tokenizer_strict: bool,
    pub prompt_ids_digest: &'static str,
    pub generated_ids_digest: &'static str,
    pub decoded_text_digest: &'static str,
    pub selected_backend: &'static str,
    pub selected_kernel_identity: &'static str,
    pub dense_hook_identity: &'static str,
    pub q_norm_input_boundary: &'static str,
    pub q_norm_input_tensor_identity: &'static str,
    pub fallback_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarQNormInputReceiptComparison {
    pub passed: bool,
    pub failed_fields: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseQ8SidecarQNormInputProofGate {
    pub role: &'static str,
    pub status: &'static str,
    pub source_boundary_status: &'static str,
    pub exact_tensor_name: &'static str,
    pub exact_tensor_role: &'static str,
    pub selected_materialization_boundary: &'static str,
    pub required_receipts: &'static [DenseQ8SidecarQNormInputProofReceiptRequirement],
    pub blockers: &'static [DenseQ8SidecarQNormInputProofBlocker],
    pub proof_ready: bool,
    pub missing_runtime_hook: bool,
    pub missing_receipt_field: bool,
    pub missing_comparator: bool,
    pub comparator_contract_defined: bool,
    pub tensor_identity_unrecorded: bool,
    pub accumulator_order_unproven: bool,
    pub artifact_gap: bool,
    pub runtime_execution_enabled: bool,
    pub default_runtime_changed: bool,
    pub packed_q8_sidecar_default_enabled: bool,
    pub allocation_reduction_claim: bool,
    pub speedup_claim: bool,
    pub sustained_throughput_claim: bool,
    pub q4_q5_runtime_claim: bool,
    pub server_or_accelerator_claim: bool,
    pub qwen35_claim: bool,
    pub bitnet_qk256_claim: bool,
    pub next_required_slice: &'static str,
}

pub const DENSE_Q8_SIDECAR_FUSED_CONSUMER_EXACT_BLOCKING_OPS: &[&str] = &[
    "dense_q8_sidecar_linear_forward(&Tensor, Option<&Tensor>, &DenseLinearPackedQ8Payload) -> candle_core::Result<Tensor>",
    "Tensor::from_vec(output, output_shape, input.device()) transfers owned Vec storage into Candle",
    "MultiHeadAttention::reshape_qkv_heads consumes q_proj as a Candle Tensor and applies Tensor::reshape plus Tensor::transpose",
    "Qwen q_norm/k_norm paths consume attention heads as Candle Tensors through candle_nn::LayerNorm::forward",
    "RoPE/cache/trace/workspace paths preserve Tensor-shaped attention-head semantics before score computation",
];

pub const DENSE_Q8_SIDECAR_FUSED_CONSUMER_REQUIRED_API: &str = "typed fused Q projection consumer accepting packed-Q8 matvec output slices and applying reshape, q_norm, RoPE, trace/workspace identity, and attention-head handoff without materializing an intermediate returned Candle Tensor";

pub const DENSE_Q8_SIDECAR_FUSED_Q_PROJECTION_STAGES:
    &[DenseQ8SidecarFusedQProjectionStageContract] = &[
    DenseQ8SidecarFusedQProjectionStageContract {
        stage: "packed_q8_matvec_output_slice",
        consumes: "DenseLinearPackedQ8Payload plus input rows",
        produces: "&mut [f32] q projection rows",
        fused_consumer_must_own: true,
        candle_tensor_semantics_required_today: false,
        optional: false,
    },
    DenseQ8SidecarFusedQProjectionStageContract {
        stage: "q_proj_reshape",
        consumes: "[batch, seq, n_heads * head_dim]",
        produces: "[batch, seq, n_heads, head_dim]",
        fused_consumer_must_own: true,
        candle_tensor_semantics_required_today: true,
        optional: false,
    },
    DenseQ8SidecarFusedQProjectionStageContract {
        stage: "q_proj_transpose",
        consumes: "[batch, seq, n_heads, head_dim]",
        produces: "[batch, n_heads, seq, head_dim]",
        fused_consumer_must_own: true,
        candle_tensor_semantics_required_today: true,
        optional: false,
    },
    DenseQ8SidecarFusedQProjectionStageContract {
        stage: "optional_q_norm",
        consumes: "[batch, n_heads, seq, head_dim]",
        produces: "[batch, n_heads, seq, head_dim]",
        fused_consumer_must_own: true,
        candle_tensor_semantics_required_today: true,
        optional: true,
    },
    DenseQ8SidecarFusedQProjectionStageContract {
        stage: "q_rope",
        consumes: "[batch, n_heads, seq, head_dim] plus position",
        produces: "[batch, n_heads, seq, head_dim]",
        fused_consumer_must_own: true,
        candle_tensor_semantics_required_today: true,
        optional: false,
    },
    DenseQ8SidecarFusedQProjectionStageContract {
        stage: "trace_workspace_identity",
        consumes: "projection, heads, q_norm, q_rope identity",
        produces: "trace/workspace source identity",
        fused_consumer_must_own: true,
        candle_tensor_semantics_required_today: true,
        optional: false,
    },
    DenseQ8SidecarFusedQProjectionStageContract {
        stage: "attention_head_handoff",
        consumes: "[batch, n_heads, seq, head_dim]",
        produces: "AttentionHeads.q-compatible handoff before scores",
        fused_consumer_must_own: true,
        candle_tensor_semantics_required_today: true,
        optional: false,
    },
];

pub const DENSE_Q8_SIDECAR_FUSED_Q_PROJECTION_RECEIPT_FIELDS: &[&str] = &[
    "model.sha256",
    "tokenizer.source=gguf_metadata",
    "tokenizer.strict=true",
    "prompt_ids",
    "generated_ids",
    "decoded_text",
    "selected_backend=cpu-rust",
    "selected_kernel identity",
    "dense_hook identity",
    "fallback_used=false",
];

pub const DENSE_Q8_SIDECAR_TYPED_FUSED_Q_PROJECTION_IMPLEMENTATION_BLOCKERS:
    &[DenseQ8SidecarTypedFusedQProjectionImplementationBlocker] = &[
    DenseQ8SidecarTypedFusedQProjectionImplementationBlocker {
        blocker: "q_heads_tensor_semantics",
        category: "tensor-layout",
        exact_api_or_surface: "MultiHeadAttention::reshape_qkv_heads returns AttentionHeads { q: Tensor, k: Tensor, v: Tensor } after Tensor::reshape and Tensor::transpose",
        required_before_runtime_execution: "Introduce a typed AttentionHeads buffer/view that preserves [batch, heads, seq, head_dim] layout and can feed all downstream attention stages without first constructing an intermediate returned Candle Tensor.",
    },
    DenseQ8SidecarTypedFusedQProjectionImplementationBlocker {
        blocker: "q_norm_tensor_api",
        category: "API",
        exact_api_or_surface: "candle_nn::LayerNorm::forward(&Tensor) -> Result<Tensor> in MultiHeadAttention::apply_qk_norms",
        required_before_runtime_execution: "Add a behavior-equivalent q_norm path for the typed Q-head buffer, or prove that materializing only at the q_norm boundary preserves the SLM-CPU-099 no-intermediate-Candle-Tensor contract.",
    },
    DenseQ8SidecarTypedFusedQProjectionImplementationBlocker {
        blocker: "rope_tensor_api",
        category: "API",
        exact_api_or_surface: "RotaryEmbedding::apply(&Tensor, position) -> Result<Tensor>",
        required_before_runtime_execution: "Add a typed Q-head RoPE application that matches the existing split-layout RoPE tables, position handling, device/dtype behavior, and trace summaries before attention scores.",
    },
    DenseQ8SidecarTypedFusedQProjectionImplementationBlocker {
        blocker: "trace_workspace_tensor_identity",
        category: "lifetime",
        exact_api_or_surface: "TransformerAttentionOutputSourceTensors stores q_projection, q_heads, q_norm, and q_rope as Candle Tensor values",
        required_before_runtime_execution: "Define how typed fused Q buffers are represented in checkpoint traces and workspace source-tensor receipts without losing the current diagnostic identity or requiring a returned intermediate Tensor.",
    },
    DenseQ8SidecarTypedFusedQProjectionImplementationBlocker {
        blocker: "attention_handoff_tensor_contract",
        category: "API",
        exact_api_or_surface: "prepare_attention_scores consumes q as &Tensor and uses Tensor matmul/transpose/dtype operations",
        required_before_runtime_execution: "Either extend the typed Q buffer through score computation or provide an explicitly proven single materialization point after q_norm/RoPE that does not claim SLM-CPU-099 fused-consumer completion.",
    },
    DenseQ8SidecarTypedFusedQProjectionImplementationBlocker {
        blocker: "receipt_safety_evidence",
        category: "receipt-safety",
        exact_api_or_surface: "SLM-CPU-099 receipt gate requires repeated Qwen3 Q8_0 before/after receipts with identical IDs/text/backend/dense-hook/fallback=false",
        required_before_runtime_execution: "Capture behavior-preserving before/after Qwen3 Q8_0 appliance receipts before enabling any runtime-adjacent fused Q consumer path or claiming allocation/timing improvement.",
    },
];

pub const DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_VIEW_STAGES:
    &[DenseQ8SidecarTypedAttentionHeadViewStageContract] = &[
    DenseQ8SidecarTypedAttentionHeadViewStageContract {
        stage: "q_projection_output_slice",
        consumes: "&mut [f32] rows from the exact packed-Q8 q_proj matvec",
        produces: "logical [batch, seq, n_heads * head_dim] projection view",
        storage_or_view_contract: "contiguous row-major projection rows owned by the fused Q consumer workspace",
        candle_tensor_semantics_required_today: false,
        optional: false,
    },
    DenseQ8SidecarTypedAttentionHeadViewStageContract {
        stage: "logical_head_view",
        consumes: "[batch, seq, n_heads * head_dim] projection view",
        produces: "[batch, n_heads, seq, head_dim] logical Q-head view",
        storage_or_view_contract: "head-major logical strides over the same projection storage without a returned intermediate Candle Tensor",
        candle_tensor_semantics_required_today: false,
        optional: false,
    },
    DenseQ8SidecarTypedAttentionHeadViewStageContract {
        stage: "optional_q_norm_handoff",
        consumes: "[batch, n_heads, seq, head_dim] logical Q-head view",
        produces: "q_norm-compatible typed buffer or explicit materialization blocker",
        storage_or_view_contract: "must preserve Qwen per-head q_norm semantics before RoPE",
        candle_tensor_semantics_required_today: true,
        optional: true,
    },
    DenseQ8SidecarTypedAttentionHeadViewStageContract {
        stage: "q_rope_handoff",
        consumes: "q_norm output or logical Q-head view plus position",
        produces: "RoPE-compatible typed Q-head buffer or explicit materialization blocker",
        storage_or_view_contract: "must preserve split-layout RoPE position, dtype, and trace summaries",
        candle_tensor_semantics_required_today: true,
        optional: false,
    },
    DenseQ8SidecarTypedAttentionHeadViewStageContract {
        stage: "trace_workspace_identity",
        consumes: "projection, logical heads, optional q_norm, and q_rope typed identities",
        produces: "checkpoint trace/workspace source identity without losing diagnostic surfaces",
        storage_or_view_contract: "must map to the current TransformerAttentionOutputSourceTensors receipt fields or explicitly block",
        candle_tensor_semantics_required_today: true,
        optional: false,
    },
    DenseQ8SidecarTypedAttentionHeadViewStageContract {
        stage: "attention_score_handoff",
        consumes: "typed Q-head buffer after q_norm/RoPE",
        produces: "prepare_attention_scores-compatible Q input or explicit materialization blocker",
        storage_or_view_contract: "must preserve [batch, heads, seq, head_dim] score semantics and GQA interaction",
        candle_tensor_semantics_required_today: true,
        optional: false,
    },
];

pub const DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_VIEW_BLOCKERS:
    &[DenseQ8SidecarTypedAttentionHeadViewBlocker] = &[
    DenseQ8SidecarTypedAttentionHeadViewBlocker {
        blocker: "q_norm_requires_tensor_or_typed_norm",
        category: "API",
        exact_api_or_surface: "MultiHeadAttention::apply_qk_norms calls candle_nn::LayerNorm::forward(&Tensor) for q_norm",
        required_before_runtime_execution: "Add a behavior-equivalent typed q_norm over the logical Q-head buffer or record an explicit single-materialization point before q_norm with Qwen3 Q8_0 before/after receipts.",
    },
    DenseQ8SidecarTypedAttentionHeadViewBlocker {
        blocker: "rope_requires_tensor_or_typed_rope",
        category: "API",
        exact_api_or_surface: "MultiHeadAttention::apply_rotary_embeddings calls RotaryEmbedding::apply(&Tensor, position)",
        required_before_runtime_execution: "Add a typed RoPE path for the logical Q-head buffer that preserves split-layout tables, position indexing, dtype behavior, and trace summaries.",
    },
    DenseQ8SidecarTypedAttentionHeadViewBlocker {
        blocker: "trace_source_identity_requires_tensor_mapping",
        category: "receipt-safety",
        exact_api_or_surface: "TransformerAttentionOutputSourceTensors currently stores q_projection, q_heads, q_norm, and q_rope as Candle Tensor values",
        required_before_runtime_execution: "Define typed trace fingerprints for the Q projection/head/norm/RoPE surfaces without weakening checkpoint comparability or receipt provenance.",
    },
    DenseQ8SidecarTypedAttentionHeadViewBlocker {
        blocker: "attention_scores_require_tensor_or_typed_score_path",
        category: "API",
        exact_api_or_surface: "MultiHeadAttention::prepare_attention_scores consumes q as &Tensor and uses Tensor matmul/transpose/dtype operations",
        required_before_runtime_execution: "Either extend the typed Q-head buffer through attention score computation or prove a single materialization point after RoPE that preserves generated IDs and dense-hook identity.",
    },
    DenseQ8SidecarTypedAttentionHeadViewBlocker {
        blocker: "receipt_safety_evidence",
        category: "receipt-safety",
        exact_api_or_surface: "Runtime-adjacent fused Q execution requires repeated Qwen3 Q8_0 before/after appliance receipts",
        required_before_runtime_execution: "Capture repeated before/after receipts proving identical model SHA, strict tokenizer authority, prompt IDs, generated IDs, decoded text, backend/kernel identity, dense-hook identity, and fallback_used=false.",
    },
];

pub const DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_CONSUMER_STAGES:
    &[DenseQ8SidecarTypedAttentionHeadConsumerStageContract] = &[
    DenseQ8SidecarTypedAttentionHeadConsumerStageContract {
        stage: "projection_slice_ingress",
        consumes: "&mut [f32] rows from the exact packed-Q8 q_proj matvec",
        required_surface: "typed projection-row slice with [batch, seq, n_heads * head_dim] metadata",
        current_status: "representable_without_candle_tensor",
        candle_tensor_materialization_required_today: false,
    },
    DenseQ8SidecarTypedAttentionHeadConsumerStageContract {
        stage: "logical_head_view_ingress",
        consumes: "typed projection-row slice",
        required_surface: "typed [batch, n_heads, seq, head_dim] Q-head logical view",
        current_status: "representable_without_candle_tensor",
        candle_tensor_materialization_required_today: false,
    },
    DenseQ8SidecarTypedAttentionHeadConsumerStageContract {
        stage: "q_norm_consumer",
        consumes: "typed Q-head logical view",
        required_surface: "behavior-equivalent typed q_norm or proven materialization boundary",
        current_status: "blocked_by_layernorm_tensor_api",
        candle_tensor_materialization_required_today: true,
    },
    DenseQ8SidecarTypedAttentionHeadConsumerStageContract {
        stage: "rope_consumer",
        consumes: "q_norm output or typed Q-head logical view",
        required_surface: "behavior-equivalent typed RoPE preserving split-layout table and position semantics",
        current_status: "blocked_by_rotary_embedding_tensor_api",
        candle_tensor_materialization_required_today: true,
    },
    DenseQ8SidecarTypedAttentionHeadConsumerStageContract {
        stage: "trace_identity_consumer",
        consumes: "typed projection, head, q_norm, and q_rope identities",
        required_surface: "typed trace fingerprints equivalent to TransformerAttentionOutputSourceTensors",
        current_status: "blocked_by_tensor_receipt_identity_gap",
        candle_tensor_materialization_required_today: true,
    },
    DenseQ8SidecarTypedAttentionHeadConsumerStageContract {
        stage: "attention_score_consumer",
        consumes: "typed Q-head buffer after q_norm/RoPE",
        required_surface: "typed score path or proven score-handoff materialization boundary",
        current_status: "blocked_by_prepare_attention_scores_tensor_api",
        candle_tensor_materialization_required_today: true,
    },
    DenseQ8SidecarTypedAttentionHeadConsumerStageContract {
        stage: "receipt_safety_gate",
        consumes: "runtime-adjacent fused Q execution candidate",
        required_surface: "repeated Qwen3 Q8_0 and Qwen2.5 Q8_0 before/after receipts",
        current_status: "blocked_until_behavior_oracles_pass",
        candle_tensor_materialization_required_today: true,
    },
];

pub const DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_CONSUMER_BLOCKERS:
    &[DenseQ8SidecarTypedAttentionHeadConsumerBlocker] = &[
    DenseQ8SidecarTypedAttentionHeadConsumerBlocker {
        blocker: "q_norm_typed_consumer_absent",
        category: "API",
        exact_api_or_surface: "MultiHeadAttention::apply_qk_norms calls candle_nn::LayerNorm::forward(&Tensor) for q_norm",
        required_before_runtime_execution: "Add a behavior-equivalent typed q_norm over the logical Q-head view, or prove a materialization boundary at q_norm input with strict before/after Qwen3 and Qwen2.5 CPU receipts.",
    },
    DenseQ8SidecarTypedAttentionHeadConsumerBlocker {
        blocker: "rope_typed_consumer_absent",
        category: "API",
        exact_api_or_surface: "MultiHeadAttention::apply_rotary_embeddings calls RotaryEmbedding::apply(&Tensor, position)",
        required_before_runtime_execution: "Add a typed RoPE consumer preserving split-layout tables, position indexing, dtype behavior, and trace summaries before attention scores.",
    },
    DenseQ8SidecarTypedAttentionHeadConsumerBlocker {
        blocker: "trace_identity_typed_receipt_gap",
        category: "receipt-safety",
        exact_api_or_surface: "TransformerAttentionOutputSourceTensors stores q_projection, q_heads, q_norm, and q_rope as Candle Tensor values",
        required_before_runtime_execution: "Define typed trace fingerprints for projection, head, q_norm, and q_rope surfaces without weakening checkpoint comparability or receipt provenance.",
    },
    DenseQ8SidecarTypedAttentionHeadConsumerBlocker {
        blocker: "attention_score_typed_path_absent",
        category: "API",
        exact_api_or_surface: "MultiHeadAttention::prepare_attention_scores consumes q as &Tensor and uses Tensor matmul, transpose, dtype, and GQA score operations",
        required_before_runtime_execution: "Add a typed attention-score path or prove a single score-handoff materialization boundary after q_norm/RoPE with strict generated-ID and dense-hook preservation.",
    },
    DenseQ8SidecarTypedAttentionHeadConsumerBlocker {
        blocker: "accumulator_order_unproven",
        category: "accumulator-order",
        exact_api_or_surface: "Typed q_norm/RoPE/score execution could change floating-point operation order relative to Candle Tensor operations",
        required_before_runtime_execution: "Capture before/after CPU behavior receipts and any focused numerical parity evidence before claiming an allocation or timing improvement.",
    },
    DenseQ8SidecarTypedAttentionHeadConsumerBlocker {
        blocker: "receipt_safety_evidence",
        category: "receipt-safety",
        exact_api_or_surface: "Runtime-adjacent fused Q execution requires repeated Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU behavior-oracle receipts",
        required_before_runtime_execution: "Prove identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, and fallback_used=false before runtime enablement.",
    },
];

pub const DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_CONSUMER_CANDIDATE_MATERIALIZATION_POINTS:
    &[&str] = &[
    "q_norm_input_candle_tensor_boundary",
    "after_q_norm_before_rope_candle_tensor_boundary",
    "after_q_rope_before_attention_scores_candle_tensor_boundary",
];

pub const DENSE_Q8_SIDECAR_Q_NORM_MATERIALIZATION_REJECTED_POINTS: &[&str] = &[
    "after_q_norm_before_rope_candle_tensor_boundary",
    "after_q_rope_before_attention_scores_candle_tensor_boundary",
];

pub const DENSE_Q8_SIDECAR_Q_NORM_MATERIALIZATION_PRESERVED_CONSUMERS: &[&str] = &[
    "candle_nn::LayerNorm::forward(&Tensor) for q_norm",
    "RotaryEmbedding::apply(&Tensor, position) for RoPE",
    "TransformerAttentionOutputSourceTensors q_heads/q_norm/q_rope Tensor identity",
    "MultiHeadAttention::prepare_attention_scores(&Tensor, ...) Tensor score handoff",
];

pub const DENSE_Q8_SIDECAR_Q_NORM_INPUT_PROOF_REQUIRED_RECEIPTS:
    &[DenseQ8SidecarQNormInputProofReceiptRequirement] = &[
    DenseQ8SidecarQNormInputProofReceiptRequirement {
        model_id: "qwen3-0.6b-q8_0",
        model_architecture: "qwen3",
        quant_format: "Q8_0",
        required_before_receipt: "strict CPU eager_f32_candle receipt before q_norm_input materialization candidate",
        required_after_receipt: "strict CPU receipt after q_norm_input materialization candidate",
        required_fields: DENSE_Q8_SIDECAR_FUSED_Q_PROJECTION_RECEIPT_FIELDS,
    },
    DenseQ8SidecarQNormInputProofReceiptRequirement {
        model_id: "qwen2.5-0.5b-instruct-q8_0",
        model_architecture: "qwen2",
        quant_format: "Q8_0",
        required_before_receipt: "strict CPU eager_f32_candle receipt before q_norm_input materialization candidate",
        required_after_receipt: "strict CPU receipt after q_norm_input materialization candidate",
        required_fields: DENSE_Q8_SIDECAR_FUSED_Q_PROJECTION_RECEIPT_FIELDS,
    },
];

pub const DENSE_Q8_SIDECAR_Q_NORM_INPUT_RECEIPT_COMPARATOR_FIELDS: &[&str] = &[
    "model_id",
    "model_sha256",
    "tokenizer_source=gguf_metadata",
    "tokenizer_strict=true",
    "prompt_ids_digest",
    "generated_ids_digest",
    "decoded_text_digest",
    "selected_backend=cpu-rust",
    "selected_kernel_identity",
    "dense_hook_identity",
    "q_norm_input_boundary=q_norm_input_candle_tensor_boundary",
    "q_norm_input_tensor_identity",
    "fallback_used=false",
];

pub const DENSE_Q8_SIDECAR_Q_NORM_INPUT_RECEIPT_COMPARATOR_REMAINING_BLOCKERS: &[&str] = &[
    "qwen3_q8_before_after_receipts_missing",
    "qwen25_q8_before_after_receipts_missing",
    "accumulator_order_unproven",
];

pub const DENSE_Q8_SIDECAR_Q_NORM_INPUT_PROOF_BLOCKERS: &[DenseQ8SidecarQNormInputProofBlocker] = &[
    DenseQ8SidecarQNormInputProofBlocker {
        blocker: "qwen3_q8_before_after_receipts_missing",
        category: "artifact-gap",
        exact_api_or_surface: "No SLM-CPU-105 Qwen3 Q8_0 before/after strict CPU receipt pair exists for q_norm_input_candle_tensor_boundary",
        required_before_proof: "Collect before and after receipts with identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, CPU backend/kernel identity, dense hook identity, and fallback_used=false.",
    },
    DenseQ8SidecarQNormInputProofBlocker {
        blocker: "qwen25_q8_before_after_receipts_missing",
        category: "artifact-gap",
        exact_api_or_surface: "No SLM-CPU-105 Qwen2.5 Q8_0 before/after strict CPU receipt pair exists for q_norm_input_candle_tensor_boundary",
        required_before_proof: "Collect the same before/after receipt contract for the second Qwen dense SLM before treating the boundary as proven.",
    },
    DenseQ8SidecarQNormInputProofBlocker {
        blocker: "accumulator_order_unproven",
        category: "accumulator-order",
        exact_api_or_surface: "Packed-Q8 sidecar q_proj accumulation feeding q_norm_input may differ from the eager_f32_candle path before Candle LayerNorm consumes the materialized Tensor",
        required_before_proof: "Record generated-ID/text equivalence and focused numerical evidence before claiming behavior preservation or any allocation/timing improvement.",
    },
];

pub const DENSE_Q8_SIDECAR_Q_NORM_INPUT_RUNTIME_HOOK_TENSOR_IDENTITY_FIELDS: &[&str] = &[
    "q_norm_input.boundary=q_norm_input_candle_tensor_boundary",
    "q_norm_input.source_tensor=layers.0.attention.q_proj.weight",
    "q_norm_input.source_stage=attention.q_proj.reshape_q_heads",
    "q_norm_input.shape",
    "q_norm_input.dtype",
    "q_norm_input.dense_hook_identity",
    "q_norm_input.tensor_fingerprint_sha256_f32_le",
];

pub const DENSE_Q8_SIDECAR_Q_NORM_INPUT_RUNTIME_HOOK_REMAINING_BLOCKERS: &[&str] = &[
    "qwen3_q8_before_after_receipts_missing",
    "qwen25_q8_before_after_receipts_missing",
    "accumulator_order_unproven",
];

pub const DENSE_Q8_SIDECAR_TYPED_Q_NORM_ROPE_CONSUMER_STAGES:
    &[DenseQ8SidecarTypedQNormRopeConsumerStageContract] = &[
    DenseQ8SidecarTypedQNormRopeConsumerStageContract {
        stage: "typed_q_head_view_ingress",
        consumes: "typed [batch, n_heads, seq, head_dim] Q-head logical view",
        required_surface: "stable logical Q-head view carrying exact packed-Q8 q_proj source identity",
        current_status: "representable_without_candle_tensor",
        selected_materialization_boundary: None,
        candle_tensor_materialization_required_today: false,
    },
    DenseQ8SidecarTypedQNormRopeConsumerStageContract {
        stage: "typed_q_norm_consumer",
        consumes: "typed Q-head logical view plus optional Qwen q_norm weights",
        required_surface: "behavior-equivalent typed q_norm matching candle_nn::LayerNorm::forward accumulation, epsilon, dtype, and per-head axis semantics",
        current_status: "blocked_by_layernorm_tensor_api_and_accumulator_order_receipt_gap",
        selected_materialization_boundary: None,
        candle_tensor_materialization_required_today: true,
    },
    DenseQ8SidecarTypedQNormRopeConsumerStageContract {
        stage: "typed_rope_consumer",
        consumes: "typed q_norm output or typed Q-head logical view plus RoPE tables and position",
        required_surface: "behavior-equivalent typed RoPE preserving Qwen split-layout rotation, position index, dtype, and trace summaries",
        current_status: "blocked_by_rotary_embedding_tensor_api",
        selected_materialization_boundary: None,
        candle_tensor_materialization_required_today: true,
    },
    DenseQ8SidecarTypedQNormRopeConsumerStageContract {
        stage: "trace_workspace_identity_handoff",
        consumes: "typed q_head, q_norm, and q_rope identities",
        required_surface: "typed trace fingerprints equivalent to TransformerAttentionOutputSourceTensors q_heads/q_norm/q_rope fields",
        current_status: "blocked_by_tensor_receipt_identity_gap",
        selected_materialization_boundary: None,
        candle_tensor_materialization_required_today: true,
    },
    DenseQ8SidecarTypedQNormRopeConsumerStageContract {
        stage: "attention_score_handoff",
        consumes: "typed Q-head buffer after q_norm and RoPE",
        required_surface: "typed score path or a single proven materialization boundary before prepare_attention_scores",
        current_status: "blocked_by_prepare_attention_scores_tensor_api",
        selected_materialization_boundary: None,
        candle_tensor_materialization_required_today: true,
    },
    DenseQ8SidecarTypedQNormRopeConsumerStageContract {
        stage: "receipt_safety_gate",
        consumes: "any runtime-adjacent typed q_norm/RoPE consumer candidate",
        required_surface: "before/after Qwen3 Q8_0 and Qwen2.5 Q8_0 CPU receipts with identical generated IDs, decoded text, dense hook identity, and fallback=false",
        current_status: "blocked_until_behavior_oracles_pass",
        selected_materialization_boundary: None,
        candle_tensor_materialization_required_today: true,
    },
];

pub const DENSE_Q8_SIDECAR_TYPED_Q_NORM_ROPE_CONSUMER_BLOCKERS:
    &[DenseQ8SidecarTypedQNormRopeConsumerBlocker] = &[
    DenseQ8SidecarTypedQNormRopeConsumerBlocker {
        blocker: "typed_q_norm_kernel_absent",
        category: "API",
        exact_api_or_surface: "MultiHeadAttention::apply_qk_norms calls candle_nn::LayerNorm::forward(&Tensor) for q_norm",
        required_before_runtime_execution: "Add a typed q_norm kernel over the logical Q-head view that matches Candle LayerNorm/RMSNorm behavior, axis choice, epsilon, dtype, and f32 accumulator order, or prove materialization at q_norm input with strict Qwen3/Qwen2.5 before/after receipts.",
    },
    DenseQ8SidecarTypedQNormRopeConsumerBlocker {
        blocker: "typed_rope_kernel_absent",
        category: "API",
        exact_api_or_surface: "MultiHeadAttention::apply_rotary_embeddings calls RotaryEmbedding::apply(&Tensor, position)",
        required_before_runtime_execution: "Add typed RoPE over the logical Q-head view preserving split-layout tables, position indexing, head_dim rotation, dtype behavior, and checkpoint trace summaries.",
    },
    DenseQ8SidecarTypedQNormRopeConsumerBlocker {
        blocker: "trace_identity_typed_surface_absent",
        category: "receipt-safety",
        exact_api_or_surface: "TransformerAttentionOutputSourceTensors stores q_heads, q_norm, and q_rope as Candle Tensor values",
        required_before_runtime_execution: "Define typed trace fingerprints for q_head, q_norm, and q_rope that remain comparable with reference checkpoint packs and strict receipts.",
    },
    DenseQ8SidecarTypedQNormRopeConsumerBlocker {
        blocker: "score_handoff_typed_surface_absent",
        category: "API",
        exact_api_or_surface: "MultiHeadAttention::prepare_attention_scores consumes q as &Tensor and uses Tensor matmul, transpose, dtype, and GQA score operations",
        required_before_runtime_execution: "Add a typed score handoff or prove exactly one materialization boundary after q_norm/RoPE before attention scores without claiming packed-Q8 sidecar default-runtime promotion.",
    },
    DenseQ8SidecarTypedQNormRopeConsumerBlocker {
        blocker: "single_materialization_boundary_unproven",
        category: "layout",
        exact_api_or_surface: "candidate boundaries: q_norm input, after q_norm before RoPE, or after q_rope before attention scores",
        required_before_runtime_execution: "Select one boundary only after strict receipts prove identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, and fallback_used=false.",
    },
    DenseQ8SidecarTypedQNormRopeConsumerBlocker {
        blocker: "accumulator_order_receipt_absent",
        category: "accumulator-order",
        exact_api_or_surface: "typed q_norm and RoPE execution can change floating-point operation order relative to Candle Tensor operations",
        required_before_runtime_execution: "Record numerical parity evidence and before/after Qwen3 Q8_0 plus Qwen2.5 Q8_0 CPU receipts before enabling runtime-adjacent execution or claiming allocation/timing improvement.",
    },
    DenseQ8SidecarTypedQNormRopeConsumerBlocker {
        blocker: "receipt_safety_evidence",
        category: "receipt-safety",
        exact_api_or_surface: "SLM-CPU-103 runtime-adjacent changes require repeated strict CPU behavior-oracle receipts",
        required_before_runtime_execution: "Preserve Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU behavior oracles with identical generated IDs/text/backend/kernel/dense-hook/fallback=false before any default-runtime or timing claim.",
    },
];

pub fn dense_q8_sidecar_fused_consumer_boundary() -> DenseQ8SidecarFusedConsumerBoundary {
    DenseQ8SidecarFusedConsumerBoundary {
        role: "attention.q_proj.fused_output_consumer",
        status: "blocked_by_downstream_candle_tensor_consumers",
        reason: "The packed-Q8 inner matvec helpers can fill caller-provided output slices, but the exact sidecar tensor currently feeds Candle Tensor consumers: q_proj output is reshaped, transposed, optionally q_norm-normalized, RoPE-transformed, traced, and recorded before attention scores. Avoiding the returned Candle Tensor would require a typed fused projection consumer covering those downstream semantics, not only a reusable matvec output buffer.",
        exact_tensor_name: SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR,
        exact_tensor_role: "AttentionQ",
        sidecar_inner_matvec_accepts_output_slice: true,
        can_avoid_returned_candle_tensor_for_current_consumer: false,
        downstream_consumers_require_tensor_semantics: true,
        exact_blocking_ops: DENSE_Q8_SIDECAR_FUSED_CONSUMER_EXACT_BLOCKING_OPS,
        required_missing_api: DENSE_Q8_SIDECAR_FUSED_CONSUMER_REQUIRED_API,
        appliance_oracle_required_before_claim: true,
    }
}

pub fn dense_q8_sidecar_fused_q_projection_consumer_contract()
-> DenseQ8SidecarFusedQProjectionConsumerContract {
    DenseQ8SidecarFusedQProjectionConsumerContract {
        role: "attention.q_proj.typed_fused_consumer_contract",
        status: "contract_defined_runtime_disabled",
        source_boundary_status: "blocked_by_downstream_candle_tensor_consumers",
        exact_tensor_name: SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR,
        exact_tensor_role: "AttentionQ",
        shape: DenseQ8SidecarFusedQProjectionShapeContract {
            input_rank: 3,
            projected_rank: 3,
            attention_heads_rank: 4,
            projected_shape: "[batch, seq, n_heads * head_dim]",
            attention_heads_shape: "[batch, n_heads, seq, head_dim]",
            head_handoff_shape: "AttentionHeads.q",
        },
        stages: DENSE_Q8_SIDECAR_FUSED_Q_PROJECTION_STAGES,
        receipt: DenseQ8SidecarFusedQProjectionReceiptContract {
            required_before_runtime_execution: true,
            required_before_allocation_claim: true,
            required_before_speedup_claim: true,
            required_fields: DENSE_Q8_SIDECAR_FUSED_Q_PROJECTION_RECEIPT_FIELDS,
            gate: "repeated_qwen3_q8_before_after_receipts_with_identical_generated_ids_text_backend_dense_hook_and_fallback_false",
        },
        owns_packed_q8_matvec_output_slice: true,
        intermediate_returned_candle_tensor_allowed: false,
        runtime_execution_enabled: false,
        allocation_reduction_claim: false,
        speedup_claim: false,
        default_runtime_changed: false,
        required_missing_implementation: "behavior-preserving fused Q projection consumer implementation plus before/after Qwen3 Q8_0 receipts",
    }
}

pub fn dense_q8_sidecar_typed_fused_q_projection_implementation_gate()
-> DenseQ8SidecarTypedFusedQProjectionImplementationGate {
    let contract = dense_q8_sidecar_fused_q_projection_consumer_contract();
    DenseQ8SidecarTypedFusedQProjectionImplementationGate {
        role: "attention.q_proj.typed_fused_consumer_implementation_gate",
        status: "blocked_runtime_disabled",
        source_contract_status: contract.status,
        exact_tensor_name: contract.exact_tensor_name,
        exact_tensor_role: contract.exact_tensor_role,
        attempted_runtime_implementation: false,
        can_own_packed_q8_matvec_output_slice: true,
        can_preserve_downstream_tensor_semantics_without_intermediate_tensor: false,
        runtime_execution_enabled: false,
        default_runtime_changed: false,
        allocation_reduction_claim: false,
        speedup_claim: false,
        blockers: DENSE_Q8_SIDECAR_TYPED_FUSED_Q_PROJECTION_IMPLEMENTATION_BLOCKERS,
        receipt_gate: contract.receipt,
        next_required_slice: "typed attention-head buffer/view plus q_norm/RoPE/trace/score-consumer API, followed by behavior-preserving Qwen3 Q8_0 before/after receipts",
    }
}

pub fn dense_q8_sidecar_typed_attention_head_view_gate() -> DenseQ8SidecarTypedAttentionHeadViewGate
{
    let source = dense_q8_sidecar_typed_fused_q_projection_implementation_gate();
    DenseQ8SidecarTypedAttentionHeadViewGate {
        role: "attention.q_proj.typed_attention_head_view_gate",
        status: "contract_defined_runtime_disabled",
        source_gate_status: source.status,
        exact_tensor_name: source.exact_tensor_name,
        exact_tensor_role: source.exact_tensor_role,
        layout: DenseQ8SidecarTypedAttentionHeadViewLayoutContract {
            projected_rank: 3,
            attention_heads_rank: 4,
            projected_shape: "[batch, seq, n_heads * head_dim]",
            attention_heads_shape: "[batch, n_heads, seq, head_dim]",
            logical_storage_order: "projection-row-major-with-head-major-logical-view",
            head_stride_contract: "head index splits the final projection dimension before logical seq/head_dim addressing",
        },
        stages: DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_VIEW_STAGES,
        blockers: DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_VIEW_BLOCKERS,
        receipt_gate: source.receipt_gate,
        can_represent_q_heads_without_candle_tensor: true,
        can_feed_current_attention_score_api_without_materialization: false,
        selected_materialization_point: None,
        runtime_execution_enabled: false,
        default_runtime_changed: false,
        allocation_reduction_claim: false,
        speedup_claim: false,
        next_required_slice: "typed q_norm/RoPE or explicit single-materialization score-handoff boundary plus repeated behavior-preserving Qwen3 Q8_0 receipts",
    }
}

pub fn dense_q8_sidecar_typed_attention_head_consumer_gate()
-> DenseQ8SidecarTypedAttentionHeadConsumerGate {
    let source = dense_q8_sidecar_typed_attention_head_view_gate();
    DenseQ8SidecarTypedAttentionHeadConsumerGate {
        role: "attention.q_proj.typed_attention_head_consumer_gate",
        status: "blocked_runtime_disabled",
        source_gate_status: source.status,
        exact_tensor_name: source.exact_tensor_name,
        exact_tensor_role: source.exact_tensor_role,
        stages: DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_CONSUMER_STAGES,
        blockers: DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_CONSUMER_BLOCKERS,
        receipt_gate: source.receipt_gate,
        can_consume_projection_output_slice: true,
        can_apply_logical_head_view_without_candle_tensor: true,
        can_apply_q_norm_without_candle_tensor: false,
        can_apply_rope_without_candle_tensor: false,
        can_feed_attention_scores_without_candle_tensor: false,
        first_blocking_stage: "q_norm_consumer",
        accepted_single_materialization_point: None,
        candidate_materialization_points:
            DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_CONSUMER_CANDIDATE_MATERIALIZATION_POINTS,
        runtime_execution_enabled: false,
        default_runtime_changed: false,
        allocation_reduction_claim: false,
        speedup_claim: false,
        next_required_slice: "typed q_norm/RoPE implementation or a proven single-materialization boundary with strict Qwen3/Qwen2.5 CPU receipts",
    }
}

pub fn dense_q8_sidecar_typed_q_norm_rope_consumer_gate() -> DenseQ8SidecarTypedQNormRopeConsumerGate
{
    let source = dense_q8_sidecar_typed_attention_head_consumer_gate();
    DenseQ8SidecarTypedQNormRopeConsumerGate {
        role: "attention.q_proj.typed_q_norm_rope_consumer_gate",
        status: "blocked_runtime_disabled",
        source_gate_status: source.status,
        exact_tensor_name: source.exact_tensor_name,
        exact_tensor_role: source.exact_tensor_role,
        stages: DENSE_Q8_SIDECAR_TYPED_Q_NORM_ROPE_CONSUMER_STAGES,
        blockers: DENSE_Q8_SIDECAR_TYPED_Q_NORM_ROPE_CONSUMER_BLOCKERS,
        receipt_gate: source.receipt_gate,
        can_consume_logical_q_head_view: true,
        can_apply_typed_q_norm_without_candle_tensor: false,
        can_apply_typed_rope_without_candle_tensor: false,
        can_preserve_trace_identity_without_tensor_mapping: false,
        can_feed_attention_scores_without_candle_tensor: false,
        first_blocking_stage: "typed_q_norm_consumer",
        accepted_single_materialization_point: None,
        candidate_materialization_points:
            DENSE_Q8_SIDECAR_TYPED_ATTENTION_HEAD_CONSUMER_CANDIDATE_MATERIALIZATION_POINTS,
        runtime_execution_enabled: false,
        default_runtime_changed: false,
        packed_q8_sidecar_default_enabled: false,
        allocation_reduction_claim: false,
        speedup_claim: false,
        next_required_slice: "typed q_norm kernel, typed RoPE kernel, or one proven materialization boundary before attention scores with strict Qwen3/Qwen2.5 CPU receipts",
    }
}

pub fn dense_q8_sidecar_q_norm_materialization_boundary_gate()
-> DenseQ8SidecarQNormMaterializationBoundaryGate {
    let source = dense_q8_sidecar_typed_q_norm_rope_consumer_gate();
    DenseQ8SidecarQNormMaterializationBoundaryGate {
        role: "attention.q_proj.q_norm_input_materialization_boundary_gate",
        status: "boundary_selected_runtime_disabled",
        source_gate_status: source.status,
        exact_tensor_name: source.exact_tensor_name,
        exact_tensor_role: source.exact_tensor_role,
        accepted_single_materialization_point: "q_norm_input_candle_tensor_boundary",
        rejected_materialization_points: DENSE_Q8_SIDECAR_Q_NORM_MATERIALIZATION_REJECTED_POINTS,
        materializes_before_stage: "typed_q_norm_consumer",
        preserved_candle_consumers: DENSE_Q8_SIDECAR_Q_NORM_MATERIALIZATION_PRESERVED_CONSUMERS,
        receipt_gate: source.receipt_gate,
        runtime_execution_enabled: false,
        default_runtime_changed: false,
        packed_q8_sidecar_default_enabled: false,
        allocation_reduction_claim: false,
        speedup_claim: false,
        sustained_throughput_claim: false,
        q4_q5_runtime_claim: false,
        qwen3_q8_before_after_receipts_required: true,
        qwen25_q8_before_after_receipts_required: true,
        next_required_slice: "prove the q_norm-input materialization boundary with strict before/after Qwen3 Q8_0 and Qwen2.5 Q8_0 receipts before enabling any runtime-adjacent packed-Q8 sidecar consumer or claiming allocation/timing improvement",
    }
}

pub fn dense_q8_sidecar_q_norm_input_receipt_comparator_gate()
-> DenseQ8SidecarQNormInputReceiptComparatorGate {
    DenseQ8SidecarQNormInputReceiptComparatorGate {
        role: "attention.q_proj.q_norm_input_receipt_identity_comparator",
        status: "comparator_contract_defined_proof_blocked_on_runtime_hook_and_receipts",
        selected_materialization_boundary: "q_norm_input_candle_tensor_boundary",
        required_identity_fields: DENSE_Q8_SIDECAR_Q_NORM_INPUT_RECEIPT_COMPARATOR_FIELDS,
        fail_closed_on_missing_field: true,
        fail_closed_on_mismatch: true,
        fail_closed_on_fallback: true,
        compares_qwen3_q8: true,
        compares_qwen25_q8: true,
        runtime_execution_enabled: false,
        allocation_reduction_claim: false,
        speedup_claim: false,
        sustained_throughput_claim: false,
        q4_q5_runtime_claim: false,
        server_or_accelerator_claim: false,
        qwen35_claim: false,
        bitnet_qk256_claim: false,
        remaining_blockers: DENSE_Q8_SIDECAR_Q_NORM_INPUT_RECEIPT_COMPARATOR_REMAINING_BLOCKERS,
    }
}

pub fn dense_q8_sidecar_q_norm_input_runtime_hook_gate() -> DenseQ8SidecarQNormInputRuntimeHookGate
{
    let source = dense_q8_sidecar_q_norm_materialization_boundary_gate();
    let comparator = dense_q8_sidecar_q_norm_input_receipt_comparator_gate();
    DenseQ8SidecarQNormInputRuntimeHookGate {
        role: "attention.q_proj.q_norm_input_runtime_disabled_hook_gate",
        status: "runtime_disabled_hook_and_tensor_identity_surface_defined",
        source_comparator_status: comparator.status,
        exact_tensor_name: source.exact_tensor_name,
        exact_tensor_role: source.exact_tensor_role,
        selected_materialization_boundary: source.accepted_single_materialization_point,
        hook_identity: "layers.0.attention.q_proj.weight:q_norm_input_candle_tensor_boundary:runtime_disabled",
        hook_scope: "exact Qwen3 Q8_0 layers.0.attention.q_proj.weight packed-Q8 sidecar path only",
        hook_runtime_enabled: false,
        hook_default_enabled: false,
        tensor_identity_surface_defined: true,
        tensor_identity_fields: DENSE_Q8_SIDECAR_Q_NORM_INPUT_RUNTIME_HOOK_TENSOR_IDENTITY_FIELDS,
        after_receipt_field: "dense_q8_hook.q_norm_input_tensor_identity",
        preserves_eager_f32_default: true,
        packed_q8_sidecar_default_enabled: false,
        required_receipts: DENSE_Q8_SIDECAR_Q_NORM_INPUT_PROOF_REQUIRED_RECEIPTS,
        remaining_blockers: DENSE_Q8_SIDECAR_Q_NORM_INPUT_RUNTIME_HOOK_REMAINING_BLOCKERS,
        allocation_reduction_claim: false,
        speedup_claim: false,
        sustained_throughput_claim: false,
        q4_q5_runtime_claim: false,
        server_or_accelerator_claim: false,
        qwen35_claim: false,
        bitnet_qk256_claim: false,
    }
}

pub fn compare_dense_q8_sidecar_q_norm_input_receipts(
    before: &DenseQ8SidecarQNormInputReceiptIdentity,
    after: &DenseQ8SidecarQNormInputReceiptIdentity,
) -> DenseQ8SidecarQNormInputReceiptComparison {
    let mut failed_fields = Vec::new();

    if before.model_id.is_empty() || after.model_id.is_empty() || before.model_id != after.model_id
    {
        failed_fields.push("model_id");
    }
    if before.model_sha256.is_empty()
        || after.model_sha256.is_empty()
        || before.model_sha256 != after.model_sha256
    {
        failed_fields.push("model_sha256");
    }
    if before.tokenizer_source != "gguf_metadata"
        || after.tokenizer_source != "gguf_metadata"
        || before.tokenizer_source != after.tokenizer_source
    {
        failed_fields.push("tokenizer_source");
    }
    if !before.tokenizer_strict || !after.tokenizer_strict {
        failed_fields.push("tokenizer_strict");
    }
    if before.prompt_ids_digest.is_empty()
        || after.prompt_ids_digest.is_empty()
        || before.prompt_ids_digest != after.prompt_ids_digest
    {
        failed_fields.push("prompt_ids_digest");
    }
    if before.generated_ids_digest.is_empty()
        || after.generated_ids_digest.is_empty()
        || before.generated_ids_digest != after.generated_ids_digest
    {
        failed_fields.push("generated_ids_digest");
    }
    if before.decoded_text_digest.is_empty()
        || after.decoded_text_digest.is_empty()
        || before.decoded_text_digest != after.decoded_text_digest
    {
        failed_fields.push("decoded_text_digest");
    }
    if before.selected_backend != "cpu-rust"
        || after.selected_backend != "cpu-rust"
        || before.selected_backend != after.selected_backend
    {
        failed_fields.push("selected_backend");
    }
    if before.selected_kernel_identity.is_empty()
        || after.selected_kernel_identity.is_empty()
        || before.selected_kernel_identity != after.selected_kernel_identity
    {
        failed_fields.push("selected_kernel_identity");
    }
    if before.dense_hook_identity.is_empty()
        || after.dense_hook_identity.is_empty()
        || before.dense_hook_identity != after.dense_hook_identity
    {
        failed_fields.push("dense_hook_identity");
    }
    if before.q_norm_input_boundary != "q_norm_input_candle_tensor_boundary"
        || after.q_norm_input_boundary != "q_norm_input_candle_tensor_boundary"
        || before.q_norm_input_boundary != after.q_norm_input_boundary
    {
        failed_fields.push("q_norm_input_boundary");
    }
    if before.q_norm_input_tensor_identity.is_empty()
        || after.q_norm_input_tensor_identity.is_empty()
        || before.q_norm_input_tensor_identity != after.q_norm_input_tensor_identity
    {
        failed_fields.push("q_norm_input_tensor_identity");
    }
    if before.fallback_used || after.fallback_used {
        failed_fields.push("fallback_used");
    }

    failed_fields.sort_unstable();
    failed_fields.dedup();

    DenseQ8SidecarQNormInputReceiptComparison { passed: failed_fields.is_empty(), failed_fields }
}

pub fn dense_q8_sidecar_q_norm_input_proof_gate() -> DenseQ8SidecarQNormInputProofGate {
    let source = dense_q8_sidecar_q_norm_materialization_boundary_gate();
    let runtime_hook = dense_q8_sidecar_q_norm_input_runtime_hook_gate();
    let comparator = dense_q8_sidecar_q_norm_input_receipt_comparator_gate();
    DenseQ8SidecarQNormInputProofGate {
        role: "attention.q_proj.q_norm_input_materialization_proof_gate",
        status: "blocked_before_after_receipts_missing_runtime_hook_defined",
        source_boundary_status: source.status,
        exact_tensor_name: source.exact_tensor_name,
        exact_tensor_role: source.exact_tensor_role,
        selected_materialization_boundary: source.accepted_single_materialization_point,
        required_receipts: DENSE_Q8_SIDECAR_Q_NORM_INPUT_PROOF_REQUIRED_RECEIPTS,
        blockers: DENSE_Q8_SIDECAR_Q_NORM_INPUT_PROOF_BLOCKERS,
        proof_ready: false,
        missing_runtime_hook: !runtime_hook.tensor_identity_surface_defined,
        missing_receipt_field: false,
        missing_comparator: false,
        comparator_contract_defined: comparator.fail_closed_on_missing_field
            && comparator.fail_closed_on_mismatch
            && comparator.fail_closed_on_fallback,
        tensor_identity_unrecorded: !runtime_hook.tensor_identity_surface_defined,
        accumulator_order_unproven: true,
        artifact_gap: true,
        runtime_execution_enabled: false,
        default_runtime_changed: false,
        packed_q8_sidecar_default_enabled: false,
        allocation_reduction_claim: false,
        speedup_claim: false,
        sustained_throughput_claim: false,
        q4_q5_runtime_claim: false,
        server_or_accelerator_claim: false,
        qwen35_claim: false,
        bitnet_qk256_claim: false,
        next_required_slice: "collect Qwen3 Q8_0 and Qwen2.5 Q8_0 before/after strict CPU receipts that include q_norm_input tensor identity and pass the fail-closed comparator before proving the selected boundary",
    }
}

impl TransformerForwardWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn model_forward_calls(&self) -> usize {
        self.model_forward_calls
    }

    pub fn block_forward_calls(&self) -> usize {
        self.block_forward_calls
    }

    pub fn feed_forward_calls(&self) -> usize {
        self.feed_forward_calls
    }

    pub fn last_input_shape(&self) -> &[usize] {
        &self.last_input_shape
    }

    pub fn last_output_shape(&self) -> &[usize] {
        &self.last_output_shape
    }

    pub fn tensor_reuse_enabled(&self) -> bool {
        self.tensor_reuse_enabled
    }

    pub fn workspace_owned_output_count(&self) -> usize {
        self.workspace_owned_output_count
    }

    pub fn model_workspace_owned_output_count(&self) -> usize {
        self.model_workspace_owned_output_count
    }

    pub fn down_proj_output_storage_attempts(&self) -> usize {
        self.down_proj_output_storage_attempts
    }

    pub fn model_output_storage_attempts(&self) -> usize {
        self.model_output_storage_attempts
    }

    pub fn final_norm_output_storage_attempts(&self) -> usize {
        self.final_norm_output_storage_attempts
    }

    pub fn layer_output_storage_attempts(&self) -> usize {
        self.layer_output_storage_attempts
    }

    pub fn model_output_surface(&self) -> Option<&TransformerWorkspaceOutputSurface> {
        self.model_output_surface.as_ref()
    }

    pub fn model_forward_source_tensors(&self) -> Option<&TransformerModelForwardSourceTensors> {
        self.model_forward_source_tensors.as_ref()
    }

    pub fn final_block_source_tensors(&self) -> Option<&TransformerFinalBlockSourceTensors> {
        self.final_block_source_tensors.as_ref()
    }

    pub fn penultimate_block_source_tensors(&self) -> Option<&TransformerFinalBlockSourceTensors> {
        self.penultimate_block_source_tensors.as_ref()
    }

    pub fn antepenultimate_block_source_tensors(
        &self,
    ) -> Option<&TransformerFinalBlockSourceTensors> {
        self.antepenultimate_block_source_tensors.as_ref()
    }

    pub fn pre_antepenultimate_block_source_tensors(
        &self,
    ) -> Option<&TransformerFinalBlockSourceTensors> {
        self.pre_antepenultimate_block_source_tensors.as_ref()
    }

    pub fn earlier_block_source_tensors(&self) -> Option<&TransformerFinalBlockSourceTensors> {
        self.earlier_block_source_tensors.as_ref()
    }

    pub fn block_source_tensors(&self) -> &[TransformerFinalBlockSourceTensors] {
        &self.block_source_tensors
    }

    pub fn attention_output_source_tensors(&self) -> &[TransformerAttentionOutputSourceTensors] {
        &self.attention_output_source_tensors
    }

    pub fn qkv_projection_source_tensors(&self) -> &[TransformerQkvProjectionSourceTensors] {
        &self.qkv_projection_source_tensors
    }

    pub fn first_output_surface(&self) -> Option<&TransformerWorkspaceOutputSurface> {
        self.feed_forward_output_surface.as_ref()
    }

    pub fn final_norm_output_surface(&self) -> Option<&TransformerWorkspaceOutputStorageBoundary> {
        self.final_norm_output_surface.as_ref()
    }

    pub fn layer_output_surface(&self) -> Option<&TransformerWorkspaceOutputStorageBoundary> {
        self.layer_output_surface.as_ref()
    }

    pub fn reuse_status(&self) -> &'static str {
        if self.tensor_reuse_enabled {
            "typed_transformer_forward_workspace_reuse_enabled"
        } else if self.feed_forward_output_surface.is_some() {
            "dense_linear_output_storage_blocked_by_candle_tensor_ops"
        } else {
            "api_boundary_present_owned_tensor_reuse_not_enabled"
        }
    }

    fn record_model_input(&mut self, tensor: &Tensor) {
        self.model_forward_calls += 1;
        self.last_input_shape = tensor.dims().to_vec();
    }

    fn record_model_output(&mut self, tensor: &Tensor) {
        self.last_output_shape = tensor.dims().to_vec();
        self.model_output_storage_attempts += 1;
        self.model_output_surface = Some(TransformerWorkspaceOutputSurface {
            name: "model.forward.output",
            storage_owner: "TransformerForwardWorkspace",
            status: "model_forward_output_storage_api_surface_present_reuse_blocked_by_candle_tensor_ops",
            reason: "TransformerModel::forward_with_workspace now moves the final Candle Tensor through a TransformerForwardWorkspace-owned model output slot, preserving behavior while keeping reusable caller-filled output storage blocked by Candle tensor operations that still return owned Tensors",
            next_api_hook: "add or adopt final-norm/layer-output caller-output-storage APIs before replacing the final TransformerModel::forward output allocation with reusable workspace-backed storage",
            last_shape: tensor.dims().to_vec(),
            linear_weight_shape: Vec::new(),
            linear_bias_shape: None,
            weight_accessible: false,
            bias_accessible: false,
            can_fill_caller_output_storage: false,
        });
    }

    fn record_model_forward_source_tensors(
        &mut self,
        prior_layer_output: &Tensor,
        final_norm_output: &Tensor,
    ) {
        self.model_forward_source_tensors = Some(TransformerModelForwardSourceTensors {
            prior_layer_output: prior_layer_output.clone(),
            final_norm_output: final_norm_output.clone(),
        });
    }

    fn record_final_block_source_tensors(
        &mut self,
        layer_idx: usize,
        block_input: &Tensor,
        attention_output: &Tensor,
        post_attention_residual: &Tensor,
        feed_forward_output: &Tensor,
        block_output: &Tensor,
    ) {
        self.earlier_block_source_tensors = self.pre_antepenultimate_block_source_tensors.clone();
        self.pre_antepenultimate_block_source_tensors =
            self.antepenultimate_block_source_tensors.clone();
        self.antepenultimate_block_source_tensors = self.penultimate_block_source_tensors.clone();
        self.penultimate_block_source_tensors = self.final_block_source_tensors.clone();
        let source = TransformerFinalBlockSourceTensors {
            layer_idx,
            block_input: block_input.clone(),
            attention_output: attention_output.clone(),
            post_attention_residual: post_attention_residual.clone(),
            feed_forward_output: feed_forward_output.clone(),
            block_output: block_output.clone(),
        };
        self.block_source_tensors.push(source.clone());
        self.final_block_source_tensors = Some(source);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_attention_output_source_tensors(
        &mut self,
        layer_idx: usize,
        attention_input: &Tensor,
        q_projection: &Tensor,
        k_projection: &Tensor,
        v_projection: &Tensor,
        q_heads: &Tensor,
        k_heads: &Tensor,
        v_heads: &Tensor,
        q_norm: &Tensor,
        k_norm: &Tensor,
        q_rope: &Tensor,
        k_rope: &Tensor,
        k_context: &Tensor,
        v_context: &Tensor,
        expanded_k: &Tensor,
        expanded_v: &Tensor,
        scores: &Tensor,
        probabilities: &Tensor,
        value_mix_output_heads: &Tensor,
        output_projection_input: &Tensor,
        sub_layernorm_output: Option<&Tensor>,
        attention_output: &Tensor,
    ) {
        self.attention_output_source_tensors.push(TransformerAttentionOutputSourceTensors {
            layer_idx,
            attention_input: attention_input.clone(),
            q_projection: q_projection.clone(),
            k_projection: k_projection.clone(),
            v_projection: v_projection.clone(),
            q_heads: q_heads.clone(),
            k_heads: k_heads.clone(),
            v_heads: v_heads.clone(),
            q_norm: q_norm.clone(),
            k_norm: k_norm.clone(),
            q_rope: q_rope.clone(),
            k_rope: k_rope.clone(),
            k_context: k_context.clone(),
            v_context: v_context.clone(),
            expanded_k: expanded_k.clone(),
            expanded_v: expanded_v.clone(),
            scores: scores.clone(),
            probabilities: probabilities.clone(),
            value_mix_output_heads: value_mix_output_heads.clone(),
            output_projection_input: output_projection_input.clone(),
            sub_layernorm_output: sub_layernorm_output.cloned(),
            attention_output: attention_output.clone(),
        });
    }

    fn record_qkv_projection_source_tensors(
        &mut self,
        source: TransformerQkvProjectionSourceTensors,
    ) {
        self.qkv_projection_source_tensors.push(source);
    }

    fn record_block_input(&mut self, tensor: &Tensor) {
        self.block_forward_calls += 1;
        self.last_input_shape = tensor.dims().to_vec();
    }

    fn record_block_output(&mut self, tensor: &Tensor) {
        self.last_output_shape = tensor.dims().to_vec();
    }

    fn record_feed_forward_input(&mut self, tensor: &Tensor) {
        self.feed_forward_calls += 1;
        self.last_input_shape = tensor.dims().to_vec();
    }

    fn record_feed_forward_output(&mut self, tensor: &Tensor) {
        self.last_output_shape = tensor.dims().to_vec();
    }

    fn record_down_proj_output_storage_boundary(
        &mut self,
        tensor: &Tensor,
        boundary: TransformerWorkspaceOutputSurface,
    ) {
        self.down_proj_output_storage_attempts += 1;
        self.last_output_shape = tensor.dims().to_vec();
        self.feed_forward_output_surface = Some(boundary);
    }

    fn record_layer_output_storage_boundary(
        &mut self,
        tensor: &Tensor,
        residual_input: &Tensor,
        branch_output: &Tensor,
    ) {
        let boundary =
            LayerOutputStorageApiBoundary::from_candle_residual_add("transformer.block.output");
        self.layer_output_storage_attempts += 1;
        self.last_output_shape = tensor.dims().to_vec();
        self.layer_output_surface = Some(TransformerWorkspaceOutputStorageBoundary {
            name: boundary.role,
            storage_owner: "TransformerForwardWorkspace",
            status: boundary.status,
            reason: boundary.reason,
            next_api_hook: boundary.next_api_hook,
            last_shape: tensor.dims().to_vec(),
            operation_family: "candle_core::Tensor residual_add",
            operation_detail: "residual_add_owned_tensor_output",
            residual_input_shape: Some(residual_input.dims().to_vec()),
            branch_output_shape: Some(branch_output.dims().to_vec()),
            weight_shape: None,
            bias_shape: None,
            epsilon: None,
            input_accessible: true,
            weight_accessible: false,
            bias_accessible: false,
            residual_add_involved: boundary.residual_add_involved,
            caller_output_helper_status: "layer_output_storage_helper_blocked_by_owned_candle_residual_add_output",
            can_fill_caller_output_storage: boundary.can_fill_caller_output_storage,
            exact_blocking_ops: Some(boundary.exact_blocking_ops),
            public_api_return_type: Some(boundary.public_api_return_type),
            required_missing_api: Some(boundary.required_missing_api),
            public_api_accepts_output_storage: Some(boundary.public_api_accepts_output_storage),
            backend_internal_in_place_api_exposed: Some(
                boundary.backend_internal_in_place_api_exposed,
            ),
        });
    }

    fn record_final_norm_output_storage_boundary(&mut self, tensor: &Tensor, norm: &LayerNorm) {
        let boundary =
            NormOutputStorageApiBoundary::from_candle_layer_norm("model.final_norm.output", norm);
        self.final_norm_output_storage_attempts += 1;
        self.last_output_shape = tensor.dims().to_vec();
        self.final_norm_output_surface = Some(TransformerWorkspaceOutputStorageBoundary {
            name: boundary.role,
            storage_owner: "TransformerForwardWorkspace",
            status: boundary.status,
            reason: boundary.reason,
            next_api_hook: boundary.next_api_hook,
            last_shape: tensor.dims().to_vec(),
            operation_family: if boundary.remove_mean {
                "candle_nn::LayerNorm::forward"
            } else {
                "candle_nn::RmsNorm::forward"
            },
            operation_detail: boundary.norm_kind,
            residual_input_shape: None,
            branch_output_shape: None,
            weight_shape: Some(boundary.weight_shape),
            bias_shape: boundary.bias_shape,
            epsilon: Some(boundary.epsilon),
            input_accessible: boundary.input_accessible,
            weight_accessible: boundary.weight_accessible,
            bias_accessible: boundary.bias_accessible,
            residual_add_involved: false,
            caller_output_helper_status: boundary.caller_output_helper_status,
            can_fill_caller_output_storage: boundary.can_fill_caller_output_storage,
            exact_blocking_ops: None,
            public_api_return_type: None,
            required_missing_api: None,
            public_api_accepts_output_storage: None,
            backend_internal_in_place_api_exposed: None,
        });
    }

    fn store_feed_forward_output(&mut self, tensor: Tensor) {
        let last_shape = tensor.dims().to_vec();
        self.last_output_shape = last_shape.clone();
        if let Some(surface) = self.feed_forward_output_surface.as_mut() {
            surface.last_shape = last_shape;
        }
        self.workspace_owned_output_count += 1;
        self.feed_forward_output_slot = Some(tensor);
    }

    fn take_feed_forward_output(&mut self) -> Result<Tensor> {
        self.feed_forward_output_slot.take().ok_or_else(|| {
            BitNetError::Validation(
                "TransformerForwardWorkspace feed-forward output slot must be populated before take"
                    .to_string(),
            )
        })
    }

    fn store_model_output(&mut self, tensor: Tensor) {
        let last_shape = tensor.dims().to_vec();
        self.last_output_shape = last_shape.clone();
        if let Some(surface) = self.model_output_surface.as_mut() {
            surface.last_shape = last_shape;
        }
        self.model_workspace_owned_output_count += 1;
        self.model_output_slot = Some(tensor);
    }

    fn take_model_output(&mut self) -> Result<Tensor> {
        self.model_output_slot.take().ok_or_else(|| {
            BitNetError::Validation(
                "TransformerForwardWorkspace model output slot must be populated before take"
                    .to_string(),
            )
        })
    }
}

impl TransformerBlock {
    pub fn new(config: &BitNetConfig, vb: VarBuilder, layer_idx: usize) -> Result<Self> {
        let block_start = Instant::now();
        let trace_model_init = qwen_trace_events_enabled();
        let device = vb.device().clone();
        let hidden_size = config.model.hidden_size;
        // PATCH 1: Use RMSNorm epsilon from config header for ALL norms (per-layer + final)
        let eps = eps_from_config(config);

        tracing::debug!("TransformerBlock using RMSNorm eps={} (from header)", eps);

        qwen_trace_model_init_event(trace_model_init, "model_init.block_start", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\",\"hidden_size\":{},\"eps\":{}",
                qwen_trace_elapsed_ms(block_start),
                layer_idx,
                qwen_trace_device_kind(&device),
                hidden_size,
                eps
            )
        });
        qwen_trace_model_init_event(trace_model_init, "model_init.block_attention_start", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\"",
                qwen_trace_elapsed_ms(block_start),
                layer_idx,
                qwen_trace_device_kind(&device)
            )
        });
        let attention = MultiHeadAttention::new(config, vb.pp("attention"), layer_idx)?;
        qwen_trace_model_init_event(trace_model_init, "model_init.block_attention_finish", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\"",
                qwen_trace_elapsed_ms(block_start),
                layer_idx,
                qwen_trace_device_kind(&device)
            )
        });
        qwen_trace_model_init_event(
            trace_model_init,
            "model_init.block_feed_forward_start",
            || {
                format!(
                    "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(block_start),
                    layer_idx,
                    qwen_trace_device_kind(&device)
                )
            },
        );
        let feed_forward = FeedForward::new(config, vb.pp("feed_forward"), layer_idx)?;
        qwen_trace_model_init_event(
            trace_model_init,
            "model_init.block_feed_forward_finish",
            || {
                format!(
                    "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(block_start),
                    layer_idx,
                    qwen_trace_device_kind(&device)
                )
            },
        );
        qwen_trace_model_init_event(
            trace_model_init,
            "model_init.block_attention_norm_start",
            || {
                format!(
                    "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(block_start),
                    layer_idx,
                    qwen_trace_device_kind(&device)
                )
            },
        );
        let attention_norm = norm_with_optional_bias(
            config.model.norm_type,
            hidden_size,
            eps,
            vb.pp("attention_norm"),
        )?;
        qwen_trace_model_init_event(
            trace_model_init,
            "model_init.block_attention_norm_finish",
            || {
                format!(
                    "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(block_start),
                    layer_idx,
                    qwen_trace_device_kind(&device)
                )
            },
        );
        qwen_trace_model_init_event(trace_model_init, "model_init.block_ffn_norm_start", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\"",
                qwen_trace_elapsed_ms(block_start),
                layer_idx,
                qwen_trace_device_kind(&device)
            )
        });
        let ffn_norm = norm_with_optional_bias(
            config.model.norm_type,
            hidden_size,
            eps,
            vb.pp("post_attention_layernorm"),
        )?;
        qwen_trace_model_init_event(trace_model_init, "model_init.block_ffn_norm_finish", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\"",
                qwen_trace_elapsed_ms(block_start),
                layer_idx,
                qwen_trace_device_kind(&device)
            )
        });
        qwen_trace_model_init_event(trace_model_init, "model_init.block_finish", || {
            format!(
                "\"elapsed_ms\":{},\"layer\":{},\"device\":\"{}\"",
                qwen_trace_elapsed_ms(block_start),
                layer_idx,
                qwen_trace_device_kind(&device)
            )
        });

        Ok(Self { attention, feed_forward, attention_norm, ffn_norm })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
    ) -> Result<Tensor> {
        self.forward_impl(x, kv_cache, raw_tensors, dense_linear_hooks, None, None)
    }

    pub fn forward_with_workspace(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        workspace: &mut TransformerForwardWorkspace,
    ) -> Result<Tensor> {
        workspace.record_block_input(x);
        let output =
            self.forward_impl(x, kv_cache, raw_tensors, dense_linear_hooks, None, Some(workspace))?;
        workspace.record_block_output(&output);
        Ok(output)
    }

    pub fn forward_with_no_bias_callsite_descriptor(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        prompt_bound_no_bias_descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    ) -> Result<Tensor> {
        self.forward_impl(
            x,
            kv_cache,
            raw_tensors,
            dense_linear_hooks,
            Some(prompt_bound_no_bias_descriptor),
            None,
        )
    }

    pub fn forward_with_workspace_and_no_bias_callsite_descriptor(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        workspace: &mut TransformerForwardWorkspace,
        prompt_bound_no_bias_descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    ) -> Result<Tensor> {
        workspace.record_block_input(x);
        let output = self.forward_impl(
            x,
            kv_cache,
            raw_tensors,
            dense_linear_hooks,
            Some(prompt_bound_no_bias_descriptor),
            Some(workspace),
        )?;
        workspace.record_block_output(&output);
        Ok(output)
    }

    fn forward_impl(
        &self,
        x: &Tensor,
        kv_cache: Option<&mut LayerKVCache>,
        raw_tensors: &HashMap<String, Tensor>,
        dense_linear_hooks: &DenseLinearRuntimeHookRegistry,
        prompt_bound_no_bias_descriptor: Option<
            &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
        >,
        mut workspace: Option<&mut TransformerForwardWorkspace>,
    ) -> Result<Tensor> {
        let trace_forward = qwen_trace_events_enabled();
        let block_start = Instant::now();
        qwen_trace_runtime_event(trace_forward, "block.forward_start", || {
            format!(
                "\"layer\":{},\"dims\":[{}],\"device\":\"{}\"",
                self.attention.layer_idx,
                qwen_trace_dims_json(x.dims()),
                qwen_trace_device_kind(x.device())
            )
        });
        // Debug input activation norms
        if debug_attn_enabled() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] input: {norm:.6e}");
        }

        // Pre-norm attention
        let block_input_for_source = workspace.as_ref().map(|_| x.clone());
        let residual = x;

        // RMSNorm diagnostics (Layer 0 only) - attention norm
        // User's diagnostic: log mean(x^2) and rms = sqrt(mean(x^2) + eps) before/after norm
        if debug_rmsnorm_enabled() {
            static ATTN_NORM_LOGGED: std::sync::Once = std::sync::Once::new();
            ATTN_NORM_LOGGED.call_once(|| {
                if let Ok(mean_sq) =
                    x.sqr().and_then(|s| s.mean_all()).and_then(|m| m.to_scalar::<f32>())
                {
                    // Note: RMSNorm formula is: rms = sqrt(mean(x^2) + eps), y = (x / rms) * weight
                    // The actual eps value is in the LayerNorm (handled by candle)
                    let rms_approx = mean_sq.sqrt(); // Approximate (actual includes eps inside sqrt)
                    tracing::info!(
                        "RMSNorm (attn, layer 0) - input mean(x^2): {:.6e}, approx_rms: {:.6e}",
                        mean_sq,
                        rms_approx
                    );
                    if !rms_approx.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (attn) - input has non-finite values!");
                    }
                }
            });
        }

        let attention_norm_start = Instant::now();
        qwen_trace_runtime_event(trace_forward, "block.attention_norm_start", || {
            format!(
                "\"layer\":{},\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\",\"remove_mean\":{},\"bias_present\":{}",
                self.attention.layer_idx,
                qwen_trace_dims_json(x.dims()),
                x.dtype(),
                qwen_trace_device_kind(x.device()),
                self.attention_norm.remove_mean(),
                self.attention_norm.bias().is_some()
            )
        });
        let x = if trace_forward
            && !self.attention_norm.remove_mean()
            && self.attention_norm.bias().is_none()
        {
            let x_dtype = x.dtype();
            let internal_dtype = match x_dtype {
                DType::F16 | DType::BF16 => DType::F32,
                dtype => dtype,
            };
            let hidden_size = x.dim(D::Minus1)?;

            qwen_trace_runtime_event(trace_forward, "block.attention_norm_manual_start", || {
                format!(
                    "\"layer\":{},\"elapsed_ms\":{},\"path\":\"rms_norm_manual_trace\",\"hidden_size\":{},\"input_dtype\":\"{:?}\",\"internal_dtype\":\"{:?}\",\"eps\":{},\"weight_dims\":[{}],\"weight_device\":\"{}\"",
                    self.attention.layer_idx,
                    qwen_trace_elapsed_ms(attention_norm_start),
                    hidden_size,
                    x_dtype,
                    internal_dtype,
                    qwen_trace_number(self.attention_norm.eps()),
                    qwen_trace_dims_json(self.attention_norm.weight().dims()),
                    qwen_trace_device_kind(self.attention_norm.weight().device())
                )
            });

            let to_dtype_start = Instant::now();
            qwen_trace_runtime_event(trace_forward, "block.attention_norm_to_dtype_start", || {
                format!(
                    "\"layer\":{},\"elapsed_ms\":{},\"from\":\"{:?}\",\"to\":\"{:?}\"",
                    self.attention.layer_idx,
                    qwen_trace_elapsed_ms(attention_norm_start),
                    x_dtype,
                    internal_dtype
                )
            });
            let norm_input = x.to_dtype(internal_dtype)?;
            qwen_trace_runtime_event(trace_forward, "block.attention_norm_to_dtype_finish", || {
                format!(
                    "\"layer\":{},\"elapsed_ms\":{},\"op_ms\":{},\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                    self.attention.layer_idx,
                    qwen_trace_elapsed_ms(attention_norm_start),
                    qwen_trace_elapsed_ms(to_dtype_start),
                    qwen_trace_dims_json(norm_input.dims()),
                    norm_input.dtype(),
                    qwen_trace_device_kind(norm_input.device())
                )
            });

            let fused_start = Instant::now();
            qwen_trace_runtime_event(trace_forward, "block.attention_norm_fused_rms_start", || {
                format!(
                    "\"layer\":{},\"elapsed_ms\":{},\"path\":\"candle_ops_rms_norm\",\"hidden_size\":{},\"input_dims\":[{}],\"input_dtype\":\"{:?}\",\"weight_dims\":[{}],\"weight_dtype\":\"{:?}\",\"eps\":{},\"device\":\"{}\"",
                    self.attention.layer_idx,
                    qwen_trace_elapsed_ms(attention_norm_start),
                    hidden_size,
                    qwen_trace_dims_json(norm_input.dims()),
                    norm_input.dtype(),
                    qwen_trace_dims_json(self.attention_norm.weight().dims()),
                    self.attention_norm.weight().dtype(),
                    qwen_trace_number(self.attention_norm.eps()),
                    qwen_trace_device_kind(norm_input.device())
                )
            });
            let output = qwen_trace_rms_norm_fused(
                &norm_input,
                x_dtype,
                self.attention_norm.weight(),
                self.attention_norm.eps(),
            )?;
            qwen_trace_runtime_event(
                trace_forward,
                "block.attention_norm_fused_rms_finish",
                || {
                    format!(
                        "\"layer\":{},\"elapsed_ms\":{},\"op_ms\":{},\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                        self.attention.layer_idx,
                        qwen_trace_elapsed_ms(attention_norm_start),
                        qwen_trace_elapsed_ms(fused_start),
                        qwen_trace_dims_json(output.dims()),
                        output.dtype(),
                        qwen_trace_device_kind(output.device())
                    )
                },
            );
            output
        } else {
            qwen_trace_runtime_event(
                trace_forward,
                "block.attention_norm_forward_call_start",
                || {
                    format!(
                        "\"layer\":{},\"elapsed_ms\":{},\"path\":\"candle_layer_norm\"",
                        self.attention.layer_idx,
                        qwen_trace_elapsed_ms(attention_norm_start)
                    )
                },
            );
            self.attention_norm.forward(x)?
        };
        qwen_trace_runtime_event(trace_forward, "block.attention_norm_finish", || {
            format!(
                "\"layer\":{},\"norm_ms\":{},\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                self.attention.layer_idx,
                qwen_trace_elapsed_ms(attention_norm_start),
                qwen_trace_dims_json(x.dims()),
                x.dtype(),
                qwen_trace_device_kind(x.device())
            )
        });
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.attention_norm", Some(self.attention.layer_idx), &x)?;
        }

        // Probe A2: LayerNorm gamma RMS + LN output RMS (layer 0, step 0 only)
        if trace_rms_enabled() && self.attention.layer_idx == 0 {
            static LN0_LOGGED: std::sync::Once = std::sync::Once::new();
            LN0_LOGGED.call_once(|| {
                let _ = (|| -> candle_core::Result<()> {
                    // Get gamma (weight) from LayerNorm
                    let gamma_vec = self.attention_norm.weight().to_vec1::<f32>()?;
                    let g_rms = (gamma_vec.iter().map(|x| x * x).sum::<f32>()
                        / gamma_vec.len().max(1) as f32)
                        .sqrt();

                    // Get LN output RMS
                    let ln_vec = x.flatten_all()?.to_vec1::<f32>()?;
                    let ln_rms = (ln_vec.iter().map(|x| x * x).sum::<f32>()
                        / ln_vec.len().max(1) as f32)
                        .sqrt();
                    eprintln!("trace: ln0_gamma_rms={:.6} ln0_out_rms={:.6}", g_rms, ln_rms);
                    Ok(())
                })();
            });
        }

        // Tracepoint 2: Attention norm output (layer-specific)
        #[cfg(feature = "trace")]
        {
            let trace_name = format!("t0/blk{}/attn_norm", self.attention.layer_idx);
            bitnet_trace::dump_trace(
                &trace_name,
                &x,
                Some(0),
                Some(self.attention.layer_idx as isize),
                Some("attn_norm"),
            )
            .map_err(BitNetError::from)?;
        }

        // Check norm output
        if debug_rmsnorm_enabled() {
            static ATTN_NORM_OUT_LOGGED: std::sync::Once = std::sync::Once::new();
            ATTN_NORM_OUT_LOGGED.call_once(|| {
                if let Ok(norm_out) = x
                    .sqr()
                    .and_then(|s| s.mean_all())
                    .and_then(|m| m.sqrt())
                    .and_then(|r| r.to_scalar::<f32>())
                {
                    tracing::info!("RMSNorm (attn, layer 0) - output L2 norm: {:.6e}", norm_out);
                    if !norm_out.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (attn) - output is non-finite!");
                    }
                }
            });
        }

        let attention_start = Instant::now();
        qwen_trace_runtime_event(trace_forward, "block.attention_start", || {
            format!("\"layer\":{}", self.attention.layer_idx)
        });
        let x = self.attention.forward(
            &x,
            kv_cache,
            raw_tensors,
            dense_linear_hooks,
            workspace.as_deref_mut(),
        )?;
        let attention_output_for_source = workspace.as_ref().map(|_| x.clone());
        qwen_trace_runtime_event(trace_forward, "block.attention_finish", || {
            format!(
                "\"layer\":{},\"attention_ms\":{}",
                self.attention.layer_idx,
                qwen_trace_elapsed_ms(attention_start)
            )
        });
        let x = (x + residual)?;
        let post_attention_residual_for_source = workspace.as_ref().map(|_| x.clone());
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.post_attention_residual", Some(self.attention.layer_idx), &x)?;
        }

        // Debug post-attention activation norms
        if debug_attn_enabled() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] post-attn: {norm:.6e}");
        }

        // Pre-norm FFN
        let residual = &x;

        // RMSNorm diagnostics (Layer 0 only) - FFN norm
        if debug_rmsnorm_enabled() {
            static FFN_NORM_LOGGED: std::sync::Once = std::sync::Once::new();
            FFN_NORM_LOGGED.call_once(|| {
                if let Ok(mean_sq) =
                    x.sqr().and_then(|s| s.mean_all()).and_then(|m| m.to_scalar::<f32>())
                {
                    let rms_approx = mean_sq.sqrt();
                    tracing::info!(
                        "RMSNorm (ffn, layer 0) - input mean(x^2): {:.6e}, approx_rms: {:.6e}",
                        mean_sq,
                        rms_approx
                    );
                    if !rms_approx.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (ffn) - input has non-finite values!");
                    }
                }
            });
        }

        let ffn_norm_start = Instant::now();
        qwen_trace_runtime_event(trace_forward, "block.ffn_norm_start", || {
            format!("\"layer\":{}", self.attention.layer_idx)
        });
        let x = self.ffn_norm.forward(&x)?;
        qwen_trace_runtime_event(trace_forward, "block.ffn_norm_finish", || {
            format!(
                "\"layer\":{},\"norm_ms\":{}",
                self.attention.layer_idx,
                qwen_trace_elapsed_ms(ffn_norm_start)
            )
        });
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.ffn_norm", Some(self.attention.layer_idx), &x)?;
        }

        // Check norm output
        if debug_rmsnorm_enabled() {
            static FFN_NORM_OUT_LOGGED: std::sync::Once = std::sync::Once::new();
            FFN_NORM_OUT_LOGGED.call_once(|| {
                if let Ok(norm_out) = x
                    .sqr()
                    .and_then(|s| s.mean_all())
                    .and_then(|m| m.sqrt())
                    .and_then(|r| r.to_scalar::<f32>())
                {
                    tracing::info!("RMSNorm (ffn, layer 0) - output L2 norm: {:.6e}", norm_out);
                    if !norm_out.is_finite() {
                        tracing::warn!("⚠️  RMSNorm (ffn) - output is non-finite!");
                    }
                }
            });
        }

        let feed_forward_start = Instant::now();
        qwen_trace_runtime_event(trace_forward, "block.feed_forward_start", || {
            format!("\"layer\":{}", self.attention.layer_idx)
        });
        let feed_forward_output = match (workspace.as_mut(), prompt_bound_no_bias_descriptor) {
            (Some(workspace), Some(descriptor)) => {
                self.feed_forward.forward_with_workspace_and_no_bias_callsite_descriptor(
                    &x,
                    raw_tensors,
                    dense_linear_hooks,
                    workspace,
                    descriptor,
                )?
            }
            (Some(workspace), None) => self.feed_forward.forward_with_workspace(
                &x,
                raw_tensors,
                dense_linear_hooks,
                workspace,
            )?,
            (None, Some(descriptor)) => {
                self.feed_forward.forward_with_no_bias_callsite_descriptor(
                    &x,
                    raw_tensors,
                    dense_linear_hooks,
                    descriptor,
                )?
            }
            (None, None) => self.feed_forward.forward(&x, raw_tensors, dense_linear_hooks)?,
        };
        qwen_trace_runtime_event(trace_forward, "block.feed_forward_finish", || {
            format!(
                "\"layer\":{},\"feed_forward_ms\":{}",
                self.attention.layer_idx,
                qwen_trace_elapsed_ms(feed_forward_start)
            )
        });
        let x = (&feed_forward_output + residual)?;
        if let Some(workspace) = workspace.as_mut() {
            workspace.record_layer_output_storage_boundary(&x, residual, &feed_forward_output);
            if let (Some(block_input), Some(attention_output), Some(post_attention_residual)) = (
                block_input_for_source.as_ref(),
                attention_output_for_source.as_ref(),
                post_attention_residual_for_source.as_ref(),
            ) {
                workspace.record_final_block_source_tensors(
                    self.attention.layer_idx,
                    block_input,
                    attention_output,
                    post_attention_residual,
                    &feed_forward_output,
                    &x,
                );
            }
        }
        if qwen_trace_layer_enabled(self.attention.layer_idx) {
            qwen_trace_tensor("block.output", Some(self.attention.layer_idx), &x)?;
        }

        // Debug post-FFN activation norms
        if debug_attn_enabled() {
            let norm = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[norm] post-ffn: {norm:.6e}");
        }

        qwen_trace_runtime_event(trace_forward, "block.forward_finish", || {
            format!(
                "\"layer\":{},\"block_ms\":{},\"dims\":[{}],\"device\":\"{}\"",
                self.attention.layer_idx,
                qwen_trace_elapsed_ms(block_start),
                qwen_trace_dims_json(x.dims()),
                qwen_trace_device_kind(x.device())
            )
        });

        Ok(x)
    }
}

fn eps_from_config(config: &BitNetConfig) -> f64 {
    config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5)
}

/// KV Cache for a single layer
pub struct LayerKVCache {
    pub k: Tensor,
    pub v: Tensor,
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub n_kv_heads: usize, // Store the number of KV heads for validation
}

impl LayerKVCache {
    pub fn new(
        batch_size: usize,
        n_kv_heads: usize, // Changed from n_heads to n_kv_heads
        max_seq_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let k =
            Tensor::zeros(&[batch_size, n_kv_heads, max_seq_len, head_dim], DType::F32, device)?;
        let v =
            Tensor::zeros(&[batch_size, n_kv_heads, max_seq_len, head_dim], DType::F32, device)?;

        Ok(Self { k, v, seq_len: 0, max_seq_len, n_kv_heads })
    }

    /// Append new K/V tensors to the cache
    ///
    /// **Performance note**: The clones on first append (lines 1130-1131) are necessary
    /// because we accept `&Tensor` but need to store owned tensors. Candle's `Tensor::clone()`
    /// is cheap - it only increments the Arc reference count, not a deep data copy.
    /// Subsequent appends use `Tensor::cat` which allocates new tensors regardless.
    ///
    /// To eliminate these clones would require API changes to accept owned tensors,
    /// which would complicate calling code.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        // Expect shapes: k: [B,HKV,T_new,Hd], v: [B,HKV,T_new,Hd] where HKV = n_kv_heads
        let new_seq_len = k_new.dims()[2];

        // Validate that the incoming tensors have the expected number of KV heads
        let k_heads = k_new.dims()[1];
        if k_heads != self.n_kv_heads {
            return Err(BitNetError::Validation(format!(
                "KV cache expects {} heads, but received K tensor with {} heads",
                self.n_kv_heads, k_heads
            )));
        }

        if self.seq_len == 0 {
            // First append: clone is necessary (Arc increment only, not deep copy)
            self.k = k_new.clone();
            self.v = v_new.clone();
        } else {
            // Concatenate along time dimension (dim=2)
            if self.seq_len + new_seq_len > self.max_seq_len {
                return Err(BitNetError::from(candle_core::Error::Msg(
                    "KV cache overflow".to_string(),
                )));
            }
            // Tensor::cat allocates new tensor - no optimization possible here
            self.k = Tensor::cat(&[&self.k, k_new], 2)?;
            self.v = Tensor::cat(&[&self.v, v_new], 2)?;
        }

        self.seq_len += new_seq_len;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
    }
}

/// Full KV Cache for all layers
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(config: &BitNetConfig, batch_size: usize, device: &Device) -> Result<Self> {
        Self::new_with_max_seq_len(config, batch_size, device, config.model.max_position_embeddings)
    }

    pub fn new_with_max_seq_len(
        config: &BitNetConfig,
        batch_size: usize,
        device: &Device,
        max_seq_len: usize,
    ) -> Result<Self> {
        let n_layers = config.model.num_layers;
        let n_heads = config.model.num_heads;
        let hidden_size = config.model.hidden_size;
        let model_max_seq_len = config.model.max_position_embeddings;

        if max_seq_len == 0 {
            return Err(BitNetError::Validation(
                "KVCache: max_seq_len must be greater than zero".to_string(),
            ));
        }
        if max_seq_len > model_max_seq_len {
            return Err(BitNetError::Validation(format!(
                "KVCache: requested max_seq_len {max_seq_len} exceeds model context {model_max_seq_len}"
            )));
        }

        // Validate shape assumptions before calculating dimensions
        if config.model.attention_head_dim.is_none() && !hidden_size.is_multiple_of(n_heads) {
            return Err(BitNetError::Validation(format!(
                "KVCache: hidden_size {} not divisible by num_heads {}",
                hidden_size, n_heads
            )));
        }

        let n_kv_heads = config.model.num_key_value_heads.max(1).min(n_heads);
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(BitNetError::Validation(format!(
                "KVCache: num_heads {} not divisible by num_key_value_heads {}",
                n_heads, n_kv_heads
            )));
        }

        let head_dim = config.model.attention_head_dim.unwrap_or_else(|| hidden_size / n_heads);

        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(LayerKVCache::new(batch_size, n_kv_heads, max_seq_len, head_dim, device)?);
        }

        Ok(Self { layers })
    }

    pub fn estimated_f32_bytes_for_max_seq_len(
        config: &BitNetConfig,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<u128> {
        let n_layers = config.model.num_layers as u128;
        let n_heads = config.model.num_heads;
        let hidden_size = config.model.hidden_size;
        if config.model.attention_head_dim.is_none() && !hidden_size.is_multiple_of(n_heads) {
            return Err(BitNetError::Validation(format!(
                "KVCache: hidden_size {} not divisible by num_heads {}",
                hidden_size, n_heads
            )));
        }
        let n_kv_heads = config.model.num_key_value_heads.max(1).min(n_heads);
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(BitNetError::Validation(format!(
                "KVCache: num_heads {} not divisible by num_key_value_heads {}",
                n_heads, n_kv_heads
            )));
        }
        let head_dim = config.model.attention_head_dim.unwrap_or_else(|| hidden_size / n_heads);
        Ok(n_layers
            .saturating_mul(2)
            .saturating_mul(batch_size as u128)
            .saturating_mul(n_kv_heads as u128)
            .saturating_mul(max_seq_len as u128)
            .saturating_mul(head_dim as u128)
            .saturating_mul(std::mem::size_of::<f32>() as u128))
    }

    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut LayerKVCache> {
        self.layers.get_mut(idx)
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}

/// Complete Transformer Model
pub struct TransformerModel {
    pub config: BitNetConfig,
    pub embed_tokens: candle_nn::Embedding,
    pub embed_transposed: bool, // True if embeddings are stored as [hidden, vocab]
    pub embed_tied_weight: Option<Tensor>, // Cached transposed embedding weight for tied models [H, V]
    pub layers: Vec<TransformerBlock>,
    pub norm: LayerNorm,
    pub lm_head: Option<Linear>,        // Optional for tied weights
    pub lm_head_weight: Option<Tensor>, // Direct access to lm_head weight for transposed handling
    pub lm_head_transposed: bool,       // True if lm_head is stored as [hidden, vocab]
    device: Device,
    raw_tensors: HashMap<String, Tensor>, // Store raw tensors for QK256 dispatch
    dense_linear_hooks: DenseLinearRuntimeHookRegistry,
}

impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tensors(config, vb, HashMap::new())
    }

    pub fn new_with_tensors(
        config: BitNetConfig,
        vb: VarBuilder,
        raw_tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::new_with_tensors_and_dense_linear_hooks(
            config,
            vb,
            raw_tensors,
            DenseLinearRuntimeHookRegistry::default(),
        )
    }

    pub fn new_with_tensors_and_dense_linear_hooks(
        config: BitNetConfig,
        vb: VarBuilder,
        raw_tensors: HashMap<String, Tensor>,
        dense_linear_hooks: DenseLinearRuntimeHookRegistry,
    ) -> Result<Self> {
        let init_start = Instant::now();
        let trace_model_init = qwen_trace_events_enabled();
        let device = vb.device().clone();
        let vocab_size = config.model.vocab_size;
        let hidden_size = config.model.hidden_size;
        let n_layers = config.model.num_layers;
        qwen_trace_model_init_event(trace_model_init, "model_init.start", || {
            format!(
                "\"vocab_size\":{},\"hidden_size\":{},\"layers\":{},\"device\":\"{}\"",
                vocab_size,
                hidden_size,
                n_layers,
                qwen_trace_device_kind(&device)
            )
        });

        qwen_trace_model_init_event(trace_model_init, "model_init.embedding_start", || {
            format!("\"elapsed_ms\":{}", qwen_trace_elapsed_ms(init_start))
        });
        let embed_tokens = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))?;
        qwen_trace_model_init_event(trace_model_init, "model_init.embedding_finish", || {
            format!(
                "\"elapsed_ms\":{},\"embedding_dims\":\"{:?}\"",
                qwen_trace_elapsed_ms(init_start),
                embed_tokens.embeddings().dims()
            )
        });

        // Read transpose flag for embeddings (1-element tensor)
        qwen_trace_model_init_event(
            trace_model_init,
            "model_init.embed_transposed_flag_start",
            || format!("\"elapsed_ms\":{}", qwen_trace_elapsed_ms(init_start)),
        );
        let embed_transposed = match vb.get((1,), "embed_tokens.transposed") {
            Ok(t) => {
                let vals = t.to_vec1::<f32>()?;
                vals.first().copied().unwrap_or(0.0) > 0.5
            }
            Err(_) => false, // If flag doesn't exist, assume not transposed
        };
        qwen_trace_model_init_event(
            trace_model_init,
            "model_init.embed_transposed_flag_finish",
            || {
                format!(
                    "\"elapsed_ms\":{},\"embed_transposed\":{}",
                    qwen_trace_elapsed_ms(init_start),
                    embed_transposed
                )
            },
        );

        if embed_transposed {
            tracing::info!(
                "Embeddings are transposed [hidden, vocab] - will handle efficiently at runtime"
            );
        }

        qwen_trace_model_init_event(trace_model_init, "model_init.layers_start", || {
            format!("\"elapsed_ms\":{},\"layers\":{}", qwen_trace_elapsed_ms(init_start), n_layers)
        });
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let layer_start = Instant::now();
            qwen_trace_model_init_event(trace_model_init, "model_init.layer_start", || {
                format!("\"elapsed_ms\":{},\"layer\":{}", qwen_trace_elapsed_ms(init_start), i)
            });
            let layer = TransformerBlock::new(&config, vb.pp(format!("layers.{}", i)), i)?;
            qwen_trace_model_init_event(trace_model_init, "model_init.layer_finish", || {
                format!(
                    "\"elapsed_ms\":{},\"layer\":{},\"layer_ms\":{}",
                    qwen_trace_elapsed_ms(init_start),
                    i,
                    qwen_trace_elapsed_ms(layer_start)
                )
            });
            layers.push(layer);
        }
        qwen_trace_model_init_event(trace_model_init, "model_init.layers_finish", || {
            format!("\"elapsed_ms\":{},\"layers\":{}", qwen_trace_elapsed_ms(init_start), n_layers)
        });

        // Use RMSNorm epsilon from config header (CRITICAL: must match per-layer norms)
        let eps = config.model.rms_norm_eps.map(|e| e as f64).unwrap_or(1e-5);
        tracing::info!("Final norm using RMSNorm eps={} (from header)", eps);

        qwen_trace_model_init_event(trace_model_init, "model_init.final_norm_start", || {
            format!("\"elapsed_ms\":{},\"eps\":{}", qwen_trace_elapsed_ms(init_start), eps)
        });
        let norm =
            norm_with_optional_bias(config.model.norm_type, hidden_size, eps, vb.pp("final_norm"))?;
        qwen_trace_model_init_event(trace_model_init, "model_init.final_norm_finish", || {
            format!("\"elapsed_ms\":{}", qwen_trace_elapsed_ms(init_start))
        });

        // Try to load lm_head, but it's optional (can be tied to embeddings)
        // Try to create the linear layer, catching errors if weights don't exist
        qwen_trace_model_init_event(trace_model_init, "model_init.lm_head_start", || {
            format!("\"elapsed_ms\":{}", qwen_trace_elapsed_ms(init_start))
        });
        let lm_head_trace = LinearInitTrace {
            enabled: trace_model_init,
            init_start,
            layer_idx: None,
            device: &device,
            scope: "output_head",
            name: "lm_head",
        };
        let (lm_head, lm_head_weight, lm_head_transposed) = match linear_with_optional_bias_traced(
            hidden_size,
            vocab_size,
            vb.pp("lm_head"),
            lm_head_trace,
        ) {
            Ok(layer) => {
                // Also get the weight tensor directly for transposed handling
                // Note: weight dimensions might be transposed
                let weight = vb
                    .get((vocab_size, hidden_size), "lm_head.weight")
                    .or_else(|_| vb.get((hidden_size, vocab_size), "lm_head.weight"))
                    .ok();

                // Read transpose flag for lm_head
                let transposed = match vb.get((1,), "lm_head.transposed") {
                    Ok(t) => {
                        let vals = t.to_vec1::<f32>()?;
                        vals.first().copied().unwrap_or(0.0) > 0.5
                    }
                    Err(_) => false, // If flag doesn't exist, assume not transposed
                };

                if transposed {
                    tracing::info!(
                        "LM head is transposed [hidden, vocab] - will handle efficiently at runtime"
                    );
                }
                (Some(layer), weight, transposed)
            }
            Err(err) => match vb
                .get((vocab_size, hidden_size), "lm_head.weight")
                .or_else(|_| vb.get((vocab_size, hidden_size), "output.weight"))
            {
                Ok(weight) => {
                    tracing::warn!(
                        "lm_head linear construction failed ({err}); recovered canonical \
                         lm_head/output weight [{}, {}] through direct lookup",
                        vocab_size,
                        hidden_size
                    );
                    qwen_trace_model_init_event(
                        trace_model_init,
                        "model_init.linear_bias_finish",
                        || {
                            format!(
                                "\"elapsed_ms\":{},\"layer\":null,\"device\":\"{}\",\"scope\":\"output_head\",\"linear\":\"lm_head\",\"bias_ms\":0,\"present\":false,\"recovery_path\":\"direct_lm_head_or_output_weight\"",
                                qwen_trace_elapsed_ms(init_start),
                                qwen_trace_device_kind(&device)
                            )
                        },
                    );
                    (Some(Linear::new(weight.clone(), None)), Some(weight), false)
                }
                Err(_) => match vb.get((hidden_size, vocab_size), "lm_head.weight") {
                    Ok(weight) => {
                        tracing::info!(
                            "LM head is stored transposed [hidden, vocab] - using direct matmul path"
                        );
                        qwen_trace_model_init_event(
                            trace_model_init,
                            "model_init.linear_bias_finish",
                            || {
                                format!(
                                    "\"elapsed_ms\":{},\"layer\":null,\"device\":\"{}\",\"scope\":\"output_head\",\"linear\":\"lm_head\",\"bias_ms\":0,\"present\":false,\"recovery_path\":\"direct_transposed_lm_head_weight\"",
                                    qwen_trace_elapsed_ms(init_start),
                                    qwen_trace_device_kind(&device)
                                )
                            },
                        );
                        (None, Some(weight), true)
                    }
                    Err(_) => match vb.get((hidden_size, vocab_size), "output.weight") {
                        Ok(weight) => {
                            tracing::warn!(
                                "lm_head linear construction failed ({err}); recovered GGUF \
                                 output.weight with hidden/vocab dims by reshaping to token-major \
                                 [{}, {}] without transposing values",
                                vocab_size,
                                hidden_size
                            );
                            let weight = weight.reshape((vocab_size, hidden_size))?;
                            qwen_trace_model_init_event(
                                trace_model_init,
                                "model_init.linear_bias_finish",
                                || {
                                    format!(
                                        "\"elapsed_ms\":{},\"layer\":null,\"device\":\"{}\",\"scope\":\"output_head\",\"linear\":\"lm_head\",\"bias_ms\":0,\"present\":false,\"recovery_path\":\"direct_transposed_output_weight_reshape\"",
                                        qwen_trace_elapsed_ms(init_start),
                                        qwen_trace_device_kind(&device)
                                    )
                                },
                            );
                            (Some(Linear::new(weight.clone(), None)), Some(weight), false)
                        }
                        Err(_) => {
                            tracing::info!(
                                "lm_head/output weight not found after linear construction failed ({err}); \
                                 will use tied weights"
                            );
                            qwen_trace_model_init_event(
                                trace_model_init,
                                "model_init.linear_bias_finish",
                                || {
                                    format!(
                                        "\"elapsed_ms\":{},\"layer\":null,\"device\":\"{}\",\"scope\":\"output_head\",\"linear\":\"lm_head\",\"bias_ms\":0,\"present\":false,\"recovery_path\":\"tied_embedding_output_head\"",
                                        qwen_trace_elapsed_ms(init_start),
                                        qwen_trace_device_kind(&device)
                                    )
                                },
                            );
                            (None, None, false)
                        }
                    },
                },
            },
        };
        qwen_trace_model_init_event(trace_model_init, "model_init.lm_head_finish", || {
            format!(
                "\"elapsed_ms\":{},\"lm_head_present\":{},\"lm_head_weight_present\":{},\"lm_head_transposed\":{}",
                qwen_trace_elapsed_ms(init_start),
                lm_head.is_some(),
                lm_head_weight.is_some(),
                lm_head_transposed
            )
        });

        // PATCH 2: Optimize tied weights by pre-transposing embeddings once at load
        // NOTE: embed_tokens.embeddings() ALWAYS returns [V,H] (Candle's internal format)
        // regardless of how they were stored in GGUF. We need [H,V] for tied weights.
        let (embed_transposed, embed_tied_weight) = if lm_head.is_none() && lm_head_weight.is_none()
        {
            // No dedicated lm_head, we'll use tied weights - pre-transpose for efficiency
            let embed_weight = embed_tokens.embeddings();
            tracing::info!(
                "Embedding matrix from Candle: {:?} (always [V,H] internally)",
                embed_weight.dims()
            );

            // Always transpose [V,H] -> [H,V] for tied weights, regardless of embed_transposed flag
            // The embed_transposed flag tells us how GGUF stored it, but Candle normalizes to [V,H]
            tracing::info!("Pre-transposing tied embeddings [V,H] -> [H,V] for logits computation");
            qwen_trace_model_init_event(
                trace_model_init,
                "model_init.tied_embedding_transpose_start",
                || {
                    format!(
                        "\"elapsed_ms\":{},\"embedding_dims\":\"{:?}\"",
                        qwen_trace_elapsed_ms(init_start),
                        embed_weight.dims()
                    )
                },
            );
            let transposed_weight = embed_weight.transpose(0, 1)?; // [H, V]
            tracing::info!("Transposed weight shape: {:?}", transposed_weight.dims());
            qwen_trace_model_init_event(
                trace_model_init,
                "model_init.tied_embedding_transpose_finish",
                || {
                    format!(
                        "\"elapsed_ms\":{},\"transposed_dims\":\"{:?}\"",
                        qwen_trace_elapsed_ms(init_start),
                        transposed_weight.dims()
                    )
                },
            );
            (embed_transposed, Some(transposed_weight)) // Cache transposed weight
        } else {
            // Dedicated lm_head exists, no need to optimize embeddings
            (embed_transposed, None)
        };
        qwen_trace_event(
            "model_config",
            &format!(
                "\"vocab_size\":{},\"hidden_size\":{},\"layers\":{},\"heads\":{},\"kv_heads\":{},\"norm_type\":\"{:?}\",\"rms_norm_eps\":{},\"rope_theta\":{},\"embed_transposed\":{},\"lm_head_present\":{},\"lm_head_weight_present\":{},\"lm_head_transposed\":{}",
                vocab_size,
                hidden_size,
                n_layers,
                config.model.num_heads,
                config.model.num_key_value_heads,
                config.model.norm_type,
                config
                    .model
                    .rms_norm_eps
                    .map(|value| qwen_trace_number(value as f64))
                    .unwrap_or_else(|| "null".to_string()),
                config
                    .model
                    .rope_theta
                    .map(|value| qwen_trace_number(value as f64))
                    .unwrap_or_else(|| "null".to_string()),
                embed_transposed,
                lm_head.is_some(),
                lm_head_weight.is_some(),
                lm_head_transposed
            ),
        );
        qwen_trace_model_init_event(trace_model_init, "model_init.finish", || {
            format!("\"elapsed_ms\":{}", qwen_trace_elapsed_ms(init_start))
        });

        Ok(Self {
            config,
            embed_tokens,
            embed_transposed,
            embed_tied_weight,
            layers,
            norm,
            lm_head,
            lm_head_weight,
            lm_head_transposed,
            device,
            raw_tensors,
            dense_linear_hooks,
        })
    }

    pub fn dense_linear_runtime_hook_boundary(
        &self,
        tensor_name: &str,
    ) -> DenseLinearRuntimeHookBoundary {
        dense_linear_runtime_hook_boundary(tensor_name, &self.dense_linear_hooks)
    }

    pub fn dense_linear_runtime_hook_boundaries(&self) -> Vec<DenseLinearRuntimeHookBoundary> {
        let mut tensor_names: Vec<_> = self.dense_linear_hooks.keys().cloned().collect();
        tensor_names.sort();
        tensor_names
            .into_iter()
            .map(|tensor_name| {
                dense_linear_runtime_hook_boundary(&tensor_name, &self.dense_linear_hooks)
            })
            .collect()
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let trace_embed = qwen_trace_events_enabled();
        let embed_start = Instant::now();
        qwen_trace_runtime_event(trace_embed, "model.embed_start", || {
            format!(
                "\"elapsed_ms\":{},\"token_count\":{},\"hidden_size\":{},\"embed_transposed\":{},\"device\":\"{}\"",
                qwen_trace_elapsed_ms(embed_start),
                tokens.len(),
                self.config.model.hidden_size,
                self.embed_transposed,
                qwen_trace_device_kind(&self.device)
            )
        });

        let token_tensor_start = Instant::now();
        qwen_trace_runtime_event(trace_embed, "model.embed_token_tensor_start", || {
            format!(
                "\"elapsed_ms\":{},\"token_count\":{},\"device\":\"{}\"",
                qwen_trace_elapsed_ms(embed_start),
                tokens.len(),
                qwen_trace_device_kind(&self.device)
            )
        });
        let token_ids = Tensor::from_vec(tokens.to_vec(), &[1, tokens.len()], &self.device)?;
        qwen_trace_runtime_event(trace_embed, "model.embed_token_tensor_finish", || {
            format!(
                "\"elapsed_ms\":{},\"op_ms\":{},\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                qwen_trace_elapsed_ms(embed_start),
                qwen_trace_elapsed_ms(token_tensor_start),
                qwen_trace_dims_json(token_ids.dims()),
                token_ids.dtype(),
                qwen_trace_device_kind(token_ids.device())
            )
        });

        // Get dimensions
        let batch_size = token_ids.dims()[0];
        let seq_len = token_ids.dims()[1];
        let hidden_size = self.config.model.hidden_size;

        // Flatten to [B*S] for index_select
        let flatten_start = Instant::now();
        qwen_trace_runtime_event(trace_embed, "model.embed_flatten_start", || {
            format!(
                "\"elapsed_ms\":{},\"dims\":[{}],\"device\":\"{}\"",
                qwen_trace_elapsed_ms(embed_start),
                qwen_trace_dims_json(token_ids.dims()),
                qwen_trace_device_kind(token_ids.device())
            )
        });
        let flat_ids = token_ids.flatten_all()?;
        qwen_trace_runtime_event(trace_embed, "model.embed_flatten_finish", || {
            format!(
                "\"elapsed_ms\":{},\"op_ms\":{},\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                qwen_trace_elapsed_ms(embed_start),
                qwen_trace_elapsed_ms(flatten_start),
                qwen_trace_dims_json(flat_ids.dims()),
                flat_ids.dtype(),
                qwen_trace_device_kind(flat_ids.device())
            )
        });

        if self.embed_transposed {
            // Column-gather path for [hidden, vocab] storage
            // This avoids materializing the full transpose
            let weight_start = Instant::now();
            qwen_trace_runtime_event(trace_embed, "model.embed_weight_start", || {
                format!(
                    "\"elapsed_ms\":{},\"path\":\"column_gather\",\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_device_kind(&self.device)
                )
            });
            let weight = self.embed_tokens.embeddings();
            qwen_trace_runtime_event(trace_embed, "model.embed_weight_finish", || {
                format!(
                    "\"elapsed_ms\":{},\"op_ms\":{},\"path\":\"column_gather\",\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_elapsed_ms(weight_start),
                    qwen_trace_dims_json(weight.dims()),
                    weight.dtype(),
                    qwen_trace_device_kind(weight.device())
                )
            });

            // index_select on dim=1 gathers columns from [H, V]
            // Result: [H, B*S]
            let index_select_start = Instant::now();
            qwen_trace_runtime_event(trace_embed, "model.embed_index_select_start", || {
                format!(
                    "\"elapsed_ms\":{},\"path\":\"column_gather\",\"dim\":1,\"weight_dims\":[{}],\"id_dims\":[{}],\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_dims_json(weight.dims()),
                    qwen_trace_dims_json(flat_ids.dims()),
                    qwen_trace_device_kind(weight.device())
                )
            });
            let cols = weight.index_select(&flat_ids, 1)?;
            qwen_trace_runtime_event(trace_embed, "model.embed_index_select_finish", || {
                format!(
                    "\"elapsed_ms\":{},\"op_ms\":{},\"path\":\"column_gather\",\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_elapsed_ms(index_select_start),
                    qwen_trace_dims_json(cols.dims()),
                    cols.dtype(),
                    qwen_trace_device_kind(cols.device())
                )
            });

            // Transpose to [B*S, H] (small transpose, only B*S elements)
            let transpose_start = Instant::now();
            qwen_trace_runtime_event(trace_embed, "model.embed_transpose_start", || {
                format!(
                    "\"elapsed_ms\":{},\"dims\":[{}],\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_dims_json(cols.dims()),
                    qwen_trace_device_kind(cols.device())
                )
            });
            let embeddings = cols.t()?;
            qwen_trace_runtime_event(trace_embed, "model.embed_transpose_finish", || {
                format!(
                    "\"elapsed_ms\":{},\"op_ms\":{},\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_elapsed_ms(transpose_start),
                    qwen_trace_dims_json(embeddings.dims()),
                    embeddings.dtype(),
                    qwen_trace_device_kind(embeddings.device())
                )
            });

            // Reshape to [B, S, H]
            let reshape_start = Instant::now();
            qwen_trace_runtime_event(trace_embed, "model.embed_reshape_start", || {
                format!(
                    "\"elapsed_ms\":{},\"target_dims\":[{},{},{}],\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    batch_size,
                    seq_len,
                    hidden_size,
                    qwen_trace_device_kind(embeddings.device())
                )
            });
            let output = embeddings.reshape(&[batch_size, seq_len, hidden_size])?;
            qwen_trace_runtime_event(trace_embed, "model.embed_reshape_finish", || {
                format!(
                    "\"elapsed_ms\":{},\"op_ms\":{},\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_elapsed_ms(reshape_start),
                    qwen_trace_dims_json(output.dims()),
                    output.dtype(),
                    qwen_trace_device_kind(output.device())
                )
            });
            qwen_trace_runtime_event(trace_embed, "model.embed_finish", || {
                format!(
                    "\"elapsed_ms\":{},\"path\":\"column_gather\",\"dims\":[{}],\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_dims_json(output.dims()),
                    qwen_trace_device_kind(output.device())
                )
            });
            Ok(output)
        } else {
            // Row-gather path for standard [vocab, hidden] storage
            let weight_start = Instant::now();
            qwen_trace_runtime_event(trace_embed, "model.embed_weight_start", || {
                format!(
                    "\"elapsed_ms\":{},\"path\":\"row_gather\",\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_device_kind(&self.device)
                )
            });
            let weight = self.embed_tokens.embeddings();
            qwen_trace_runtime_event(trace_embed, "model.embed_weight_finish", || {
                format!(
                    "\"elapsed_ms\":{},\"op_ms\":{},\"path\":\"row_gather\",\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_elapsed_ms(weight_start),
                    qwen_trace_dims_json(weight.dims()),
                    weight.dtype(),
                    qwen_trace_device_kind(weight.device())
                )
            });

            let rows = if tokens.len() == 1 {
                let vocab_size = weight.dims().first().copied().ok_or_else(|| {
                    BitNetError::Validation("embedding weight must expose a vocab dimension".into())
                })?;
                let token_id = tokens[0] as usize;
                if token_id >= vocab_size {
                    return Err(BitNetError::Validation(format!(
                        "single-token embedding id {token_id} is outside vocab size {vocab_size}"
                    )));
                }

                // Avoid Candle CUDA index_select for the strict one-token Qwen3 frontier.
                let narrow_start = Instant::now();
                qwen_trace_runtime_event(
                    trace_embed,
                    "model.embed_single_token_narrow_start",
                    || {
                        format!(
                            "\"elapsed_ms\":{},\"path\":\"row_gather_single_token_narrow\",\"dim\":0,\"weight_dims\":[{}],\"device\":\"{}\"",
                            qwen_trace_elapsed_ms(embed_start),
                            qwen_trace_dims_json(weight.dims()),
                            qwen_trace_device_kind(weight.device())
                        )
                    },
                );
                let rows = weight.narrow(0, token_id, 1)?;
                qwen_trace_runtime_event(
                    trace_embed,
                    "model.embed_single_token_narrow_finish",
                    || {
                        format!(
                            "\"elapsed_ms\":{},\"op_ms\":{},\"path\":\"row_gather_single_token_narrow\",\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                            qwen_trace_elapsed_ms(embed_start),
                            qwen_trace_elapsed_ms(narrow_start),
                            qwen_trace_dims_json(rows.dims()),
                            rows.dtype(),
                            qwen_trace_device_kind(rows.device())
                        )
                    },
                );
                rows
            } else {
                // index_select on dim=0 gathers rows from [V, H]
                // Result: [B*S, H]
                let index_select_start = Instant::now();
                qwen_trace_runtime_event(trace_embed, "model.embed_index_select_start", || {
                    format!(
                        "\"elapsed_ms\":{},\"path\":\"row_gather\",\"dim\":0,\"weight_dims\":[{}],\"id_dims\":[{}],\"device\":\"{}\"",
                        qwen_trace_elapsed_ms(embed_start),
                        qwen_trace_dims_json(weight.dims()),
                        qwen_trace_dims_json(flat_ids.dims()),
                        qwen_trace_device_kind(weight.device())
                    )
                });
                let rows = weight.index_select(&flat_ids, 0)?;
                qwen_trace_runtime_event(trace_embed, "model.embed_index_select_finish", || {
                    format!(
                        "\"elapsed_ms\":{},\"op_ms\":{},\"path\":\"row_gather\",\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                        qwen_trace_elapsed_ms(embed_start),
                        qwen_trace_elapsed_ms(index_select_start),
                        qwen_trace_dims_json(rows.dims()),
                        rows.dtype(),
                        qwen_trace_device_kind(rows.device())
                    )
                });
                rows
            };

            // Reshape to [B, S, H]
            let reshape_start = Instant::now();
            qwen_trace_runtime_event(trace_embed, "model.embed_reshape_start", || {
                format!(
                    "\"elapsed_ms\":{},\"target_dims\":[{},{},{}],\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    batch_size,
                    seq_len,
                    hidden_size,
                    qwen_trace_device_kind(rows.device())
                )
            });
            let output = rows.reshape(&[batch_size, seq_len, hidden_size])?;
            qwen_trace_runtime_event(trace_embed, "model.embed_reshape_finish", || {
                format!(
                    "\"elapsed_ms\":{},\"op_ms\":{},\"dims\":[{}],\"dtype\":\"{:?}\",\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_elapsed_ms(reshape_start),
                    qwen_trace_dims_json(output.dims()),
                    output.dtype(),
                    qwen_trace_device_kind(output.device())
                )
            });
            qwen_trace_runtime_event(trace_embed, "model.embed_finish", || {
                format!(
                    "\"elapsed_ms\":{},\"path\":\"row_gather\",\"dims\":[{}],\"device\":\"{}\"",
                    qwen_trace_elapsed_ms(embed_start),
                    qwen_trace_dims_json(output.dims()),
                    qwen_trace_device_kind(output.device())
                )
            });
            Ok(output)
        }
    }

    /// Teacher-forcing forward: full sequence `[B,T] -> [B,T,V]` logits
    ///
    /// This implementation mirrors the incremental decoding path by
    /// processing tokens step-by-step with a KV cache. This ensures that
    /// rotary (or absolute) positional encodings are applied per layer with
    /// the correct positions and that a causal mask prevents attending to
    /// future tokens.
    pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
        // Token ids expected shape: [B,T]
        let (batch_size, seq_len) = token_ids.dims2()?;

        // Embed the entire sequence once.
        let flat_ids = token_ids.flatten_all()?;
        let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
        let hidden = self.embed(&ids_vec)?;
        let hidden_size = self.config.model.hidden_size;
        let hidden = hidden.reshape(&[batch_size, seq_len, hidden_size])?;

        // Probe A1: Embedding RMS (step 0 only)
        if trace_rms_enabled() {
            static EMB_LOGGED: std::sync::Once = std::sync::Once::new();
            EMB_LOGGED.call_once(|| {
                let _ = (|| -> candle_core::Result<()> {
                    let emb_vec = hidden.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
                    let rms = (emb_vec.iter().map(|x| x * x).sum::<f32>()
                        / emb_vec.len().max(1) as f32)
                        .sqrt();
                    eprintln!("trace: emb_rms={:.6}", rms);
                    Ok(())
                })();
            });
        }

        // Tracepoint 1: Embeddings output (after embed, before layers)
        #[cfg(feature = "trace")]
        {
            use bitnet_trace::dump_trace;
            // Extract first token's embedding for tracing [B, 1, H]
            let first_token_emb = hidden.narrow(1, 0, 1)?;
            let _ = dump_trace(
                "embeddings",
                &first_token_emb,
                Some(0),            // seq=0 (prefill step)
                Some(-1),           // layer=-1 (pre-layer operation)
                Some("embeddings"), // stage name
            );
        }

        // Create per-layer KV cache so that rotary/absolute positional
        // encodings use the proper positions during iterative decoding.
        let mut kv_cache = KVCache::new(&self.config, batch_size, &self.device)?;

        // Collect logits for each position.
        let mut logits_steps = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // Select the current token's embedding as [B, 1, H] (keep seq dim for attention)
            let step_hidden = hidden.narrow(1, t, 1)?;

            // Run through all layers using the incremental path which applies
            // positional encoding per layer and causal masking internally.
            let step_hidden = self.forward(step_hidden, Some(&mut kv_cache))?;

            // Tracepoint: All layers output for this position
            #[cfg(feature = "trace")]
            {
                use bitnet_trace::dump_trace;
                let _ = dump_trace(
                    &format!("t{}_all_layers_out", t),
                    &step_hidden,
                    Some(t),                // seq=t (current position)
                    Some(-2),               // layer=-2 (post-all-layers)
                    Some("all_layers_out"), // stage name
                );
            }

            // Project to vocabulary logits for this step.
            let step_logits = self.logits(&step_hidden)?;

            // Trace logits for this position
            #[cfg(feature = "trace")]
            {
                use bitnet_trace::dump_trace;
                let _ = dump_trace(
                    &format!("t{}_logits", t),
                    &step_logits,
                    Some(t),        // seq=t (current position)
                    Some(-1),       // layer=-1 (post-layers stage)
                    Some("logits"), // stage name
                );
            }

            logits_steps.push(step_logits);
        }

        // Stack logits: handle both [B,V] and [B,1,V] shapes
        let logits = if logits_steps[0].dims().len() == 2 {
            // logits are [B, V], stack them to [B, T, V]
            let logits_2d: Vec<_> = logits_steps
                .iter()
                .map(|t| t.unsqueeze(1))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Tensor::cat(&logits_2d, 1)?
        } else {
            // logits are [B, 1, V], concatenate along time dimension
            Tensor::cat(&logits_steps, 1)?
        };

        // Tracepoint 5: Final logits (first token only)
        #[cfg(feature = "trace")]
        {
            // Extract first token's logits for tracing [B, 1, V]
            let first_token_logits = logits.narrow(1, 0, 1)?;
            bitnet_trace::dump_trace(
                "t0/logits",
                &first_token_logits,
                Some(0),
                Some(-1),
                Some("logits"),
            )
            .map_err(BitNetError::from)?;
        }

        Ok(logits)
    }

    /// Forward pass through transformer layers
    ///
    /// **Performance note**: Accepts ownership of `hidden` to avoid cloning on hot path.
    /// Caller should pass owned tensor or use `.clone()` explicitly if needed.
    pub fn forward(&self, hidden: Tensor, kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
        self.forward_impl(hidden, kv_cache, None, None)
    }

    pub fn forward_with_workspace(
        &self,
        hidden: Tensor,
        kv_cache: Option<&mut KVCache>,
        workspace: &mut TransformerForwardWorkspace,
    ) -> Result<Tensor> {
        workspace.record_model_input(&hidden);
        let output = self.forward_impl(hidden, kv_cache, Some(workspace), None)?;
        workspace.record_model_output(&output);
        workspace.store_model_output(output);
        workspace.take_model_output()
    }

    pub fn forward_with_no_bias_callsite_descriptor(
        &self,
        hidden: Tensor,
        kv_cache: Option<&mut KVCache>,
        prompt_bound_no_bias_descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    ) -> Result<Tensor> {
        self.forward_impl(hidden, kv_cache, None, Some(prompt_bound_no_bias_descriptor))
    }

    pub fn forward_with_workspace_and_no_bias_callsite_descriptor(
        &self,
        hidden: Tensor,
        kv_cache: Option<&mut KVCache>,
        workspace: &mut TransformerForwardWorkspace,
        prompt_bound_no_bias_descriptor: &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
    ) -> Result<Tensor> {
        workspace.record_model_input(&hidden);
        let output = self.forward_impl(
            hidden,
            kv_cache,
            Some(workspace),
            Some(prompt_bound_no_bias_descriptor),
        )?;
        workspace.record_model_output(&output);
        workspace.store_model_output(output);
        workspace.take_model_output()
    }

    fn forward_impl(
        &self,
        hidden: Tensor,
        mut kv_cache: Option<&mut KVCache>,
        mut workspace: Option<&mut TransformerForwardWorkspace>,
        prompt_bound_no_bias_descriptor: Option<
            &DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary,
        >,
    ) -> Result<Tensor> {
        let trace_forward = qwen_trace_events_enabled();
        let forward_start = Instant::now();
        qwen_trace_runtime_event(trace_forward, "model.forward_start", || {
            format!(
                "\"dims\":[{}],\"device\":\"{}\",\"layers\":{}",
                qwen_trace_dims_json(hidden.dims()),
                qwen_trace_device_kind(hidden.device()),
                self.layers.len()
            )
        });
        let mut x = hidden; // Take ownership - no clone needed!

        if let Some(descriptor) = prompt_bound_no_bias_descriptor {
            let matching_layer_count = (0..self.layers.len())
                .filter(|layer_idx| {
                    prompt_bound_no_bias_descriptor_targets_feed_forward_down_proj_layer(
                        descriptor, *layer_idx,
                    )
                })
                .count();
            if matching_layer_count != 1 {
                return Err(BitNetError::Validation(format!(
                    "prompt-bound no-bias descriptor target {} must match exactly one feed_forward.down_proj layer before model forward",
                    descriptor.tensor_name
                )));
            }
        }

        // Tracepoint 1: Embeddings (incremental path - single token)
        // This captures the embedding for the current token being processed
        #[cfg(feature = "trace")]
        {
            // For incremental path, hidden is already [B, H] (single token)
            // Trace it directly without narrowing (unlike forward_full which has [B, T, H])
            bitnet_trace::dump_trace("t0/embeddings", &x, Some(0), Some(-1), Some("embeddings"))
                .map_err(BitNetError::from)?;
        }

        // Debug input activation norm
        if debug_attn_enabled()
            && let Ok(norm) = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            eprintln!("[norm] input: {:.6e}", norm);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_start = Instant::now();
            qwen_trace_runtime_event(trace_forward, "model.forward_layer_start", || {
                format!("\"layer\":{},\"dims\":[{}]", i, qwen_trace_dims_json(x.dims()))
            });
            let layer_cache = kv_cache.as_mut().and_then(|c| c.layer_mut(i));
            let layer_no_bias_descriptor = prompt_bound_no_bias_descriptor.filter(|descriptor| {
                prompt_bound_no_bias_descriptor_targets_feed_forward_down_proj_layer(descriptor, i)
            });
            x = if let Some(workspace) = workspace.as_mut() {
                if let Some(descriptor) = layer_no_bias_descriptor {
                    layer.forward_with_workspace_and_no_bias_callsite_descriptor(
                        &x,
                        layer_cache,
                        &self.raw_tensors,
                        &self.dense_linear_hooks,
                        workspace,
                        descriptor,
                    )?
                } else {
                    layer.forward_with_workspace(
                        &x,
                        layer_cache,
                        &self.raw_tensors,
                        &self.dense_linear_hooks,
                        workspace,
                    )?
                }
            } else if let Some(descriptor) = layer_no_bias_descriptor {
                layer.forward_with_no_bias_callsite_descriptor(
                    &x,
                    layer_cache,
                    &self.raw_tensors,
                    &self.dense_linear_hooks,
                    descriptor,
                )?
            } else {
                layer.forward(&x, layer_cache, &self.raw_tensors, &self.dense_linear_hooks)?
            };

            // Debug layer activation norms (show all layers when debugging)
            if debug_attn_enabled()
                && let Ok(norm) = x.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
            {
                eprintln!("[norm] layer {i}: {:.6e}", norm);
            }
            qwen_trace_runtime_event(trace_forward, "model.forward_layer_finish", || {
                format!(
                    "\"layer\":{},\"layer_ms\":{},\"dims\":[{}],\"device\":\"{}\"",
                    i,
                    qwen_trace_elapsed_ms(layer_start),
                    qwen_trace_dims_json(x.dims()),
                    qwen_trace_device_kind(x.device())
                )
            });
        }

        let final_norm_start = Instant::now();
        qwen_trace_runtime_event(trace_forward, "model.final_norm_start", || {
            format!("\"dims\":[{}]", qwen_trace_dims_json(x.dims()))
        });
        let normalized = self.norm.forward(&x)?;
        if let Some(workspace) = workspace.as_mut() {
            workspace.record_model_forward_source_tensors(&x, &normalized);
            workspace.record_final_norm_output_storage_boundary(&normalized, &self.norm);
        }
        qwen_trace_runtime_event(trace_forward, "model.final_norm_finish", || {
            format!(
                "\"norm_ms\":{},\"dims\":[{}],\"device\":\"{}\"",
                qwen_trace_elapsed_ms(final_norm_start),
                qwen_trace_dims_json(normalized.dims()),
                qwen_trace_device_kind(normalized.device())
            )
        });
        qwen_trace_tensor("model.final_norm", None, &normalized)?;
        if debug_attn_enabled()
            && let Ok(norm) = normalized.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
        {
            eprintln!("[norm] final: {:.6e}", norm);
        }

        qwen_trace_runtime_event(trace_forward, "model.forward_finish", || {
            format!(
                "\"forward_ms\":{},\"dims\":[{}],\"device\":\"{}\"",
                qwen_trace_elapsed_ms(forward_start),
                qwen_trace_dims_json(normalized.dims()),
                qwen_trace_device_kind(normalized.device())
            )
        });

        Ok(normalized)
    }

    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let vocab_size = self.config.model.vocab_size;
        let has_tied_qk256_output = self.raw_tensors.contains_key(TIED_EMBED_QK256_KEY);
        qwen_trace_tensor("lm_head.input_hidden", None, hidden)?;
        qwen_trace_event(
            "lm_head.metadata",
            &format!(
                "\"lm_head_present\":{},\"lm_head_transposed\":{},\"embed_transposed\":{},\"has_cached_tied_weight\":{},\"has_tied_qk256_output\":{}",
                self.lm_head.is_some(),
                self.lm_head_transposed,
                self.embed_transposed,
                self.embed_tied_weight.is_some(),
                has_tied_qk256_output
            ),
        );

        match hidden.rank() {
            2 => {
                // [B, H] - last token only
                let (b, _h) = (hidden.dims()[0], hidden.dims()[1]);

                let logits = if self.lm_head_transposed {
                    if let Some(ref weight) = self.lm_head_weight {
                        hidden.matmul(weight)?.reshape(&[b, vocab_size])?
                    } else if let Some(ref lm_head) = self.lm_head {
                        let logits = lm_head.forward(hidden)?; // [B, V]
                        logits.reshape(&[b, vocab_size])?
                    } else {
                        return Err(BitNetError::Validation(
                            "lm_head is marked transposed but lm_head.weight is unavailable".into(),
                        ));
                    }
                } else if let Some(ref lm_head) = self.lm_head {
                    // Use dedicated LM head if available
                    let logits = lm_head.forward(hidden)?; // [B, V]
                    logits.reshape(&[b, vocab_size])?
                } else if let Some(qk256_tensor) = self.raw_tensors.get(TIED_EMBED_QK256_KEY) {
                    static LOGGED_QK256_TIED: std::sync::Once = std::sync::Once::new();
                    LOGGED_QK256_TIED.call_once(|| {
                        tracing::info!(
                            "LM head tied to raw QK256 token embeddings for BitNet.cpp parity"
                        );
                    });
                    let inline_scale = qk256_inline_scale(&self.raw_tensors, TIED_EMBED_QK256_KEY)?;
                    forward_qk256_with_scale(
                        hidden,
                        qk256_tensor,
                        TIED_EMBED_QK256_KEY,
                        inline_scale,
                    )?
                } else {
                    // Tied weights: use embedding matrix
                    static LOGGED: std::sync::Once = std::sync::Once::new();
                    LOGGED.call_once(|| {
                        tracing::info!("LM head tied to input embeddings");
                    });

                    let result = if self.embed_transposed {
                        // Embeddings are [hidden, vocab]
                        let embeddings = self.embed_tokens.embeddings();
                        hidden.matmul(embeddings)? // [B, V]
                    } else if let Some(ref cached_weight) = self.embed_tied_weight {
                        // Use pre-transposed cached weight [H, V] - avoids per-step transpose!
                        hidden.matmul(cached_weight)? // [B, V]
                    } else {
                        // Fallback: transpose on-demand (should be rare after optimization)
                        let embeddings = self.embed_tokens.embeddings();
                        let w = embeddings.transpose(0, 1)?; // [H, V]
                        hidden.matmul(&w)? // [B, V]
                    };

                    // Debug: sanity check tied embeddings orientation (runs once)
                    if std::env::var("BITNET_DEBUG_LOGITS").is_ok() {
                        static SANITY_LOGGED: std::sync::Once = std::sync::Once::new();
                        SANITY_LOGGED.call_once(|| {
                            if let Ok(mean_val) = result.mean_all().and_then(|m| m.to_scalar::<f32>())
                                && let Ok(std_val) = result.broadcast_sub(&result.mean_all().unwrap())
                                    .and_then(|d| d.sqr())
                                    .and_then(|s| s.mean_all())
                                    .and_then(|v| v.sqrt())
                                    .and_then(|s| s.to_scalar::<f32>())
                            {
                                tracing::info!("tied logits sanity check - mean/std: {:.4}/{:.4}", mean_val, std_val);

                                // Float sanity check: compare with non-quantized path
                                if let Ok(emb) = self.embed_tokens.embeddings().transpose(0, 1)
                                    && let Ok(ref_logits) = hidden.matmul(&emb)
                                    && let Ok(ref_mean) = ref_logits.mean_all().and_then(|m| m.to_scalar::<f32>())
                                    && let Ok(ref_std) = ref_logits.broadcast_sub(&ref_logits.mean_all().unwrap())
                                        .and_then(|d| d.sqr())
                                        .and_then(|s| s.mean_all())
                                        .and_then(|v| v.sqrt())
                                        .and_then(|s| s.to_scalar::<f32>())
                                {
                                    tracing::info!("float ref logits - mean/std: {:.4}/{:.4}", ref_mean, ref_std);
                                    tracing::info!("correlation check: quantized vs float stats should be similar");
                                }
                            }
                        });
                    }

                    result
                };

                // Debug logits std
                if debug_attn_enabled()
                    && let Ok(mean) = logits.mean_all()
                    && let Ok(diff) = logits.broadcast_sub(&mean)
                    && let Ok(variance) = diff.sqr()?.mean_all()
                    && let Ok(std_val) = variance.sqrt()?.to_scalar::<f32>()
                {
                    eprintln!("[norm] logits std: {:.6e}", std_val);
                }
                qwen_trace_tensor("lm_head.logits", None, &logits)?;

                // Tracepoint 5: Logits (incremental path - single token)
                // This captures the final logits for the current token [B, V]
                #[cfg(feature = "trace")]
                {
                    // For incremental path, logits are [B, V] (single token)
                    // Trace directly without narrowing (unlike forward_full which has [B, T, V])
                    bitnet_trace::dump_trace(
                        "t0/logits",
                        &logits,
                        Some(0),
                        Some(-1),
                        Some("logits"),
                    )
                    .map_err(BitNetError::from)?;
                }

                Ok(logits)
            }
            3 => {
                // [B, T, H] - all timesteps
                let (b, t, h) = (hidden.dims()[0], hidden.dims()[1], hidden.dims()[2]);

                if self.lm_head_transposed {
                    if let Some(ref weight) = self.lm_head_weight {
                        // LM head weight is stored as [hidden, vocab].
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(weight)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else if let Some(ref lm_head) = self.lm_head {
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = lm_head.forward(&hidden_2d)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else {
                        Err(BitNetError::Validation(
                            "lm_head is marked transposed but lm_head.weight is unavailable".into(),
                        ))
                    }
                } else if let Some(ref lm_head) = self.lm_head {
                    // Use dedicated LM head if available
                    // Standard path: LM head weight is [vocab, hidden]
                    // Flatten to 2D for proper matmul
                    let hidden_2d = hidden.reshape(&[b * t, h])?;
                    let logits_2d = lm_head.forward(&hidden_2d)?;
                    Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                } else if let Some(qk256_tensor) = self.raw_tensors.get(TIED_EMBED_QK256_KEY) {
                    static LOGGED_QK256_TIED: std::sync::Once = std::sync::Once::new();
                    LOGGED_QK256_TIED.call_once(|| {
                        tracing::info!(
                            "LM head tied to raw QK256 token embeddings for BitNet.cpp parity"
                        );
                    });
                    let inline_scale = qk256_inline_scale(&self.raw_tensors, TIED_EMBED_QK256_KEY)?;
                    let logits = forward_qk256_with_scale(
                        hidden,
                        qk256_tensor,
                        TIED_EMBED_QK256_KEY,
                        inline_scale,
                    )?;
                    Ok(logits.reshape(&[b, t, vocab_size])?)
                } else {
                    // Tied weights: use embedding matrix
                    static LOGGED: std::sync::Once = std::sync::Once::new();
                    LOGGED.call_once(|| {
                        tracing::info!("LM head tied to input embeddings");
                    });

                    if self.embed_transposed {
                        // Embeddings are [hidden, vocab], flatten hidden for matmul
                        let embeddings = self.embed_tokens.embeddings();
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(embeddings)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else if let Some(ref cached_weight) = self.embed_tied_weight {
                        // Use pre-transposed cached weight [H, V] - avoids per-step transpose!
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(cached_weight)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    } else {
                        // Fallback: transpose on-demand (should be rare after optimization)
                        let embeddings = self.embed_tokens.embeddings();
                        let w = embeddings.transpose(0, 1)?; // [H, V]
                        let hidden_2d = hidden.reshape(&[b * t, h])?;
                        let logits_2d = hidden_2d.matmul(&w)?;
                        Ok(logits_2d.reshape(&[b, t, vocab_size])?)
                    }
                }
            }
            _ => Err(BitNetError::Validation("unexpected hidden rank".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::config::ModelConfig;
    use candle_nn::RmsNorm;
    use serial_test::serial;
    use std::collections::HashMap;

    /// Helper to compute RMS (root mean square) of a tensor
    fn compute_rms(tensor: &Tensor) -> candle_core::Result<f64> {
        let squared = tensor.sqr()?;
        let mean = squared.mean_all()?;
        let rms = mean.sqrt()?.to_scalar::<f32>()? as f64;
        Ok(rms)
    }

    #[test]
    fn layer_output_storage_boundary_names_exact_candle_residual_add_blocker() {
        let boundary =
            LayerOutputStorageApiBoundary::from_candle_residual_add("transformer.block.output");

        assert_eq!(boundary.status, "layer_output_storage_blocked_by_candle_tensor_add_ops");
        assert_eq!(boundary.public_api_return_type, "Result<Tensor>");
        assert!(!boundary.public_api_accepts_output_storage);
        assert!(!boundary.backend_internal_in_place_api_exposed);
        assert!(
            boundary.exact_blocking_ops.contains(&"Tensor::add(&self, &Tensor) -> Result<Tensor>")
        );
        assert!(
            boundary
                .exact_blocking_ops
                .contains(&"Tensor::broadcast_add(&self, &Tensor) -> Result<Tensor>")
        );
        assert_eq!(
            boundary.required_missing_api,
            "Tensor residual-add API accepting caller-provided output storage, e.g. add_out/broadcast_add_out(&self, rhs, &mut output)"
        );
        assert!(boundary.required_shape_contract.contains("same shape, dtype, and device"));
        assert!(boundary.ownership_contract.contains("TransformerForwardWorkspace"));
        assert!(boundary.behavior_preservation_gate.contains("Qwen3 Q8_0"));
        assert!(boundary.behavior_preservation_gate.contains("Qwen2.5 Q8_0"));
        assert_eq!(
            boundary.runtime_slice_status,
            "runtime_slice_blocked_by_missing_caller_output_storage_api"
        );
        assert!(boundary.runtime_slice_blocker.contains("requires an output-storage API"));
        assert!(boundary.candle_api_evidence.contains(
            &"candle-core-0.10.2/src/tensor.rs binary_op!(add, Add) returns Result<Tensor>"
        ));
        assert!(!boundary.can_fill_caller_output_storage);
    }

    #[test]
    fn logits_output_storage_boundary_names_exact_candle_logits_blocker() {
        let boundary = LogitsOutputStorageApiBoundary::from_candle_logits("model.logits");

        assert_eq!(boundary.status, "logits_output_storage_blocked_by_candle_tensor_ops");
        assert_eq!(boundary.public_api_return_type, "Result<Tensor>");
        assert!(!boundary.public_api_accepts_output_storage);
        assert!(!boundary.backend_internal_in_place_api_exposed);
        assert!(
            boundary
                .exact_blocking_ops
                .contains(&"candle_nn::Linear::forward(&self, &Tensor) -> Result<Tensor>")
        );
        assert!(
            boundary
                .exact_blocking_ops
                .contains(&"Tensor::matmul(&self, &Tensor) -> Result<Tensor>")
        );
        assert!(boundary.required_missing_api.contains("caller-provided output storage"));
        assert!(
            boundary
                .fused_selection_blocking_ops
                .contains(&"Tensor::argmax(&self, dim) -> Result<Tensor> selects after full logits Tensor materialization")
        );
        assert!(
            boundary
                .fused_selection_blocking_ops
                .contains(&"Tensor::sort_last_dim(&self, asc) -> Result<(Tensor, Tensor)> sorts after full logits Tensor materialization")
        );
        assert!(boundary.device_argmax_available_after_logits_tensor);
        assert!(boundary.topk_sort_available_after_logits_tensor);
        assert!(!boundary.can_fuse_output_head_and_selection);
        assert!(!boundary.can_fill_caller_output_storage);
    }

    #[test]
    fn test_relu2_activation_squares_positive_values() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::from_slice(&[-2.0f32, -0.5, 0.0, 2.0, 3.0], (5,), &device)?;
        let output = apply_ffn_activation(&input, ActivationType::Relu2)?;
        let values = output.to_vec1::<f32>()?;

        assert_eq!(values, vec![0.0, 0.0, 0.0, 4.0, 9.0]);
        Ok(())
    }

    #[test]
    fn test_layer_norm_with_standard_gamma() -> candle_core::Result<()> {
        // Test that RMSNorm behaves correctly with standard gamma (RMS ≈ 1.0)
        let device = Device::Cpu;
        let hidden_size = 2560;
        let eps = 1e-5;

        // Create input tensor [1, 1, 2560]
        let input_data: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let x = i as f32 / hidden_size as f32;
                ((x * 10.0).sin() + (x * 20.0).cos()) * 0.5
            })
            .collect();

        let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

        // Create standard gamma (all ones)
        let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;

        // Apply RMSNorm
        let rms_norm = RmsNorm::new(gamma, eps);
        let output = rms_norm.forward(&input)?;

        // Verify output RMS is reasonable (should be close to 1.0)
        let output_rms = compute_rms(&output)?;

        assert!(
            output_rms > 0.5 && output_rms < 2.0,
            "Output RMS should be reasonable with standard gamma, got {:.6e}",
            output_rms
        );

        // Verify no NaN/Inf
        let vec_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let has_nan = vec_data.iter().any(|x| x.is_nan());
        let has_inf = vec_data.iter().any(|x| x.is_infinite());
        assert!(!has_nan, "Output should not contain NaN");
        assert!(!has_inf, "Output should not contain Inf");

        Ok(())
    }

    #[test]
    fn attention_f16_dot_input_uses_f16_roundtrip_values() -> Result<()> {
        let device = Device::Cpu;
        let input =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 1, 4), &device)?;

        let output = attention_f16_dot_input(&input)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(values, vec![1.0, -2.0, 3.125, -4.25]);
        Ok(())
    }

    #[test]
    fn attention_score_key_input_preserves_f32_values() -> Result<()> {
        let device = Device::Cpu;
        let input =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 1, 4), &device)?;

        let output = attention_score_key_input(&input)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(values, vec![1.0003, -2.0007, 3.1259, -4.2509]);
        Ok(())
    }

    #[test]
    fn exact_q8_sidecar_runtime_hook_matches_dense_linear_reference() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[0.5f32, 1.0, 1.5, 2.0], (2, 2), &device)?;
        let linear = Linear::new(weight, None);
        let input = Tensor::from_slice(&[2.0f32, 3.0], (1, 1, 2), &device)?;

        let mut packed = Vec::new();
        packed.extend_from_slice(&f32_to_fp16(0.5).to_le_bytes());
        for value in [1i8, 2, 3, 4] {
            packed.push(value as u8);
        }
        packed.resize(34, 0);

        let mut hooks = DenseLinearRuntimeHookRegistry::default();
        hooks.insert(
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR.to_string(),
            DenseLinearRuntimeHookDescriptor {
                tensor_name: "blk.0.attn_q.weight".to_string(),
                role: "AttentionQ".to_string(),
                sidecar_payload_sha256: Some("sha256:test".to_string()),
                packed_q8_payload: Some(DenseLinearPackedQ8Payload {
                    tensor_name: "blk.0.attn_q.weight".to_string(),
                    packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
                    q8_block_size: 32,
                    q8_block_count: 1,
                    matrix_rows: 2,
                    matrix_cols: 2,
                }),
                payload_order_matches_runtime_shape: true,
                source_order_q8_matvec_candidate: false,
                source_order_input_dim: None,
                source_order_output_dim: None,
                runtime_compute_enabled: true,
                receipt_bound_no_bias_selector: None,
            },
        );

        let instrumentation_before = dense_q8_sidecar_instrumentation_snapshot();
        let Some(output) = maybe_forward_dense_q8_sidecar_linear(
            &input,
            &linear,
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR,
            &hooks,
        )?
        else {
            return Err(BitNetError::Validation(
                "expected exact Q8 sidecar runtime hook to run".to_string(),
            ));
        };

        assert_eq!(output.dims(), &[1, 1, 2]);
        assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, vec![4.0, 9.0]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            linear.forward(&input)?.flatten_all()?.to_vec1::<f32>()?
        );
        let instrumentation_after = dense_q8_sidecar_instrumentation_snapshot();
        assert!(
            instrumentation_after.selector_dispatch_calls
                > instrumentation_before.selector_dispatch_calls
        );
        assert!(
            instrumentation_after.selector_selected_calls
                > instrumentation_before.selector_selected_calls
        );
        assert!(
            instrumentation_after.input_materialization_calls
                > instrumentation_before.input_materialization_calls
        );
        assert!(
            instrumentation_after.input_values_materialized
                >= instrumentation_before.input_values_materialized + 2
        );
        assert!(
            instrumentation_after.bias_materialization_calls
                > instrumentation_before.bias_materialization_calls
        );
        assert!(
            instrumentation_after.packed_matvec_calls > instrumentation_before.packed_matvec_calls
        );
        assert!(
            instrumentation_after.packed_matvec_input_rows
                > instrumentation_before.packed_matvec_input_rows
        );
        assert!(
            instrumentation_after.packed_matvec_output_values
                >= instrumentation_before.packed_matvec_output_values + 2
        );
        assert!(
            instrumentation_after.output_tensor_construction_calls
                > instrumentation_before.output_tensor_construction_calls
        );
        Ok(())
    }

    #[test]
    fn exact_q8_sidecar_runtime_hook_records_declined_selector_path() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[0.5f32, 1.0, 1.5, 2.0], (2, 2), &device)?;
        let linear = Linear::new(weight, None);
        let input = Tensor::from_slice(&[2.0f32, 3.0], (1, 1, 2), &device)?;
        let mut hooks = DenseLinearRuntimeHookRegistry::default();
        hooks.insert(
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR.to_string(),
            DenseLinearRuntimeHookDescriptor {
                tensor_name: "blk.0.attn_q.weight".to_string(),
                role: "AttentionQ".to_string(),
                sidecar_payload_sha256: Some("sha256:test".to_string()),
                packed_q8_payload: None,
                payload_order_matches_runtime_shape: false,
                source_order_q8_matvec_candidate: false,
                source_order_input_dim: None,
                source_order_output_dim: None,
                runtime_compute_enabled: false,
                receipt_bound_no_bias_selector: None,
            },
        );

        let instrumentation_before = dense_q8_sidecar_instrumentation_snapshot();
        let output = maybe_forward_dense_q8_sidecar_linear(
            &input,
            &linear,
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR,
            &hooks,
        )?;

        assert!(output.is_none());
        let instrumentation_after = dense_q8_sidecar_instrumentation_snapshot();
        assert!(
            instrumentation_after.selector_dispatch_calls
                > instrumentation_before.selector_dispatch_calls
        );
        assert!(
            instrumentation_after.selector_declined_calls
                > instrumentation_before.selector_declined_calls
        );
        Ok(())
    }

    #[test]
    fn exact_q8_sidecar_runtime_hook_selects_source_order_for_payload_order_mismatch() -> Result<()>
    {
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[0.5f32, 1.0, 1.5, 2.0], (2, 2), &device)?;
        let linear = Linear::new(weight, None);
        let input = Tensor::from_slice(&[2.0f32, 3.0], (1, 1, 2), &device)?;

        let mut packed = Vec::new();
        packed.extend_from_slice(&f32_to_fp16(0.5).to_le_bytes());
        packed.extend(std::iter::repeat_n(1_u8, 32));

        let mut hooks = DenseLinearRuntimeHookRegistry::default();
        hooks.insert(
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR.to_string(),
            DenseLinearRuntimeHookDescriptor {
                tensor_name: "blk.0.attn_q.weight".to_string(),
                role: "AttentionQ".to_string(),
                sidecar_payload_sha256: Some("sha256:test".to_string()),
                packed_q8_payload: Some(DenseLinearPackedQ8Payload {
                    tensor_name: "blk.0.attn_q.weight".to_string(),
                    packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
                    q8_block_size: 32,
                    q8_block_count: 1,
                    matrix_rows: 2,
                    matrix_cols: 2,
                }),
                payload_order_matches_runtime_shape: false,
                source_order_q8_matvec_candidate: true,
                source_order_input_dim: Some(2),
                source_order_output_dim: Some(2),
                runtime_compute_enabled: true,
                receipt_bound_no_bias_selector: None,
            },
        );

        let descriptor = hooks.get(SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR).ok_or_else(|| {
            BitNetError::Config(format!(
                "missing test hook descriptor for {SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR}"
            ))
        })?;
        let boundary = DenseLinearRuntimeHookBoundary::from_sidecar_descriptor(
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR,
            descriptor,
        );
        assert_eq!(boundary.selected_path, "source_order_q8_0_qproj_matvec");
        assert_eq!(boundary.selected_kernel, "dense-q8-source-order-qproj-matvec");
        assert!(boundary.sidecar_payload_contract_valid);
        assert!(!boundary.sidecar_payload_order_matches_runtime_shape);
        assert!(boundary.source_order_q8_matvec_candidate);
        assert_eq!(
            boundary.source_order_candidate_receipt_identity.as_deref(),
            Some("layers.0.attention.q_proj.weight:source_order_q8_0_qproj_matvec:runtime_enabled")
        );
        assert_eq!(boundary.source_order_input_dim, Some(2));
        assert_eq!(boundary.source_order_output_dim, Some(2));
        assert!(boundary.source_order_candidate_runtime_enabled);
        assert!(!boundary.runtime_compute_enabled);
        assert!(!boundary.eager_f32_runtime_preserved);
        assert!(boundary.dense_runtime_replaced);

        let output = maybe_forward_dense_q8_sidecar_linear(
            &input,
            &linear,
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR,
            &hooks,
        )?;
        assert!(output.is_some());
        Ok(())
    }

    #[test]
    fn source_order_qproj_candidate_matvec_uses_runtime_row_mapping() -> Result<()> {
        let mut packed = Vec::new();
        packed.extend_from_slice(&f32_to_fp16(0.5).to_le_bytes());
        for value in [1i8, 2, 3, 4, 5, 6] {
            packed.push(value as u8);
        }
        packed.resize(34, 0);
        let payload = DenseLinearPackedQ8Payload {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
            q8_block_size: 32,
            q8_block_count: 1,
            matrix_rows: 2,
            matrix_cols: 3,
        };
        let input = [10.0f32, 20.0];
        let mut output = vec![0.0f32; 3];

        dense_q8_source_order_qproj_matvec_into(&input, None, &payload, 2, 3, &mut output)?;

        assert_eq!(output, vec![25.0, 55.0, 85.0]);
        Ok(())
    }

    #[test]
    fn source_order_qproj_candle_slice_compare_uses_runtime_weight_rows() -> Result<()> {
        let mut packed = Vec::new();
        packed.extend_from_slice(&f32_to_fp16(0.5).to_le_bytes());
        for value in [1i8, 2, 3, 4, 5, 6] {
            packed.push(value as u8);
        }
        packed.resize(34, 0);
        let payload = DenseLinearPackedQ8Payload {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
            q8_block_size: 32,
            q8_block_count: 1,
            matrix_rows: 2,
            matrix_cols: 3,
        };
        let input = [10.0f32, 20.0];
        let candle_weight = [0.5f32, 2.0, 1.0, 2.5, 1.5, 3.0];

        let (candle_output, terms_json) = dense_q8_source_order_qproj_candle_slice_compare_entry(
            &input,
            0.0,
            &payload,
            &candle_weight,
            2,
            3,
            1,
        )?;

        assert_eq!(candle_output, 60.0);
        assert!(terms_json.contains("\"source_weight_idx\":1"));
        assert!(terms_json.contains("\"candle_weight_idx\":2"));
        assert!(terms_json.contains("\"source_order_weight_value\":1.000000000"));
        assert!(terms_json.contains("\"candle_weight_value\":1.000000000"));
        Ok(())
    }

    #[test]
    fn source_order_qproj_accumulator_audit_uses_runtime_row_mapping() -> Result<()> {
        let mut packed = Vec::new();
        packed.extend_from_slice(&f32_to_fp16(0.5).to_le_bytes());
        for value in [1i8, 2, 3, 4, 5, 6] {
            packed.push(value as u8);
        }
        packed.resize(34, 0);
        let payload = DenseLinearPackedQ8Payload {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
            q8_block_size: 32,
            q8_block_count: 1,
            matrix_rows: 2,
            matrix_cols: 3,
        };
        let input = [10.0f32, 20.0, 30.0];

        let (candidate_output, terms_json) =
            dense_q8_source_order_qproj_accumulator_audit_entry(&input, 0.0, &payload, 3, 2, 1)?;

        assert_eq!(candidate_output, 160.0);
        assert!(terms_json.contains("\"weight_idx\":3"));
        assert!(terms_json.contains("\"q\":4"));
        Ok(())
    }

    #[test]
    fn source_order_qproj_row_mapping_proof_reconciles_runtime_weight_rows() -> Result<()> {
        let mut packed = Vec::new();
        packed.extend_from_slice(&f32_to_fp16(0.5).to_le_bytes());
        for value in [1i8, 2, 3, 4, 5, 6] {
            packed.push(value as u8);
        }
        packed.resize(34, 0);
        let payload = DenseLinearPackedQ8Payload {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
            q8_block_size: 32,
            q8_block_count: 1,
            matrix_rows: 2,
            matrix_cols: 3,
        };
        let input = [10.0f32, 20.0, 30.0];
        let candle_weight = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0];

        let (mapped_output, candle_output, terms_json) =
            dense_q8_source_order_qproj_row_mapping_proof_entry(
                &input,
                0.0,
                &payload,
                &candle_weight,
                3,
                2,
                1,
            )?;

        assert_eq!(mapped_output, 160.0);
        assert_eq!(candle_output, 160.0);
        assert!(terms_json.contains("\"source_order_weight_idx\":1"));
        assert!(terms_json.contains("\"runtime_weight_idx\":3"));
        assert!(terms_json.contains("\"mapped_weight_value\":2.000000000"));
        assert!(terms_json.contains("\"candle_weight_value\":2.000000000"));
        Ok(())
    }

    #[test]
    fn exact_q8_sidecar_runtime_hook_matches_reference_across_q8_blocks() -> Result<()> {
        let device = Device::Cpu;
        let mut weight_values = Vec::new();
        for row in 0..2 {
            for col in 0..64 {
                let q = if row == 0 { (col as i8) - 32 } else { 31 - (col as i8) };
                weight_values.push(f32::from(q) * 0.25);
            }
        }
        let weight = Tensor::from_vec(weight_values, (2, 64), &device)?;
        let linear = Linear::new(weight, None);
        let input_values: Vec<f32> = (0..64)
            .map(|idx| match idx % 5 {
                0 => -1.0,
                1 => -0.5,
                2 => 0.25,
                3 => 0.75,
                _ => 1.5,
            })
            .collect();
        let input = Tensor::from_vec(input_values, (1, 1, 64), &device)?;

        let mut packed = Vec::new();
        for block_idx in 0..4 {
            packed.extend_from_slice(&f32_to_fp16(0.25).to_le_bytes());
            for offset in 0..32 {
                let flat_idx = block_idx * 32 + offset;
                let row = flat_idx / 64;
                let col = flat_idx % 64;
                let q = if row == 0 { (col as i8) - 32 } else { 31 - (col as i8) };
                packed.push(q as u8);
            }
        }

        let mut hooks = DenseLinearRuntimeHookRegistry::default();
        hooks.insert(
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR.to_string(),
            DenseLinearRuntimeHookDescriptor {
                tensor_name: "blk.0.attn_q.weight".to_string(),
                role: "AttentionQ".to_string(),
                sidecar_payload_sha256: Some("sha256:test".to_string()),
                packed_q8_payload: Some(DenseLinearPackedQ8Payload {
                    tensor_name: "blk.0.attn_q.weight".to_string(),
                    packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
                    q8_block_size: 32,
                    q8_block_count: 4,
                    matrix_rows: 2,
                    matrix_cols: 64,
                }),
                payload_order_matches_runtime_shape: true,
                source_order_q8_matvec_candidate: false,
                source_order_input_dim: None,
                source_order_output_dim: None,
                runtime_compute_enabled: true,
                receipt_bound_no_bias_selector: None,
            },
        );

        let Some(output) = maybe_forward_dense_q8_sidecar_linear(
            &input,
            &linear,
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR,
            &hooks,
        )?
        else {
            return Err(BitNetError::Validation(
                "expected exact Q8 sidecar runtime hook to run".to_string(),
            ));
        };

        assert_eq!(output.dims(), &[1, 1, 2]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            linear.forward(&input)?.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn exact_q8_sidecar_runtime_hook_matches_reference_when_rows_split_q8_blocks() -> Result<()> {
        let device = Device::Cpu;
        let matrix_rows = 2;
        let matrix_cols = 40;
        let q8_block_size = 32;
        let q8_block_count = 3;
        let scales = [0.125f32, 0.25, 0.5];
        let mut weight_values = Vec::new();
        let mut packed = Vec::new();

        for (block_idx, scale) in scales.iter().enumerate().take(q8_block_count) {
            packed.extend_from_slice(&f32_to_fp16(*scale).to_le_bytes());
            for offset in 0..q8_block_size {
                let flat_idx = block_idx * q8_block_size + offset;
                let q = ((flat_idx % 17) as i8) - 8;
                packed.push(q as u8);
                if flat_idx < matrix_rows * matrix_cols {
                    weight_values.push(*scale * f32::from(q));
                }
            }
        }

        let weight = Tensor::from_vec(weight_values, (matrix_rows, matrix_cols), &device)?;
        let linear = Linear::new(weight, None);
        let input_values: Vec<f32> = (0..matrix_cols)
            .map(|idx| match idx % 6 {
                0 => -1.25,
                1 => -0.75,
                2 => -0.25,
                3 => 0.25,
                4 => 0.75,
                _ => 1.25,
            })
            .collect();
        let input = Tensor::from_vec(input_values, (1, 1, matrix_cols), &device)?;

        let mut hooks = DenseLinearRuntimeHookRegistry::default();
        hooks.insert(
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR.to_string(),
            DenseLinearRuntimeHookDescriptor {
                tensor_name: "blk.0.attn_q.weight".to_string(),
                role: "AttentionQ".to_string(),
                sidecar_payload_sha256: Some("sha256:test".to_string()),
                packed_q8_payload: Some(DenseLinearPackedQ8Payload {
                    tensor_name: "blk.0.attn_q.weight".to_string(),
                    packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
                    q8_block_size,
                    q8_block_count,
                    matrix_rows,
                    matrix_cols,
                }),
                payload_order_matches_runtime_shape: true,
                source_order_q8_matvec_candidate: false,
                source_order_input_dim: None,
                source_order_output_dim: None,
                runtime_compute_enabled: true,
                receipt_bound_no_bias_selector: None,
            },
        );

        let Some(output) = maybe_forward_dense_q8_sidecar_linear(
            &input,
            &linear,
            SLM_CPU_067_EXACT_Q8_RUNTIME_TENSOR,
            &hooks,
        )?
        else {
            return Err(BitNetError::Validation(
                "expected exact Q8 sidecar runtime hook to run".to_string(),
            ));
        };

        assert_eq!(output.dims(), &[1, 1, matrix_rows]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            linear.forward(&input)?.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn dense_q8_sidecar_block_aligned_matvec_fills_caller_output_slice() -> Result<()> {
        let mut packed = Vec::new();
        packed.extend_from_slice(&f32_to_fp16(0.5).to_le_bytes());
        packed.extend(std::iter::repeat_n(1_u8, 32));
        packed.extend_from_slice(&f32_to_fp16(0.25).to_le_bytes());
        packed.extend(std::iter::repeat_n(2_u8, 32));
        let payload = DenseLinearPackedQ8Payload {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
            q8_block_size: 32,
            q8_block_count: 2,
            matrix_rows: 2,
            matrix_cols: 32,
        };
        let input = vec![1.0f32; 32];
        let bias = [1.0f32, -2.0];
        let mut caller_output = [f32::NAN; 2];

        dense_q8_sidecar_matvec_block_aligned_into(
            &input,
            Some(&bias),
            &payload,
            &mut caller_output,
        );

        assert_eq!(caller_output, [17.0, 14.0]);
        Ok(())
    }

    #[test]
    fn dense_q8_sidecar_generic_matvec_fills_caller_output_slice() -> Result<()> {
        let matrix_rows = 2;
        let matrix_cols = 40;
        let q8_block_size = 32;
        let q8_block_count = 3;
        let scales = [0.125f32, 0.25, 0.5];
        let mut packed = Vec::new();
        for (block_idx, scale) in scales.iter().enumerate().take(q8_block_count) {
            packed.extend_from_slice(&f32_to_fp16(*scale).to_le_bytes());
            for offset in 0..q8_block_size {
                let flat_idx = block_idx * q8_block_size + offset;
                let q = ((flat_idx % 17) as i8) - 8;
                packed.push(q as u8);
            }
        }
        let payload = DenseLinearPackedQ8Payload {
            tensor_name: "blk.0.attn_q.weight".to_string(),
            packed_q8_bytes: std::sync::Arc::from(packed.into_boxed_slice()),
            q8_block_size,
            q8_block_count,
            matrix_rows,
            matrix_cols,
        };
        let input_values: Vec<f32> = (0..matrix_cols)
            .map(|idx| match idx % 6 {
                0 => -1.25,
                1 => -0.75,
                2 => -0.25,
                3 => 0.25,
                4 => 0.75,
                _ => 1.25,
            })
            .collect();
        let mut expected = Vec::new();
        dense_q8_sidecar_matvec_generic(&input_values, None, &payload, &mut expected);
        let mut caller_output = vec![f32::NAN; expected.len()];

        dense_q8_sidecar_matvec_generic_into(&input_values, None, &payload, &mut caller_output);

        assert_eq!(caller_output, expected);
        Ok(())
    }

    #[test]
    fn attention_score_qk_inputs_match_reference_precision_contract() -> Result<()> {
        let device = Device::Cpu;
        let query =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 1, 4), &device)?;
        let key = Tensor::from_slice(&[5.0003f32, 6.0007, -7.1259, 8.2509], (1, 1, 1, 4), &device)?;

        let q_values = attention_f16_dot_input(&query)?.flatten_all()?.to_vec1::<f32>()?;
        let k_values = attention_score_key_input(&key)?.flatten_all()?.to_vec1::<f32>()?;
        let score = q_values.iter().zip(k_values.iter()).fold(0.0f32, |sum, (q, k)| sum + q * k);

        assert_eq!(q_values, vec![1.0, -2.0, 3.125, -4.25]);
        assert_eq!(k_values, vec![5.0003, 6.0007, -7.1259, 8.2509]);
        assert_eq!(score, -64.33586);

        Ok(())
    }

    fn slm_cpu_244_test_prompt_descriptor(
        model_architecture: &'static str,
        model_sha256: &'static str,
        candidate_path: &'static str,
    ) -> DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary {
        let callsite_identity =
            dense_linear_no_bias_feed_forward_apply_linear_callsite_identity(0, "down_proj");
        let descriptor = DenseLinearNoBiasPromptSessionDescriptor::from_prompt_session(
            DenseLinearNoBiasPromptSessionDescriptorInput {
                tensor_name: "layers.0.feed_forward.down_proj.weight",
                callsite_identity: callsite_identity.as_str(),
                model_sha256,
                model_architecture,
                quant_format: "Q8_0",
                tokenizer_source: "gguf_metadata",
                tokenizer_strict: true,
                runtime_api: "cpu",
                selected_backend: "cpu-rust",
                fallback_used: false,
                prompt_ids: &[1, 2, 3],
                prompt_ids_digest: "sha256:prompt",
                selected_path: "eager_f32_candle",
                selected_kernel: "dense-f32-candle-linear",
                candidate_path,
                candidate_kernel: SLM_CPU_APPLY_LINEAR_NO_BIAS_CANDIDATE_KERNEL,
                bias_present: Some(false),
                explicit_runtime_gate_requested: true,
            },
        );
        DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_prompt_session_descriptor(
            &descriptor,
        )
    }

    #[test]
    fn no_bias_apply_linear_dispatch_executes_candidate_when_descriptor_matches() -> Result<()> {
        reset_dense_linear_no_bias_candidate_instrumentation();
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[0.5f32, 1.0, 1.5, 2.0], (2, 2), &device)?;
        let linear = Linear::new(weight, None);
        let input = Tensor::from_slice(&[2.0f32, 3.0], (1, 1, 2), &device)?;
        let descriptor = slm_cpu_244_test_prompt_descriptor(
            "qwen3",
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            SLM_CPU_195_NO_BIAS_CANDIDATE_PATH,
        );

        let Some(output) = maybe_forward_feed_forward_no_bias_candidate_linear(
            &input,
            &linear,
            "down_proj",
            "layers.0.feed_forward.down_proj.weight",
            Some(&descriptor),
        )?
        else {
            return Err(BitNetError::Validation(
                "expected no-bias candidate dispatch to execute".to_string(),
            ));
        };

        assert_eq!(output.dims(), &[1, 1, 2]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            linear.forward(&input)?.flatten_all()?.to_vec1::<f32>()?
        );
        let snapshot = dense_linear_no_bias_candidate_instrumentation_snapshot();
        assert!(snapshot.selector_dispatch_calls >= 1);
        assert!(snapshot.selector_selected_calls >= 1);
        assert!(snapshot.candidate_forward_calls >= 1);
        Ok(())
    }

    #[test]
    fn no_bias_apply_linear_dispatch_preserves_default_without_descriptor() -> Result<()> {
        reset_dense_linear_no_bias_candidate_instrumentation();
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[0.5f32, 1.0, 1.5, 2.0], (2, 2), &device)?;
        let linear = Linear::new(weight, None);
        let input = Tensor::from_slice(&[2.0f32, 3.0], (1, 1, 2), &device)?;

        let output = maybe_forward_feed_forward_no_bias_candidate_linear(
            &input,
            &linear,
            "down_proj",
            "layers.0.feed_forward.down_proj.weight",
            None,
        )?;

        assert!(output.is_none());
        Ok(())
    }

    #[test]
    fn no_bias_apply_linear_dispatch_fails_closed_when_bias_is_present() -> Result<()> {
        reset_dense_linear_no_bias_candidate_instrumentation();
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[0.5f32, 1.0, 1.5, 2.0], (2, 2), &device)?;
        let bias = Tensor::zeros(2, DType::F32, &device)?;
        let linear = Linear::new(weight, Some(bias));
        let input = Tensor::from_slice(&[2.0f32, 3.0], (1, 1, 2), &device)?;
        let descriptor = slm_cpu_244_test_prompt_descriptor(
            "qwen2",
            SLM_CPU_QWEN25_Q8_MODEL_SHA256,
            SLM_CPU_QWEN25_NO_BIAS_CANDIDATE_PATH,
        );

        let error = maybe_forward_feed_forward_no_bias_candidate_linear(
            &input,
            &linear,
            "down_proj",
            "layers.0.feed_forward.down_proj.weight",
            Some(&descriptor),
        )
        .expect_err("bias-present linear must fail closed");

        let message = error.to_string();
        assert!(
            message.contains("bias_present_true"),
            "expected bias_present_true blocker, got {message}"
        );
        let snapshot = dense_linear_no_bias_candidate_instrumentation_snapshot();
        assert!(snapshot.selector_dispatch_calls >= 1);
        assert!(snapshot.selector_error_calls >= 1);
        Ok(())
    }

    #[test]
    fn attention_value_mix_uses_f16_roundtrip_values() -> Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::from_slice(&[0.25f32, 0.25, 0.25, 0.25], (1, 1, 1, 4), &device)?;
        let values =
            Tensor::from_slice(&[1.0003f32, -2.0007, 3.1259, -4.2509], (1, 1, 4, 1), &device)?;

        let rounded_values = attention_f16_dot_input(&values)?;
        let mixed = weights.matmul(&rounded_values)?;
        let mixed = mixed.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(mixed, vec![-0.53125]);
        Ok(())
    }

    #[test]
    fn test_layer_norm_with_small_gamma() -> candle_core::Result<()> {
        // Test RMSNorm with gamma RMS ≈ 0.018 (our model's case)
        let device = Device::Cpu;
        let hidden_size = 2560;
        let eps = 1e-5;

        // Create input tensor [1, 1, 2560]
        let input_data: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let x = i as f32 / hidden_size as f32;
                ((x * 10.0).sin() + (x * 20.0).cos()) * 0.5
            })
            .collect();

        let input = Tensor::from_slice(&input_data, (1, 1, hidden_size), &device)?;

        // Create gamma with RMS ≈ 1/√2560 ≈ 0.01976
        let target_rms = 1.0 / (hidden_size as f64).sqrt();
        let gamma_data: Vec<f32> = vec![target_rms as f32; hidden_size];
        let gamma = Tensor::from_slice(&gamma_data, hidden_size, &device)?;

        // Verify gamma RMS
        let gamma_rms = compute_rms(&gamma)?;
        assert!(
            (gamma_rms - target_rms).abs() < 0.001,
            "Gamma RMS should be close to {:.6e}, got {:.6e}",
            target_rms,
            gamma_rms
        );

        // Apply RMSNorm
        let rms_norm = RmsNorm::new(gamma, eps);
        let output = rms_norm.forward(&input)?;

        // Verify output RMS is smaller but reasonable
        let output_rms = compute_rms(&output)?;

        assert!(
            output_rms > 0.001 && output_rms < 0.1,
            "Output RMS should be reasonable with small gamma, got {:.6e}",
            output_rms
        );

        // Verify no NaN/Inf
        let vec_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let has_nan = vec_data.iter().any(|x| x.is_nan());
        let has_inf = vec_data.iter().any(|x| x.is_infinite());
        assert!(!has_nan, "Output should not contain NaN");
        assert!(!has_inf, "Output should not contain Inf");

        Ok(())
    }

    #[test]
    #[serial]
    fn test_layer_norm_with_optional_bias() -> candle_core::Result<()> {
        // Test layer_norm_with_optional_bias helper with no-bias LayerNorm path
        let device = Device::Cpu;
        let hidden_size = 128;
        let eps = 1e-5;

        // Create VarBuilder with only weight (no bias)
        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let weight = Tensor::ones(hidden_size, DType::F32, &device)?;
        tensors.insert("weight".to_string(), weight);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        // Create LayerNorm (should use no-bias LayerNorm path due to missing bias)
        let layer_norm = layer_norm_with_optional_bias(hidden_size, eps, vb)?;

        // Test forward pass
        let input_data: Vec<f32> =
            (0..hidden_size).map(|i| (i as f32 / hidden_size as f32).sin()).collect();
        let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

        let output = layer_norm.forward(&input)?;

        // Verify output shape
        assert_eq!(output.shape(), input.shape());

        // Verify no NaN/Inf
        let vec_data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let has_nan = vec_data.iter().any(|x| x.is_nan());
        let has_inf = vec_data.iter().any(|x| x.is_infinite());
        assert!(!has_nan, "Output should not contain NaN");
        assert!(!has_inf, "Output should not contain Inf");

        Ok(())
    }

    #[test]
    fn test_linear_with_optional_bias_uses_no_bias_path() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let input = Tensor::from_slice(&[5.0f32, 6.0], (1, 2), &device)?;

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), weight.clone());
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let no_bias = linear_with_optional_bias(2, 2, vb)?;
        let no_bias_output = no_bias.forward(&input)?;

        let explicit_zero_bias = Linear::new(weight, Some(Tensor::zeros(2, DType::F32, &device)?));
        let zero_bias_output = explicit_zero_bias.forward(&input)?;

        assert_eq!(no_bias_output.to_vec2::<f32>()?, zero_bias_output.to_vec2::<f32>()?);
        assert_eq!(no_bias_output.to_vec2::<f32>()?, vec![vec![17.0, 39.0]]);

        Ok(())
    }

    #[test]
    fn test_rms_norm_type_does_not_remove_mean() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let hidden_size = 3;
        let eps = 1e-5;

        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), Tensor::ones(hidden_size, DType::F32, &device)?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let norm = norm_with_optional_bias(NormType::RmsNorm, hidden_size, eps, vb)?;

        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, hidden_size), &device)?;
        let output = norm.forward(&input)?;
        let values = output.to_vec2::<f32>()?;

        assert!(
            values[0].iter().all(|value| *value > 0.0),
            "RMSNorm should preserve positive sign instead of mean-centering: {values:?}"
        );

        Ok(())
    }

    #[test]
    fn transposed_lm_head_uses_dedicated_output_weight() -> Result<()> {
        use std::collections::HashMap;

        let device = Device::Cpu;
        let vocab_size = 4;
        let hidden_size = 2;
        let mut config = BitNetConfig::default();
        config.model.vocab_size = vocab_size;
        config.model.hidden_size = hidden_size;
        config.model.num_layers = 0;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::zeros((vocab_size, hidden_size), DType::F32, &device)?,
        );
        tensors.insert(
            "final_norm.weight".to_string(),
            Tensor::ones(hidden_size, DType::F32, &device)?,
        );
        tensors.insert(
            "final_norm.bias".to_string(),
            Tensor::zeros(hidden_size, DType::F32, &device)?,
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            Tensor::from_slice(
                &[
                    1.0f32, 0.0, 0.0, 0.0, // hidden dim 0 -> token 0
                    0.0, 1.0, 0.0, 0.0, // hidden dim 1 -> token 1
                ],
                (hidden_size, vocab_size),
                &device,
            )?,
        );
        tensors.insert(
            "lm_head.transposed".to_string(),
            Tensor::from_slice(&[1.0f32], (1,), &device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(
            model.lm_head_transposed,
            "transposed lm_head flag must survive model construction"
        );
        assert!(
            model.lm_head_weight.is_some(),
            "transposed lm_head must keep its dedicated output weight"
        );
        assert!(
            model.embed_tied_weight.is_none(),
            "dedicated transposed lm_head must not fall back to tied embeddings"
        );

        let hidden = Tensor::from_slice(&[2.0f32, 3.0], (1, hidden_size), &device)?;
        let logits = model.logits(&hidden)?;
        let values = logits.to_vec2::<f32>()?;
        assert_eq!(values, vec![vec![2.0, 3.0, 0.0, 0.0]]);

        Ok(())
    }

    #[test]
    fn tied_embedding_logits_use_cached_embedding_transpose() -> Result<()> {
        use std::collections::HashMap;

        let device = Device::Cpu;
        let vocab_size = 4;
        let hidden_size = 3;
        let mut config = BitNetConfig::default();
        config.model.vocab_size = vocab_size;
        config.model.hidden_size = hidden_size;
        config.model.num_layers = 0;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_slice(
                &[
                    1.0f32, 0.0, 0.0, // token 0
                    0.0, 1.0, 0.0, // token 1
                    0.0, 0.0, 1.0, // token 2
                    1.0, 1.0, 1.0, // token 3
                ],
                (vocab_size, hidden_size),
                &device,
            )?,
        );
        tensors.insert(
            "final_norm.weight".to_string(),
            Tensor::ones(hidden_size, DType::F32, &device)?,
        );
        tensors.insert(
            "final_norm.bias".to_string(),
            Tensor::zeros(hidden_size, DType::F32, &device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(
            model.lm_head.is_none() && model.lm_head_weight.is_none(),
            "missing lm_head must use tied embeddings"
        );
        assert!(
            model.embed_tied_weight.is_some(),
            "tied embedding logits should cache [hidden, vocab] weight"
        );

        let hidden_2d = Tensor::from_slice(&[2.0f32, 3.0, 5.0], (1, hidden_size), &device)?;
        let logits_2d = model.logits(&hidden_2d)?;
        assert_eq!(logits_2d.to_vec2::<f32>()?, vec![vec![2.0, 3.0, 5.0, 10.0]]);

        let hidden_3d = Tensor::from_slice(
            &[
                2.0f32, 3.0, 5.0, // step 0
                7.0, 11.0, 13.0, // step 1
            ],
            (1, 2, hidden_size),
            &device,
        )?;
        let logits_3d = model.logits(&hidden_3d)?;
        assert_eq!(
            logits_3d.to_vec3::<f32>()?,
            vec![vec![vec![2.0, 3.0, 5.0, 10.0], vec![7.0, 11.0, 13.0, 31.0]]]
        );

        Ok(())
    }

    #[test]
    fn kv_cache_can_be_bounded_below_model_context() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.num_layers = 2;
        config.model.num_heads = 4;
        config.model.num_key_value_heads = 2;
        config.model.hidden_size = 16;
        config.model.max_position_embeddings = 128;

        let cache = KVCache::new_with_max_seq_len(&config, 1, &device, 12)?;

        assert_eq!(cache.layers.len(), 2);
        for layer in &cache.layers {
            assert_eq!(layer.max_seq_len, 12);
            assert_eq!(layer.k.dims(), &[1, 2, 12, 4]);
            assert_eq!(layer.v.dims(), &[1, 2, 12, 4]);
        }
        let bytes_per_f32 = std::mem::size_of::<f32>() as u128;
        assert_eq!(
            KVCache::estimated_f32_bytes_for_max_seq_len(&config, 1, 12)?,
            2u128 * 2 * 2 * 12 * 4 * bytes_per_f32
        );

        Ok(())
    }

    #[test]
    fn kv_cache_rejects_bounded_context_above_model_context() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.num_layers = 1;
        config.model.num_heads = 2;
        config.model.num_key_value_heads = 1;
        config.model.hidden_size = 8;
        config.model.max_position_embeddings = 16;

        let err = match KVCache::new_with_max_seq_len(&config, 1, &device, 17) {
            Ok(_) => {
                return Err(BitNetError::Validation(
                    "bounded KV cache accepted a request past model context".to_string(),
                ));
            }
            Err(err) => err,
        };
        assert!(err.to_string().contains("exceeds model context"), "unexpected error: {err}");

        Ok(())
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_layer_norm_requires_bias_when_guard_enabled() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let hidden_size = 64;
        let eps = 1e-5;

        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let weight = Tensor::ones(hidden_size, DType::F32, &device)?;
        tensors.insert("weight".to_string(), weight);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let mut scope = bitnet_test_support::EnvScope::new();
        scope.set("BITNET_REQUIRE_LAYER_NORM_BIAS", "1");

        let err = layer_norm_with_optional_bias(hidden_size, eps, vb)
            .expect_err("missing bias should error when BITNET_REQUIRE_LAYER_NORM_BIAS=1");
        assert!(
            err.to_string().contains("LayerNorm bias tensor is required"),
            "unexpected error: {err}"
        );

        Ok(())
    }

    fn tiny_bitnet_config() -> BitNetConfig {
        BitNetConfig {
            model: ModelConfig {
                hidden_size: 2,
                vocab_size: 8,
                num_heads: 1,
                num_key_value_heads: 1,
                num_layers: 1,
                intermediate_size: 2,
                max_position_embeddings: 8,
                rms_norm_eps: Some(1e-5),
                norm_type: NormType::LayerNorm,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn identity_2(device: &Device) -> candle_core::Result<Tensor> {
        Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2], device)
    }

    #[test]
    #[serial]
    fn attention_applies_bitnet_sub_layernorm_before_output_projection() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_bitnet_config();
        let mut tensors = HashMap::new();
        for name in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"] {
            tensors.insert(name.to_string(), identity_2(&device)?);
        }
        tensors.insert("sub_layernorm.weight".to_string(), Tensor::ones(2, DType::F32, &device)?);
        tensors.insert("sub_layernorm.bias".to_string(), Tensor::zeros(2, DType::F32, &device)?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let attention = MultiHeadAttention::new(&config, vb, 0)?;
        let x = Tensor::from_vec(vec![1.0f32, 3.0], &[1, 1, 2], &device)?;
        let output = attention.forward(
            &x,
            None,
            &HashMap::new(),
            &DenseLinearRuntimeHookRegistry::new(),
            None,
        )?;
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;

        assert!(
            values[0] < -0.99 && values[0] > -1.01,
            "attention sub-layernorm should center first value near -1, got {}",
            values[0]
        );
        assert!(
            values[1] > 0.99 && values[1] < 1.01,
            "attention sub-layernorm should center second value near 1, got {}",
            values[1]
        );

        Ok(())
    }

    #[test]
    #[serial]
    fn feed_forward_applies_bitnet_sub_layernorm_before_down_projection() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_bitnet_config();
        let mut tensors = HashMap::new();
        for name in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            tensors.insert(name.to_string(), identity_2(&device)?);
        }
        tensors.insert("sub_layernorm.weight".to_string(), Tensor::ones(2, DType::F32, &device)?);
        tensors.insert("sub_layernorm.bias".to_string(), Tensor::zeros(2, DType::F32, &device)?);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let feed_forward = FeedForward::new(&config, vb, 0)?;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 2], &device)?;
        let output =
            feed_forward.forward(&x, &HashMap::new(), &DenseLinearRuntimeHookRegistry::new())?;
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;

        assert!(
            values[0] < -0.99 && values[0] > -1.01,
            "feed-forward sub-layernorm should center first value near -1, got {}",
            values[0]
        );
        assert!(
            values[1] > 0.99 && values[1] < 1.01,
            "feed-forward sub-layernorm should center second value near 1, got {}",
            values[1]
        );

        Ok(())
    }

    #[test]
    fn feed_forward_uses_relu2_activation_when_configured() -> Result<()> {
        let device = Device::Cpu;
        let mut config = tiny_bitnet_config();
        config.model.activation_type = ActivationType::Relu2;
        let mut tensors = HashMap::new();
        for name in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            tensors.insert(name.to_string(), identity_2(&device)?);
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let feed_forward = FeedForward::new(&config, vb, 0)?;
        let x = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 2], &device)?;
        let output =
            feed_forward.forward(&x, &HashMap::new(), &DenseLinearRuntimeHookRegistry::new())?;
        let values: Vec<f32> = output.flatten_all()?.to_vec1()?;

        assert!(
            (values[0] - 1.0).abs() < 1e-5,
            "relu2 gate should produce first output 1, got {}",
            values[0]
        );
        assert!(
            (values[1] - 8.0).abs() < 1e-5,
            "relu2 gate should square ReLU before multiplying by up projection, got {}",
            values[1]
        );

        Ok(())
    }

    #[test]
    fn qk256_inline_scale_reads_sibling_raw_tensor() -> Result<()> {
        let device = Device::Cpu;
        let mut raw_tensors = HashMap::new();
        raw_tensors.insert(
            "layers.0.attention.q_proj.weight.qk256_scale".to_string(),
            Tensor::from_vec(vec![0.25f32], &[1], &device)?,
        );

        let scale = qk256_inline_scale(&raw_tensors, "layers.0.attention.q_proj.weight.qk256_qs")?;

        assert_eq!(scale, Some(0.25));

        Ok(())
    }

    #[test]
    fn logits_prefer_raw_qk256_tied_embeddings_when_present() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 256;
        config.model.vocab_size = 2;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 256;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::zeros((2, 256), DType::F32, &device)?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(256, DType::F32, &device)?);

        let mut raw_tensors = HashMap::new();
        let mut packed = vec![0x00u8; 64];
        packed.extend(std::iter::repeat_n(0xAAu8, 64));
        raw_tensors.insert(
            TIED_EMBED_QK256_KEY.to_string(),
            Tensor::from_raw_buffer(&packed, DType::U8, &[2, 64], &device)?,
        );
        raw_tensors.insert(
            "embed_tokens.weight.qk256_scale".to_string(),
            Tensor::from_vec(vec![1.0f32], &[1], &device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, raw_tensors)?;
        let hidden = Tensor::ones((1, 256), DType::F32, &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;

        assert!(
            logits[0][0] < -200.0,
            "first raw QK256 tied logit should come from packed code-0 row, got {}",
            logits[0][0]
        );
        assert!(
            logits[0][1] > 200.0,
            "second raw QK256 tied logit should come from packed code-2 row, got {}",
            logits[0][1]
        );

        Ok(())
    }

    #[test]
    fn transformer_uses_dedicated_lm_head_when_present() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 4;
        config.model.vocab_size = 3;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 4;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_vec(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, //
                    0.0, 10.0, 0.0, 0.0, //
                    0.0, 0.0, 10.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(4, DType::F32, &device)?);
        tensors.insert(
            "lm_head.weight".to_string(),
            Tensor::from_vec(
                vec![
                    0.0f32, 0.0, 0.0, 0.0, //
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(model.lm_head.is_some(), "dedicated lm_head.weight must be loaded");
        assert!(
            model.embed_tied_weight.is_none(),
            "explicit lm_head.weight must not fall back to tied embeddings"
        );

        let hidden = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;
        assert!(
            logits[0][1] > logits[0][0],
            "logits should come from dedicated lm_head, got {:?}",
            logits[0]
        );

        Ok(())
    }

    #[test]
    fn transformer_uses_dedicated_output_weight_when_present() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 4;
        config.model.vocab_size = 3;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 4;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_vec(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, //
                    0.0, 10.0, 0.0, 0.0, //
                    0.0, 0.0, 10.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(4, DType::F32, &device)?);
        tensors.insert(
            "output.weight".to_string(),
            Tensor::from_vec(
                vec![
                    0.0f32, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, //
                    1.0, 0.0, 0.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(
            model.lm_head.is_some(),
            "dedicated GGUF output.weight must be loaded as the lm head"
        );
        assert!(
            model.embed_tied_weight.is_none(),
            "explicit output.weight must not fall back to tied embeddings"
        );

        let hidden = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;
        assert!(
            logits[0][2] > logits[0][0],
            "logits should come from dedicated output.weight, got {:?}",
            logits[0]
        );

        Ok(())
    }

    #[test]
    fn transformer_reshapes_gguf_output_weight_hidden_vocab_without_transpose() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 4;
        config.model.vocab_size = 3;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 4;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_vec(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, //
                    0.0, 10.0, 0.0, 0.0, //
                    0.0, 0.0, 10.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(4, DType::F32, &device)?);
        tensors.insert(
            "output.weight".to_string(),
            Tensor::from_vec(
                vec![
                    0.0f32, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, //
                    1.0, 0.0, 0.0, 0.0,
                ],
                (4, 3),
                &device,
            )?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(
            model.lm_head.is_some(),
            "GGUF output.weight with hidden/vocab dims must be reshaped into a dedicated lm head"
        );
        assert!(
            !model.lm_head_transposed,
            "GGUF output.weight hidden/vocab dims are token-major storage, not a transposed head"
        );
        assert!(
            model.embed_tied_weight.is_none(),
            "explicit output.weight must not fall back to tied embeddings"
        );
        assert_eq!(
            model.lm_head_weight.as_ref().map(|weight| weight.dims().to_vec()),
            Some(vec![3, 4]),
            "reshaped output.weight should be stored as [vocab, hidden]"
        );

        let hidden = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;
        assert_eq!(logits, vec![vec![0.0, 0.0, 1.0]]);
        assert!(
            logits[0][2] > logits[0][0],
            "hidden/vocab output.weight must be reshaped without transposing values, got {:?}",
            logits[0]
        );

        let hidden = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            (1, 2, 4),
            &device,
        )?;
        let logits = model.logits(&hidden)?.to_vec3::<f32>()?;
        assert_eq!(logits, vec![vec![vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 0.0]]]);

        Ok(())
    }

    #[test]
    fn transformer_keeps_hidden_vocab_lm_head_transposed() -> Result<()> {
        let device = Device::Cpu;
        let mut config = BitNetConfig::default();
        config.model.hidden_size = 4;
        config.model.vocab_size = 3;
        config.model.num_layers = 0;
        config.model.num_heads = 1;
        config.model.num_key_value_heads = 1;
        config.model.intermediate_size = 4;
        config.model.rms_norm_eps = Some(1e-5);
        config.model.norm_type = NormType::RmsNorm;

        let mut tensors = HashMap::new();
        tensors.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_vec(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, //
                    0.0, 10.0, 0.0, 0.0, //
                    0.0, 0.0, 10.0, 0.0,
                ],
                (3, 4),
                &device,
            )?,
        );
        tensors.insert("final_norm.weight".to_string(), Tensor::ones(4, DType::F32, &device)?);
        tensors.insert(
            "lm_head.weight".to_string(),
            Tensor::from_vec(
                vec![
                    0.0f32, 1.0, 0.0, //
                    0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0,
                ],
                (4, 3),
                &device,
            )?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = TransformerModel::new_with_tensors(config, vb, HashMap::new())?;

        assert!(
            model.lm_head.is_none(),
            "hidden/vocab lm_head.weight should use direct matmul, not Linear"
        );
        assert!(
            model.lm_head_transposed,
            "hidden/vocab lm_head.weight must remain marked as a true transposed head"
        );
        assert_eq!(
            model.lm_head_weight.as_ref().map(|weight| weight.dims().to_vec()),
            Some(vec![4, 3])
        );

        let hidden = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &device)?;
        let logits = model.logits(&hidden)?.to_vec2::<f32>()?;
        assert_eq!(logits, vec![vec![0.0, 1.0, 0.0]]);

        let hidden = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0,
            ],
            (1, 2, 4),
            &device,
        )?;
        let logits = model.logits(&hidden)?.to_vec3::<f32>()?;
        assert_eq!(logits, vec![vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0]]]);

        Ok(())
    }

    #[test]
    fn test_rmsnorm_formula_consistency() -> candle_core::Result<()> {
        // Verify RMSNorm formula: output = (x / sqrt(mean(x²) + eps)) * gamma
        let device = Device::Cpu;
        let hidden_size = 256;
        let eps = 1e-5;

        // Create input
        let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 / 100.0).sin()).collect();
        let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

        // Create gamma
        let gamma = Tensor::ones(hidden_size, DType::F32, &device)?;

        // Apply RMSNorm via Candle
        let rms_norm = RmsNorm::new(gamma.clone(), eps);
        let output_candle = rms_norm.forward(&input)?;

        // Manually compute RMSNorm
        let squared = input.sqr()?;
        let mean_squared = squared.mean_keepdim(1)?; // Mean over last dimension
        let rms_denominator = (mean_squared + eps)?.sqrt()?;
        let normalized = input.broadcast_div(&rms_denominator)?;
        let output_manual = normalized.broadcast_mul(&gamma)?;

        // Compare outputs
        let diff = (output_candle.sub(&output_manual))?.abs()?;
        let diff_vec: Vec<f32> = diff.flatten_all()?.to_vec1()?;
        let max_diff = diff_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;

        assert!(
            max_diff < 1e-5,
            "Candle's RMSNorm should match manual computation: max_diff={:.6e}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_rmsnorm_output_scale_relationship() -> candle_core::Result<()> {
        // Test that output RMS scales proportionally with gamma RMS
        let device = Device::Cpu;
        let hidden_size = 256;
        let eps = 1e-5;

        // Create same input for both tests
        let input_data: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let x = i as f32 / hidden_size as f32;
                ((x * 10.0).sin() + (x * 20.0).cos()) * 2.0
            })
            .collect();
        let input = Tensor::from_slice(&input_data, (1, hidden_size), &device)?;

        // Test 1: Standard gamma (RMS ≈ 1.0)
        let gamma_std = Tensor::ones(hidden_size, DType::F32, &device)?;
        let rms_norm_std = RmsNorm::new(gamma_std.clone(), eps);
        let output_std = rms_norm_std.forward(&input)?;
        let output_std_rms = compute_rms(&output_std)?;

        // Test 2: Small gamma (RMS ≈ 0.02)
        let target_rms = 0.02;
        let gamma_small =
            Tensor::from_slice(&vec![target_rms as f32; hidden_size], hidden_size, &device)?;
        let rms_norm_small = RmsNorm::new(gamma_small.clone(), eps);
        let output_small = rms_norm_small.forward(&input)?;
        let output_small_rms = compute_rms(&output_small)?;

        // Verify scaling relationship
        let gamma_std_rms = compute_rms(&gamma_std)?;
        let gamma_small_rms = compute_rms(&gamma_small)?;
        let expected_ratio = gamma_small_rms / gamma_std_rms;
        let actual_ratio = output_small_rms / output_std_rms;

        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "Output RMS should scale with gamma RMS: expected ratio {:.6}, got {:.6}",
            expected_ratio,
            actual_ratio
        );

        Ok(())
    }

    fn slm_cpu_211_test_gate(
        model_sha256: &str,
        candidate_path: &'static str,
    ) -> DenseLinearNoBiasApplyLinearBeforeAfterReceiptGate {
        DenseLinearNoBiasApplyLinearBeforeAfterReceiptGate {
            tensor_name: "layers.0.feed_forward.down_proj.weight".to_string(),
            role_id: "layers.0.feed_forward.down_proj".to_string(),
            model_sha256: model_sha256.to_string(),
            quant_format: "Q8_0",
            manifest_sha256: "sha256:manifest".to_string(),
            layer_idx: 0,
            scope: "feed_forward",
            linear: "down_proj",
            bias_present: Some(false),
            runtime_gate_name: "BITNET_DENSE_LINEAR_NO_BIAS_RUNTIME",
            runtime_gate_requested_enabled: false,
            selected_path: "eager_f32_candle",
            selected_kernel: "dense-f32-candle-linear",
            candidate_path,
            candidate_kernel: "dense-f32-candle-linear-no-bias-candidate",
            runtime_api: "cpu",
            selected_backend: "cpu-rust",
            fallback_used: false,
            before_after_receipts_present: true,
            descriptor_callsite_identity_preserved: true,
            prompt_ids_digest_preserved: true,
            generated_ids_digest_preserved: true,
            decoded_text_digest_preserved: true,
            prompt_ids_digest: "sha256:prompt".to_string(),
            generated_ids_digest: "sha256:generated".to_string(),
            decoded_text_digest: "sha256:text".to_string(),
            normal_inference_runtime_selection_enabled: false,
            candidate_execution_enabled: false,
            decision: "before_after_receipt_gate_ready_runtime_disabled",
            reason: "strict_warm_session_identity_preserved",
            remaining_runtime_selection_blocker: "candidate_execution_still_disabled_until_explicit_runtime_selection_pr",
            fail_closed_conditions: Vec::new(),
            allocation_reduction_claim: false,
            timing_improvement_claim: false,
            speedup_claim: false,
        }
    }

    fn slm_cpu_216_ready_pair_gate(
        model_sha256: &str,
        model_architecture: &'static str,
        candidate_path: &'static str,
    ) -> DenseLinearNoBiasCandidateOffOnReceiptPairGate {
        let gate = slm_cpu_211_test_gate(model_sha256, candidate_path);
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            model_architecture,
            "gguf_metadata",
            true,
            "slm-cpu-209:before-after",
            true,
        );
        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                descriptor.tensor_name.clone(),
                "bitnet_transformer::FeedForward::apply_linear:layers.0.feed_forward.down_proj.weight",
                true,
                true,
                true,
            );

        DenseLinearNoBiasCandidateOffOnReceiptPairGate::from_per_callsite_emitter(
            &emitter, true, true, true, true, true,
        )
    }

    fn slm_cpu_220_ready_off_on_boundary(
        model_sha256: &str,
        model_architecture: &'static str,
        candidate_path: &'static str,
    ) -> DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary {
        let pair_gate =
            slm_cpu_216_ready_pair_gate(model_sha256, model_architecture, candidate_path);
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                true,
                true,
                true,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: true,
                    candidate_off_on_strict_receipts_present: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );
        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: true,
                    candidate_off_strict_receipt_present: true,
                    candidate_on_strict_receipt_present: true,
                    strict_receipts_bind_owner_identity: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary::from_same_callsite_receipt_emitter_boundary(
            &emitter,
            DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptInputs {
                candidate_off_strict_receipt_artifact_present: true,
                candidate_on_strict_receipt_artifact_present: true,
                candidate_off_receipt_binds_owner_identity: true,
                candidate_on_receipt_binds_owner_identity: true,
                candidate_off_on_same_callsite_identity: true,
                prompt_ids_preserved: true,
                generated_ids_preserved: true,
                decoded_text_preserved: true,
            },
        )
    }

    fn slm_cpu_223_blocked_strict_artifact_capture_boundary(
        model_sha256: &str,
        model_architecture: &'static str,
        candidate_path: &'static str,
    ) -> DenseLinearNoBiasStrictArtifactCaptureBoundary {
        let pair_gate =
            slm_cpu_216_ready_pair_gate(model_sha256, model_architecture, candidate_path);
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                false,
                false,
                false,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: false,
                    candidate_off_on_strict_receipts_present: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: true,
                    candidate_off_strict_receipt_present: false,
                    candidate_on_strict_receipt_present: false,
                    strict_receipts_bind_owner_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let receipts =
            DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary::from_same_callsite_receipt_emitter_boundary(
                &emitter,
                DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptInputs {
                    candidate_off_strict_receipt_artifact_present: false,
                    candidate_on_strict_receipt_artifact_present: false,
                    candidate_off_receipt_binds_owner_identity: false,
                    candidate_on_receipt_binds_owner_identity: false,
                    candidate_off_on_same_callsite_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: true,
                    default_runtime_path_preserved: true,
                },
            );
        let artifact_pair =
            DenseLinearNoBiasStrictReceiptArtifactPairBoundary::from_receipt_gated_candidate_execution_boundary(
                &execution,
                DenseLinearNoBiasStrictReceiptArtifactPairInputs {
                    candidate_off_strict_receipt_artifact_path: None,
                    candidate_on_strict_receipt_artifact_path: None,
                    candidate_off_strict_receipt_artifact_present: false,
                    candidate_on_strict_receipt_artifact_present: false,
                    candidate_off_receipt_binds_gate_identity: false,
                    candidate_on_receipt_binds_gate_identity: false,
                    candidate_off_receipt_binds_descriptor_identity: false,
                    candidate_on_receipt_binds_descriptor_identity: false,
                    candidate_off_receipt_binds_owner_callsite_identity: false,
                    candidate_on_receipt_binds_owner_callsite_identity: false,
                    candidate_off_on_same_callsite_identity: false,
                    candidate_off_on_same_prompt_digest: false,
                    candidate_off_on_same_generated_digest: false,
                    candidate_off_on_same_decoded_text_digest: false,
                    candidate_off_on_same_model_backend_identity: false,
                    default_runtime_path_preserved: true,
                },
            );

        DenseLinearNoBiasStrictArtifactCaptureBoundary::from_strict_receipt_artifact_pair_boundary(
            &artifact_pair,
            DenseLinearNoBiasStrictArtifactCaptureInputs {
                candidate_off_capture_artifact_validated: false,
                candidate_on_capture_artifact_validated: false,
                candidate_off_capture_command_recorded: false,
                candidate_on_capture_command_recorded: false,
                candidate_off_on_capture_same_callsite_identity: false,
                candidate_off_on_capture_same_prompt_digest: false,
                candidate_off_on_capture_same_generated_digest: false,
                candidate_off_on_capture_same_decoded_text_digest: false,
                candidate_off_on_capture_same_model_backend_identity: false,
                capture_blocker_recorded: true,
                default_runtime_path_preserved: true,
            },
        )
    }

    fn slm_cpu_223_ready_strict_artifact_capture_boundary(
        model_sha256: &str,
        model_architecture: &'static str,
        candidate_path: &'static str,
    ) -> DenseLinearNoBiasStrictArtifactCaptureBoundary {
        let receipts =
            slm_cpu_220_ready_off_on_boundary(model_sha256, model_architecture, candidate_path);
        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: true,
                    default_runtime_path_preserved: true,
                },
            );
        let artifact_pair =
            DenseLinearNoBiasStrictReceiptArtifactPairBoundary::from_receipt_gated_candidate_execution_boundary(
                &execution,
                DenseLinearNoBiasStrictReceiptArtifactPairInputs {
                    candidate_off_strict_receipt_artifact_path: Some("ci/slm-cpu/candidate-off.json"),
                    candidate_on_strict_receipt_artifact_path: Some("ci/slm-cpu/candidate-on.json"),
                    candidate_off_strict_receipt_artifact_present: true,
                    candidate_on_strict_receipt_artifact_present: true,
                    candidate_off_receipt_binds_gate_identity: true,
                    candidate_on_receipt_binds_gate_identity: true,
                    candidate_off_receipt_binds_descriptor_identity: true,
                    candidate_on_receipt_binds_descriptor_identity: true,
                    candidate_off_receipt_binds_owner_callsite_identity: true,
                    candidate_on_receipt_binds_owner_callsite_identity: true,
                    candidate_off_on_same_callsite_identity: true,
                    candidate_off_on_same_prompt_digest: true,
                    candidate_off_on_same_generated_digest: true,
                    candidate_off_on_same_decoded_text_digest: true,
                    candidate_off_on_same_model_backend_identity: true,
                    default_runtime_path_preserved: true,
                },
            );

        DenseLinearNoBiasStrictArtifactCaptureBoundary::from_strict_receipt_artifact_pair_boundary(
            &artifact_pair,
            DenseLinearNoBiasStrictArtifactCaptureInputs {
                candidate_off_capture_artifact_validated: true,
                candidate_on_capture_artifact_validated: true,
                candidate_off_capture_command_recorded: true,
                candidate_on_capture_command_recorded: true,
                candidate_off_on_capture_same_callsite_identity: true,
                candidate_off_on_capture_same_prompt_digest: true,
                candidate_off_on_capture_same_generated_digest: true,
                candidate_off_on_capture_same_decoded_text_digest: true,
                candidate_off_on_capture_same_model_backend_identity: true,
                capture_blocker_recorded: false,
                default_runtime_path_preserved: true,
            },
        )
    }

    fn slm_cpu_230_ready_runtime_hook_attachment(
        model_sha256: &str,
        model_architecture: &'static str,
        candidate_path: &'static str,
        receipt_pair_identity: &'static str,
    ) -> DenseLinearNoBiasRuntimeHookAttachmentBoundary {
        let capture = slm_cpu_223_ready_strict_artifact_capture_boundary(
            model_sha256,
            model_architecture,
            candidate_path,
        );
        let pair =
            DenseLinearNoBiasStrictCaptureArtifactPairBoundary::from_strict_artifact_capture_boundary(
                &capture,
                DenseLinearNoBiasStrictCaptureArtifactPairInputs {
                    candidate_off_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/candidate-off-strict-capture.json",
                    ),
                    candidate_on_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/candidate-on-strict-capture.json",
                    ),
                    candidate_off_strict_capture_artifact_present: true,
                    candidate_on_strict_capture_artifact_present: true,
                    candidate_off_capture_command_recorded: true,
                    candidate_on_capture_command_recorded: true,
                    candidate_off_capture_binds_gate_identity: true,
                    candidate_on_capture_binds_gate_identity: true,
                    candidate_off_capture_binds_descriptor_identity: true,
                    candidate_on_capture_binds_descriptor_identity: true,
                    candidate_off_capture_binds_owner_callsite_identity: true,
                    candidate_on_capture_binds_owner_callsite_identity: true,
                    candidate_off_on_capture_same_callsite_identity: true,
                    candidate_off_on_capture_same_prompt_digest: true,
                    candidate_off_on_capture_same_generated_digest: true,
                    candidate_off_on_capture_same_decoded_text_digest: true,
                    candidate_off_on_capture_same_model_backend_identity: true,
                    capture_prerequisite_blocker_recorded: false,
                    default_runtime_path_preserved: true,
                },
            );
        let attempt = DenseLinearNoBiasRuntimeAttemptBoundary::from_strict_capture_artifact_pair(
            &pair,
            DenseLinearNoBiasRuntimeAttemptInputs {
                explicit_candidate_execution_gate_requested: true,
                runtime_hook_registry_attachment_present: false,
                runtime_hook_descriptor_binds_selector_identity: false,
                runtime_hook_descriptor_binds_strict_capture_pair: false,
                apply_linear_dispatch_wired_to_no_bias_candidate: false,
                feed_forward_down_proj_scope_preserved: true,
                default_runtime_path_preserved: true,
            },
        );
        let mut gate = slm_cpu_211_test_gate(model_sha256, candidate_path);
        gate.runtime_gate_requested_enabled = true;
        let selector = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            model_architecture,
            "gguf_metadata",
            true,
            receipt_pair_identity,
            true,
        );
        let hook = DenseLinearRuntimeHookDescriptor {
            tensor_name: selector.tensor_name.clone(),
            role: "FeedForwardDown".to_string(),
            sidecar_payload_sha256: None,
            packed_q8_payload: None,
            payload_order_matches_runtime_shape: false,
            source_order_q8_matvec_candidate: false,
            source_order_input_dim: None,
            source_order_output_dim: None,
            runtime_compute_enabled: false,
            receipt_bound_no_bias_selector: Some(selector),
        };
        let registry = DenseLinearRuntimeHookRegistry::from([(attempt.tensor_name.clone(), hook)]);

        DenseLinearNoBiasRuntimeHookAttachmentBoundary::from_runtime_attempt_and_registry(
            &attempt, &registry,
        )
    }

    #[test]
    fn no_bias_apply_linear_receipt_bound_selector_carries_qwen3_identity_runtime_disabled() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );

        assert_eq!(descriptor.decision, "receipt_bound_selector_descriptor_ready_runtime_disabled");
        assert_eq!(descriptor.model_architecture, "qwen3");
        assert_eq!(descriptor.tokenizer_source, "gguf_metadata");
        assert!(descriptor.tokenizer_strict);
        assert!(descriptor.descriptor_ready_for_apply_linear_callsite);
        assert!(descriptor.preserves_normal_inference());
        assert!(!descriptor.candidate_execution_enabled);
        assert!(!descriptor.normal_inference_runtime_selection_enabled);
        assert!(!descriptor.allocation_reduction_claim);
        assert!(!descriptor.timing_improvement_claim);
        assert!(!descriptor.speedup_claim);

        let hook = DenseLinearRuntimeHookDescriptor {
            tensor_name: descriptor.tensor_name.clone(),
            role: "FeedForwardDown".to_string(),
            sidecar_payload_sha256: None,
            packed_q8_payload: None,
            payload_order_matches_runtime_shape: false,
            source_order_q8_matvec_candidate: false,
            source_order_input_dim: None,
            source_order_output_dim: None,
            runtime_compute_enabled: false,
            receipt_bound_no_bias_selector: Some(descriptor.clone()),
        };
        assert!(!hook.runtime_compute_enabled);
        assert_eq!(
            hook.receipt_bound_no_bias_selector
                .as_ref()
                .map(|selector| selector.model_sha256.as_str()),
            Some(SLM_CPU_195_QWEN3_Q8_MODEL_SHA256)
        );
    }

    #[test]
    fn no_bias_apply_linear_receipt_bound_selector_carries_qwen25_identity_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let gate =
            slm_cpu_211_test_gate(qwen25_sha, "qwen25_feed_forward_down_proj_no_bias_candidate");
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen2",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen25:before-after",
            true,
        );

        assert_eq!(descriptor.decision, "receipt_bound_selector_descriptor_ready_runtime_disabled");
        assert_eq!(descriptor.model_architecture, "qwen2");
        assert_eq!(descriptor.model_sha256, qwen25_sha);
        assert!(descriptor.qwen2_candidate_policy_present);
        assert!(descriptor.descriptor_ready_for_apply_linear_callsite);
        assert!(descriptor.preserves_normal_inference());
        assert!(!descriptor.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_apply_linear_receipt_bound_selector_fails_closed_on_missing_identity() {
        let mut gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        gate.fallback_used = true;
        gate.prompt_ids_digest.clear();

        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate, "qwen2", "unknown", false, "", false,
        );

        assert_eq!(descriptor.decision, "blocked_fail_closed");
        assert!(!descriptor.descriptor_ready_for_apply_linear_callsite);
        assert!(descriptor.fail_closed_conditions.contains(&"fallback_used"));
        assert!(descriptor.fail_closed_conditions.contains(&"prompt_ids_digest_not_preserved"));
        assert!(descriptor.fail_closed_conditions.contains(&"tokenizer_source_not_gguf_metadata"));
        assert!(descriptor.fail_closed_conditions.contains(&"tokenizer_not_strict"));
        assert!(
            descriptor
                .fail_closed_conditions
                .contains(&"before_after_receipt_pair_identity_missing")
        );
        assert!(descriptor.fail_closed_conditions.contains(&"qwen2_candidate_policy_missing"));
        assert_eq!(descriptor.selected_path, "eager_f32_candle");
        assert_eq!(descriptor.selected_kernel, "dense-f32-candle-linear");
        assert!(!descriptor.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_selector_propagation_boundary_records_missing_hook_mutation_point() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );

        let boundary =
            DenseLinearNoBiasSelectorPropagationBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                false,
                false,
                false,
            );

        assert_eq!(boundary.decision, "blocked_fail_closed");
        assert_eq!(
            boundary.reason,
            "receipt_bound_selector_identity_cannot_reach_apply_linear_before_candidate_execution"
        );
        assert_eq!(
            boundary.remaining_runtime_selection_blocker,
            "session_hook_registry_mutation_point_or_per_callsite_candidate_receipt_emitter"
        );
        assert_eq!(
            boundary.prompt_digest_lifetime,
            "available_after_warm_session_prompt_execution"
        );
        assert_eq!(
            boundary.hook_registry_owner,
            "bitnet_models::bitnet::dense_q8_runtime_hooks_from_sidecars"
        );
        assert!(boundary.descriptor_ready_for_apply_linear_callsite);
        assert!(!boundary.hook_registry_selector_present);
        assert!(!boundary.hook_registry_mutation_point_present);
        assert!(!boundary.per_callsite_receipt_emitter_present);
        assert!(!boundary.can_attach_after_prompt_digests_known);
        assert!(!boundary.can_attach_before_same_prompt_candidate_execution);
        assert!(
            boundary.fail_closed_conditions.contains(&"hook_registry_selector_identity_missing")
        );
        assert!(
            boundary.fail_closed_conditions.contains(&"session_selector_mutation_point_missing")
        );
        assert!(
            boundary
                .fail_closed_conditions
                .contains(&"per_callsite_candidate_receipt_emitter_missing")
        );
        assert!(boundary.preserves_normal_inference());
        assert!(!boundary.candidate_execution_enabled);
        assert!(!boundary.normal_inference_runtime_selection_enabled);
        assert!(!boundary.allocation_reduction_claim);
        assert!(!boundary.timing_improvement_claim);
        assert!(!boundary.speedup_claim);
    }

    #[test]
    fn no_bias_selector_propagation_boundary_can_model_future_safe_attachment_runtime_disabled() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );

        let boundary =
            DenseLinearNoBiasSelectorPropagationBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                true,
                true,
                false,
            );

        assert_eq!(boundary.decision, "selector_propagation_boundary_ready_runtime_disabled");
        assert!(boundary.can_attach_after_prompt_digests_known);
        assert!(boundary.can_attach_before_same_prompt_candidate_execution);
        assert!(boundary.fail_closed_conditions.is_empty());
        assert!(boundary.preserves_normal_inference());
        assert!(!boundary.candidate_execution_enabled);
        assert!(!boundary.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_selector_propagation_boundary_fails_closed_on_descriptor_identity_drift() {
        let mut gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        gate.generated_ids_digest.clear();
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );

        let boundary =
            DenseLinearNoBiasSelectorPropagationBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                true,
                true,
                false,
            );

        assert_eq!(boundary.decision, "blocked_fail_closed");
        assert!(!boundary.can_attach_after_prompt_digests_known);
        assert!(!boundary.can_attach_before_same_prompt_candidate_execution);
        assert!(boundary.fail_closed_conditions.contains(&"generated_ids_digest_not_preserved"));
        assert!(boundary.fail_closed_conditions.contains(&"generated_ids_digest_missing"));
        assert!(
            boundary
                .fail_closed_conditions
                .contains(&"receipt_bound_selector_descriptor_not_ready")
        );
        assert!(boundary.preserves_normal_inference());
        assert!(!boundary.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_per_callsite_emitter_binds_descriptor_identity_runtime_disabled() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );

        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                "layers.0.feed_forward.down_proj.weight",
                "bitnet_transformer::FeedForward::apply_linear:layers.0.feed_forward.down_proj.weight",
                false,
                false,
                false,
            );

        assert_eq!(emitter.decision, "per_callsite_candidate_receipt_emitter_defined_fail_closed");
        assert!(emitter.per_callsite_receipt_emitter_present);
        assert!(emitter.per_callsite_identity_matches_descriptor);
        assert_eq!(emitter.model_architecture, "qwen3");
        assert_eq!(emitter.tokenizer_source, "gguf_metadata");
        assert!(emitter.tokenizer_strict);
        assert_eq!(emitter.runtime_api, "cpu");
        assert_eq!(emitter.selected_backend, "cpu-rust");
        assert!(!emitter.fallback_used);
        assert!(emitter.fail_closed_conditions.contains(&"explicit_runtime_gate_not_requested"));
        assert!(emitter.fail_closed_conditions.contains(&"candidate_off_on_receipts_missing"));
        assert!(emitter.fail_closed_conditions.contains(&"generated_id_preservation_not_proven"));
        assert!(emitter.preserves_normal_inference());
        assert!(!emitter.candidate_execution_enabled);
        assert!(!emitter.normal_inference_runtime_selection_enabled);
        assert!(!emitter.allocation_reduction_claim);
        assert!(!emitter.timing_improvement_claim);
        assert!(!emitter.speedup_claim);
    }

    #[test]
    fn no_bias_per_callsite_emitter_can_model_ready_attachment_without_execution() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );

        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                descriptor.tensor_name.clone(),
                "bitnet_transformer::FeedForward::apply_linear:layers.0.feed_forward.down_proj.weight",
                true,
                true,
                true,
            );

        assert_eq!(
            emitter.decision,
            "per_callsite_candidate_receipt_emitter_ready_runtime_disabled"
        );
        assert!(emitter.fail_closed_conditions.is_empty());
        assert!(emitter.explicit_runtime_gate_requested);
        assert!(emitter.candidate_off_on_receipts_present);
        assert!(emitter.generated_id_preservation_proven);
        assert!(emitter.per_callsite_receipt_emitter_present);
        assert!(emitter.preserves_normal_inference());
        assert!(!emitter.candidate_execution_enabled);
        assert!(!emitter.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_per_callsite_emitter_rejects_wrong_tensor_or_missing_identity() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );

        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                "layers.0.feed_forward.up_proj.weight",
                "",
                true,
                true,
                true,
            );

        assert_eq!(emitter.decision, "blocked_fail_closed");
        assert!(!emitter.per_callsite_receipt_emitter_present);
        assert!(!emitter.per_callsite_identity_matches_descriptor);
        assert!(emitter.fail_closed_conditions.contains(&"callsite_tensor_name_mismatch"));
        assert!(emitter.fail_closed_conditions.contains(&"callsite_identity_missing"));
        assert!(emitter.preserves_normal_inference());
        assert!(!emitter.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_candidate_off_on_receipt_pair_gate_blocks_missing_candidate_on() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );
        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                descriptor.tensor_name.clone(),
                "bitnet_transformer::FeedForward::apply_linear:layers.0.feed_forward.down_proj.weight",
                true,
                false,
                false,
            );

        let pair_gate = DenseLinearNoBiasCandidateOffOnReceiptPairGate::from_per_callsite_emitter(
            &emitter, true, false, true, true, true,
        );

        assert_eq!(pair_gate.decision, "candidate_off_on_receipt_pair_gate_defined_fail_closed");
        assert_eq!(pair_gate.reason, "candidate_off_on_receipt_pair_incomplete");
        assert_eq!(
            pair_gate.remaining_runtime_selection_blocker,
            "candidate_on_strict_warm_session_receipt_artifact"
        );
        assert!(pair_gate.candidate_off_receipt_present);
        assert!(!pair_gate.candidate_on_receipt_present);
        assert!(pair_gate.fail_closed_conditions.contains(&"candidate_on_receipt_missing"));
        assert!(pair_gate.fail_closed_conditions.contains(&"candidate_off_on_receipts_missing"));
        assert!(pair_gate.fail_closed_conditions.contains(&"generated_id_preservation_not_proven"));
        assert!(pair_gate.preserves_normal_inference());
        assert!(!pair_gate.candidate_execution_enabled);
        assert!(!pair_gate.normal_inference_runtime_selection_enabled);
        assert!(!pair_gate.allocation_reduction_claim);
        assert!(!pair_gate.timing_improvement_claim);
        assert!(!pair_gate.speedup_claim);
    }

    #[test]
    fn no_bias_candidate_off_on_receipt_pair_gate_models_ready_pair_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let gate =
            slm_cpu_211_test_gate(qwen25_sha, "qwen25_feed_forward_down_proj_no_bias_candidate");
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen2",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen25:before-after",
            true,
        );
        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                descriptor.tensor_name.clone(),
                "bitnet_transformer::FeedForward::apply_linear:layers.0.feed_forward.down_proj.weight",
                true,
                true,
                true,
            );

        let pair_gate = DenseLinearNoBiasCandidateOffOnReceiptPairGate::from_per_callsite_emitter(
            &emitter, true, true, true, true, true,
        );

        assert_eq!(pair_gate.decision, "candidate_off_on_receipt_pair_gate_ready_runtime_disabled");
        assert_eq!(pair_gate.model_architecture, "qwen2");
        assert_eq!(pair_gate.model_sha256, qwen25_sha);
        assert!(pair_gate.per_callsite_receipt_emitter_present);
        assert!(pair_gate.per_callsite_identity_matches_descriptor);
        assert!(pair_gate.explicit_runtime_gate_requested);
        assert!(pair_gate.candidate_off_receipt_present);
        assert!(pair_gate.candidate_on_receipt_present);
        assert!(pair_gate.prompt_ids_preserved);
        assert!(pair_gate.generated_ids_preserved);
        assert!(pair_gate.decoded_text_preserved);
        assert!(pair_gate.fail_closed_conditions.is_empty());
        assert!(pair_gate.preserves_normal_inference());
        assert!(!pair_gate.candidate_execution_enabled);
        assert!(!pair_gate.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_prompt_session_descriptor_reaches_callsite_without_generated_text_binding() {
        let tensor_name = "layers.0.feed_forward.down_proj.weight";
        let callsite_identity =
            dense_linear_no_bias_feed_forward_apply_linear_callsite_identity(0, "down_proj");
        let descriptor = DenseLinearNoBiasPromptSessionDescriptor::from_prompt_session(
            DenseLinearNoBiasPromptSessionDescriptorInput {
                tensor_name,
                callsite_identity: callsite_identity.as_str(),
                model_sha256: SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
                model_architecture: "qwen3",
                quant_format: "Q8_0",
                tokenizer_source: "gguf_metadata",
                tokenizer_strict: true,
                runtime_api: "cpu",
                selected_backend: "cpu-rust",
                fallback_used: false,
                prompt_ids: &[151644, 872, 198, 19],
                prompt_ids_digest: "prompt-digest",
                selected_path: "eager_f32_candle",
                selected_kernel: "dense-f32-candle-linear",
                candidate_path: "qwen3_feed_forward_down_proj_no_bias_candidate",
                candidate_kernel: "dense-f32-candle-linear-no-bias-candidate",
                bias_present: Some(false),
                explicit_runtime_gate_requested: true,
            },
        );
        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_prompt_session_descriptor(
                &descriptor,
            );

        assert_eq!(
            descriptor.decision,
            "prompt_session_descriptor_ready_for_apply_linear_runtime_disabled"
        );
        assert!(descriptor.descriptor_ready_for_apply_linear_callsite);
        assert!(!descriptor.generated_ids_bound_before_decode);
        assert!(!descriptor.decoded_text_bound_before_decode);
        assert!(descriptor.preserves_normal_inference());
        assert_eq!(
            emitter.decision,
            "per_callsite_prompt_session_descriptor_ready_runtime_disabled"
        );
        assert_eq!(emitter.tensor_name, tensor_name);
        assert_eq!(emitter.callsite_identity, callsite_identity);
        assert!(emitter.per_callsite_receipt_emitter_present);
        assert!(emitter.per_callsite_identity_matches_descriptor);
        assert!(emitter.explicit_runtime_gate_requested);
        assert_eq!(emitter.prompt_ids_digest, "prompt-digest");
        assert!(emitter.generated_ids_digest.is_empty());
        assert!(emitter.decoded_text_digest.is_empty());
        assert!(!emitter.candidate_off_on_receipts_present);
        assert!(!emitter.generated_id_preservation_proven);
        assert!(!emitter.candidate_execution_enabled);
        assert!(emitter.preserves_normal_inference());

        let fail_closed_conditions =
            feed_forward_no_bias_apply_linear_descriptor_fail_closed_conditions(
                &emitter,
                "down_proj",
                tensor_name,
            );
        assert!(fail_closed_conditions.is_empty());
    }

    #[test]
    fn no_bias_per_callsite_dispatch_descriptor_records_apply_linear_argument_blocker() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );

        let boundary =
            DenseLinearNoBiasPerCallsiteDispatchDescriptorBoundary::from_candidate_off_on_pair_gate(
                &pair_gate,
                DenseLinearNoBiasPerCallsiteDispatchDescriptorInputs {
                    prompt_bound_candidate_descriptor_argument_present: false,
                    prompt_bound_session_descriptor_constructed: false,
                    descriptor_identity_reaches_apply_linear_callsite: false,
                    prompt_digest_available_at_apply_linear: false,
                    generated_text_digests_available_at_apply_linear: false,
                    feed_forward_apply_linear_no_bias_dispatch_branch_present: false,
                    dispatch_calls_no_bias_candidate_forward: false,
                    candidate_on_receipt_emitted_at_apply_linear_callsite: false,
                    feed_forward_down_proj_scope_preserved: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(boundary.decision, "per_callsite_dispatch_descriptor_blocked_fail_closed");
        assert_eq!(
            boundary.remaining_runtime_selection_blocker,
            "feed_forward_apply_linear_prompt_bound_candidate_descriptor_argument"
        );
        assert!(boundary.candidate_off_on_receipt_pair_gate_ready);
        assert!(boundary.explicit_runtime_gate_requested);
        assert!(!boundary.prompt_bound_candidate_descriptor_argument_present);
        assert!(!boundary.prompt_bound_session_descriptor_constructed);
        assert!(!boundary.descriptor_identity_reaches_apply_linear_callsite);
        assert!(!boundary.feed_forward_apply_linear_no_bias_dispatch_branch_present);
        assert!(!boundary.dispatch_calls_no_bias_candidate_forward);
        assert!(!boundary.candidate_execution_attempt_allowed);
        assert!(!boundary.candidate_execution_enabled_by_default);
        assert!(!boundary.normal_inference_runtime_selection_enabled);
        assert!(boundary.fail_closed_conditions.contains(
            &"feed_forward_apply_linear_prompt_bound_candidate_descriptor_argument_missing"
        ));
        assert!(
            boundary
                .fail_closed_conditions
                .contains(&"prompt_bound_session_descriptor_not_constructed")
        );
        assert!(
            boundary
                .fail_closed_conditions
                .contains(&"generated_text_digests_not_available_before_apply_linear_dispatch")
        );
        assert!(
            boundary
                .fail_closed_conditions
                .contains(&"feed_forward_apply_linear_no_bias_dispatch_branch_missing")
        );
        assert!(boundary.preserves_normal_inference());
        assert!(!boundary.allocation_reduction_claim);
        assert!(!boundary.timing_improvement_claim);
        assert!(!boundary.speedup_claim);
    }

    #[test]
    fn no_bias_per_callsite_dispatch_descriptor_names_session_descriptor_blocker() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );

        let boundary =
            DenseLinearNoBiasPerCallsiteDispatchDescriptorBoundary::from_candidate_off_on_pair_gate(
                &pair_gate,
                DenseLinearNoBiasPerCallsiteDispatchDescriptorInputs {
                    prompt_bound_candidate_descriptor_argument_present: true,
                    prompt_bound_session_descriptor_constructed: false,
                    descriptor_identity_reaches_apply_linear_callsite: false,
                    prompt_digest_available_at_apply_linear: false,
                    generated_text_digests_available_at_apply_linear: false,
                    feed_forward_apply_linear_no_bias_dispatch_branch_present: false,
                    dispatch_calls_no_bias_candidate_forward: false,
                    candidate_on_receipt_emitted_at_apply_linear_callsite: false,
                    feed_forward_down_proj_scope_preserved: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(
            boundary.remaining_runtime_selection_blocker,
            "prompt_bound_session_descriptor_construction"
        );
        assert!(boundary.prompt_bound_candidate_descriptor_argument_present);
        assert!(!boundary.prompt_bound_session_descriptor_constructed);
        assert!(!boundary.descriptor_identity_reaches_apply_linear_callsite);
        assert!(
            boundary
                .fail_closed_conditions
                .contains(&"prompt_bound_session_descriptor_not_constructed")
        );
        assert!(boundary.preserves_normal_inference());
        assert!(!boundary.candidate_execution_attempt_allowed);
        assert!(!boundary.candidate_execution_enabled_by_default);
    }

    #[test]
    fn no_bias_per_callsite_dispatch_descriptor_names_digest_lifetime_blocker() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );

        let boundary =
            DenseLinearNoBiasPerCallsiteDispatchDescriptorBoundary::from_candidate_off_on_pair_gate(
                &pair_gate,
                DenseLinearNoBiasPerCallsiteDispatchDescriptorInputs {
                    prompt_bound_candidate_descriptor_argument_present: true,
                    prompt_bound_session_descriptor_constructed: true,
                    descriptor_identity_reaches_apply_linear_callsite: true,
                    prompt_digest_available_at_apply_linear: true,
                    generated_text_digests_available_at_apply_linear: false,
                    feed_forward_apply_linear_no_bias_dispatch_branch_present: false,
                    dispatch_calls_no_bias_candidate_forward: false,
                    candidate_on_receipt_emitted_at_apply_linear_callsite: false,
                    feed_forward_down_proj_scope_preserved: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(boundary.decision, "per_callsite_dispatch_descriptor_blocked_fail_closed");
        assert_eq!(
            boundary.remaining_runtime_selection_blocker,
            "generated_text_digest_lifetime_before_apply_linear_dispatch"
        );
        assert!(boundary.candidate_off_on_receipt_pair_gate_ready);
        assert!(boundary.prompt_bound_candidate_descriptor_argument_present);
        assert!(boundary.prompt_bound_session_descriptor_constructed);
        assert!(boundary.descriptor_identity_reaches_apply_linear_callsite);
        assert!(boundary.prompt_digest_available_at_apply_linear);
        assert!(!boundary.generated_text_digests_available_at_apply_linear);
        assert!(!boundary.feed_forward_apply_linear_no_bias_dispatch_branch_present);
        assert!(!boundary.candidate_execution_attempt_allowed);
        assert!(
            boundary
                .fail_closed_conditions
                .contains(&"generated_text_digests_not_available_before_apply_linear_dispatch")
        );
        assert!(boundary.preserves_normal_inference());
    }

    #[test]
    fn no_bias_per_callsite_dispatch_descriptor_models_future_ready_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let pair_gate = slm_cpu_216_ready_pair_gate(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );

        let boundary =
            DenseLinearNoBiasPerCallsiteDispatchDescriptorBoundary::from_candidate_off_on_pair_gate(
                &pair_gate,
                DenseLinearNoBiasPerCallsiteDispatchDescriptorInputs {
                    prompt_bound_candidate_descriptor_argument_present: true,
                    prompt_bound_session_descriptor_constructed: true,
                    descriptor_identity_reaches_apply_linear_callsite: true,
                    prompt_digest_available_at_apply_linear: true,
                    generated_text_digests_available_at_apply_linear: true,
                    feed_forward_apply_linear_no_bias_dispatch_branch_present: true,
                    dispatch_calls_no_bias_candidate_forward: true,
                    candidate_on_receipt_emitted_at_apply_linear_callsite: true,
                    feed_forward_down_proj_scope_preserved: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(boundary.decision, "per_callsite_dispatch_descriptor_ready_runtime_disabled");
        assert_eq!(
            boundary.remaining_runtime_selection_blocker,
            "fresh_candidate_off_on_execution_receipts_from_apply_linear"
        );
        assert_eq!(boundary.model_architecture, "qwen2");
        assert_eq!(boundary.model_sha256, qwen25_sha);
        assert_eq!(boundary.candidate_path, "qwen25_feed_forward_down_proj_no_bias_candidate");
        assert!(boundary.candidate_off_on_receipt_pair_gate_ready);
        assert!(boundary.prompt_bound_candidate_descriptor_argument_present);
        assert!(boundary.prompt_bound_session_descriptor_constructed);
        assert!(boundary.descriptor_identity_reaches_apply_linear_callsite);
        assert!(boundary.feed_forward_apply_linear_no_bias_dispatch_branch_present);
        assert!(boundary.dispatch_calls_no_bias_candidate_forward);
        assert!(boundary.candidate_on_receipt_emitted_at_apply_linear_callsite);
        assert!(boundary.candidate_execution_attempt_allowed);
        assert!(!boundary.candidate_execution_enabled_by_default);
        assert!(!boundary.normal_inference_runtime_selection_enabled);
        assert!(boundary.fail_closed_conditions.is_empty());
        assert!(boundary.preserves_normal_inference());
        assert!(!boundary.allocation_reduction_claim);
        assert!(!boundary.timing_improvement_claim);
        assert!(!boundary.speedup_claim);
    }

    #[test]
    fn no_bias_apply_linear_descriptor_argument_accepts_exact_down_proj_callsite() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );
        let callsite_identity =
            feed_forward_apply_linear_callsite_identity(&descriptor.tensor_name);
        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                descriptor.tensor_name.clone(),
                callsite_identity,
                true,
                true,
                true,
            );

        let fail_closed_conditions =
            feed_forward_no_bias_apply_linear_descriptor_fail_closed_conditions(
                &emitter,
                "down_proj",
                &descriptor.tensor_name,
            );

        assert!(fail_closed_conditions.is_empty());
        assert!(emitter.per_callsite_receipt_emitter_present);
        assert!(emitter.per_callsite_identity_matches_descriptor);
        assert!(emitter.preserves_normal_inference());
        assert!(!emitter.candidate_execution_enabled);
        assert!(!emitter.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_prompt_bound_descriptor_targets_only_matching_down_proj_layer() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );
        let callsite_identity =
            dense_linear_no_bias_feed_forward_apply_linear_callsite_identity(0, "down_proj");
        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                descriptor.tensor_name.clone(),
                callsite_identity.clone(),
                true,
                true,
                true,
            );

        assert_eq!(
            callsite_identity,
            "bitnet_transformer::FeedForward::apply_linear:layers.0.feed_forward.down_proj.weight"
        );
        assert!(prompt_bound_no_bias_descriptor_targets_feed_forward_down_proj_layer(&emitter, 0));
        assert!(!prompt_bound_no_bias_descriptor_targets_feed_forward_down_proj_layer(&emitter, 1));
        assert_eq!(emitter.tensor_name, "layers.0.feed_forward.down_proj.weight");
        assert!(emitter.preserves_normal_inference());
        assert!(!emitter.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_apply_linear_descriptor_argument_rejects_wrong_callsite() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );
        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                descriptor.tensor_name.clone(),
                feed_forward_apply_linear_callsite_identity(&descriptor.tensor_name),
                true,
                true,
                true,
            );

        let fail_closed_conditions =
            feed_forward_no_bias_apply_linear_descriptor_fail_closed_conditions(
                &emitter,
                "gate_proj",
                "layers.0.feed_forward.gate_proj.weight",
            );

        assert!(fail_closed_conditions.contains(&"feed_forward_projection_not_down_proj"));
        assert!(fail_closed_conditions.contains(&"prompt_bound_descriptor_tensor_name_mismatch"));
        assert!(
            fail_closed_conditions.contains(&"prompt_bound_descriptor_callsite_identity_mismatch")
        );
        assert!(!emitter.candidate_execution_enabled);
        assert!(emitter.preserves_normal_inference());
    }

    #[test]
    fn no_bias_candidate_off_on_receipt_pair_gate_rejects_generated_id_drift() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );
        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                descriptor.tensor_name.clone(),
                "bitnet_transformer::FeedForward::apply_linear:layers.0.feed_forward.down_proj.weight",
                true,
                true,
                true,
            );

        let pair_gate = DenseLinearNoBiasCandidateOffOnReceiptPairGate::from_per_callsite_emitter(
            &emitter, true, true, true, false, true,
        );

        assert_eq!(pair_gate.decision, "candidate_off_on_receipt_pair_gate_defined_fail_closed");
        assert!(!pair_gate.generated_ids_preserved);
        assert!(pair_gate.fail_closed_conditions.contains(&"generated_ids_not_preserved"));
        assert!(pair_gate.preserves_normal_inference());
        assert!(!pair_gate.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_candidate_on_behavior_gate_blocks_missing_runtime_attachment() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );

        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, false, false, false,
            );

        assert_eq!(
            behavior_gate.decision,
            "candidate_on_behavior_evidence_gate_defined_fail_closed"
        );
        assert_eq!(
            behavior_gate.remaining_runtime_selection_blocker,
            "candidate_on_apply_linear_runtime_attachment_point"
        );
        assert!(behavior_gate.candidate_off_on_pair_gate_ready);
        assert!(!behavior_gate.candidate_on_behavior_evidence_present);
        assert!(!behavior_gate.candidate_on_runtime_attachment_point_present);
        assert!(!behavior_gate.candidate_on_receipt_fields_complete);
        assert!(
            behavior_gate
                .fail_closed_conditions
                .contains(&"candidate_on_behavior_evidence_missing")
        );
        assert!(
            behavior_gate
                .fail_closed_conditions
                .contains(&"candidate_on_runtime_attachment_point_missing")
        );
        assert!(
            behavior_gate
                .fail_closed_conditions
                .contains(&"candidate_on_receipt_fields_incomplete")
        );
        assert!(behavior_gate.preserves_normal_inference());
        assert!(!behavior_gate.candidate_execution_enabled);
        assert!(!behavior_gate.normal_inference_runtime_selection_enabled);
        assert!(!behavior_gate.allocation_reduction_claim);
        assert!(!behavior_gate.timing_improvement_claim);
        assert!(!behavior_gate.speedup_claim);
    }

    #[test]
    fn no_bias_candidate_on_behavior_gate_models_ready_evidence_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let pair_gate = slm_cpu_216_ready_pair_gate(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );

        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );

        assert_eq!(behavior_gate.decision, "candidate_on_behavior_evidence_ready_runtime_disabled");
        assert_eq!(
            behavior_gate.reason,
            "candidate_on_behavior_preserves_strict_warm_session_identity"
        );
        assert_eq!(behavior_gate.model_architecture, "qwen2");
        assert_eq!(behavior_gate.model_sha256, qwen25_sha);
        assert_eq!(behavior_gate.candidate_path, "qwen25_feed_forward_down_proj_no_bias_candidate");
        assert!(behavior_gate.candidate_off_on_pair_gate_ready);
        assert!(behavior_gate.candidate_on_behavior_evidence_present);
        assert!(behavior_gate.candidate_on_runtime_attachment_point_present);
        assert!(behavior_gate.candidate_on_receipt_fields_complete);
        assert!(behavior_gate.prompt_ids_preserved);
        assert!(behavior_gate.generated_ids_preserved);
        assert!(behavior_gate.decoded_text_preserved);
        assert!(behavior_gate.default_runtime_path_preserved);
        assert!(behavior_gate.fail_closed_conditions.is_empty());
        assert!(behavior_gate.preserves_normal_inference());
        assert!(!behavior_gate.candidate_execution_enabled);
        assert!(!behavior_gate.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_candidate_on_behavior_gate_rejects_incomplete_pair_gate() {
        let gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let descriptor = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-209:qwen3:before-after",
            true,
        );
        let emitter =
            DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary::from_receipt_bound_selector_descriptor(
                &descriptor,
                descriptor.tensor_name.clone(),
                "bitnet_transformer::FeedForward::apply_linear:layers.0.feed_forward.down_proj.weight",
                true,
                true,
                true,
            );
        let pair_gate = DenseLinearNoBiasCandidateOffOnReceiptPairGate::from_per_callsite_emitter(
            &emitter, true, false, true, true, true,
        );

        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );

        assert_eq!(behavior_gate.decision, "blocked_fail_closed");
        assert_eq!(
            behavior_gate.remaining_runtime_selection_blocker,
            "candidate_off_on_receipt_pair_gate"
        );
        assert!(!behavior_gate.candidate_off_on_pair_gate_ready);
        assert!(
            behavior_gate.fail_closed_conditions.contains(&"candidate_off_on_pair_gate_not_ready")
        );
        assert!(behavior_gate.fail_closed_conditions.contains(&"candidate_on_receipt_missing"));
        assert!(behavior_gate.preserves_normal_inference());
        assert!(!behavior_gate.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_candidate_runtime_attachment_boundary_records_missing_runtime_owner() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );

        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                false,
                false,
                false,
            );

        assert_eq!(attachment.decision, "candidate_runtime_attachment_defined_fail_closed");
        assert_eq!(attachment.remaining_runtime_selection_blocker, "candidate_runtime_owner");
        assert!(attachment.candidate_on_behavior_gate_ready);
        assert!(attachment.explicit_runtime_gate_requested);
        assert!(attachment.apply_linear_candidate_attachment_wired);
        assert!(!attachment.candidate_runtime_owner_present);
        assert!(!attachment.candidate_receipt_emitter_wired);
        assert!(!attachment.candidate_compute_callable);
        assert!(attachment.fail_closed_conditions.contains(&"candidate_runtime_owner_missing"));
        assert!(attachment.fail_closed_conditions.contains(&"candidate_receipt_emitter_not_wired"));
        assert!(attachment.preserves_normal_inference());
        assert!(!attachment.candidate_execution_enabled);
        assert!(!attachment.normal_inference_runtime_selection_enabled);
        assert!(!attachment.allocation_reduction_claim);
        assert!(!attachment.timing_improvement_claim);
        assert!(!attachment.speedup_claim);
    }

    #[test]
    fn no_bias_candidate_runtime_attachment_boundary_models_ready_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let pair_gate = slm_cpu_216_ready_pair_gate(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );

        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                true,
                true,
                true,
            );

        assert_eq!(attachment.decision, "candidate_runtime_attachment_ready_runtime_disabled");
        assert_eq!(attachment.model_architecture, "qwen2");
        assert_eq!(attachment.model_sha256, qwen25_sha);
        assert_eq!(attachment.candidate_path, "qwen25_feed_forward_down_proj_no_bias_candidate");
        assert!(attachment.candidate_on_behavior_gate_ready);
        assert!(attachment.explicit_runtime_gate_requested);
        assert!(attachment.apply_linear_candidate_attachment_wired);
        assert!(attachment.candidate_runtime_owner_present);
        assert!(attachment.candidate_receipt_emitter_wired);
        assert!(attachment.candidate_compute_callable);
        assert!(attachment.default_runtime_path_preserved);
        assert!(attachment.fail_closed_conditions.is_empty());
        assert!(attachment.preserves_normal_inference());
        assert!(!attachment.candidate_execution_enabled);
        assert!(!attachment.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_candidate_runtime_attachment_boundary_rejects_incomplete_behavior_gate() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, false, false, false,
            );

        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                true,
                true,
                true,
            );

        assert_eq!(attachment.decision, "blocked_fail_closed");
        assert_eq!(
            attachment.remaining_runtime_selection_blocker,
            "candidate_on_behavior_evidence_gate"
        );
        assert!(!attachment.candidate_on_behavior_gate_ready);
        assert!(
            attachment.fail_closed_conditions.contains(&"candidate_on_behavior_gate_not_ready")
        );
        assert!(
            attachment.fail_closed_conditions.contains(&"candidate_on_behavior_evidence_missing")
        );
        assert!(attachment.preserves_normal_inference());
        assert!(!attachment.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_candidate_runtime_owner_boundary_records_receipt_emitter_blocker() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                false,
                false,
                false,
            );

        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: false,
                    candidate_off_on_strict_receipts_present: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );

        assert_eq!(owner.decision, "candidate_runtime_owner_defined_fail_closed");
        assert_eq!(
            owner.remaining_runtime_selection_blocker,
            "same_callsite_candidate_on_receipt_emitter"
        );
        assert!(owner.candidate_runtime_attachment_boundary_defined);
        assert!(owner.apply_linear_runtime_owner_present);
        assert!(owner.owner_has_apply_linear_inputs);
        assert!(owner.owner_has_linear_weight_access);
        assert!(owner.candidate_compute_callable);
        assert!(!owner.same_callsite_candidate_on_receipt_emitter_wired);
        assert!(!owner.candidate_off_on_strict_receipts_present);
        assert!(!owner.prompt_ids_preserved);
        assert!(!owner.generated_ids_preserved);
        assert!(!owner.decoded_text_preserved);
        assert!(!owner.fail_closed_conditions.contains(&"candidate_runtime_owner_missing"));
        assert!(
            owner
                .fail_closed_conditions
                .contains(&"same_callsite_candidate_on_receipt_emitter_missing")
        );
        assert!(owner.fail_closed_conditions.contains(&"candidate_off_on_strict_receipts_missing"));
        assert!(owner.preserves_normal_inference());
        assert!(!owner.candidate_execution_enabled);
        assert!(!owner.normal_inference_runtime_selection_enabled);
        assert!(!owner.allocation_reduction_claim);
        assert!(!owner.timing_improvement_claim);
        assert!(!owner.speedup_claim);
    }

    #[test]
    fn no_bias_candidate_runtime_owner_boundary_models_ready_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let pair_gate = slm_cpu_216_ready_pair_gate(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                true,
                true,
                true,
            );

        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: true,
                    candidate_off_on_strict_receipts_present: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        assert_eq!(
            owner.decision,
            "candidate_runtime_owner_and_receipt_emitter_ready_runtime_disabled"
        );
        assert_eq!(owner.model_architecture, "qwen2");
        assert_eq!(owner.model_sha256, qwen25_sha);
        assert_eq!(owner.candidate_path, "qwen25_feed_forward_down_proj_no_bias_candidate");
        assert!(owner.candidate_runtime_attachment_boundary_defined);
        assert!(owner.apply_linear_runtime_owner_present);
        assert!(owner.same_callsite_candidate_on_receipt_emitter_wired);
        assert!(owner.candidate_off_on_strict_receipts_present);
        assert!(owner.prompt_ids_preserved);
        assert!(owner.generated_ids_preserved);
        assert!(owner.decoded_text_preserved);
        assert!(owner.fail_closed_conditions.is_empty());
        assert!(owner.preserves_normal_inference());
        assert!(!owner.candidate_execution_enabled);
        assert!(!owner.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_candidate_runtime_owner_boundary_rejects_missing_owner_inputs() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                true,
                true,
                true,
            );

        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: false,
                    owner_has_apply_linear_inputs: false,
                    owner_has_linear_weight_access: false,
                    candidate_compute_callable: false,
                    same_callsite_candidate_on_receipt_emitter_wired: true,
                    candidate_off_on_strict_receipts_present: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        assert_eq!(owner.decision, "candidate_runtime_owner_blocked_fail_closed");
        assert_eq!(owner.remaining_runtime_selection_blocker, "apply_linear_runtime_owner");
        assert!(!owner.apply_linear_runtime_owner_present);
        assert!(!owner.owner_has_apply_linear_inputs);
        assert!(!owner.owner_has_linear_weight_access);
        assert!(!owner.candidate_compute_callable);
        assert!(owner.fail_closed_conditions.contains(&"apply_linear_runtime_owner_missing"));
        assert!(
            owner.fail_closed_conditions.contains(&"runtime_owner_missing_apply_linear_inputs")
        );
        assert!(owner.preserves_normal_inference());
        assert!(!owner.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_same_callsite_receipt_emitter_boundary_records_fresh_receipt_blocker() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                false,
                false,
                false,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: false,
                    candidate_off_on_strict_receipts_present: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );

        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: true,
                    candidate_off_strict_receipt_present: false,
                    candidate_on_strict_receipt_present: false,
                    strict_receipts_bind_owner_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );

        assert_eq!(emitter.decision, "same_callsite_candidate_receipt_emitter_defined_fail_closed");
        assert_eq!(
            emitter.remaining_runtime_selection_blocker,
            "fresh_candidate_off_on_strict_receipts"
        );
        assert!(emitter.runtime_owner_boundary_defined);
        assert!(emitter.same_callsite_candidate_receipt_emitter_present);
        assert!(!emitter.candidate_off_strict_receipt_present);
        assert!(!emitter.candidate_on_strict_receipt_present);
        assert!(!emitter.strict_receipts_bind_owner_identity);
        assert!(!emitter.prompt_ids_preserved);
        assert!(!emitter.generated_ids_preserved);
        assert!(!emitter.decoded_text_preserved);
        assert!(
            !emitter
                .fail_closed_conditions
                .contains(&"same_callsite_candidate_on_receipt_emitter_missing")
        );
        assert!(
            !emitter.fail_closed_conditions.contains(&"candidate_off_on_strict_receipts_missing")
        );
        assert!(emitter.fail_closed_conditions.contains(&"candidate_off_strict_receipt_missing"));
        assert!(emitter.fail_closed_conditions.contains(&"candidate_on_strict_receipt_missing"));
        assert!(
            emitter
                .fail_closed_conditions
                .contains(&"strict_receipts_do_not_bind_runtime_owner_identity")
        );
        assert!(
            emitter.fail_closed_conditions.contains(&"generated_ids_preservation_receipt_missing")
        );
        assert!(emitter.preserves_normal_inference());
        assert!(!emitter.candidate_execution_enabled);
        assert!(!emitter.normal_inference_runtime_selection_enabled);
        assert!(!emitter.allocation_reduction_claim);
        assert!(!emitter.timing_improvement_claim);
        assert!(!emitter.speedup_claim);
    }

    #[test]
    fn no_bias_same_callsite_receipt_emitter_boundary_models_ready_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let pair_gate = slm_cpu_216_ready_pair_gate(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                true,
                true,
                true,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: true,
                    candidate_off_on_strict_receipts_present: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: true,
                    candidate_off_strict_receipt_present: true,
                    candidate_on_strict_receipt_present: true,
                    strict_receipts_bind_owner_identity: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        assert_eq!(
            emitter.decision,
            "same_callsite_candidate_receipt_emitter_ready_runtime_disabled"
        );
        assert_eq!(emitter.model_architecture, "qwen2");
        assert_eq!(emitter.model_sha256, qwen25_sha);
        assert_eq!(emitter.candidate_path, "qwen25_feed_forward_down_proj_no_bias_candidate");
        assert!(emitter.runtime_owner_boundary_defined);
        assert!(emitter.apply_linear_runtime_owner_present);
        assert!(emitter.owner_has_apply_linear_inputs);
        assert!(emitter.owner_has_linear_weight_access);
        assert!(emitter.candidate_compute_callable);
        assert!(emitter.same_callsite_candidate_receipt_emitter_present);
        assert!(emitter.candidate_off_strict_receipt_present);
        assert!(emitter.candidate_on_strict_receipt_present);
        assert!(emitter.strict_receipts_bind_owner_identity);
        assert!(emitter.prompt_ids_preserved);
        assert!(emitter.generated_ids_preserved);
        assert!(emitter.decoded_text_preserved);
        assert!(emitter.fail_closed_conditions.is_empty());
        assert!(emitter.preserves_normal_inference());
        assert!(!emitter.candidate_execution_enabled);
        assert!(!emitter.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_same_callsite_receipt_emitter_boundary_rejects_missing_emitter() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                true,
                true,
                true,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: true,
                    candidate_off_on_strict_receipts_present: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: false,
                    candidate_off_strict_receipt_present: true,
                    candidate_on_strict_receipt_present: true,
                    strict_receipts_bind_owner_identity: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        assert_eq!(emitter.decision, "same_callsite_candidate_receipt_emitter_blocked_fail_closed");
        assert_eq!(
            emitter.remaining_runtime_selection_blocker,
            "same_callsite_candidate_receipt_emitter"
        );
        assert!(emitter.runtime_owner_boundary_defined);
        assert!(!emitter.same_callsite_candidate_receipt_emitter_present);
        assert!(
            emitter
                .fail_closed_conditions
                .contains(&"same_callsite_candidate_receipt_emitter_missing")
        );
        assert!(emitter.preserves_normal_inference());
        assert!(!emitter.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_same_callsite_off_on_receipt_boundary_records_candidate_on_artifact_blocker() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                false,
                false,
                false,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: false,
                    candidate_off_on_strict_receipts_present: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: true,
                    candidate_off_strict_receipt_present: false,
                    candidate_on_strict_receipt_present: false,
                    strict_receipts_bind_owner_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );

        let receipts =
            DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary::from_same_callsite_receipt_emitter_boundary(
                &emitter,
                DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptInputs {
                    candidate_off_strict_receipt_artifact_present: true,
                    candidate_on_strict_receipt_artifact_present: false,
                    candidate_off_receipt_binds_owner_identity: true,
                    candidate_on_receipt_binds_owner_identity: false,
                    candidate_off_on_same_callsite_identity: false,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );

        assert_eq!(
            receipts.decision,
            "same_callsite_candidate_off_on_strict_receipts_blocked_fail_closed"
        );
        assert_eq!(
            receipts.remaining_runtime_selection_blocker,
            "candidate_on_strict_receipt_artifact"
        );
        assert!(receipts.same_callsite_receipt_emitter_ready);
        assert!(receipts.candidate_off_strict_receipt_artifact_present);
        assert!(!receipts.candidate_on_strict_receipt_artifact_present);
        assert!(receipts.candidate_off_receipt_binds_owner_identity);
        assert!(!receipts.candidate_on_receipt_binds_owner_identity);
        assert!(!receipts.candidate_off_on_same_callsite_identity);
        assert!(!receipts.generated_ids_preserved);
        assert!(!receipts.decoded_text_preserved);
        assert!(!receipts.fail_closed_conditions.contains(&"candidate_on_strict_receipt_missing"));
        assert!(
            receipts
                .fail_closed_conditions
                .contains(&"candidate_on_strict_receipt_artifact_missing")
        );
        assert!(
            receipts
                .fail_closed_conditions
                .contains(&"candidate_on_receipt_does_not_bind_owner_identity")
        );
        assert!(
            receipts
                .fail_closed_conditions
                .contains(&"candidate_off_on_callsite_identity_mismatch")
        );
        assert!(receipts.preserves_normal_inference());
        assert!(!receipts.candidate_execution_enabled);
        assert!(!receipts.normal_inference_runtime_selection_enabled);
        assert!(!receipts.allocation_reduction_claim);
        assert!(!receipts.timing_improvement_claim);
        assert!(!receipts.speedup_claim);
    }

    #[test]
    fn no_bias_same_callsite_off_on_receipt_boundary_models_ready_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let pair_gate = slm_cpu_216_ready_pair_gate(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                true,
                true,
                true,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: true,
                    candidate_off_on_strict_receipts_present: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );
        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: true,
                    candidate_off_strict_receipt_present: true,
                    candidate_on_strict_receipt_present: true,
                    strict_receipts_bind_owner_identity: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        let receipts =
            DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary::from_same_callsite_receipt_emitter_boundary(
                &emitter,
                DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptInputs {
                    candidate_off_strict_receipt_artifact_present: true,
                    candidate_on_strict_receipt_artifact_present: true,
                    candidate_off_receipt_binds_owner_identity: true,
                    candidate_on_receipt_binds_owner_identity: true,
                    candidate_off_on_same_callsite_identity: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        assert_eq!(
            receipts.decision,
            "same_callsite_candidate_off_on_strict_receipts_ready_runtime_disabled"
        );
        assert_eq!(receipts.model_architecture, "qwen2");
        assert_eq!(receipts.model_sha256, qwen25_sha);
        assert_eq!(receipts.candidate_path, "qwen25_feed_forward_down_proj_no_bias_candidate");
        assert!(receipts.same_callsite_receipt_emitter_ready);
        assert!(receipts.candidate_off_strict_receipt_artifact_present);
        assert!(receipts.candidate_on_strict_receipt_artifact_present);
        assert!(receipts.candidate_off_receipt_binds_owner_identity);
        assert!(receipts.candidate_on_receipt_binds_owner_identity);
        assert!(receipts.candidate_off_on_same_callsite_identity);
        assert!(receipts.prompt_ids_preserved);
        assert!(receipts.generated_ids_preserved);
        assert!(receipts.decoded_text_preserved);
        assert!(receipts.fail_closed_conditions.is_empty());
        assert!(receipts.preserves_normal_inference());
        assert!(!receipts.candidate_execution_enabled);
        assert!(!receipts.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_same_callsite_off_on_receipt_boundary_rejects_incomplete_emitter() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                true,
                true,
                true,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: true,
                    candidate_off_on_strict_receipts_present: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );
        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: false,
                    candidate_off_strict_receipt_present: true,
                    candidate_on_strict_receipt_present: true,
                    strict_receipts_bind_owner_identity: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        let receipts =
            DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary::from_same_callsite_receipt_emitter_boundary(
                &emitter,
                DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptInputs {
                    candidate_off_strict_receipt_artifact_present: true,
                    candidate_on_strict_receipt_artifact_present: true,
                    candidate_off_receipt_binds_owner_identity: true,
                    candidate_on_receipt_binds_owner_identity: true,
                    candidate_off_on_same_callsite_identity: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                },
            );

        assert_eq!(receipts.decision, "blocked_fail_closed");
        assert_eq!(
            receipts.remaining_runtime_selection_blocker,
            "same_callsite_candidate_receipt_emitter_boundary"
        );
        assert!(!receipts.same_callsite_receipt_emitter_ready);
        assert!(
            receipts
                .fail_closed_conditions
                .contains(&"same_callsite_candidate_receipt_emitter_missing")
        );
        assert!(
            receipts
                .fail_closed_conditions
                .contains(&"same_callsite_receipt_emitter_boundary_not_ready")
        );
        assert!(receipts.preserves_normal_inference());
        assert!(!receipts.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_receipt_gated_candidate_execution_records_off_on_boundary_blocker() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                false,
                false,
                false,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: false,
                    candidate_off_on_strict_receipts_present: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: true,
                    candidate_off_strict_receipt_present: false,
                    candidate_on_strict_receipt_present: false,
                    strict_receipts_bind_owner_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let receipts =
            DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary::from_same_callsite_receipt_emitter_boundary(
                &emitter,
                DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptInputs {
                    candidate_off_strict_receipt_artifact_present: false,
                    candidate_on_strict_receipt_artifact_present: false,
                    candidate_off_receipt_binds_owner_identity: false,
                    candidate_on_receipt_binds_owner_identity: false,
                    candidate_off_on_same_callsite_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );

        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(execution.decision, "blocked_fail_closed");
        assert_eq!(
            execution.remaining_runtime_selection_blocker,
            "same_callsite_candidate_off_on_strict_receipts"
        );
        assert!(!execution.off_on_strict_receipt_boundary_ready);
        assert!(!execution.candidate_execution_attempt_allowed);
        assert!(
            execution.fail_closed_conditions.contains(&"off_on_strict_receipt_boundary_not_ready")
        );
        assert!(
            execution
                .fail_closed_conditions
                .contains(&"candidate_on_strict_receipt_artifact_missing")
        );
        assert!(execution.preserves_normal_inference());
        assert!(!execution.candidate_execution_enabled);
        assert!(!execution.normal_inference_runtime_selection_enabled);
        assert!(!execution.allocation_reduction_claim);
        assert!(!execution.timing_improvement_claim);
        assert!(!execution.speedup_claim);
    }

    #[test]
    fn no_bias_receipt_gated_candidate_execution_blocks_missing_explicit_gate() {
        let receipts = slm_cpu_220_ready_off_on_boundary(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );

        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: false,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(execution.decision, "receipt_gated_candidate_execution_blocked_fail_closed");
        assert_eq!(
            execution.remaining_runtime_selection_blocker,
            "explicit_candidate_execution_gate"
        );
        assert!(execution.off_on_strict_receipt_boundary_ready);
        assert!(!execution.candidate_execution_attempt_allowed);
        assert!(
            execution
                .fail_closed_conditions
                .contains(&"explicit_candidate_execution_gate_not_requested")
        );
        assert!(execution.preserves_normal_inference());
        assert!(!execution.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_receipt_gated_candidate_execution_models_ready_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let receipts = slm_cpu_220_ready_off_on_boundary(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );

        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(
            execution.decision,
            "receipt_gated_candidate_execution_prereqs_ready_runtime_disabled"
        );
        assert_eq!(execution.model_architecture, "qwen2");
        assert_eq!(execution.model_sha256, qwen25_sha);
        assert_eq!(execution.candidate_path, "qwen25_feed_forward_down_proj_no_bias_candidate");
        assert!(execution.off_on_strict_receipt_boundary_ready);
        assert!(execution.explicit_gate_identity_present);
        assert!(execution.descriptor_identity_present);
        assert!(execution.owner_callsite_identity_present);
        assert!(execution.prompt_generated_text_digests_bound);
        assert!(execution.candidate_execution_attempt_allowed);
        assert!(execution.fail_closed_conditions.is_empty());
        assert!(execution.preserves_normal_inference());
        assert!(!execution.candidate_execution_enabled);
        assert!(!execution.normal_inference_runtime_selection_enabled);
        assert!(!execution.allocation_reduction_claim);
        assert!(!execution.timing_improvement_claim);
        assert!(!execution.speedup_claim);
    }

    #[test]
    fn no_bias_strict_receipt_artifact_pair_records_execution_boundary_blocker() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                false,
                false,
                false,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: false,
                    candidate_off_on_strict_receipts_present: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: true,
                    candidate_off_strict_receipt_present: false,
                    candidate_on_strict_receipt_present: false,
                    strict_receipts_bind_owner_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let receipts =
            DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary::from_same_callsite_receipt_emitter_boundary(
                &emitter,
                DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptInputs {
                    candidate_off_strict_receipt_artifact_present: false,
                    candidate_on_strict_receipt_artifact_present: false,
                    candidate_off_receipt_binds_owner_identity: false,
                    candidate_on_receipt_binds_owner_identity: false,
                    candidate_off_on_same_callsite_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: true,
                    default_runtime_path_preserved: true,
                },
            );

        let artifact_pair =
            DenseLinearNoBiasStrictReceiptArtifactPairBoundary::from_receipt_gated_candidate_execution_boundary(
                &execution,
                DenseLinearNoBiasStrictReceiptArtifactPairInputs {
                    candidate_off_strict_receipt_artifact_path: None,
                    candidate_on_strict_receipt_artifact_path: None,
                    candidate_off_strict_receipt_artifact_present: false,
                    candidate_on_strict_receipt_artifact_present: false,
                    candidate_off_receipt_binds_gate_identity: false,
                    candidate_on_receipt_binds_gate_identity: false,
                    candidate_off_receipt_binds_descriptor_identity: false,
                    candidate_on_receipt_binds_descriptor_identity: false,
                    candidate_off_receipt_binds_owner_callsite_identity: false,
                    candidate_on_receipt_binds_owner_callsite_identity: false,
                    candidate_off_on_same_callsite_identity: false,
                    candidate_off_on_same_prompt_digest: false,
                    candidate_off_on_same_generated_digest: false,
                    candidate_off_on_same_decoded_text_digest: false,
                    candidate_off_on_same_model_backend_identity: false,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(artifact_pair.decision, "blocked_fail_closed");
        assert_eq!(
            artifact_pair.remaining_runtime_selection_blocker,
            "receipt_gated_candidate_execution_boundary"
        );
        assert!(!artifact_pair.receipt_gated_candidate_execution_boundary_ready);
        assert!(!artifact_pair.candidate_execution_attempt_allowed);
        assert!(
            artifact_pair
                .fail_closed_conditions
                .contains(&"receipt_gated_candidate_execution_boundary_not_ready")
        );
        assert!(
            artifact_pair
                .fail_closed_conditions
                .contains(&"candidate_on_strict_receipt_artifact_missing")
        );
        assert!(artifact_pair.preserves_normal_inference());
        assert!(!artifact_pair.candidate_execution_enabled);
        assert!(!artifact_pair.normal_inference_runtime_selection_enabled);
        assert!(!artifact_pair.allocation_reduction_claim);
        assert!(!artifact_pair.timing_improvement_claim);
        assert!(!artifact_pair.speedup_claim);
    }

    #[test]
    fn no_bias_strict_receipt_artifact_pair_blocks_missing_candidate_on_artifact() {
        let receipts = slm_cpu_220_ready_off_on_boundary(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: true,
                    default_runtime_path_preserved: true,
                },
            );

        let artifact_pair =
            DenseLinearNoBiasStrictReceiptArtifactPairBoundary::from_receipt_gated_candidate_execution_boundary(
                &execution,
                DenseLinearNoBiasStrictReceiptArtifactPairInputs {
                    candidate_off_strict_receipt_artifact_path: Some("ci/slm-cpu/qwen3-candidate-off.json"),
                    candidate_on_strict_receipt_artifact_path: None,
                    candidate_off_strict_receipt_artifact_present: true,
                    candidate_on_strict_receipt_artifact_present: false,
                    candidate_off_receipt_binds_gate_identity: true,
                    candidate_on_receipt_binds_gate_identity: false,
                    candidate_off_receipt_binds_descriptor_identity: true,
                    candidate_on_receipt_binds_descriptor_identity: false,
                    candidate_off_receipt_binds_owner_callsite_identity: true,
                    candidate_on_receipt_binds_owner_callsite_identity: false,
                    candidate_off_on_same_callsite_identity: false,
                    candidate_off_on_same_prompt_digest: true,
                    candidate_off_on_same_generated_digest: false,
                    candidate_off_on_same_decoded_text_digest: false,
                    candidate_off_on_same_model_backend_identity: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(
            artifact_pair.decision,
            "same_callsite_strict_receipt_artifact_pair_blocked_fail_closed"
        );
        assert_eq!(
            artifact_pair.remaining_runtime_selection_blocker,
            "candidate_on_strict_receipt_artifact"
        );
        assert!(artifact_pair.receipt_gated_candidate_execution_boundary_ready);
        assert!(artifact_pair.candidate_off_strict_receipt_artifact_present);
        assert!(!artifact_pair.candidate_on_strict_receipt_artifact_present);
        assert_eq!(
            artifact_pair.candidate_off_strict_receipt_artifact_path.as_deref(),
            Some("ci/slm-cpu/qwen3-candidate-off.json")
        );
        assert!(!artifact_pair.candidate_execution_attempt_allowed);
        assert!(
            artifact_pair
                .fail_closed_conditions
                .contains(&"candidate_on_strict_receipt_artifact_missing")
        );
        assert!(artifact_pair.preserves_normal_inference());
        assert!(!artifact_pair.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_strict_receipt_artifact_pair_models_ready_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let receipts = slm_cpu_220_ready_off_on_boundary(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );
        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: true,
                    default_runtime_path_preserved: true,
                },
            );

        let artifact_pair =
            DenseLinearNoBiasStrictReceiptArtifactPairBoundary::from_receipt_gated_candidate_execution_boundary(
                &execution,
                DenseLinearNoBiasStrictReceiptArtifactPairInputs {
                    candidate_off_strict_receipt_artifact_path: Some("ci/slm-cpu/qwen25-candidate-off.json"),
                    candidate_on_strict_receipt_artifact_path: Some("ci/slm-cpu/qwen25-candidate-on.json"),
                    candidate_off_strict_receipt_artifact_present: true,
                    candidate_on_strict_receipt_artifact_present: true,
                    candidate_off_receipt_binds_gate_identity: true,
                    candidate_on_receipt_binds_gate_identity: true,
                    candidate_off_receipt_binds_descriptor_identity: true,
                    candidate_on_receipt_binds_descriptor_identity: true,
                    candidate_off_receipt_binds_owner_callsite_identity: true,
                    candidate_on_receipt_binds_owner_callsite_identity: true,
                    candidate_off_on_same_callsite_identity: true,
                    candidate_off_on_same_prompt_digest: true,
                    candidate_off_on_same_generated_digest: true,
                    candidate_off_on_same_decoded_text_digest: true,
                    candidate_off_on_same_model_backend_identity: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(
            artifact_pair.decision,
            "same_callsite_strict_receipt_artifact_pair_ready_runtime_disabled"
        );
        assert_eq!(artifact_pair.model_architecture, "qwen2");
        assert_eq!(artifact_pair.model_sha256, qwen25_sha);
        assert_eq!(artifact_pair.candidate_path, "qwen25_feed_forward_down_proj_no_bias_candidate");
        assert!(artifact_pair.receipt_gated_candidate_execution_boundary_ready);
        assert!(artifact_pair.candidate_off_strict_receipt_artifact_present);
        assert!(artifact_pair.candidate_on_strict_receipt_artifact_present);
        assert!(artifact_pair.candidate_off_receipt_binds_gate_identity);
        assert!(artifact_pair.candidate_on_receipt_binds_gate_identity);
        assert!(artifact_pair.candidate_off_receipt_binds_descriptor_identity);
        assert!(artifact_pair.candidate_on_receipt_binds_descriptor_identity);
        assert!(artifact_pair.candidate_off_receipt_binds_owner_callsite_identity);
        assert!(artifact_pair.candidate_on_receipt_binds_owner_callsite_identity);
        assert!(artifact_pair.candidate_off_on_same_callsite_identity);
        assert!(artifact_pair.candidate_off_on_same_prompt_digest);
        assert!(artifact_pair.candidate_off_on_same_generated_digest);
        assert!(artifact_pair.candidate_off_on_same_decoded_text_digest);
        assert!(artifact_pair.candidate_off_on_same_model_backend_identity);
        assert!(artifact_pair.candidate_execution_attempt_allowed);
        assert!(artifact_pair.fail_closed_conditions.is_empty());
        assert!(artifact_pair.preserves_normal_inference());
        assert!(!artifact_pair.candidate_execution_enabled);
        assert!(!artifact_pair.normal_inference_runtime_selection_enabled);
        assert!(!artifact_pair.allocation_reduction_claim);
        assert!(!artifact_pair.timing_improvement_claim);
        assert!(!artifact_pair.speedup_claim);
    }

    #[test]
    fn no_bias_strict_artifact_capture_records_pair_boundary_blocker() {
        let pair_gate = slm_cpu_216_ready_pair_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let behavior_gate =
            DenseLinearNoBiasCandidateOnBehaviorEvidenceGate::from_candidate_off_on_pair_gate(
                &pair_gate, true, true, true,
            );
        let attachment =
            DenseLinearNoBiasCandidateRuntimeAttachmentBoundary::from_candidate_on_behavior_gate(
                &behavior_gate,
                true,
                true,
                false,
                false,
                false,
            );
        let owner =
            DenseLinearNoBiasCandidateRuntimeOwnerBoundary::from_runtime_attachment_boundary(
                &attachment,
                DenseLinearNoBiasCandidateRuntimeOwnerInputs {
                    apply_linear_runtime_owner_present: true,
                    owner_has_apply_linear_inputs: true,
                    owner_has_linear_weight_access: true,
                    candidate_compute_callable: true,
                    same_callsite_candidate_on_receipt_emitter_wired: false,
                    candidate_off_on_strict_receipts_present: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let emitter =
            DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary::from_runtime_owner_boundary(
                &owner,
                DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterInputs {
                    same_callsite_candidate_receipt_emitter_present: true,
                    candidate_off_strict_receipt_present: false,
                    candidate_on_strict_receipt_present: false,
                    strict_receipts_bind_owner_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let receipts =
            DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary::from_same_callsite_receipt_emitter_boundary(
                &emitter,
                DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptInputs {
                    candidate_off_strict_receipt_artifact_present: false,
                    candidate_on_strict_receipt_artifact_present: false,
                    candidate_off_receipt_binds_owner_identity: false,
                    candidate_on_receipt_binds_owner_identity: false,
                    candidate_off_on_same_callsite_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                },
            );
        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: true,
                    default_runtime_path_preserved: true,
                },
            );
        let artifact_pair =
            DenseLinearNoBiasStrictReceiptArtifactPairBoundary::from_receipt_gated_candidate_execution_boundary(
                &execution,
                DenseLinearNoBiasStrictReceiptArtifactPairInputs {
                    candidate_off_strict_receipt_artifact_path: None,
                    candidate_on_strict_receipt_artifact_path: None,
                    candidate_off_strict_receipt_artifact_present: false,
                    candidate_on_strict_receipt_artifact_present: false,
                    candidate_off_receipt_binds_gate_identity: false,
                    candidate_on_receipt_binds_gate_identity: false,
                    candidate_off_receipt_binds_descriptor_identity: false,
                    candidate_on_receipt_binds_descriptor_identity: false,
                    candidate_off_receipt_binds_owner_callsite_identity: false,
                    candidate_on_receipt_binds_owner_callsite_identity: false,
                    candidate_off_on_same_callsite_identity: false,
                    candidate_off_on_same_prompt_digest: false,
                    candidate_off_on_same_generated_digest: false,
                    candidate_off_on_same_decoded_text_digest: false,
                    candidate_off_on_same_model_backend_identity: false,
                    default_runtime_path_preserved: true,
                },
            );

        let capture =
            DenseLinearNoBiasStrictArtifactCaptureBoundary::from_strict_receipt_artifact_pair_boundary(
                &artifact_pair,
                DenseLinearNoBiasStrictArtifactCaptureInputs {
                    candidate_off_capture_artifact_validated: false,
                    candidate_on_capture_artifact_validated: false,
                    candidate_off_capture_command_recorded: false,
                    candidate_on_capture_command_recorded: false,
                    candidate_off_on_capture_same_callsite_identity: false,
                    candidate_off_on_capture_same_prompt_digest: false,
                    candidate_off_on_capture_same_generated_digest: false,
                    candidate_off_on_capture_same_decoded_text_digest: false,
                    candidate_off_on_capture_same_model_backend_identity: false,
                    capture_blocker_recorded: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(capture.decision, "blocked_fail_closed");
        assert_eq!(
            capture.remaining_runtime_selection_blocker,
            "strict_receipt_artifact_pair_boundary"
        );
        assert!(!capture.strict_receipt_artifact_pair_boundary_ready);
        assert!(!capture.candidate_execution_prereqs_complete);
        assert!(
            capture
                .fail_closed_conditions
                .contains(&"strict_receipt_artifact_pair_boundary_not_ready")
        );
        assert!(capture.preserves_normal_inference());
        assert!(!capture.candidate_execution_enabled);
        assert!(!capture.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_strict_artifact_capture_blocks_missing_candidate_on_capture() {
        let receipts = slm_cpu_220_ready_off_on_boundary(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: true,
                    default_runtime_path_preserved: true,
                },
            );
        let artifact_pair =
            DenseLinearNoBiasStrictReceiptArtifactPairBoundary::from_receipt_gated_candidate_execution_boundary(
                &execution,
                DenseLinearNoBiasStrictReceiptArtifactPairInputs {
                    candidate_off_strict_receipt_artifact_path: Some("ci/slm-cpu/qwen3-candidate-off.json"),
                    candidate_on_strict_receipt_artifact_path: Some("ci/slm-cpu/qwen3-candidate-on.json"),
                    candidate_off_strict_receipt_artifact_present: true,
                    candidate_on_strict_receipt_artifact_present: true,
                    candidate_off_receipt_binds_gate_identity: true,
                    candidate_on_receipt_binds_gate_identity: true,
                    candidate_off_receipt_binds_descriptor_identity: true,
                    candidate_on_receipt_binds_descriptor_identity: true,
                    candidate_off_receipt_binds_owner_callsite_identity: true,
                    candidate_on_receipt_binds_owner_callsite_identity: true,
                    candidate_off_on_same_callsite_identity: true,
                    candidate_off_on_same_prompt_digest: true,
                    candidate_off_on_same_generated_digest: true,
                    candidate_off_on_same_decoded_text_digest: true,
                    candidate_off_on_same_model_backend_identity: true,
                    default_runtime_path_preserved: true,
                },
            );

        let capture =
            DenseLinearNoBiasStrictArtifactCaptureBoundary::from_strict_receipt_artifact_pair_boundary(
                &artifact_pair,
                DenseLinearNoBiasStrictArtifactCaptureInputs {
                    candidate_off_capture_artifact_validated: true,
                    candidate_on_capture_artifact_validated: false,
                    candidate_off_capture_command_recorded: true,
                    candidate_on_capture_command_recorded: true,
                    candidate_off_on_capture_same_callsite_identity: true,
                    candidate_off_on_capture_same_prompt_digest: true,
                    candidate_off_on_capture_same_generated_digest: false,
                    candidate_off_on_capture_same_decoded_text_digest: false,
                    candidate_off_on_capture_same_model_backend_identity: true,
                    capture_blocker_recorded: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(capture.decision, "same_callsite_strict_artifact_capture_blocked_fail_closed");
        assert_eq!(capture.remaining_runtime_selection_blocker, "candidate_on_capture_artifact");
        assert!(capture.strict_receipt_artifact_pair_boundary_ready);
        assert!(capture.candidate_off_capture_artifact_validated);
        assert!(!capture.candidate_on_capture_artifact_validated);
        assert!(!capture.candidate_execution_prereqs_complete);
        assert!(
            capture.fail_closed_conditions.contains(&"candidate_on_capture_artifact_not_validated")
        );
        assert!(capture.preserves_normal_inference());
        assert!(!capture.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_strict_artifact_capture_models_ready_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let receipts = slm_cpu_220_ready_off_on_boundary(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );
        let execution =
            DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary::from_off_on_strict_receipt_boundary(
                &receipts,
                DenseLinearNoBiasReceiptGatedCandidateExecutionInputs {
                    explicit_gate_identity_present: true,
                    descriptor_identity_present: true,
                    owner_callsite_identity_present: true,
                    prompt_generated_text_digests_bound: true,
                    explicit_candidate_execution_gate_requested: true,
                    default_runtime_path_preserved: true,
                },
            );
        let artifact_pair =
            DenseLinearNoBiasStrictReceiptArtifactPairBoundary::from_receipt_gated_candidate_execution_boundary(
                &execution,
                DenseLinearNoBiasStrictReceiptArtifactPairInputs {
                    candidate_off_strict_receipt_artifact_path: Some("ci/slm-cpu/qwen25-candidate-off.json"),
                    candidate_on_strict_receipt_artifact_path: Some("ci/slm-cpu/qwen25-candidate-on.json"),
                    candidate_off_strict_receipt_artifact_present: true,
                    candidate_on_strict_receipt_artifact_present: true,
                    candidate_off_receipt_binds_gate_identity: true,
                    candidate_on_receipt_binds_gate_identity: true,
                    candidate_off_receipt_binds_descriptor_identity: true,
                    candidate_on_receipt_binds_descriptor_identity: true,
                    candidate_off_receipt_binds_owner_callsite_identity: true,
                    candidate_on_receipt_binds_owner_callsite_identity: true,
                    candidate_off_on_same_callsite_identity: true,
                    candidate_off_on_same_prompt_digest: true,
                    candidate_off_on_same_generated_digest: true,
                    candidate_off_on_same_decoded_text_digest: true,
                    candidate_off_on_same_model_backend_identity: true,
                    default_runtime_path_preserved: true,
                },
            );

        let capture =
            DenseLinearNoBiasStrictArtifactCaptureBoundary::from_strict_receipt_artifact_pair_boundary(
                &artifact_pair,
                DenseLinearNoBiasStrictArtifactCaptureInputs {
                    candidate_off_capture_artifact_validated: true,
                    candidate_on_capture_artifact_validated: true,
                    candidate_off_capture_command_recorded: true,
                    candidate_on_capture_command_recorded: true,
                    candidate_off_on_capture_same_callsite_identity: true,
                    candidate_off_on_capture_same_prompt_digest: true,
                    candidate_off_on_capture_same_generated_digest: true,
                    candidate_off_on_capture_same_decoded_text_digest: true,
                    candidate_off_on_capture_same_model_backend_identity: true,
                    capture_blocker_recorded: false,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(
            capture.decision,
            "same_callsite_strict_artifact_capture_ready_runtime_disabled"
        );
        assert_eq!(capture.model_architecture, "qwen2");
        assert_eq!(capture.model_sha256, qwen25_sha);
        assert_eq!(capture.candidate_path, "qwen25_feed_forward_down_proj_no_bias_candidate");
        assert!(capture.strict_receipt_artifact_pair_boundary_ready);
        assert!(capture.candidate_off_capture_artifact_validated);
        assert!(capture.candidate_on_capture_artifact_validated);
        assert!(capture.candidate_execution_prereqs_complete);
        assert!(capture.fail_closed_conditions.is_empty());
        assert!(capture.preserves_normal_inference());
        assert!(!capture.candidate_execution_enabled);
        assert!(!capture.normal_inference_runtime_selection_enabled);
        assert!(!capture.allocation_reduction_claim);
        assert!(!capture.timing_improvement_claim);
        assert!(!capture.speedup_claim);
    }

    #[test]
    fn no_bias_strict_capture_artifact_pair_records_capture_boundary_blocker() {
        let capture = slm_cpu_223_blocked_strict_artifact_capture_boundary(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );

        let pair =
            DenseLinearNoBiasStrictCaptureArtifactPairBoundary::from_strict_artifact_capture_boundary(
                &capture,
                DenseLinearNoBiasStrictCaptureArtifactPairInputs {
                    candidate_off_strict_capture_artifact_path: None,
                    candidate_on_strict_capture_artifact_path: None,
                    candidate_off_strict_capture_artifact_present: false,
                    candidate_on_strict_capture_artifact_present: false,
                    candidate_off_capture_command_recorded: false,
                    candidate_on_capture_command_recorded: false,
                    candidate_off_capture_binds_gate_identity: false,
                    candidate_on_capture_binds_gate_identity: false,
                    candidate_off_capture_binds_descriptor_identity: false,
                    candidate_on_capture_binds_descriptor_identity: false,
                    candidate_off_capture_binds_owner_callsite_identity: false,
                    candidate_on_capture_binds_owner_callsite_identity: false,
                    candidate_off_on_capture_same_callsite_identity: false,
                    candidate_off_on_capture_same_prompt_digest: false,
                    candidate_off_on_capture_same_generated_digest: false,
                    candidate_off_on_capture_same_decoded_text_digest: false,
                    candidate_off_on_capture_same_model_backend_identity: false,
                    capture_prerequisite_blocker_recorded: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(pair.decision, "blocked_fail_closed");
        assert_eq!(pair.remaining_runtime_selection_blocker, "strict_artifact_capture_boundary");
        assert!(!pair.strict_artifact_capture_boundary_ready);
        assert!(!pair.strict_capture_artifact_pair_validated);
        assert!(!pair.candidate_execution_prereqs_complete);
        assert!(
            pair.fail_closed_conditions.contains(&"strict_artifact_capture_boundary_not_ready")
        );
        assert!(pair.preserves_normal_inference());
        assert!(!pair.candidate_execution_enabled);
        assert!(!pair.normal_inference_runtime_selection_enabled);
    }

    #[test]
    fn no_bias_strict_capture_artifact_pair_blocks_missing_candidate_on_artifact() {
        let capture = slm_cpu_223_ready_strict_artifact_capture_boundary(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );

        let pair =
            DenseLinearNoBiasStrictCaptureArtifactPairBoundary::from_strict_artifact_capture_boundary(
                &capture,
                DenseLinearNoBiasStrictCaptureArtifactPairInputs {
                    candidate_off_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen3-candidate-off-strict-capture.json",
                    ),
                    candidate_on_strict_capture_artifact_path: None,
                    candidate_off_strict_capture_artifact_present: true,
                    candidate_on_strict_capture_artifact_present: false,
                    candidate_off_capture_command_recorded: true,
                    candidate_on_capture_command_recorded: true,
                    candidate_off_capture_binds_gate_identity: true,
                    candidate_on_capture_binds_gate_identity: true,
                    candidate_off_capture_binds_descriptor_identity: true,
                    candidate_on_capture_binds_descriptor_identity: true,
                    candidate_off_capture_binds_owner_callsite_identity: true,
                    candidate_on_capture_binds_owner_callsite_identity: true,
                    candidate_off_on_capture_same_callsite_identity: true,
                    candidate_off_on_capture_same_prompt_digest: true,
                    candidate_off_on_capture_same_generated_digest: false,
                    candidate_off_on_capture_same_decoded_text_digest: false,
                    candidate_off_on_capture_same_model_backend_identity: true,
                    capture_prerequisite_blocker_recorded: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(pair.decision, "strict_capture_artifact_pair_blocked_fail_closed");
        assert_eq!(
            pair.remaining_runtime_selection_blocker,
            "candidate_on_strict_capture_artifact"
        );
        assert!(pair.strict_artifact_capture_boundary_ready);
        assert!(pair.candidate_off_strict_capture_artifact_present);
        assert!(!pair.candidate_on_strict_capture_artifact_present);
        assert!(!pair.strict_capture_artifact_pair_validated);
        assert!(
            pair.fail_closed_conditions.contains(&"candidate_on_strict_capture_artifact_missing")
        );
        assert!(pair.preserves_normal_inference());
        assert!(!pair.candidate_execution_enabled);
    }

    #[test]
    fn no_bias_strict_capture_artifact_pair_validated_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let capture = slm_cpu_223_ready_strict_artifact_capture_boundary(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );

        let pair =
            DenseLinearNoBiasStrictCaptureArtifactPairBoundary::from_strict_artifact_capture_boundary(
                &capture,
                DenseLinearNoBiasStrictCaptureArtifactPairInputs {
                    candidate_off_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen25-candidate-off-strict-capture.json",
                    ),
                    candidate_on_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen25-candidate-on-strict-capture.json",
                    ),
                    candidate_off_strict_capture_artifact_present: true,
                    candidate_on_strict_capture_artifact_present: true,
                    candidate_off_capture_command_recorded: true,
                    candidate_on_capture_command_recorded: true,
                    candidate_off_capture_binds_gate_identity: true,
                    candidate_on_capture_binds_gate_identity: true,
                    candidate_off_capture_binds_descriptor_identity: true,
                    candidate_on_capture_binds_descriptor_identity: true,
                    candidate_off_capture_binds_owner_callsite_identity: true,
                    candidate_on_capture_binds_owner_callsite_identity: true,
                    candidate_off_on_capture_same_callsite_identity: true,
                    candidate_off_on_capture_same_prompt_digest: true,
                    candidate_off_on_capture_same_generated_digest: true,
                    candidate_off_on_capture_same_decoded_text_digest: true,
                    candidate_off_on_capture_same_model_backend_identity: true,
                    capture_prerequisite_blocker_recorded: false,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(pair.decision, "strict_capture_artifact_pair_validated_runtime_disabled");
        assert_eq!(pair.model_architecture, "qwen2");
        assert_eq!(pair.model_sha256, qwen25_sha);
        assert_eq!(pair.candidate_path, "qwen25_feed_forward_down_proj_no_bias_candidate");
        assert!(pair.strict_artifact_capture_boundary_ready);
        assert!(pair.candidate_off_strict_capture_artifact_present);
        assert!(pair.candidate_on_strict_capture_artifact_present);
        assert!(pair.strict_capture_artifact_pair_validated);
        assert!(pair.candidate_execution_prereqs_complete);
        assert!(pair.fail_closed_conditions.is_empty());
        assert!(pair.preserves_normal_inference());
        assert!(!pair.candidate_execution_enabled);
        assert!(!pair.normal_inference_runtime_selection_enabled);
        assert!(!pair.allocation_reduction_claim);
        assert!(!pair.timing_improvement_claim);
        assert!(!pair.speedup_claim);
    }

    #[test]
    fn no_bias_runtime_attempt_records_missing_runtime_attachment_blocker() {
        let capture = slm_cpu_223_ready_strict_artifact_capture_boundary(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let pair =
            DenseLinearNoBiasStrictCaptureArtifactPairBoundary::from_strict_artifact_capture_boundary(
                &capture,
                DenseLinearNoBiasStrictCaptureArtifactPairInputs {
                    candidate_off_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen3-candidate-off-strict-capture.json",
                    ),
                    candidate_on_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen3-candidate-on-strict-capture.json",
                    ),
                    candidate_off_strict_capture_artifact_present: true,
                    candidate_on_strict_capture_artifact_present: true,
                    candidate_off_capture_command_recorded: true,
                    candidate_on_capture_command_recorded: true,
                    candidate_off_capture_binds_gate_identity: true,
                    candidate_on_capture_binds_gate_identity: true,
                    candidate_off_capture_binds_descriptor_identity: true,
                    candidate_on_capture_binds_descriptor_identity: true,
                    candidate_off_capture_binds_owner_callsite_identity: true,
                    candidate_on_capture_binds_owner_callsite_identity: true,
                    candidate_off_on_capture_same_callsite_identity: true,
                    candidate_off_on_capture_same_prompt_digest: true,
                    candidate_off_on_capture_same_generated_digest: true,
                    candidate_off_on_capture_same_decoded_text_digest: true,
                    candidate_off_on_capture_same_model_backend_identity: true,
                    capture_prerequisite_blocker_recorded: false,
                    default_runtime_path_preserved: true,
                },
            );

        let attempt = DenseLinearNoBiasRuntimeAttemptBoundary::from_strict_capture_artifact_pair(
            &pair,
            DenseLinearNoBiasRuntimeAttemptInputs {
                explicit_candidate_execution_gate_requested: true,
                runtime_hook_registry_attachment_present: false,
                runtime_hook_descriptor_binds_selector_identity: false,
                runtime_hook_descriptor_binds_strict_capture_pair: false,
                apply_linear_dispatch_wired_to_no_bias_candidate: false,
                feed_forward_down_proj_scope_preserved: true,
                default_runtime_path_preserved: true,
            },
        );

        assert_eq!(attempt.decision, "candidate_execution_attempt_blocked_fail_closed");
        assert_eq!(
            attempt.remaining_runtime_selection_blocker,
            "receipt_bound_selector_runtime_hook_registry_attachment"
        );
        assert!(attempt.strict_capture_artifact_pair_validated);
        assert!(attempt.explicit_candidate_execution_gate_requested);
        assert!(!attempt.runtime_hook_registry_attachment_present);
        assert!(!attempt.candidate_execution_attempt_allowed);
        assert!(!attempt.candidate_execution_enabled);
        assert!(
            attempt
                .fail_closed_conditions
                .contains(&"receipt_bound_selector_not_attached_to_runtime_hook_registry")
        );
        assert!(
            attempt
                .fail_closed_conditions
                .contains(&"apply_linear_no_bias_candidate_dispatch_not_wired")
        );
        assert!(attempt.preserves_normal_inference());
        assert!(!attempt.allocation_reduction_claim);
        assert!(!attempt.timing_improvement_claim);
        assert!(!attempt.speedup_claim);
    }

    #[test]
    fn no_bias_runtime_attempt_prereqs_ready_still_keeps_runtime_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let capture = slm_cpu_223_ready_strict_artifact_capture_boundary(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
        );
        let pair =
            DenseLinearNoBiasStrictCaptureArtifactPairBoundary::from_strict_artifact_capture_boundary(
                &capture,
                DenseLinearNoBiasStrictCaptureArtifactPairInputs {
                    candidate_off_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen25-candidate-off-strict-capture.json",
                    ),
                    candidate_on_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen25-candidate-on-strict-capture.json",
                    ),
                    candidate_off_strict_capture_artifact_present: true,
                    candidate_on_strict_capture_artifact_present: true,
                    candidate_off_capture_command_recorded: true,
                    candidate_on_capture_command_recorded: true,
                    candidate_off_capture_binds_gate_identity: true,
                    candidate_on_capture_binds_gate_identity: true,
                    candidate_off_capture_binds_descriptor_identity: true,
                    candidate_on_capture_binds_descriptor_identity: true,
                    candidate_off_capture_binds_owner_callsite_identity: true,
                    candidate_on_capture_binds_owner_callsite_identity: true,
                    candidate_off_on_capture_same_callsite_identity: true,
                    candidate_off_on_capture_same_prompt_digest: true,
                    candidate_off_on_capture_same_generated_digest: true,
                    candidate_off_on_capture_same_decoded_text_digest: true,
                    candidate_off_on_capture_same_model_backend_identity: true,
                    capture_prerequisite_blocker_recorded: false,
                    default_runtime_path_preserved: true,
                },
            );

        let attempt = DenseLinearNoBiasRuntimeAttemptBoundary::from_strict_capture_artifact_pair(
            &pair,
            DenseLinearNoBiasRuntimeAttemptInputs {
                explicit_candidate_execution_gate_requested: true,
                runtime_hook_registry_attachment_present: true,
                runtime_hook_descriptor_binds_selector_identity: true,
                runtime_hook_descriptor_binds_strict_capture_pair: true,
                apply_linear_dispatch_wired_to_no_bias_candidate: true,
                feed_forward_down_proj_scope_preserved: true,
                default_runtime_path_preserved: true,
            },
        );

        assert_eq!(attempt.decision, "candidate_execution_attempt_prereqs_ready_runtime_disabled");
        assert_eq!(
            attempt.remaining_runtime_selection_blocker,
            "fresh_candidate_off_on_execution_receipts"
        );
        assert!(attempt.candidate_execution_attempt_allowed);
        assert!(!attempt.candidate_execution_enabled);
        assert!(!attempt.normal_inference_runtime_selection_enabled);
        assert!(attempt.fail_closed_conditions.is_empty());
        assert!(attempt.preserves_normal_inference());
        assert!(!attempt.allocation_reduction_claim);
        assert!(!attempt.timing_improvement_claim);
        assert!(!attempt.speedup_claim);
    }

    #[test]
    fn no_bias_runtime_hook_attachment_binds_selector_identity_runtime_disabled() {
        let capture = slm_cpu_223_ready_strict_artifact_capture_boundary(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let pair =
            DenseLinearNoBiasStrictCaptureArtifactPairBoundary::from_strict_artifact_capture_boundary(
                &capture,
                DenseLinearNoBiasStrictCaptureArtifactPairInputs {
                    candidate_off_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen3-candidate-off-strict-capture.json",
                    ),
                    candidate_on_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen3-candidate-on-strict-capture.json",
                    ),
                    candidate_off_strict_capture_artifact_present: true,
                    candidate_on_strict_capture_artifact_present: true,
                    candidate_off_capture_command_recorded: true,
                    candidate_on_capture_command_recorded: true,
                    candidate_off_capture_binds_gate_identity: true,
                    candidate_on_capture_binds_gate_identity: true,
                    candidate_off_capture_binds_descriptor_identity: true,
                    candidate_on_capture_binds_descriptor_identity: true,
                    candidate_off_capture_binds_owner_callsite_identity: true,
                    candidate_on_capture_binds_owner_callsite_identity: true,
                    candidate_off_on_capture_same_callsite_identity: true,
                    candidate_off_on_capture_same_prompt_digest: true,
                    candidate_off_on_capture_same_generated_digest: true,
                    candidate_off_on_capture_same_decoded_text_digest: true,
                    candidate_off_on_capture_same_model_backend_identity: true,
                    capture_prerequisite_blocker_recorded: false,
                    default_runtime_path_preserved: true,
                },
            );
        let attempt = DenseLinearNoBiasRuntimeAttemptBoundary::from_strict_capture_artifact_pair(
            &pair,
            DenseLinearNoBiasRuntimeAttemptInputs {
                explicit_candidate_execution_gate_requested: true,
                runtime_hook_registry_attachment_present: false,
                runtime_hook_descriptor_binds_selector_identity: false,
                runtime_hook_descriptor_binds_strict_capture_pair: false,
                apply_linear_dispatch_wired_to_no_bias_candidate: false,
                feed_forward_down_proj_scope_preserved: true,
                default_runtime_path_preserved: true,
            },
        );
        let mut gate = slm_cpu_211_test_gate(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        gate.runtime_gate_requested_enabled = true;
        let selector = DenseLinearNoBiasReceiptBoundSelectorDescriptor::from_before_after_gate(
            &gate,
            "qwen3",
            "gguf_metadata",
            true,
            "slm-cpu-228:qwen3:candidate-off-on-strict-capture",
            true,
        );
        let hook = DenseLinearRuntimeHookDescriptor {
            tensor_name: selector.tensor_name.clone(),
            role: "FeedForwardDown".to_string(),
            sidecar_payload_sha256: None,
            packed_q8_payload: None,
            payload_order_matches_runtime_shape: false,
            source_order_q8_matvec_candidate: false,
            source_order_input_dim: None,
            source_order_output_dim: None,
            runtime_compute_enabled: false,
            receipt_bound_no_bias_selector: Some(selector),
        };
        let registry = DenseLinearRuntimeHookRegistry::from([(attempt.tensor_name.clone(), hook)]);

        let attachment =
            DenseLinearNoBiasRuntimeHookAttachmentBoundary::from_runtime_attempt_and_registry(
                &attempt, &registry,
            );

        assert_eq!(attachment.decision, "runtime_hook_attachment_ready_runtime_disabled");
        assert_eq!(
            attachment.remaining_runtime_selection_blocker,
            "fresh_candidate_off_on_execution_receipts"
        );
        assert!(attachment.strict_capture_artifact_pair_validated);
        assert!(attachment.explicit_candidate_execution_gate_requested);
        assert!(attachment.runtime_hook_registry_attachment_present);
        assert!(attachment.runtime_hook_descriptor_binds_selector_identity);
        assert!(attachment.runtime_hook_descriptor_binds_strict_capture_pair);
        assert!(attachment.registry_key_matches_tensor_name);
        assert!(attachment.descriptor_ready_for_apply_linear_callsite);
        assert!(attachment.feed_forward_down_proj_scope_preserved);
        assert!(attachment.default_runtime_path_preserved);
        assert!(!attachment.candidate_execution_attempt_allowed);
        assert!(!attachment.candidate_execution_enabled);
        assert!(!attachment.normal_inference_runtime_selection_enabled);
        assert!(attachment.fail_closed_conditions.is_empty());
        assert!(attachment.preserves_normal_inference());
        assert!(!attachment.allocation_reduction_claim);
        assert!(!attachment.timing_improvement_claim);
        assert!(!attachment.speedup_claim);
    }

    #[test]
    fn no_bias_runtime_hook_attachment_fails_closed_without_selector_descriptor() {
        let capture = slm_cpu_223_ready_strict_artifact_capture_boundary(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
        );
        let pair =
            DenseLinearNoBiasStrictCaptureArtifactPairBoundary::from_strict_artifact_capture_boundary(
                &capture,
                DenseLinearNoBiasStrictCaptureArtifactPairInputs {
                    candidate_off_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen3-candidate-off-strict-capture.json",
                    ),
                    candidate_on_strict_capture_artifact_path: Some(
                        "ci/slm-cpu/qwen3-candidate-on-strict-capture.json",
                    ),
                    candidate_off_strict_capture_artifact_present: true,
                    candidate_on_strict_capture_artifact_present: true,
                    candidate_off_capture_command_recorded: true,
                    candidate_on_capture_command_recorded: true,
                    candidate_off_capture_binds_gate_identity: true,
                    candidate_on_capture_binds_gate_identity: true,
                    candidate_off_capture_binds_descriptor_identity: true,
                    candidate_on_capture_binds_descriptor_identity: true,
                    candidate_off_capture_binds_owner_callsite_identity: true,
                    candidate_on_capture_binds_owner_callsite_identity: true,
                    candidate_off_on_capture_same_callsite_identity: true,
                    candidate_off_on_capture_same_prompt_digest: true,
                    candidate_off_on_capture_same_generated_digest: true,
                    candidate_off_on_capture_same_decoded_text_digest: true,
                    candidate_off_on_capture_same_model_backend_identity: true,
                    capture_prerequisite_blocker_recorded: false,
                    default_runtime_path_preserved: true,
                },
            );
        let attempt = DenseLinearNoBiasRuntimeAttemptBoundary::from_strict_capture_artifact_pair(
            &pair,
            DenseLinearNoBiasRuntimeAttemptInputs {
                explicit_candidate_execution_gate_requested: true,
                runtime_hook_registry_attachment_present: false,
                runtime_hook_descriptor_binds_selector_identity: false,
                runtime_hook_descriptor_binds_strict_capture_pair: false,
                apply_linear_dispatch_wired_to_no_bias_candidate: false,
                feed_forward_down_proj_scope_preserved: true,
                default_runtime_path_preserved: true,
            },
        );
        let hook = DenseLinearRuntimeHookDescriptor {
            tensor_name: attempt.tensor_name.clone(),
            role: "FeedForwardDown".to_string(),
            sidecar_payload_sha256: None,
            packed_q8_payload: None,
            payload_order_matches_runtime_shape: false,
            source_order_q8_matvec_candidate: false,
            source_order_input_dim: None,
            source_order_output_dim: None,
            runtime_compute_enabled: false,
            receipt_bound_no_bias_selector: None,
        };
        let registry = DenseLinearRuntimeHookRegistry::from([(attempt.tensor_name.clone(), hook)]);

        let attachment =
            DenseLinearNoBiasRuntimeHookAttachmentBoundary::from_runtime_attempt_and_registry(
                &attempt, &registry,
            );

        assert_eq!(attachment.decision, "runtime_hook_attachment_blocked_fail_closed");
        assert_eq!(
            attachment.remaining_runtime_selection_blocker,
            "runtime_hook_descriptor_selector_identity"
        );
        assert!(attachment.runtime_hook_registry_attachment_present);
        assert!(!attachment.runtime_hook_descriptor_binds_selector_identity);
        assert!(!attachment.runtime_hook_descriptor_binds_strict_capture_pair);
        assert!(!attachment.descriptor_ready_for_apply_linear_callsite);
        assert!(!attachment.candidate_execution_attempt_allowed);
        assert!(!attachment.candidate_execution_enabled);
        assert!(
            attachment
                .fail_closed_conditions
                .contains(&"runtime_hook_descriptor_selector_identity_missing")
        );
        assert!(
            attachment
                .fail_closed_conditions
                .contains(&"runtime_hook_descriptor_strict_capture_pair_identity_missing")
        );
        assert!(attachment.preserves_normal_inference());
        assert!(!attachment.allocation_reduction_claim);
        assert!(!attachment.timing_improvement_claim);
        assert!(!attachment.speedup_claim);
    }

    #[test]
    fn no_bias_candidate_execution_receipt_gate_blocks_without_fresh_execution_pair() {
        let attachment = slm_cpu_230_ready_runtime_hook_attachment(
            SLM_CPU_195_QWEN3_Q8_MODEL_SHA256,
            "qwen3",
            "qwen3_feed_forward_down_proj_no_bias_candidate",
            "slm-cpu-228:qwen3:candidate-off-on-strict-capture",
        );

        let receipt_gate =
            DenseLinearNoBiasCandidateExecutionReceiptGate::from_runtime_hook_attachment(
                &attachment,
                DenseLinearNoBiasCandidateExecutionReceiptInputs {
                    candidate_off_execution_receipt_present: false,
                    candidate_on_execution_receipt_present: false,
                    candidate_off_execution_binds_registry_attachment: false,
                    candidate_on_execution_binds_registry_attachment: false,
                    candidate_off_on_same_callsite_identity: false,
                    candidate_off_on_same_prompt_digest: false,
                    candidate_off_on_same_generated_digest: false,
                    candidate_off_on_same_decoded_text_digest: false,
                    candidate_off_on_same_model_backend_identity: false,
                    prompt_ids_preserved: false,
                    generated_ids_preserved: false,
                    decoded_text_preserved: false,
                    execution_receipt_blocker_recorded: true,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(receipt_gate.decision, "candidate_execution_receipt_pair_blocked_fail_closed");
        assert_eq!(
            receipt_gate.reason,
            "runtime_hook_attachment_ready_but_fresh_execution_receipts_are_missing"
        );
        assert_eq!(
            receipt_gate.remaining_runtime_selection_blocker,
            "candidate_on_execution_receipt"
        );
        assert!(receipt_gate.runtime_hook_attachment_ready);
        assert!(receipt_gate.explicit_candidate_execution_gate_requested);
        assert!(receipt_gate.runtime_hook_registry_attachment_present);
        assert!(receipt_gate.runtime_hook_descriptor_binds_selector_identity);
        assert!(receipt_gate.runtime_hook_descriptor_binds_strict_capture_pair);
        assert!(receipt_gate.registry_key_matches_tensor_name);
        assert!(receipt_gate.descriptor_ready_for_apply_linear_callsite);
        assert!(!receipt_gate.candidate_off_execution_receipt_present);
        assert!(!receipt_gate.candidate_on_execution_receipt_present);
        assert!(!receipt_gate.candidate_execution_receipt_pair_ready);
        assert!(!receipt_gate.candidate_execution_enabled_by_default);
        assert!(!receipt_gate.normal_inference_runtime_selection_enabled);
        assert!(
            receipt_gate.fail_closed_conditions.contains(&"candidate_on_execution_receipt_missing")
        );
        assert!(
            receipt_gate
                .fail_closed_conditions
                .contains(&"candidate_execution_generated_id_text_preservation")
                || receipt_gate.fail_closed_conditions.contains(&"generated_ids_not_preserved")
        );
        assert!(receipt_gate.preserves_normal_inference());
        assert!(!receipt_gate.allocation_reduction_claim);
        assert!(!receipt_gate.timing_improvement_claim);
        assert!(!receipt_gate.speedup_claim);
    }

    #[test]
    fn no_bias_candidate_execution_receipt_gate_ready_still_default_disabled() {
        let qwen25_sha = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
        let attachment = slm_cpu_230_ready_runtime_hook_attachment(
            qwen25_sha,
            "qwen2",
            "qwen25_feed_forward_down_proj_no_bias_candidate",
            "slm-cpu-228:qwen25:candidate-off-on-strict-capture",
        );

        let receipt_gate =
            DenseLinearNoBiasCandidateExecutionReceiptGate::from_runtime_hook_attachment(
                &attachment,
                DenseLinearNoBiasCandidateExecutionReceiptInputs {
                    candidate_off_execution_receipt_present: true,
                    candidate_on_execution_receipt_present: true,
                    candidate_off_execution_binds_registry_attachment: true,
                    candidate_on_execution_binds_registry_attachment: true,
                    candidate_off_on_same_callsite_identity: true,
                    candidate_off_on_same_prompt_digest: true,
                    candidate_off_on_same_generated_digest: true,
                    candidate_off_on_same_decoded_text_digest: true,
                    candidate_off_on_same_model_backend_identity: true,
                    prompt_ids_preserved: true,
                    generated_ids_preserved: true,
                    decoded_text_preserved: true,
                    execution_receipt_blocker_recorded: false,
                    default_runtime_path_preserved: true,
                },
            );

        assert_eq!(
            receipt_gate.decision,
            "candidate_execution_receipt_pair_ready_default_disabled"
        );
        assert_eq!(
            receipt_gate.reason,
            "candidate_off_on_execution_receipts_preserve_registry_bound_identity"
        );
        assert!(receipt_gate.candidate_execution_receipt_pair_ready);
        assert!(receipt_gate.candidate_off_execution_receipt_present);
        assert!(receipt_gate.candidate_on_execution_receipt_present);
        assert!(receipt_gate.candidate_off_execution_binds_registry_attachment);
        assert!(receipt_gate.candidate_on_execution_binds_registry_attachment);
        assert!(receipt_gate.prompt_ids_preserved);
        assert!(receipt_gate.generated_ids_preserved);
        assert!(receipt_gate.decoded_text_preserved);
        assert!(!receipt_gate.candidate_execution_enabled_by_default);
        assert!(!receipt_gate.normal_inference_runtime_selection_enabled);
        assert!(receipt_gate.fail_closed_conditions.is_empty());
        assert!(receipt_gate.preserves_normal_inference());
        assert!(!receipt_gate.allocation_reduction_claim);
        assert!(!receipt_gate.timing_improvement_claim);
        assert!(!receipt_gate.speedup_claim);
    }
}
