//! QK256 linear dispatch for BitNet transformer layers.
//!
//! The public `forward_qk256` entry point is used by the transformer whenever a
//! `.qk256_qs` raw tensor is present. It records coverage counters for the
//! BitNet linear path and, when the selected backend is CUDA, attempts the CUDA
//! QK256 kernel before falling back according to strict-mode policy.

use bitnet_common::{BitNetError, Result};
#[cfg(feature = "opencl")]
use bitnet_kernels::a770_opencl_runtime::{
    A770OpenClQk256ScaledGemv, A770OpenClQk256ScaledGemvDebug, run_a770_qk256_i8s_scaled_gemv,
    run_a770_qk256_i8s_scaled_gemv_debug,
};
#[cfg(feature = "cuda")]
use bitnet_kernels::cuda::{
    CUDA_QK256_GEMV_KERNEL_ID, CudaBitnetContext, CudaBitnetLinearBackend, PackedQk256Weights,
};
use bitnet_qk256_layout_core::{
    Qk256InputShape, Qk256Layout, parse_input_shape, parse_qk256_layout, validate_input_cols,
};
use bitnet_quantization::i2s_qk256::quantize_row_i8_s_activation;
use candle_core::Tensor;
#[cfg(feature = "cuda")]
use std::cell::RefCell;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

mod cpu_hot_path;

const NOT_CLAIMED_OPENCL_QK256: &[&str] = &[
    "a770_qk256_opencl_claim_grade_execution",
    "a770_qk256_opencl_performance",
    "activation_quantization_residency",
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
];

static BITNET_LINEAR_TOTAL: AtomicU64 = AtomicU64::new(0);
static BITNET_LINEAR_ON_CUDA: AtomicU64 = AtomicU64::new(0);
static BITNET_LINEAR_ON_A770_OPENCL: AtomicU64 = AtomicU64::new(0);
static BITNET_LINEAR_CPU_FALLBACK: AtomicU64 = AtomicU64::new(0);
static BITNET_LINEAR_UNSUPPORTED: AtomicU64 = AtomicU64::new(0);
static BITNET_LINEAR_A770_OPENCL_CPU_FALLBACK: AtomicU64 = AtomicU64::new(0);
static BITNET_LINEAR_A770_OPENCL_UNSUPPORTED: AtomicU64 = AtomicU64::new(0);
static A770_OPENCL_HOST_TO_DEVICE_BYTES: AtomicU64 = AtomicU64::new(0);
static A770_OPENCL_DEVICE_TO_HOST_BYTES: AtomicU64 = AtomicU64::new(0);
static A770_OPENCL_KERNEL_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
static A770_OPENCL_LAST_DEVICE: Mutex<Option<A770OpenClRuntimeDevice>> = Mutex::new(None);
static CUDA_QK256_HOST_TO_DEVICE_BYTES: AtomicU64 = AtomicU64::new(0);
static CUDA_QK256_DEVICE_TO_HOST_BYTES: AtomicU64 = AtomicU64::new(0);
static CUDA_QK256_HOST_TO_DEVICE_MICROS: AtomicU64 = AtomicU64::new(0);
static CUDA_QK256_HOST_TO_DEVICE_TIME_SAMPLES: AtomicU64 = AtomicU64::new(0);
static CUDA_QK256_DEVICE_TO_HOST_MICROS: AtomicU64 = AtomicU64::new(0);
static CUDA_QK256_DEVICE_TO_HOST_TIME_SAMPLES: AtomicU64 = AtomicU64::new(0);
static CUDA_QK256_KERNEL_TIME_MICROS: AtomicU64 = AtomicU64::new(0);
static CUDA_QK256_KERNEL_TIME_SAMPLES: AtomicU64 = AtomicU64::new(0);
static QK256_F32_SCALAR_GEMV_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
static QK256_F32_AVX2_GEMV_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
static QK256_I8S_SCALED_SCALAR_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
static QK256_I8S_SCALED_AVX2_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
static QK256_FLAT_BYTES_EXTRACTED_COUNT: AtomicU64 = AtomicU64::new(0);
static QK256_INPUT_ROWS_MATERIALIZED_COUNT: AtomicU64 = AtomicU64::new(0);
static QK256_OUTPUT_ROWS_ALLOCATED_COUNT: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "cuda")]
thread_local! {
    static CUDA_QK256_CONTEXT: RefCell<Option<CudaBitnetContext>> = RefCell::new(None);
}

/// Coverage counters for BitNet QK256 linear dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256DispatchCoverageCounters {
    /// Total BitNet linear dispatch points observed by this crate.
    pub bitnet_linear_layers_total: u64,
    /// Dispatch points routed through the CUDA QK256 kernel.
    pub bitnet_linear_layers_on_cuda: u64,
    /// Dispatch points routed through the selected-device A770 OpenCL QK256 kernel.
    pub bitnet_linear_layers_on_a770_opencl: u64,
    /// Dispatch points that used CPU fallback while a CUDA backend was requested.
    pub bitnet_linear_layers_cpu_fallback: u64,
    /// Unsupported operations that prevent a full CUDA inference claim.
    pub unsupported_ops: Vec<String>,
    /// Human-readable claim boundary for partial routing.
    pub execution_claim: &'static str,
}

/// Aggregate A770 OpenCL QK256 runtime counters for receipt accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256A770OpenClRuntimeStats {
    /// Host-to-device bytes copied for A770 OpenCL QK256 GEMV calls.
    pub host_to_device_bytes: u64,
    /// Device-to-host bytes copied for A770 OpenCL QK256 GEMV calls.
    pub device_to_host_bytes: u64,
    /// Number of selected-device A770 OpenCL QK256 kernel invocations.
    pub kernel_invocations: u64,
    /// Last selected OpenCL device observed by this process.
    pub last_device: Option<A770OpenClRuntimeDevice>,
}

/// Diagnostic CPU-vs-A770 replay for one QK256 projection under identical inputs.
///
/// This is not the production dispatch entry point. It avoids the global dispatch
/// counters and is intended only for focused receipts that need to compare the
/// CPU scalar oracle and the selected-device A770 OpenCL candidate on the same
/// materialized input row(s).
#[derive(Debug, Clone)]
pub struct Qk256CpuA770DispatchReplay {
    /// Number of materialized input rows replayed.
    pub input_rows: usize,
    /// Number of output rows per input row.
    pub output_rows: usize,
    /// Number of input columns.
    pub cols: usize,
    /// Packed QK256 byte stride for each output row.
    pub row_stride_bytes: usize,
    /// Inline BitNet.cpp weight scale used by the scaled I2_S x I8_S policy.
    pub inline_scale: Option<f32>,
    /// CPU scalar replay output tensor.
    pub cpu_output: Tensor,
    /// Host-side replay of the OpenCL kernel's numeric expression policy.
    pub opencl_policy_output: Tensor,
    /// A770 OpenCL replay output tensor when the replay ran successfully.
    pub a770_output: Option<Tensor>,
    /// Compact diagnostic trace for sampled host-side output expression variants.
    pub device_expression_trace: Option<Qk256DeviceExpressionTrace>,
    /// Compact selected-device debug-kernel trace for sampled intermediates.
    pub device_intermediate_trace: Option<Qk256DeviceIntermediateTrace>,
    /// Optional raw focused operands for replaying a single QK256 output row
    /// through selected-device production instrumentation.
    pub focused_operands: Option<Qk256FocusedRawOperands>,
    /// Optional complete packed projection operands for a bounded source packet.
    /// This remains diagnostic-only and does not change production dispatch.
    pub full_projection_operands: Option<Qk256FullProjectionRawOperands>,
    /// CPU replay stats.
    pub cpu: Qk256CpuDispatchReplayStats,
    /// A770 replay stats.
    pub a770: Qk256A770DispatchReplayStats,
}

/// Raw diagnostic operands for one focused QK256 replay row.
#[derive(Debug, Clone, PartialEq)]
pub struct Qk256FocusedRawOperands {
    /// Materialized input row index used for the activation vector.
    pub input_row_index: usize,
    /// Output row index within the projection matrix.
    pub output_index: usize,
    /// Number of input columns.
    pub cols: usize,
    /// Packed QK256 byte stride for each output row.
    pub row_stride_bytes: usize,
    /// Scope of the packed QK256 bytes.
    pub packed_qk256_scope: &'static str,
    /// Sum of the prequantized I8_S activation row.
    pub activation_sum: i32,
    /// Raw `f32` bits for the activation scale.
    pub activation_scale_bits: u32,
    /// Raw `f32` bits for the BitNet inline weight scale.
    pub weight_scale_bits: u32,
    /// Quantized I8_S activation row.
    pub activations_i8: Vec<i8>,
    /// Packed QK256 bytes for the selected output row.
    pub packed_qk256: Vec<u8>,
}

/// Complete raw operands for one QK256 projection and one activation row.
///
/// The packet carries every packed output row, preserving the logical matrix
/// shape used by the QK256 loader rather than any transposed GGUF metadata
/// shape. It is intentionally opt-in through the existing raw-operands replay
/// environment and remains a diagnostic source contract.
#[derive(Debug, Clone, PartialEq)]
pub struct Qk256FullProjectionRawOperands {
    /// Materialized input row index used for the activation vector.
    pub input_row_index: usize,
    /// Number of packed output rows in the projection matrix.
    pub rows: usize,
    /// Number of input columns.
    pub cols: usize,
    /// Packed QK256 byte stride for each output row.
    pub row_stride_bytes: usize,
    /// Scope of the packed QK256 bytes.
    pub packed_qk256_scope: &'static str,
    /// Sum of the prequantized I8_S activation row.
    pub activation_sum: i32,
    /// Raw `f32` bits for the activation scale.
    pub activation_scale_bits: u32,
    /// Raw `f32` bits for the BitNet inline weight scale.
    pub weight_scale_bits: u32,
    /// Quantized I8_S activation row.
    pub activations_i8: Vec<i8>,
    /// Complete packed QK256 projection rows.
    pub packed_qk256: Vec<u8>,
}

/// Compact diagnostic trace for selected QK256 output expression policy.
#[derive(Debug, Clone)]
pub struct Qk256DeviceExpressionTrace {
    /// Materialized input row index used for the samples.
    pub input_row_index: usize,
    /// Maximum number of output rows sampled.
    pub sample_limit: usize,
    /// Number of output rows sampled.
    pub sample_count: usize,
    /// Sampled output expression rows.
    pub samples: Vec<Qk256DeviceExpressionSample>,
}

/// Host-side variants for one selected QK256 output row.
#[derive(Debug, Clone, PartialEq)]
pub struct Qk256DeviceExpressionSample {
    /// Output row index within the projection matrix.
    pub output_index: usize,
    /// Integer dot product before activation-sum correction.
    pub int_dot: i32,
    /// Sum of the prequantized I8_S activation row.
    pub activation_sum: i32,
    /// `int_dot - activation_sum`.
    pub adjusted_dot: i32,
    /// I8_S activation scale.
    pub activation_scale: f32,
    /// Raw `f32` bits for the activation scale.
    pub activation_scale_bits: u32,
    /// BitNet inline weight scale.
    pub weight_scale: f32,
    /// Raw `f32` bits for the weight scale.
    pub weight_scale_bits: u32,
    /// Host policy expression: `((adjusted as f32) / activation_scale) * weight_scale`.
    pub div_then_mul: f32,
    /// Reassociated expression: `((adjusted as f32) * weight_scale) / activation_scale`.
    pub mul_then_div: f32,
    /// Reassociated expression: `(adjusted as f32) * (weight_scale / activation_scale)`.
    pub reciprocal_then_mul: f32,
    /// f64 diagnostic expression rounded back to f32.
    pub f64_div_then_mul_cast: f32,
}

/// Compact diagnostic trace from a bounded selected-device debug kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct Qk256DeviceIntermediateTrace {
    /// True when the crate was built with the OpenCL debug dependency.
    pub compiled_opencl: bool,
    /// True when the debug kernel was attempted.
    pub attempted: bool,
    /// True when sampled intermediates were captured successfully.
    pub success: bool,
    /// Error string when the debug capture failed.
    pub error: Option<String>,
    /// Materialized input row index used for the samples.
    pub input_row_index: usize,
    /// Maximum number of output rows requested.
    pub sample_limit: usize,
    /// Number of output rows sampled.
    pub sample_count: usize,
    /// OpenCL platform index selected for execution.
    pub platform_index: Option<usize>,
    /// OpenCL device index selected for execution.
    pub device_index: Option<usize>,
    /// OpenCL platform name.
    pub platform_name: Option<String>,
    /// Selected OpenCL device name.
    pub runtime_device: Option<String>,
    /// Selected OpenCL device vendor.
    pub vendor: Option<String>,
    /// Selected OpenCL driver version.
    pub driver_version: Option<String>,
    /// Host-to-device bytes uploaded by the debug capture.
    pub host_to_device_bytes: usize,
    /// Device-to-host bytes read by the debug capture.
    pub device_to_host_bytes: usize,
    /// Number of debug-kernel invocations.
    pub kernel_invocations: usize,
    /// Sampled device-side intermediate rows.
    pub samples: Vec<Qk256DeviceIntermediateSample>,
}

/// Device-side intermediate values for one selected QK256 output row.
#[derive(Debug, Clone, PartialEq)]
pub struct Qk256DeviceIntermediateSample {
    /// Output row index within the projection matrix.
    pub output_index: usize,
    /// Integer dot product before activation-sum correction.
    pub int_dot: i32,
    /// Sum of the prequantized I8_S activation row as seen by the device.
    pub activation_sum: i32,
    /// `int_dot - activation_sum` as seen by the device.
    pub adjusted_dot: i32,
    /// Raw `f32` bits for the activation scale as seen by the device.
    pub activation_scale_bits: u32,
    /// Raw `f32` bits for the weight scale as seen by the device.
    pub weight_scale_bits: u32,
    /// Raw `f32` bits for `(float)adjusted_dot`.
    pub adjusted_f32_bits: u32,
    /// Raw `f32` bits for the debug kernel output expression.
    pub output_bits: u32,
    /// Debug kernel output expression value.
    pub output: f32,
    /// Raw `f32` bits for device-side `(adjusted_f32 / activation_scale) * weight_scale`.
    pub div_then_mul_bits: u32,
    /// Device-side `(adjusted_f32 / activation_scale) * weight_scale`.
    pub div_then_mul: f32,
    /// Raw `f32` bits for device-side `(adjusted_f32 * weight_scale) / activation_scale`.
    pub mul_then_div_bits: u32,
    /// Device-side `(adjusted_f32 * weight_scale) / activation_scale`.
    pub mul_then_div: f32,
    /// Raw `f32` bits for device-side `adjusted_f32 * (weight_scale / activation_scale)`.
    pub reciprocal_then_mul_bits: u32,
    /// Device-side `adjusted_f32 * (weight_scale / activation_scale)`.
    pub reciprocal_then_mul: f32,
    /// Raw `f32` bits for volatile device-side div-then-mul replay.
    pub volatile_div_then_mul_bits: u32,
    /// Volatile device-side div-then-mul replay.
    pub volatile_div_then_mul: f32,
}

/// Diagnostic CPU replay stats for one QK256 projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256CpuDispatchReplayStats {
    /// Number of CPU scalar GEMV invocations in the replay.
    pub scalar_invocations: u64,
    /// Diagnostic execution path.
    pub execution_path: &'static str,
}

/// Diagnostic A770 replay stats for one QK256 projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256A770DispatchReplayStats {
    /// True when this crate was compiled with the OpenCL replay dependency.
    pub compiled_opencl: bool,
    /// True when the replay attempted selected-device A770 OpenCL execution.
    pub attempted: bool,
    /// True when every replay row ran successfully on A770 OpenCL.
    pub success: bool,
    /// Host-to-device bytes copied by the diagnostic replay.
    pub host_to_device_bytes: u64,
    /// Device-to-host bytes copied by the diagnostic replay.
    pub device_to_host_bytes: u64,
    /// OpenCL kernel invocations in the diagnostic replay.
    pub kernel_invocations: u64,
    /// Last selected OpenCL device observed by the diagnostic replay.
    pub last_device: Option<A770OpenClRuntimeDevice>,
    /// Replay error when A770 execution was unavailable or failed.
    pub error: Option<String>,
    /// Diagnostic execution path.
    pub execution_path: &'static str,
}

/// Selected OpenCL device identity observed by A770 QK256 dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct A770OpenClRuntimeDevice {
    /// OpenCL platform index selected for execution.
    pub platform_index: usize,
    /// OpenCL device index selected for execution.
    pub device_index: usize,
    /// OpenCL platform name.
    pub platform_name: String,
    /// Selected OpenCL device name.
    pub runtime_device: String,
    /// Selected OpenCL device vendor.
    pub vendor: String,
    /// Selected OpenCL driver version.
    pub driver_version: String,
}

/// Diagnostic counters for CPU QK256 hot-path reality audits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256CpuHotPathCounters {
    /// No-inline-scale QK256 F32 GEMV calls that selected scalar execution.
    pub qk256_f32_scalar_gemv_invocations: u64,
    /// No-inline-scale QK256 F32 GEMV calls that selected AVX2/FMA execution.
    pub qk256_f32_avx2_gemv_invocations: u64,
    /// Inline-scaled BitNet I2_S × I8_S GEMV calls that used the scalar oracle.
    pub qk256_i8s_scaled_scalar_invocations: u64,
    /// Inline-scaled BitNet I2_S × I8_S GEMV calls that used AVX2/FMA execution.
    pub qk256_i8s_scaled_avx2_invocations: u64,
    /// QK256 tensor-to-flat-byte materializations observed in this process.
    pub qk256_flat_bytes_extracted_count: u64,
    /// Input activation rows materialized into Rust Vec storage.
    pub input_rows_materialized_count: u64,
    /// Output rows allocated as Rust Vec storage.
    pub output_rows_allocated_count: u64,
    /// Requested CPU kernel label, if one was provided through BITNET_CPU_KERNEL.
    pub requested_kernel: Option<String>,
    /// Diagnostic selected hot-path label inferred from observed QK256 calls.
    pub selected_kernel: Option<String>,
    /// Diagnostic path summary for scaled/no-scale QK256 execution.
    pub qk256_execution_path: &'static str,
}

/// CUDA weight residency summary for QK256 routed inference receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256CudaWeightResidency {
    /// Number of CUDA-resident QK256 weight handles.
    pub weight_handle_count: usize,
    /// True when every observed handle was uploaded exactly once.
    pub weights_uploaded_once: bool,
    /// True if the CUDA context recorded any per-token weight upload.
    pub per_token_weight_upload: bool,
}

/// Aggregate CUDA QK256 runtime counters for receipt timing and transfer accounting.
#[derive(Debug, Clone, PartialEq)]
pub struct Qk256CudaRuntimeStats {
    /// Host-to-device activation bytes copied for QK256 CUDA GEMV calls.
    pub host_to_device_bytes: u64,
    /// Aggregate host-to-device activation copy time for QK256 CUDA GEMV calls.
    pub host_to_device_ms: Option<f64>,
    /// Number of host-to-device copy timing samples.
    pub host_to_device_time_samples: u64,
    /// Device-to-host output bytes copied for QK256 CUDA GEMV calls.
    pub device_to_host_bytes: u64,
    /// Aggregate device-to-host output copy time for QK256 CUDA GEMV calls.
    pub device_to_host_ms: Option<f64>,
    /// Number of device-to-host copy timing samples.
    pub device_to_host_time_samples: u64,
    /// Aggregate measured CUDA event time for QK256 kernel launches, in milliseconds.
    pub kernel_time_ms: Option<f64>,
    /// Number of kernel launches with a measured CUDA event time.
    pub kernel_time_samples: u64,
}

/// Describes which QK256 runtime is currently used by this dispatch crate.
///
/// OpenCL/oneAPI features currently compile route dependencies only. The GGML
/// QK256 no-scale GEMV remains the CPU implementation until a format-correct
/// OpenCL QK256 runtime is wired here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qk256DispatchStatus {
    /// True when the crate was compiled with the `opencl` feature.
    pub compiled_opencl: bool,
    /// True when the crate was compiled with the `oneapi` feature.
    pub compiled_oneapi: bool,
    /// Runtime backend used by QK256 dispatch today.
    pub runtime_backend: &'static str,
    /// True only when this crate can support an accelerator execution claim.
    pub accelerator_claimable: bool,
    /// Missing runtime component that blocks accelerator promotion.
    pub blocker: Option<&'static str>,
    /// Claim identifiers that this status must not promote.
    pub not_claims: &'static [&'static str],
}

/// Returns the non-promoting QK256 dispatch status for proof receipts.
pub fn qk256_dispatch_status() -> Qk256DispatchStatus {
    let compiled_opencl = cfg!(feature = "opencl");
    let compiled_oneapi = cfg!(feature = "oneapi");
    let (runtime_backend, blocker) = if compiled_oneapi {
        ("cpu_qk256_reference", Some("oneapi_qk256_runtime_not_wired"))
    } else if compiled_opencl {
        (
            "a770_opencl_qk256_i8s_scaled_candidate",
            Some("activation_quantization_cpu_resident_and_partial_qk256_only"),
        )
    } else {
        ("cpu_qk256_reference", Some("cpu_qk256_dispatch_only"))
    };

    Qk256DispatchStatus {
        compiled_opencl,
        compiled_oneapi,
        runtime_backend,
        accelerator_claimable: false,
        blocker,
        not_claims: NOT_CLAIMED_OPENCL_QK256,
    }
}

/// Snapshot the current QK256 dispatch coverage counters.
pub fn qk256_dispatch_coverage() -> Qk256DispatchCoverageCounters {
    let cpu_fallback = BITNET_LINEAR_CPU_FALLBACK.load(Ordering::Relaxed);
    let unsupported = BITNET_LINEAR_UNSUPPORTED.load(Ordering::Relaxed);
    let on_cuda = BITNET_LINEAR_ON_CUDA.load(Ordering::Relaxed);
    let on_a770_opencl = BITNET_LINEAR_ON_A770_OPENCL.load(Ordering::Relaxed);
    let a770_cpu_fallback = BITNET_LINEAR_A770_OPENCL_CPU_FALLBACK.load(Ordering::Relaxed);
    let a770_unsupported = BITNET_LINEAR_A770_OPENCL_UNSUPPORTED.load(Ordering::Relaxed);
    let mut unsupported_ops = Vec::new();
    if cpu_fallback > 0 {
        unsupported_ops.push("qk256_cpu_fallback".to_string());
    }
    if a770_cpu_fallback > 0 || a770_unsupported > 0 {
        unsupported_ops.push("qk256_a770_opencl_not_routed".to_string());
    }
    if unsupported > 0 {
        unsupported_ops.push("qk256_strict_cuda_unsupported".to_string());
    }

    Qk256DispatchCoverageCounters {
        bitnet_linear_layers_total: BITNET_LINEAR_TOTAL.load(Ordering::Relaxed),
        bitnet_linear_layers_on_cuda: on_cuda,
        bitnet_linear_layers_on_a770_opencl: on_a770_opencl,
        bitnet_linear_layers_cpu_fallback: cpu_fallback,
        unsupported_ops,
        execution_claim: if on_cuda > 0 {
            "cuda_inference_contribution"
        } else if on_a770_opencl > 0 {
            "a770_opencl_qk256_contribution"
        } else if a770_cpu_fallback > 0 || a770_unsupported > 0 || a770_opencl_backend_requested() {
            "a770_opencl_not_routed"
        } else if cuda_bitnet_backend_requested() {
            "cuda_bitnet_not_routed"
        } else {
            "cpu_reference"
        },
    }
}

/// Snapshot A770 OpenCL QK256 timing and transfer counters for the current process.
pub fn qk256_a770_opencl_runtime_stats() -> Qk256A770OpenClRuntimeStats {
    Qk256A770OpenClRuntimeStats {
        host_to_device_bytes: A770_OPENCL_HOST_TO_DEVICE_BYTES.load(Ordering::Relaxed),
        device_to_host_bytes: A770_OPENCL_DEVICE_TO_HOST_BYTES.load(Ordering::Relaxed),
        kernel_invocations: A770_OPENCL_KERNEL_INVOCATIONS.load(Ordering::Relaxed),
        last_device: A770_OPENCL_LAST_DEVICE.lock().ok().and_then(|device| device.clone()),
    }
}

/// Snapshot CPU QK256 hot-path diagnostic counters for the current process.
pub fn qk256_cpu_hot_path_counters() -> Qk256CpuHotPathCounters {
    let f32_scalar = QK256_F32_SCALAR_GEMV_INVOCATIONS.load(Ordering::Relaxed);
    let f32_avx2 = QK256_F32_AVX2_GEMV_INVOCATIONS.load(Ordering::Relaxed);
    let scaled_scalar = QK256_I8S_SCALED_SCALAR_INVOCATIONS.load(Ordering::Relaxed);
    let scaled_avx2 = QK256_I8S_SCALED_AVX2_INVOCATIONS.load(Ordering::Relaxed);

    Qk256CpuHotPathCounters {
        qk256_f32_scalar_gemv_invocations: f32_scalar,
        qk256_f32_avx2_gemv_invocations: f32_avx2,
        qk256_i8s_scaled_scalar_invocations: scaled_scalar,
        qk256_i8s_scaled_avx2_invocations: scaled_avx2,
        qk256_flat_bytes_extracted_count: QK256_FLAT_BYTES_EXTRACTED_COUNT.load(Ordering::Relaxed),
        input_rows_materialized_count: QK256_INPUT_ROWS_MATERIALIZED_COUNT.load(Ordering::Relaxed),
        output_rows_allocated_count: QK256_OUTPUT_ROWS_ALLOCATED_COUNT.load(Ordering::Relaxed),
        requested_kernel: cpu_hot_path::requested_cpu_kernel_label(),
        selected_kernel: cpu_hot_path::selected_cpu_hot_path_label(
            f32_scalar,
            f32_avx2,
            scaled_scalar,
            scaled_avx2,
        ),
        qk256_execution_path: cpu_hot_path::qk256_execution_path_label(
            f32_scalar,
            f32_avx2,
            scaled_scalar,
            scaled_avx2,
        ),
    }
}

/// Reset dispatch coverage counters.
///
/// This is public so CLI and integration tests can scope receipt counters to a
/// single run without relying on process lifetime.
pub fn reset_qk256_dispatch_coverage() {
    BITNET_LINEAR_TOTAL.store(0, Ordering::Relaxed);
    BITNET_LINEAR_ON_CUDA.store(0, Ordering::Relaxed);
    BITNET_LINEAR_ON_A770_OPENCL.store(0, Ordering::Relaxed);
    BITNET_LINEAR_CPU_FALLBACK.store(0, Ordering::Relaxed);
    BITNET_LINEAR_UNSUPPORTED.store(0, Ordering::Relaxed);
    BITNET_LINEAR_A770_OPENCL_CPU_FALLBACK.store(0, Ordering::Relaxed);
    BITNET_LINEAR_A770_OPENCL_UNSUPPORTED.store(0, Ordering::Relaxed);
    A770_OPENCL_HOST_TO_DEVICE_BYTES.store(0, Ordering::Relaxed);
    A770_OPENCL_DEVICE_TO_HOST_BYTES.store(0, Ordering::Relaxed);
    A770_OPENCL_KERNEL_INVOCATIONS.store(0, Ordering::Relaxed);
    if let Ok(mut device) = A770_OPENCL_LAST_DEVICE.lock() {
        *device = None;
    }
    CUDA_QK256_HOST_TO_DEVICE_BYTES.store(0, Ordering::Relaxed);
    CUDA_QK256_DEVICE_TO_HOST_BYTES.store(0, Ordering::Relaxed);
    CUDA_QK256_HOST_TO_DEVICE_MICROS.store(0, Ordering::Relaxed);
    CUDA_QK256_HOST_TO_DEVICE_TIME_SAMPLES.store(0, Ordering::Relaxed);
    CUDA_QK256_DEVICE_TO_HOST_MICROS.store(0, Ordering::Relaxed);
    CUDA_QK256_DEVICE_TO_HOST_TIME_SAMPLES.store(0, Ordering::Relaxed);
    CUDA_QK256_KERNEL_TIME_MICROS.store(0, Ordering::Relaxed);
    CUDA_QK256_KERNEL_TIME_SAMPLES.store(0, Ordering::Relaxed);
    QK256_F32_SCALAR_GEMV_INVOCATIONS.store(0, Ordering::Relaxed);
    QK256_F32_AVX2_GEMV_INVOCATIONS.store(0, Ordering::Relaxed);
    QK256_I8S_SCALED_SCALAR_INVOCATIONS.store(0, Ordering::Relaxed);
    QK256_I8S_SCALED_AVX2_INVOCATIONS.store(0, Ordering::Relaxed);
    QK256_FLAT_BYTES_EXTRACTED_COUNT.store(0, Ordering::Relaxed);
    QK256_INPUT_ROWS_MATERIALIZED_COUNT.store(0, Ordering::Relaxed);
    QK256_OUTPUT_ROWS_ALLOCATED_COUNT.store(0, Ordering::Relaxed);
    reset_cuda_qk256_context();
}

/// Snapshot CUDA QK256 weight residency for the current thread-local proof run.
pub fn qk256_cuda_weight_residency() -> Option<Qk256CudaWeightResidency> {
    cuda_qk256_weight_residency()
}

/// Snapshot CUDA QK256 timing and transfer counters for the current process.
pub fn qk256_cuda_runtime_stats() -> Qk256CudaRuntimeStats {
    let samples = CUDA_QK256_KERNEL_TIME_SAMPLES.load(Ordering::Relaxed);
    let kernel_time_micros = CUDA_QK256_KERNEL_TIME_MICROS.load(Ordering::Relaxed);
    let host_to_device_samples = CUDA_QK256_HOST_TO_DEVICE_TIME_SAMPLES.load(Ordering::Relaxed);
    let host_to_device_micros = CUDA_QK256_HOST_TO_DEVICE_MICROS.load(Ordering::Relaxed);
    let device_to_host_samples = CUDA_QK256_DEVICE_TO_HOST_TIME_SAMPLES.load(Ordering::Relaxed);
    let device_to_host_micros = CUDA_QK256_DEVICE_TO_HOST_MICROS.load(Ordering::Relaxed);
    Qk256CudaRuntimeStats {
        host_to_device_bytes: CUDA_QK256_HOST_TO_DEVICE_BYTES.load(Ordering::Relaxed),
        host_to_device_ms: (host_to_device_samples > 0)
            .then(|| host_to_device_micros as f64 / 1000.0),
        host_to_device_time_samples: host_to_device_samples,
        device_to_host_bytes: CUDA_QK256_DEVICE_TO_HOST_BYTES.load(Ordering::Relaxed),
        device_to_host_ms: (device_to_host_samples > 0)
            .then(|| device_to_host_micros as f64 / 1000.0),
        device_to_host_time_samples: device_to_host_samples,
        kernel_time_ms: (samples > 0).then(|| kernel_time_micros as f64 / 1000.0),
        kernel_time_samples: samples,
    }
}

/// Record a BitNet linear CPU fallback outside the QK256 raw-tensor path.
pub fn record_bitnet_linear_cpu_fallback() {
    BITNET_LINEAR_TOTAL.fetch_add(1, Ordering::Relaxed);
    if cuda_bitnet_backend_requested() {
        BITNET_LINEAR_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
    if a770_opencl_backend_requested() {
        BITNET_LINEAR_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
        BITNET_LINEAR_A770_OPENCL_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
}

/// Record a BitNet linear dispatch point that strict CUDA cannot support.
pub fn record_bitnet_linear_unsupported() {
    BITNET_LINEAR_TOTAL.fetch_add(1, Ordering::Relaxed);
    if a770_opencl_backend_requested() {
        BITNET_LINEAR_A770_OPENCL_UNSUPPORTED.fetch_add(1, Ordering::Relaxed);
    } else {
        BITNET_LINEAR_UNSUPPORTED.fetch_add(1, Ordering::Relaxed);
    }
}

/// True when the run selected or requested the RTX 5070 Ti CUDA BitNet lane.
pub fn cuda_bitnet_backend_requested() -> bool {
    backend_env_matches("BITNET_SELECTED_BACKEND")
        || backend_env_matches("BITNET_REQUESTED_BACKEND")
        || backend_env_matches("BITNET_BACKEND")
}

/// True when strict mode forbids CPU fallback for the CUDA BitNet lane.
pub fn strict_cuda_bitnet_backend_requested() -> bool {
    (cuda_bitnet_backend_requested() && env_truthy("BITNET_STRICT_MODE"))
        || env_truthy("BITNET_STRICT_CUDA_BACKEND")
}

/// True when the run selected or requested the Intel Arc A770 OpenCL BitNet lane.
pub fn a770_opencl_backend_requested() -> bool {
    a770_opencl_backend_env_matches("BITNET_SELECTED_BACKEND")
        || a770_opencl_backend_env_matches("BITNET_REQUESTED_BACKEND")
        || a770_opencl_backend_env_matches("BITNET_BACKEND")
}

/// True when strict mode forbids CPU fallback for the A770 OpenCL BitNet lane.
pub fn strict_a770_opencl_backend_requested() -> bool {
    (a770_opencl_backend_requested() && env_truthy("BITNET_STRICT_MODE"))
        || env_truthy("BITNET_STRICT_A770_OPENCL_BACKEND")
}

/// Runs I2_S QK256 forward pass for input tensor shapes [B, T, H] or [B, H].
pub fn forward_qk256(input: &Tensor, qk256_tensor: &Tensor, weight_name: &str) -> Result<Tensor> {
    forward_qk256_with_scale(input, qk256_tensor, weight_name, None)
}

/// Runs I2_S QK256 forward pass with an optional BitNet.cpp inline tensor scale.
pub fn forward_qk256_with_scale(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
    inline_scale: Option<f32>,
) -> Result<Tensor> {
    BITNET_LINEAR_TOTAL.fetch_add(1, Ordering::Relaxed);

    if a770_opencl_backend_requested() {
        if inline_scale.is_some() {
            #[cfg(feature = "opencl")]
            {
                match forward_qk256_a770_opencl(input, qk256_tensor, weight_name, inline_scale) {
                    Ok(output) => {
                        BITNET_LINEAR_ON_A770_OPENCL.fetch_add(1, Ordering::Relaxed);
                        return Ok(output);
                    }
                    Err(err) if strict_a770_opencl_backend_requested() => {
                        BITNET_LINEAR_A770_OPENCL_UNSUPPORTED.fetch_add(1, Ordering::Relaxed);
                        return Err(BitNetError::Validation(format!(
                            "strict A770 OpenCL BitNet linear dispatch failed for {weight_name}: {err}"
                        )));
                    }
                    Err(err) => {
                        BITNET_LINEAR_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
                        BITNET_LINEAR_A770_OPENCL_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(
                            "A770 OpenCL QK256 dispatch failed for {}; using CPU fallback: {}",
                            weight_name,
                            err
                        );
                    }
                }
            }

            #[cfg(not(feature = "opencl"))]
            {
                if strict_a770_opencl_backend_requested() {
                    BITNET_LINEAR_A770_OPENCL_UNSUPPORTED.fetch_add(1, Ordering::Relaxed);
                    return Err(BitNetError::Validation(format!(
                        "strict A770 OpenCL BitNet linear dispatch requested for {weight_name}, but bitnet-qk256-dispatch was built without the opencl feature"
                    )));
                }
                BITNET_LINEAR_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
                BITNET_LINEAR_A770_OPENCL_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            if strict_a770_opencl_backend_requested() {
                BITNET_LINEAR_A770_OPENCL_UNSUPPORTED.fetch_add(1, Ordering::Relaxed);
                return Err(BitNetError::Validation(format!(
                    "strict A770 OpenCL BitNet linear dispatch requested for {weight_name}, but the OpenCL QK256 runtime currently requires an inline BitNet scale"
                )));
            }
            BITNET_LINEAR_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
            BITNET_LINEAR_A770_OPENCL_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                "A770 OpenCL QK256 dispatch requested for {}; using CPU fallback because inline BitNet scale is absent",
                weight_name
            );
        }
    }

    if cuda_bitnet_backend_requested() {
        #[cfg(feature = "cuda")]
        {
            match forward_qk256_cuda(input, qk256_tensor, weight_name, inline_scale) {
                Ok(output) => {
                    BITNET_LINEAR_ON_CUDA.fetch_add(1, Ordering::Relaxed);
                    return Ok(output);
                }
                Err(err) if strict_cuda_bitnet_backend_requested() => {
                    return Err(BitNetError::Validation(format!(
                        "strict CUDA BitNet linear dispatch failed for {weight_name}: {err}"
                    )));
                }
                Err(err) => {
                    BITNET_LINEAR_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        "CUDA QK256 dispatch failed for {}; using CPU fallback: {}",
                        weight_name,
                        err
                    );
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            if strict_cuda_bitnet_backend_requested() {
                return Err(BitNetError::Validation(format!(
                    "strict CUDA BitNet linear dispatch requested for {weight_name}, but bitnet-qk256-dispatch was built without the cuda feature"
                )));
            }
            BITNET_LINEAR_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
        }
    }

    forward_qk256_cpu(input, qk256_tensor, weight_name, inline_scale)
}

/// Replay one scaled QK256 projection with the CPU scalar oracle and, when
/// available, the selected-device A770 OpenCL candidate.
///
/// The replay is diagnostic-only: it does not increment the global dispatch,
/// CPU hot-path, or A770 runtime counters used by production receipts.
pub fn replay_qk256_cpu_vs_a770_with_scale(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
    inline_scale: Option<f32>,
) -> Result<Qk256CpuA770DispatchReplay> {
    use bitnet_quantization::i2s_qk256::gemv_qk256_bitnet_i8s_scaled;

    let weight_scale = inline_scale.ok_or_else(|| {
        BitNetError::Validation(format!(
            "QK256 dispatch replay requires an inline BitNet scale for {weight_name}"
        ))
    })?;
    if !weight_scale.is_finite() {
        return Err(BitNetError::Validation(format!(
            "QK256 dispatch replay inline scale is not finite for {weight_name}: {weight_scale}"
        )));
    }

    let prepared = prepare_qk256_forward_untracked(input, qk256_tensor, weight_name)?;
    let output_row_count = prepared.shape.batch_size * prepared.shape.seq_len;
    let mut cpu_flat = Vec::with_capacity(output_row_count * prepared.layout.rows);
    let mut opencl_policy_flat = Vec::with_capacity(output_row_count * prepared.layout.rows);
    let mut cpu_scalar_invocations = 0u64;
    let mut device_expression_trace = None;
    let mut device_intermediate_trace = None;
    let capture_raw_operands = env_truthy("BITNET_QKV_PROJECTION_DISPATCH_REPLAY_RAW_OPERANDS");
    let raw_operand_input_row =
        env_usize("BITNET_QKV_PROJECTION_DISPATCH_REPLAY_RAW_OPERAND_INPUT_ROW").unwrap_or(0);
    let raw_operand_output_index =
        env_usize("BITNET_QKV_PROJECTION_DISPATCH_REPLAY_RAW_OPERAND_OUTPUT_INDEX").unwrap_or(0);
    let mut focused_operands = None;
    let mut full_projection_operands = None;

    for (input_row_index, input_row) in prepared.input_rows.iter().enumerate() {
        let mut output_row = vec![0.0f32; prepared.layout.rows];
        gemv_qk256_bitnet_i8s_scaled(
            &prepared.flat_bytes,
            input_row,
            &mut output_row,
            prepared.layout.rows,
            prepared.layout.cols,
            prepared.layout.row_stride_bytes,
            weight_scale,
        )
        .map_err(|err| {
            BitNetError::Validation(format!(
                "QK256 CPU dispatch replay failed for {weight_name}: {err}"
            ))
        })?;
        cpu_scalar_invocations += 1;
        cpu_flat.extend_from_slice(&output_row);

        let mut opencl_policy_row = vec![0.0f32; prepared.layout.rows];
        gemv_qk256_opencl_linear_i8s_scaled(
            &prepared.flat_bytes,
            input_row,
            &mut opencl_policy_row,
            prepared.layout.rows,
            prepared.layout.cols,
            prepared.layout.row_stride_bytes,
            weight_scale,
        )?;
        opencl_policy_flat.extend_from_slice(&opencl_policy_row);

        if device_expression_trace.is_none() {
            let (q, activation_scale, activation_sum) =
                quantize_row_i8_s_activation(input_row, prepared.layout.cols);
            if capture_raw_operands
                && focused_operands.is_none()
                && input_row_index == raw_operand_input_row
                && raw_operand_output_index < prepared.layout.rows
            {
                let row_start = raw_operand_output_index * prepared.layout.row_stride_bytes;
                let row_end = row_start + prepared.layout.row_stride_bytes;
                focused_operands = Some(Qk256FocusedRawOperands {
                    input_row_index,
                    output_index: raw_operand_output_index,
                    cols: prepared.layout.cols,
                    row_stride_bytes: prepared.layout.row_stride_bytes,
                    packed_qk256_scope: "single_output_row",
                    activation_sum,
                    activation_scale_bits: activation_scale.to_bits(),
                    weight_scale_bits: weight_scale.to_bits(),
                    activations_i8: q.clone(),
                    packed_qk256: prepared.flat_bytes[row_start..row_end].to_vec(),
                });
            }
            if capture_raw_operands
                && full_projection_operands.is_none()
                && input_row_index == raw_operand_input_row
            {
                full_projection_operands = Some(Qk256FullProjectionRawOperands {
                    input_row_index,
                    rows: prepared.layout.rows,
                    cols: prepared.layout.cols,
                    row_stride_bytes: prepared.layout.row_stride_bytes,
                    packed_qk256_scope: "full_projection_output_rows",
                    activation_sum,
                    activation_scale_bits: activation_scale.to_bits(),
                    weight_scale_bits: weight_scale.to_bits(),
                    activations_i8: q.clone(),
                    packed_qk256: prepared.flat_bytes.clone(),
                });
            }
            device_expression_trace = Some(qk256_device_expression_trace_for_row(
                &prepared.flat_bytes,
                &q,
                prepared.layout.rows,
                prepared.layout.cols,
                prepared.layout.row_stride_bytes,
                activation_sum,
                activation_scale,
                weight_scale,
                input_row_index,
                8,
            )?);
            device_intermediate_trace = Some(qk256_device_intermediate_trace_for_row(
                &prepared.flat_bytes,
                &q,
                prepared.layout.rows,
                prepared.layout.cols,
                prepared.layout.row_stride_bytes,
                activation_sum,
                activation_scale,
                weight_scale,
                input_row_index,
                8,
            ));
        }
    }

    let cpu_output = tensor_from_flat_output(cpu_flat, &prepared.shape, &prepared.layout, input)?;
    let opencl_policy_output =
        tensor_from_flat_output(opencl_policy_flat, &prepared.shape, &prepared.layout, input)?;
    let (a770_output, a770) = replay_qk256_a770_opencl_untracked(&prepared, input, weight_scale);

    Ok(Qk256CpuA770DispatchReplay {
        input_rows: output_row_count,
        output_rows: prepared.layout.rows,
        cols: prepared.layout.cols,
        row_stride_bytes: prepared.layout.row_stride_bytes,
        inline_scale,
        cpu_output,
        opencl_policy_output,
        a770_output,
        device_expression_trace,
        device_intermediate_trace,
        focused_operands,
        full_projection_operands,
        cpu: Qk256CpuDispatchReplayStats {
            scalar_invocations: cpu_scalar_invocations,
            execution_path: "cpu_qk256_i2s_i8s_scaled_scalar_replay",
        },
        a770,
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "opencl")]
fn qk256_device_intermediate_trace_for_row(
    qs_data: &[u8],
    q: &[i8],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
    activation_sum: i32,
    activation_scale: f32,
    weight_scale: f32,
    input_row_index: usize,
    sample_limit: usize,
) -> Qk256DeviceIntermediateTrace {
    match run_a770_qk256_i8s_scaled_gemv_debug(A770OpenClQk256ScaledGemvDebug {
        activations_i8: q,
        packed_qk256: qs_data,
        rows,
        cols,
        row_stride_bytes,
        activation_sum,
        activation_scale,
        weight_scale,
        sample_limit,
    }) {
        Ok(result) => Qk256DeviceIntermediateTrace {
            compiled_opencl: true,
            attempted: true,
            success: true,
            error: None,
            input_row_index,
            sample_limit,
            sample_count: result.samples.len(),
            platform_index: Some(result.platform_index),
            device_index: Some(result.device_index),
            platform_name: Some(result.platform_name),
            runtime_device: Some(result.runtime_device),
            vendor: Some(result.vendor),
            driver_version: Some(result.driver_version),
            host_to_device_bytes: result.host_to_device_bytes,
            device_to_host_bytes: result.device_to_host_bytes,
            kernel_invocations: result.kernel_invocations,
            samples: result
                .samples
                .into_iter()
                .map(|sample| Qk256DeviceIntermediateSample {
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
        },
        Err(err) => Qk256DeviceIntermediateTrace {
            compiled_opencl: true,
            attempted: true,
            success: false,
            error: Some(err.to_string()),
            input_row_index,
            sample_limit,
            sample_count: 0,
            platform_index: None,
            device_index: None,
            platform_name: None,
            runtime_device: None,
            vendor: None,
            driver_version: None,
            host_to_device_bytes: 0,
            device_to_host_bytes: 0,
            kernel_invocations: 0,
            samples: Vec::new(),
        },
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "opencl"))]
fn qk256_device_intermediate_trace_for_row(
    _qs_data: &[u8],
    _q: &[i8],
    _rows: usize,
    _cols: usize,
    _row_stride_bytes: usize,
    _activation_sum: i32,
    _activation_scale: f32,
    _weight_scale: f32,
    input_row_index: usize,
    sample_limit: usize,
) -> Qk256DeviceIntermediateTrace {
    Qk256DeviceIntermediateTrace {
        compiled_opencl: false,
        attempted: false,
        success: false,
        error: Some("bitnet-qk256-dispatch was built without the opencl feature".to_string()),
        input_row_index,
        sample_limit,
        sample_count: 0,
        platform_index: None,
        device_index: None,
        platform_name: None,
        runtime_device: None,
        vendor: None,
        driver_version: None,
        host_to_device_bytes: 0,
        device_to_host_bytes: 0,
        kernel_invocations: 0,
        samples: Vec::new(),
    }
}

fn gemv_qk256_opencl_linear_i8s_scaled(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
    weight_scale: f32,
) -> Result<()> {
    if y_out.len() != rows {
        return Err(BitNetError::Validation(format!(
            "OpenCL-policy QK256 replay y_out length {} != rows {}",
            y_out.len(),
            rows
        )));
    }
    if x.len() < cols {
        return Err(BitNetError::Validation(format!(
            "OpenCL-policy QK256 replay x length {} < cols {}",
            x.len(),
            cols
        )));
    }
    let expected_total = rows.checked_mul(row_stride_bytes).ok_or_else(|| {
        BitNetError::Validation("OpenCL-policy QK256 replay packed length overflow".to_string())
    })?;
    if qs_data.len() < expected_total {
        return Err(BitNetError::Validation(format!(
            "OpenCL-policy QK256 replay data too short: {} < {}",
            qs_data.len(),
            expected_total
        )));
    }

    let (q, activation_scale, activation_sum) = quantize_row_i8_s_activation(x, cols);
    for (row, output) in y_out.iter_mut().enumerate().take(rows) {
        let row_base = row * row_stride_bytes;
        let mut int_dot = 0i32;
        for (col, &q_value) in q.iter().enumerate().take(cols) {
            let block = col / 256;
            let offset = col - block * 256;
            let chunk = offset / 128;
            let lane = (offset - chunk * 128) / 32;
            let gp = offset & 31;
            let byte_index = row_base + block * 64 + chunk * 32 + gp;
            let packed = qs_data[byte_index];
            let code = ((packed >> (6 - lane * 2)) & 0x03) as i32;
            int_dot += code * q_value as i32;
        }
        *output = (((int_dot - activation_sum) as f32) / activation_scale) * weight_scale;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qk256_device_expression_trace_for_row(
    qs_data: &[u8],
    q: &[i8],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
    activation_sum: i32,
    activation_scale: f32,
    weight_scale: f32,
    input_row_index: usize,
    sample_limit: usize,
) -> Result<Qk256DeviceExpressionTrace> {
    let expected_total = rows.checked_mul(row_stride_bytes).ok_or_else(|| {
        BitNetError::Validation("OpenCL expression trace packed length overflow".to_string())
    })?;
    if qs_data.len() < expected_total {
        return Err(BitNetError::Validation(format!(
            "OpenCL expression trace data too short: {} < {}",
            qs_data.len(),
            expected_total
        )));
    }
    if q.len() < cols {
        return Err(BitNetError::Validation(format!(
            "OpenCL expression trace activation length {} < cols {}",
            q.len(),
            cols
        )));
    }

    let mut samples = Vec::with_capacity(sample_limit.min(rows));
    for output_index in 0..rows.min(sample_limit) {
        let int_dot =
            qk256_opencl_int_dot_for_row(qs_data, q, output_index, cols, row_stride_bytes)?;
        let adjusted_dot = int_dot - activation_sum;
        let adjusted_f32 = adjusted_dot as f32;
        let div_then_mul = (adjusted_f32 / activation_scale) * weight_scale;
        let mul_then_div = (adjusted_f32 * weight_scale) / activation_scale;
        let reciprocal_then_mul = adjusted_f32 * (weight_scale / activation_scale);
        let f64_div_then_mul_cast =
            ((adjusted_dot as f64 / activation_scale as f64) * weight_scale as f64) as f32;
        samples.push(Qk256DeviceExpressionSample {
            output_index,
            int_dot,
            activation_sum,
            adjusted_dot,
            activation_scale,
            activation_scale_bits: activation_scale.to_bits(),
            weight_scale,
            weight_scale_bits: weight_scale.to_bits(),
            div_then_mul,
            mul_then_div,
            reciprocal_then_mul,
            f64_div_then_mul_cast,
        });
    }

    Ok(Qk256DeviceExpressionTrace {
        input_row_index,
        sample_limit,
        sample_count: samples.len(),
        samples,
    })
}

fn qk256_opencl_int_dot_for_row(
    qs_data: &[u8],
    q: &[i8],
    row: usize,
    cols: usize,
    row_stride_bytes: usize,
) -> Result<i32> {
    let row_base = row * row_stride_bytes;
    let mut int_dot = 0i32;
    for (col, &q_value) in q.iter().enumerate().take(cols) {
        let block = col / 256;
        let offset = col - block * 256;
        let chunk = offset / 128;
        let lane = (offset - chunk * 128) / 32;
        let gp = offset & 31;
        let byte_index = row_base + block * 64 + chunk * 32 + gp;
        let packed = qs_data.get(byte_index).copied().ok_or_else(|| {
            BitNetError::Validation(format!(
                "OpenCL expression trace byte index {byte_index} out of range {}",
                qs_data.len()
            ))
        })?;
        let code = ((packed >> (6 - lane * 2)) & 0x03) as i32;
        int_dot += code * q_value as i32;
    }
    Ok(int_dot)
}

#[cfg(feature = "opencl")]
fn forward_qk256_a770_opencl(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
    inline_scale: Option<f32>,
) -> Result<Tensor> {
    let weight_scale = inline_scale.ok_or_else(|| {
        BitNetError::Validation(
            "A770 OpenCL QK256 dispatch requires an inline BitNet scale".to_string(),
        )
    })?;
    if !weight_scale.is_finite() {
        return Err(BitNetError::Validation(format!(
            "QK256 inline scale is not finite: {weight_scale}"
        )));
    }

    let prepared = prepare_qk256_forward(input, qk256_tensor, weight_name)?;
    let output_row_count = prepared.shape.batch_size * prepared.shape.seq_len;
    QK256_OUTPUT_ROWS_ALLOCATED_COUNT.fetch_add(output_row_count as u64, Ordering::Relaxed);
    let mut output_rows = Vec::with_capacity(output_row_count * prepared.layout.rows);

    for input_row in &prepared.input_rows {
        let (q, activation_scale, activation_sum) =
            quantize_row_i8_s_activation(input_row, prepared.layout.cols);
        let result = run_a770_qk256_i8s_scaled_gemv(A770OpenClQk256ScaledGemv {
            activations_i8: &q,
            packed_qk256: &prepared.flat_bytes,
            rows: prepared.layout.rows,
            cols: prepared.layout.cols,
            row_stride_bytes: prepared.layout.row_stride_bytes,
            activation_sum,
            activation_scale,
            weight_scale,
        })?;
        A770_OPENCL_HOST_TO_DEVICE_BYTES
            .fetch_add(result.host_to_device_bytes as u64, Ordering::Relaxed);
        A770_OPENCL_DEVICE_TO_HOST_BYTES
            .fetch_add(result.device_to_host_bytes as u64, Ordering::Relaxed);
        A770_OPENCL_KERNEL_INVOCATIONS
            .fetch_add(result.kernel_invocations as u64, Ordering::Relaxed);
        record_a770_opencl_runtime_device(
            result.platform_index,
            result.device_index,
            &result.platform_name,
            &result.runtime_device,
            &result.vendor,
            &result.driver_version,
        );
        output_rows.extend_from_slice(&result.output);
    }

    tensor_from_flat_output(output_rows, &prepared.shape, &prepared.layout, input)
}

#[cfg(feature = "opencl")]
fn replay_qk256_a770_opencl_untracked(
    prepared: &PreparedQk256Forward,
    input: &Tensor,
    weight_scale: f32,
) -> (Option<Tensor>, Qk256A770DispatchReplayStats) {
    let mut output_rows = Vec::with_capacity(prepared.input_rows.len() * prepared.layout.rows);
    let mut stats = Qk256A770DispatchReplayStats {
        compiled_opencl: true,
        attempted: true,
        success: false,
        host_to_device_bytes: 0,
        device_to_host_bytes: 0,
        kernel_invocations: 0,
        last_device: None,
        error: None,
        execution_path: "a770_opencl_qk256_i2s_i8s_scaled_replay",
    };

    for input_row in &prepared.input_rows {
        let (q, activation_scale, activation_sum) =
            quantize_row_i8_s_activation(input_row, prepared.layout.cols);
        let result = match run_a770_qk256_i8s_scaled_gemv(A770OpenClQk256ScaledGemv {
            activations_i8: &q,
            packed_qk256: &prepared.flat_bytes,
            rows: prepared.layout.rows,
            cols: prepared.layout.cols,
            row_stride_bytes: prepared.layout.row_stride_bytes,
            activation_sum,
            activation_scale,
            weight_scale,
        }) {
            Ok(result) => result,
            Err(err) => {
                stats.error = Some(err.to_string());
                return (None, stats);
            }
        };

        stats.host_to_device_bytes += result.host_to_device_bytes as u64;
        stats.device_to_host_bytes += result.device_to_host_bytes as u64;
        stats.kernel_invocations += result.kernel_invocations as u64;
        stats.last_device = Some(A770OpenClRuntimeDevice {
            platform_index: result.platform_index,
            device_index: result.device_index,
            platform_name: result.platform_name,
            runtime_device: result.runtime_device,
            vendor: result.vendor,
            driver_version: result.driver_version,
        });
        output_rows.extend_from_slice(&result.output);
    }

    let output =
        match tensor_from_flat_output(output_rows, &prepared.shape, &prepared.layout, input) {
            Ok(output) => output,
            Err(err) => {
                stats.error = Some(err.to_string());
                return (None, stats);
            }
        };
    stats.success = true;
    (Some(output), stats)
}

#[cfg(not(feature = "opencl"))]
fn replay_qk256_a770_opencl_untracked(
    _prepared: &PreparedQk256Forward,
    _input: &Tensor,
    _weight_scale: f32,
) -> (Option<Tensor>, Qk256A770DispatchReplayStats) {
    (
        None,
        Qk256A770DispatchReplayStats {
            compiled_opencl: false,
            attempted: false,
            success: false,
            host_to_device_bytes: 0,
            device_to_host_bytes: 0,
            kernel_invocations: 0,
            last_device: None,
            error: Some("bitnet-qk256-dispatch was built without the opencl feature".to_string()),
            execution_path: "a770_opencl_qk256_i2s_i8s_scaled_replay_unavailable",
        },
    )
}

#[cfg(feature = "opencl")]
fn record_a770_opencl_runtime_device(
    platform_index: usize,
    device_index: usize,
    platform_name: &str,
    runtime_device: &str,
    vendor: &str,
    driver_version: &str,
) {
    if let Ok(mut device) = A770_OPENCL_LAST_DEVICE.lock() {
        *device = Some(A770OpenClRuntimeDevice {
            platform_index,
            device_index,
            platform_name: platform_name.to_string(),
            runtime_device: runtime_device.to_string(),
            vendor: vendor.to_string(),
            driver_version: driver_version.to_string(),
        });
    }
}

fn forward_qk256_cpu(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
    inline_scale: Option<f32>,
) -> Result<Tensor> {
    use bitnet_quantization::i2s_qk256::{
        QK256_AVX2_GEMV_KERNEL_ID, QK256_SCALAR_GEMV_KERNEL_ID, gemv_qk256_bitnet_i8s_scaled,
        gemv_qk256_with_kernel_selection,
    };

    let prepared = prepare_qk256_forward(input, qk256_tensor, weight_name)?;
    let output_row_count = prepared.shape.batch_size * prepared.shape.seq_len;
    QK256_OUTPUT_ROWS_ALLOCATED_COUNT.fetch_add(output_row_count as u64, Ordering::Relaxed);
    let mut output_rows = vec![vec![0.0f32; prepared.layout.rows]; output_row_count];

    if std::env::var("BITNET_TRACE_RMS").as_deref() == Ok("1") && weight_name.contains("layers.0.")
    {
        static DIM_LOGGED: std::sync::Once = std::sync::Once::new();
        DIM_LOGGED.call_once(|| {
            eprintln!(
                "trace_qk256: weight={} rows={} cols={} row_stride_bytes={} qk256_shape={:?}",
                weight_name,
                prepared.layout.rows,
                prepared.layout.cols,
                prepared.layout.row_stride_bytes,
                qk256_tensor.dims()
            );
        });
    }

    for (row_index, input_row) in prepared.input_rows.iter().enumerate() {
        let gemv_result = if let Some(scale) = inline_scale {
            QK256_I8S_SCALED_SCALAR_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
            gemv_qk256_bitnet_i8s_scaled(
                &prepared.flat_bytes,
                input_row,
                &mut output_rows[row_index],
                prepared.layout.rows,
                prepared.layout.cols,
                prepared.layout.row_stride_bytes,
                scale,
            )
        } else {
            match gemv_qk256_with_kernel_selection(
                &prepared.flat_bytes,
                input_row,
                &mut output_rows[row_index],
                prepared.layout.rows,
                prepared.layout.cols,
                prepared.layout.row_stride_bytes,
                None,
                false,
            ) {
                Ok(selection) => {
                    match selection.selected_kernel {
                        QK256_AVX2_GEMV_KERNEL_ID => {
                            QK256_F32_AVX2_GEMV_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
                        }
                        QK256_SCALAR_GEMV_KERNEL_ID => {
                            QK256_F32_SCALAR_GEMV_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
                        }
                        _ => {}
                    }
                    Ok(())
                }
                Err(err) => Err(err),
            }
        };
        gemv_result.map_err(|e| {
            BitNetError::Validation(format!(
                "QK256 GEMV failed for {} at row {}: {}",
                weight_name, row_index, e
            ))
        })?;
    }

    tensor_from_flat_output(
        output_rows.into_iter().flatten().collect(),
        &prepared.shape,
        &prepared.layout,
        input,
    )
}

#[cfg(feature = "cuda")]
fn forward_qk256_cuda(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
    inline_scale: Option<f32>,
) -> Result<Tensor> {
    if let Some(scale) = inline_scale
        && !scale.is_finite()
    {
        return Err(BitNetError::Validation(format!("QK256 inline scale is not finite: {scale}")));
    }
    let prepared = prepare_qk256_activation(input, qk256_tensor, weight_name)?;
    let mut input_flat = Vec::with_capacity(prepared.input_rows.len() * prepared.layout.cols);
    for row in &prepared.input_rows {
        input_flat.extend_from_slice(row);
    }
    let mut output_flat =
        vec![0.0f32; prepared.shape.batch_size * prepared.shape.seq_len * prepared.layout.rows];
    let seq_len = prepared.shape.batch_size * prepared.shape.seq_len;

    with_cuda_qk256_context(|context| {
        let handle = if let Some(handle) = context.weight_handle(weight_name) {
            handle.clone()
        } else {
            let flat_bytes = extract_qk256_flat_bytes(qk256_tensor, &prepared.layout, weight_name)?;
            let weights = PackedQk256Weights::from_strict_gguf_no_scale(
                prepared.layout.rows,
                prepared.layout.cols,
                prepared.layout.row_stride_bytes,
                flat_bytes,
            )
            .map_err(BitNetError::from)?;
            context.upload_qk256_weights(weight_name, &weights).map_err(BitNetError::from)?
        };
        let mut stats =
            bitnet_kernels::cuda::CudaBitnetKernelInvocationStats::new(CUDA_QK256_GEMV_KERNEL_ID);
        let result = context
            .qk256_gemv(&handle, &input_flat, &mut output_flat, seq_len, inline_scale, &mut stats)
            .map_err(BitNetError::from);
        if result.is_ok() {
            record_cuda_qk256_runtime_stats(&stats);
        }
        result
    })?;

    tensor_from_flat_output(output_flat, &prepared.shape, &prepared.layout, input)
}

#[cfg(feature = "cuda")]
fn with_cuda_qk256_context<T>(f: impl FnOnce(&mut CudaBitnetContext) -> Result<T>) -> Result<T> {
    CUDA_QK256_CONTEXT.with(|cell| {
        let mut context = cell.borrow_mut();
        if context.is_none() {
            *context =
                Some(CudaBitnetContext::new(cuda_qk256_device_index()).map_err(BitNetError::from)?);
        }
        let context = context.as_mut().ok_or_else(|| {
            BitNetError::Validation(
                "failed to initialize persistent CUDA QK256 context".to_string(),
            )
        })?;
        f(context)
    })
}

#[cfg(feature = "cuda")]
fn cuda_qk256_device_index() -> usize {
    std::env::var("BITNET_RTX5070TI_CUDA_DEVICE_INDEX")
        .or_else(|_| std::env::var("CUDA_VISIBLE_DEVICE_INDEX"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

#[cfg(feature = "cuda")]
fn reset_cuda_qk256_context() {
    CUDA_QK256_CONTEXT.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

#[cfg(not(feature = "cuda"))]
fn reset_cuda_qk256_context() {}

#[cfg(feature = "cuda")]
fn cuda_qk256_weight_residency() -> Option<Qk256CudaWeightResidency> {
    CUDA_QK256_CONTEXT.with(|cell| {
        cell.borrow().as_ref().map(|context| {
            let fields = context.receipt_fields();
            Qk256CudaWeightResidency {
                weight_handle_count: fields.weight_handle_count,
                weights_uploaded_once: fields.weights_uploaded_once,
                per_token_weight_upload: fields.per_token_weight_upload,
            }
        })
    })
}

#[cfg(not(feature = "cuda"))]
fn cuda_qk256_weight_residency() -> Option<Qk256CudaWeightResidency> {
    None
}

struct PreparedQk256Forward {
    layout: Qk256Layout,
    shape: Qk256InputShape,
    flat_bytes: Vec<u8>,
    input_rows: Vec<Vec<f32>>,
}

struct PreparedQk256Activation {
    layout: Qk256Layout,
    shape: Qk256InputShape,
    input_rows: Vec<Vec<f32>>,
}

fn prepare_qk256_forward(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
) -> Result<PreparedQk256Forward> {
    prepare_qk256_forward_with_tracking(input, qk256_tensor, weight_name, true)
}

fn prepare_qk256_forward_untracked(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
) -> Result<PreparedQk256Forward> {
    prepare_qk256_forward_with_tracking(input, qk256_tensor, weight_name, false)
}

fn prepare_qk256_forward_with_tracking(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
    track_counters: bool,
) -> Result<PreparedQk256Forward> {
    let prepared =
        prepare_qk256_activation_with_tracking(input, qk256_tensor, weight_name, track_counters)?;
    let flat_bytes = extract_qk256_flat_bytes_with_tracking(
        qk256_tensor,
        &prepared.layout,
        weight_name,
        track_counters,
    )?;

    Ok(PreparedQk256Forward {
        layout: prepared.layout,
        shape: prepared.shape,
        flat_bytes,
        input_rows: prepared.input_rows,
    })
}

fn prepare_qk256_activation_with_tracking(
    input: &Tensor,
    qk256_tensor: &Tensor,
    weight_name: &str,
    track_counters: bool,
) -> Result<PreparedQk256Activation> {
    let qk256_dims = qk256_tensor.dims();
    let layout = parse_qk256_layout(weight_name, qk256_dims)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;

    debug_assert!(
        layout.row_stride_bytes.is_multiple_of(64),
        "QK256 row_stride_bytes must be multiple of 64"
    );

    let shape =
        parse_input_shape(input.dims()).map_err(|e| BitNetError::Validation(e.to_string()))?;

    validate_input_cols(weight_name, shape.cols, layout.cols)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;

    let input_flat = input.reshape(&[shape.batch_size * shape.seq_len, layout.cols])?;
    let input_rows = input_flat.to_vec2::<f32>().map_err(|e| {
        BitNetError::Validation(format!(
            "Failed to convert input to f32 for {}: {}",
            weight_name, e
        ))
    })?;
    if track_counters {
        QK256_INPUT_ROWS_MATERIALIZED_COUNT.fetch_add(input_rows.len() as u64, Ordering::Relaxed);
    }

    Ok(PreparedQk256Activation { layout, shape, input_rows })
}

fn extract_qk256_flat_bytes_with_tracking(
    qk256_tensor: &Tensor,
    layout: &Qk256Layout,
    weight_name: &str,
    track_counters: bool,
) -> Result<Vec<u8>> {
    let bytes_2d = qk256_tensor.to_vec2::<u8>().map_err(|e| {
        BitNetError::Validation(format!("Failed to extract QK256 bytes for {}: {}", weight_name, e))
    })?;
    let mut flat_bytes = Vec::with_capacity(layout.rows * layout.row_stride_bytes);
    for row in bytes_2d {
        flat_bytes.extend_from_slice(&row);
    }
    if track_counters {
        QK256_FLAT_BYTES_EXTRACTED_COUNT.fetch_add(1, Ordering::Relaxed);
    }
    Ok(flat_bytes)
}

fn tensor_from_flat_output(
    output_flat: Vec<f32>,
    shape: &Qk256InputShape,
    layout: &Qk256Layout,
    input: &Tensor,
) -> Result<Tensor> {
    let output_tensor = if shape.input_rank == 3 {
        Tensor::from_vec(
            output_flat,
            (shape.batch_size, shape.seq_len, layout.rows),
            input.device(),
        )?
    } else {
        Tensor::from_vec(output_flat, (shape.batch_size, layout.rows), input.device())?
    };

    Ok(output_tensor)
}

#[cfg(feature = "cuda")]
fn record_cuda_qk256_runtime_stats(stats: &bitnet_kernels::cuda::CudaBitnetKernelInvocationStats) {
    CUDA_QK256_HOST_TO_DEVICE_BYTES.fetch_add(stats.host_to_device_bytes, Ordering::Relaxed);
    CUDA_QK256_DEVICE_TO_HOST_BYTES.fetch_add(stats.device_to_host_bytes, Ordering::Relaxed);
    if let Some(host_to_device_ms) = stats.host_to_device_ms {
        let micros = (host_to_device_ms.max(0.0) * 1000.0).round() as u64;
        CUDA_QK256_HOST_TO_DEVICE_MICROS.fetch_add(micros, Ordering::Relaxed);
        CUDA_QK256_HOST_TO_DEVICE_TIME_SAMPLES
            .fetch_add(stats.host_to_device_time_samples.max(1), Ordering::Relaxed);
    }
    if let Some(device_to_host_ms) = stats.device_to_host_ms {
        let micros = (device_to_host_ms.max(0.0) * 1000.0).round() as u64;
        CUDA_QK256_DEVICE_TO_HOST_MICROS.fetch_add(micros, Ordering::Relaxed);
        CUDA_QK256_DEVICE_TO_HOST_TIME_SAMPLES
            .fetch_add(stats.device_to_host_time_samples.max(1), Ordering::Relaxed);
    }
    if let Some(kernel_time_ms) = stats.kernel_time_ms {
        let micros = (kernel_time_ms.max(0.0) * 1000.0).round() as u64;
        CUDA_QK256_KERNEL_TIME_MICROS.fetch_add(micros, Ordering::Relaxed);
        CUDA_QK256_KERNEL_TIME_SAMPLES.fetch_add(1, Ordering::Relaxed);
    }
}

fn backend_env_matches(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        matches!(value.to_ascii_lowercase().as_str(), "nvidia-rtx-5070-ti-cuda" | "cuda")
    })
}

fn a770_opencl_backend_env_matches(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        let value = value.to_ascii_lowercase();
        matches!(value.as_str(), "intel-a770-opencl" | "intel-arc-a770-opencl" | "a770-opencl")
            || (value.contains("a770") && value.contains("opencl"))
    })
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|value| value.parse::<usize>().ok())
}
