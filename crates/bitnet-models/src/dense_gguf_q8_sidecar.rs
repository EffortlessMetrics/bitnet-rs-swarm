//! Inert Q8_0 GGUF sidecar descriptors.
//!
//! This module carries packed Q8_0 tensor metadata from strict GGUF loading
//! toward future dense-linear runtime work. It deliberately does not replace
//! the eager F32 Candle tensors used for model execution.

use crate::dense_gguf_descriptors::{DenseGgufTensorRole, classify_dense_tensor_role};
use crate::formats::gguf::{GgufTensorType, TensorInfo};
use crate::names::is_projection_weight;
use bitnet_common::{BitNetError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub const DENSE_GGUF_Q8_SIDECAR_REGISTRY_ARTIFACT_KIND: &str = "dense_gguf_q8_sidecar_registry";
pub const DENSE_Q8_PAYLOAD_ENABLE_ENV: &str = "BITNET_DENSE_Q8_PAYLOAD_ENABLE";
pub const DENSE_Q8_PAYLOAD_TENSOR_ENV: &str = "BITNET_DENSE_Q8_PAYLOAD_TENSOR";
pub const DENSE_Q8_RUNTIME_ENABLE_ENV: &str = "BITNET_DENSE_Q8_RUNTIME_ENABLE";
pub const DENSE_Q8_RUNTIME_TENSOR_ENV: &str = "BITNET_DENSE_Q8_RUNTIME_TENSOR";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseGgufQ8SidecarDescriptor {
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub tensor_type: String,
    pub source_shape: Vec<usize>,
    pub runtime_candle_shape: Vec<usize>,
    pub source_offset: u64,
    pub source_size_bytes: u64,
    pub value_count: usize,
    pub q8_block_size: usize,
    pub q8_block_count: usize,
    pub q8_payload_bytes: usize,
    pub packed_q8_bytes_sha256: String,
    #[serde(skip_serializing, skip_deserializing, default)]
    pub packed_q8_bytes: Option<Arc<[u8]>>,
    pub shape_reshaped_without_transpose: bool,
    pub eager_f32_runtime_preserved: bool,
    pub runtime_compute_enabled: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub next_runtime_api_hook: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseGgufQ8SidecarRegistry {
    pub schema: u64,
    pub artifact_kind: String,
    pub descriptors: Vec<DenseGgufQ8SidecarDescriptor>,
    pub eager_f32_runtime_preserved: bool,
    pub runtime_compute_enabled: bool,
    pub dense_runtime_replaced: bool,
    pub speedup_claim: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub next_runtime_api_hook: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8PayloadOrderProofStatus {
    MatchesRuntimeShape,
    BlockedPendingPayloadReorderProof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8PayloadReorderContractStatus {
    NativePayloadOrder,
    BlockedRequiresDequantizeRequantize,
    BlockedUnsupportedShapeRank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8SourceOrderKernelContractStatus {
    NativePayloadOrder,
    RuntimeDisabledSourceOrderMatvecCandidate,
    BlockedUnsupportedShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseQ8SourceOrderMatvecPrototypeStatus {
    RuntimeDisabledPrototypeAvailable,
    BlockedMissingPayload,
    BlockedUnsupportedContract,
    NativePayloadOrderUsesExistingGate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseQ8PayloadOrderProof {
    pub schema: u64,
    pub artifact_kind: String,
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub source_shape: Vec<usize>,
    pub runtime_candle_shape: Vec<usize>,
    pub source_payload_order_matches_runtime_shape: bool,
    pub runtime_shape_requires_reorder: bool,
    pub q8_block_size: usize,
    pub q8_block_count: usize,
    pub q8_payload_bytes: usize,
    pub packed_q8_bytes_sha256: String,
    pub proof_status: DenseQ8PayloadOrderProofStatus,
    pub runtime_selection_allowed: bool,
    pub blocker: Option<String>,
    pub next_safe_step: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseQ8PayloadReorderContract {
    pub schema: u64,
    pub artifact_kind: String,
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub source_shape: Vec<usize>,
    pub runtime_candle_shape: Vec<usize>,
    pub q8_block_size: usize,
    pub q8_block_count: usize,
    pub source_row_major_block_span: Option<usize>,
    pub runtime_row_major_block_span: Option<usize>,
    pub source_payload_order_matches_runtime_shape: bool,
    pub pure_byte_reorder_possible: bool,
    pub requires_dequantize_requantize: bool,
    pub runtime_selection_allowed: bool,
    pub contract_status: DenseQ8PayloadReorderContractStatus,
    pub blocker: Option<String>,
    pub next_safe_step: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseQ8SourceOrderKernelContract {
    pub schema: u64,
    pub artifact_kind: String,
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub source_shape: Vec<usize>,
    pub runtime_candle_shape: Vec<usize>,
    pub q8_block_size: usize,
    pub q8_block_count: usize,
    pub source_input_dim: Option<usize>,
    pub source_output_dim: Option<usize>,
    pub runtime_input_dim: Option<usize>,
    pub runtime_output_dim: Option<usize>,
    pub source_row_major_block_span: Option<usize>,
    pub consumes_source_order_payload_directly: bool,
    pub requires_dequantize_requantize: bool,
    pub requires_output_accumulator: bool,
    pub runtime_selection_allowed: bool,
    pub proof_gate_required_before_runtime_selection: bool,
    pub contract_status: DenseQ8SourceOrderKernelContractStatus,
    pub blocker: Option<String>,
    pub next_safe_step: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseQ8SourceOrderMatvecPrototypeProof {
    pub schema: u64,
    pub artifact_kind: String,
    pub tensor_name: String,
    pub role: DenseGgufTensorRole,
    pub source_shape: Vec<usize>,
    pub runtime_candle_shape: Vec<usize>,
    pub q8_block_size: usize,
    pub q8_block_count: usize,
    pub input_len: usize,
    pub output_len: usize,
    pub consumes_source_order_payload_directly: bool,
    pub dequantizes_inside_matvec: bool,
    pub materializes_full_f32_weights: bool,
    pub compares_against_eager_f32_reference: bool,
    pub max_abs_diff_vs_eager_source_order_f32: Option<f32>,
    pub fused_output_sha256: Option<String>,
    pub eager_output_sha256: Option<String>,
    pub runtime_selection_allowed: bool,
    pub default_runtime_preserved: bool,
    pub generated_id_preservation_required_before_runtime_use: bool,
    pub prototype_status: DenseQ8SourceOrderMatvecPrototypeStatus,
    pub blocker: Option<String>,
    pub next_safe_step: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseQ8SourceOrderMatvecPrototype {
    pub proof: DenseQ8SourceOrderMatvecPrototypeProof,
    pub fused_output: Vec<f32>,
    pub eager_output: Vec<f32>,
}

impl Default for DenseGgufQ8SidecarRegistry {
    fn default() -> Self {
        Self {
            schema: 1,
            artifact_kind: DENSE_GGUF_Q8_SIDECAR_REGISTRY_ARTIFACT_KIND.to_string(),
            descriptors: Vec::new(),
            eager_f32_runtime_preserved: true,
            runtime_compute_enabled: false,
            dense_runtime_replaced: false,
            speedup_claim: false,
            generated_id_preservation_required_before_runtime_use: true,
            next_runtime_api_hook: "add a behavior-preserving dense linear dispatch API that can select a packed Q8_0 sidecar only after generated-ID and strict-receipt equivalence gates pass".to_string(),
        }
    }
}

impl DenseGgufQ8SidecarRegistry {
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }

    pub fn descriptor_for_tensor(
        &self,
        tensor_name: &str,
    ) -> Option<&DenseGgufQ8SidecarDescriptor> {
        self.descriptors.iter().find(|descriptor| descriptor.tensor_name == tensor_name)
    }

    pub fn try_push_tensor(&mut self, info: &TensorInfo, data: &[u8]) -> Result<()> {
        self.try_push_tensor_with_payload_candidate(info, data, None)
    }

    pub fn try_push_tensor_with_payload_candidate(
        &mut self,
        info: &TensorInfo,
        data: &[u8],
        payload_candidate_tensor: Option<&str>,
    ) -> Result<()> {
        let Some(descriptor) = DenseGgufQ8SidecarDescriptor::from_tensor_with_payload_candidate(
            info,
            data,
            payload_candidate_tensor,
        )?
        else {
            return Ok(());
        };
        self.descriptors.push(descriptor);
        Ok(())
    }
}

impl DenseGgufQ8SidecarDescriptor {
    pub fn from_tensor(info: &TensorInfo, data: &[u8]) -> Result<Option<Self>> {
        Self::from_tensor_with_payload_candidate(info, data, None)
    }

    pub fn from_tensor_with_payload_candidate(
        info: &TensorInfo,
        data: &[u8],
        payload_candidate_tensor: Option<&str>,
    ) -> Result<Option<Self>> {
        if info.tensor_type != GgufTensorType::Q8_0 {
            return Ok(None);
        }

        let role = classify_dense_tensor_role(&info.name);
        if !is_dense_linear_sidecar_role(role) {
            return Ok(None);
        }

        let value_count = checked_element_count(&info.shape, &info.name)?;
        let q8_block_size = GgufTensorType::Q8_0.block_size();
        let q8_block_count = value_count.div_ceil(q8_block_size);
        let q8_payload_bytes =
            q8_block_count.checked_mul(GgufTensorType::Q8_0.element_size()).ok_or_else(|| {
                BitNetError::Validation(format!(
                    "Q8_0 sidecar descriptor '{}' byte count overflows for {q8_block_count} blocks",
                    info.name
                ))
            })?;
        if data.len() < q8_payload_bytes {
            return Err(BitNetError::Validation(format!(
                "Q8_0 sidecar descriptor '{}' has {} bytes, expected at least {}",
                info.name,
                data.len(),
                q8_payload_bytes
            )));
        }

        let runtime_candle_shape = dense_q8_runtime_shape(info, role);
        let shape_reshaped_without_transpose = runtime_candle_shape != info.shape;
        let packed_q8_bytes = (payload_candidate_tensor == Some(info.name.as_str()))
            .then(|| Arc::<[u8]>::from(data[..q8_payload_bytes].to_vec().into_boxed_slice()));

        Ok(Some(Self {
            tensor_name: info.name.clone(),
            role,
            tensor_type: "q8_0".to_string(),
            source_shape: info.shape.clone(),
            runtime_candle_shape,
            source_offset: info.offset,
            source_size_bytes: info.size,
            value_count,
            q8_block_size,
            q8_block_count,
            q8_payload_bytes,
            packed_q8_bytes_sha256: bytes_sha256(&data[..q8_payload_bytes]),
            packed_q8_bytes,
            shape_reshaped_without_transpose,
            eager_f32_runtime_preserved: true,
            runtime_compute_enabled: false,
            dense_runtime_replaced: false,
            speedup_claim: false,
            generated_id_preservation_required_before_runtime_use: true,
            next_runtime_api_hook: "dense_linear_dispatch_q8_sidecar_candidate".to_string(),
        }))
    }

    pub fn payload_order_matches_runtime_shape(&self) -> bool {
        !self.shape_reshaped_without_transpose
    }

    pub fn payload_order_proof(&self) -> DenseQ8PayloadOrderProof {
        let source_payload_order_matches_runtime_shape = self.payload_order_matches_runtime_shape();
        let proof_status = if source_payload_order_matches_runtime_shape {
            DenseQ8PayloadOrderProofStatus::MatchesRuntimeShape
        } else {
            DenseQ8PayloadOrderProofStatus::BlockedPendingPayloadReorderProof
        };
        let blocker = (!source_payload_order_matches_runtime_shape).then(|| {
            format!(
                "GGUF source shape {:?} is represented at runtime as Candle shape {:?}; packed Q8_0 bytes remain in source payload order and require a tensor-specific reorder/runtime-shape proof before packed sidecar selection",
                self.source_shape, self.runtime_candle_shape
            )
        });
        let next_safe_step = if source_payload_order_matches_runtime_shape {
            "eligible for generated-ID before/after receipt gate before runtime selection"
                .to_string()
        } else {
            "produce a tensor-specific payload reorder proof, or keep the selector fail-closed on eager_f32_candle".to_string()
        };

        DenseQ8PayloadOrderProof {
            schema: 1,
            artifact_kind: "dense_q8_payload_order_proof".to_string(),
            tensor_name: self.tensor_name.clone(),
            role: self.role,
            source_shape: self.source_shape.clone(),
            runtime_candle_shape: self.runtime_candle_shape.clone(),
            source_payload_order_matches_runtime_shape,
            runtime_shape_requires_reorder: !source_payload_order_matches_runtime_shape,
            q8_block_size: self.q8_block_size,
            q8_block_count: self.q8_block_count,
            q8_payload_bytes: self.q8_payload_bytes,
            packed_q8_bytes_sha256: self.packed_q8_bytes_sha256.clone(),
            proof_status,
            runtime_selection_allowed: source_payload_order_matches_runtime_shape,
            blocker,
            next_safe_step,
        }
    }

    pub fn payload_reorder_contract(&self) -> DenseQ8PayloadReorderContract {
        let source_payload_order_matches_runtime_shape = self.payload_order_matches_runtime_shape();
        let source_row_major_block_span = self.source_shape.last().copied();
        let runtime_row_major_block_span = self.runtime_candle_shape.last().copied();

        let contract_status = if source_payload_order_matches_runtime_shape {
            DenseQ8PayloadReorderContractStatus::NativePayloadOrder
        } else if self.source_shape.len() == 2 && self.runtime_candle_shape.len() == 2 {
            DenseQ8PayloadReorderContractStatus::BlockedRequiresDequantizeRequantize
        } else {
            DenseQ8PayloadReorderContractStatus::BlockedUnsupportedShapeRank
        };
        let requires_dequantize_requantize = matches!(
            contract_status,
            DenseQ8PayloadReorderContractStatus::BlockedRequiresDequantizeRequantize
        );
        let runtime_selection_allowed =
            matches!(contract_status, DenseQ8PayloadReorderContractStatus::NativePayloadOrder);
        let pure_byte_reorder_possible = false;
        let blocker = match contract_status {
            DenseQ8PayloadReorderContractStatus::NativePayloadOrder => None,
            DenseQ8PayloadReorderContractStatus::BlockedRequiresDequantizeRequantize => {
                Some(format!(
                    "Q8_0 scales are attached to {}-value contiguous source row-major blocks; source shape {:?} would be consumed as runtime shape {:?}, so a pure byte reorder would regroup values under the wrong block scales. Runtime selection requires a dequantize/requantize proof or a kernel that consumes source-order Q8_0 blocks directly.",
                    self.q8_block_size, self.source_shape, self.runtime_candle_shape
                ))
            }
            DenseQ8PayloadReorderContractStatus::BlockedUnsupportedShapeRank => Some(format!(
                "Q8_0 payload reorder is defined only for native-order tensors and explicit 2D transpose blockers; source shape {:?} and runtime shape {:?} are unsupported for runtime selection",
                self.source_shape, self.runtime_candle_shape
            )),
        };
        let next_safe_step = match contract_status {
            DenseQ8PayloadReorderContractStatus::NativePayloadOrder => {
                "eligible for generated-ID before/after receipt gate before runtime selection"
                    .to_string()
            }
            DenseQ8PayloadReorderContractStatus::BlockedRequiresDequantizeRequantize => {
                "prove dequantize/requantize equivalence for the exact tensor, or implement a source-order Q8_0 kernel; keep eager_f32_candle selected until that proof exists"
                    .to_string()
            }
            DenseQ8PayloadReorderContractStatus::BlockedUnsupportedShapeRank => {
                "add an explicit tensor-rank-specific reorder contract, or keep eager_f32_candle selected"
                    .to_string()
            }
        };

        DenseQ8PayloadReorderContract {
            schema: 1,
            artifact_kind: "dense_q8_payload_reorder_contract".to_string(),
            tensor_name: self.tensor_name.clone(),
            role: self.role,
            source_shape: self.source_shape.clone(),
            runtime_candle_shape: self.runtime_candle_shape.clone(),
            q8_block_size: self.q8_block_size,
            q8_block_count: self.q8_block_count,
            source_row_major_block_span,
            runtime_row_major_block_span,
            source_payload_order_matches_runtime_shape,
            pure_byte_reorder_possible,
            requires_dequantize_requantize,
            runtime_selection_allowed,
            contract_status,
            blocker,
            next_safe_step,
        }
    }

    pub fn source_order_kernel_contract(&self) -> DenseQ8SourceOrderKernelContract {
        let is_2d_transpose = self.source_shape.len() == 2
            && self.runtime_candle_shape.len() == 2
            && self.source_shape[0] == self.runtime_candle_shape[1]
            && self.source_shape[1] == self.runtime_candle_shape[0]
            && self.source_shape != self.runtime_candle_shape;
        let native_payload_order = self.payload_order_matches_runtime_shape();

        let contract_status = if native_payload_order {
            DenseQ8SourceOrderKernelContractStatus::NativePayloadOrder
        } else if is_2d_transpose {
            DenseQ8SourceOrderKernelContractStatus::RuntimeDisabledSourceOrderMatvecCandidate
        } else {
            DenseQ8SourceOrderKernelContractStatus::BlockedUnsupportedShape
        };
        let source_input_dim = self.source_shape.first().copied();
        let source_output_dim = self.source_shape.get(1).copied();
        let runtime_output_dim = self.runtime_candle_shape.first().copied();
        let runtime_input_dim = self.runtime_candle_shape.get(1).copied();
        let consumes_source_order_payload_directly = matches!(
            contract_status,
            DenseQ8SourceOrderKernelContractStatus::RuntimeDisabledSourceOrderMatvecCandidate
        );
        let requires_output_accumulator = consumes_source_order_payload_directly;
        let proof_gate_required_before_runtime_selection = consumes_source_order_payload_directly;
        let runtime_selection_allowed = false;
        let blocker = match contract_status {
            DenseQ8SourceOrderKernelContractStatus::NativePayloadOrder => None,
            DenseQ8SourceOrderKernelContractStatus::RuntimeDisabledSourceOrderMatvecCandidate => {
                Some(format!(
                    "source-order Q8_0 matvec candidate for source shape {:?} and runtime shape {:?} must prove accumulator order, block-scale decode, generated-ID preservation, and receipt identity before runtime selection",
                    self.source_shape, self.runtime_candle_shape
                ))
            }
            DenseQ8SourceOrderKernelContractStatus::BlockedUnsupportedShape => Some(format!(
                "source-order Q8_0 kernel contract is defined only for native payload order or explicit 2D transpose candidates; source shape {:?} and runtime shape {:?} are unsupported",
                self.source_shape, self.runtime_candle_shape
            )),
        };
        let next_safe_step = match contract_status {
            DenseQ8SourceOrderKernelContractStatus::NativePayloadOrder => {
                "use the native payload-order receipt gate; no source-order transpose kernel is needed"
                    .to_string()
            }
            DenseQ8SourceOrderKernelContractStatus::RuntimeDisabledSourceOrderMatvecCandidate => {
                "add a runtime-disabled source-order Q8_0 matvec implementation and compare it against eager_f32_candle with the accepted Qwen3/Qwen2.5 behavior oracle before enabling selection"
                    .to_string()
            }
            DenseQ8SourceOrderKernelContractStatus::BlockedUnsupportedShape => {
                "add an explicit tensor-shape contract, or keep eager_f32_candle selected"
                    .to_string()
            }
        };

        DenseQ8SourceOrderKernelContract {
            schema: 1,
            artifact_kind: "dense_q8_source_order_kernel_contract".to_string(),
            tensor_name: self.tensor_name.clone(),
            role: self.role,
            source_shape: self.source_shape.clone(),
            runtime_candle_shape: self.runtime_candle_shape.clone(),
            q8_block_size: self.q8_block_size,
            q8_block_count: self.q8_block_count,
            source_input_dim,
            source_output_dim,
            runtime_input_dim,
            runtime_output_dim,
            source_row_major_block_span: self.source_shape.last().copied(),
            consumes_source_order_payload_directly,
            requires_dequantize_requantize: false,
            requires_output_accumulator,
            runtime_selection_allowed,
            proof_gate_required_before_runtime_selection,
            contract_status,
            blocker,
            next_safe_step,
        }
    }

    pub fn source_order_matvec_prototype(
        &self,
        input: &[f32],
    ) -> Result<DenseQ8SourceOrderMatvecPrototype> {
        let contract = self.source_order_kernel_contract();
        let status = match contract.contract_status {
            DenseQ8SourceOrderKernelContractStatus::RuntimeDisabledSourceOrderMatvecCandidate => {
                if self.packed_q8_bytes.is_some() {
                    DenseQ8SourceOrderMatvecPrototypeStatus::RuntimeDisabledPrototypeAvailable
                } else {
                    DenseQ8SourceOrderMatvecPrototypeStatus::BlockedMissingPayload
                }
            }
            DenseQ8SourceOrderKernelContractStatus::NativePayloadOrder => {
                DenseQ8SourceOrderMatvecPrototypeStatus::NativePayloadOrderUsesExistingGate
            }
            DenseQ8SourceOrderKernelContractStatus::BlockedUnsupportedShape => {
                DenseQ8SourceOrderMatvecPrototypeStatus::BlockedUnsupportedContract
            }
        };

        let input_len = contract.source_input_dim.unwrap_or_default();
        let output_len = contract.source_output_dim.unwrap_or_default();
        let blocker = match status {
            DenseQ8SourceOrderMatvecPrototypeStatus::RuntimeDisabledPrototypeAvailable => None,
            DenseQ8SourceOrderMatvecPrototypeStatus::BlockedMissingPayload => Some(format!(
                "source-order Q8_0 matvec prototype for '{}' requires exact packed payload bytes; set {}=1 and {}={} during strict GGUF load",
                self.tensor_name, DENSE_Q8_PAYLOAD_ENABLE_ENV, DENSE_Q8_PAYLOAD_TENSOR_ENV, self.tensor_name
            )),
            DenseQ8SourceOrderMatvecPrototypeStatus::BlockedUnsupportedContract => {
                contract.blocker.clone()
            }
            DenseQ8SourceOrderMatvecPrototypeStatus::NativePayloadOrderUsesExistingGate => Some(
                "native payload order should use the existing native packed-Q8 gate, not the source-order transpose prototype"
                    .to_string(),
            ),
        };

        if status != DenseQ8SourceOrderMatvecPrototypeStatus::RuntimeDisabledPrototypeAvailable {
            return Ok(DenseQ8SourceOrderMatvecPrototype {
                proof: DenseQ8SourceOrderMatvecPrototypeProof {
                    schema: 1,
                    artifact_kind: "dense_q8_source_order_matvec_prototype_proof".to_string(),
                    tensor_name: self.tensor_name.clone(),
                    role: self.role,
                    source_shape: self.source_shape.clone(),
                    runtime_candle_shape: self.runtime_candle_shape.clone(),
                    q8_block_size: self.q8_block_size,
                    q8_block_count: self.q8_block_count,
                    input_len,
                    output_len,
                    consumes_source_order_payload_directly: false,
                    dequantizes_inside_matvec: false,
                    materializes_full_f32_weights: false,
                    compares_against_eager_f32_reference: false,
                    max_abs_diff_vs_eager_source_order_f32: None,
                    fused_output_sha256: None,
                    eager_output_sha256: None,
                    runtime_selection_allowed: false,
                    default_runtime_preserved: true,
                    generated_id_preservation_required_before_runtime_use: true,
                    prototype_status: status,
                    blocker,
                    next_safe_step: contract.next_safe_step,
                },
                fused_output: Vec::new(),
                eager_output: Vec::new(),
            });
        }

        let Some(source_input_dim) = contract.source_input_dim else {
            return Err(BitNetError::Validation(format!(
                "source-order Q8_0 matvec prototype '{}' has no source input dimension",
                self.tensor_name
            )));
        };
        let Some(source_output_dim) = contract.source_output_dim else {
            return Err(BitNetError::Validation(format!(
                "source-order Q8_0 matvec prototype '{}' has no source output dimension",
                self.tensor_name
            )));
        };
        if input.len() != source_input_dim {
            return Err(BitNetError::Validation(format!(
                "source-order Q8_0 matvec prototype '{}' input length {} does not match source input dim {}",
                self.tensor_name,
                input.len(),
                source_input_dim
            )));
        }
        let payload = self.packed_q8_bytes.as_deref().ok_or_else(|| {
            BitNetError::Validation(format!(
                "source-order Q8_0 matvec prototype '{}' is missing packed payload bytes",
                self.tensor_name
            ))
        })?;
        let fused_output = source_order_q8_0_matvec(
            payload,
            self.q8_block_count,
            self.q8_block_size,
            source_input_dim,
            source_output_dim,
            input,
            &self.tensor_name,
        )?;
        let eager_output = source_order_eager_f32_matvec(
            payload,
            self.q8_block_count,
            self.q8_block_size,
            source_input_dim,
            source_output_dim,
            input,
            &self.tensor_name,
        )?;
        let max_abs_diff = max_abs_diff(&fused_output, &eager_output)?;

        Ok(DenseQ8SourceOrderMatvecPrototype {
            proof: DenseQ8SourceOrderMatvecPrototypeProof {
                schema: 1,
                artifact_kind: "dense_q8_source_order_matvec_prototype_proof".to_string(),
                tensor_name: self.tensor_name.clone(),
                role: self.role,
                source_shape: self.source_shape.clone(),
                runtime_candle_shape: self.runtime_candle_shape.clone(),
                q8_block_size: self.q8_block_size,
                q8_block_count: self.q8_block_count,
                input_len: source_input_dim,
                output_len: source_output_dim,
                consumes_source_order_payload_directly: true,
                dequantizes_inside_matvec: true,
                materializes_full_f32_weights: false,
                compares_against_eager_f32_reference: true,
                max_abs_diff_vs_eager_source_order_f32: Some(max_abs_diff),
                fused_output_sha256: Some(f32_values_sha256(&fused_output)),
                eager_output_sha256: Some(f32_values_sha256(&eager_output)),
                runtime_selection_allowed: false,
                default_runtime_preserved: true,
                generated_id_preservation_required_before_runtime_use: true,
                prototype_status: status,
                blocker: Some(
                    "runtime selection remains disabled until this prototype is validated against real Qwen3/Qwen2.5 behavior receipts with unchanged generated IDs and receipt identity"
                        .to_string(),
                ),
                next_safe_step:
                    "capture exact-model before/after behavior receipts before any selector use"
                        .to_string(),
            },
            fused_output,
            eager_output,
        })
    }
}

pub fn dense_q8_payload_candidate_tensor_from_env() -> Option<String> {
    let enabled = std::env::var(DENSE_Q8_PAYLOAD_ENABLE_ENV).ok()?;
    if !matches!(enabled.as_str(), "1" | "true" | "TRUE" | "yes" | "YES") {
        return None;
    }
    std::env::var(DENSE_Q8_PAYLOAD_TENSOR_ENV).ok().and_then(|tensor| {
        let tensor = tensor.trim();
        (!tensor.is_empty()).then(|| tensor.to_string())
    })
}

pub fn dense_q8_runtime_compute_tensor_from_env() -> Option<String> {
    let enabled = std::env::var(DENSE_Q8_RUNTIME_ENABLE_ENV).ok()?;
    if !matches!(enabled.as_str(), "1" | "true" | "TRUE" | "yes" | "YES") {
        return None;
    }
    std::env::var(DENSE_Q8_RUNTIME_TENSOR_ENV).ok().and_then(|tensor| {
        let tensor = tensor.trim();
        (!tensor.is_empty()).then(|| tensor.to_string())
    })
}

fn is_dense_linear_sidecar_role(role: DenseGgufTensorRole) -> bool {
    matches!(
        role,
        DenseGgufTensorRole::TokenEmbedding
            | DenseGgufTensorRole::Output
            | DenseGgufTensorRole::AttentionQ
            | DenseGgufTensorRole::AttentionK
            | DenseGgufTensorRole::AttentionV
            | DenseGgufTensorRole::AttentionOutput
            | DenseGgufTensorRole::MlpGate
            | DenseGgufTensorRole::MlpUp
            | DenseGgufTensorRole::MlpDown
    )
}

fn dense_q8_runtime_shape(info: &TensorInfo, role: DenseGgufTensorRole) -> Vec<usize> {
    if info.shape.len() == 2
        && ((matches!(role, DenseGgufTensorRole::TokenEmbedding | DenseGgufTensorRole::Output)
            && embedding_is_transposed(&info.shape))
            || is_projection_weight(&info.name))
    {
        vec![info.shape[1], info.shape[0]]
    } else {
        info.shape.clone()
    }
}

fn embedding_is_transposed(shape: &[usize]) -> bool {
    shape.len() == 2 && shape[0] < shape[1] && shape[1] >= 32768
}

fn checked_element_count(shape: &[usize], tensor_name: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            BitNetError::Validation(format!(
                "Q8_0 sidecar descriptor '{tensor_name}' element count overflows for shape {shape:?}"
            ))
        })
    })
}

fn source_order_q8_0_matvec(
    bytes: &[u8],
    q8_block_count: usize,
    q8_block_size: usize,
    source_input_dim: usize,
    source_output_dim: usize,
    input: &[f32],
    tensor_name: &str,
) -> Result<Vec<f32>> {
    let expected_values =
        checked_matvec_value_count(source_input_dim, source_output_dim, tensor_name)?;
    let expected_blocks = expected_values.div_ceil(q8_block_size);
    if expected_blocks != q8_block_count {
        return Err(BitNetError::Validation(format!(
            "source-order Q8_0 matvec prototype '{tensor_name}' expected {expected_blocks} blocks from shape, descriptor has {q8_block_count}"
        )));
    }
    let expected_bytes =
        q8_block_count.checked_mul(GgufTensorType::Q8_0.element_size()).ok_or_else(|| {
            BitNetError::Validation(format!(
                "source-order Q8_0 matvec prototype '{tensor_name}' byte count overflows for {q8_block_count} blocks"
            ))
        })?;
    if bytes.len() < expected_bytes {
        return Err(BitNetError::Validation(format!(
            "source-order Q8_0 matvec prototype '{tensor_name}' has {} bytes, expected at least {expected_bytes}",
            bytes.len()
        )));
    }
    if input.len() != source_input_dim {
        return Err(BitNetError::Validation(format!(
            "source-order Q8_0 matvec prototype '{tensor_name}' input length {} does not match source input dim {source_input_dim}",
            input.len()
        )));
    }

    let mut output = vec![0.0f32; source_output_dim];
    for (source_input_idx, input_value) in input.iter().enumerate().take(source_input_dim) {
        let source_row_start = source_input_idx * source_output_dim;
        for (source_output_idx, output_value) in output.iter_mut().enumerate() {
            let weight_idx = source_row_start + source_output_idx;
            let weight =
                q8_0_value_at(bytes, weight_idx, expected_values, q8_block_size, tensor_name)?;
            *output_value += weight * *input_value;
        }
    }
    Ok(output)
}

fn source_order_eager_f32_matvec(
    bytes: &[u8],
    q8_block_count: usize,
    q8_block_size: usize,
    source_input_dim: usize,
    source_output_dim: usize,
    input: &[f32],
    tensor_name: &str,
) -> Result<Vec<f32>> {
    let expected_values =
        checked_matvec_value_count(source_input_dim, source_output_dim, tensor_name)?;
    let expected_blocks = expected_values.div_ceil(q8_block_size);
    if expected_blocks != q8_block_count {
        return Err(BitNetError::Validation(format!(
            "source-order eager f32 matvec prototype '{tensor_name}' expected {expected_blocks} blocks from shape, descriptor has {q8_block_count}"
        )));
    }
    let mut values = Vec::with_capacity(expected_values);
    for weight_idx in 0..expected_values {
        values.push(q8_0_value_at(bytes, weight_idx, expected_values, q8_block_size, tensor_name)?);
    }
    let mut output = vec![0.0f32; source_output_dim];
    for (source_input_idx, input_value) in input.iter().enumerate().take(source_input_dim) {
        let source_row_start = source_input_idx * source_output_dim;
        for source_output_idx in 0..source_output_dim {
            output[source_output_idx] +=
                values[source_row_start + source_output_idx] * *input_value;
        }
    }
    Ok(output)
}

fn checked_matvec_value_count(
    source_input_dim: usize,
    source_output_dim: usize,
    tensor_name: &str,
) -> Result<usize> {
    source_input_dim.checked_mul(source_output_dim).ok_or_else(|| {
        BitNetError::Validation(format!(
            "source-order Q8_0 matvec prototype '{tensor_name}' shape overflows: {source_input_dim}x{source_output_dim}"
        ))
    })
}

fn q8_0_value_at(
    bytes: &[u8],
    value_idx: usize,
    expected_values: usize,
    q8_block_size: usize,
    tensor_name: &str,
) -> Result<f32> {
    if q8_block_size != GgufTensorType::Q8_0.block_size() {
        return Err(BitNetError::Validation(format!(
            "source-order Q8_0 matvec prototype '{tensor_name}' block size {q8_block_size} does not match GGUF Q8_0 block size {}",
            GgufTensorType::Q8_0.block_size()
        )));
    }
    if value_idx >= expected_values {
        return Err(BitNetError::Validation(format!(
            "source-order Q8_0 matvec prototype '{tensor_name}' value index {value_idx} exceeds {expected_values}"
        )));
    }
    let block_idx = value_idx / q8_block_size;
    let code_idx = value_idx % q8_block_size;
    let block_offset =
        block_idx.checked_mul(GgufTensorType::Q8_0.element_size()).ok_or_else(|| {
            BitNetError::Validation(format!(
                "source-order Q8_0 matvec prototype '{tensor_name}' block offset overflows for block {block_idx}"
            ))
        })?;
    if block_offset + 2 + code_idx >= bytes.len() {
        return Err(BitNetError::Validation(format!(
            "source-order Q8_0 matvec prototype '{tensor_name}' value index {value_idx} reads past payload"
        )));
    }
    let scale_bits = u16::from_le_bytes([bytes[block_offset], bytes[block_offset + 1]]);
    let scale = half::f16::from_bits(scale_bits).to_f32();
    let q = bytes[block_offset + 2 + code_idx] as i8;
    Ok(scale * f32::from(q))
}

fn f32_values_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> Result<f32> {
    if lhs.len() != rhs.len() {
        return Err(BitNetError::Validation(format!(
            "source-order Q8_0 matvec output length mismatch: {} vs {}",
            lhs.len(),
            rhs.len()
        )));
    }
    Ok(lhs.iter().zip(rhs).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max))
}

fn bytes_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q8_info(name: &str, shape: Vec<usize>, size: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            shape,
            tensor_type: GgufTensorType::Q8_0,
            offset: 128,
            size,
        }
    }

    fn q8_0_blob(scale: f32, code: i8) -> Vec<u8> {
        let mut blob = Vec::with_capacity(GgufTensorType::Q8_0.element_size());
        blob.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        for _ in 0..32 {
            blob.push(code as u8);
        }
        blob
    }

    #[test]
    fn dense_gguf_q8_sidecar_descriptor_is_metadata_only_and_inert() {
        let info = q8_info("blk.0.attn_q.weight", vec![2, 64], 136);
        let mut data = vec![0u8; 136];
        data.extend_from_slice(&[1, 2, 3, 4]);

        let descriptor = match DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data) {
            Ok(Some(descriptor)) => descriptor,
            result => {
                assert!(false, "expected sidecar descriptor, got {result:?}");
                return;
            }
        };

        assert_eq!(descriptor.role, DenseGgufTensorRole::AttentionQ);
        assert_eq!(descriptor.value_count, 128);
        assert_eq!(descriptor.q8_block_size, 32);
        assert_eq!(descriptor.q8_block_count, 4);
        assert_eq!(descriptor.q8_payload_bytes, 136);
        assert_eq!(descriptor.packed_q8_bytes_sha256, bytes_sha256(&data[..136]));
        assert!(descriptor.packed_q8_bytes.is_none());
        assert_ne!(descriptor.packed_q8_bytes_sha256, bytes_sha256(&data));
        assert_eq!(descriptor.runtime_candle_shape, vec![64, 2]);
        assert!(descriptor.shape_reshaped_without_transpose);
        assert!(descriptor.eager_f32_runtime_preserved);
        assert!(!descriptor.runtime_compute_enabled);
        assert!(!descriptor.dense_runtime_replaced);
        assert!(!descriptor.speedup_claim);
        assert!(descriptor.generated_id_preservation_required_before_runtime_use);
    }

    #[test]
    fn dense_gguf_q8_sidecar_registry_ignores_non_linear_q8_tensors() {
        let mut registry = DenseGgufQ8SidecarRegistry::default();
        let info = q8_info("blk.0.attn_norm.weight", vec![64], 68);
        let data = vec![0u8; 68];

        assert!(registry.try_push_tensor(&info, &data).is_ok());

        assert!(registry.is_empty());
        assert!(registry.eager_f32_runtime_preserved);
        assert!(!registry.runtime_compute_enabled);
        assert!(!registry.dense_runtime_replaced);
        assert!(!registry.speedup_claim);
    }

    #[test]
    fn dense_gguf_q8_sidecar_embedding_shape_matches_loader_heuristic() {
        let transposed = q8_info("token_embd.weight", vec![64, 32768], 2_228_224);
        let token_major = q8_info("token_embd.weight", vec![32768, 64], 2_228_224);
        let data = vec![0u8; 2_228_224];

        let transposed_descriptor =
            match DenseGgufQ8SidecarDescriptor::from_tensor(&transposed, &data) {
                Ok(Some(descriptor)) => descriptor,
                result => {
                    assert!(false, "expected transposed descriptor, got {result:?}");
                    return;
                }
            };
        let token_major_descriptor =
            match DenseGgufQ8SidecarDescriptor::from_tensor(&token_major, &data) {
                Ok(Some(descriptor)) => descriptor,
                result => {
                    assert!(false, "expected token-major descriptor, got {result:?}");
                    return;
                }
            };

        assert_eq!(transposed_descriptor.runtime_candle_shape, vec![32768, 64]);
        assert!(transposed_descriptor.shape_reshaped_without_transpose);
        assert_eq!(token_major_descriptor.runtime_candle_shape, vec![32768, 64]);
        assert!(!token_major_descriptor.shape_reshaped_without_transpose);
    }

    #[test]
    fn dense_gguf_q8_payload_order_proof_blocks_transposed_projection_payload() -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![1024, 2048], 2_228_224);
        let data = vec![0u8; 2_228_224];
        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data)?
            .ok_or_else(|| BitNetError::Validation("expected Q8 sidecar descriptor".to_string()))?;

        let proof = descriptor.payload_order_proof();

        assert_eq!(proof.tensor_name, "blk.0.attn_q.weight");
        assert_eq!(proof.source_shape, vec![1024, 2048]);
        assert_eq!(proof.runtime_candle_shape, vec![2048, 1024]);
        assert_eq!(
            proof.proof_status,
            DenseQ8PayloadOrderProofStatus::BlockedPendingPayloadReorderProof
        );
        assert!(!proof.source_payload_order_matches_runtime_shape);
        assert!(proof.runtime_shape_requires_reorder);
        assert!(!proof.runtime_selection_allowed);
        assert!(
            proof
                .blocker
                .as_deref()
                .is_some_and(|blocker| blocker.contains("require a tensor-specific reorder"))
        );
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_payload_order_proof_allows_matching_projection_payload() -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![896, 896], 852_992);
        let data = vec![0u8; 852_992];
        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data)?
            .ok_or_else(|| BitNetError::Validation("expected Q8 sidecar descriptor".to_string()))?;

        let proof = descriptor.payload_order_proof();

        assert_eq!(proof.source_shape, vec![896, 896]);
        assert_eq!(proof.runtime_candle_shape, vec![896, 896]);
        assert_eq!(proof.proof_status, DenseQ8PayloadOrderProofStatus::MatchesRuntimeShape);
        assert!(proof.source_payload_order_matches_runtime_shape);
        assert!(!proof.runtime_shape_requires_reorder);
        assert!(proof.runtime_selection_allowed);
        assert!(proof.blocker.is_none());
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_payload_reorder_contract_blocks_qwen3_qproj_byte_reorder() -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![1024, 2048], 2_228_224);
        let data = vec![0u8; 2_228_224];
        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data)?
            .ok_or_else(|| BitNetError::Validation("expected Q8 sidecar descriptor".to_string()))?;

        let contract = descriptor.payload_reorder_contract();

        assert_eq!(contract.source_shape, vec![1024, 2048]);
        assert_eq!(contract.runtime_candle_shape, vec![2048, 1024]);
        assert_eq!(
            contract.contract_status,
            DenseQ8PayloadReorderContractStatus::BlockedRequiresDequantizeRequantize
        );
        assert_eq!(contract.source_row_major_block_span, Some(2048));
        assert_eq!(contract.runtime_row_major_block_span, Some(1024));
        assert!(!contract.source_payload_order_matches_runtime_shape);
        assert!(!contract.pure_byte_reorder_possible);
        assert!(contract.requires_dequantize_requantize);
        assert!(!contract.runtime_selection_allowed);
        assert!(
            contract
                .blocker
                .as_deref()
                .is_some_and(|blocker| blocker.contains("wrong block scales"))
        );
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_payload_reorder_contract_allows_native_payload_order() -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![896, 896], 852_992);
        let data = vec![0u8; 852_992];
        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data)?
            .ok_or_else(|| BitNetError::Validation("expected Q8 sidecar descriptor".to_string()))?;

        let contract = descriptor.payload_reorder_contract();

        assert_eq!(
            contract.contract_status,
            DenseQ8PayloadReorderContractStatus::NativePayloadOrder
        );
        assert!(contract.source_payload_order_matches_runtime_shape);
        assert!(!contract.pure_byte_reorder_possible);
        assert!(!contract.requires_dequantize_requantize);
        assert!(contract.runtime_selection_allowed);
        assert!(contract.blocker.is_none());
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_source_order_kernel_contract_accepts_qwen3_qproj_candidate() -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![1024, 2048], 2_228_224);
        let data = vec![0u8; 2_228_224];
        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data)?
            .ok_or_else(|| BitNetError::Validation("expected Q8 sidecar descriptor".to_string()))?;

        let contract = descriptor.source_order_kernel_contract();

        assert_eq!(
            contract.contract_status,
            DenseQ8SourceOrderKernelContractStatus::RuntimeDisabledSourceOrderMatvecCandidate
        );
        assert_eq!(contract.source_input_dim, Some(1024));
        assert_eq!(contract.source_output_dim, Some(2048));
        assert_eq!(contract.runtime_input_dim, Some(1024));
        assert_eq!(contract.runtime_output_dim, Some(2048));
        assert_eq!(contract.source_row_major_block_span, Some(2048));
        assert!(contract.consumes_source_order_payload_directly);
        assert!(!contract.requires_dequantize_requantize);
        assert!(contract.requires_output_accumulator);
        assert!(contract.proof_gate_required_before_runtime_selection);
        assert!(!contract.runtime_selection_allowed);
        assert!(
            contract
                .blocker
                .as_deref()
                .is_some_and(|blocker| blocker.contains("accumulator order"))
        );
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_source_order_kernel_contract_uses_native_payload_gate_when_matching()
    -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![896, 896], 852_992);
        let data = vec![0u8; 852_992];
        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data)?
            .ok_or_else(|| BitNetError::Validation("expected Q8 sidecar descriptor".to_string()))?;

        let contract = descriptor.source_order_kernel_contract();

        assert_eq!(
            contract.contract_status,
            DenseQ8SourceOrderKernelContractStatus::NativePayloadOrder
        );
        assert!(!contract.consumes_source_order_payload_directly);
        assert!(!contract.requires_dequantize_requantize);
        assert!(!contract.proof_gate_required_before_runtime_selection);
        assert!(!contract.runtime_selection_allowed);
        assert!(contract.blocker.is_none());
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_source_order_matvec_prototype_matches_eager_source_order_reference()
    -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![2, 64], 136);
        let mut data = Vec::new();
        data.extend_from_slice(&q8_0_blob(1.0, 1));
        data.extend_from_slice(&q8_0_blob(1.0, 1));
        data.extend_from_slice(&q8_0_blob(1.0, 2));
        data.extend_from_slice(&q8_0_blob(1.0, 2));
        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor_with_payload_candidate(
            &info,
            &data,
            Some("blk.0.attn_q.weight"),
        )?
        .ok_or_else(|| BitNetError::Validation("expected Q8 sidecar descriptor".to_string()))?;

        let prototype = descriptor.source_order_matvec_prototype(&[3.0, 1.0])?;

        assert_eq!(
            prototype.proof.prototype_status,
            DenseQ8SourceOrderMatvecPrototypeStatus::RuntimeDisabledPrototypeAvailable
        );
        assert_eq!(prototype.proof.input_len, 2);
        assert_eq!(prototype.proof.output_len, 64);
        assert!(prototype.proof.consumes_source_order_payload_directly);
        assert!(prototype.proof.dequantizes_inside_matvec);
        assert!(!prototype.proof.materializes_full_f32_weights);
        assert!(prototype.proof.compares_against_eager_f32_reference);
        assert_eq!(prototype.proof.max_abs_diff_vs_eager_source_order_f32, Some(0.0));
        assert_eq!(prototype.fused_output, vec![5.0; 64]);
        assert_eq!(prototype.eager_output, prototype.fused_output);
        assert_eq!(prototype.proof.fused_output_sha256, prototype.proof.eager_output_sha256);
        assert!(!prototype.proof.runtime_selection_allowed);
        assert!(prototype.proof.default_runtime_preserved);
        assert!(prototype.proof.generated_id_preservation_required_before_runtime_use);
        assert!(
            prototype
                .proof
                .blocker
                .as_deref()
                .is_some_and(|blocker| blocker.contains("runtime selection remains disabled"))
        );
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_source_order_matvec_prototype_blocks_without_payload() -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![2, 64], 136);
        let data = vec![0u8; 136];
        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data)?
            .ok_or_else(|| BitNetError::Validation("expected Q8 sidecar descriptor".to_string()))?;

        let prototype = descriptor.source_order_matvec_prototype(&[3.0, 1.0])?;

        assert_eq!(
            prototype.proof.prototype_status,
            DenseQ8SourceOrderMatvecPrototypeStatus::BlockedMissingPayload
        );
        assert!(prototype.fused_output.is_empty());
        assert!(prototype.eager_output.is_empty());
        assert!(!prototype.proof.runtime_selection_allowed);
        assert!(prototype.proof.default_runtime_preserved);
        assert!(
            prototype
                .proof
                .blocker
                .as_deref()
                .is_some_and(|blocker| blocker.contains(DENSE_Q8_PAYLOAD_ENABLE_ENV))
        );
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_source_order_matvec_prototype_rejects_wrong_input_len() -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![2, 64], 136);
        let mut data = Vec::new();
        data.extend_from_slice(&q8_0_blob(1.0, 1));
        data.extend_from_slice(&q8_0_blob(1.0, 1));
        data.extend_from_slice(&q8_0_blob(1.0, 2));
        data.extend_from_slice(&q8_0_blob(1.0, 2));
        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor_with_payload_candidate(
            &info,
            &data,
            Some("blk.0.attn_q.weight"),
        )?
        .ok_or_else(|| BitNetError::Validation("expected Q8 sidecar descriptor".to_string()))?;

        let err = descriptor
            .source_order_matvec_prototype(&[1.0])
            .expect_err("wrong input length should fail");

        assert!(err.to_string().contains("input length 1 does not match source input dim 2"));
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_sidecar_rejects_truncated_payload() {
        let info = q8_info("blk.0.ffn_down.weight", vec![4, 32], 135);
        let data = vec![0u8; 135];

        let err = match DenseGgufQ8SidecarDescriptor::from_tensor(&info, &data) {
            Err(err) => err,
            result => {
                assert!(false, "expected truncated payload error, got {result:?}");
                return;
            }
        };

        assert!(err.to_string().contains("expected at least 136"));
    }

    #[test]
    fn dense_gguf_q8_sidecar_can_carry_exact_payload_candidate() -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![2, 64], 136);
        let mut data = vec![0u8; 136];
        data.extend_from_slice(&[1, 2, 3, 4]);

        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor_with_payload_candidate(
            &info,
            &data,
            Some("blk.0.attn_q.weight"),
        )
        .and_then(|descriptor| {
            descriptor.ok_or_else(|| {
                BitNetError::Validation("expected dense Q8 sidecar descriptor".to_string())
            })
        })?;

        let Some(payload) = descriptor.packed_q8_bytes.as_ref() else {
            return Err(BitNetError::Validation("expected packed payload bytes".to_string()));
        };
        assert_eq!(payload.len(), 136);
        assert_eq!(&payload[..], &data[..136]);
        assert_eq!(descriptor.packed_q8_bytes_sha256, bytes_sha256(payload));
        assert!(descriptor.eager_f32_runtime_preserved);
        assert!(!descriptor.runtime_compute_enabled);
        Ok(())
    }

    #[test]
    fn dense_gguf_q8_sidecar_payload_candidate_requires_exact_tensor_name() -> Result<()> {
        let info = q8_info("blk.0.attn_q.weight", vec![2, 64], 136);
        let data = vec![0u8; 136];

        let descriptor = DenseGgufQ8SidecarDescriptor::from_tensor_with_payload_candidate(
            &info,
            &data,
            Some("blk.0.attn_k.weight"),
        )
        .and_then(|descriptor| {
            descriptor.ok_or_else(|| {
                BitNetError::Validation("expected dense Q8 sidecar descriptor".to_string())
            })
        })?;

        assert!(descriptor.packed_q8_bytes.is_none());
        Ok(())
    }
}
