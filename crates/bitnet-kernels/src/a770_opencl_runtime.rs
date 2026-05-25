//! Selected-device Intel Arc A770 OpenCL runtime helpers.
//!
//! These helpers are the first bridge from A770 fixture parity into reusable
//! runtime code. They intentionally cover only the grouped QK256 I2_S x
//! prequantized I8_S scaled GEMV contract. Activation quantization remains a
//! CPU-side input to this helper, so this is not a full residency claim.

use bitnet_common::{KernelError, Result};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, ClMem};
use opencl3::platform::get_platforms;
use opencl3::program::{
    CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT, CL_PROGRAM_BINARY_TYPE_EXECUTABLE,
    CL_PROGRAM_BINARY_TYPE_LIBRARY, CL_PROGRAM_BINARY_TYPE_NONE, Program,
};
use opencl3::types::CL_BLOCKING;
use std::fmt::Display;

const QK256_I2S_I8S_SCALED_GEMV_SRC: &str = r#"
__kernel void qk256_i2s_i8s_scaled_gemv(
    __global const char* q,
    __global const uchar* qs,
    __global float* y,
    const uint rows,
    const uint cols,
    const uint row_stride_bytes,
    const int activation_sum,
    const float activation_scale,
    const float weight_scale
) {
    const uint row = get_global_id(0);
    if (row >= rows) return;

    int int_dot = 0;
    const uint row_base = row * row_stride_bytes;

    for (uint col = 0; col < cols; col++) {
        const uint block = col / 256;
        const uint offset = col - block * 256;
        const uint chunk = offset / 128;
        const uint lane = (offset - chunk * 128) / 32;
        const uint gp = offset & 31;
        const uint byte_index = row_base + block * 64 + chunk * 32 + gp;
        const uchar packed = qs[byte_index];
        const uchar code = (packed >> (6 - lane * 2)) & 0x03;
        int_dot += ((int)code) * ((int)q[col]);
    }

    y[row] = (((float)(int_dot - activation_sum)) / activation_scale) * weight_scale;
}
"#;

const QK256_I2S_I8S_SCALED_GEMV_DEBUG_SRC: &str = r#"
__kernel void qk256_i2s_i8s_scaled_gemv_debug(
    __global const char* q,
    __global const uchar* qs,
    __global int* int_values,
    __global uint* bit_values,
    const uint rows,
    const uint cols,
    const uint row_stride_bytes,
    const int activation_sum,
    const float activation_scale,
    const float weight_scale,
    const uint sample_limit
) {
    const uint row = get_global_id(0);
    if (row >= rows || row >= sample_limit) return;

    int int_dot = 0;
    const uint row_base = row * row_stride_bytes;

    for (uint col = 0; col < cols; col++) {
        const uint block = col / 256;
        const uint offset = col - block * 256;
        const uint chunk = offset / 128;
        const uint lane = (offset - chunk * 128) / 32;
        const uint gp = offset & 31;
        const uint byte_index = row_base + block * 64 + chunk * 32 + gp;
        const uchar packed = qs[byte_index];
        const uchar code = (packed >> (6 - lane * 2)) & 0x03;
        int_dot += ((int)code) * ((int)q[col]);
    }

    const int adjusted_dot = int_dot - activation_sum;
    const float adjusted_f32 = (float)adjusted_dot;
    const float output = (adjusted_f32 / activation_scale) * weight_scale;
    const float div_then_mul = (adjusted_f32 / activation_scale) * weight_scale;
    const float mul_then_div = (adjusted_f32 * weight_scale) / activation_scale;
    const float reciprocal_then_mul = adjusted_f32 * (weight_scale / activation_scale);
    volatile float volatile_adjusted_f32 = adjusted_f32;
    volatile float volatile_activation_scale = activation_scale;
    volatile float volatile_weight_scale = weight_scale;
    const float volatile_div_then_mul =
        (volatile_adjusted_f32 / volatile_activation_scale) * volatile_weight_scale;

    const uint int_base = row * 3;
    int_values[int_base + 0] = int_dot;
    int_values[int_base + 1] = activation_sum;
    int_values[int_base + 2] = adjusted_dot;

    const uint bit_base = row * 8;
    bit_values[bit_base + 0] = as_uint(activation_scale);
    bit_values[bit_base + 1] = as_uint(weight_scale);
    bit_values[bit_base + 2] = as_uint(adjusted_f32);
    bit_values[bit_base + 3] = as_uint(output);
    bit_values[bit_base + 4] = as_uint(div_then_mul);
    bit_values[bit_base + 5] = as_uint(mul_then_div);
    bit_values[bit_base + 6] = as_uint(reciprocal_then_mul);
    bit_values[bit_base + 7] = as_uint(volatile_div_then_mul);
}
"#;

/// Runtime request for the selected-device A770 QK256 scaled GEMV.
#[derive(Debug)]
pub struct A770OpenClQk256ScaledGemv<'a> {
    /// Prequantized I8_S activation row.
    pub activations_i8: &'a [i8],
    /// GGML grouped QK256 I2_S weight bytes.
    pub packed_qk256: &'a [u8],
    /// Number of output rows.
    pub rows: usize,
    /// Number of input columns.
    pub cols: usize,
    /// Packed byte stride for each output row.
    pub row_stride_bytes: usize,
    /// Sum of the prequantized I8_S activation row.
    pub activation_sum: i32,
    /// I8_S activation scale.
    pub activation_scale: f32,
    /// BitNet.cpp inline weight scale.
    pub weight_scale: f32,
}

/// Runtime request for a bounded selected-device A770 QK256 debug capture.
///
/// This uses a separate diagnostic kernel that writes sampled integer and
/// `f32` bit intermediates. It is not a production dispatch path.
#[derive(Debug)]
pub struct A770OpenClQk256ScaledGemvDebug<'a> {
    /// Prequantized I8_S activation row.
    pub activations_i8: &'a [i8],
    /// GGML grouped QK256 I2_S weight bytes.
    pub packed_qk256: &'a [u8],
    /// Number of output rows.
    pub rows: usize,
    /// Number of input columns.
    pub cols: usize,
    /// Packed byte stride for each output row.
    pub row_stride_bytes: usize,
    /// Sum of the prequantized I8_S activation row.
    pub activation_sum: i32,
    /// I8_S activation scale.
    pub activation_scale: f32,
    /// BitNet.cpp inline weight scale.
    pub weight_scale: f32,
    /// Maximum number of output rows to sample.
    pub sample_limit: usize,
}

/// Runtime result for the selected-device A770 QK256 scaled GEMV.
#[derive(Debug, Clone, PartialEq)]
pub struct A770OpenClQk256ScaledGemvResult {
    /// Output values in row order.
    pub output: Vec<f32>,
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
    /// Host-to-device bytes uploaded for this invocation.
    pub host_to_device_bytes: usize,
    /// Device-to-host bytes read for this invocation.
    pub device_to_host_bytes: usize,
    /// Number of OpenCL kernel invocations.
    pub kernel_invocations: usize,
}

/// One sampled row from the selected-device A770 QK256 debug kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct A770OpenClQk256DebugSample {
    /// Output row index within the projection matrix.
    pub output_index: usize,
    /// Integer dot product before activation-sum correction.
    pub int_dot: i32,
    /// Sum of the prequantized I8_S activation row as seen by the kernel.
    pub activation_sum: i32,
    /// `int_dot - activation_sum` as seen by the kernel.
    pub adjusted_dot: i32,
    /// Raw `f32` bits for the activation scale as seen by the kernel.
    pub activation_scale_bits: u32,
    /// Raw `f32` bits for the weight scale as seen by the kernel.
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

/// Runtime result for the selected-device A770 QK256 debug kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct A770OpenClQk256ScaledGemvDebugResult {
    /// Sampled output rows in row order.
    pub samples: Vec<A770OpenClQk256DebugSample>,
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
    /// Host-to-device bytes uploaded for this invocation.
    pub host_to_device_bytes: usize,
    /// Device-to-host bytes read for this invocation.
    pub device_to_host_bytes: usize,
    /// Number of OpenCL kernel invocations.
    pub kernel_invocations: usize,
}

/// Compiled-program evidence for the selected-device A770 QK256 debug kernel.
///
/// This records OpenCL compiler/build artifacts only. It is not a production
/// dispatch path and does not prove BitNet inference or QK256 policy readiness.
#[derive(Debug, Clone, PartialEq)]
pub struct A770OpenClQk256CompilerBinaryEvidence {
    /// OpenCL platform index selected for compilation.
    pub platform_index: usize,
    /// OpenCL device index selected for compilation.
    pub device_index: usize,
    /// OpenCL platform name.
    pub platform_name: String,
    /// Selected OpenCL device name.
    pub runtime_device: String,
    /// Selected OpenCL device vendor.
    pub vendor: String,
    /// Selected OpenCL driver version.
    pub driver_version: String,
    /// OpenCL program build options reported by the driver.
    pub build_options: String,
    /// OpenCL program build log reported by the driver.
    pub build_log: String,
    /// OpenCL binary type reported for the selected device.
    pub binary_type: String,
    /// Kernel names reported by the compiled program.
    pub kernel_names: String,
    /// Number of devices attached to the compiled program.
    pub program_device_count: usize,
    /// Binary sizes reported by OpenCL.
    pub binary_sizes: Vec<usize>,
    /// Deterministic FNV-1a hashes of each OpenCL program binary.
    pub binary_fnv1a64: Vec<String>,
    /// Short hex previews of each OpenCL program binary.
    pub binary_prefix_hex: Vec<String>,
    /// Driver-returned OpenCL program binaries.
    pub binaries: Vec<Vec<u8>>,
    /// Number of source bytes compiled.
    pub source_bytes: usize,
    /// Deterministic FNV-1a hash of the OpenCL C source.
    pub source_fnv1a64: String,
    /// Whether the source contains the volatile strict-f32 replay expression.
    pub strict_f32_barrier_source_present: bool,
    /// Whether the driver exposed any non-empty compiled program binary.
    pub program_binary_captured: bool,
    /// Whether this receipt includes real disassembly text.
    pub disassembly_captured: bool,
    /// Compact classification for the captured compiler evidence.
    pub classification: String,
}

/// Run grouped QK256 I2_S x prequantized I8_S scaled GEMV on the selected A770.
pub fn run_a770_qk256_i8s_scaled_gemv(
    request: A770OpenClQk256ScaledGemv<'_>,
) -> Result<A770OpenClQk256ScaledGemvResult> {
    validate_request(&request)?;
    let selected = find_a770_device()?;
    let context = Context::from_device(&selected.device).map_err(gpu_err("create context"))?;
    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
            .map_err(gpu_err("create command queue"))?;
    let program =
        Program::create_and_build_from_source(&context, QK256_I2S_I8S_SCALED_GEMV_SRC, "")
            .map_err(gpu_err("build qk256_i2s_i8s_scaled_gemv program"))?;
    let kernel =
        Kernel::create(&program, "qk256_i2s_i8s_scaled_gemv").map_err(gpu_err("create kernel"))?;

    let mut output = vec![0.0f32; request.rows];
    let mut buf_q = unsafe {
        Buffer::<i8>::create(
            &context,
            CL_MEM_READ_ONLY,
            request.activations_i8.len(),
            std::ptr::null_mut(),
        )
        .map_err(gpu_err("create activation buffer"))?
    };
    let mut buf_qs = unsafe {
        Buffer::<u8>::create(
            &context,
            CL_MEM_READ_ONLY,
            request.rows * request.row_stride_bytes,
            std::ptr::null_mut(),
        )
        .map_err(gpu_err("create packed weight buffer"))?
    };
    let buf_out = unsafe {
        Buffer::<f32>::create(&context, CL_MEM_WRITE_ONLY, output.len(), std::ptr::null_mut())
            .map_err(gpu_err("create output buffer"))?
    };

    let weight_bytes = &request.packed_qk256[..request.rows * request.row_stride_bytes];
    unsafe {
        queue
            .enqueue_write_buffer(&mut buf_q, CL_BLOCKING, 0, request.activations_i8, &[])
            .map_err(gpu_err("write activation buffer"))?;
        queue
            .enqueue_write_buffer(&mut buf_qs, CL_BLOCKING, 0, weight_bytes, &[])
            .map_err(gpu_err("write packed weight buffer"))?;
    }

    let rows = request.rows as u32;
    let cols = request.cols as u32;
    let row_stride_bytes = request.row_stride_bytes as u32;
    let event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&buf_q.get())
            .set_arg(&buf_qs.get())
            .set_arg(&buf_out.get())
            .set_arg(&rows)
            .set_arg(&cols)
            .set_arg(&row_stride_bytes)
            .set_arg(&request.activation_sum)
            .set_arg(&request.activation_scale)
            .set_arg(&request.weight_scale)
            .set_global_work_sizes(&[request.rows])
            .enqueue_nd_range(&queue)
            .map_err(gpu_err("enqueue qk256_i2s_i8s_scaled_gemv kernel"))?
    };
    event.wait().map_err(gpu_err("wait for qk256_i2s_i8s_scaled_gemv kernel"))?;

    unsafe {
        queue
            .enqueue_read_buffer(&buf_out, CL_BLOCKING, 0, &mut output, &[])
            .map_err(gpu_err("read output buffer"))?;
    }

    Ok(A770OpenClQk256ScaledGemvResult {
        output,
        platform_index: selected.platform_index,
        device_index: selected.device_index,
        platform_name: selected.platform_name,
        runtime_device: selected.device_name,
        vendor: selected.vendor,
        driver_version: selected.driver_version,
        host_to_device_bytes: std::mem::size_of_val(request.activations_i8) + weight_bytes.len(),
        device_to_host_bytes: std::mem::size_of::<f32>() * request.rows,
        kernel_invocations: 1,
    })
}

/// Run a bounded diagnostic QK256 intermediate capture on the selected A770.
pub fn run_a770_qk256_i8s_scaled_gemv_debug(
    request: A770OpenClQk256ScaledGemvDebug<'_>,
) -> Result<A770OpenClQk256ScaledGemvDebugResult> {
    let gemv_request = A770OpenClQk256ScaledGemv {
        activations_i8: request.activations_i8,
        packed_qk256: request.packed_qk256,
        rows: request.rows,
        cols: request.cols,
        row_stride_bytes: request.row_stride_bytes,
        activation_sum: request.activation_sum,
        activation_scale: request.activation_scale,
        weight_scale: request.weight_scale,
    };
    validate_request(&gemv_request)?;
    if request.sample_limit == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "A770 QK256 OpenCL debug sample_limit must be non-zero".to_string(),
        }
        .into());
    }

    let sample_count = request.rows.min(request.sample_limit);
    let selected = find_a770_device()?;
    let context = Context::from_device(&selected.device).map_err(gpu_err("create context"))?;
    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
            .map_err(gpu_err("create command queue"))?;
    let program =
        Program::create_and_build_from_source(&context, QK256_I2S_I8S_SCALED_GEMV_DEBUG_SRC, "")
            .map_err(gpu_err("build qk256_i2s_i8s_scaled_gemv_debug program"))?;
    let kernel = Kernel::create(&program, "qk256_i2s_i8s_scaled_gemv_debug")
        .map_err(gpu_err("create debug kernel"))?;

    let mut int_values = vec![0i32; sample_count * 3];
    let mut bit_values = vec![0u32; sample_count * 8];
    let mut buf_q = unsafe {
        Buffer::<i8>::create(
            &context,
            CL_MEM_READ_ONLY,
            request.activations_i8.len(),
            std::ptr::null_mut(),
        )
        .map_err(gpu_err("create debug activation buffer"))?
    };
    let mut buf_qs = unsafe {
        Buffer::<u8>::create(
            &context,
            CL_MEM_READ_ONLY,
            request.rows * request.row_stride_bytes,
            std::ptr::null_mut(),
        )
        .map_err(gpu_err("create debug packed weight buffer"))?
    };
    let buf_int = unsafe {
        Buffer::<i32>::create(&context, CL_MEM_WRITE_ONLY, int_values.len(), std::ptr::null_mut())
            .map_err(gpu_err("create debug int buffer"))?
    };
    let buf_bits = unsafe {
        Buffer::<u32>::create(&context, CL_MEM_WRITE_ONLY, bit_values.len(), std::ptr::null_mut())
            .map_err(gpu_err("create debug bits buffer"))?
    };

    let weight_bytes = &request.packed_qk256[..request.rows * request.row_stride_bytes];
    unsafe {
        queue
            .enqueue_write_buffer(&mut buf_q, CL_BLOCKING, 0, request.activations_i8, &[])
            .map_err(gpu_err("write debug activation buffer"))?;
        queue
            .enqueue_write_buffer(&mut buf_qs, CL_BLOCKING, 0, weight_bytes, &[])
            .map_err(gpu_err("write debug packed weight buffer"))?;
    }

    let rows = request.rows as u32;
    let cols = request.cols as u32;
    let row_stride_bytes = request.row_stride_bytes as u32;
    let sample_limit = sample_count as u32;
    let event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&buf_q.get())
            .set_arg(&buf_qs.get())
            .set_arg(&buf_int.get())
            .set_arg(&buf_bits.get())
            .set_arg(&rows)
            .set_arg(&cols)
            .set_arg(&row_stride_bytes)
            .set_arg(&request.activation_sum)
            .set_arg(&request.activation_scale)
            .set_arg(&request.weight_scale)
            .set_arg(&sample_limit)
            .set_global_work_sizes(&[sample_count])
            .enqueue_nd_range(&queue)
            .map_err(gpu_err("enqueue qk256_i2s_i8s_scaled_gemv_debug kernel"))?
    };
    event.wait().map_err(gpu_err("wait for qk256_i2s_i8s_scaled_gemv_debug kernel"))?;

    unsafe {
        queue
            .enqueue_read_buffer(&buf_int, CL_BLOCKING, 0, &mut int_values, &[])
            .map_err(gpu_err("read debug int buffer"))?;
        queue
            .enqueue_read_buffer(&buf_bits, CL_BLOCKING, 0, &mut bit_values, &[])
            .map_err(gpu_err("read debug bits buffer"))?;
    }

    let samples = (0..sample_count)
        .map(|output_index| {
            let int_base = output_index * 3;
            let bit_base = output_index * 8;
            let output_bits = bit_values[bit_base + 3];
            let div_then_mul_bits = bit_values[bit_base + 4];
            let mul_then_div_bits = bit_values[bit_base + 5];
            let reciprocal_then_mul_bits = bit_values[bit_base + 6];
            let volatile_div_then_mul_bits = bit_values[bit_base + 7];
            A770OpenClQk256DebugSample {
                output_index,
                int_dot: int_values[int_base],
                activation_sum: int_values[int_base + 1],
                adjusted_dot: int_values[int_base + 2],
                activation_scale_bits: bit_values[bit_base],
                weight_scale_bits: bit_values[bit_base + 1],
                adjusted_f32_bits: bit_values[bit_base + 2],
                output_bits,
                output: f32::from_bits(output_bits),
                div_then_mul_bits,
                div_then_mul: f32::from_bits(div_then_mul_bits),
                mul_then_div_bits,
                mul_then_div: f32::from_bits(mul_then_div_bits),
                reciprocal_then_mul_bits,
                reciprocal_then_mul: f32::from_bits(reciprocal_then_mul_bits),
                volatile_div_then_mul_bits,
                volatile_div_then_mul: f32::from_bits(volatile_div_then_mul_bits),
            }
        })
        .collect();

    Ok(A770OpenClQk256ScaledGemvDebugResult {
        samples,
        platform_index: selected.platform_index,
        device_index: selected.device_index,
        platform_name: selected.platform_name,
        runtime_device: selected.device_name,
        vendor: selected.vendor,
        driver_version: selected.driver_version,
        host_to_device_bytes: std::mem::size_of_val(request.activations_i8) + weight_bytes.len(),
        device_to_host_bytes: std::mem::size_of_val(int_values.as_slice())
            + std::mem::size_of_val(bit_values.as_slice()),
        kernel_invocations: 1,
    })
}

/// Capture OpenCL compiler/build evidence for the selected-device A770 QK256
/// debug kernel.
///
/// The OpenCL program binary is driver-specific and is recorded through size,
/// stable hash, and prefix only. This does not disassemble the binary and must
/// not be used as a production QK256 policy change by itself.
pub fn capture_a770_qk256_debug_compiler_binary_evidence()
-> Result<A770OpenClQk256CompilerBinaryEvidence> {
    let selected = find_a770_device()?;
    let context = Context::from_device(&selected.device).map_err(gpu_err("create context"))?;
    let program =
        Program::create_and_build_from_source(&context, QK256_I2S_I8S_SCALED_GEMV_DEBUG_SRC, "")
            .map_err(gpu_err("build qk256_i2s_i8s_scaled_gemv_debug evidence program"))?;

    let device_id = selected.device.id();
    let build_options = program.get_build_options(device_id).unwrap_or_default();
    let build_log = program.get_build_log(device_id).unwrap_or_default();
    let binary_type = program
        .get_build_binary_type(device_id)
        .map(binary_type_name)
        .unwrap_or("unknown")
        .to_owned();
    let kernel_names = program.kernel_names().to_owned();
    let program_device_count = program.get_num_devices().unwrap_or(0) as usize;
    let binary_sizes =
        program.get_binary_sizes().unwrap_or_default().into_iter().collect::<Vec<_>>();
    let binaries = program.get_binaries().unwrap_or_default();
    let binary_fnv1a64 = binaries.iter().map(|binary| fnv1a64_hex(binary)).collect::<Vec<_>>();
    let binary_prefix_hex =
        binaries.iter().map(|binary| hex_prefix(binary, 32)).collect::<Vec<_>>();
    let program_binary_captured = binaries.iter().any(|binary| !binary.is_empty());
    let strict_f32_barrier_source_present = QK256_I2S_I8S_SCALED_GEMV_DEBUG_SRC
        .contains("volatile_div_then_mul")
        && QK256_I2S_I8S_SCALED_GEMV_DEBUG_SRC.contains("volatile float");
    let disassembly_captured = false;
    let classification = compiler_binary_evidence_classification(
        program_binary_captured,
        disassembly_captured,
        strict_f32_barrier_source_present,
    )
    .to_owned();

    Ok(A770OpenClQk256CompilerBinaryEvidence {
        platform_index: selected.platform_index,
        device_index: selected.device_index,
        platform_name: selected.platform_name,
        runtime_device: selected.device_name,
        vendor: selected.vendor,
        driver_version: selected.driver_version,
        build_options,
        build_log,
        binary_type,
        kernel_names,
        program_device_count,
        binary_sizes,
        binary_fnv1a64,
        binary_prefix_hex,
        binaries,
        source_bytes: QK256_I2S_I8S_SCALED_GEMV_DEBUG_SRC.len(),
        source_fnv1a64: fnv1a64_hex(QK256_I2S_I8S_SCALED_GEMV_DEBUG_SRC.as_bytes()),
        strict_f32_barrier_source_present,
        program_binary_captured,
        disassembly_captured,
        classification,
    })
}

fn validate_request(request: &A770OpenClQk256ScaledGemv<'_>) -> Result<()> {
    if request.rows == 0 || request.cols == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "A770 QK256 OpenCL request dimensions must be non-zero".to_string(),
        }
        .into());
    }
    if !request.cols.is_multiple_of(256) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL cols must be a multiple of 256, got {}",
                request.cols
            ),
        }
        .into());
    }
    let expected_stride = (request.cols / 256) * 64;
    if request.row_stride_bytes != expected_stride {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL row_stride_bytes {} != expected {} for cols={}",
                request.row_stride_bytes, expected_stride, request.cols
            ),
        }
        .into());
    }
    if request.activations_i8.len() < request.cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL activation length {} < cols {}",
                request.activations_i8.len(),
                request.cols
            ),
        }
        .into());
    }
    let expected_bytes = request.rows * request.row_stride_bytes;
    if request.packed_qk256.len() < expected_bytes {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL packed bytes {} < expected {}",
                request.packed_qk256.len(),
                expected_bytes
            ),
        }
        .into());
    }
    if !request.activation_scale.is_finite() || request.activation_scale == 0.0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL activation scale must be finite and non-zero: {}",
                request.activation_scale
            ),
        }
        .into());
    }
    if !request.weight_scale.is_finite() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "A770 QK256 OpenCL weight scale is not finite: {}",
                request.weight_scale
            ),
        }
        .into());
    }
    Ok(())
}

#[derive(Debug)]
struct SelectedA770Device {
    platform_index: usize,
    device_index: usize,
    platform_name: String,
    device_name: String,
    vendor: String,
    driver_version: String,
    device: Device,
}

fn find_a770_device() -> Result<SelectedA770Device> {
    let platforms = get_platforms().map_err(gpu_err("enumerate OpenCL platforms"))?;
    for (platform_index, platform) in platforms.iter().enumerate() {
        let platform_name = platform.name().unwrap_or_else(|_| "unknown".to_owned());
        let devices = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap_or_default();
        for (device_index, device_id) in devices.iter().enumerate() {
            let device = Device::new(*device_id);
            let device_name = device.name().unwrap_or_default();
            let vendor = device.vendor().unwrap_or_default();
            if !is_intel_vendor(&vendor) || !is_a770_device_name(&device_name) {
                continue;
            }
            let driver_version = device.driver_version().unwrap_or_default();
            return Ok(SelectedA770Device {
                platform_index,
                device_index,
                platform_name,
                device_name,
                vendor,
                driver_version,
                device,
            });
        }
    }
    Err(KernelError::GpuError {
        reason: "Intel Arc A770 OpenCL device was not visible".to_string(),
    }
    .into())
}

fn is_intel_vendor(value: &str) -> bool {
    value.to_ascii_lowercase().contains("intel")
}

fn is_a770_device_name(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    (lower.contains("arc") && lower.contains("a770")) || lower.contains("56a0")
}

fn gpu_err<E: Display>(context: &'static str) -> impl FnOnce(E) -> KernelError {
    move |err| KernelError::GpuError { reason: format!("A770 OpenCL {context}: {err}") }
}

fn binary_type_name(binary_type: u32) -> &'static str {
    match binary_type {
        CL_PROGRAM_BINARY_TYPE_NONE => "none",
        CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT => "compiled_object",
        CL_PROGRAM_BINARY_TYPE_LIBRARY => "library",
        CL_PROGRAM_BINARY_TYPE_EXECUTABLE => "executable",
        _ => "unknown",
    }
}

fn compiler_binary_evidence_classification(
    program_binary_captured: bool,
    disassembly_captured: bool,
    strict_f32_barrier_source_present: bool,
) -> &'static str {
    match (program_binary_captured, disassembly_captured, strict_f32_barrier_source_present) {
        (false, _, _) => "a770_qk256_opencl_compiler_binary_evidence_missing_program_binary",
        (true, false, false) => {
            "a770_qk256_opencl_compiler_binary_evidence_missing_strict_f32_source_context"
        }
        (true, false, true) => {
            "a770_qk256_opencl_compiler_binary_evidence_program_binary_captured_disassembly_missing"
        }
        (true, true, true) => {
            "a770_qk256_opencl_compiler_binary_evidence_program_binary_and_disassembly_captured"
        }
        (true, true, false) => {
            "a770_qk256_opencl_compiler_binary_evidence_disassembly_captured_missing_strict_f32_source_context"
        }
    }
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn hex_prefix(bytes: &[u8], limit: usize) -> String {
    let mut out = String::new();
    for byte in bytes.iter().take(limit) {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(
        activations_i8: &'a [i8],
        packed_qk256: &'a [u8],
    ) -> A770OpenClQk256ScaledGemv<'a> {
        A770OpenClQk256ScaledGemv {
            activations_i8,
            packed_qk256,
            rows: 1,
            cols: 256,
            row_stride_bytes: 64,
            activation_sum: 0,
            activation_scale: 1.0,
            weight_scale: 0.25,
        }
    }

    #[test]
    fn validation_rejects_non_qk256_cols() {
        let q = vec![0; 128];
        let weights = vec![0; 64];
        let mut req = request(&q, &weights);
        req.cols = 128;
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn validation_rejects_bad_stride() {
        let q = vec![0; 256];
        let weights = vec![0; 64];
        let mut req = request(&q, &weights);
        req.row_stride_bytes = 32;
        assert!(validate_request(&req).is_err());
    }

    #[test]
    fn validation_accepts_minimal_qk256_contract() -> Result<()> {
        let q = vec![0; 256];
        let weights = vec![0; 64];
        validate_request(&request(&q, &weights))
    }

    #[test]
    fn debug_validation_rejects_zero_sample_limit() {
        let q = vec![0; 256];
        let weights = vec![0; 64];
        let req = A770OpenClQk256ScaledGemvDebug {
            activations_i8: &q,
            packed_qk256: &weights,
            rows: 1,
            cols: 256,
            row_stride_bytes: 64,
            activation_sum: 0,
            activation_scale: 1.0,
            weight_scale: 0.25,
            sample_limit: 0,
        };
        assert!(run_a770_qk256_i8s_scaled_gemv_debug(req).is_err());
    }

    #[test]
    fn compiler_binary_evidence_classifies_binary_without_disassembly() {
        assert_eq!(
            compiler_binary_evidence_classification(true, false, true),
            "a770_qk256_opencl_compiler_binary_evidence_program_binary_captured_disassembly_missing"
        );
    }

    #[test]
    fn compiler_binary_evidence_classifies_missing_binary() {
        assert_eq!(
            compiler_binary_evidence_classification(false, false, true),
            "a770_qk256_opencl_compiler_binary_evidence_missing_program_binary"
        );
    }

    #[test]
    fn deterministic_fnv_and_hex_prefix_are_stable() {
        assert_eq!(fnv1a64_hex(b""), "cbf29ce484222325");
        assert_eq!(hex_prefix(&[0xab, 0xcd, 0xef], 2), "abcd");
    }
}
