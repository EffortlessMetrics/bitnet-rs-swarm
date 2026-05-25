use std::{env, error::Error, io, path::PathBuf};

use bitnet_kernels::a770_opencl_runtime::{
    A770OpenClQk256ProductionReplay, A770OpenClQk256ProductionReplayResult,
    A770OpenClQk256ProductionReplaySample, run_a770_qk256_i8s_scaled_gemv_production_replay,
};

const RECEIPT_ENV: &str = "BITNET_A770_OPENCL_PRODUCTION_REPLAY_RECEIPT";
const DEFAULT_RECEIPT: &str = "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-replay-instrumentation.json";
const ROWS: usize = 2;
const COLS: usize = 256;
const ROW_STRIDE_BYTES: usize = 64;
const SAMPLE_LIMIT: usize = 2;

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let fixture = fixture();
    let replay =
        run_a770_qk256_i8s_scaled_gemv_production_replay(A770OpenClQk256ProductionReplay {
            activations_i8: &fixture.activations_i8,
            packed_qk256: &fixture.packed_qk256,
            rows: ROWS,
            cols: COLS,
            row_stride_bytes: ROW_STRIDE_BYTES,
            activation_sum: fixture.activation_sum,
            activation_scale: fixture.activation_scale,
            weight_scale: fixture.weight_scale,
            sample_limit: SAMPLE_LIMIT,
        })?;
    let receipt = receipt_to_json(&fixture, &replay)?;
    if let Some(parent) = args.receipt.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.receipt, &receipt)?;
    println!("{receipt}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    receipt: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut receipt = env::var_os(RECEIPT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_RECEIPT));
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--receipt" => {
                    receipt = PathBuf::from(
                        args.next()
                            .ok_or_else(|| io_error("--receipt requires a path argument"))?,
                    );
                }
                "--help" | "-h" => {
                    println!(
                        "Usage: a770-opencl-production-replay-instrumentation [--receipt <path>]\n\nRuns diagnostic production replay instrumentation for selected Intel Arc A770 OpenCL QK256 scaled GEMV."
                    );
                    std::process::exit(0);
                }
                other => return Err(io_error(format!("unknown argument {other:?}"))),
            }
        }
        Ok(Self { receipt })
    }
}

#[derive(Debug, Clone)]
struct Fixture {
    activations_i8: Vec<i8>,
    packed_qk256: Vec<u8>,
    activation_sum: i32,
    activation_scale: f32,
    weight_scale: f32,
    host_rows: Vec<HostRow>,
}

#[derive(Debug, Clone)]
struct HostRow {
    output_index: usize,
    int_dot: i32,
    adjusted_dot: i32,
    div_then_mul_bits: u32,
    reciprocal_path_final_bits: u32,
}

fn fixture() -> Fixture {
    let activations_i8 =
        (0..COLS).map(|index| (((index * 37 + 11) % 127) as i16 - 63) as i8).collect::<Vec<_>>();
    let activation_sum = activations_i8.iter().map(|value| i32::from(*value)).sum::<i32>();
    let activation_scale = f32::from_bits(0x4214_0000);
    let weight_scale = f32::from_bits(0x3bfd_70a4);
    let mut packed_qk256 = vec![0u8; ROWS * ROW_STRIDE_BYTES];

    for row in 0..ROWS {
        for col in 0..COLS {
            let code = ((row * 5 + col * 3 + 1) & 0x03) as u8;
            write_qk256_code(&mut packed_qk256, row, col, code);
        }
    }

    let host_rows = (0..ROWS)
        .map(|row| {
            let int_dot = host_int_dot(&activations_i8, &packed_qk256, row);
            let adjusted_dot = int_dot - activation_sum;
            let adjusted_f32 = adjusted_dot as f32;
            let div_then_mul = (adjusted_f32 / activation_scale) * weight_scale;
            let reciprocal_path_final = adjusted_f32 * (1.0f32 / activation_scale) * weight_scale;
            HostRow {
                output_index: row,
                int_dot,
                adjusted_dot,
                div_then_mul_bits: div_then_mul.to_bits(),
                reciprocal_path_final_bits: reciprocal_path_final.to_bits(),
            }
        })
        .collect::<Vec<_>>();

    Fixture {
        activations_i8,
        packed_qk256,
        activation_sum,
        activation_scale,
        weight_scale,
        host_rows,
    }
}

fn write_qk256_code(packed_qk256: &mut [u8], row: usize, col: usize, code: u8) {
    let offset = col % 256;
    let chunk = offset / 128;
    let lane = (offset - chunk * 128) / 32;
    let gp = offset & 31;
    let byte_index = row * ROW_STRIDE_BYTES + chunk * 32 + gp;
    packed_qk256[byte_index] |= (code & 0x03) << (6 - lane * 2);
}

fn read_qk256_code(packed_qk256: &[u8], row: usize, col: usize) -> u8 {
    let offset = col % 256;
    let chunk = offset / 128;
    let lane = (offset - chunk * 128) / 32;
    let gp = offset & 31;
    let byte_index = row * ROW_STRIDE_BYTES + chunk * 32 + gp;
    (packed_qk256[byte_index] >> (6 - lane * 2)) & 0x03
}

fn host_int_dot(activations_i8: &[i8], packed_qk256: &[u8], row: usize) -> i32 {
    (0..COLS)
        .map(|col| {
            i32::from(read_qk256_code(packed_qk256, row, col)) * i32::from(activations_i8[col])
        })
        .sum()
}

fn receipt_to_json(
    fixture: &Fixture,
    replay: &A770OpenClQk256ProductionReplayResult,
) -> Result<String, Box<dyn Error>> {
    let all_store_match_replay =
        replay.samples.iter().all(|sample| sample.output_store_matches_replay_output);
    let all_store_match_final =
        replay.samples.iter().all(|sample| sample.output_store_matches_final_scaled_value);
    let classification = if replay.samples.is_empty() {
        "a770_qk256_production_replay_instrumentation_missing_context"
    } else if all_store_match_replay && all_store_match_final {
        "a770_qk256_production_replay_instrumentation_output_store_matches_replay"
    } else {
        "a770_qk256_production_replay_instrumentation_output_store_differs_from_replay"
    };

    Ok(format!(
        concat!(
            "{{\n",
            "  \"campaign\": \"intel-a770\",\n",
            "  \"work_item\": \"A770-062\",\n",
            "  \"proof_family\": \"a770_opencl_qk256_production_replay_instrumentation\",\n",
            "  \"proof_stage\": \"diagnostic_production_replay_instrumentation_captured\",\n",
            "  \"requested_backend\": \"intel-arc-a770\",\n",
            "  \"selected_backend\": \"intel-arc-a770-opencl\",\n",
            "  \"runtime_api\": \"opencl\",\n",
            "  \"runtime_device\": \"{}\",\n",
            "  \"platform_index\": {},\n",
            "  \"device_index\": {},\n",
            "  \"platform_name\": \"{}\",\n",
            "  \"vendor\": \"{}\",\n",
            "  \"driver_version\": \"{}\",\n",
            "  \"kernel_name\": \"qk256_i2s_i8s_scaled_gemv\",\n",
            "  \"replay_kernel_name\": \"qk256_i2s_i8s_scaled_gemv_production_replay\",\n",
            "  \"classification\": \"{}\",\n",
            "  \"source_receipts\": {{\n",
            "    \"production_lowered_operation_sequence\": \"ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-lowered-operation-sequence.json\",\n",
            "    \"production_disassembly_evidence\": \"ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-kernel-disassembly-evidence.json\",\n",
            "    \"focused_qk256_replay\": \"ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json\"\n",
            "  }},\n",
            "  \"fixture\": {{\n",
            "    \"fixture_source\": \"diagnostic_minimal_qk256_fixture\",\n",
            "    \"focused_first_mismatch_operands_available\": false,\n",
            "    \"focused_case_id\": \"a770_summary_seed770024_keywords_014\",\n",
            "    \"rows\": {},\n",
            "    \"cols\": {},\n",
            "    \"row_stride_bytes\": {},\n",
            "    \"sample_limit\": {},\n",
            "    \"activation_sum\": {},\n",
            "    \"activation_scale_bits\": {},\n",
            "    \"weight_scale_bits\": {},\n",
            "    \"activation_bytes\": {},\n",
            "    \"packed_weight_bytes\": {}\n",
            "  }},\n",
            "  \"captured_intermediates\": {{\n",
            "    \"adjusted_dot\": true,\n",
            "    \"activation_scale\": true,\n",
            "    \"weight_scale\": true,\n",
            "    \"reciprocal_path_intermediate_bits\": true,\n",
            "    \"final_scaled_value_bits\": true,\n",
            "    \"output_store_bits\": true\n",
            "  }},\n",
            "  \"samples\": {},\n",
            "  \"all_output_store_matches_replay_output\": {},\n",
            "  \"all_output_store_matches_final_scaled_value\": {},\n",
            "  \"production_replay_instrumentation_captured\": true,\n",
            "  \"host_to_device_bytes\": {},\n",
            "  \"device_to_host_bytes\": {},\n",
            "  \"kernel_invocations\": {},\n",
            "  \"fallback_used\": false,\n",
            "  \"cpu_fallback_allowed\": false,\n",
            "  \"bitnet_inference\": false,\n",
            "  \"qk256_decode\": false,\n",
            "  \"production_qk256_policy_change\": false,\n",
            "  \"claim_allowed\": false,\n",
            "  \"diagnostic_only\": true,\n",
            "  \"performance_claim\": false,\n",
            "  \"full_residency_claim\": false,\n",
            "  \"next_diagnostic\": \"capture focused production operands for the first mismatch before any production QK256 policy change\",\n",
            "  \"must_not_claim\": [\n",
            "    \"CPU/A770 answer parity is proven\",\n",
            "    \"Reference parity is proven\",\n",
            "    \"Strict A770 answer readiness is proven\",\n",
            "    \"Broad A770 answer quality is proven\",\n",
            "    \"Official BitNet QK256 production semantics are proven\",\n",
            "    \"Production QK256 dispatch policy changed\",\n",
            "    \"BitNet inference works on A770\",\n",
            "    \"A770 trusted partial acceleration is claim-grade\",\n",
            "    \"Full A770 residency is proven\",\n",
            "    \"A770 performance speedup is proven\"\n",
            "  ]\n",
            "}}\n"
        ),
        json_escape(&replay.runtime_device),
        replay.platform_index,
        replay.device_index,
        json_escape(&replay.platform_name),
        json_escape(&replay.vendor),
        json_escape(&replay.driver_version),
        classification,
        ROWS,
        COLS,
        ROW_STRIDE_BYTES,
        SAMPLE_LIMIT,
        fixture.activation_sum,
        fixture.activation_scale.to_bits(),
        fixture.weight_scale.to_bits(),
        fixture.activations_i8.len(),
        fixture.packed_qk256.len(),
        samples_json(&replay.samples, &fixture.host_rows)?,
        all_store_match_replay,
        all_store_match_final,
        replay.host_to_device_bytes,
        replay.device_to_host_bytes,
        replay.kernel_invocations
    ))
}

fn samples_json(
    samples: &[A770OpenClQk256ProductionReplaySample],
    host_rows: &[HostRow],
) -> Result<String, Box<dyn Error>> {
    let rows = samples
        .iter()
        .map(|sample| {
            let host = host_rows
                .iter()
                .find(|row| row.output_index == sample.output_index)
                .ok_or_else(|| {
                    io_error(format!(
                        "missing host row for sample output_index {}",
                        sample.output_index
                    ))
                })?;
            let production_matches_host_div_then_mul =
                sample.production_output_bits == host.div_then_mul_bits;
            let production_matches_host_reciprocal_final =
                sample.production_output_bits == host.reciprocal_path_final_bits;
            Ok(format!(
                concat!(
                    "{{",
                    "\"output_index\":{},",
                    "\"int_dot\":{},",
                    "\"host_int_dot\":{},",
                    "\"activation_sum\":{},",
                    "\"adjusted_dot\":{},",
                    "\"host_adjusted_dot\":{},",
                    "\"activation_scale_bits\":{},",
                    "\"weight_scale_bits\":{},",
                    "\"adjusted_f32_bits\":{},",
                    "\"reciprocal_activation_scale_bits\":{},",
                    "\"adjusted_mul_reciprocal_bits\":{},",
                    "\"final_scaled_value_bits\":{},",
                    "\"div_then_mul_bits\":{},",
                    "\"weight_over_activation_bits\":{},",
                    "\"reciprocal_then_mul_bits\":{},",
                    "\"replay_output_bits\":{},",
                    "\"production_output_bits\":{},",
                    "\"host_div_then_mul_bits\":{},",
                    "\"host_reciprocal_path_final_bits\":{},",
                    "\"output_store_matches_replay_output\":{},",
                    "\"output_store_matches_final_scaled_value\":{},",
                    "\"production_matches_host_div_then_mul\":{},",
                    "\"production_matches_host_reciprocal_final\":{}",
                    "}}"
                ),
                sample.output_index,
                sample.int_dot,
                host.int_dot,
                sample.activation_sum,
                sample.adjusted_dot,
                host.adjusted_dot,
                sample.activation_scale_bits,
                sample.weight_scale_bits,
                sample.adjusted_f32_bits,
                sample.reciprocal_activation_scale_bits,
                sample.adjusted_mul_reciprocal_bits,
                sample.final_scaled_value_bits,
                sample.div_then_mul_bits,
                sample.weight_over_activation_bits,
                sample.reciprocal_then_mul_bits,
                sample.replay_output_bits,
                sample.production_output_bits,
                host.div_then_mul_bits,
                host.reciprocal_path_final_bits,
                sample.output_store_matches_replay_output,
                sample.output_store_matches_final_scaled_value,
                production_matches_host_div_then_mul,
                production_matches_host_reciprocal_final
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?
        .join(",\n    ");
    Ok(format!("[\n    {rows}\n  ]"))
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn io_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}
