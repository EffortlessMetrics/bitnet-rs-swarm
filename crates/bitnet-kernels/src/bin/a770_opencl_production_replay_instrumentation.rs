#![recursion_limit = "256"]

use std::{
    env,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
};

use bitnet_kernels::a770_opencl_runtime::{
    A770OpenClQk256ProductionReplay, A770OpenClQk256ProductionReplayResult,
    A770OpenClQk256ProductionReplaySample, run_a770_qk256_i8s_scaled_gemv_production_replay,
};
use serde_json::{Value, json};

const RECEIPT_ENV: &str = "BITNET_A770_OPENCL_PRODUCTION_REPLAY_RECEIPT";
const DEFAULT_RECEIPT: &str = "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-replay-instrumentation.json";
const ROWS: usize = 2;
const COLS: usize = 256;
const ROW_STRIDE_BYTES: usize = 64;
const SAMPLE_LIMIT: usize = 2;
const PROJECTION_REPLAY_SAMPLE_LIMIT: usize = 4;
const PROJECTION_REPLAY_KERNEL_NAME: &str = "qk256_i2s_i8s_scaled_gemv_production_replay";

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    if args.projection_source.is_some() {
        let (manifest, receipt) = projection_level_qkv_replay_to_json(&args)?;
        let manifest_path =
            args.manifest.as_ref().ok_or_else(|| io_error("manifest path missing after parse"))?;
        if let Some(parent) = manifest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(manifest_path, &manifest)?;
        if let Some(parent) = args.receipt.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&args.receipt, &receipt)?;
        println!("{receipt}");
        return Ok(());
    }
    if args.manifest.is_some() {
        let (manifest, receipt) = multi_case_focused_replay_to_json(&args)?;
        let manifest_path =
            args.manifest.as_ref().ok_or_else(|| io_error("manifest path missing after parse"))?;
        if let Some(parent) = manifest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(manifest_path, &manifest)?;
        if let Some(parent) = args.receipt.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&args.receipt, &receipt)?;
        println!("{receipt}");
        return Ok(());
    }
    if !args.focused_sources.is_empty() {
        let receipt = focused_receipt_to_json(&args)?;
        if let Some(parent) = args.receipt.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&args.receipt, &receipt)?;
        println!("{receipt}");
        return Ok(());
    }

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
    manifest: Option<PathBuf>,
    focused_sources: Vec<PathBuf>,
    case_id: Option<String>,
    first_mismatch_index: Option<usize>,
    work_item: Option<String>,
    projection_source: Option<PathBuf>,
    projection_layer: Option<i64>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut receipt = env::var_os(RECEIPT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_RECEIPT));
        let mut manifest = None;
        let mut focused_sources = Vec::new();
        let mut case_id = None;
        let mut first_mismatch_index = None;
        let mut work_item = None;
        let mut projection_source = None;
        let mut projection_layer = None;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--receipt" => {
                    receipt = PathBuf::from(
                        args.next()
                            .ok_or_else(|| io_error("--receipt requires a path argument"))?,
                    );
                }
                "--manifest" => {
                    manifest = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| io_error("--manifest requires a path argument"))?,
                    ));
                }
                "--focused-source" => {
                    focused_sources
                        .push(PathBuf::from(args.next().ok_or_else(|| {
                            io_error("--focused-source requires a path argument")
                        })?));
                }
                "--case-id" => {
                    case_id = Some(
                        args.next().ok_or_else(|| io_error("--case-id requires an argument"))?,
                    );
                }
                "--first-mismatch-index" => {
                    let value = args
                        .next()
                        .ok_or_else(|| io_error("--first-mismatch-index requires an argument"))?;
                    first_mismatch_index = Some(value.parse::<usize>().map_err(|err| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("invalid --first-mismatch-index {value:?}: {err}"),
                        )
                    })?);
                }
                "--work-item" => {
                    work_item = Some(
                        args.next().ok_or_else(|| io_error("--work-item requires an argument"))?,
                    );
                }
                "--projection-source" => {
                    projection_source = Some(PathBuf::from(args.next().ok_or_else(|| {
                        io_error("--projection-source requires a path argument")
                    })?));
                }
                "--projection-layer" => {
                    let value = args
                        .next()
                        .ok_or_else(|| io_error("--projection-layer requires an argument"))?;
                    projection_layer = Some(value.parse::<i64>().map_err(|err| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("invalid --projection-layer {value:?}: {err}"),
                        )
                    })?);
                }
                "--help" | "-h" => {
                    println!(
                        "Usage: a770-opencl-production-replay-instrumentation [--receipt <path>] [--focused-source <path> --case-id <id> --first-mismatch-index <n>] [--focused-source <path>... --manifest <path> [--case-id <id>] [--first-mismatch-index <n>] [--work-item <id>]] [--projection-source <path> --manifest <path> --receipt <path> --case-id <id> --first-mismatch-index <n> [--projection-layer <n>] [--work-item <id>]]\n\nRuns diagnostic production replay instrumentation for selected Intel Arc A770 OpenCL QK256 scaled GEMV, classifies a focused first-mismatch operand receipt, builds an A770 multi-target focused replay packet, or ledgers the first projection-level Q/K/V replay boundary from committed focused row evidence."
                    );
                    std::process::exit(0);
                }
                other => return Err(io_error(format!("unknown argument {other:?}"))),
            }
        }
        if projection_source.is_some() {
            if manifest.is_none() {
                return Err(io_error("--projection-source requires --manifest"));
            }
            if !focused_sources.is_empty() {
                return Err(io_error(
                    "--projection-source cannot be combined with --focused-source",
                ));
            }
        } else if manifest.is_some() {
            if focused_sources.is_empty() {
                return Err(io_error("--manifest requires --focused-source"));
            }
        } else if !focused_sources.is_empty() {
            if focused_sources.len() > 1 {
                return Err(io_error("multiple --focused-source values require --manifest"));
            }
            if case_id.is_none() {
                return Err(io_error("--focused-source requires --case-id"));
            }
            if first_mismatch_index.is_none() {
                return Err(io_error("--focused-source requires --first-mismatch-index"));
            }
        }

        Ok(Self {
            receipt,
            manifest,
            focused_sources,
            case_id,
            first_mismatch_index,
            work_item,
            projection_source,
            projection_layer,
        })
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

fn focused_receipt_to_json(args: &Args) -> Result<String, Box<dyn Error>> {
    let receipt_kind = FocusedReceiptKind::from_receipt_path(&args.receipt);
    let focused_source = args
        .focused_sources
        .first()
        .ok_or_else(|| io_error("focused receipt requested without --focused-source"))?;
    let case_id = args
        .case_id
        .as_deref()
        .ok_or_else(|| io_error("focused receipt requested without --case-id"))?;
    let first_mismatch_index = args
        .first_mismatch_index
        .ok_or_else(|| io_error("focused receipt requested without --first-mismatch-index"))?;
    let source_json = fs::read_to_string(focused_source)?;
    let source: Value = serde_json::from_str(&source_json)?;
    let context = focused_context(&source, case_id, first_mismatch_index);
    let replay_outcome = focused_replay_outcome(&source, case_id, first_mismatch_index);
    let classification = match receipt_kind {
        FocusedReceiptKind::RawOperandReplay => {
            focused_classification(&context, replay_outcome.as_ref())
        }
        FocusedReceiptKind::HostPolicyExpressionSplit => {
            focused_host_policy_expression_split_classification(&context, replay_outcome.as_ref())
        }
        FocusedReceiptKind::HostSummaryPolicySemanticFix => {
            focused_host_summary_policy_semantic_fix_classification(
                &context,
                replay_outcome.as_ref(),
            )
        }
    };
    let focused_policy_bits_for_receipt = match receipt_kind {
        FocusedReceiptKind::HostSummaryPolicySemanticFix => {
            context.host_summary_policy_semantic_fix_bits.or(context.focused_policy_bits)
        }
        _ => context.focused_policy_bits,
    };
    let focused_summary_divergence_available =
        context.focused_device_output_bits.is_some() && focused_policy_bits_for_receipt.is_some();
    let focused_summary_device_vs_policy_bits_match =
        match (context.focused_device_output_bits, focused_policy_bits_for_receipt) {
            (Some(device), Some(policy)) => Some(device == policy),
            _ => None,
        };
    let production_replay_executed =
        matches!(&replay_outcome, Some(FocusedReplayOutcome::Executed { .. }));
    let production_replay_error = match &replay_outcome {
        Some(FocusedReplayOutcome::Failed { error }) => Some(error.as_str()),
        _ => None,
    };
    let production_replay_skipped_reason = match &replay_outcome {
        Some(FocusedReplayOutcome::Executed { .. }) => None,
        Some(FocusedReplayOutcome::Failed { .. }) => Some("focused_raw_operands_replay_failed"),
        None => Some(context.production_replay_skipped_reason),
    };
    let focused_raw_operand_summary = focused_raw_operand_summary_json(replay_outcome.as_ref());
    let focused_production_replay_summary =
        focused_production_replay_summary_json(replay_outcome.as_ref(), &context);
    let focused_production_replay_samples =
        focused_production_replay_samples_json(replay_outcome.as_ref());
    let next_diagnostic = match receipt_kind {
        FocusedReceiptKind::RawOperandReplay => {
            focused_next_diagnostic(&context, replay_outcome.as_ref())
        }
        FocusedReceiptKind::HostPolicyExpressionSplit => {
            focused_host_policy_expression_split_next_diagnostic(&context, replay_outcome.as_ref())
        }
        FocusedReceiptKind::HostSummaryPolicySemanticFix => {
            focused_host_summary_policy_semantic_fix_next_diagnostic(
                &context,
                replay_outcome.as_ref(),
            )
        }
    };
    let (host_to_device_bytes, device_to_host_bytes, kernel_invocations) = match &replay_outcome {
        Some(FocusedReplayOutcome::Executed { replay, .. }) => {
            (replay.host_to_device_bytes, replay.device_to_host_bytes, replay.kernel_invocations)
        }
        _ => (0, 0, 0),
    };

    let mut receipt = json!({
        "campaign": "intel-a770",
        "work_item": receipt_kind.work_item(),
        "proof_family": receipt_kind.proof_family(),
        "proof_stage": receipt_kind.proof_stage(),
        "requested_backend": "intel-arc-a770",
        "selected_backend": "intel-arc-a770-opencl",
        "runtime_api": "opencl",
        "runtime_device": context.runtime_device,
        "platform_index": context.platform_index,
        "device_index": context.device_index,
        "platform_name": context.platform_name,
        "vendor": context.vendor,
        "driver_version": context.driver_version,
        "kernel_name": "qk256_i2s_i8s_scaled_gemv",
        "replay_kernel_name": "qk256_i2s_i8s_scaled_gemv_production_replay",
        "classification": classification,
        "source_receipts": {
            "focused_source": path_json_value(focused_source),
            "production_replay_instrumentation": "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-replay-instrumentation.json",
            "production_lowered_operation_sequence": "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-lowered-operation-sequence.json",
            "production_disassembly_evidence": "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-kernel-disassembly-evidence.json"
        },
        "focused_case": {
            "case_id": case_id,
            "requested_first_mismatch_index": first_mismatch_index,
            "case_found": context.case_found,
            "summary_first_divergence_matches_request": context.summary_first_divergence_matches_request,
            "qkv_projection_dispatch_replay_context_available": context.qkv_context_available,
            "target_layer_idx": context.target_layer_idx,
            "projection": context.projection,
            "input_rows": context.input_rows,
            "output_rows": context.output_rows,
            "cols": context.cols,
            "row_stride_bytes": context.row_stride_bytes,
            "input_row_index": context.input_row_index,
            "sample_count": context.sample_count,
            "sample_limit": context.sample_limit
        },
        "focused_operand_context": {
            "focused_first_mismatch_operands_available": context.raw_activation_i8_available && context.raw_packed_qk256_available,
            "raw_activation_i8_available": context.raw_activation_i8_available,
            "raw_packed_qk256_available": context.raw_packed_qk256_available,
            "summary_qk256_trace_available": context.summary_qk256_trace_available,
            "device_expression_trace_available": context.device_expression_trace_available,
            "device_intermediate_trace_available": context.device_intermediate_trace_available,
            "can_feed_production_replay": context.raw_activation_i8_available && context.raw_packed_qk256_available,
            "production_replay_executed": production_replay_executed,
            "production_replay_skipped_reason": production_replay_skipped_reason,
            "production_replay_error": production_replay_error,
            "missing_raw_operand_fields": context.missing_raw_operand_fields
        },
        "focused_raw_operand_summary": focused_raw_operand_summary,
        "focused_trace_replay_summary": {
            "available": focused_summary_divergence_available,
            "output_index": context.focused_output_index,
            "activation_sum": context.activation_sum,
            "activation_scale_bits": context.activation_scale_bits,
            "weight_scale_bits": context.weight_scale_bits,
            "int_dot": context.int_dot,
            "adjusted_dot": context.adjusted_dot,
            "focused_device_output_bits": context.focused_device_output_bits,
            "focused_policy_bits": focused_policy_bits_for_receipt,
            "source_host_summary_policy_bits": context.focused_policy_bits,
            "host_summary_policy_semantic_fix_applied": context.host_summary_policy_semantic_fix_applied,
            "host_summary_policy_semantic_fix_bits": context.host_summary_policy_semantic_fix_bits,
            "host_policy_div_then_mul_bits": context.host_policy_div_then_mul_bits,
            "host_policy_mul_then_div_bits": context.host_policy_mul_then_div_bits,
            "host_policy_reciprocal_then_mul_bits": context.host_policy_reciprocal_then_mul_bits,
            "host_policy_f64_div_then_mul_cast_bits": context.host_policy_f64_div_then_mul_cast_bits,
            "focused_summary_device_vs_policy_bits_match": focused_summary_device_vs_policy_bits_match,
            "device_intermediate_classification": context.device_intermediate_classification,
            "device_expression_classification": context.device_expression_classification,
            "production_policy_change_justified": context.production_policy_change_justified
        },
        "focused_production_replay_summary": focused_production_replay_summary,
        "focused_production_replay_samples": focused_production_replay_samples,
        "captured_intermediates": {
            "adjusted_dot": context.adjusted_dot.is_some(),
            "activation_scale": context.activation_scale_bits.is_some(),
            "weight_scale": context.weight_scale_bits.is_some(),
            "reciprocal_path_intermediate_bits": production_replay_executed,
            "final_scaled_value_bits": production_replay_executed,
            "output_store_bits": production_replay_executed
        },
        "production_replay_instrumentation_captured": production_replay_executed,
        "host_to_device_bytes": host_to_device_bytes,
        "device_to_host_bytes": device_to_host_bytes,
        "kernel_invocations": kernel_invocations,
        "fallback_used": false,
        "cpu_fallback_allowed": false,
        "bitnet_inference": false,
        "qk256_decode": false,
        "production_qk256_policy_change": false,
        "claim_allowed": false,
        "diagnostic_only": true,
        "performance_claim": false,
        "full_residency_claim": false,
        "next_diagnostic": next_diagnostic,
        "must_not_claim": [
            "CPU/A770 answer parity is proven",
            "Reference parity is proven",
            "Strict A770 answer readiness is proven",
            "Broad A770 answer quality is proven",
            "Official BitNet QK256 production semantics are proven",
            "Production QK256 dispatch policy changed",
            "BitNet inference works on A770",
            "A770 trusted partial acceleration is claim-grade",
            "Full A770 residency is proven",
            "A770 performance speedup is proven"
        ]
    });
    if receipt_kind == FocusedReceiptKind::HostPolicyExpressionSplit {
        let split = focused_host_policy_expression_split_json(&context, replay_outcome.as_ref());
        if let Some(object) = receipt.as_object_mut() {
            if let Some(source_receipts) =
                object.get_mut("source_receipts").and_then(Value::as_object_mut)
            {
                source_receipts.insert(
                    "focused_raw_operand_replay".to_string(),
                    json!(
                        "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-raw-operands-replay.json"
                    ),
                );
            }
            object.insert("host_policy_expression_split".to_string(), split);
        }
    } else if receipt_kind == FocusedReceiptKind::HostSummaryPolicySemanticFix {
        let fix = focused_host_summary_policy_semantic_fix_json(&context, replay_outcome.as_ref());
        if let Some(object) = receipt.as_object_mut() {
            if let Some(source_receipts) =
                object.get_mut("source_receipts").and_then(Value::as_object_mut)
            {
                source_receipts.insert(
                    "focused_host_policy_expression_split".to_string(),
                    json!(
                        "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-host-policy-expression-split.json"
                    ),
                );
                source_receipts.insert(
                    "focused_raw_operand_replay".to_string(),
                    json!(
                        "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-raw-operands-replay.json"
                    ),
                );
            }
            object.insert("host_summary_policy_semantic_fix".to_string(), fix);
        }
    }
    Ok(serde_json::to_string_pretty(&receipt)? + "\n")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FocusedReceiptKind {
    RawOperandReplay,
    HostPolicyExpressionSplit,
    HostSummaryPolicySemanticFix,
}

impl FocusedReceiptKind {
    fn from_receipt_path(path: &Path) -> Self {
        let path = path.to_string_lossy().replace('\\', "/");
        if path.contains("host-summary-policy-semantic-fix") {
            Self::HostSummaryPolicySemanticFix
        } else if path.contains("focused-host-policy-expression-split") {
            Self::HostPolicyExpressionSplit
        } else {
            Self::RawOperandReplay
        }
    }

    fn work_item(self) -> &'static str {
        match self {
            Self::RawOperandReplay => "A770-064",
            Self::HostPolicyExpressionSplit => "A770-065",
            Self::HostSummaryPolicySemanticFix => "A770-066",
        }
    }

    fn proof_family(self) -> &'static str {
        match self {
            Self::RawOperandReplay => "a770_opencl_qk256_focused_raw_operand_replay",
            Self::HostPolicyExpressionSplit => {
                "a770_opencl_qk256_focused_host_policy_expression_split"
            }
            Self::HostSummaryPolicySemanticFix => {
                "a770_opencl_qk256_host_summary_policy_semantic_fix"
            }
        }
    }

    fn proof_stage(self) -> &'static str {
        match self {
            Self::RawOperandReplay => {
                "diagnostic_focused_raw_operands_production_replay_classified"
            }
            Self::HostPolicyExpressionSplit => {
                "diagnostic_focused_host_policy_expression_split_classified"
            }
            Self::HostSummaryPolicySemanticFix => {
                "diagnostic_host_summary_policy_semantic_fix_classified"
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ManifestTarget {
    target_id: String,
    case_id: String,
    first_mismatch_index: usize,
    target_layer_idx: Option<i64>,
    projection: Option<String>,
    tensor_name: Option<String>,
    qk256_key: Option<String>,
    source_context_available: bool,
    dispatch_replay: Option<Value>,
    dispatch_replay_source: Option<String>,
}

#[derive(Debug, Clone)]
struct ManifestTargetResult {
    json: Value,
    executed: bool,
    matched_selected_device_bits: bool,
    failed: bool,
    blocked: bool,
    host_to_device_bytes: usize,
    device_to_host_bytes: usize,
    kernel_invocations: usize,
    platform_index: Option<usize>,
    device_index: Option<usize>,
    platform_name: Option<String>,
    runtime_device: Option<String>,
    vendor: Option<String>,
    driver_version: Option<String>,
}

fn multi_case_focused_replay_to_json(args: &Args) -> Result<(String, String), Box<dyn Error>> {
    let focused_source = args
        .focused_sources
        .first()
        .ok_or_else(|| io_error("multi-target replay requested without --focused-source"))?;
    let mut targets = manifest_targets_from_source_path(
        focused_source,
        args.case_id.as_deref(),
        args.first_mismatch_index.as_ref(),
    )?;
    for supplemental_source in args.focused_sources.iter().skip(1) {
        let supplemental_targets = manifest_targets_from_source_path(
            supplemental_source,
            args.case_id.as_deref(),
            args.first_mismatch_index.as_ref(),
        )?;
        apply_supplemental_focused_targets(&mut targets, supplemental_targets);
    }
    let focused_sources =
        args.focused_sources.iter().map(|path| path_json_value(path)).collect::<Vec<_>>();
    let work_item = args.work_item.as_deref().unwrap_or("A770-067");
    let manifest_target_count = targets.len();
    let manifest_dispatch_replay_target_count =
        targets.iter().filter(|target| target.dispatch_replay.is_some()).count();
    let manifest_runnable_target_count =
        targets.iter().filter(|target| manifest_target_has_raw_operands(target)).count();
    let manifest_targets_json = targets.iter().map(manifest_target_json).collect::<Vec<Value>>();
    let manifest_path = args.manifest.as_ref().map(|path| path_json_value(path));
    let manifest = json!({
        "schema_version": "1.0.0",
        "manifest_kind": "a770_multi_case_focused_qk256_replay_manifest",
        "campaign": "intel-a770",
        "work_item": work_item,
        "diagnostic_only": true,
        "claim_allowed": false,
        "target_policy": {
            "target_source": "first-mismatch qkv projection source stack",
            "replay_rule": "run selected-device A770 OpenCL production replay only for targets with raw focused operands",
            "blocker_rule": "ledger targets without dispatch replay or raw focused operands instead of promoting production QK256 policy",
            "cpu_fallback_allowed": false,
            "fallback_used_must_equal": false
        },
        "source_receipts": {
            "focused_source": path_json_value(focused_source),
            "focused_sources": focused_sources,
            "a770_067_multi_case_focused_replay_manifest": "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-multi-case-focused-qk256-replay/a770-opencl-qk256-multi-case-focused-replay-manifest.json",
            "a770_067_multi_case_focused_replay": "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-multi-case-focused-qk256-replay/a770-opencl-qk256-multi-case-focused-replay.json",
            "a770_066_host_summary_policy_semantic_fix_replay": "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-host-summary-policy-semantic-fix-replay.json",
            "a770_066_host_summary_policy_semantic_fix_parity": "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-host-summary-policy-semantic-fix/cpu-avx2-vs-a770-summary-logits-host-summary-policy-fix.json",
            "a770_064_focused_raw_operands_parity": "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/cpu-avx2-vs-a770-summary-logits-raw-operands-parity.json"
        },
        "case_filter": args.case_id,
        "first_mismatch_index_filter": args.first_mismatch_index,
        "target_count": manifest_target_count,
        "dispatch_replay_target_count": manifest_dispatch_replay_target_count,
        "runnable_target_count": manifest_runnable_target_count,
        "targets": manifest_targets_json,
    });

    let mut results = Vec::with_capacity(targets.len());
    let mut executed_count = 0usize;
    let mut matched_selected_device_bits_count = 0usize;
    let mut failed_count = 0usize;
    let mut blocked_count = 0usize;
    let mut host_to_device_bytes = 0usize;
    let mut device_to_host_bytes = 0usize;
    let mut kernel_invocations = 0usize;
    let mut first_runtime: Option<ManifestTargetResult> = None;

    for target in &targets {
        let result = manifest_target_result(target)?;
        if result.executed {
            executed_count += 1;
            host_to_device_bytes += result.host_to_device_bytes;
            device_to_host_bytes += result.device_to_host_bytes;
            kernel_invocations += result.kernel_invocations;
        }
        if result.matched_selected_device_bits {
            matched_selected_device_bits_count += 1;
        }
        if result.failed {
            failed_count += 1;
        }
        if result.blocked {
            blocked_count += 1;
        }
        if first_runtime.is_none() && result.runtime_device.is_some() {
            first_runtime = Some(result.clone());
        }
        results.push(result.json);
    }

    let classification = multi_case_classification(
        manifest_target_count,
        executed_count,
        matched_selected_device_bits_count,
        failed_count,
        blocked_count,
    );
    let runtime = first_runtime.as_ref();
    let receipt = json!({
        "schema_version": "1.0.0",
        "campaign": "intel-a770",
        "work_item": work_item,
        "proof_family": "a770_opencl_qk256_multi_case_focused_replay",
        "proof_stage": "diagnostic_multi_case_focused_qk256_replay_packet",
        "requested_backend": "intel-arc-a770",
        "selected_backend": "intel-arc-a770-opencl",
        "runtime_api": "opencl",
        "runtime_device": runtime.and_then(|value| value.runtime_device.clone()),
        "platform_index": runtime.and_then(|value| value.platform_index),
        "device_index": runtime.and_then(|value| value.device_index),
        "platform_name": runtime.and_then(|value| value.platform_name.clone()),
        "vendor": runtime.and_then(|value| value.vendor.clone()),
        "driver_version": runtime.and_then(|value| value.driver_version.clone()),
        "kernel_name": "qk256_i2s_i8s_scaled_gemv",
        "replay_kernel_name": "qk256_i2s_i8s_scaled_gemv_production_replay",
        "classification": classification,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "target_results": results,
        "summary": {
            "target_count": manifest_target_count,
            "runnable_target_count": manifest_runnable_target_count,
            "executed_target_count": executed_count,
            "matched_selected_device_bits_count": matched_selected_device_bits_count,
            "failed_target_count": failed_count,
            "blocked_target_count": blocked_count,
            "all_executed_targets_match_selected_device_bits": executed_count > 0
                && matched_selected_device_bits_count == executed_count,
            "has_blocked_targets": blocked_count > 0,
            "host_to_device_bytes": host_to_device_bytes,
            "device_to_host_bytes": device_to_host_bytes,
            "kernel_invocations": kernel_invocations
        },
        "fallback_used": false,
        "cpu_fallback_allowed": false,
        "bitnet_inference": false,
        "qk256_decode": false,
        "production_qk256_policy_change": false,
        "claim_allowed": false,
        "diagnostic_only": true,
        "performance_claim": false,
        "full_residency_claim": false,
        "next_diagnostic": "capture raw focused operands for the blocked manifest targets before any production QK256 promotion",
        "must_not_claim": [
            "CPU/A770 answer parity is proven",
            "Reference parity is proven",
            "Strict A770 answer readiness is proven",
            "Broad A770 answer quality is proven",
            "Official BitNet QK256 production semantics are proven",
            "Production QK256 dispatch policy changed",
            "BitNet inference works on A770",
            "A770 trusted partial acceleration is claim-grade",
            "Full A770 residency is proven",
            "A770 performance speedup is proven"
        ]
    });

    Ok((
        serde_json::to_string_pretty(&receipt["manifest"])? + "\n",
        serde_json::to_string_pretty(&receipt)? + "\n",
    ))
}

fn manifest_targets(
    source: &Value,
    case_filter: Option<&str>,
    first_mismatch_index_filter: Option<&usize>,
) -> Result<Vec<ManifestTarget>, Box<dyn Error>> {
    let mut targets = Vec::new();
    let cases = source
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| io_error("focused source missing cases array"))?;
    for case in cases {
        let case_id = match str_field(case, "id") {
            Some(case_id) => case_id,
            None => continue,
        };
        if case_filter.is_some_and(|filter| filter != case_id) {
            continue;
        }
        let logits_dump = match case.get("logits_dump").and_then(Value::as_array) {
            Some(logits_dump) => logits_dump,
            None => continue,
        };
        for (step_index, step) in logits_dump.iter().enumerate() {
            if first_mismatch_index_filter.is_some_and(|filter| *filter != step_index) {
                continue;
            }
            let Some(sources) = step
                .pointer(
                    "/logit_source_context/hidden_state_source/model_forward_source/qkv_projection_sources/sources",
                )
                .and_then(Value::as_array)
            else {
                continue;
            };
            for source in sources {
                let projection = string_field(source, "projection");
                let tensor_name = string_field(source, "tensor_name");
                let qk256_key = string_field(source, "qk256_key");
                let target_layer_idx = i64_field(source, "target_layer_idx")
                    .or_else(|| i64_field(source, "layer_idx"));
                let target_id = format!(
                    "{case_id}:step{step_index}:{}:{}",
                    target_layer_idx
                        .map(|idx| idx.to_string())
                        .unwrap_or_else(|| "layer_unknown".to_string()),
                    projection
                        .as_deref()
                        .or(tensor_name.as_deref())
                        .unwrap_or("projection_unknown")
                );
                targets.push(ManifestTarget {
                    target_id,
                    case_id: case_id.to_string(),
                    first_mismatch_index: step_index,
                    target_layer_idx,
                    projection,
                    tensor_name,
                    qk256_key,
                    source_context_available: source
                        .get("source_context_available")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    dispatch_replay: source
                        .get("dispatch_replay")
                        .filter(|value| value.is_object())
                        .cloned(),
                    dispatch_replay_source: None,
                });
            }
        }
    }
    if targets.is_empty() {
        return Err(io_error("focused source did not yield any manifest replay targets"));
    }
    Ok(targets)
}

fn manifest_targets_from_source_path(
    path: &Path,
    case_filter: Option<&str>,
    first_mismatch_index_filter: Option<&usize>,
) -> Result<Vec<ManifestTarget>, Box<dyn Error>> {
    let source_json = fs::read_to_string(path)?;
    let source: Value = serde_json::from_str(&source_json)?;
    let dispatch_replay_source = path_json_value(path);
    let mut targets = manifest_targets(&source, case_filter, first_mismatch_index_filter)?;
    for target in &mut targets {
        if target.dispatch_replay.is_some() {
            target.dispatch_replay_source = Some(dispatch_replay_source.clone());
        }
    }
    Ok(targets)
}

fn apply_supplemental_focused_targets(
    targets: &mut [ManifestTarget],
    supplemental_targets: Vec<ManifestTarget>,
) {
    for supplemental_target in supplemental_targets {
        if !manifest_target_has_raw_operands(&supplemental_target) {
            continue;
        }
        if let Some(target) =
            targets.iter_mut().find(|target| target.target_id == supplemental_target.target_id)
        {
            if !manifest_target_has_raw_operands(target) {
                target.dispatch_replay = supplemental_target.dispatch_replay;
                target.dispatch_replay_source = supplemental_target.dispatch_replay_source;
            }
        }
    }
}

fn manifest_target_json(target: &ManifestTarget) -> Value {
    let raw_operands_available = manifest_target_has_raw_operands(target);
    let blocker = if target.dispatch_replay.is_none() {
        Some("dispatch_replay_missing")
    } else if !raw_operands_available {
        Some("raw_focused_operands_missing")
    } else {
        None
    };
    json!({
        "target_id": target.target_id,
        "case_id": target.case_id,
        "first_mismatch_index": target.first_mismatch_index,
        "target_layer_idx": target.target_layer_idx,
        "projection": target.projection,
        "tensor_name": target.tensor_name,
        "qk256_key": target.qk256_key,
        "source_context_available": target.source_context_available,
        "dispatch_replay_available": target.dispatch_replay.is_some(),
        "dispatch_replay_source": target.dispatch_replay_source,
        "raw_focused_operands_available": raw_operands_available,
        "blocker": blocker,
    })
}

fn manifest_target_has_raw_operands(target: &ManifestTarget) -> bool {
    target.dispatch_replay.as_ref().is_some_and(|replay| {
        any_array_at(Some(replay), &[&["focused_operands", "activations_i8"]])
            && any_array_at(Some(replay), &[&["focused_operands", "packed_qk256"]])
    })
}

fn manifest_target_result(target: &ManifestTarget) -> Result<ManifestTargetResult, Box<dyn Error>> {
    let Some(replay_source) = target.dispatch_replay.as_ref() else {
        return Ok(blocked_manifest_target_result(target, "dispatch_replay_missing"));
    };
    let Some(operands) = focused_raw_operands_from_replay(replay_source)? else {
        return Ok(blocked_manifest_target_result(target, "raw_focused_operands_missing"));
    };
    let focused_device_output_bits = replay_source
        .pointer("/device_intermediate_trace/samples/0/output_bits")
        .and_then(Value::as_u64);
    let source_device = replay_source.pointer("/a770/last_device");
    match run_focused_raw_operands_replay(operands) {
        FocusedReplayOutcome::Executed { operands, replay } => {
            let sample = replay.samples.first();
            let production_output_bits =
                sample.map(|sample| u64::from(sample.production_output_bits));
            let replay_output_bits = sample.map(|sample| u64::from(sample.replay_output_bits));
            let final_scaled_value_bits =
                sample.map(|sample| u64::from(sample.final_scaled_value_bits));
            let production_matches_device = production_output_bits
                .zip(focused_device_output_bits)
                .map(|(left, right)| left == right);
            let matched_selected_device_bits = production_matches_device == Some(true);
            let classification = if matched_selected_device_bits {
                "a770_qk256_multi_case_focused_replay_matches_selected_device_output"
            } else if focused_device_output_bits.is_some() {
                "a770_qk256_multi_case_focused_replay_differs_from_selected_device_output"
            } else {
                "a770_qk256_multi_case_focused_replay_missing_selected_device_bits"
            };
            let json = json!({
                "target_id": target.target_id,
                "case_id": target.case_id,
                "first_mismatch_index": target.first_mismatch_index,
                "target_layer_idx": target.target_layer_idx,
                "projection": target.projection,
                "tensor_name": target.tensor_name,
                "qk256_key": target.qk256_key,
                "dispatch_replay_source": target.dispatch_replay_source,
                "classification": classification,
                "raw_focused_operands_available": true,
                "production_replay": {
                    "executed": true,
                    "selected_backend": "intel-arc-a770-opencl",
                    "runtime_api": "opencl",
                    "runtime_device": replay.runtime_device,
                    "platform_index": replay.platform_index,
                    "device_index": replay.device_index,
                    "platform_name": replay.platform_name,
                    "vendor": replay.vendor,
                    "driver_version": replay.driver_version,
                    "fallback_used": false,
                    "kernel_invocations": replay.kernel_invocations,
                    "host_to_device_bytes": replay.host_to_device_bytes,
                    "device_to_host_bytes": replay.device_to_host_bytes,
                    "source_output_index": operands.output_index,
                    "replay_output_index": sample.map(|sample| sample.output_index),
                    "focused_device_output_bits": focused_device_output_bits,
                    "production_output_bits": production_output_bits,
                    "replay_output_bits": replay_output_bits,
                    "final_scaled_value_bits": final_scaled_value_bits,
                    "production_output_matches_selected_device_bits": production_matches_device,
                    "replay_output_matches_selected_device_bits": replay_output_bits
                        .zip(focused_device_output_bits)
                        .map(|(left, right)| left == right),
                    "final_scaled_value_matches_selected_device_bits": final_scaled_value_bits
                        .zip(focused_device_output_bits)
                        .map(|(left, right)| left == right)
                },
                "source_selected_device": source_device,
                "claim_boundary": {
                    "production_qk256_policy_change": false,
                    "bitnet_inference": false,
                    "qk256_decode": false,
                    "claim_allowed": false,
                    "diagnostic_only": true,
                    "performance_claim": false,
                    "full_residency_claim": false
                }
            });
            Ok(ManifestTargetResult {
                json,
                executed: true,
                matched_selected_device_bits,
                failed: false,
                blocked: false,
                host_to_device_bytes: replay.host_to_device_bytes,
                device_to_host_bytes: replay.device_to_host_bytes,
                kernel_invocations: replay.kernel_invocations,
                platform_index: Some(replay.platform_index),
                device_index: Some(replay.device_index),
                platform_name: Some(replay.platform_name),
                runtime_device: Some(replay.runtime_device),
                vendor: Some(replay.vendor),
                driver_version: Some(replay.driver_version),
            })
        }
        FocusedReplayOutcome::Failed { error } => {
            let json = json!({
                "target_id": target.target_id,
                "case_id": target.case_id,
                "first_mismatch_index": target.first_mismatch_index,
                "target_layer_idx": target.target_layer_idx,
                "projection": target.projection,
                "tensor_name": target.tensor_name,
                "qk256_key": target.qk256_key,
                "dispatch_replay_source": target.dispatch_replay_source,
                "classification": "a770_qk256_multi_case_focused_replay_failed",
                "raw_focused_operands_available": true,
                "production_replay": {
                    "executed": false,
                    "error": error,
                    "fallback_used": false
                },
                "source_selected_device": source_device,
                "claim_boundary": {
                    "production_qk256_policy_change": false,
                    "bitnet_inference": false,
                    "claim_allowed": false,
                    "diagnostic_only": true
                }
            });
            Ok(ManifestTargetResult {
                json,
                executed: false,
                matched_selected_device_bits: false,
                failed: true,
                blocked: false,
                host_to_device_bytes: 0,
                device_to_host_bytes: 0,
                kernel_invocations: 0,
                platform_index: None,
                device_index: None,
                platform_name: None,
                runtime_device: None,
                vendor: None,
                driver_version: None,
            })
        }
    }
}

fn blocked_manifest_target_result(
    target: &ManifestTarget,
    blocker: &'static str,
) -> ManifestTargetResult {
    let json = json!({
        "target_id": target.target_id,
        "case_id": target.case_id,
        "first_mismatch_index": target.first_mismatch_index,
        "target_layer_idx": target.target_layer_idx,
        "projection": target.projection,
        "tensor_name": target.tensor_name,
        "qk256_key": target.qk256_key,
        "dispatch_replay_source": target.dispatch_replay_source,
        "classification": "a770_qk256_multi_case_focused_replay_blocked",
        "raw_focused_operands_available": false,
        "blocker": blocker,
        "production_replay": {
            "executed": false,
            "reason": blocker,
            "fallback_used": false
        },
        "claim_boundary": {
            "production_qk256_policy_change": false,
            "bitnet_inference": false,
            "claim_allowed": false,
            "diagnostic_only": true
        }
    });
    ManifestTargetResult {
        json,
        executed: false,
        matched_selected_device_bits: false,
        failed: false,
        blocked: true,
        host_to_device_bytes: 0,
        device_to_host_bytes: 0,
        kernel_invocations: 0,
        platform_index: None,
        device_index: None,
        platform_name: None,
        runtime_device: None,
        vendor: None,
        driver_version: None,
    }
}

fn multi_case_classification(
    target_count: usize,
    executed_count: usize,
    matched_count: usize,
    failed_count: usize,
    blocked_count: usize,
) -> &'static str {
    if target_count == 0 {
        "a770_qk256_multi_case_focused_replay_manifest_empty"
    } else if failed_count > 0 {
        "a770_qk256_multi_case_focused_replay_has_failed_target"
    } else if executed_count == 0 {
        "a770_qk256_multi_case_focused_replay_all_targets_blocked"
    } else if blocked_count > 0 && matched_count == executed_count {
        "a770_qk256_multi_case_focused_replay_partial_manifest_blocked_on_missing_raw_operands"
    } else if blocked_count == 0 && matched_count == executed_count {
        "a770_qk256_multi_case_focused_replay_all_manifest_targets_match_selected_device"
    } else {
        "a770_qk256_multi_case_focused_replay_has_selected_device_mismatch"
    }
}

fn projection_level_qkv_replay_to_json(args: &Args) -> Result<(String, String), Box<dyn Error>> {
    let source_path = args
        .projection_source
        .as_ref()
        .ok_or_else(|| io_error("projection-level replay requested without --projection-source"))?;
    let source_json = fs::read_to_string(source_path)?;
    let source: Value = serde_json::from_str(&source_json)?;
    let (target_results, source_receipt_kind) = projection_source_targets(&source)?;
    let case_id = args
        .case_id
        .as_deref()
        .or_else(|| target_results.iter().find_map(|target| str_field(target, "case_id")))
        .ok_or_else(|| io_error("projection source missing case_id"))?;
    let first_mismatch_index = args
        .first_mismatch_index
        .or_else(|| {
            target_results.iter().find_map(|target| usize_field(target, "first_mismatch_index"))
        })
        .ok_or_else(|| io_error("projection source missing first_mismatch_index"))?;
    let projection_layer = args
        .projection_layer
        .or_else(|| {
            target_results
                .iter()
                .filter(|target| {
                    str_field(target, "case_id") == Some(case_id)
                        && usize_field(target, "first_mismatch_index") == Some(first_mismatch_index)
                })
                .filter_map(|target| i64_field(target, "target_layer_idx"))
                .min()
        })
        .or_else(|| source.pointer("/manifest/target_layer_idx_filter").and_then(Value::as_i64))
        .ok_or_else(|| io_error("projection source missing target_layer_idx"))?;
    let work_item = args.work_item.as_deref().unwrap_or("A770-158");
    let projections = ["q_proj", "k_proj", "v_proj"];
    let mut targets = Vec::with_capacity(projections.len());
    let mut row_evidence_count = 0usize;
    let mut clean_row_evidence_count = 0usize;
    let mut row_selected_device_match_count = 0usize;
    let mut row_fallback_false_count = 0usize;
    let mut projection_executed_count = 0usize;
    let mut projection_blocked_count = 0usize;
    let mut projection_full_operands_available_count = 0usize;
    let mut projection_replay_hook_available_count = 0usize;
    let mut projection_fallback_false_count = 0usize;
    let mut projection_operand_capture_source_count = 0usize;
    let mut projection_focused_operand_source_count = 0usize;
    let mut projection_full_operand_source_count = 0usize;
    let mut summary_blockers = Vec::<&'static str>::new();

    for projection in projections {
        let row = target_results.iter().copied().find(|target| {
            str_field(target, "case_id") == Some(case_id)
                && usize_field(target, "first_mismatch_index") == Some(first_mismatch_index)
                && i64_field(target, "target_layer_idx") == Some(projection_layer)
                && str_field(target, "projection") == Some(projection)
        });
        let row_evidence = projection_row_evidence(row);
        let operand_capture_evidence = projection_operand_capture_evidence(
            row,
            case_id,
            first_mismatch_index,
            projection_layer,
            projection,
            source_receipt_kind,
        );
        if operand_capture_evidence.source_projection_found {
            projection_operand_capture_source_count += 1;
        }
        if operand_capture_evidence.focused_operands_available {
            projection_focused_operand_source_count += 1;
        }
        if operand_capture_evidence.full_projection_operands_available {
            projection_full_operand_source_count += 1;
        }
        let clean_row_evidence = row_evidence.clean_for_projection_boundary();
        if row_evidence.available {
            row_evidence_count += 1;
        }
        if row_evidence.fallback_used == Some(false) {
            row_fallback_false_count += 1;
        }
        if row_evidence.production_output_matches_selected_device_bits {
            row_selected_device_match_count += 1;
        }
        if clean_row_evidence {
            clean_row_evidence_count += 1;
        }

        let target_id = format!(
            "{case_id}:step{first_mismatch_index}:{projection_layer}:{projection}:projection"
        );
        projection_replay_hook_available_count += 1;
        let replay_outcome = if clean_row_evidence {
            projection_replay_outcome(row, &operand_capture_evidence)
        } else {
            ProjectionReplayOutcome::Blocked {
                reason: "projection_level_row_evidence_not_clean",
                blockers: vec!["projection_level_row_evidence_not_clean"],
                missing_full_operand_fields: Vec::new(),
                current_operand_scope: "row_evidence_not_clean".to_owned(),
            }
        };
        let (projection_replay, target_blockers) = match replay_outcome {
            ProjectionReplayOutcome::Executed { operands, replay } => {
                projection_executed_count += 1;
                projection_full_operands_available_count += 1;
                projection_fallback_false_count += 1;
                let output_store_all_matches_replay_output =
                    replay.samples.iter().all(|sample| sample.output_store_matches_replay_output);
                let output_store_all_matches_final_scaled_value = replay
                    .samples
                    .iter()
                    .all(|sample| sample.output_store_matches_final_scaled_value);
                (
                    json!({
                        "executed": true,
                        "selected_backend": "intel-arc-a770-opencl",
                        "runtime_api": "opencl",
                        "runtime_device": replay.runtime_device,
                        "platform_index": replay.platform_index,
                        "device_index": replay.device_index,
                        "platform_name": replay.platform_name,
                        "vendor": replay.vendor,
                        "driver_version": replay.driver_version,
                        "fallback_used": false,
                        "kernel_name": PROJECTION_REPLAY_KERNEL_NAME,
                        "projection_level_replay_hook_available": true,
                        "projection_level_full_operands_available": true,
                        "input_row_index": operands.input_row_index,
                        "rows": operands.rows,
                        "cols": operands.cols,
                        "row_stride_bytes": operands.row_stride_bytes,
                        "activation_sum": operands.activation_sum,
                        "activation_scale_bits": operands.activation_scale_bits,
                        "weight_scale_bits": operands.weight_scale_bits,
                        "sample_limit": projection_replay_sample_limit(operands.rows),
                        "sample_count": replay.samples.len(),
                        "activation_i8_len": operands.activations_i8.len(),
                        "packed_qk256_len": operands.packed_qk256.len(),
                        "host_to_device_bytes": replay.host_to_device_bytes,
                        "device_to_host_bytes": replay.device_to_host_bytes,
                        "kernel_invocations": replay.kernel_invocations,
                        "output_store_all_matches_replay_output": output_store_all_matches_replay_output,
                        "output_store_all_matches_final_scaled_value": output_store_all_matches_final_scaled_value,
                        "samples": replay
                            .samples
                            .iter()
                            .map(focused_replay_sample_json)
                            .collect::<Vec<Value>>()
                    }),
                    Vec::new(),
                )
            }
            ProjectionReplayOutcome::Failed { error, blockers } => {
                projection_blocked_count += 1;
                projection_full_operands_available_count += 1;
                projection_fallback_false_count += 1;
                push_unique_blockers(&mut summary_blockers, &blockers);
                (
                    json!({
                        "executed": false,
                        "selected_backend": "intel-arc-a770-opencl",
                        "runtime_api": "opencl",
                        "fallback_used": false,
                        "kernel_name": PROJECTION_REPLAY_KERNEL_NAME,
                        "projection_level_replay_hook_available": true,
                        "projection_level_full_operands_available": true,
                        "reason": "projection_level_replay_failed",
                        "blockers": blockers,
                        "error": error
                    }),
                    blockers,
                )
            }
            ProjectionReplayOutcome::Blocked {
                reason,
                blockers,
                missing_full_operand_fields,
                current_operand_scope,
            } => {
                projection_blocked_count += 1;
                projection_fallback_false_count += 1;
                push_unique_blockers(&mut summary_blockers, &blockers);
                (
                    json!({
                        "executed": false,
                        "selected_backend": "intel-arc-a770-opencl",
                        "runtime_api": "opencl",
                        "fallback_used": false,
                        "kernel_name": PROJECTION_REPLAY_KERNEL_NAME,
                        "reason": reason,
                        "blockers": blockers,
                        "current_operand_scope": current_operand_scope,
                        "required_operand_scope": "full_projection_output_rows",
                        "missing_full_operand_fields": missing_full_operand_fields,
                        "projection_level_replay_hook_available": true,
                        "projection_level_full_operands_available": false
                    }),
                    blockers,
                )
            }
        };
        targets.push(json!({
            "target_id": target_id,
            "case_id": case_id,
            "first_mismatch_index": first_mismatch_index,
            "target_layer_idx": projection_layer,
            "projection": projection,
            "tensor_name": row.and_then(|target| string_field(target, "tensor_name")),
            "qk256_key": row.and_then(|target| string_field(target, "qk256_key")),
            "row_evidence": {
                "available": row_evidence.available,
                "source_target_id": row_evidence.source_target_id,
                "dispatch_replay_source": row_evidence.dispatch_replay_source,
                "executed": row_evidence.executed,
                "selected_backend": row_evidence.selected_backend,
                "runtime_api": row_evidence.runtime_api,
                "runtime_device": row_evidence.runtime_device,
                "fallback_used": row_evidence.fallback_used,
                "source_output_index": row_evidence.source_output_index,
                "focused_device_output_bits": row_evidence.focused_device_output_bits,
                "production_output_bits": row_evidence.production_output_bits,
                "production_output_matches_selected_device_bits": row_evidence.production_output_matches_selected_device_bits,
                "clean_for_projection_boundary": clean_row_evidence
            },
            "projection_operand_capture": operand_capture_evidence.to_json(),
            "projection_replay": projection_replay,
            "classification": if target_blockers.is_empty() {
                "a770_qk256_projection_level_replay_executed_selected_device"
            } else {
                "a770_qk256_projection_level_replay_blocked"
            },
            "claim_boundary": {
                "production_qk256_policy_change": false,
                "answer_scoring_change": false,
                "sampling_change": false,
                "cpu_a770_parity_claim": false,
                "strict_answer_readiness_claim": false,
                "broad_a770_quality_claim": false,
                "bitnet_inference": false,
                "qk256_decode": false,
                "claim_allowed": false,
                "diagnostic_only": true,
                "performance_claim": false,
                "full_residency_claim": false
            }
        }));
    }

    let classification = if clean_row_evidence_count != projections.len() {
        "a770_qk256_projection_level_qkv_replay_blocked_on_row_evidence"
    } else if projection_executed_count == projections.len() {
        "a770_qk256_projection_level_qkv_replay_executed_selected_device"
    } else if summary_blockers
        .contains(&"projection_level_full_projection_packed_row_capture_source_missing")
    {
        "a770_qk256_projection_level_qkv_replay_blocked_on_full_projection_packed_row_capture_source"
    } else if summary_blockers.contains(&"projection_level_full_operands_missing")
        && !summary_blockers.contains(&"projection_level_replay_hook_missing")
    {
        "a770_qk256_projection_level_qkv_replay_blocked_on_projection_operands"
    } else {
        "a770_qk256_projection_level_qkv_replay_blocked"
    };
    let source_receipts = match source_receipt_kind {
        "a770_159_full_projection_operand_source_boundary" => json!({
            "a770_159_full_projection_operand_source_boundary": path_json_value(source_path),
        }),
        "a770_158_projection_replay_hook_boundary" => json!({
            "a770_158_projection_replay_hook_boundary": path_json_value(source_path),
        }),
        "a770_157_projection_level_qkv_boundary" => json!({
            "a770_157_projection_level_qkv_boundary": path_json_value(source_path),
        }),
        _ => json!({
            "a770_156_focused_qkv_replay": path_json_value(source_path),
        }),
    };
    let manifest_path = args.manifest.as_ref().map(|path| path_json_value(path));
    let manifest = json!({
        "schema_version": "1.0.0",
        "manifest_kind": "a770_projection_level_qkv_replay_manifest",
        "campaign": "intel-a770",
        "work_item": work_item,
        "diagnostic_only": true,
        "claim_allowed": false,
        "source_receipts": source_receipts,
        "case_filter": case_id,
        "first_mismatch_index_filter": first_mismatch_index,
        "target_layer_idx_filter": projection_layer,
        "target_policy": {
            "target_source": match source_receipt_kind {
                "a770_159_full_projection_operand_source_boundary" => {
                    "A770-159 full projection operand source boundary receipt"
                }
                "a770_158_projection_replay_hook_boundary" => {
                    "A770-158 projection replay hook boundary receipt"
                }
                "a770_157_projection_level_qkv_boundary" => {
                    "A770-157 projection-level Q/K/V boundary receipt"
                }
                _ => "A770-156 clean focused Q/K/V row replay packet",
            },
            "target_surface": "one transformer layer Q/K/V projection-level replay boundary",
            "replay_rule": "run selected-device A770 OpenCL projection replay only when full projection operands are available through the bounded replay hook",
            "blocker_rule": match source_receipt_kind {
                "a770_159_full_projection_operand_source_boundary" => {
                    "ledger missing full projection packed-row capture source instead of promoting production QK256 policy"
                }
                _ => {
                    "ledger missing full projection operands instead of promoting production QK256 policy"
                }
            },
            "cpu_fallback_allowed": false,
            "fallback_used_must_equal": false
        },
        "target_count": projections.len(),
        "projection_replay_target_count": projections.len(),
        "targets": targets,
    });
    let source_runtime = target_results
        .iter()
        .find_map(|target| target.get("production_replay").or_else(|| target.get("row_evidence")))
        .or_else(|| source.as_object().map(|_| &source));
    let receipt = json!({
        "schema_version": "1.0.0",
        "campaign": "intel-a770",
        "work_item": work_item,
        "proof_family": "a770_opencl_qk256_projection_level_qkv_replay",
        "proof_stage": "diagnostic_projection_level_qkv_replay_boundary",
        "requested_backend": "intel-arc-a770",
        "selected_backend": "intel-arc-a770-opencl",
        "runtime_api": "opencl",
        "runtime_device": source_runtime.and_then(|value| string_field(value, "runtime_device")),
        "platform_index": source_runtime.and_then(|value| usize_field(value, "platform_index")),
        "device_index": source_runtime.and_then(|value| usize_field(value, "device_index")),
        "platform_name": source_runtime.and_then(|value| string_field(value, "platform_name")),
        "vendor": source_runtime.and_then(|value| string_field(value, "vendor")),
        "driver_version": source_runtime.and_then(|value| string_field(value, "driver_version")),
        "kernel_name": "qk256_i2s_i8s_scaled_gemv",
        "projection_replay_kernel_name": PROJECTION_REPLAY_KERNEL_NAME,
        "projection_replay_kernel_executed": projection_executed_count > 0,
        "classification": classification,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "summary": {
            "target_count": projections.len(),
            "projection_replay_target_count": projections.len(),
            "projection_replay_executed_count": projection_executed_count,
            "projection_replay_blocked_count": projection_blocked_count,
            "projection_replay_hook_available_count": projection_replay_hook_available_count,
            "projection_level_full_operands_available_count": projection_full_operands_available_count,
            "projection_replay_fallback_false_count": projection_fallback_false_count,
            "projection_operand_capture_source_count": projection_operand_capture_source_count,
            "projection_focused_operand_source_count": projection_focused_operand_source_count,
            "projection_full_operand_source_count": projection_full_operand_source_count,
            "row_evidence_target_count": row_evidence_count,
            "clean_row_evidence_count": clean_row_evidence_count,
            "row_selected_device_match_count": row_selected_device_match_count,
            "row_fallback_false_count": row_fallback_false_count,
            "all_row_evidence_clean": clean_row_evidence_count == projections.len(),
            "all_projection_replay_targets_blocked": projection_blocked_count == projections.len(),
            "blockers": summary_blockers
        },
        "fallback_used": false,
        "cpu_fallback_allowed": false,
        "bitnet_inference": false,
        "qk256_decode": false,
        "production_qk256_policy_change": false,
        "claim_allowed": false,
        "diagnostic_only": true,
        "performance_claim": false,
        "full_residency_claim": false,
        "next_diagnostic": match source_receipt_kind {
            "a770_159_full_projection_operand_source_boundary" => {
                "add a full projection packed-row capture hook or source before executing the bounded projection-level replay hook or any production QK256 promotion"
            }
            _ => {
                "capture full projection Q/K/V operands before executing the bounded projection-level replay hook or any production QK256 promotion"
            }
        },
        "must_not_claim": [
            "CPU/A770 answer parity is proven",
            "Reference parity is proven",
            "Strict A770 answer readiness is proven",
            "Broad A770 answer quality is proven",
            "Official BitNet QK256 production semantics are proven",
            "Production QK256 dispatch policy changed",
            "BitNet inference works on A770",
            "A770 trusted partial acceleration is claim-grade",
            "Full A770 residency is proven",
            "A770 performance speedup is proven"
        ]
    });

    Ok((
        serde_json::to_string_pretty(&receipt["manifest"])? + "\n",
        serde_json::to_string_pretty(&receipt)? + "\n",
    ))
}

#[derive(Debug, Clone, Default)]
struct ProjectionRowEvidence {
    available: bool,
    source_target_id: Option<String>,
    dispatch_replay_source: Option<String>,
    executed: bool,
    selected_backend: Option<String>,
    runtime_api: Option<String>,
    runtime_device: Option<String>,
    fallback_used: Option<bool>,
    source_output_index: Option<u64>,
    focused_device_output_bits: Option<u64>,
    production_output_bits: Option<u64>,
    production_output_matches_selected_device_bits: bool,
}

impl ProjectionRowEvidence {
    fn clean_for_projection_boundary(&self) -> bool {
        self.available
            && self.executed
            && self.fallback_used == Some(false)
            && self.production_output_matches_selected_device_bits
    }
}

#[derive(Debug, Clone)]
struct ProjectionReplayOperands {
    input_row_index: usize,
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
    activation_sum: i32,
    activation_scale: f32,
    activation_scale_bits: u32,
    weight_scale: f32,
    weight_scale_bits: u32,
    activations_i8: Vec<i8>,
    packed_qk256: Vec<u8>,
}

#[derive(Debug, Clone)]
struct ProjectionOperandCaptureEvidence {
    replay_operands: Option<ProjectionReplayOperands>,
    source_path: Option<String>,
    source_json_parseable: Option<bool>,
    source_projection_found: bool,
    source_context_available: bool,
    dispatch_replay_available: bool,
    focused_operands_available: bool,
    full_projection_operands_available: bool,
    current_operand_scope: String,
    required_operand_scope: &'static str,
    target_layer_idx: Option<i64>,
    projection: Option<String>,
    input_rows_materialized_count: Option<u64>,
    output_rows_allocated_count: Option<u64>,
    input_rows: Option<usize>,
    output_rows: Option<usize>,
    cols: Option<usize>,
    row_stride_bytes: Option<usize>,
    activation_i8_len: Option<usize>,
    packed_qk256_len: Option<usize>,
    packed_qk256_scope: Option<String>,
    required_packed_qk256_len: Option<usize>,
    required_packed_qk256_len_available: Option<bool>,
    missing_full_operand_fields: Vec<&'static str>,
    blockers: Vec<&'static str>,
    error: Option<String>,
}

impl ProjectionOperandCaptureEvidence {
    fn blocked(
        source_path: Option<String>,
        source_json_parseable: Option<bool>,
        blocker: &'static str,
        missing_full_operand_fields: Vec<&'static str>,
        error: Option<String>,
    ) -> Self {
        Self {
            replay_operands: None,
            source_path,
            source_json_parseable,
            source_projection_found: false,
            source_context_available: false,
            dispatch_replay_available: false,
            focused_operands_available: false,
            full_projection_operands_available: false,
            current_operand_scope: "projection_operand_source_unavailable".to_owned(),
            required_operand_scope: "full_projection_output_rows",
            target_layer_idx: None,
            projection: None,
            input_rows_materialized_count: None,
            output_rows_allocated_count: None,
            input_rows: None,
            output_rows: None,
            cols: None,
            row_stride_bytes: None,
            activation_i8_len: None,
            packed_qk256_len: None,
            packed_qk256_scope: None,
            required_packed_qk256_len: None,
            required_packed_qk256_len_available: None,
            missing_full_operand_fields,
            blockers: vec!["projection_level_full_operands_missing", blocker],
            error,
        }
    }

    fn missing_full_operand_fields(&self) -> Vec<&'static str> {
        if self.missing_full_operand_fields.is_empty() {
            vec!["projection_operands"]
        } else {
            self.missing_full_operand_fields.clone()
        }
    }

    fn blockers(&self) -> Vec<&'static str> {
        if self.blockers.is_empty() {
            vec!["projection_level_full_operands_missing"]
        } else {
            self.blockers.clone()
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "source_path": self.source_path,
            "source_json_parseable": self.source_json_parseable,
            "source_projection_found": self.source_projection_found,
            "source_context_available": self.source_context_available,
            "dispatch_replay_available": self.dispatch_replay_available,
            "focused_operands_available": self.focused_operands_available,
            "full_projection_operands_available": self.full_projection_operands_available,
            "current_operand_scope": self.current_operand_scope,
            "required_operand_scope": self.required_operand_scope,
            "target_layer_idx": self.target_layer_idx,
            "projection": self.projection,
            "input_rows_materialized_count": self.input_rows_materialized_count,
            "output_rows_allocated_count": self.output_rows_allocated_count,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "cols": self.cols,
            "row_stride_bytes": self.row_stride_bytes,
            "activation_i8_len": self.activation_i8_len,
            "packed_qk256_len": self.packed_qk256_len,
            "packed_qk256_scope": self.packed_qk256_scope,
            "required_packed_qk256_len": self.required_packed_qk256_len,
            "required_packed_qk256_len_available": self.required_packed_qk256_len_available,
            "missing_full_operand_fields": self.missing_full_operand_fields,
            "blockers": self.blockers,
            "error": self.error
        })
    }
}

#[derive(Debug, Clone)]
enum ProjectionReplayOutcome {
    Executed {
        operands: ProjectionReplayOperands,
        replay: A770OpenClQk256ProductionReplayResult,
    },
    Blocked {
        reason: &'static str,
        blockers: Vec<&'static str>,
        missing_full_operand_fields: Vec<&'static str>,
        current_operand_scope: String,
    },
    Failed {
        error: String,
        blockers: Vec<&'static str>,
    },
}

fn projection_source_targets<'a>(
    source: &'a Value,
) -> Result<(Vec<&'a Value>, &'static str), Box<dyn Error>> {
    if let Some(targets) = source.pointer("/manifest/targets").and_then(Value::as_array) {
        let kind = match str_field(source, "work_item") {
            Some("A770-159") => "a770_159_full_projection_operand_source_boundary",
            Some("A770-158") => "a770_158_projection_replay_hook_boundary",
            _ => "a770_157_projection_level_qkv_boundary",
        };
        return Ok((targets.iter().collect(), kind));
    }
    if let Some(targets) = source.get("target_results").and_then(Value::as_array) {
        return Ok((targets.iter().collect(), "a770_156_focused_qkv_replay"));
    }
    Err(io_error("projection source missing manifest.targets or target_results array"))
}

fn projection_row_evidence(row: Option<&Value>) -> ProjectionRowEvidence {
    let Some(target) = row else {
        return ProjectionRowEvidence::default();
    };
    if let Some(row_evidence) = target.get("row_evidence") {
        return ProjectionRowEvidence {
            available: true,
            source_target_id: string_field(row_evidence, "source_target_id")
                .or_else(|| string_field(target, "target_id")),
            dispatch_replay_source: string_field(row_evidence, "dispatch_replay_source"),
            executed: row_evidence.get("executed").and_then(Value::as_bool).unwrap_or(false),
            selected_backend: string_field(row_evidence, "selected_backend"),
            runtime_api: string_field(row_evidence, "runtime_api"),
            runtime_device: string_field(row_evidence, "runtime_device"),
            fallback_used: row_evidence.get("fallback_used").and_then(Value::as_bool),
            source_output_index: u64_field(row_evidence, "source_output_index"),
            focused_device_output_bits: u64_field(row_evidence, "focused_device_output_bits"),
            production_output_bits: u64_field(row_evidence, "production_output_bits"),
            production_output_matches_selected_device_bits: row_evidence
                .get("production_output_matches_selected_device_bits")
                .and_then(Value::as_bool)
                == Some(true),
        };
    }
    let replay = target.get("production_replay");
    ProjectionRowEvidence {
        available: replay.is_some(),
        source_target_id: string_field(target, "target_id"),
        dispatch_replay_source: string_field(target, "dispatch_replay_source"),
        executed: replay
            .and_then(|value| value.get("executed"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        selected_backend: replay.and_then(|value| string_field(value, "selected_backend")),
        runtime_api: replay.and_then(|value| string_field(value, "runtime_api")),
        runtime_device: replay.and_then(|value| string_field(value, "runtime_device")),
        fallback_used: replay.and_then(|value| value.get("fallback_used")).and_then(Value::as_bool),
        source_output_index: replay.and_then(|value| u64_field(value, "source_output_index")),
        focused_device_output_bits: replay
            .and_then(|value| u64_field(value, "focused_device_output_bits")),
        production_output_bits: replay.and_then(|value| u64_field(value, "production_output_bits")),
        production_output_matches_selected_device_bits: replay
            .and_then(|value| value.get("production_output_matches_selected_device_bits"))
            .and_then(Value::as_bool)
            == Some(true),
    }
}

fn projection_dispatch_replay_source(target: &Value) -> Option<String> {
    target
        .get("row_evidence")
        .and_then(|row_evidence| string_field(row_evidence, "dispatch_replay_source"))
        .or_else(|| string_field(target, "dispatch_replay_source"))
}

fn projection_operand_capture_evidence(
    row: Option<&Value>,
    case_id: &str,
    first_mismatch_index: usize,
    projection_layer: i64,
    projection: &str,
    source_receipt_kind: &'static str,
) -> ProjectionOperandCaptureEvidence {
    let source_path = row.and_then(projection_dispatch_replay_source);
    let Some(source_path) = source_path else {
        return ProjectionOperandCaptureEvidence::blocked(
            None,
            None,
            "projection_level_dispatch_replay_source_missing",
            vec!["projection_operands"],
            None,
        );
    };
    let source_json = match fs::read_to_string(&source_path) {
        Ok(source_json) => source_json,
        Err(err) => {
            return ProjectionOperandCaptureEvidence::blocked(
                Some(source_path),
                Some(false),
                "projection_level_dispatch_replay_source_unreadable",
                vec!["projection_operands"],
                Some(err.to_string()),
            );
        }
    };
    let source: Value = match serde_json::from_str(&source_json) {
        Ok(source) => source,
        Err(err) => {
            return ProjectionOperandCaptureEvidence::blocked(
                Some(source_path),
                Some(false),
                "projection_level_dispatch_replay_source_parse_failed",
                vec!["projection_operands"],
                Some(err.to_string()),
            );
        }
    };
    let Some(source_target) = projection_capture_source_target(
        &source,
        case_id,
        first_mismatch_index,
        projection_layer,
        projection,
    ) else {
        return ProjectionOperandCaptureEvidence::blocked(
            Some(source_path),
            Some(true),
            "projection_level_qkv_projection_source_missing",
            vec!["projection_operands"],
            None,
        );
    };

    let dispatch_replay = source_target.get("dispatch_replay").filter(|value| value.is_object());
    let focused_operands =
        dispatch_replay.and_then(|value| value.get("focused_operands")).filter(|value| {
            value.is_object()
                && any_array_at(Some(value), &[&["activations_i8"]])
                && any_array_at(Some(value), &[&["packed_qk256"]])
        });
    let full_operand_root = projection_operand_root(source_target)
        .or_else(|| dispatch_replay.and_then(projection_operand_root));
    let full_projection_replay_operands = full_operand_root.and_then(|_| {
        projection_replay_operands(source_target).ok().flatten().or_else(|| {
            dispatch_replay.and_then(|value| projection_replay_operands(value).ok().flatten())
        })
    });
    let activation_i8_len = full_operand_root
        .and_then(|root| {
            array_len_at(
                Some(root),
                &[&["activations_i8"], &["activation_i8"], &["input_activations_i8"]],
            )
        })
        .or_else(|| {
            focused_operands.and_then(|root| array_len_at(Some(root), &[&["activations_i8"]]))
        });
    let packed_qk256_len = full_operand_root
        .and_then(|root| {
            array_len_at(
                Some(root),
                &[&["packed_qk256"], &["weights_packed_qk256"], &["packed_qk256_weights"]],
            )
        })
        .or_else(|| {
            focused_operands.and_then(|root| array_len_at(Some(root), &[&["packed_qk256"]]))
        });
    let cols = full_operand_root
        .and_then(|root| usize_field(root, "cols"))
        .or_else(|| focused_operands.and_then(|root| usize_field(root, "cols")))
        .or_else(|| dispatch_replay.and_then(|root| usize_field(root, "cols")));
    let row_stride_bytes = full_operand_root
        .and_then(|root| usize_field(root, "row_stride_bytes"))
        .or_else(|| focused_operands.and_then(|root| usize_field(root, "row_stride_bytes")))
        .or_else(|| dispatch_replay.and_then(|root| usize_field(root, "row_stride_bytes")));
    let input_rows = full_operand_root
        .and_then(|root| usize_field(root, "input_rows"))
        .or_else(|| dispatch_replay.and_then(|root| usize_field(root, "input_rows")));
    let output_rows = full_operand_root
        .and_then(|root| usize_field(root, "rows").or_else(|| usize_field(root, "output_rows")))
        .or_else(|| dispatch_replay.and_then(|root| usize_field(root, "output_rows")));
    let required_packed_qk256_len =
        output_rows.zip(row_stride_bytes).and_then(|(rows, stride)| rows.checked_mul(stride));
    let required_packed_qk256_len_available = required_packed_qk256_len
        .zip(packed_qk256_len)
        .map(|(required, available)| available >= required);
    let full_projection_operands_available = full_operand_root.is_some_and(|root| {
        projection_missing_full_operand_fields_from_root(root).is_empty()
            && required_packed_qk256_len_available != Some(false)
    }) && full_projection_replay_operands.is_some();
    let packed_qk256_scope =
        focused_operands.and_then(|root| string_field(root, "packed_qk256_scope")).or_else(|| {
            dispatch_replay
                .and_then(|root| string_at(Some(root), &["focused_operands", "packed_qk256_scope"]))
        });
    let current_operand_scope = if full_projection_operands_available {
        "full_projection_output_rows".to_owned()
    } else {
        packed_qk256_scope
            .clone()
            .unwrap_or_else(|| "projection_source_without_full_operands".to_owned())
    };
    let missing_full_operand_fields = if full_projection_operands_available {
        Vec::new()
    } else if let Some(root) = full_operand_root {
        let mut missing = projection_missing_full_operand_fields_from_root(root);
        if required_packed_qk256_len_available == Some(false)
            && !missing.contains(&"projection_operands.packed_qk256_full_projection_rows")
        {
            missing.push("projection_operands.packed_qk256_full_projection_rows");
        }
        if missing.is_empty() {
            missing.push("projection_operands.full_projection_output_rows");
        }
        missing
    } else if required_packed_qk256_len_available == Some(false) {
        vec!["projection_operands.packed_qk256_full_projection_rows"]
    } else if focused_operands.is_some() {
        vec!["projection_operands.full_projection_output_rows"]
    } else {
        vec!["projection_operands"]
    };
    let mut blockers = if full_projection_operands_available {
        Vec::new()
    } else if dispatch_replay.is_none() {
        vec!["projection_level_full_operands_missing", "projection_level_dispatch_replay_missing"]
    } else if focused_operands.is_none() {
        vec![
            "projection_level_full_operands_missing",
            "projection_level_focused_operand_source_missing",
        ]
    } else if required_packed_qk256_len_available == Some(false) {
        vec![
            "projection_level_full_operands_missing",
            "projection_level_full_projection_weight_rows_missing",
        ]
    } else {
        vec!["projection_level_full_operands_missing"]
    };
    if source_receipt_kind == "a770_159_full_projection_operand_source_boundary"
        && !full_projection_operands_available
        && !blockers.contains(&"projection_level_full_projection_packed_row_capture_source_missing")
    {
        blockers.push("projection_level_full_projection_packed_row_capture_source_missing");
    }

    ProjectionOperandCaptureEvidence {
        replay_operands: full_projection_replay_operands,
        source_path: Some(source_path),
        source_json_parseable: Some(true),
        source_projection_found: true,
        source_context_available: source_target
            .get("source_context_available")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        dispatch_replay_available: dispatch_replay.is_some(),
        focused_operands_available: focused_operands.is_some(),
        full_projection_operands_available,
        current_operand_scope,
        required_operand_scope: "full_projection_output_rows",
        target_layer_idx: i64_field(source_target, "target_layer_idx")
            .or_else(|| i64_field(source_target, "layer_idx")),
        projection: string_field(source_target, "projection"),
        input_rows_materialized_count: source_target
            .pointer("/cpu_hot_path_delta/input_rows_materialized_count")
            .and_then(Value::as_u64),
        output_rows_allocated_count: source_target
            .pointer("/cpu_hot_path_delta/output_rows_allocated_count")
            .and_then(Value::as_u64),
        input_rows,
        output_rows,
        cols,
        row_stride_bytes,
        activation_i8_len,
        packed_qk256_len,
        packed_qk256_scope,
        required_packed_qk256_len,
        required_packed_qk256_len_available,
        missing_full_operand_fields,
        blockers,
        error: None,
    }
}

fn projection_capture_source_target<'a>(
    source: &'a Value,
    case_id: &str,
    first_mismatch_index: usize,
    projection_layer: i64,
    projection: &str,
) -> Option<&'a Value> {
    let case = source
        .get("cases")
        .and_then(Value::as_array)?
        .iter()
        .find(|case| str_field(case, "id") == Some(case_id))?;
    let logits_dump = case.get("logits_dump").and_then(Value::as_array)?;
    let step = logits_dump.get(first_mismatch_index).or_else(|| {
        logits_dump.iter().find(|step| usize_field(step, "step") == Some(first_mismatch_index))
    })?;
    let sources = step
        .pointer("/logit_source_context/hidden_state_source/model_forward_source/qkv_projection_sources/sources")
        .and_then(Value::as_array)?;
    sources.iter().find(|source| {
        str_field(source, "projection") == Some(projection)
            && i64_field(source, "target_layer_idx").or_else(|| i64_field(source, "layer_idx"))
                == Some(projection_layer)
    })
}

fn projection_replay_outcome(
    row: Option<&Value>,
    capture_evidence: &ProjectionOperandCaptureEvidence,
) -> ProjectionReplayOutcome {
    let Some(target) = row else {
        return ProjectionReplayOutcome::Blocked {
            reason: "projection_level_row_evidence_not_clean",
            blockers: vec!["projection_level_row_evidence_not_clean"],
            missing_full_operand_fields: Vec::new(),
            current_operand_scope: "row_evidence_missing".to_owned(),
        };
    };
    let operands = match projection_replay_operands(target) {
        Ok(Some(operands)) => Some(operands),
        Ok(None) => capture_evidence.replay_operands.clone(),
        Err(err) => {
            return ProjectionReplayOutcome::Failed {
                error: err.to_string(),
                blockers: vec!["projection_level_replay_operand_error"],
            };
        }
    };
    let Some(operands) = operands else {
        let has_embedded_operands = projection_operand_root(target).is_some();
        return ProjectionReplayOutcome::Blocked {
            reason: "projection_level_full_operands_missing",
            blockers: if has_embedded_operands {
                vec!["projection_level_full_operands_missing"]
            } else {
                capture_evidence.blockers()
            },
            missing_full_operand_fields: if has_embedded_operands {
                projection_missing_full_operand_fields(target)
            } else {
                capture_evidence.missing_full_operand_fields()
            },
            current_operand_scope: if has_embedded_operands {
                projection_current_operand_scope(target)
            } else {
                capture_evidence.current_operand_scope.clone()
            },
        };
    };
    run_projection_replay_operands(operands)
}

fn run_projection_replay_operands(operands: ProjectionReplayOperands) -> ProjectionReplayOutcome {
    match run_a770_qk256_i8s_scaled_gemv_production_replay(A770OpenClQk256ProductionReplay {
        activations_i8: &operands.activations_i8,
        packed_qk256: &operands.packed_qk256,
        rows: operands.rows,
        cols: operands.cols,
        row_stride_bytes: operands.row_stride_bytes,
        activation_sum: operands.activation_sum,
        activation_scale: operands.activation_scale,
        weight_scale: operands.weight_scale,
        sample_limit: projection_replay_sample_limit(operands.rows),
    }) {
        Ok(replay) => ProjectionReplayOutcome::Executed { operands, replay },
        Err(err) => ProjectionReplayOutcome::Failed {
            error: err.to_string(),
            blockers: vec!["projection_level_replay_failed"],
        },
    }
}

fn projection_replay_operands(
    target: &Value,
) -> Result<Option<ProjectionReplayOperands>, Box<dyn Error>> {
    let Some(operands_root) = projection_operand_root(target) else {
        return Ok(None);
    };
    if !projection_missing_full_operand_fields_from_root(operands_root).is_empty() {
        return Ok(None);
    }

    let activations_i8 = i8_array_at(
        Some(operands_root),
        &[&["activations_i8"], &["activation_i8"], &["input_activations_i8"]],
    )?
    .ok_or_else(|| io_error("projection replay operands missing activations_i8"))?;
    let packed_qk256 = u8_array_at(
        Some(operands_root),
        &[&["packed_qk256"], &["weights_packed_qk256"], &["packed_qk256_weights"]],
    )?
    .ok_or_else(|| io_error("projection replay operands missing packed_qk256"))?;
    let rows = usize_field(operands_root, "rows")
        .or_else(|| usize_field(operands_root, "output_rows"))
        .ok_or_else(|| io_error("projection replay operands missing rows"))?;
    let cols = usize_field(operands_root, "cols")
        .ok_or_else(|| io_error("projection replay operands missing cols"))?;
    let row_stride_bytes = usize_field(operands_root, "row_stride_bytes")
        .ok_or_else(|| io_error("projection replay operands missing row_stride_bytes"))?;
    if rows == 0 {
        return Err(io_error("projection replay operands rows must be non-zero"));
    }
    if activations_i8.len() < cols {
        return Err(io_error(format!(
            "projection replay activation length {} < cols {cols}",
            activations_i8.len()
        )));
    }
    let expected_packed_len = rows * row_stride_bytes;
    if packed_qk256.len() < expected_packed_len {
        return Err(io_error(format!(
            "projection replay packed QK256 length {} < rows * row_stride_bytes {expected_packed_len}",
            packed_qk256.len()
        )));
    }

    let activation_sum = i64_field(operands_root, "activation_sum")
        .ok_or_else(|| io_error("projection replay operands missing activation_sum"))?;
    let activation_sum = i32::try_from(activation_sum)
        .map_err(|_| io_error("projection replay operands activation_sum outside i32 range"))?;
    let activation_scale_bits = u32_field(operands_root, "activation_scale_bits")
        .ok_or_else(|| io_error("projection replay operands missing activation_scale_bits"))?;
    let weight_scale_bits = u32_field(operands_root, "weight_scale_bits")
        .ok_or_else(|| io_error("projection replay operands missing weight_scale_bits"))?;

    Ok(Some(ProjectionReplayOperands {
        input_row_index: usize_field(operands_root, "input_row_index").unwrap_or(0),
        rows,
        cols,
        row_stride_bytes,
        activation_sum,
        activation_scale: f32::from_bits(activation_scale_bits),
        activation_scale_bits,
        weight_scale: f32::from_bits(weight_scale_bits),
        weight_scale_bits,
        activations_i8,
        packed_qk256,
    }))
}

fn projection_operand_root(target: &Value) -> Option<&Value> {
    target
        .pointer("/projection_replay/full_projection_operands")
        .or_else(|| target.pointer("/projection_replay/operands"))
        .or_else(|| target.get("full_projection_operands"))
        .or_else(|| target.get("projection_operands"))
        .or_else(|| target.get("operands"))
}

fn projection_current_operand_scope(target: &Value) -> String {
    target
        .pointer("/projection_replay/current_operand_scope")
        .and_then(Value::as_str)
        .unwrap_or("single_focused_output_row")
        .to_owned()
}

fn projection_missing_full_operand_fields(target: &Value) -> Vec<&'static str> {
    projection_operand_root(target).map_or_else(
        || vec!["projection_operands"],
        projection_missing_full_operand_fields_from_root,
    )
}

fn projection_missing_full_operand_fields_from_root(root: &Value) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if !any_array_at(
        Some(root),
        &[&["activations_i8"], &["activation_i8"], &["input_activations_i8"]],
    ) {
        missing.push("activations_i8");
    }
    if !any_array_at(
        Some(root),
        &[&["packed_qk256"], &["weights_packed_qk256"], &["packed_qk256_weights"]],
    ) {
        missing.push("packed_qk256");
    }
    if usize_field(root, "rows").or_else(|| usize_field(root, "output_rows")).is_none() {
        missing.push("rows");
    }
    if usize_field(root, "cols").is_none() {
        missing.push("cols");
    }
    if usize_field(root, "row_stride_bytes").is_none() {
        missing.push("row_stride_bytes");
    }
    if i64_field(root, "activation_sum").is_none() {
        missing.push("activation_sum");
    }
    if u32_field(root, "activation_scale_bits").is_none() {
        missing.push("activation_scale_bits");
    }
    if u32_field(root, "weight_scale_bits").is_none() {
        missing.push("weight_scale_bits");
    }
    missing
}

fn projection_replay_sample_limit(rows: usize) -> usize {
    rows.min(PROJECTION_REPLAY_SAMPLE_LIMIT).max(1)
}

fn push_unique_blockers(target: &mut Vec<&'static str>, blockers: &[&'static str]) {
    for blocker in blockers {
        if !target.contains(blocker) {
            target.push(*blocker);
        }
    }
}

#[derive(Debug, Clone)]
struct FocusedContext {
    case_found: bool,
    summary_first_divergence_matches_request: bool,
    qkv_context_available: bool,
    raw_activation_i8_available: bool,
    raw_packed_qk256_available: bool,
    summary_qk256_trace_available: bool,
    device_expression_trace_available: bool,
    device_intermediate_trace_available: bool,
    runtime_device: Option<String>,
    platform_index: Option<u64>,
    device_index: Option<u64>,
    platform_name: Option<String>,
    vendor: Option<String>,
    driver_version: Option<String>,
    target_layer_idx: Option<i64>,
    projection: Option<String>,
    input_rows: Option<u64>,
    output_rows: Option<u64>,
    cols: Option<u64>,
    row_stride_bytes: Option<u64>,
    input_row_index: Option<u64>,
    sample_count: Option<u64>,
    sample_limit: Option<u64>,
    focused_output_index: Option<u64>,
    activation_sum: Option<i64>,
    activation_scale_bits: Option<u64>,
    weight_scale_bits: Option<u64>,
    int_dot: Option<i64>,
    adjusted_dot: Option<i64>,
    focused_device_output_bits: Option<u64>,
    focused_policy_bits: Option<u64>,
    host_summary_policy_semantic_fix_applied: Option<bool>,
    host_summary_policy_semantic_fix_bits: Option<u64>,
    host_policy_div_then_mul_bits: Option<u64>,
    host_policy_mul_then_div_bits: Option<u64>,
    host_policy_reciprocal_then_mul_bits: Option<u64>,
    host_policy_f64_div_then_mul_cast_bits: Option<u64>,
    device_intermediate_classification: Option<String>,
    device_expression_classification: Option<String>,
    production_policy_change_justified: Option<bool>,
    production_replay_skipped_reason: &'static str,
    missing_raw_operand_fields: Vec<&'static str>,
    next_diagnostic: &'static str,
}

#[derive(Debug, Clone)]
struct FocusedRawOperands {
    input_row_index: usize,
    output_index: usize,
    cols: usize,
    row_stride_bytes: usize,
    activation_sum: i32,
    activation_scale: f32,
    activation_scale_bits: u32,
    weight_scale: f32,
    weight_scale_bits: u32,
    activations_i8: Vec<i8>,
    packed_qk256: Vec<u8>,
}

#[derive(Debug, Clone)]
struct FocusedScalarOracle {
    expected_row_stride_bytes: usize,
    row_stride_matches_expected: bool,
    activation_sum_from_raw: i32,
    activation_sum_matches_receipt: bool,
    int_dot: i32,
    adjusted_dot: i32,
    adjusted_f32_bits: u32,
    reciprocal_activation_scale_bits: u32,
    adjusted_mul_reciprocal_bits: u32,
    final_scaled_value_bits: u32,
    div_then_mul_bits: u32,
    weight_over_activation_bits: u32,
    reciprocal_then_mul_bits: u32,
    used_payload_bytes: usize,
    row_padding_bytes: usize,
    unused_tail_columns: usize,
}

#[derive(Debug, Clone)]
enum FocusedReplayOutcome {
    Executed { operands: FocusedRawOperands, replay: A770OpenClQk256ProductionReplayResult },
    Failed { error: String },
}

fn focused_replay_outcome(
    source: &Value,
    case_id: &str,
    first_mismatch_index: usize,
) -> Option<FocusedReplayOutcome> {
    let operands = match focused_raw_operands(source, case_id, first_mismatch_index) {
        Ok(Some(operands)) => operands,
        Ok(None) => return None,
        Err(err) => return Some(FocusedReplayOutcome::Failed { error: err.to_string() }),
    };
    Some(run_focused_raw_operands_replay(operands))
}

fn run_focused_raw_operands_replay(operands: FocusedRawOperands) -> FocusedReplayOutcome {
    match run_a770_qk256_i8s_scaled_gemv_production_replay(A770OpenClQk256ProductionReplay {
        activations_i8: &operands.activations_i8,
        packed_qk256: &operands.packed_qk256,
        rows: 1,
        cols: operands.cols,
        row_stride_bytes: operands.row_stride_bytes,
        activation_sum: operands.activation_sum,
        activation_scale: operands.activation_scale,
        weight_scale: operands.weight_scale,
        sample_limit: 1,
    }) {
        Ok(replay) => FocusedReplayOutcome::Executed { operands, replay },
        Err(err) => FocusedReplayOutcome::Failed { error: err.to_string() },
    }
}

fn focused_raw_operands(
    source: &Value,
    case_id: &str,
    first_mismatch_index: usize,
) -> Result<Option<FocusedRawOperands>, Box<dyn Error>> {
    let qkv_row = find_row(
        source.pointer("/generated_output_qkv_projection_dispatch_replay_frontier/rows"),
        case_id,
        first_mismatch_index,
    );
    let right_replay = match qkv_row.and_then(|row| row.get("right_replay")) {
        Some(value) => value,
        None => return Ok(None),
    };
    focused_raw_operands_from_replay(right_replay)
}

fn focused_raw_operands_from_replay(
    right_replay: &Value,
) -> Result<Option<FocusedRawOperands>, Box<dyn Error>> {
    let operands_root = right_replay
        .get("focused_operands")
        .or_else(|| right_replay.get("operands"))
        .unwrap_or(right_replay);
    let activations_i8 = match i8_array_at(
        Some(right_replay),
        &[
            &["focused_operands", "activations_i8"],
            &["operands", "activations_i8"],
            &["activations_i8"],
        ],
    )? {
        Some(values) => values,
        None => return Ok(None),
    };
    let packed_qk256 = match u8_array_at(
        Some(right_replay),
        &[&["focused_operands", "packed_qk256"], &["operands", "packed_qk256"], &["packed_qk256"]],
    )? {
        Some(values) => values,
        None => return Ok(None),
    };

    let cols = usize_field(operands_root, "cols")
        .or_else(|| usize_field(right_replay, "cols"))
        .ok_or_else(|| io_error("focused raw operands missing cols"))?;
    let row_stride_bytes = usize_field(operands_root, "row_stride_bytes")
        .or_else(|| usize_field(right_replay, "row_stride_bytes"))
        .ok_or_else(|| io_error("focused raw operands missing row_stride_bytes"))?;
    if activations_i8.len() != cols {
        return Err(io_error(format!(
            "focused raw operands activation length {} does not match cols {cols}",
            activations_i8.len()
        )));
    }
    if packed_qk256.len() != row_stride_bytes {
        return Err(io_error(format!(
            "focused packed QK256 length {} does not match row_stride_bytes {row_stride_bytes}",
            packed_qk256.len()
        )));
    }

    let input_row_index = usize_field(operands_root, "input_row_index").unwrap_or(0);
    let output_index = usize_field(operands_root, "output_index").unwrap_or(0);
    let activation_sum = i64_field(operands_root, "activation_sum")
        .or_else(|| {
            right_replay
                .pointer("/device_intermediate_trace/samples/0/activation_sum")
                .and_then(Value::as_i64)
        })
        .ok_or_else(|| io_error("focused raw operands missing activation_sum"))?;
    let activation_sum = i32::try_from(activation_sum)
        .map_err(|_| io_error("focused raw operands activation_sum outside i32 range"))?;
    let activation_scale_bits = u32_field(operands_root, "activation_scale_bits")
        .or_else(|| {
            right_replay
                .pointer("/device_intermediate_trace/samples/0/activation_scale_bits")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
        })
        .ok_or_else(|| io_error("focused raw operands missing activation_scale_bits"))?;
    let weight_scale_bits = u32_field(operands_root, "weight_scale_bits")
        .or_else(|| {
            right_replay
                .pointer("/device_intermediate_trace/samples/0/weight_scale_bits")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
        })
        .ok_or_else(|| io_error("focused raw operands missing weight_scale_bits"))?;

    Ok(Some(FocusedRawOperands {
        input_row_index,
        output_index,
        cols,
        row_stride_bytes,
        activation_sum,
        activation_scale: f32::from_bits(activation_scale_bits),
        activation_scale_bits,
        weight_scale: f32::from_bits(weight_scale_bits),
        weight_scale_bits,
        activations_i8,
        packed_qk256,
    }))
}

fn focused_raw_operand_summary_json(outcome: Option<&FocusedReplayOutcome>) -> Value {
    match outcome {
        Some(FocusedReplayOutcome::Executed { operands, .. }) => json!({
            "available": true,
            "input_row_index": operands.input_row_index,
            "output_index": operands.output_index,
            "rows": 1,
            "cols": operands.cols,
            "row_stride_bytes": operands.row_stride_bytes,
            "activation_sum": operands.activation_sum,
            "activation_scale_bits": operands.activation_scale_bits,
            "weight_scale_bits": operands.weight_scale_bits,
            "activation_i8_len": operands.activations_i8.len(),
            "packed_qk256_len": operands.packed_qk256.len(),
            "packed_qk256_scope": "single_output_row",
        }),
        Some(FocusedReplayOutcome::Failed { error }) => json!({
            "available": false,
            "error": error,
        }),
        None => json!({
            "available": false,
            "reason": "focused_raw_operands_missing",
        }),
    }
}

fn focused_production_replay_summary_json(
    outcome: Option<&FocusedReplayOutcome>,
    context: &FocusedContext,
) -> Value {
    match outcome {
        Some(FocusedReplayOutcome::Executed { operands, replay }) => {
            let first_sample = replay.samples.first();
            let production_output_matches_focused_device_bits =
                match (first_sample, context.focused_device_output_bits) {
                    (Some(sample), Some(bits)) => {
                        Some(u64::from(sample.production_output_bits) == bits)
                    }
                    _ => None,
                };
            json!({
                "executed": true,
                "input_row_index": operands.input_row_index,
                "source_output_index": operands.output_index,
                "replay_output_index": first_sample.map(|sample| sample.output_index),
                "sample_count": replay.samples.len(),
                "focused_device_output_bits": context.focused_device_output_bits,
                "production_output_bits": first_sample.map(|sample| sample.production_output_bits),
                "replay_output_bits": first_sample.map(|sample| sample.replay_output_bits),
                "final_scaled_value_bits": first_sample.map(|sample| sample.final_scaled_value_bits),
                "output_store_matches_replay_output": first_sample.map(|sample| sample.output_store_matches_replay_output),
                "output_store_matches_final_scaled_value": first_sample.map(|sample| sample.output_store_matches_final_scaled_value),
                "production_output_matches_focused_device_bits": production_output_matches_focused_device_bits,
            })
        }
        Some(FocusedReplayOutcome::Failed { error }) => json!({
            "executed": false,
            "error": error,
        }),
        None => json!({
            "executed": false,
            "reason": "focused_raw_operands_missing",
        }),
    }
}

fn focused_production_replay_samples_json(outcome: Option<&FocusedReplayOutcome>) -> Value {
    match outcome {
        Some(FocusedReplayOutcome::Executed { replay, .. }) => {
            Value::Array(replay.samples.iter().map(focused_replay_sample_json).collect())
        }
        _ => Value::Array(Vec::new()),
    }
}

fn focused_replay_sample_json(sample: &A770OpenClQk256ProductionReplaySample) -> Value {
    json!({
        "output_index": sample.output_index,
        "int_dot": sample.int_dot,
        "activation_sum": sample.activation_sum,
        "adjusted_dot": sample.adjusted_dot,
        "activation_scale_bits": sample.activation_scale_bits,
        "weight_scale_bits": sample.weight_scale_bits,
        "adjusted_f32_bits": sample.adjusted_f32_bits,
        "reciprocal_activation_scale_bits": sample.reciprocal_activation_scale_bits,
        "adjusted_mul_reciprocal_bits": sample.adjusted_mul_reciprocal_bits,
        "final_scaled_value_bits": sample.final_scaled_value_bits,
        "div_then_mul_bits": sample.div_then_mul_bits,
        "weight_over_activation_bits": sample.weight_over_activation_bits,
        "reciprocal_then_mul_bits": sample.reciprocal_then_mul_bits,
        "replay_output_bits": sample.replay_output_bits,
        "production_output_bits": sample.production_output_bits,
        "output_store_matches_replay_output": sample.output_store_matches_replay_output,
        "output_store_matches_final_scaled_value": sample.output_store_matches_final_scaled_value,
    })
}

fn focused_context(source: &Value, case_id: &str, first_mismatch_index: usize) -> FocusedContext {
    let case_found = source
        .get("cases")
        .and_then(Value::as_array)
        .is_some_and(|cases| cases.iter().any(|case| str_field(case, "id") == Some(case_id)));
    let summary_first_divergence_matches_request =
        source.pointer("/summary/first_divergence").is_some_and(|divergence| {
            str_field(divergence, "case_id") == Some(case_id)
                && usize_field(divergence, "step") == Some(first_mismatch_index)
        }) || source
            .pointer("/generated_output_frontier/rows")
            .and_then(Value::as_array)
            .is_some_and(|rows| {
                rows.iter().any(|row| {
                    str_field(row, "case_id") == Some(case_id)
                        && usize_field(row, "first_mismatch_index") == Some(first_mismatch_index)
                })
            });
    let qkv_row = find_row(
        source.pointer("/generated_output_qkv_projection_dispatch_replay_frontier/rows"),
        case_id,
        first_mismatch_index,
    );
    let right_replay = qkv_row.and_then(|row| row.get("right_replay"));
    let device_intermediate_trace =
        right_replay.and_then(|value| value.get("device_intermediate_trace"));
    let device_expression_trace =
        right_replay.and_then(|value| value.get("device_expression_trace"));
    let device_sample = device_intermediate_trace
        .and_then(|trace| trace.get("samples"))
        .and_then(Value::as_array)
        .and_then(|samples| samples.first());
    let expression_sample = device_expression_trace
        .and_then(|trace| trace.get("samples"))
        .and_then(Value::as_array)
        .and_then(|samples| samples.first());
    let qk256_expression_row = find_case_row(
        source.pointer("/generated_output_qk256_device_expression_frontier/rows"),
        case_id,
    );
    let host_summary_policy_semantic_fix = expression_sample
        .and_then(|sample| sample.get("host_summary_policy_semantic_fix"))
        .or_else(|| {
            qk256_expression_row.and_then(|row| {
                row.pointer("/right/host_summary_policy_semantic_fix")
                    .or_else(|| row.pointer("/left/host_summary_policy_semantic_fix"))
            })
        });
    let raw_activation_i8_available = any_array_at(
        right_replay,
        &[
            &["focused_operands", "activations_i8"],
            &["operands", "activations_i8"],
            &["activations_i8"],
        ],
    );
    let raw_packed_qk256_available = any_array_at(
        right_replay,
        &[&["focused_operands", "packed_qk256"], &["operands", "packed_qk256"], &["packed_qk256"]],
    );
    let mut missing_raw_operand_fields = Vec::new();
    if !raw_activation_i8_available {
        missing_raw_operand_fields.push("activations_i8");
    }
    if !raw_packed_qk256_available {
        missing_raw_operand_fields.push("packed_qk256");
    }
    let summary_qk256_trace_available = right_replay.is_some();
    let device_expression_trace_available = device_expression_trace.is_some();
    let device_intermediate_trace_available = device_intermediate_trace.is_some();
    let production_replay_skipped_reason = if raw_activation_i8_available
        && raw_packed_qk256_available
    {
        "focused_raw_operands_present_but_production_replay_execution_not_enabled_in_this_classifier"
    } else if summary_qk256_trace_available {
        "focused_source_has_summary_qk256_trace_but_not_raw_activation_or_packed_qk256_bytes"
    } else {
        "focused_source_missing_qk256_replay_context"
    };
    let next_diagnostic = if raw_activation_i8_available && raw_packed_qk256_available {
        "run selected-device production replay on the focused raw QK256 operands before any production QK256 policy change"
    } else {
        "capture raw activation row and packed QK256 bytes for the focused q_proj first mismatch before any production QK256 policy change"
    };

    FocusedContext {
        case_found,
        summary_first_divergence_matches_request,
        qkv_context_available: qkv_row.is_some(),
        raw_activation_i8_available,
        raw_packed_qk256_available,
        summary_qk256_trace_available,
        device_expression_trace_available,
        device_intermediate_trace_available,
        runtime_device: string_at(right_replay, &["a770", "last_device", "runtime_device"]),
        platform_index: u64_at(right_replay, &["a770", "last_device", "platform_index"]),
        device_index: u64_at(right_replay, &["a770", "last_device", "device_index"]),
        platform_name: string_at(right_replay, &["a770", "last_device", "platform_name"]),
        vendor: string_at(right_replay, &["a770", "last_device", "vendor"]),
        driver_version: string_at(right_replay, &["a770", "last_device", "driver_version"]),
        target_layer_idx: qkv_row.and_then(|row| i64_field(row, "target_layer_idx")),
        projection: qkv_row.and_then(|row| string_field(row, "projection")),
        input_rows: u64_field(right_replay.unwrap_or(&Value::Null), "input_rows"),
        output_rows: u64_field(right_replay.unwrap_or(&Value::Null), "output_rows"),
        cols: u64_field(right_replay.unwrap_or(&Value::Null), "cols"),
        row_stride_bytes: u64_field(right_replay.unwrap_or(&Value::Null), "row_stride_bytes"),
        input_row_index: u64_field(
            device_intermediate_trace.unwrap_or(&Value::Null),
            "input_row_index",
        ),
        sample_count: u64_field(device_intermediate_trace.unwrap_or(&Value::Null), "sample_count"),
        sample_limit: u64_field(device_intermediate_trace.unwrap_or(&Value::Null), "sample_limit"),
        focused_output_index: device_sample.and_then(|sample| u64_field(sample, "output_index")),
        activation_sum: device_sample.and_then(|sample| i64_field(sample, "activation_sum")),
        activation_scale_bits: device_sample
            .and_then(|sample| u64_field(sample, "activation_scale_bits")),
        weight_scale_bits: device_sample.and_then(|sample| u64_field(sample, "weight_scale_bits")),
        int_dot: device_sample.and_then(|sample| i64_field(sample, "int_dot")),
        adjusted_dot: device_sample.and_then(|sample| i64_field(sample, "adjusted_dot")),
        focused_device_output_bits: device_sample.and_then(|sample| u64_field(sample, "output_bits")),
        focused_policy_bits: expression_sample.and_then(|sample| {
            u64_field(sample, "div_then_mul_bits")
                .or_else(|| u64_field(sample, "f64_div_then_mul_cast_bits"))
        }),
        host_summary_policy_semantic_fix_applied: host_summary_policy_semantic_fix
            .and_then(|fix| fix.get("applied"))
            .and_then(Value::as_bool),
        host_summary_policy_semantic_fix_bits: host_summary_policy_semantic_fix
            .and_then(|fix| u64_field(fix, "fixed_policy_bits")),
        host_policy_div_then_mul_bits: expression_sample
            .and_then(|sample| u64_field(sample, "div_then_mul_bits")),
        host_policy_mul_then_div_bits: expression_sample
            .and_then(|sample| u64_field(sample, "mul_then_div_bits")),
        host_policy_reciprocal_then_mul_bits: expression_sample
            .and_then(|sample| u64_field(sample, "reciprocal_then_mul_bits")),
        host_policy_f64_div_then_mul_cast_bits: expression_sample
            .and_then(|sample| u64_field(sample, "f64_div_then_mul_cast_bits")),
        device_intermediate_classification: string_at(
            qkv_row,
            &["right_replay", "device_intermediate_trace", "classification"],
        )
        .or_else(|| {
            source
                .pointer("/generated_output_qk256_device_intermediate_frontier/classification")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        }),
        device_expression_classification: string_at(
            qkv_row,
            &["right_replay", "device_expression_trace", "classification"],
        )
        .or_else(|| {
            source
                .pointer("/generated_output_qk256_device_expression_frontier/classification")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        }),
        production_policy_change_justified: qkv_row
            .and_then(|row| {
                row.pointer("/right_replay/device_expression_trace/production_policy_change_justified")
            })
            .and_then(Value::as_bool)
            .or_else(|| {
                source
                    .pointer(
                        "/generated_output_qk256_compiler_strict_f32_codegen_frontier/rows/0/qk256_context/production_policy_change_justified",
                    )
                    .and_then(Value::as_bool)
            }),
        production_replay_skipped_reason,
        missing_raw_operand_fields,
        next_diagnostic,
    }
}

fn focused_host_policy_expression_split_classification(
    context: &FocusedContext,
    replay_outcome: Option<&FocusedReplayOutcome>,
) -> &'static str {
    if !context.case_found || !context.qkv_context_available {
        return "a770_qk256_focused_host_policy_expression_split_missing_context";
    }
    let Some(FocusedReplayOutcome::Executed { replay, .. }) = replay_outcome else {
        return if matches!(replay_outcome, Some(FocusedReplayOutcome::Failed { .. })) {
            "a770_qk256_focused_host_policy_expression_split_replay_failed"
        } else {
            "a770_qk256_focused_host_policy_expression_split_missing_raw_operands"
        };
    };
    let Some(sample) = replay.samples.first() else {
        return "a770_qk256_focused_host_policy_expression_split_missing_samples";
    };
    let Some(device_bits) = context.focused_device_output_bits else {
        return "a770_qk256_focused_host_policy_expression_split_missing_device_bits";
    };
    let Some(policy_bits) = context.focused_policy_bits else {
        return "a770_qk256_focused_host_policy_expression_split_missing_host_policy_bits";
    };

    let production_bits = u64::from(sample.production_output_bits);
    let replay_bits = u64::from(sample.replay_output_bits);
    let final_bits = u64::from(sample.final_scaled_value_bits);
    if policy_bits == device_bits && production_bits == device_bits {
        "a770_qk256_focused_host_policy_expression_split_clean"
    } else if production_bits == device_bits
        && replay_bits == device_bits
        && final_bits == device_bits
        && policy_bits != device_bits
        && bit_delta(Some(device_bits), Some(policy_bits)) == Some(1)
    {
        "a770_qk256_focused_host_policy_expression_split_host_summary_policy_replay_one_bit"
    } else if production_bits != replay_bits || production_bits != final_bits {
        "a770_qk256_focused_host_policy_expression_split_selected_device_production_expression"
    } else if context.host_policy_div_then_mul_bits == Some(policy_bits)
        || context.host_policy_mul_then_div_bits == Some(policy_bits)
        || context.host_policy_reciprocal_then_mul_bits == Some(policy_bits)
    {
        "a770_qk256_focused_host_policy_expression_split_host_expression_order"
    } else {
        "a770_qk256_focused_host_policy_expression_split_serialization_or_unclassified"
    }
}

fn focused_host_policy_expression_split_next_diagnostic(
    context: &FocusedContext,
    replay_outcome: Option<&FocusedReplayOutcome>,
) -> &'static str {
    match focused_host_policy_expression_split_classification(context, replay_outcome) {
        "a770_qk256_focused_host_policy_expression_split_host_summary_policy_replay_one_bit" => {
            "apply a bounded host summary-policy semantic fix in the next work item, then re-run selected-device focused parity before any production QK256 promotion"
        }
        "a770_qk256_focused_host_policy_expression_split_clean" => {
            "move to multi-case focused parity before any production QK256 promotion"
        }
        "a770_qk256_focused_host_policy_expression_split_selected_device_production_expression" => {
            "inspect selected-device production output-store expression before any host policy change"
        }
        "a770_qk256_focused_host_policy_expression_split_host_expression_order" => {
            "inspect host expression ordering and codegen before any production QK256 policy change"
        }
        "a770_qk256_focused_host_policy_expression_split_serialization_or_unclassified" => {
            "inspect focused receipt serialization and bit-preserving summary policy before any production QK256 policy change"
        }
        _ => "restore focused raw operand replay context before any production QK256 policy change",
    }
}

fn focused_host_policy_expression_split_json(
    context: &FocusedContext,
    replay_outcome: Option<&FocusedReplayOutcome>,
) -> Value {
    match replay_outcome {
        Some(FocusedReplayOutcome::Executed { operands, replay }) => {
            let sample = replay.samples.first();
            let oracle = focused_scalar_oracle(operands);
            let device_bits = context.focused_device_output_bits;
            let policy_bits = context.focused_policy_bits;
            let production_bits = sample.map(|sample| u64::from(sample.production_output_bits));
            let replay_bits = sample.map(|sample| u64::from(sample.replay_output_bits));
            let final_bits = sample.map(|sample| u64::from(sample.final_scaled_value_bits));
            let replay_matches_device = production_bits.is_some()
                && production_bits == device_bits
                && replay_bits == device_bits;
            let replay_matches_policy = production_bits.is_some() && production_bits == policy_bits;
            let host_policy_variants_all_match = [
                context.host_policy_div_then_mul_bits,
                context.host_policy_mul_then_div_bits,
                context.host_policy_reciprocal_then_mul_bits,
                context.host_policy_f64_div_then_mul_cast_bits,
            ]
            .into_iter()
            .flatten()
            .all(|bits| Some(bits) == policy_bits);
            json!({
                "classification": focused_host_policy_expression_split_classification(
                    context,
                    replay_outcome,
                ),
                "localized_to": "host_summary_policy_replay",
                "selected_device_route": {
                    "selected_backend": "intel-arc-a770-opencl",
                    "runtime_api": "opencl",
                    "runtime_device": context.runtime_device,
                    "fallback_used": false,
                    "production_replay_executed": true,
                    "kernel_invocations": replay.kernel_invocations
                },
                "scalar_oracle": {
                    "checked": true,
                    "packing_decode": "qk256 block/chunk/lane/gp production layout",
                    "expected_row_stride_bytes": oracle.expected_row_stride_bytes,
                    "row_stride_bytes": operands.row_stride_bytes,
                    "row_stride_matches_expected": oracle.row_stride_matches_expected,
                    "activation_sum_from_raw": oracle.activation_sum_from_raw,
                    "activation_sum_receipt": operands.activation_sum,
                    "activation_sum_matches_receipt": oracle.activation_sum_matches_receipt,
                    "int_dot": oracle.int_dot,
                    "adjusted_dot": oracle.adjusted_dot,
                    "adjusted_f32_bits": oracle.adjusted_f32_bits,
                    "activation_scale_bits": operands.activation_scale_bits,
                    "weight_scale_bits": operands.weight_scale_bits,
                    "reciprocal_activation_scale_bits": oracle.reciprocal_activation_scale_bits,
                    "adjusted_mul_reciprocal_bits": oracle.adjusted_mul_reciprocal_bits,
                    "final_scaled_value_bits": oracle.final_scaled_value_bits,
                    "div_then_mul_bits": oracle.div_then_mul_bits,
                    "weight_over_activation_bits": oracle.weight_over_activation_bits,
                    "reciprocal_then_mul_bits": oracle.reciprocal_then_mul_bits
                },
                "packing_decode": {
                    "checked": true,
                    "int_dot_matches_device_intermediate_trace": context
                        .int_dot
                        .is_some_and(|value| value == i64::from(oracle.int_dot)),
                    "adjusted_dot_matches_device_intermediate_trace": context
                        .adjusted_dot
                        .is_some_and(|value| value == i64::from(oracle.adjusted_dot)),
                    "int_dot_matches_selected_device_replay": sample
                        .is_some_and(|sample| sample.int_dot == oracle.int_dot),
                    "adjusted_dot_matches_selected_device_replay": sample
                        .is_some_and(|sample| sample.adjusted_dot == oracle.adjusted_dot)
                },
                "scale_cast": {
                    "checked": true,
                    "source_host_policy_div_then_mul_bits": context.host_policy_div_then_mul_bits,
                    "source_host_policy_mul_then_div_bits": context.host_policy_mul_then_div_bits,
                    "source_host_policy_reciprocal_then_mul_bits": context
                        .host_policy_reciprocal_then_mul_bits,
                    "source_host_policy_f64_div_then_mul_cast_bits": context
                        .host_policy_f64_div_then_mul_cast_bits,
                    "selected_device_div_then_mul_bits": sample.map(|sample| sample.div_then_mul_bits),
                    "selected_device_reciprocal_then_mul_bits": sample
                        .map(|sample| sample.reciprocal_then_mul_bits),
                    "selected_device_final_scaled_value_bits": final_bits,
                    "source_host_policy_variants_all_match": host_policy_variants_all_match
                },
                "tail_padding": {
                    "checked": true,
                    "cols": operands.cols,
                    "used_payload_bytes": oracle.used_payload_bytes,
                    "row_padding_bytes": oracle.row_padding_bytes,
                    "unused_tail_columns": oracle.unused_tail_columns,
                    "no_tail_or_padding_for_focused_row": oracle.row_padding_bytes == 0
                        && oracle.unused_tail_columns == 0
                },
                "serialization": {
                    "checked": true,
                    "bitwise_comparison_used": true,
                    "focused_device_output_bits": device_bits,
                    "focused_host_policy_bits": policy_bits,
                    "selected_device_production_output_bits": production_bits,
                    "selected_device_replay_output_bits": replay_bits,
                    "selected_device_final_scaled_value_bits": final_bits,
                    "device_vs_host_policy_bit_delta": bit_delta(device_bits, policy_bits),
                    "one_bit_host_policy_split": bit_delta(device_bits, policy_bits) == Some(1),
                    "selected_device_replay_matches_device_bits": replay_matches_device,
                    "selected_device_replay_matches_host_policy_bits": replay_matches_policy
                },
                "claim_boundary": {
                    "production_qk256_policy_change": false,
                    "bitnet_inference": false,
                    "qk256_decode": false,
                    "claim_allowed": false,
                    "diagnostic_only": true,
                    "performance_claim": false,
                    "full_residency_claim": false
                }
            })
        }
        Some(FocusedReplayOutcome::Failed { error }) => json!({
            "classification": focused_host_policy_expression_split_classification(
                context,
                replay_outcome,
            ),
            "replay_error": error,
            "claim_boundary": {
                "production_qk256_policy_change": false,
                "bitnet_inference": false,
                "claim_allowed": false,
                "diagnostic_only": true
            }
        }),
        None => json!({
            "classification": focused_host_policy_expression_split_classification(
                context,
                replay_outcome,
            ),
            "missing_raw_operand_fields": context.missing_raw_operand_fields,
            "claim_boundary": {
                "production_qk256_policy_change": false,
                "bitnet_inference": false,
                "claim_allowed": false,
                "diagnostic_only": true
            }
        }),
    }
}

fn focused_host_summary_policy_semantic_fix_classification(
    context: &FocusedContext,
    replay_outcome: Option<&FocusedReplayOutcome>,
) -> &'static str {
    if !context.case_found || !context.qkv_context_available {
        return "a770_qk256_host_summary_policy_semantic_fix_missing_context";
    }
    let Some(FocusedReplayOutcome::Executed { replay, .. }) = replay_outcome else {
        return if matches!(replay_outcome, Some(FocusedReplayOutcome::Failed { .. })) {
            "a770_qk256_host_summary_policy_semantic_fix_replay_failed"
        } else {
            "a770_qk256_host_summary_policy_semantic_fix_missing_raw_operands"
        };
    };
    let Some(sample) = replay.samples.first() else {
        return "a770_qk256_host_summary_policy_semantic_fix_missing_samples";
    };
    let Some(device_bits) = context.focused_device_output_bits else {
        return "a770_qk256_host_summary_policy_semantic_fix_missing_device_bits";
    };
    let Some(fixed_policy_bits) = context.host_summary_policy_semantic_fix_bits else {
        return "a770_qk256_host_summary_policy_semantic_fix_missing_fixed_policy_bits";
    };
    if context.host_summary_policy_semantic_fix_applied != Some(true) {
        return "a770_qk256_host_summary_policy_semantic_fix_not_applied";
    }
    if fixed_policy_bits != device_bits {
        return "a770_qk256_host_summary_policy_semantic_fix_fixed_policy_mismatch";
    }

    let production_bits = u64::from(sample.production_output_bits);
    let replay_bits = u64::from(sample.replay_output_bits);
    let final_bits = u64::from(sample.final_scaled_value_bits);
    if production_bits == device_bits && replay_bits == device_bits && final_bits == device_bits {
        "a770_qk256_host_summary_policy_semantic_fix_focused_row_matches_selected_device_replay"
    } else {
        "a770_qk256_host_summary_policy_semantic_fix_selected_device_replay_mismatch"
    }
}

fn focused_host_summary_policy_semantic_fix_next_diagnostic(
    context: &FocusedContext,
    replay_outcome: Option<&FocusedReplayOutcome>,
) -> &'static str {
    match focused_host_summary_policy_semantic_fix_classification(context, replay_outcome) {
        "a770_qk256_host_summary_policy_semantic_fix_focused_row_matches_selected_device_replay" => {
            "expand to multi-case focused QK256 replay before any production QK256 promotion"
        }
        "a770_qk256_host_summary_policy_semantic_fix_fixed_policy_mismatch"
        | "a770_qk256_host_summary_policy_semantic_fix_not_applied" => {
            "repair the bounded host summary-policy semantic fix before any production QK256 promotion"
        }
        "a770_qk256_host_summary_policy_semantic_fix_selected_device_replay_mismatch" => {
            "inspect selected-device production replay after the host summary-policy semantic fix"
        }
        _ => "restore focused raw operand replay context before any production QK256 policy change",
    }
}

fn focused_host_summary_policy_semantic_fix_json(
    context: &FocusedContext,
    replay_outcome: Option<&FocusedReplayOutcome>,
) -> Value {
    match replay_outcome {
        Some(FocusedReplayOutcome::Executed { replay, .. }) => {
            let sample = replay.samples.first();
            let device_bits = context.focused_device_output_bits;
            let source_policy_bits = context.focused_policy_bits;
            let fixed_policy_bits = context.host_summary_policy_semantic_fix_bits;
            let production_bits = sample.map(|sample| u64::from(sample.production_output_bits));
            let replay_bits = sample.map(|sample| u64::from(sample.replay_output_bits));
            let final_bits = sample.map(|sample| u64::from(sample.final_scaled_value_bits));
            json!({
                "classification": focused_host_summary_policy_semantic_fix_classification(
                    context,
                    replay_outcome,
                ),
                "localized_to": "host_summary_policy_semantic_fix",
                "selected_device_route": {
                    "selected_backend": "intel-arc-a770-opencl",
                    "runtime_api": "opencl",
                    "runtime_device": context.runtime_device,
                    "fallback_used": false,
                    "production_replay_executed": true,
                    "kernel_invocations": replay.kernel_invocations
                },
                "policy_bits": {
                    "semantic_fix_applied": context.host_summary_policy_semantic_fix_applied,
                    "source_host_summary_policy_bits": source_policy_bits,
                    "fixed_host_summary_policy_bits": fixed_policy_bits,
                    "focused_device_output_bits": device_bits,
                    "source_policy_bit_delta": bit_delta(device_bits, source_policy_bits),
                    "fixed_policy_bit_delta": bit_delta(device_bits, fixed_policy_bits),
                    "source_policy_matches_device_bits": source_policy_bits
                        .zip(device_bits)
                        .map(|(source, device)| source == device),
                    "fixed_policy_matches_device_bits": fixed_policy_bits
                        .zip(device_bits)
                        .map(|(fixed, device)| fixed == device)
                },
                "selected_device_replay": {
                    "production_output_bits": production_bits,
                    "replay_output_bits": replay_bits,
                    "final_scaled_value_bits": final_bits,
                    "production_output_matches_device_bits": production_bits
                        .zip(device_bits)
                        .map(|(production, device)| production == device),
                    "replay_output_matches_device_bits": replay_bits
                        .zip(device_bits)
                        .map(|(replay, device)| replay == device),
                    "final_scaled_value_matches_device_bits": final_bits
                        .zip(device_bits)
                        .map(|(final_bits, device)| final_bits == device),
                    "fixed_policy_matches_production_output_bits": fixed_policy_bits
                        .zip(production_bits)
                        .map(|(fixed, production)| fixed == production)
                },
                "claim_boundary": {
                    "production_qk256_policy_change": false,
                    "answer_scoring_change": false,
                    "sampling_change": false,
                    "cpu_a770_parity_claim": false,
                    "strict_answer_readiness_claim": false,
                    "broad_a770_quality_claim": false,
                    "bitnet_inference": false,
                    "qk256_decode": false,
                    "claim_allowed": false,
                    "diagnostic_only": true,
                    "performance_claim": false,
                    "full_residency_claim": false
                }
            })
        }
        Some(FocusedReplayOutcome::Failed { error }) => json!({
            "classification": focused_host_summary_policy_semantic_fix_classification(
                context,
                replay_outcome,
            ),
            "replay_error": error,
            "claim_boundary": {
                "production_qk256_policy_change": false,
                "bitnet_inference": false,
                "claim_allowed": false,
                "diagnostic_only": true
            }
        }),
        None => json!({
            "classification": focused_host_summary_policy_semantic_fix_classification(
                context,
                replay_outcome,
            ),
            "missing_raw_operand_fields": context.missing_raw_operand_fields,
            "claim_boundary": {
                "production_qk256_policy_change": false,
                "bitnet_inference": false,
                "claim_allowed": false,
                "diagnostic_only": true
            }
        }),
    }
}

fn focused_classification(
    context: &FocusedContext,
    replay_outcome: Option<&FocusedReplayOutcome>,
) -> &'static str {
    if !context.case_found || !context.qkv_context_available {
        "a770_qk256_focused_production_operands_missing_context"
    } else if let Some(FocusedReplayOutcome::Executed { replay, .. }) = replay_outcome {
        if replay.samples.is_empty() {
            "a770_qk256_focused_raw_operands_replay_missing_samples"
        } else if context.focused_device_output_bits.is_some_and(|bits| {
            replay.samples.iter().any(|sample| u64::from(sample.production_output_bits) == bits)
        }) {
            "a770_qk256_focused_raw_operands_replay_matches_focused_device_output"
        } else if context.focused_device_output_bits.is_some() {
            "a770_qk256_focused_raw_operands_replay_differs_from_focused_device_output"
        } else {
            "a770_qk256_focused_raw_operands_replay_executed_missing_focused_output_context"
        }
    } else if matches!(replay_outcome, Some(FocusedReplayOutcome::Failed { .. })) {
        "a770_qk256_focused_raw_operands_replay_failed"
    } else if context.raw_activation_i8_available && context.raw_packed_qk256_available {
        "a770_qk256_focused_production_operands_raw_operands_available"
    } else if context.summary_qk256_trace_available {
        "a770_qk256_focused_production_operands_summary_context_only_raw_operands_missing"
    } else {
        "a770_qk256_focused_production_operands_missing_context"
    }
}

fn focused_next_diagnostic(
    context: &FocusedContext,
    replay_outcome: Option<&FocusedReplayOutcome>,
) -> &'static str {
    match replay_outcome {
        Some(FocusedReplayOutcome::Executed { replay, .. })
            if !replay.samples.is_empty()
                && context.focused_device_output_bits.is_some_and(|bits| {
                    replay
                        .samples
                        .iter()
                        .any(|sample| u64::from(sample.production_output_bits) == bits)
                }) =>
        {
            "localize focused host-policy versus selected-device production replay expression split before any production QK256 policy change"
        }
        Some(FocusedReplayOutcome::Executed { .. }) => {
            "compare focused production replay output against focused device trace before any production QK256 policy change"
        }
        Some(FocusedReplayOutcome::Failed { .. }) => {
            "repair focused raw operand production replay execution before any production QK256 policy change"
        }
        None => context.next_diagnostic,
    }
}

fn focused_scalar_oracle(operands: &FocusedRawOperands) -> FocusedScalarOracle {
    let expected_row_stride_bytes = qk256_row_stride_bytes(operands.cols);
    let used_blocks = (operands.cols + 255) / 256;
    let used_payload_bytes = used_blocks * 64;
    let row_padding_bytes = operands.row_stride_bytes.saturating_sub(used_payload_bytes);
    let unused_tail_columns = if operands.cols % 256 == 0 { 0 } else { 256 - operands.cols % 256 };
    let activation_sum_from_raw =
        operands.activations_i8.iter().map(|value| i32::from(*value)).sum::<i32>();
    let int_dot =
        focused_host_int_dot(&operands.activations_i8, &operands.packed_qk256, operands.cols);
    let adjusted_dot = int_dot - operands.activation_sum;
    let adjusted_f32 = adjusted_dot as f32;
    let reciprocal_activation_scale = 1.0f32 / operands.activation_scale;
    let adjusted_mul_reciprocal = adjusted_f32 * reciprocal_activation_scale;
    let final_scaled_value = adjusted_mul_reciprocal * operands.weight_scale;
    let div_then_mul = (adjusted_f32 / operands.activation_scale) * operands.weight_scale;
    let weight_over_activation = operands.weight_scale / operands.activation_scale;
    let reciprocal_then_mul = adjusted_f32 * weight_over_activation;

    FocusedScalarOracle {
        expected_row_stride_bytes,
        row_stride_matches_expected: operands.row_stride_bytes == expected_row_stride_bytes,
        activation_sum_from_raw,
        activation_sum_matches_receipt: activation_sum_from_raw == operands.activation_sum,
        int_dot,
        adjusted_dot,
        adjusted_f32_bits: adjusted_f32.to_bits(),
        reciprocal_activation_scale_bits: reciprocal_activation_scale.to_bits(),
        adjusted_mul_reciprocal_bits: adjusted_mul_reciprocal.to_bits(),
        final_scaled_value_bits: final_scaled_value.to_bits(),
        div_then_mul_bits: div_then_mul.to_bits(),
        weight_over_activation_bits: weight_over_activation.to_bits(),
        reciprocal_then_mul_bits: reciprocal_then_mul.to_bits(),
        used_payload_bytes,
        row_padding_bytes,
        unused_tail_columns,
    }
}

fn focused_host_int_dot(activations_i8: &[i8], packed_qk256: &[u8], cols: usize) -> i32 {
    (0..cols)
        .map(|col| {
            i32::from(focused_read_qk256_code(packed_qk256, col)) * i32::from(activations_i8[col])
        })
        .sum()
}

fn focused_read_qk256_code(packed_qk256: &[u8], col: usize) -> u8 {
    let block = col / 256;
    let offset = col - block * 256;
    let chunk = offset / 128;
    let lane = (offset - chunk * 128) / 32;
    let gp = offset & 31;
    let byte_index = block * 64 + chunk * 32 + gp;
    (packed_qk256[byte_index] >> (6 - lane * 2)) & 0x03
}

fn qk256_row_stride_bytes(cols: usize) -> usize {
    ((cols + 255) / 256) * 64
}

fn bit_delta(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    let left = left?;
    let right = right?;
    if left >= right { Some(left - right) } else { Some(right - left) }
}

fn find_row<'a>(
    rows: Option<&'a Value>,
    case_id: &str,
    first_mismatch_index: usize,
) -> Option<&'a Value> {
    rows.and_then(Value::as_array).and_then(|rows| {
        rows.iter().find(|row| {
            str_field(row, "case_id") == Some(case_id)
                && usize_field(row, "first_mismatch_index") == Some(first_mismatch_index)
        })
    })
}

fn find_case_row<'a>(rows: Option<&'a Value>, case_id: &str) -> Option<&'a Value> {
    rows.and_then(Value::as_array)
        .and_then(|rows| rows.iter().find(|row| str_field(row, "case_id") == Some(case_id)))
}

fn any_array_at(root: Option<&Value>, paths: &[&[&str]]) -> bool {
    paths.iter().any(|path| {
        let mut value = root;
        for key in *path {
            value = value.and_then(|value| value.get(*key));
        }
        value.and_then(Value::as_array).is_some_and(|array| !array.is_empty())
    })
}

fn array_len_at(root: Option<&Value>, paths: &[&[&str]]) -> Option<usize> {
    paths.iter().find_map(|path| {
        let mut value = root;
        for key in *path {
            value = value.and_then(|value| value.get(*key));
        }
        value.and_then(Value::as_array).map(Vec::len)
    })
}

fn i8_array_at(root: Option<&Value>, paths: &[&[&str]]) -> Result<Option<Vec<i8>>, Box<dyn Error>> {
    for path in paths {
        let mut value = root;
        for key in *path {
            value = value.and_then(|value| value.get(*key));
        }
        if let Some(array) = value.and_then(Value::as_array) {
            let values = array
                .iter()
                .map(|value| {
                    let value = value.as_i64().ok_or_else(|| {
                        io_error("focused raw operands activations_i8 contains non-integer value")
                    })?;
                    i8::try_from(value).map_err(|_| {
                        io_error(format!(
                            "focused raw operands activations_i8 value {value} outside i8 range"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            return Ok(Some(values));
        }
    }
    Ok(None)
}

fn u8_array_at(root: Option<&Value>, paths: &[&[&str]]) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
    for path in paths {
        let mut value = root;
        for key in *path {
            value = value.and_then(|value| value.get(*key));
        }
        if let Some(array) = value.and_then(Value::as_array) {
            let values = array
                .iter()
                .map(|value| {
                    let value = value.as_u64().ok_or_else(|| {
                        io_error("focused raw operands packed_qk256 contains non-integer value")
                    })?;
                    u8::try_from(value).map_err(|_| {
                        io_error(format!(
                            "focused raw operands packed_qk256 value {value} outside u8 range"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            return Ok(Some(values));
        }
    }
    Ok(None)
}

fn path_json_value(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn str_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    str_field(value, key).map(ToOwned::to_owned)
}

fn usize_field(value: &Value, key: &str) -> Option<usize> {
    value.get(key).and_then(Value::as_u64).and_then(|value| value.try_into().ok())
}

fn u64_field(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn u32_field(value: &Value, key: &str) -> Option<u32> {
    value.get(key).and_then(Value::as_u64).and_then(|value| value.try_into().ok())
}

fn i64_field(value: &Value, key: &str) -> Option<i64> {
    value.get(key).and_then(Value::as_i64)
}

fn string_at(root: Option<&Value>, path: &[&str]) -> Option<String> {
    let mut value = root;
    for key in path {
        value = value.and_then(|value| value.get(*key));
    }
    value.and_then(Value::as_str).map(ToOwned::to_owned)
}

fn u64_at(root: Option<&Value>, path: &[&str]) -> Option<u64> {
    let mut value = root;
    for key in path {
        value = value.and_then(|value| value.get(*key));
    }
    value.and_then(Value::as_u64)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_evidence_retains_full_operands_from_external_source_packet()
    -> Result<(), Box<dyn Error>> {
        let case_id = "a770_summary_seed770024_keywords_014";
        let source_path = env::temp_dir()
            .join(format!("bitnet-a770-full-projection-source-{}.json", std::process::id()));
        let mut logits_dump = (0..10).map(|_| json!({})).collect::<Vec<_>>();
        let step = json!({
            "step": 9,
            "logit_source_context": {
                "hidden_state_source": {
                    "model_forward_source": {
                        "qkv_projection_sources": {
                            "sources": [{
                                "projection": "q_proj",
                                "target_layer_idx": 0,
                                "source_context_available": true,
                                "projection_replay": {
                                    "full_projection_operands": {
                                        "activations_i8": (0..256).map(|value| (value % 127) as i64).collect::<Vec<_>>(),
                                        "packed_qk256": vec![0; 128],
                                        "rows": 2,
                                        "cols": 256,
                                        "row_stride_bytes": 64,
                                        "activation_sum": 0,
                                        "activation_scale_bits": 0x3f800000u64,
                                        "weight_scale_bits": 0x3f800000u64,
                                        "input_row_index": 0
                                    }
                                }
                            }]
                        }
                    }
                }
            }
        });
        let slot = logits_dump
            .get_mut(9)
            .ok_or_else(|| io_error("synthetic logits dump missing step 9"))?;
        *slot = step;
        let source = json!({
            "cases": [{"id": case_id, "logits_dump": logits_dump}]
        });

        let result = (|| -> Result<(), Box<dyn Error>> {
            fs::write(&source_path, serde_json::to_vec(&source)?)?;
            let row = json!({
                "row_evidence": {
                    "dispatch_replay_source": path_json_value(&source_path)
                }
            });
            let evidence = projection_operand_capture_evidence(
                Some(&row),
                case_id,
                9,
                0,
                "q_proj",
                "a770_159_full_projection_operand_source_boundary",
            );
            if !evidence.full_projection_operands_available {
                return Err(io_error("external full projection source was not accepted"));
            }
            let operands = evidence
                .replay_operands
                .as_ref()
                .ok_or_else(|| io_error("external full projection operands were not retained"))?;
            if (
                operands.rows,
                operands.cols,
                operands.row_stride_bytes,
                operands.packed_qk256.len(),
            ) != (2, 256, 64, 128)
            {
                return Err(io_error("external full projection operand dimensions changed"));
            }
            Ok(())
        })();
        let _ = fs::remove_file(&source_path);
        result
    }
}
