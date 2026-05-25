use std::{env, error::Error, io, path::PathBuf};

use bitnet_kernels::a770_opencl_runtime::{
    A770OpenClQk256CompilerBinaryEvidence, capture_a770_qk256_debug_compiler_binary_evidence,
};

const RECEIPT_ENV: &str = "BITNET_A770_OPENCL_COMPILER_EVIDENCE_RECEIPT";

fn main() -> Result<(), Box<dyn Error>> {
    let receipt_path = receipt_path_from_args()?;
    let evidence = capture_a770_qk256_debug_compiler_binary_evidence()?;
    let receipt = evidence_to_json(&evidence);
    if let Some(path) = receipt_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &receipt)?;
    }
    println!("{receipt}");
    Ok(())
}

fn receipt_path_from_args() -> Result<Option<PathBuf>, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mut receipt = env::var_os(RECEIPT_ENV).map(PathBuf::from);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--receipt" => {
                let path =
                    args.next().ok_or_else(|| io_error("--receipt requires a path argument"))?;
                receipt = Some(PathBuf::from(path));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: a770-opencl-compiler-evidence [--receipt <path>]\n\nCaptures selected Intel Arc A770 OpenCL compiler binary evidence for the diagnostic QK256 debug kernel."
                );
                std::process::exit(0);
            }
            other => return Err(io_error(format!("unknown argument {other:?}"))),
        }
    }
    Ok(receipt)
}

fn evidence_to_json(evidence: &A770OpenClQk256CompilerBinaryEvidence) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"campaign\": \"intel-a770\",\n",
            "  \"work_item\": \"A770-056\",\n",
            "  \"proof_family\": \"a770_opencl_qk256_compiler_binary_evidence\",\n",
            "  \"proof_stage\": \"diagnostic_compiler_binary_captured\",\n",
            "  \"requested_backend\": \"intel-arc-a770\",\n",
            "  \"selected_backend\": \"intel-arc-a770-opencl\",\n",
            "  \"runtime_api\": \"opencl\",\n",
            "  \"runtime_device\": \"{}\",\n",
            "  \"platform_index\": {},\n",
            "  \"device_index\": {},\n",
            "  \"platform_name\": \"{}\",\n",
            "  \"vendor\": \"{}\",\n",
            "  \"driver_version\": \"{}\",\n",
            "  \"kernel_source\": \"bitnet_kernels::a770_opencl_runtime::QK256_I2S_I8S_SCALED_GEMV_DEBUG_SRC\",\n",
            "  \"kernel_name\": \"qk256_i2s_i8s_scaled_gemv_debug\",\n",
            "  \"classification\": \"{}\",\n",
            "  \"build_options\": \"{}\",\n",
            "  \"build_log\": \"{}\",\n",
            "  \"binary_type\": \"{}\",\n",
            "  \"kernel_names\": \"{}\",\n",
            "  \"program_device_count\": {},\n",
            "  \"binary_sizes\": {},\n",
            "  \"binary_fnv1a64\": {},\n",
            "  \"binary_prefix_hex\": {},\n",
            "  \"source_bytes\": {},\n",
            "  \"source_fnv1a64\": \"{}\",\n",
            "  \"strict_f32_barrier_source_present\": {},\n",
            "  \"program_binary_captured\": {},\n",
            "  \"disassembly_captured\": {},\n",
            "  \"fallback_used\": false,\n",
            "  \"cpu_fallback_allowed\": false,\n",
            "  \"bitnet_inference\": false,\n",
            "  \"qk256_decode\": false,\n",
            "  \"production_qk256_policy_change\": false,\n",
            "  \"claim_allowed\": false,\n",
            "  \"diagnostic_only\": true,\n",
            "  \"performance_claim\": false,\n",
            "  \"full_residency_claim\": false,\n",
            "  \"next_diagnostic\": \"capture vendor disassembly or offline compiler evidence for the strict-f32 barrier before any production QK256 policy change\",\n",
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
        json_escape(&evidence.runtime_device),
        evidence.platform_index,
        evidence.device_index,
        json_escape(&evidence.platform_name),
        json_escape(&evidence.vendor),
        json_escape(&evidence.driver_version),
        json_escape(&evidence.classification),
        json_escape(&evidence.build_options),
        json_escape(&evidence.build_log),
        json_escape(&evidence.binary_type),
        json_escape(&evidence.kernel_names),
        evidence.program_device_count,
        usize_array_json(&evidence.binary_sizes),
        string_array_json(&evidence.binary_fnv1a64),
        string_array_json(&evidence.binary_prefix_hex),
        evidence.source_bytes,
        json_escape(&evidence.source_fnv1a64),
        evidence.strict_f32_barrier_source_present,
        evidence.program_binary_captured,
        evidence.disassembly_captured
    )
}

fn usize_array_json(values: &[usize]) -> String {
    let joined = values.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
    format!("[{joined}]")
}

fn string_array_json(values: &[String]) -> String {
    let joined = values
        .iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{joined}]")
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
            c if c.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\u{:04x}", c as u32);
            }
            c => escaped.push(c),
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
    fn json_escape_handles_quotes_and_control_chars() {
        assert_eq!(json_escape("a\"b\\c\n"), "a\\\"b\\\\c\\n");
    }

    #[test]
    fn array_json_formats_scalars_and_strings() {
        assert_eq!(usize_array_json(&[1, 2]), "[1, 2]");
        assert_eq!(
            string_array_json(&["ab".to_owned(), "c\"d".to_owned()]),
            "[\"ab\", \"c\\\"d\"]"
        );
    }
}
