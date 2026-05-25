use std::{
    env,
    error::Error,
    io,
    path::{Path, PathBuf},
};

const DEFAULT_ASM_PATH: &str = "ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-disassembly/ocloc-dump/.text.qk256_i2s_i8s_scaled_gemv_debug.asm";
const KERNEL_NAME: &str = "qk256_i2s_i8s_scaled_gemv_debug";

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let asm = std::fs::read_to_string(&args.asm)?;
    let frontier = inspect_strict_f32_frontier(&asm, &args.asm);
    let receipt = frontier_to_json(&frontier);
    if let Some(path) = &args.receipt {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &receipt)?;
    }
    println!("{receipt}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    asm: PathBuf,
    receipt: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut asm = PathBuf::from(DEFAULT_ASM_PATH);
        let mut receipt = None;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--asm" => {
                    asm = PathBuf::from(
                        args.next().ok_or_else(|| io_error("--asm requires a path argument"))?,
                    );
                }
                "--receipt" => {
                    receipt = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| io_error("--receipt requires a path argument"))?,
                    ));
                }
                "--help" | "-h" => {
                    println!(
                        "Usage: a770-opencl-strict-f32-disassembly-frontier [--asm <path>] [--receipt <path>]\n\nInspects committed A770 OpenCL QK256 debug-kernel assembly for the lowered strict-f32 barrier operation sequence."
                    );
                    std::process::exit(0);
                }
                other => return Err(io_error(format!("unknown argument {other:?}"))),
            }
        }
        Ok(Self { asm, receipt })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StrictF32DisassemblyFrontier {
    asm_path: PathBuf,
    asm_bytes: usize,
    asm_lines: usize,
    asm_fnv1a64: String,
    kernel_name: &'static str,
    f32_mul_count: usize,
    f32_mov_count: usize,
    direct_div_count: usize,
    ugm_d32_store_count: usize,
    ugm_d32_load_count: usize,
    strict_f32_barrier_store_load_sequence: bool,
    finite_guard_sequence_present: bool,
    classification: &'static str,
    evidence_lines: Vec<String>,
}

fn inspect_strict_f32_frontier(asm: &str, asm_path: &Path) -> StrictF32DisassemblyFrontier {
    let normalized = asm.replace("\r\n", "\n");
    let lines = normalized.lines().collect::<Vec<_>>();
    let f32_mul_count = count_lines(&lines, |line| line.contains("mul") && line.contains(":f"));
    let f32_mov_count = count_lines(&lines, |line| line.contains("mov") && line.contains(":f"));
    let direct_div_count =
        count_lines(&lines, |line| line.contains("div") || line.contains("math.fdiv"));
    let ugm_d32_store_count = count_lines(&lines, |line| line.contains("store.ugm.d32.a64"));
    let ugm_d32_load_count = count_lines(&lines, |line| line.contains("load.ugm.d32.a64"));
    let finite_guard_sequence_present = normalized.contains("0x4F800000:f")
        && normalized.contains("0x2F800000:f")
        && normalized.contains("0x64000000:ud");
    let strict_f32_barrier_store_load_sequence =
        has_ordered_store_then_load(&lines) && ugm_d32_store_count >= 4 && ugm_d32_load_count >= 4;
    let classification = classify_strict_f32_sequence(
        !normalized.trim().is_empty(),
        f32_mul_count,
        direct_div_count,
        strict_f32_barrier_store_load_sequence,
        finite_guard_sequence_present,
    );
    let evidence_lines = collect_evidence_lines(&lines);

    StrictF32DisassemblyFrontier {
        asm_path: asm_path.to_path_buf(),
        asm_bytes: normalized.len(),
        asm_lines: lines.len(),
        asm_fnv1a64: fnv1a64_hex(normalized.as_bytes()),
        kernel_name: KERNEL_NAME,
        f32_mul_count,
        f32_mov_count,
        direct_div_count,
        ugm_d32_store_count,
        ugm_d32_load_count,
        strict_f32_barrier_store_load_sequence,
        finite_guard_sequence_present,
        classification,
        evidence_lines,
    }
}

fn classify_strict_f32_sequence(
    asm_available: bool,
    f32_mul_count: usize,
    direct_div_count: usize,
    strict_f32_barrier_store_load_sequence: bool,
    finite_guard_sequence_present: bool,
) -> &'static str {
    if !asm_available {
        return "a770_qk256_strict_f32_disassembly_frontier_missing_context";
    }
    if strict_f32_barrier_store_load_sequence && f32_mul_count > 0 && finite_guard_sequence_present
    {
        return "a770_qk256_strict_f32_disassembly_frontier_barrier_preserving_f32_sequence";
    }
    if f32_mul_count > 0 && direct_div_count == 0 {
        return "a770_qk256_strict_f32_disassembly_frontier_compiler_runtime_reassociation_or_collapse";
    }
    if f32_mul_count == 0 && direct_div_count == 0 {
        return "a770_qk256_strict_f32_disassembly_frontier_missing_context";
    }
    "a770_qk256_strict_f32_disassembly_frontier_unrecognized_lowered_sequence"
}

fn count_lines(lines: &[&str], predicate: impl Fn(&str) -> bool) -> usize {
    lines.iter().filter(|line| predicate(line)).count()
}

fn has_ordered_store_then_load(lines: &[&str]) -> bool {
    let first_store = lines.iter().position(|line| line.contains("store.ugm.d32.a64"));
    let first_load = lines.iter().position(|line| line.contains("load.ugm.d32.a64"));
    matches!((first_store, first_load), (Some(store), Some(load)) if store < load)
}

fn collect_evidence_lines(lines: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let interesting = line.contains("store.ugm.d32.a64")
            || line.contains("load.ugm.d32.a64")
            || (line.contains("mul") && line.contains(":f"))
            || line.contains("0x4F800000:f")
            || line.contains("0x2F800000:f")
            || line.contains("0x64000000:ud");
        if interesting {
            out.push(format!("{}: {}", index + 1, line.trim()));
        }
        if out.len() >= 24 {
            break;
        }
    }
    out
}

fn frontier_to_json(frontier: &StrictF32DisassemblyFrontier) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"campaign\": \"intel-a770\",\n",
            "  \"work_item\": \"A770-058\",\n",
            "  \"proof_family\": \"a770_opencl_qk256_strict_f32_disassembly_frontier\",\n",
            "  \"proof_stage\": \"diagnostic_strict_f32_barrier_sequence_classified\",\n",
            "  \"requested_backend\": \"intel-arc-a770\",\n",
            "  \"selected_backend\": \"intel-arc-a770-opencl\",\n",
            "  \"runtime_api\": \"opencl\",\n",
            "  \"kernel_name\": \"{}\",\n",
            "  \"classification\": \"{}\",\n",
            "  \"asm_path\": \"{}\",\n",
            "  \"asm_bytes\": {},\n",
            "  \"asm_lines\": {},\n",
            "  \"asm_fnv1a64\": \"{}\",\n",
            "  \"f32_mul_count\": {},\n",
            "  \"f32_mov_count\": {},\n",
            "  \"direct_div_count\": {},\n",
            "  \"ugm_d32_store_count\": {},\n",
            "  \"ugm_d32_load_count\": {},\n",
            "  \"strict_f32_barrier_store_load_sequence\": {},\n",
            "  \"finite_guard_sequence_present\": {},\n",
            "  \"evidence_lines\": {},\n",
            "  \"fallback_used\": false,\n",
            "  \"cpu_fallback_allowed\": false,\n",
            "  \"bitnet_inference\": false,\n",
            "  \"qk256_decode\": false,\n",
            "  \"production_qk256_policy_change\": false,\n",
            "  \"claim_allowed\": false,\n",
            "  \"diagnostic_only\": true,\n",
            "  \"performance_claim\": false,\n",
            "  \"full_residency_claim\": false,\n",
            "  \"next_diagnostic\": \"inspect production-policy impact only after strict-f32 disassembly context is reviewed\",\n",
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
        frontier.kernel_name,
        frontier.classification,
        json_escape(&path_json_value(&frontier.asm_path)),
        frontier.asm_bytes,
        frontier.asm_lines,
        frontier.asm_fnv1a64,
        frontier.f32_mul_count,
        frontier.f32_mov_count,
        frontier.direct_div_count,
        frontier.ugm_d32_store_count,
        frontier.ugm_d32_load_count,
        frontier.strict_f32_barrier_store_load_sequence,
        frontier.finite_guard_sequence_present,
        string_array_json(&frontier.evidence_lines)
    )
}

fn string_array_json(values: &[String]) -> String {
    let joined = values
        .iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{joined}]")
}

fn path_json_value(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
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

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn io_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_missing_context() {
        assert_eq!(
            classify_strict_f32_sequence(false, 0, 0, false, false),
            "a770_qk256_strict_f32_disassembly_frontier_missing_context"
        );
    }

    #[test]
    fn classifies_barrier_preserving_store_load_sequence() {
        let asm = concat!(
            "send.ugm (16|M0) null r58 r54:2 0x0 0x080E0584 // store.ugm.d32.a64.wb.wb\n",
            "send.ugm (16|M0) null r66 r74:2 0x0 0x080E0584 // store.ugm.d32.a64.wb.wb\n",
            "send.ugm (16|M0) null r78 r86:2 0x0 0x080E0584 // store.ugm.d32.a64.wb.wb\n",
            "send.ugm (16|M16) null r82 r88:2 0x0 0x080E0584 // store.ugm.d32.a64.wb.wb\n",
            "mov r100.0<1>:f 0x4F800000:f\n",
            "cmp null<1>:ud r96.0<0;1,0>:ud 0x64000000:ud\n",
            "send.ugm (16|M0) r90 r66 null:0 0x0 0x08280580 // load.ugm.d32.a64.ca.ca\n",
            "send.ugm (16|M16) r92 r70 null:0 0x0 0x08280580 // load.ugm.d32.a64.ca.ca\n",
            "send.ugm (16|M0) r11 r58 null:0 0x0 0x08280580 // load.ugm.d32.a64.ca.ca\n",
            "send.ugm (16|M16) r13 r62 null:0 0x0 0x08280580 // load.ugm.d32.a64.ca.ca\n",
            "mov r3.0<1>:f 0x2F800000:f\n",
            "mul (16|M0) r47.0<1>:f r54.0<1;1,0>:f r9.3<0;1,0>:f\n"
        );

        let frontier = inspect_strict_f32_frontier(asm, Path::new("sample.asm"));

        assert_eq!(
            frontier.classification,
            "a770_qk256_strict_f32_disassembly_frontier_barrier_preserving_f32_sequence"
        );
        assert!(frontier.strict_f32_barrier_store_load_sequence);
        assert!(frontier.finite_guard_sequence_present);
        assert_eq!(frontier.direct_div_count, 0);
    }

    #[test]
    fn classifies_reassociation_when_barrier_sequence_is_absent() {
        assert_eq!(
            classify_strict_f32_sequence(true, 3, 0, false, false),
            "a770_qk256_strict_f32_disassembly_frontier_compiler_runtime_reassociation_or_collapse"
        );
    }

    #[test]
    fn json_receipt_keeps_claim_boundary_closed() {
        let frontier = inspect_strict_f32_frontier(
            "mul (16|M0) r1.0<1>:f r2.0<1;1,0>:f r3.0<0;1,0>:f\n",
            Path::new("sample.asm"),
        );
        let json = frontier_to_json(&frontier);
        assert!(json.contains("\"claim_allowed\": false"));
        assert!(json.contains("\"diagnostic_only\": true"));
        assert!(json.contains("\"production_qk256_policy_change\": false"));
        assert!(json.contains("\"bitnet_inference\": false"));
        assert!(json.contains("\"qk256_decode\": false"));
    }
}
