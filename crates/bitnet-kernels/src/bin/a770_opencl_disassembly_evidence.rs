use std::{
    env,
    error::Error,
    ffi::OsString,
    io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use bitnet_kernels::a770_opencl_runtime::{
    A770OpenClQk256CompilerBinaryEvidence, capture_a770_qk256_debug_compiler_binary_evidence,
    capture_a770_qk256_production_compiler_binary_evidence,
};

const RECEIPT_ENV: &str = "BITNET_A770_OPENCL_DISASSEMBLY_EVIDENCE_RECEIPT";
const ARTIFACT_DIR_ENV: &str = "BITNET_A770_OPENCL_DISASSEMBLY_ARTIFACT_DIR";
const OCLOC_ENV: &str = "BITNET_A770_OCLOC";
const DEFAULT_OCLOC_DEVICE: &str = "dg2-g10";

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let evidence = collect_disassembly_evidence(&args)?;
    let receipt = evidence_to_json(&evidence);
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
    receipt: Option<PathBuf>,
    artifact_dir: PathBuf,
    ocloc: Option<PathBuf>,
    device: String,
    kernel: KernelFlavor,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut receipt = env::var_os(RECEIPT_ENV).map(PathBuf::from);
        let mut artifact_dir = env::var_os(ARTIFACT_DIR_ENV).map(PathBuf::from);
        let mut ocloc = env::var_os(OCLOC_ENV).map(PathBuf::from);
        let mut device = DEFAULT_OCLOC_DEVICE.to_owned();
        let mut kernel = KernelFlavor::Debug;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--receipt" => {
                    let path = args
                        .next()
                        .ok_or_else(|| io_error("--receipt requires a path argument"))?;
                    receipt = Some(PathBuf::from(path));
                }
                "--artifact-dir" => {
                    let path = args
                        .next()
                        .ok_or_else(|| io_error("--artifact-dir requires a path argument"))?;
                    artifact_dir = Some(PathBuf::from(path));
                }
                "--ocloc" => {
                    let path =
                        args.next().ok_or_else(|| io_error("--ocloc requires a path argument"))?;
                    ocloc = Some(PathBuf::from(path));
                }
                "--device" => {
                    device = args.next().ok_or_else(|| io_error("--device requires a value"))?;
                }
                "--kernel" => {
                    kernel = KernelFlavor::parse(
                        &args.next().ok_or_else(|| io_error("--kernel requires a value"))?,
                    )?;
                }
                "--help" | "-h" => {
                    println!(
                        "Usage: a770-opencl-disassembly-evidence [--receipt <path>] [--artifact-dir <dir>] [--ocloc <path>] [--device <ocloc-device>] [--kernel debug|production]\n\nCaptures selected Intel Arc A770 OpenCL program binary disassembly evidence for the diagnostic QK256 debug kernel or production QK256 kernel."
                    );
                    std::process::exit(0);
                }
                other => return Err(io_error(format!("unknown argument {other:?}"))),
            }
        }

        let artifact_dir = artifact_dir.unwrap_or_else(|| default_artifact_dir(receipt.as_deref()));
        Ok(Self { receipt, artifact_dir, ocloc, device, kernel })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelFlavor {
    Debug,
    Production,
}

impl KernelFlavor {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "debug" => Ok(Self::Debug),
            "production" => Ok(Self::Production),
            other => Err(io_error(format!(
                "unknown --kernel value {other:?}; expected debug or production"
            ))),
        }
    }

    fn kernel_name(self) -> &'static str {
        match self {
            Self::Debug => "qk256_i2s_i8s_scaled_gemv_debug",
            Self::Production => "qk256_i2s_i8s_scaled_gemv",
        }
    }

    fn binary_file_name(self) -> &'static str {
        match self {
            Self::Debug => "qk256_i2s_i8s_scaled_gemv_debug.bin",
            Self::Production => "qk256_i2s_i8s_scaled_gemv.bin",
        }
    }

    fn kernel_source_label(self) -> &'static str {
        match self {
            Self::Debug => {
                "bitnet_kernels::a770_opencl_runtime::QK256_I2S_I8S_SCALED_GEMV_DEBUG_SRC"
            }
            Self::Production => {
                "bitnet_kernels::a770_opencl_runtime::QK256_I2S_I8S_SCALED_GEMV_SRC"
            }
        }
    }

    fn work_item(self) -> &'static str {
        match self {
            Self::Debug => "A770-057",
            Self::Production => "A770-060",
        }
    }

    fn proof_family(self) -> &'static str {
        match self {
            Self::Debug => "a770_opencl_qk256_compiler_disassembly_evidence",
            Self::Production => "a770_opencl_qk256_production_kernel_disassembly_evidence",
        }
    }

    fn proof_stage(self) -> &'static str {
        match self {
            Self::Debug => "diagnostic_compiler_disassembly_captured",
            Self::Production => "diagnostic_production_kernel_disassembly_context_captured",
        }
    }

    fn next_diagnostic(self) -> &'static str {
        match self {
            Self::Debug => {
                "inspect lowered strict-f32 barrier operation sequence before any production QK256 policy change"
            }
            Self::Production => {
                "inspect production-kernel lowered operation sequence and replay context before any production QK256 policy change"
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct A770OpenClQk256DisassemblyEvidence {
    kernel: KernelFlavor,
    compiler: A770OpenClQk256CompilerBinaryEvidence,
    artifact_dir: PathBuf,
    binary_path: Option<PathBuf>,
    binary_index: Option<usize>,
    ocloc_path: Option<PathBuf>,
    ocloc_device: String,
    ocloc_command: Option<Vec<String>>,
    ocloc_exit_code: Option<i32>,
    ocloc_stdout: String,
    ocloc_stderr: String,
    dump_dir: Option<PathBuf>,
    kernel_asm_path: Option<PathBuf>,
    kernel_asm_bytes: Option<usize>,
    kernel_asm_fnv1a64: Option<String>,
    kernel_asm_prefix: Option<String>,
    kernel_asm_trailing_whitespace_trimmed: bool,
    disassembly_captured: bool,
    classification: String,
}

impl A770OpenClQk256DisassemblyEvidence {
    fn compiler_flavor(&self) -> KernelFlavor {
        self.kernel
    }
}

fn collect_disassembly_evidence(
    args: &Args,
) -> Result<A770OpenClQk256DisassemblyEvidence, Box<dyn Error>> {
    let compiler = match args.kernel {
        KernelFlavor::Debug => capture_a770_qk256_debug_compiler_binary_evidence()?,
        KernelFlavor::Production => capture_a770_qk256_production_compiler_binary_evidence()?,
    };
    std::fs::create_dir_all(&args.artifact_dir)?;

    let (binary_index, binary_path) = if let Some((index, binary)) =
        compiler.binaries.iter().enumerate().find(|(_, b)| !b.is_empty())
    {
        let path = args.artifact_dir.join(args.kernel.binary_file_name());
        std::fs::write(&path, binary)?;
        (Some(index), Some(path))
    } else {
        (None, None)
    };

    let ocloc_path = resolve_ocloc_path(args.ocloc.as_deref());
    let mut ocloc_command = None;
    let mut ocloc_exit_code = None;
    let mut ocloc_stdout = String::new();
    let mut ocloc_stderr = String::new();
    let mut dump_dir = None;
    let mut kernel_asm_path = None;
    let mut kernel_asm_bytes = None;
    let mut kernel_asm_fnv1a64 = None;
    let mut kernel_asm_prefix = None;
    let mut kernel_asm_trailing_whitespace_trimmed = false;

    if let (Some(binary_path), Some(ocloc_path)) = (&binary_path, &ocloc_path) {
        let disasm_dump_dir = args.artifact_dir.join("ocloc-dump");
        if disasm_dump_dir.exists() {
            std::fs::remove_dir_all(&disasm_dump_dir)?;
        }
        std::fs::create_dir_all(&disasm_dump_dir)?;
        dump_dir = Some(disasm_dump_dir.clone());

        let command_args = vec![
            OsString::from("disasm"),
            OsString::from("-file"),
            binary_path.as_os_str().to_owned(),
            OsString::from("-device"),
            OsString::from(&args.device),
            OsString::from("-dump"),
            disasm_dump_dir.as_os_str().to_owned(),
        ];
        ocloc_command = Some(command_to_vec(ocloc_path, &command_args));
        let output = Command::new(ocloc_path).args(&command_args).output()?;
        ocloc_exit_code = output.status.code();
        ocloc_stdout = compact_output(&output.stdout);
        ocloc_stderr = compact_output(&output.stderr);

        if output.status.success() {
            if let Some(path) = find_kernel_asm(&disasm_dump_dir, args.kernel.kernel_name())? {
                let bytes = std::fs::read(&path)?;
                let (normalized, trimmed) = normalize_asm_bytes(&bytes);
                if trimmed {
                    std::fs::write(&path, &normalized)?;
                }
                kernel_asm_trailing_whitespace_trimmed = trimmed;
                kernel_asm_bytes = Some(normalized.len());
                kernel_asm_fnv1a64 = Some(fnv1a64_hex(&normalized));
                kernel_asm_prefix = Some(text_prefix(&String::from_utf8_lossy(&normalized), 512));
                kernel_asm_path = Some(path);
            }
        }
    }

    let disassembly_captured = kernel_asm_path.is_some();
    let classification = disassembly_evidence_classification(
        args.kernel,
        compiler.program_binary_captured,
        compiler.strict_f32_barrier_source_present,
        ocloc_path.is_some(),
        ocloc_exit_code.map(|code| code == 0).unwrap_or(false),
        disassembly_captured,
    )
    .to_owned();

    Ok(A770OpenClQk256DisassemblyEvidence {
        kernel: args.kernel,
        compiler,
        artifact_dir: args.artifact_dir.clone(),
        binary_path,
        binary_index,
        ocloc_path,
        ocloc_device: args.device.clone(),
        ocloc_command,
        ocloc_exit_code,
        ocloc_stdout,
        ocloc_stderr,
        dump_dir,
        kernel_asm_path,
        kernel_asm_bytes,
        kernel_asm_fnv1a64,
        kernel_asm_prefix,
        kernel_asm_trailing_whitespace_trimmed,
        disassembly_captured,
        classification,
    })
}

fn evidence_to_json(evidence: &A770OpenClQk256DisassemblyEvidence) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"campaign\": \"intel-a770\",\n",
            "  \"work_item\": \"{}\",\n",
            "  \"proof_family\": \"{}\",\n",
            "  \"proof_stage\": \"{}\",\n",
            "  \"requested_backend\": \"intel-arc-a770\",\n",
            "  \"selected_backend\": \"intel-arc-a770-opencl\",\n",
            "  \"runtime_api\": \"opencl\",\n",
            "  \"runtime_device\": \"{}\",\n",
            "  \"platform_index\": {},\n",
            "  \"device_index\": {},\n",
            "  \"platform_name\": \"{}\",\n",
            "  \"vendor\": \"{}\",\n",
            "  \"driver_version\": \"{}\",\n",
            "  \"kernel_source\": \"{}\",\n",
            "  \"kernel_name\": \"{}\",\n",
            "  \"classification\": \"{}\",\n",
            "  \"build_options\": \"{}\",\n",
            "  \"build_log\": \"{}\",\n",
            "  \"binary_type\": \"{}\",\n",
            "  \"kernel_names\": \"{}\",\n",
            "  \"program_device_count\": {},\n",
            "  \"binary_sizes\": {},\n",
            "  \"binary_fnv1a64\": {},\n",
            "  \"binary_prefix_hex\": {},\n",
            "  \"binary_index\": {},\n",
            "  \"binary_path\": {},\n",
            "  \"artifact_dir\": \"{}\",\n",
            "  \"source_bytes\": {},\n",
            "  \"source_fnv1a64\": \"{}\",\n",
            "  \"strict_f32_barrier_source_present\": {},\n",
            "  \"program_binary_captured\": {},\n",
            "  \"ocloc_available\": {},\n",
            "  \"ocloc_path\": {},\n",
            "  \"ocloc_device\": \"{}\",\n",
            "  \"ocloc_command\": {},\n",
            "  \"ocloc_exit_code\": {},\n",
            "  \"ocloc_stdout\": \"{}\",\n",
            "  \"ocloc_stderr\": \"{}\",\n",
            "  \"dump_dir\": {},\n",
            "  \"kernel_asm_path\": {},\n",
            "  \"kernel_asm_bytes\": {},\n",
            "  \"kernel_asm_fnv1a64\": {},\n",
            "  \"kernel_asm_prefix\": {},\n",
            "  \"kernel_asm_trailing_whitespace_trimmed\": {},\n",
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
            "  \"next_diagnostic\": \"{}\",\n",
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
        evidence.compiler_flavor().work_item(),
        evidence.compiler_flavor().proof_family(),
        evidence.compiler_flavor().proof_stage(),
        json_escape(&evidence.compiler.runtime_device),
        evidence.compiler.platform_index,
        evidence.compiler.device_index,
        json_escape(&evidence.compiler.platform_name),
        json_escape(&evidence.compiler.vendor),
        json_escape(&evidence.compiler.driver_version),
        evidence.compiler_flavor().kernel_source_label(),
        evidence.compiler_flavor().kernel_name(),
        json_escape(&evidence.classification),
        json_escape(&evidence.compiler.build_options),
        json_escape(&evidence.compiler.build_log),
        json_escape(&evidence.compiler.binary_type),
        json_escape(&evidence.compiler.kernel_names),
        evidence.compiler.program_device_count,
        usize_array_json(&evidence.compiler.binary_sizes),
        string_array_json(&evidence.compiler.binary_fnv1a64),
        string_array_json(&evidence.compiler.binary_prefix_hex),
        option_usize_json(evidence.binary_index),
        option_path_json(evidence.binary_path.as_deref()),
        json_escape(&path_json_value(&evidence.artifact_dir)),
        evidence.compiler.source_bytes,
        json_escape(&evidence.compiler.source_fnv1a64),
        evidence.compiler.strict_f32_barrier_source_present,
        evidence.compiler.program_binary_captured,
        evidence.ocloc_path.is_some(),
        option_path_json(evidence.ocloc_path.as_deref()),
        json_escape(&evidence.ocloc_device),
        option_string_array_json(evidence.ocloc_command.as_deref()),
        option_i32_json(evidence.ocloc_exit_code),
        json_escape(&evidence.ocloc_stdout),
        json_escape(&evidence.ocloc_stderr),
        option_path_json(evidence.dump_dir.as_deref()),
        option_path_json(evidence.kernel_asm_path.as_deref()),
        option_usize_json(evidence.kernel_asm_bytes),
        option_string_json(evidence.kernel_asm_fnv1a64.as_deref()),
        option_string_json(evidence.kernel_asm_prefix.as_deref()),
        evidence.kernel_asm_trailing_whitespace_trimmed,
        evidence.disassembly_captured,
        json_escape(evidence.compiler_flavor().next_diagnostic())
    )
}

fn default_artifact_dir(receipt: Option<&Path>) -> PathBuf {
    receipt
        .and_then(Path::parent)
        .map(|parent| parent.join("a770-opencl-qk256-compiler-disassembly"))
        .unwrap_or_else(|| PathBuf::from("target/a770-opencl-qk256-compiler-disassembly"))
}

fn resolve_ocloc_path(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return path.exists().then(|| path.to_path_buf());
    }
    for path in common_ocloc_paths() {
        if path.exists() {
            return Some(path);
        }
    }
    let names = if cfg!(windows) { ["ocloc.exe", "ocloc"] } else { ["ocloc", "ocloc"] };
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        for name in names {
            let candidate = dir.join(name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

fn common_ocloc_paths() -> Vec<PathBuf> {
    vec![
        PathBuf::from(r"C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\bin\ocloc.exe"),
        PathBuf::from(r"C:\Program Files (x86)\Intel\oneAPI\ocloc\2024.2\bin\ocloc.exe"),
        PathBuf::from(r"C:\Program Files (x86)\Intel\oneAPI\compiler\2024.0\lib\ocloc\ocloc.exe"),
        PathBuf::from(
            r"C:\Program Files (x86)\Intel\oneAPI\compiler\2023.2.0\windows\lib\ocloc\gen12+\ocloc.exe",
        ),
        PathBuf::from("/usr/bin/ocloc"),
    ]
}

fn find_kernel_asm(dir: &Path, kernel_name: &str) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let mut stack = vec![dir.to_path_buf()];
    let mut candidates = Vec::new();
    while let Some(path) = stack.pop() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.ends_with(".asm") {
                candidates.push(path);
            }
        }
    }
    candidates.sort();
    let expected_file_name = format!(".text.{kernel_name}.asm");
    Ok(candidates
        .into_iter()
        .find(|path| path.file_name().is_some_and(|name| name == expected_file_name.as_str())))
}

fn command_to_vec(program: &Path, args: &[OsString]) -> Vec<String> {
    let mut command = vec![path_json_value(program)];
    command.extend(args.iter().map(|arg| arg.to_string_lossy().to_string()));
    command
}

fn compact_output(bytes: &[u8]) -> String {
    text_prefix(&String::from_utf8_lossy(bytes).replace("\r\n", "\n"), 2048)
}

fn text_prefix(text: &str, limit: usize) -> String {
    let mut out = String::new();
    for ch in text.chars().take(limit) {
        out.push(ch);
    }
    if text.chars().count() > limit {
        out.push_str("...");
    }
    out
}

fn normalize_asm_bytes(bytes: &[u8]) -> (Vec<u8>, bool) {
    let text = String::from_utf8_lossy(bytes).replace("\r\n", "\n");
    let mut normalized = String::new();
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            normalized.push('\n');
        }
        let trimmed_line = line.trim_end_matches([' ', '\t']);
        normalized.push_str(trimmed_line);
    }
    let normalized = normalized.into_bytes();
    let trimmed = normalized != bytes;
    (normalized, trimmed)
}

fn disassembly_evidence_classification(
    kernel: KernelFlavor,
    program_binary_captured: bool,
    strict_f32_barrier_source_present: bool,
    ocloc_available: bool,
    ocloc_success: bool,
    kernel_asm_captured: bool,
) -> &'static str {
    if kernel == KernelFlavor::Production {
        if !program_binary_captured {
            return "a770_qk256_production_kernel_disassembly_evidence_missing_program_binary";
        }
        if !ocloc_available {
            return "a770_qk256_production_kernel_disassembly_evidence_ocloc_missing";
        }
        if !ocloc_success {
            return "a770_qk256_production_kernel_disassembly_evidence_disasm_failed";
        }
        if !kernel_asm_captured {
            return "a770_qk256_production_kernel_disassembly_evidence_kernel_asm_missing";
        }
        return "a770_qk256_production_kernel_disassembly_evidence_captured";
    }
    if !program_binary_captured {
        return "a770_qk256_opencl_disassembly_evidence_missing_program_binary";
    }
    if !strict_f32_barrier_source_present {
        return "a770_qk256_opencl_disassembly_evidence_missing_strict_f32_source_context";
    }
    if !ocloc_available {
        return "a770_qk256_opencl_disassembly_evidence_ocloc_missing";
    }
    if !ocloc_success {
        return "a770_qk256_opencl_disassembly_evidence_disasm_failed";
    }
    if !kernel_asm_captured {
        return "a770_qk256_opencl_disassembly_evidence_kernel_asm_missing";
    }
    "a770_qk256_opencl_disassembly_evidence_captured"
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

fn option_string_array_json(values: Option<&[String]>) -> String {
    values.map(string_array_json).unwrap_or_else(|| "null".to_owned())
}

fn option_path_json(path: Option<&Path>) -> String {
    match path {
        Some(path) => {
            let value = path_json_value(path);
            option_string_json(Some(&value))
        }
        None => "null".to_owned(),
    }
}

fn option_string_json(value: Option<&str>) -> String {
    value.map(|value| format!("\"{}\"", json_escape(value))).unwrap_or_else(|| "null".to_owned())
}

fn option_usize_json(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "null".to_owned())
}

fn option_i32_json(value: Option<i32>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "null".to_owned())
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

#[allow(dead_code)]
fn _assert_output_is_used(_: &Output) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_requires_binary_first() {
        assert_eq!(
            disassembly_evidence_classification(KernelFlavor::Debug, false, true, true, true, true),
            "a770_qk256_opencl_disassembly_evidence_missing_program_binary"
        );
    }

    #[test]
    fn classification_splits_ocloc_and_disassembly_failures() {
        assert_eq!(
            disassembly_evidence_classification(
                KernelFlavor::Debug,
                true,
                true,
                false,
                false,
                false
            ),
            "a770_qk256_opencl_disassembly_evidence_ocloc_missing"
        );
        assert_eq!(
            disassembly_evidence_classification(
                KernelFlavor::Debug,
                true,
                true,
                true,
                false,
                false
            ),
            "a770_qk256_opencl_disassembly_evidence_disasm_failed"
        );
        assert_eq!(
            disassembly_evidence_classification(KernelFlavor::Debug, true, true, true, true, false),
            "a770_qk256_opencl_disassembly_evidence_kernel_asm_missing"
        );
        assert_eq!(
            disassembly_evidence_classification(KernelFlavor::Debug, true, true, true, true, true),
            "a770_qk256_opencl_disassembly_evidence_captured"
        );
    }

    #[test]
    fn production_classification_ignores_debug_strict_f32_source_requirement() {
        assert_eq!(
            disassembly_evidence_classification(
                KernelFlavor::Production,
                true,
                false,
                true,
                true,
                true
            ),
            "a770_qk256_production_kernel_disassembly_evidence_captured"
        );
        assert_eq!(
            disassembly_evidence_classification(
                KernelFlavor::Production,
                true,
                false,
                true,
                true,
                false
            ),
            "a770_qk256_production_kernel_disassembly_evidence_kernel_asm_missing"
        );
    }

    #[test]
    fn json_helpers_escape_and_format_options() {
        assert_eq!(json_escape("a\"b\\c\n"), "a\\\"b\\\\c\\n");
        assert_eq!(option_usize_json(Some(7)), "7");
        assert_eq!(option_usize_json(None), "null");
        assert_eq!(option_string_json(Some("a\"b")), "\"a\\\"b\"");
    }

    #[test]
    fn text_prefix_marks_truncation() {
        assert_eq!(text_prefix("abcdef", 3), "abc...");
        assert_eq!(text_prefix("abc", 3), "abc");
    }

    #[test]
    fn normalize_asm_bytes_trims_line_end_whitespace() {
        let (normalized, trimmed) = normalize_asm_bytes(b"mov   \r\nnop\t\n");
        assert!(trimmed);
        assert_eq!(normalized, b"mov\nnop\n");
    }

    #[test]
    fn find_kernel_asm_requires_exact_kernel_file_name() {
        let root =
            std::env::temp_dir().join(format!("a770-find-kernel-asm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let debug = root.join(".text.qk256_i2s_i8s_scaled_gemv_debug.asm");
        let production = root.join(".text.qk256_i2s_i8s_scaled_gemv.asm");
        std::fs::write(&debug, "debug").unwrap();

        assert_eq!(find_kernel_asm(&root, "qk256_i2s_i8s_scaled_gemv").unwrap(), None);

        std::fs::write(&production, "production").unwrap();
        assert_eq!(find_kernel_asm(&root, "qk256_i2s_i8s_scaled_gemv").unwrap(), Some(production));
        assert_eq!(find_kernel_asm(&root, "qk256_i2s_i8s_scaled_gemv_debug").unwrap(), Some(debug));

        let _ = std::fs::remove_dir_all(&root);
    }
}
