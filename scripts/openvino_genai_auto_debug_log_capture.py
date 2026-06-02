#!/usr/bin/env python3
"""Capture OpenVINO GenAI AUTO debug logs and emit selected-device evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openvino_genai_auto_debug_log_parser import build_debug_log_evidence


ARTIFACT_KIND = "lunar_lake_openvino_auto_genai_debug_log_evidence"
DEFAULT_PHASE_SCRIPT = Path(__file__).resolve().parent / "openvino_genai_phase_receipt.py"
ATTENTION_BACKEND_WARNING = (
    "Paged Attention backend initialization failed. Falling back to SDPA backend."
)
AUTO_FALLBACK_DISABLED_MARKERS = ("enable_startup_fallback", "enable_running_fallback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--phase-json-out", required=True, type=Path)
    parser.add_argument("--debug-log-out", required=True, type=Path)
    parser.add_argument("--evidence-json-out", required=True, type=Path)
    parser.add_argument("--phase-script", default=DEFAULT_PHASE_SCRIPT, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--machine-id", default="intel-258v")
    parser.add_argument("--item", default="LNL258V-NPU-AUTO-SOURCE-001")
    parser.add_argument("--devices", nargs="+", default=["AUTO"])
    parser.add_argument("--openvino-log-level", default="2")
    parser.add_argument("--created-utc")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write unavailable evidence and exit zero when the stateful LLM block is absent.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def auto_device_block(phase_receipt: dict[str, Any]) -> dict[str, Any] | None:
    devices = (phase_receipt.get("generation") or {}).get("devices") or []
    for device in devices:
        if isinstance(device, dict) and str(device.get("runtime_device", "")).upper() == "AUTO":
            return device
    return None


def requested_devices_from_phase(phase_receipt: dict[str, Any], fallback: list[str]) -> list[str]:
    devices = (phase_receipt.get("generation") or {}).get("devices") or []
    requested = [
        str(device.get("runtime_device"))
        for device in devices
        if isinstance(device, dict) and device.get("runtime_device") is not None
    ]
    return requested or fallback


def matching_log_lines(text: str, markers: tuple[str, ...]) -> list[dict[str, Any]]:
    matches = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if any(marker in line for marker in markers):
            matches.append({"line": line_no, "text": line.rstrip()})
    return matches


def same_run_answer_and_fallback(auto_device: dict[str, Any] | None, debug_log_text: str) -> dict[str, Any]:
    cases = []
    if auto_device:
        for case in auto_device.get("cases") or []:
            if not isinstance(case, dict):
                continue
            answer_gate = case.get("answer_gate") or {}
            cases.append(
                {
                    "id": case.get("id"),
                    "answer_gate_passed": bool(answer_gate.get("passed")),
                    "fallback_used": bool(case.get("fallback_used")),
                    "selected_device_visibility_status_from_phase_receipt": case.get(
                        "selected_device_visibility_status"
                    ),
                    "openvino_runtime_auto_selected_device_proof_from_phase_receipt": bool(
                        case.get("openvino_runtime_auto_selected_device_proof")
                    ),
                }
            )

    attention_warning_lines = matching_log_lines(debug_log_text, (ATTENTION_BACKEND_WARNING,))
    auto_fallback_disabled_lines = matching_log_lines(debug_log_text, AUTO_FALLBACK_DISABLED_MARKERS)

    return {
        "phase_receipt_runtime_device": "AUTO",
        "phase_receipt_fallback_used": bool(auto_device.get("fallback_used")) if auto_device else False,
        "phase_receipt_fallback_status": (
            auto_device.get("fallback_status")
            if auto_device
            else "auto_device_block_missing_from_phase_receipt"
        ),
        "case_count": len(cases),
        "all_answer_gates_passed": bool(cases) and all(case["answer_gate_passed"] for case in cases),
        "cases": cases,
        "attention_backend_warning_observed": bool(attention_warning_lines),
        "attention_backend_warning_lines": attention_warning_lines,
        "attention_backend_warning_boundary": (
            "The SDPA warning is recorded as an attention-backend implementation fallback, "
            "not as application route/device fallback; application fallback_used remains "
            "false in the phase receipt."
        ),
        "auto_startup_running_fallback_disabled_observed": bool(auto_fallback_disabled_lines),
        "auto_startup_running_fallback_disabled_lines": auto_fallback_disabled_lines,
    }


def run_phase_script(args: argparse.Namespace) -> list[str]:
    if "AUTO" not in [device.upper() for device in args.devices]:
        raise ValueError("--devices must include AUTO for selected-device evidence capture")

    command = [
        str(args.python),
        str(args.phase_script),
        "--model-dir",
        str(args.model_dir),
        "--machine-id",
        str(args.machine_id),
        "--devices",
        *args.devices,
        "--json-out",
        str(args.phase_json_out),
    ]
    env = os.environ.copy()
    env["OPENVINO_LOG_LEVEL"] = str(args.openvino_log_level)
    args.phase_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.debug_log_out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    args.debug_log_out.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return command


def build_evidence_receipt(args: argparse.Namespace, command: list[str]) -> dict[str, Any]:
    phase_receipt = load_json(args.phase_json_out)
    auto_device = auto_device_block(phase_receipt)
    debug_log_text = args.debug_log_out.read_text(encoding="utf-8")
    parsed_evidence = build_debug_log_evidence(debug_log_text)
    phase_record = file_record(args.phase_json_out)
    log_record = file_record(args.debug_log_out)
    phase_record.update(
        {
            "artifact_kind": phase_receipt.get("artifact_kind"),
            "source_item": phase_receipt.get("item"),
            "runtime_api": phase_receipt.get("runtime_api"),
            "requested_devices": requested_devices_from_phase(phase_receipt, args.devices),
            "phase_receipt_selected_device_visibility_status": (
                auto_device.get("selected_device_visibility_status") if auto_device else "not_exposed"
            ),
            "phase_receipt_openvino_runtime_auto_selected_device_proof": bool(
                auto_device.get("openvino_runtime_auto_selected_device_proof") if auto_device else False
            ),
        }
    )
    log_record.update(
        {
            "captured_streams": "stdout_and_stderr_combined",
            "openvino_log_level_env": str(args.openvino_log_level),
            "openvino_log_level_reason": (
                "Numeric OpenVINO log level is required for GenAI debug output that can expose "
                "stateful LLM model block execution devices."
            ),
        }
    )

    return {
        "schema_version": "1.0.0",
        "artifact_kind": ARTIFACT_KIND,
        "campaign": "intel-258v-platform",
        "item": args.item,
        "linked_issues": [1251, 1149, 1242, 1248, 1119, 1064],
        "created_utc": args.created_utc or utc_now(),
        "machine_id": args.machine_id,
        "proof_stage": "genai_runtime_auto_debug_log_receipt_source_no_policy_change",
        "source_phase_receipt": phase_record,
        "debug_log": log_record,
        "command": {
            "environment": {"OPENVINO_LOG_LEVEL": str(args.openvino_log_level)},
            "argv": command,
        },
        "environment": phase_receipt.get("environment", {}),
        "model_tuple": {
            "model_family": phase_receipt.get("model_family"),
            "model_architecture": phase_receipt.get("model_architecture"),
            "quantization": phase_receipt.get("quantization"),
            "prompt_template": phase_receipt.get("prompt_template"),
            "tokenizer_source": phase_receipt.get("tokenizer_source"),
            "model": phase_receipt.get("model"),
        },
        "genai_debug_log_evidence": parsed_evidence,
        "same_run_answer_and_fallback": same_run_answer_and_fallback(auto_device, debug_log_text),
        "claim_boundary": {
            "may_claim": [
                "A repeatable receipt-source path captured OpenVINO GenAI runtime AUTO debug logs.",
                "The parser interpreted only the accepted stateful LLM model block in the captured log.",
            ],
            "must_not_claim": [
                "No route policy changed.",
                "No OpenVINO GPU, NPU, or AUTO route promotion changed.",
                "No low_power promotion, battery-mode evidence, power advantage, speedup, or benchmark-qualified advantage is proven.",
                "No native OpenCL, native NPU, acceleration, broad dense SLM quality, model-format equivalence, or BitNet QK256/I2_S behavior claim is proven.",
                "Generated phase receipts with selected_device_visibility_status=not_exposed are not converted into selected-device proof.",
            ],
        },
    }


def main() -> int:
    args = parse_args()
    try:
        command = run_phase_script(args)
    except subprocess.CalledProcessError as exc:
        print(f"phase script failed with exit code {exc.returncode}; debug log was still captured", file=sys.stderr)
        return exc.returncode

    receipt = build_evidence_receipt(args, command)
    args.evidence_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_json_out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    visibility_status = receipt["genai_debug_log_evidence"].get("visibility_status")
    if visibility_status != "exposed_by_genai_debug_log" and not args.allow_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
